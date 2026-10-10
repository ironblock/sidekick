"""Latency and resident memory of a Core ML embedder on each compute path,
for comparison with tools/bench_onnx.py.

Each compute path (cpu, gpu, ane) runs in a fresh process: load time (on
the ANE the first load includes its compile), resident memory after load
and after the runs (ps RSS), and the median and p90 latency of one
prediction per length. A Core ML artifact takes one input per prediction
at a static bucket, so a length runs in the smallest bucket that fits it,
padded, as the server runs it. Inputs are real token ids from the model's
parity reference (its longest case, cut to each length). The load average
is recorded before and after each path.

Usage:
    python tools/bench_coreml.py <model-dir> --refs <refs-dir>
        [--lengths 16,64,256] [--units cpu,gpu,ane] [--runs 30] [--json out.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

UNITS = {"cpu": "CPU_ONLY", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE"}


def rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024


def child(args):
    import numpy as np
    import coremltools as ct
    manifest = tomllib.loads((args.model_dir / "manifest.toml").read_text())
    io = manifest["io"]
    ref = json.loads((args.refs / manifest["id"] / "reference.json").read_text())
    rows = [json.loads(c["ids"]) if isinstance(c["ids"], str) else c["ids"] for c in ref["cases"]]
    longest = max(rows, key=len)
    buckets = sorted(manifest["buckets"])
    out = {"units": args.units, "load_avg_before": os.getloadavg(), "rss_before_mb": rss_mb(), "load_ms": {},
           "runs": []}
    models = {}
    for length in args.lengths:
        bucket = next((b for b in buckets if b >= length), None)
        if bucket is None:
            continue
        if bucket not in models:
            path = args.model_dir / manifest["artifact"].replace("{seq}", str(bucket))
            t = time.perf_counter()
            models[bucket] = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, UNITS[args.units]))
            out["load_ms"][bucket] = (time.perf_counter() - t) * 1000
        ids = longest[:length - 1] + longest[-1:] if len(longest) >= length else longest
        feed = {io["input_ids"]: np.zeros((1, bucket), np.int32), io["attention_mask"]: np.zeros((1, bucket), np.int32)}
        feed[io["input_ids"]][0, :len(ids)] = ids
        feed[io["attention_mask"]][0, :len(ids)] = 1
        m = models[bucket]
        for _ in range(3):
            m.predict(feed)
        ms = []
        for _ in range(args.runs):
            t = time.perf_counter()
            m.predict(feed)
            ms.append((time.perf_counter() - t) * 1000)
        out["runs"].append({"tokens": len(ids), "bucket": bucket, "median_ms": float(np.median(ms)),
                            "p90_ms": float(np.percentile(ms, 90))})
    out["rss_after_mb"] = rss_mb()
    out["load_avg_after"] = os.getloadavg()
    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--lengths", default="16,64,256")
    p.add_argument("--units", default="cpu,gpu,ane")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.lengths = [int(x) for x in str(args.lengths).split(",")]
    if args.child:
        return child(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_onnx import machine
    report = {"machine": machine(), "model_dir": args.model_dir.name, "paths": []}
    for units in args.units.split(","):
        cmd = [sys.executable, __file__, str(args.model_dir), "--refs", str(args.refs), "--child", "--units", units,
               "--lengths", ",".join(map(str, args.lengths)), "--runs", str(args.runs)]
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        report["paths"].append(res)
        loads = ", ".join(f"{b}: {ms:.0f} ms" for b, ms in res["load_ms"].items())
        print(f"{units}: load {loads}; RSS {res['rss_after_mb']:.0f} MB after runs; "
              f"load average {res['load_avg_before'][0]:.1f} -> {res['load_avg_after'][0]:.1f}")
        for r in res["runs"]:
            print(f"  {r['tokens']:4d} tokens (bucket {r['bucket']}): median {r['median_ms']:.1f} ms (p90 {r['p90_ms']:.1f})")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
