"""Latency and resident memory of an ONNX embedder on ONNX Runtime's CPU.

Each configuration runs in a fresh process, so its memory is its own:
session load time, resident memory after load and after the runs (ps
RSS), and the median and p90 latency of a forward pass at each length and
batch size, after warm-up. Inputs are real token ids from the model's parity
reference (its longest case, cut to each length), so the timing sees real
attention patterns rather than padding.

Usage:
    python tools/bench_onnx.py <model-dir> --refs <refs-dir> [--model-file F]
        [--lengths 16,64,256] [--batches 1,8] [--threads 0,4,8] [--runs 30]
        [--json out.json]

    --threads: intra-op threads; 0 is ONNX Runtime's default (one per core)

Record the machine with every number: the output carries the CPU model and
core counts.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
import tomllib
from pathlib import Path


def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024


def machine():
    def sysctl(k):
        return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
    if platform.system() == "Darwin":
        return {"cpu": sysctl("machdep.cpu.brand_string"), "cores": sysctl("hw.ncpu"),
                "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
                "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"), "os": platform.platform()}
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), platform.processor())
    import os
    return {"cpu": model, "cores": os.cpu_count(), "os": platform.platform()}


def child(args):
    """One configuration, in this process."""
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sidekick_convert.onnx_export import Runner
    manifest = tomllib.loads((args.model_dir / "manifest.toml").read_text())
    pooling = None if manifest.get("pooling", "none") == "none" else manifest["pooling"]
    ref = json.loads((args.refs / manifest["id"] / "reference.json").read_text())
    rows = [json.loads(c["ids"]) if isinstance(c["ids"], str) else c["ids"] for c in ref["cases"]]
    longest = max(rows, key=len)
    before = rss_mb(__import__("os").getpid())
    t = time.perf_counter()
    runner = Runner(args.model_dir / (args.model_file or manifest["artifact"]), pooling,
                    threads=args.threads or None)
    load_ms = (time.perf_counter() - t) * 1000
    loaded = rss_mb(__import__("os").getpid())
    out = {"ort": ort.__version__, "threads": args.threads, "load_ms": load_ms,
           "rss_before_mb": before, "rss_loaded_mb": loaded, "runs": []}
    for length in args.lengths:
        ids = longest[:length - 1] + longest[-1:] if len(longest) >= length else longest
        for batch in args.batches:
            rows_ = [ids] * batch
            for _ in range(3):
                runner.run(rows_)
            ms = []
            for _ in range(args.runs):
                t = time.perf_counter()
                runner.run(rows_)
                ms.append((time.perf_counter() - t) * 1000)
            out["runs"].append({"tokens": len(ids), "batch": batch, "median_ms": float(np.median(ms)),
                                "p90_ms": float(np.percentile(ms, 90)),
                                "ms_per_input": float(np.median(ms)) / batch})
    out["rss_after_mb"] = rss_mb(__import__("os").getpid())
    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--model-file", default=None)
    p.add_argument("--lengths", default="16,64,256")
    p.add_argument("--batches", default="1,8")
    p.add_argument("--threads", default="0")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.lengths = [int(x) for x in str(args.lengths).split(",")]
    args.batches = [int(x) for x in str(args.batches).split(",")]
    if args.child:
        args.threads = int(args.threads)
        return child(args)
    report = {"machine": machine(), "model_dir": args.model_dir.name, "model_file": args.model_file, "configs": []}
    for threads in [int(x) for x in args.threads.split(",")]:
        cmd = [sys.executable, __file__, str(args.model_dir), "--refs", str(args.refs), "--child",
               "--threads", str(threads), "--lengths", ",".join(map(str, args.lengths)),
               "--batches", ",".join(map(str, args.batches)), "--runs", str(args.runs)]
        if args.model_file:
            cmd += ["--model-file", args.model_file]
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        report["configs"].append(res)
        print(f"threads {threads or 'default'}: load {res['load_ms']:.0f} ms, RSS {res['rss_loaded_mb']:.0f} MB "
              f"loaded / {res['rss_after_mb']:.0f} MB after runs")
        for r in res["runs"]:
            print(f"  {r['tokens']:4d} tokens x {r['batch']}: median {r['median_ms']:.1f} ms "
                  f"(p90 {r['p90_ms']:.1f}), {r['ms_per_input']:.1f} ms/input")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
