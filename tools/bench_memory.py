"""Memory footprint of an embedder on one path (onnx-cpu, or Core ML's cpu,
gpu or ane), measured three ways, in a fresh process per run:

- the process's own footprint (phys_footprint, what `footprint` and
  Activity Monitor's Memory column report), and its lifetime peak, from
  proc_pid_rusage: before loading, after loading, and after one serving pass
  (one prediction at each length);
- the whole machine's memory in use (vm_stat: wired, active, inactive and
  compressor-held pages, plus speculative and free) at the same three
  points. Core ML keeps part of a Neural Engine model in system services,
  which the process's own count misses; the machine-wide change catches it,
  with the noise of whatever else runs;
- the artifact's size on disk.

Each run writes one JSON object (--json) with every raw value, the machine
and the load average, for a results generator to read.

Usage:
    python tools/bench_memory.py <model-dir> --path onnx-cpu|cpu|gpu|ane --refs <refs-dir>
        [--corpus main|long] [--model-file F] [--lengths 16,64,256] [--json out.json]

    --corpus long takes the token ids from the long-input reference
    (reference-long.json), for lengths past 512.
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
# --corpus: the parity reference (tools/parity_reference.py) to take token ids from.
REFERENCE = {"main": "reference.json", "long": "reference-long.json"}


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
        "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
        "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
        "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
        "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
        "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
        "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")]


def footprint():
    """(phys_footprint, lifetime peak phys_footprint) of this process, bytes."""
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = _RUsageInfoV4()
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info.phys_footprint, info.lifetime_max_phys_footprint


def vm_stat():
    """vm_stat's page counts, as bytes, plus the in-use total (wired +
    active + inactive + compressor-held)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = {m.group(1).strip().lower().replace(" ", "_").replace('"', ""): int(m.group(2)) * PAGE
             for m in re.finditer(r"^(.+?):\s+(\d+)\.?$", out, re.M) if not m.group(1).startswith("Mach")}
    pages["in_use"] = sum(pages.get(k, 0) for k in ("pages_wired_down", "pages_active", "pages_inactive",
                                                     "pages_occupied_by_compressor"))
    return pages


def snapshot():
    fp, peak = footprint()
    return {"phys_footprint": fp, "peak_phys_footprint": peak, "vm": vm_stat(), "load_avg": os.getloadavg(),
            "t": time.time()}


def artifact_files(model_dir, manifest, path, model_file):
    if path == "onnx-cpu":
        f = model_dir / (model_file or manifest["artifact"])
        files = [f] + [p for p in (Path(str(f) + "_data"), Path(str(f) + ".data")) if p.exists()]
        return [p.resolve() for p in files]
    return [model_dir / manifest["artifact"].replace("{seq}", str(b)) for b in manifest["buckets"]]


def size_on_disk(paths):
    total = 0
    for p in paths:
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total


def child(args):
    import numpy as np
    manifest = tomllib.loads((args.model_dir / "manifest.toml").read_text())
    ref = json.loads((args.refs / manifest["id"] / REFERENCE[args.corpus]).read_text())
    rows = [json.loads(c["ids"]) if isinstance(c["ids"], str) else c["ids"] for c in ref["cases"]]
    longest = max(rows, key=len)
    inputs = [longest[:n - 1] + longest[-1:] if len(longest) >= n else longest for n in args.lengths]
    if args.path == "onnx-cpu":
        import onnxruntime  # noqa: F401  (imported before the baseline, as the runtime is)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sidekick_convert.onnx_export import Runner
    else:
        import coremltools as ct
    out = {"model": manifest["id"], "path": args.path, "model_file": args.model_file,
           "artifact_bytes": size_on_disk(artifact_files(args.model_dir, manifest, args.path, args.model_file)),
           "before": snapshot()}
    t = time.perf_counter()
    if args.path == "onnx-cpu":
        pooling = None if manifest.get("pooling", "none") == "none" else manifest["pooling"]
        runner = Runner(args.model_dir / (args.model_file or manifest["artifact"]), pooling)
        run = lambda ids: runner.run([ids])  # noqa: E731
    else:
        units = {"cpu": "CPU_ONLY", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE"}[args.path]
        io, buckets, models = manifest["io"], sorted(manifest["buckets"]), {}
        # the server loads a bucket on first use: load each bucket the lengths land in
        for b in sorted({next(x for x in buckets if x >= len(i)) for i in inputs}):
            path = args.model_dir / manifest["artifact"].replace("{seq}", str(b))
            models[b] = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))

        def run(ids):
            b = next(x for x in buckets if x >= len(ids))
            feed = {io["input_ids"]: np.zeros((1, b), np.int32), io["attention_mask"]: np.zeros((1, b), np.int32)}
            feed[io["input_ids"]][0, :len(ids)] = ids
            feed[io["attention_mask"]][0, :len(ids)] = 1
            return models[b].predict(feed)
    out["load_ms"] = (time.perf_counter() - t) * 1000
    out["loaded"] = snapshot()
    for ids in inputs:
        run(ids)
    out["served"] = snapshot()
    print(json.dumps(out))


def mb(x):
    return x / 2**20


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--path", required=True, choices=("onnx-cpu", "cpu", "gpu", "ane"))
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--corpus", choices=sorted(REFERENCE), default="main",
                   help="the reference whose longest case is cut to each length: main (reference.json) or "
                        "long (reference-long.json, for lengths past 512)")
    p.add_argument("--model-file", default=None)
    p.add_argument("--lengths", default="16,64,256")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    if args.child:
        return child(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_onnx import machine
    cmd = [sys.executable, __file__, str(args.model_dir), "--path", args.path, "--refs", str(args.refs),
           "--corpus", args.corpus,
           "--lengths", ",".join(map(str, args.lengths)), "--child"]
    if args.model_file:
        cmd += ["--model-file", args.model_file]
    res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    res["machine"] = machine()
    b, l, s = res["before"], res["loaded"], res["served"]
    print(f"{res['model']} {args.path} {args.model_file or ''}: disk {mb(res['artifact_bytes']):.0f} MB; "
          f"load {res['load_ms']:.0f} ms; footprint +{mb(l['phys_footprint'] - b['phys_footprint']):.0f} MB loaded, "
          f"+{mb(s['phys_footprint'] - b['phys_footprint']):.0f} MB served; peak +"
          f"{mb(s['peak_phys_footprint'] - b['phys_footprint']):.0f} MB; machine in use "
          f"+{mb(l['vm']['in_use'] - b['vm']['in_use']):.0f} MB loaded, "
          f"+{mb(s['vm']['in_use'] - b['vm']['in_use']):.0f} MB served; load average {b['load_avg'][0]:.1f}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
