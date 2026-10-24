"""companion.py --primary <mlx model> --ane <model_dir> --refs <refs_dir> --out <json> [--seconds 60] [--n 128]

The companion test: a GPU LLM (MLX, in this process) decodes beside the spike's
ANE decoder (skrun's "loop" task, a child process). Phases, each after a
cooldown, with whole-system power logged throughout (powerlog.py):
  idle            nothing runs (baseline power)
  gpu_alone       the primary generates back to back
  ane_alone       the ANE model generates back to back
  gpu_with_ane    both at once
  gpu_alone_again the primary alone (drift)
The primary's decode rate comes from mlx_lm's own generation_tps per run; the
ANE model's from skrun. Every child has a time cap (ANE hangs)."""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
WORK = Path(os.environ.get("SPIKE_WORK", ".dev-data/spike-ane-decoder"))   # skrun binary, results, Core ML home
(WORK / "results" / "raw").mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--primary", required=True)
ap.add_argument("--ane", required=True)
ap.add_argument("--refs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--primary-tokens", type=int, default=200)
ap.add_argument("--cooldown", type=float, default=15)
a = ap.parse_args()

from mlx_lm import load, stream_generate
import mlx.core as mx

power_path = Path(a.out).with_suffix(".power.jsonl")
power = subprocess.Popen([sys.executable, str(HERE / "powerlog.py"), str(power_path), "1"])
phases = []
model, tok = load(a.primary)
PROMPT = tok.apply_chat_template([{"role": "user", "content": "Write a detailed essay about the history of the printing press."}],
                                 add_generation_prompt=True, tokenize=False)
g = json.load(open(Path(a.refs) / "greedy.json"))
job = WORK / "results" / "job_companion_loop.json"
stop = WORK / "results" / "companion.stop"


def gpu_loop(seconds):
    runs, t0 = [], time.time()
    while time.time() - t0 < seconds:
        last, s = None, time.time()
        for r in stream_generate(model, tok, PROMPT, max_tokens=a.primary_tokens):
            last = r
        runs.append({"start": s, "end": time.time(), "decode_tok_s": last.generation_tps,
                     "prompt_tok_s": last.prompt_tps, "tokens": last.generation_tokens,
                     "peak_memory_GB": last.peak_memory})
    return runs


def ane_start(seconds, out):
    if stop.exists():
        stop.unlink()
    json.dump({"tasks": [{"type": "loop", "name": "loop", "ids": g[0]["ids"], "n": a.n, "seconds": seconds,
                          "stop_file": str(stop)}]}, open(job, "w"))
    return subprocess.Popen([str(WORK / "skrun"), a.ane, "cpu_and_ne", str(job), out],
                            env={"CFFIXED_USER_HOME": str(WORK / "home"), "PATH": "/usr/bin:/bin"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def ane_wait(p, cap, out):
    try:
        p.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        p.kill()
        raise SystemExit("ANE child over its time cap: possible hang; stopping")
    if p.returncode != 0:
        raise SystemExit(f"ANE child failed: {p.returncode}")
    return json.load(open(out + ".json"))["tasks"][0]["runs"]


def phase(name, fn):
    time.sleep(a.cooldown)
    t0 = time.time()
    rec = {"phase": name, "start": t0, **fn()}
    rec["end"] = time.time()
    phases.append(rec)
    summ = {k: v for k, v in rec.items() if not isinstance(v, list)}
    print(json.dumps(summ), flush=True)


def stats(runs, key="decode_tok_s"):
    xs = [r[key] for r in runs]
    return {"runs": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs),
            "min": min(xs), "max": max(xs)} if xs else {}


# warm both (first MLX run compiles kernels; skrun's first load reads Core ML's cache)
gpu_loop(1)
w = ane_start(5, str(WORK / "results" / "raw" / "companion_warm"))
ane_wait(w, 600, str(WORK / "results" / "raw" / "companion_warm"))

phase("idle", lambda: (time.sleep(a.seconds), {})[1])
phase("gpu_alone", lambda: {"gpu_runs": gpu_loop(a.seconds)})


def ane_alone():
    out = str(WORK / "results" / "raw" / "companion_ane_alone")
    return {"ane_runs": ane_wait(ane_start(a.seconds, out), a.seconds + 600, out)}


phase("ane_alone", ane_alone)


def both():
    out = str(WORK / "results" / "raw" / "companion_both")
    p = ane_start(a.seconds + 300, out)
    time.sleep(3)                       # let the companion load and start generating
    runs = gpu_loop(a.seconds)
    stop.touch()
    return {"gpu_runs": runs, "ane_runs": ane_wait(p, 900, out)}


phase("gpu_with_ane", both)
phase("gpu_alone_again", lambda: {"gpu_runs": gpu_loop(a.seconds)})
time.sleep(2)
power.terminate()

# power per phase from the battery controller's accumulators (mean mW over the window)
samples = [json.loads(l) for l in open(power_path) if '"error"' not in l]
report = {"primary": a.primary, "ane_model": a.ane, "seconds": a.seconds, "phases": []}
# The controller updates its accumulators about once a minute, each update
# covering the time since the one before; only update windows lying wholly
# inside a phase (polled 1 s apart, so with 1 s of slack) count for it.
ups = [s for i, s in enumerate(samples)
       if i == 0 or s["SystemPowerInAccumulatorCount"] != samples[i - 1]["SystemPowerInAccumulatorCount"]]
windows = [(p["t"], q["t"], q["AccumulatedSystemPowerIn"] - p["AccumulatedSystemPowerIn"],
            q["SystemPowerInAccumulatorCount"] - p["SystemPowerInAccumulatorCount"]) for p, q in zip(ups, ups[1:])]
report["power_windows"] = windows
for ph in phases:
    inside = [w for w in windows if w[0] >= ph["start"] - 1 and w[1] <= ph["end"] + 1]
    pw = sum(w[2] for w in inside) / sum(w[3] for w in inside) / 1000 if inside else None
    report["phases"].append({"phase": ph["phase"], "start": ph["start"], "end": ph["end"],
                             "seconds": round(ph["end"] - ph["start"], 1),
                             "system_power_W": pw, "power_windows_used": len(inside),
                             "gpu_decode": stats(ph.get("gpu_runs", [])),
                             "ane_decode": stats(ph.get("ane_runs", [])),
                             "ane_ttft_ms": stats(ph.get("ane_runs", []), "ttft_ms"),
                             "raw": {k: v for k, v in ph.items() if k.endswith("_runs")}})
json.dump(report, open(a.out, "w"), indent=1)
for p in report["phases"]:
    print(p["phase"], {k: v for k, v in p.items() if k not in ("raw", "phase")})
