"""What a companion model costs the model it runs beside.

sidekick can serve a small model (a classifier) next to a larger one that
the GPU serves, a local LLM say. Served on the GPU, the companion competes
with the larger model for it; served on the ANE (a chunked artifact, D37,
with the operator choosing its compute units, D38), it leaves the GPU alone
but answers more slowly. This measures the trade on one machine, in three
phases of equal length:

1. idle: the primary alone;
2. gpu: the companion at a fixed request rate on cpu_and_gpu;
3. ane: the same on cpu_and_ne.

In each phase it runs the primary's own benchmark command back to back and
reads its tokens per second from the output; the companion runs as
crates/sidekick-embed's `chain_timing --rate`, which sends requests at a
fixed rate and reports their latency. Every phase's start and end are
printed and written to the report, so a power trace taken across the whole
run (`sudo powermetrics`, see below) can be split by phase with
`--power`.

Usage:
    python tools/companion_bench.py --primary-cmd '<command>' --tps-regex '<regex>'
        --models-dir <dir> --model <id> [--tokens 1024] [--rate 2] [--seconds 120]
        [--chain-timing target/release/examples/chain_timing] [--out report.json]
        [--power power.txt]

    --primary-cmd   one run of the primary's decode benchmark, e.g.
                    llama.cpp's `llama-bench -m <model.gguf> -p 0 -n 128 -r 1`
                    or mlx-lm's `mlx_lm.generate --model <dir> --prompt hi
                    --max-tokens 256`
    --tps-regex     a regex whose first group is the decode tokens/s in that
                    command's output, e.g. 'tg128 .*?([0-9.]+) ±' (llama-bench)
                    or 'Generation: .*?([0-9.]+) tokens-per-sec' (mlx-lm)
    --models-dir    a sidekick models directory holding the companion
    --model         the companion's classifier id (an agentjev-format model,
                    which chain_timing drives)
    --tokens        the companion's request length
    --rate          companion requests per second
    --seconds       length of each phase
    --power         a powermetrics log covering the run: also report each
                    phase's mean CPU, GPU and ANE power

Power, in a second terminal before starting (it needs sudo, so the person
running the benchmark starts it):

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -o power.txt

Stop it with Ctrl-C after the run, then pass `--power power.txt`, or run
this script again with `--report report.json --power power.txt` to add it.

Build chain_timing first:
    cargo build --release -p sidekick-embed --features coreml --example chain_timing

Warm up first: the companion's first load on the ANE compiles each chunk
(minutes for a large bucket); run `chain_timing` once for each compute unit
before measuring, so the phases don't include it.
"""

import argparse
import datetime
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def primary_runs(cmd, regex, seconds):
    """Run the primary's benchmark back to back for `seconds`; its tokens/s
    per run."""
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        m = re.search(regex, r.stdout + r.stderr)
        if r.returncode != 0 or not m:
            raise SystemExit(f"primary command failed or printed no match for {regex!r}:\n{r.stdout[-2000:]}"
                             f"\n{r.stderr[-2000:]}")
        out.append(float(m.group(1)))
    return out


def companion(args, units):
    """chain_timing at a fixed rate, for the phase's length, in the
    background."""
    cmd = [str(args.chain_timing), "--models-dir", str(args.models_dir), "--model", args.model, "--units", units,
           "--tokens", str(args.tokens), "--rate", str(args.rate), "--seconds", str(args.seconds)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def phase(args, name, units):
    start = datetime.datetime.now().astimezone()
    print(f"phase {name}: start {start.isoformat(timespec='seconds')}", flush=True)
    proc = companion(args, units) if units else None
    tps = primary_runs(args.primary_cmd, args.tps_regex, args.seconds)
    companion_out = proc.communicate()[0] if proc else ""
    end = datetime.datetime.now().astimezone()
    result = {"phase": name, "units": units, "start": start.isoformat(), "end": end.isoformat(),
              "primary_tps": tps, "primary_tps_median": statistics.median(tps)}
    m = re.search(r"(\d+) predictions in ([0-9.]+)s .*?p50 ([0-9.]+) ms, p90 ([0-9.]+) ms, p99 ([0-9.]+) ms",
                  companion_out)
    if units and not m:
        raise SystemExit(f"companion output not understood:\n{companion_out}")
    if m:
        n, wall, p50, p90, p99 = m.groups()
        result["companion"] = {"predictions": int(n), "rate": int(n) / float(wall), "p50_ms": float(p50),
                               "p90_ms": float(p90), "p99_ms": float(p99)}
    print(f"phase {name}: primary {result['primary_tps_median']:.1f} tok/s (median of {len(tps)})"
          + (f"; companion {result['companion']['rate']:.2f}/s, p50 {result['companion']['p50_ms']:.0f} ms, "
             f"p99 {result['companion']['p99_ms']:.0f} ms" if m else ""), flush=True)
    return result


_POWER = re.compile(r"^(CPU|GPU|ANE) Power: (\d+) mW", re.M)


def power_by_phase(path, phases):
    """Mean CPU, GPU and ANE power (mW) per phase from a powermetrics log."""
    text = Path(path).read_text(errors="replace")
    samples = []
    for block in text.split("*** Sampled system activity")[1:]:
        when = re.match(r" \((.+?)\)", block)
        if not when:
            continue
        stamp = datetime.datetime.strptime(when.group(1).split(" (")[0].strip(), "%a %b %d %H:%M:%S %Y %z")
        samples.append((stamp, {k: int(v) for k, v in _POWER.findall(block)}))
    for p in phases:
        start, end = datetime.datetime.fromisoformat(p["start"]), datetime.datetime.fromisoformat(p["end"])
        inside = [s for t, s in samples if start <= t <= end]
        p["power_mw"] = {k: statistics.mean(s.get(k, 0) for s in inside) for k in ("CPU", "GPU", "ANE")} if inside else None
        p["power_samples"] = len(inside)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--primary-cmd")
    ap.add_argument("--tps-regex")
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--chain-timing", type=Path, default=REPO / "target/release/examples/chain_timing")
    ap.add_argument("--out", type=Path, default=Path("companion_report.json"))
    ap.add_argument("--report", type=Path, help="add --power to an existing report instead of measuring")
    ap.add_argument("--power", type=Path)
    args = ap.parse_args()
    if args.report:
        report = json.loads(args.report.read_text())
    else:
        for need in ("primary_cmd", "tps_regex", "models_dir", "model"):
            if getattr(args, need) is None:
                ap.error(f"--{need.replace('_', '-')} is required")
        report = {"tokens": args.tokens, "rate": args.rate, "seconds": args.seconds, "model": args.model,
                  "phases": [phase(args, "idle", None), phase(args, "gpu", "cpu_and_gpu"),
                             phase(args, "ane", "cpu_and_ne")]}
    if args.power:
        power_by_phase(args.power, report["phases"])
        for p in report["phases"]:
            w = p.get("power_mw")
            print(f"phase {p['phase']}: " + (", ".join(f"{k} {v / 1000:.2f} W" for k, v in w.items()) if w
                                             else "no power samples in this phase"))
    out = args.report or args.out
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report -> {out}")


if __name__ == "__main__":
    sys.exit(main())
