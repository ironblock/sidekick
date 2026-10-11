"""Build docs/data/models.json, the results file behind the model
comparison page (docs/site/), from the parity suite's reports and the
benchmarks.

One record per (model, task, path, precision, chip, runtime). Every value
comes from an input; one nobody measured is null, never estimated.

Inputs:
- parity reports (`parity --json`), in the directories given with
  --reports, searched recursively. A report is read only if
  docs/data/runs.toml describes it (by file name): the report records the
  chip and results, runs.toml the runtime, OS, precision, date and whether
  the machine was idle.
- docs/data/carried.toml: Core ML results that so far exist only as
  docs/MODELS.md tables.
- docs/data/models.toml: each model's name, family, task, checkpoint and
  default served path.
- --bench: tools/bench_onnx.py, bench_coreml.py and bench_memory.py output
  (fixed-length latency, footprint, peak and size on disk), attached to the
  matching rows.
- --threads: thread-scaling runs (sidekickd over HTTP), attached as
  latency per thread count.

Usage:
    python tools/models_table.py --reports DIR [--reports DIR ...]
        [--bench DIR] [--threads DIR] [--out docs/data/models.json] [--check]

    --check: write nothing; fail if the output would differ from --out

Standard library only. The output is sorted and formatted the same way on
every run, so a regenerated file diffs cleanly.
"""

import argparse
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"

TASKS = ["embed", "classify", "rerank"]
PATHS = ["coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
PRECISIONS = ["fp32", "fp16", "fp16 weights", "int8"]
REPORT_PATH = {"ane": "coreml-ane", "gpu": "coreml-gpu", "cpu": "coreml-cpu", "onnx-cpu": "onnx-cpu"}

FIELDS = [
    "model", "name", "family", "task", "checkpoint", "path", "precision", "chip", "os", "runtime",
    "sidekick", "date", "served", "grade", "error_kind", "error", "worst_case", "stress", "mean",
    "drift", "rank_flips", "matryoshka", "ms", "ms_source", "ms_idle", "ms_tokens", "latency",
    "footprint_mb", "footprint_measured", "peak_mb", "peak_measured", "disk_mb", "disk_measured",
    "machine_mb", "machine_measured", "note", "source",
]


def load_toml(path):
    return tomllib.loads(path.read_text())


def finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def rounded(x, digits=4):
    """Round a measurement for publication: significant digits for small
    errors, fixed decimals otherwise."""
    if x is None:
        return None
    if x == 0:
        return 0
    if abs(x) < 1e-3:
        return float(f"{x:.3g}")
    return round(x, digits)


def blank(model, models, **kw):
    m = models.get(model)
    if m is None:
        raise SystemExit(f"model {model!r} isn't in docs/data/models.toml")
    row = {f: None for f in FIELDS}
    row.update(model=model, name=m["name"], family=m["family"], task=m["task"], checkpoint=m.get("source"))
    row.update(kw)
    return row


def served(row, models):
    """The path a build is served on by default: a Core ML build on its
    manifest's compute units on Apple silicon, an ONNX build on onnx-cpu in
    fp32."""
    if row["path"] == "onnx-cpu":
        return row["precision"] == "fp32"
    return row["chip"].startswith("Apple") and row["path"] == models[row["model"]]["served"]


def from_reports(dirs, runs, models):
    rows, used = [], []
    for d in dirs:
        for f in sorted(Path(d).rglob("*.json")):
            meta = runs["runs"].get(f.name)
            if meta is None:
                continue
            used.append(f.name)
            report = json.loads(f.read_text())
            chip = runs["chips"].get(report.get("chip"))
            if chip is None:
                raise SystemExit(f"{f.name}: chip {report.get('chip')!r} isn't in runs.toml [chips]")
            common = dict(chip=chip, os=meta.get("os"), runtime=meta.get("runtime"), sidekick=meta.get("sidekick"),
                          date=meta.get("date"), precision=meta.get("precision"), source=f.name)
            for m in report.get("models", []):
                for p in m["paths"]:
                    row = blank(m["id"], models, path=REPORT_PATH[p["path"]], **common)
                    worst = finite(p.get("worst"))
                    row.update(grade=p["letter"], error_kind="1-cos",
                               error=rounded(1 - worst) if worst is not None else None,
                               worst_case=p.get("worst_case"), stress=one_minus(p.get("stress")),
                               mean=one_minus(p.get("mean")), drift=rounded(finite(p.get("drift"))),
                               rank_flips=p.get("rank_flips"),
                               matryoshka={k: rounded(1 - v) for k, v in sorted(p["matryoshka"].items(), key=lambda kv: -int(kv[0]))}
                               if p.get("matryoshka") else None)
                    grading_ms(row, p, meta)
                    row["note"] = failures_note(p, m)
                    rows.append(row)
            for c in report.get("classifiers", []):
                for p in c["paths"]:
                    row = blank(c["id"], models, path=REPORT_PATH[p["path"]], **common)
                    row.update(grade=p["letter"], error_kind="dp", error=rounded(finite(p.get("worst_dp"))),
                               worst_case=p.get("worst_case"), stress=rounded(finite(p.get("stress_dp"))),
                               mean=rounded(finite(p.get("mean_dp"))), rank_flips=len(p.get("flips") or []))
                    grading_ms(row, p, meta)
                    row["note"] = failures_note(p, c)
                    rows.append(row)
    return rows, used


def one_minus(cos):
    cos = finite(cos)
    return rounded(1 - cos) if cos is not None else None


def grading_ms(row, p, meta):
    ms = finite(p.get("median_ms"))
    if ms is not None:
        row.update(ms=round(ms, 1), ms_source="grading: median per input over the corpus", ms_idle=meta.get("idle"))


def failures_note(p, m):
    notes = list(p.get("failures") or [])
    return "; ".join(f"Gate failed: {n}" for n in notes) or None


def from_carried(carried, models):
    d = carried["defaults"]
    rows = []
    for c in carried["row"]:
        row = blank(c["model"], models, path=c["path"], precision=c.get("precision", d["precision"]),
                    chip=d["chip"], os=d["os"], runtime=d["runtime"], date=d["date"], source=d["source"])
        kind = c.get("error", "cos")
        worst = c.get("worst")
        row.update(grade=c["grade"], error_kind="1-cos" if kind == "cos" else "dp",
                   error=None if worst is None else rounded(1 - worst if kind == "cos" else worst),
                   worst_case=c.get("worst_case"), drift=c.get("drift"), rank_flips=c.get("rank_flips"),
                   note=c.get("note"))
        if c.get("ms") is not None:
            row.update(ms=c["ms"], ms_source="grading: median per input (docs/MODELS.md table)", ms_idle=None,
                       ms_tokens=c.get("ms_tokens"))
        rows.append(row)
    return rows


def precision_of(model_file):
    if not model_file:
        return "fp32"
    if "fp16w" in model_file:
        return "fp16 weights"
    if "int8" in model_file:
        return "int8"
    return "fp32"


def index(rows):
    return {(r["model"], r["path"], r["precision"], r["chip"]): r for r in rows}


def machine_chip(m, chips):
    name = m.get("cpu") or m.get("Model name")
    return chips.get(name, name)


def attach_bench(rows, bench, runs):
    """Fixed-length latency and memory from tools/bench_*.py, on the rows of
    the same model, path, precision and chip."""
    idx = index(rows)
    chips = runs["chips"]
    bench = Path(bench)
    used = []
    for f in sorted(bench.glob("onnx-*.json")) + sorted(bench.glob("coreml-*.json")):
        r = json.loads(f.read_text())
        chip = machine_chip(r["machine"], chips)
        if f.name.startswith("onnx-"):
            model = model_of(f.name[len("onnx-"):-len(".json")])
            if model is None:
                continue
            key = (model, "onnx-cpu", precision_of(r.get("model_file")), chip)
            row = idx.get(key)
            if row is None:
                continue
            for c in r["configs"]:
                for run in c["runs"]:
                    add_latency(row, tokens=run["tokens"], batch=run["batch"], threads=c["threads"] or None,
                                ms=run["median_ms"], runtime=f"ONNX Runtime {c['ort']}, from Python",
                                load_avg=c.get("load_avg_before"), source=f.name)
        else:
            model = model_of(f.name[len("coreml-"):-len(".json")])
            for p in r["paths"]:
                row = idx.get((model, f"coreml-{p['units']}", "fp16", chip))
                if row is None:
                    continue
                for run in p["runs"]:
                    add_latency(row, tokens=run["tokens"], batch=1, threads=None, ms=run["median_ms"],
                                runtime=f"Core ML, bucket {run['bucket']}, from Python",
                                load_avg=p.get("load_avg_before"), source=f.name)
        used.append(f.name)
    memory = {}
    for f in sorted((bench / "memory").glob("*.json")):
        r = json.loads(f.read_text())
        precision = precision_of(r.get("model_file")) if r["path"] == "onnx-cpu" else "fp16"
        if r["path"] == "onnx-cpu" and r.get("model_file") is None and "published" in f.name:
            continue
        key = (r["model"], "onnx-cpu" if r["path"] == "onnx-cpu" else f"coreml-{r['path']}", precision,
               machine_chip(r["machine"], chips))
        memory.setdefault(key, []).append((f.name, r))
    allow = set(runs.get("memory", {}).get("machine_wide", []))
    for key, runs_ in sorted(memory.items()):
        row = idx.get(key)
        if row is None:
            continue
        names = [n for n, _ in runs_]
        foot = statistics.median((r["served"]["phys_footprint"] - r["before"]["phys_footprint"]) / 2**20 for _, r in runs_)
        peak = statistics.median((r["served"]["peak_phys_footprint"] - r["before"]["phys_footprint"]) / 2**20 for _, r in runs_)
        who = "a Python process running ONNX Runtime" if key[1] == "onnx-cpu" else "a Python process running Core ML"
        n = f"median of {len(runs_)} runs" if len(runs_) > 1 else "one run"
        row.update(footprint_mb=round(foot), peak_mb=round(peak),
                   footprint_measured=f"phys_footprint change of {who}, after load and one prediction per length; {n}",
                   peak_measured=f"lifetime peak phys_footprint of {who} above its pre-load footprint; {n}",
                   disk_mb=round(runs_[0][1]["artifact_bytes"] / 2**20),
                   disk_measured=("the ONNX graph and its external data" if key[1] == "onnx-cpu"
                                  else "every bucket of the Core ML install, shared by its three paths"))
        tag = f"{key[1]}:{key[0]}"
        if tag in allow:
            machine = statistics.median((r["served"]["vm"]["in_use"] - r["before"]["vm"]["in_use"]) / 2**20 for _, r in runs_)
            row.update(machine_mb=round(machine),
                       machine_measured=f"change in the whole machine's memory in use (vm_stat), {n}: "
                                        "includes what system services hold for the Neural Engine")
        row["source"] = ", ".join([row["source"]] + names)
        used.extend(names)
    # An ONNX export is the same files on every chip: its measured size
    # applies to that model and precision everywhere.
    sizes = {(r["model"], r["precision"]): r["disk_mb"] for r in rows
             if r["path"] == "onnx-cpu" and r["disk_mb"] is not None}
    for r in rows:
        if r["path"] == "onnx-cpu" and r["disk_mb"] is None and (r["model"], r["precision"]) in sizes:
            r["disk_mb"] = sizes[(r["model"], r["precision"])]
            r["disk_measured"] = "the ONNX graph and its external data, the same files on every chip"
    return used


def model_of(stem):
    aliases = {"qwen3-fp16w": "qwen3-embedding-0.6b", "qwen3-int8": "qwen3-embedding-0.6b",
               "qwen3-published-int8": None}
    return aliases.get(stem, stem)


def attach_threads(rows, threads, runs):
    """Latency by thread count from sidekickd over HTTP (one file per
    machine), on the onnx-cpu fp32 rows of that chip."""
    idx = index(rows)
    used = []
    for f in sorted(Path(threads).glob("*.json")):
        if "-check" in f.stem:
            continue          # a re-run of a few settings, to confirm the main run
        r = json.loads(f.read_text())
        if "runs" not in r or "machine" not in r:
            continue
        chip = machine_chip(r["machine"], runs["chips"])
        for run in r["runs"]:
            row = idx.get((run["model"], "onnx-cpu", "fp32", chip))
            if row is None:
                continue
            for length in ("short", "long"):
                add_latency(row, tokens=run[length]["tokens"], batch=1, threads=run["threads"],
                            ms=run[length]["median_ms"], runtime="sidekickd over HTTP",
                            load_avg=run.get("load_before"), source=f.name)
        used.append(f.name)
    return used


def add_latency(row, *, tokens, batch, threads, ms, runtime, load_avg, source):
    entry = {"tokens": tokens, "batch": batch, "threads": threads, "ms": round(ms, 1), "how": runtime,
             "load_avg": round(load_avg[0], 1) if load_avg else None, "source": source}
    row["latency"] = sorted((row["latency"] or []) + [entry],
                            key=lambda e: (e["how"], e["threads"] or 0, e["batch"], e["tokens"]))


def order(r):
    return (TASKS.index(r["task"]), r["name"].lower(), r["chip"], PATHS.index(r["path"]),
            PRECISIONS.index(r["precision"]), r["runtime"] or "")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reports", action="append", default=[], help="a directory of parity --json reports")
    p.add_argument("--bench", help="tools/bench_*.py output (with a memory/ subdirectory)")
    p.add_argument("--threads", help="thread-scaling runs (sidekickd over HTTP)")
    p.add_argument("--out", type=Path, default=DATA / "models.json")
    p.add_argument("--check", action="store_true", help="fail if --out isn't what these inputs build")
    args = p.parse_args()

    models = load_toml(DATA / "models.toml")["models"]
    runs = load_toml(DATA / "runs.toml")
    rows, used = from_reports(args.reports, runs, models)
    n_reports = len(rows)
    carried = from_carried(load_toml(DATA / "carried.toml"), models)
    have = {(r["model"], r["path"], r["precision"], r["chip"]) for r in rows}
    rows += [c for c in carried if (c["model"], c["path"], c["precision"], c["chip"]) not in have]
    if args.bench:
        used += attach_bench(rows, args.bench, runs)
    if args.threads:
        used += attach_threads(rows, args.threads, runs)
    missing = sorted(set(runs["runs"]) - set(used))
    if missing:
        print(f"note: runs.toml describes reports not found here: {', '.join(missing)}", file=sys.stderr)
    for r in rows:
        r["served"] = served(r, models)
    rows.sort(key=order)
    keys = [(r["model"], r["path"], r["precision"], r["chip"], r["runtime"]) for r in rows]
    if len(keys) != len(set(keys)):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise SystemExit(f"duplicate rows: {dup}")
    dates = sorted(d for d in (r["date"] for r in rows) if d)
    cargo = tomllib.loads((REPO / "Cargo.toml").read_text())
    out = {"sidekick": cargo["workspace"]["package"]["version"], "as_of": dates[-1] if dates else None,
           "fields": FIELDS, "rows": rows}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            raise SystemExit(f"{args.out} is out of date: rerun tools/models_table.py with the same inputs")
        print(f"{args.out} is up to date ({len(rows)} rows)")
        return
    args.out.write_text(text)
    print(f"{len(rows)} rows ({n_reports} from reports, {len(rows) - n_reports} carried from MODELS.md) -> {args.out}")


if __name__ == "__main__":
    main()
