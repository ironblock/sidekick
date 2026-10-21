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
  the machine was idle. An embedder graded on the long-input corpus
  (fixtures/parity/corpus_long.toml, the report's `long` section) gets
  that result as a note on the same row; the row's grade is the main
  corpus's, and a gate failed on long inputs fails it.
- docs/data/carried.toml: Core ML results that so far exist only as
  docs/MODELS.md tables.
- fixtures/parity/expectations.toml `[grades]`: the letter scales, which
  the parity suite also reads. Every letter here is computed from a row's
  numbers with them; a report's own letter must agree, and `--check`
  (with `--docs` too) fails on any row whose stored letter doesn't.
- docs/data/models.toml: each model's name, family, task, checkpoint and
  default served path.
- --bench: tools/bench_onnx.py, bench_coreml.py and bench_memory.py output
  (fixed-length latency, footprint, peak and size on disk), attached to the
  matching rows.
- --threads: thread-scaling runs (sidekickd over HTTP), attached as
  latency per thread count.
- each build's committed manifest (examples/manifests/, with an onnx.toml
  beside it for an ONNX build that differs, and examples/classifiers/):
  its `max_seq_len`, on every row of that build, and an embedder's
  `[prefixes]`, for docs/MODELS.md's prompts table.
- benchmark scores, published beside the rows as the results file's
  `benchmarks`: docs/data/benchmarks.mteb.json (MTEB, collected by
  tools/fetch_benchmarks.py) and docs/data/benchmarks.toml (curated,
  mostly self-reported). Every score must name its benchmark and version,
  category, metric, source (pinned to a revision) and the date it was
  read; a category mean must cover all of its tasks. `--check` refuses one
  that doesn't.
- docs/data/mteb_transfer.json (tools/mteb_transfer.py): MTEB tasks run
  through sidekick's builds beside their published scores, published as
  `benchmarks.transfer`. `--check` refuses a run without its provenance
  (revision, path, cap, build, chip, OS, mteb version, date), published
  scores from another results revision than benchmarks.mteb.json's, and a
  score without a run, its task's metric, or its count of inputs over the
  cap.

It also writes the data tables in docs/MODELS.md from the results file
(those between `<!-- generated: NAME -->` markers), and checks the
hand-written classifier and reranker tables' grades against it, so the
document and the page can't drift apart. To change a published number,
change its input (a report, runs.toml, carried.toml or a benchmarks file)
and rerun this.

Usage:
    python tools/models_table.py --reports DIR [--reports DIR ...]
        [--bench DIR] [--threads DIR] [--out docs/data/models.json] [--check]
    python tools/models_table.py --docs [--check]

    --check: write nothing; fail if the output would differ from --out
             or from docs/MODELS.md
    --docs:  regenerate docs/MODELS.md's tables, and the results file's
             benchmark scores, from in-repo inputs, reading no reports
             (what CI checks)

Standard library only. The output is sorted and formatted the same way on
every run, so a regenerated file diffs cleanly.
"""

import argparse
import datetime
import decimal
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"
EXPECTATIONS = REPO / "fixtures" / "parity" / "expectations.toml"

TASKS = ["embed", "classify", "rerank"]
PATHS = ["coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
PRECISIONS = ["fp32", "fp16", "fp16 weights", "int8"]
REPORT_PATH = {"ane": "coreml-ane", "gpu": "coreml-gpu", "cpu": "coreml-cpu", "onnx-cpu": "onnx-cpu"}

FIELDS = [
    "model", "name", "family", "task", "checkpoint", "license_note", "path", "precision", "chip", "os", "runtime",
    "sidekick", "threads", "date", "served", "grade", "error_kind", "error", "error_text", "ratio",
    "gate_failed", "worst_case", "stress", "mean",
    "drift", "bias", "rank_flips", "matryoshka", "ms", "ms_source", "ms_idle", "ms_tokens", "latency",
    "footprint_mb", "footprint_range_mb", "footprint_measured", "peak_mb", "peak_measured", "disk_mb", "disk_measured",
    "machine_mb", "machine_measured", "max_seq_len", "note", "source",
]


def load_toml(path):
    return tomllib.loads(path.read_text())


def finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def rounded(x, digits=4, keeps=None):
    """Round a measurement for publication: significant digits for small
    errors, fixed decimals otherwise. With `keeps`, more digits until
    keeps(rounded value) holds, so rounding never moves a value across a
    grade boundary."""
    if x is None:
        return None
    if x == 0:
        return 0
    v = float(f"{x:.3g}") if abs(x) < 1e-3 else round(x, digits)
    sig = 4
    while keeps is not None and not keeps(v) and sig <= 17:
        v, sig = float(f"{x:.{sig}g}"), sig + 1
    return v


# --- grades ------------------------------------------------------------------
#
# One definition of the letters: [grades] in fixtures/parity/expectations.toml,
# which the parity suite (crates/sidekick-embed/examples/parity) reads too.

def load_grades():
    return load_toml(EXPECTATIONS)["grades"]


def scale_letter(scale, v, higher_is_better):
    """A, B or C by the first bound `v` meets (inclusive), else D."""
    for letter in "ABC":
        if (v >= scale[letter]) if higher_is_better else (v <= scale[letter]):
            return letter
    return "D"


def absolute_letter(grades, kind, error):
    """The letter for a worst error alone: 1 − cosine (embedders) or |Δp|."""
    if kind == "1-cos":
        return scale_letter(grades["cosine"], 1 - error, True)
    return scale_letter(grades["dp"], error, False)


def grade_of(row, grades):
    """The letter a row's numbers earn, by the parity suite's rules: F for a
    failed gate or a non-finite error; embedders by worst cosine;
    classifiers by the better of worst |Δp| and the ratio to their fp16
    ceiling, with a graded flip capping the grade at C."""
    if row["gate_failed"] or row["error"] is None:
        return "F"
    best = absolute_letter(grades, row["error_kind"], row["error"])
    if row["error_kind"] == "1-cos":
        return best
    if row["ratio"] is not None:
        best = min(best, scale_letter(grades["ratio"], row["ratio"], False))
    return max(best, "C") if row["rank_flips"] else best


def publish_error(grades, kind, error):
    """`error` rounded for publication, on the same side of every bound."""
    if error is None:
        return None
    exact = absolute_letter(grades, kind, error)
    return rounded(error, keeps=lambda v: absolute_letter(grades, kind, v) == exact)


def publish_ratio(grades, ratio):
    if ratio is None:
        return None
    exact = scale_letter(grades["ratio"], ratio, False)
    v, digits = round(ratio, 2), 3
    while scale_letter(grades["ratio"], v, False) != exact and digits <= 12:
        v, digits = round(ratio, digits), digits + 1
    return v


def error_text(row):
    """The one way a row's error is written, in the tables and on the page:
    a cosine for embedders (1 − cosine as "1 − 2e-12" when it rounds to 1),
    |Δp| for classifiers."""
    e = row["error"]
    if e is None:
        return ""
    if row["error_kind"] == "1-cos":
        return cosine(e)
    if e == 0:
        return "0"
    if e < 1e-2:
        mantissa, exponent = f"{e:.3g}".split("e") if "e" in f"{e:.3g}" else (None, None)
        if mantissa is None:
            m, x = f"{e:.2e}".split("e")
            return f"{float(m):g}e{int(x)}"
        return f"{float(mantissa):g}e{int(exponent)}"
    return f"{e:.3g}"


def ratio_note(row, grades):
    if row["ratio"] is None:
        return None
    return (f"p99 {row['ratio']:g}× its fp16 ceiling "
            f"({scale_letter(grades['ratio'], row['ratio'], False)} on that scale)")


def finish(row, grades, reported=None):
    """Fill a row's letter and error text from its numbers; a report's own
    letter must be the same."""
    row["grade"] = grade_of(row, grades)
    if reported is not None and reported != row["grade"]:
        raise SystemExit(f"{row['source']}: {row['model']} {row['path']}: the report's grade {reported} isn't what "
                         f"its numbers earn under [grades] ({row['grade']}); were they graded with other scales?")
    row["error_text"] = error_text(row)
    row["note"] = join_notes(ratio_note(row, grades), row["note"])


def blank(model, models, **kw):
    m = models.get(model)
    if m is None:
        raise SystemExit(f"model {model!r} isn't in docs/data/models.toml")
    row = {f: None for f in FIELDS}
    row.update(model=model, name=m["name"], family=m["family"], task=m["task"], checkpoint=m.get("source"),
               license_note=m.get("license_note"))
    row.update(kw)
    return row


def served(row, models):
    """The path a build is served on by default: a Core ML build on its
    manifest's compute units on Apple silicon, an ONNX build on onnx-cpu in
    fp32."""
    if row["path"] == "onnx-cpu":
        return row["precision"] == "fp32"
    return row["chip"].startswith("Apple") and row["path"] == models[row["model"]]["served"]


def from_reports(dirs, runs, models, grades):
    rows, used, seen = [], [], {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*.json")):
            meta = runs["runs"].get(f.name)
            if meta is None:
                continue
            # runs.toml keys reports by file name, so a copy of one in two
            # places is read once; two different reports by one name are an
            # error.
            text = f.read_text()
            if f.name in seen:
                if seen[f.name] != text:
                    raise SystemExit(f"two different reports named {f.name}")
                continue
            seen[f.name] = text
            used.append(f.name)
            report = json.loads(text)
            chip = runs["chips"].get(report.get("chip"))
            if chip is None:
                raise SystemExit(f"{f.name}: chip {report.get('chip')!r} isn't in runs.toml [chips]")
            common = dict(chip=chip, os=meta.get("os"), runtime=meta.get("runtime"), sidekick=meta.get("sidekick"),
                          threads=meta.get("threads"), date=meta.get("date"), precision=meta.get("precision"),
                          source=f.name)
            notes = runs.get("notes", {})
            for m in report.get("models", []):
                for p in m["paths"]:
                    row = blank(m["id"], models, path=REPORT_PATH[p["path"]], **common)
                    worst = finite(p.get("worst"))
                    row.update(error_kind="1-cos", gate_failed=bool(p.get("failures")),
                               error=publish_error(grades, "1-cos", 1 - worst) if worst is not None else None,
                               worst_case=p.get("worst_case"), stress=one_minus(p.get("stress")),
                               mean=one_minus(p.get("mean")), drift=rounded(finite(p.get("drift"))),
                               rank_flips=p.get("rank_flips"),
                               matryoshka={k: rounded(1 - v) for k, v in sorted(p["matryoshka"].items(), key=lambda kv: -int(kv[0]))}
                               if p.get("matryoshka") else None)
                    grading_ms(row, p, meta)
                    long, long_failed = long_note(m, p["path"], grades, f.name)
                    row["gate_failed"] = row["gate_failed"] or long_failed
                    row["note"] = join_notes(notes.get(f"{f.name}:{m['id']}:{row['path']}"), long,
                                             failures_note(p, m))
                    row["threads"] = runs.get("threads", {}).get(f"{f.name}:{m['id']}", row["threads"])
                    # a gate failed on long inputs fails the row (F), whatever the main corpus's letter
                    finish(row, grades, None if long_failed else p["letter"])
                    rows.append(row)
            for c in report.get("classifiers", []):
                for p in c["paths"]:
                    row = blank(c["id"], models, path=REPORT_PATH[p["path"]], **common)
                    # a path the runtime won't serve keeps its failed gates under "reported"
                    row.update(error_kind="dp", gate_failed=bool(p.get("failures") or p.get("reported")),
                               error=publish_error(grades, "dp", finite(p.get("worst_dp"))),
                               ratio=publish_ratio(grades, finite(p.get("ratio"))),
                               worst_case=p.get("worst_case"), stress=rounded(finite(p.get("stress_dp"))),
                               mean=rounded(finite(p.get("mean_dp"))), rank_flips=len(p.get("flips") or []))
                    grading_ms(row, p, meta)
                    row["note"] = join_notes(notes.get(f"{f.name}:{c['id']}:{row['path']}"), failures_note(p, c))
                    finish(row, grades, p["letter"])
                    rows.append(row)
    return rows, used


def long_note(m, path, grades, source):
    """An embedder's result on the long-input corpus (the report's `long`
    section, fixtures/parity/corpus_long.toml) for one path, as a note, and
    whether it failed a gate. The row's own grade stays the main corpus's,
    which every model is graded on; a failed gate here fails the row."""
    p = next((g for g in (m.get("long") or {}).get("paths", []) if g["path"] == path), None)
    if p is None:
        return None, False
    tokens = [c["tokens"] for c in m["long"]["cases"]]
    worst = finite(p.get("worst"))
    failed = bool(p.get("failures"))
    error = publish_error(grades, "1-cos", 1 - worst) if worst is not None else None
    letter = "F" if failed or error is None else absolute_letter(grades, "1-cos", error)
    if letter != p["letter"]:
        raise SystemExit(f"{source}: {m['id']} {path}: the long corpus's grade {p['letter']} isn't what its "
                         f"numbers earn under [grades] ({letter})")
    text = (f"Long inputs ({len(tokens)} cases of {min(tokens):,}–{max(tokens):,} tokens, "
            f"fixtures/parity/corpus_long.toml): {letter}, worst cosine {cosine(error) if error is not None else 'non-finite'}")
    for f in p.get("failures") or []:
        text += f"; gate failed: {f}"
    return text, failed


def one_minus(cos):
    cos = finite(cos)
    return rounded(1 - cos) if cos is not None else None


def grading_ms(row, p, meta):
    ms = finite(p.get("median_ms"))
    if ms is not None:
        row.update(ms=round(ms, 1), ms_source="grading: median per input over the corpus", ms_idle=meta.get("idle"))


def join_notes(*notes):
    return ". ".join(n for n in notes if n) or None


def failures_note(p, m):
    notes = list(p.get("failures") or [])
    return "; ".join(f"Gate failed: {n}" for n in notes) or None


def from_carried(carried, models, grades):
    d = carried["defaults"]
    rows = []
    for c in carried["row"]:
        row = blank(c["model"], models, path=c["path"], precision=c.get("precision", d["precision"]),
                    chip=d["chip"], os=d["os"], runtime=d["runtime"], date=d["date"], source=d["source"])
        kind = c.get("error", "cos")
        worst = c.get("worst")
        if "grade" in c:
            raise SystemExit(f"carried.toml: {c['model']} {c['path']} stores a grade; give its numbers instead")
        row.update(error_kind="1-cos" if kind == "cos" else "dp", gate_failed=False,
                   error=None if worst is None else rounded(1 - worst if kind == "cos" else worst),
                   ratio=c.get("ratio"), worst_case=c.get("worst_case"), drift=c.get("drift"),
                   bias=c.get("bias"), rank_flips=c.get("rank_flips"), note=c.get("note"))
        finish(row, grades)
        if c["model"] in carried.get("disk", {}):
            row.update(disk_mb=carried["disk"][c["model"]],
                       disk_measured="every bucket of the Core ML install, shared by its three paths "
                                     "(measured on the installed directory)")
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
        foots = [(r["served"]["phys_footprint"] - r["before"]["phys_footprint"]) / 2**20 for _, r in runs_]
        foot = statistics.median(foots)
        peak = statistics.median((r["served"]["peak_phys_footprint"] - r["before"]["phys_footprint"]) / 2**20 for _, r in runs_)
        who = "a Python process running ONNX Runtime" if key[1] == "onnx-cpu" else "a Python process running Core ML"
        n = f"median of {len(runs_)} runs" if len(runs_) > 1 else "one run"
        row.update(footprint_mb=round(foot), peak_mb=round(peak),
                   footprint_range_mb=[round(min(foots)), round(max(foots))] if len(foots) > 1 else None,
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


# --- build caps --------------------------------------------------------------
#
# Each build's max_seq_len, from its committed manifest: the longest input it
# takes. Benchmarks ran at the model's own context, which is often longer.

MANIFESTS = REPO / "examples" / "manifests"
CLASSIFIERS = REPO / "examples" / "classifiers"


def build_caps(models):
    """{model: {"coreml": n, "onnx": n}} from examples/manifests/<id>/
    (onnx.toml overrides the ONNX build's) or examples/classifiers/<id>/."""
    caps = {}
    for mid in models:
        embed, classify = MANIFESTS / mid / "manifest.toml", CLASSIFIERS / mid / "classifier.toml"
        if embed.exists():
            n = load_toml(embed).get("max_seq_len")
            sidecar = embed.with_name("onnx.toml")
            onnx = load_toml(sidecar).get("max_seq_len", n) if sidecar.exists() else n
            caps[mid] = {"coreml": n, "onnx": onnx}
        elif classify.exists():
            caps[mid] = {"coreml": load_toml(classify).get("max_seq_len"), "onnx": None}
    return caps


def cap_of(row, caps):
    c = caps.get(row["model"])
    if c is None:
        return None
    return c["onnx"] if row["path"] == "onnx-cpu" else c["coreml"]


# --- benchmark scores ---------------------------------------------------------
#
# Published as the results file's `benchmarks`, one record per (model,
# benchmark, version, category, metric), from two inputs:
# - docs/data/benchmarks.mteb.json, written by tools/fetch_benchmarks.py from
#   the MTEB results repository at a pinned commit: per (model, benchmark),
#   each category's mean over its tasks, null unless every task is present;
# - docs/data/benchmarks.toml, curated by hand: what a model's card or its
#   authors report.
# Values are published on a 0–100 scale. Nothing is averaged here: a mean is
# the benchmark's own (over its tasks), never one across benchmarks.

BENCH_TOML = DATA / "benchmarks.toml"
BENCH_MTEB = DATA / "benchmarks.mteb.json"
MEAN = "Mean (task)"
# Category order in the tables; others follow alphabetically.
CATEGORIES = [MEAN, "Retrieval", "Classification", "Clustering", "PairClassification", "Reranking", "STS",
              "Summarization", "BitextMining", "MultilabelClassification", "InstructionReranking"]
SOURCE_KINDS = {
    "mteb": "Run with the mteb package, from the MTEB results repository",
    "mixed": "Run with the mteb package, from the MTEB results repository, with some tasks imported from the "
             "model card (self-reported)",
    "self-reported": "Reported by the model's authors",
    "independent": "Measured by a third party",
}
CURATED_KINDS = ("self-reported", "independent")
REQUIRED = ("model", "benchmark", "version", "category", "metric", "value", "scale", "source_kind", "source_url",
            "read_on")
OPTIONAL = ("trained_on", "note")
NUMBER = re.compile(r"-?\d+(\.\d+)?$")
SHA = r"[0-9a-f]{40}"
# A model card, dataset or repository file, pinned to a revision or commit.
PINNED = [
    re.compile(rf"^https://huggingface\.co/(datasets/|spaces/)?[\w.-]+(/[\w.-]+)?/(blob|raw|resolve|tree)/{SHA}(/|$)"),
    re.compile(rf"^https://github\.com/[\w.-]+/[\w.-]+/(blob|tree|commit)/{SHA}(/|$)"),
    re.compile(rf"^https://raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/{SHA}/"),
]
# A complete mean must agree with its categories to the file's rounding.
MEAN_TOLERANCE = 1e-5


def pinned(url):
    """Whether a source URL names a fixed revision: a Hugging Face or
    GitHub URL at a commit. Any other page can change under the number, so
    it's refused too; cite the repository file it's built from instead."""
    return isinstance(url, str) and any(p.match(url) for p in PINNED)


def points_text(value, scale):
    """A curated value, a string as the source writes it ("0.860", "74.30"),
    on the 0–100 scale with the same precision ("86.0", "74.30")."""
    d = decimal.Decimal(value)
    return format(d.scaleb(2) if scale == "fraction" else d, "f")


def check_value(where, value, category):
    lo = -100 if category == "InstructionReranking" else 0      # p-MRR runs from -100 to 100
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not lo <= value <= 100:
        return [f"{where}: value {value!r} isn't a number from {lo} to 100"]
    return []


def check_date(where, value):
    """A read_on: an ISO date, not in the future."""
    try:
        day = datetime.date.fromisoformat(str(value))
    except ValueError:
        return [f"{where}: read_on {value!r} isn't a YYYY-MM-DD date"]
    return [f"{where}: read_on {value} is in the future"] if day > datetime.date.today() else []


def blank_text(v):
    return v is None or (isinstance(v, str) and not v.strip())


def curated_problems(entries, models):
    """docs/data/benchmarks.toml's entries: what --check refuses."""
    problems = []
    for i, e in enumerate(entries):
        where = f"benchmarks.toml entry {i + 1} ({e.get('model')}, {e.get('benchmark')}, {e.get('category')})"
        missing = [k for k in REQUIRED if blank_text(e.get(k)) or e.get(k) == []]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        ids = e["model"] if isinstance(e["model"], list) else [e["model"]]
        for m in ids:
            if not isinstance(m, str) or m not in models:
                problems.append(f"{where}: model {m!r} isn't in docs/data/models.toml")
        texts = [k for k in REQUIRED if k not in ("model", "read_on") and not isinstance(e[k], str)]
        if texts:
            problems.append(f"{where}: {', '.join(texts)} must be {'a string' if len(texts) == 1 else 'strings'} "
                            "(a value as the source writes it, quoted)")
            continue
        if e["scale"] not in ("percent", "fraction"):
            problems.append(f"{where}: scale {e['scale']!r} isn't percent or fraction")
        elif not NUMBER.match(e["value"]):
            problems.append(f"{where}: value {e['value']!r} isn't a decimal number")
        else:
            problems += check_value(where, float(points_text(e["value"], e["scale"])), e["category"])
        if e["source_kind"] not in CURATED_KINDS:
            problems.append(f"{where}: source_kind {e['source_kind']!r} isn't one of {', '.join(CURATED_KINDS)}")
        if not pinned(e["source_url"]):
            problems.append(f"{where}: source_url {e['source_url']} isn't pinned to a revision or commit")
        problems += check_date(where, e["read_on"])
        if "trained_on" in e and not isinstance(e["trained_on"], bool):
            problems.append(f"{where}: trained_on {e['trained_on']!r} isn't true or false")
        if "note" in e and (not isinstance(e["note"], str) or not e["note"].strip()):
            problems.append(f"{where}: note is empty or not a string")
        unknown = set(e) - set(REQUIRED) - set(OPTIONAL)
        if unknown:
            problems.append(f"{where}: unknown fields {', '.join(sorted(unknown))}")
    return problems


REVISION_MATCHES = ("exact", "same-weights", "unverified", "different", "model card only")
CONTEXT_SOURCES = ("mteb loader", "sentence-transformers config", "advertised")


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def mteb_type_problems(e):
    """An MTEB entry's fields of the wrong JSON type, or outside their
    closed sets."""
    out = []
    for k in ("model", "benchmark", "mteb_version", "task_set_hash", "source_kind", "source_url", "read_on",
              "sidekick_revision"):
        if not isinstance(e[k], str):
            out.append(f"{k} {e[k]!r} isn't a string")
    for k in ("n_tasks", "of"):
        if not is_int(e[k]):
            out.append(f"{k} {e[k]!r} isn't an integer")
    for k in ("mean_task", "mean_task_type"):
        if e.get(k) is not None and not is_number(e[k]):
            out.append(f"{k} {e[k]!r} isn't a number")
    revs = e["evaluated_revisions"]
    if not isinstance(revs, dict) or not all(isinstance(r, str) and is_int(n) for r, n in revs.items()):
        out.append("evaluated_revisions isn't a map of revision to task count")
    if e.get("revision_match") is not None and e["revision_match"] not in REVISION_MATCHES:
        out.append(f"revision_match {e['revision_match']!r} isn't one of {', '.join(REVISION_MATCHES)}")
    if e.get("model_context") is not None and not is_int(e["model_context"]):
        out.append(f"model_context {e['model_context']!r} isn't an integer")
    if (e.get("model_context") is None) != (e.get("model_context_source") is None):
        out.append("model_context and model_context_source must be given together")
    elif e.get("model_context_source") is not None and e["model_context_source"] not in CONTEXT_SOURCES:
        out.append(f"model_context_source {e['model_context_source']!r} isn't one of {', '.join(CONTEXT_SOURCES)}")
    if e.get("prompting") is not None and not isinstance(e["prompting"], str):
        out.append(f"prompting {e['prompting']!r} isn't a string")
    if not isinstance(e["categories"], dict):
        return out + ["categories isn't a map"]
    for name, c in e["categories"].items():
        if not isinstance(c, dict):
            out.append(f"{name} isn't a map")
            continue
        if c.get("value") is not None and not is_number(c["value"]):
            out.append(f"{name}: value {c['value']!r} isn't a number")
        for k in ("n_tasks", "of", "n_trained_on", "n_model_card"):
            if not is_int(c.get(k)):
                out.append(f"{name}: {k} {c.get(k)!r} isn't an integer")
        if not isinstance(c.get("metric"), str):
            out.append(f"{name}: metric {c.get('metric')!r} isn't a string")
    return out


def mteb_problems(data, models):
    """docs/data/benchmarks.mteb.json's entries: what --check refuses."""
    problems = []
    if data.get("scale") != "fraction":
        problems.append("benchmarks.mteb.json: scale isn't fraction")
    revision = (data.get("results") or {}).get("revision")
    if not isinstance(revision, str) or not re.fullmatch(SHA, revision):
        problems.append("benchmarks.mteb.json: no results revision")
    problems += [f"benchmarks.mteb.json: {k} isn't a string" for k in ("mteb_version", "read_on")
                 if not isinstance(data.get(k), str)]
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict) or not all(
            isinstance(b, dict) and is_int(b.get("n_tasks")) and isinstance(b.get("task_set_hash"), str)
            and isinstance(b.get("categories"), dict) and all(is_int(n) for n in b["categories"].values())
            for b in benchmarks.values()):
        return problems + ["benchmarks.mteb.json: `benchmarks` isn't a map of benchmark to its task set"]
    if not isinstance(data.get("entries"), list):
        return problems + ["benchmarks.mteb.json: `entries` isn't a list"]
    for i, e in enumerate(data["entries"]):
        if not isinstance(e, dict):
            problems.append(f"benchmarks.mteb.json entry {i + 1} isn't an object")
            continue
        where = f"benchmarks.mteb.json entry {i + 1} ({e.get('model')}, {e.get('benchmark')})"
        need = ("model", "benchmark", "mteb_version", "task_set_hash", "source_kind", "source_url", "read_on",
                "sidekick_revision", "evaluated_revisions", "n_tasks", "of", "categories")
        missing = [k for k in need if blank_text(e.get(k)) or e.get(k) == {}]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        wrong = mteb_type_problems(e)
        if wrong:
            problems += [f"{where}: {w}" for w in wrong]
            continue
        if sum(e["evaluated_revisions"].values()) != e["n_tasks"]:
            problems.append(f"{where}: evaluated_revisions count {sum(e['evaluated_revisions'].values())} tasks, "
                            f"not its {e['n_tasks']}")
        if e["model"] not in models:
            problems.append(f"{where}: model {e['model']!r} isn't in docs/data/models.toml")
        b = benchmarks.get(e["benchmark"])
        if b is None or b.get("task_set_hash") != e["task_set_hash"]:
            problems.append(f"{where}: task set {e['task_set_hash']} isn't the file's {e['benchmark']}")
            continue
        if e["of"] != b["n_tasks"]:
            problems.append(f"{where}: of {e['of']}, the benchmark has {b['n_tasks']} tasks")
        if set(e["categories"]) != set(b["categories"]):
            problems.append(f"{where}: categories {sorted(e['categories'])}, the benchmark's are "
                            f"{sorted(b['categories'])}")
        if e["source_kind"] not in ("mteb", "mixed"):
            problems.append(f"{where}: source_kind {e['source_kind']!r} isn't mteb or mixed")
        if not pinned(e["source_url"]):
            problems.append(f"{where}: source_url {e['source_url']} isn't pinned to a revision or commit")
        elif revision and revision not in e["source_url"]:
            problems.append(f"{where}: source_url isn't the file's results revision {revision}")
        problems += check_date(where, e["read_on"])
        cats = e["categories"]
        means = [(MEAN, e.get("mean_task"), e["n_tasks"], e["of"])]
        card = 0
        for name, c in cats.items():
            cw = f"{where} {name}"
            if not c.get("metric"):
                problems.append(f"{cw}: no metric")
            if b["categories"].get(name) != c.get("of"):
                problems.append(f"{cw}: of {c.get('of')}, the benchmark has {b['categories'].get(name)}")
            n = c.get("n_tasks")
            for field in ("n_trained_on", "n_model_card"):
                if not isinstance(c.get(field), int) or not 0 <= c[field] <= (n or 0):
                    problems.append(f"{cw}: {field} {c.get(field)!r} isn't from 0 to its {n} tasks")
            card += c.get("n_model_card") or 0
            means.append((name, c.get("value"), n, c.get("of")))
        if sum(c.get("n_tasks") or 0 for c in cats.values()) != e["n_tasks"]:
            problems.append(f"{where}: its categories' tasks don't add up to its {e['n_tasks']}")
        if (e["source_kind"] == "mixed") != (card > 0):
            problems.append(f"{where}: source_kind {e['source_kind']} with {card} tasks from the model card")
        any_score = False
        for name, value, n, of in means:
            if value is not None and n != of:
                problems.append(f"{where} {name}: a score over {n} of {of} tasks; a mean needs every task")
            elif value is None and n == of:
                problems.append(f"{where} {name}: all {of} tasks present but no score")
            elif value is not None:
                any_score = True
                problems += check_value(f"{where} {name}", value * 100, name)
        values = [c.get("value") for c in cats.values()]
        if e.get("mean_task") is not None and None not in values:
            by_task = sum(c["value"] * c["of"] for c in cats.values()) / sum(c["of"] for c in cats.values())
            if abs(by_task - e["mean_task"]) > MEAN_TOLERANCE:
                problems.append(f"{where}: mean_task {e['mean_task']} isn't its categories' task-weighted mean "
                                f"{by_task:.6f}")
            by_type = sum(values) / len(values)
            if e.get("mean_task_type") is None or abs(by_type - e["mean_task_type"]) > MEAN_TOLERANCE:
                problems.append(f"{where}: mean_task_type {e.get('mean_task_type')} isn't its categories' mean "
                                f"{by_type:.6f}")
        elif e.get("mean_task_type") is not None:
            problems.append(f"{where}: a mean_task_type without every category")
        if any_score and not e.get("revision_match"):
            problems.append(f"{where}: scores without a revision_match")
    return problems


def mteb_records(data):
    """benchmarks.mteb.json as score records, one per category (and the
    benchmark's mean over tasks)."""
    out = []
    for e in data["entries"]:
        common = dict(model=e["model"], benchmark=e["benchmark"],
                      version=f"mteb {e['mteb_version']}, task set {e['task_set_hash']}",
                      source_kind=e["source_kind"], source_url=e["source_url"], read_on=e["read_on"],
                      revision_match=e.get("revision_match"), evaluated_revisions=e["evaluated_revisions"],
                      prompting=e.get("prompting"), model_context=e.get("model_context"),
                      model_context_source=e.get("model_context_source"), note=None)
        cats = [(MEAN, {"value": e.get("mean_task"), "n_tasks": e["n_tasks"], "of": e["of"],
                        "metric": "mean of the tasks' main metrics",
                        "n_trained_on": sum(c["n_trained_on"] for c in e["categories"].values()),
                        "n_model_card": sum(c["n_model_card"] for c in e["categories"].values())})]
        cats += sorted(e["categories"].items(), key=lambda kv: category_order(kv[0]))
        for name, c in cats:
            value = None if c["value"] is None else round(c["value"] * 100, 2)
            out.append(dict(common, category=name, metric=c["metric"], value=value,
                            value_text=None if value is None else f"{value:.2f}",
                            n_tasks=c["n_tasks"], of=c["of"], trained_on=c["n_trained_on"],
                            model_card_tasks=c["n_model_card"]))
    return out


def curated_records(entries):
    out = []
    for e in entries:
        text = points_text(e["value"], e["scale"])
        for m in e["model"] if isinstance(e["model"], list) else [e["model"]]:
            out.append(dict(model=m, benchmark=e["benchmark"], version=e["version"], source_kind=e["source_kind"],
                            source_url=e["source_url"], read_on=str(e["read_on"]), revision_match=None,
                            evaluated_revisions=None, prompting=None, model_context=None,
                            model_context_source=None, note=e.get("note"), category=e["category"],
                            metric=e["metric"], value=float(text), value_text=text, n_tasks=None, of=None,
                            trained_on=1 if e.get("trained_on") else 0, model_card_tasks=None))
    return out


def category_order(name):
    return (CATEGORIES.index(name) if name in CATEGORIES else len(CATEGORIES), name)


# --- MTEB scores measured on sidekick's builds -------------------------------
#
# docs/data/mteb_transfer.json, written by tools/mteb_transfer.py: MTEB tasks
# run through sidekickd (each build's tokenizer, graph and input cap) and
# through sentence-transformers in fp32 (the adapter check), beside the
# published score for the same model and task, read from the same pinned
# mteb/results commit as benchmarks.mteb.json.

TRANSFER = DATA / "mteb_transfer.json"
TRANSFER_PATHS = ["reference", "coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
RUN_FIELDS = {"model", "path", "revision", "chip", "os", "mteb_version", "date"}
BUILD_FIELDS = {"backend", "compute_units", "max_seq_len", "build", "sidekick"}
SCORE_FIELDS = {"model", "path", "task", "category", "metric", "score", "seconds", "inputs", "over_cap", "longest",
                "published", "published_revision", "published_source", "published_trained_on"}


def transfer_problems(data, models, mteb):
    """docs/data/mteb_transfer.json: what --check refuses."""
    where = "mteb_transfer.json"
    problems = []
    if data.get("scale") != "fraction":
        problems.append(f"{where}: scale isn't fraction")
    if data.get("benchmark") not in mteb["benchmarks"]:
        problems.append(f"{where}: benchmark {data.get('benchmark')!r} isn't one benchmarks.mteb.json collects")
    if data.get("mteb_version") != mteb["mteb_version"]:
        problems.append(f"{where}: mteb {data.get('mteb_version')}, the published scores' is {mteb['mteb_version']}")
    if data.get("results") != mteb["results"]:
        problems.append(f"{where}: published scores aren't from benchmarks.mteb.json's results revision")
    problems += check_date(where, data.get("read_on"))
    tasks, runs, scores = data.get("tasks"), data.get("runs"), data.get("scores")
    if not isinstance(tasks, dict) or not isinstance(runs, list) or not isinstance(scores, list):
        return problems + [f"{where}: tasks, runs or scores missing"]
    for name, t in tasks.items():
        if not isinstance(t, dict) or not all(isinstance(t.get(k), str) for k in ("category", "metric")) or not all(
                isinstance(t.get(k), list) and t[k] for k in ("splits", "subsets")):
            problems.append(f"{where} task {name}: needs category, metric, splits and subsets")
    have = set()
    for r in runs:
        w = f"{where} run ({r.get('model')}, {r.get('path')})"
        fields = RUN_FIELDS | (set() if r.get("path") == "reference" else BUILD_FIELDS) | (
            {"runtime"} if r.get("path") == "reference" else set())
        missing = sorted(k for k in fields if blank_text(r.get(k)))
        extra = sorted(set(r) - fields)
        if missing or extra:
            problems.append(f"{w}: " + "; ".join(([f"missing {', '.join(missing)}"] if missing else [])
                                                  + ([f"unknown fields {', '.join(extra)}"] if extra else [])))
            continue
        if r["model"] not in models:
            problems.append(f"{w}: model isn't in docs/data/models.toml")
        elif r["revision"] != models[r["model"]].get("source", "").partition("@")[2]:
            problems.append(f"{w}: revision {r['revision']} isn't the model's pinned source")
        if r["path"] not in TRANSFER_PATHS:
            problems.append(f"{w}: path isn't one of {', '.join(TRANSFER_PATHS)}")
        if r["mteb_version"] != mteb["mteb_version"]:
            problems.append(f"{w}: mteb {r['mteb_version']}, the published scores' is {mteb['mteb_version']}")
        if r["path"] != "reference" and (not is_int(r["max_seq_len"]) or r["max_seq_len"] < 1):
            problems.append(f"{w}: max_seq_len {r['max_seq_len']!r} isn't a positive integer")
        problems += check_date(w, r["date"])
        if (r["model"], r["path"]) in have:
            problems.append(f"{w}: listed twice")
        have.add((r["model"], r["path"]))
    seen = set()
    for s in scores:
        w = f"{where} score ({s.get('model')}, {s.get('path')}, {s.get('task')})"
        missing = sorted(k for k in SCORE_FIELDS - {"over_cap", "longest", "published", "published_revision",
                                                    "published_source", "published_trained_on"}
                         if blank_text(s.get(k)))
        extra = sorted(set(s) - SCORE_FIELDS - {"vs_reference"})
        if missing or extra:
            problems.append(f"{w}: " + "; ".join(([f"missing {', '.join(missing)}"] if missing else [])
                                                  + ([f"unknown fields {', '.join(extra)}"] if extra else [])))
            continue
        key = (s["model"], s["path"], s["task"])
        if key in seen:
            problems.append(f"{w}: listed twice")
        seen.add(key)
        if (s["model"], s["path"]) not in have:
            problems.append(f"{w}: no run describes it")
        t = tasks.get(s["task"])
        if t is None:
            problems.append(f"{w}: task isn't in `tasks`")
        elif (s["category"], s["metric"]) != (t["category"], t["metric"]):
            problems.append(f"{w}: category or metric isn't the task's")
        for k in ("score", "published"):
            if s.get(k) is not None and (not is_number(s[k]) or not -1 <= s[k] <= 1):
                problems.append(f"{w}: {k} {s[k]!r} isn't a fraction")
        if not is_int(s["inputs"]) or s["inputs"] < 1:
            problems.append(f"{w}: inputs {s['inputs']!r} isn't a positive integer")
        elif s["path"] != "reference" and not (is_int(s.get("over_cap")) and 0 <= s["over_cap"] <= s["inputs"]):
            problems.append(f"{w}: over_cap {s.get('over_cap')!r} isn't from 0 to its {s['inputs']} inputs")
        if s.get("published") is not None and (s.get("published_source") not in ("mteb", "model card")
                                               or not isinstance(s.get("published_revision"), str)):
            problems.append(f"{w}: a published score needs its revision and a source of mteb or model card")
        v = s.get("vs_reference")
        if v is not None and not (isinstance(v, dict) and is_int(v.get("inputs"))
                                  and all(is_number(v.get(k)) and -1 <= v[k] <= 1.000001
                                          for k in ("worst_cosine", "mean_cosine"))):
            problems.append(f"{w}: vs_reference needs inputs, worst_cosine and mean_cosine")
    return problems


def benchmark_scores(models):
    """The results file's `benchmarks`, after checking its inputs."""
    curated = load_toml(BENCH_TOML).get("scores", [])
    mteb = json.loads(BENCH_MTEB.read_text())
    transfer = json.loads(TRANSFER.read_text())
    problems = curated_problems(curated, models) + mteb_problems(mteb, models) + transfer_problems(
        transfer, models, mteb)
    if problems:
        raise SystemExit("benchmark scores refused:\n  " + "\n  ".join(problems))
    scores = mteb_records(mteb) + curated_records(curated)
    keys = [(s["model"], s["benchmark"], s["version"], s["category"], s["metric"]) for s in scores]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit("benchmark scores refused:\n  " + "\n  ".join(f"duplicate score: {k}" for k in dup))
    order = list(models)
    benchmarks = list(mteb["benchmarks"])
    scores.sort(key=lambda s: (order.index(s["model"]), s["source_kind"] not in ("mteb", "mixed"),
                               benchmarks.index(s["benchmark"]) if s["benchmark"] in benchmarks else len(benchmarks),
                               s["benchmark"], category_order(s["category"]), s["metric"]))
    return {
        "mteb": {"results": mteb["results"], "mteb_version": mteb["mteb_version"], "read_on": mteb["read_on"],
                 "benchmarks": mteb["benchmarks"]},
        "source_kinds": SOURCE_KINDS,
        "scores": scores,
        "transfer": {k: v for k, v in transfer.items() if k != "about"},
    }


# --- docs/MODELS.md tables ---------------------------------------------------
#
# The data tables in docs/MODELS.md are generated from the results file. Each
# sits between `<!-- generated: NAME -->` and `<!-- /generated -->` markers,
# indented like its markers, and is rewritten in place. The classifier and
# reranker tables carry prose in their cells, so they stay hand-written; their
# grades and errors are checked against the results file instead.

MODELS_MD = REPO / "docs" / "MODELS.md"
M1 = "Apple M1 Max"
RYZEN, THREADRIPPER = "AMD Ryzen 7 5800X", "AMD Ryzen Threadripper 3970X"
XEON = "Intel Xeon W-3235"
COREML = [("coreml-cpu", "Core ML CPU"), ("coreml-gpu", "Core ML GPU"), ("coreml-ane", "Core ML ANE")]


def cosine(error):
    """A worst-case cosine as the tables print it, from 1 − cosine."""
    if error is None:
        return ""
    if error < 1e-8:
        return f"1 − {error:.0e}"
    if error >= 1e-2:
        return f"{1 - error:.3f}"
    return f"{1 - error:.{max(5, math.ceil(-math.log10(error)))}f}"


def grade_cell(r, bold=False):
    if r is None:
        return ""
    g = f"**{r['grade']}**" if bold else r["grade"]
    return f"{g} {r['error_text']}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["|" + "|".join(f" {c} " if c else " " for c in row) + "|" for row in rows]
    return lines


def ordered(rows, models, keep):
    """The rows `keep` selects, in docs/data/models.toml's model order, then
    by precision."""
    order = list(models)
    return sorted((r for r in rows if keep(r)),
                  key=lambda r: (order.index(r["model"]), PRECISIONS.index(r["precision"])))


def label(r, graded=False):
    name = r["name"] if r["precision"] in ("fp32", "fp16") else f"{r['name']}, {r['precision']}"
    return f"{name} (graded {r['grade']})" if graded and r["grade"] in ("D", "F") else name


def ms_text(ms):
    return f"{ms:,.0f}" if ms >= 100 else f"{ms:.1f}"


def coreml_grades(rows, models):
    """Core ML embedders on the M1 Max: the parity suite's grades."""
    idx = index(rows)
    out = []
    for ane in ordered(rows, models, lambda r: r["task"] == "embed" and r["path"] == "coreml-ane" and r["chip"] == M1):
        cpu = idx.get((ane["model"], "coreml-cpu", "fp16", M1))
        gpu = idx.get((ane["model"], "coreml-gpu", "fp16", M1))
        bias = ane["bias"]
        bias = "" if bias is None else f" ({'+' if bias > 0 else '−' if bias < 0 else ''}{abs(bias):.3f})"
        case = f" ({ane['worst_case']})" if ane["worst_case"] else ""
        ms = (f"{ane['ms']:.1f} ({cpu['ms']:.1f})" if ane["ms"] is not None and cpu and cpu["ms"] is not None
              else "not measured")
        out.append([ane["name"], grade_cell(cpu), grade_cell(gpu), grade_cell(ane, bold=True) + case,
                    f"{ane['drift']:.3f}{bias}", str(ane["rank_flips"]), ms])
    return table(["model", "CPU", "GPU", "ANE: worst case", "ANE: similarity drift (bias)", "ANE: rank flips", "ms"],
                 out)


def onnx_accuracy(rows, models):
    """onnx-cpu grades on every chip, beside the Core ML paths on the M1 Max.
    The two AMD machines share a column; they must agree."""
    idx = index(rows)
    coreml = {r["model"] for r in rows if r["path"].startswith("coreml")}
    out = []
    for r in ordered(rows, models, lambda r: r["path"] == "onnx-cpu" and r["chip"] == M1):
        key = (r["model"], "onnx-cpu", r["precision"])
        amd = {grade_cell(idx[key + (c,)], bold=True) for c in (RYZEN, THREADRIPPER) if key + (c,) in idx}
        if len(amd) > 1:
            raise SystemExit(f"{r['model']} ({r['precision']}): the Ryzen and Threadripper grades differ: {amd}")
        if r["precision"] != "fp32":
            core = ["", "", ""]
        elif r["model"] in coreml:
            core = [grade_cell(idx.get((r["model"], p, "fp16", M1))) for p, _ in COREML]
        else:
            core = ["not converted"] * 3
        out.append([label(r), grade_cell(r, bold=True), amd.pop() if amd else "",
                    grade_cell(idx.get(key + (XEON,)), bold=True)] + core)
    return table(["model", "onnx-cpu, M1 Max", "onnx-cpu, Ryzen 7 5800X and Threadripper 3970X",
                  "onnx-cpu, Xeon W-3235"] + [h for _, h in COREML], out)


def latency(rows, models):
    """Fixed-length latency from Python on the M1 Max, one input at a time,
    at each runtime's default threads."""
    idx = index(rows)

    def runs(r, how):
        if r is None:
            return {}
        return {e["tokens"]: e["ms"] for e in r["latency"] or []
                if e["how"].startswith(how) and e["batch"] == 1 and e["threads"] is None}

    out = []
    for r in ordered(rows, models, lambda r: r["path"] == "onnx-cpu" and r["chip"] == M1 and runs(r, "ONNX Runtime")):
        cells = [runs(r, "ONNX Runtime")]
        cells += [runs(idx.get((r["model"], p, "fp16", M1)), "Core ML") if r["precision"] == "fp32" else {}
                  for p, _ in COREML]
        tokens = sorted(cells[0])
        out.append([label(r, graded=True), " / ".join(map(str, tokens))]
                   + [" / ".join(ms_text(c[t]) for t in tokens) if c else "" for c in cells])
    return table(["model", "tokens", "onnx-cpu (fp32)", "Core ML CPU (fp16)", "Core ML GPU", "Core ML ANE"], out)


THREAD_CHIPS = [(M1, "M1 Max", "8P + 2E"), (RYZEN, "Ryzen 7 5800X", "8 cores, 16 threads"),
                (THREADRIPPER, "Threadripper 3970X", "32 cores"), (XEON, "Xeon W-3235", "12 cores, 24 threads")]
THREAD_MODELS = ["gte-modernbert-base", "embeddinggemma-300m", "qwen3-embedding-0.6b"]


def threads(rows, models):
    """sidekickd's onnx-cpu latency for one long input by intra-op threads."""
    idx = index(rows)
    body, counts = [], set()
    for chip, short, cores in THREAD_CHIPS:
        first = True
        for model in THREAD_MODELS:
            r = idx.get((model, "onnx-cpu", "fp32", chip))
            ms = {e["threads"]: e["ms"] for e in (r["latency"] or []) if e["how"] == "sidekickd over HTTP"
                  and e["tokens"] > 100} if r else {}
            if not ms:
                continue
            counts |= set(ms)
            body.append((f"{short} ({cores}), {r['name']}" if first else f"{short}, {r['name']}", ms))
            first = False
    counts = sorted(counts)
    return table(["machine, model"] + [str(n) for n in counts],
                 [[name] + [f"{round(ms[n]):,}" if n in ms else "" for n in counts] for name, ms in body])


def memory(rows, models):
    """Size on disk and the process's footprint once loaded, M1 Max."""
    idx = index(rows)

    def mb(x, sign=""):
        return "" if x is None else f"{sign}{max(x, 0):,} MB"

    def footprint(r):
        if r is None or r["footprint_mb"] is None:
            return ""
        lo, hi = r["footprint_range_mb"] or (r["footprint_mb"],) * 2
        if hi - lo > max(64, 0.05 * r["footprint_mb"]):   # runs disagree: show the range
            return f"+{lo / 1024:.1f}–{hi / 1024:.1f} GB"
        return mb(r["footprint_mb"], "+")

    out = []
    for r in ordered(rows, models, lambda r: r["path"] == "onnx-cpu" and r["chip"] == M1 and r["footprint_mb"] is not None):
        core = [idx.get((r["model"], p, "fp16", M1)) if r["precision"] == "fp32" else None for p, _ in COREML]
        disk = next((c["disk_mb"] for c in core if c and c["disk_mb"] is not None), None)
        out.append([label(r), mb(r["disk_mb"]), footprint(r), mb(disk)] + [footprint(c) for c in core])
    return table(["model", "onnx-cpu: disk", "onnx-cpu: footprint", "Core ML: disk (all buckets)", "Core ML CPU",
                  "Core ML GPU", "Core ML ANE (process)"], out)


def bound(x):
    return f"{x:g}" if x >= 1e-3 else f"{x:.0e}".replace("e-0", "e-")


def cosine_scale(rows, models):
    """The embedder letters, from [grades]."""
    c = load_grades()["cosine"]
    return table(["grade", "worst-case cosine", "meaning"],
                 [["**A**", f"≥ {bound(c['A'])}", "indistinguishable from fp32"],
                  ["**B**", f"≥ {bound(c['B'])}", ""], ["**C**", f"≥ {bound(c['C'])}", ""],
                  ["**D**", f"below {bound(c['C'])}", ""], ["**F**", "—", "a hard gate failed"]])


def classifier_scales(rows, models):
    """The classifier letters, from [grades]: absolute |Δp| and the ratio to
    the fp16 ceiling."""
    g = load_grades()
    d, r = g["dp"], g["ratio"]
    return table(["grade", "worst-case \\|Δp\\|", "p99 \\|Δp\\| ÷ the fp16 ceiling's"],
                 [[f"**{x}**", f"≤ {bound(d[x])}", f"≤ {r[x]:g}×"] for x in "ABC"]
                 + [["**D**", f"above {bound(d['C'])}", f"above {r['C']:g}×"],
                    ["**F**", "—", "a hard gate failed"]])


SHORT_PATH = {"coreml-ane": "ANE", "coreml-gpu": "GPU", "coreml-cpu": "CPU"}
REVISION_TEXT = {"exact": "sidekick's revision", "same-weights": "another revision, same weights",
                 "unverified": "another revision, weights not checked (gated)",
                 "different": "another revision, different weights", "model card only": "model card only"}


def build_grades(idx, model, models):
    """A model's grade on its served Core ML path and on onnx-cpu (fp32),
    on the M1 Max, with each build's max_seq_len."""
    served = models[model]["served"]
    core, onnx = idx.get((model, served, "fp16", M1)), idx.get((model, "onnx-cpu", "fp32", M1))
    grades = ", ".join(([f"{SHORT_PATH[served]} **{core['grade']}**"] if core else [])
                       + ([f"onnx-cpu **{onnx['grade']}**"] if onnx else []))
    caps = [r["max_seq_len"] for r in (core, onnx) if r and r["max_seq_len"]]
    if not caps:
        return grades, ""
    if len(set(caps)) == 1:
        return grades, f"{caps[0]:,}"
    return grades, f"{core['max_seq_len']:,} (onnx-cpu {onnx['max_seq_len']:,})"


def score_text(s):
    """A score as the tables print it (its value_text: 2 decimals for MTEB,
    the source's own precision otherwise); † and a count for tasks in the
    model's training data; for an incomplete MTEB category, how many of its
    tasks there are."""
    if s is None:
        return ""
    if s["value"] is None:
        return f"({s['n_tasks']} of {s['of']} tasks)" if s["n_tasks"] else ""
    text = s["value_text"]
    if s["trained_on"]:
        text += f" †{s['trained_on']}" if s["of"] is not None else " †"
    return text


def mteb_table(benchmark, every_embedder=False):
    """One MTEB benchmark: each model's categories beside its grades and
    caps. With every_embedder, embedders without a score get an empty row."""
    def render(rows, models, bench):
        idx = index(rows)
        scores = [s for s in bench["scores"] if s["benchmark"] == benchmark and s["source_kind"] in ("mteb", "mixed")]
        cats = sorted(bench["mteb"]["benchmarks"][benchmark]["categories"], key=category_order)
        by = {(s["model"], s["category"]): s for s in scores}
        have = {s["model"] for s in scores}
        out = []
        for model, m in models.items():
            if model not in have and not (every_embedder and m["task"] == "embed"):
                continue
            grades, caps = build_grades(idx, model, models)
            mean = by.get((model, MEAN))
            context = ""
            if mean and mean["model_context"]:
                context = f"{mean['model_context']:,}"
                if mean["model_context_source"] == "advertised":
                    context = f"≤ {context} (advertised)"
            revision = ""
            if mean:
                revision = REVISION_TEXT.get(mean["revision_match"], mean["revision_match"] or "")
                if mean["model_card_tasks"]:
                    revision += (f"; {mean['model_card_tasks']} of {mean['of']} tasks from the model card "
                                 "(the authors' own runs, not mteb's)")
            out.append([m["name"], grades, f"{caps} / {context}" if mean else caps,
                        (mean["prompting"] or "") if mean else ""]
                       + [score_text(by.get((model, c))) for c in [MEAN] + cats] + [revision])
        return table(["model", "grade (M1 Max)", "max tokens: sidekick / MTEB run", "prompts in the MTEB run",
                      "mean (task)"] + cats + ["revision evaluated"], out)
    return render


def self_reported(rows, models, bench):
    """Scores from benchmarks.toml: each on its own benchmark."""
    idx = index(rows)
    out = []
    for s in bench["scores"]:
        if s["source_kind"] in ("mteb", "mixed"):
            continue
        grades, caps = build_grades(idx, s["model"], models)
        where = "model card" if "huggingface.co" in s["source_url"] else "documentation"
        kind = "" if s["source_kind"] == "self-reported" else f" ({s['source_kind']})"
        out.append([models[s["model"]]["name"], grades, caps, f"{s['benchmark']} ({s['version']})",
                    s["category"], s["metric"], score_text(s), f"[{where}]({s['source_url']}){kind}, read {s['read_on']}",
                    s["note"] or ""])
    return table(["model", "grade (M1 Max)", "max tokens", "benchmark", "category", "metric", "score", "source",
                  "note"], out)


def prompts(rows, models):
    """Each embedder's prompts by `input_type`, from its committed manifest's
    [prefixes]: what sidekick puts before every input (D9)."""
    def cell(prompt):
        if not prompt:
            return "none"
        # A code span shows the prompt's characters, its trailing space
        # included; a newline is shown escaped, and a pipe escaped for the table.
        return "`" + prompt.replace("\n", "\\n").replace("|", "\\|") + "`"
    out = []
    for mid, m in models.items():
        manifest = MANIFESTS / mid / "manifest.toml"
        if m.get("task") != "embed" or not manifest.exists():
            continue
        p = load_toml(manifest).get("prefixes", {})
        out.append([m["name"], cell(p.get("query", "")), cell(p.get("document", ""))])
    return table(["model", "`query`", "`document`, and no `input_type`"], out)


def mteb_source(rows, models, bench):
    """Where the MTEB scores came from, as a sentence."""
    m = bench["mteb"]
    return [f"MTEB results: [mteb/results at {m['results']['revision'][:12]}]({m['results']['url']}), "
            f"benchmarks as mteb {m['mteb_version']} defines them, read {m['read_on']}."]


def points(v):
    return f"{v * 100:.2f}"


def delta(v, ref):
    d = round(v * 100, 2) - round(ref * 100, 2)
    return f"{'+' if d > 0 else '−' if d < 0 else '±'}{abs(d):.2f}"


def transfer_scores(rows, models, bench):
    """Each model and task: the published score, then each build's score
    (and its difference from the published one), and the share of the
    task's inputs over each build's cap."""
    t = bench["transfer"]
    idx = index(rows)
    caps = {(r["model"], r["path"]): r.get("max_seq_len") for r in t["runs"]}
    by = {(s["model"], s["path"], s["task"]): s for s in t["scores"]}
    builds = [p for p in TRANSFER_PATHS[1:] if any(s["path"] == p for s in t["scores"])]
    out = []
    for model, m in models.items():
        for task, meta in t["tasks"].items():
            cells = [by.get((model, p, task)) for p in builds]
            if not any(cells):
                continue
            first = next(c for c in cells if c)
            pub = first["published"]
            pub_text = "" if pub is None else points(pub) + (" (model card)" if first["published_source"] ==
                                                               "model card" else "")
            ref = by.get((model, "reference", task))
            ref_text = "" if ref is None else points(ref["score"]) + (
                f" ({delta(ref['score'], pub)})" if pub is not None else "")
            row = [m["name"], build_grades(idx, model, models)[0], f"{task} ({meta['category']})", pub_text,
                   ref_text]
            over = []
            for p, c in zip(builds, cells):
                if c is None:
                    row.append("")
                    over.append("")
                    continue
                row.append(points(c["score"]) + (f" ({delta(c['score'], pub)})" if pub is not None else ""))
                share = c["over_cap"] / c["inputs"]
                over.append(f"{share * 100:.1f}% over {caps[(model, p)]:,}" if c["over_cap"]
                            else f"none over {caps[(model, p)]:,}")
            row.append(" / ".join(over))
            out.append(row)
    names = {"coreml-ane": "ANE", "coreml-gpu": "GPU", "coreml-cpu": "Core ML CPU", "onnx-cpu": "onnx-cpu"}
    return table(["model", "grade (M1 Max)", "task (category)", "MTEB run", "sentence-transformers fp32, here"]
                 + [f"sidekick {names[p]}" for p in builds]
                 + ["inputs over the cap: " + " / ".join(names[p] for p in builds)], out)


def vector_cosine(c):
    """A cosine between two fp32 vectors: within 1e-7 of 1 is all fp32
    rounding can tell apart."""
    return "1.0000000" if 1 - c < 1e-7 else cosine(1 - c)


def transfer_check(rows, models, bench):
    """The adapter check: each build's vectors against sentence-transformers
    in fp32, input by input, on the tasks where both were kept."""
    t = bench["transfer"]
    by = {(s["model"], s["path"], s["task"]): s for s in t["scores"]}
    out = []
    for s in t["scores"]:
        ref, v = by.get((s["model"], "reference", s["task"])), s.get("vs_reference")
        if s["path"] == "reference" or ref is None or v is None:
            continue
        out.append([models[s["model"]]["name"], s["path"], s["task"], f"{v['inputs']:,}",
                    "none" if not s["over_cap"] else f"{s['over_cap']:,}", points(ref["score"]),
                    points(s["score"]), vector_cosine(v["worst_cosine"]), vector_cosine(v["mean_cosine"])])
    return table(["model", "path", "task", "inputs", "over the cap", "sentence-transformers fp32, here",
                  "sidekick", "worst cosine", "mean cosine"], out)


def transfer_source(rows, models, bench):
    """Where and how the transfer runs were made, as a sentence."""
    t = bench["transfer"]
    builds = [r for r in t["runs"] if r["path"] != "reference"]
    refs = [r for r in t["runs"] if r["path"] == "reference"]
    chips = sorted({r["chip"] for r in t["runs"]})
    oses = sorted({r["os"] for r in t["runs"]})
    versions = sorted({r["sidekick"] for r in builds})
    dates = sorted({r["date"] for r in t["runs"]})
    runtimes = sorted({r["runtime"] for r in refs})
    return [f"Measured on {', '.join(chips)} ({', '.join(oses)}) with sidekick {', '.join(versions)}, "
            f"{dates[0]}{' to ' + dates[-1] if dates[-1] != dates[0] else ''}; reference: {'; '.join(runtimes)}. "
            f"Tasks as mteb {t['mteb_version']} defines {t['benchmark']}; published scores from "
            f"[mteb/results at {t['results']['revision'][:12]}]({t['results']['url']}). "
            f"Data: [mteb_transfer.json](data/mteb_transfer.json)."]


TABLES = {"coreml-grades": coreml_grades, "onnx-accuracy": onnx_accuracy, "latency": latency,
          "threads": threads, "memory": memory, "cosine-scale": cosine_scale,
          "classifier-scales": classifier_scales, "prompts": prompts}
# Tables from the results file's benchmark scores.
BENCH_TABLES = {"mteb-eng-v2": mteb_table("MTEB(eng, v2)", every_embedder=True),
                "mteb-multilingual-v2": mteb_table("MTEB(Multilingual, v2)"),
                "mteb-eng-v1": mteb_table("MTEB(eng, v1)"),
                "nanobeir-multilingual": mteb_table("NanoBEIR-multilingual"),
                "self-reported": self_reported, "mteb-source": mteb_source,
                "mteb-transfer": transfer_scores, "mteb-transfer-check": transfer_check,
                "mteb-transfer-source": transfer_source}
MARKER = re.compile(r"^( *)<!-- generated: ([a-z0-9-]+) -->$")


def render_docs(text, rows, models, bench):
    """docs/MODELS.md with every generated table rewritten."""
    tables = dict(TABLES, **{k: (lambda f: lambda r, m: f(r, m, bench))(f) for k, f in BENCH_TABLES.items()})
    lines, out, seen = text.split("\n"), [], set()
    i = 0
    while i < len(lines):
        out.append(lines[i])
        m = MARKER.match(lines[i])
        i += 1
        if not m:
            continue
        indent, name = m.groups()
        if name not in tables:
            raise SystemExit(f"docs/MODELS.md: no generated table named {name!r}")
        end = next((j for j in range(i, len(lines)) if lines[j].strip() == "<!-- /generated -->"), None)
        if end is None:
            raise SystemExit(f"docs/MODELS.md: the {name} table has no end marker")
        out += [indent + line for line in tables[name](rows, models)]
        i = end
        seen.add(name)
    if seen != set(tables):
        raise SystemExit(f"docs/MODELS.md: missing generated tables: {sorted(set(tables) - seen)}")
    return "\n".join(out)


HAND_WRITTEN = re.compile(r"as `([^`]+)`")
CELL = re.compile(r"^\**([A-F])\**\s+([0-9.]+(?:e-?\d+)?)")
RATIO = re.compile(r"([0-9.]+)× ceiling(?:, ([A-D]) level)?")
LETTER = re.compile(r"^\**([A-F])\**(?:[,:\s]|$)")
INTEL_HEADER = ["model", "Core ML CPU", "Core ML GPU"]


def check_hand_written(text, rows):
    """The classifier and reranker tables' CPU, GPU and ANE cells against
    the M1 Max rows: same grade, same error. Returns the problems."""
    idx = index(rows)
    problems, header = [], None
    for line in text.split("\n"):
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        m = HAND_WRITTEN.search(cells[0])
        if not m or "ANE" not in header:
            continue
        for col, path in (("CPU", "coreml-cpu"), ("GPU", "coreml-gpu"), ("ANE", "coreml-ane")):
            r, c = idx.get((m.group(1), path, "fp16", M1)), CELL.match(cells[header.index(col)])
            if r is None or c is None:
                problems.append(f"{m.group(1)} {col}: no {'row in models.json' if r is None else 'grade in the cell'}")
            elif c.group(1) != r["grade"] or not math.isclose(float(c.group(2)), r["error"], rel_tol=1e-9):
                problems.append(f"{m.group(1)} {col}: the table says {c.group(1)} {c.group(2)}, "
                                f"models.json {r['grade']} {r['error']}")
            else:
                problems += check_ratio(f"{m.group(1)} {col}", cells[header.index(col)], r)
    return problems + check_intel(text, rows)


def check_ratio(where, cell, row):
    """A cell's "N× ceiling" (and "X level") against the row's ratio."""
    m = RATIO.search(cell)
    if m is None:
        return []
    if row["ratio"] is None or not math.isclose(float(m.group(1)), row["ratio"], rel_tol=1e-9):
        return [f"{where}: the table says {m.group(1)}× ceiling, models.json {row['ratio']}"]
    if m.group(2) and m.group(2) != scale_letter(load_grades()["ratio"], row["ratio"], False):
        return [f"{where}: the table calls {m.group(1)}× {m.group(2)} level"]
    return []


def check_intel(text, rows):
    """The Intel Mac table: each cell's letter against the Xeon rows, and no
    "not graded" cell where models.json has a grade."""
    idx = index(rows)
    problems, header = [], None
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if header != INTEL_HEADER or set(cells[0]) <= set("-"):
            continue
        model = cells[0].split()[0]
        for col, path in (("Core ML CPU", "coreml-cpu"), ("Core ML GPU", "coreml-gpu")):
            r, cell = idx.get((model, path, "fp16", XEON)), cells[header.index(col)]
            m = LETTER.match(cell)
            if m is None:
                if r is not None:
                    problems.append(f"{model} {col} (Intel): the table says {cell!r}, models.json {r['grade']}")
            elif r is None:
                problems.append(f"{model} {col} (Intel): the table says {m.group(1)}, models.json has no row")
            elif m.group(1) != r["grade"]:
                problems.append(f"{model} {col} (Intel): the table says {m.group(1)}, models.json {r['grade']}")
    return problems


def check_grades(results, rows):
    """Every row's letter and error text against what its numbers give."""
    grades, problems = load_grades(), []
    if results.get("grades") != grades:
        problems.append("models.json's [grades] isn't fixtures/parity/expectations.toml's: rerun tools/models_table.py")
    for r in rows:
        where = f"{r['model']} {r['path']} {r['precision']} {r['chip']}"
        if grade_of(r, grades) != r["grade"]:
            problems.append(f"{where}: stored {r['grade']}, its numbers earn {grade_of(r, grades)}")
        if error_text(r) != r["error_text"]:
            problems.append(f"{where}: error text {r['error_text']!r}, expected {error_text(r)!r}")
    return problems


def check_caps(rows, caps):
    """Every row's max_seq_len against its build's committed manifest."""
    return [f"{r['model']} {r['path']} {r['chip']}: max_seq_len {r.get('max_seq_len')}, its manifest says "
            f"{cap_of(r, caps)}: rerun tools/models_table.py with the reports"
            for r in rows if r.get("max_seq_len") != cap_of(r, caps)]


def docs(results, models, check):
    """docs/MODELS.md's tables from the results file, and the results
    file's benchmark scores from their inputs."""
    text_json = results.read_text()
    data = json.loads(text_json)
    rows = data["rows"]
    bench = benchmark_scores(models)
    if data.get("benchmarks") != bench:
        if check:
            raise SystemExit(f"{results}'s benchmark scores are out of date: rerun tools/models_table.py --docs")
        data["benchmarks"] = bench
        results.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"rewrote the benchmark scores in {results}")
    text = MODELS_MD.read_text()
    new = render_docs(text, rows, models, bench)
    problems = check_grades(data, rows) + check_hand_written(new, rows) + check_caps(rows, build_caps(models))
    if problems:
        raise SystemExit("docs/MODELS.md disagrees with the results file:\n  " + "\n  ".join(problems))
    if check:
        if new != text:
            raise SystemExit("docs/MODELS.md's generated tables are out of date: rerun tools/models_table.py --docs")
        print("docs/MODELS.md's tables are up to date")
    elif new != text:
        MODELS_MD.write_text(new)
        print(f"rewrote the generated tables in {MODELS_MD.relative_to(REPO)}")


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
    p.add_argument("--docs", action="store_true",
                   help="only regenerate docs/MODELS.md's tables from --out, reading no reports")
    args = p.parse_args()

    models = load_toml(DATA / "models.toml")["models"]
    if args.docs:
        docs(args.out, models, args.check)
        return
    runs = load_toml(DATA / "runs.toml")
    grades = load_grades()
    rows, used = from_reports(args.reports, runs, models, grades)
    n_reports = len(rows)
    carried = from_carried(load_toml(DATA / "carried.toml"), models, grades)
    have = {(r["model"], r["path"], r["precision"], r["chip"]) for r in rows}
    rows += [c for c in carried if (c["model"], c["path"], c["precision"], c["chip"]) not in have]
    if args.bench:
        used += attach_bench(rows, args.bench, runs)
    if args.threads:
        used += attach_threads(rows, args.threads, runs)
    missing = sorted(set(runs["runs"]) - set(used))
    if missing:
        print(f"note: runs.toml describes reports not found here: {', '.join(missing)}", file=sys.stderr)
    caps = build_caps(models)
    for r in rows:
        r["served"] = served(r, models)
        r["max_seq_len"] = cap_of(r, caps)
    rows.sort(key=order)
    keys = [(r["model"], r["path"], r["precision"], r["chip"], r["runtime"]) for r in rows]
    if len(keys) != len(set(keys)):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise SystemExit(f"duplicate rows: {dup}")
    dates = sorted(d for d in (r["date"] for r in rows) if d)
    cargo = tomllib.loads((REPO / "Cargo.toml").read_text())
    out = {"sidekick": cargo["workspace"]["package"]["version"], "as_of": dates[-1] if dates else None,
           "grades": grades, "fields": FIELDS, "rows": rows, "benchmarks": benchmark_scores(models)}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            raise SystemExit(f"{args.out} is out of date: rerun tools/models_table.py with the same inputs")
        print(f"{args.out} is up to date ({len(rows)} rows)")
    else:
        args.out.write_text(text)
        print(f"{len(rows)} rows ({n_reports} from reports, {len(rows) - n_reports} carried from MODELS.md) -> {args.out}")
    docs(args.out, models, args.check)


if __name__ == "__main__":
    main()
