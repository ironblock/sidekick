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

It also writes the data tables in docs/MODELS.md from the results file
(those between `<!-- generated: NAME -->` markers), and checks the
hand-written classifier and reranker tables' grades against it, so the
document and the page can't drift apart. To change a published number,
change its input (a report, runs.toml or carried.toml) and rerun this.

Usage:
    python tools/models_table.py --reports DIR [--reports DIR ...]
        [--bench DIR] [--threads DIR] [--out docs/data/models.json] [--check]
    python tools/models_table.py --docs [--check]

    --check: write nothing; fail if the output would differ from --out
             or from docs/MODELS.md
    --docs:  only regenerate docs/MODELS.md's tables from --out, reading
             no reports (what CI checks)

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
    "machine_mb", "machine_measured", "note", "source",
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


TABLES = {"coreml-grades": coreml_grades, "onnx-accuracy": onnx_accuracy, "latency": latency,
          "threads": threads, "memory": memory, "cosine-scale": cosine_scale,
          "classifier-scales": classifier_scales}
MARKER = re.compile(r"^( *)<!-- generated: ([a-z-]+) -->$")


def render_docs(text, rows, models):
    """docs/MODELS.md with every generated table rewritten."""
    lines, out, seen = text.split("\n"), [], set()
    i = 0
    while i < len(lines):
        out.append(lines[i])
        m = MARKER.match(lines[i])
        i += 1
        if not m:
            continue
        indent, name = m.groups()
        if name not in TABLES:
            raise SystemExit(f"docs/MODELS.md: no generated table named {name!r}")
        end = next((j for j in range(i, len(lines)) if lines[j].strip() == "<!-- /generated -->"), None)
        if end is None:
            raise SystemExit(f"docs/MODELS.md: the {name} table has no end marker")
        out += [indent + line for line in TABLES[name](rows, models)]
        i = end
        seen.add(name)
    if seen != set(TABLES):
        raise SystemExit(f"docs/MODELS.md: missing generated tables: {sorted(set(TABLES) - seen)}")
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


def docs(results, models, check):
    data = json.loads(results.read_text())
    rows = data["rows"]
    text = MODELS_MD.read_text()
    new = render_docs(text, rows, models)
    problems = check_grades(data, rows) + check_hand_written(new, rows)
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
           "grades": grades, "fields": FIELDS, "rows": rows}
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
