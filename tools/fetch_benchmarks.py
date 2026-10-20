"""Collect MTEB benchmark scores for sidekick's models into
docs/data/benchmarks.mteb.json, which tools/models_table.py publishes
beside each build's fidelity grade (docs/MODELS.md and the comparison
page).

Source: the `mteb/results` dataset on Hugging Face, a flat parquet copy of
the MTEB results repository (github.com/embeddings-benchmark/results, the
per-task JSON behind the MTEB leaderboard), read at a pinned dataset
commit. Benchmark definitions (task lists, task types, evaluation splits
and subsets, each task's main metric) come from the `mteb` package, whose
version is recorded with every score. Re-running this with a newer
--results pin (and, if wanted, a newer mteb) refreshes every score; nothing
here is copied by hand. CI never runs it: CI checks the file it wrote.

Scores are aggregated as the leaderboard aggregates them, so they match it:
- a task's score is the mean of its main metric over the benchmark's
  (split, subset) cells, and a task missing any cell is missing;
- a category (task type) is the plain mean over the benchmark's tasks of
  that type, and null unless every one of them is present;
- the benchmark's mean over tasks, and its mean over categories, are null
  unless every task is present.
Per task, one revision of the model supplies the score: sidekick's pinned
revision if it has the task, else the revision with the most tasks, with
`external` (scores imported from the model card, so self-reported) last.
Every entry records which revisions supplied its tasks, how many tasks
were imported from the model card, and how many are flagged `trained_on`
(the task's training data is in the model's declared training data).

For a revision other than sidekick's pin, the weights are compared by the
Hugging Face LFS hashes of the weight files both revisions carry
(`same-weights`, `different`, or `unverified` when the repository is gated).
Each entry also records the longest input the run embedded
(`model_context`) and where that figure comes from: the mteb loader's
setting or the evaluated revision's sentence-transformers config, or,
where neither states it, the model's advertised context (an upper bound).

Usage (needs mteb and duckdb, in a Python 3.12 virtualenv):
    python -I tools/fetch_benchmarks.py [--results SHA] [--out docs/data/benchmarks.mteb.json]
        [--read-on YYYY-MM-DD] [--cross-check]

    --results:     the mteb/results dataset commit to read
    --read-on:     the date recorded as read; by default the existing
                   file's when --results is the pin it records (so a rerun
                   at the same pin writes the same bytes), else today
    --cross-check: also compare each complete mean with the live MTEB
                   leaderboard's (not pinned, so a difference after a newer
                   pin is expected); prints, changes nothing
"""

import argparse
import collections
import datetime
import hashlib
import json
import logging
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"

# The mteb/results dataset commit read by default. Bump it (and rerun) to
# refresh the scores.
RESULTS = "3b098007622127a024fb9065b23b6898aa936b34"

# Benchmarks collected, by their mteb names. MTEB(eng, v2) is the primary
# English benchmark; MTEB(eng, v1) is its 56-task predecessor, which some
# models are complete on and v2 isn't; NanoBEIR-multilingual is a small
# retrieval benchmark. Scores from different benchmarks are never combined.
BENCHMARKS = ["MTEB(eng, v2)", "MTEB(Multilingual, v2)", "MTEB(eng, v1)", "NanoBEIR-multilingual"]

# sidekick's model tasks whose checkpoints MTEB evaluates.
TASKS = ("embed", "rerank")

# How the MTEB runs prompt each model, which sidekick's single prefix per
# input_type may not reproduce. From each model's mteb implementation.
PROMPTING = {
    "sentence-transformers/all-MiniLM-L6-v2": "none",
    "BAAI/bge-small-en-v1.5": "query prefix, retrieval and reranking only",
    "intfloat/e5-small-v2": "query and passage prefixes, retrieval and reranking only",
    "Alibaba-NLP/gte-modernbert-base": "none",
    "google/embeddinggemma-300m": "a prompt per task type",
    "google/embeddinggemma-2": "a prompt per task type",
    "codefuse-ai/F2LLM-v2-160M": "an instruction per task",
    "LiquidAI/LFM2.5-Embedding-350M": "query and document prefixes, retrieval and reranking only",
    "Qwen/Qwen3-Embedding-0.6B": "an instruction per task",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "none",
}

# Revision directories holding scores imported from a model card rather than
# run with mteb.
MODEL_CARD = ("external", "no_revision_available")

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
SHA256 = re.compile(r"[0-9a-f]{64}")


def benchmark_cells(mteb, name):
    """(task, type, split, subset, main metric) for every cell a benchmark
    scores, and the benchmark's task-set hash: the first 12 hex digits of
    the sha256 of its sorted task names, one per line."""
    b = mteb.get_benchmark(name)
    cells = [(t.metadata.name, t.metadata.type, split, subset, t.metadata.main_score)
             for t in b.tasks for split in t.eval_splits for subset in t.hf_subsets]
    names = sorted({t.metadata.name for t in b.tasks})
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:12]
    return cells, digest


def metric_text(metrics):
    """A category's metric: the main metric of most of its tasks, and any
    other with its count."""
    counts = collections.Counter(metrics).most_common()
    head = counts[0][0]
    rest = [f"{m} on {n} {'task' if n == 1 else 'tasks'}" for m, n in counts[1:]]
    return head + (f" ({', '.join(rest)})" if rest else "")


def query_scores(con, results, models, cells):
    """Per (benchmark, model, task, revision): the mean main score over the
    cells present, how many cells, and whether any is flagged trained_on."""
    con.execute("create or replace temp table bt(benchmark varchar, task_name varchar, split varchar, subset varchar)")
    con.executemany("insert into bt values (?, ?, ?, ?)",
                    [(b, t, sp, ss) for b, cs in cells.items() for t, _, sp, ss, _ in cs])
    con.execute("create or replace temp table wanted(model_name varchar)")
    con.executemany("insert into wanted values (?)", [(m,) for m in models])
    base = f"hf://datasets/mteb/results@{results}/data/train-*.parquet"
    return con.execute(f"""
        select bt.benchmark, r.model_name, r.task_name, r.model_revision,
               avg(r.score), count(*), bool_or(coalesce(r.trained_on, false))
        from '{base}' r
        join bt on r.task_name = bt.task_name and r.split = bt.split and r.subset = bt.subset
        where r.model_name in (select model_name from wanted)
        group by 1, 2, 3, 4
        order by 1, 2, 3, 4""").fetchall()


def aggregate(rows, cells, pin):
    """One benchmark's entry for one model, from its query rows."""
    per_rev = collections.Counter(r[3] for r in rows)

    def priority(rev):
        return (rev != pin, rev in MODEL_CARD, -per_rev[rev], rev)

    need = collections.Counter(c[0] for c in cells)
    chosen = {}
    for _, _, task, rev, score, n, trained in sorted(rows, key=lambda r: priority(r[3])):
        if n == need[task]:            # a task counts only with every cell present
            chosen.setdefault(task, (rev, score, trained))
    types = {c[0]: c[1] for c in cells}
    metrics = {c[0]: c[4] for c in cells}
    categories = {}
    for ttype in sorted(set(types.values())):
        names = sorted(t for t, ty in types.items() if ty == ttype)
        got = [chosen[t] for t in names if t in chosen]
        complete = len(got) == len(names)
        categories[ttype] = {
            "value": round(sum(g[1] for g in got) / len(got), 6) if complete else None,
            "n_tasks": len(got), "of": len(names),
            "metric": metric_text([metrics[t] for t in names]),
            "n_trained_on": sum(1 for g in got if g[2]),
            "n_model_card": sum(1 for g in got if g[0] in MODEL_CARD),
        }
    complete = len(chosen) == len(types)
    return {
        "mean_task": round(sum(v[1] for v in chosen.values()) / len(types), 6) if complete else None,
        "mean_task_type": (round(sum(c["value"] for c in categories.values()) / len(categories), 6)
                           if complete else None),
        "n_tasks": len(chosen), "of": len(types),
        "evaluated_revisions": dict(sorted(collections.Counter(v[0] for v in chosen.values()).items())),
        "categories": categories,
    }


GATED = (401, 403)


def hf_json(url, missing_ok=False):
    """A Hugging Face JSON resource, or None if the repository is gated
    (401/403), or the file is absent (404) and missing_ok. Any other failure
    (a timeout, DNS, a server error) stops the run: publishing it as
    "unverified" would hide it."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as err:
        if err.code in GATED or (missing_ok and err.code == 404):
            return None
        raise SystemExit(f"{url}: HTTP {err.code}")
    except (urllib.error.URLError, TimeoutError) as err:
        raise SystemExit(f"{url}: {err}")


def hf_tree(repo, rev):
    """{file: LFS sha256} for a revision's top-level weight files, or None
    if the repository is gated."""
    files = hf_json(f"https://huggingface.co/api/models/{repo}/tree/{rev}")
    if files is None:
        return None
    # A gated repository answers with masked hashes ("****..."): unreadable.
    return {f["path"]: f["lfs"]["oid"] for f in files
            if f.get("lfs") and f["path"] in WEIGHT_FILES and SHA256.fullmatch(f["lfs"]["oid"])}


def revision_match(repo, pin, revisions, cache):
    """How the revisions MTEB evaluated relate to sidekick's pin."""
    run = [r for r in revisions if r not in MODEL_CARD]
    if not run:
        return "model card only"
    if all(r == pin for r in run):
        return "exact"
    verdicts = set()
    for rev in run:
        if rev == pin:
            continue
        for key in ((repo, pin), (repo, rev)):
            if key not in cache:
                cache[key] = hf_tree(*key)
        a, b = cache[(repo, pin)], cache[(repo, rev)]
        common = set(a or {}) & set(b or {})
        if a is None or b is None or not common:
            verdicts.add("unverified")
        else:
            verdicts.add("same-weights" if all(a[f] == b[f] for f in common) else "different")
    for v in ("different", "unverified", "same-weights"):
        if v in verdicts:
            return v


def model_context(mteb, name, revisions):
    """The longest input the MTEB run embedded, and where that comes from.
    mteb's wrappers load text models with sentence-transformers, which
    truncates at the max_seq_length the mteb loader passes, else at the one
    in the evaluated revision's sentence_bert_config.json. Where neither
    states one (no such config, or a gated repository), the figure is the
    model's advertised max_tokens, labelled "advertised": an upper bound,
    not necessarily what the run used."""
    try:
        meta = mteb.get_model_meta(name)
    except Exception:
        return None, None
    kw = getattr(meta, "loader_kwargs", None) or {}
    if kw.get("max_seq_length"):
        return int(kw["max_seq_length"]), "mteb loader"
    run = [r for r, _ in sorted(revisions.items(), key=lambda kv: -kv[1]) if r not in MODEL_CARD]
    if run:
        config = hf_json(f"https://huggingface.co/{name}/raw/{run[0]}/sentence_bert_config.json", missing_ok=True)
        if config and config.get("max_seq_length"):
            return int(config["max_seq_length"]), "sentence-transformers config"
    return (int(meta.max_tokens), "advertised") if meta.max_tokens else (None, None)


def cross_check(entries):
    """Compare complete means with the live leaderboard backend."""
    seen = {}
    for e in entries:
        if e["mean_task"] is None:
            continue
        name = e["mteb_model"]
        if name not in seen:
            try:
                with urllib.request.urlopen(f"https://mteb-leaderboard-backend.hf.space/v1/models/{name}/scores",
                                            timeout=60) as r:
                    seen[name] = {row["benchmarkName"]: row for row in json.load(r)["rows"]}
            except (urllib.error.URLError, TimeoutError, KeyError) as err:
                print(f"cross-check: {name}: {err}", file=sys.stderr)
                seen[name] = {}
        live = seen[name].get(e["benchmark"])
        if live is None or live.get("meanTask") is None:
            print(f"cross-check: {name} {e['benchmark']}: no live value")
            continue
        diff = abs(live["meanTask"] - e["mean_task"])
        print(f"cross-check: {name} {e['benchmark']}: pinned {e['mean_task']:.6f}, live {live['meanTask']:.6f}"
              + ("" if diff < 1e-6 else f"  DIFFERS by {diff:.2e}"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default=RESULTS, help="mteb/results dataset commit")
    p.add_argument("--out", type=Path, default=DATA / "benchmarks.mteb.json")
    p.add_argument("--read-on", help="the date to record as read (default: the existing file's when --results "
                                      "is its pin, so a rerun at the same pin is byte-identical; otherwise today)")
    p.add_argument("--cross-check", action="store_true")
    args = p.parse_args()

    if args.read_on is None:
        old = json.loads(args.out.read_text()) if args.out.exists() else {}
        same = old.get("results", {}).get("revision") == args.results
        args.read_on = old["read_on"] if same else datetime.date.today().isoformat()

    logging.disable(logging.WARNING)
    import duckdb
    import mteb

    models = tomllib.loads((DATA / "models.toml").read_text())["models"]
    pins = {}
    for mid, m in models.items():
        if m["task"] in TASKS and "@" in m.get("source", ""):
            repo, rev = m["source"].split("@", 1)
            pins.setdefault(repo, []).append((mid, rev))

    cells, hashes = {}, {}
    for b in BENCHMARKS:
        cells[b], hashes[b] = benchmark_cells(mteb, b)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = query_scores(con, args.results, sorted(pins), cells)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append(r)

    source_url = f"https://huggingface.co/datasets/mteb/results/tree/{args.results}"
    trees, entries = {}, []
    for repo in sorted(pins):
        evaluated = collections.Counter()
        for (b, name), found in by.items():
            if name == repo:
                evaluated.update(r[3] for r in found)
        context, context_source = model_context(mteb, repo, evaluated)
        for mid, pin in pins[repo]:
            for b in BENCHMARKS:
                found = by.get((b, repo))
                if not found:
                    continue
                agg = aggregate(found, cells[b], pin)
                if agg["n_tasks"] == 0:
                    continue
                card = sum(c["n_model_card"] for c in agg["categories"].values())
                entries.append({
                    "model": mid, "mteb_model": repo, "benchmark": b,
                    "mteb_version": mteb.__version__, "task_set_hash": hashes[b],
                    "sidekick_revision": pin,
                    "revision_match": revision_match(repo, pin, agg["evaluated_revisions"], trees),
                    "prompting": PROMPTING.get(repo),
                    "model_context": context, "model_context_source": context_source,
                    "source_kind": "mixed" if card else "mteb",
                    "source_url": source_url, "read_on": args.read_on,
                    **agg,
                })
    order = list(models)
    entries.sort(key=lambda e: (order.index(e["model"]), BENCHMARKS.index(e["benchmark"])))

    out = {
        "about": "Generated by tools/fetch_benchmarks.py from the mteb/results dataset; do not edit by hand.",
        "results": {"dataset": "mteb/results", "revision": args.results, "url": source_url},
        "mteb_version": mteb.__version__,
        "read_on": args.read_on,
        "scale": "fraction",
        "benchmarks": {b: {"task_set_hash": hashes[b], "n_tasks": len({c[0] for c in cells[b]}),
                           "categories": {t: len({c[0] for c in cells[b] if c[1] == t})
                                          for t in sorted({c[1] for c in cells[b]})}}
                       for b in BENCHMARKS},
        "entries": entries,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(entries)} entries ({len({e['model'] for e in entries})} models) -> {args.out}")
    if args.cross_check:
        cross_check(entries)


if __name__ == "__main__":
    main()
