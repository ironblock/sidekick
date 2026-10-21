"""Measure whether published MTEB scores carry over to sidekick's builds:
run MTEB tasks through a running `sidekickd`, and compare each score with
the one the MTEB results repository publishes for the same model and task.

docs/MODELS.md publishes each embedder's MTEB scores beside its builds'
fidelity grades (tools/fetch_benchmarks.py). Those scores were measured
with sentence-transformers, at each model's full context, with MTEB's
prompts. This tool measures the same tasks on a sidekick build: its Core ML
or ONNX graph, its tokenizer, and its input cap (`max_seq_len`).

How a task is run (`run`):
- The encoder is mteb's own wrapper for the model (mteb's model registry,
  at the version below), loaded at sidekick's pinned revision, so mteb
  chooses every prompt exactly as it did for the published run: a prompt
  per task type (EmbeddingGemma), an instruction per task (Qwen3), or a
  query prefix on retrieval and reranking only (bge). Only the wrapper's
  sentence-transformers model is replaced: its `encode` resolves the prompt
  as sentence-transformers would, puts it before each text, and sends the
  text to `/v1/embeddings` with `input_type: "none"` (sidekick adds no
  prompt of its own) and `truncate: "END"` (an input over the build's cap
  is cut to its first `max_seq_len` tokens, as a capped build serves it).
- Each distinct input is also counted with `/tokenize` (same text, same
  `input_type`), to report the share over the build's cap.
- A task's score is its main metric, averaged over the evaluation splits
  and subsets MTEB(eng, v2) scores it on, as the published score is.

The adapter check (`reference`, then `run --reference`): the same task with
the unmodified wrapper, sentence-transformers in fp32 on the CPU. On a task
whose inputs all fit the build's cap, a sidekick score that matches it
shows the adapter reproduces mteb's run, so a difference elsewhere belongs
to the build or the cap. `run --reference` also compares every input's
vector with the reference's (worst and mean cosine).

A published run may have used another revision of the checkpoint than
sidekick's pin. `reference --revision <rev>` compares that revision's files
with the pin's, file by file (LFS sha256 or git blob id), and runs the
tasks given at it; `record` keeps both as `revision_checks`, not as the
fp32 run.

`record` collects the runs into docs/data/mteb_transfer.json, with each
published per-task score read from the same pinned mteb/results commit as
docs/data/benchmarks.mteb.json, at the revision tools/fetch_benchmarks.py's
`choose` picks for it there, and tools/models_table.py --docs publishes them
in docs/MODELS.md.

Usage (a Python 3.12 virtualenv with mteb==2.26.14, sentence-transformers
and duckdb; the models' checkpoints at sidekick's pinned revisions):

    # sidekickd serving the build under test, on a port of your choosing
    python tools/mteb_transfer.py run --url http://127.0.0.1:8811 \\
        --model bge-small-en-v1.5 --path coreml-ane --out <dir> \\
        [--tasks STSBenchmark ...] [--build <installed model dir>] [--reference <dir>]
    python tools/mteb_transfer.py reference --model bge-small-en-v1.5 --out <dir> [--tasks STSBenchmark]
        [--revision <rev>] [--batch N]
    python tools/mteb_transfer.py record <dir> [<dir> ...] [--out docs/data/mteb_transfer.json]

    --path:      the compute path the daemon serves the model on: coreml-ane,
                 coreml-gpu, coreml-cpu or onnx-cpu (checked against /v1/models)
    --build:     the installed model directory, to record a fingerprint of the
                 build's files (sha256 over its weights, tokenizer and manifest)
    --reference: a `reference` output directory; compare vectors with it

Each run writes one JSON file per task to --out, with the daemon's version
as it reports it (its `sidekick-version` response header). A Core ML model's
`compute_units` come from its manifest or the daemon's config
(`[models."<id>"] compute_units`); /v1/models reports what is in use.
Requests carry 32 inputs; raise the daemon's `request_timeout_secs` for a
build that takes long inputs (32 inputs of 4,096 tokens take minutes on
onnx-cpu). On an M1 Max, the six default tasks took 7 minutes for
bge-small-en-v1.5 on onnx-cpu, 26 for EmbeddingGemma and 85 for
Qwen3-Embedding-0.6B (at 4,096 tokens); on the ANE, 3, 8 and 28.
"""

import argparse
import base64
import datetime
import hashlib
import http.client
import json
import logging
import platform
import subprocess
import sys
import time
import tomllib
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"
OUT = DATA / "mteb_transfer.json"

MTEB_VERSION = "2.26.14"
BENCHMARK = "MTEB(eng, v2)"
# Fast MTEB(eng, v2) tasks across its categories, and two with inputs past
# 512 tokens: ArguAna (arguments as queries and documents) and STS22.v2
# (news articles; English subset).
TASKS = ["Banking77Classification", "STSBenchmark", "ArXivHierarchicalClusteringP2P", "AskUbuntuDupQuestions",
         "ArguAna", "STS22.v2"]
CHECK_TASKS = ["STSBenchmark"]
PATHS = {("coreml", "cpu_and_ne"): "coreml-ane", ("coreml", "cpu_and_gpu"): "coreml-gpu",
         ("coreml", "cpu_only"): "coreml-cpu", ("onnx", "cpu"): "onnx-cpu"}
BATCH = 32


def models_toml():
    return tomllib.loads((DATA / "models.toml").read_text())["models"]


def checkpoint(model):
    """(repo, revision) sidekick converts `model` from."""
    m = models_toml().get(model)
    if m is None or "@" not in m.get("source", ""):
        raise SystemExit(f"{model}: not in docs/data/models.toml with a pinned source")
    repo, rev = m["source"].split("@", 1)
    return repo, rev


def benchmark_tasks(names):
    """The tasks as MTEB(eng, v2) defines them (its splits and subsets)."""
    import mteb
    if mteb.__version__ != MTEB_VERSION:
        raise SystemExit(f"mteb {mteb.__version__}: the published scores' benchmark definitions are mteb "
                         f"{MTEB_VERSION}'s")
    by = {t.metadata.name: t for t in mteb.get_benchmark(BENCHMARK).tasks}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"not in {BENCHMARK}: {', '.join(missing)}")
    return [by[n] for n in names]


def task_score(task, result):
    """The task's main metric, averaged over the benchmark's (split, subset)
    cells, as tools/fetch_benchmarks.py averages a published score."""
    values = []
    for split in task.eval_splits:
        cells = {s["hf_subset"]: s["main_score"] for s in result.scores[split]}
        for subset in task.hf_subsets:
            values.append(cells[subset])
    return sum(values) / len(values)


def load_wrapper(model, revision=None):
    """mteb's own wrapper for the model, at sidekick's pinned revision (or
    `revision`), on the CPU in fp32. The wrapper takes its prompts from that
    revision's sentence-transformers config."""
    import mteb
    repo, rev = checkpoint(model)
    meta = mteb.get_model_meta(repo).model_copy(update={"revision": revision or rev})
    wrapper = meta.load_model(device="cpu")
    wrapper.model.float()
    return wrapper


class Sidekick:
    """/v1/embeddings and /tokenize on one model, over one connection."""

    def __init__(self, url, model):
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.model = u.hostname, u.port or 80, model
        self.conn = None
        self.version = None
        self.reset()

    def reset(self):
        self.counts, self.vectors = {}, {}

    def call(self, method, path, body=None):
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
            try:
                data = None if body is None else json.dumps(body).encode()
                self.conn.request(method, path, body=data, headers={"content-type": "application/json"})
                r = self.conn.getresponse()
                payload = r.read()
            except (http.client.HTTPException, ConnectionError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
                continue
            if r.status != 200:
                raise SystemExit(f"{method} {path}: HTTP {r.status}: {payload[:500]!r}")
            version = r.getheader("sidekick-version")
            if version:
                if self.version not in (None, version):
                    raise SystemExit(f"the daemon's version changed during the run: {self.version}, {version}")
                self.version = version
            return json.loads(payload)

    def info(self):
        for m in self.call("GET", "/v1/models")["data"]:
            if m["id"] == self.model:
                return m
        raise SystemExit(f"{self.model} isn't served at {self.host}:{self.port}")

    def count(self, text):
        if text not in self.counts:
            r = self.call("POST", "/tokenize", {"model": self.model, "prompt": text, "input_type": "none"})
            self.counts[text] = (r["count"], r["max_model_len"])

    def embed(self, texts, keep=False):
        out = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            r = self.call("POST", "/v1/embeddings", {"model": self.model, "input": chunk, "input_type": "none",
                                                     "truncate": "END", "encoding_format": "base64"})
            out += [base64.b64decode(d["embedding"]) for d in sorted(r["data"], key=lambda d: d["index"])]
            for t in chunk:
                self.count(t)
        import numpy as np
        vecs = np.stack([np.frombuffer(b, dtype="<f4") for b in out]) if out else np.zeros((0, 0), "f4")
        if keep:
            for t, v in zip(texts, vecs):
                self.vectors[t] = v
        return vecs


def full_texts(st, inputs, prompt_name, prompt):
    """The texts as sentence-transformers would embed them: the prompt
    (explicit, named, or the model's default) before each."""
    p = st._resolve_prompt(prompt, prompt_name)
    return [(p or "") + t for t in inputs]


def patch_sidekick(wrapper, client, keep):
    """Route the wrapper's sentence-transformers encode to sidekick. mteb's
    wrappers call encode, encode_query or encode_document; the latter two
    name their prompt and call encode."""
    st = wrapper.model

    def encode(inputs=None, prompt_name=None, prompt=None, **kwargs):
        inputs = kwargs.pop("sentences", inputs)
        single = isinstance(inputs, str)
        texts = full_texts(st, [inputs] if single else list(inputs), prompt_name, prompt)
        vecs = client.embed(texts, keep=keep)
        return vecs[0] if single else vecs

    st.encode = encode


def patch_capture(wrapper, store):
    """Keep every vector the unmodified encode returns, by its full text."""
    import numpy as np
    import torch
    st = wrapper.model
    original = st.encode

    def encode(inputs=None, prompt_name=None, prompt=None, **kwargs):
        inputs = kwargs.pop("sentences", inputs)
        single = isinstance(inputs, str)
        texts = full_texts(st, [inputs] if single else list(inputs), prompt_name, prompt)
        p = st._resolve_prompt(prompt, prompt_name)
        out = original(inputs, prompt=p, **kwargs)
        arr = out.float().cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out, dtype=np.float32)
        for t, v in zip(texts, arr.reshape(len(texts), -1)):
            store[t] = v
        return out

    st.encode = encode


def evaluate(wrapper, task, batch=BATCH):
    import mteb
    t = time.perf_counter()
    result = mteb.evaluate(wrapper, task, cache=None, overwrite_strategy="always", show_progress_bar=False,
                           encode_kwargs={"batch_size": batch})
    return task_score(task, result.task_results[0]), time.perf_counter() - t


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return chip, f"macOS {platform.mac_ver()[0]}"


def fingerprint(build):
    """sha256 over the build's weights, tokenizer and manifest: (relative
    path, file sha256) pairs in path order, first 12 hex digits."""
    build = Path(build).resolve()
    files = sorted(p for p in build.rglob("*") if p.is_file() and (
        p.name in ("weight.bin", "tokenizer.json", "manifest.toml", "model.mil")
        or p.suffix == ".onnx" or p.name.endswith((".onnx_data", ".onnx.data"))))
    h = hashlib.sha256()
    for p in files:
        f = hashlib.sha256()
        with p.open("rb") as fh:
            while block := fh.read(1 << 24):
                f.update(block)
        h.update(f"{p.relative_to(build)}\0{f.hexdigest()}\n".encode())
    return h.hexdigest()[:12]


def write(out, name, record):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")


def cmd_run(args):
    import mteb
    import numpy as np
    client = Sidekick(args.url, args.model)
    info = client.info()
    path = PATHS.get((info.get("backend"), info.get("compute_units")))
    if path != args.path:
        raise SystemExit(f"{args.model} is served as {info.get('backend')}/{info.get('compute_units')}, not "
                         f"{args.path}")
    repo, rev = checkpoint(args.model)
    chip, os_ = machine()
    common = {"model": args.model, "mteb_model": repo, "revision": rev, "path": path,
              "backend": info["backend"], "compute_units": info["compute_units"], "max_seq_len": info["max_seq_len"],
              "build": fingerprint(args.build) if args.build else None,
              "chip": chip, "os": os_, "mteb_version": mteb.__version__, "benchmark": BENCHMARK}
    wrapper = load_wrapper(args.model)
    for task in benchmark_tasks(args.tasks):
        name = task.metadata.name
        ref = None
        if args.reference:
            f = Path(args.reference) / f"{args.model}__{name}.npz"
            ref = np.load(f) if f.exists() else None
        client.reset()
        patch_sidekick(wrapper, client, keep=ref is not None)
        score, seconds = evaluate(wrapper, task)
        counts = list(client.counts.values())
        cap = info["max_seq_len"]
        record = dict(common, sidekick=client.version, task=name, category=task.metadata.type,
                      metric=task.metadata.main_score,
                      score=round(score, 6), seconds=round(seconds, 1), inputs=len(counts),
                      over_cap=sum(1 for c, _ in counts if c > cap), longest=max((c for c, _ in counts), default=0),
                      date=datetime.date.today().isoformat())
        if ref is not None:
            texts = json.loads(str(ref["texts"]))
            cos = []
            for t, v in zip(texts, ref["vectors"]):
                w = client.vectors.get(t)
                if w is not None:
                    cos.append(float(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))))
            record["vs_reference"] = {"inputs": len(cos), "worst_cosine": min(cos), "mean_cosine": sum(cos) / len(cos)}
        print(json.dumps({k: record[k] for k in ("model", "path", "task", "score", "seconds", "inputs", "over_cap")}
                         | ({"vs_reference": record["vs_reference"]} if "vs_reference" in record else {})), flush=True)
        write(Path(args.out), f"{args.model}__{path}__{name}", record)


def cmd_reference(args):
    import mteb
    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    repo, pin = checkpoint(args.model)
    rev = args.revision or pin
    chip, os_ = machine()
    out = Path(args.out)
    wrapper = load_wrapper(args.model, rev)
    runtime = (f"sentence-transformers {sentence_transformers.__version__}, transformers {transformers.__version__}, "
               f"torch {torch.__version__}, fp32 on the CPU")
    common = {"model": args.model, "mteb_model": repo, "revision": rev, "path": "reference", "runtime": runtime,
              "max_seq_len": wrapper.model.max_seq_length, "chip": chip, "os": os_,
              "mteb_version": mteb.__version__, "benchmark": BENCHMARK}
    if rev != pin:
        # Another revision than the one sidekick converts: what differs
        # between the two, file by file. Its runs aren't the reference.
        write(out, f"{args.model}__revision-{rev[:12]}", dict(common, pin=pin, files=compare_revisions(repo, pin, rev)))
    for task in benchmark_tasks(args.tasks):
        name = task.metadata.name
        store = {}
        patch_capture(wrapper, store)
        score, seconds = evaluate(wrapper, task, args.batch)
        texts = list(store)
        record = dict(common, task=name, category=task.metadata.type, metric=task.metadata.main_score,
                      score=round(score, 6), seconds=round(seconds, 1), inputs=len(texts),
                      date=datetime.date.today().isoformat())
        print(json.dumps({k: record[k] for k in ("model", "revision", "task", "score", "seconds", "inputs")}),
              flush=True)
        if rev == pin:
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"{args.model}__{name}.npz", texts=json.dumps(texts),
                     vectors=np.stack([store[t] for t in texts]))
            write(out, f"{args.model}__reference__{name}", record)
        else:
            write(out, f"{args.model}__revision-{rev[:12]}__{name}", record)


def compare_revisions(repo, a, b):
    """{"same": [...], "different": [...], "only_in_<rev>": [...]}: every
    file of two revisions, compared by its LFS sha256 (or git blob id)."""
    from huggingface_hub import HfApi
    api = HfApi()

    def tree(rev):
        return {f.path: (f.lfs.sha256 if f.lfs else f.blob_id)
                for f in api.list_repo_tree(repo, revision=rev, recursive=True) if hasattr(f, "size")}

    ta, tb = tree(a), tree(b)
    return {"same": sorted(p for p in ta if tb.get(p) == ta[p]),
            "different": sorted(p for p in ta if p in tb and tb[p] != ta[p]),
            "only_in_pin": sorted(set(ta) - set(tb)), "only_in_revision": sorted(set(tb) - set(ta))}


def published(models, tasks, results):
    """{(sidekick model, task): published score} at the pinned mteb/results
    commit, each task's revision chosen as tools/fetch_benchmarks.py chooses
    it for docs/data/benchmarks.mteb.json (its `choose`, over the whole
    benchmark, so the most-tasks tiebreak counts the same tasks)."""
    import duckdb
    import mteb
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fetch_benchmarks as fb
    cells, _ = fb.benchmark_cells(mteb, BENCHMARK)
    pins = {checkpoint(m)[0]: (m, checkpoint(m)[1]) for m in models}
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = fb.query_scores(con, results, sorted(pins), {BENCHMARK: cells})
    out = {}
    for repo, (model, pin) in pins.items():
        chosen = fb.choose([r for r in rows if r[1] == repo], cells, pin)
        for task in tasks:
            if task in chosen:
                rev, score, trained = chosen[task]
                out[(model, task)] = {"score": round(score, 6), "revision": rev,
                                      "source": "model card" if rev in fb.MODEL_CARD else "mteb",
                                      "trained_on": bool(trained)}
    return out


def cmd_record(args):
    import mteb
    records, revisions = [], {}
    for d in args.dirs:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text())
            if r["revision"] == checkpoint(r["model"])[1]:
                records.append(r)
            elif "files" in r:              # another revision's comparison with the pin
                revisions.setdefault((r["model"], r["revision"]), {"scores": []}).update(
                    {k: r[k] for k in ("model", "pin", "revision", "files", "runtime", "max_seq_len", "chip", "os",
                                       "mteb_version")})
            else:                           # a task run at that revision
                revisions.setdefault((r["model"], r["revision"]), {"scores": []})["scores"].append(
                    {k: r[k] for k in ("task", "score", "inputs", "date")})
    if not records:
        raise SystemExit("no runs found")
    bench = json.loads((DATA / "benchmarks.mteb.json").read_text())
    results = bench["results"]["revision"]
    models = sorted({r["model"] for r in records}, key=list(models_toml()).index)
    tasks = [t for t in TASKS if t in {r["task"] for r in records}]
    tasks += sorted({r["task"] for r in records} - set(tasks))
    pub = published(models, tasks, results)
    meta = {t.metadata.name: t for t in benchmark_tasks(tasks)}
    old = json.loads(args.out.read_text()) if args.out.exists() else {}
    keep_date = old.get("results", {}).get("revision") == results
    runs, scores = {}, []
    for r in records:
        key = (r["model"], r["path"])
        if r["path"] == "reference":
            run = {k: r[k] for k in ("model", "path", "revision", "runtime", "max_seq_len", "chip", "os",
                                     "mteb_version", "date")}
        else:
            run = {k: r[k] for k in ("model", "path", "revision", "backend", "compute_units", "max_seq_len", "build",
                                     "sidekick", "chip", "os", "mteb_version", "date")}
        prev = runs.get(key)
        if prev is not None:
            if {k: v for k, v in prev.items() if k != "date"} != {k: v for k, v in run.items() if k != "date"}:
                raise SystemExit(f"{key}: runs of different builds or setups: {prev} and {run}")
            prev["date"] = max(prev["date"], run["date"])     # the last day it ran
        else:
            runs[key] = run
        p = pub.get((r["model"], r["task"]))
        s = {"model": r["model"], "path": r["path"], "task": r["task"], "category": r["category"],
             "metric": r["metric"], "score": r["score"], "seconds": r["seconds"], "inputs": r["inputs"],
             **({} if r["path"] == "reference" else {"over_cap": r["over_cap"], "longest": r["longest"]}),
             "published": p["score"] if p else None, "published_revision": p["revision"] if p else None,
             "published_source": p["source"] if p else None,
             "published_trained_on": p["trained_on"] if p else None}
        if "vs_reference" in r:
            s["vs_reference"] = {k: round(v, 8) if isinstance(v, float) else v for k, v in r["vs_reference"].items()}
        scores.append(s)
    order = list(models_toml())
    path_order = ["reference", "coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
    scores.sort(key=lambda s: (order.index(s["model"]), tasks.index(s["task"]), path_order.index(s["path"])))
    out = {
        "about": "Written by tools/mteb_transfer.py record; do not edit by hand.",
        "benchmark": BENCHMARK,
        "mteb_version": mteb.__version__,
        "results": bench["results"],
        "read_on": old["read_on"] if keep_date and old.get("read_on") else datetime.date.today().isoformat(),
        "scale": "fraction",
        "tasks": {t: {"category": meta[t].metadata.type, "metric": meta[t].metadata.main_score,
                      "splits": list(meta[t].eval_splits), "subsets": list(meta[t].hf_subsets)} for t in tasks},
        "runs": sorted(runs.values(), key=lambda r: (order.index(r["model"]), path_order.index(r["path"]))),
        "scores": scores,
        # Other revisions than sidekick's pin (the published runs'),
        # compared with it file by file, and any tasks run at them.
        "revision_checks": sorted(
            ({**v, "scores": sorted(v["scores"], key=lambda s: tasks.index(s["task"]) if s["task"] in tasks else 99)}
             for v in revisions.values()), key=lambda v: (order.index(v["model"]), v["revision"])),
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(scores)} scores, {len(runs)} runs -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run tasks through a running sidekickd")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--path", required=True, choices=sorted(set(PATHS.values())))
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", nargs="+", default=TASKS)
    r.add_argument("--build", help="the installed model directory, to fingerprint")
    r.add_argument("--reference", help="a `reference` output directory to compare vectors with")
    f = sub.add_parser("reference", help="the same tasks with sentence-transformers, fp32 on the CPU")
    f.add_argument("--model", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--tasks", nargs="*", default=CHECK_TASKS)
    f.add_argument("--revision", help="another revision of the checkpoint (the published run's, say): compare its "
                                      "files with sidekick's pin, and run the tasks at it")
    f.add_argument("--batch", type=int, default=BATCH,
                   help="inputs per sentence-transformers batch (lower it for long inputs at a long context)")
    c = sub.add_parser("record", help="collect runs into docs/data/mteb_transfer.json")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    logging.disable(logging.WARNING)
    {"run": cmd_run, "reference": cmd_reference, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
