"""Generate the model card (README.md) for a sidekick build's Hugging Face
repository (#137; docs/PUBLISHING.md).

One repository per model per backend (Core ML, ONNX), holding every build of
that model for that backend, each in a directory named for its variant
(`fp16/`, `fp16-max2048/`, ...) laid out exactly as sidekick installs it.
The card is generated from sidekick's committed data, never written by hand:
- docs/data/models.toml: the model's name, task, checkpoint and served path;
- the committed manifest (examples/manifests/, examples/classifiers/): the
  buckets, length cap, prefixes and the converter that produces the build;
- docs/data/models.json: the fidelity grades per path and chip, and the
  benchmark scores;
- docs/data/licences.toml: the licence, training data and publishing tier,
  from which the notice (tier 2a) or the consent terms (tier 2b) are built;
- docs/data/ledger.json: each build's files, sha256 and size, what produced
  it, and the grades its exact files were graded with;
- docs/data/publishing.toml: publisher exports listed as ONNX alternatives.

Usage:
    python3 tools/model_card.py MODEL [--backend coreml|onnx] [--build ID ...]
        [--owner HF_ACCOUNT] [--out DIR]
    python3 tools/model_card.py --check

Without --out the card is printed. With --out, DIR receives the README.md,
one `<variant>.sha256` per build (`shasum -a 256 -c` format, paths relative
to the variant's directory), a NOTICE where the licence requires one, and
for a tier-2b model TERMS.md, the consent terms fetch shows and records.
Copies of documents the licence requires (licences.toml's `ship_copies`)
are fetched from their source at upload, not generated here.
--check renders every publishable model and backend the ledger holds builds
for, and fails on any that can't be rendered, or on a tier-3 build marked
published.

Standard library only.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger as ledger_mod  # noqa: E402
import licences as licences_mod  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"
GITHUB = "https://github.com/ironblock/sidekick"
OWNER = "<owner>"
FETCH = "sidekickd fetch"

PATH_NAME = {"coreml-ane": "Neural Engine", "coreml-gpu": "GPU", "coreml-cpu": "CPU", "onnx-cpu": "CPU (ONNX Runtime)"}
BACKEND_NAME = {"coreml": "Core ML", "onnx": "ONNX"}
BACKEND_PATHS = {"coreml": ("coreml-ane", "coreml-gpu", "coreml-cpu"), "onnx": ("onnx-cpu",)}
CHIP_NAME = {}  # filled from runs.toml's [chips]
PIPELINE = {"embed": "feature-extraction", "rerank": "text-ranking"}
COMPUTE_UNITS = {None: "coreml-ane", "cpu_and_ne": "coreml-ane", "cpu_and_gpu": "coreml-gpu",
                 "cpu_only": "coreml-cpu", "all": "coreml-ane"}
# Converters the committed manifest doesn't name (the conversion library's
# BERT recipe), by model.
CONVERTER = {"all-minilm-l6-v2": "tools/convert_bert_embedder.py", "e5-small-v2": "tools/convert_bert_embedder.py"}

CONCERN_TEXT = {
    "copyleft-data": "share-alike (CC BY-SA) terms",
    "unlicensed-data": "no licence that a primary source states (scraped or user-generated content)",
    "undisclosed-data": "sources the model's authors don't disclose",
    "unnamed-synthetic-generator": "synthetic data from a generator the authors don't name, whose output terms can't be checked",
}


# --- inputs ------------------------------------------------------------------

def load_inputs():
    models = tomllib.loads((DATA / "models.toml").read_text())["models"]
    results = json.loads((DATA / "models.json").read_text())
    runs = tomllib.loads((DATA / "runs.toml").read_text())
    CHIP_NAME.update(runs.get("chips", {}))
    return {
        "models": models,
        "results": results,
        "licences": licences_mod.load()["models"],
        "ledger": ledger_mod.load(),
        "publishing": tomllib.loads((DATA / "publishing.toml").read_text()),
    }


def committed_manifest(model, backend):
    """(parsed manifest, its text) for the model's committed manifest; for
    ONNX, the onnx.toml beside it where one exists."""
    base = REPO / "examples" / "manifests" / model
    names = (["onnx.toml"] if backend == "onnx" else []) + ["manifest.toml"]
    paths = [base / n for n in names] + [REPO / "examples" / "classifiers" / model / "classifier.toml"]
    for p in paths:
        if p.exists():
            text = p.read_text()
            return tomllib.loads(text), text
    return {}, ""


def split_source(source):
    repo, _, rev = source.partition("@")
    return repo, rev


def hf_url(repo, rev=None):
    return f"https://huggingface.co/{repo}" + (f"/tree/{rev}" if rev else "")


def repo_id(owner, model, backend):
    return f"{owner}/{model}-{backend}"


def mb(n):
    if n is None:
        return "unknown"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.1f} kB" if n >= 1e3 else f"{n} B"


def chip(name):
    return CHIP_NAME.get(name, name)


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def yq(s):
    """A YAML double-quoted scalar (JSON strings are valid YAML)."""
    return json.dumps(s, ensure_ascii=False)


# --- the parts of a card -----------------------------------------------------

def task_of(m, manifest):
    if m["task"] == "classify":
        return manifest.get("task", "text-classification")
    return PIPELINE.get(m["task"], m["task"])


def result_rows(results, model, backend):
    rows = [r for r in results["rows"] if r["model"] == model and r["path"] in BACKEND_PATHS[backend]]
    return sorted(rows, key=lambda r: (r["chip"] != "Apple M1 Max", r["chip"],
                                       BACKEND_PATHS[backend].index(r["path"]), r["precision"]))


def measure(r):
    if r["error_text"] is None:
        return "not graded"
    if r["error_kind"] == "dp":
        return f"worst |Δp| {r['error_text']}" + (f", {r['ratio']}× its fp16 ceiling" if r.get("ratio") else "")
    return f"worst cosine {r['error_text']}"


def consent_terms(m, lic):
    """The terms a user accepts to download a tier-2b build: built from the
    audit's concerns and the datasets behind them. Plain Markdown, used
    both as the gate's prompt and on the card."""
    repo, rev = split_source(m["source"])
    items = []
    nc = [d for d in lic["training_data"] if d["terms"] == "non-commercial"]
    if nc:
        names = "; ".join(f"{d['name']}: {d['licence']}" if "(" in d["licence"] else f"{d['name']} ({d['licence']})"
                          for d in nc)
        items.append(
            f"The original model was trained on data whose terms restrict it to non-commercial use: {names}. "
            "Whether such terms bind a model trained on the data is legally unsettled. You decide whether your "
            "use is permitted, and the publisher of this repository gives no assurance that it is.")
    for t in lic["custom_terms"]:
        name = t["name"].split(" (")[0]
        if "Policy" in name:
            items.append(f"You won't use these files for anything the [{name}]({t['url']}) prohibits, and you "
                         "will pass it on to anyone you give them to.")
        else:
            items.append(f"You receive and use these files under the [{name}]({t['url']}), and you will give a "
                         "copy of it to anyone you give them to.")
    if "use-policy" in lic["concerns"] and not lic["custom_terms"]:
        items.append("You will follow the Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy).")
    items.append(
        f"These files are a conversion of [{repo}]({hf_url(repo, rev)}), licensed to you under its licence "
        f"({lic['licence']}), and are provided as is, without warranty of any kind.")
    items.append(
        "Hugging Face shares your account's username and email address with this repository's owner when you "
        "accept, and records that you accepted.")
    return items


def terms_text(m, lic):
    """TERMS.md: the consent terms on their own, the text fetch shows
    before a tier-2b download and whose sha256 it records with the
    acceptance."""
    return (f"# Terms for {m['name']}\n\nBy downloading or using these files you agree that:\n\n"
            + "\n".join(f"{i}. {t}" for i, t in enumerate(consent_terms(m, lic), 1)) + "\n")


def notice(m, lic):
    """The tier notice at the top of the card."""
    tier = lic["tier"]
    if tier == "1":
        return (f"**Licence: {lic['licence']}.** The model's licence and all of its declared training data are "
                "permissive.")
    if tier == "2a":
        kinds = [CONCERN_TEXT[c] for c in lic["concerns"] if c in CONCERN_TEXT]
        sa = [d["name"] for d in lic["training_data"] if d["terms"] == "share-alike"]
        lines = [f"**Notice.** These weights are published under the original model's licence, "
                 f"{lic['licence']}, but its training data includes " + "; ".join(kinds) + "."]
        if sa:
            lines.append(f"Share-alike datasets: {', '.join(sa)}.")
        lines.append(f"{lic['reason']} Whether a dataset's terms reach a model trained on it is legally "
                     "unsettled; the [training data](#training-data-and-licences) is listed below with each "
                     "source's licence, so you can judge for yourself.")
        return " ".join(lines)
    if tier == "2b":
        terms = consent_terms(m, lic)
        return ("**This repository is gated.** To download it you accept these terms on Hugging Face, and "
                f"`{FETCH}` asks you to accept them again before it downloads anything:\n\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(terms, 1))
                + f"\n\nWhy: {lic['reason']}")
    raise SystemExit(f"{m['name']} is tier 3: sidekick doesn't publish its builds")


def carried_files(lic):
    files = list(lic.get("ship_files", []))
    if lic.get("notice") and "NOTICE" not in files:
        files.append("NOTICE")
    files += [c["file"] for c in lic.get("ship_copies", [])]
    return files


def modified(backend, builds, lic):
    quant = sorted({b["quantization"] for b in builds}) or (["fp32"] if backend == "onnx" else ["fp16"])
    text = (f"**Modified files.** Every file here except `tokenizer.json` is modified from the original model: "
            f"its weights converted to {BACKEND_NAME[backend]} in {' and '.join(quant)}, with the graph rewrites "
            f"sidekick's converter applies to keep each path accurate "
            f"([docs/CONVERTING.md]({GITHUB}/blob/main/docs/CONVERTING.md)), and a sidekick manifest added.")
    files = carried_files(lic)
    if files:
        text += f" The repository also carries {', '.join(f'`{f}`' for f in files)}, as the licence requires."
    return text


def front_matter(model, m, manifest, lic, backend, builds):
    repo, _ = split_source(m["source"])
    lines = ["---"]
    hf = licences_mod.hf_license(lic)
    lines.append(f"license: {hf}")
    if hf == "other":
        lines.append(f"license_name: {yq(lic['licence'])}")
        lines.append(f"license_link: {lic['licence_sources'][-1]}")
    lines.append(f"base_model: {repo}")
    if any(b["quantization"] != "fp32" for b in builds) or backend == "coreml":
        lines.append("base_model_relation: quantized")
    lines.append(f"library_name: {backend}")
    lines.append(f"pipeline_tag: {task_of(m, manifest)}")
    tags = ["sidekick", backend]
    tags += ["apple-silicon", "neural-engine"] if backend == "coreml" else ["cpu"]
    tags += {"embed": ["embeddings", "sentence-embeddings"], "classify": ["text-classification"],
             "rerank": ["reranker", "cross-encoder"]}[m["task"]]
    tags.append(f"licence-tier-{lic['tier']}")
    lines.append("tags:")
    lines += [f"- {t}" for t in dict.fromkeys(tags)]
    ds = licences_mod.hf_datasets(lic)
    if ds:
        lines.append("datasets:")
        lines += [f"- {d}" for d in ds]
    if lic["tier"] == "2b":
        lines.append(f"extra_gated_heading: {yq('Accept the terms to download ' + m['name'])}")
        prompt = "By requesting access you agree that:\n\n" + "\n".join(
            f"{i}. {t}" for i, t in enumerate(consent_terms(m, lic), 1))
        lines.append("extra_gated_prompt: |-")
        lines += ["  " + ln if ln else "" for ln in prompt.split("\n")]
        lines.append("extra_gated_fields:")
        lines.append(f"  {yq('I have read the terms above and accept them')}: checkbox")
        lines.append(f"extra_gated_button_content: {yq('Accept and request access')}")
    lines.append("---")
    return "\n".join(lines)


def build_size(b):
    distinct = sum(e["size"] or 0 for e in b["files"])
    on_disk = sum((e["size"] or 0) * len(e["paths"]) for e in b["files"])
    unknown = any(e["size"] is None for e in b["files"])
    return (None, None) if unknown else (distinct, on_disk)


def served_path(m, manifest, backend):
    if backend == "onnx":
        return "onnx-cpu"
    return COMPUTE_UNITS.get(manifest.get("compute_units"), m.get("served", "coreml-ane"))


def use_for(rows, backend, served):
    """Where each graded path stands, by chip: "Apple M1 Max: GPU A
    (served), Neural Engine C, CPU C". `rows` are published result rows or
    a build's own grades (both have chip, path and grade)."""
    by_chip = {}
    for r in sorted(rows, key=lambda r: (not chip(r["chip"]).startswith("Apple M1"), chip(r["chip"]),
                                         BACKEND_PATHS[backend].index(r["path"]))):
        if r["grade"] is None:
            continue
        name = PATH_NAME[r["path"]] + (" (served)" if backend == "coreml" and r["path"] == served else "")
        entry = f"{name} {r['grade']}"
        if entry not in by_chip.setdefault(r["chip"], []):
            by_chip[r["chip"]].append(entry)
    return "; ".join(f"{chip(c)}: {', '.join(v)}" for c, v in by_chip.items())


def downloads(model, m, manifest, backend, builds, rows):
    served = served_path(m, manifest, backend)
    out = []
    if builds:
        trows = []
        for b in builds:
            distinct, on_disk = build_size(b)
            shape = (f"buckets {', '.join(map(str, b['buckets']))}" + (f", {b['chunks']} chunks each" if b["chunks"] > 1 else "")
                     if b["buckets"] else "dynamic length")
            size = mb(distinct) + (f" ({mb(on_disk)} installed)" if on_disk and on_disk != distinct else "")
            trows.append([f"`{b['variant']}/`", b["quantization"], shape, b["max_seq_len"] or "", size])
        out.append(table(["directory", "weights", "shape", "max tokens", "download"], trows))
        out.append("")
    else:
        out.append("No build of this model for this backend is in sidekick's checksum ledger yet. The table below "
                   "lists the grades its builds were given; a build's files appear here when it is recorded.\n")
    graded = [g for b in builds for g in b["grades"]]
    covered = {g["chip"] for g in graded} | {chip(g["chip"]) for g in graded}
    merged = graded + [r for r in rows if r["chip"] not in covered]
    out.append(f"Download one directory: each is a complete sidekick model directory. Which path to run it on, by "
               f"the grades under [Fidelity](#fidelity) (a build's own where its files were graded, else the "
               f"published ones): {use_for(merged, backend, served) or 'not graded yet'}.")
    if backend == "coreml":
        out.append("\nCore ML builds are compiled models (`.mlmodelc`), the exact files that were graded. "
                   "The first load of each bucket on a path compiles it for that device, which can take from "
                   "seconds to minutes; later loads are fast.")
    alts = [a for a in INPUTS["publishing"].get("alternatives", []) if a["model"] == model]
    if backend == "onnx" and alts:
        out.append("\n**Alternatives.** The publisher's own ONNX export also grades A against the checkpoint, "
                   "and sidekick serves it as it serves its own (manifest: `pooling` and `[io]` as in the "
                   "committed onnx manifest):\n")
        out.append(table(["file", "revision", "sha256", "size", "grade"],
                         [[f"[{a['repo']}/{a['file']}](https://huggingface.co/{a['repo']}/blob/{a['revision']}/{a['file']})",
                           a["revision"][:12], f"`{a['sha256']}`", mb(a["size"]), f"{a['grade']} ({a['worst']})"]
                          for a in alts]))
        for a in alts:
            if a.get("note"):
                out.append(f"\n{a['note']}")
    return "\n".join(out)


def request_example(model, m, manifest):
    fmt = manifest.get("classify", {}).get("format")
    if m["task"] == "embed":
        body = {"model": model, "input": ["The battery lasts for days.", "It broke on day two."]}
        if manifest.get("prefixes", {}).get("query"):
            body["input_type"] = "query"
        url = "/v1/embeddings"
    elif m["task"] == "rerank":
        body = {"model": model, "query": "Will it rain tomorrow?",
                "documents": ["It rained all day yesterday.", "Tomorrow brings heavy showers."]}
        url = "/v1/rerank"
    elif fmt == "nli":
        body = {"model": model, "input": "The config sets port 8080 and enables TLS.",
                "candidate_labels": ["the port is 8080", "the server logs to a file"]}
        url = "/v1/classify"
    elif fmt == "laya":
        body = {"model": model, "input": "Please reset my password, I can't log in.", "question_type": "choice",
                "instructions": "Which team should handle this ticket?",
                "candidate_labels": ["billing", "account", "other"]}
        url = "/v1/classify"
    elif manifest.get("task", "text-classification") == "text-classification":
        body = {"model": model, "input": ["The battery lasts for days.", "It broke on day two."]}
        url = "/v1/classify"
    else:
        return (f"Its request format (`format = \"{fmt}\"`) is described in "
                f"[docs/design/classify.md]({GITHUB}/blob/main/docs/design/classify.md).")
    return ("```sh\ncurl -s localhost:8790" + url + " -H 'content-type: application/json' -d '"
            + json.dumps(body, ensure_ascii=False) + "'\n```")


def usage(model, m, manifest, backend, builds, lic, owner):
    rid = repo_id(owner, model, backend)
    variant = builds[0]["variant"] if builds else ("fp32" if backend == "onnx" else "fp16")
    accept = f" --accept-terms {model}" if lic["tier"] == "2b" else ""
    flag = " --backend onnx" if backend == "onnx" else ""
    out = [f"With [sidekick]({GITHUB}) installed, fetch the build into sidekick's models directory and start the "
           "daemon. Fetch downloads each distinct file once, checks its sha256 against sidekick's checksum ledger "
           "before placing it, and hard-links a file that several buckets share:",
           "", "```sh", f"{FETCH} {model}{flag}{accept}", "sidekickd", "```", "",
           "Then:", "", request_example(model, m, manifest), "",
           "**Without sidekick's fetch,** download one directory with the Hugging Face CLI, check it, and move it "
           "into place:", "", "```sh",
           f"hf download {rid} --include '{variant}/*' '{variant}.sha256' --local-dir ./{model}-download",
           f"cd ./{model}-download/{variant} && shasum -a 256 -c ../{variant}.sha256 && cd -",
           f"mv ./{model}-download/{variant} \"$HOME/Library/Application Support/sidekick/models/{model}\"",
           "```", "", "That is sidekick's models directory on macOS; on Linux, or with another directory in "
           "sidekick's config, move it there instead."]
    if any(b["layout"] == "shared" for b in builds):
        out += ["", "A shared-weights build (one weights file per chunk, used by every bucket) arrives as copies "
                "this way; `python3 -m sidekick_convert.shared_weights link <model dir>`, from sidekick's `tools/`, "
                "turns them back into hard links."]
    if backend == "onnx":
        out += ["", "ONNX builds run on ONNX Runtime's CPU backend, on macOS and Linux; the daemon needs its `onnx` "
                "feature and the ONNX Runtime library its release archives ship."]
    return "\n".join(out)


def fidelity(model, m, manifest, backend, builds, rows, results):
    g = results["grades"]
    out = ["sidekick's parity suite grades every path against the original checkpoint run in fp32 (PyTorch), on a "
           "fixed corpus of edge cases (empty, single-token, bucket-boundary, over-length and multilingual inputs), "
           "and fails a build outright on any hard gate (bucket and padding invariance, determinism, a finite "
           "output). The letters:", ""]
    if m["task"] == "embed":
        out.append(f"- **A**: worst-case cosine to fp32 at least {g['cosine']['A']} (indistinguishable from fp32); "
                   f"**B** ≥ {g['cosine']['B']}; **C** ≥ {g['cosine']['C']}; **D** below; **F**: a gate failed.")
    else:
        out.append(f"- **A**: worst |Δp| (probability) at most {g['dp']['A']}; **B** ≤ {g['dp']['B']}; "
                   f"**C** ≤ {g['dp']['C']}; **D** above; **F**: a gate failed. A model whose reference carries "
                   f"an ideal-fp16 ceiling gets the better of that and its p99 as a multiple of the ceiling "
                   f"(A ≤ {g['ratio']['A']}×, B ≤ {g['ratio']['B']}×, C ≤ {g['ratio']['C']}×). A flipped "
                   "prediction caps the grade at C.")
    committed = manifest.get("buckets")
    differ = [b["variant"] for b in builds if backend == "coreml" and committed and b["buckets"] != committed]
    lead = "Published grades ([docs/MODELS.md]({}/blob/main/docs/MODELS.md))".format(GITHUB)
    if differ:
        lead += (f", for the model's committed build (buckets {', '.join(map(str, committed))}); "
                 f"the files here are a different build of it, whose own grades follow")
    out += ["", lead + ":", ""]
    out.append(table(["chip", "OS", "path", "weights", "grade", "measure", "date"],
                     [[chip(r["chip"]), r["os"] or "",
                       PATH_NAME[r["path"]] + (" (served)" if r["served"] and backend == "coreml" else ""),
                       r["precision"], f"**{r['grade']}**" if r["grade"] else "", measure(r), r["date"] or ""]
                      for r in rows]))
    notes = [f"- {chip(r['chip'])}, {PATH_NAME[r['path']]}: {r['note']}" for r in rows if r.get("note")]
    if notes:
        out += ["", *notes]
    runs = [r for r in results["benchmarks"]["transfer"].get("runs", [])
            if r["model"] == model and r.get("build") in {b["fingerprint"] for b in builds}]
    for r in runs:
        b = next(b for b in builds if b["fingerprint"] == r["build"])
        out += ["", f"sidekick also ran MTEB tasks through the files of `{b['variant']}/` (build fingerprint "
                f"`{r['build']}`, {PATH_NAME[r['path']]}, {chip(r['chip'])}, {r['date']}): "
                f"[docs/MODELS.md]({GITHUB}/blob/main/docs/MODELS.md#measured-on-sidekicks-builds) compares "
                "its scores with the original's."]
    for b in builds:
        if b["grades"]:
            out += ["", f"The files of `{b['variant']}/` are the ones these reports graded (matched by each "
                    "bucket's weights sha256):", ""]
            out.append(table(["chip", "OS", "path", "grade", "report"],
                             [[chip(x["chip"]), x["os"] or "", PATH_NAME[x["path"]], f"**{x['grade']}**",
                               x["report"] or ""] for x in b["grades"]]))
    return "\n".join(out)


def benchmarks(model, results):
    scores = [s for s in results["benchmarks"]["scores"] if s["model"] == model]
    scores = [s for s in scores if s["value_text"] is not None]
    heads = [s for s in scores if s["category"] == "Mean (task)"] or scores[:6]
    if not heads:
        return ""
    kinds = results["benchmarks"]["source_kinds"]
    rows = [[s["benchmark"], s["category"], s["metric"], s["value_text"], kinds.get(s["source_kind"], s["source_kind"]),
             f"[source]({s['source_url']})"] for s in heads]
    return ("These are the original model's published scores, not measurements of this build; the fidelity grades "
            "above say how closely this build reproduces the original. Every score, with its caveats, is in "
            f"[docs/MODELS.md]({GITHUB}/blob/main/docs/MODELS.md#benchmark-scores).\n\n"
            + table(["benchmark", "category", "metric", "score", "source", ""], rows))


def provenance(m, b):
    p = b["produced"]
    repo, rev = split_source(b["source"])
    rows = [["source", f"[{repo}]({hf_url(repo, rev)}) at `{rev}`"],
            ["converter", f"`{p['converter'] or 'unknown'}` at commit `{p['converter_commit']}`" if p["converter_commit"]
             else f"`{p['converter'] or 'unknown'}`, commit not recorded"]]
    for k, label in (("coremltools", "coremltools"), ("torch", "PyTorch"), ("python", "Python"), ("os", "OS"),
                     ("xcode", "Xcode"), ("chip", "machine"), ("date", "converted")):
        if p.get(k):
            rows.append([label, p[k]])
    rows.append(["ledger id", f"`{b['id']}`"])
    rows.append(["digest", f"`{b['digest']}` (sha256 over every path and file hash)"])
    rows.append(["fingerprint", f"`{b['fingerprint']}` (weights, programs, tokenizer and manifest; as in "
                                "tools/mteb_transfer.py)"])
    out = table(["", f"`{b['variant']}/`"], rows)
    if b.get("note"):
        out += f"\n\n{b['note']}"
    return out


def checksums(b):
    rows = [[f"`{e['sha256']}`", mb(e["size"]) if e["size"] is not None else "unknown",
             "<br>".join(f"`{p}`" for p in e["paths"])] for e in b["files"]]
    shared = [e for e in b["files"] if len(e["paths"]) > 1]
    lead = (f"`{b['variant']}/`: {len(b['files'])} distinct files" +
            (f", {len(shared)} of them at several paths (stored once)" if shared else "") +
            f". The same list is in `{b['variant']}.sha256` and in sidekick's "
            f"[docs/data/ledger.json]({GITHUB}/blob/main/docs/data/ledger.json).")
    return lead + "\n\n" + table(["sha256", "size", "paths"], rows)


def convert_yourself(model, m, manifest_text, backend, builds):
    conv = (builds[0]["produced"]["converter"] if builds else None) or ledger_mod.converter_of(manifest_text) \
        or CONVERTER.get(model)
    cmd = None
    lines = manifest_text.splitlines()
    for i, ln in enumerate(lines):
        mm = re.match(r"^#\s+(python3? tools/convert_\S+\.py .*)$", ln)
        if mm:
            cmd = mm.group(1).replace("<models>", "<models dir>")
            break
    if not cmd:
        mm = re.search(r"`(tools/convert_[a-z0-9_]+\.py[^`]*)`", manifest_text)
        extra = mm.group(1).split(" ", 1)[1] if mm and " " in mm.group(1) else ""
        cmd = f"python tools/{conv.removeprefix('tools/') if conv else 'convert_<model>.py'} <checkpoint dir> " \
              f"<models dir>/{model}" + (f" {extra}" if extra else "")
    if backend == "onnx":
        cmd += " --format onnx"
    repo, rev = split_source(m["source"])
    return "\n".join([
        f"Every build here can be reproduced from the checkpoint with sidekick's converter, `{conv or 'see below'}`, "
        f"which runs the same gates the published build passed. Download [{repo}]({hf_url(repo, rev)}) at revision "
        f"`{rev}`, then, from a checkout of [sidekick]({GITHUB}):", "", "```sh", cmd,
        f"python3 tools/ledger.py verify <models dir>/{model}", "```", "",
        "`ledger.py verify` compares the result with this card's checksums and names the layer that differs: "
        "the weights, the compiled program, the tokenizer or the manifest. "
        + ("A Core ML compile on another macOS or Xcode version can legitimately change the compiled program "
           "while the weights stay identical. " if backend == "coreml" else "")
        + f"Requirements and each converter's options: [docs/CONVERTING.md]({GITHUB}/blob/main/docs/CONVERTING.md)."])


def training(lic):
    out = []
    base = lic["base_model"] or "none declared"
    if "@" in base:
        r, rev = base.split("@", 1)
        base = f"[{r}]({hf_url(r, rev)}) at `{rev[:12]}`"
    out.append(f"- **Model licence:** {lic['licence']} ([source]({lic['licence_sources'][0]}))")
    out.append(f"- **Base model:** {base}" + (f", {lic['base_model_licence']}" if lic["base_model_licence"] else ""))
    out.append(f"- **Training data disclosure:** {lic['training_disclosure']}")
    out.append(f"- **Publishing tier:** {lic['tier']} (sidekick's licence audit, "
               f"[docs/data/licences.toml]({GITHUB}/blob/main/docs/data/licences.toml), read {lic['read']})")
    out.append("")
    out.append(table(["dataset", "licence (original source)", "terms", "stage"],
                     [[f"[{d['name']}]({d['source']})", d["licence"], d["terms"], d["stage"]]
                      for d in lic["training_data"]]))
    if lic["custom_terms"]:
        out += ["", "Terms that travel with the weights:", ""]
        out += [f"- [{t['name']}]({t['url']}): {t['requires']}" for t in lic["custom_terms"]]
    out += ["", f"The base model's own pretraining data doesn't count toward the tier: {lic['base_pretraining']}"]
    if lic["unverified"]:
        out += ["", "Not confirmed from a primary source:", ""]
        out += [f"- {u}" for u in lic["unverified"]]
    return "\n".join(out)


# --- the card ----------------------------------------------------------------

INPUTS = {}


def select_builds(model, backend, ids=None):
    builds = [b for b in INPUTS["ledger"]["builds"] if b["model"] == model and b["backend"] == backend]
    if ids:
        known = {b["id"] for b in builds}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"not in the ledger for {model} ({backend}): {', '.join(missing)}")
        builds = [b for b in builds if b["id"] in ids]
    return builds


def render(model, backend="coreml", build_ids=None, owner=OWNER):
    if not INPUTS:
        INPUTS.update(load_inputs())
    m = INPUTS["models"].get(model)
    if m is None:
        raise SystemExit(f"{model} isn't in docs/data/models.toml")
    lic = INPUTS["licences"].get(model)
    if lic is None:
        raise SystemExit(f"{model} has no licence verdict in docs/data/licences.toml: not publishable")
    if lic["tier"] == "3":
        raise SystemExit(f"{model} is tier 3 ({lic['reason']})")
    manifest, manifest_text = committed_manifest(model, backend)
    builds = select_builds(model, backend, build_ids)
    rows = result_rows(INPUTS["results"], model, backend)
    repo, rev = split_source(m["source"])
    bname = BACKEND_NAME[backend]
    intro = (f"[{repo}]({hf_url(repo, rev)}) converted to {bname} and graded by "
             f"[sidekick]({GITHUB}), a local inference daemon for macOS that serves embeddings, classifiers and "
             "rerankers beside Apple's on-device Foundation Models. ")
    intro += ("Each build is a set of static-shape programs, one per sequence-length bucket, compiled for "
              "Apple silicon's Neural Engine, GPU and CPU (and the CPU and GPU of Intel Macs)."
              if backend == "coreml" else
              "Each build is one dynamic-shape ONNX graph for ONNX Runtime's CPU backend.")
    parts = [
        front_matter(model, m, manifest, lic, backend, builds),
        "",
        f"# {m['name']} ({bname}, for sidekick)",
        "",
        intro,
        "",
        notice(m, lic),
        "",
        modified(backend, builds, lic),
        "",
        "## Which build to download",
        "",
        downloads(model, m, manifest, backend, builds, rows),
        "",
        "## Use it with sidekick",
        "",
        usage(model, m, manifest, backend, builds, lic, owner),
        "",
        "## Fidelity",
        "",
        fidelity(model, m, manifest, backend, builds, rows, INPUTS["results"]),
    ]
    bench = benchmarks(model, INPUTS["results"])
    if bench:
        parts += ["", "## Benchmark scores", "", bench]
    for b in builds:
        parts += ["", f"## Provenance: `{b['variant']}/`", "", provenance(m, b)]
    parts += ["", "## Checksums", ""]
    parts += [checksums(b) for b in builds] or ["Recorded when a build is."]
    parts += ["", "## Convert it yourself", "", convert_yourself(model, m, manifest_text, backend, builds)]
    parts += ["", "## Training data and licences", "", training(lic)]
    parts += ["", "---", "",
              f"Generated by `tools/model_card.py` from sidekick's committed data. Report a problem with this build "
              f"at {GITHUB}/issues."]
    return "\n".join(parts) + "\n", builds, lic


def sums_text(b):
    lines = []
    for e in b["files"]:
        for p in e["paths"]:
            lines.append((p, f"{e['sha256']}  {p}"))
    return "\n".join(t for _, t in sorted(lines)) + "\n"


def write(out, model, backend, build_ids, owner):
    card, builds, lic = render(model, backend, build_ids, owner)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(card)
    for b in builds:
        (out / f"{b['variant']}.sha256").write_text(sums_text(b))
    if lic.get("notice"):
        (out / "NOTICE").write_text(lic["notice"] + "\n")
    if lic["tier"] == "2b":
        (out / "TERMS.md").write_text(terms_text(INPUTS["models"][model], lic))
    print(f"wrote {out}/README.md" + (f" and {len(builds)} checksum file(s)" if builds else ""))


def check():
    INPUTS.update(load_inputs())
    problems = []
    seen = sorted({(b["model"], b["backend"]) for b in INPUTS["ledger"]["builds"]})
    n = 0
    for model, backend in seen:
        lic = INPUTS["licences"].get(model)
        if lic and lic["tier"] == "3":
            for b in INPUTS["ledger"]["builds"]:
                if b["model"] == model and b["published"]:
                    problems.append(f"{b['id']} is marked published, but {model} is tier 3")
            continue
        try:
            render(model, backend)
            n += 1
        except SystemExit as e:
            problems.append(f"{model} ({backend}): {e}")
    if problems:
        raise SystemExit("model cards:\n  " + "\n  ".join(problems))
    print(f"model cards: {n} render")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", nargs="?")
    p.add_argument("--backend", choices=("coreml", "onnx"), default="coreml")
    p.add_argument("--build", action="append", help="a ledger build id to include (default: every build)")
    p.add_argument("--owner", default=OWNER, help="the Hugging Face account the repository belongs to")
    p.add_argument("--out", help="write README.md, checksum files and any NOTICE here")
    p.add_argument("--check", action="store_true", help="render every publishable card the ledger allows")
    args = p.parse_args()
    if args.check:
        check()
        return
    if not args.model:
        p.error("name a model, or pass --check")
    if args.out:
        write(args.out, args.model, args.backend, args.build, args.owner)
    else:
        sys.stdout.write(render(args.model, args.backend, args.build, args.owner)[0])


if __name__ == "__main__":
    main()
