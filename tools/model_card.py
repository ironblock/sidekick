"""Generate the model card (README.md) for a sidekick build's Hugging Face
repository (#137; docs/PUBLISHING.md, "Model cards").

One repository per model per backend (Core ML, ONNX), holding every build of
that model for that backend, each in a directory named for its variant
(`fp16/`, `fp16-max2048/`, ...) laid out exactly as sidekick installs it.

A card holds only what is particular to its model and builds: what the
model is, its licence notice, which build to download, the one fetch
command, its grades, its benchmark scores, its training data, the upstream
model, and the builds' checksums. Everything that is the same for every
model (how fetch, serving and installation work, what a grade means, the
licence-tier policy, how to convert a model yourself) lives once in
sidekick's own docs, and the card links there. Those links are pinned to a
release tag (`--ref`, default `v<workspace version>`): a card is published
with a sidekick release and tagged with it on the Hub, so it links the docs
of that release, as release notes do, and a heading renamed later never
breaks a published card. --check resolves every link against the working
tree, which is what the tag will hold.

The card is generated from committed data, never written by hand:
- docs/data/models.toml: the model's name, family, task and checkpoint;
- the committed manifest (examples/manifests/, examples/classifiers/): its
  dimensions, compute units and converter;
- docs/data/models.json: the fidelity grades per path and chip, and the
  benchmark scores;
- docs/data/licences.toml: the licence, training data and publishing tier,
  from which the notice (tier 2a) or the consent terms (tier 2b) are built;
- docs/data/ledger.json: each build's files, sha256 and size, what produced
  it, and the grades its exact files were graded with;
- docs/data/publishing.toml: publisher exports listed as ONNX alternatives.

Usage:
    python3 tools/model_card.py MODEL [--backend coreml|onnx] [--build ID ...]
        [--ref vX.Y.Z] [--out DIR]
    python3 tools/model_card.py --check

Without --out the card is printed. With --out, DIR receives the README.md,
one `<variant>.sha256` per build (`shasum -a 256 -c` format, paths relative
to the variant's directory), a NOTICE where the licence requires one, and
for a tier-2b model TERMS.md, the consent terms fetch shows and records.
It also writes LICENSE from the model's committed licence text
(licences.toml's `licence_text`, docs/data/licence-texts/) and NOTICE from
its upstream NOTICE (`notice_text`).
Copies of documents the licence requires (licences.toml's `ship_copies`)
are fetched from their source at upload, not generated here.
--check renders every publishable model and backend the ledger holds builds
for, resolves every link each card makes into sidekick's docs, and fails on
a card that can't be rendered, a link that doesn't resolve, or a tier-3
build marked published.

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

PATH_NAME = {"coreml-ane": "Neural Engine", "coreml-gpu": "GPU", "coreml-cpu": "CPU", "onnx-cpu": "CPU"}
BACKEND_NAME = {"coreml": "Core ML", "onnx": "ONNX"}
BACKEND_PATHS = {"coreml": ("coreml-ane", "coreml-gpu", "coreml-cpu"), "onnx": ("onnx-cpu",)}
CHIP_NAME = {}  # filled from runs.toml's [chips]
PIPELINE = {"embed": "feature-extraction", "rerank": "text-ranking"}
COMPUTE_UNITS = {None: "coreml-ane", "cpu_and_ne": "coreml-ane", "cpu_and_gpu": "coreml-gpu",
                 "cpu_only": "coreml-cpu", "all": "coreml-ane"}
WEIGHTS_NAME = {"fp32": "32-bit float", "fp16": "16-bit float", "fp16 weights": "16-bit float, computed in 32-bit",
                "int8": "8-bit integer"}
TERMS_NAME = {"permissive": "Permissive", "share-alike": "Share-alike", "non-commercial": "Non-commercial",
              "unknown": "Unknown"}
DISCLOSURE_NAME = {"full": "fully disclosed", "partial": "partly disclosed", "undisclosed": "undisclosed"}
TASK_NAME = {"embed": "embedding model", "rerank": "reranker", "text-classification": "classifier",
             "zero-shot-classification": "zero-shot classifier", "text-ranking": "reranker"}
SOURCE_KIND = {"mteb": "MTEB results", "mixed": "MTEB results, some tasks self-reported",
               "self-reported": "the model's authors", "independent": "an independent evaluation"}

# Where the card points for everything that isn't particular to it: (file,
# heading anchor). --check resolves each one against the working tree.
DOCS = {
    "fetch": ("docs/RUNNING.md", "fetching-a-published-build"),
    "by-hand": ("docs/RUNNING.md", "installing-a-published-build-by-hand"),
    "embed": ("docs/API.md", "embeddings"),
    "classify": ("docs/API.md", "classification"),
    "rerank": ("docs/API.md", "rerank"),
    "onnx": ("docs/MODELS.md", "onnx-runtime-on-the-cpu-onnx-cpu"),
    "grades-embed": ("docs/MODELS.md", "confidence-grades-the-parity-suite"),
    "grades-classify": ("docs/MODELS.md", "classifiers-v1classify"),
    "grades-rerank": ("docs/MODELS.md", "rerankers-v1rerank"),
    "benchmarks": ("docs/MODELS.md", "benchmark-scores"),
    "transfer": ("docs/MODELS.md", "measured-on-sidekicks-builds"),
    "licences": ("docs/MODELS.md", "licences-and-training-data"),
    "tiers": ("docs/PUBLISHING.md", "what-is-published-licence-tiers"),
    "consent": ("docs/PUBLISHING.md", "consent-for-tier-2b-gated-repositories"),
    "ledger": ("docs/PUBLISHING.md", "the-checksum-ledger"),
    "compiled": ("docs/PUBLISHING.md", "compiled-programs-not-packages"),
    "reproduce": ("docs/CONVERTING.md", "reproducing-a-published-build"),
    "shared": ("docs/DECISIONS.md", "d49--one-weights-file-per-chunk-shared-by-every-bucket"),
}


# --- inputs ------------------------------------------------------------------

def owner():
    """The Hugging Face organization the repositories belong to."""
    return tomllib.loads((DATA / "publishing.toml").read_text())["owner"]


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


def default_ref():
    """The release tag a card's links are pinned to: the workspace version's."""
    return "v" + tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"]["package"]["version"]


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


def count(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


class Links:
    """Links into sidekick's docs at one ref, remembered for --check."""

    def __init__(self, ref):
        self.ref = ref
        self.used = set()

    def __call__(self, key, text):
        path, anchor = DOCS[key]
        self.used.add(key)
        return f"[{text}]({GITHUB}/blob/{self.ref}/{path}#{anchor})"


# --- anchors -----------------------------------------------------------------

def slug(heading):
    """GitHub's anchor for a Markdown heading."""
    h = heading.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path):
    """Every heading anchor GitHub generates for a Markdown file, with its
    -1, -2 suffixes for repeated headings; code blocks are skipped."""
    seen, out, fenced = {}, set(), False
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        m = None if fenced else re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def link_problems(keys):
    problems = []
    cache = {}
    for key in sorted(keys):
        path, anchor = DOCS[key]
        if not (REPO / path).exists():
            problems.append(f"link {key}: {path} doesn't exist")
            continue
        found = cache.setdefault(path, anchors(REPO / path))
        if anchor not in found:
            problems.append(f"link {key}: {path} has no heading #{anchor}")
    return problems


# --- the parts of a card -----------------------------------------------------

def task_of(m, manifest):
    if m["task"] == "classify":
        return manifest.get("task", "text-classification")
    return PIPELINE.get(m["task"], m["task"])


def result_rows(results, model, backend):
    rows = [r for r in results["rows"] if r["model"] == model and r["path"] in BACKEND_PATHS[backend]]
    return sorted(rows, key=lambda r: (r["chip"] != "Apple M1 Max", r["chip"],
                                       BACKEND_PATHS[backend].index(r["path"]), r["precision"]))


def consent_terms(m, lic):
    """The terms a user accepts to download a tier-2b build: built from the
    audit's concerns and the datasets behind them. Plain Markdown, used
    as the gate's prompt, on the card, and as TERMS.md."""
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


def notice(m, lic, link):
    """The licence notice for the model's tier, as a blockquote."""
    tier = lic["tier"]
    head = f"**Licence: {lic['licence']} · publishing tier {tier}.**"
    if tier == "1":
        body = [f"> {head} The licence and all of the declared training data are permissive. "
                + link("tiers", "How sidekick decides what it publishes") + "."]
    elif tier == "2a":
        body = [f"> {head} {lic['reason']} Whether a dataset's terms reach a model trained on it is legally "
                "unsettled; the [training data](#training-data) below gives each source's licence. "
                + link("tiers", "How sidekick decides what it publishes") + "."]
    elif tier == "2b":
        terms = consent_terms(m, lic)
        body = [f"> {head} **This repository is gated.** To download it, accept these terms on Hugging Face; "
                "`sidekickd fetch` shows them and asks for `--accept-terms` too.", ">"]
        body += [f"> {i}. {t}" for i, t in enumerate(terms, 1)]
        body += [">", f"> Why: {lic['reason']} " + link("consent", "About the gate") + "."]
    else:
        raise SystemExit(f"{m['name']} is tier 3: sidekick doesn't publish its builds")
    return "\n".join(body)


def carried_files(lic):
    """Files other than LICENSE and an upstream NOTICE that the licence asks
    a published build to carry."""
    files = list(lic.get("ship_files", []))
    if lic.get("notice") and "NOTICE" not in files:
        files.append("NOTICE")
    files += [c["file"] for c in lic.get("ship_copies", [])]
    return files


def licence_file(model, m, lic):
    """The LICENSE a published build carries: the committed text named by
    licence_text, byte for byte; for the SPDX MIT template (a model whose
    repository publishes no licence file), with a header saying so and its
    copyright line naming the model's authors."""
    name = lic.get("licence_text")
    if not name:
        return None
    text = (licences_mod.TEXTS / name).read_text()
    if name != "mit.txt":
        return text
    repo, rev = split_source(m["source"])
    card = lic["licence_sources"][0]
    header = (f"{repo} is licensed under the MIT License by its authors, as its model card at revision {rev} "
              f"states:\n{card}\nIts repository publishes no licence file or copyright line of its own. Below is "
              "the MIT License's text from the SPDX License List, its copyright line naming the model's "
              "authors.\n\n")
    line = "Copyright (c) <year> <copyright holders>"
    assert line in text, "the SPDX MIT text changed: update licence_file()"
    return header + text.replace(line, f"Copyright (c) the authors of {repo}")


def licence_rows(lic):
    """The "Original model" rows naming the licence files this repository
    carries, and where each came from."""
    texts = licences_mod.load_texts()["texts"]
    rows = []
    name = lic.get("licence_text")
    if name:
        src = texts[name]["source"]
        if name == "apache-2.0.txt":
            what = (f"the [Apache License 2.0]({src}) as the Apache Software Foundation publishes it; the model's "
                    "repository carries no licence file")
        elif name == "mit.txt":
            what = (f"the [MIT License]({src}), naming the model's authors; the model's repository carries no "
                    "licence file or copyright line")
        else:
            what = f"the upstream licence text, with its copyright line, from [{short_source(src)}]({src})"
        rows.append(["Licence file", f"`LICENSE`: {what}"])
    if lic.get("notice_text"):
        src = texts[lic["notice_text"]]["source"]
        rows.append(["Notice file", f"`NOTICE`: the upstream NOTICE, from [{short_source(src)}]({src})"])
    return rows


def short_source(url):
    """"FlagOpen/FlagEmbedding" from a pinned GitHub or Hugging Face link."""
    parts = url.split("/")
    return "/".join(parts[3:5]) if len(parts) > 4 else url


def front_matter(model, m, manifest, lic, backend, builds):
    repo, _ = split_source(m["source"])
    lines = ["---"]
    hf = licences_mod.hf_license(lic)
    lines.append(f"license: {hf}")
    if hf == "other":
        lines.append(f"license_name: {yq(lic['licence'])}")
        lines.append(f"license_link: {lic['licence_sources'][-1]}")
    lines.append(f"base_model: {repo}")
    # The Hub has no "converted" relation. "quantized" is the one it uses for
    # format conversions (GGUF, at any precision); finetune, adapter and
    # merge would each claim training that didn't happen.
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


def summary(model, m, manifest, backend, builds, rows):
    repo, rev = split_source(m["source"])
    what = TASK_NAME.get(m["task"] if m["task"] != "classify" else task_of(m, manifest), "model")
    dims = f" ({manifest['dims']} dimensions)" if manifest.get("dims") else ""
    article = "an" if m["family"][0].lower() in "aeiou" else "a"
    where = ("for Apple silicon's Neural Engine, GPU and CPU" if backend == "coreml"
             else "for ONNX Runtime's CPU backend on macOS and Linux")
    held = (f"one build, `{builds[0]['variant']}/`" if len(builds) == 1
            else f"{len(builds)} builds" if builds else "no build yet")
    return (f"[{repo}]({hf_url(repo, rev)}), {article} {m['family']} {what}{dims}, converted to "
            f"{BACKEND_NAME[backend]} by [sidekick]({GITHUB}) {where}. This repository holds {held}, "
            "graded against the original model on every compute path it runs on.")


def served_path(m, manifest, backend):
    if backend == "onnx":
        return "onnx-cpu"
    return COMPUTE_UNITS.get(manifest.get("compute_units"), m.get("served", "coreml-ane"))


def build_size(b):
    if any(e["size"] is None for e in b["files"]):
        return None
    return sum(e["size"] for e in b["files"])


def runs_on(b, rows, backend, served):
    """Each compute path's grade on Apple silicon (any chip where nothing
    was graded there), the default marked: the build's own grade where its
    files were graded, else the published one."""
    def apple(g):
        return chip(g["chip"]).startswith("Apple")

    own = [g for g in b["grades"] if g["grade"]]
    published = [r for r in rows if r["grade"] and r["precision"] == b["quantization"]]
    if any(apple(g) for g in own + published):
        own, published = [g for g in own if apple(g)], [r for r in published if apple(r)]
    grades = {}
    for g in own:
        grades.setdefault(g["path"], set()).add(g["grade"])
    for r in published:
        if r["path"] not in {g["path"] for g in own}:
            grades.setdefault(r["path"], set()).add(r["grade"])
    out = []
    for p in BACKEND_PATHS[backend]:
        if p in grades:
            worst = max(grades[p])
            name = PATH_NAME[p] + (" (default)" if backend == "coreml" and p == served else "")
            out.append(f"{name} {worst}")
    return ", ".join(out) or "not graded yet"


def download(model, m, manifest, backend, builds, rows, lic, link):
    served = served_path(m, manifest, backend)
    out = []
    if builds:
        trows = []
        for b in builds:
            shape = ", ".join(map(str, b["buckets"])) if b["buckets"] else "any length"
            if b["chunks"] > 1:
                shape += f" ({b['chunks']} programs each)"
            shared = ""
            if backend == "coreml":
                shared = "Yes, one file for all buckets" if b["layout"] == "shared" else "No, one per bucket"
            cap = b["max_seq_len"]
            if not cap and not b["buckets"]:
                cap = next((r["max_seq_len"] for r in rows if r.get("max_seq_len")), manifest.get("max_seq_len"))
            elif not cap:
                cap = manifest.get("max_seq_len") if manifest.get("buckets") == b["buckets"] else max(b["buckets"])
            row = [f"`{b['variant']}/`", WEIGHTS_NAME.get(b["quantization"], b["quantization"]), shape,
                   cap or "", mb(build_size(b))]
            if backend == "coreml":
                row.append(shared)
            row.append(runs_on(b, rows, backend, served))
            trows.append(row)
        header = ["Build", "Weights", "Sequence buckets" if backend == "coreml" else "Input length", "Max tokens",
                  "Download"] + (["Shared weights"] if backend == "coreml" else []) + ["Runs on (grade)"]
        out.append(table(header, trows))
    else:
        out.append("No build of this model for this backend is recorded in sidekick's checksum ledger yet.")
    variants = {b["variant"] for b in builds}
    cmd = f"sidekickd fetch {model}"
    cmd += " --backend onnx" if backend == "onnx" else ""
    cmd += f" --variant {builds[0]['variant']}" if len(variants) > 1 else ""
    cmd += f" --accept-terms {model}" if lic["tier"] == "2b" else ""
    since = link.ref.removeprefix("v") if re.match(r"^v\d", link.ref) else None
    out += ["", "```sh", cmd, "```", "",
            (f"Needs sidekick {since} or later. " if since else "")
            + " · ".join([link("fetch", "How fetch works"), link("by-hand", "Installing without fetch"),
                          link({"embed": "embed", "classify": "classify", "rerank": "rerank"}[m["task"]],
                               "Calling the model")])]
    if backend == "coreml":
        out[-1] += " · " + link("compiled", "Why compiled programs")
    alts = [a for a in INPUTS["publishing"].get("alternatives", []) if a["model"] == model]
    if backend == "onnx" and alts:
        out += ["", "The publisher's own export also grades A, and sidekick serves it the same way "
                + f"({link('onnx', 'ONNX Runtime on the CPU')}):", ""]
        out.append(table(["Export", "Revision", "sha256", "Size", "Grade"],
                         [[f"[{a['repo']} `{a['file']}`](https://huggingface.co/{a['repo']}/blob/{a['revision']}/{a['file']})",
                           f"`{a['revision'][:12]}`", f"`{a['sha256']}`", mb(a["size"]), a["grade"]]
                          for a in alts]))
    return "\n".join(out)


def measure_header(m):
    return "Worst cosine" if m["task"] == "embed" else "Worst \\|Δp\\|"


def fidelity(model, m, manifest, backend, builds, rows, link):
    served = served_path(m, manifest, backend) if backend == "coreml" else None
    several = len(builds) > 1
    # (path, chip, report) -> the build whose files that report graded
    graded = {(g["path"], chip(g["chip"]), g["report"]): b for b in builds for g in b["grades"]}
    weights = len({r["precision"] for r in rows} | {b["quantization"] for b in builds}) > 1

    def path_name(path, default):
        return PATH_NAME[path] + (" (default)" if default else "")

    def files(b):
        return (f"This build (`{b['variant']}/`)" if several else "This build") if b else "Unconfirmed"

    trows, covered = [], set()
    for r in rows:
        b = next((graded[(r["path"], chip(r["chip"]), rep)] for (path, c, rep) in graded
                  if path == r["path"] and c == chip(r["chip"]) and rep and rep in (r["source"] or "")
                  and graded[(path, c, rep)]["quantization"] == r["precision"]), None)
        if b:
            covered.add((r["path"], chip(r["chip"]), b["id"]))
        measure = r["error_text"] or "—"
        if r["error_kind"] == "dp" and r.get("ratio"):
            measure += f" ({r['ratio']}× the fp16 ceiling)"
        trows.append([path_name(r["path"], r["served"] and backend == "coreml"), chip(r["chip"]), r["os"] or ""]
                     + ([WEIGHTS_NAME.get(r["precision"], r["precision"])] if weights else [])
                     + [f"**{r['grade']}**" if r["grade"] else "", measure, files(b)])
    for b in builds:
        for g in b["grades"]:
            key = (g["path"], chip(g["chip"]), b["id"])
            if key not in covered:
                covered.add(key)
                trows.append([path_name(g["path"], g["path"] == served), chip(g["chip"]),
                              (g["os"] or "").split(" (")[0]]
                             + ([WEIGHTS_NAME.get(b["quantization"], b["quantization"])] if weights else [])
                             + [f"**{g['grade']}**", "—", files(b)])
    kind = {"embed": "grades-embed", "classify": "grades-classify", "rerank": "grades-rerank"}[m["task"]]
    header = ["Compute path", "Chip", "OS"] + (["Weights"] if weights else []) + \
        ["Grade", measure_header(m), "Graded files"]
    out = [f"Graded against the original model in fp32 by sidekick's parity suite. {link(kind, 'What the grades mean')}.",
           "", table(header, trows), "",
           "*This build*: the grading report's file hashes match this repository's files. *Unconfirmed*: "
           "graded on a build of the same model that can't be matched to these files by hash."]
    notes = [f"- **{PATH_NAME[r['path']]}, {chip(r['chip'])}:** {r['note']}" for r in rows if r.get("note")]
    if notes:
        out += ["", *notes]
    return "\n".join(out)


def benchmarks(model, builds, results, link):
    scores = [s for s in results["benchmarks"]["scores"] if s["model"] == model and s["value_text"] is not None]
    heads = [s for s in scores if s["category"] == "Mean (task)"] or scores[:6]
    runs = [r for r in results["benchmarks"]["transfer"].get("runs", [])
            if r["model"] == model and r.get("build") in {b["fingerprint"] for b in builds}]
    if not heads and not runs:
        return ""
    out = []
    if heads:
        out.append("The original model's published scores, not this build's. "
                   + link("benchmarks", "Every score, with its caveats") + ".")
        out += ["", table(["Benchmark", "Score", "Measure", "Source"],
                          [[s["benchmark"], s["value_text"],
                            ("Mean over its tasks" if s["category"] == "Mean (task)" else f"{s['category']}, {s['metric']}"),
                            f"[{SOURCE_KIND.get(s['source_kind'], s['source_kind'])}]({s['source_url']})"]
                           for s in heads])]
    for r in runs:
        b = next(b for b in builds if b["fingerprint"] == r["build"])
        out += ["", f"sidekick also ran MTEB tasks through `{b['variant']}/` itself: "
                + link("transfer", "its scores beside the original's") + "."]
    return "\n".join(out)


def training(lic):
    out = [f"Training data: {DISCLOSURE_NAME[lic['training_disclosure']]}. Each dataset under its original "
           "source's licence:", ""]
    out.append(table(["Dataset", "Licence", "Terms", "Stage"],
                     [[f"[{d['name']}]({d['source']})", d["licence"], TERMS_NAME[d["terms"]], d["stage"]]
                      for d in lic["training_data"]]))
    if lic["custom_terms"]:
        out += ["", table(["Terms that travel with the weights", "What they require"],
                          [[f"[{t['name']}]({t['url']})", t["requires"]] for t in lic["custom_terms"]])]
    if lic["unverified"]:
        out += ["", "<details><summary>Not confirmed from a primary source</summary>", ""]
        out += [f"- {u}" for u in lic["unverified"]]
        out += ["", "</details>"]
    return "\n".join(out)


def original(model, m, manifest_text, backend, builds, lic, link):
    repo, rev = split_source(m["source"])
    base = lic["base_model"] or "None declared"
    if "@" in base:
        r, brev = base.split("@", 1)
        base = f"[{r}]({hf_url(r, brev)})"
    if lic["base_model_licence"]:
        base += f" ({lic['base_model_licence']})"
    conv = next((b["produced"]["converter"] for b in builds if b["produced"]["converter"]), None) \
        or ledger_mod.converter_of(manifest_text)
    weights = sorted({WEIGHTS_NAME.get(b["quantization"], b["quantization"]).split(",")[0] for b in builds}) \
        or (["32-bit float"] if backend == "onnx" else ["16-bit float"])
    rows = [["Model", f"[{repo}]({hf_url(repo, rev)})"],
            ["Revision", f"`{rev}`"],
            ["Licence", f"[{lic['licence']}]({lic['licence_sources'][0]})"],
            *licence_rows(lic),
            ["Base model", base],
            ["Changes in this repository",
             f"Converted to {BACKEND_NAME[backend]} in {' and '.join(weights).lower()}"
             + (f" by `{conv}`" if conv else "") + ", with a sidekick manifest added. "
             + link("reproduce", "Reproduce it")]]
    files = carried_files(lic)
    if files:
        rows.append(["Also included, as the licence requires", ", ".join(f"`{f}`" for f in files)])
    return table(["", ""], rows)


LEDGER_TEXT = "sidekick's checksum ledger"


def build_details(b, link):
    p = b["produced"]
    made = []
    if p["converter"]:
        made.append(["Converter", f"`{p['converter']}`" + (f" at `{p['converter_commit'][:12]}`" if p["converter_commit"] else "")])
    for k, label in (("coremltools", "coremltools"), ("torch", "PyTorch"), ("python", "Python"), ("os", "OS"),
                     ("xcode", "Xcode"), ("chip", "Converted on"), ("date", "Converted")):
        if p.get(k):
            made.append([label, p[k]])
    if b["backend"] == "coreml":
        made.append(["Shared weights", "Yes: " + link("shared", "one weights file for every bucket")
                     if b["layout"] == "shared" else "No: one weights file per bucket"])
    if b["evidence"] == "sha256sums":
        made.append(["Recorded from", "an archived copy's checksums"])
    shared = [e for e in b["files"] if len(e["paths"]) > 1]
    rows = [[f"`{e['sha256']}`", mb(e["size"]), "<br>".join(f"`{p}`" for p in e["paths"])] for e in b["files"]]
    paths = sum(len(e["paths"]) for e in b["files"])
    out = [f"<details><summary><code>{b['variant']}/</code>: ledger build <code>{b['id']}</code>, "
           f"{count(paths, 'file')}{f', {len(shared)} shared' if shared else ''}</summary>", "",
           f"`sidekickd fetch` checks every file against {link('ledger', LEDGER_TEXT)}; "
           f"`{b['variant']}.sha256` has the same list for `shasum -a 256 -c`.", ""]
    if made:
        out += [table(["Built with", ""], made), ""]
    out += [table(["sha256", "Size", "Path"], rows), "", "</details>"]
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


def render(model, backend="coreml", build_ids=None, owner=None, ref=None, links=None):
    """The card, its builds and the licence entry. `links` collects which
    docs links the card makes, for --check."""
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
    link = links if links is not None else Links(ref or default_ref())
    manifest, manifest_text = committed_manifest(model, backend)
    builds = select_builds(model, backend, build_ids)
    rows = result_rows(INPUTS["results"], model, backend)
    parts = [
        front_matter(model, m, manifest, lic, backend, builds), "",
        f"# {m['name']} for sidekick ({BACKEND_NAME[backend]})", "",
        summary(model, m, manifest, backend, builds, rows), "",
        notice(m, lic, link), "",
        "## Download", "",
        download(model, m, manifest, backend, builds, rows, lic, link), "",
        "## Fidelity", "",
        fidelity(model, m, manifest, backend, builds, rows, link),
    ]
    bench = benchmarks(model, builds, INPUTS["results"], link)
    if bench:
        parts += ["", "## Benchmark scores", "", bench]
    parts += ["", "## Training data", "", training(lic)]
    parts += ["", "## Original model", "", original(model, m, manifest_text, backend, builds, lic, link)]
    if builds:
        parts += ["", "## Checksums", ""]
        parts += ["\n\n".join(build_details(b, link) for b in builds)]
    parts += ["", "---", "",
              f"Generated by `tools/model_card.py` from sidekick's committed data. Report a problem at "
              f"{GITHUB}/issues."]
    return "\n".join(parts) + "\n", builds, lic


def sums_text(b):
    lines = []
    for e in b["files"]:
        for p in e["paths"]:
            lines.append((p, f"{e['sha256']}  {p}"))
    return "\n".join(t for _, t in sorted(lines)) + "\n"


def write(out, model, backend, build_ids, owner, ref=None):
    card, builds, lic = render(model, backend, build_ids, owner, ref)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(card)
    for b in builds:
        (out / f"{b['variant']}.sha256").write_text(sums_text(b))
    licence = licence_file(model, INPUTS["models"][model], lic)
    if licence:
        (out / "LICENSE").write_text(licence)
    if lic.get("notice_text"):
        (out / "NOTICE").write_bytes((licences_mod.TEXTS / lic["notice_text"]).read_bytes())
    elif lic.get("notice"):
        (out / "NOTICE").write_text(lic["notice"] + "\n")
    if lic["tier"] == "2b":
        (out / "TERMS.md").write_text(terms_text(INPUTS["models"][model], lic))
    print(f"wrote {out}/README.md" + (f" and {len(builds)} checksum file(s)" if builds else ""))


def check():
    INPUTS.update(load_inputs())
    problems = []
    link = Links(default_ref())
    seen = sorted({(b["model"], b["backend"]) for b in INPUTS["ledger"]["builds"]})
    n = 0
    org = owner()
    for b in INPUTS["ledger"]["builds"]:
        for pub in b["published"]:
            if pub["repo"] != repo_id(org, b["model"], b["backend"]):
                problems.append(f"{b['id']} is published at {pub['repo']}, not {repo_id(org, b['model'], b['backend'])}")
    for model, backend in seen:
        lic = INPUTS["licences"].get(model)
        if lic and lic["tier"] == "3":
            for b in INPUTS["ledger"]["builds"]:
                if b["model"] == model and b["published"]:
                    problems.append(f"{b['id']} is marked published, but {model} is tier 3")
            continue
        try:
            render(model, backend, links=link)
            licence_file(model, INPUTS["models"][model], lic)
            n += 1
        except SystemExit as e:
            problems.append(f"{model} ({backend}): {e}")
    problems += link_problems(link.used)
    if problems:
        raise SystemExit("model cards:\n  " + "\n  ".join(problems))
    print(f"model cards: {n} render, {len(link.used)} doc links resolve")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model", nargs="?")
    p.add_argument("--backend", choices=("coreml", "onnx"), default="coreml")
    p.add_argument("--build", action="append", help="a ledger build id to include (default: every build)")
    p.add_argument("--owner", help="the Hugging Face organization (default: docs/data/publishing.toml's owner)")
    p.add_argument("--ref", help="the sidekick release tag the card's doc links are pinned to "
                                 "(default: v<workspace version>)")
    p.add_argument("--out", help="write README.md, checksum files and any NOTICE here")
    p.add_argument("--check", action="store_true", help="render every publishable card the ledger allows")
    args = p.parse_args()
    if args.check:
        check()
        return
    if not args.model:
        p.error("name a model, or pass --check")
    if args.out:
        write(args.out, args.model, args.backend, args.build, args.owner, args.ref)
    else:
        sys.stdout.write(render(args.model, args.backend, args.build, args.owner, args.ref)[0])


if __name__ == "__main__":
    main()
