"""Build the causal-LM converter's committed text fixtures
(fixtures/causal-lm/; docs/CONVERTING.md, "Causal language models").

    python tools/make_causal_lm_fixture.py calibration
    python tools/make_causal_lm_fixture.py evaluation
    python tools/make_causal_lm_fixture.py tokens <hf-model-dir> <model id>

`calibration` writes calibration.jsonl, the texts GPTQ and the range
calibration see, one document per line, each with its source and licence:
- `repository`: spans of this repository's own files at REPO_COMMIT (MIT),
  none from a file the evaluation set reads, the parity corpus or the chat
  benchmark's items;
- `wikipedia`: the lead section of each article in WIKIPEDIA, as plain text
  (MediaWiki's TextExtracts), pinned to the revision it was read from
  (CC BY-SA 4.0, attributed to the article's contributors). No article the
  chat benchmark draws on (fixtures/tasks/chat/) is allowed;
- `chat`: conversations written for this fixture (MIT), kept from the
  existing file: they are authored here, not generated.
Fetching reads each article's current revision, so a rebuild picks up newer
revisions; the committed file, not this tool, is the fixture, and the
converter records its sha256.

`evaluation` writes evaluation.jsonl: the teacher-forced texts and greedy
prompts the converter grades a build on, which are #173's (the spike that
first graded MiniCPM5-2B on the ANE), so grades stay comparable with it:
spans of this repository's files at EVAL_COMMIT, a run of numbers, and six
chat prompts.

`tokens` writes <model id>.tokens.json: every document's token ids under the
checkpoint's tokenizer.json (chats through its chat template, with the
template arguments the model is served with), and the sha256 of the two
text files and of the tokenizer. The converter re-encodes and refuses a
mismatch, so a changed tokenizer or text can't go unnoticed.

Requires `git` (spans are read with `git show`), network access for
`calibration`, and tokenizers and transformers for `tokens`.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures" / "causal-lm"
CALIBRATION = FIXTURES / "calibration.jsonl"
EVALUATION = FIXTURES / "evaluation.jsonl"
BENCHMARK = REPO / "fixtures" / "tasks" / "chat"

REPO_COMMIT = "466d7b4"
"""The commit the calibration's repository spans are read at."""

SPAN_CHARS = 6000

REPO_SPANS = [
    ("README.md", 0), ("docs/API.md", 0), ("docs/API.md", 14000), ("docs/BACKENDS.md", 0),
    ("docs/CONVERTING.md", 2000), ("docs/CONVERTING.md", 32000), ("docs/CONVERTING.md", 62000),
    ("docs/PUBLISHING.md", 4000), ("docs/design/classify.md", 5000), ("docs/design/classify.md", 30000),
    ("docs/design/open-llm.md", 3000), ("docs/design/open-llm.md", 26000), ("docs/design/coreml-cache.md", 2000),
    ("crates/sidekick-core/src/manifest.rs", 10000), ("crates/sidekick-core/src/manifest.rs", 90000),
    ("crates/sidekick-embed/src/classify_input.rs", 20000), ("crates/sidekick-embed/src/warm.rs", 4000),
    ("crates/sidekick-fm/src/backend.rs", 12000), ("crates/sidekick-server/src/api/wire.rs", 3000),
    ("crates/sidekick-server/src/fetch/mod.rs", 6000), ("crates/sidekick-fm/swift/bridge.swift", 8000),
    ("tools/sidekick_convert/core.py", 12000), ("tools/sidekick_convert/fp16sim.py", 3000),
    ("tools/ledger.py", 2000), ("tools/release_gate.py", 20000), (".github/workflows/release.yml", 0),
    ("Cargo.toml", 0),
]
"""(path, first character) of each repository span, SPAN_CHARS long. None
reads a file the evaluation set reads (EVAL_SPANS), fixtures/, or
docs/data/ (which holds the chat benchmark's replies)."""

WIKIPEDIA = [
    # science
    "Mitochondrion", "DNA replication", "Enzyme", "Neuron", "Immune system", "Vaccine", "Antimicrobial resistance",
    "Periodic table", "Chemical bond", "Thermodynamics", "Entropy", "Quantum mechanics", "General relativity",
    "Speed of light", "Electromagnetism", "Superconductivity", "Laser", "Nuclear fusion", "Volcano", "Earthquake",
    "Glacier", "Ocean current", "Monsoon", "Tornado", "Water cycle", "Carbon cycle", "Evolution",
    "Natural selection", "CRISPR gene editing", "Fungus", "Lichen", "Blue whale", "Monarch butterfly",
    "Emperor penguin", "African elephant",
    # astronomy
    "Moon", "Mars", "Jupiter", "Sun", "Milky Way", "Exoplanet", "Comet", "Hubble Space Telescope",
    # mathematics and computing
    "Prime number", "Pythagorean theorem", "Calculus", "Linear algebra", "Probability theory", "Graph theory",
    "Algorithm", "Sorting algorithm", "Hash table", "Binary search", "Public-key cryptography", "Compiler",
    "Operating system", "Linux", "Git", "HTTP", "Domain Name System", "Internet protocol suite", "World Wide Web",
    "SQL", "JSON", "Regular expression", "Rust (programming language)", "C (programming language)", "JavaScript",
    "Machine learning", "Unicode", "UTF-8", "Floating-point arithmetic", "Virtual memory", "Solid-state drive",
    "Bluetooth", "Wi-Fi", "Large language model",
    # history
    "Roman Republic", "Ancient Egypt", "Ming dynasty", "Renaissance", "French Revolution", "American Civil War",
    "Silk Road", "Ottoman Empire", "Viking Age", "Magna Carta", "Cold War", "Meiji Restoration", "Inca Empire",
    "Mughal Empire", "Apollo 11",
    # arts
    "Baroque music", "Ludwig van Beethoven", "William Shakespeare", "Hamlet", "Leonardo da Vinci", "Mona Lisa",
    "Cubism", "Haiku", "Opera", "Ballet", "Hip-hop", "Film noir",
    # places
    "Lake Baikal", "Sahara", "Nile", "Himalayas", "Andes", "Kyoto", "Istanbul", "Patagonia", "Galápagos Islands",
    "Mount Everest", "Danube", "Antarctica",
    # everyday life, food and sport
    "Bread", "Cheese", "Tea", "Chocolate", "Bicycle", "Basketball", "Association football", "Tennis",
    "Olympic Games", "Yoga", "Sleep", "Recycling", "Solar power", "Wind power", "Lithium-ion battery", "Library",
    # economics and society
    "Inflation", "Supply and demand", "Central bank", "Stock market", "Game theory", "Democracy", "United Nations",
    "Universal Declaration of Human Rights", "Museum", "Postage stamp",
    # medicine
    "Hypertension", "Influenza", "Malaria", "Anesthesia", "X-ray", "Cholera",
]

EVAL_COMMIT = "4643aa4"
"""The commit #173's evaluation spans were read at (the last to change those
files before its references were made)."""

EVAL_TOKENS = 256
EVAL_SPANS = [
    ("prose_running", "docs/RUNNING.md", 2000), ("prose_decisions", "docs/DECISIONS.md", 60000),
    ("prose_integrating", "docs/INTEGRATING.md", 3000), ("rust_model", "crates/sidekick-coreml/src/model.rs", 4000),
    ("rust_chat", "crates/sidekick-core/src/chat.rs", 3000), ("toml_corpus", "fixtures/parity/corpus.toml", 1000),
    ("numbers", None, None), ("models_table", "docs/MODELS.md", 30000),
]
EVAL_CHARS = 4000
GREEDY_PROMPTS = [
    "Explain in two sentences why a key-value cache speeds up autoregressive decoding.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List three differences between a GPU and a neural processing unit.",
    "Summarize: The daemon serves embeddings, classification and reranking on device, "
    "beside a larger model on the GPU.",
    "What is the capital of France, and what river runs through it?",
    "Translate to French: The weather is nice today, so we will walk to the market.",
]

USER_AGENT = "sidekick-calibration-fixture/1.0 (https://github.com/ironblock/sidekick)"


def git_show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(REPO)}: {len(entries)} entries")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def benchmark_articles():
    """Normalized titles of the Wikipedia articles the chat benchmark uses."""
    titles = set()
    for f in BENCHMARK.glob("*.jsonl"):
        for e in read_jsonl(f):
            src = e.get("source", "")
            if "wikipedia.org" in src:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(src).query)
                titles.add(normalize(q["title"][0]))
    return titles


def normalize(title):
    return urllib.parse.unquote(title).replace("_", " ").strip().casefold()


def fetch_leads(titles):
    """{requested title: (resolved title, revision id, lead text)}, 20 per
    request (TextExtracts' limit for intro extracts)."""
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        params = {"action": "query", "format": "json", "formatversion": "2", "redirects": "1",
                  "prop": "extracts|revisions", "exintro": "1", "explaintext": "1", "exlimit": "max",
                  "rvprop": "ids", "titles": "|".join(batch)}
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        resolved = {t: t for t in batch}
        for step in ("normalized", "redirects"):
            for m in data["query"].get(step, []):
                for k, v in resolved.items():
                    if v == m["from"]:
                        resolved[k] = m["to"]
        pages = {p["title"]: p for p in data["query"]["pages"]}
        for t in batch:
            p = pages.get(resolved[t])
            if p is None or p.get("missing") or not p.get("extract"):
                raise SystemExit(f"no lead section for {t!r}")
            out[t] = (p["title"], p["revisions"][0]["revid"], p["extract"].strip())
    return out


def calibration(args):
    taken = benchmark_articles()
    clash = [t for t in WIKIPEDIA if normalize(t) in taken]
    if clash:
        raise SystemExit(f"articles the chat benchmark uses: {clash}")
    eval_files = {p for _, p, _ in EVAL_SPANS if p}
    for path, _ in REPO_SPANS:
        if path in eval_files or path.startswith(("fixtures/", "docs/data/")):
            raise SystemExit(f"{path}: the evaluation set or a benchmark reads it")
    entries = []
    for path, start in REPO_SPANS:
        text = git_show(REPO_COMMIT, path)[start:start + SPAN_CHARS]
        entries.append({"id": f"repo-{len(entries) + 1:03d}", "kind": "repository", "text": text,
                        "source": f"sidekick repository at {REPO_COMMIT}, {path}, "
                                  f"characters {start}-{start + len(text)}",
                        "licence": "MIT"})
    leads = fetch_leads(WIKIPEDIA)
    resolved = [leads[t][0] for t in WIKIPEDIA]
    clash = [t for t in resolved if normalize(t) in taken]
    if clash or len(set(resolved)) != len(resolved):
        raise SystemExit(f"after redirects: benchmark articles {clash}, or a duplicate")
    for i, t in enumerate(WIKIPEDIA, 1):
        title, revid, text = leads[t]
        url_title = urllib.parse.quote(title.replace(" ", "_"), safe="()_,'-")
        entries.append({"id": f"wiki-{i:03d}", "kind": "wikipedia", "text": text,
                        "source": f"https://en.wikipedia.org/w/index.php?title={url_title}&oldid={revid}",
                        "licence": "CC-BY-SA-4.0",
                        "attribution": f"Wikipedia contributors, \"{title}\", English Wikipedia (lead section, "
                                       "as plain text)"})
    chats = [e for e in read_jsonl(args.chats or CALIBRATION) if e["kind"] == "chat"] if (
        args.chats or CALIBRATION.exists()) else []
    if not chats:
        raise SystemExit("no chat entries: pass --chats <jsonl> the first time")
    entries += chats
    write_jsonl(CALIBRATION, entries)


def evaluation(args):
    entries = []
    for name, path, start in EVAL_SPANS:
        if path is None:
            text = " ".join(f"{i * 37 % 1000}" for i in range(1, 600))
            source = "generated: i * 37 mod 1000 for i = 1..599, space-separated"
        else:
            text = git_show(EVAL_COMMIT, path)[start:start + EVAL_CHARS]
            source = f"sidekick repository at {EVAL_COMMIT}, {path}, characters {start}-{start + len(text)}"
        entries.append({"id": name, "kind": "teacher-forced", "tokens": EVAL_TOKENS, "text": text, "source": source,
                        "licence": "MIT"})
    for i, p in enumerate(GREEDY_PROMPTS, 1):
        entries.append({"id": f"greedy-{i}", "kind": "greedy", "messages": [{"role": "user", "content": p}],
                        "source": "authored (#173)", "licence": "MIT"})
    write_jsonl(EVALUATION, entries)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(args):
    sys.path.insert(0, str(REPO / "tools"))
    from sidekick_convert import causal_lm
    src = Path(args.src).expanduser()
    tok = causal_lm.Tokens.load(src, args.model_id)
    out = {"tokenizer_sha256": sha256(src / "tokenizer.json"), "template_kwargs": tok.template_kwargs,
           "calibration_sha256": sha256(CALIBRATION), "evaluation_sha256": sha256(EVALUATION),
           "calibration": {e["id"]: tok.encode_entry(e) for e in read_jsonl(CALIBRATION)},
           "evaluation": {e["id"]: tok.encode_entry(e) for e in read_jsonl(EVALUATION)}}
    path = FIXTURES / f"{args.model_id}.tokens.json"
    with open(path, "w") as f:
        f.write("{\n")
        keys = list(out)
        for k in keys:
            v = out[k]
            if isinstance(v, dict) and k in ("calibration", "evaluation"):
                f.write(f" {json.dumps(k)}: {{\n")
                items = list(v.items())
                for j, (doc, ids) in enumerate(items):
                    f.write(f"  {json.dumps(doc)}: {json.dumps(ids, separators=(',', ':'))}"
                            f"{',' if j < len(items) - 1 else ''}\n")
                f.write(" }" + ("," if k != keys[-1] else "") + "\n")
            else:
                f.write(f" {json.dumps(k)}: {json.dumps(v)}" + ("," if k != keys[-1] else "") + "\n")
        f.write("}\n")
    n = sum(map(len, out["calibration"].values()))
    print(f"wrote {path.relative_to(REPO)}: calibration {n} tokens in {len(out['calibration'])} documents")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibration")
    c.add_argument("--chats", help="a JSONL of chat entries to use instead of the existing file's")
    sub.add_parser("evaluation")
    t = sub.add_parser("tokens")
    t.add_argument("src", help="local Hugging Face snapshot")
    t.add_argument("model_id", help="the model id (its examples/chat/<id>/chat.toml gives the template arguments)")
    args = ap.parse_args()
    {"calibration": calibration, "evaluation": evaluation, "tokens": tokens}[args.cmd](args)


if __name__ == "__main__":
    main()
