"""The spike's fixed text sets, built deterministically from files in this
repository (prose, Rust, TOML) plus chat-formatted prompts, so nothing is
fetched. Calibration and evaluation texts never overlap.

eval_texts(tok, n_tokens): token-id lists of exactly n_tokens each.
calib_texts(tok): token-id lists for the RMSNorm range calibration.
prompts(): chat prompts for greedy continuation."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _read(rel, start, n_chars):
    t = (REPO / rel).read_text(encoding="utf-8")
    return t[start:start + n_chars]


def _sources():
    # (name, text): evaluation sources, each long enough for 320+ tokens
    return [
        ("prose_running", _read("docs/RUNNING.md", 2000, 4000)),
        ("prose_decisions", _read("docs/DECISIONS.md", 60000, 4000)),
        ("prose_integrating", _read("docs/INTEGRATING.md", 3000, 4000)),
        ("rust_model", _read("crates/sidekick-coreml/src/model.rs", 4000, 4000)),
        ("rust_chat", _read("crates/sidekick-core/src/chat.rs", 3000, 4000)),
        ("toml_corpus", _read("fixtures/parity/corpus.toml", 1000, 4000)),
        ("numbers", " ".join(f"{i * 37 % 1000}" for i in range(1, 600))),
        ("models_table", _read("docs/MODELS.md", 30000, 4000)),
    ]


def _calib_sources():
    return [
        _read("docs/DECISIONS.md", 120000, 3000),
        _read("docs/MODELS.md", 90000, 3000),
        _read("crates/sidekick-embed/src/lib.rs", 2000, 3000),
        _read("README.md", 0, 3000),
    ]


CHATS = [
    "Explain in two sentences why a key-value cache speeds up autoregressive decoding.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List three differences between a GPU and a neural processing unit.",
    "Summarize: The daemon serves embeddings, classification and reranking on device, beside a larger model on the GPU.",
    "What is the capital of France, and what river runs through it?",
    "Translate to French: The weather is nice today, so we will walk to the market.",
]


def chat_ids(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True)


def eval_texts(tok, n_tokens):
    out = []
    for name, text in _sources():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < n_tokens:
            raise ValueError(f"{name}: {len(ids)} tokens < {n_tokens}")
        out.append((name, ids[:n_tokens]))
    return out


def calib_texts(tok):
    ids = [tok(t, add_special_tokens=False)["input_ids"][:512] for t in _calib_sources()]
    ids += [chat_ids(tok, c) + tok("Sure. Here is the answer you asked for.", add_special_tokens=False)["input_ids"]
            for c in CALIB_CHATS]
    return ids


CALIB_CHATS = [
    "Describe how a hash map handles collisions.",
    "Give me a haiku about autumn leaves.",
    "How many minutes are there in a week? Show the arithmetic.",
]


def prompts():
    return CHATS


# GPTQ calibration (gptq.py): spans of repository files that the eval texts
# don't read, plus chat-formatted prompts with a short answer
_GPTQ_SOURCES = [
    ("docs/DECISIONS.md", 0), ("docs/DECISIONS.md", 20000), ("docs/DECISIONS.md", 100000),
    ("docs/DECISIONS.md", 160000), ("docs/DECISIONS.md", 200000), ("docs/DECISIONS.md", 240000),
    ("docs/MODELS.md", 0), ("docs/MODELS.md", 60000), ("docs/MODELS.md", 130000),
    ("docs/CONVERTING.md", 0), ("docs/CONVERTING.md", 40000), ("docs/API.md", 0), ("docs/BACKENDS.md", 0),
    ("docs/PUBLISHING.md", 0), ("crates/sidekick-core/src/manifest.rs", 20000),
    ("crates/sidekick-core/src/manifest.rs", 120000), ("crates/sidekick-embed/src/classify_input.rs", 10000),
    ("crates/sidekick-fm/src/backend.rs", 5000), ("crates/sidekick-server/src/warm.rs", 2000),
    ("crates/sidekick-embed/src/placement.rs", 2000),
]
_GPTQ_CHATS = [
    "Write a short email declining a meeting invitation politely.",
    "What are the main causes of inflation? Answer briefly.",
    "Give a JSON object with the fields name, age and city for a fictional person.",
    "Explain recursion to a ten-year-old.",
    "Suggest a title for an article about saving energy at home.",
    "List five tags for a photo of a mountain lake at sunrise.",
]


def gptq_texts(tok, n_tokens, seg=512):
    out = []
    for rel, start in _GPTQ_SOURCES:
        ids = tok(_read(rel, start, max(4000, 8 * seg)), add_special_tokens=False)["input_ids"][:seg]
        if len(ids) >= 64:
            out.append(ids)
        if sum(map(len, out)) >= n_tokens:
            break
    for c in _GPTQ_CHATS:
        out.append(chat_ids(tok, c) + tok("<think>\nThe user asks for something short. I will answer directly.\n"
                                          "</think>\nHere is a concise answer.", add_special_tokens=False)["input_ids"])
    return out
