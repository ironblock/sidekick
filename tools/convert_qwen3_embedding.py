"""Convert a Qwen3-class causal-decoder embedding model into ANE-resident
Core ML artifacts. Validated on codefuse-ai/F2LLM-v2-160M; the same recipe
covers Qwen3/Qwen3-Embedding-derived last-token embedders (dims read from
config, prefixes read from config_sentence_transformers.json).

Produces one static-shape .mlmodelc per sequence-length bucket, with
LAST-TOKEN pooling baked into the graph, matching the model's manifest.

Usage:
    python tools/convert_qwen3_embedding.py <hf-model-dir> <install-dir> [buckets...] [--time]

    install-dir:  model directory the daemon scans. Its name is the model id:
                  the manifest is copied from examples/manifests/<name>/manifest.toml

Requires: torch, transformers >= 4.51 (Qwen3), tokenizers, coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the Qwen3 backbone
with a last-token pooling head. Qwen3 was the first CAUSAL DECODER used for
embeddings here (docs/DECISIONS.md D20). What it handles:

A. CAUSAL + PADDING MASK, fp16-safe: transformers' create_causal_mask fills
   with finfo.min (-inf in fp16); the backbone builds the same causal and
   key-padding mask with -30000. Right padding is correct for a causal
   model: the last real token never sees the pads.
B. LAST-TOKEN POOLING in-graph, without a data-dependent index:
   mask * (1 - shift_left(mask)) is 1 exactly at the last real position, and
   a masked sum picks it. The embedding IS the trailing EOS token, so the
   server's truncation keeps it (D20).
C. RoPE and grouped-query attention without shape arithmetic (traceable
   rotate_half and repeat_kv, also where transformers' sdpa path calls them).
D. PRECISION REWRITE (D20 amendment): the coarse native silu becomes
   TanhSilu, and attention's small inputs are rescaled by powers of two,
   undone before the residual add. Calibrated on the texts below, minus any
   the graded parity corpus holds (the scales are the same either way).

No fp16 range rewrite: QK-norm keeps activations small (max ~420).

Gates, per bucket: fp32 exactness of the rewritten wrapper (cosine >=
0.99999); no fused attention, silu or gelu op; compute plan; parity >= 0.999
on CPU_AND_NE and CPU_ONLY; finite output; pad invariance. Every gate treats
NaN as a failure.
"""

import json

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import qwen3
from sidekick_convert.heads.pool import Pool

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]
# alternate query/document so both prefix paths are tested
QUERY_FLAGS = [True, False, True, False, False]

# Varied text for constraint D's activation statistics: prose, code, numbers
# and URLs, punctuation, non-English, and degenerate repetition. The query
# prompt is added where marked.
CALIBRATION_TEXTS = [(s, q) for s, q in zip(PARITY_SENTENCES, QUERY_FLAGS)] + [
    ("def add(a, b):\n    return a + b  # simple helper\n", False),
    ("Order #48213 shipped 2026-09-14; see https://example.com/track?id=48213&ref=a1b2.", False),
    ("Wait... what?! (No, really — \"that\" isn't it.) [1] {2} <3>", False),
    ("Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。", False),
    (" ".join(["buffalo"] * 60), False),
    ("3.14159 2.71828 1.41421 6.02214076e23 299792458", True),
]


def load_prompts(src):
    cfg = json.loads((src / "config_sentence_transformers.json").read_text())
    p = cfg.get("prompts", {})
    return p.get("query", ""), p.get("document", "")


def main():
    args = cli.parse(__doc__.split("\n\n")[0])
    model_id = args.install_dir.name
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    qprefix, dprefix = load_prompts(args.src)
    full = lambda text, is_query: (qprefix if is_query else dprefix) + text
    backbone = qwen3.load(args.src, tok)
    print(f"dims={backbone.hidden_size} qprefix={qprefix[:40]!r} dprefix={dprefix!r}")
    calibration = core.Calibration.without_graded([full(t, q) for t, q in CALIBRATION_TEXTS])
    job = recipes.embedder(
        model_id=model_id, src=args.src, buckets=args.buckets, backbone=backbone, head=Pool("last_token"),
        tok=tok, texts=[full(s, q) for s, q in zip(PARITY_SENTENCES, QUERY_FLAGS)], calibration=calibration,
        rewrites=[qwen3.precision_rewrite(calibration, tok)], strict_max_seq_len=False,
        truncate=True,   # the long text is 520 tokens: gate the server's truncation, which keeps the EOS
        timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
