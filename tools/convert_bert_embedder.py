"""Convert a BERT-family sentence-transformers embedder into ANE-resident Core
ML artifacts. For models such as sentence-transformers/all-MiniLM-L6-v2 and
intfloat/e5-small-v2.

Produces one static-shape .mlmodelc per sequence-length bucket, with the
model's pooling (mean or CLS) baked into the graph and output `embedding`,
statically (1, hidden size). sidekick L2-normalizes it and applies the
manifest's prefixes (e5's "query: " / "passage: "); the graph does neither.

Usage:
    python tools/convert_bert_embedder.py <hf-model-dir> <install-dir> [buckets...]
        [--pooling cls|mean] [--twice-gelu] [--tokenizer-sha256 HEX] [--time]

    hf-model-dir: local snapshot of the sentence-transformers checkpoint
    install-dir:  model directory the daemon scans. Its name is the model id:
                  the manifest is copied from examples/manifests/<name>/manifest.toml
    buckets:      default: the manifest's (e.g. 64 128 256 for all-MiniLM-L6-v2)
    --pooling:    override the checkpoint's sentence-transformers pooling
    --twice-gelu: opt-in activation rewrite (docs/CONVERTING.md); changes the
                  artifact, so measure it per model
    --tokenizer-sha256: fail unless the installed tokenizer.json has this
                  SHA-256 (what parity references were generated against)

Requires: torch, transformers, tokenizers, coremltools, numpy (arm64-native
Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the BERT backbone
with explicit attention and a finite mask, and a pooling head:
- mean pooling masks by the attention_mask input, never by token id
  (sidekick pads with id 0), and sums at 1/32 to stay in fp16 range;
- tokenizer.json is copied, or cleaned of padding and truncation, which
  sidekick does itself;
- the manifest must agree with the checkpoint: max_seq_len equals
  sentence-transformers' max_seq_length and the largest bucket, dims the
  hidden size, pooling the checkpoint's.
Gates per bucket as in docs/CONVERTING.md: fp32 exactness, no fused
attention, compute plan, parity >= 0.999 on CPU_AND_NE and CPU_ONLY, pad
invariance. The parity suite (D26) grades the result against
sentence-transformers.
"""

from sidekick_convert import cli, core, manifest, recipes, tokenizer
from sidekick_convert.backbones import bert
from sidekick_convert.heads.pool import Pool

_LONG = ("The river rose steadily through the night, and by morning the lower fields were under water. "
         "Farmers moved their animals to higher ground while volunteers filled sandbags near the bridge.")
GATE_TEXTS = [
    "How do I change the battery in my smoke detector?",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Quantum entanglement links the states of particles across distance.",
    "Der Zug nach München fährt um halb neun vom Gleis vier ab.",
    "¿Dónde está la estación de autobuses más cercana?",
    "def mean(xs):\n    return sum(xs) / len(xs)",
    "Order #58231 shipped on 2026-03-14; tracking: https://example.com/t/58231",
    " ".join([_LONG] * 2),     # ~75 tokens
    " ".join([_LONG] * 5),     # ~180
    " ".join([_LONG] * 11),    # ~400
]


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=[
        ("--pooling", {"choices": ["cls", "mean"], "help": "override the checkpoint's pooling"}),
        ("--twice-gelu", {"action": "store_true", "help": "opt-in erf-GELU rewrite (changes the artifact)"}),
        ("--tokenizer-sha256", {"help": "expected SHA-256 of the installed tokenizer.json"}),
    ])
    model_id = args.install_dir.name
    m = manifest.load(manifest.embedder_path(model_id))
    buckets = args.buckets or m["buckets"]
    pooling = args.pooling or {"cls": "cls", "mean": "mean"}.get(tokenizer.st_pooling(args.src))
    if pooling is None:
        raise SystemExit(f"{args.src}: sentence-transformers pooling {tokenizer.st_pooling(args.src)!r} "
                         "isn't cls or mean; pass --pooling")
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean",
                                           expected_sha256=args.tokenizer_sha256))
    backbone = bert.load(args.src, tok, attention="explicit", token_types="zeros")
    texts = [t for t in GATE_TEXTS if len(tokenizer.encode(tok, t)) <= max(buckets)]
    name, value, factor = backbone.check_linear_range(texts, tok)
    print(f"{pooling} pooling; gate set: {len(texts)} texts; largest linear output {value:.1f} at {name}, "
          f"{factor:.1f}x under the ANE linear's 2^15")
    job = recipes.embedder(model_id=model_id, src=args.src, buckets=buckets, backbone=backbone,
                           head=Pool(pooling), tok=tok, texts=texts,
                           rewrites=[bert.twice_gelu] if args.twice_gelu else (), timing=args.time, **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
