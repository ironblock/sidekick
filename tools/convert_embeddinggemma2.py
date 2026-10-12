"""Convert google/embeddinggemma-2's text tower for sidekick.

Produces one static-shape .mlmodelc per sequence-length bucket: the
checkpoint loaded text-only, its output mean-pooled in the graph (the
server normalizes, as sentence-transformers does). `--format onnx` instead
exports one fp32 model.onnx for ONNX Runtime's CPU backend
(docs/DECISIONS.md D40; tools/sidekick_convert/onnx_export.py).

Usage:
    python tools/convert_embeddinggemma2.py <hf-model-dir> <install-dir> [buckets...] [--format onnx]

    install-dir: model directory the daemon scans, named embeddinggemma-2;
                 the manifest derives from examples/manifests/embeddinggemma-2

Requires: transformers >= 5.19 (EmbeddingGemma 2's own model code) and
sentence-transformers >= 6.1, tokenizers, torch, numpy, coremltools (plus
Xcode for `xcrun coremlcompiler`); onnx and onnxruntime for --format onnx.
The library's other converters are validated on transformers 4.57, so this
one runs from its own environment (docs/CONVERTING.md).

The architecture, why conversion builds the attention masks itself, and
the fp16 range rewrite: backbones/embedding_gemma2.py. The model card warns
that fp16 gives NaN or degraded embeddings. Converted straight to fp16, the
GPU path is unharmed (cosine 0.999999) and the ANE nearly so (0.99994), but
Core ML's CPU path loses 5% (0.947): RMSNorm inputs reach ~1,600, and their
squares overflow. With the range rewrite, measured on an M1 Max at buckets
128, 512 and 2,048: CPU_ONLY 0.99992, CPU_AND_GPU 0.999999, CPU_AND_NE
0.999995.

CPU_ONLY isn't bucket-invariant: Core ML's CPU softmax rounds differently
once the key axis passes 512, so a CPU embedding moves by 1 - cos ~6e-5
between buckets 512 and 1,024 (backbones/embedding_gemma2.py). The GPU
and ANE paths are, within their gates. The matmul softmax fixes the CPU
but broke the ANE at bucket 256 (cosine 0.44), so it isn't used.

The pooled sum needs its own scale: the output projection reaches ~4,000
per token (3,962 on the calibration texts), so a channel's sum over 2,048
tokens can pass fp16's 65504. pool_prescale() keeps the worst case at
32,768 per bucket: 1/16 at 128 tokens, 1/256 at 2,048.

Sliding layers attend where |q - k| <= 512, so the gate set includes a text
past 1,024 tokens, where the band is live.
"""

import math

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import embedding_gemma2
from sidekick_convert.gates import EmbeddingGates
from sidekick_convert.heads.pool import Pool

MODEL_ID = "embeddinggemma-2"
QUERY = "task: search result | query: "
DOCUMENT = "title: none | text: "

_LONG = " ".join(f"Paragraph {i} describes step {i % 9} of a long procedure in some detail, with numbers like "
                 f"{i * 37 % 1000} and names such as Alder, Birch and Cedar." for i in range(60))
GATE_TEXTS = [
    QUERY + "How do I reset my password?",
    DOCUMENT + "A cat sat on the mat.",
    DOCUMENT + "def add(a, b):\n    return a + b  # simple helper\n",
    DOCUMENT + "Order #48213 shipped 2026-09-14; see https://example.com/track?id=48213&ref=a1b2.",
    DOCUMENT + "Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。",
    DOCUMENT + " ".join(["buffalo"] * 300),
    DOCUMENT + _LONG,                                   # past 1,024 tokens: the sliding band is live
]
# calibration adds a text that fills the largest bucket
CALIBRATION_EXTRA = [DOCUMENT + " ".join(["buffalo"] * 2040)]

TOKEN_MAX = 4096.0     # bound on |output| per token (3,962 measured)
SUM_MAX = 32768.0      # cap on |pooled sum| in fp16


def pool_prescale(seq):
    """Power-of-two scale on the pooled sum, so seq tokens at TOKEN_MAX stay
    under SUM_MAX."""
    return 2.0 ** -math.ceil(math.log2(seq * TOKEN_MAX / SUM_MAX))


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None)
    m_buckets = args.buckets or [128, 256, 512, 1024, 2048]
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    from transformers import AutoTokenizer
    if AutoTokenizer.from_pretrained(args.src).pad_token_id != 0:
        raise SystemExit("the server pads input_ids with 0, and this tokenizer's pad id isn't 0")
    backbone = embedding_gemma2.load(args.src, tok)
    texts = [t for t in GATE_TEXTS if len(tokenizer.encode(tok, t)) <= max(m_buckets)]
    longest = max(len(tokenizer.encode(tok, t)) for t in texts)
    if max(m_buckets) > 2 * backbone.window and longest <= 2 * backbone.window:
        raise SystemExit(f"no gate text is longer than the sliding band ({2 * backbone.window}); the longest is {longest}")
    print(f"gate set: {len(texts)} texts, longest {longest} tokens; sliding window {backbone.window} each side")
    calibration = core.Calibration.without_graded(GATE_TEXTS + CALIBRATION_EXTRA)
    rewrites = [embedding_gemma2.traceable_helpers, embedding_gemma2.fp16_range_rewrite(calibration, tok)]
    job = recipes.embedder(model_id=MODEL_ID, src=args.src, buckets=m_buckets, backbone=backbone,
                           head=Pool("mean", prescale=pool_prescale), tok=tok, texts=texts,
                           calibration=calibration, rewrites=rewrites,
                           gates=EmbeddingGates(parity={"CPU_AND_NE": 0.999, "CPU_ONLY": 0.999, "CPU_AND_GPU": 0.999},
                                                pad_id_range=(1000, 40000)),
                           strict_max_seq_len=False, **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
