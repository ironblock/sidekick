"""Convert google/embeddinggemma-2's text tower for sidekick.

`--format onnx` exports one fp32 model.onnx for ONNX Runtime's CPU backend
(docs/DECISIONS.md D40; tools/sidekick_convert/onnx_export.py): the
checkpoint loaded text-only, its output mean-pooled and normalized by the
server, as sentence-transformers does. The Core ML conversion is not
written yet: the model card warns that fp16 gives NaN or degraded
embeddings, and its ranges are being measured first.

Usage:
    python tools/convert_embeddinggemma2.py <hf-model-dir> <install-dir> --format onnx [buckets...]

    install-dir: model directory the daemon scans, named embeddinggemma-2;
                 the manifest derives from examples/manifests/embeddinggemma-2

Requires: transformers >= 5.19 (EmbeddingGemma 2's own model code),
tokenizers, torch, numpy; onnx and onnxruntime for --format onnx. The
library's other converters are validated on transformers 4.57, so this
one runs from its own environment.

The architecture and why export builds the attention masks itself:
backbones/embedding_gemma2.py. Sliding layers attend where |q - k| <= 512,
so the gate set includes a text past 1,024 tokens, where the band is live.
"""

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import embedding_gemma2
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


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None)
    if args.format != "onnx":
        raise SystemExit("EmbeddingGemma 2 converts to ONNX only so far (--format onnx)")
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
    job = recipes.embedder(model_id=MODEL_ID, src=args.src, buckets=m_buckets, backbone=backbone,
                           head=Pool("mean"), tok=tok, texts=texts, strict_max_seq_len=False,
                           **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
