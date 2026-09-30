"""Convert a BERT text-classification checkpoint into ANE-resident Core ML
classifier artifacts for sidekick's `POST /v1/classify`
(docs/design/classify.md). Validated on
nlptown/bert-base-multilingual-uncased-sentiment.

Produces one static-shape .mlmodelc per sequence-length bucket whose output
is the model's logits, statically (1, num_labels), in id2label order.

Usage:
    python tools/convert_bert_classifier.py <hf-model-dir> <install-dir> [buckets...] [--time]

    hf-model-dir: local snapshot of the checkpoint (config.json,
                  model.safetensors, vocab.txt or tokenizer.json)
    install-dir:  classifier directory the daemon scans. Its name is the
                  classifier id: the manifest is copied from
                  examples/classifiers/<name>/classifier.toml, e.g.
                  "~/Library/Application Support/sidekick/models/nlptown-sentiment"
    buckets:      default 128 256 512

Requires: torch, transformers, tokenizers, coremltools, numpy (arm64-native
Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the BERT backbone
with the checkpoint's own sequence-classification head:
- explicit attention with a finite -30000 mask (D25, D15); the fused
  attention op is refused;
- static shapes per bucket, with position_ids and token_type_ids buffers;
- tokenizer.json from the checkpoint, or built from vocab.txt when it ships
  only that (nlptown does), without padding or truncation (sidekick does
  both itself);
- the ANE linear's 2^15 limit (D25 amendment), checked on the gate texts;
  BERT is post-norm, so there is no residual rewrite, only the check;
- the manifest checked against the checkpoint (labels in id2label order,
  problem_type, max_seq_len, source revision).

Gates, per bucket: the fp32 wrapper reproduces the checkpoint's forward (max
|dlogit| <= 1e-3); no fused attention op; compute plan (every
linear/matmul on the ANE and >= 80% of ops); on CPU_AND_NE and CPU_ONLY,
finite logits, argmax agreement with fp32 wherever fp32's top-2 margin is
>= 0.05, max raw |dp| <= 0.02, and pad invariance. Every gate treats NaN as a
failure.

Measured on nlptown (M1 Max, macOS 27.0), with tools/classifier_reference.py
and tools/measure_classifier.py on the 51-input parity corpus against the
checkpoint in fp32: argmax agreement 100% on the ANE, CPU and GPU (46 cases
above the 0.05-logit margin; the 5 near-ties agree too); raw |dp| max 0.0026
on the ANE, 0.0034 on the CPU, 0.0007 on the GPU; 294 of 304 operations on
the ANE; ANE latency 4.3 / 10.4 / 27.3 ms at buckets 128 / 256 / 512 (CPU
18.9 / 32.3 / 62.3 ms). The generated tokenizer.json matches transformers'
slow BertTokenizer on accented, CJK, Cyrillic, emoji and special-token text.
"""

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import bert
from sidekick_convert.heads.sequence import SequenceClassification

_REVIEW = ("The blender arrived quickly and works well, although the lid is a bit loose and "
           "the instructions were confusing at first.")
GATE_TEXTS = [
    "Absolutely terrible. It broke after two days and support never answered.",
    "Not bad, not great. It does the job.",
    "Great value for the money, I would buy it again!",
    "Das Produkt ist in Ordnung, aber die Lieferung hat zu lange gedauert.",
    "Producto excelente, llegó antes de lo previsto.",
    "Service client décevant, je ne recommande pas.",
    "Prodotto perfetto, lo consiglio a tutti.",
    "Het werkt prima, maar de batterij is snel leeg.",
    " ".join([_REVIEW] * 6),
    " ".join([_REVIEW] * 14),
    " ".join([_REVIEW] * 17),
]


def main():
    args = cli.parse(__doc__.split("\n\n")[0])
    model_id = args.install_dir.name
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean"))
    backbone = bert.load(args.src, tok, task="sequence-classification", attention="explicit",
                         token_types="zeros")
    name, value, factor = backbone.check_linear_range(GATE_TEXTS, tok)
    print(f"gate set: {len(GATE_TEXTS)} texts; largest linear output {value:.1f} at {name}, "
          f"{factor:.1f}x under the ANE linear's 2^15")
    job = recipes.classifier(model_id=model_id, src=args.src, buckets=args.buckets, backbone=backbone,
                             head=SequenceClassification(), tok=tok, texts=GATE_TEXTS, timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
