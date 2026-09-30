"""Convert a BERT-family text-classification or text-ranking checkpoint into
ANE-resident Core ML artifacts for sidekick's `POST /v1/classify` and
`/v1/rerank` (docs/design/classify.md, docs/design/rerank.md). Validated on
nlptown/bert-base-multilingual-uncased-sentiment.

Produces one static-shape .mlmodelc per sequence-length bucket whose output
is the model's logits, statically (1, num_labels), in id2label order. A
reranker (a manifest with task = "text-ranking") scores one (query,
document) pair per prediction: it takes token_type_ids as an input when its
manifest names it, and returns its raw (1, 1) logit.

Usage:
    python tools/convert_bert_classifier.py <hf-model-dir> <install-dir> [buckets...]
        [--twice-gelu] [--tokenizer-sha256 HEX] [--time]

    hf-model-dir: local snapshot of the checkpoint (config.json,
                  model.safetensors, vocab.txt or tokenizer.json)
    install-dir:  classifier directory the daemon scans. Its name is the
                  classifier id: the manifest is copied from
                  examples/classifiers/<name>/classifier.toml, e.g.
                  "~/Library/Application Support/sidekick/models/nlptown-sentiment"
    buckets:      default: the manifest's
    --twice-gelu: opt-in activation rewrite (docs/CONVERTING.md): the erf
                  GELU from erf, mul and add instead of Core ML's coarse
                  native gelu. Changes the artifact; measure it per model.
    --tokenizer-sha256: fail unless the installed tokenizer.json has this
                  SHA-256 (what parity references were generated against).

Requires: torch, transformers, tokenizers, coremltools, numpy (arm64-native
Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the BERT backbone
with the checkpoint's own sequence-classification head:
- explicit attention with a finite -30000 mask (D25, D15); the fused
  attention op is refused;
- static shapes per bucket, with position_ids and token_type_ids buffers;
- tokenizer.json from the checkpoint, or built from vocab.txt when it ships
  only that (nlptown does), without padding or truncation (sidekick does
  both itself); a reranker's pairs are encoded from that file;
- the ANE linear's 2^15 limit (D25 amendment), checked on the gate texts;
  BERT is post-norm, so there is no residual rewrite, only the check;
- the manifest checked against the checkpoint (labels in id2label order,
  problem_type, max_seq_len, source revision). A reranker's problem_type is
  the activation vLLM derives from the checkpoint, and token_type_ids must
  be named exactly when pairs carry segment ids.

Gates, per bucket: the fp32 wrapper reproduces the checkpoint's forward (max
|dlogit| <= 1e-3); no fused attention op; compute plan (every
linear/matmul on the ANE and >= 80% of ops); on CPU_AND_NE and CPU_ONLY,
finite logits, argmax agreement with fp32 wherever fp32's top-2 margin is
>= 0.05, max raw |dp| <= 0.02, and pad invariance. A reranker's score is
gated the same way in sigmoid space (see RANKING_GATES). Every gate treats
NaN as a failure.

Measured on nlptown (M1 Max, macOS 27.0), with tools/classifier_reference.py
and tools/measure_classifier.py on the 51-input parity corpus against the
checkpoint in fp32: argmax agreement 100% on the ANE, CPU and GPU (46 cases
above the 0.05-logit margin; the 5 near-ties agree too); raw |dp| max 0.0026
on the ANE, 0.0034 on the CPU, 0.0007 on the GPU; 294 of 304 operations on
the ANE; ANE latency 4.3 / 10.4 / 27.3 ms at buckets 128 / 256 / 512 (CPU
18.9 / 32.3 / 62.3 ms). The generated tokenizer.json matches transformers'
slow BertTokenizer on accented, CJK, Cyrillic, emoji and special-token text.
"""

from sidekick_convert import cli, core, manifest, recipes, tokenizer
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


_DEHYDRATION = ("Dehydration happens when the body loses more fluid than it takes in. Early signs include "
                "thirst, a dry mouth, dark urine and tiredness; severe cases bring dizziness and confusion.")
# A reranker's score is a raw logit (problem_type regression, identity
# activation), |x| ~ 10 for ms-marco-MiniLM-L6-v2, so a probability-scale gate
# on it would measure the wrong unit. Cross-encoders train with a binary
# cross-entropy, so their error is gated in sigmoid space, as the parity suite
# grades them, with |dlogit| reported alongside. ms-marco-MiniLM-L6-v2 on these
# pairs: |dp| 0.0029 on the ANE and 0.0053 on CPU_ONLY (|dlogit| 0.027, 0.092).
RANKING_GATES = {"activation": "sigmoid"}
RANKING_PAIRS = [
    ("how do I reset my password",
     "To reset your password, open Settings, choose Account, and follow the link we email you."),
    ("how do I reset my password", "Our office is closed on public holidays."),
    ("what is the capital of france", "Paris is the capital and most populous city of France."),
    ("what is the capital of france", "The Eiffel Tower was completed in 1889 for the World's Fair."),
    ("best way to store fresh basil",
     "Keep basil stems in a glass of water at room temperature, loosely covered with a bag."),
    ("wie spät ist es in tokio", "Tokio liegt in der Zeitzone UTC+9 und kennt keine Sommerzeit."),
    ("symptoms of dehydration", " ".join([_DEHYDRATION] * 5)),
    ("symptoms of dehydration", " ".join([_DEHYDRATION] * 12)),
]


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=[
        ("--twice-gelu", {"action": "store_true", "help": "opt-in erf-GELU rewrite (changes the artifact)"}),
        ("--tokenizer-sha256", {"help": "expected SHA-256 of the installed tokenizer.json"}),
    ])
    model_id = args.install_dir.name
    m = manifest.load(manifest.classifier_path(model_id))
    buckets = args.buckets or m["buckets"]
    ranking = m.get("task") == "text-ranking"
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean",
                                           expected_sha256=args.tokenizer_sha256))
    token_types = "input" if "token_type_ids" in m.get("classify", {}).get("io", {}) else "zeros"
    backbone = bert.load(args.src, tok, task="sequence-classification", attention="explicit",
                         token_types=token_types)
    gate = [p for p in RANKING_PAIRS if len(tokenizer.encode_pair(tok, *p)[0]) <= max(buckets)] if ranking \
        else GATE_TEXTS
    name, value, factor = backbone.check_linear_range(gate, tok)
    print(f"gate set: {len(gate)} {'pairs' if ranking else 'texts'}; largest linear output {value:.1f} at "
          f"{name}, {factor:.1f}x under the ANE linear's 2^15")
    job = recipes.classifier(
        model_id=model_id, src=args.src, buckets=buckets, backbone=backbone, head=SequenceClassification(),
        tok=tok, texts=None if ranking else gate, pairs=gate if ranking else None,
        expected_problem_type=manifest.vllm_problem_type(backbone.config) if ranking else None,
        gate_overrides=RANKING_GATES if ranking else None,
        rewrites=[bert.twice_gelu] if args.twice_gelu else (), timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
