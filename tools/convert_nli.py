"""Convert an NLI cross-encoder into Core ML artifacts for sidekick's
zero-shot `format = "nli"` (docs/design/classify.md).

Produces one static-shape .mlmodelc per sequence-length bucket that scores
one (premise, hypothesis) pair and returns the checkpoint's NLI logits,
statically (1, n_classes), in id2label order. sidekick builds one hypothesis
per candidate label from the manifest's template and scores the labels as
transformers' ZeroShotClassificationPipeline does.

Usage:
    python tools/convert_nli.py <hf-model-dir> <install-dir> [buckets...] [--twice-gelu] [--time]

    install-dir: classifier directory the daemon scans. Its name is the
                 classifier id: the manifest is copied from
                 examples/classifiers/<name>/classifier.toml

Requires: torch, transformers 4.57, tokenizers, coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The recipe follows the checkpoint's model_type:
- bert, roberta, xlm-roberta: the BERT backbone with the checkpoint's own
  classification head (explicit attention, finite mask, static positions;
  RoBERTa's offset by padding_idx + 1);
- deberta-v2 (DeBERTa-v3): the DeBERTa backbone, its relative-position
  gathers rewritten as matmuls (techniques/relative_shift.py), with the
  checkpoint's ContextPooler and classifier.
A BERT-family model whose manifest serves it on the ANE (no compute_units)
gets the BERT backbone's ANE rewrites: the erf GELU, the matmul softmax
(bucket invariance) and the linear-floor rescale. Measured on
nli-MiniLM2-L6-H768, they take its ANE path from F to B, so it ships both
ways: nli-minilm2-l6-h768 on the GPU, nli-minilm2-l6-h768-ane on the ANE.
The pair is tokenizer.json's own (premise, hypothesis) encoding. The
graph takes token_type_ids only when the model reads pair segment ids
(type_vocab_size > 1 and nonzero second-segment ids): DeBERTa-v3's
tokenizer emits them, but its embeddings have no token-type table.

Gates, per bucket, on gate pairs built from the default template: the fp32
wrapper reproduces the checkpoint's forward, no fused attention, and on the
path the manifest serves (its compute_units; the compute plan too when
that is the ANE) finite logits, argmax agreement with fp32 above the 0.05
margin, max |dp| of the n-way softmax <= 0.02, and pad invariance. The
other Core ML paths are reported.
"""

import json

from sidekick_convert import cli, core, manifest, recipes, tokenizer
from sidekick_convert.backbones import bert, deberta_v2
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.heads.sequence import SequenceClassification

_LOG = ("2026-10-12T09:14:03Z INFO worker started job=reindex shard=7\n"
        "2026-10-12T09:14:05Z WARN retrying request to storage (attempt 2 of 5)\n")
_TRACE = ("Traceback (most recent call last):\n"
          '  File "app/sync.py", line 88, in run\n'
          "    batch = client.fetch(cursor)\n"
          '  File "app/client.py", line 41, in fetch\n'
          "    raise TimeoutError(f\"no response after {self.timeout}s\")\n"
          "TimeoutError: no response after 30s\n")
GATE_PREMISES = [
    "A man is playing a guitar on a crowded street corner.",
    "The meeting has been moved to Thursday because the director is travelling.",
    "I tried the new flag, but it turns out this version of the compiler doesn't support it.",
    _TRACE,
    "All 214 tests passed in 3.2 seconds. Build artifacts were uploaded.",
    "Der Zug nach Berlin hatte heute Morgen zwei Stunden Verspätung.",
    _LOG * 4,      # 190-220 tokens with a hypothesis: bucket 256
    _LOG * 8,      # 380-430: bucket 512
    _LOG * 16,     # 750-860: bucket 1024, for models that take longer inputs
    _LOG * 36,     # 1,700-1,950: bucket 2048
]
GATE_LABELS = ["music", "travel", "error", "success", "a delay"]


def gate_pairs(template, tok, max_len):
    """(premise, hypothesis) pairs for every gate premise and label, those
    that fit the largest bucket."""
    pairs = [(p, template.format(label)) for p in GATE_PREMISES for label in GATE_LABELS]
    return [p for p in pairs if len(tokenizer.encode_pair(tok, *p)[0]) <= max_len]


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=[
        ("--twice-gelu", {"action": "store_true", "help": "opt-in erf-GELU rewrite (changes the artifact)"}),
    ])
    model_id = args.install_dir.name
    m = manifest.load(manifest.classifier_path(model_id))
    if m.get("classify", {}).get("format") != "nli":
        raise SystemExit(f"{model_id}: not an nli-format manifest")
    buckets = args.buckets or m["buckets"]
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean"))
    model_type = json.loads((args.src / "config.json").read_text())["model_type"]
    if model_type == "deberta-v2":
        backbone = deberta_v2.load(args.src, tok, task="sequence-classification")
        twice = deberta_v2.twice_gelu
    elif model_type in ("bert", *bert.OFFSET_MODEL_TYPES):
        config = json.loads((args.src / "config.json").read_text())
        _, types = tokenizer.encode_pair(tok, "a", "b")
        reads = config.get("type_vocab_size", 0) > 1 and any(types)
        backbone = bert.load(args.src, tok, task="sequence-classification", attention="explicit",
                             token_types="input" if reads else "zeros")
        twice = bert.twice_gelu
    else:
        raise SystemExit(f"model_type {model_type!r} has no NLI recipe (bert, roberta, xlm-roberta, deberta-v2)")
    template = m["classify"]["nli"].get("hypothesis_template", "This example is {}.")
    gate = gate_pairs(template, tok, max(buckets))
    rewrites = [twice] if args.twice_gelu else []
    if manifest.served_path(m) == "CPU_AND_NE":
        # served on the ANE: the BERT backbone's ANE rewrites (its docstring)
        if model_type == "deberta-v2":
            raise SystemExit("DeBERTa-v3 NLI models are served on the GPU (compute_units = \"cpu_and_gpu\"): "
                             "on the ANE they are slower and fail bucket invariance")
        rewrites = [bert.twice_gelu, bert.matmul_softmax_rewrite, bert.linear_floor_rewrite(gate, tok)]
    name, value, factor = backbone.check_linear_range(gate, tok)
    print(f"gate set: {len(gate)} pairs; largest linear output {value:.1f} at {name}, "
          f"{factor:.1f}x under the ANE linear's 2^15")
    job = recipes.classifier(
        model_id=model_id, src=args.src, buckets=buckets, backbone=backbone, head=SequenceClassification(),
        tok=tok, pairs=gate, rewrites=rewrites, timing=args.time,
        gate_overrides=ClassifierGates.paths(manifest.served_path(m), report=("CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU")),
        **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
