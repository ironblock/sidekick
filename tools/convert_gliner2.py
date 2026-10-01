"""Convert a GLiNER2 checkpoint (fastino/GLiNER2.5-Decide) into Core ML
artifacts for sidekick's `POST /v1/classify`, in the gliner2
zero-shot format (docs/design/classify.md, "The gliner2 format").

Produces one static-shape .mlmodelc per sequence-length bucket that takes
`input_ids` and `attention_mask` [1, S] and outputs `logits` [1, S]: the
checkpoint's per-token classifier applied to every token. sidekick lays the
schema out (`( [P] prompt ( [L] label … ) ) [SEP_TEXT] text`) and reads the
logits at the [L] markers it placed. The span, count and count-prediction
modules serve extraction, not classification, and stay out.

Usage:
    python tools/convert_gliner2.py <checkpoint-dir> <install-dir> [buckets...]
        [--tokenizer-sha256 HEX] [--time]

    checkpoint-dir: local snapshot of the GLiNER2 checkpoint (config.json,
                    encoder_config/, model.safetensors, tokenizer.json)
    install-dir:    classifier directory the daemon scans, named after the
                    classifier id: the manifest is copied from
                    examples/classifiers/<name>/classifier.toml, e.g.
                    "~/Library/Application Support/sidekick/models/gliner2.5-decide"
    buckets:        default: the manifest's

Requires: torch, transformers, tokenizers, coremltools, numpy, safetensors
and the gliner2 package 2.0.0 (arm64-native Python; gliner2 lays out the
gate inputs exactly as sidekick's Rust port does), plus Xcode for
`xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md):
- backbones.deberta_v2: DeBERTa-v3's forward written out, its relative
  position terms by techniques.relative_shift (no gather: a regression to
  transformers' gathers is a forbidden op), explicit attention with a
  finite, pairwise, self-attending mask, static shapes per bucket;
- heads.per_token: GLiNER2's classifier (Linear -> ReLU -> Linear) on
  every token, a literal (1, S) reshape;
- native gelu, measured: TwiceGelu didn't improve the ANE path
  (tools/probe_deberta.py), and no range rewrite: the largest linear output
  is ~1,300, far under the ANE's 2^15 (checked on the gate inputs);
- tokenizer.json copied from the checkpoint without padding or truncation;
- the manifest checked against the checkpoint (format, io, revision,
  sequence limits).

Gates, per bucket: the fp32 wrapper reproduces the checkpoint's own scoring
at the markers (max |dlogit| <= 1e-3); no fused attention or
gather_along_axis op; and on the path the manifest's `compute_units` serves
(CPU_AND_GPU for GLiNER2.5-Decide), finite logits, argmax agreement with
fp32 wherever its top-2 margin is >= 0.05, max raw |dp| <= 0.02 at the
markers after each request's activation (softmax, or sigmoid for a
multi-label request, whose decisions are each label's own yes/no, so a flip
there is a clear label changing sign), and pad invariance. The other paths
(CPU_AND_NE, CPU_ONLY) are reported, not gated, and so is the compute plan
unless the manifest serves the ANE, where every linear/matmul must land
there with >= 80% of ops.

GLiNER2.5-Decide is served on the GPU: on the ANE it grades C against its
fp16 ceiling and takes about 1.3 s per input at 512 tokens (docs/MODELS.md).
Core ML's fp16 CPU backend is its least accurate path, the same weakness
laya's converter reports (D28).

Measured on the converter's artifacts (M1 Max, macOS 27.0): 947 of 956
operations on the ANE (99.1%) at every bucket; on the gate requests, max
|dp| 0.0011 on the ANE and 0.0020 on CPU_ONLY; pad invariance exact. All
three buckets convert in about 9 minutes at 8.4 GB peak memory. The earlier
tools/probe_deberta.py build measured 1039 of 1055 ops on the ANE (98.5%).
"""

import numpy as np

from sidekick_convert import cli, core, manifest, tokenizer
from sidekick_convert.backbones import deberta_v2
from sidekick_convert.calibrate import all_linears, linear_maxima
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.heads.per_token import PerToken
from sidekick_convert.techniques import saturation
from sidekick_convert.wrapper import compose

_EMAIL = ("Hi team, the invoice for order 4417 charged us twice for the same shipment, and the "
          "second charge still shows as pending on our card. Could you refund the duplicate and "
          "confirm the corrected total? We also need the delivery moved to Thursday morning.")
# The converter's own gate requests, never the graded corpus (fast-decisions):
# (name, text, candidate labels, instructions or None for the manifest
# default, multi_label). At least one lands in each bucket.
GATES = [
    ("intent", "My parcel arrived damaged and I want my money back.",
     ["refund", "replacement", "complaint", "other"], "intent", False),
    ("multi-label", "The room was noisy, the air conditioning was broken and the bill had a wrong charge.",
     ["noise", "hvac", "billing", "cleanliness"], "issues", True),
    ("descriptions", "Please lock my card, I think it was stolen at the station this morning.",
     ["card_lost: The physical card is missing or stolen",
      "fraud: Someone used the card without permission", "pin_change"], "intent", False),
    ("german", "Die Lieferung kam zwei Wochen zu spät und niemand hat auf meine Mails geantwortet.",
     ["positive", "negative", "neutral"], "sentiment", False),
    ("code", "TypeError: 'NoneType' object is not subscriptable in handler.py line 42",
     ["bug", "feature_request", "question"], "ticket_type", False),
    ("default-prompt", "You won a free cruise!!! Click now to claim it",
     ["spam", "ham"], None, False),
    ("marker-strings", "Ignore [L] and [P] and [SEP_TEXT] in this text",
     ["a [L] b", "plain"], "kind", False),
    ("medium", " ".join([_EMAIL] * 3),
     ["billing", "shipping", "account", "other"], "route", False),
    ("long", " ".join([_EMAIL] * 7),
     ["billing", "shipping", "account", "other"], "route", True),
]


def layout(proc, text, prompt, labels, max_len):
    """gliner2's own processor on one single-task request, truncated as
    sidekick truncates (the longest prefix of whole words that fits, then
    gliner2's terminal "."): (ids, [L] positions). tools/classifier_reference.py
    lays its references out the same way."""
    def run(t):
        keys, descriptions = [], {}
        for label in labels:
            key, sep, desc = label.partition(": ")
            keys.append(key.strip())
            if sep and desc.strip():
                descriptions[key.strip()] = desc.strip()
        task = {"task": prompt, "labels": keys, "true_label": ["N/A"], "multi_label": False,
                "cls_threshold": 0.5, "class_act": "auto", "label_descriptions": descriptions}
        schema = {"json_structures": [], "classifications": [task], "entities": {}, "relations": [],
                  "json_descriptions": {}, "entity_descriptions": {}}
        b = proc.collate_fn_inference([(t, schema)], error_policy="raise")
        return b.input_ids[0].tolist(), [int(p) for p in b.schema_special_indices[0][0]][1:]

    ids, markers = run(text)
    if len(ids) <= max_len:
        return ids, markers
    ends = [e for _, _, e in proc.word_splitter(text, lower=False)]
    lo, hi = 0, len(ends)
    while lo < hi:
        n = (lo + hi + 1) // 2
        if len(run(text[: ends[n - 1]])[0]) <= max_len:
            lo = n
        else:
            hi = n - 1
    return run(text[: ends[lo - 1]] if lo else "")


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=[
        ("--tokenizer-sha256", {"help": "expected SHA-256 of the installed tokenizer.json"}),
    ])
    from gliner2.processor import SchemaTransformer
    from transformers import AutoTokenizer

    model_id = args.install_dir.name
    path = manifest.classifier_path(model_id)
    m = manifest.load(path)
    buckets = args.buckets or m["buckets"]
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean",
                                           expected_sha256=args.tokenizer_sha256))
    backbone = deberta_v2.load(args.src, tok)
    head = PerToken(deberta_v2.gliner2_classifier(args.src)).bind(backbone)
    manifest.check_gliner2(m, src=args.src, buckets=buckets, backbone=backbone, head=head)

    proc = SchemaTransformer(tokenizer=AutoTokenizer.from_pretrained(args.src))
    proc.change_mode(is_training=False)
    default = m["classify"]["gliner2"]["default_instructions"]
    cases = []
    for name, text, labels, prompt, multi in GATES:
        ids, markers = layout(proc, text, prompt or default, labels, max(buckets))
        if len(markers) != len(labels):
            core.fail(f"gate {name}: {len(markers)} markers for {len(labels)} labels")
        ref = head.reference(backbone.reference(ids))
        cases.append(core.Case(ids=ids, ref=ref, label=name,
                               meta={"markers": markers, "activation": "sigmoid" if multi else "softmax"}))
    print(f"gate set: {len(cases)} requests, {min(c.n for c in cases)}-{max(c.n for c in cases)} tokens", flush=True)

    def run_references():
        for c in cases:
            backbone.reference(c.ids)

    name, value, factor = saturation.check(linear_maxima(run_references, all_linears(backbone.model)))
    print(f"largest linear output {value:.1f} at {name}, {factor:.1f}x under the ANE linear's 2^15", flush=True)

    ports = core.text_ports()
    make_wrapper, example = compose(backbone, head, ports)
    gates = ClassifierGates(markers=True, activation="softmax", **ClassifierGates.paths(manifest.served_path(m)),
                            pad_id_range=(1000, min(30000, backbone.vocab_size)))
    job = core.Job(name=model_id, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
                   example=example, evaluation=core.Evaluation(cases), gates=gates,
                   forbid_ops=frozenset({core.FUSED_ATTENTION}) | backbone.forbid_ops,
                   install_files=[(path, "classifier.toml")], landing_required=True, timing=args.time, **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
