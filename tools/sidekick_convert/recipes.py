"""Recipes: a backbone, a head and a committed manifest, made into a core.Job.

`embedder()` and `classifier()` are the two shapes every converter so far
takes. They build the evaluation cases (token ids from tokenizer.json, fp32
references from the checkpoint's own unpadded forward), check the manifest
against the checkpoint, and compose the per-bucket wrapper. Anything
model-specific (gate texts, rewrites, extra ports) stays in the converter or
its backbone.
"""

from . import manifest as _manifest
from . import tokenizer as _tok
from .core import Case, Evaluation, Job, FUSED_ATTENTION, text_ports
from .gates import ClassifierGates, EmbeddingGates
from .wrapper import compose


def evaluation(backbone, head, tok, texts, max_len, pairs=None, truncate=False):
    """Cases for gate texts (or (a, b) text pairs). A text longer than the
    largest bucket is an error, since it would silently gate nothing, unless
    `truncate`: then it is cut as the server cuts it, keeping the first
    max_len - 1 tokens and the final one (the EOS that last-token pooling
    reads, docs/DECISIONS.md D20), and its reference is computed on that."""
    cases = []
    for i, item in enumerate(pairs if pairs is not None else texts):
        if pairs is not None:
            ids, types = _tok.encode_pair(tok, *item)
            extra = {"token_type_ids": types}
        else:
            ids, types, extra = _tok.encode(tok, item), None, {}
        if len(ids) > max_len and truncate and types is None:
            ids = ids[:max_len - 1] + ids[-1:]
        if len(ids) > max_len:
            raise SystemExit(f"gate input {i} is {len(ids)} tokens, longer than the largest bucket {max_len}")
        ref = head.reference(backbone.reference(ids, types))
        cases.append(Case(ids=ids, ref=ref, extra=extra, label=str(i)))
    return Evaluation(cases)


def _apply(backbone, rewrites):
    for rewrite in rewrites:
        rewrite(backbone)


def embedder(*, model_id, src, buckets, backbone, head, tok, texts, calibration=None, gates=None,
             forbid_ops=frozenset({FUSED_ATTENTION}), rewrites=(), strict_max_seq_len=True,
             truncate=False, negative_control=False, timing=False, int8_embedding=False,
             ignore_ane_weight_cap=False, chunks=None, chunk_identity_all=False):
    """An embedding job; installs examples/manifests/<model_id>/manifest.toml.
    `rewrites` (functions of the backbone) run after the fp32 references are
    computed from the unmodified checkpoint."""
    path = _manifest.embedder_path(model_id)
    m = _manifest.load(path)
    head.bind(backbone)
    _manifest.check_embedder(m, src=src, buckets=buckets, backbone=backbone, head=head,
                             strict_max_seq_len=strict_max_seq_len)
    ports = text_ports(token_type_ids=getattr(backbone, "token_types", None) == "input")
    cases = evaluation(backbone, head, tok, texts, max(buckets), truncate=truncate)
    _apply(backbone, rewrites)
    make_wrapper, example = compose(backbone, head, ports)
    return Job(name=model_id, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
               example=example, evaluation=cases,
               gates=gates or EmbeddingGates(pad_id_range=(1000, min(30000, backbone.vocab_size))),
               calibration=calibration, forbid_ops=frozenset(forbid_ops) | backbone.forbid_ops,
               install_files=[(path, "manifest.toml")],
               negative_control=negative_control, timing=timing, int8_embedding=int8_embedding,
               ignore_ane_weight_cap=ignore_ane_weight_cap, chunks=chunks, chunk_identity_all=chunk_identity_all,
               backbone=backbone, head=head)


def classifier(*, model_id, src, buckets, backbone, head, tok, texts=None, pairs=None, calibration=None,
               gates=None, gate_overrides=None, expected_problem_type=None, rewrites=(),
               strict_max_seq_len=True, landing_required=True, negative_control=False, timing=False,
               int8_embedding=False, ignore_ane_weight_cap=False, chunks=None, chunk_identity_all=False):
    """A classification job; installs examples/classifiers/<model_id>/classifier.toml.
    The gates compare outputs after the manifest's activation; `gate_overrides`
    adjusts ClassifierGates' thresholds. `rewrites` run after the fp32
    references, as in embedder()."""
    path = _manifest.classifier_path(model_id)
    m = _manifest.load(path)
    head.bind(backbone)
    token_type_input = getattr(backbone, "token_types", None) == "input"
    _manifest.check_classifier(m, src=src, buckets=buckets, backbone=backbone, head=head, tok=tok,
                               token_type_input=token_type_input, expected_problem_type=expected_problem_type,
                               strict_max_seq_len=strict_max_seq_len)
    activation = {"single_label": "softmax" if head.num_labels > 1 else "sigmoid",
                  "multi_label": "sigmoid", "regression": "identity"}[m.get("problem_type", "single_label")]
    ports = text_ports(token_type_ids=token_type_input)
    cases = evaluation(backbone, head, tok, texts, max(buckets), pairs=pairs)
    _apply(backbone, rewrites)
    make_wrapper, example = compose(backbone, head, ports)
    return Job(name=model_id, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
               example=example, evaluation=cases, forbid_ops=frozenset({FUSED_ATTENTION}) | backbone.forbid_ops,
               gates=gates or ClassifierGates(**{"activation": activation,
                                                 "pad_id_range": (1000, min(30000, backbone.vocab_size)),
                                                 **(gate_overrides or {})}),
               calibration=calibration, install_files=[(path, "classifier.toml")],
               negative_control=negative_control, timing=timing, landing_required=landing_required,
               int8_embedding=int8_embedding, ignore_ane_weight_cap=ignore_ane_weight_cap, chunks=chunks,
               chunk_identity_all=chunk_identity_all, backbone=backbone, head=head)
