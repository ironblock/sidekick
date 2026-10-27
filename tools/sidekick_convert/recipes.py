"""Recipes: a backbone, a head and a committed manifest, made into a core.Job.

`embedder()` and `classifier()` are the two shapes every encoder converter
takes. They build the evaluation cases (token ids from tokenizer.json, fp32
references from the checkpoint's own unpadded forward), check the manifest
against the checkpoint, and compose the per-bucket wrapper. Anything
model-specific (gate texts, rewrites, extra ports) stays in the converter or
its backbone. `causal_lm()` makes a causal_lm.Job instead, for the
decoder's own pipeline.
"""

from pathlib import Path

from . import manifest as _manifest
from . import tokenizer as _tok
from .core import Case, Evaluation, Job, FUSED_ATTENTION, text_ports
from .gates import ClassifierGates, EmbeddingGates
from .wrapper import compose


def evaluation(backbone, head, tok, texts, max_len, pairs=None, truncate=False):
    """Cases for gate texts (or (a, b) text pairs). A text longer than
    `max_len` (the largest bucket, or an ONNX build's max_seq_len) is an
    error, since it would silently gate nothing, unless `truncate`: then it
    is cut as the server cuts it, keeping the first max_len - 1 tokens and
    the final one (the EOS that last-token pooling reads, docs/DECISIONS.md
    D20), and its reference is computed on that."""
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
            raise SystemExit(f"gate input {i} is {len(ids)} tokens, longer than the limit {max_len}")
        ref = head.reference(backbone.reference(ids, types))
        cases.append(Case(ids=ids, ref=ref, extra=extra, label=str(i)))
    return Evaluation(cases)


def _apply(backbone, rewrites):
    for rewrite in rewrites:
        rewrite(backbone)


def embedder(*, model_id, src, buckets, backbone, head, tok, texts, calibration=None, gates=None,
             forbid_ops=frozenset({FUSED_ATTENTION}), rewrites=(), strict_max_seq_len=True,
             truncate=False, negative_control=False, timing=False, int8_embedding=False,
             ignore_ane_weight_cap=False, chunks=None, chunk_identity_all=False, shared_weights=True,
             format="coreml", onnx_weights="fp32", keep_packages=None):
    """An embedding job; installs examples/manifests/<model_id>/manifest.toml.
    `rewrites` (functions of the backbone) run after the fp32 references are
    computed from the unmodified checkpoint. For `format="onnx"` they don't
    run: they are the ANE's precision rewrites, exact in fp32, and only add
    operations on the CPU. An ONNX job checks the manifest as the ONNX build
    installs it (no buckets, onnx.toml's max_seq_len) and cuts its gate
    texts at that max_seq_len rather than at the largest bucket."""
    path = _manifest.embedder_path(model_id)
    m = _manifest.load(path)
    head.bind(backbone)
    if format == "onnx":
        m = _manifest.onnx_view(m, _manifest.onnx_sidecar_text(path))
        checked, max_len = [], m["max_seq_len"]
    else:
        checked, max_len = buckets, max(buckets)
    _manifest.check_embedder(m, src=src, buckets=checked, backbone=backbone, head=head,
                             strict_max_seq_len=strict_max_seq_len)
    ports = text_ports(token_type_ids=getattr(backbone, "token_types", None) == "input")
    cases = evaluation(backbone, head, tok, texts, max_len, truncate=truncate)
    if format != "onnx":
        _apply(backbone, rewrites)
    make_wrapper, example = compose(backbone, head, ports)
    return Job(name=model_id, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
               example=example, evaluation=cases,
               gates=gates or EmbeddingGates(pad_id_range=(1000, min(30000, backbone.vocab_size))),
               calibration=calibration, forbid_ops=frozenset(forbid_ops) | backbone.forbid_ops,
               install_files=[(path, "manifest.toml")],
               negative_control=negative_control, timing=timing, int8_embedding=int8_embedding,
               ignore_ane_weight_cap=ignore_ane_weight_cap, chunks=chunks, chunk_identity_all=chunk_identity_all,
               shared_weights=shared_weights, backbone=backbone, head=head, format=format, onnx_weights=onnx_weights,
               keep_packages=keep_packages)


def classifier(*, model_id, src, buckets, backbone, head, tok, texts=None, pairs=None, calibration=None,
               gates=None, gate_overrides=None, expected_problem_type=None, rewrites=(),
               strict_max_seq_len=True, landing_required=True, negative_control=False, timing=False,
               int8_embedding=False, ignore_ane_weight_cap=False, chunks=None, chunk_identity_all=False,
               shared_weights=True, format="coreml", onnx_weights="fp32", keep_packages=None):
    """A classification job; installs examples/classifiers/<model_id>/classifier.toml.
    The gates compare outputs after the manifest's activation; `gate_overrides`
    adjusts ClassifierGates' thresholds. `rewrites` run after the fp32
    references, as in embedder()."""
    if format != "coreml":
        raise SystemExit("classifiers convert to Core ML only; ONNX export covers embedders")
    path = _manifest.classifier_path(model_id)
    m = _manifest.load(path)
    head.bind(backbone)
    token_type_input = getattr(backbone, "token_types", None) == "input"
    if m.get("classify", {}).get("format") == "nli":
        # an (premise, hypothesis) pair per prediction, gated on the model's
        # own n-way softmax
        _manifest.check_nli(m, src=src, buckets=buckets, backbone=backbone, head=head, tok=tok,
                            token_type_input=token_type_input)
    else:
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
               chunk_identity_all=chunk_identity_all, shared_weights=shared_weights, backbone=backbone,
               head=head, keep_packages=keep_packages)


CAUSAL_LM = {
    "minicpm5-2b": {},
}
"""Per-model departures from causal_lm.Recipe's defaults. MiniCPM5-2B takes
the defaults: R1 and R2 rotations, GPTQ with act-order onto 8-bit
palettes per 16 output channels, 40,960 calibration tokens in 2,048-token
sequences of which 8 are long chat requests, the head as four
convolutions, and no projection left in fp16."""


def causal_lm(*, model_id, src, work, draw=None, stability_draw=None, tokens=None, chat_sequences=None,
              fp16_projections=None, plan=True, screen_only=False, keep_packages=False):
    """A causal-LM job (causal_lm.run); installs examples/chat/<model_id>/chat.toml
    with its [build] table. `stability_draw` -1 skips the second draw."""
    from .causal_lm import Job, Recipe, manifest_path
    if not manifest_path(model_id).exists():
        raise SystemExit(f"no manifest at {manifest_path(model_id)}: name the install dir after the model id")
    recipe = Recipe(**CAUSAL_LM.get(model_id, {}))
    if draw is not None:
        recipe.draw_seed = draw
    if tokens is not None:
        recipe.tokens = tokens
    if chat_sequences is not None:
        recipe.chat_sequences = chat_sequences
    if fp16_projections is not None:
        names = []
        for item in filter(None, fp16_projections.split(",")):
            layer, proj = item.strip().split(".")
            names += [f"{layer}.{p}" for p in ("q", "k", "v", "o", "gate", "up", "down")] if proj == "*" else [
                f"{int(layer)}.{proj}"]
        recipe.fp16_projections = tuple(names)
    if stability_draw is not None:
        recipe.stability_seed = None if stability_draw < 0 else stability_draw
    if recipe.stability_seed == recipe.draw_seed:
        raise SystemExit("the stability draw must differ from the build's draw")
    return Job(model_id=model_id, src=str(Path(src).resolve()), work=str(Path(work).resolve()), recipe=recipe,
               plan=plan, screen_only=screen_only, keep_packages=keep_packages)
