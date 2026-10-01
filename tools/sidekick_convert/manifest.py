"""The committed manifest a converter installs, validated against the checkpoint.

Manifests are reviewed decisions (buckets, max_batch, prefixes,
problem_type) and parity references check them, so converters copy the
committed file (examples/manifests/<id>/manifest.toml for embedders,
examples/classifiers/<id>/classifier.toml for classifiers) and never
generate one. Before converting, the manifest is checked against the
checkpoint and the recipe; any mismatch stops the conversion. A check runs
only where its field exists.
"""

import tomllib
from pathlib import Path

from .core import REPO
from . import tokenizer as _tok


def embedder_path(model_id):
    return REPO / "examples" / "manifests" / model_id / "manifest.toml"


def classifier_path(model_id):
    return REPO / "examples" / "classifiers" / model_id / "classifier.toml"


def load(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no manifest at {path}: name the install dir after the model id")
    return tomllib.loads(path.read_text())


# A manifest's `compute_units` ("cpu_and_ne" when absent; docs/design/
# classify.md, "Compute units") and the Core ML compute units it names.
COMPUTE_UNITS = {"cpu_and_ne": "CPU_AND_NE", "cpu_and_gpu": "CPU_AND_GPU", "cpu_only": "CPU_ONLY", "all": "ALL"}


def served_path(m):
    """The Core ML compute units sidekick serves a manifest's model on: what
    a converter gates."""
    value = m.get("compute_units", "cpu_and_ne")
    if value not in COMPUTE_UNITS:
        raise SystemExit(f"manifest compute_units {value!r} is not one of {', '.join(COMPUTE_UNITS)}")
    return COMPUTE_UNITS[value]


def _check(errors, ok, message):
    if not ok:
        errors.append(message)


def _common(m, errors, *, src, buckets, backbone, strict_max_seq_len):
    msl = m.get("max_seq_len")
    mb = m.get("buckets")
    if mb is not None and msl is not None:
        _check(errors, mb[-1] == msl, f"buckets[-1] {mb[-1]} != max_seq_len {msl}")
    if mb is not None:
        _check(errors, set(buckets) <= set(mb), f"converting buckets {buckets} not all in the manifest's {mb}")
    st_msl = _tok.st_config(src).get("max_seq_length")
    if msl is not None and st_msl is not None:
        if strict_max_seq_len:
            _check(errors, msl == st_msl, f"max_seq_len {msl} != sentence-transformers max_seq_length {st_msl}")
        else:
            _check(errors, msl <= st_msl, f"max_seq_len {msl} > sentence-transformers max_seq_length {st_msl}")
    mpe = getattr(backbone.config, "max_position_embeddings", None)
    offset = getattr(backbone, "position_offset", 0) or 0
    if msl is not None and mpe is not None:
        _check(errors, msl + offset <= mpe, f"max_seq_len {msl} (+ offset {offset}) exceeds position embeddings {mpe}")
    rev = (m.get("source") or {}).get("revision")
    snap = _tok.snapshot_revision(src)
    if rev is not None and snap is not None:
        _check(errors, rev == snap, f"source.revision {rev} != snapshot revision {snap}")


def check_embedder(m, *, src, buckets, backbone, head, dims=None, strict_max_seq_len=True):
    """Embedder manifest vs the checkpoint: sequence limits, dims, pooling, io."""
    errors = []
    _common(m, errors, src=src, buckets=buckets, backbone=backbone, strict_max_seq_len=strict_max_seq_len)
    want_dims = dims if dims is not None else backbone.hidden_size
    _check(errors, m.get("dims") == want_dims, f"dims {m.get('dims')} != the head's output size {want_dims}")
    st_mode = _tok.st_pooling(src)
    if st_mode is not None and hasattr(head, "st_mode"):
        _check(errors, st_mode == head.st_mode(), f"sentence-transformers pooling {st_mode!r} != the head's "
                                                  f"{head.st_mode()!r}")
    io = m.get("io", {})
    _check(errors, io.get("output") == head.output, f"io.output {io.get('output')!r} != the head's {head.output!r}")
    if errors:
        raise SystemExit("manifest check failed:\n  " + "\n  ".join(errors))


def problem_type(config):
    """transformers' text-classification activation rule (D28): unset
    activates like single_label."""
    return {"multi_label_classification": "multi_label",
            "regression": "regression"}.get(config.problem_type, "single_label")


def vllm_problem_type(config):
    """A reranker's problem_type: the activation vLLM applies to it (get_act_fn
    in vllm/model_executor/layers/pooler/activations.py). An explicit
    config.problem_type wins; otherwise sentence-transformers' CrossEncoder
    activation (activation_fn, or the older sbert_ce_default_activation_function):
    Identity is regression, Sigmoid single_label; with neither, single_label."""
    explicit = getattr(config, "problem_type", None)
    if explicit:
        return {"regression": "regression", "single_label_classification": "single_label",
                "multi_label_classification": "multi_label"}[explicit]
    st = getattr(config, "sentence_transformers", None) or {}
    fn = st.get("activation_fn") or getattr(config, "sbert_ce_default_activation_function", None)
    if fn:
        if fn.endswith("Identity"):
            return "regression"
        if fn.endswith("Sigmoid"):
            return "single_label"
        raise SystemExit(f"activation {fn} has no problem_type equivalent")
    return "single_label"


def check_classifier(m, *, src, buckets, backbone, head, tok, token_type_input, expected_problem_type=None,
                     strict_max_seq_len=True):
    """Classifier manifest vs the checkpoint: labels, problem_type, token types."""
    errors = []
    _common(m, errors, src=src, buckets=buckets, backbone=backbone, strict_max_seq_len=strict_max_seq_len)
    labels = m.get("classify", {}).get("labels")
    ranking = m.get("task") == "text-ranking"
    if ranking:
        # one output, its relevance score, whatever id2label calls it (LABEL_0)
        _check(errors, head.num_labels == 1, f"a reranker needs num_labels 1, not {head.num_labels}")
        _check(errors, labels == ["score"], f"a reranker's labels must be ['score'], not {labels}")
    elif labels is not None:
        _check(errors, labels == head.labels, f"labels {labels} != id2label {head.labels}")
    want = expected_problem_type or problem_type(backbone.config)
    got = m.get("problem_type", "single_label")
    _check(errors, got == want, f"problem_type {got!r} != the checkpoint's {want!r}")
    io = m.get("classify", {}).get("io", {})
    named = "token_type_ids" in io
    if ranking:
        _, types = _tok.encode_pair(tok, "a", "b")
        _check(errors, named == any(types), f"token_type_ids named={named} but pair segment ids are "
                                            f"{'nonzero' if any(types) else 'all zero'}")
    else:
        _check(errors, not named, "token_type_ids is named, but only text-ranking manifests may name it "
                                  "(older daemons would feed a text-classification model no segment ids)")
    _check(errors, named == token_type_input, f"token_type_ids named={named} but the recipe's "
                                              f"token_type_input={token_type_input}")
    _check(errors, io.get("output") == head.output, f"io.output {io.get('output')!r} != the head's {head.output!r}")
    if errors:
        raise SystemExit("manifest check failed:\n  " + "\n  ".join(errors))


def check_gliner2(m, *, src, buckets, backbone, head):
    """A gliner2-format manifest (docs/design/classify.md) vs the checkpoint
    and the per-token head: zero-shot, the format's section and io, and the
    common sequence-limit and revision checks."""
    errors = []
    _common(m, errors, src=src, buckets=buckets, backbone=backbone, strict_max_seq_len=True)
    c = m.get("classify", {})
    _check(errors, m.get("task") == "zero-shot-classification", f"task {m.get('task')!r} != 'zero-shot-classification'")
    _check(errors, c.get("format") == "gliner2", f"classify.format {c.get('format')!r} != 'gliner2'")
    _check(errors, not c.get("labels"), "a zero-shot model has no fixed labels")
    _check(errors, (c.get("max_labels") or 0) >= 2, "classify.max_labels must be >= 2")
    _check(errors, bool((c.get("gliner2") or {}).get("default_instructions", "").strip()),
           "[classify.gliner2] needs default_instructions")
    _check(errors, m.get("problem_type", "single_label") == "single_label",
           "a gliner2 model's problem_type is single_label (requests opt into multi_label)")
    io = c.get("io", {})
    _check(errors, set(io) == {"input_ids", "attention_mask", "output"},
           f"[classify.io] must name input_ids, attention_mask and output only, not {sorted(io)}")
    _check(errors, io.get("output") == head.output, f"io.output {io.get('output')!r} != the head's {head.output!r}")
    if errors:
        raise SystemExit("manifest check failed:\n  " + "\n  ".join(errors))


def check_agentjev(m, *, src, buckets, backbone, head, temperatures=None):
    """An agentjev-format manifest (docs/design/classify.md) vs the checkpoint
    and the candidate head: zero-shot, the tree's io names, max_labels equal
    to the head's slots, and the common sequence-limit and revision checks.
    With `temperatures` (the checkpoint's temperatures.json), every
    calibration key must carry its question type's temperature."""
    errors = []
    _common(m, errors, src=src, buckets=buckets, backbone=backbone, strict_max_seq_len=True)
    c = m.get("classify", {})
    _check(errors, m.get("task") == "zero-shot-classification", f"task {m.get('task')!r} != 'zero-shot-classification'")
    _check(errors, c.get("format") == "agentjev", f"classify.format {c.get('format')!r} != 'agentjev'")
    _check(errors, not c.get("labels"), "a zero-shot model has no fixed labels")
    _check(errors, c.get("max_labels") == head.kmax, f"classify.max_labels {c.get('max_labels')} != the head's {head.kmax} slots")
    _check(errors, m.get("problem_type", "single_label") == "single_label", "an agentjev model's problem_type is single_label")
    io = c.get("io", {})
    _check(errors, set(io) == {"input_ids", "attention_mask", "seg", "position_ids", "marker_pos", "output"},
           f"[classify.io] must name input_ids, attention_mask, seg, position_ids, marker_pos and output, not {sorted(io)}")
    ports = {p.name for p in head.ports()}
    for key in ("input_ids", "attention_mask", "seg", "position_ids"):
        _check(errors, io.get(key) == key, f"io.{key} {io.get(key)!r} != the graph's {key!r}")
    _check(errors, io.get("marker_pos") in ports, f"io.marker_pos {io.get('marker_pos')!r} isn't one of the graph's inputs")
    _check(errors, io.get("output") == head.output, f"io.output {io.get('output')!r} != the head's {head.output!r}")
    if temperatures is not None:
        types = {"noul": "boolean", "choice": "choice", "score": "score"}
        for key, t in c.get("calibration", {}).items():
            want = temperatures.get(types.get(key.split(":")[0], ""), {}).get("temperature")
            _check(errors, want is not None and abs(t - want) < 1e-9,
                   f"calibration {key!r} = {t} isn't the checkpoint's {want}")
        reached = {"noul:2", "choice:2", "choice:3-5", "choice:6-10", "choice:11+", "score:2", "score:3-5", "score:6-10"}
        _check(errors, reached <= set(c.get("calibration", {})),
               f"calibration lacks {sorted(reached - set(c.get('calibration', {})))}: AgentJev's temperatures apply at every label count")
    if errors:
        raise SystemExit("manifest check failed:\n  " + "\n  ".join(errors))
