"""BERT-family encoders: BERT, RoBERTa and XLM-R (post-norm, absolute
positions, token types).

What conversion needs (docs/DECISIONS.md D15, D25, D28):
- ATTENTION. "explicit" loads transformers' eager path (matmul -> softmax ->
  matmul) and replaces its extended attention mask, which fills with
  finfo(float32).min (-inf in fp16, NaN through softmax), by a finite
  additive -30000 of the same geometry. "fused" loads the sdpa path, which
  coremltools turns into Core ML's fused attention op: the ANE ignores its
  mask here, and bge-small's artifact is correct only through a Core ML
  fallback (D25). It exists only to keep bge-small's existing artifact
  byte-identical; new models use "explicit", and moving bge-small over is a
  separate, measured change.
- STATIC SHAPES. position_ids is a buffer, never computed from a traced size.
  RoBERTa and XLM-R number positions from padding_idx + 1, so their buffer
  is offset by 2 (position_offset="auto" keys on model_type, never on the
  tokenizer or pad id: paraphrase-multilingual-MiniLM is model_type "bert"
  with an XLM-R tokenizer and takes no offset).
- TOKEN TYPES, one of:
  - "zeros": a zero buffer, as a single-text classifier sees them;
  - "input": an int32 token_type_ids port, for pair models (rerankers);
  - "none": pass nothing, so the embeddings use their own registered zeros.
    bge-small's artifact was built this way.
- ACTIVATION. The native gelu op is coarse on the ANE (D17). twice_gelu()
  is an opt-in rewrite: 2 * erf-GELU built from erf, mul and add, with the
  0.5 folded into each layer's output.dense weight. Measure it per model
  (on laya it cut ANE flips; Core ML's CPU erf is coarser than its gelu).
- RANGE. BERT is post-norm: there is no residual stream to rescale, so the
  ANE linear's 2^15 limit (D25 amendment) is only checked, over the gate
  texts, and the conversion fails when it doesn't hold.
- ANE REWRITES, opt-in, measured on nli-MiniLM2-L6-H768 (an NLI
  cross-encoder) on an M1 Max: on the ANE it failed bucket invariance
  (Delta-p 5.9e-3 between buckets) and its worst Delta-p was 0.019.
  matmul_softmax_rewrite() writes every layer out with the softmax
  denominator from the value matmul (techniques/attention.py), which made
  it exactly bucket-invariant (C, 0.013); twice_gelu() then took it to
  5.4e-3; linear_floor_rewrite() to 4.9e-3 (B), each at the same latency.
  On roberta-large-mnli the three give an exactly bucket-invariant C
  (p99 2.01x its fp16 ceiling).
"""

import torch

from ..techniques import activations, attention, masks, precision, saturation
from ..calibrate import all_linears, linear_maxima
from . import Backbone

OFFSET_MODEL_TYPES = ("roberta", "xlm-roberta", "camembert")


def _finite_extended_mask(self, attention_mask, input_shape=None, dtype=None, **_):
    # the geometry of ModuleUtilsMixin.get_extended_attention_mask for an
    # encoder, (bsz, 1, 1, seq), with a finite additive constant. The call
    # count lets load() prove the patch is on the forward path.
    self._finite_mask_calls = getattr(self, "_finite_mask_calls", 0) + 1
    return masks.key_padding(attention_mask)


def _verify_finite_mask(model, special_ids):
    """One padded forward must go through the patched mask. transformers 5
    builds BERT's mask in masking_utils instead, which would bypass the patch
    and leave finfo.min (-inf in fp16) in the graph: refuse rather than
    convert with it."""
    base = model.base_model
    before = getattr(base, "_finite_mask_calls", 0)
    ids = torch.tensor([list(special_ids) + [0, 0]])
    am = torch.tensor([[1] * len(special_ids) + [0, 0]])
    with torch.no_grad():
        base(input_ids=ids, attention_mask=am)
    if getattr(base, "_finite_mask_calls", 0) == before:
        import transformers
        raise SystemExit(f"transformers {transformers.__version__} doesn't build BERT's attention mask through "
                         "get_extended_attention_mask, so the finite-mask patch would be bypassed. The BERT "
                         "backbone is validated with transformers 4.57.")


class BertBackbone(Backbone):
    softmax = "native"     # "matmul" after matmul_softmax_rewrite()

    def __init__(self, *, token_types, position_offset, **kw):
        super().__init__(**kw)
        self.token_types = token_types
        self.position_offset = position_offset

    def buffers(self, seq):
        position_ids = torch.arange(seq, dtype=torch.long)
        if self.position_offset:
            position_ids = position_ids + self.position_offset
        b = {"position_ids": position_ids.unsqueeze(0)}
        if self.token_types == "zeros":
            b["token_type_ids"] = torch.zeros((1, seq), dtype=torch.long)
        return b

    def call(self, w, x):
        if self.softmax == "matmul":
            return self._call_written_out(w, x)
        kw = {"input_ids": x["input_ids"].long(), "attention_mask": x["attention_mask"].long()}
        if self.token_types == "zeros":
            kw["token_type_ids"] = w.token_type_ids
        elif self.token_types == "input":
            kw["token_type_ids"] = x["token_type_ids"].long()
        kw["position_ids"] = w.position_ids
        return getattr(w, self.attr)(**kw)

    def _call_written_out(self, w, x):
        """The encoder layer by layer, attention by matmul_softmax, then the
        checkpoint's own head: BERT's pooler and classifier, or RoBERTa's
        classification head (which reads the first token itself)."""
        from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
        task = getattr(w, self.attr)
        base, seq = task.base_model, w.seq
        types = {"zeros": getattr(w, "token_type_ids", None), "none": None}.get(self.token_types)
        if self.token_types == "input":
            types = x["token_type_ids"].long()
        h = base.embeddings(input_ids=x["input_ids"].long(), token_type_ids=types, position_ids=w.position_ids)
        add = masks.key_padding(x["attention_mask"].long())
        for layer in base.encoder.layer:
            att = layer.attention.self
            heads, hd = att.num_attention_heads, att.attention_head_size

            def split(t):
                return t.reshape(1, seq, heads, hd).transpose(1, 2)
            ctx = attention.matmul_softmax(split(att.query(h)), split(att.key(h)), split(att.value(h)), add,
                                           hd ** -0.5, seq)
            a = layer.attention.output(ctx.transpose(1, 2).reshape(1, seq, heads * hd), h)
            h = layer.output(layer.intermediate(a), a)
        if task is base:
            return BaseModelOutput(last_hidden_state=h)
        pooler = getattr(base, "pooler", None)
        return SequenceClassifierOutput(logits=task.classifier(pooler(h) if pooler is not None else h))

    def reference(self, ids, token_type_ids=None):
        t = torch.tensor([ids])
        kw = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if token_type_ids is not None:
            kw["token_type_ids"] = torch.tensor([token_type_ids])
        with torch.no_grad():
            return self.model(**kw)

    def check_linear_range(self, texts, tok):
        """The ANE linear's 2^15 rule, checked over `texts`, each a string or
        an (a, b) pair (a gate, so the evaluation texts will do: it decides
        nothing about the artifact). Returns (name, value, headroom factor)
        of the largest output."""
        from ..tokenizer import encode, encode_pair

        def run():
            for item in texts:
                if isinstance(item, tuple):
                    self.reference(*encode_pair(tok, *item))
                else:
                    self.reference(encode(tok, item))

        return saturation.check(linear_maxima(run, all_linears(self.model)))


def load(src, tok, *, task=None, attention="explicit", token_types="zeros", position_offset="auto"):
    """Load a BERT-family checkpoint for conversion.

    task: None for the bare encoder (pooling heads), "sequence-classification"
    for the checkpoint's own classification head (classifiers, rerankers).
    """
    from transformers import AutoModel, AutoModelForSequenceClassification
    if attention not in ("explicit", "fused"):
        raise ValueError(f"attention must be explicit or fused, not {attention!r}")
    if token_types not in ("zeros", "input", "none"):
        raise ValueError(f"token_types must be zeros, input or none, not {token_types!r}")
    cls = {None: AutoModel, "sequence-classification": AutoModelForSequenceClassification}[task]
    impl = "eager" if attention == "explicit" else "sdpa"
    model = cls.from_pretrained(src, dtype=torch.float32, attn_implementation=impl).eval()
    config = model.config
    if position_offset == "auto":
        position_offset = config.pad_token_id + 1 if config.model_type in OFFSET_MODEL_TYPES else 0
    from ..tokenizer import special_ids
    specials = tuple(special_ids(tok))
    if attention == "explicit":
        for m in {id(model): model, id(model.base_model): model.base_model}.values():
            m.get_extended_attention_mask = _finite_extended_mask.__get__(m)
        _verify_finite_mask(model, specials)
    return BertBackbone(family="bert", model=model, config=config, special_ids=specials,
                        token_types=token_types, position_offset=position_offset)


def matmul_softmax_rewrite(backbone):
    """A rewrite: the encoder written out layer by layer, each softmax's
    denominator computed by the value matmul (techniques/attention.py), which
    makes the ANE's output bucket-invariant. Exact in fp32; changes the
    artifact. The key-padding mask never masks a whole row, so no row needs
    self-attention."""
    if backbone.config.model_type not in ("bert", *OFFSET_MODEL_TYPES):
        raise SystemExit(f"matmul_softmax_rewrite: model_type {backbone.config.model_type!r} isn't BERT-family")
    backbone.softmax = "matmul"


class _Scaled(torch.nn.Module):
    """inner(x) * s, a power-of-two scale on an activation's output."""

    def __init__(self, inner, s):
        super().__init__()
        self.inner = inner
        self.s = float(s)

    def forward(self, x):
        return self.inner(x) * self.s


def linear_floor_rewrite(pairs, tok, report=print):
    """A rewrite: the ANE linear's small-input floor (techniques/precision.py)
    for every layer's two output linears. output.dense's input is the
    activation's output, scaled by a calibrated power of two; the
    attention's output.dense input is the context, scaled through the value
    projection; each linear is then descaled, its bias scaled up so the
    descale leaves it unchanged. Calibrated on `pairs` (a converter's gate
    pairs, never the graded corpus). Exact in fp32."""
    from ..calibrate import collect
    from ..tokenizer import encode_pair

    def rewrite(backbone):
        base = backbone.model.base_model
        sites = {}
        for i, layer in enumerate(base.encoder.layer):
            for name, lin in (("o", layer.output.dense), ("a", layer.attention.output.dense)):
                sites[f"{i}.{name}_in"] = (lin, "in")
                sites[f"{i}.{name}_out"] = (lin, "out")
        stats = collect(lambda: [backbone.reference(*encode_pair(tok, *p)) for p in pairs], sites)
        scales = []
        for i, layer in enumerate(base.encoder.layer):
            so = precision.input_scale(stats[f"{i}.o_in"], stats[f"{i}.o_out"])
            sa = precision.input_scale(stats[f"{i}.a_in"], stats[f"{i}.a_out"])
            layer.intermediate.intermediate_act_fn = _Scaled(layer.intermediate.intermediate_act_fn, so)
            v = layer.attention.self.value
            with torch.no_grad():
                layer.output.dense.bias.mul_(so)
                v.weight.mul_(sa)
                v.bias.mul_(sa)
                layer.attention.output.dense.bias.mul_(sa)
            layer.output.dense = precision.Descale(layer.output.dense, 1.0 / so)
            layer.attention.output.dense = precision.Descale(layer.attention.output.dense, 1.0 / sa)
            scales.append(f"{so:g}/{sa:g}")
        report(f"linear-floor scales (mlp/attention) per layer: {' '.join(scales)}")
    return rewrite


def twice_gelu(backbone):
    """Opt-in rewrite: every layer's erf GELU becomes techniques.activations.
    TwiceGelu (2 * gelu), and output.dense takes the 0.5. Exact in fp32; the
    native gelu op is then forbidden in the converted graph."""
    if getattr(backbone.config, "hidden_act", None) != "gelu":
        raise SystemExit(f"twice_gelu needs hidden_act 'gelu' (erf), not {backbone.config.hidden_act!r}")
    for layer in backbone.model.base_model.encoder.layer:
        layer.intermediate.intermediate_act_fn = activations.TwiceGelu()
        with torch.no_grad():
            layer.output.dense.weight.mul_(1.0 / activations.TwiceGelu.GAIN)
    backbone.forbid_ops.add("gelu")
