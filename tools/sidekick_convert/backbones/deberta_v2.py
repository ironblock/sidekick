"""DeBERTa-v2/v3 encoders: disentangled relative-position attention, no
absolute positions (DeBERTa-v3, GLiNER2's encoder).

What conversion needs (measured with tools/probe_deberta.py on macOS 27):
- ATTENTION, written out here rather than patched. transformers' attention
  doesn't trace under coremltools 9 (its TorchScript helpers and arithmetic
  on tensor sizes hit the `int` conversion bug). Made traceable, its two
  relative-position gathers per layer (content-to-position, c2p, and
  position-to-content, p2c) become `gather_along_axis` on the CPU, taking
  their matmuls along: a hand-off per gather and a failing plan. call()
  computes them with techniques.relative_shift instead: per bucket and
  layer, the position projections expanded over the 2L - 1 distances are
  constant buffers (`buffers(seq)`), and each term is a matmul plus a
  reshape/slice skew. Then techniques.attention.explicit().
  The buffers cost 2 terms x layers x (2L - 1) x hidden in fp16: for
  DeBERTa-v3-large (24 layers, hidden 1024) about 25 MB at bucket 128,
  50 MB at 256 and 100 MB at 512, about 176 MB across the three, on top
  of the ~870 MB of weights each bucket already carries.
- MASKS. transformers fills masked scores with finfo.min (-inf in fp16) and
  masks pairwise, so a pad query's whole row is masked. Here the pairwise
  mask is finite (masks.MASK_ADD) and self-attending, which is exact for
  real tokens. The embeddings' own pad zeroing (embeddings * mask) is kept.
- STATIC SHAPES. Every size is the bucket's Python int; no tensor size is
  read in the traced path.
- ACTIVATION. hidden_act is the erf gelu. twice_gelu() is the opt-in
  rewrite, as for BERT; on GLiNER2.5-Decide it didn't improve the ANE path,
  so measure it per model.
- RANGE. DeBERTa is post-norm, like BERT: the ANE linear's 2^15 limit is
  only checked (GLiNER2.5-Decide peaks at about 1,300).

Checkpoints: a DeBERTa-v2/v3 snapshot, or a GLiNER2 checkpoint, whose
encoder config is in encoder_config/ and whose weights are prefixed
"encoder." in model.safetensors. gliner2_classifier() rebuilds a GLiNER2
checkpoint's per-token classifier for heads.per_token.

Measured on an M1 Max, macOS 27.0:
- deberta-v3-small through this backbone, CLS pooled, at 128 and 512: 242 of
  251 operations on the ANE, with no gather and no fused attention. Cosine
  against transformers is 0.9999998 on the ANE and 0.9999952 on the CPU.
- GLiNER2.5-Decide with the same rewrite (tools/probe_deberta.py): its
  classifier logits agree with gliner2's own fp32 scoring on every argmax at
  128/256/512, max |dp| 0.007-0.009 on the ANE.
"""

import math
from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position

from ..calibrate import all_linears, linear_maxima
from ..techniques import activations, attention, masks, relative_shift, saturation
from . import Backbone


def relative_index(distance, span, max_position, buckets):
    """transformers' row of the relative-position table for distance q - k:
    log-bucketed when the model has position buckets, then shifted by the
    span and clamped to the table (DisentangledSelfAttention's c2p_pos)."""
    d = torch.as_tensor(distance, dtype=torch.long)
    if buckets > 0 and max_position > 0:
        d = make_log_bucket_position(d, buckets, max_position).long()
    return torch.clamp(d + span, 0, 2 * span - 1)


class DebertaV2Backbone(Backbone):
    def __init__(self, **kw):
        super().__init__(**kw)
        att = self.model.encoder.layer[0].attention.self
        self.terms = tuple(t for t in ("c2p", "p2c") if t in att.pos_att_type)
        self.heads = att.num_attention_heads
        self.head_size = att.attention_head_size
        self.scale = 1.0 / math.sqrt(self.head_size * (1 + len(self.terms)))

    def _relative_rows(self, seq):
        enc = self.model.encoder
        att = enc.layer[0].attention.self

        def index(distance):
            return relative_index(distance, att.pos_ebd_size, enc.max_relative_positions, enc.position_buckets)

        return {"c2p": relative_shift.query_side_index(index, seq), "p2c": relative_shift.key_side_index(index, seq)}

    def buffers(self, seq):
        """position_ids (unused by the v3 embeddings, kept for the base
        contract), token_type_ids when the model has token types, and per
        layer the expanded relative-position projections: c2p_<i> =
        key_proj(rel)[c2p rows], p2c_<i> = query_proj(rel)[p2c rows], each
        (2L - 1, hidden). Weight-only, so they are constants of the bucket."""
        b = super().buffers(seq)
        if self.config.type_vocab_size > 0:
            b["token_type_ids"] = torch.zeros((1, seq), dtype=torch.long)
        rows = self._relative_rows(seq)
        enc = self.model.encoder
        with torch.no_grad():
            rel = enc.get_rel_embedding()
            for i, layer in enumerate(enc.layer):
                att = layer.attention.self
                for term in self.terms:
                    proj = {"c2p": att.key_proj if att.share_att_key else getattr(att, "pos_key_proj", None),
                            "p2c": att.query_proj if att.share_att_key else getattr(att, "pos_query_proj", None)}[term]
                    b[f"{term}_{i}"] = proj(rel[rows[term]]).detach().clone()
        return b

    def _split(self, t, seq):
        """(1, n, hidden) -> (1, H, n, D), with n a Python int."""
        return t.reshape(1, seq, self.heads, self.head_size).transpose(1, 2)

    def _attention(self, w, i, layer, h, add, seq):
        att = layer.attention.self
        q = self._split(att.query_proj(h), seq)
        k = self._split(att.key_proj(h), seq)
        v = self._split(att.value_proj(h), seq)
        bias = add
        n = 2 * seq - 1
        if "c2p" in self.terms:
            bias = bias + relative_shift.query_side(q, self._split(getattr(w, f"c2p_{i}").unsqueeze(0), n), seq) * self.scale
        if "p2c" in self.terms:
            bias = bias + relative_shift.key_side(k, self._split(getattr(w, f"p2c_{i}").unsqueeze(0), n), seq) * self.scale
        ctx = attention.explicit(q, k, v, bias, self.scale)
        ctx = ctx.transpose(1, 2).reshape(1, seq, self.heads * self.head_size)
        return layer.attention.output(ctx, h)

    def call(self, w, x):
        m = getattr(w, self.attr)
        seq = w.seq
        mask = x["attention_mask"].long()
        kw = {"input_ids": x["input_ids"].long(), "position_ids": w.position_ids, "mask": mask}
        if self.config.type_vocab_size > 0:
            kw["token_type_ids"] = w.token_type_ids
        h = m.embeddings(**kw)
        real = mask.to(torch.float32)
        pairwise = real[:, None, :, None] * real[:, None, None, :]
        add = masks.self_attending((1.0 - pairwise) * masks.MASK_ADD, seq)
        for i, layer in enumerate(m.encoder.layer):
            a = self._attention(w, i, layer, h, add, seq)
            h = layer.output(layer.intermediate(a), a)
        return BaseModelOutput(last_hidden_state=h)

    def reference(self, ids, token_type_ids=None):
        """transformers' own forward, unpadded."""
        t = torch.tensor([ids])
        kw = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if token_type_ids is not None:
            kw["token_type_ids"] = torch.tensor([token_type_ids])
        with torch.no_grad():
            return self.model(**kw)

    def check_linear_range(self, texts, tok):
        """The ANE linear's 2^15 rule over `texts` (see BertBackbone)."""
        from ..tokenizer import encode

        def run():
            for text in texts:
                self.reference(encode(tok, text))

        return saturation.check(linear_maxima(run, all_linears(self.model)))


def _check_supported(config, model):
    if not getattr(config, "relative_attention", False):
        raise SystemExit("the DeBERTa-v2 backbone needs relative_attention")
    terms = set(config.pos_att_type or [])
    if not terms or not terms <= {"c2p", "p2c"}:
        raise SystemExit(f"pos_att_type must be c2p and/or p2c, not {config.pos_att_type!r}")
    if getattr(config, "conv_kernel_size", 0) > 0:
        raise SystemExit("DeBERTa-v2's convolution layer (conv_kernel_size > 0, e.g. deberta-v2-xlarge) isn't handled")
    if getattr(model, "z_steps", 0) > 1:
        raise SystemExit("DeBERTa's enhanced mask decoder steps (z_steps) aren't handled")


def _gliner2_state(src, prefix):
    from safetensors.torch import load_file
    return {k[len(prefix):]: v.clone() for k, v in load_file(Path(src) / "model.safetensors").items()
            if k.startswith(prefix)}


def load(src, tok, *, task=None):
    """Load a DeBERTa-v2/v3 snapshot, or a GLiNER2 checkpoint's encoder, in fp32.
    Only the bare encoder (task=None) is handled: pooling and per-token heads."""
    from transformers import AutoConfig, AutoModel
    if task is not None:
        raise SystemExit(f"the DeBERTa-v2 backbone loads the bare encoder only, not task {task!r}")
    src = Path(src)
    if (src / "encoder_config").is_dir():
        config = AutoConfig.from_pretrained(src / "encoder_config")
        model = AutoModel.from_config(config)
        missing, unexpected = model.load_state_dict(_gliner2_state(src, "encoder."), strict=False)
        if unexpected or any("position_ids" not in name for name in missing):
            raise SystemExit(f"{src}: encoder weights don't match encoder_config: "
                             f"missing {missing}, unexpected {unexpected}")
    else:
        model = AutoModel.from_pretrained(src, dtype=torch.float32)
        config = model.config
    model = model.float().eval()
    _check_supported(config, model)
    from ..tokenizer import special_ids
    # The relative shift replaces transformers' gathers; one in the graph
    # means a regression that would put the attention on the CPU.
    return DebertaV2Backbone(family="deberta-v2", model=model, config=config,
                             special_ids=tuple(special_ids(tok)), forbid_ops={"gather_along_axis"})


def gliner2_classifier(src):
    """A GLiNER2 checkpoint's per-token classifier, Linear -> ReLU -> Linear
    (gliner2's create_mlp with dropout 0), with its weights, in fp32."""
    sd = _gliner2_state(src, "classifier.")
    hidden, inner = sd["0.weight"].shape[1], sd["0.weight"].shape[0]
    mlp = torch.nn.Sequential(torch.nn.Linear(hidden, inner), torch.nn.ReLU(), torch.nn.Linear(inner, 1))
    mlp.load_state_dict(sd)
    return mlp.float().eval()


def twice_gelu(backbone):
    """Opt-in rewrite, as bert.twice_gelu: each layer's erf GELU becomes
    activations.TwiceGelu and output.dense takes the 1/GAIN; the native gelu
    op is then forbidden. Exact in fp32."""
    if getattr(backbone.config, "hidden_act", None) != "gelu":
        raise SystemExit(f"twice_gelu needs hidden_act 'gelu' (erf), not {backbone.config.hidden_act!r}")
    for layer in backbone.model.encoder.layer:
        layer.intermediate.intermediate_act_fn = activations.TwiceGelu()
        with torch.no_grad():
            layer.output.dense.weight.mul_(1.0 / activations.TwiceGelu.GAIN)
    backbone.forbid_ops.add("gelu")
