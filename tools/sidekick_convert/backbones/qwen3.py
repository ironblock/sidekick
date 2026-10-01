"""Qwen3 causal decoders used as embedders (F2LLM, Qwen3-Embedding):
pre-norm, RoPE, grouped-query attention with QK-norm, SwiGLU MLP, last-token
pooling (docs/DECISIONS.md D20 and its amendment).

What conversion needs:
- ATTENTION. transformers' sdpa path calls F.scaled_dot_product_attention
  with an explicit `scale=`, so coremltools lowers it to explicit
  matmul -> softmax -> matmul, never the fused op; the forbidden-op gate
  proves it on every build.
- MASK. create_causal_mask fills with finfo.min (-inf in fp16). The patch
  builds the same causal AND key-padding mask with -30000. Right padding is
  correct for a causal model: the last real token never sees the pads.
- TRACEABLE rotate_half and repeat_kv, installed where the sdpa path calls
  them too (transformers' sdpa integration keeps its own repeat_kv).
- NO RANGE REWRITE. QK-norm keeps activations small (max ~420 on
  F2LLM-v2-160M).
- PRECISION REWRITE, precision_rewrite() (D20 amendment):
  - the coarse native silu becomes TanhSilu (2 * silu), with up_proj taking
    the 1/2, so the MLP adds no op;
  - attention's inputs are small in the early layers (q/k/v at rms
    0.06-0.18, o_proj at 0.02-0.35), under the ANE linear's small-input
    floor: power-of-two scales on the input norm's weight (q/k RMSNorm eps
    x s^2, since q and k are re-normalized per head) and on v_proj bring both
    to rms ~1, and Descale undoes o_proj's before the residual add. The
    MLP's inputs needed no rescale.
"""

import torch

from ..techniques import activations, precision, traceable
from ..calibrate import collect
from . import Backbone

MASK_ADD = -30000.0
QK_MAX = 150.0      # cap on |q|, |k| entering their RMSNorms, which square them


def _fp16_safe_causal_mask(config=None, input_embeds=None, attention_mask=None, **kw):
    # lower-triangular (causal) AND key-not-pad, additive with MASK_ADD.
    # Batch-1 static seq; returns (bsz, 1, seq, seq).
    embeds = input_embeds
    seq = embeds.shape[1]
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.float32))  # 1 where k<=q
    if attention_mask is not None:
        keep = attention_mask.to(torch.float32)                      # (bsz, seq) 1=real
        allowed = causal.unsqueeze(0) * keep[:, None, :]             # (bsz, q, k)
    else:
        allowed = causal.unsqueeze(0)
    add = (1.0 - allowed).unsqueeze(1) * MASK_ADD                    # (bsz, 1, q, k)
    return add.to(embeds.dtype)


def install_patches():
    import transformers.integrations.sdpa_attention as _sdpa_attention
    import transformers.models.qwen3.modeling_qwen3 as _qwen3
    traceable.install(_qwen3)
    traceable.install(_sdpa_attention, names=("repeat_kv",))
    _qwen3.create_causal_mask = _fp16_safe_causal_mask


class Qwen3Backbone(Backbone):
    def call(self, w, x):
        return getattr(w, self.attr)(
            input_ids=x["input_ids"].long(),
            attention_mask=x["attention_mask"].long(),
            position_ids=w.position_ids,
        )

    def reference(self, ids, token_type_ids=None):
        t = torch.tensor([ids])
        with torch.no_grad():
            return self.model(input_ids=t, attention_mask=torch.ones_like(t))


def load(src, tok):
    """Load a Qwen3 decoder for conversion (transformers' sdpa path, which
    emits explicit attention because it passes a scale)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa").eval()
    model.config.use_cache = False
    install_patches()
    from ..tokenizer import special_ids
    return Qwen3Backbone(family="qwen3", model=model, config=model.config, special_ids=tuple(special_ids(tok)),
                         forbid_ops={"silu", "gelu"})


def precision_rewrite(calibration, tok, texts_for=lambda cal: cal.texts, report=print):
    """A rewrite: calibrate attention's input statistics on `calibration`
    (texts_for(calibration) yields the full texts, prompts included), then
    swap in TanhSilu and rescale attention's inputs. Every factor is a power
    of two, so the fp32 graph is unchanged (the fp32 gate proves it)."""
    from ..core import Calibration
    from ..tokenizer import encode
    if not isinstance(calibration, Calibration):
        raise TypeError("precision_rewrite calibrates on a core.Calibration, never on evaluation cases")

    def rewrite(backbone):
        model = backbone.model
        sites = {}
        for i, layer in enumerate(model.layers):
            a = layer.self_attn
            sites.update({f"{i}.qkv_in": (a.q_proj, "in"), f"{i}.q": (a.q_proj, "out"), f"{i}.k": (a.k_proj, "out"),
                          f"{i}.o_in": (a.o_proj, "in"), f"{i}.o_out": (a.o_proj, "out")})
        def run():
            for text in texts_for(calibration):
                backbone.reference(encode(tok, text))

        stats = collect(run, sites)
        scales = []
        for i, layer in enumerate(model.layers):
            with torch.no_grad():
                layer.mlp.up_proj.weight.mul_(1.0 / activations.TanhSilu.GAIN)
            layer.mlp.act_fn = activations.TanhSilu()
            a = layer.self_attn
            s_in = precision.input_scale(stats[f"{i}.qkv_in"])
            qk = max(stats[f"{i}.q"].max, stats[f"{i}.k"].max)
            while s_in > 1.0 and qk * s_in > QK_MAX:
                s_in /= 2.0
            s_o = precision.input_scale(stats[f"{i}.o_in"], stats[f"{i}.o_out"])
            with torch.no_grad():
                layer.input_layernorm.weight.mul_(s_in)
                a.v_proj.weight.mul_(s_o / s_in)
            a.q_norm.variance_epsilon *= s_in * s_in
            a.k_norm.variance_epsilon *= s_in * s_in
            a.o_proj = precision.Descale(a.o_proj, 1.0 / s_o)
            scales.append(f"L{i} {s_in:g}/{s_o:g}")
        report("attention input scales (q/k/v in, o_proj in): " + ", ".join(scales))
    return rewrite
