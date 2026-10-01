"""Gemma3 text encoders run bidirectionally (EmbeddingGemma): RMSNorm with
pre- and post-branch norms, RoPE, grouped-query attention with QK-norm,
alternating full and sliding-window attention, GeGLU-tanh MLP
(docs/DECISIONS.md D17 and its amendment).

What conversion needs:
- ATTENTION. transformers' sdpa path passes an explicit scale, so attention
  converts to explicit ops, never the fused one. The masks are built by the
  head and passed as transformers' prepared-mask dict (heads/gemma_st.py).
- TRACEABLE rotate_half and repeat_kv (also where the sdpa path calls them).
- fp16 RANGE REWRITE (fp16_range_rewrite). The residual stream grows to
  ~1.5e5, past fp16's 65504, and RMSNorm squares its input. The embedding and
  each layer's two residual-branch outputs take 1/K (K a power of two,
  calibrated); every RMSNorm gets a calibrated power-of-two input pre-scale
  with eps compensated as eps * s^2, exact because RMSNorm is
  scale-invariant; eps is floored at 1e-4 (1e-6 is fp16-subnormal and
  flushes to zero on the ANE).
- MLP PRECISION REWRITE, part of the same rewrite: the down projection's
  input (rms 0.004-0.03 in late layers) sits under the ANE linear's
  small-input floor, so a power-of-two scale folded into up_proj brings it
  to rms ~1, divided out in the scale-invariant post-feedforward norm; and
  the coarse native gelu becomes TanhGelu, exact to gelu_pytorch_tanh.
"""

import numpy as np
import torch

from ..techniques import activations, traceable
from ..techniques.precision import pow2
from . import Backbone

EPS_FLOOR = 1e-4          # smallest fp16-safe rmsnorm eps (1e-6 flushes to 0)
RESIDUAL_MAX_TARGET = 8192.0   # keep |residual| <= this after 1/K scaling
NORM_SQ_MAX = 30000.0     # keep (|x|*s)^2 under this inside every rmsnorm
DOWN_IN_MAX = 2048.0      # cap on |down_proj input| after the MLP rescale
DOWN_OUT_MAX = 16384.0    # cap on |down_proj output| after the MLP rescale


class SafeRMSNorm(torch.nn.Module):
    """Gemma3RMSNorm rewritten for fp16 range, numerically identical in fp32.

    Computes x*rsqrt(mean(x^2)+eps)*(1+w) as y*rsqrt(mean(y^2)+eps*s^2)*(1+w)
    with y = x*mult, where s = mult*in_scale is the total power-of-two scale
    relative to the raw (unscaled-model) activation. `out_scale` folds the
    1/K residual-branch scaling into the norms that feed residual adds.
    """

    def __init__(self, orig, mult, eps_eff, out_scale=1.0):
        super().__init__()
        self.register_buffer("weight", orig.weight.detach().clone())
        self.mult = float(mult)
        self.eps_eff = float(eps_eff)
        self.out_scale = float(out_scale)

    def forward(self, x):
        y = x * self.mult
        n = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps_eff)
        return n * ((1.0 + self.weight) * self.out_scale)


class ScaledEmbedding(torch.nn.Module):
    def __init__(self, inner, scale):
        super().__init__()
        self.inner = inner
        self.scale = float(scale)

    def forward(self, input_ids):
        return self.inner(input_ids) * self.scale


def install_patches():
    import transformers.integrations.sdpa_attention as _sdpa_attention
    import transformers.models.gemma3.modeling_gemma3 as _gemma3
    traceable.install(_gemma3)
    traceable.install(_sdpa_attention, names=("repeat_kv",))


class Gemma3Backbone(Backbone):
    def reference(self, ids, token_type_ids=None):
        t = torch.tensor([ids])
        with torch.no_grad():
            return self.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False)

    @property
    def window(self):
        """config.sliding_window after transformers halves it for bidirectional
        models (config.json's 512 becomes 257): sliding layers attend iff
        |q - k| < window."""
        return int(self.config.sliding_window)


def load(src, tok):
    from transformers import AutoModel
    install_patches()
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa").eval()
    model.config.use_cache = False
    from ..tokenizer import special_ids
    return Gemma3Backbone(family="gemma3", model=model, config=model.config, special_ids=tuple(special_ids(tok)),
                          forbid_ops={"gelu"})


def _calibrate(model, run):
    """fp32 activation stats: (max_abs, min/max mean-square) at every RMSNorm
    input, and (max_abs, sum of squares, count) at every MLP down_proj input."""
    stats = {}

    def pre_hook(name):
        def f(mod, args):
            t = args[0].detach()
            msq = t.pow(2).mean(-1)
            rec = stats.setdefault(name, [0.0, 0.0, float("inf")])
            rec[0] = max(rec[0], float(t.abs().max()))
            rec[1] = max(rec[1], float(msq.max()))
            rec[2] = min(rec[2], float(msq.min()))
        return f

    def rms_hook(name):
        def f(mod, args):
            t = args[0].detach()
            rec = stats.setdefault(name, [0.0, 0.0, 0])
            rec[0] = max(rec[0], float(t.abs().max()))
            rec[1] += float(t.pow(2).sum())
            rec[2] += t.numel()
        return f

    handles = []
    for i, layer in enumerate(model.layers):
        for attr in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            handles.append(getattr(layer, attr).register_forward_pre_hook(pre_hook(f"L{i}.{attr}")))
        handles.append(layer.self_attn.q_norm.register_forward_pre_hook(pre_hook(f"L{i}.q_norm")))
        handles.append(layer.self_attn.k_norm.register_forward_pre_hook(pre_hook(f"L{i}.k_norm")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(rms_hook(f"L{i}.down_proj")))
    handles.append(model.norm.register_forward_pre_hook(pre_hook("final")))
    with torch.no_grad():
        run()
    for h in handles:
        h.remove()
    return stats


def norm_scale(rec):
    """Power-of-two input scale s for a rmsnorm: mean(y^2)~=1, squares in range."""
    max_abs, max_msq, min_msq = rec
    s = pow2(1.0 / (max(min_msq, 1e-30) * max_msq) ** 0.25)
    while (max_abs * s) ** 2 > NORM_SQ_MAX:
        s /= 2.0
    if min_msq * s * s < 5e-4:
        raise SystemExit(f"rmsnorm dynamic range too wide for fp16: {rec}")
    return s


def mlp_scale(down_in, down_out, gain):
    """Power-of-two up_proj scale m for one layer: brings the down_proj input,
    raw * gain * m, to rms ~1 within fp16 headroom for the down_proj input and
    output. Never below 1."""
    max_abs, sumsq, count = down_in
    m = pow2(1.0 / (gain * (sumsq / count) ** 0.5))
    while m > 1.0 and (max_abs * gain * m > DOWN_IN_MAX or down_out[0] * gain * m > DOWN_OUT_MAX):
        m /= 2.0
    return max(m, 1.0)


def fp16_range_rewrite(calibration, tok, report=print):
    """A rewrite: calibrate on `calibration` (a core.Calibration), then apply
    the 1/K residual rewrite and the MLP precision rewrite."""
    from ..core import Calibration
    from ..tokenizer import encode
    if not isinstance(calibration, Calibration):
        raise TypeError("fp16_range_rewrite calibrates on a core.Calibration, never on evaluation cases")

    def rewrite(backbone):
        model = backbone.model
        rms_eps = model.config.rms_norm_eps

        def run():
            for text in calibration.texts:
                backbone.reference(encode(tok, text))

        stats = _calibrate(model, run)
        residual_max = max(rec[0] for name, rec in stats.items()
                           if name.endswith(("input_layernorm", "pre_feedforward_layernorm", "final")))
        k = pow2(1.0)
        while residual_max / k > RESIDUAL_MAX_TARGET:
            k *= 2.0

        def safe(orig, rec, in_scale, out_scale=1.0):
            s = norm_scale(rec)
            eps_eff = max(rms_eps * s * s, EPS_FLOOR)
            return SafeRMSNorm(orig, mult=s / in_scale, eps_eff=eps_eff, out_scale=out_scale)

        mlp_scales = []
        for i, layer in enumerate(model.layers):
            # the MLP branch reaches post_feedforward_layernorm scaled by
            # TanhGelu.GAIN * m, where m is folded into up_proj
            m = mlp_scale(stats[f"L{i}.down_proj"], stats[f"L{i}.post_feedforward_layernorm"],
                          activations.TanhGelu.GAIN)
            mlp_scales.append(activations.TanhGelu.GAIN * m)
            with torch.no_grad():
                layer.mlp.up_proj.weight.mul_(m)
            layer.mlp.act_fn = activations.TanhGelu()
            layer.input_layernorm = safe(layer.input_layernorm, stats[f"L{i}.input_layernorm"], 1 / k)
            layer.pre_feedforward_layernorm = safe(
                layer.pre_feedforward_layernorm, stats[f"L{i}.pre_feedforward_layernorm"], 1 / k)
            # branch-output norms: output folds the 1/K step. The attention
            # branch arrives unscaled, the MLP branch scaled by GAIN * m.
            layer.post_attention_layernorm = safe(
                layer.post_attention_layernorm, stats[f"L{i}.post_attention_layernorm"], 1.0, 1 / k)
            layer.post_feedforward_layernorm = safe(
                layer.post_feedforward_layernorm, stats[f"L{i}.post_feedforward_layernorm"],
                mlp_scales[-1], 1 / k)
            layer.self_attn.q_norm = safe(layer.self_attn.q_norm, stats[f"L{i}.q_norm"], 1.0)
            layer.self_attn.k_norm = safe(layer.self_attn.k_norm, stats[f"L{i}.k_norm"], 1.0)
        model.norm = safe(model.norm, stats["final"], 1 / k)
        model.embed_tokens = ScaledEmbedding(model.embed_tokens, 1 / k)
        report(f"residual scale K={k:g}; down_proj input scales {[int(s) for s in mlp_scales]}")
    return rewrite
