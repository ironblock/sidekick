"""EmbeddingGemma 2 (google/embeddinggemma-2): the text tower of a Gemma 4
based embedder, loaded text-only (the vision and audio towers dropped).

The architecture, as transformers 5.19 implements it
(models/embedding_gemma2):
- 24 bidirectional layers at hidden size 512. Sliding layers attend where
  |q - k| <= sliding_window (512, inclusive, not halved: 1,025 positions);
  full layers (5, 11, 17, 23) attend everywhere, at head_dim 512 with one
  KV head against the sliding layers' 256 and two. RoPE theta is 10k
  local, 1M global. Attention scaling is 1.0, with q and k RMSNorms.
- Per-layer inputs: a linear projection of the token embeddings to 24 x 512
  (no separate per-layer table), each layer gating its hidden state
  against its slice.
- A 512 -> 768 projection inside the model; the output is [B, S, 768].

What export needs: the masks. transformers builds them with mask
functions that don't trace; the model accepts them prepared, as a mapping
from layer type to an additive 4-D mask, so `onnx_call` builds the
key-padding mask and the sliding band from the input's length (and `call`,
for Core ML, from a per-bucket band buffer), and the fp32 gate checks them
against the checkpoint's own forward.

What fp16 needs (fp16_range_rewrite), from fp32 activation ranges:
- RMSNorm squares its input, and several inputs pass fp16's square root of
  65504 (255.9): post_per_layer_input_norm reaches ~1,600, v_norm ~700, and
  q_norm, k_norm and pre_feedforward_layernorm ~180. Every RMSNorm gets a
  calibrated power-of-two input pre-scale with eps compensated as eps * s^2,
  exact because RMSNorm is scale-invariant.
- The residual stream stays under ~200 and attention scores under ~25
  (scaling 1.0 after q and k norms), so neither needs a rescale.
- gelu_pytorch_tanh, in the MLP and in the per-layer-input gate, becomes
  TanhGelu (2 * gelu from tanh, mul and add; Core ML's gelu op is coarse on
  the ANE); its gain of 2 is divided out in the pre-scale of the norm each
  branch ends in (post_feedforward_layernorm, post_per_layer_input_norm).
- The output projection reaches ~4,000 per token, so the mean pool's sum
  over a long bucket needs its own pre-scale (the converter's head).

Bucket invariance on Core ML's CPU path: its softmax rounds differently
once the key axis passes 512 (measured on an M1 Max: a single softmax op
over 20 real keys differs in 2,177 of 3,200 probabilities, by up to one
fp16 ulp, between 512 and 1,024 keys; matmul, linear, RMSNorm and the
pooling sum don't). A CPU embedding moves by 1 - cos 6.4e-5 between buckets
512 and 1,024. The library's matmul softmax (techniques/attention.py),
which computes the denominator with the value matmul, made the CPU output
bit-identical across buckets 128 and 1,024, but it broke the ANE at bucket
256 (cosine 0.44, against 0.999995 at 128 and 1,024), a compile-context
failure like the one docs/DECISIONS.md records for laya, so it isn't used.

Requires transformers >= 5.19.
"""

import torch

from ..techniques import activations, traceable
from . import Backbone

MASK_ADD = -30000.0
# the norms that follow a gelu branch, which see TanhGelu's gain
AFTER_GELU = ("post_feedforward_layernorm", "post_per_layer_input_norm")


class EmbeddingGemma2Backbone(Backbone):
    @property
    def hidden_size(self):
        """The output width: the 768 of the in-model projection, not the
        transformer's 512."""
        return int(self.config.text_config.embedding_dim)

    @property
    def vocab_size(self):
        return int(self.config.text_config.vocab_size)

    @property
    def window(self):
        return int(self.config.text_config.sliding_window)

    def buffers(self, seq):
        idx = torch.arange(seq, dtype=torch.long)
        band = ((idx[:, None] - idx[None, :]).abs() > self.window).to(torch.float32)
        return {"position_ids": idx.unsqueeze(0), "band_mask": band.reshape(1, 1, seq, seq) * MASK_ADD}

    def call(self, w, x):
        """The text model at one bucket, with the masks `onnx_call` builds."""
        pad = (1.0 - x["attention_mask"].to(torch.float32)).reshape(1, 1, 1, w.seq) * MASK_ADD
        return getattr(w, self.attr).language_model(
            input_ids=x["input_ids"].long(),
            attention_mask={"full_attention": pad, "sliding_attention": pad + w.band_mask},
            position_ids=w.position_ids,
        )

    def reference(self, ids, token_type_ids=None):
        """The checkpoint's own forward (its own masks), fp32, unpadded."""
        t = torch.tensor([ids])
        with torch.no_grad():
            return self.model(input_ids=t, attention_mask=torch.ones_like(t))

    def onnx_call(self, model, input_ids, attention_mask):
        """The text model on dynamic shapes, with the masks transformers
        would build: key padding everywhere, plus the |q - k| > window band
        on sliding layers."""
        seq = input_ids.shape[1]
        idx = torch.arange(seq, device=input_ids.device)
        pad = (1.0 - attention_mask.to(torch.float32))[:, None, None, :] * MASK_ADD
        band = ((idx[:, None] - idx[None, :]).abs() > self.window).to(torch.float32)[None, None] * MASK_ADD
        return model.language_model(
            input_ids=input_ids,
            attention_mask={"full_attention": pad, "sliding_attention": pad + band},
            position_ids=idx[None, :].expand(input_ids.shape[0], -1),
        ).last_hidden_state


class SafeRMSNorm(torch.nn.Module):
    """EmbeddingGemma2RMSNorm rewritten for fp16 range, identical in fp32:
    y = x * mult, then y * rsqrt(mean(y^2) + eps_eff) * weight, where
    mult = s / gain for an input arriving `gain` times its raw value and
    eps_eff = eps * s^2."""

    def __init__(self, orig, mult, eps_eff):
        super().__init__()
        self.register_buffer("weight", orig.weight.detach().clone() if orig.with_scale else None)
        self.mult = float(mult)
        self.eps_eff = float(eps_eff)

    def forward(self, x):
        y = x * self.mult
        n = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps_eff)
        return n if self.weight is None else n * self.weight


def _norms(text_model):
    """(owner, attribute, name) for every RMSNorm in the text model."""
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RMSNorm
    out = []
    for name, mod in text_model.named_modules():
        for attr, child in mod.named_children():
            if isinstance(child, EmbeddingGemma2RMSNorm):
                out.append((mod, attr, f"{name}.{attr}" if name else attr))
    return out


def fp16_range_rewrite(calibration, tok, report=print):
    """A rewrite: TanhGelu for every gelu, and a calibrated power-of-two
    pre-scale on every RMSNorm (calibrated on `calibration`, a
    core.Calibration)."""
    from ..core import Calibration
    from ..tokenizer import encode
    from .gemma3 import norm_scale
    if not isinstance(calibration, Calibration):
        raise TypeError("fp16_range_rewrite calibrates on a core.Calibration, never on evaluation cases")

    def rewrite(backbone):
        text = backbone.model.language_model
        eps = float(text.config.rms_norm_eps)
        norms = _norms(text)
        stats = {}

        def hook(name):
            def f(mod, args):
                t = args[0].detach().float()
                msq = t.pow(2).mean(-1)
                rec = stats.setdefault(name, [0.0, 0.0, float("inf")])
                rec[0] = max(rec[0], float(t.abs().max()))
                rec[1] = max(rec[1], float(msq.max()))
                rec[2] = min(rec[2], float(msq.min()))
            return f

        handles = [getattr(owner, attr).register_forward_pre_hook(hook(name)) for owner, attr, name in norms]
        for t in calibration.texts:
            backbone.reference(encode(tok, t))
        for h in handles:
            h.remove()
        for layer in text.layers:
            layer.mlp.act_fn = activations.TanhGelu()
            layer.ple_block.act_fn = activations.TanhGelu()
        scales = {}
        for owner, attr, name in norms:
            s = norm_scale(stats[name])
            gain = activations.TanhGelu.GAIN if attr in AFTER_GELU else 1.0
            setattr(owner, attr, SafeRMSNorm(getattr(owner, attr), s / gain, eps * s * s))
            scales[name] = s
        report(f"rmsnorm pre-scales: {len(scales)} norms, {min(scales.values()):g} to {max(scales.values()):g}")
    return rewrite


def traceable_helpers(backbone):
    """A rewrite: rotate_half and repeat_kv without shape arithmetic, which
    coremltools can't convert under static shapes (techniques/traceable.py).
    The ONNX export traces the stock helpers, so a rewrite (Core ML only)
    rather than a load-time patch."""
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as _eg2
    traceable.install(_eg2)


def load(src, tok):
    import transformers
    from transformers import AutoConfig, AutoModel
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 19):
        raise SystemExit(f"EmbeddingGemma 2 needs transformers >= 5.19, not {transformers.__version__}")
    config = AutoConfig.from_pretrained(src, vision_config=None, audio_config=None)
    model = AutoModel.from_pretrained(src, config=config, dtype=torch.float32, attn_implementation="eager").eval()
    from ..tokenizer import special_ids
    return EmbeddingGemma2Backbone(family="embedding_gemma2", model=model, config=model.config,
                                   special_ids=tuple(special_ids(tok)), forbid_ops={"gelu"})
