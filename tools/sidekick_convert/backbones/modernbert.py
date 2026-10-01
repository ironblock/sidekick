"""ModernBERT encoders (gte-modernbert, laya's encoder): pre-norm, RoPE,
alternating global and sliding-window attention, GeGLU MLP.

What conversion needs (docs/DECISIONS.md D25 and its amendment):
- ATTENTION. "explicit" loads transformers' eager path (matmul -> softmax ->
  matmul). "fused" loads sdpa, which coremltools turns into Core ML's fused
  attention op; on the ANE that op ignores ModernBERT's mask, because
  transformers builds the masks before the CPU-only embedding gather, which
  puts them on the CPU. It exists only as a negative control.
- MASKS. transformers builds a global mask and a sliding-window mask
  (|q - k| <= local_attention // 2) with finfo.min, -inf in fp16. The patch
  builds the same two masks with -30000. With `self_attending_pads`, every
  query may also attend to itself, so no row is fully masked (a pad query
  far past the last real token has its whole window masked); exact for real
  tokens, and needed with matmul_softmax, whose CPU path NaNs such rows.
- TRACEABLE RoPE. Stock rotate_half slices at x.shape[-1] // 2; the
  traceable version from techniques.traceable replaces it.
- STATIC SHAPES. Eager and sdpa keep full static shapes (only
  flash_attention_2 unpads); position_ids is a buffer.
- THE ANE LINEAR'S 2^15 LIMIT. The massive activation (dimension 251 on
  delimiter tokens) is written by an MLP output projection past 32,768
  (gte: layer 15, ~50,000). residual_k() runs the residual stream at 1/K,
  exact in fp32: the embedding norm's weight takes 1/K, layer 0's Wqkv (which
  reads the embedding directly) takes K, both output projections of every
  layer take 1/K, and every LayerNorm's eps takes 1/K^2 (they are
  scale-invariant). K is calibrated (smallest power of two under 0.85 x 2^15)
  or pinned.

Opt-in rewrites measured on laya (docs/CONVERTING.md): twice_gelu (the
encoder MLP's erf GELU from erf, mul and add, 0.5 folded into Wi's gate rows)
and matmul_softmax (the softmax denominator from the value matmul, which made
laya's ANE output bucket-invariant). Each changes the artifact, so a model
takes one only as a separate, measured change.

- CONFIG FROM TRANSFORMERS 5. Checkpoints saved by transformers 5 carry RoPE
  theta in a `rope_parameters` block ({"full_attention": {"rope_theta": ..},
  "sliding_attention": {..}}), which transformers 4.57 ignores, falling back
  to 160000 / 10000. mmBERT's are 160000 / 160000, so a 4.57 load would
  silently build the wrong model, and an fp32 reference built the same way
  can't catch it. load() applies the block, and verify_config() checks every
  layer's attention type and rotary frequencies against the file.

The patches are installed on transformers' ModernBERT module and class, and
so apply to every ModernBERT model in the process.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..techniques import activations, attention, masks, saturation, traceable
from ..calibrate import linear_maxima
from . import Backbone

MASK_ADD = masks.MASK_ADD


def _make_update_attention_mask(self_attending):
    def _fp16_safe_update_attention_mask(self, attention_mask, output_attentions=False):
        # byte-for-byte the geometry transformers builds in
        # ModernBertModel._update_attention_mask, with MASK_ADD instead of
        # finfo(dtype).min so masked logits stay fp16-representable. Returns
        # (global_mask, sliding_window_mask), both (bsz, 1, seq, seq) additive.
        seq = attention_mask.shape[-1]
        keypad = (1.0 - attention_mask.to(torch.float32))  # 1 at pad positions
        big = (keypad * MASK_ADD)[:, None, None, :]         # (bsz, 1, 1, seq)
        global_mask = big.expand(attention_mask.shape[0], 1, seq, seq).contiguous()
        rows = torch.arange(seq).unsqueeze(0)
        distance = torch.abs(rows - rows.T)
        window_bad = (distance > self.config.local_attention // 2)[None, None]
        sliding_mask = global_mask.masked_fill(window_bad, MASK_ADD)
        if self_attending:
            eye = torch.eye(seq, dtype=torch.bool)
            global_mask = global_mask.masked_fill(eye, 0.0)
            sliding_mask = sliding_mask.masked_fill(eye, 0.0)
        return global_mask, sliding_mask
    return _fp16_safe_update_attention_mask


def install_patches(self_attending_pads=False):
    import transformers.models.modernbert.modeling_modernbert as _mb
    _mb.rotate_half = traceable.rotate_half
    _mb.ModernBertModel._update_attention_mask = _make_update_attention_mask(self_attending_pads)


class ModernBertBackbone(Backbone):
    @property
    def encoder(self):
        return self.model.base_model

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

    def linears(self):
        return linears(self.encoder)


def linears(encoder):
    """A ModernBERT encoder's linears, keyed (layer, name) as residual_k groups them."""
    out = {}
    for i, layer in enumerate(encoder.layers):
        for name, lin in (("Wqkv", layer.attn.Wqkv), ("attn.Wo", layer.attn.Wo),
                          ("Wi", layer.mlp.Wi), ("mlp.Wo", layer.mlp.Wo)):
            out[(i, name)] = lin
    return out


def file_config(config_dir):
    """The checkpoint's config.json as written, or {}."""
    p = Path(config_dir) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _rope_thetas(raw):
    """(global theta, local theta) from the file, transformers 5's
    rope_parameters first, then the 4.x keys; None where absent."""
    rp = raw.get("rope_parameters") or {}
    full = (rp.get("full_attention") or {}).get("rope_theta", raw.get("global_rope_theta"))
    sliding = (rp.get("sliding_attention") or {}).get("rope_theta", raw.get("local_rope_theta"))
    return full, sliding


def resolve_config(config, raw):
    """Apply the file's transformers-5 rope_parameters, which 4.57 ignores."""
    full, sliding = _rope_thetas(raw)
    if full is not None:
        config.global_rope_theta = full
    if sliding is not None:
        config.local_rope_theta = sliding
    return config


def verify_config(encoder, raw):
    """Fail unless every layer's attention type and rotary frequencies match
    the checkpoint's config file."""
    full, sliding = _rope_thetas(raw)
    layer_types = raw.get("layer_types")
    for i, layer in enumerate(encoder.layers):
        is_global = layer.attn.local_attention == (-1, -1)
        if layer_types is not None:
            want = "full_attention" if is_global else "sliding_attention"
            if layer_types[i] != want:
                raise SystemExit(f"layer {i} builds as {want}, but config.json's layer_types says {layer_types[i]}")
        theta = full if is_global else (sliding if sliding is not None else full)
        if theta is None:
            continue
        d = int(layer.attn.head_dim)
        want = 1.0 / (float(theta) ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        got = layer.attn.rotary_emb.inv_freq.float()
        if not torch.allclose(got, want, rtol=1e-6, atol=0):
            raise SystemExit(f"layer {i}: rotary theta doesn't match config.json ({theta}); the installed "
                             "transformers read the config differently from the one that saved it")


def load(src, tok, *, task=None, attention="explicit", self_attending_pads=False, model=None,
         config_dir=None):
    """Load a ModernBERT checkpoint for conversion. `model` passes an
    already-built encoder (laya builds its own around it) instead of loading
    one from `src`; its config.json is read from `config_dir`."""
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    if attention not in ("explicit", "fused"):
        raise ValueError(f"attention must be explicit or fused, not {attention!r}")
    impl = "eager" if attention == "explicit" else "sdpa"
    raw = file_config(config_dir or src)
    if model is None:
        cls = {None: AutoModel, "sequence-classification": AutoModelForSequenceClassification}[task]
        config = resolve_config(AutoConfig.from_pretrained(src), raw)
        model = cls.from_pretrained(src, config=config, dtype=torch.float32, attn_implementation=impl).eval()
    else:
        model.config._attn_implementation = impl
    verify_config(model.base_model, raw)
    install_patches(self_attending_pads)
    from ..tokenizer import special_ids
    return ModernBertBackbone(family="modernbert", model=model, config=model.config,
                              special_ids=tuple(special_ids(tok)))


def maxima(encoder, run):
    """fp32 max |output| of every linear of a ModernBERT encoder during run(),
    keyed (layer, name)."""
    return linear_maxima(run, linears(encoder))


def split_maxima(m):
    """(largest output that doesn't scale with the residual, largest that does):
    only the output projections (attn.Wo, mlp.Wo) scale with 1/K; Wqkv and Wi
    read scale-invariant norms."""
    fixed = max(v for (i, n), v in m.items() if not n.endswith("Wo"))
    scaled = max(v for (i, n), v in m.items() if n.endswith("Wo"))
    return fixed, scaled


def residual_rewrite(backbone, k):
    """The residual stream at 1/k, exact in fp32."""
    residual_rewrite_encoder(backbone.encoder, k)


def residual_rewrite_encoder(enc, k):
    """residual_rewrite() on a bare ModernBERT encoder."""
    if k == 1:
        return
    with torch.no_grad():
        def scale(module, factor):
            module.weight.mul_(factor)
            if getattr(module, "bias", None) is not None:
                module.bias.mul_(factor)

        scale(enc.embeddings.norm, 1.0 / k)
        for layer in enc.layers:
            if isinstance(layer.attn_norm, torch.nn.Identity):
                layer.attn.Wqkv.weight.mul_(k)   # layer 0 reads the embedding directly
            else:
                layer.attn_norm.eps /= k * k
            layer.mlp_norm.eps /= k * k
            scale(layer.attn.Wo, 1.0 / k)
            scale(layer.mlp.Wo, 1.0 / k)
        enc.final_norm.eps /= k * k


def residual_k(calibration, tok, k="auto", report=print):
    """A rewrite: calibrate every encoder linear's largest output on
    `calibration` (a core.Calibration), choose K (or check a pinned K's
    headroom), and apply residual_rewrite."""
    from ..core import Calibration
    from ..tokenizer import encode
    if not isinstance(calibration, Calibration):
        raise TypeError("residual_k calibrates on a core.Calibration, never on evaluation cases")

    def rewrite(backbone):
        def run():
            for text in calibration.texts:
                ids = encode(tok, text)
                backbone.reference(ids if len(ids) <= 512 else ids[:511] + ids[-1:])  # as the server truncates

        fixed, scaled = split_maxima(maxima(backbone.encoder, run))
        if k == "auto":
            chosen, headroom = saturation.choose_k(fixed, scaled)
        else:
            chosen, headroom = k, saturation.headroom_at(fixed, scaled, k)
            if headroom < 1.0 / saturation.HEADROOM:
                raise SystemExit(f"pinned K={k} leaves only {headroom:.2f}x headroom under 2^15")
        residual_rewrite(backbone, chosen)
        report(f"residual scale K={chosen}; largest calibrated linear output is {headroom:.2f}x under the "
               f"ANE linear's {saturation.ANE_LINEAR_MAX:.0f}")
    return rewrite


def twice_gelu(backbone):
    """Opt-in rewrite: every encoder MLP's erf GELU becomes TwiceGelu, with the
    0.5 folded into Wi's gate rows (ModernBERT's MLP is Wo(act(input) * gate)).
    Exact in fp32; the native gelu op is then forbidden."""
    if getattr(backbone.config, "hidden_activation", None) != "gelu":
        raise SystemExit(f"twice_gelu needs hidden_activation 'gelu' (erf), "
                         f"not {backbone.config.hidden_activation!r}")
    ff = int(backbone.config.intermediate_size)
    for layer in backbone.encoder.layers:
        layer.mlp.act = activations.TwiceGelu()
        with torch.no_grad():
            layer.mlp.Wi.weight[ff:].mul_(1.0 / activations.TwiceGelu.GAIN)
            if layer.mlp.Wi.bias is not None:
                layer.mlp.Wi.bias[ff:].mul_(1.0 / activations.TwiceGelu.GAIN)
    backbone.forbid_ops.add("gelu")


def _matmul_softmax_attention(module, qkv, attention_mask, sliding_window_mask, position_ids, local_attention,
                              bs, dim, output_attentions=False, **_kwargs):
    # transformers' eager_attention_forward with the softmax and value matmul
    # replaced by techniques.attention.matmul_softmax
    import transformers.models.modernbert.modeling_modernbert as _mb
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = _mb.apply_rotary_pos_emb(query, key, cos, sin)
    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask
    seq = int(query.shape[-2])
    attn_output = attention.matmul_softmax(query, key, value, attention_mask, module.head_dim ** -0.5, seq)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output.view(bs, -1, dim),)


def matmul_softmax(backbone):
    """Opt-in rewrite: every encoder attention uses techniques.attention.
    matmul_softmax. Needs the backbone loaded with self_attending_pads=True."""
    import transformers.models.modernbert.modeling_modernbert as _mb
    _mb.MODERNBERT_ATTENTION_FUNCTION["eager"] = _matmul_softmax_attention
