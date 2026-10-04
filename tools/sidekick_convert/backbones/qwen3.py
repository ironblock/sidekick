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
  - the coarse native silu becomes a silu built from accurate ANE ops (2 *
    silu), with up_proj taking the 1/2, so the MLP adds no op: TanhSilu by
    default, StableSilu (no fp16 cancellation for negative inputs, D39) in
    the F2LLM and agent-jev converters;
  - attention's inputs are small in the early layers (q/k/v at rms
    0.06-0.18, o_proj at 0.02-0.35), under the ANE linear's small-input
    floor: power-of-two scales on the input norm's weight (q/k RMSNorm eps
    x s^2, since q and k are re-normalized per head) and on v_proj bring both
    to rms ~1, and Descale undoes o_proj's before the residual add;
  - optionally (mlp_down) down_proj's input too, via up_proj's rows: once
    the silu is StableSilu, it is the next-largest ANE error on F2LLM
    (scales up to 16 in the early layers) and on agent-jev.
"""

import types
from pathlib import Path

import torch

from ..techniques import activations, attention, masks, precision, traceable
from ..techniques.onehot import positions, positions_onehot
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


def precision_rewrite(calibration, tok, texts_for=lambda cal: cal.texts, report=print, silu=None, mlp_down=False):
    """A rewrite: calibrate attention's input statistics on `calibration`
    (texts_for(calibration) yields the full texts, prompts included), then
    swap in a silu built from accurate ANE ops (`silu`, an activations
    class: TanhSilu by default) and rescale attention's inputs. With
    `mlp_down`, also rescale down_proj's input (up_proj's rows by s, its
    output by 1/s), capped by down_proj's measured peaks so a layer that
    builds an attention sink keeps s = 1. Every factor is a power of two, so
    the fp32 graph is unchanged (the fp32 gate proves it)."""
    silu = silu or activations.TanhSilu
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
                          f"{i}.o_in": (a.o_proj, "in"), f"{i}.o_out": (a.o_proj, "out"),
                          f"{i}.down_in": (layer.mlp.down_proj, "in"), f"{i}.down_out": (layer.mlp.down_proj, "out")})
        def run():
            for text in texts_for(calibration):
                backbone.reference(encode(tok, text))

        stats = collect(run, sites)
        scales = []
        for i, layer in enumerate(model.layers):
            s_d = precision.input_scale(stats[f"{i}.down_in"], stats[f"{i}.down_out"]) if mlp_down else 1.0
            with torch.no_grad():
                layer.mlp.up_proj.weight.mul_(s_d / silu.GAIN)
            layer.mlp.act_fn = silu()
            if s_d != 1.0:
                layer.mlp.down_proj = precision.Descale(layer.mlp.down_proj, 1.0 / s_d)
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
            scales.append(f"L{i} {s_in:g}/{s_o:g}" + (f"/{s_d:g}" if mlp_down else ""))
        report("input scales (q/k/v in, o_proj in" + (", down_proj in" if mlp_down else "") + "): " + ", ".join(scales))
    return rewrite


class Qwen3TreeBackbone(Qwen3Backbone):
    """A Qwen3 decoder over a tree layout (the agentjev format,
    docs/design/classify.md): a shared prefix then sibling branches, each
    branch seeing the prefix and its own earlier tokens only. Ports beyond
    the text ones: `seg` [1, S] (0 prefix, c branch c, -1 pads) and
    `position_ids` [1, S] (every branch continues from the prefix's end).

    The mask is built in-graph from `seg` (masks.tree). RoPE reads constant
    cos/sin tables through a one-hot of `position_ids` rather than computing
    angles: an angle like 2,047 radians stored in fp16 is off by up to 0.5,
    while the tables are exact constants. Positions never exceed the bucket,
    so one table row per position is enough.

    The matmul_softmax() rewrite computes attention's softmax with the value
    matmul (techniques.attention.matmul_softmax), as fev's constraint D does.
    Core ML's own softmax, after an in-graph score matmul, rounds differently
    on the CPU for different key lengths below 1,024, so the same input
    differed between buckets; the matmul form is bit-identical across them
    (tools/repro_cpu_softmax_length.py)."""

    def buffers(self, seq):
        rope = self.model.rotary_emb
        angle = torch.arange(seq, dtype=torch.float64)[:, None] * rope.inv_freq.double()[None, :]
        emb = torch.cat([angle, angle], dim=-1)
        return {"rope_positions": positions(seq),
                "rope_cos": (emb.cos() * rope.attention_scaling).float(),
                "rope_sin": (emb.sin() * rope.attention_scaling).float(),
                **masks.tree_buffers(seq)}

    def call(self, w, x):
        m = getattr(w, self.attr)
        h = self._layers(m.layers, m.embed_tokens(x["input_ids"].long()), w, x)
        return types.SimpleNamespace(last_hidden_state=m.norm(h))

    def _layers(self, layers, h, w, x):
        """`layers` over the residual stream `h`, with the tree mask and RoPE
        built from `x` and the wrapper's buffers."""
        seq = w.seq
        bias = masks.tree(x["seg"], x["attention_mask"], w.tree_causal, w.tree_eye, seq)
        onehot = positions_onehot(x["position_ids"], w.rope_positions, torch.float32)   # [1, S, S]
        rope = (onehot @ w.rope_cos, onehot @ w.rope_sin)                              # [1, S, head_dim]
        for layer in layers:
            layer.self_attn.sidekick_seq = seq      # the key length, a Python int, for _matmul_softmax_attention
            h = layer(h, attention_mask=bias, position_embeddings=rope)
            h = h[0] if isinstance(h, tuple) else h
        for layer in layers:
            layer.self_attn.sidekick_seq = None     # eager runs (references) use the keys' own length
        return h

    # Chunking (chunking.py, D37): each chunk rebuilds the mask and RoPE
    # from the int32 ports, so only the residual stream crosses a boundary.
    chunk_ports = ("input_ids", "attention_mask", "seg", "position_ids")

    def chunk_layers(self):
        return list(self.model.layers)

    def chunk_parts(self, lo, hi):
        layers = self.chunk_layers()
        parts = {"layers": torch.nn.ModuleList(layers[lo:hi])}
        if lo == 0:
            parts["embed_tokens"] = self.model.embed_tokens
        if hi == len(layers):
            parts["norm"] = self.model.norm
        return parts

    def chunk_call(self, w, x):
        h = w.embed_tokens(x["input_ids"].long()) if w.first else x["hidden_in"]
        h = self._layers(w.layers, h, w, x)
        return w.norm(h) if w.last else h


MATMUL_SOFTMAX = "sidekick_matmul_softmax"


def _matmul_softmax_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """transformers' attention-function interface (as sdpa_attention_forward),
    with the softmax computed by the value matmul. Traced, the key length
    comes from module.sidekick_seq, set per bucket by Qwen3TreeBackbone.call:
    a size read from a traced tensor can't be converted under static shapes.
    Run eagerly (an fp32 reference), it is the keys' own length."""
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    seq = getattr(module, "sidekick_seq", None) or int(key.shape[-2])
    o = attention.matmul_softmax(query, k, v, attention_mask, scaling, seq)
    return o.transpose(1, 2).contiguous(), None


def matmul_softmax(backbone):
    """A rewrite: every layer's attention computes its softmax with the value
    matmul (_matmul_softmax_attention) instead of transformers' sdpa path,
    which converts to Core ML's softmax. Apply it after the fp32 references
    are computed, so the fp32 gate proves it exact.

    Process-global in part: it registers _matmul_softmax_attention under a
    new name in transformers' ALL_ATTENTION_FUNCTIONS (adding a name changes
    no other model), and switches this backbone's config to it. A model is
    affected only through its own config, so F2LLM's converter, which
    doesn't call it, and the other Qwen3 tests keep sdpa. Don't apply it to a
    backbone shared between tests or converters."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[MATMUL_SOFTMAX] = _matmul_softmax_attention
    backbone.model.config._attn_implementation = MATMUL_SOFTMAX
    return backbone


def load_tree(src, tok, prefix=""):
    """A Qwen3 decoder for the tree layout, from a checkpoint whose backbone
    tensors are named `<prefix><Qwen3Model name>` in model.safetensors (AgentJev:
    "path_encoder.backbone."). Built without allocating weights and given the
    checkpoint's tensors, so the model never exists twice; RoPE's inv_freq,
    which isn't stored, is rebuilt from the config. The checkpoint's other
    tensors (a task head) are returned for the head to load."""
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    config = Qwen3Config.from_pretrained(src)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    weights = load_file(str(Path(src) / "model.safetensors"))
    with torch.device("meta"):
        model = Qwen3Model(config)
    model.load_state_dict({k[len(prefix):]: v.float() for k, v in weights.items() if k.startswith(prefix)},
                          strict=True, assign=True)
    model.rotary_emb = Qwen3RotaryEmbedding(config=config)
    model.eval().requires_grad_(False)
    install_patches()
    from ..tokenizer import special_ids
    rest = {k: v for k, v in weights.items() if not k.startswith(prefix)}
    ids = tuple(special_ids(tok)) if tok is not None else ()
    return Qwen3TreeBackbone(family="qwen3", model=model, config=config, special_ids=ids), rest
