"""Nandi, FrontiersMind's Llama-style causal decoder (the backbone of
Lumma-fev-0.1b): pre-norm RMSNorm, RoPE, grouped-query attention without
QK-norm, a SwiGLU MLP, a factorized token embedding (a rank-r table, then a
linear to the hidden size) and layer sharing (each layer applied
`layer_sharing_repeats` times in a row).

The checkpoint's own modeling code needs transformers 5 and remote code, so
the backbone is written out here in plain torch from config.json and the
weights, with the checkpoint's module names. The converter's fp32 gate
compares it with the checkpoint's own forward.

What conversion needs:
- ATTENTION, written out: explicit, or the matmul softmax
  (techniques.attention) for bucket invariance. A finite additive mask:
  causal AND key-padding, with every query attending to itself so a pad row
  is never fully masked. RoPE's cos and sin are constant buffers per bucket.
  Real tokens never see the right padding, because the mask is causal.
- fp16 RANGE: no linear output comes near the ANE's 2^15 (the largest is
  ~200), but the residual stream reaches ~870 and every RMSNorm squares its
  input, past fp16's 65504. fp16_norms() gives each RMSNorm that needs it a
  calibrated power-of-two input pre-scale with eps compensated as eps * s^2,
  exact in fp32 because RMSNorm is scale-invariant (gemma3's SafeRMSNorm
  technique, without its eps floor).
- tanh_silu(): Core ML's native silu is coarse on the ANE (D20 amendment);
  TanhSilu computes 2 * silu, and up_proj takes the 1/2.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..techniques import activations, attention, masks
from ..techniques.precision import pow2

NORM_SQ_MAX = 30000.0     # keep (|x| * s)^2 under this inside every RMSNorm


class RMSNorm(torch.nn.Module):
    """weight * x * rsqrt(mean((x * mult)^2) + eps) * mult: the checkpoint's
    RMSNorm when mult = 1; with a power-of-two mult and eps scaled by mult^2,
    the same function in fp32 with its squares kept in fp16 range."""

    def __init__(self, d, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d))
        self.eps = float(eps)
        self.mult = 1.0

    def forward(self, x):
        y = x * self.mult if self.mult != 1.0 else x
        return self.weight * (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps))


class Layer(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        d, ff = c["hidden_size"], c["intermediate_size"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.input_layernorm = RMSNorm(d, c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(d, c["rms_norm_eps"])
        self.q_proj = torch.nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.gate_proj = torch.nn.Linear(d, ff, bias=False)
        self.up_proj = torch.nn.Linear(d, ff, bias=False)
        self.down_proj = torch.nn.Linear(ff, d, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x, cos, sin, add, seq, softmax):
        b, s = 1, int(seq)   # static: a shape read from a traced tensor converts as an op coremltools rejects
        y = self.input_layernorm(x)
        q = self.q_proj(y).reshape(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).reshape(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).reshape(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        half = self.head_dim // 2
        q, k = q * cos + rotate_half(q, half) * sin, k * cos + rotate_half(k, half) * sin
        rep = self.heads // self.kv_heads
        if rep > 1:
            k = k[:, :, None].expand(b, self.kv_heads, rep, s, self.head_dim).reshape(b, self.heads, s, self.head_dim)
            v = v[:, :, None].expand(b, self.kv_heads, rep, s, self.head_dim).reshape(b, self.heads, s, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        if softmax == "matmul":
            a = attention.matmul_softmax(q, k, v, add, scale, seq)
        else:
            a = attention.explicit(q, k, v, add, scale)
        x = x + self.o_proj(a.transpose(1, 2).reshape(b, s, self.heads * self.head_dim))
        y = self.post_attention_layernorm(x)
        return x + self.down_proj(self.act(self.gate_proj(y)) * self.up_proj(y))


def rotate_half(x, half):
    """`half` is static (head_dim / 2): a size read from a traced tensor converts badly."""
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Nandi(torch.nn.Module):
    """The decoder: ids -> final-norm hidden states [1, S, hidden]."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        factorized = c.get("factorized_embedding", False)
        self.embed_tokens = torch.nn.Embedding(c["vocab_size"], c["embedding_rank"] if factorized else c["hidden_size"])
        self.embedding_proj = (torch.nn.Linear(c["embedding_rank"], c["hidden_size"], bias=False)
                               if factorized else None)
        self.layers = torch.nn.ModuleList(Layer(c) for _ in range(c["num_hidden_layers"]))
        self.repeats = c.get("layer_sharing_repeats", 1) if c.get("layer_sharing") else 1
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.softmax = "explicit"

    def rope(self, seq):
        """cos, sin [1, 1, seq, head_dim] for positions 0..seq-1, as the
        checkpoint computes them (fp32)."""
        hd = self.c["head_dim"]
        inv = 1.0 / (self.c["rope_parameters"]["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
        ang = torch.arange(seq).float()[:, None] * inv[None, :]
        emb = torch.cat((ang, ang), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, input_ids, attention_mask, cos, sin, seq):
        x = self.embed_tokens(input_ids)
        if self.embedding_proj is not None:
            x = self.embedding_proj(x)
        add = masks.self_attending(masks.causal(seq) + masks.key_padding(attention_mask, x.dtype), seq)
        for layer in self.layers:
            for _ in range(self.repeats):
                x = layer(x, cos, sin, add, seq, self.softmax)
        return self.norm(x)


def load(src, prefix="lm."):
    """The decoder from a Lumma-fev snapshot (config.json's backbone_config,
    model.safetensors' `lm.` weights) in fp32, plus the remaining state dict
    (the head's weights)."""
    from safetensors.torch import load_file
    cfg = json.loads((Path(src) / "config.json").read_text())
    c = cfg["backbone_config"]
    model = Nandi(c)
    state = {k: v.float() for k, v in load_file(str(Path(src) / "model.safetensors")).items()}
    mine = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        n = k[len(prefix):].replace("self_attn.", "").replace("mlp.", "")
        mine[n] = v
    missing, unexpected = model.load_state_dict(mine, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Nandi state dict mismatch: missing {missing}, unexpected {unexpected}")
    norms_ = model.embed_tokens.weight.norm(dim=-1)
    if not bool((norms_ > 0).all()):
        raise SystemExit("a token embedding is all zero: the fp16 RMSNorm rewrite assumes none is")
    rest = {k: v for k, v in state.items() if not k.startswith(prefix)}
    return model.eval(), cfg, rest


def norms(model):
    """Every RMSNorm, by name, in execution order."""
    out = {}
    for i, layer in enumerate(model.layers):
        out[f"L{i}.input_layernorm"] = layer.input_layernorm
        out[f"L{i}.post_attention_layernorm"] = layer.post_attention_layernorm
    out["final"] = model.norm
    return out


def norm_stats(model, run):
    """fp32 (max |x|, max mean(x^2), min mean(x^2)) at every RMSNorm input
    during run(), over every application of a shared layer."""
    stats, handles = {}, []
    for name, mod in norms(model).items():
        def hook(m, args, name=name):
            t = args[0].detach()
            msq = t.pow(2).mean(-1)
            rec = stats.setdefault(name, [0.0, 0.0, float("inf")])
            rec[0] = max(rec[0], float(t.abs().max()))
            rec[1] = max(rec[1], float(msq.max()))
            rec[2] = min(rec[2], float(msq.min()))
        handles.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        run()
    for h in handles:
        h.remove()
    return stats


def fp16_norms(model, run, report=print):
    """Give each RMSNorm whose input would overflow fp16 when squared a
    power-of-two input scale s < 1, the largest with (max |x| s)^2 under
    NORM_SQ_MAX (calibrated during run()), and eps * s^2 in place of eps.
    Exact in fp32. Norms that don't need it keep s = 1.

    eps is not floored: the inputs' mean squares span more than 4 decades
    (layer 0 reads the raw embedding), so a floor would change the
    small-norm tokens. A scaled eps that fp16 flushes to zero on the ANE
    changes nothing unless a hidden state is entirely zero, which no token's
    embedding is (load() checks it)."""
    stats = norm_stats(model, run)
    scales = {}
    for name, mod in norms(model).items():
        max_abs = stats[name][0]
        s = 1.0
        while (max_abs * s) ** 2 > NORM_SQ_MAX:
            s /= 2.0
        mod.mult = s
        mod.eps = mod.eps * s * s
        scales[name] = s
    report("RMSNorm input scales: " + (", ".join(f"{k} {v:g}" for k, v in scales.items() if v != 1.0) or "none"))
    return stats


def tanh_silu(model):
    """Every MLP's silu becomes TanhSilu (2 * silu), with up_proj taking 1/2."""
    for layer in model.layers:
        with torch.no_grad():
            layer.up_proj.weight.mul_(1.0 / activations.TanhSilu.GAIN)
        layer.act = activations.TanhSilu()


def fold_embedding(model):
    """The factorized embedding folded into one table, exact up to fp32
    rounding: table @ proj^T, [vocab, hidden], and no projection. Core ML
    keeps the first linear after the (CPU) gather on the CPU however it is
    written, and the ANE serves every linear of the graph only without it.
    The cost is size: the table grows from vocab x rank to vocab x hidden."""
    if model.embedding_proj is None:
        return
    with torch.no_grad():
        table = model.embed_tokens.weight @ model.embedding_proj.weight.t()
    emb = torch.nn.Embedding(table.shape[0], table.shape[1])
    emb.weight.data = table
    model.embed_tokens, model.embedding_proj = emb, None
