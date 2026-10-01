"""Attention written out as explicit ops.

Traced values are named after the Python locals they are bound to, and those
names reach the converted program. Keep a converter's binding structure
when refactoring it, or its model.mil changes text though not math.

Never let coremltools emit Core ML's fused `scaled_dot_product_attention`
(docs/DECISIONS.md D25). On the ANE it ignores a mask computed outside its
own ANE procedure: a mask built on the CPU, fed as an input, or built in an
earlier procedure. Where it looks correct (bge-small) it is correct only
through a Core ML fallback that the iOS26 opset no longer has, and its CPU
version NaNs fully masked rows. coremltools emits the fused op for
F.scaled_dot_product_attention without `scale=`; with `scale=`, or written
out as below, attention becomes matmul -> softmax -> matmul. Job.forbid_ops
refuses the fused op by default.

`explicit()` is that form. `matmul_softmax()` is an opt-in variant measured
on laya: the ANE's reduce_sum, which Core ML's softmax uses, is the one op
not bit-identical across compiled buckets (linear, layer_norm, matmul, exp
and reduce_max are), so computing the denominator with the value matmul made
laya's ANE output exactly bucket-invariant across 128/256/512. It changes
output by design, and some inputs regressed on the ANE (a compile-context
effect in one MLP, not the softmax arithmetic), so a model switches to it
only as a separate, measured change. It needs masks.self_attending() on any
mask that can fully mask a row, or that row NaNs on the CPU and spreads.
"""

import torch

from .reduce import blocked_max


def explicit(q, k, v, add, scale):
    """softmax(q k^T * scale + add) v, for (B, H, S, D) q/k/v and an additive mask."""
    scores = (q @ k.transpose(-1, -2)) * scale + add
    return torch.softmax(scores, dim=-1) @ v


def matmul_softmax(q, k, v, add, scale, seq):
    """explicit() with the softmax denominator computed by the value matmul:
    e = exp(w - rowmax(w)); [num | den] = e @ [V | 1]; out = num / den,
    divided before the heads are merged. `seq` is the key length as a Python
    int. The mask is added before the row max. Opt-in."""
    w = (q @ k.transpose(-1, -2)) * scale + add
    e = torch.exp(w - blocked_max(w, seq))
    o = torch.matmul(e, torch.cat([v, torch.ones_like(v[..., :1])], dim=-1))
    return o[..., :-1] / o[..., -1:]


def transformer_encoder_layer(layer, x, add, seq, softmax="native"):
    """nn.TransformerEncoderLayer(norm_first=True, batch_first=True) with its
    attention written out, so neither its fast path nor the fused attention op
    is ever traced (laya's head). Shapes come from module constants, never
    from traced sizes. `add` is an additive (1, 1, 1, S) or (1, 1, S, S) mask;
    softmax="matmul" uses matmul_softmax (opt-in)."""
    attn = layer.self_attn
    heads, d = int(attn.num_heads), int(attn.embed_dim)
    hd = d // heads
    y = layer.norm1(x)
    q, k, v = torch.nn.functional.linear(y, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q.reshape(1, seq, heads, hd).transpose(1, 2)
    k = k.reshape(1, seq, heads, hd).transpose(1, 2)
    v = v.reshape(1, seq, heads, hd).transpose(1, 2)
    attend = matmul_softmax if softmax == "matmul" else explicit
    extra = (seq,) if softmax == "matmul" else ()
    o = attend(q, k, v, add, hd ** -0.5, *extra).transpose(1, 2).reshape(1, seq, d)
    x = x + attn.out_proj(o)
    return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
