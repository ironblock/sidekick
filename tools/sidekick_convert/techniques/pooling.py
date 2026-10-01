"""Pooling inside the graph (docs/DECISIONS.md D15).

A raw per-token output keeps a symbolic sequence dimension that the ANE/CPU
path rejects, so every pooled output ends in a reshape to a literal
(1, dims). Pass `dims` as a Python int: a size read from a tensor traces as
arithmetic that coremltools 9 can't convert under static shapes.

fp16 range: sums over 512 tokens, and sums of squares, can pass 65504.
masked_mean() sums at `prescale` (default 1/32) and divides the scale back
out; l2() scales before squaring, which the normalization cancels.
sidekick normalizes pooled vectors again in f32, so a positive scale on an
embedding output is harmless.
"""

import torch
import torch.nn.functional as F

PRESCALE = 1.0 / 32.0


def cls(hidden, dims):
    """Position 0 ([CLS] or BOS)."""
    return hidden[:, 0, :].reshape(1, dims)


def masked_mean(hidden, attention_mask, dims, prescale=PRESCALE):
    """Mean over the positions whose attention_mask is 1, the
    sentence-transformers mean. Masks by the attention_mask input, never by
    token id: sidekick pads with id 0, and the pad-invariance gate fills pads
    with random ids."""
    m = attention_mask.to(hidden.dtype)
    summed = (hidden * (m * prescale).unsqueeze(-1)).sum(dim=1)
    count = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
    return (summed / (count * prescale)).reshape(1, dims)


def last_token(hidden, attention_mask, dims):
    """The last real position of a right-padded input, selected without a
    data-dependent index: mask * (1 - shift_left(mask)) is 1 exactly there,
    and a masked sum picks it (F2LLM, docs/DECISIONS.md D20). Written as the
    F2LLM converter wrote it, whose locals name the converted values."""
    mask_f = attention_mask.to(hidden.dtype)                    # (1, seq)
    shifted = F.pad(mask_f[:, 1:], (0, 1), value=0.0)          # mask[i+1], last=0
    last_onehot = mask_f * (1.0 - shifted)                     # 1 at last real pos
    pooled = (last_onehot.unsqueeze(-1) * hidden).sum(dim=1)   # (1, dims)
    return pooled.reshape(1, dims)


def l2(y, prescale=PRESCALE):
    """L2-normalize the last dimension, squaring at `prescale`."""
    y = y * prescale
    return y / torch.linalg.vector_norm(y, dim=-1, keepdim=True)
