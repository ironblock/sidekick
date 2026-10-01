"""Attention masks that stay finite in fp16.

transformers fills masked logits with torch.finfo(dtype).min. That is -inf
once the graph runs in fp16 (the ANE's only precision), and softmax over -inf
turns to NaN (docs/DECISIONS.md D15). Masks here use MASK_ADD = -30000, which
fp16 represents and which still zeroes a masked key after softmax.

Two failure modes to design against:
- A query row whose keys are ALL masked. Explicit softmax still returns a
  finite (uniform) row, but Core ML's fused attention on the CPU returns NaN
  when |fill| x sqrt(head_dim) > 65504, and some softmax rewrites divide by
  zero. `self_attending()` lets every query see itself, so no row is fully
  masked. It is exact for real tokens, which always see themselves anyway.
- A mask the ANE's fused attention never reads (D25): don't use the fused
  op (techniques.attention).
"""

import torch

MASK_ADD = -30000.0


def key_padding(attention_mask, dtype=torch.float32):
    """(B, S) 1/0 mask -> additive (B, 1, 1, S): 0 for real keys, MASK_ADD for
    pads. The geometry of transformers' encoder extended attention mask."""
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * MASK_ADD


def band(seq, keep):
    """(1, 1, S, S) additive mask from `keep(distance)`, a boolean function of
    |q - k| (e.g. lambda d: d <= 64 for a +-64 sliding window)."""
    idx = torch.arange(seq)
    d = (idx[:, None] - idx[None, :]).abs()
    return (~keep(d)).to(torch.float32).reshape(1, 1, seq, seq) * MASK_ADD


def causal(seq):
    """(1, 1, S, S) additive causal mask: key k is visible to query q iff k <= q."""
    idx = torch.arange(seq)
    return (idx[None, :] > idx[:, None]).to(torch.float32).reshape(1, 1, seq, seq) * MASK_ADD


def self_attending(mask, seq=None):
    """Zero the diagonal of an (..., S, S) additive mask, so every query can
    attend to itself and no row is fully masked. Pass `seq` as a Python int
    when the mask is computed from an input: reading the size from a traced
    tensor records arithmetic that coremltools 9 can't convert."""
    if seq is None:
        seq = int(mask.shape[-1])
    eye = torch.eye(seq, dtype=torch.bool)
    return mask.masked_fill(eye, 0.0)


def tree_buffers(seq):
    """Constants for tree(): the (1, 1, S, S) causal visibility (1 where key
    k <= query q) and identity, as floats."""
    idx = torch.arange(seq)
    visible = (idx[None, :] <= idx[:, None]).to(torch.float32).reshape(1, 1, seq, seq)
    return {"tree_causal": visible, "tree_eye": torch.eye(seq).reshape(1, 1, seq, seq)}


def tree(seg, attention_mask, causal_visible, eye, seq):
    """(1, 1, S, S) additive mask for a tree layout (the agentjev format): a
    shared prefix (segment 0), then sibling branches (segments 1, 2, ...),
    with pads at segment -1. Query i sees key j iff j <= i, both are real
    tokens (segment >= 0, and attention_mask 1 at j), and j is in the prefix
    or in i's own branch. Siblings never see each other, so each branch computes exactly
    its own path. Every query also sees itself (self_attending), so a pad's
    row is never empty and random pad ids change nothing.

    Float arithmetic throughout: coremltools has no conversion for boolean
    `|`, and comparisons of int32 inputs convert cleanly. `seq` is a Python
    int (see self_attending)."""
    sj, si = seg.reshape(1, 1, 1, seq), seg.reshape(1, 1, seq, 1)
    shared_or_own = torch.clamp((sj == 0).to(torch.float32) + (sj == si).to(torch.float32), max=1.0)
    # both ends real: a pad query (segment -1) sees nothing but itself
    real = ((sj >= 0).to(torch.float32) * (si >= 0).to(torch.float32)
            * (attention_mask.reshape(1, 1, 1, seq) == 1).to(torch.float32))
    allowed = torch.clamp(causal_visible * real * shared_or_own + eye, max=1.0)
    return (1.0 - allowed) * MASK_ADD
