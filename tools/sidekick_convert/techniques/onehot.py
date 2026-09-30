"""Selections from int32 inputs without data-dependent gathers.

A gather indexed by an input traces to an op whose result shape or placement
depends on data; comparing the input with a constant position vector gives
a one-hot matrix instead, and a matmul selects with it. The graph stays
static and on the ANE (laya's classifier interface, docs/DECISIONS.md D28).
The constant vectors are buffers the caller registers once per bucket.
"""

import torch


def positions(seq):
    """Buffer for positions_onehot: (1, 1, seq) arange."""
    return torch.arange(seq, dtype=torch.long).reshape(1, 1, seq)


def positions_onehot(pos, positions_buffer, dtype):
    """pos (1, K) int32 positions -> (1, K, seq) one-hot. A -1 pad matches
    nothing, so its row is zero."""
    return (pos.long().unsqueeze(-1) == positions_buffer).to(dtype)


def indices(n):
    """Buffer for index_onehot: (1, n) arange."""
    return torch.arange(n, dtype=torch.long).reshape(1, n)


def index_onehot(i, indices_buffer, dtype):
    """i (1,) int32 -> (1, n) one-hot row."""
    return (i.long().reshape(1, 1) == indices_buffer).to(dtype)
