"""Relative-position attention terms without gathers.

Some attentions add a term that reads a table by the distance between query
and key: score[r, c] += x_r · T[idx(r - c)] (DeBERTa's content-to-position)
or x_c · T[idx(r - c)] (its position-to-content). transformers builds them
with torch.gather over an index that depends only on r - c. On macOS 27 every
such gather becomes a `gather_along_axis` on the CPU, and the matmul feeding
it goes with it: an ANE/CPU hand-off per gather, and a plan that fails the
heavy-op rule (deberta-v3-small: 12 gathers, 6 matmuls on the CPU).

For a static bucket of L tokens, r - c takes the 2L - 1 values
-(L-1)..(L-1). So the table is expanded over those distances once, a
weight-only constant per bucket (`query_side_index()` / `key_side_index()`
pick its rows), and the term is read off one matmul by a relative shift
(`skew()`), which is only reshape and slice. Exact in fp32 (tests); on the
ANE, deberta-v3-small and DeBERTa-v3-large (GLiNER2.5-Decide) run 98.5% of
their operations there, with one hand-off.

The expanded table has 2L - 1 rows per term and layer: the same order as the
model's own position table up to bucket 256, and twice it at 512.
Computing the expansion in-graph instead costs as much as the model's own
position projections, and coremltools folds weight-only expressions into
constants anyway.

L is always a Python int: a size read from a tensor traces as arithmetic
that coremltools 9 can't convert under static shapes.
"""

import torch


def skew(x, seq):
    """(..., L, 2L - 1) -> (..., L, L) with out[r, c] = x[r, c - r + L - 1].

    Column e of x holds distance d = (L - 1) - e from row r's point of view, so
    out[r, c] is x at distance r - c. Only reshape and slice."""
    if seq == 1:
        return x
    # flatten/unflatten on the last two dims: no size is read from x
    flat = torch.flatten(x, start_dim=-2)[..., seq - 1 : seq - 1 + seq * (2 * seq - 2)]
    return torch.unflatten(flat, -1, (seq, 2 * seq - 2))[..., :seq]


def distances(seq):
    """The 2L - 1 distances r - c, in the column order skew() expects: (L-1) .. -(L-1)."""
    return (seq - 1) - torch.arange(2 * seq - 1)


def query_side_index(index_of_distance, seq):
    """Rows of a table T to expand for score[r, c] = x_r · T[idx(r - c)]:
    T[query_side_index(idx, L)] then query_side()."""
    return index_of_distance(distances(seq))


def key_side_index(index_of_distance, seq):
    """Rows of T to expand for score[r, c] = x_c · T[idx(r - c)]:
    T[key_side_index(idx, L)] then key_side()."""
    return index_of_distance(-distances(seq))


def query_side(xq, expanded, seq):
    """score[r, c] = xq[r] · T[idx(r - c)], for xq (..., L, D) and the
    expanded table (..., 2L - 1, D) from query_side_index()."""
    return skew(xq @ expanded.transpose(-1, -2), seq)


def key_side(xk, expanded, seq):
    """score[r, c] = xk[c] · T[idx(r - c)], for xk (..., L, D) and the
    expanded table from key_side_index()."""
    return skew(xk @ expanded.transpose(-1, -2), seq).transpose(-1, -2)
