"""GPTQ onto 8-bit palettes: the compressed weight format a causal LM's
weights take on the ANE (docs/DECISIONS.md D53; docs/design/open-llm.md,
"The converter").

Why palettes, and why GPTQ (#173, MiniCPM5-2B on an M1 Max, macOS 27):
- fp16 weights (4.2 GiB for a 2B model) pass the ANE's resident budget of
  roughly 2-3 GiB, and programs then reload on almost every call;
- the ANE runs only per-channel grids: per-block int8 falls back to the CPU
  without saying so, and per-block int4 doesn't load. Per-channel int8 and
  8-bit palettes run entirely on the ANE;
- rounding each weight to its nearest grid point (round to nearest) loses
  the model's top-1 agreement badly; the loss is the rounding, not the
  ANE's arithmetic, since the same weights score almost the same in fp32
  PyTorch as on the ANE. GPTQ (Frantar et al., 2022) rounds the input
  channels in order and moves each rounding error onto the channels not yet
  rounded, weighted by the inverse Hessian of the layer's inputs (H = X^T X
  over calibration tokens), so the layer's output error is much smaller on
  the same grid. With act-order the channels go in order of decreasing
  Hessian diagonal (GPTQ's desc_act).
- 4-bit is ruled out on this model: the best 4-bit result was 87% top-1.

The grid is an 8-bit palette (a 256-entry lookup table) per group of
GROUP output channels: coremltools' per_grouped_channel palettization. The
palette is fixed before GPTQ by k-means on the weights, as coremltools'
k-means palettizer computes it (scikit-learn k-means++ on the group's fp16
values); GPTQ only chooses each weight's entry. The converter then stores
the weights with the palettizer's `unique` mode, which keeps the values
exactly, since each group holds at most 256 distinct values.

Reproducibility: k-means runs single-threaded with a fixed seed, and GPTQ
in float64 in a fixed order, so a rerun on the same machine and library
versions gives the same weights bit for bit.
"""

import numpy as np
import torch

GROUP = 16
"""Output channels per palette."""

ENTRIES = 256
"""Palette entries: 8 bits."""


def f16(x):
    return x.half().float()


def kmeans_palettes(W, group=GROUP, entries=ENTRIES, seed=0):
    """An `entries`-value palette per group of `group` output channels of W
    [out, in], as coremltools' k-means palettizer computes one: scikit-learn
    k-means++ on the group's distinct fp16 values, weighted by their counts,
    sorted. A group with fewer distinct values than entries repeats its
    largest. Single-threaded, so the result doesn't depend on the machine's
    thread count. Returns fp16 values as float32 [out / group, entries]."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    o, i = W.shape
    if o % group:
        raise ValueError(f"{o} output channels don't split into groups of {group}")
    out = np.zeros((o // group, entries), dtype=np.float16)
    x = W.reshape(o // group, group * i).half().numpy()
    with threadpool_limits(limits=1):
        for r in range(o // group):
            v, cnt = np.unique(x[r], return_counts=True)
            n = min(len(v), entries)
            km = KMeans(n, init="k-means++", tol=1e-2, n_init=1, random_state=seed).fit(
                v.reshape(-1, 1).astype(np.float32), sample_weight=cnt)
            c = np.sort(km.cluster_centers_.reshape(-1)).astype(np.float16)
            out[r, :n] = c
            out[r, n:] = c[-1]
    return torch.from_numpy(out.astype(np.float32))


class Palette:
    """The rounding grid: each output channel's group palette. GPTQ calls
    round() once per input channel."""

    def __init__(self, W, palettes, group=GROUP):
        self.palettes = palettes.double()                          # [G, entries]
        rows = torch.arange(W.shape[0]) // group
        self.row = self.palettes[rows]                             # [out, entries]
        self.mid = (self.row[:, 1:] + self.row[:, :-1]) / 2
        self.codes = torch.zeros(W.shape, dtype=torch.uint8)

    def nearest(self, w):
        """Indices of the entries nearest to w ([out] or [out, k])."""
        w = w if w.dim() == 2 else w.unsqueeze(1)
        return torch.searchsorted(self.mid, w.contiguous()).clamp(max=self.palettes.shape[1] - 1)

    def round(self, w, col):
        idx = self.nearest(w)
        self.codes[:, col] = idx.squeeze(1).to(torch.uint8)
        return torch.gather(self.row, 1, idx).squeeze(1)

    def round_to_nearest(self, W):
        """Every weight to its nearest entry: the baseline GPTQ is measured against."""
        return torch.gather(self.row, 1, self.nearest(W.double())).float()

    def result(self):
        return self.codes, self.palettes.float()


def gptq(W, H, grid, damp=0.01, block=128, act_order=True, report=print):
    """GPTQ of W [out, in] under H [in, in] onto `grid` (a Palette built on
    W). Returns grid.result(): (codes [out, in] uint8, palettes)."""
    W = W.clone().double()
    H = H.clone().double()
    n_in = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True, stable=True) if act_order else torch.arange(n_in)
    W, H = W[:, perm], H[perm][:, perm]
    eye = torch.mean(torch.diag(H)) * torch.eye(n_in, dtype=H.dtype)
    while True:                         # raise the damping until H is numerically positive definite
        try:
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H + damp * eye)), upper=True)
            break
        except torch._C._LinAlgError:
            damp *= 2
            report(f"  GPTQ damping raised to {damp}")
    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            wq = grid.round(w, int(perm[i1 + i]))
            err = (w - wq) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return grid.result()


def dequantize(codes, palettes, group=GROUP):
    """Weights [out, in] from codes [out, in] and group palettes."""
    rows = torch.arange(codes.shape[0]) // group
    return torch.gather(palettes[rows], 1, codes.long())


def output_error(Wq, W, H):
    """The relative output error of Wq against W under H:
    sqrt(tr(D H D^T) / tr(W H W^T)), D = Wq - W."""
    D = (Wq - W).double()
    W = W.double()
    return float(torch.sqrt(torch.einsum("oi,ij,oj->", D, H, D) / torch.einsum("oi,ij,oj->", W, H, W)))


class Hessian:
    """Accumulates X^T X over the inputs a module sees ([1, C, 1, S] or
    [S, C]), in float64. `mean()` is H / tokens."""

    def __init__(self):
        self.sum, self.tokens = None, 0

    def add(self, x):
        X = x.reshape(x.shape[1], -1).double() if x.dim() == 4 else x.double().T     # [in, S]
        h = X @ X.T
        self.sum = h if self.sum is None else self.sum.add_(h)
        self.tokens += X.shape[1]

    def mean(self):
        return self.sum / self.tokens


class Recorder(torch.nn.Module):
    """Wraps a module and adds its input to a Hessian on every call."""

    def __init__(self, module, hessian):
        super().__init__()
        self.module, self.hessian = module, hessian

    def forward(self, x):
        self.hessian.add(x)
        return self.module(x)
