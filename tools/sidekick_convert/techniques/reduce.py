"""Reductions that are correct on every compute path.

On macOS 27, Core ML's CPU reduce_max over 256 or more elements returns
max(x, 0), and reduce_min returns min(x, 0): a row of all-negative values
comes back as 0. Lengths up to 255 are fine, as are the ANE and the GPU. A
small reduce op can land on the CPU even under CPU_AND_NE, so any explicit
max over a long axis is a correctness trap, not only softmax's.
blocked_max() takes the max of 128-wide slices and combines them with
elementwise maximum, which is exact everywhere, and bucket-invariant because
max is exact.
"""

import torch

BLOCK = 128


def blocked_max(w, n, block=BLOCK):
    """max over the last dimension, keepdim=True. `n` is that dimension's
    size as a Python int: a size read from a tensor traces as arithmetic
    that coremltools can't convert under static shapes. This is the form
    measured on laya (the matmul softmax's row max)."""
    m = None
    for b in range(0, int(n), block):
        mb = w[..., b:b + block].max(dim=-1, keepdim=True).values
        m = mb if m is None else torch.maximum(m, mb)
    return m
