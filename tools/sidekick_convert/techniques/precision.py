"""The ANE linear's small-input precision floor (docs/DECISIONS.md D17, D19).

The ANE's linear op has an ABSOLUTE precision floor on its input: its
relative error is ~3e-4 / rms(input), 0.04% at rms 1 but 2% at rms 0.016.
fp16 itself doesn't have this limit (the same graph on the GPU is exact to
~1e-6). The fix is a power-of-two scale that brings a linear's input to
rms ~1, cancelled downstream:
- fold it into whatever produces the input (a norm's weight, rows of the
  previous projection), exact because powers of two are exact in fp32;
- undo it after the linear with Descale, an explicit multiply, when no
  scale-invariant norm follows (a residual add does). Dividing the weights
  instead would push small ones into fp16's subnormal range.

Scales are chosen from fp32 statistics (calibrate.Stat) with headroom caps
on the rescaled input and output, and are never below 1.
"""

import numpy as np
import torch

IN_MAX = 2048.0      # cap on |linear input| after a rescale
OUT_MAX = 16384.0    # cap on |linear output| after a rescale


def pow2(x):
    """Nearest power of two (exact in floating point)."""
    return 2.0 ** round(np.log2(x))


def input_scale(stat, out=None, gain=1.0, in_max=IN_MAX, out_max=OUT_MAX):
    """Power-of-two scale bringing `stat` (arriving already scaled by `gain`)
    to rms ~1, within headroom for the input and, if given, the output. >= 1."""
    s = pow2(1.0 / (gain * stat.rms))
    while s > 1.0 and (stat.max * gain * s > in_max
                       or (out is not None and out.max * gain * s > out_max)):
        s /= 2.0
    return max(s, 1.0)


class Descale(torch.nn.Module):
    """inner(x) * inv: undoes a rescale before a residual add."""

    def __init__(self, inner, inv):
        super().__init__()
        self.inner = inner
        self.inv = float(inv)

    def forward(self, x):
        return self.inner(x) * self.inv
