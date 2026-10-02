"""Reproduce Core ML's length-dependent CPU matmul accumulation, standalone.

============================== THE CPU RULE ===============================
On CPU_ONLY, Core ML's fp16 matmul is accurate at every contraction length
(within about half an fp16 ulp of the exact result here), but past 1,024 it
sums in a different order: the same real values, zero-padded to 1,024 and to
2,048, differ by about 2e-3 (754 real keys; the outputs reach ~46, where an
fp16 ulp is 0.03). Up to 1,024 the order doesn't depend on the length: 384
real keys padded to 512, 1,024 and 2,048 give bit-identical results.

Slicing the contraction does not help. The same matmul written as fixed
512-wide slices summed in order agrees with itself at every length, but it
is about 13x less accurate (0.20 from the exact result, against 0.015 for
the single matmul), and it differs from the single matmul even when only
one slice holds data. Core ML computes a sliced matmul with a less precise
kernel.
===========================================================================

Why it matters: a classifier converted with one artifact per sequence
bucket is checked for giving the same output for the same input in every
bucket that holds it (the parity suite's bucket-invariance gate, exact on
the CPU). Attention's value matmul contracts over the key length, which is
the bucket. Lumma-fev-0.1b (buckets up to 2,048) is bit-identical on the CPU
in every bucket up to 1,024, and for every input of 512 tokens or fewer in
every bucket; inputs past 512 tokens move by up to 0.021 in probability
between its 1,024 and 2,048 buckets. That is a reordering inside the
matmul's own rounding error, and slicing would trade it for a large
accuracy loss, so a CPU bucket over 1,024 can't be made exactly invariant
this way.

This script builds e @ v models (e [1, H, 8, S], v [1, H, S, D], the first N
keys real and the rest zero), as one matmul and as 512-wide slices, at
pairs of lengths, and reports each form's difference between the two
lengths and its largest error against the exact (float64) product of the
same fp16 inputs. No downloads; it runs in about a minute per compute unit.

Measured on an M1 Max, macOS 27.0, coremltools 9.0, torch 2.13, fp16.

Usage:
    python tools/repro_cpu_matmul_accumulation.py [--units CPU_ONLY,CPU_AND_GPU,CPU_AND_NE]

Exits 1 if the single matmul differs between lengths on the CPU on this
machine, so the same command tracks the behavior across macOS updates.

Requires: torch, coremltools, numpy (arm64-native Python).
"""

import argparse
import sys

import numpy as np
import torch
import coremltools as ct

UNITS = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
         "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE}
CASES = ((384, 512, 1024), (754, 1024, 2048))   # (real keys, shorter length, longer length)
H, Q, D = 4, 8, 53          # heads, query rows, head_dim + the [V | 1] column
SLICE = 512


class Single(torch.nn.Module):
    def forward(self, e, v):
        return e @ v


class Sliced(torch.nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, e, v):
        out = None
        for b in range(0, self.seq, SLICE):
            part = e[..., b:b + SLICE] @ v[:, :, b:b + SLICE]
            out = part if out is None else out + part
        return out


def data(n, seq, seed=0):
    rng = np.random.default_rng(seed)
    e = np.zeros((1, H, Q, seq), np.float32)
    v = np.zeros((1, H, seq, D), np.float32)
    e[..., :n] = rng.random((1, H, Q, n)).astype(np.float32)          # softmax numerators, in [0, 1)
    v[:, :, :n] = rng.standard_normal((1, H, n, D)).astype(np.float32)
    return e, v


def predict(module, seq, n, unit):
    e, v = data(n, seq)
    traced = torch.jit.trace(module.eval(), (torch.from_numpy(e), torch.from_numpy(v)))
    model = ct.convert(traced, inputs=[ct.TensorType(name="e", shape=e.shape), ct.TensorType(name="v", shape=v.shape)],
                       convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_units=UNITS[unit])
    feed = {"e": e, "v": v}     # kept referenced while Core ML reads it
    out = np.asarray(next(iter(model.predict(feed).values())), np.float64)
    exact = e.astype(np.float16).astype(np.float64) @ v.astype(np.float16).astype(np.float64)
    return out, float(np.abs(out - exact).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", default=",".join(UNITS), help="compute units to run, comma-separated")
    units = ap.parse_args().units.split(",")
    reproduced = False
    print(f"{'unit':10s} {'form':7s} {'keys':>5s} {'lengths':>13s} {'between lengths':>16s} {'error vs exact':>16s}")
    for unit in units:
        for n, short, long in CASES:
            for name, make in (("single", lambda s: Single()), ("sliced", lambda s: Sliced(s))):
                a, err_a = predict(make(short), short, n, unit)
                b, err_b = predict(make(long), long, n, unit)
                diff = float(np.abs(a - b).max())
                print(f"{unit:10s} {name:7s} {n:5d} {f'{short} vs {long}':>13s} {diff:16.3e} {max(err_a, err_b):16.3e}")
                reproduced |= unit == "CPU_ONLY" and name == "single" and diff > 0
    print("CPU single matmul is length-dependent past 1,024: reproduced" if reproduced
          else "not reproduced on this machine")
    return 1 if reproduced else 0


if __name__ == "__main__":
    sys.exit(main())
