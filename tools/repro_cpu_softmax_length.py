"""Reproduce Core ML's length-dependent CPU softmax in attention, standalone.

============================== THE CPU RULE ===============================
On CPU_ONLY, attention written as matmul -> softmax, the softmax reading the
score matmul's output in the same program, gives different probabilities for
the same real keys at different key lengths, below 1,024 too. With 106 real
keys, the rest masked at -30000, the 512- and 1,024-key versions differ by
up to 1.5e-3 in a probability (the same row of real scores; the masked keys
contribute exactly zero). The softmax op alone, fed the same scores as an
input, is bit-identical across lengths: the difference appears only when it
follows the in-graph score matmul.

The softmax computed with the value matmul, exp(w - rowmax(w)) followed by
one matmul against [V | 1] (sidekick_convert.techniques.attention
.matmul_softmax), is bit-identical across 512 and 1,024 keys, at the same
accuracy against the exact result. Up to 1,024, Core ML's CPU matmul doesn't
depend on the length (tools/repro_cpu_matmul_accumulation.py).

On the GPU both forms are length-independent. There the matmul form's
attention output was less accurate in this synthetic case (9.1e-3 against
the exact result, 5.5e-3 with Core ML's softmax), though on agent-jev's
first two layers the GPU's error was unchanged (5.34e-3 and 5.32e-3), so a
model served on the GPU checks its grade after switching.
===========================================================================

Why it matters: a classifier converted with one artifact per sequence
bucket is checked for giving the same output for the same input in every
bucket that holds it (the parity suite's bucket-invariance gate, exact on
the CPU). agent-jev (a Qwen3 decoder converted through transformers' sdpa
path, which becomes matmul -> softmax -> matmul) moved by up to 0.019 in
probability between its 512 and 1,024 buckets on the CPU. Lumma-fev, whose
converter uses the matmul softmax (tools/convert_fev.py, constraint D), is
bit-identical there. agent-jev's converter now uses it too.

This script builds q k^T * scale + mask -> softmax models, and the same with
the value matmul and in the matmul-softmax form, at pairs of key lengths,
with identical real data and the rest masked. It reports each form's
difference between the two lengths and its error against the exact
(float64) result. No downloads; it runs in about two minutes per compute
unit.

Measured on an M1 Max, macOS 27.0, coremltools 9.0, torch 2.13, fp16.

Usage:
    python tools/repro_cpu_softmax_length.py [--units CPU_ONLY,CPU_AND_GPU] [--lengths 512,1024]

Exits 1 if the softmax that follows the score matmul differs between
lengths on the CPU on this machine, so the same command tracks the behavior
across macOS updates.

Requires: torch, coremltools, numpy (arm64-native Python).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HEADS, HEAD_DIM, REAL = 16, 128, 106
MASK = -30000.0


def blocked_max(w, n, block=128):
    m = None
    for b in range(0, n, block):
        mb = w[..., b:b + block].max(dim=-1, keepdim=True).values
        m = mb if m is None else torch.maximum(m, mb)
    return m


class Attention(torch.nn.Module):
    """form: "softmax" (probabilities), "softmax+pv" (attention output), "scores-in"
    (softmax of a scores input), or "matmul" / "matmul+pv" (the matmul softmax)."""

    def __init__(self, form, seq):
        super().__init__()
        self.form, self.seq = form, seq

    def forward(self, q, k, v, add):
        if self.form == "scores-in":
            return torch.softmax(q, -1)
        w = (q @ k.transpose(-1, -2)) * HEAD_DIM ** -0.5 + add
        if self.form.startswith("matmul"):
            e = torch.exp(w - blocked_max(w, self.seq))
            o = e @ torch.cat([v, torch.ones_like(v[..., :1])], -1)
            return o[..., :-1] / o[..., -1:] if self.form == "matmul+pv" else e / o[..., -1:]
        p = torch.softmax(w, -1)
        return p @ v if self.form == "softmax+pv" else p


def inputs(seq, rng_seed=0):
    """The same real q/k/v for every length; keys past REAL masked, and each
    query sees the real keys up to itself (and itself), as a causal tree does."""
    rng = np.random.default_rng(rng_seed)
    q = np.zeros((1, HEADS, seq, HEAD_DIM), np.float32)
    k, v = np.zeros_like(q), np.zeros_like(q)
    q[:, :, :REAL] = rng.standard_normal((1, HEADS, REAL, HEAD_DIM)) * 2
    k[:, :, :REAL] = rng.standard_normal((1, HEADS, REAL, HEAD_DIM)) * 2
    v[:, :, :REAL] = rng.standard_normal((1, HEADS, REAL, HEAD_DIM))
    i = np.arange(seq)
    add = np.where((i[None, :] <= i[:, None]) & (i[None, :] < REAL), 0.0, MASK).astype(np.float32)
    add[i, i] = 0.0
    return {"q": q, "k": k, "v": v, "add": add[None, None]}


def run(form, seq, units):
    feeds = inputs(seq)
    if form == "scores-in":
        w = (feeds["q"] @ feeds["k"].transpose(0, 1, 3, 2)) * HEAD_DIM ** -0.5 + feeds["add"]
        feeds = {"q": w.astype(np.float32), "k": feeds["k"], "v": feeds["v"], "add": feeds["add"]}
    model = Attention(form, seq).eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.from_numpy(x) for x in feeds.values()))
        exact = model(*(torch.from_numpy(x).double() for x in feeds.values())).numpy()
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.float32) for n, x in feeds.items()],
                    outputs=[ct.TensorType(name="out")], convert_to="mlprogram",
                    minimum_deployment_target=ct.target.macOS15)
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "m.mlpackage"
        ml.save(str(pkg))
        out = ct.models.MLModel(str(pkg), compute_units=getattr(ct.ComputeUnit, units)).predict(feeds)["out"]
    cols = HEAD_DIM if form.endswith("pv") else REAL
    real = (slice(None), slice(None), slice(0, REAL), slice(0, cols))
    return out[real].astype(np.float64), exact[real]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_GPU")
    ap.add_argument("--lengths", default="512,1024")
    args = ap.parse_args()
    a_len, b_len = (int(x) for x in args.lengths.split(","))
    torch.set_grad_enabled(False)
    differs = False
    print(f"{REAL} real keys of {HEADS} heads x {HEAD_DIM}, the rest masked at {MASK:g}; lengths {a_len} and {b_len}")
    for units in args.units.split(","):
        for form in ("softmax", "softmax+pv", "scores-in", "matmul", "matmul+pv"):
            a, ea = run(form, a_len, units)
            b, eb = run(form, b_len, units)
            d = float(np.abs(a - b).max())
            err = max(float(np.abs(a - ea).max()), float(np.abs(b - eb).max()))
            print(f"  {units:12s} {form:11s} between lengths {d:.2e}  vs exact {err:.2e}", flush=True)
            if units == "CPU_ONLY" and form == "softmax" and d > 0:
                differs = True
    print("CPU softmax after the score matmul is length-dependent here" if differs else
          "CPU softmax after the score matmul is length-independent here")
    sys.exit(1 if differs else 0)


if __name__ == "__main__":
    main()
