"""Reproduce Core ML's bucket-dependent attention scores on an Intel Mac's
CPU, standalone.

============================== THE INTEL RULE =============================
On an Intel Mac's CPU (CPU_ONLY), the attention-score matmul q k^T gives
different values for the same real (query, key) pair at different key
lengths, but only for keys near the end of the shorter length. With 127
real keys and query rows, the rest masked, the 128-key version differs from
the 256- and 512-key versions by one fp16 ulp (9.8e-4 at scores up to 12)
in 4 of 193,548 real scores, all in key columns 123 to 126 (this script's
data; other data put them in columns 120 to 124); the 256- and 512-key
versions agree. With 86 real keys every score is bit-identical. The softmax
and a linear are bit-identical across lengths in the same test, and the
probabilities @ V matmul (a contraction over the keys) moves one value by
7.6e-6.

The affected columns are the last 8 of the 128-wide output, which fits a
matmul kernel that computes the remainder of its column tiling (128 =
5 x 24 + 8) differently from a full tile: at 256 keys those columns sit
inside a full tile, and the remainder moves to 240-255, where a 255-key
input differs from its 512-key version instead. Computing the scores as
(k q^T)^T gives the same pattern: coremltools folds the transposes into
the same kernel. Both values are correct fp16 roundings; this is a
determinism property, not an accuracy one.

On Apple silicon (CPU and GPU) every op is bit-identical across lengths.
On the Intel Mac's GPU (AMD, Metal) none of this holds: q k^T differs
across all columns (up to 1.95e-3), and the softmax, probabilities @ V and a
linear differ too, by up to 2e-3; a whole BERT is not bucket-invariant
there either (nlptown-sentiment: Δp up to 2.6e-4), with Core ML logging
"MPSGraph doesn't support padded tensors on Intel macs". The parity suite's
GPU gate (Δp 1e-3) admits it.
===========================================================================

Why it matters: a classifier with one artifact per sequence bucket is
checked for giving the same output for the same input in every bucket that
holds it (the parity suite's bucket-invariance gate, exact on the CPU). On
an Intel Xeon W-3235, nlptown-sentiment's Core ML CPU path moves by up to
Δp 1.9e-4 between buckets, only for inputs that nearly fill their bucket
(127 and 128 tokens against 256 and 512, 255 tokens against 512); the
other 83 of 88 comparisons are bit-identical. bge-small-en-v1.5 shows the
same effect inside the embedder gate (bucket cosine 0.99999910 against
0.999999).

This script builds q k^T (and controls) with identical real data at key
lengths 128, 256 and 512, fp16, as sidekick's converters build them, and
compares each op's real region across lengths. On a Mac with coremltools
it runs the programs itself. An Intel Mac without coremltools runs them
with tools/coreml_run.swift: build them elsewhere with --emit, copy the
directory over, run the harness on each program, copy the outputs back and
--compare them.

Measured: Intel Xeon W-3235 (macOS 26.6.2) and Apple M1 Max (macOS 27.0),
coremltools 9.0, torch 2.13.

Usage:
    python tools/repro_cpu_matmul_tail.py [--real 127] [--units CPU_ONLY,CPU_AND_GPU]
    python tools/repro_cpu_matmul_tail.py --emit DIR [--real 127]
        # then on the other Mac, for each program:
        #   swift tools/coreml_run.swift DIR/<op>_<S>.mlmodelc cpu DIR/<op>_<S>.feeds.json \\
        #       DIR/<op>_<S>.cpu.out.json 2
    python tools/repro_cpu_matmul_tail.py --compare DIR [--real 127]

Exits 1 if q k^T differs between lengths on the CPU, so the same command
tracks the behavior across macOS updates.

Requires: torch, coremltools, numpy (to build); numpy (to compare).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

H, D, HID = 12, 64, 768
LENGTHS = (128, 256, 512)
MASK = -30000.0


def data(real):
    rng = np.random.default_rng(74)
    return {"q": rng.standard_normal((1, H, real, D)).astype(np.float32) * 0.5,
            "k": rng.standard_normal((1, H, 512, D)).astype(np.float32) * 0.5,
            "v": rng.standard_normal((1, H, 512, D)).astype(np.float32),
            "x": rng.standard_normal((1, 512, HID)).astype(np.float32),
            "w": (rng.standard_normal((HID, HID)) / np.sqrt(HID)).astype(np.float32),
            "scores": rng.standard_normal((1, H, real, real)).astype(np.float32) * 3}


def ops():
    import torch

    class QK(torch.nn.Module):            # output width = the key length
        def forward(self, q, k):
            return q @ k.transpose(-1, -2)

    class Softmax(torch.nn.Module):
        def forward(self, scores, mask):
            return torch.softmax(scores + mask, dim=-1)

    class PV(torch.nn.Module):            # contraction over the keys
        def forward(self, p, v):
            return p @ v

    class Linear(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.lin = torch.nn.Linear(HID, HID, bias=False)
            self.lin.weight.data = torch.from_numpy(w)

        def forward(self, x):
            return self.lin(x)

    return {"qk": QK, "softmax": Softmax, "pv": PV, "linear": Linear}


def feeds(op, S, d, real):
    m = np.full((1, 1, 1, S), MASK, np.float32)
    m[..., :real] = 0
    if op == "qk":
        return {"q": d["q"], "k": d["k"][:, :, :S].copy()}
    if op == "softmax":
        s = np.zeros((1, H, real, S), np.float32)
        s[..., :real] = d["scores"]
        return {"scores": s, "mask": m}
    if op == "pv":
        e = np.exp(d["scores"] - d["scores"].max(-1, keepdims=True))
        p = np.zeros((1, H, real, S), np.float32)
        p[..., :real] = e / e.sum(-1, keepdims=True)
        return {"p": p, "v": d["v"][:, :, :S].copy()}
    return {"x": d["x"][:, :S].copy()}


def real_region(op, y, real):
    """The outputs that depend only on real data: scores against real keys,
    and the rows of real positions."""
    return y[..., :real] if op in ("qk", "softmax") else (y[:, :real] if op == "linear" else y)


def convert(op, S, d, real):
    import coremltools as ct
    import torch
    cls = ops()[op]
    mod = (cls(d["w"]) if op == "linear" else cls()).eval()
    f = feeds(op, S, d, real)
    with torch.no_grad():
        traced = torch.jit.trace(mod, tuple(torch.from_numpy(a) for a in f.values()))
    return ct.convert(traced, inputs=[ct.TensorType(name=n, shape=a.shape) for n, a in f.items()],
                      outputs=[ct.TensorType(name="y")], convert_to="mlprogram",
                      minimum_deployment_target=ct.target.macOS15, skip_model_load=True), f


def emit(out, real):
    out.mkdir(parents=True, exist_ok=True)
    d = data(real)
    for op in ops():
        for S in LENGTHS:
            ml, f = convert(op, S, d, real)
            pkg = out / f"{op}_{S}.mlpackage"
            ml.save(str(pkg))
            subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out)], check=True,
                           stdout=subprocess.DEVNULL)
            shutil.rmtree(pkg)
            (out / f"{op}_{S}.feeds.json").write_text(json.dumps([{n: {
                "shape": list(a.shape), "dtype": "float32", "data": a.ravel().astype(float).tolist()}
                for n, a in f.items()}]))
    print(f"wrote {len(ops()) * len(LENGTHS)} programs and feeds to {out}")


def report(results, real):
    """results: {units: {op: {S: array}}}. Returns True if q k^T differs on the CPU."""
    differs = False
    for units, by_op in results.items():
        print(f"== {units}")
        for op, by_len in by_op.items():
            base = real_region(op, by_len[LENGTHS[0]], real)
            cells = []
            for S in LENGTHS[1:]:
                y = real_region(op, by_len[S], real)
                delta = np.abs(y - base)
                cols = np.where(delta.reshape(-1, delta.shape[-1]).max(0) > 0)[0] if op == "qk" else []
                where = f" (key columns {cols.min()}-{cols.max()})" if len(cols) else ""
                cells.append(f"{S}: {delta.max():.3e}, {int((delta > 0).sum())} differ{where}")
                if op == "qk" and units.lower().startswith("cpu") and delta.max() > 0:
                    differs = True
            print(f"  {op:8s} vs {LENGTHS[0]} keys: " + "; ".join(cells))
    return differs


def out_shape(op, S, real):
    return {"qk": (1, H, real, S), "softmax": (1, H, real, S), "pv": (1, H, real, D), "linear": (1, S, HID)}[op]


def compare(d, real):
    results = {}
    for f in sorted(d.glob("*.*.out.json")):
        op, S = f.name.split(".")[0].rsplit("_", 1)
        units = f.name.split(".")[1]
        y = np.array(json.loads(f.read_text())["outputs"][0]["y"]).reshape(out_shape(op, int(S), real))
        results.setdefault(units, {}).setdefault(op, {})[int(S)] = y
    return report(results, real)


def run_here(units_list, real):
    import coremltools as ct
    d = data(real)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for op in ops():
            for S in LENGTHS:
                ml, f = convert(op, S, d, real)
                pkg = Path(tmp) / f"{op}_{S}.mlpackage"
                ml.save(str(pkg))
                for units in units_list:
                    model = ct.models.MLModel(str(pkg), compute_units=getattr(ct.ComputeUnit, units))
                    y = model.predict(f)["y"]
                    results.setdefault(units, {}).setdefault(op, {})[S] = np.asarray(y, np.float64)
    return report(results, real)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--real", type=int, default=127, help="real keys (and query rows); the rest are masked")
    p.add_argument("--units", default="CPU_ONLY,CPU_AND_GPU")
    p.add_argument("--emit", type=Path, help="write the programs and feeds for tools/coreml_run.swift")
    p.add_argument("--compare", type=Path, help="compare tools/coreml_run.swift outputs in this directory")
    args = p.parse_args()
    if args.emit:
        emit(args.emit, args.real)
        return 0
    differs = compare(args.compare, args.real) if args.compare else run_here(args.units.split(","), args.real)
    print("q k^T on the CPU " + ("DIFFERS between key lengths" if differs else "is the same at every key length"))
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
