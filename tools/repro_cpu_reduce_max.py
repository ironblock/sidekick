"""Reproduce Core ML's CPU reduce_max defect, standalone.

=============================== THE CPU RULE ===============================
On CPU_ONLY, Core ML's reduce_max over an axis of 256 or more elements
returns max(x, 0) instead of max(x): a row whose largest element is
negative comes back as 0. reduce_min mirrors it, returning min(x, 0). Axes
of up to 255 elements are correct (measured at 128, 129, 160, 192 and
255), and so are the Neural Engine and the GPU. It looks like the
reduction's accumulator starts at 0 instead of at -inf (+inf for min) once
the axis is long enough to be split.
============================================================================

Why it matters: a numerically stable softmax written out as
exp(w - rowmax(w)) is the common case. When every score in a row is
negative, the CPU subtracts 0 instead of the row max, and if the scores are
below about -17 every exp underflows in fp16: the row sums to 0 and the
division gives NaN. laya's head attention (tools/convert_laya.py) is such a
row. Core ML's own softmax op is unaffected, and so is anything that
doesn't reduce with max or min.

Workaround, measured: take the max in blocks of at most 128 elements
(slices, max within each block, then an elementwise maximum across
blocks). The result is exact on every compute unit, and because max is
exact, it's also independent of the axis length, which keeps
bucket-invariant graphs bucket-invariant.

This script builds identity-linear -> reduce models with synthetic rows
(half with a negative max) at several axis lengths, runs them on each
compute unit, and reports wrong rows. No downloads; it runs in about a
minute. The compute unit that actually ran each reduction is read from the
compute plan, since a small model can land on the CPU even under
CPU_AND_NE.

Measured on an M1 Max, macOS 27.0, coremltools 9.0, torch 2.13, fp16.

Usage:
    python tools/repro_cpu_reduce_max.py              # run the cases, print a table
    python tools/repro_cpu_reduce_max.py --check M    # list a model's risky reductions
                                                      # (.mlmodelc or .mlpackage)

The plain run exits 1 if the defect reproduces on this machine, so the
same command tracks it across macOS updates. `--check` exits 1 if the
model has a reduce_max or reduce_min over 256 or more elements that the
compute plan places on the CPU, or could (no device assigned). It reads the
program and the compute plan and never runs the model.

Requires: torch, coremltools, numpy (arm64-native Python).
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

UNITS = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
         "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU}
LENGTHS = (128, 192, 255, 256, 512, 1024)
ROWS = 256
BLOCK = 128
RISKY = 256


class Reduce(torch.nn.Module):
    """Identity linear (keeps the graph off the elementwise-only CPU path),
    then a reduction over the last axis."""

    def __init__(self, n, how):
        super().__init__()
        self.lin = torch.nn.Linear(n, n, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(n))
        self.how, self.n = how, n

    def forward(self, x):
        x = self.lin(x)
        if self.how == "max":
            return x.max(dim=-1, keepdim=True).values
        if self.how == "min":
            return x.min(dim=-1, keepdim=True).values
        # the workaround: max within blocks of BLOCK, then across blocks
        m = None
        for b in range(0, self.n, BLOCK):
            mb = x[..., b:b + BLOCK].max(dim=-1, keepdim=True).values
            m = mb if m is None else torch.maximum(m, mb)
        return m


def rows(n, seed=0):
    """ROWS rows of length n: small noise around a per-row offset, so about
    half the rows lie entirely below 0 (and half entirely above)."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 3, (1, ROWS, n)) + rng.normal(0, 200, (1, ROWS, 1))).astype(np.float32)


def convert(module, n, workdir):
    x = torch.from_numpy(rows(n))
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), (x,))
    ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=(1, ROWS, n), dtype=np.float32)],
                    outputs=[ct.TensorType(name="y")], convert_to="mlprogram",
                    minimum_deployment_target=ct.target.macOS15)
    path = Path(workdir) / f"reduce_{module.how}_{n}.mlpackage"
    ml.save(str(path))
    return path


def reduce_devices(model):
    """Device of each reduce op in the compute plan, by op name."""
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(),
                                        compute_units=model.compute_unit)
    out = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name.split(".")[-1] in ("reduce_max", "reduce_min"):
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__ if usage else "none"
            out.append(dev.replace("ML", "").replace("ComputeDevice", ""))
    return sorted(set(out)) or ["-"]


def run_cases():
    print(f"{'reduction':10s} {'axis':>5s} {'unit':12s} {'ran on':14s} wrong rows (of {ROWS})")
    reproduced = False
    with tempfile.TemporaryDirectory() as workdir:
        for how in ("max", "min", "blocked"):
            for n in LENGTHS:
                if how != "max" and n not in (255, 256, 512):
                    continue
                x = rows(n)
                ref = x.max(axis=-1) if how in ("max", "blocked") else x.min(axis=-1)
                path = convert(Reduce(n, how), n, workdir)
                for label, unit in UNITS.items():
                    m = ct.models.MLModel(str(path), compute_units=unit)
                    y = np.asarray(m.predict({"x": x})["y"], dtype=np.float64).reshape(ref.shape)
                    finite = np.isfinite(y)
                    wrong = ~finite | (np.abs(y - ref) > 0.5)
                    note = ""
                    if wrong.any():
                        r = np.argmax(wrong.reshape(-1))
                        note = f"  e.g. got {y.reshape(-1)[r]:.1f}, expected {ref.reshape(-1)[r]:.1f}"
                        if how != "blocked" and label == "CPU_ONLY":
                            reproduced = True
                    ran = ",".join(reduce_devices(m))
                    print(f"{how:10s} {n:5d} {label:12s} {ran:14s} {int(wrong.sum()):4d}{note}")
    print("\nCPU reduce_max/reduce_min defect reproduces" if reproduced
          else "\nCPU reduce_max/reduce_min defect does not reproduce")
    return 1 if reproduced else 0


SHAPE = re.compile(r"tensor<\w+, \[([0-9, ]*)\]> (\w+)\b")
AXES = re.compile(r"(\w+) = const\(\)\[name = string\(\"[^\"]*\"\), val = tensor<int32, \[\d+\]>\(\[([-0-9, ]*)\]\)\]")
REDUCE = re.compile(r"= (reduce_max|reduce_min)\(([^)]*)\)\[name = string\(\"([^\"]+)\"\)")


def check(model_path):
    """Risky reductions: reduce_max/reduce_min over >= RISKY elements that
    the compute plan places on the CPU, or leaves unassigned. Shapes come
    from the compiled program text (model.mil), devices from the plan."""
    from coremltools.models.compute_plan import MLComputePlan
    path = Path(model_path)
    if path.suffix == ".mlpackage":
        compiled = Path(ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY).get_compiled_model_path())
    else:
        compiled = path
    mil = (compiled / "model.mil").read_text()
    shapes = {name: [int(d) for d in dims.split(",") if d.strip()] for dims, name in SHAPE.findall(mil)}
    axes_of = {name: [int(a) for a in vals.split(",") if a.strip()] for name, vals in AXES.findall(mil)}
    lengths = {}
    for kind, args, op_name in REDUCE.findall(mil):
        kw = dict(a.strip().split(" = ", 1) for a in args.split(",") if " = " in a)
        shape, axes = shapes.get(kw.get("x")), axes_of.get(kw.get("axes"))
        n = int(np.prod([shape[a] for a in axes])) if shape and axes else None
        lengths[kw.get("x")] = (kind, op_name, n)
    if not lengths:
        print("no reduce_max or reduce_min ops")
        return 0
    risky = 0
    for label in ("CPU_ONLY", "CPU_AND_NE"):
        try:
            plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=UNITS[label])
        except Exception as e:  # noqa: BLE001
            print(f"{label}: compute plan unreadable ({e})")
            return 2
        for op in plan.model_structure.program.functions["main"].block.operations:
            if op.operator_name.split(".")[-1] not in ("reduce_max", "reduce_min"):
                continue
            x = op.inputs["x"].bindings[0].name
            kind, op_name, n = lengths.get(x, ("?", "?", None))
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__ if usage else "none"
            dev = dev.replace("ML", "").replace("ComputeDevice", "")
            bad = (n is None or n >= RISKY) and dev not in ("NeuralEngine", "GPU")
            risky += bad
            print(f"{label}: {kind} {op_name} over {n if n else '?'} elements on {dev}"
                  + ("  <- affected" if bad else ""))
    if risky:
        print(f"{risky} reduction(s) over >= {RISKY} elements (or of unknown length) can run on the CPU; "
              f"take the max in blocks of {BLOCK} (see the module docstring)")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", metavar="MODEL", help="list a compiled model's CPU-placed reduce_max/min ops")
    args = ap.parse_args()
    sys.exit(check(args.check) if args.check else run_cases())


if __name__ == "__main__":
    main()
