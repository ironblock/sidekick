"""plan_default.py <mlmodelc> [units]: Core ML's device per operation for a
compiled program's default function (coremltools' MLComputePlan reads only
that one). Reading a plan compiles the program, so use it on small probes."""
import collections
import json
import sys

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

path = sys.argv[1]
units = getattr(ct.ComputeUnit, sys.argv[2] if len(sys.argv) > 2 else "CPU_AND_NE")
plan = MLComputePlan.load_from_path(path=path, compute_units=units)
prog = plan.model_structure.program
fn = prog.functions[next(iter(prog.functions))] if "main" not in prog.functions else prog.functions["main"]
count = collections.Counter()
off = collections.Counter()


def walk(block):
    for op in block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            count["none"] += 1
            continue
        dev = type(u.preferred_compute_device).__name__
        count[dev] += 1
        if "NeuralEngine" not in dev:
            off[f"{op.operator_name}@{dev}"] += 1
        for b in op.blocks:
            walk(b)


walk(fn.block)
print("RESULT", json.dumps({"functions": list(prog.functions), "devices": count, "off_ane": off}))
