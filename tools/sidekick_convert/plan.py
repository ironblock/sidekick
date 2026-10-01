"""Core ML's compute plan: which device each operation is assigned to,
read without running the model (docs/DECISIONS.md D24).

A plan can be unreadable: it fails to load, or loads with every operation
unassigned. Core ML's cache of compiled bundles
(~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache) can hold a
broken entry for an artifact's path, and plans for that path then come back
empty until the entry is gone. `read()` retries once from an APFS clone at a
new path, as the parity suite does (D26), and fails if that is empty too.
"""

import dataclasses
import shutil
import subprocess
import tempfile
from pathlib import Path

from .core import FUSED_ATTENTION, GateFailure

HEAVY = ("linear", "matmul", "conv")


@dataclasses.dataclass
class Op:
    name: str          # "linear", not "ios18.linear"
    device: str        # "NeuralEngine", "CPU", "GPU", or "none" (unassigned)
    outputs: list
    attn_mask: str = None


def _device(plan, op):
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is None:
        return "none"
    return type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")


def _load(path):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    ops = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        name = op.operator_name.split(".")[-1]
        if name == "const":
            continue
        mask = op.inputs.get("attn_mask") if name == FUSED_ATTENTION else None
        ops.append(Op(name, _device(plan, op), [o.name for o in op.outputs],
                      mask.bindings[0].name if mask is not None and mask.bindings else None))
    if ops and all(o.device == "none" for o in ops):
        raise GateFailure(f"compute plan for {path} assigns none of its {len(ops)} operations to a device")
    return ops


def read(compiled):
    """Per-op devices of a compiled model, in program order, for CPU_AND_NE."""
    try:
        return _load(compiled)
    except Exception as first:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / Path(compiled).name
            subprocess.run(["cp", "-c", "-R", str(compiled), str(clone)], check=False)
            if not clone.exists():
                shutil.copytree(compiled, clone)
            try:
                return _load(clone)
            except Exception as second:
                raise GateFailure(
                    f"compute plan unreadable ({first}); from a clone at a new path too ({second}). "
                    "Core ML's bundle cache (~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache) "
                    "may hold a broken entry") from second


def masked_fused_attention(ops):
    """Fused attention ops that run natively on the ANE with a mask computed
    outside their ANE procedure: the ANE ignores such a mask (D25). Returns
    (index, mask producer) pairs. A mask from a model input, a CPU op, or an
    ANE op with a non-ANE op between it and the attention counts as outside."""
    produced_by = {name: i for i, op in enumerate(ops) for name in op.outputs}
    bad = []
    for i, op in enumerate(ops):
        if op.name != FUSED_ATTENTION or op.device != "NeuralEngine" or op.attn_mask is None:
            continue
        j = produced_by.get(op.attn_mask)
        if j is None:
            bad.append((i, "model input"))
            continue
        between = any(o.device not in ("NeuralEngine", "none") for o in ops[j + 1:i])
        if ops[j].device != "NeuralEngine" or between:
            bad.append((i, f"{ops[j].name}@{ops[j].device}"))
    return bad


def summarize(ops):
    assigned = [o for o in ops if o.device != "none"]
    ane = sum(o.device == "NeuralEngine" for o in assigned)
    off = {}
    for o in assigned:
        if o.device != "NeuralEngine":
            off[o.name] = off.get(o.name, 0) + 1
    unassigned_heavy = sorted({o.name for o in ops if o.device == "none" and o.name in HEAVY + (FUSED_ATTENTION,)})
    return {"ane": ane, "assigned": len(assigned), "off": off,
            "heavy_off": sorted({n for n in off if n in HEAVY}),
            "unassigned_heavy": unassigned_heavy,
            "masked_fused_attention": masked_fused_attention(ops)}


def gate(compiled, min_ane=0.8):
    """The ane_check verdict: every compute-heavy op on the ANE, at least
    `min_ane` of assigned ops on the ANE, and no fused attention that would
    drop its mask. Returns the summary; raises GateFailure."""
    s = summarize(read(compiled))
    share = s["ane"] / s["assigned"] if s["assigned"] else 0.0
    s["share"] = share
    if s["heavy_off"] or share < min_ane:
        raise GateFailure(f"compute plan: {s['ane']}/{s['assigned']} ops on the ANE ({share:.1%}); "
                          f"off the ANE: {s['off']}")
    if s["masked_fused_attention"]:
        raise GateFailure(f"compute plan: fused attention on the ANE reads a mask built outside its "
                          f"procedure, which the ANE ignores (D25): {s['masked_fused_attention'][:3]}")
    return s


def describe(s):
    note = ""
    if s.get("unassigned_heavy"):
        note = f"; not placed natively (Core ML fallback): {s['unassigned_heavy']}"
    return (f"compute plan {s['ane']}/{s['assigned']} ops on the ANE ({s['share']:.1%}); "
            f"off: {s['off']}{note}")
