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


def _walk(plan, block, ops):
    # every operation of the block and the blocks nested in it, constants
    # included (they have no device), as sidekick_coreml::compute_plan walks
    for op in block.operations:
        name = op.operator_name.split(".")[-1]
        mask = op.inputs.get("attn_mask") if name == FUSED_ATTENTION else None
        ops.append(Op(name, _device(plan, op), [o.name for o in op.outputs],
                      mask.bindings[0].name if mask is not None and mask.bindings else None))
        for nested in getattr(op, "blocks", None) or ():
            _walk(plan, nested, ops)


def _load(path, units):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=str(path), compute_units=getattr(ct.ComputeUnit, units))
    ops = []
    _walk(plan, plan.model_structure.program.functions["main"].block, ops)
    if ops and all(o.device == "none" for o in ops):
        raise GateFailure(f"compute plan for {path} assigns none of its {len(ops)} operations to a device")
    return ops


def read(compiled, units="CPU_AND_NE"):
    """Per-op devices of a compiled model, in program order, for the compute
    units Core ML is asked to use (a coremltools ComputeUnit name)."""
    try:
        return _load(compiled, units)
    except Exception as first:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / Path(compiled).name
            subprocess.run(["cp", "-c", "-R", str(compiled), str(clone)], check=False)
            if not clone.exists():
                shutil.copytree(compiled, clone)
            try:
                return _load(clone, units)
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
    gpu = sum(o.device == "GPU" for o in assigned)
    return {"ane": ane, "gpu": gpu, "cpu": len(assigned) - ane - gpu, "unassigned": len(ops) - len(assigned),
            "total": len(ops), "assigned": len(assigned), "off": off,
            "heavy_off": sorted({n for n in off if n in HEAVY}),
            "unassigned_heavy": unassigned_heavy,
            "masked_fused_attention": masked_fused_attention(ops)}


MAX_ANE_PROGRAM_WEIGHT_BYTES = 1 << 30
"""The most weight one ML program can carry and still run on the Neural
Engine: 1 GiB. Past it, Core ML runs the whole program on the CPU, with no
error and nothing in the log. Measured on an M1 Max under macOS 27.0, where
a Qwen3 backbone with 0.964 GiB of weights ran 1,526 of its 1,535
operations on the ANE, and with 1.022 GiB none of them. The true limit lies
in that bracket, and other chips or OS versions may set it elsewhere: the
compute-plan gate still judges where Core ML actually places a program."""

_CAP_OPTIONS = ("convert it in chunks under the limit (--chunks auto, D37), serve it on the GPU "
                '(compute_units = "cpu_and_gpu"), convert with --int8-embedding, or pass '
                "--ignore-ane-weight-cap to try anyway")


def weights_bytes(compiled):
    """The compiled model's weight files, in bytes; for a chain, its largest
    chunk's."""
    if is_chain(compiled):
        return max(weights_bytes(p) for p in compiled.paths)
    return sum(f.stat().st_size for f in (Path(compiled) / "weights").rglob("*") if f.is_file())


def over_cap(model, seq, size, chunk=None):
    """Why a program of `size` bytes of weights won't run on the ANE."""
    what = f"{seq} bucket" if chunk is None else f"{seq} bucket's chunk {chunk}"
    return (f"{model}'s {what} has {size / 2**30:.3f} GiB of weights, over the Neural Engine's 1 GiB "
            f"per-program limit (MAX_ANE_PROGRAM_WEIGHT_BYTES), so Core ML would run it on the CPU. "
            f"Options: {_CAP_OPTIONS}.")


def combine(summaries):
    """One bucket's summary from its chunks' (chunking.py, D37): the counts
    summed, the off-ANE ops merged, and each chunk's own summary under
    "chunks"."""
    s = {k: sum(c[k] for c in summaries) for k in ("ane", "gpu", "cpu", "unassigned", "total", "assigned")}
    off = {}
    for c in summaries:
        for name, n in c["off"].items():
            off[name] = off.get(name, 0) + n
    s.update(off=off, heavy_off=sorted({n for c in summaries for n in c["heavy_off"]}),
             unassigned_heavy=sorted({n for c in summaries for n in c["unassigned_heavy"]}),
             masked_fused_attention=[(i, *m) for i, c in enumerate(summaries) for m in c["masked_fused_attention"]],
             chunks=summaries)
    return s


def is_chain(compiled):
    """A bucket compiled as a chain of chunks (chunking.Chain)."""
    return hasattr(compiled, "paths")


def _summary(compiled, units="CPU_AND_NE"):
    if is_chain(compiled):
        return combine([summarize(read(p, units)) for p in compiled.paths])
    return summarize(read(compiled, units))


def gate(compiled, min_ane=0.8):
    """The ane_check verdict: every compute-heavy op on the ANE, at least
    `min_ane` of assigned ops on the ANE, and no fused attention that would
    drop its mask. Returns the summary; raises GateFailure. (core.run checks
    the weights against MAX_ANE_PROGRAM_WEIGHT_BYTES before this.) A chain
    is judged on its chunks' summed counts."""
    s = _summary(compiled)
    share = s["ane"] / s["assigned"] if s["assigned"] else 0.0
    s["share"], s["units"] = share, "CPU_AND_NE"
    if s["heavy_off"] or share < min_ane:
        hint = ""
        size = weights_bytes(compiled)
        if s["ane"] == 0 and size > 0.9 * MAX_ANE_PROGRAM_WEIGHT_BYTES:
            hint = (f"; its {size / 2**30:.3f} GiB of weights are near the Neural Engine's 1 GiB per-program "
                    f"limit (MAX_ANE_PROGRAM_WEIGHT_BYTES), which shows up this way. Options: {_CAP_OPTIONS}")
        raise GateFailure(f"compute plan: {s['ane']}/{s['assigned']} ops on the ANE ({share:.1%}); "
                          f"off the ANE: {s['off']}{hint}")
    if s["masked_fused_attention"]:
        raise GateFailure(f"compute plan: fused attention on the ANE reads a mask built outside its "
                          f"procedure, which the ANE ignores (D25): {s['masked_fused_attention'][:3]}")
    return s


def report(compiled, units="CPU_AND_NE"):
    """The summary without the verdict, for a model sidekick doesn't serve on
    the ANE, read for the units it is served with: a low ANE share passes,
    and an unreadable plan returns None."""
    try:
        s = _summary(compiled, units)
    except GateFailure:
        return None
    s["share"] = s["ane"] / s["assigned"] if s["assigned"] else 0.0
    s["units"] = units
    return s


def describe(s):
    note = ""
    if s.get("unassigned_heavy"):
        note = f"; not placed natively (Core ML fallback): {s['unassigned_heavy']}"
    units = "" if s.get("units", "CPU_AND_NE") == "CPU_AND_NE" else f" for {s['units']}"
    devices = "" if units == "" else f" (GPU {s['gpu']}, CPU {s['cpu']})"
    return (f"compute plan{units} {s['ane']}/{s['assigned']} ops on the ANE ({s['share']:.1%}){devices}; "
            f"off: {s['off']}{note}")


def machine():
    """(chip, macOS build) of this machine, as a recorded plan names them."""
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    return run("sysctl", "-n", "machdep.cpu.brand_string"), run("sw_vers", "-buildVersion")


def _toml_key(key):
    return key if key.replace("_", "").replace("-", "").isalnum() else '"' + key.replace('"', '\\"') + '"'


def _counts(s):
    off = ", ".join(f"{_toml_key(k)} = {v}" for k, v in sorted(s["off"].items()))
    return [f"ane = {s['ane']}", f"gpu = {s['gpu']}", f"cpu = {s['cpu']}", f"unassigned = {s['unassigned']}",
            f"total = {s['total']}", f"off_ane_ops = {{ {off} }}"]


ALTERNATIVE_UNITS = ("CPU_AND_NE", "CPU_AND_GPU")
"""The compute units a converter also records a plan for, besides the ones
the manifest serves with, so an operator who moves the model (D38) still
gets its placement reported."""


def placement_toml(plans, chip, macos_build, date, alternatives=None):
    """The installed manifest's [placement] table: the compute plan each
    bucket was converted with, read on this machine, so the daemon can report
    placement without a second compile. `plans`: {bucket: summary}, every
    summary read for the same units. The counts are sidekick_coreml's
    PlanSummary: the main function with nested blocks, constants unassigned,
    total = ane + gpu + cpu + unassigned. A chained bucket's counts are its
    chunks' sums, with each chunk's own under it (D37). `alternatives`:
    {units: {bucket: summary}}, plans read for other compute units, under
    [placement.alternatives.<units>], which daemons before 0.7.0 ignore."""
    units = {s["units"] for s in plans.values()}
    if len(units) != 1:
        raise ValueError(f"plans read for different compute units: {sorted(units)}")
    lines = ["", "[placement]", f'compute_units = "{units.pop().lower()}"', f'chip = "{chip}"',
             f'macos_build = "{macos_build}"', f'date = "{date}"']
    for seq in sorted(plans):
        lines += _bucket_tables("placement", seq, plans[seq])
    for units, by_bucket in sorted((alternatives or {}).items()):
        for seq in sorted(by_bucket):
            lines += _bucket_tables(f"placement.alternatives.{units.lower()}", seq, by_bucket[seq])
    return "\n".join(lines) + "\n"


def _bucket_tables(prefix, seq, s):
    lines = ["", f"[{prefix}.buckets.{seq}]", *_counts(s)]
    for c in s.get("chunks", ()):
        lines += ["", f"[[{prefix}.buckets.{seq}.chunks]]", *_counts(c)]
    return lines
