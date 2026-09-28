"""Reproduce Core ML's fused-attention mask bugs, standalone.

coremltools lowers PyTorch's F.scaled_dot_product_attention to Core ML's
fused `scaled_dot_product_attention` op at the iOS18/macOS15 opset and later
(unless `scale=` is passed). That op has one Neural Engine defect and two CPU
defects. This script builds a single attention layer with random weights for
each case (no downloads; it runs in seconds) and measures it on CPU_AND_NE
and CPU_ONLY.

=============================== THE ANE RULE ===============================
The Neural Engine's native fused attention IGNORES ITS attn_mask WHEN THE
MASK IS AN INPUT OF THE ANE PROCEDURE THAT RUNS THE ATTENTION, i.e. when no
op inside that procedure computes it. The mask may be:
  - a model input,
  - the output of an op placed on the CPU (ModernBERT: its mask plumbing is
    built before the CPU-only embedding gather, so it lands on the CPU), or
  - the output of an earlier ANE procedure (a CPU op in between splits the
    ANE work).
The output then equals unmasked attention exactly, whatever the mask holds.
Tried with mask shapes (1,1,S,S), (1,H,S,S) and (1,1,1,S); fp16 and fp32
masks; fills -inf, -30000, -10000 and -100; head dims 32 and 64; 6 and 12
heads; sequence lengths 128, 256 and 512; with and without q/k/v biases; and
at the macOS15 and iOS26 opsets.

A mask computed by ops inside the attention's own ANE procedure is honoured.

Other fused-attention graphs work only through a fallback. Examples are
bge-small (whose mask is also built on the CPU) and any mask passed through
one extra ANE op such as a clamp. For these, Core ML evidently builds no
native ANE plan: no ANE bundle is cached, the compute plan reports the
attention op with no device, and the mask works. What triggers the fallback
isn't known. At the iOS26 opset those same models fail to load on
CPU_AND_NE ("Failed to build the model execution plan ... error code: -14"),
so the fallback is what keeps them correct at the macOS15 opset.

In a compute plan, the rule reads: `scaled_dot_product_attention` is
assigned to the Neural Engine, and its `attn_mask` comes from a model input
or from an op on the CPU. `--check` applies it to a compiled model. At run
time, pad invariance catches it (same text, different pad contents, the mask
hides the pads, so outputs must match). sidekick's `ane_check` gates on pad
invariance.

Workarounds, all measured: explicit matmul -> softmax -> matmul attention
(transformers' attn_implementation="eager"), or passing `scale=` to
F.scaled_dot_product_attention, which makes coremltools emit explicit ops.
============================================================================

The two CPU defects (CPU_ONLY runs the same fused op on the CPU):
  - A query row whose keys are ALL masked returns NaN when
    |fill| x sqrt(head_dim) > 65504, the fp16 maximum (measured at head dims
    32, 64 and 128). With head dim 64 the threshold is ~8188, so the usual
    -30000 (or -inf) gives NaN, while explicit attention gives a finite
    uniform row. Sliding-window models hit this whenever a pad query's whole
    window is padding, and the NaN then spreads to every token in the next
    layer.
  - q/k/v taken straight from a packed projection with
    `.view(B, S, 3, H, D).transpose(3, 1).unbind(dim=2)` (a split along a
    middle axis) and fed directly to the op give wrong output, far from the
    fp32 reference though still pad-invariant. `.permute(2, 0, 3, 1, 4)`
    (split along the leading axis) is correct, as is explicit attention on
    the same layout. ModernBERT escapes this because RoPE runs between the
    split and the attention.

Measured on an M1 Max, macOS 27.0, coremltools 9.0, torch 2.13, fp16,
seq 128, 12 heads x 64. Metrics are NaN-safe: a non-finite output is
reported as such, never folded into a min() or a mean.

Usage:
    python tools/repro_sdpa_mask.py              # run every case, print a table
    python tools/repro_sdpa_mask.py --check M    # apply the ANE rule to a model
                                                 # (.mlmodelc or .mlpackage)

`--check` exits 1 if any attention op matches the rule, and 2 if the compute
plan is unreadable. It reads the compute plan and never runs the model. It
detects a mask from a model input or a CPU op directly; an ANE-built mask in
another procedure is inferred from a non-ANE op sitting between the mask and
the attention in program order.

An unreadable plan is one that fails to load, or loads with no operation
assigned to any device. It must not pass: an attention op with no device
normally means Core ML's fallback, so an all-unassigned plan would clear
every op. Core ML's cache of compiled bundles
(~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache) can hold a broken
entry for an artifact's path. Plans for that path then come back
all-unassigned, or fail with "internal failure", until the entry is gone. A
copy at a new path reads normally (`cp -c -R <model> <new path>` makes an
APFS clone).

Requires: torch, coremltools >= 8 (arm64-native Python), Apple Silicon.
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

SEQ, HEADS, HEAD_DIM = 128, 12, 64
DIM = HEADS * HEAD_DIM
VOCAB = 1000
N_REAL = 9          # real tokens; the other positions are padding
MASKED = -30000.0   # fp16-safe additive mask value, as sidekick's converters use
SDPA = "scaled_dot_product_attention"

warnings.filterwarnings("ignore", category=RuntimeWarning)       # coremltools' value-range inference
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


def _randn(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _linear(seed, n_out=DIM):
    lin = torch.nn.Linear(DIM, n_out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(_randn(n_out, DIM, seed=seed) / DIM ** 0.5)
    return lin


def split_heads(t):
    return t.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)


def merge_heads(t):
    return t.transpose(1, 2).reshape(1, SEQ, DIM)


def key_padding_mask(keypad):
    """(1, S) float, 1 at pads -> additive (1, 1, S, S)."""
    return (keypad * MASKED)[:, None, None, :].expand(1, 1, SEQ, SEQ)


# ---------------------------------------------------------------------------
# Models. Each takes numpy-able inputs in a fixed order and returns (1, S, DIM).

class Attention(torch.nn.Module):
    """q/k/v linears -> attention -> output linear. `attention` is "fused"
    (coremltools emits the fused op), "explicit" (matmul -> softmax ->
    matmul), or "scale" (F.scaled_dot_product_attention with scale=, which
    coremltools lowers to explicit ops)."""

    def __init__(self, attention="fused"):
        super().__init__()
        self.q, self.k, self.v, self.o = (_linear(s) for s in (1, 2, 3, 4))
        self.attention = attention

    def attend(self, h, mask):
        q, k, v = split_heads(self.q(h)), split_heads(self.k(h)), split_heads(self.v(h))
        if self.attention == "explicit":
            a = torch.softmax(q @ k.transpose(-1, -2) * HEAD_DIM ** -0.5 + mask, dim=-1) @ v
        elif self.attention == "scale":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=HEAD_DIM ** -0.5)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(merge_heads(a))


class MaskInput(Attention):
    """The additive mask is a model input and feeds the attention directly."""
    inputs = ("x", "mask")

    def forward(self, x, mask):
        return self.attend(x, mask)


class MaskInputClamped(Attention):
    """As MaskInput, plus one clamp on the mask right before the attention."""
    inputs = ("x", "mask")

    def forward(self, x, mask):
        return self.attend(x, mask.clamp(min=MASKED))


class MaskBuiltOnCpu(Attention):
    """ModernBERT's op order: the mask is built from the int32 attention mask
    first, then the embedding gather (CPU-only) runs, so the partitioner puts
    the mask plumbing on the CPU with the gather."""
    inputs = ("ids", "attention_mask")

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        with torch.no_grad():
            self.emb.weight.copy_(_randn(VOCAB, DIM, seed=5))
        self.norm = torch.nn.LayerNorm(DIM)

    def forward(self, ids, attention_mask):
        mask = key_padding_mask(1.0 - attention_mask.to(torch.float32)).contiguous()
        return self.attend(self.norm(self.emb(ids.long())), mask)


class MaskFromOtherAneProcedure(Attention):
    """The mask is built by ANE ops, but a CPU-only gather between it and the
    attention splits the ANE work into two procedures."""
    inputs = ("x", "keypad", "ids")

    def __init__(self):
        super().__init__()
        self.pre = _linear(6)
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        with torch.no_grad():
            self.emb.weight.copy_(_randn(VOCAB, DIM, seed=7))

    def forward(self, x, keypad, ids):
        mask = key_padding_mask(torch.relu(keypad))
        h = self.pre(x) + self.emb(ids.long())
        return self.attend(h, mask)


class MaskInsideProcedure(torch.nn.Module):
    """The additive mask is built by ANE ops right before the attention, in
    the same ANE procedure (only the int32 -> fp16 cast runs on the CPU)."""
    inputs = ("x", "attention_mask")

    def __init__(self):
        super().__init__()
        self.qkv, self.o = _linear(1, 3 * DIM), _linear(4)

    def forward(self, x, attention_mask):
        q, k, v = self.qkv(x).view(1, SEQ, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        m = attention_mask.to(torch.float32)[:, None, None, :]
        mask = ((1.0 - m) * MASKED).expand(1, 1, SEQ, SEQ).contiguous()
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(a.transpose(1, 2).reshape(1, SEQ, DIM))


class PackedQkv(torch.nn.Module):
    """q/k/v straight from a packed projection split along a middle axis
    (ModernBERT's layout, without its RoPE in between)."""
    inputs = ("x", "mask")

    def __init__(self, layout):
        super().__init__()
        self.qkv, self.layout = _linear(1, 3 * DIM), layout

    def forward(self, x, mask):
        qkv = self.qkv(x).view(1, SEQ, 3, HEADS, HEAD_DIM)
        if self.layout == "transpose_unbind":
            q, k, v = qkv.transpose(3, 1).unbind(dim=2)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(1, SEQ, DIM)


# ---------------------------------------------------------------------------
# Inputs. `pads` picks the content of the padded positions ("a" or "b");
# `masked=False` builds the same input with nothing masked (the unmasked
# reference). `rows_fully_masked` also masks every key for the last queries.

def make_feed(names, pads="a", masked=True, rows_fully_masked=False):
    rng = np.random.default_rng(0)
    pad_rng = np.random.default_rng({"a": 1, "b": 2}[pads])
    keypad = np.zeros((1, SEQ), np.float32)
    if masked:
        keypad[0, N_REAL:] = 1.0
    feed = {}
    for name in names:
        if name == "x":
            x = rng.standard_normal((1, SEQ, DIM)).astype(np.float32)
            x[0, N_REAL:] = 3.0 * pad_rng.standard_normal((SEQ - N_REAL, DIM))
            feed[name] = x
        elif name == "ids":
            ids = rng.integers(0, VOCAB, (1, SEQ)).astype(np.int32)
            ids[0, N_REAL:] = pad_rng.integers(0, VOCAB, SEQ - N_REAL)
            feed[name] = ids
        elif name == "mask":
            mask = np.broadcast_to((keypad * MASKED)[:, None, None, :], (1, 1, SEQ, SEQ)).copy()
            if rows_fully_masked:
                mask[..., SEQ - 8:, :] = MASKED
            feed[name] = mask
        elif name == "attention_mask":
            feed[name] = (1 - keypad).astype(np.int32)
        elif name == "keypad":
            feed[name] = keypad.copy()
    return feed


# ---------------------------------------------------------------------------
# Conversion, compute plan, metrics.

def convert(module, names, workdir):
    feed = make_feed(names)
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), tuple(torch.from_numpy(feed[n]) for n in names))
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=feed[n].shape, dtype=feed[n].dtype) for n in names],
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    pkg = Path(tempfile.mkdtemp(dir=workdir)) / "model.mlpackage"
    ml.save(str(pkg))
    cpu = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    return ml, cpu


def _device(plan, op):
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is None:
        return "none"
    return type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")


class PlanUnreadable(Exception):
    """The compute plan failed to load, or assigned no operation to a device."""


STALE_CACHE_HINT = ("Core ML's bundle cache (~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache) "
                    "may hold a broken entry for this path; re-check a copy at a new path "
                    "(`cp -c -R <model> <new path>`).")


def attention_masks(compiled_path, label=None):
    """For every fused attention op: (its device, where its mask comes from,
    whether the rule predicts the mask is ignored). Raises PlanUnreadable
    rather than returning a plan it can't trust."""
    label = label or compiled_path
    try:
        plan = MLComputePlan.load_from_path(path=str(compiled_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        raise PlanUnreadable(f"compute plan for {label} failed to load: {e}. {STALE_CACHE_HINT}") from e
    ops = [op for op in plan.model_structure.program.functions["main"].block.operations
           if not op.operator_name.endswith(".const")]
    devices = [_device(plan, op) for op in ops]
    if ops and all(d == "none" for d in devices):
        raise PlanUnreadable(f"compute plan for {label} assigns none of its {len(ops)} operations "
                             f"to a device. {STALE_CACHE_HINT}")
    produced_by = {out.name: i for i, op in enumerate(ops) for out in op.outputs}
    rows = []
    for i, op in enumerate(ops):
        if not op.operator_name.endswith(SDPA):
            continue
        mask_arg = op.inputs.get("attn_mask")
        if mask_arg is None:
            rows.append((devices[i], "no mask", False))
            continue
        name = mask_arg.bindings[0].name
        if name not in produced_by:
            source, outside = "model input", True
        else:
            j = produced_by[name]
            between = {d for d in devices[j + 1:i] if d not in ("NeuralEngine", "none")}
            source = f"{ops[j].operator_name.split('.')[-1]}@{devices[j]}"
            outside = devices[j] != "NeuralEngine" or bool(between)
            if devices[j] == "NeuralEngine" and between:
                source += ", CPU op between"
        rows.append((devices[i], source, devices[i] == "NeuralEngine" and outside))
    return rows


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def drop_fraction(out, masked_ref, unmasked_ref):
    """0 when the output matches masked attention, 1 when it matches unmasked."""
    out, m, u = (np.asarray(t, np.float64).ravel() for t in (out, masked_ref, unmasked_ref))
    if not np.isfinite(out).all():
        return float("nan")
    return float(np.linalg.norm(out - m) / np.linalg.norm(u - m))


def torch_out(module, names, feed):
    with torch.no_grad():
        return module(*(torch.from_numpy(feed[n]) for n in names)).numpy()


def measure(module, workdir):
    names = module.inputs
    ml_ne, ml_cpu = convert(module, names, workdir)
    plan = attention_masks(ml_ne.get_compiled_model_path())
    fa, fb = make_feed(names, "a"), make_feed(names, "b")
    ref_m, ref_u = torch_out(module, names, fa), torch_out(module, names, make_feed(names, "a", masked=False))
    real = (slice(None), slice(0, N_REAL))
    result = {"plan": plan}
    for label, model in (("NE", ml_ne), ("CPU", ml_cpu)):
        oa, ob = model.predict(fa)["y"], model.predict(fb)["y"]
        result[label] = (drop_fraction(oa[real], ref_m[real], ref_u[real]), cosine(oa[real], ob[real]))
    return result, ml_cpu


def verdict(drop, pad_inv):
    if np.isnan(drop) or np.isnan(pad_inv):
        return "NON-FINITE"
    if drop > 0.5:
        return "MASK IGNORED"
    if drop < 0.1 and pad_inv >= 0.99999:
        return "mask honoured"
    return "partly masked"


def run_all():
    cases = [
        ("mask is a model input", MaskInput()),
        ("mask built on the CPU (ModernBERT)", MaskBuiltOnCpu()),
        ("mask from another ANE procedure", MaskFromOtherAneProcedure()),
        ("mask built inside the procedure", MaskInsideProcedure()),
        ("model-input mask + one clamp", MaskInputClamped()),
        ("model-input mask, explicit attention", MaskInput("explicit")),
        ("model-input mask, scale= passed", MaskInput("scale")),
    ]
    print(f"Core ML fused attention, seq {SEQ}, {HEADS}x{HEAD_DIM} heads, fp16, macOS15 target, "
          f"{N_REAL} real tokens. drop: 0 = masked attention, 1 = unmasked. pad-inv: cosine "
          f"between two pad contents (1.0 = mask hides the pads).\n")
    header = f"{'case':<37} {'attention op <- mask source':<52} {'ANE drop / pad-inv':<22} {'CPU drop / pad-inv':<22} verdict (ANE)"
    print(header)
    print("-" * len(header))
    reproduced = True
    cpu_models = {}
    with tempfile.TemporaryDirectory() as wd:
        for label, module in cases:
            res, cpu_models[label] = measure(module, wd)
            plan = "; ".join(f"{d} <- {s}" for d, s, _ in res["plan"]) or "no fused attention op"
            (nd, npi), (cd, cpi) = res["NE"], res["CPU"]
            v = verdict(nd, npi)
            predicted = any(flag for *_, flag in res["plan"])
            reproduced &= (v == "MASK IGNORED") == predicted
            print(f"{label:<37} {plan:<52} {nd:6.3f} / {npi:.6f}    {cd:6.3f} / {cpi:.6f}    {v}", flush=True)

        print("\nCPU defects (CPU_ONLY):")
        feed = make_feed(MaskInput.inputs, rows_fully_masked=True)

        def bad_rows(label):
            out = cpu_models[label].predict(feed)["y"]
            return sorted({int(r) for r in np.argwhere(~np.isfinite(out))[:, 1]}) or "none"

        print(f"  fully-masked query rows (last 8, fill {MASKED:.0f}, head dim {HEAD_DIM}): non-finite rows "
              f"{bad_rows(cases[0][0])} with the fused op, {bad_rows('model-input mask, explicit attention')} "
              f"with explicit attention")
        for layout in ("transpose_unbind", "permute"):
            p = PackedQkv(layout)
            res, cpu = measure(p, wd)
            fa = make_feed(p.inputs)
            c = cosine(cpu.predict(fa)["y"][:, :N_REAL], torch_out(p, p.inputs, fa)[:, :N_REAL])
            print(f"  packed q/k/v via {layout:<17} CPU cosine vs fp32 reference {c:.6f}, pad-inv {res['CPU'][1]:.6f}")
    print("\nThe compute-plan rule predicted every ANE verdict." if reproduced else
          "\nNOTE: at least one ANE verdict differs from the compute-plan rule's prediction.")


def check(path):
    path = Path(path).expanduser()
    compiled = path
    if path.suffix == ".mlpackage":
        compiled = Path(ct.utils.compile_model(str(path)))
    rows = attention_masks(compiled, label=path)
    if not rows:
        print(f"{path}: no fused {SDPA} ops")
        return 0
    flagged = 0
    for i, (device, source, at_risk) in enumerate(rows):
        flagged += at_risk
        if at_risk:
            note = "MASK LIKELY IGNORED on the ANE"
        elif device == "NeuralEngine":
            note = "ok: native on the ANE, mask built in its own procedure"
        else:
            note = "ok: not run as a native ANE op"
        print(f"{SDPA} #{i}: {device} <- mask from {source}: {note}")
    print(f"{flagged} of {len(rows)} fused attention ops match the rule")
    return 1 if flagged else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", metavar="MODEL", help="apply the ANE rule to a .mlmodelc or .mlpackage")
    args = parser.parse_args()
    try:
        if args.check:
            sys.exit(check(args.check))
        run_all()
    except PlanUnreadable as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
