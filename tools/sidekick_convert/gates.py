"""Gates: what a converted bucket must satisfy before it is installed.

Two families of gates, one per kind of output:
- `EmbeddingGates`: cosine against the fp32 reference (the vector is
  normalized by the server, so scale is free).
- `ClassifierGates`: logits, graded after the model's activation (D28).

Each has a torch phase, run on the wrapper before converting (fp32
exactness: the rewritten, padded, static-shape graph must reproduce the
checkpoint's own unpadded forward), and a Core ML phase, run on the compiled
artifact:
- the compute plan (plan.gate: ane_check's verdict, D24);
- accuracy per compute path, with CPU_AND_NE being what sidekick serves and
  CPU_ONLY a second, independent execution of the same graph;
- pad invariance: the same input with random pad ids must give the same
  output, which catches a dropped attention mask (D25) or a mixer that reads
  pad states (D19);
- finite output.
Every metric is NaN-safe (metrics.py): a non-finite value fails.
Latency is measured only when asked (`--time`): it depends on machine load,
accuracy doesn't.
"""

import dataclasses
import time

import numpy as np

from . import plan as _plan
from .core import GateFailure
from .metrics import activate, cosine, finite, largest, max_abs_diff, worst

PATHS = {"CPU_AND_NE": "CPU_AND_NE", "CPU_ONLY": "CPU_ONLY"}


# Every prediction input, referenced for the life of the process. Core ML
# keeps a prediction's inputs bound to its execution stream and releases them
# about a second after the stream goes idle, on a queue of its own;
# coremltools backs them with the NumPy arrays, so an input Python has
# already freed crashes the process then (EXC_BAD_ACCESS on
# MLE5ExecutionStream's reset queue). Gates make few, small predictions.
_INPUTS = []


class _Model:
    """A compiled model whose predict() keeps its inputs referenced."""

    def __init__(self, model):
        self._model = model

    def predict(self, feed):
        _INPUTS.append(feed)
        return self._model.predict(feed)


def _model(compiled, path):
    import coremltools as ct
    return _Model(ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, path)))


def _run_torch(wrapper, feed, ports):
    import torch
    with torch.no_grad():
        return wrapper(*(torch.from_numpy(feed[p.name]) for p in ports))[0].numpy()


def _latency_ms(model, feed, warm=3, n=10):
    for _ in range(warm):
        model.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(feed)
    return (time.perf_counter() - t0) / n * 1e3


def _pad_ids(seq, lo_hi, seed=0):
    return np.random.default_rng(seed).integers(lo_hi[0], lo_hi[1], seq)


@dataclasses.dataclass
class EmbeddingGates:
    fp32_min_cos: float = 0.99999
    parity: dict = dataclasses.field(default_factory=lambda: {"CPU_AND_NE": 0.999, "CPU_ONLY": 0.999})
    pad_min_cos: float = 0.99999
    plan_min_ane: float = 0.8
    pad_id_range: tuple = (1000, 30000)

    def torch(self, wrapper, seq, cases, ports):
        w = worst(cosine(c.ref, _run_torch(wrapper, c.feed(seq, ports), ports)) for c in cases)
        if not w >= self.fp32_min_cos:
            raise GateFailure(f"bucket {seq}: fp32 wrapper vs the checkpoint, worst cosine {w:.7f} "
                              f"< {self.fp32_min_cos}")
        return {"fp32": w}

    def describe_torch(self, r):
        return f"fp32 wrapper vs the checkpoint, worst cosine {r.get('fp32', float('nan')):.7f}"

    def coreml(self, compiled, seq, cases, job, timing):
        out = {"plan": _plan.gate(compiled, self.plan_min_ane)}
        padded = next((c for c in cases if c.n < seq), None)
        for path, gate in self.parity.items():
            m = _model(compiled, path)
            outs = [m.predict(c.feed(seq, job.ports))[job.output][0] for c in cases]
            if not all(finite(o) for o in outs):
                raise GateFailure(f"bucket {seq} [{path}]: non-finite output")
            w = worst(cosine(c.ref, o) for c, o in zip(cases, outs))
            if not w >= gate:
                raise GateFailure(f"bucket {seq} [{path}]: parity cosine {w:.6f} < {gate}")
            r = {"n": len(cases), "worst": w}
            if padded is not None:
                a = m.predict(padded.feed(seq, job.ports))[job.output][0]
                b = m.predict(padded.feed(seq, job.ports, _pad_ids(seq, self.pad_id_range)))[job.output][0]
                r["pad"] = cosine(a, b)
                if not r["pad"] >= self.pad_min_cos:
                    raise GateFailure(f"bucket {seq} [{path}]: output depends on pad content (cosine "
                                      f"{r['pad']:.7f}); the attention mask is being dropped or a mixer "
                                      "reads pad states")
            if timing:
                r["ms"] = _latency_ms(m, cases[0].feed(seq, job.ports))
            out[path] = r
        return out

    def describe_coreml(self, r):
        lines = [_plan.describe(r["plan"])] if "plan" in r else []
        for path in self.parity:
            if path in r:
                p = r[path]
                lines.append(f"[{path}] n={p['n']} parity cos={p['worst']:.6f}"
                             + (f" pad {p['pad']:.7f}" if "pad" in p else "")
                             + (f" {p['ms']:.1f}ms" if "ms" in p else ""))
        return lines


@dataclasses.dataclass
class ClassifierGates:
    """Logit outputs. `activation` is the manifest's (softmax, sigmoid or
    identity); probabilities are compared after it. An argmax flip counts only
    where the fp32 top-2 margin is at least `margin` (single-output models
    have no argmax). Paths in `gated_paths` fail on flips or on max |dp| above
    `dp_gate`; paths in `report_paths` only report. `pad_value`, when set, is
    what padded output slots must hold (laya's -1e4); a case's reference then
    covers only its first len(ref) slots.

    What counts as a flip follows the decision the activation serves: for
    softmax, the argmax; for sigmoid over several labels (multi-label), each
    label's own yes/no, the sign of its logit. A label whose fp32 logit is
    within `margin` of the decision boundary is a near-tie either way. A
    single output (a reranker's score) has no decision to flip.

    `markers` grades what a per-token head serves (gliner2): a case's
    reference covers its real tokens, and its served logits are the output
    at case.meta["markers"], activated by case.meta["activation"] when set
    (a multi-label request) or `activation`. Flips, |dp| and pad invariance
    are all measured on those logits, not on every token.

    `plan_required` makes the compute plan a gate (every heavy op and
    `plan_min_ane` of all ops on the ANE). Without it the plan is only
    reported, for a model served off the ANE; `paths()` builds the paths
    from the manifest's served path."""
    fp32_tol: float = 1e-3
    activation: str = "softmax"
    margin: float = 0.05
    dp_gate: float = 0.02
    gated_paths: tuple = ("CPU_AND_NE", "CPU_ONLY")
    report_paths: tuple = ()
    pad_tol: float = 1e-3
    pad_value: float = None
    markers: bool = False
    plan_min_ane: float = 0.8
    pad_id_range: tuple = (1000, 30000)
    plan_required: bool = True
    served: str = None  # the compute units the model is served with; default: the first gated path

    @staticmethod
    def paths(served, report=("CPU_AND_NE", "CPU_ONLY")):
        """Keyword arguments for a model served on `served` (manifest.served_path):
        that path gated, the rest of `report` reported, and the compute plan a
        gate only for a model served on the ANE."""
        return {"gated_paths": (served,), "report_paths": tuple(p for p in report if p != served),
                "plan_required": served == "CPU_AND_NE", "served": served}

    def units(self):
        """The compute units the model is served with, which its compute plan
        is read for."""
        return self.served or (self.gated_paths[0] if self.gated_paths else "CPU_AND_NE")

    def _split(self, out, ref):
        k = len(ref)
        got = np.asarray(out, dtype=np.float64)
        slots_ok = self.pad_value is None or bool(np.all(got[k:] == self.pad_value))
        return got[:k], slots_ok

    def _served(self, case, out):
        """(served logits, slots ok, their reference) for one case's output."""
        if not self.markers:
            got, ok = self._split(out, case.ref)
            return got, ok, np.asarray(case.ref, dtype=np.float64)
        m = list(case.meta["markers"])
        return (np.asarray(out, dtype=np.float64)[m], True, np.asarray(case.ref, dtype=np.float64)[m])

    def _decisions(self, got, ref, activation):
        """(flips, near-ties) of one case's served logits against fp32."""
        if len(ref) < 2:
            return 0, 0
        if activation == "sigmoid":
            clear = np.abs(ref) >= self.margin
            flips = int(np.sum(clear & (np.sign(got) != np.sign(ref))))
            return flips, int(np.sum(~clear))
        top2 = np.sort(ref)[-2:]
        if top2[1] - top2[0] >= self.margin:
            return int(np.argmax(got) != np.argmax(ref)), 0
        return 0, 1

    def _activation(self, case):
        return case.meta.get("activation", self.activation) if self.markers else self.activation

    def torch(self, wrapper, seq, cases, ports):
        diffs = []
        for c in cases:
            got, slots_ok, ref = self._served(c, _run_torch(wrapper, c.feed(seq, ports), ports))
            if not slots_ok:
                raise GateFailure(f"bucket {seq}: fp32 wrapper output malformed (padded slots)")
            diffs.append(max_abs_diff(got, ref))
        d = largest(diffs)
        if not d <= self.fp32_tol:
            raise GateFailure(f"bucket {seq}: fp32 wrapper vs the checkpoint, max |dlogit| {d:.2e} "
                              f"> {self.fp32_tol}")
        return {"fp32": d}

    def describe_torch(self, r):
        return f"fp32 wrapper vs the checkpoint, max |dlogit| {r.get('fp32', float('nan')):.1e}"

    def coreml(self, compiled, seq, cases, job, timing):
        out = {"plan": _plan.gate(compiled, self.plan_min_ane) if self.plan_required
               else _plan.report(compiled, self.units())}
        padded = next((c for c in cases if c.n < seq), None)
        for path in tuple(self.gated_paths) + tuple(self.report_paths):
            m = _model(compiled, path)
            flips, ties, dps, dls = 0, 0, [], []
            for c in cases:
                got, slots_ok, ref = self._served(c, m.predict(c.feed(seq, job.ports))[job.output][0])
                if not (finite(got) and slots_ok):
                    raise GateFailure(f"bucket {seq} [{path}]: non-finite or unpadded logits")
                activation = self._activation(c)
                f, t = self._decisions(got, ref, activation)
                flips, ties = flips + f, ties + t
                dps.append(max_abs_diff(activate(got, activation), activate(ref, activation)))
                dls.append(max_abs_diff(got, ref))
            r = {"n": len(cases), "flips": flips, "near_ties": ties, "dp_max": largest(dps),
                 "dlogit_max": largest(dls)}
            bad = flips or not r["dp_max"] <= self.dp_gate
            if bad:
                message = (f"bucket {seq} [{path}]: {flips} decision flips above margin {self.margin}, "
                           f"max |dp| {r['dp_max']:.4f} (gate {self.dp_gate})")
                if path in self.gated_paths:
                    raise GateFailure(message)
                print(f"WARNING, report only: {message}", flush=True)
            if padded is not None:
                a, _, _ = self._served(padded, m.predict(padded.feed(seq, job.ports))[job.output][0])
                b, _, _ = self._served(padded, m.predict(padded.feed(seq, job.ports, _pad_ids(seq, self.pad_id_range)))
                                       [job.output][0])
                r["pad"] = max_abs_diff(a, b)
                if not r["pad"] <= self.pad_tol:
                    raise GateFailure(f"bucket {seq} [{path}]: logits depend on pad content "
                                      f"(max |dlogit| {r['pad']})")
            if timing:
                r["ms"] = _latency_ms(m, cases[0].feed(seq, job.ports))
            out[path] = r
        return out

    def describe_coreml(self, r):
        lines = [_plan.describe(r["plan"])] if r.get("plan") else []
        for path in tuple(self.gated_paths) + tuple(self.report_paths):
            if path in r:
                p = r[path]
                lines.append(f"[{path}] n={p['n']} max |dp| {p['dp_max']:.4f} max |dlogit| {p['dlogit_max']:.3f} "
                             f"flips {p['flips']} near-ties {p['near_ties']}"
                             + (f" pad {p['pad']:.1e}" if "pad" in p else "")
                             + (f" {p['ms']:.1f}ms" if "ms" in p else ""))
        return lines
