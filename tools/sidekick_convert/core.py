"""The conversion driver: ports, gate inputs, and the per-bucket loop.

A converter builds a `Job` and hands it to `run()`. For every bucket, `run()`:
1. builds the static-shape wrapper (`Job.make_wrapper(seq)`), so a recipe can
   differ per bucket;
2. runs the torch gates (fp32 exactness against the checkpoint's own forward);
3. traces it and converts with coremltools to an ML program (macOS 15 opset,
   int32 inputs with static shapes, docs/DECISIONS.md D15 and D27);
4. refuses the graph if it contains a forbidden op (Core ML's fused
   attention by default, D25);
5. compiles it with `xcrun coremlcompiler` and runs the Core ML gates on the
   compiled artifact, the bytes that get installed;
6. installs `model_{seq}.mlmodelc`.

Two inputs are kept apart by type, so no recipe can mix them up:
- `Calibration` decides rewrites (residual K, input rescales). It must never
  come from the graded parity corpus (fixtures/parity/corpus.toml), or the
  grades would flatter the model.
- `Evaluation` holds the gate cases and their fp32 references.
"""

import dataclasses
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PARITY_CORPUS = REPO / "fixtures" / "parity" / "corpus.toml"
FUSED_ATTENTION = "scaled_dot_product_attention"


class GateFailure(Exception):
    """A gate failed. Fatal, except in a negative control, which reports it."""


@dataclasses.dataclass(frozen=True)
class Port:
    """One Core ML input: int32, static shape as a function of the bucket."""
    name: str
    shape: Callable[[int], tuple]

    def __call__(self, seq):
        return tuple(self.shape(seq))


def sequence_port(name):
    return Port(name, lambda seq: (1, seq))


def text_ports(token_type_ids=False):
    """input_ids and attention_mask, plus token_type_ids for pair models, all
    [1, S] int32."""
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_type_ids else [])
    return [sequence_port(n) for n in names]


def bucket_of(n, buckets):
    """The smallest bucket that fits n tokens, as the server picks it."""
    return next(b for b in sorted(buckets) if n <= b)


@dataclasses.dataclass
class Case:
    """One evaluation input and its fp32 reference output.

    `extra` holds per-port values for ports beyond the text ones: a list for
    token_type_ids (padded with 0), or an array for fixed-shape ports
    (laya's marker_pos, qtype)."""
    ids: list
    ref: np.ndarray
    extra: dict = dataclasses.field(default_factory=dict)
    label: str = ""

    @property
    def n(self):
        return len(self.ids)

    def feed(self, seq, ports, pad_ids=None):
        """int32 inputs for bucket `seq`: right-padded with id 0, as the server
        pads, or with `pad_ids` (pad-invariance gate)."""
        n = len(self.ids)
        if n > seq:
            raise ValueError(f"case {self.label!r} is {n} tokens, longer than bucket {seq}")
        out = {}
        for port in ports:
            if port.name == "input_ids":
                x = np.zeros((1, seq), dtype=np.int32)
                x[0, :n] = self.ids
                if pad_ids is not None:
                    x[0, n:] = np.asarray(pad_ids)[: seq - n]
            elif port.name == "attention_mask":
                x = np.zeros((1, seq), dtype=np.int32)
                x[0, :n] = 1
            elif port.name == "token_type_ids":
                x = np.zeros((1, seq), dtype=np.int32)
                x[0, :n] = self.extra.get("token_type_ids", [0] * n)
            else:
                x = np.asarray(self.extra[port.name], dtype=np.int32).reshape(port(seq))
            out[port.name] = x
        return out


def _parity_corpus_texts():
    if not PARITY_CORPUS.exists():
        return set()
    corpus = tomllib.loads(PARITY_CORPUS.read_text())
    return {c["text"] for c in corpus.get("case", []) if "text" in c}


def _graded(texts):
    """Texts that are, or end with, a graded parity-corpus text: a model's
    prompt prefix followed by a corpus text is still the corpus text. Short
    corpus texts (under 12 characters: "a", "42") only match exactly."""
    corpus = _parity_corpus_texts()
    long = [c for c in corpus if len(c) >= 12]
    return [t for t in texts if t in corpus or any(t.endswith(c) for c in long)]


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Inputs that decide rewrites. Never the graded parity corpus (D26):
    calibrating on what is graded would flatter the grades. A converter's own
    gate texts, or a committed calibration set, are fine.

    `legacy_graded` is the one exception: a reason string that keeps graded
    texts in an existing model's calibration, because dropping them would
    change an artifact that is already graded and shipped. It is logged on
    every run, and removing it is a separate, measured change."""
    texts: tuple
    legacy_graded: str = None

    def __post_init__(self):
        object.__setattr__(self, "texts", tuple(self.texts))
        graded = _graded(self.texts)
        if graded and not self.legacy_graded:
            raise ValueError(f"calibration texts taken from the graded parity corpus: {graded[:3]}")
        if graded:
            print(f"WARNING: calibration keeps {len(graded)} graded parity-corpus text(s): {self.legacy_graded}",
                  flush=True)

    @classmethod
    def without_graded(cls, texts, report=print):
        """A Calibration of `texts` minus any the graded parity corpus holds,
        reporting what it dropped (a converter's gate texts can overlap the
        corpus, which was seeded from them)."""
        dropped = _graded(texts)
        if dropped:
            report(f"calibration: dropped {len(dropped)} text(s) that the graded parity corpus holds")
        return cls([t for t in texts if t not in dropped])


@dataclasses.dataclass
class Evaluation:
    """Gate cases with fp32 references. Used only to judge, never to calibrate."""
    cases: list

    def fitting(self, seq):
        return [c for c in self.cases if c.n <= seq]

    def landing(self, seq, buckets):
        return [c for c in self.cases if c.n <= max(buckets) and bucket_of(c.n, buckets) == seq]


@dataclasses.dataclass
class Job:
    """Everything `run()` needs to convert one model into per-bucket artifacts."""
    name: str
    buckets: list
    ports: list
    output: str
    make_wrapper: Callable
    example: Callable
    evaluation: Evaluation
    gates: object
    calibration: Calibration = None
    forbid_ops: frozenset = frozenset({FUSED_ATTENTION})
    install_files: list = dataclasses.field(default_factory=list)   # (source path, installed name)
    negative_control: bool = False
    timing: bool = False
    landing_required: bool = False
    gate_cases: str = "fitting"   # "fitting": every case that fits a bucket; "landing": those the server
                                  # would send to it (for models whose accuracy varies by bucket)


def trace_convert(wrapper, ports, seq, example, output):
    """jit.trace, then coremltools to an ML program (macOS 15 opset), int32
    static-shape inputs."""
    import torch
    import coremltools as ct
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(torch.from_numpy(example[p.name]) for p in ports))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=p.name, shape=p(seq), dtype=np.int32) for p in ports],
        outputs=[ct.TensorType(name=output)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )


def mil_op_types(mlmodel):
    """Every op type in the converted ML program."""
    return {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
            for block in fn.block_specializations.values() for op in block.operations}


def compile_mlmodelc(pkg, out_dir):
    """xcrun coremlcompiler; returns the compiled .mlmodelc inside out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)], check=True,
                   stdout=subprocess.DEVNULL)
    return next(out_dir.glob("*.mlmodelc"))


def _gate(job, fn, *args):
    try:
        return fn(*args)
    except GateFailure as e:
        if not job.negative_control:
            raise
        print(f"negative control, expected: {e}", flush=True)
        return {}


def run(job, install_dir):
    """Convert, gate and install every bucket, then install the job's files."""
    install_dir = Path(install_dir).expanduser()
    install_dir.mkdir(parents=True, exist_ok=True)
    if job.negative_control:
        print(f"{job.name}: NEGATIVE CONTROL. Gate failures are reported, not fatal; "
              "never install the result where the daemon looks.", flush=True)
    reports = {}
    with tempfile.TemporaryDirectory() as work:
        for seq in job.buckets:
            cases = (job.evaluation.landing(seq, job.buckets) if job.gate_cases == "landing"
                     else job.evaluation.fitting(seq))
            if not cases:
                raise GateFailure(f"{job.name}: no evaluation case fits bucket {seq}")
            if job.landing_required and not job.evaluation.landing(seq, job.buckets):
                raise GateFailure(f"{job.name}: no evaluation case lands in bucket {seq}")
            wrapper = job.make_wrapper(seq).eval()
            report = {"torch": _gate(job, job.gates.torch, wrapper, seq, cases, job.ports)}
            print(f"bucket {seq}: {job.gates.describe_torch(report['torch'])}; converting...", flush=True)
            mlmodel = trace_convert(wrapper, job.ports, seq, job.example(seq), job.output)
            found = mil_op_types(mlmodel) & set(job.forbid_ops)
            if found:
                _gate(job, _raise, GateFailure(
                    f"bucket {seq}: the converted graph contains {sorted(found)}; "
                    "see docs/CONVERTING.md for why each is forbidden"))
            pkg = Path(work) / f"model_{seq}.mlpackage"
            mlmodel.save(str(pkg))
            compiled = compile_mlmodelc(pkg, Path(work) / f"compiled_{seq}")
            report["coreml"] = _gate(job, job.gates.coreml, compiled, seq, cases, job, job.timing)
            for line in job.gates.describe_coreml(report["coreml"]):
                print(f"bucket {seq}: {line}", flush=True)
            dest = install_dir / f"model_{seq}.mlmodelc"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(compiled), dest)
            print(f"bucket {seq} -> {dest}", flush=True)
            reports[seq] = report
    for source, name in job.install_files:
        shutil.copy(source, install_dir / name)
    if job.install_files:
        print(f"installed {', '.join(n for _, n in job.install_files)} -> {install_dir}", flush=True)
    return reports


def _raise(e):
    raise e


def fail(message):
    """Stop a converter with a message (not a gate: always fatal)."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)
