"""The conversion driver: ports, gate inputs, and the per-bucket loop.

A converter builds a `Job` and hands it to `run()`. For every bucket, `run()`:
1. builds the static-shape wrapper (`Job.make_wrapper(seq)`), so a recipe can
   differ per bucket;
2. runs the torch gates (fp32 exactness against the checkpoint's own forward);
3. traces it and converts with coremltools to an ML program (macOS 15 opset,
   int32 inputs with static shapes, docs/DECISIONS.md D15 and D27);
4. refuses the graph if it contains a forbidden op (Core ML's fused
   attention by default, D25), and with `int8_embedding` stores the
   token-embedding table in int8;
5. compiles it with `xcrun coremlcompiler`; refuses a model served on the
   ANE whose weights pass the Neural Engine's per-program limit
   (plan.MAX_ANE_PROGRAM_WEIGHT_BYTES; `ignore_ane_weight_cap` makes that a
   warning, recorded in the report and the installed manifest); and runs the
   Core ML gates on the compiled artifact, the bytes that get installed;
6. installs `model_{seq}.mlmodelc`, and records in the installed manifest
   the compute plan each bucket was read with ([placement]), so the daemon
   can report placement without compiling the model a second time.

A job with `chunks` (chunking.py, D37) converts each bucket as a chain of
programs split at layer boundaries: the fp32 gate also checks the composed
chunks against the unchunked wrapper, each chunk is converted, compiled and
checked against the weight limit on its own, the unchunked program is
converted too so the chain's GPU output can be required bit-identical to
it, and the usual Core ML gates run on the chain. It installs
`model_{seq}.{chunk}.mlmodelc` and a manifest whose `artifact` and
[chunking] table say so.

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
    # Values the gates read but the model doesn't take, e.g. gliner2's
    # "markers" (the positions its labels' logits are read at) and the
    # case's "activation". `extra` stays per port: feed() reads it by port.
    meta: dict = dataclasses.field(default_factory=dict)

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
    int8_embedding: bool = False  # store the token-embedding table in int8 (see int8_embedding())
    ignore_ane_weight_cap: bool = False  # convert past MAX_ANE_PROGRAM_WEIGHT_BYTES, with a warning
    chunks: object = None         # None, "auto", a chunk count or layer cuts (chunking.plan_cuts); needs
    backbone: object = None       # the backbone and head the wrapper was composed from
    head: object = None


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


def int8_embedding(mlmodel):
    """The converted program with its token-embedding table, the largest
    constant a gather reads, stored in int8: linear symmetric, one scale per
    row. Opt-in (Job(int8_embedding=True)), and graded like any rewrite. It
    halves the largest single weight of a big-vocabulary model, which can
    bring the program under Core ML's ~1 GiB cap for the ANE (plan.py). The
    lookup is a gather, off the ANE either way, so the ANE's arithmetic is
    unchanged. Returns (model, table name, shape)."""
    import coremltools.optimize.coreml as cto
    tables = [op.x.op for fn in mlmodel._mil_program.functions.values() for op in fn.operations
              if op.op_type == "gather" and op.x.op is not None and op.x.op.op_type == "const"
              and len(op.x.shape) == 2]
    if not tables:
        raise GateFailure("int8_embedding: the converted program has no gather from a constant table")
    table = max(tables, key=lambda t: t.outputs[0].shape[0])
    config = cto.OptimizationConfig(op_name_configs={table.name: cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=1)})
    return cto.linear_quantize_weights(mlmodel, config), table.name, tuple(table.outputs[0].shape)


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


def check_ane_weights(job, seq, compiled, chunk=None):
    """The weight-size check for a model served on the ANE (the gates make
    its compute plan a gate), for one program: a bucket, or one chunk of
    it. Returns None when the weights fit, or the bypass note when the job
    ignores the limit; raises GateFailure otherwise."""
    from . import plan
    if not getattr(job.gates, "plan_required", True):
        return None
    size = plan.weights_bytes(compiled)
    if size <= plan.MAX_ANE_PROGRAM_WEIGHT_BYTES:
        return None
    message = plan.over_cap(job.name, seq, size, chunk)
    if not job.ignore_ane_weight_cap:
        raise GateFailure(message)
    print(f"WARNING, --ignore-ane-weight-cap: {message} Converting anyway; the compute-plan gate still "
          "checks where Core ML places it.", flush=True)
    what = f"bucket {seq}" if chunk is None else f"bucket {seq}'s chunk {chunk}"
    return (f"{what} has {size / 2**30:.3f} GiB of weights, over the Neural Engine's 1 GiB "
            "per-program limit (MAX_ANE_PROGRAM_WEIGHT_BYTES)")


def record_placement(manifest, plans):
    """Append the [placement] table (plan.placement_toml) for the buckets whose
    compute plan the gates read, to an installed manifest. A bucket whose
    plan was unreadable is left out; the daemon reports it as not recorded."""
    if not plans:
        return
    import datetime
    from . import plan
    chip, build = plan.machine()
    with open(manifest, "a") as f:
        f.write(plan.placement_toml(plans, chip, build, datetime.date.today().isoformat()))


def note_bypass(manifest, notes):
    """Append the weight-limit bypass to an installed manifest, as comments,
    so a bypassed artifact stays visible."""
    lines = ["", "# Converted with --ignore-ane-weight-cap:"] + [f"# - {n}" for n in notes]
    with open(manifest, "a") as f:
        f.write("\n".join(lines) + "\n")


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
    reports, bypassed, plans = {}, [], {}
    cuts = _plan_chunks(job)
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
            if cuts:
                reports[seq] = _run_chunked(job, seq, cases, wrapper, cuts, report, work, install_dir, bypassed,
                                            plans)
                continue
            mlmodel = trace_convert(wrapper, job.ports, seq, job.example(seq), job.output)
            found = mil_op_types(mlmodel) & set(job.forbid_ops)
            if found:
                _gate(job, _raise, GateFailure(
                    f"bucket {seq}: the converted graph contains {sorted(found)}; "
                    "see docs/CONVERTING.md for why each is forbidden"))
            if job.int8_embedding:
                mlmodel, table, shape = int8_embedding(mlmodel)
                print(f"bucket {seq}: embedding table {table} {shape} stored in int8", flush=True)
            pkg = Path(work) / f"model_{seq}.mlpackage"
            mlmodel.save(str(pkg))
            compiled = compile_mlmodelc(pkg, Path(work) / f"compiled_{seq}")
            bypass = _gate(job, check_ane_weights, job, seq, compiled)
            if bypass:
                report["ane_weight_cap"] = f"bypassed: {bypass}"
                bypassed.append(bypass)
            report["coreml"] = _gate(job, job.gates.coreml, compiled, seq, cases, job, job.timing)
            if (report["coreml"] or {}).get("plan"):
                plans[seq] = report["coreml"]["plan"]
            for line in job.gates.describe_coreml(report["coreml"]):
                print(f"bucket {seq}: {line}", flush=True)
            dest = install_dir / f"model_{seq}.mlmodelc"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(compiled), dest)
            print(f"bucket {seq} -> {dest}", flush=True)
            reports[seq] = report
    for source, name in job.install_files:
        shutil.copy(source, install_dir / name)
        if name.endswith(".toml") and cuts:
            from . import chunking
            dest = install_dir / name
            budget = chunking.CHUNK_WEIGHT_BUDGET_BYTES if job.chunks == "auto" else None
            dest.write_text(chunking.manifest_text(dest.read_text(), len(cuts) + 1, budget))
        if name.endswith(".toml"):
            record_placement(install_dir / name, plans)
            if bypassed:
                note_bypass(install_dir / name, bypassed)
    if job.install_files:
        print(f"installed {', '.join(n for _, n in job.install_files)} -> {install_dir}", flush=True)
    if bypassed:
        print(f"WARNING: installed past the Neural Engine's weight limit (--ignore-ane-weight-cap): "
              f"{'; '.join(bypassed)}", flush=True)
    return reports


def _raise(e):
    raise e


CHUNK_FP32_TOL = 1e-5
"""How far the composed chunks may be from the unchunked wrapper in fp32:
the same operations in the same order, so in practice 0."""


def _plan_chunks(job):
    """The layer cuts every bucket is split at (planned for the largest
    bucket, whose buffers every chunk repeats), or None."""
    if not job.chunks:
        return None
    from . import chunking
    if job.backbone is None or job.head is None:
        raise ValueError(f"{job.name}: a chunked Job needs its backbone and head")
    seq = max(job.buckets)
    cuts = chunking.plan_cuts(job.backbone, job.head, seq, job.chunks)
    if not cuts:
        print(f"{job.name}: --chunks {job.chunks}: every bucket fits one program; converting unchunked", flush=True)
        return None
    sizes = chunking.chunk_sizes(job.backbone, job.head, seq, cuts)
    print(f"{job.name}: {len(cuts) + 1} chunks, starting at layers {[0] + cuts}; planned fp16 weights "
          f"{', '.join(f'{b / 2**30:.3f}' for b in sizes)} GiB at {seq} tokens", flush=True)
    return cuts


def _run_chunked(job, seq, cases, wrapper, cuts, report, work, install_dir, bypassed, plans):
    """One bucket as a chain of chunks (D37): see the module docstring."""
    import torch
    from . import chunking, plan
    from .gates import _model
    from .metrics import largest, max_abs_diff
    chunk_list = chunking.chunks(job.backbone, job.head, job.ports, seq, cuts, job.output)
    diffs = []
    for c in cases:
        feed = c.feed(seq, job.ports)
        with torch.no_grad():
            whole = wrapper(*(torch.from_numpy(feed[p.name]) for p in job.ports))
        diffs.append(max_abs_diff(chunking.run_torch(chunk_list, feed).numpy(), whole.numpy()))
    d = largest(diffs)
    report["torch"]["chunks_fp32"] = d
    if not d <= CHUNK_FP32_TOL:
        _gate(job, _raise, GateFailure(f"bucket {seq}: the composed chunks differ from the unchunked wrapper "
                                       f"in fp32 by {d:.2e} (gate {CHUNK_FP32_TOL})"))
    print(f"bucket {seq}: fp32 chunks vs the unchunked wrapper, max |diff| {d:.1e}", flush=True)

    paths, inputs = [], []
    example = job.example(seq)
    for c in chunk_list:
        mlmodel = chunking.trace_convert(c, seq, example, job.backbone.hidden_size)
        found = mil_op_types(mlmodel) & set(job.forbid_ops)
        if found:
            _gate(job, _raise, GateFailure(
                f"bucket {seq} chunk {c.index}: the converted graph contains {sorted(found)}; "
                "see docs/CONVERTING.md for why each is forbidden"))
        if job.int8_embedding and c.index == 0:
            mlmodel, table, shape = int8_embedding(mlmodel)
            print(f"bucket {seq} chunk 0: embedding table {table} {shape} stored in int8", flush=True)
        pkg = Path(work) / f"model_{seq}.{c.index}.mlpackage"
        mlmodel.save(str(pkg))
        del mlmodel
        compiled = compile_mlmodelc(pkg, Path(work) / f"compiled_{seq}_{c.index}")
        shutil.rmtree(pkg)
        bypass = _gate(job, check_ane_weights, job, seq, compiled, c.index)
        if bypass:
            report.setdefault("ane_weight_cap", []).append(f"bypassed: {bypass}")
            bypassed.append(bypass)
        print(f"bucket {seq} chunk {c.index}: layers [{c.lo}, {c.hi}), {plan.weights_bytes(compiled) / 2**30:.3f} "
              f"GiB, inputs {list(c.inputs)} -> {c.output}", flush=True)
        paths.append(compiled)
        inputs.append(c.inputs)
    chain = chunking.Chain(paths, inputs, job.output)

    # The chain's plumbing, exactly: on the GPU it must give the unchunked
    # program's output bit for bit.
    whole = trace_convert(wrapper, job.ports, seq, example, job.output)
    if job.int8_embedding:
        whole, _, _ = int8_embedding(whole)
    pkg = Path(work) / f"whole_{seq}.mlpackage"
    whole.save(str(pkg))
    del whole
    whole_compiled = compile_mlmodelc(pkg, Path(work) / f"whole_compiled_{seq}")
    shutil.rmtree(pkg)
    a, b = _model(whole_compiled, "CPU_AND_GPU"), _model(chain, "CPU_AND_GPU")
    worst = 0.0
    for c in cases:
        feed = c.feed(seq, job.ports)
        x = np.asarray(a.predict(feed)[job.output], dtype=np.float32)
        y = np.asarray(b.predict(feed)[job.output], dtype=np.float32)
        worst = max(worst, float(np.abs(x - y).max()) if np.array_equal(np.isnan(x), np.isnan(y)) else np.inf)
    del a, b
    shutil.rmtree(whole_compiled.parent, ignore_errors=True)
    report["chain_gpu_vs_unchunked"] = worst
    if worst != 0.0:
        _gate(job, _raise, GateFailure(f"bucket {seq}: on the GPU the chain differs from the unchunked program "
                                       f"by {worst:.2e}; it must be bit-identical"))
    print(f"bucket {seq}: on the GPU the chain is bit-identical to the unchunked program", flush=True)

    report["coreml"] = _gate(job, job.gates.coreml, chain, seq, cases, job, job.timing)
    if (report["coreml"] or {}).get("plan"):
        plans[seq] = report["coreml"]["plan"]
    for line in job.gates.describe_coreml(report["coreml"]):
        print(f"bucket {seq}: {line}", flush=True)
    for c, compiled in zip(chunk_list, paths):
        dest = install_dir / chunking.artifact_name(seq, c.index)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
        print(f"bucket {seq} chunk {c.index} -> {dest}", flush=True)
    return report


def fail(message):
    """Stop a converter with a message (not a gate: always fatal)."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)
