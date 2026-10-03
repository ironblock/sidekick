"""Chunked buckets: one bucket as an ordered chain of programs, each under
the Neural Engine's per-program weight limit (docs/DECISIONS.md D37).

Core ML runs an ML program on the ANE only if its weights are under about
1 GiB (plan.MAX_ANE_PROGRAM_WEIGHT_BYTES, D32). A bucket over it is split,
in torch and before tracing, at layer boundaries:
- the first chunk holds the embedding and the first layers, and takes
  `input_ids`;
- every chunk after it takes `hidden_in`, the residual stream, as an fp16
  [1, S, H] input, and every chunk before the last outputs it as
  `hidden_out`;
- every chunk takes the int32 ports its backbone reads (the mask and
  position inputs) and rebuilds what it derives from them, so nothing else
  crosses a boundary;
- the last chunk holds the final norm and the head, takes the head's own
  ports, and produces the model's output.

The cuts are layer indices: chunk i runs layers [cuts[i-1], cuts[i]).
`plan_cuts` chooses them ("auto": the fewest chunks whose fp16 weights fit
CHUNK_WEIGHT_BUDGET_BYTES, balanced), or takes a chunk count or explicit
cuts. A chunk keeps every rewrite of its backbone, and the fp32 gate checks
the composed chunks against the unchunked wrapper.
"""

import dataclasses
import types

import numpy as np
import torch

HIDDEN_IN, HIDDEN_OUT = "hidden_in", "hidden_out"

CHUNK_WEIGHT_BUDGET_BYTES = int(0.9 * 2**30)
"""The most fp16 weight `auto` puts in one chunk: 0.9 GiB. The ANE's limit
was measured between 0.964 and 1.022 GiB on one chip and one OS (D32); the
budget keeps a margin below the largest program measured on the ANE, for
other chips. Every compiled chunk is still checked against the limit."""


def _module_bytes(module):
    return sum(t.numel() * 2 for t in module.parameters())


def _buffer_bytes(tensors):
    # float buffers become fp16 constants; integer ones stay 4 bytes
    return sum(t.numel() * (2 if t.is_floating_point() else 4) for t in tensors)


def weight_plan(backbone, head, seq):
    """(first, per-layer, last, per-chunk) fp16 bytes: what only the first
    chunk holds (the embedding), each layer, what only the last holds (the
    final norm and the head), and what every chunk repeats (the bucket's
    buffers)."""
    layers = backbone.chunk_layers()
    parts = backbone.chunk_parts(0, len(layers))
    first = _module_bytes(parts["embed_tokens"])
    last = _module_bytes(parts["norm"]) + sum(_module_bytes(m) for m in head_modules(head))
    every = _buffer_bytes(backbone.buffers(seq).values())
    return first, [_module_bytes(layer) for layer in layers], last, every


def head_modules(head):
    """The head's modules, as head.register() attaches them to a wrapper."""
    probe = torch.nn.Module()
    head.register(probe, 1)
    return list(probe.children())


def _chunk_bytes(cuts, first, per_layer, last, every):
    bounds = [0] + list(cuts) + [len(per_layer)]
    sizes = [every + sum(per_layer[lo:hi]) for lo, hi in zip(bounds, bounds[1:])]
    sizes[0] += first
    sizes[-1] += last
    return sizes


def _balanced(n, first, per_layer, last, every):
    """The cuts into n chunks that minimize the largest chunk."""
    layers = len(per_layer)
    best = None

    def search(start, left, cuts):
        nonlocal best
        if left == 1:
            sizes = _chunk_bytes(cuts, first, per_layer, last, every)
            if best is None or max(sizes) < best[0]:
                best = (max(sizes), list(cuts))
            return
        for c in range(start + 1, layers - left + 2):
            search(c, left - 1, cuts + [c])

    search(0, n, [])
    return best[1]


def plan_cuts(backbone, head, seq, spec="auto", budget=CHUNK_WEIGHT_BUDGET_BYTES):
    """Cuts for `spec`: "auto" (the fewest chunks under `budget`, balanced;
    none when the whole bucket fits),
    a chunk count (balanced), or a list of layer indices. Sized for bucket
    `seq`, whose buffers every chunk repeats; plan for the largest bucket."""
    if backbone.chunk_ports is None:
        raise ValueError(f"the {backbone.family} backbone can't be chunked")
    first, per_layer, last, every = weight_plan(backbone, head, seq)
    layers = len(per_layer)
    if isinstance(spec, (list, tuple)):
        cuts = sorted(int(c) for c in spec)
        if not cuts or cuts[0] <= 0 or cuts[-1] >= layers or len(set(cuts)) != len(cuts):
            raise ValueError(f"chunk cuts {spec} must be distinct layers between 1 and {layers - 1}")
        return cuts
    if spec == "auto":
        if max(_chunk_bytes([], first, per_layer, last, every)) <= budget:
            return []
        for n in range(2, layers + 1):
            cuts = _balanced(n, first, per_layer, last, every)
            if max(_chunk_bytes(cuts, first, per_layer, last, every)) <= budget:
                return cuts
        raise ValueError(f"no split into whole layers fits {budget / 2**30:.2f} GiB per chunk")
    n = int(spec)
    if not 2 <= n <= layers:
        raise ValueError(f"a chunk count must be between 2 and {layers}, got {n}")
    return _balanced(n, first, per_layer, last, every)


def parse_spec(text):
    """--chunks: "auto", a count ("2"), or cuts ("10,20")."""
    if text in (None, "", "1"):
        return None
    if text == "auto":
        return "auto"
    if "," in text:
        return [int(c) for c in text.split(",")]
    return int(text)


def chunk_sizes(backbone, head, seq, cuts):
    """Each chunk's planned fp16 weight bytes."""
    return _chunk_bytes(cuts, *weight_plan(backbone, head, seq))


@dataclasses.dataclass
class Chunk:
    """One chunk to convert: its module, its inputs in order, its output."""
    index: int
    lo: int
    hi: int
    module: torch.nn.Module
    inputs: tuple
    output: str


class ChunkWrapper(torch.nn.Module):
    """A chunk's traced module: the backbone's parts for layers [lo, hi),
    the bucket's buffers, and the head on the last chunk. Its forward takes
    `names` as positional inputs, like wrapper.Wrapper."""

    def __init__(self, backbone, head, names, seq, lo, hi):
        super().__init__()
        self.seq, self.lo, self.hi = seq, lo, hi
        self.first, self.last = lo == 0, hi == len(backbone.chunk_layers())
        for name, module in backbone.chunk_parts(lo, hi).items():
            setattr(self, name, module)
        for name, tensor in backbone.buffers(seq).items():
            self.register_buffer(name, tensor)
        if self.last:
            head.register(self, seq)
        self._names, self._backbone, self._head = tuple(names), backbone, head

    def forward(self, *inputs):
        x = dict(zip(self._names, inputs))
        h = self._backbone.chunk_call(self, x)
        if not self.last:
            return h
        return self._head.forward(self, x, _Finished(self._backbone, h))


class _Finished:
    """The backbone as the head sees it in the last chunk: its call returns
    the chunk's final, normed states."""

    def __init__(self, backbone, h):
        self._backbone, self._h = backbone, h

    def call(self, w, x):
        return types.SimpleNamespace(last_hidden_state=self._h)

    def __getattr__(self, name):
        return getattr(self._backbone, name)


def chunks(backbone, head, ports, seq, cuts, output):
    """The chunks of bucket `seq` split at `cuts`, each with its inputs:
    the backbone's ports it reads, `hidden_in` after the first, and the
    head's own ports on the last."""
    head.bind(backbone)
    names = [p.name for p in ports]
    own = [n for n in names if n in backbone.chunk_ports]
    head_only = [n for n in names if n not in backbone.chunk_ports]
    bounds = [0] + list(cuts) + [len(backbone.chunk_layers())]
    out = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        first, last = lo == 0, i == len(bounds) - 2
        inputs = (own if first else [HIDDEN_IN] + [n for n in own if n != "input_ids"]) + (head_only if last else [])
        out.append(Chunk(i, lo, hi, ChunkWrapper(backbone, head, inputs, seq, lo, hi).eval(), tuple(inputs),
                         output if last else HIDDEN_OUT))
    return out


def run_torch(chunk_list, feed):
    """The composed chunks on an int32 feed {port: array}, in fp32."""
    h = None
    with torch.no_grad():
        for c in chunk_list:
            args = [h if n == HIDDEN_IN else torch.from_numpy(feed[n]) for n in c.inputs]
            h = c.module(*args)
    return h


def trace_convert(chunk, seq, feed, hidden_size):
    """jit.trace and coremltools, as core.trace_convert, with `hidden_in`
    an fp16 [1, S, H] input and `hidden_out` an fp16 output."""
    import coremltools as ct
    hidden = (1, seq, hidden_size)
    with torch.no_grad():
        args = tuple(torch.zeros(hidden) if n == HIDDEN_IN else torch.from_numpy(feed[n]) for n in chunk.inputs)
        traced = torch.jit.trace(chunk.module, args)
    inputs = [ct.TensorType(name=n, shape=hidden, dtype=np.float16) if n == HIDDEN_IN
              else ct.TensorType(name=n, shape=feed[n].shape, dtype=np.int32) for n in chunk.inputs]
    output = (ct.TensorType(name=HIDDEN_OUT, dtype=np.float16) if chunk.output == HIDDEN_OUT
              else ct.TensorType(name=chunk.output))
    return ct.convert(traced, inputs=inputs, outputs=[output], convert_to="mlprogram",
                      minimum_deployment_target=ct.target.macOS15, skip_model_load=True)


@dataclasses.dataclass
class Chain:
    """A bucket's compiled chunks, in order, with each chunk's inputs: what
    the gates, the compute plan and the parity checks read in place of one
    compiled path."""
    paths: list
    inputs: list
    output: str

    def __len__(self):
        return len(self.paths)


class ChainModel:
    """A chain loaded for one compute unit, with predict() like a compiled
    model's: each chunk's output feeds the next chunk's `hidden_in`. Every
    input is kept referenced (gates._INPUTS explains why)."""

    def __init__(self, chain, units, keep):
        import coremltools as ct
        self.chain, self.keep = chain, keep
        self.models = [ct.models.CompiledMLModel(str(p), compute_units=getattr(ct.ComputeUnit, units))
                       for p in chain.paths]

    def predict(self, feed):
        h = None
        for i, (model, names) in enumerate(zip(self.models, self.chain.inputs)):
            f = {n: (h if n == HIDDEN_IN else feed[n]) for n in names}
            self.keep.append(f)
            out = model.predict(f)
            self.keep.append(out)
            if i == len(self.models) - 1:
                return out
            h = out[HIDDEN_OUT]


def manifest_text(text, chunks, budget):
    """An installed manifest's text for a chunked conversion: `artifact`
    with a `{chunk}` placeholder, and the [chunking] table (D37)."""
    old = 'artifact = "model_{seq}.mlmodelc"'
    if old not in text:
        raise ValueError(f"the manifest's artifact isn't {old!r}, so the chunked name can't be derived")
    text = text.replace(old, 'artifact = "model_{seq}.{chunk}.mlmodelc"')
    budget_line = "" if budget is None else f"weight_budget_bytes = {budget}\n"
    return text.rstrip("\n") + f"\n\n[chunking]\nchunks = {chunks}\n{budget_line}"


def artifact_name(seq, index):
    return f"model_{seq}.{index}.mlmodelc"
