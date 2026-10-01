"""Ideal fp16: what a model loses to fp16 storage alone.

A converted model is an fp16 program: its weights are stored in fp16 and
every operation writes its output as an fp16 tensor. An ideal fp16 engine
runs that program and loses nothing else: each operation computes exactly
(here, in fp32) from its fp16 inputs, and its result is rounded to fp16
(round to nearest even) before anything reads it. A real engine loses more,
to its own arithmetic: accumulation precision, coarse activations, the ANE
linear's small-input floor (docs/CONVERTING.md). The ideal engine's error
against the fp32 reference is the ceiling a real path is graded against.

The definition (docs/CONVERTING.md, "Ideal fp16"):
- Every operation that depends on the input is rounded: its floating-point
  output is stored in fp16. Rounding happens at the dispatcher, below
  Python, so it reaches every operation however the model calls it
  (F.linear, @, torch.bmm, Tensor.softmax, +) with no per-architecture
  list. A fused kernel (F.layer_norm, F.gelu, softmax, a linear with its
  bias) is one operation, rounded once. Fused attention is not: it runs as
  its math decomposition, so scores and probabilities are stored like any
  other tensor.
- Everything that does not depend on the input is a constant: computed
  exactly, as a converter folds it in fp32, and stored in fp16 once, where
  an input-dependent operation reads it. Weights, buffers, and tables built
  from them (RoPE's cos and sin from positions x inv_freq) are constants.
  Rounding RoPE's angles op by op would put position 500 off by up to 0.25
  radians, a loss no converted program has.
- Python scalars (eps, a 1/sqrt(d) written as a float) are applied exactly.

`ops` restricts the output rounding to some aten operations, by name, for
diagnosis only (which operations cost what; CLASSES groups them). Constants
are always stored in fp16. A ceiling for grading rounds every operation.
"""

import contextlib

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary

# aten operation names by class, for diagnosis with `ops=`
CLASSES = {
    "linear": {"addmm", "mm", "linear"},
    "matmul": {"bmm", "matmul", "baddbmm"},
    "softmax": {"_softmax", "_safe_softmax", "softmax"},
    "layer_norm": {"native_layer_norm", "layer_norm"},
}

_FLOATS = (torch.float32, torch.float64)


def round_fp16(t):
    """A float tensor rounded to fp16 and back; anything else unchanged."""
    if isinstance(t, torch.Tensor) and t.dtype in _FLOATS:
        return t.to(torch.float16).to(t.dtype)
    return t


def _round_in_place(t):
    if isinstance(t, torch.Tensor) and t.dtype in _FLOATS:
        t.copy_(t.to(torch.float16))


class IdealFp16(TorchDispatchMode):
    """The dispatch mode behind ideal_fp16(). A tensor is input-dependent
    once input() marks it or an operation derives it from one that is."""

    def __init__(self, ops=None):
        super().__init__()
        self.ops = ops
        self._dynamic = WeakTensorKeyDictionary()

    def input(self, t):
        """Mark a model input as input-dependent; a float input is stored in
        fp16, as a copy."""
        if not isinstance(t, torch.Tensor):
            return t
        t = round_fp16(t.detach().clone()) if t.dtype in _FLOATS else t
        self._dynamic[t] = True
        return t

    def _is_dynamic(self, x):
        return isinstance(x, torch.Tensor) and x in self._dynamic

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not any(self._is_dynamic(x) for x in pytree.tree_leaves((args, kwargs))):
            return func(*args, **kwargs)  # a constant, folded exactly

        # constants an input-dependent operation reads are stored in fp16;
        # one it writes in place is rounded where it is
        schema = func._schema
        args, kwargs = list(args), dict(kwargs)
        for i, arg in enumerate(schema.arguments):
            if i < len(args):
                get, put = (lambda i=i: args[i]), (lambda v, i=i: args.__setitem__(i, v))
            elif arg.name in kwargs:
                get, put = (lambda n=arg.name: kwargs[n]), (lambda v, n=arg.name: kwargs.__setitem__(n, v))
            else:
                continue
            value = get()
            written = arg.alias_info is not None and arg.alias_info.is_write
            for t in pytree.tree_leaves(value):
                if isinstance(t, torch.Tensor) and not self._is_dynamic(t) and written:
                    _round_in_place(t)
            if not written:
                put(pytree.tree_map(lambda t: t if self._is_dynamic(t) else round_fp16(t), value))

        out = func(*args, **kwargs)
        if not schema.returns:
            return out
        rounding =self.ops is None or func.overloadpacket.__name__ in self.ops
        single = len(schema.returns) == 1  # one return may be a list (unbind, split)
        results = (out,) if single else out
        rounded = []
        for ret, value in zip(schema.returns, results):
            aliased = ret.alias_info is not None
            leaves, spec = pytree.tree_flatten(value)
            new = []
            for t in leaves:
                if isinstance(t, torch.Tensor):
                    if rounding and aliased and ret.alias_info.is_write:
                        _round_in_place(t)  # an in-place or out= result
                    elif rounding and not aliased:
                        t = round_fp16(t)  # a view of a stored tensor is already fp16
                    self._dynamic[t] = True
                new.append(t)
            rounded.append(pytree.tree_unflatten(new, spec))
        return rounded[0] if single else type(out)(rounded)


@contextlib.contextmanager
def ideal_fp16(ops=None):
    """Within the context, input-dependent operations run as an ideal fp16
    engine. Pass each model input through the yielded mode's input():

        with fp16sim.ideal_fp16() as sim:
            out = model(input_ids=sim.input(ids), attention_mask=sim.input(mask))

    Run the model in eval mode."""
    mode = IdealFp16(ops)
    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad(), mode:
        yield mode


def run(module, *args, ops=None, **kwargs):
    """module(*args, **kwargs) as an ideal fp16 engine, every tensor argument
    an input (see ideal_fp16)."""
    with ideal_fp16(ops) as sim:
        args, kwargs = pytree.tree_map(sim.input, (args, kwargs))
        return module(*args, **kwargs)
