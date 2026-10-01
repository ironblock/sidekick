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
  bias) is one operation, rounded once. Attention is not: fused attention
  runs in the explicit form every converted program uses (D25), matmul,
  scale, mask, softmax, matmul, each output stored, whatever attention
  implementation the model loaded with. (PyTorch's own math decomposition
  scales q and k separately, two roundings no converted program has.)
  nn.TransformerEncoderLayer's fast path, one fused operation per layer, is
  turned off for the same reason.
- A normalization is one operation: fp32 inside, its output stored in fp16.
  A converter computes a norm whole or pre-scales it, so an intermediate
  such as RMSNorm's x^2 (4.4e7 for a 6,600 activation) never sits in fp16;
  rounding it would zero the token and model a loss no converted program
  has. Normalizations are the modules whose class name ends in RMSNorm or
  LayerNorm (run() and ideal_fp16(module=) find them).
- Everything that does not depend on the input is a constant: computed
  exactly, as a converter folds it in fp32, and stored in fp16 once, where
  an input-dependent operation reads it. Weights, buffers, and tables built
  from them (RoPE's cos and sin from positions x inv_freq) are constants.
  Rounding RoPE's angles op by op would put position 500 off by up to 0.25
  radians, a loss no converted program has.
- Python scalars (eps, a 1/sqrt(d) written as a float) are applied exactly.

A ceiling exists only if fp16 storage keeps every value meaningful. An
infinity or NaN may flow only through operations that handle it exactly
(masking arithmetic, softmax, comparisons, data movement): that is how a
finfo.min mask fill, -inf in fp16, does its job. Reaching any other
input-dependent operation (a reduction, a matmul, a square root) it would
corrupt the result while the output can stay finite, so the run raises
Fp16Overflow, naming the operation and where the value came from: the model
has no ideal-fp16 ceiling as run.

`ops` restricts the output rounding to some aten operations, by name, for
diagnosis only (which operations cost what; CLASSES groups them). Constants
are always stored in fp16. A ceiling for grading rounds every operation.
"""

import contextlib
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils.weak import WeakTensorKeyDictionary

# aten operation names by class, for diagnosis with `ops=`
CLASSES = {
    "linear": {"addmm", "mm", "linear"},
    "matmul": {"bmm", "matmul", "baddbmm"},
    "softmax": {"_softmax", "_safe_softmax", "softmax"},
    "layer_norm": {"native_layer_norm", "layer_norm"},
}

_FLOATS = (torch.float32, torch.float64)
_FP16_OVERFLOW = 65520.0  # the smallest magnitude fp16 rounds to infinity

# normalization modules, by class-name suffix: each runs as one operation
NORMS = ("RMSNorm", "LayerNorm")

# operations that handle an infinity or NaN in their inputs exactly
_NONFINITE_OK = frozenset({
    "add", "sub", "rsub", "mul", "neg", "masked_fill", "where", "maximum", "minimum", "clamp", "clamp_min",
    "clamp_max", "max", "min", "amax", "amin", "exp", "_softmax", "_safe_softmax", "softmax",
    "eq", "ne", "lt", "le", "gt", "ge", "isinf", "isnan", "isfinite", "logical_not", "logical_and",
    "logical_or", "any", "all",
    "view", "_unsafe_view", "reshape", "expand", "permute", "transpose", "t", "slice", "select", "index",
    "index_select", "gather", "cat", "stack", "clone", "contiguous", "_to_copy", "copy_", "copy", "alias",
    "detach", "unsqueeze", "squeeze", "split", "split_with_sizes", "unbind", "narrow", "repeat", "flatten",
    "unflatten", "as_strided", "lift_fresh", "fill_", "zero_",
})


class Fp16Overflow(ArithmeticError):
    """fp16 storage produced an infinity or NaN that reached an operation it
    corrupts: the model has no ideal-fp16 ceiling as run."""

# every scaled_dot_product_attention reaches this op under ideal_fp16(),
# which allows only the flash backend; it runs as explicit attention instead
_FUSED_ATTENTION = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default


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
        self._origin = WeakTensorKeyDictionary()  # where a non-finite value came from
        self._exact = 0                           # depth inside a normalization

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

        name = func.overloadpacket.__name__
        self._check_finite(func, name, args, kwargs)

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

        if func is _FUSED_ATTENTION:
            out = self._explicit_attention(*args, **kwargs)
        else:
            out = func(*args, **kwargs)
        if not schema.returns:
            return out
        rounding = self._exact == 0 and (self.ops is None or name in self.ops)
        inherited = self._inherited(args, kwargs)
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
                        origin = self._overflow(name, t, inherited)
                        _round_in_place(t)  # an in-place or out= result
                    elif rounding and not aliased:
                        origin = self._overflow(name, t, inherited)
                        t = round_fp16(t)  # a view of a stored tensor is already fp16
                    else:
                        origin = inherited
                    self._dynamic[t] = True
                    if origin is not None:
                        self._origin[t] = origin
                new.append(t)
            rounded.append(pytree.tree_unflatten(new, spec))
        return rounded[0] if single else type(out)(rounded)

    def _check_finite(self, func, name, args, kwargs):
        """Raise Fp16Overflow when a non-finite input-dependent value reaches
        an operation that doesn't handle it exactly. Fused attention handles
        it in its mask only."""
        if name in _NONFINITE_OK:
            return
        checked = args[:3] if func is _FUSED_ATTENTION else (args, kwargs)
        for t in pytree.tree_leaves(checked):
            if (self._is_dynamic(t) and t.dtype in _FLOATS and t.numel()
                    and not bool(torch.isfinite(t).all())):
                origin = self._origin.get(t) or "a non-finite input"
                raise Fp16Overflow(f"{name} reads an infinity or NaN ({origin}): the model has no "
                                   "ideal-fp16 ceiling as run")

    def _inherited(self, args, kwargs):
        for t in pytree.tree_leaves((args, kwargs)):
            if isinstance(t, torch.Tensor) and t in self._origin:
                return self._origin[t]
        return None

    def _overflow(self, name, t, inherited):
        """Where a value about to be stored in fp16 turns non-finite, if it
        does: this operation, when its exact result passes fp16's range."""
        if t.dtype not in _FLOATS or not t.numel():
            return inherited
        big = t.abs().amax()
        if bool(big < _FP16_OVERFLOW):
            return inherited
        finite = t[torch.isfinite(t)].abs()
        if finite.numel() and bool(finite.amax() >= _FP16_OVERFLOW):
            return f"{name} stored |x| up to {float(finite.amax()):.3g} in fp16, past 65,504"
        return inherited or f"{name} computed a non-finite value"

    def _norm_enter(self, module, args):
        self._exact += 1

    def _norm_exit(self, module, args, out):
        """A normalization's output, stored in fp16 once."""
        self._exact -= 1
        if self._exact:
            return out
        name = type(module).__name__
        with _disable_current_modes():
            def store(t):
                if not (isinstance(t, torch.Tensor) and self._is_dynamic(t)):
                    return t
                origin = self._overflow(name, t, self._origin.get(t))
                t = round_fp16(t)
                self._dynamic[t] = True
                if origin is not None:
                    self._origin[t] = origin
                return t
            return pytree.tree_map(store, out)

    def _store(self, name, t):
        return round_fp16(t) if self.ops is None or name in self.ops else t

    def _explicit_attention(self, query, key, value, dropout_p=0.0, is_causal=False, *, attn_mask=None,
                            scale=None):
        """scaled_dot_product_attention as a converted program computes it:
        matmul, scale, mask, softmax, matmul, each output stored in fp16.
        Returns (output, logsumexp), as the fused operation does."""
        if dropout_p:
            raise ValueError("ideal fp16 runs a model in eval mode: attention dropout must be 0")
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])
        if key.shape[-3] != query.shape[-3]:  # grouped-query attention: each kv head serves a group
            group = query.shape[-3] // key.shape[-3]
            key, value = key.repeat_interleave(group, dim=-3), value.repeat_interleave(group, dim=-3)
        scores = self._store("bmm", torch.matmul(query, key.transpose(-2, -1)))
        scores = self._store("mul", scores * scale)
        if is_causal:
            keep = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool).tril()
            scores = self._store("masked_fill", scores.masked_fill(~keep, float("-inf")))
        if attn_mask is not None:
            scores = self._store("add", scores + attn_mask)
        probs = self._store("_softmax", torch.softmax(scores, dim=-1))
        return self._store("bmm", torch.matmul(probs, value)), torch.logsumexp(scores, dim=-1)


@contextlib.contextmanager
def ideal_fp16(ops=None, module=None):
    """Within the context, input-dependent operations run as an ideal fp16
    engine. Pass the model as `module`, so its normalizations run as single
    operations, and each model input through the yielded mode's input():

        with fp16sim.ideal_fp16(module=model) as sim:
            out = model(input_ids=sim.input(ids), attention_mask=sim.input(mask))

    Run the model in eval mode. Raises Fp16Overflow when the model has no
    ideal-fp16 ceiling (see the module docstring)."""
    mode = IdealFp16(ops)
    hooks = []
    for m in module.modules() if module is not None else ():
        if type(m).__name__.endswith(NORMS):
            hooks += [m.register_forward_pre_hook(mode._norm_enter), m.register_forward_hook(mode._norm_exit)]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.no_grad(), mode:
            yield mode
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for h in hooks:
            h.remove()


def run(module, *args, ops=None, **kwargs):
    """module(*args, **kwargs) as an ideal fp16 engine, every tensor argument
    an input (see ideal_fp16)."""
    with ideal_fp16(ops, module) as sim:
        args, kwargs = pytree.tree_map(sim.input, (args, kwargs))
        return module(*args, **kwargs)
