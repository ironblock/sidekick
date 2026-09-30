"""Traceable replacements for shape arithmetic.

Under static input shapes, coremltools 9 crashes on the 'int' op that
jit.trace emits for arithmetic on tensor sizes ("only 0-dimensional arrays
can be converted to Python scalars"). Stock rotate_half slices at
x.shape[-1] // 2 and stock repeat_kv reshapes to num_kv_heads * n_rep, and
both trace to that op (docs/DECISIONS.md D17 constraint 8). The versions
here are exact and need no shape arithmetic.

The general rule, in any traced code: never compute with x.size() or
x.shape; pass sizes as Python ints and use literal dims in reshapes. That
includes a pooled output's final reshape(1, dims).
"""

import torch


def rotate_half(x):
    """Identical to transformers' rotate_half for even head dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    """Identical layout to transformers' repeat_kv, via expand + flatten."""
    if n_rep == 1:
        return hidden_states
    return hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)


def install(module, names=("rotate_half", "repeat_kv")):
    """Point a transformers modeling module's helpers at these versions.
    Patch every module that holds its own reference: the sdpa integration
    keeps a separate repeat_kv, for example."""
    for name in names:
        if hasattr(module, name):
            setattr(module, name, globals()[name])
