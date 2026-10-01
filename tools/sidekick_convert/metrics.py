"""NaN-safe metrics for gates and reports.

A non-finite value is a failure, never a number to fold away: Python's
`min(worst, nan)` returns `worst`, which once hid a NaN-producing CPU path
behind "parity 1.000000" (docs/DECISIONS.md D25). Every helper here returns
NaN when its input is non-finite, and every gate treats NaN as failing.
"""

import numpy as np


def finite(*arrays):
    """True when every element of every array is finite."""
    return all(np.isfinite(np.asarray(a, dtype=np.float64)).all() for a in arrays)


def cosine(a, b):
    """Cosine similarity in float64; NaN when either side is non-finite or zero."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if not finite(a, b):
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def worst(values):
    """min() that keeps NaN. An empty sequence is NaN too: a gate with no
    cases has measured nothing."""
    values = [float(v) for v in values]
    if not values or any(np.isnan(v) for v in values):
        return float("nan")
    return min(values)


def largest(values):
    """max() that keeps NaN; NaN for an empty sequence."""
    values = [float(v) for v in values]
    if not values or any(np.isnan(v) for v in values):
        return float("nan")
    return max(values)


def max_abs_diff(a, b):
    """max |a - b| in float64; NaN when either side is non-finite."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not finite(a, b):
        return float("nan")
    return float(np.abs(a - b).max())


def activate(z, activation):
    """A classifier's activation in float64: "softmax", "sigmoid" or "identity"
    (transformers' text-classification rule, docs/DECISIONS.md D28)."""
    z = np.asarray(z, dtype=np.float64)
    if activation == "softmax":
        e = np.exp(z - z.max())
        return e / e.sum()
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if activation == "identity":
        return z
    raise ValueError(f"unknown activation {activation!r}")
