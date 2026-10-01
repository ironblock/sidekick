"""Activation statistics for the rewrites that need them.

Rewrites that depend on activation ranges (the residual K, input rescales)
are calibrated in fp32 PyTorch on a `core.Calibration`: unpadded forwards,
one text at a time. Never on the graded parity corpus (core.Calibration
refuses its texts).
"""

import torch


class Stat:
    """rms over every element seen, and max |x|."""

    def __init__(self):
        self.sumsq, self.count, self.max = 0.0, 0, 0.0

    def add(self, t):
        t = t.detach().double()
        self.sumsq += float(t.pow(2).sum())
        self.count += t.numel()
        self.max = max(self.max, float(t.abs().max()))

    @property
    def rms(self):
        return (self.sumsq / self.count) ** 0.5 if self.count else float("nan")

    def __repr__(self):
        return f"Stat(rms={self.rms:.4g}, max={self.max:.4g})"


def collect(run, sites):
    """Run `run()` with hooks at `sites`, {name: (module, where)}, where is
    "in" (the module's first input) or "out" (its output). Returns {name: Stat}."""
    stats = {name: Stat() for name in sites}
    hooks = []
    for name, (module, where) in sites.items():
        if where == "in":
            hooks.append(module.register_forward_pre_hook(lambda m, args, n=name: stats[n].add(args[0])))
        elif where == "out":
            hooks.append(module.register_forward_hook(lambda m, args, out, n=name: stats[n].add(out)))
        else:
            raise ValueError(f"site {name}: where must be 'in' or 'out', not {where!r}")
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hooks:
            h.remove()
    return stats


def linear_maxima(run, modules):
    """max |output| of each linear in `modules` ({name: nn.Linear}) during `run()`."""
    stats = collect(run, {name: (m, "out") for name, m in modules.items()})
    return {name: s.max for name, s in stats.items()}


def all_linears(model):
    """{qualified name: nn.Linear} for every linear in a model."""
    return {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
