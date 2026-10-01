"""The ANE linear's 2^15 output limit (docs/DECISIONS.md D25 amendment).

The ANE's linear op saturates above 2^15 = 32,768: an output of 32,000 comes
back exact and 33,000 as inf, whether it comes from one term or thousands.
Its add, mul and layer_norm handle fp16's full range, so this limit is the
linear's own, and fp16's 65,504 isn't the one that matters.

Rule: keep every calibrated linear output at or below HEADROOM x 2^15.

Where a massive activation crosses it (ModernBERT's dimension 251, written by
layer 15's MLP output projection at up to ~51,500), run the residual stream
at 1/K: the output projections take 1/K, whatever reads the embedding
directly takes K, and scale-invariant norms need only eps / K^2. Which
weights take K is architecture-specific, so each backbone applies it
(backbones.modernbert.residual_rewrite). K is the smallest power of two that
fits, or pinned by the recipe (laya pins K = 2 for headroom). Larger K costs
precision on every path, so it is kept minimal.

A post-norm model (BERT) has no residual stream to scale, so its recipe only
checks the rule and fails when it doesn't hold.
"""

from ..core import GateFailure

ANE_LINEAR_MAX = 32768.0
HEADROOM = 0.85
K_MAX = 8


def check(maxima, headroom=HEADROOM):
    """Fail if any calibrated linear output ({name: max |out|}) is past the
    rule. Returns (largest name, its value, headroom factor)."""
    top = max(maxima, key=maxima.get)
    if maxima[top] > headroom * ANE_LINEAR_MAX:
        raise GateFailure(f"linear {top} reaches {maxima[top]:.0f}, past {headroom} x 2^15 "
                          "(the ANE linear saturates above 2^15)")
    return top, maxima[top], ANE_LINEAR_MAX / maxima[top]


def choose_k(fixed, scaled, headroom=HEADROOM, k_max=K_MAX):
    """Smallest power of two K <= k_max with every output in range, where
    `fixed` is the largest output that doesn't scale with the residual stream
    and `scaled` the largest one that scales with 1/K. Returns (K, headroom
    factor over the largest output at that K)."""
    limit = headroom * ANE_LINEAR_MAX
    if fixed > limit:
        raise GateFailure(f"an unscaled linear output reaches {fixed:.0f}, past the ANE linear's range")
    k = 1
    while scaled / k > limit:
        k *= 2
        if k > k_max:
            raise GateFailure(f"output projections reach {scaled:.0f}; no K <= {k_max} fits")
    return k, ANE_LINEAR_MAX / max(fixed, scaled / k)


def headroom_at(fixed, scaled, k):
    """Headroom factor under 2^15 at a pinned K."""
    return ANE_LINEAR_MAX / max(fixed, scaled / k)
