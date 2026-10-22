"""Verdict rules, metrics and statistics for scoring the verification corpus.

Per-window aggregation (design §7.5), over the windows a checker scored:
- **three-way** (`nli_probs` from a three-way model, or a reader's per-window
  verdicts as 0/1): E = max entailment, C = max contradiction. S if E ≥ τ
  and E ≥ C; else C if C ≥ τ; else NM.
- **two-way** (e − c): E = max over windows of sigmoid(e − c), the
  two-class entailment probability. S if E ≥ τ, else not-S. A two-way
  checker never answers C or NM, and its three-way result is "not
  applicable", never zero.
- **gated**: the same, over only the windows holding an anchor; with none,
  the answer is NM (not-S) and is counted as "no anchor".

Scores (design §2.3–2.5, §6):
- triple accuracy (all three variants right), three-way and binary kept
  apart; binary maps S → supported and C, NM → unsupported;
- per-variant accuracy and the 3×3 confusion matrix;
- AUROC of E for S against {C, NM}, and of C for C against NM;
- a cluster bootstrap over claim shapes (10,000 resamples, seed 133), the
  paired difference's SE, MDE = 2.8 × that SE;
- the control gate: 6/6 short and 5/6 long in every group a checker answers;
- one of three verdicts per number (measured, measured negative,
  instrument-limited), by §2.5's rules. A comparison whose CI contains 0 but
  is wider than ±MDE fits none of them and is reported as inconclusive.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

S, C, NM, NOT_S = "S", "C", "NM", "not-S"
LABELS3 = (S, C, NM)
CHANCE = {"three-way": 1 / 27, "binary": 1 / 8}


# ---------------------------------------------------------------- aggregation


def three_way(window_probs: list[dict], tau: float) -> dict:
    """Aggregate per-window {entailment, neutral, contradiction}."""
    if not window_probs:
        return {"verdict": NM, "E": 0.0, "C": 0.0, "windows": 0}
    e = max(p["entailment"] for p in window_probs)
    c = max(p["contradiction"] for p in window_probs)
    if e >= tau and e >= c:
        v = S
    elif c >= tau:
        v = C
    else:
        v = NM
    return {"verdict": v, "E": e, "C": c, "windows": len(window_probs)}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def two_way(window_entail: list[float], tau: float) -> dict:
    """Aggregate per-window p(entailment) = sigmoid(e − c)."""
    if not window_entail:
        return {"verdict": NOT_S, "E": 0.0, "windows": 0}
    e = max(window_entail)
    return {"verdict": S if e >= tau else NOT_S, "E": e, "windows": len(window_entail)}


def gate(values: list, mask: list[bool]) -> tuple[list, bool]:
    """The windows a gated checker scores, and whether none held an anchor."""
    kept = [x for x, m in zip(values, mask) if m]
    return kept, not kept


def reader_probs(verdict: str | None) -> dict:
    """A reader's per-window verdict as probabilities for the three-way rule
    (a missing or unparsable answer counts as neither)."""
    return {"entailment": float(verdict == S), "neutral": float(verdict == NM),
            "contradiction": float(verdict == C)}


def binary(verdict: str | None) -> str | None:
    if verdict is None:
        return None
    return S if verdict == S else NOT_S


# ---------------------------------------------------------------- metrics


def triple_correct(verdicts: dict[str, str | None], mode: str) -> bool:
    """verdicts: {"s": .., "c": .., "nm": ..}; a missing verdict is wrong."""
    if mode == "three-way":
        return verdicts.get("s") == S and verdicts.get("c") == C and verdicts.get("nm") == NM
    return (binary(verdicts.get("s")) == S and binary(verdicts.get("c")) == NOT_S
            and binary(verdicts.get("nm")) == NOT_S)


def confusion(pairs: list[tuple[str, str | None]]) -> dict:
    """3×3 (gold × answer), plus a column for no answer."""
    m = {g: {a: 0 for a in (*LABELS3, "none")} for g in LABELS3}
    for gold, ans in pairs:
        m[gold][ans if ans in LABELS3 else "none"] += 1
    return m


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann–Whitney AUROC, ties counted half."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # ranks are 1-based
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- bootstrap


def shape_bootstrap(correct: dict[str, bool], shape_of: dict[str, str], resamples: int = 10000,
                    seed: int = 133) -> list[float]:
    """Triple accuracy under a cluster bootstrap over shapes. `correct` is
    per triple id. Returns the resampled accuracies."""
    return paired_bootstrap({"x": correct}, shape_of, resamples, seed)["x"]


def paired_bootstrap(correct: dict[str, dict[str, bool]], shape_of: dict[str, str], resamples: int = 10000,
                     seed: int = 133) -> dict[str, list[float]]:
    """The same shape draws for every checker, so differences are paired."""
    shapes = sorted({shape_of[t] for t in shape_of})
    per = {k: defaultdict(lambda: [0, 0]) for k in correct}
    for k, cm in correct.items():
        for t, ok in cm.items():
            cell = per[k][shape_of[t]]
            cell[0] += 1
            cell[1] += int(ok)
    r = random.Random(seed)
    out = {k: [] for k in correct}
    for _ in range(resamples):
        draw = [shapes[r.randrange(len(shapes))] for _ in shapes]
        for k in correct:
            n = sum(per[k][s][0] for s in draw)
            out[k].append(sum(per[k][s][1] for s in draw) / n if n else float("nan"))
    return out


def sd(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(point: float, draws: list[float]) -> dict:
    lo, hi = percentile(draws, 0.025), percentile(draws, 0.975)
    return {"value": point, "ci95": [lo, hi], "half_width": (hi - lo) / 2, "se": sd(draws)}


def compare(x: dict[str, bool], y: dict[str, bool], shape_of: dict[str, str], resamples: int = 10000,
            seed: int = 133, mde_factor: float = 2.8) -> dict:
    """Paired comparison of two checkers on their common triples."""
    common = sorted(set(x) & set(y))
    so = {t: shape_of[t] for t in common}
    b = paired_bootstrap({"x": {t: x[t] for t in common}, "y": {t: y[t] for t in common}}, so, resamples, seed)
    diff = [a - c for a, c in zip(b["x"], b["y"])]
    point = (sum(x[t] for t in common) - sum(y[t] for t in common)) / len(common)
    s = summary(point, diff)
    s["mde"] = mde_factor * s["se"]
    s["triples"] = len(common)
    return s


# ---------------------------------------------------------------- the control gate


def control_gate(results: list[dict], answers: tuple[str, ...], short: int = 6, long: int = 5,
                 exclude_not_run: bool = True) -> dict:
    """results: [{"group", "form", "gold", "verdict"}]. `answers` is the
    groups the checker answers (two-way checkers: positive and null). A
    two-way answer of not-S counts as right for the null group.

    With `exclude_not_run` (scoring.toml `[gate]`, a declared deviation), a
    control marked `not_run` (over the checker's context) is left out. A
    group and form with none run is "not run" (`passed` None) and doesn't
    count against the gate; one with some run must still reach its share of
    those run. Without it, a control not run counts as wrong."""
    if not exclude_not_run:
        results = [{**r, "not_run": False} for r in results]
    cells = defaultdict(lambda: [0, 0, 0])
    for r in results:
        if r["group"] not in answers:
            continue
        cell = cells[(r["group"], r["form"])]
        if r.get("not_run"):
            cell[2] += 1
            continue
        right = r["verdict"] == r["gold"] or (r["gold"] == NM and r["verdict"] == NOT_S)
        cell[0] += 1
        cell[1] += int(right)
    out, ok = {}, True
    for g in answers:
        for form, need in (("short", short), ("long", long)):
            n, right, skipped = cells.get((g, form), [0, 0, 0])
            if n == 0 and skipped:
                out[f"{g}/{form}"] = {"right": 0, "of": 0, "not_run": skipped, "need": need, "passed": None}
                continue
            need_here = need if not skipped else math.ceil(need * n / (n + skipped))
            passed = n > 0 and right >= need_here
            ok &= passed
            out[f"{g}/{form}"] = {"right": right, "of": n, "not_run": skipped, "need": need_here, "passed": passed}
    return {"passed": ok, "groups": out}


# ---------------------------------------------------------------- verdicts (§2.5)


MEASURED = "measured"
NEGATIVE = "measured negative"
LIMITED = "instrument-limited"
INCONCLUSIVE = "inconclusive"


def single_verdict(gate_passed: bool) -> dict:
    if not gate_passed:
        return {"verdict": LIMITED, "why": "failed a control group"}
    return {"verdict": MEASURED, "why": "passed its controls; read against its ceiling and the MDE"}


def comparison_verdict(cmp: dict, gate_x: bool, gate_y: bool, ceiling_x: float | None, score_y: float,
                       ceiling_y: float | None = None, score_x: float | None = None,
                       reader_fails_share: float | None = None) -> dict:
    """§2.5 for X against Y. `reader_fails_share` is the share of the
    discordant triples (one checker right, the other wrong) that the second
    reader, given the losing checker's inputs, also fails; a majority makes
    the comparison instrument-limited."""
    why = []
    if not gate_x or not gate_y:
        why.append("a checker failed a control group")
    if ceiling_x is not None and ceiling_x < score_y:
        why.append("X's ceiling is below Y's score")
    if ceiling_y is not None and score_x is not None and ceiling_y < score_x:
        why.append("Y's ceiling is below X's score")
    if reader_fails_share is not None and reader_fails_share > 0.5:
        why.append("the second reader, given the same inputs, also fails most of the triples in question")
    if why:
        return {"verdict": LIMITED, "why": "; ".join(why)}
    lo, hi = cmp["ci95"]
    if lo > 0 or hi < 0:
        return {"verdict": MEASURED, "why": "the paired difference's CI excludes 0"}
    if -cmp["mde"] <= lo and hi <= cmp["mde"]:
        return {"verdict": NEGATIVE, "why": f"the CI lies within ±MDE ({cmp['mde']:.3f}); nothing smaller is ruled out"}
    return {"verdict": INCONCLUSIVE, "why": "the CI contains 0 and reaches beyond ±MDE"}
