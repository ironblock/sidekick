"""Score the verification checkers' runs into docs/data/verification-scores.json,
and render its MODELS.md section (design §3).

    python3 tools/tasks/scoring_report.py --runs <runs dir> \\
        [--batch <judge batch dir> --responses <responses.jsonl>] \\
        [--out docs/data/verification-scores.json] [--models-md docs/MODELS.md]

Inputs: the runs written by tools/tasks/scoring_run.py; optionally the judge
batch (tools/tasks/scoring_export.py) and its responses, which give J and
the second reader in each form; and the committed ceilings
(docs/data/verification-ceilings.json).

Per checker: the control gate (§6.1), triple accuracy (three-way, or
binary for two-way checkers, never mixed) with a shape-cluster bootstrap CI
and the score as a share of the checker's own ceiling, per-variant accuracy,
the confusion matrix, AUROC, descriptive cells by obstacle, decision rule,
claim type and length, and a §2.5 verdict. Comparisons: each gated checker
against its ungated form, and each checker against the second reader that
shadows it, as paired differences with the MDE. The report leads with the
MDE: this round is a pilot, and no ranking is claimed. Its notes (also in
the MODELS.md section) say what each kind of checker is asked: the LLM
readers' prompt defines the corpus's rules for current evidence, while the
nli checkers get the claim and evidence only; the 512-token reader's windows
match nli-deberta-v3-base's. J's and the second reader's parse rates,
lenient (scored) and strict, are reported per reader form.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, scoring_export, scoring_rules as rules, scoring_set  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "verification-scores.json"
CEILINGS = REPO / "docs" / "data" / "verification-ceilings.json"
SCHEMA = "sidekick-verification-scores/1"
MD_START = "<!-- verification-scores:start -->"
MD_END = "<!-- verification-scores:end -->"
READER_FORMS = {"window-512": "R512", "window-2048": "R2048", "chunk-400": "R400", "full": "J"}
LEAD = ("This round is a pilot: {triples} triples from {shapes} claim shapes. Its minimum detectable effect "
        "(MDE, 2.8 × the bootstrap SE of a paired difference) is about {mde} points of triple accuracy, so "
        "differences between checkers smaller than that cannot be seen. Results are per checker, each with its "
        "verdict; no ranking is claimed until the corpus reaches about 150 claims.")
NOTES = [
    "The task, as the LLM judge (J), Foundation Models and the second reader are asked it, includes the corpus's "
    "rules for which evidence is current: a later record of the same thing supersedes an earlier one (by timestamp "
    "where records carry one); a `was: X …, wrong because Y` note refutes X; an override applies over what it "
    "overrides; a tool's help and output speak for the version that produced them. These rules define the labels "
    "for every item alike. The nli checkers, MiniCheck and HHEM receive only the claim and the evidence, as their "
    "interfaces take nothing else.",
    "The 512-token second reader sees windows cut with nli-deberta-v3-base's tokenizer, so it shadows that model "
    "(and deberta-v3-large-zeroshot-v2.0-c, whose cuts differ slightly) closely, and nli-minilm2-l6-h768 and "
    "roberta-large-mnli only approximately: their own tokenizers cut the evidence at other places.",
    "J's and the second reader's verdicts are read from the final answer only. The scored (lenient) rule accepts "
    "one JSON object at the end of the answer, with prose before it; the strict rule needs the answer to be exactly "
    "that object. Both parse rates are reported per reader form; a failure counts as wrong. A verdict object inside "
    "a Markdown code fence fails both rules and is counted on its own line, so its frequency can be reviewed.",
]


# ---------------------------------------------------------------- readers


def reader_runs(requests: list[dict], responses: dict, tau: float) -> list[dict]:
    """J and the second reader as run records, aggregated by the max rule."""
    runs = []
    rates = scoring_export.parse_rates(requests, responses)
    for form, short in READER_FORMS.items():
        wins = scoring_export.reader_windows(requests, responses, form)
        if not wins:
            continue
        for gated in ((False, True) if form.startswith("window") else (False,)):
            out = []
            for item, ws in sorted(wins.items()):
                use = [w for w in ws if w["anchored"]] if gated else ws
                agg = rules.three_way([rules.reader_probs(w["verdict"]) for w in use], tau)
                out.append({"item": item, "verdict": agg["verdict"], "E": agg["E"], "C": agg["C"],
                            "windows": len(ws), "scored_windows": len(use), "no_anchor": gated and not use,
                            "parse_failures": sum(1 for w in ws if not w["parsed"])})
            name = short + ("g" if gated else "")
            runs.append({"checker": name, "family": name, "mode": "three-way", "items": out, "reader_form": form,
                         "provenance": {"model": responses["model"], "missing": len(responses["missing"]),
                                        "bad_sha256": len(responses["bad_sha"]),
                                        "errors": len(responses.get("errors", [])),
                                        "bad_prompt": len(responses.get("bad_prompt", []))},
                         "parse_rates": rates.get(form)})
    return runs


SHADOW = {"A3": "R512", "A3g": "R512g", "A2": "R2048", "A2g": "R2048g", "M": "R400", "F": "J", "H": "J"}


# ---------------------------------------------------------------- scoring one checker


def ceiling_of(run: dict, ceilings: dict | None) -> float | None:
    if ceilings is None:
        return None
    fam = run["family"]
    model = run["checker"].split(":", 1)[1] if ":" in run["checker"] else None
    binary = run["mode"] == "binary"
    if fam == "J":
        return 1.0
    if fam in ("F", "H"):  # whole-haystack checkers: the share of triples that fit (all variants run)
        rs = [i for i in run["items"] if not i["item"].startswith("ctl:")]
        by = defaultdict(list)
        for i in rs:
            by[scoring_set.split_item(i["item"])[0]].append(not i.get("not_run"))
        return sum(all(v) for v in by.values()) / len(by) if by else None
    lookup = {"R512": "nli-deberta-v3-base", "R512g": "nli-deberta-v3-base", "R2048": "bge-m3-zeroshot-v2.0-c",
              "R2048g": "bge-m3-zeroshot-v2.0-c", "R400": "minicheck-deberta-v3-large"}
    model = lookup.get(fam, model)
    ch = ceilings["checkers"].get(model)
    if ch is None:
        return None
    src = ch["binary"] if binary else ch
    key = "gated_window_reachable" if fam.endswith("g") else "window_reachable"
    return src.get(key, src["window_reachable"])["overall"]["share"]


def score_run(run: dict, triples: list[dict], cfg: dict, ceilings: dict | None) -> dict:
    by_item = {i["item"]: i for i in run["items"]}
    mode = run["mode"]
    answers = ("positive", "contradiction", "null") if mode == "three-way" else ("positive", "null")
    ctl = []
    for item, r in by_item.items():
        if item.startswith("ctl:"):
            _, _cid, group, form = item.split(":")
            ctl.append({"group": group, "form": form, "gold": {"positive": "S", "contradiction": "C",
                                                               "null": "NM"}[group], "verdict": r["verdict"],
                        "not_run": bool(r.get("not_run"))})
    gate = rules.control_gate(ctl, answers, cfg["gate"]["short"], cfg["gate"]["long"])
    correct, shape_of, pairs, cells = {}, {}, [], defaultdict(list)
    e_pos, e_neg, c_pos, c_neg = [], [], [], []
    per_variant = {"S": [0, 0], "C": [0, 0], "NM": [0, 0]}
    for t in triples:
        vs = {v: by_item.get(scoring_set.item_id(t["id"], v), {}).get("verdict") for v in scoring_set.VARIANTS}
        if all(by_item.get(scoring_set.item_id(t["id"], v)) is None for v in scoring_set.VARIANTS):
            continue
        ok = rules.triple_correct(vs, mode)
        correct[t["id"]] = ok
        shape_of[t["id"]] = t["claim"]["shape"]
        for dim, val in (("obstacle", t["config"]["obstacle"]), ("mechanism", t["claim"]["mechanism"]),
                         ("type", t["claim"]["type"]), ("length", t["config"]["length"])):
            cells[(dim, val)].append(ok)
        for v in scoring_set.VARIANTS:
            g = scoring_set.GOLD[v]
            ans = vs[v]
            r = by_item.get(scoring_set.item_id(t["id"], v), {})
            right = ans == g if mode == "three-way" else rules.binary(ans) == (rules.S if g == "S" else rules.NOT_S)
            per_variant[g][0] += 1
            per_variant[g][1] += int(right)
            pairs.append((g, ans))
            if r.get("E") is not None:
                (e_pos if g == "S" else e_neg).append(r["E"])
            if r.get("C") is not None and g in ("C", "NM"):
                (c_pos if g == "C" else c_neg).append(r["C"])
    if not correct:
        return {"checker": run["checker"], "mode": mode, "gate": gate, "triples": 0}
    point = sum(correct.values()) / len(correct)
    draws = rules.shape_bootstrap(correct, shape_of, cfg["resamples"], cfg["seed"])
    acc = rules.summary(point, draws)
    acc["chance"] = rules.CHANCE[mode]
    ceil = ceiling_of(run, ceilings)
    acc["ceiling"] = ceil
    acc["share_of_ceiling"] = round(point / ceil, 4) if ceil else None
    acc.update(rules.single_verdict(gate["passed"]))
    key = "triple_accuracy" if mode == "three-way" else "binary_triple_accuracy"
    out = {"checker": run["checker"], "family": run["family"], "mode": mode, "gate": gate, "triples": len(correct),
           key: acc,
           "per_variant": {g: {"right": r, "of": n, "accuracy": round(r / n, 4) if n else None}
                           for g, (n, r) in per_variant.items()},
           "auroc": {"E_S_vs_rest": rules.auroc(e_pos, e_neg),
                     "C_C_vs_NM": rules.auroc(c_pos, c_neg) if mode == "three-way" else None},
           "descriptive": {f"{d}:{v}": {"triples": len(oks), "accuracy": round(sum(oks) / len(oks), 4)}
                           for (d, v), oks in sorted(cells.items())},
           "no_anchor": sum(1 for i in run["items"] if i.get("no_anchor")),
           "not_run": sum(1 for i in run["items"] if i.get("not_run")),
           "parse_failures": sum(1 for i in run["items"] if i.get("parse_ok") is False)
                             + sum(i.get("parse_failures", 0) for i in run["items"]),
           "provenance": run.get("provenance", {}), "_correct": correct}
    if mode == "three-way":
        out["confusion"] = rules.confusion(pairs)
        out["binary_view"] = {"triples_right": sum(rules.triple_correct(
            {v: by_item.get(scoring_set.item_id(t, v), {}).get("verdict") for v in scoring_set.VARIANTS}, "binary")
            for t in correct), "of": len(correct)}
    else:
        out["three_way"] = "not applicable"
    return out


# ---------------------------------------------------------------- the report


def build(runs: list[dict], triples: list[dict], cfg: dict, ceilings: dict | None) -> dict:
    scored = {r["checker"]: score_run(r, triples, cfg, ceilings) for r in runs}
    shape_of = {t["id"]: t["claim"]["shape"] for t in triples}
    comparisons = []

    def pair(x, y, why):
        sx, sy = scored[x], scored[y]
        if not sx.get("_correct") or not sy.get("_correct") or sx["mode"] != sy["mode"]:
            return
        cmp = rules.compare(sx["_correct"], sy["_correct"], shape_of, cfg["resamples"], cfg["seed"],
                            cfg["mde_factor"])
        k = "triple_accuracy" if sx["mode"] == "three-way" else "binary_triple_accuracy"
        loser = x if cmp["value"] < 0 else y
        shadow = SHADOW.get(scored[loser]["family"])
        share = None
        if shadow and shadow in scored and scored[shadow]["gate"]["passed"]:
            winner = y if loser == x else x
            disc = [t for t in sx["_correct"] if scored[winner]["_correct"].get(t) and
                    not scored[loser]["_correct"].get(t)]
            rc = scored[shadow].get("_correct", {})
            share = sum(1 for t in disc if not rc.get(t, False)) / len(disc) if disc else None
        v = rules.comparison_verdict(cmp, sx["gate"]["passed"], sy["gate"]["passed"], sx[k].get("ceiling"),
                                     sy[k]["value"], sy[k].get("ceiling"), sx[k]["value"], share)
        comparisons.append({"x": x, "y": y, "what": why, **cmp, **v})

    for name, s in scored.items():
        fam = s.get("family", "")
        if fam in ("A3g", "A2g", "R512g", "R2048g"):
            pair(name, name.replace(fam, fam[:-1], 1), "gated against ungated")
        shadow = SHADOW.get(fam)
        if shadow in scored and scored[shadow]["gate"]["passed"]:
            pair(name, shadow, "checker against the second reader given the same inputs")
    mdes = [c["mde"] for c in comparisons]
    mde = round(100 * sum(mdes) / len(mdes)) if mdes else None
    for s in scored.values():
        s.pop("_correct", None)
    parse_rates = {r["reader_form"]: r["parse_rates"] for r in runs if r.get("reader_form") and r.get("parse_rates")}
    return {"schema": SCHEMA, "status": "pilot",
            "lead": LEAD.format(triples=len(triples), shapes=len(set(shape_of.values())),
                                mde=mde if mde is not None else "20–25 (a priori)"),
            "notes": list(NOTES), "reader_parse_rates": parse_rates,
            "set": cfg["set"], "tau": cfg["tau"], "mde_points_mean": mde,
            "ceilings_sha256": canon.sha256_file(CEILINGS) if CEILINGS.exists() else None,
            "checkers": scored, "comparisons": comparisons}


def render_md(scores: dict) -> str:
    """The MODELS.md section."""
    lines = [MD_START, "", "### Verification checkers (pilot)", "", scores["lead"], ""]
    lines += [f"- {n}" for n in scores.get("notes", [])] + [""]
    rates = scores.get("reader_parse_rates") or {}
    if rates:
        lines += ["| reader form | requests | parsed (lenient, scored) | parsed (strict) "
                  "| fenced (scored as failures) |", "|---|---|---|---|---|"]
        lines += [f"| {f} | {r['requests']} | {r['lenient']:.3f} | {r['strict']:.3f} | {r.get('fenced', 0):.3f} |"
                  for f, r in rates.items()]
        lines.append("")
    lines += ["| checker | mode | controls | triple accuracy | 95% CI | chance | share of ceiling | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(scores["checkers"].items()):
        acc = s.get("triple_accuracy") or s.get("binary_triple_accuracy")
        if not acc:
            continue
        lines.append(f"| {name} | {s['mode']} | {'pass' if s['gate']['passed'] else 'fail'} | {acc['value']:.3f} | "
                     f"{acc['ci95'][0]:.3f}–{acc['ci95'][1]:.3f} | {acc['chance']:.3f} | "
                     f"{acc['share_of_ceiling'] if acc['share_of_ceiling'] is not None else '–'} | {acc['verdict']} |")
    if scores["comparisons"]:
        lines += ["", "| X | Y | what | difference | 95% CI | MDE | verdict |", "|---|---|---|---|---|---|---|"]
        for c in scores["comparisons"]:
            lines.append(f"| {c['x']} | {c['y']} | {c['what']} | {c['value']:+.3f} | {c['ci95'][0]:+.3f}–"
                         f"{c['ci95'][1]:+.3f} | {c['mde']:.3f} | {c['verdict']} |")
    lines += ["", "Two-way checkers report binary triple accuracy (S against not-S), never three-way. Generated by "
              "tools/tasks/scoring_report.py from docs/data/verification-scores.json.", "", MD_END]
    return "\n".join(lines) + "\n"


def splice_md(text: str, section: str) -> str:
    if MD_START in text and MD_END in text:
        a, b = text.index(MD_START), text.index(MD_END) + len(MD_END) + 1
        return text[:a] + section + text[b:]
    return text.rstrip("\n") + "\n\n" + section


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--batch", type=Path, help="the judge batch directory (with requests.jsonl)")
    ap.add_argument("--responses", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--models-md", type=Path, help="splice the section into this MODELS.md")
    a = ap.parse_args(argv)
    cfg = scoring_set.config()
    triples, _claims, _lock = scoring_set.load(cfg["set"])
    runs = [json.loads(p.read_text()) for p in sorted(a.runs.glob("*.json"))]
    if not runs and not a.responses:
        raise SystemExit("no runs to score")
    if a.responses:
        reqs = canon.read_jsonl(a.batch / "requests.jsonl")
        resp = scoring_export.read_responses(a.responses, reqs)
        runs += reader_runs(reqs, resp, cfg["tau"])
    ceilings = json.loads(CEILINGS.read_text()) if CEILINGS.exists() else None
    scores = build(runs, triples, cfg, ceilings)
    canon.write_json(a.out, scores)
    print(f"wrote {a.out}")
    if a.models_md:
        a.models_md.write_text(splice_md(a.models_md.read_text(), render_md(scores)))
        print(f"updated {a.models_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
