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

Deviations from the pre-registered design are declared in scoring.toml
(`[[deviation]]`) and listed in the header and the section; one with a
`beside` form (chunk-400) puts its note beside every number of that form.
Verdicts are never relabelled. Beside each failed checker's verdict, the
diagnosis (design §7) is computed from the runs: whether the second reader
passed its controls through the same windows and max rule, the scope
(max-aggregation at τ), what gating changed, and Foundation Models' answers
on NOT MENTIONED variants.
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
        "(MDE, 2.8 × the bootstrap SE of a paired difference) is about {mde} of triple accuracy, so "
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
    "one verdict object at the end of the answer, with prose before it (a JSON record quoted from the evidence is "
    "prose); the strict rule needs the answer to be exactly that object. Both are reported per reader form, with "
    "the lenient rule's literal reading beside them; a failure counts as wrong. A verdict object inside a Markdown "
    "code fence fails both rules and is counted on its own line, so its frequency can be reviewed. An answer cut "
    "off by the output budget (finish_reason `length`) is wrong, and is listed by id.",
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


SHADOW = {"M": "R400", "F": "J", "H": "J"}


def shadow_of(checker: str, family: str, cfg: dict) -> str | None:
    """The second-reader form a checker stands beside: an nli checker by its
    window size (512 → R512, 2048 → R2048; gated → the gated form),
    MiniCheck by its chunks, the whole-haystack checkers by J."""
    if family in ("A3", "A3g", "A2", "A2g"):
        model = checker.split(":", 1)[-1]
        seq = next((m["max_seq_len"] for m in cfg.get("nli", []) if m["id"] == model), None)
        if seq is None:
            return None
        return f"R{seq}" + ("g" if family.endswith("g") else "")
    return SHADOW.get(family)


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
    gate = rules.control_gate(ctl, answers, cfg["gate"]["short"], cfg["gate"]["long"],
                              cfg["gate"].get("exclude_not_run", False))
    correct, verdicts, shape_of, pairs, cells = {}, {}, {}, [], defaultdict(list)
    e_pos, e_neg, c_pos, c_neg = [], [], [], []
    per_variant = {"S": [0, 0], "C": [0, 0], "NM": [0, 0]}
    for t in triples:
        vs = {v: by_item.get(scoring_set.item_id(t["id"], v), {}).get("verdict") for v in scoring_set.VARIANTS}
        if all(by_item.get(scoring_set.item_id(t["id"], v)) is None for v in scoring_set.VARIANTS):
            continue
        ok = rules.triple_correct(vs, mode)
        correct[t["id"]] = ok
        verdicts[t["id"]] = {scoring_set.VARIANT_NAME[v]: vs[v] for v in scoring_set.VARIANTS}
        shape_of[t["id"]] = t["claim"]["shape"]
        for dim, val in (("obstacle", t["config"]["obstacle"]), ("mechanism", t["claim"]["mechanism"]),
                         ("type", t["claim"]["type"]), ("length", t["config"]["length"]),
                         ("nm_kind", t.get("nm_kind", "removed"))):
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
           "provenance": run.get("provenance", {}), "_correct": correct, "_verdicts": verdicts}
    if mode == "three-way":
        out["confusion"] = rules.confusion(pairs)
        out["_correct_binary"] = {t: rules.triple_correct(
            {v: by_item.get(scoring_set.item_id(t, v), {}).get("verdict") for v in scoring_set.VARIANTS}, "binary")
            for t in correct}
        out["binary_view"] = {"triples_right": sum(out["_correct_binary"].values()), "of": len(correct)}
    else:
        out["_correct_binary"] = correct
        out["three_way"] = "not applicable"
    return out


# ---------------------------------------------------------------- the report


def build(runs: list[dict], triples: list[dict], cfg: dict, ceilings: dict | None) -> dict:
    scored = {r["checker"]: score_run(r, triples, cfg, ceilings) for r in runs}
    shape_of = {t["id"]: t["claim"]["shape"] for t in triples}
    comparisons = []

    def pair(x, y, why):
        """X against Y on the same triples. When their modes differ (a
        three-way reader shadowing a two-way checker), both are compared on
        binary terms, as the two-way checker answers."""
        sx, sy = scored[x], scored[y]
        if not sx.get("_correct") or not sy.get("_correct"):
            return
        mode = sx["mode"] if sx["mode"] == sy["mode"] else "binary"
        key = "_correct" if sx["mode"] == sy["mode"] else "_correct_binary"
        cx, cy = sx[key], sy[key]
        cmp = rules.compare(cx, cy, shape_of, cfg["resamples"], cfg["seed"], cfg["mde_factor"])
        k = "triple_accuracy" if mode == "three-way" else "binary_triple_accuracy"
        loser = x if cmp["value"] < 0 else y
        shadow = shadow_of(loser, scored[loser]["family"], cfg)
        share = None
        if shadow and shadow in scored and scored[shadow]["gate"]["passed"]:
            cw, cl = (cy, cx) if loser == x else (cx, cy)
            disc = [t for t in cx if cw.get(t) and not cl.get(t)]
            rc = scored[shadow].get(key, {})
            share = sum(1 for t in disc if not rc.get(t, False)) / len(disc) if disc else None

        def acc(s, c):  # (value, ceiling) on the comparison's terms
            own = s.get(k)
            return sum(c.values()) / len(c), (own or {}).get("ceiling")

        (vx, cx_ceil), (vy, cy_ceil) = acc(sx, cx), acc(sy, cy)
        v = rules.comparison_verdict(cmp, sx["gate"]["passed"], sy["gate"]["passed"], cx_ceil, vy, cy_ceil, vx, share)
        comparisons.append({"x": x, "y": y, "what": why, "mode": mode, **cmp, **v})

    for name, s in scored.items():
        fam = s.get("family", "")
        if fam in ("A3g", "A2g", "R512g", "R2048g"):
            pair(name, name.replace(fam, fam[:-1], 1), "gated against ungated")
        shadow = shadow_of(name, fam, cfg)
        if shadow in scored and scored[shadow]["gate"]["passed"]:
            pair(name, shadow, "checker against the second reader given the same inputs")
    # The MDE the report leads with: the median over the paired comparisons
    # whose difference varies at all (two checkers answering identically on
    # every triple give an SE, and so an MDE, of 0).
    mdes = sorted(c["mde"] for c in comparisons if c["se"] > 0)
    mde = round(100 * mdes[len(mdes) // 2]) if mdes else None
    mde_range = [round(100 * mdes[0]), round(100 * mdes[-1])] if mdes else None
    # Each checker's answers on the items the ceilings carry label notes for
    # (the version-scoped 045 and 050, and 020 near a position boundary).
    label_items = []
    for note in (ceilings or {}).get("label_notes", []):
        row = {"n": note["n"], "id": note["id"], "note": note["note"], "answers": {}}
        for name, s in sorted(scored.items()):
            v = s.get("_verdicts", {}).get(note["id"])
            if v is not None:
                row["answers"][name] = {**v, "triple_right": s["_correct"][note["id"]]}
        label_items.append(row)
    findings = gray_zone_findings(scored, triples, cfg)
    for s in scored.values():
        s.pop("_correct", None)
        s.pop("_correct_binary", None)
        s.pop("_verdicts", None)
    parse_rates = {r["reader_form"]: dict(r["parse_rates"]) for r in runs
                   if r.get("reader_form") and r.get("parse_rates")}
    deviations = [dict(d) for d in cfg.get("deviation", [])]
    # Each deviation that must stand beside the numbers of a form (chunk-400).
    form_of = {r["checker"]: r.get("reader_form") for r in runs}
    for d in deviations:
        if not d.get("beside"):
            continue
        for name, s in scored.items():
            if form_of.get(name) == d["beside"]:
                s.setdefault("notes", []).append(d["note"])
        for c in comparisons:
            if d["beside"] in (form_of.get(c["x"]), form_of.get(c["y"])):
                c.setdefault("notes", []).append(d["note"])
        if d["beside"] in parse_rates:
            parse_rates[d["beside"]].setdefault("notes", []).append(d["note"])
    diag = diagnosis(scored, runs, comparisons, cfg)
    return {"schema": SCHEMA, "status": "pilot",
            "lead": LEAD.format(triples=len(triples), shapes=len(set(shape_of.values())),
                                mde=(f"{mde} points ({mde_range[0]}–{mde_range[1]} across the paired comparisons)"
                                     if mde is not None else "20–25 points (a priori)")),
            "mde_points_range": mde_range,
            "deviations": [{"id": d["id"], "text": d["text"]} for d in deviations],
            "diagnosis": diag,
            "notes": list(NOTES), "reader_parse_rates": parse_rates,
            "set": cfg["set"], "tau": cfg["tau"], "mde_points_median": mde,
            "ceilings_sha256": canon.sha256_file(CEILINGS) if CEILINGS.exists() else None,
            "checkers": scored, "comparisons": comparisons, "label_items": label_items, "findings": findings}


def gray_zone_findings(scored: dict, triples: list[dict], cfg: dict) -> list[dict]:
    """The reviewer's findings for the next corpus round (scoring.toml
    `[[finding]]`), each with what the judge did on it. A finding naming an
    `nm_kind` says which of J's NOT MENTIONED → S misses fall in that group.
    This round's labels and scores don't change."""
    nums = scoring_set.review_numbers(triples)
    kind = {t["id"]: t.get("nm_kind", "removed") for t in triples}
    j = scored.get("J", {}).get("_verdicts", {})
    out = []
    for f in cfg.get("finding", []):
        row = {"id": f["id"], "text": f["text"]}
        if f.get("nm_kind") and j:
            miss = sorted((nums[t], t) for t, v in j.items() if v["NM"] == "S")
            inside = [n for n, t in miss if kind[t] == f["nm_kind"]]
            if miss:
                row["judge"] = (f"J's NOT MENTIONED → S misses are on {', '.join(f'{n:03d}' for n, _ in miss)}; "
                                f"{len(inside)} of {len(miss)} fall in this group"
                                + (", where a visible nonzero exit supports “the call failed”."
                                   if inside else "."))
            row["triples"] = sum(1 for t in kind.values() if t == f["nm_kind"])
        out.append(row)
    return out


WINDOW_FAMILIES = ("A3", "A3g", "A2", "A2g", "M")
READER_OF_WINDOWS = ("R512", "R512g", "R2048", "R2048g")


def attribution_text(a: dict, header: bool = False) -> str:
    """One `[[attribution]]` entry as the report words it, by its status:
    "attributed" (cause attributed by the audit), "candidates" (cause not
    yet attributed; candidates from the audit) or "unattributed" (the text
    says so itself). `header` is the line under the family's label;
    otherwise the checker's own line, after "instrument-limited; "."""
    status = a.get("status", "attributed")
    if status == "candidates":
        lead = "candidate: " if header else "cause not yet attributed; candidates from the audit: "
        return (("cause not yet attributed; " if header else "") + lead + a["candidate"] + ". " + a["support"])
    if status == "unattributed":
        return a["text"]
    return a["text"] if header else f"cause attributed by the audit: {a['label']}: {a['text']}"


def diagnosis(scored: dict, runs: list[dict], comparisons: list[dict], cfg: dict) -> list[str]:
    """The diagnosis beside the instrument-limited verdicts (design §7). The
    verdicts themselves are never changed; this says why, from the runs and
    the audit's declared attributions (scoring.toml `[audit]`,
    `[[attribution]]`). Each failed checker gets "instrument-limited; cause
    not yet attributed" with the open candidates, unless an attribution
    names its family. Also attaches each failed checker's own lines as
    `diagnosis`."""
    tau = cfg["tau"]
    audit = cfg.get("audit", {})
    attributions = cfg.get("attribution", [])
    out = []
    readers = [scored[n] for n in READER_OF_WINDOWS if n in scored]
    passed = [r for r in readers if r["gate"]["passed"]]
    failed_window = [s for s in scored.values() if s.get("family") in WINDOW_FAMILIES and not s["gate"]["passed"]]
    harness = None
    if passed and failed_window:
        right = sum(v["right"] for v in passed[0]["gate"]["groups"].values())
        of = sum(v["of"] for v in passed[0]["gate"]["groups"].values())
        harness = (f"The second reader passes {right}/{of} controls through identical windows and the same max rule "
                   f"({', '.join(r['checker'] for r in passed)}). "
                   + audit.get("second_reader", "That rules out the information missing from the windows and the "
                               "max rule losing it, not each checker's own interface, label mapping, tokenizer cuts "
                               "or τ."))
        out.append(harness)
    if audit.get("construction"):
        out.append(audit["construction"])
    scope = (f"Scope: these results hold under max-aggregation at τ = {tau}, not for NLI on tool output in "
             "general. A calibrated threshold, or a contradiction rule that needs corroboration, was not applied; "
             "tuning waits until the corpus can be tuned on dev.")
    gating = []
    for c in comparisons:
        if c["what"] == "gated against ungated" and c["x"] in scored and c["y"] in scored:
            gx, ux = scored[c["x"]], scored[c["y"]]
            k = "triple_accuracy" if gx["mode"] == "three-way" else "binary_triple_accuracy"
            if gx.get(k) and ux.get(k) and gx[k]["value"] != ux[k]["value"]:
                gating.append(f"{c['y'].split(':', 1)[-1]}: {ux[k]['value']:.2f} ungated, {gx[k]['value']:.2f} gated")
    if failed_window:
        out.append(scope + (" Gating changes the result (" + "; ".join(gating) + ")." if gating else ""))
    fm = next((r for r in runs if r.get("family") == "F"), None)
    fm_line = None
    if fm:
        nm = [i for i in fm["items"] if i["item"].endswith(":nm") and not i.get("not_run")]
        if nm:
            k = sum(1 for i in nm if i.get("verdict") == "C")
            fm_line = (f"Foundation Models answers C on {k} of the {len(nm)} NOT MENTIONED variants that fit its "
                       "context: it reads an absence as a contradiction (a behaviour, not only a score).")
            out.append(fm_line)
    failed_fams = {s.get("family") for s in scored.values() if not s["gate"]["passed"]}
    open_ = audit.get("open", "interface, label mapping, tokenizer cuts, τ, domain")
    for a in attributions:
        if failed_fams & set(a["families"]):
            out.append(f"{a['label']}: {attribution_text(a, header=True)}")
    if failed_fams - {f for a in attributions for f in a["families"]}:
        out.append(f"Otherwise: instrument-limited; cause not yet attributed (open: {open_}).")
    for s in scored.values():
        if s["gate"]["passed"]:
            continue
        att = next((a for a in attributions if s.get("family") in a["families"]), None)
        lines = ["instrument-limited; " + (attribution_text(att) if att
                                           else f"cause not yet attributed (open: {open_})")]
        if s.get("family") in WINDOW_FAMILIES:
            lines += [x for x in (harness, scope) if x]
        if s.get("family") == "F" and fm_line:
            lines.append(fm_line)
        s["diagnosis"] = lines
    return out


def render_md(scores: dict) -> str:
    """The MODELS.md section."""
    lines = [MD_START, "", "### Verification checkers (pilot)", "", scores["lead"], ""]
    if scores.get("deviations"):
        lines += ["Deviations from the pre-registered design:", ""]
        lines += [f"{n}. {d['text']}" for n, d in enumerate(scores["deviations"], 1)] + [""]
    if scores.get("diagnosis"):
        lines += ["Diagnosis beside the instrument-limited verdicts (the verdicts are unchanged):", ""]
        lines += [f"- {d}" for d in scores["diagnosis"]] + [""]
    lines += [f"- {n}" for n in scores.get("notes", [])] + [""]
    rp = scores.get("reader_provenance")
    if rp:
        bits = [f"`{rp['registry_id']}`" if rp.get("registry_id") else None,
                f"served as `{rp['served_name']}`" if rp.get("served_name") else None]
        bits += [f"{k.replace('_', ' ')}: {v}" for k, v in rp.items() if k not in ("registry_id", "served_name")]
        lines += ["J and the second reader: " + "; ".join(b for b in bits if b) + ".", ""]

    def note(x):
        return f" ({'; '.join(x['notes'])})" if x.get("notes") else ""

    rates = scores.get("reader_parse_rates") or {}
    if rates:
        lines += ["Reader answers, by form (lenient is the scored rule; a failure counts as wrong):", "",
                  "| reader form | requests | strict | lenient only | failed | fenced | lenient, literal reading "
                  "| output budget ran out |", "|---|---|---|---|---|---|---|---|"]
        for f, r in rates.items():
            c = r.get("counts", {})
            cut = r.get("budget_truncations", [])
            lines.append(f"| {f}{note(r)} | {r['requests']} | {c.get('strict', '–')} | {c.get('lenient_only', '–')} | "
                         f"{c.get('failed', '–')} | {c.get('fenced', '–')} | {c.get('lenient_literal', '–')} | "
                         f"{len(cut)}{' (' + ', '.join(f'`{i}`' for i in cut) + ')' if cut else ''} |")
        lines.append("")
    lines += ["| checker | mode | controls | triple accuracy | 95% CI | chance | ceiling | share of ceiling | "
              "S / C / NM accuracy | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(scores["checkers"].items()):
        acc = s.get("triple_accuracy") or s.get("binary_triple_accuracy")
        if not acc:
            continue
        pv = s["per_variant"]
        share = f"{acc['share_of_ceiling']:.3f}" if acc.get("share_of_ceiling") is not None else "–"
        ceil = f"{acc['ceiling']:.3f}" if acc.get("ceiling") is not None else "–"
        lines.append(f"| {name}{note(s)} | {s['mode']} | {'pass' if s['gate']['passed'] else 'fail'} | "
                     f"{acc['value']:.3f} ({round(acc['value'] * s['triples'])}/{s['triples']}) | "
                     f"{acc['ci95'][0]:.3f}–{acc['ci95'][1]:.3f} | {acc['chance']:.3f} | {ceil} | {share} | "
                     f"{pv['S']['accuracy']:.2f} / {pv['C']['accuracy']:.2f} / {pv['NM']['accuracy']:.2f} | "
                     f"{acc['verdict']} |")
    if scores["comparisons"]:
        lines += ["", "| X | Y | what | difference | 95% CI | MDE | verdict |", "|---|---|---|---|---|---|---|"]
        for c in scores["comparisons"]:
            lines.append(f"| {c['x']} | {c['y']} | {c['what']}{' (binary terms)' if c.get('mode') == 'binary' else ''}"
                         f"{note(c)} | {c['value']:+.3f} | {c['ci95'][0]:+.3f}–"
                         f"{c['ci95'][1]:+.3f} | {c['mde']:.3f} | {c['verdict']} |")
    cols = [("obstacle", "distractor"), ("obstacle", "repeated"), ("mechanism", "plain"), ("mechanism", "sequence"),
            ("mechanism", "supersede"), ("mechanism", "precedence"), ("mechanism", "version"),
            ("mechanism", "timestamp")]
    lines += ["", "By obstacle and decision rule (triples right; descriptive only, no comparisons):", "",
              "| checker | " + " | ".join(v for _, v in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, s in sorted(scores["checkers"].items()):
        d = s.get("descriptive", {})
        cells = []
        for dim, v in cols:
            x = d.get(f"{dim}:{v}")
            cells.append(f"{round(x['accuracy'] * x['triples'])}/{x['triples']}" if x else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if scores.get("findings"):
        lines += ["", "Findings for the next corpus round (this round's labels and scores don't change):", ""]
        for f in scores["findings"]:
            lines.append(f"- {f['text']}" + (f" {f['judge']}" if f.get("judge") else ""))
    kinds = [("removed", "NOT MENTIONED removes the call"),
             ("empty-output-nonzero-exit", "keeps it, output emptied, nonzero exit")]
    lines += ["", "By NOT MENTIONED kind (triples right; the two groups scored apart, descriptive only):", "",
              "| checker | " + " | ".join(f"{k} ({d})" for k, d in kinds) + " |", "|---|" + "---|" * len(kinds)]
    for name, s in sorted(scores["checkers"].items()):
        d = s.get("descriptive", {})
        cells = []
        for k, _ in kinds:
            x = d.get(f"nm_kind:{k}")
            cells.append(f"{round(x['accuracy'] * x['triples'])}/{x['triples']}" if x else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for item in scores.get("label_items", []):
        lines += ["", f"Item {item['n']:03d} (`{item['id']}`): {item['note']}", "",
                  "| checker | S | C | NM | triple right |", "|---|---|---|---|---|"]
        for name, a in item["answers"].items():
            cells = " | ".join(a[v] if a[v] is not None else "not run" for v in ("S", "C", "NM"))
            lines.append(f"| {name} | {cells} | {'yes' if a['triple_right'] else 'no'} |")
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
    ap.add_argument("--reader", action="append", default=[], metavar="KEY=VALUE",
                    help="the readers' model provenance, given at run time and never kept in code: registry_id, "
                         "quantization, context, reasoning_effort, sampler (the served name is read from the "
                         "responses). Name no host, endpoint or machine.")
    a = ap.parse_args(argv)
    cfg = scoring_set.config()
    triples, _claims, _lock = scoring_set.load(cfg["set"])
    runs = [json.loads(p.read_text()) for p in sorted(a.runs.glob("*.json"))]
    if not runs and not a.responses:
        raise SystemExit("no runs to score")
    reader_prov = None
    if a.responses:
        reqs = canon.read_jsonl(a.batch / "requests.jsonl")
        resp = scoring_export.read_responses(a.responses, reqs)
        runs += reader_runs(reqs, resp, cfg["tau"])
        reader_prov = {"served_name": resp["model"], **dict(x.split("=", 1) for x in a.reader)}
    ceilings = json.loads(CEILINGS.read_text()) if CEILINGS.exists() else None
    scores = build(runs, triples, cfg, ceilings)
    if reader_prov:
        scores["reader_provenance"] = reader_prov
    canon.write_json(a.out, scores)
    print(f"wrote {a.out}")
    if a.models_md:
        a.models_md.write_text(splice_md(a.models_md.read_text(), render_md(scores)))
        print(f"updated {a.models_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
