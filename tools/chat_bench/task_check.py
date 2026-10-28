"""The causal-LM task check (docs/DECISIONS.md D54; docs/CONVERTING.md,
"Grading"): a converted chat model against the checkpoint it came from, on
the same seeded sample of the chat benchmark, scored by the benchmark's own
scorers.

    SAMPLE=extraction=24,titles=20,tags_vocab=20,instructions=20
    python tools/chat_bench/run.py minicpm5-2b-fp32 --sample $SAMPLE --out <dir>
    python tools/chat_bench/run.py minicpm5-2b-ane --base-url http://127.0.0.1:<port> --sample $SAMPLE --out <dir>
    python tools/chat_bench/task_check.py <dir>/minicpm5-2b-ane.json <dir>/minicpm5-2b-fp32.json [--json out.json]

The candidate is the build served by sidekickd; the reference is the
checkpoint in fp32 transformers on the CPU (serve_ref.py). Both run greedy,
prompt-only JSON, with the same requests. Never run the reference while
sidekickd holds the build: they would compete for memory and the ANE's
neighbours.

For each family it reports the headline metric of both runs (titles:
ROUGE-L; fixed-vocabulary tags: F1; extraction: field accuracy, item mean;
instructions: items passing every check) and the candidate's difference,
with the share of replies identical to the reference's. A family passes
when the candidate is no worse than the reference by more than its
tolerance (TOLERANCE); better is never a failure. The check passes when
every family does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_bench import score, tasks  # noqa: E402

TOLERANCE = {
    "titles": ("rouge_l", 0.03, "ROUGE-L, at most 3 points below"),
    "tags_vocab": ("f1", 0.03, "F1, at most 3 points below"),
    "tags_free": ("concept_recall", 0.03, "concept recall, at most 3 points below"),
    "extraction": ("item_field_acc", 0.02, "field accuracy (item mean), at most 2 points below"),
    "instructions": ("all_checks", None, "items passing every check, at most 1 fewer"),
}


def check(candidate: dict, reference: dict) -> dict:
    keys_c = [(r["family"], r["item"]) for r in candidate["records"]]
    keys_r = [(r["family"], r["item"]) for r in reference["records"]]
    if keys_c != keys_r:
        raise SystemExit("the two runs answer different requests: run both with the same --sample and --seed")
    cfg, items, schemas = tasks.load_config(), tasks.load_items(), tasks.load_schemas()
    emb = score.Embedder(None, "none", Path("/nonexistent"))
    fc, pc = score.score_run(candidate, cfg, items, schemas, emb)
    fr, pr = score.score_run(reference, cfg, items, schemas, emb)
    content_c = {(r["family"], r["item"]): r["content"] for r in candidate["records"]}
    content_r = {(r["family"], r["item"]): r["content"] for r in reference["records"]}
    out = {"families": {}, "passed": True}
    for fam in fc:
        key, tol, rule = TOLERANCE[fam]
        vc, vr = fc[fam][key]["value"], fr[fam][key]["value"]
        n = len(pc[fam])
        same = sum(content_c[(fam, i)] == content_r[(fam, i)] for i in pc[fam]) / n
        if tol is None:                                 # count items: at most one fewer
            ic = sum(bool(v[key]) for v in pc[fam].values())
            ir = sum(bool(v[key]) for v in pr[fam].values())
            ok = ic >= ir - 1
            detail = {"candidate_items": ic, "reference_items": ir}
        else:
            ok = vc >= vr - tol
            detail = {}
        out["families"][fam] = {"metric": key, "candidate": vc, "reference": vr, "difference": vc - vr,
                                "rule": rule, "passed": ok, "n": n, "identical_replies": round(same, 4), **detail}
        out["passed"] &= ok
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--json", type=Path, help="write the result here")
    a = ap.parse_args()
    res = check(json.loads(a.candidate.read_text()), json.loads(a.reference.read_text()))
    for fam, f in res["families"].items():
        extra = f" ({f['candidate_items']} against {f['reference_items']} items)" if "candidate_items" in f else ""
        print(f"{fam:13s} {f['metric']:15s} candidate {f['candidate']:.3f}  reference {f['reference']:.3f}  "
              f"difference {f['difference']:+.3f}{extra}  identical replies {f['identical_replies']:.0%}  "
              f"{'pass' if f['passed'] else 'FAIL'} ({f['rule']})")
    print("task check:", "pass" if res["passed"] else "FAIL")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
