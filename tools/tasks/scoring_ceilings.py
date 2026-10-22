"""The ceilings, computed before any checker runs (design §2.2).

No model runs: only the generator's slot records, the claims' probes and
each checker's tokenizer.

- **Window-reachable** (A, M): the share of triples all of whose variants
  (for binary scoring, SUPPORTED alone: `binary`) have their deciding spans (tools/tasks/scoring_windows.py) inside one of
  the checker's own windows. Per checker, since W and the tokenizer differ,
  and by decision rule (the claim's mechanism), obstacle and length. For
  MiniCheck, over its 400-token chunks approximated at line granularity.
- **Gate** (A3g, A2g): the share of triples whose deciding spans contain an
  anchor (`gate`), and, stricter, whose deciding spans share a window with
  an anchor in it (`gated_window_reachable`).
- **Context** (F): read from Foundation Models at run time (sidekickd's
  `/health` reports `chat.context_limit`); `context_ceiling` computes it
  from the F run's records. Here, only an approx-v1 estimate.
- **Label notes:** the version-scoped items (045 and 050 in the review
  index), with the stricter NOT MENTIONED reading in which the
  older-version note and the version line must share a window; and 020,
  labelled `start` under approx-v1, with its position by a real tokenizer.

    python3 tools/tasks/scoring_ceilings.py --tokenizer <id>=<tokenizer.json> ... \\
        [--out docs/data/verification-ceilings.json]

Needs the `tokenizers` package. Each tokenizer is checked against its
sha256 in fixtures/tasks/verification/scoring.toml.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, scoring_controls, scoring_set, scoring_windows as sw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "data" / "verification-ceilings.json"
SCHEMA = "sidekick-verification-ceilings/1"
LABEL_NOTES = {
    50: "version scope: NOT MENTIONED keeps the jj 0.30.0 note and `jj --version` printing 0.46.0; the note "
        "names its own version, so a reader that respects that scope answers NOT MENTIONED from the note alone "
        "(the primary ceiling). `nm_strict` is the stricter reading: the note and the version line in one window.",
    45: "version scope: the same construction with a dasel 2.8.1 note beside `dasel version` printing 3.11.2.",
    20: "labelled `start` under approx-v1 (position_at 0.318, 0.012 inside the boundary); flagged, not excluded. "
        "`position_by_tokenizer` gives the needle's position by each tokenizer's real token count.",
}


def variant_reach(t: dict, claim: dict, v: str, cuts: list[sw.Window], text: str, anchors: list[str]) -> dict:
    groups = sw.deciding_groups(t, claim, v, text)
    mask = sw.gated(cuts, anchors)
    return {"window": sw.reachable(cuts, groups),
            "gate": sw.gate_reachable(text, groups, anchors),
            "gated_window": sw.reachable(cuts, groups, mask),
            "windows": len(cuts), "anchored_windows": sum(mask)}


def share(rows: list[dict], key: str) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r[key])
    return {"triples": n, "reachable": k, "share": round(k / n, 4) if n else None}


def breakdown(rows: list[dict], key: str) -> dict:
    out = {"overall": share(rows, key)}
    for dim in ("mechanism", "obstacle", "length"):
        cells = defaultdict(list)
        for r in rows:
            cells[r[dim]].append(r)
        out[f"by_{dim}"] = {k: share(v, key) for k, v in sorted(cells.items())}
    return out


def ceilings_for(name: str, tok: sw.Tokenizer, triples: list[dict], claims: dict, cut) -> dict:
    """cut(text, claim) -> windows. Returns per-triple rows and breakdowns."""
    rows = []
    for t in triples:
        claim = scoring_set.claim_for(t, claims)
        anchors = sw.anchors(t["claim"]["text"])
        per = {}
        for v in scoring_set.VARIANTS:
            text = scoring_set.evidence(t, v)
            per[v] = variant_reach(t, claim, v, cut(text, t["claim"]["text"]), text, anchors)
        row = {"id": t["id"], "mechanism": t["claim"]["mechanism"], "obstacle": t["config"]["obstacle"],
               "length": t["config"]["length"],
               "variants": {scoring_set.VARIANT_NAME[v]: {k: per[v][k] for k in ("window", "gate", "gated_window")}
                            for v in per},
               "windows": {scoring_set.VARIANT_NAME[v]: per[v]["windows"] for v in per},
               "anchored_windows": {scoring_set.VARIANT_NAME[v]: per[v]["anchored_windows"] for v in per}}
        for key in ("window", "gate", "gated_window"):
            row[key] = all(per[v][key] for v in per)
            # Binary (two-way) scoring needs only SUPPORTED's spans: not-S on
            # C and NM needs nothing a window could lose.
            row[f"{key}_binary"] = per["s"][key]
        if claim["mechanism"] == "version":
            text = scoring_set.evidence(t, "nm")
            row["nm_strict"] = sw.reachable(cut(text, t["claim"]["text"]), sw.nm_strict_groups(claim, text))
        rows.append(row)
    return {"checker": name, "tokenizer_sha256": tok.sha256,
            "window_reachable": breakdown(rows, "window"), "gate": breakdown(rows, "gate"),
            "gated_window_reachable": breakdown(rows, "gated_window"),
            "binary": {"window_reachable": breakdown(rows, "window_binary"), "gate": breakdown(rows, "gate_binary"),
                       "gated_window_reachable": breakdown(rows, "gated_window_binary")},
            "triples": rows}


def control_checks(controls: list[dict], tok: sw.Tokenizer, cut) -> dict:
    """Each short control is one window; each long one keeps its block whole
    in some window."""
    out = {"short_single_window": True, "long_block_reachable": True, "failures": []}
    for c in controls:
        ws = cut(c["evidence"], c["claim"])
        if c["form"] == "short" and len(ws) != 1:
            out["short_single_window"] = False
            out["failures"].append(f"{c['item']}: {len(ws)} windows")
        span = scoring_controls.deciding_span(c)
        if c["form"] == "long" and span and not any(w.holds(span) for w in ws):
            out["long_block_reachable"] = False
            out["failures"].append(f"{c['item']}: block split across windows")
    return out


def position_by_tokenizer(t: dict, tok: sw.Tokenizer) -> float:
    text = scoring_set.evidence(t, "s")
    at = t["position_at"] * len(text)
    offs = tok.offsets(text)
    return round(sum(1 for s, _ in offs if s < at) / max(1, len(offs)), 3)


def context_ceiling(records: list[dict]) -> dict:
    """F's context ceiling from its run: the share of triples all of whose
    variants fit the model's context (records: {"item", "not_run"})."""
    by = defaultdict(list)
    for r in records:
        if not r["item"].startswith("ctl:"):
            by[scoring_set.split_item(r["item"])[0]].append(not r.get("not_run"))
    n = len(by)
    k = sum(1 for v in by.values() if all(v))
    return {"triples": n, "fit": k, "share": round(k / n, 4) if n else None}


def context_estimate(triples: list[dict], limit: int, overhead: int) -> dict:
    """Approx-v1 estimate only; the real count is the framework's."""
    k = sum(1 for t in triples if max(t["tokens"].values()) + overhead <= limit)
    return {"assumed_context_tokens": limit, "prompt_overhead_tokens": overhead, "fit": k,
            "triples": len(triples), "share": round(k / len(triples), 4), "unit": canon.LENGTH_UNIT}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", action="append", default=[], metavar="ID=PATH",
                    help="a checker id from scoring.toml (or `minicheck`) and its tokenizer.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--minicheck-exact", action="store_true",
                    help="cut MiniCheck's chunks with its own sentence chunker (needs nltk and its punkt_tab data)")
    a = ap.parse_args(argv)
    cfg = scoring_set.config()
    paths = dict(t.split("=", 1) for t in a.tokenizer)
    triples, claims, lock = scoring_set.load(cfg["set"])
    controls = scoring_controls.build()
    numbers = scoring_set.review_numbers(triples)
    result = {"schema": SCHEMA, "set": cfg["set"], "set_sha256": lock["set_sha256"],
              "controls_sha256": canon.sha256_file(scoring_controls.CONTROLS),
              "scoring_sha256": canon.sha256_file(scoring_set.SCORING),
              "overlap": cfg["overlap"], "margin": cfg["margin"], "checkers": {}, "controls": {}}
    toks = {}
    for m in cfg["nli"]:
        if m["id"] not in paths:
            raise SystemExit(f"--tokenizer {m['id']}=<path> is required")
        tok = toks.setdefault(m["sha256_tokenizer"], sw.HFTokenizer(Path(paths[m["id"]]), m["sha256_tokenizer"]))

        def cut(text, claim, tok=tok, m=m):
            return sw.windows(text, tok, claim, m["max_seq_len"], cfg["overlap"], cfg["margin"])

        c = ceilings_for(m["id"], tok, triples, claims, cut)
        c.update({"max_seq_len": m["max_seq_len"], "ways": m["ways"], "unit": "window"})
        result["checkers"][m["id"]] = c
        result["controls"][m["id"]] = control_checks(controls, tok, cut)
        print(f"{m['id']}: window {c['window_reachable']['overall']['reachable']}/54, "
              f"gate {c['gate']['overall']['reachable']}/54, gated {c['gated_window_reachable']['overall']['reachable']}/54")
    mc = cfg["minicheck"]
    tok = sw.HFTokenizer(Path(paths["minicheck"]), mc["sha256_tokenizer"])
    if a.minicheck_exact:
        import nltk  # MiniCheck's own sentence splitter
        from nltk.tokenize import sent_tokenize

        def mcut(text, claim):
            return sw.minicheck_chunks(text, lambda s: min(tok.count(s), mc["max_model_len"]), sent_tokenize,
                                       mc["chunk_size"])
        unit = f"chunk (exact: MiniCheck's sentence chunker, nltk {nltk.__version__} punkt_tab)"
    else:
        def mcut(text, claim):
            return sw.chunks(text, tok, mc["chunk_size"])
        unit = "chunk (approximate: line-granular; MiniCheck splits sentences with nltk)"
    c = ceilings_for(mc["id"], tok, triples, claims, mcut)
    c.update({"chunk_size": mc["chunk_size"], "unit": unit})
    for k in ("gate", "gated_window_reachable"):
        c.pop(k)  # MiniCheck has no gated variant
        c["binary"].pop(k)
    result["checkers"][mc["id"]] = c
    print(f"{mc['id']}: chunk {c['window_reachable']['overall']['reachable']}/54")
    result["checkers"]["HHEM-2.1-Open"] = {"unit": "the whole haystack", "window_reachable": {
        "overall": {"triples": len(triples), "reachable": len(triples), "share": 1.0}},
        "note": "HHEM takes the whole premise at its defaults: no window can drop a deciding span"}
    result["checkers"]["foundation-models"] = {
        "unit": "the whole haystack, where it fits the model's context",
        "context": "read at run time from sidekickd's /health (chat.context_limit); the ceiling is the share of "
                   "triples whose three variants all ran (scoring_ceilings.context_ceiling)",
        "estimate": context_estimate(triples, 4096, 600)}
    ref = toks[cfg["nli"][0]["sha256_tokenizer"]]
    notes = []
    for t in triples:
        n = numbers[t["id"]]
        if n in LABEL_NOTES:
            note = {"n": n, "id": t["id"], "note": LABEL_NOTES[n]}
            if t["claim"]["mechanism"] == "version":
                note["nm_strict_window_reachable"] = {
                    cid: r["nm_strict"] for cid, ch in result["checkers"].items()
                    for r in ch.get("triples", []) if r["id"] == t["id"] and "nm_strict" in r}
            if n == 20:
                note["position_at_approx_v1"] = t["position_at"]
                note["position_by_tokenizer"] = {"nli-deberta-v3-base": position_by_tokenizer(t, ref)}
            notes.append(note)
    result["label_notes"] = sorted(notes, key=lambda x: -x["n"])
    for ch in result["checkers"].values():
        for r in ch.get("triples", []):
            r["n"] = numbers[r["id"]]
    canon.write_json(a.out, result)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
