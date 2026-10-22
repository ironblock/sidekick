"""The verification set and its scoring configuration, as the harness reads them.

The triples are regenerated (they are never committed), and the result is
checked against the committed lock before anything is scored. Items are
`<triple id>:<variant>` for the corpus and `ctl:<control id>:<group>:<form>`
for the controls (tools/tasks/scoring_controls.py).

Scoring settings (τ, overlap, the bootstrap, the control gate, the checkers
and their pinned tokenizers) live in fixtures/tasks/verification/scoring.toml.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

from tasks import canon, gen_verification as gen

FIXTURES = gen.FIXTURES
SCORING = FIXTURES / "scoring.toml"
VARIANTS = ("s", "c", "nm")
GOLD = {"s": "S", "c": "C", "nm": "NM"}
VARIANT_NAME = {"s": "S", "c": "C", "nm": "NM"}


def config(path: Path = SCORING) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load(set_name: str | None = None) -> tuple[list[dict], dict, dict]:
    """(triples, claims by id, lock) for the set, regenerated and checked
    against its committed lock."""
    corpus = gen.load_toml(FIXTURES / "corpus.toml")
    set_name = set_name or corpus.get("set", f"review-{corpus.get('batch', 2)}")
    lock_path = FIXTURES / f"verification-{set_name}.lock.json"
    want = canon.read_json(lock_path)
    if "source_commit" in want:
        raise SystemExit(gen.historical_message(want))
    with tempfile.TemporaryDirectory() as d:
        triples, _pending, lock = gen.generate(FIXTURES / "captures.jsonl", Path(d), set_name,
                                               FIXTURES / "claims.toml", FIXTURES / "corpus.toml",
                                               FIXTURES / "captures.lock.json")
    diffs = gen.compare(lock, want)
    if diffs:
        raise SystemExit(f"{set_name} does not match its lock: {diffs[:3]}")
    claims = {c["id"]: c for c in gen.load_claims(FIXTURES / "claims.toml", corpus["max_per_shape"],
                                                  corpus.get("max_per_fact", 2))}
    return sorted(triples, key=lambda t: t["id"]), claims, lock


def claim_for(triple: dict, claims: dict) -> dict:
    """The claim as the generator used it for this triple: with
    `monotonic_calls`, the call ids it cites (in its text, probes and notes)
    are renumbered per triple, so its probes match that triple's evidence."""
    claim = claims[triple["claim"]["id"]]
    old = gen.POINTER_RE.findall(claim["text"])
    new = gen.POINTER_RE.findall(triple["claim"]["text"])
    mapping = {a: b for a, b in zip(old, new) if a != b}
    renamed = gen.renumber_calls(claim, mapping)
    if renamed["text"] != triple["claim"]["text"]:
        raise ValueError(f"{triple['id']}: cannot recover the claim's call ids")
    return renamed


def review_numbers(triples: list[dict]) -> dict[str, int]:
    """Each triple's number in the reviewer's index (gen_verification.export_review)."""
    order = sorted(triples, key=lambda t: (not t.get("worked"), t["claim"].get("type", ""), t["id"]))
    return {t["id"]: n for n, t in enumerate(order, 1)}


def evidence(triple: dict, v: str) -> str:
    return gen.expand(triple, v)


def item_id(triple_id: str, v: str) -> str:
    return f"{triple_id}:{v}"


def split_item(item: str) -> tuple[str, str]:
    tid, v = item.rsplit(":", 1)
    return tid, v


def items(triples: list[dict], controls: list[dict] | None = None) -> list[dict]:
    """Every scored unit: each triple's three variants, then the controls.
    Each has `item`, `claim`, `evidence`, `gold` and its tags."""
    out = []
    for t in triples:
        for v in VARIANTS:
            out.append({"item": item_id(t["id"], v), "triple": t["id"], "variant": VARIANT_NAME[v], "v": v,
                        "claim": t["claim"]["text"], "claim_id": t["claim"]["id"], "shape": t["claim"]["shape"],
                        "evidence": evidence(t, v), "gold": GOLD[v], "length": t["config"]["length"],
                        "obstacles": t["obstacles"]})
    for c in controls or []:
        out.append({"item": c["item"], "control": c["item"], "claim": c["claim"], "evidence": c["evidence"],
                    "gold": c["gold"], "group": c["group"], "form": c["form"]})
    return out


def in_fallback(item: dict) -> bool:
    """The second reader's fallback scope (design §6.2): the 1k and 4k
    items, the obstacle set (any triple with an obstacle or a "current"
    mechanism), and every control."""
    if "control" in item:
        return True
    return item["length"] in ("1k", "4k") or bool(item["obstacles"])
