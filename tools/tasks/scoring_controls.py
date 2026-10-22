"""Control items for the verification scoring harness (design §2.1).

Eighteen items in three groups of six, separate from the corpus, built from
the committed hermetic captures (fixtures/tasks/verification/controls.toml):

- **positive** (gold S): a capture whose text states the claim almost
  verbatim;
- **contradiction** (gold C): the same capture with one value changed
  (`c_edit`); three-way checkers only;
- **null** (gold NM, or not-S on a two-way checker): an unrelated capture
  of the same kind and about the same length.

Each item has two forms:
- **short:** the one block alone, which fits a single window;
- **long:** the block spliced at the middle of about 32k tokens (approx-v1)
  of filler, drawn as the generator draws it (captures that match none of
  the control's probes or guard literals, never from the needle's world
  family, in capture order within a family), so windowing, the gate and
  aggregation run exactly as they do in scoring.

The positive and contradiction blocks run as call `c500`, the null block
as `c600`, so a claim that cites `[call c500]` never finds an unrelated
command under that id. Filler call ids increase through the haystack,
skipping both. Everything is a function of the committed captures and the
control file's seed.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tasks import canon, gen_verification as gen, scoring_windows as sw

CONTROLS = gen.FIXTURES / "controls.toml"
GROUPS = ("positive", "contradiction", "null")
FORMS = ("short", "long")
GOLD = {"positive": "S", "contradiction": "C", "null": "NM"}
NEEDLE_CALL = "c500"
NULL_CALL = "c600"


def load(path: Path = CONTROLS) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _corpus_settings() -> dict:
    """The generator's filler settings (exclusions and world families)."""
    corpus = gen.load_toml(gen.FIXTURES / "corpus.toml")
    worlds = gen.load_toml(gen.FIXTURES / "recipes.toml")["world"]
    corpus["_parents"] = {w["id"]: w["from"] for w in worlds if "from" in w}
    no_filler = [w["id"] for w in worlds if w.get("filler") is False]
    corpus.setdefault("filler", {})
    corpus["filler"]["exclude_worlds"] = list(corpus["filler"].get("exclude_worlds", [])) + no_filler
    return corpus


def pseudo_claim(ctl: dict) -> dict:
    """The control as the generator's filler rules see a claim."""
    return {"id": ctl["id"], "block": [{"capture": ctl["capture"]}, {"capture": ctl["null"]}],
            "probes": {"support": ctl.get("probes", []), "contradict": [], "mention": []},
            "guard": ctl.get("guard", []), "touch": [], "exclude": []}


def needle(ctl: dict, group: str, caps: dict) -> str:
    """The control's one block, as rendered in the transcript."""
    if group == "null":
        src = caps[ctl["null"]]
        return gen.render_capture(src["cmd"], src, NULL_CALL)
    src = caps[ctl["capture"]]
    text = gen.render_capture(src["cmd"], src, NEEDLE_CALL)
    if group == "contradiction":
        ed = ctl["c_edit"]
        if text.count(ed["find"]) != ed.get("count", 1):
            raise ValueError(f"control {ctl['id']}: c_edit find occurs {text.count(ed['find'])} times")
        text = text.replace(ed["find"], ed["replace"])
    return text


def filler(ctl: dict, caps: dict, corpus: dict, target: int, seed) -> list[str]:
    """The long form's filler captures, in transcript order."""
    claim = pseudo_claim(ctl)
    pool = gen.filler_pool(claim, caps, corpus["filler"]["exclude_worlds"], corpus["_parents"])
    r = canon.rng(seed, "control", ctl["id"])
    r.shuffle(pool)
    out, total = [], 0
    for cid in pool:
        t = canon.approx_tokens(gen.render_capture(caps[cid]["cmd"], caps[cid], "c0"))
        if t > target * gen.MAX_BLOCK_SHARE or total + t > target * 1.05:
            continue
        out.append(cid)
        total += t
        if total >= target * 0.95:
            break
    order = gen.chronological([("filler", c) for c in out], caps, corpus)
    return [c for _, c in order]


def long_form(block: str, fill: list[str], caps: dict, seed, ctl_id: str) -> str:
    """The block at the middle of the filler (by approx-v1 tokens)."""
    sizes = [canon.approx_tokens(gen.render_capture(caps[c]["cmd"], caps[c], "c0")) for c in fill]
    cum = [0]
    for n in sizes:
        cum.append(cum[-1] + n)
    group = canon.approx_tokens(block)
    want = 0.5 * (cum[-1] + group) - group / 2
    at = min(range(len(cum)), key=lambda i: (abs(cum[i] - want), i))
    r = canon.rng(seed, "control-calls", ctl_id)
    before = sorted(r.sample(range(1, 500), at))
    after = sorted(r.sample([n for n in range(501, 1000) if n != 600], len(fill) - at))
    ids = [f"c{n}" for n in before + after]
    parts = [gen.render_capture(caps[c]["cmd"], caps[c], ids[i]) + "\n" for i, c in enumerate(fill)]
    parts.insert(at, block + "\n")
    return "".join(parts)


def build(ctl_file: dict | None = None, caps: dict | None = None) -> list[dict]:
    """All 36 control items (18 controls × short and long)."""
    ctl_file = ctl_file or load()
    if caps is None:
        caps = gen.load_captures(gen.FIXTURES / "captures.jsonl", gen.FIXTURES / "captures.lock.json")
    corpus = _corpus_settings()
    seed, target = ctl_file["seed"], ctl_file["long_tokens"]
    items = []
    for ctl in ctl_file["control"]:
        fill = filler(ctl, caps, corpus, target, seed)
        for group in GROUPS:
            block = needle(ctl, group, caps)
            for form in FORMS:
                text = block + "\n" if form == "short" else long_form(block, fill, caps, seed, ctl["id"])
                items.append({"item": f"ctl:{ctl['id']}:{group}:{form}", "control": ctl["id"], "group": group,
                              "form": form, "type": ctl["type"], "claim": ctl["claim"], "gold": GOLD[group],
                              "evidence": text, "evidence_sha256": canon.sha256_text(text),
                              "needle": block})
    return items


def mentions(text: str, ctl: dict) -> bool:
    return gen.mentions_claim(text, pseudo_claim(ctl))


def deciding_span(item: dict) -> tuple[int, int] | None:
    """Where the control's block sits in its evidence (positive and
    contradiction); the null group has nothing to find."""
    if item["group"] == "null":
        return None
    at = item["evidence"].find(item["needle"])
    return (at, at + len(item["needle"].rstrip("\n")))


def kind_of(cid: str, caps: dict) -> str:
    """A capture's kind, with help pages and help topics as one kind."""
    k = caps[cid]["kind"]
    return "documentation" if k in gen.DOC_KINDS else k


def check(ctl_file: dict, caps: dict) -> list[str]:
    """The control file's own rules (also run by the tests)."""
    problems = []
    seen = set()
    types = set()
    for ctl in ctl_file["control"]:
        cid = ctl["id"]
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        types.add(ctl["type"])
        for k in ("capture", "null"):
            if ctl[k] not in caps:
                problems.append(f"{cid}: unknown capture {ctl[k]}")
        if problems:
            continue
        pos = needle(ctl, "positive", caps)
        con = needle(ctl, "contradiction", caps)
        nul = needle(ctl, "null", caps)
        if not mentions(pos, ctl):
            problems.append(f"{cid}: the positive block matches none of its probes")
        if mentions(nul, ctl):
            problems.append(f"{cid}: the null block mentions the claim")
        if kind_of(ctl["capture"], caps) != kind_of(ctl["null"], caps):
            problems.append(f"{cid}: the null capture's kind differs")
        a, b = canon.approx_tokens(pos), canon.approx_tokens(nul)
        if not 0.5 <= b / a <= 2.0:
            problems.append(f"{cid}: the null block is {b} tokens against {a}")
        if pos == con or len([1 for x, y in zip(pos.splitlines(), con.splitlines()) if x != y]) < 1:
            problems.append(f"{cid}: the contradiction edit changes nothing")
        # A gated checker scores only windows holding an anchor; a control it
        # could never pass would test the rule, not the interface.
        anchors = sw.anchors(ctl["claim"])
        if not any(x in pos for x in anchors) or not any(x in con for x in anchors):
            problems.append(f"{cid}: no anchor of the claim ({anchors}) occurs in its block")
        if re.search(r"\[call c\d+\]", ctl["claim"]) and NEEDLE_CALL not in ctl["claim"]:
            problems.append(f"{cid}: a claim's pointer must name {NEEDLE_CALL}")
    if len(ctl_file["control"]) != 6:
        problems.append(f"{len(ctl_file['control'])} controls; the design asks for 6")
    missing = set(gen.TYPES) - types
    if missing:
        problems.append(f"claim types not covered: {sorted(missing)}")
    return problems
