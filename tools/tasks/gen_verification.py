"""Stage 2 of the verification corpus: seeded assembly of claim-versus-evidence
triples from hermetic captures (tools/tasks/capture.py).

A triple is one claim and three evidence variants that are byte-identical
outside their slots:
- SUPPORTED (s): the needle states the claim;
- CONTRADICTED (c): a minimal edit to every copy of the needle flips it;
- NOT MENTIONED (nm): every copy of the needle is removed.

Evidence is a session transcript: each block is one captured command,
`[call cN] $ <command>`, its output and `[exit N]`, or a `[notes]` block of
working notes. The needle is one or more blocks named by the claim
(fixtures/tasks/verification/claims.toml), edited per variant either
textually (`edit`: find and replace, for self-contained facts) or by taking
the variant's output from a sibling world's capture (`c_from`/`nm_from`, for
facts with dependents such as counts and sums). Filler is other captures,
chosen by a seeded draw from those that say nothing about the claim, to reach
a target length; the needle sits at the start, middle or end.

Obstacles: lengths of about 1k/4k/12k/32k tokens (`approx-v1`, see
tasks/canon.py), the needle's position, near-miss distractors, repeated
mentions, and five ways the evidence decides which record is current
(sequence, supersede marker, precedence, version scope, timestamps).

Every triple passes tools/tasks/hazards.py's checks or is refused, which
stops the run. The output is canonical JSONL plus a lock; `--check`
regenerates and compares with the committed lock.

    python3 tools/tasks/gen_verification.py --captures <dir>/captures.jsonl --out <dir> \\
        [--set review] [--update-lock | --check] [--export]

`--export` also writes a reviewer's copy: <out>/review/<set>.md (an index),
one Markdown file per triple with its claim, edits, needle locations and all
three haystacks, and <out>/review/<set>.jsonl with the expanded evidence.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, hazards, schema, scrub  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "tasks" / "verification"
GENERATOR = "sidekick-verification-gen/1"
TRIPLE_SCHEMA = "sidekick-verification-triple/1"
VARIANTS = hazards.VARIANTS
GOLD = {"s": "SUPPORTED", "c": "CONTRADICTED", "nm": "NOT_MENTIONED"}
TYPES = ("tool", "environment", "file", "result", "refutation")
OBSTACLES = ("none", "distractor", "repeated")
MECHANISMS = ("plain", "sequence", "supersede", "precedence", "version", "timestamp")
PARSED_KINDS = ("json", "jsonl", "toml")
CALL_RE = re.compile(r"\bc(\d+)\b")


class Refusal(Exception):
    pass


class Pending(Exception):
    """A claim whose captures need a recipe that has not been run yet."""


# ---------------------------------------------------------------- inputs


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_claims(path: Path, max_per_shape: int) -> list[dict]:
    raw = load_toml(path)
    claims, seen = [], set()
    for shape in raw["shape"]:
        insts = shape.get("claim", [])
        if not 1 <= len(insts) <= max_per_shape:
            raise SystemExit(f"shape {shape['id']}: {len(insts)} instances; allowed 1..{max_per_shape}")
        for c in insts:
            d = {"shape": shape["id"], "type": shape["type"], "mechanism": shape.get("mechanism", "plain"), **c}
            d.setdefault("guard", [x for x in (d.get("value"), d.get("value_c")) if x])
            if d["id"] in seen:
                raise SystemExit(f"duplicate claim id {d['id']}")
            seen.add(d["id"])
            schema.validate(d, "claim")
            claims.append(d)
    return claims


def load_captures(path: Path, lock_path: Path | None) -> dict[str, dict]:
    caps = canon.read_jsonl(path)
    for c in caps:
        schema.validate(c, "capture")
    if lock_path is not None:
        lock = canon.read_json(lock_path)
        want = {c["id"]: c["sha256"] for c in lock["captures"]}
        bad = [c["id"] for c in caps if want.get(c["id"]) != canon.record_sha256(c)]
        if bad or len(caps) != len(want):
            raise SystemExit(f"captures differ from {lock_path.name}: {bad[:5]} ({len(bad)} differ, "
                             f"{len(caps)} captured, {len(want)} locked); rerun tools/tasks/capture.py")
    return {c["id"]: c for c in caps}


# ---------------------------------------------------------------- rendering


def body(cap: dict) -> str:
    out = ""
    for stream in (cap["stdout"], cap["stderr"]):
        if stream:
            out += stream if stream.endswith("\n") else stream + "\n"
    return out


def render_capture(cmd: str, src: dict, call: str) -> str:
    return f"[call {call}] $ {cmd}\n{body(src)}[exit {src['exit']}]\n"


def render_notes(lines: list[str]) -> str:
    return "[notes]\n" + "".join(line + "\n" for line in lines) if lines else ""


def block_text(spec: dict, v: str, caps: dict, call: str | None) -> str:
    """One needle block's text in variant v, before the block separator."""
    if spec.get(f"{v}_drop"):
        return ""
    if "notes" in spec:
        text = render_notes(spec.get(f"{v}_notes", spec["notes"]))
    elif spec.get(f"{v}_empty"):
        base = caps[spec["capture"]]
        text = f"[call {call}] $ {base['cmd']}\n[exit {base['exit']}]\n"
    else:
        base = caps[spec["capture"]]
        text = render_capture(base["cmd"], caps[spec.get(f"{v}_from", spec["capture"])], call)
    for ed in spec.get("edit", []):
        if v not in ed:
            continue
        n = text.count(ed["find"])
        if n == 0 or ("count" in ed and n != ed["count"]):
            raise Refusal(f"edit: {v}: {ed['find']!r} occurs {n} times in {spec.get('capture', 'notes')}")
        text = text.replace(ed["find"], ed[v])
    return text


def block_body(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[1:-1]) if len(lines) >= 2 else ""


def split3(s: str, c: str, nm: str) -> list:
    """Split three versions of a block into fixed text and slots, by lines:
    each slot is a maximal run where any variant differs from s."""
    sl, cl, nl = (x.splitlines(keepends=True) for x in (s, c, nm))
    ops = {k: difflib.SequenceMatcher(None, sl, other, autojunk=False).get_opcodes()
           for k, other in (("c", cl), ("nm", nl))}
    touched = []
    for k in ops:
        for tag, i1, i2, _, _ in ops[k]:
            if tag != "equal":
                touched.append((i1, i2))
    touched.sort()
    merged = []
    for a, b in touched:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    def map_bound(k, i, side):
        cands = []
        for tag, i1, i2, j1, j2 in ops[k]:
            if tag == "equal" and i1 <= i <= i2:
                cands.append(j1 + (i - i1))
            elif tag != "equal":
                if i == i1:
                    cands.append(j1)
                if i == i2:
                    cands.append(j2)
        if not cands:  # both versions empty: no opcodes at all
            return 0
        return min(cands) if side == "start" else max(cands)

    pieces, last = [], 0
    other = {"c": cl, "nm": nl}
    for a, b in merged:
        if a > last:
            pieces.append("".join(sl[last:a]))
        slot = {"s": "".join(sl[a:b])}
        for k in ("c", "nm"):
            slot[k] = "".join(other[k][map_bound(k, a, "start"):map_bound(k, b, "end")])
        pieces.append(slot)
        last = b
    if last < len(sl):
        pieces.append("".join(sl[last:]))
    return pieces


# ---------------------------------------------------------------- assembly


def mentions_claim(text: str, claim: dict) -> bool:
    probes = claim.get("probes", {})
    for key in ("support", "contradict", "mention"):
        for p in probes.get(key, []):
            if re.search(p, text, re.M):
                return True
    return any(re.search(r"(?<![\w-])" + re.escape(lit) + r"(?![\w-])", text) for lit in claim.get("guard", []))


def needle_captures(claim: dict) -> set[str]:
    ids = set()
    for b in claim.get("block", []):
        for k in ("capture", "s_from", "c_from", "nm_from"):
            if k in b:
                ids.add(b[k])
    for d in claim.get("distractor", []):
        if "capture" in d:
            ids.add(d["capture"])
    return ids


def filler_pool(claim: dict, caps: dict, exclude_worlds=()) -> list[str]:
    """Captures that may pad this claim's haystack: never the needle's own,
    never a sibling world's (whose output would disagree with its base
    world's), never one the claim excludes, and never one that matches any
    of the claim's probes or guard literals."""
    exclude = needle_captures(claim) | set(claim.get("exclude", []))
    pool, seen_text = [], set()
    for cid in sorted(caps):
        if cid in exclude or caps[cid]["world"] in exclude_worlds:
            continue
        text = render_capture(caps[cid]["cmd"], caps[cid], "c0")
        key = body(caps[cid])
        if key in seen_text or mentions_claim(text, claim):
            continue
        seen_text.add(key)
        pool.append(cid)
    return pool


def nearest_boundary(cum: list[int], frac: float) -> int:
    total = cum[-1]
    return min(range(len(cum)), key=lambda i: (abs(cum[i] - frac * total), i))


def build(claim: dict, cfg: dict, caps: dict, corpus: dict) -> dict:
    """Assemble and check one triple, or raise Refusal/Pending."""
    missing = sorted(needle_captures(claim) - set(caps))
    if missing:
        if claim.get("pending"):
            raise Pending(f"{claim['id']}: waits for {claim['pending']} ({', '.join(missing)})")
        raise Refusal(f"{claim['id']}: unknown captures {missing}")
    seed = corpus["seed"]
    r = canon.rng(seed, claim["id"], cfg["length"], cfg["position"], cfg["obstacle"])
    target = corpus["lengths"][cfg["length"]]
    frac = corpus["positions"][cfg["position"]]
    specs = claim["block"]

    # Call ids the claim names, and the items other than filler, as (kind, payload).
    reserved = set(CALL_RE.findall(claim["text"]))
    for spec in specs:
        if "call" in spec:
            reserved.add(spec["call"][1:])
        for k in ("notes", "s_notes", "c_notes", "nm_notes"):
            for line in spec.get(k, []):
                reserved.update(CALL_RE.findall(line))
    needle_a = [("needle", i) for i, s in enumerate(specs) if s.get("group", "a") == "a"]
    needle_b = [("needle", i) for i, s in enumerate(specs) if s.get("group", "a") == "b"]
    extras_near = []
    if cfg["obstacle"] == "distractor":
        if not claim.get("distractor"):
            raise Refusal(f"{claim['id']}: no distractor declared")
        extras_near = [("distractor", j) for j in range(len(claim["distractor"]))]
    scoped = [("scoped", j) for j in range(len(claim.get("scoped", [])))]
    repeat = []
    if cfg["obstacle"] == "repeated":
        if "repeat" not in claim:
            raise Refusal(f"{claim['id']}: no repeatable block declared")
        repeat = [("repeat", claim["repeat"])]

    def item_tokens(kind, payload) -> int:
        if kind in ("needle", "repeat"):
            return canon.approx_tokens(block_text(specs[payload], "s", caps, "c0"))
        if kind == "distractor":
            d = claim["distractor"][payload]
            return canon.approx_tokens(render_capture(caps[d["capture"]]["cmd"], caps[d["capture"]], "c0")
                                       if "capture" in d else render_notes(d["notes"]))
        if kind == "scoped":
            return canon.approx_tokens(render_notes(claim["scoped"][payload]["notes"]))
        return canon.approx_tokens(render_capture(caps[payload]["cmd"], caps[payload], "c0"))

    fixed_items = needle_a + needle_b + extras_near + scoped + repeat
    fixed_tokens = sum(item_tokens(k, p) for k, p in fixed_items)
    if fixed_tokens > target * 1.05:
        raise Refusal(f"{claim['id']}: the needle alone is {fixed_tokens} tokens, over {cfg['length']}")

    # Filler: a seeded draw from captures that say nothing about the claim.
    pool = filler_pool(claim, caps, corpus.get("filler", {}).get("exclude_worlds", ()))
    r.shuffle(pool)
    filler, total = [], fixed_tokens
    for cid in pool:
        t = item_tokens("filler", cid)
        if total + t <= target * 1.05:
            filler.append(cid)
            total += t
        if total >= target * 0.95:
            break
    if total < target * 0.9:
        raise Refusal(f"{claim['id']}: filler pool too small for {cfg['length']} ({total} tokens)")

    # Placement by token position over the filler.
    cum = [0]
    for cid in filler:
        cum.append(cum[-1] + item_tokens("filler", cid))
    inserts = []  # (boundary index, order, item)
    anchor = claim.get("anchor", "a")
    near_before = r.random() < 0.5
    group_a = (extras_near if near_before else []) + scoped + needle_a
    group_b = needle_b
    if not needle_b:
        group_a = group_a + ([] if near_before else extras_near)
    else:
        group_b = needle_b + ([] if near_before else extras_near)
    if needle_b:
        if anchor == "b":
            ib = nearest_boundary(cum, frac)
            ia = min(nearest_boundary(cum, max(0.0, frac - 0.3)), ib)
            if ia == ib and ib < len(filler):
                ib += 1
        else:
            ia = nearest_boundary(cum, frac)
            ib = max(nearest_boundary(cum, min(1.0, frac + 0.3)), ia)
            if ia == ib and ia > 0:
                ia -= 1
        inserts += [(ia, 0, it) for it in group_a] + [(ib, 1, it) for it in group_b]
    else:
        inserts += [(nearest_boundary(cum, frac), 0, it) for it in group_a]
    for it in repeat:
        inserts.append((nearest_boundary(cum, 0.85 if frac < 0.5 else 0.15), 2, it))
    order = []
    ins = sorted(((b, o, n, it) for n, (b, o, it) in enumerate(inserts)))
    k = 0
    for i in range(len(filler) + 1):
        while k < len(ins) and ins[k][0] == i:
            order.append(ins[k][3])
            k += 1
        if i < len(filler):
            order.append(("filler", filler[i]))

    # Call ids: fixed ones from the claim, the rest drawn without collisions.
    def has_call(kind, payload):
        if kind in ("needle", "repeat"):
            return "notes" not in specs[payload]
        if kind == "distractor":
            return "capture" in claim["distractor"][payload]
        return kind == "filler"

    free = [n for n in range(1, 1000) if str(n) not in reserved]
    need = sum(1 for kind, p in order if has_call(kind, p) and not (kind == "needle" and "call" in specs[p]))
    drawn = sorted(r.sample(free, need))
    calls, di = [], 0
    for kind, p in order:
        if not has_call(kind, p):
            calls.append(None)
        elif kind == "needle" and "call" in specs[p]:
            calls.append(specs[p]["call"])  # a repeat is a re-run: it gets its own id
        else:
            calls.append(f"c{drawn[di]}")
            di += 1

    # Render into tagged pieces: (item index, str | slot).
    pieces, layout, slot_roles, bodies_spec = [], [], [], []
    for n, ((kind, p), call) in enumerate(zip(order, calls)):
        if kind in ("needle", "repeat"):
            spec = specs[p]
            texts = {v: block_text(spec, v, caps, call) for v in VARIANTS}
            texts = {v: (t + "\n" if t else "") for v, t in texts.items()}
            for piece in split3(texts["s"], texts["c"], texts["nm"]):
                if isinstance(piece, dict):
                    # A block's role, or the role of the edit that made this slot.
                    role = next((ed["role"] for ed in spec.get("edit", [])
                                 if ed.get("role") and ed["find"] in piece["s"]), spec.get("role"))
                    piece = {**piece, "role": role, "block": p}
                    slot_roles.append(role)
                pieces.append((n, piece))
            src = spec.get("capture", "notes")
            kind_of = caps[src]["kind"] if src in caps else "notes"
            if kind_of in PARSED_KINDS:
                bodies_spec.append((kind_of, {v: block_body(texts[v].rstrip("\n") + "\n") if texts[v] else ""
                                              for v in VARIANTS}))
            layout.append({"item": kind, "source": src, "call": call})
        elif kind == "distractor":
            d = claim["distractor"][p]
            text = (render_capture(caps[d["capture"]]["cmd"], caps[d["capture"]], call)
                    if "capture" in d else render_notes(d["notes"]))
            pieces.append((n, text + "\n"))
            layout.append({"item": kind, "source": d.get("capture", "notes"), "call": call})
        elif kind == "scoped":
            pieces.append((n, render_notes(claim["scoped"][p]["notes"]) + "\n"))
            layout.append({"item": kind, "source": "notes", "call": None})
        else:
            pieces.append((n, render_capture(caps[p]["cmd"], caps[p], call) + "\n"))
            layout.append({"item": kind, "source": p, "call": call})

    # Segments and slots, and each variant's evidence with its spans.
    segments, slots = [""], []
    for _, piece in pieces:
        if isinstance(piece, str):
            segments[-1] += piece
        else:
            slots.append({k: piece[k] for k in ("s", "c", "nm", "role", "block")})
            segments.append("")
    ev = {}
    for v in VARIANTS:
        text, spans, dspans, sspans, si = "", [], [], [], 0
        for n, piece in pieces:
            start = len(text)
            if isinstance(piece, str):
                text += piece
                if order[n][0] == "distractor":
                    dspans.append((start, len(text)))
                elif order[n][0] == "scoped":
                    sspans.append((start, len(text)))
            else:
                text += piece[v]
                spans.append((start, len(text), si))
                si += 1
        bodies = [(kd, b[v]) for kd, b in bodies_spec if b[v]]
        ev[v] = hazards.Evidence(text, spans, dspans, sspans, bodies)
    for v in VARIANTS:
        if expand({"segments": segments, "slots": slots}, v) != ev[v].text:
            raise Refusal(f"{claim['id']}: internal: segments do not reproduce the {v} evidence")

    problems = hazards.check(claim, ev, slot_roles)
    if problems:
        raise Refusal(f"{claim['id']} {cfg_id(cfg)}:\n  " + "\n  ".join(problems))

    obstacles = []
    if cfg["obstacle"] != "none":
        obstacles.append(cfg["obstacle"])
    if claim["mechanism"] != "plain":
        obstacles.append(f"current:{claim['mechanism']}")
    first = next((s["capture"] for s in specs if "capture" in s and not s.get("s_drop")), None)
    rec = {
        "schema": TRIPLE_SCHEMA,
        "family": "verification",
        "id": f"{claim['id']}.{cfg_id(cfg)}",
        "claim": {"id": claim["id"], "shape": claim["shape"], "type": claim["type"],
                  "mechanism": claim["mechanism"], "text": claim["text"]},
        "split": split_of(seed, claim["shape"], corpus.get("dev_share", 0.2)),
        "config": dict(cfg),
        "obstacles": obstacles,
        "haystack_kind": caps[first]["kind"] if first else "notes",
        "worked": bool(claim.get("worked")),
        "tokens": {v: canon.approx_tokens(ev[v].text) for v in VARIANTS},
        "length_unit": canon.LENGTH_UNIT,
        "segments": segments,
        "slots": slots,
        "layout": layout,
        "gold": dict(GOLD),
        "edits": claim.get("edit_note", ""),
        "evidence_sha256": {v: canon.sha256_text(ev[v].text) for v in VARIANTS},
        "generator": GENERATOR,
    }
    schema.validate(rec, "triple")
    return rec


def expand(rec: dict, v: str) -> str:
    out = rec["segments"][0]
    for slot, seg in zip(rec["slots"], rec["segments"][1:]):
        out += slot[v] + seg
    return out


def cfg_id(cfg: dict) -> str:
    return f"{cfg['length']}.{cfg['position']}.{cfg['obstacle']}"


def split_of(seed, shape: str, dev_share: float) -> str:
    return "dev" if canon.sub_seed(seed, "split", shape) % 1000 < dev_share * 1000 else "test"


# ---------------------------------------------------------------- plans


def review_plan(claims: list[dict], corpus: dict) -> list[tuple[dict, dict]]:
    """The first review batch: every worked triple at its stated config, then
    `per_type` triples per claim type, spread over lengths, positions and
    obstacles by a seeded draw."""
    rv = corpus["review"]
    seed = corpus["seed"]
    plan = []
    for c in sorted(claims, key=lambda c: c["id"]):
        if c.get("worked"):
            plan.append((c, dict(rv["worked"])))
    for t in TYPES:
        pool = sorted((c for c in claims if c["type"] == t and not c.get("worked")), key=lambda c: c["id"])
        if not pool:
            continue
        r = canon.rng(seed, "review", t)
        r.shuffle(pool)
        lengths = list(rv["lengths"])
        positions = list(rv["positions"])
        r.shuffle(lengths)
        r.shuffle(positions)
        seen = set()
        turn = r.randrange(3)  # rotates none, distractor, repeated over the type's plain claims
        for k in range(rv["per_type"]):
            c = pool[k % len(pool)]
            obstacle = "none"
            if c["mechanism"] == "plain":
                avail = {"none": True, "distractor": bool(c.get("distractor")), "repeated": "repeat" in c}
                for step in range(3):
                    cand = OBSTACLES[(turn + step) % 3]
                    if avail[cand]:
                        obstacle = cand
                        break
                turn += 1
            cfg = {"length": lengths[k % len(lengths)], "position": positions[k % len(positions)],
                   "obstacle": obstacle}
            j = 0
            while (c["id"], cfg_id(cfg)) in seen:
                j += 1
                cfg["position"] = positions[(k + j) % len(positions)]
            seen.add((c["id"], cfg_id(cfg)))
            plan.append((c, cfg))
    return plan


def fit_length(claim: dict, cfg: dict, caps: dict, corpus: dict) -> dict:
    """Move a config up to the shortest length its needle fits."""
    order = list(corpus["lengths"])
    specs = claim["block"]
    need = sum(canon.approx_tokens(block_text(s, "s", caps, "c0")) for s in specs) if \
        not (needle_captures(claim) - set(caps)) else 0
    i = order.index(cfg["length"])
    while i < len(order) - 1 and need > corpus["lengths"][order[i]] * 0.6:
        i += 1
    return {**cfg, "length": order[i]}


# ---------------------------------------------------------------- outputs


def generate(captures: Path, out: Path, set_name: str, claims_path: Path, corpus_path: Path,
             captures_lock: Path | None) -> tuple[list[dict], list[str], dict]:
    corpus = load_toml(corpus_path)
    claims = load_claims(claims_path, corpus["max_per_shape"])
    caps = load_captures(captures, captures_lock)
    if set_name != "review":
        raise SystemExit(f"unknown set {set_name!r}")
    triples, pending, refused = [], [], []
    for claim, cfg in review_plan(claims, corpus):
        try:
            cfg = fit_length(claim, cfg, caps, corpus)
            triples.append(build(claim, cfg, caps, corpus))
        except Pending as p:
            pending.append(str(p))
        except Refusal as e:
            refused.append(str(e))
    # The capture stage scanned every capture; scan the assembled evidence too,
    # since edits and notes add text of their own.
    recipes = load_toml(FIXTURES / "recipes.toml") if (FIXTURES / "recipes.toml").exists() else {}
    allow = [{**a, "capture": "*"} for a in recipes.get("scan", {}).get("allow", [])]
    clock = tuple(recipes.get("scan", {}).get("clock", ["2026-10-01", "2026-10-01"]))
    identity = scrub.machine_identity()
    for t in triples:
        for v in VARIANTS:
            found = scrub.unallowed(scrub.scan(expand(t, v), identity, clock), allow, t["id"])
            if found:
                refused.append(f"{t['id']} {v}: the refusal scan found " + "; ".join(map(str, found[:3])))
    if refused:
        raise SystemExit("refused:\n" + "\n".join(refused))
    ids = [t["id"] for t in triples]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate triple ids in the plan")
    set_sha = canon.write_jsonl(out / f"verification-{set_name}.triples.jsonl", triples)
    lock = {
        "schema": "sidekick-verification-lock/1",
        "set": set_name,
        "generator": GENERATOR,
        "length_unit": canon.LENGTH_UNIT,
        "corpus_sha256": canon.sha256_file(corpus_path),
        "claims_sha256": canon.sha256_file(claims_path),
        "captures_set_sha256": canon.sha256_file(captures),
        "triples": [{"id": t["id"], "sha256": canon.record_sha256(t), "evidence_sha256": t["evidence_sha256"]}
                    for t in sorted(triples, key=lambda t: t["id"])],
        "pending": sorted(set(pending)),
        "set_sha256": set_sha,
    }
    schema.validate(lock, "lock")
    return triples, sorted(set(pending)), lock


def compare(got: dict, want: dict) -> list[str]:
    diffs = []
    for k in ("generator", "length_unit", "corpus_sha256", "claims_sha256", "captures_set_sha256", "set_sha256",
              "pending"):
        if got.get(k) != want.get(k):
            diffs.append(f"{k}: {got.get(k)} != committed {want.get(k)}")
    g = {t["id"]: t["sha256"] for t in got["triples"]}
    w = {t["id"]: t["sha256"] for t in want["triples"]}
    for tid in sorted(set(g) | set(w)):
        if g.get(tid) != w.get(tid):
            diffs.append(f"triple {tid}: {g.get(tid)} != committed {w.get(tid)}")
    return diffs


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def export_review(triples: list[dict], out: Path, set_name: str) -> None:
    d = out / "review"
    d.mkdir(parents=True, exist_ok=True)
    for old in [*d.glob("[0-9][0-9][0-9]-*.md"), d / f"{set_name}.md", d / f"{set_name}.jsonl"]:
        old.unlink(missing_ok=True)  # a previous export's files
    index = [f"# Verification triples for review: `{set_name}`\n",
             "Each triple is one claim and three haystacks (SUPPORTED, CONTRADICTED, NOT MENTIONED) that are "
             "byte-identical outside the slots listed in its file. Lengths are approximate tokens (approx-v1).\n",
             "| # | triple | type | mechanism | obstacles | length | position | tokens (s) | split |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for n, t in enumerate(sorted(triples, key=lambda t: (not t["worked"], t["claim"]["type"], t["id"])), 1):
        name = f"{n:03d}-{t['id']}.md"
        index.append(f"| {n} | [{t['id']}]({name}) | {t['claim']['type']} | {t['claim']['mechanism']} | "
                     f"{', '.join(t['obstacles']) or '-'} | {t['config']['length']} | {t['config']['position']} | "
                     f"{t['tokens']['s']} | {t['split']} |")
        texts = {v: expand(t, v) for v in VARIANTS}
        md = [f"# {t['id']}\n",
              f"**Claim** ({t['claim']['type']}, shape `{t['claim']['shape']}`, mechanism "
              f"`{t['claim']['mechanism']}`{', worked example' if t['worked'] else ''}):\n",
              "~~~~text", t["claim"]["text"], "~~~~\n",
              f"Gold: s = SUPPORTED, c = CONTRADICTED, nm = NOT MENTIONED. Length {t['config']['length']} "
              f"(tokens s/c/nm: {t['tokens']['s']}/{t['tokens']['c']}/{t['tokens']['nm']}), position "
              f"{t['config']['position']}, obstacles: {', '.join(t['obstacles']) or 'none'}. Split: {t['split']}.\n"]
        if t["edits"]:
            md += [f"Edit: {t['edits']}\n"]
        md += ["## Needle slots and edits\n"]
        offs = {v: len(t["segments"][0]) for v in VARIANTS}
        for i, (slot, seg) in enumerate(zip(t["slots"], t["segments"][1:])):
            loc = ", ".join(f"{v} line {line_of(texts[v], offs[v])} (offset {offs[v]})" for v in VARIANTS)
            md += [f"### Slot {i + 1} (block {slot['block']}{', role ' + slot['role'] if slot['role'] else ''})\n",
                   f"Location: {loc}.\n"]
            for v in VARIANTS:
                md += [f"{v}:", "~~~~text", slot[v].rstrip("\n") if slot[v] else "(empty)", "~~~~"]
            md.append("")
            for v in VARIANTS:
                offs[v] += len(slot[v]) + len(seg)
        md += ["## Layout\n", "| # | item | source | call |", "|---|---|---|---|"]
        md += [f"| {i} | {b['item']} | `{b['source']}` | {b['call'] or ''} |" for i, b in enumerate(t["layout"])]
        for v in VARIANTS:
            md += ["", f"## Haystack: {GOLD[v]} ({v})\n", "~~~~text", texts[v].rstrip("\n"), "~~~~"]
        (d / name).write_text("\n".join(md) + "\n", encoding="utf-8")
        rows.append({"id": t["id"], "claim": t["claim"], "config": t["config"], "obstacles": t["obstacles"],
                     "gold": t["gold"], "evidence": texts, "slots": t["slots"], "layout": t["layout"]})
    (d / f"{set_name}.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    canon.write_jsonl(d / f"{set_name}.jsonl", rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", type=Path, required=True, help="captures.jsonl from tools/tasks/capture.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", default="review")
    ap.add_argument("--claims", type=Path, default=FIXTURES / "claims.toml")
    ap.add_argument("--corpus", type=Path, default=FIXTURES / "corpus.toml")
    ap.add_argument("--captures-lock", type=Path, default=FIXTURES / "captures.lock.json")
    ap.add_argument("--lock", type=Path, help="the set's lock (default: fixtures/.../verification-<set>.lock.json)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--update-lock", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--export", action="store_true", help="also write the reviewer's copy under <out>/review/")
    a = ap.parse_args(argv)
    lock_path = a.lock or FIXTURES / f"verification-{a.set}.lock.json"
    triples, pending, lock = generate(a.captures, a.out, a.set, a.claims, a.corpus, a.captures_lock)
    by_type = {}
    for t in triples:
        by_type[t["claim"]["type"]] = by_type.get(t["claim"]["type"], 0) + 1
    print(f"{len(triples)} triples ({', '.join(f'{k} {v}' for k, v in sorted(by_type.items()))}); "
          f"set sha256 {lock['set_sha256']}")
    for p in pending:
        print(f"pending: {p}")
    if a.export:
        export_review(triples, a.out, a.set)
        print(f"wrote {a.out / 'review'}")
    if a.update_lock:
        canon.write_json(lock_path, lock)
        print(f"wrote {lock_path}")
    if a.check:
        diffs = compare(lock, canon.read_json(lock_path))
        for line in diffs:
            print("MISMATCH", line)
        print("the set matches its lock" if not diffs else f"{len(diffs)} mismatch(es)")
        return 1 if diffs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
