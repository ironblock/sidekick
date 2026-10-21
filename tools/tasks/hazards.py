"""Mechanical hazard checks for a verification triple.

A triple is one claim and three evidence variants, SUPPORTED (s),
CONTRADICTED (c) and NOT MENTIONED (nm), that are byte-identical outside
their slots. The generator (tools/tasks/gen_verification.py) refuses any
triple for which `check` returns a problem. The rules:

- **Byte identity.** With every slot cut out, the three variants are equal.
- **Uniqueness.** A claim's `support` and `contradict` probes match only
  where they overlap a slot (the needle region), in every variant. The one
  exception is a version-scoped statement in a block the claim declares as
  scoped to another version, which must say which version it is about.
- **Guard literals** (the needle value and its replacement, by default)
  occur nowhere outside the slots.
- **Restatement.** SUPPORTED has a support match and no contradiction; the
  CONTRADICTED variant has a contradiction and no support anywhere, so every
  copy of the needle was edited; NOT MENTIONED has no support, contradiction
  or `mention` match anywhere.
- **Which record is current**, per the claim's mechanism:
  - `sequence`: two slots whose values swap; the later one decides.
  - `timestamp`: two records whose values swap; the later timestamp
    decides, and it must come first in the text, so position can't decide.
  - `precedence`: two slots whose values swap; the higher-precedence one
    decides, and it comes first in the text.
  - `supersede`: a refuted entry, `was: X (held at N/10), wrong because Y
    [call id]`, counts against X, not for it. For a claim about X, the
    CONTRADICTED variant's only matches of X are on such lines.
  - `version`: a declared, version-scoped statement about another version
    may say otherwise; the haystack's own version decides.
- **Near-miss distractors** match their `near` probe and neither support nor
  contradict the claim.
- **Derived values.** Each `count` or `sum` invariant holds in every variant
  where its total is shown, and every JSON, JSONL or TOML body in the needle
  still parses.
- **No defined synonyms.** The replacement value is not an alias or a
  short/long option form of the original that the haystack itself defines.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field

VARIANTS = ("s", "c", "nm")
MARKER_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?was:[ \t]", re.M)


@dataclass
class Evidence:
    """One variant's evidence and where its slots and declared blocks lie."""

    text: str
    slots: list[tuple[int, int, int]]          # (start, end, slot index)
    distractors: list[tuple[int, int]] = field(default_factory=list)
    scoped: list[tuple[int, int]] = field(default_factory=list)
    bodies: list[tuple[str, str]] = field(default_factory=list)  # (kind, text) of needle blocks


def _compile(patterns) -> list[re.Pattern]:
    return [re.compile(p, re.M) for p in patterns]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1] or (a[0] == a[1] and b[0] <= a[0] < b[1])


def _matches(pats: list[re.Pattern], text: str) -> list[tuple[int, int]]:
    out = []
    for p in pats:
        out.extend((m.start(), m.end()) for m in p.finditer(text))
    return sorted(set(out))


def _slot_of(span, ev: Evidence) -> int | None:
    for s, e, i in ev.slots:
        if _overlaps(span, (s, e)):
            return i
    return None


def _inside(span, spans) -> bool:
    return any(_overlaps(span, sp) for sp in spans)


def _on_marker_line(span, text: str) -> bool:
    line_start = text.rfind("\n", 0, span[0]) + 1
    return bool(MARKER_RE.match(text, line_start))


def _guard_pattern(lit: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(lit) + r"(?![\w-])")


def synonym_groups(text: str) -> list[set[str]]:
    """Synonyms the haystack defines: `-x, --long` option pairs in help text,
    and `[alias: a]`, `[aliases: a, b]` or `[default alias: a]` notes."""
    groups = []
    for m in re.finditer(r"^[ \t]+(-[A-Za-z0-9]), (--[A-Za-z0-9][\w-]*)", text, re.M):
        groups.append({m.group(1), m.group(2)})
    for m in re.finditer(r"^[ \t]+([\w-]+)[ \t]+.*\[(?:default )?alias(?:es)?: ([^\]]+)\]", text, re.M):
        groups.append({m.group(1), *[a.strip() for a in m.group(2).split(",")]})
    return groups


def empty_containers(kind: str, body: str) -> set[str]:
    """Keys, tables and lists left with nothing in them: a YAML `key:` with
    no value and nothing indented under it, a TOML table header with no keys
    before the next header, a JSON `{}` or `[]`."""
    out = set()
    lines = body.splitlines()
    if kind == "yaml":
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)([\w.-]+):\s*$", line)
            if not m:
                continue
            nxt = next((x for x in lines[i + 1:] if x.strip()), None)
            if nxt is None or len(nxt) - len(nxt.lstrip()) <= len(m.group(1)):
                out.add(m.group(2))
    elif kind == "toml":
        for i, line in enumerate(lines):
            m = re.match(r"^\s*\[\[?([^\]]+)\]\]?\s*$", line)
            if not m:
                continue
            nxt = next((x for x in lines[i + 1:] if x.strip() and not x.strip().startswith("#")), None)
            if nxt is None or nxt.lstrip().startswith("["):
                out.add(m.group(1))
    elif kind in ("json", "jsonl"):
        def walk(v, path):
            if isinstance(v, (dict, list)) and not v and path:
                out.add(path)
            for k, x in (v.items() if isinstance(v, dict) else enumerate(v) if isinstance(v, list) else ()):
                walk(x, f"{path}.{k}")
        for doc in ([body] if kind == "json" else [x for x in lines if x.strip()]):
            try:
                walk(json.loads(doc), "$")
            except ValueError:
                pass
    return out


def _parse_body(kind: str, body: str) -> str | None:
    try:
        if kind == "json":
            json.loads(body)
        elif kind == "jsonl":
            for line in body.splitlines():
                if line.strip():
                    json.loads(line)
        elif kind == "toml":
            tomllib.loads(body)
    except (ValueError, tomllib.TOMLDecodeError) as e:
        return str(e).splitlines()[0]
    return None


def check(claim: dict, ev: dict[str, Evidence], slot_roles: list[str | None]) -> list[str]:
    """Return every hazard the triple breaks (empty when it is well-defined)."""
    problems: list[str] = []
    probes = claim.get("probes", {})
    support = _compile(probes.get("support", []))
    contra = _compile(probes.get("contradict", []))
    mention = _compile(probes.get("mention", []))
    near = _compile(probes.get("near", []))
    mech = claim.get("mechanism", "plain")
    role = claim.get("role", "")  # supersede: "belief" (X) or "reason" (Y)
    if not support or not contra:
        problems.append("claim: needs support and contradict probes")
        return problems

    # Byte identity outside the slots.
    fixed = {}
    for v in VARIANTS:
        e = ev[v]
        parts, last = [], 0
        for s, t, _ in sorted(e.slots):
            parts.append(e.text[last:s])
            last = t
        parts.append(e.text[last:])
        fixed[v] = parts
    if not (fixed["s"] == fixed["c"] == fixed["nm"]):
        problems.append("identity: the variants differ outside their slots")

    sup = {v: _matches(support, ev[v].text) for v in VARIANTS}
    con = {v: _matches(contra, ev[v].text) for v in VARIANTS}

    # Uniqueness and guard literals.
    for v in VARIANTS:
        e = ev[v]
        for kind, spans in (("support", sup[v]), ("contradiction", con[v])):
            for sp in spans:
                if _slot_of(sp, e) is None and not (mech == "version" and _inside(sp, e.scoped)):
                    problems.append(f"uniqueness: {v}: a {kind} match outside the needle: "
                                    f"{e.text[sp[0]:sp[1]][:80]!r}")
        for lit in claim.get("guard", []):
            for m in _guard_pattern(lit).finditer(e.text):
                if _slot_of((m.start(), m.end()), e) is None:
                    problems.append(f"uniqueness: {v}: guard literal {lit!r} outside the needle")

    # Distractors are near misses: they match `near`, never the claim.
    for v in VARIANTS:
        for d in ev[v].distractors:
            seg = ev[v].text[d[0]:d[1]]
            if near and not any(p.search(seg) for p in near):
                problems.append(f"distractor: {v}: a declared distractor does not match its near probe")

    # Version scope: a scoped block names a version, and not the haystack's.
    if mech == "version":
        own = claim.get("version", "")
        for v in VARIANTS:
            for d in ev[v].scoped:
                seg = ev[v].text[d[0]:d[1]]
                if not re.search(claim.get("scope_re", r"\d+\.\d+"), seg) or (own and own in seg):
                    problems.append(f"version: {v}: the scoped block must name another version than {own}")
            if own and not re.search(re.escape(own), ev[v].text):
                problems.append(f"version: {v}: the haystack never states version {own}")

    # NOT MENTIONED: nothing about the claim anywhere.
    for kind, spans in (("support", sup["nm"]), ("contradiction", con["nm"]),
                        ("mention", _matches(mention, ev["nm"].text))):
        for sp in spans:
            if mech == "version" and _inside(sp, ev["nm"].scoped):
                continue
            problems.append(f"nm: a {kind} match remains: {ev['nm'].text[sp[0]:sp[1]][:80]!r}")

    def in_needle(spans, v):
        return [sp for sp in spans if _slot_of(sp, ev[v]) is not None]

    if mech in ("plain", "version") or (mech == "supersede" and role == "reason"):
        if not in_needle(sup["s"], "s"):
            problems.append("s: no support match in the needle")
        if in_needle(con["s"], "s"):
            problems.append("s: the needle also contradicts the claim")
        if not in_needle(con["c"], "c"):
            problems.append("c: no contradiction match in the needle")
        if in_needle(sup["c"], "c"):
            problems.append("restatement: c: a copy of the supporting needle was not edited")
    elif mech == "supersede":  # a claim about the refuted belief X
        live_s = [sp for sp in in_needle(sup["s"], "s") if not _on_marker_line(sp, ev["s"].text)]
        if not live_s:
            problems.append("s: no live (unrefuted) statement of the belief")
        if [sp for sp in sup["s"] if _on_marker_line(sp, ev["s"].text)]:
            problems.append("s: the belief is also marked refuted")
        marked = [sp for sp in sup["c"] if _on_marker_line(sp, ev["c"].text)]
        live_c = [sp for sp in sup["c"] if not _on_marker_line(sp, ev["c"].text)]
        if not marked and not in_needle(con["c"], "c"):
            problems.append("c: no refutation of the belief")
        if live_c:
            problems.append("restatement: c: the belief is still stated as live")
    elif mech in ("sequence", "timestamp", "precedence"):
        def roles(spans, v):
            return {slot_roles[_slot_of(sp, ev[v])] for sp in spans if _slot_of(sp, ev[v]) is not None}
        if roles(sup["s"], "s") != {"winner"} or roles(con["s"], "s") != {"loser"}:
            problems.append("s: support must be in the deciding slot and the stale value in the other")
        if roles(con["c"], "c") != {"winner"} or roles(sup["c"], "c") != {"loser"}:
            problems.append("c: the values must swap between the two slots")
        starts = {}
        for s, _, i in ev["s"].slots:
            if slot_roles[i] in ("winner", "loser") and slot_roles[i] not in starts:
                starts[slot_roles[i]] = s
        if len(starts) == 2:
            if mech == "sequence" and not starts["winner"] > starts["loser"]:
                problems.append("sequence: the deciding record must come later in the text")
            if mech in ("timestamp", "precedence") and not starts["winner"] < starts["loser"]:
                problems.append(f"{mech}: the deciding record must come first, so position can't decide")
        else:
            problems.append(f"{mech}: needs one winner and one loser slot")
        if mech == "timestamp":
            tre = re.compile(claim.get("time_re", r"\d{4}-\d\d-\d\dT[\d:]+Z"))
            times = {}
            for s, t, i in ev["s"].slots:
                m = tre.search(ev["s"].text, s, t)
                if m and slot_roles[i] in ("winner", "loser"):
                    times[slot_roles[i]] = m.group(0)
            if len(times) != 2 or not times["winner"] > times["loser"]:
                problems.append("timestamp: the deciding record must carry the later time")
    else:
        problems.append(f"claim: unknown mechanism {mech!r}")

    # Derived values.
    for inv in claim.get("invariants", []):
        for v in VARIANTS:
            text = ev[v].text
            tot = re.search(inv["total"], text, re.M)
            if not tot:
                continue
            items = re.findall(inv["items"], text, re.M)
            if inv["type"] == "count":
                got = len(items)
            elif inv["type"] == "sum":
                got = sum(int(x) for x in items)
            else:
                problems.append(f"claim: unknown invariant {inv['type']!r}")
                break
            if got != int(tot.group(1)):
                problems.append(f"derived: {v}: {inv['type']} of {inv['items']!r} is {got}, "
                                f"but the haystack shows {tot.group(1)}")
    for v in VARIANTS:
        for kind, body in ev[v].bodies:
            err = _parse_body(kind, body)
            if err:
                problems.append(f"derived: {v}: a {kind} body no longer parses: {err}")

    # No defined synonyms.
    val, val_c = claim.get("value"), claim.get("value_c")
    if val and val_c:
        for v in VARIANTS:
            for g in synonym_groups(ev[v].text):
                if val in g and val_c in g:
                    problems.append(f"synonym: {v}: {val_c!r} is a defined synonym of {val!r}")
    return sorted(set(problems))
