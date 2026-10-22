"""Windows, anchors and deciding spans for scoring the verification corpus.

A per-window checker (sidekick's nli models, MiniCheck) never sees a whole
haystack. This module decides what it does see, and what it would have to
see to get each variant right:

- **Windows** (design §7.5). The evidence is cut at line boundaries into
  windows of at most W text tokens, W = max_seq_len − claim tokens −
  specials − margin (8), overlapping by at least `overlap` (128) tokens, the
  generator's cap on a needle. A line longer than W is cut at token
  boundaries. Each window is then checked as the exact pair the server
  builds, so no window can be truncated (`fits`).
- **MiniCheck chunks.** MiniCheck groups sentences into chunks of at most
  400 tokens, with no overlap. Without its sentence splitter (nltk) this
  module groups lines instead; the ceiling it gives for MiniCheck is marked
  approximate.
- **Anchors** (#133, the gated check). Taken from the claim text by one
  rule: backticked spans, flag-like tokens (`-x`, `--long`), file names
  (including dot-files) and version strings. A gated checker scores only
  the windows that contain at least one anchor, as a substring.
- **Deciding spans.** For each variant, the minimal evidence that decides
  it, from the generator's slot records and the claim's probes: a probe
  match inside a slot, widened back to its block's header line (the
  `[call cN] $ <command>` or `[notes]` line that says what the text is) and
  forward to the end of its line. Per variant, the spans are groups; a
  window decides the variant if it holds one span from every group:
  - SUPPORTED: a support match (any copy); CONTRADICTED: a contradiction
    match (any copy).
  - `sequence`, `precedence`, `timestamp`: both records, the deciding
    (winner) one and the stale (loser) one.
  - `supersede`, a claim about the refuted belief: in CONTRADICTED, the
    `was:` line and the call it cites.
  - `version`: also a line stating the haystack's own version.
  - A claim's `conjuncts`: one match of each, in SUPPORTED.
  - NOT MENTIONED needs nothing: no window can show what was removed. (For
    `version` claims, `nm_strict_groups` gives the stricter reading, in
    which the older-version note and the version line must share a window.)

Tokenizers are injected (`Tokenizer`), so tests run on the standard library;
`HFTokenizer` wraps the `tokenizers` package and a tokenizer.json.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

HEADER_RE = re.compile(r"^(?:\[call c\d+\] \$ |\[notes\]$)", re.M)
MARKER_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?was:[ \t]", re.M)


# ---------------------------------------------------------------- tokenizers


class Tokenizer:
    """What windowing needs from a tokenizer."""

    sha256: str = ""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character spans of the text's tokens, without special tokens."""
        raise NotImplementedError

    def pair_len(self, text: str, hypothesis: str) -> int:
        """Length of the pair (text, hypothesis) with special tokens, as the
        server lays it out."""
        raise NotImplementedError

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def specials(self) -> int:
        return self.pair_len("a", "b") - self.count("a") - self.count("b")


class HFTokenizer(Tokenizer):
    """A tokenizer.json, through the `tokenizers` package, checked by sha256."""

    def __init__(self, path: Path, want_sha256: str | None = None):
        from tokenizers import Tokenizer as _T  # optional dependency

        data = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        if want_sha256 and self.sha256 != want_sha256:
            raise SystemExit(f"{path}: tokenizer.json sha256 {self.sha256} != pinned {want_sha256}")
        self._t = _T.from_str(data.decode("utf-8"))
        self._t.no_truncation()
        self._t.no_padding()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._t.encode(text, add_special_tokens=False)
        # Some tokenizers emit zero-width tokens (a bare "▁"); keep them, they count.
        return [tuple(o) for o in enc.offsets]

    def pair_len(self, text: str, hypothesis: str) -> int:
        return len(self._t.encode(text, hypothesis, add_special_tokens=True).ids)

    def count(self, text: str) -> int:
        return len(self._t.encode(text, add_special_tokens=False).ids)


class WordTokenizer(Tokenizer):
    """A stand-in for tests: one token per run of non-space characters, and
    3 special tokens per pair ([CLS] text [SEP] hypothesis [SEP])."""

    sha256 = "word-tokenizer"
    _RE = re.compile(r"\S+")

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in self._RE.finditer(text)]

    def pair_len(self, text: str, hypothesis: str) -> int:
        return self.count(text) + self.count(hypothesis) + 3


# ---------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    start: int        # character offsets into the evidence; end excludes a final newline
    end: int
    text: str

    def holds(self, span: tuple[int, int]) -> bool:
        return self.start <= span[0] and span[1] <= self.end


def text_budget(tok: Tokenizer, claim: str, max_seq_len: int, margin: int = 8) -> int:
    """W = max_seq_len − claim tokens − specials − margin."""
    return max_seq_len - tok.count(claim) - tok.specials() - margin


def _units(text: str, offs: list[tuple[int, int]], budget: int) -> list[tuple[int, int, int]]:
    """Lines as (start, end including the newline, tokens), with any line of
    more than `budget` tokens cut at token boundaries."""
    starts = [s for s, _ in offs]
    units, pos = [], 0
    while pos < len(text):
        nl = text.find("\n", pos)
        end = len(text) if nl < 0 else nl + 1
        lo, hi = bisect.bisect_left(starts, pos), bisect.bisect_left(starts, end)
        n = hi - lo
        if n <= budget:
            units.append((pos, end, n))
        else:
            cut = pos
            for k in range(lo, hi, budget):
                nxt = starts[k + budget] if k + budget < hi else end
                units.append((cut, nxt, min(budget, hi - k)))
                cut = nxt
        pos = end
    return units


def windows(text: str, tok: Tokenizer, claim: str, max_seq_len: int, overlap: int = 128,
            margin: int = 8) -> list[Window]:
    """Cut the evidence into windows that each fit the model with the claim."""
    budget = text_budget(tok, claim, max_seq_len, margin)
    if budget <= overlap:
        raise ValueError(f"claim leaves {budget} tokens per window, not more than the {overlap}-token overlap")
    units = _units(text, tok.offsets(text), budget)
    out, i = [], 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and total + units[j][2] <= budget:
            total += units[j][2]
            j += 1
        j = max(j, i + 1)
        while True:  # the exact pair must fit; shrink by a line if a boundary effect overflows it
            start, end = units[i][0], units[j - 1][1]
            body = text[start:end]
            body = body[:-1] if body.endswith("\n") else body
            if tok.pair_len(body, claim) <= max_seq_len or j == i + 1:
                break
            j -= 1
        if body.strip():
            out.append(Window(start, start + len(body), body))
        if j >= len(units):
            break
        # Start the next window far enough back to overlap by `overlap` tokens.
        k, back = j, 0
        while k > i + 1 and back < overlap:
            k -= 1
            back += units[k][2]
        i = k
    return out


def fits(windows_: list[Window], tok: Tokenizer, claim: str, max_seq_len: int) -> list[int]:
    """Each window's exact pair length; raises if any is over max_seq_len."""
    lens = [tok.pair_len(w.text, claim) for w in windows_]
    over = [n for n in lens if n > max_seq_len]
    if over:
        raise AssertionError(f"{len(over)} window(s) over {max_seq_len} tokens: {over[:3]}")
    return lens


def chunks(text: str, tok: Tokenizer, size: int = 400) -> list[Window]:
    """MiniCheck-style chunks, approximated at line granularity: consecutive
    lines up to `size` tokens, no overlap. (MiniCheck splits sentences with
    nltk inside each line; a chunk boundary can then fall mid-line.)"""
    units = _units(text, tok.offsets(text), size)
    out, cur, n = [], [], 0
    for u in units:
        if cur and n + u[2] > size:
            out.append(cur)
            cur, n = [], 0
        cur.append(u)
        n += u[2]
    if cur:
        out.append(cur)
    res = []
    for c in out:
        body = text[c[0][0]:c[-1][1]]
        body = body[:-1] if body.endswith("\n") else body
        if body.strip():
            res.append(Window(c[0][0], c[0][0] + len(body), body))
    return res


def minicheck_chunks(text: str, count, sent_tokenize, size: int = 400) -> list[Window]:
    """MiniCheck's own chunks (Liyan06/MiniCheck, `Inferencer.inference_per_example`
    for roberta and deberta): the document split at newlines, each line split
    into sentences by `sent_tokenize` (nltk), a newline kept as a sentence of
    its own between lines; sentences grouped while their token counts
    (`count`, without special tokens) stay within `size`; each chunk joined
    with spaces, " \\n " turned back into "\\n", stripped, and empty chunks
    dropped. Each chunk keeps MiniCheck's text and the evidence span from its
    first to its last sentence, so deciding spans can be checked against it."""
    sents: list[tuple[str, int, int]] = []  # (text, start, end) in the evidence
    pos = 0
    lines = text.split("\n")
    for n, line in enumerate(lines):
        at = 0
        for s in sent_tokenize(line):
            i = line.find(s, at)
            if i < 0:
                raise ValueError(f"sentence not found in its line: {s[:60]!r}")
            sents.append((s, pos + i, pos + i + len(s)))
            at = i + len(s)
        if n < len(lines) - 1:
            sents.append(("\n", pos + len(line), pos + len(line) + 1))
        pos += len(line) + 1
    sents = sents or [("", 0, 0)]
    groups, cur, n = [], [], 0
    for s in sents:
        k = count(s[0])
        if n + k > size:
            groups.append(cur)
            cur, n = [s], k
        else:
            cur.append(s)
            n += k
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        body = " ".join(s[0] for s in g).replace(" \n ", "\n").strip()
        if not body:
            continue
        real = [s for s in g if s[0] != "\n" and s[0].strip()]
        out.append(Window(real[0][1], real[-1][2], body))
    return out


def key(text: str, claim: str) -> str:
    """Dedupe key for one (window, claim) prediction."""
    return hashlib.sha256((text + "\x1f" + claim).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- anchors


_ANCHOR_RES = [
    re.compile(r"`([^`\n]+)`"),                                                  # backticked spans
    re.compile(r"(?<![\w-])(--[A-Za-z0-9][\w-]*|-[A-Za-z0-9])(?![\w-])"),        # flags
    re.compile(r"(?<![\w./@-])((?:[\w-]+/)*[\w-]*[A-Za-z][\w-]*(?:\.[\w-]+)*\.[A-Za-z][A-Za-z0-9]{0,4})"
               r"(?![\w(-]|\.[\w])"),                                             # file (and dotted) names
    re.compile(r"(?<!\S)(\.[A-Za-z][\w-]*)(?![\w.(])"),                           # dot-files (.jj, .git)
    re.compile(r"(?<![\w.])(\d+(?:\.\d+)+)(?![\w])"),                             # version strings
]


def anchors(claim: str) -> list[str]:
    """The claim's anchors, by the fixed rule, in order of first appearance."""
    found: list[tuple[int, str]] = []
    for rx in _ANCHOR_RES:
        for m in rx.finditer(claim):
            a = m.group(1).rstrip(".,;:")
            if a:
                found.append((m.start(1), a))
    seen, out = set(), []
    for _, a in sorted(found):
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def gated(windows_: list[Window], anchor_list: list[str]) -> list[bool]:
    return [any(a in w.text for a in anchor_list) for w in windows_]


# ---------------------------------------------------------------- deciding spans


def slot_spans(triple: dict, v: str) -> list[tuple[int, int, str | None]]:
    """(start, end, role) of each slot in variant v's evidence."""
    out, off = [], len(triple["segments"][0])
    for slot, seg in zip(triple["slots"], triple["segments"][1:]):
        out.append((off, off + len(slot[v]), slot.get("role")))
        off += len(slot[v]) + len(seg)
    return out


def _unit(text: str, headers: list[int], s: int, e: int) -> tuple[int, int]:
    i = bisect.bisect_right(headers, s) - 1
    start = headers[i] if i >= 0 else text.rfind("\n", 0, s) + 1
    nl = text.find("\n", max(s, e - 1))
    return start, (len(text) if nl < 0 else nl)


def _in_slot(span, slots) -> tuple[bool, str | None]:
    for a, b, role in slots:
        if a < b and span[0] < b and a < span[1]:
            return True, role
    return False, None


def _on_marker_line(text: str, pos: int) -> bool:
    return bool(MARKER_RE.match(text, text.rfind("\n", 0, pos) + 1))


def _matches(patterns, text):
    for p in patterns:
        for m in re.finditer(p, text, re.M):
            yield m.span()


def deciding_groups(triple: dict, claim: dict, v: str, text: str) -> list[list[tuple[int, int]]]:
    """Variant v's deciding spans, as groups (one span from each must be in
    one window). NOT MENTIONED has none."""
    if v == "nm":
        return []
    probes = claim["probes"]
    mech = claim.get("mechanism", "plain")
    headers = [m.start() for m in HEADER_RE.finditer(text)]
    slots = slot_spans(triple, v)

    def units(patterns, role=None, marker=None):
        out = []
        for sp in _matches(patterns, text):
            inside, r = _in_slot(sp, slots)
            if not inside or (role is not None and r != role):
                continue
            if marker is not None and _on_marker_line(text, sp[0]) != marker:
                continue
            out.append(_unit(text, headers, *sp))
        return sorted(set(out))

    sup, con = probes["support"], probes["contradict"]
    groups: list[list[tuple[int, int]]]
    if mech in ("sequence", "precedence", "timestamp"):
        groups = ([units(sup, "winner"), units(con, "loser")] if v == "s"
                  else [units(con, "winner"), units(sup, "loser")])
    elif mech == "supersede" and claim.get("role") == "belief":
        groups = ([units(sup, marker=False)] if v == "s"
                  else [units(sup, marker=True), units(con)])
    else:
        groups = [units(sup)] if v == "s" else [units(con)]
    if v == "s":
        for p in claim.get("conjuncts", []):
            groups.append(units([p]))
    if mech == "version":
        groups.append(version_units(claim, text, headers))
    return groups


def version_units(claim: dict, text: str, headers: list[int] | None = None) -> list[tuple[int, int]]:
    """Lines stating the haystack's own version (the claim's `version`)."""
    if headers is None:
        headers = [m.start() for m in HEADER_RE.finditer(text)]
    pat = r"(?<![\w.])" + re.escape(claim["version"]) + r"(?![\w.]*\d)"
    out = []
    for m in re.finditer(pat, text):
        if not _on_marker_line(text, m.start()) and "from notes kept under" not in _line(text, m.start()):
            out.append(_unit(text, headers, *m.span()))
    return sorted(set(out))


def _line(text: str, pos: int) -> str:
    a = text.rfind("\n", 0, pos) + 1
    b = text.find("\n", pos)
    return text[a:len(text) if b < 0 else b]


def scoped_units(claim: dict, text: str) -> list[tuple[int, int]]:
    """Where each version-scoped note sits (its rendered `[notes]` block)."""
    out = []
    for sc in claim.get("scoped", []):
        block = "[notes]\n" + "".join(line + "\n" for line in sc["notes"])
        at = text.find(block)
        if at >= 0:
            out.append((at, at + len(block) - 1))
    return out


def nm_strict_groups(claim: dict, text: str) -> list[list[tuple[int, int]]]:
    """The stricter NOT MENTIONED reading for a `version` claim: the
    older-version note and the haystack's version line share a window."""
    if claim.get("mechanism") != "version":
        return []
    return [scoped_units(claim, text), version_units(claim, text)]


def reachable(windows_: list[Window], groups: list[list[tuple[int, int]]],
              mask: list[bool] | None = None) -> bool:
    """Whether some window (among those `mask` keeps) holds one span of every
    group. With no groups, any variant is reachable."""
    if any(not g for g in groups):
        return False
    if not groups:
        return True
    for n, w in enumerate(windows_):
        if mask is not None and not mask[n]:
            continue
        if all(any(w.holds(u) for u in g) for g in groups):
            return True
    return False


def gate_reachable(text: str, groups: list[list[tuple[int, int]]], anchor_list: list[str]) -> bool:
    """The gate ceiling (design §2.2): some choice of deciding spans contains
    an anchor. With no groups, the gate's NOT MENTIONED answer is right."""
    if not groups:
        return True
    if any(not g for g in groups) or not anchor_list:
        return False
    for pick in itertools.product(*groups):
        if any(a in text[s:e] for s, e in pick for a in anchor_list):
            return True
    return False
