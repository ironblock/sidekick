"""The scrubber and the refusal scan for captured tool output.

The repository is public, so a capture must carry nothing that identifies a
machine or a person. Captures run under a fixed working root with a fixed
HOME and clock (tools/tasks/capture.py), then:

1. `scrub` maps the working root to `/work` (in both its `/tmp/…` and
   `/private/tmp/…` spellings) and applies the recipe's normalization rules.
2. `scan` looks for anything that should not be there and returns findings.
   The capture stage refuses to write a capture with any finding that the
   recipe does not allowlist with a reason.

The scan looks for:
- the real home directory, user name and host name of the machine running it;
- absolute paths under /Users/, /home/, /root/, /var/folders/, /private/;
- IPv4 addresses other than loopback, 0.0.0.0 and the documentation ranges
  (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24), and full IPv6 addresses
  other than loopback and 2001:db8::/32;
- e-mail addresses outside example.com, example.org, example.net and the
  reserved .example/.test/.invalid/.localhost names;
- MAC addresses;
- dates outside the synthetic clock's range;
- secrets: GitHub, OpenAI-style and AWS key prefixes, private-key blocks,
  and long high-entropy tokens (mixed case and digits; hex ids such as
  commit hashes are exempt).
"""

from __future__ import annotations

import getpass
import ipaddress
import math
import os
import re
import socket
from dataclasses import dataclass

PLACEHOLDER_ROOT = "/work"


@dataclass(frozen=True)
class Finding:
    rule: str
    text: str
    offset: int

    def __str__(self) -> str:
        return f"{self.rule}: {self.text!r} at {self.offset}"


def scrub(text: str, roots: list[str], rules: list[tuple[str, str]] = ()) -> str:
    spellings = set()
    for root in roots:
        root = root.rstrip("/")
        spellings.add(root)
        if root.startswith("/tmp/"):
            spellings.add("/private" + root)
        if root.startswith("/private/tmp/"):
            spellings.add(root[len("/private"):])
    for s in sorted(spellings, key=len, reverse=True):
        text = text.replace(s, PLACEHOLDER_ROOT)
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.M)
    return text


def machine_identity() -> dict:
    """The identifiers of the machine running the capture, which must never
    appear in one."""
    home = os.path.expanduser("~")
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user database entry
        user = ""
    host = socket.gethostname()
    hosts = {h for h in (host, host.split(".")[0]) if h and h not in ("localhost",)}
    return {"home": home, "user": user, "hosts": sorted(hosts)}


_PATH_RE = re.compile(r"(?<![\w.])/(?:Users|home|root|var/folders|private)/[^\s'\"`)]*")
_IPV4_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.]*\d)")
_IPV6_RE = re.compile(r"(?<![\w:])((?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}:?){1,6})(?![\w:])")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?![\w.-]*\d)")
_MAC_RE = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:])")
_DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d\d)-(\d\d)-(\d\d)(?!\d)")
_SECRET_RES = [
    ("secret:github", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("secret:openai", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("secret:aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret:private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/_=-]{25,}")
_ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
_RESERVED_TLDS = (".example", ".test", ".invalid", ".localhost")
_DOC_NETS = [ipaddress.ip_network(n) for n in
             ("127.0.0.0/8", "0.0.0.0/32", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
              "::1/128", "2001:db8::/32")]


def _entropy(s: str) -> float:
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _allowed_ip(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return True  # not an address (a version string such as 1.2.3.4000)
    return any(ip in net for net in _DOC_NETS if net.version == ip.version)


def scan(text: str, identity: dict | None = None, clock: tuple[str, str] = ("2026-10-01", "2026-10-01"),
         extra_literals: list[str] = ()) -> list[Finding]:
    """Return every finding in `text`. `clock` is the inclusive date range of
    the synthetic clock; `extra_literals` are more strings to refuse (the
    capture's own working root, for one)."""
    identity = identity if identity is not None else machine_identity()
    found: list[Finding] = []

    def literal(rule, needle, word=False):
        if not needle:
            return
        pat = re.compile((r"(?<![\w])" + re.escape(needle) + r"(?![\w])") if word else re.escape(needle),
                         re.IGNORECASE if word else 0)
        for m in pat.finditer(text):
            found.append(Finding(rule, m.group(0), m.start()))

    literal("home", identity.get("home", ""))
    if len(identity.get("user", "")) >= 3:
        literal("user", identity["user"], word=True)
    for h in identity.get("hosts", []):
        literal("host", h, word=True)
    for lit in extra_literals:
        literal("root", lit)

    for m in _PATH_RE.finditer(text):
        found.append(Finding("path", m.group(0), m.start()))
    for m in _IPV4_RE.finditer(text):
        octets = m.group(1).split(".")
        if all(int(o) <= 255 for o in octets) and not _allowed_ip(m.group(1)):
            found.append(Finding("ip", m.group(1), m.start()))
    for m in _IPV6_RE.finditer(text):
        s = m.group(1)
        if not re.search(r"\d", s) or (not re.search(r"[A-Fa-f]", s) and s.count(":") < 3):
            continue  # clock times, ratios and revset ranges such as B::D, not addresses
        try:
            ipaddress.IPv6Address(s.rstrip(":") if s.endswith(":") and not s.endswith("::") else s)
        except ValueError:
            continue
        if not _allowed_ip(s):
            found.append(Finding("ip", s, m.start()))
    for m in _EMAIL_RE.finditer(text):
        domain = m.group(1).lower()
        if domain in _ALLOWED_EMAIL_DOMAINS or domain.endswith(tuple("." + d for d in _ALLOWED_EMAIL_DOMAINS)) \
                or domain.endswith(_RESERVED_TLDS):
            continue
        found.append(Finding("email", m.group(0), m.start()))
    for m in _MAC_RE.finditer(text):
        found.append(Finding("mac", m.group(0), m.start()))
    lo, hi = clock
    for m in _DATE_RE.finditer(text):
        d = m.group(0)
        if not (lo <= d <= hi):
            found.append(Finding("date", d, m.start()))
    for rule, pat in _SECRET_RES:
        for m in pat.finditer(text):
            found.append(Finding(rule, m.group(0), m.start()))
    for m in _TOKEN_RE.finditer(text):
        # URL paths are split at slashes; each part is judged on its own.
        start = m.start()
        for s in m.group(0).split("/"):
            at, start = start, start + len(s) + 1
            if len(s) < 25 or re.fullmatch(r"[0-9a-f]+", s) or re.fullmatch(r"[0-9A-F]+", s):
                continue  # short, or a hex id such as a commit hash
            if not (re.search(r"[a-z]", s) and re.search(r"[A-Z]", s) and re.search(r"\d", s)):
                continue
            if _entropy(s) >= 4.0:
                found.append(Finding("secret:entropy", s, at))
    return sorted(found, key=lambda f: (f.offset, f.rule))


def unallowed(findings: list[Finding], allow: list[dict], capture_id: str) -> list[Finding]:
    """Drop findings that an allowlist entry covers. An entry is
    `{text, reason, capture?}`: the finding's text must equal `text`, and
    `capture` (if given) must equal the capture id or end in `*` and prefix it."""
    out = []
    for f in findings:
        ok = False
        for a in allow:
            if a["text"] != f.text:
                continue
            cap = a.get("capture", "*")
            if cap == "*" or cap == capture_id or (cap.endswith("*") and capture_id.startswith(cap[:-1])):
                ok = True
                break
        if not ok:
            out.append(f)
    return out
