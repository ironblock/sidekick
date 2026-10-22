"""Conventions shared by every task corpus: canonical JSON lines, sha256
locks and seeded sub-seeds.

- A record's canonical form is UTF-8 JSON with `ensure_ascii=False`, sorted
  keys and separators `(",", ":")`, on one line ending in a newline.
- A record's sha256 is the sha256 of that line; a set's sha256 is the sha256
  of the whole file, whose lines are sorted by `id`.
- A sub-seed is `sha256(seed ‖ part ‖ part …)` (parts joined by U+001F), so
  adding a claim or a source never reshuffles the others. Nothing reads the
  clock or depends on set iteration order.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


def canonical_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_sha256(record: dict) -> str:
    return sha256_text(canonical_line(record))


def write_jsonl(path: Path, records: list[dict]) -> str:
    """Write records sorted by id in canonical form; return the file's sha256."""
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record ids")
    body = "".join(canonical_line(r) for r in sorted(records, key=lambda r: r["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode("utf-8"))
    return sha256_text(body)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, obj) -> None:
    """Pretty, stable JSON for committed locks (sorted keys, two-space indent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sub_seed(seed: int | str, *parts: str) -> int:
    key = "\x1f".join([str(seed), *parts])
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def rng(seed: int | str, *parts: str) -> random.Random:
    return random.Random(sub_seed(seed, *parts))


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def approx_tokens(text: str) -> int:
    """A tokenizer-free length estimate (`approx-v1`): one token per digit and
    per punctuation mark, and one per started six letters of a word. It tracks
    subword tokenizers within a few tens of percent on CLI output and prose,
    which is enough to aim haystacks at 1k/4k/12k/32k; each model's exact
    count is measured at run time."""
    n = 0
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        n += (len(s) + 5) // 6 if s[0].isalpha() else 1
    return n


LENGTH_UNIT = "approx-v1"
