"""The judge prompt and its answer parser, shared by Foundation Models (F),
the LLM judge (J) and the second reader.

The prompt is fixtures/tasks/verification/judge-prompt.toml; its sha256 is
the prompt sha recorded with every request. The verdict is read from the
final answer (`content`) only, never from reasoning (`reasoning_content`),
by two rules, both reported:

- **lenient** (the scored one): after trimming whitespace, `content` ends
  with one JSON object; prose before it is allowed. Prose only, more than
  one JSON object, or anything after the final object is a failure.
- **strict:** `content`, trimmed, is exactly one JSON object.

Either way the object's `verdict` must be "supported", "contradicted" or
"not_mentioned". A failure counts as wrong and is counted. A verdict object
inside a Markdown code fence fails both rules; `fenced` recognises such an
answer so the report can count it separately.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from tasks import canon, gen_verification as gen

PROMPT = gen.FIXTURES / "judge-prompt.toml"
VERDICTS = {"supported": "S", "contradicted": "C", "not_mentioned": "NM"}
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
          "required": ["verdict"], "additionalProperties": False}
_DECODER = json.JSONDecoder()


def load(path: Path = PROMPT) -> dict:
    with open(path, "rb") as f:
        p = tomllib.load(f)
    p["sha256"] = canon.sha256_file(path)
    return p


def messages(prompt: dict, form: str, claim: str, evidence: str) -> list[dict]:
    """form: "full" (a whole haystack) or "window" (one window or chunk)."""
    user = prompt["full" if form == "full" else "window"].format(claim=claim, evidence=evidence.rstrip("\n"))
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def _verdict(obj) -> tuple[str | None, bool]:
    if not isinstance(obj, dict) or obj.get("verdict") not in VERDICTS:
        return None, False
    return VERDICTS[obj["verdict"]], True


def json_objects(text: str) -> list[tuple[int, int, object]]:
    """Top-level JSON objects in text, left to right, as (start, end, value).
    Braces that don't open a valid JSON object are prose."""
    out, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append((i, end, obj))
            i = end
        else:
            i += 1


def parse(content: str | None) -> tuple[str | None, bool]:
    """The scored (lenient) rule: (verdict as S/C/NM, parsed)."""
    if content is None:
        return None, False
    text = content.strip()
    objs = json_objects(text)
    if len(objs) != 1 or objs[0][1] != len(text):
        return None, False  # prose only, several objects, or text after the object
    return _verdict(objs[0][2])


def fenced(content: str | None) -> bool:
    """Whether the answer's final non-whitespace text is a closing ```
    fence wrapping a valid verdict object (with or without a language tag,
    prose allowed before the opening fence). Such an answer fails both
    rules; this only lets the report count how often it happens."""
    if content is None:
        return False
    text = content.strip()
    if not text.endswith("```"):
        return False
    body = text[:-3]
    opening = body.rfind("```")
    if opening < 0:
        return False
    inner = body[opening + 3:]
    tag = len(inner) - len(inner.lstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"))
    return parse_strict(inner[tag:])[1]


def parse_strict(content: str | None) -> tuple[str | None, bool]:
    """The strict rule: the trimmed content is exactly one JSON object."""
    if content is None:
        return None, False
    try:
        obj = json.loads(content.strip())
    except ValueError:
        return None, False
    return _verdict(obj)


def final_content(response: dict) -> str | None:
    """The final answer from a response line: `content`, or an OpenAI
    chat.completion's first choice's message content. Reasoning fields
    (`reasoning_content`, `reasoning`) are never read."""
    if "content" in response:
        return response["content"]
    try:
        return response["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
