"""The judge prompt and its answer parser, shared by Foundation Models (F),
the LLM judge (J) and the second reader.

The prompt is fixtures/tasks/verification/judge-prompt.toml; its sha256 is
the prompt sha recorded with every request. The verdict is read from the
final answer only (never from reasoning): after trimming whitespace and an
optional Markdown code fence, the content must be exactly one JSON object
whose `verdict` is "supported", "contradicted" or "not_mentioned". Anything
else is a parse failure, which counts as wrong and is counted.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from tasks import canon, gen_verification as gen

PROMPT = gen.FIXTURES / "judge-prompt.toml"
VERDICTS = {"supported": "S", "contradicted": "C", "not_mentioned": "NM"}
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
          "required": ["verdict"], "additionalProperties": False}
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*?)\n?```\Z", re.S)


def load(path: Path = PROMPT) -> dict:
    with open(path, "rb") as f:
        p = tomllib.load(f)
    p["sha256"] = canon.sha256_file(path)
    return p


def messages(prompt: dict, form: str, claim: str, evidence: str) -> list[dict]:
    """form: "full" (a whole haystack) or "window" (one window or chunk)."""
    user = prompt["full" if form == "full" else "window"].format(claim=claim, evidence=evidence.rstrip("\n"))
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def parse(content: str | None) -> tuple[str | None, bool]:
    """(verdict as S/C/NM, parsed) from a final answer."""
    if content is None:
        return None, False
    text = content.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except ValueError:
        return None, False
    if not isinstance(obj, dict) or obj.get("verdict") not in VERDICTS:
        return None, False
    return VERDICTS[obj["verdict"]], True


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
