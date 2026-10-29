"""Score a release-gate run of a chat model (tools/release_gate.py): the
chat benchmark's headline metric per family and mode, the guided replies
that don't satisfy their schema, and the requests whose `usage` doesn't
match the build's own tokenizer and chat template.

    python tools/chat_bench/gate_score.py <run.json> <build dir>

Reads the gate's run file ({"modes": {mode: {"records": [...]}}}, each
record as tools/chat_bench/run.py writes one, with the request's
`messages`), and prints JSON to stdout. Run it with the chat benchmark's
environment (tools/chat_bench/requirements.txt), which has the scorers'
packages and transformers, for the template.

Token counts. `prompt_tokens` must be the rendered prompt's length, as
transformers renders the build's template with its `[chat]
template_kwargs`; `completion_tokens` must be the reply's tokens re-encoded,
plus one when the reply ended on a stop token (finish_reason `stop`), which
the daemon counts. A mismatch is counted, not judged here: the gate holds
the count to its baseline.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_bench import score, tasks  # noqa: E402


def counter(build: Path):
    """(prompt_tokens(messages), completion_tokens(text)) with the build's
    tokenizer and template."""
    from transformers import PreTrainedTokenizerFast
    m = tomllib.loads((build / "chat.toml").read_text())
    chat = m.get("chat", {})
    tok = PreTrainedTokenizerFast(tokenizer_file=str(build / m["tokenizer"]),
                                  bos_token=chat.get("bos_token"), eos_token=chat.get("eos_token"))
    template = (build / chat["template"]).read_text()
    kwargs = chat.get("template_kwargs", {})

    def prompt(messages):
        ids = tok.apply_chat_template(messages, chat_template=template, add_generation_prompt=True,
                                      tokenize=True, return_dict=False, **kwargs)
        return len(ids)

    def completion(text):
        return len(tok.encode(text, add_special_tokens=False))
    return prompt, completion


def schema_ok(content: str, schema: dict) -> bool:
    import jsonschema
    value, strict = score.parse_json(content)
    if value is None or not strict:
        return False
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return False
    return True


def main() -> int:
    run = json.loads(Path(sys.argv[1]).read_text())
    build = Path(sys.argv[2])
    cfg, items, schemas = tasks.load_config(), tasks.load_items(), tasks.load_schemas()
    emb = score.Embedder(None, "none", Path("/nonexistent"))
    prompt_tokens, completion_tokens = counter(build)
    out = {"scores": {}, "schema_invalid": 0, "prompt_token_mismatches": 0, "completion_token_mismatches": 0}
    for mode, part in run["modes"].items():
        fam, _ = score.score_run(part, cfg, items, schemas, emb)
        for f, v in fam.items():
            out["scores"][f"{mode}:{f}"] = round(v[score.HEADLINE[f]]["value"], 6)
        for r in part["records"]:
            if r["status"] != 200:
                continue
            if mode == "guided" and r.get("schema") is not None and not schema_ok(r["content"], r["schema"]):
                out["schema_invalid"] += 1
            u = r.get("usage") or {}
            if u.get("prompt_tokens") != prompt_tokens(r["messages"]):
                out["prompt_token_mismatches"] += 1
            want = completion_tokens(r["content"]) + (1 if r.get("finish_reason") == "stop" else 0)
            if u.get("completion_tokens") != want:
                out["completion_token_mismatches"] += 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
