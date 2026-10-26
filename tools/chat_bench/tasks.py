"""The chat benchmark's task set (fixtures/tasks/chat/): loading the items and
turning each into the OpenAI chat request every row is sent.

A request is the same for every model. The one thing a row changes is whether
a JSON family's request carries `response_format: json_schema` (guided
generation) or relies on the prompt alone; the prompt names the schema either
way, so the two differ only by the constraint.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO / "fixtures" / "tasks" / "chat"

# The request families, in report order. A family is one kind of request;
# tags items are asked twice (fixed vocabulary, then free-form).
FAMILIES = ("titles", "tags_vocab", "tags_free", "extraction", "instructions")
JSON_FAMILIES = ("tags_vocab", "extraction")


def load_config(tasks_dir: Path = TASKS_DIR) -> dict:
    with open(tasks_dir / "tasks.toml", "rb") as f:
        return tomllib.load(f)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(tasks_dir: Path = TASKS_DIR) -> dict[str, list[dict]]:
    """Items per source file: titles, tags, extraction, instructions."""
    cfg = load_config(tasks_dir)
    return {name: load_jsonl(tasks_dir / cfg[name]["items"])
            for name in ("titles", "tags", "extraction", "instructions")}


def load_schemas(tasks_dir: Path = TASKS_DIR) -> dict[str, dict]:
    cfg = load_config(tasks_dir)
    with open(tasks_dir / cfg["extraction"]["schemas"], encoding="utf-8") as f:
        return json.load(f)


def tags_schema(cfg: dict) -> dict:
    return {"type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string",
                                                                "enum": cfg["tags"]["vocabulary"]},
                                    "minItems": 1, "maxItems": cfg["tags"]["vocab"]["max_tags"]}},
            "required": ["tags"]}


def transcript(messages: list[dict]) -> str:
    names = {"user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{names[m['role']]}: {m['content']}" for m in messages)


def requests(tasks_dir: Path = TASKS_DIR) -> list[dict]:
    """Every request of the task set, in a fixed order. Each is
    {family, item, messages, max_tokens, schema (JSON families) or None}."""
    cfg = load_config(tasks_dir)
    items = load_items(tasks_dir)
    schemas = load_schemas(tasks_dir)
    out: list[dict] = []

    def add(family, item, system, user, max_tokens, schema=None):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        out.append({"family": family, "item": item, "messages": messages,
                    "max_tokens": max_tokens, "schema": schema})

    t = cfg["titles"]
    for it in items["titles"]:
        if it["kind"] == "conversation":
            user = t["conversation"].format(transcript=transcript(it["messages"]))
        else:
            user = t["document"].format(text=it["text"])
        add("titles", it["id"], t["system"], user, t["max_tokens"])

    vocab = ", ".join(cfg["tags"]["vocabulary"])
    tv, tf = cfg["tags"]["vocab"], cfg["tags"]["free"]
    schema = tags_schema(cfg)
    for it in items["tags"]:
        add("tags_vocab", it["id"], tv["system"], tv["user"].format(vocabulary=vocab, text=it["text"]),
            tv["max_tokens"], schema)
    for it in items["tags"]:
        add("tags_free", it["id"], tf["system"], tf["user"].format(text=it["text"]), tf["max_tokens"])

    e = cfg["extraction"]
    for it in items["extraction"]:
        s = schemas[it["schema"]]
        add("extraction", it["id"], e["system"],
            e["user"].format(schema=json.dumps(s, ensure_ascii=False), text=it["text"]), e["max_tokens"], s)

    ins = cfg["instructions"]
    for it in items["instructions"]:
        add("instructions", it["id"], None, it["prompt"], ins["max_tokens"])
    return out


def task_set_lock(tasks_dir: Path = TASKS_DIR) -> dict:
    """sha256 of every task-set file and of the rendered requests: what a run
    was measured on."""
    files = sorted(p for p in tasks_dir.iterdir() if p.suffix in (".jsonl", ".json", ".toml"))
    lock = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    rendered = json.dumps(requests(tasks_dir), sort_keys=True, ensure_ascii=False).encode()
    lock["requests"] = hashlib.sha256(rendered).hexdigest()
    lock["version"] = load_config(tasks_dir)["version"]
    return lock


def refused_probes(run: dict, controls: int = 0) -> list[dict]:
    """The requests a run's model refused (HTTP 400 content_filter), plus the
    first `controls` answered fixed-vocabulary tags requests, as
    tools/chat_bench/guardrails_probe.swift reads them."""
    cfg = load_config()
    by_key = {(r["family"], r["item"]): r for r in requests()}
    refused = [(r["family"], r["item"]) for r in run["records"]
               if r["status"] == 400 and (r.get("error") or {}).get("code") == "content_filter"]
    answered = [(r["family"], r["item"]) for r in run["records"]
                if r["status"] == 200 and r["family"] == "tags_vocab"][:controls]
    out = []
    for kind, keys in (("refused", refused), ("control", answered)):
        for fam, item in keys:
            r = by_key[(fam, item)]
            p = {"id": f"{kind}:{fam}:{item}", "system": r["messages"][0]["content"],
                 "user": r["messages"][-1]["content"]}
            if fam == "tags_vocab":
                p["vocabulary"] = cfg["tags"]["vocabulary"]
            out.append(p)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "refused":
        # python tools/chat_bench/tasks.py refused <run file> [controls] > probes.json
        run = json.loads(Path(sys.argv[2]).read_text())
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        print(json.dumps(refused_probes(run, n), ensure_ascii=False, indent=1))
    else:
        print("usage: tasks.py refused <run file> [controls]", file=sys.stderr)
        sys.exit(2)
