#!/usr/bin/env python3
"""Write a chat-template fixture: what transformers' `apply_chat_template`
renders and tokenizes for a fixed set of conversations, with the checkpoint's
template and tokenizer.

    tools/make_chat_template_fixture.py <model id> <checkpoint dir> \
        --repo <org/name> --revision <sha> [--kwargs '{"enable_thinking": false}']

writes `fixtures/chat-templates/<model id>.json`. `crates/sidekick-llm/tests/
chat_templates.rs` renders every case with sidekick's renderer (minijinja,
docs/design/open-llm.md, "Tokenizer and chat template") and compares the text
always, and the token ids when the checkpoint's tokenizer is in the Hugging
Face cache. A template the renderer can't reproduce fails there.

The fixture holds the template's text, so the comparison of rendered text
needs nothing else. The checkpoint directory is read for the template and the
tokenizer only (no weights): a Hugging Face snapshot, or a sidekick model
directory with `tokenizer.json`, `tokenizer_config.json` and the template.

Pinned (the versions the committed fixtures were made with):
    transformers==4.57.6 tokenizers==0.22.2 jinja2==3.1.6
"""

import argparse
import hashlib
import json
import pathlib
import sys

PINNED = {"transformers": "4.57.6", "tokenizers": "0.22.2", "jinja2": "3.1.6"}

# The conversations. Content is plain text: special-token text in messages
# is sidekick's to neutralize, and its own tests cover it.
CASES = [
    {"id": "one-user-turn", "messages": [{"role": "user", "content": "Title this session in five words or fewer."}]},
    {
        "id": "system-and-user",
        "messages": [
            {"role": "system", "content": "You write short titles. Reply with the title only."},
            {"role": "user", "content": "We fixed the queue's timeout and added a test for it."},
        ],
    },
    {
        "id": "multi-turn",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And times 3?"},
        ],
    },
    {
        "id": "assistant-with-reasoning",
        "messages": [
            {"role": "user", "content": "Pick a colour."},
            {"role": "assistant", "content": "<think>\nAny will do.\n</think>\n\nBlue."},
            {"role": "user", "content": "Another?"},
        ],
    },
    {
        "id": "second-system-message",
        "messages": [
            {"role": "system", "content": "Rule one."},
            {"role": "user", "content": "Hi."},
            {"role": "system", "content": "Rule two."},
            {"role": "user", "content": "Hi again."},
        ],
    },
    {
        "id": "unicode-and-whitespace",
        "messages": [
            {
                "role": "user",
                "content": "  Résumé — naïve café 東京 🚀\n\n\ttabs, trailing spaces   \n\nand a final newline\n",
            }
        ],
    },
    {
        "id": "code-and-json",
        "messages": [
            {
                "role": "user",
                "content": 'Extract the fields:\n```json\n{"name": "Ada", "tags": ["a", "b"]}\n```\nfn main() { println!("{}", 1); }',
            }
        ],
    },
    {"id": "empty-user-turn", "messages": [{"role": "user", "content": ""}]},
    {
        "id": "ends-with-assistant",
        "messages": [
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi!"},
        ],
    },
]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the sidekick model id, the fixture's name")
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--kwargs", default="{}", help="template kwargs, as the manifest's [chat] template_kwargs")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import jinja2
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    versions = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "jinja2": jinja2.__version__}
    if versions != PINNED:
        sys.exit(f"pinned {PINNED}, found {versions}")

    kwargs = json.loads(args.kwargs)
    tok = AutoTokenizer.from_pretrained(str(args.checkpoint))
    template = tok.chat_template
    if not isinstance(template, str):
        sys.exit("the checkpoint has no single chat template")
    cases = []
    for case in CASES:
        text = tok.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tok.apply_chat_template(case["messages"], tokenize=True, add_generation_prompt=True, **kwargs)
        if hasattr(ids, "keys"):  # a BatchEncoding in newer releases
            ids = ids["input_ids"]
        # apply_chat_template(tokenize=True) is encode(text, add_special_tokens=False).
        assert ids == tok.encode(text, add_special_tokens=False), case["id"]
        cases.append({**case, "text": text, "ids": ids})

    fixture = {
        "format": 1,
        "model": args.model,
        "source": {"repo": args.repo, "revision": args.revision},
        "versions": versions,
        "tokenizer_sha256": sha256(args.checkpoint / "tokenizer.json"),
        "template": template,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "template_kwargs": kwargs,
        "cases": cases,
    }
    root = pathlib.Path(__file__).resolve().parent.parent
    out = args.out or root / "fixtures" / "chat-templates" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
    print(f"{out.relative_to(root) if out.is_relative_to(root) else out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
