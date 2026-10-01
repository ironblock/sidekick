"""Write Julia-1's token-id fixture from its own input builder.

fixtures/classify/julia-1.tokens.json pins sidekick's laya-format input
builder, with `option_rendering = "julia"`, to Julia-1's Python
(`julia/data.py` `sequence()`, non-strict). Each case is a sidekick request
(`input`, `candidate_labels`, `question_type`, `instructions`), plus the ids,
marker positions and qtype Julia-1's own code produces for it. The Rust test
in crates/sidekick-embed/tests/classify_tokens.rs must reproduce them
exactly. Schema: fixtures/classify/tokens.schema.json.

A sidekick label becomes a Julia-1 option the way Julia-1's typed API
(`julia/typed.py`) builds options from a question's criteria:
- choice: the label's description (the text after its first ": "), or the
  label itself when it has none;
- score: the label as given;
- noul: labels `false` then `true`, each optionally `"false: <text>"`:
  "false"/"true" when neither is described, the two descriptions when both
  are.
An empty description counts as none. Julia-1's strict mode raises on
overflow and reserved markers; sidekick truncates instead, as the
non-strict builder used here does.

Usage:
    python tools/julia_tokens.py <julia-1-snapshot> [--out PATH]

    julia-1-snapshot: local snapshot of SupersonicLabs/Julia-1 at the
                      manifest's revision (julia/data.py, tokenizer/)

Requires: transformers, tokenizers (arm64-native Python). Imports
julia/data.py from the snapshot, after checking its sha256.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "examples" / "classifiers" / "julia-1" / "classifier.toml"
OUT = REPO / "fixtures" / "classify" / "julia-1.tokens.json"
DATA_PY_SHA256 = "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e"

LONG = " ".join(f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail." for i in range(160))
TICKET = ("Hi, I was charged twice for order 4471 last week and the second charge still hasn't been "
          "refunded. Can someone look into it? Thanks, Dana")
LONG_DESC = ("handles every question, complaint, follow-up, escalation, refund, exchange, clarification "
             "and document request that customers send about billing, invoices, duplicate charges, "
             "payment methods, failed card payments, chargebacks and subscription renewals")

CASES = [
    dict(id="label-only-choice", input=TICKET, question_type="choice",
         instructions="Which team should handle this request?",
         candidate_labels=["billing", "shipping", "access"]),
    dict(id="choice-descriptions", input=TICKET, question_type="choice",
         instructions="Which team should handle this request?",
         candidate_labels=["billing: Billing and payment disputes", "shipping: Shipping and delivery",
                           "access: Account access: login and passwords"]),
    dict(id="choice-empty-description-and-bare-colon", input=TICKET, question_type="choice",
         instructions="Which label fits?",
         candidate_labels=["billing: ", "h:i", "x: a: b"]),
    dict(id="score-items", input="The product broke after one day and support never answered.",
         question_type="score", instructions="How satisfied is the customer?",
         candidate_labels=["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"]),
    dict(id="noul-undescribed", input=TICKET, question_type="noul",
         instructions="Does the customer ask for a refund?", candidate_labels=["false", "true"]),
    dict(id="noul-described", input=TICKET, question_type="noul",
         instructions="Does the customer ask for a refund?",
         candidate_labels=["false:the customer wants something else", " true: the customer wants money back "]),
    dict(id="option-cut-at-48", input=TICKET, question_type="choice",
         instructions="Which team should handle this request?",
         candidate_labels=["billing: " + LONG_DESC + ", and also " + LONG_DESC, "shipping: Shipping and delivery", "access: Account access"]),
    dict(id="k20-shrink", input=TICKET, question_type="choice",
         instructions="Which queue should this ticket go to?",
         candidate_labels=[f"queue_{i}: queue {i} handles " + LONG_DESC for i in range(20)]),
    dict(id="literal-mask", input="The customer wrote <mask> where the order number should be. <mask>",
         question_type="choice", instructions="Is the <mask> a typo or a placeholder?",
         candidate_labels=["typo: a mistyped <mask> character", "placeholder: a template <mask> left in"]),
    dict(id="long-state-truncated", input=LONG, question_type="choice",
         instructions="Which topic is discussed most?",
         candidate_labels=["topic 3: the third topic", "topic 7: the seventh topic", "other: something else"]),
]


def noul_description(label, key):
    """sidekick's noul label parsing: None if not this key, else the description or ''."""
    rest = label.strip()
    if not rest.startswith(key):
        return None
    rest = rest[len(key):]
    if not rest:
        return ""
    if not rest.startswith(":"):
        return None
    return rest[1:].strip()


def julia_options(question_type, labels):
    """sidekick's labels as Julia-1 option texts (see the module docstring)."""
    if question_type == "choice":
        out = []
        for label in labels:
            key, sep, desc = label.partition(": ")
            out.append(desc if sep and desc.strip() else key if sep else label)
        return out
    if question_type == "score":
        return list(labels)
    f, t = noul_description(labels[0], "false"), noul_description(labels[1], "true")
    if f is None or t is None or bool(f) != bool(t):
        raise SystemExit(f"noul labels {labels}: describe both or neither")
    return ["false", "true"] if not f else [f, t]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    man = tomllib.loads(MANIFEST.read_text())
    if man["classify"]["laya"]["option_rendering"] != "julia":
        raise SystemExit(f"{MANIFEST}: option_rendering is not julia")

    data_py = args.snapshot / "julia" / "data.py"
    digest = hashlib.sha256(data_py.read_bytes()).hexdigest()
    if digest != DATA_PY_SHA256:
        raise SystemExit(f"{data_py} is not Julia-1's data.py at {man['source']['revision'][:7]} (sha256 {digest})")
    spec = importlib.util.spec_from_file_location("julia_data", data_py)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)

    from transformers import PreTrainedTokenizerFast
    tok_json = args.snapshot / "tokenizer" / "tokenizer.json"
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_json))
    tok_config = json.loads((args.snapshot / "tokenizer" / "tokenizer_config.json").read_text())
    for name in ("cls_token", "sep_token", "mask_token", "pad_token"):
        setattr(tok, name, tok_config[name])

    max_len, head = man["max_seq_len"], man["classify"]["laya"]["head_max_len"]
    fx = {"format": 1, "model": man["id"], "source": man["source"],
          "tokenizer_sha256": hashlib.sha256(tok_json.read_bytes()).hexdigest(),
          "max_len": max_len, "cases": []}
    for case in CASES:
        labels = case["candidate_labels"]
        if not 2 <= len(labels) <= man["classify"]["max_labels"]:
            raise SystemExit(f"{case['id']}: {len(labels)} labels")
        row = {"state": case["input"], "question": case["instructions"], "type": case["question_type"],
               "options": julia_options(case["question_type"], labels)}
        data.validate_row(row, 1)
        enc = data.sequence(tok, row, max_len, head)
        if len(enc["markers"]) != len(labels) or len(enc["ids"]) > max_len:
            raise SystemExit(f"{case['id']}: {len(enc['markers'])} markers for {len(labels)} labels")
        fx["cases"].append(dict(case, ids=enc["ids"], markers=enc["markers"], qtype=enc["qtype"]))
        print(f"{case['id']}: {len(enc['ids'])} tokens, markers {enc['markers'][:4]}{'...' if len(labels) > 4 else ''}"
              + (", state truncated" if enc["truncated"] else ""))
    args.out.write_text(json.dumps(fx, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(fx['cases'])} cases -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
