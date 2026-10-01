"""Generate the references sidekick's classifiers are graded against, and the
committed token-id fixtures the Rust input builders must reproduce.

For one installed classifier (a directory with classifier.toml and
tokenizer.json) this builds its corpus, tokenizes every case with the model's
own Python input builder, runs the model as published in fp32 on the CPU, one
unpadded input at a time, and writes (docs/design/classify.md):

- <out>/<model id>/reference.json + reference.safetensors: per-case ids,
  markers, qtype, labels, gold labels and two oracles, each one float32
  [cases, max_labels] tensor of logits, NaN beyond each case's label count:
  "torch", the fp32 logits, and "fp16", the same model as an ideal fp16
  engine would run it (sidekick_convert.fp16sim; docs/CONVERTING.md), the
  ceiling the suite grades each path against. With D26's staleness keys
  (corpus and tokenizer hashes, the checkpoint's repo and revision). Schema: fixtures/classify/reference.schema.json. Not
  committed: it derives from model weights.
- with --tokens-fixture PATH, the token-id fixture: a small subset of cases
  with their ids and markers. Schema: fixtures/classify/tokens.schema.json.

Corpora:
- laya format: fixtures/classify/<id>.corpus.toml (laya-en's translates
  fastino/fast-decisions mechanically and adds adversarial cases;
  laya-typed-decisions' is the same with one longer case). Its [source]
  revision must be in the local Hugging Face cache (or pass --dataset).
  Inputs are built with the checkpoint's own build_sequence: laya's pinned
  rl_common.py, or for laya-typed-decisions the laya package's common.py
  (--laya-code), checked as tools/convert_laya.py checks it.
- text-classification: the embedding parity corpus
  (fixtures/parity/corpus.toml), materialized with the model's
  tokenizer.json. Inputs longer than the largest bucket are truncated to it
  (first tokens kept, special tokens preserved) and tagged "truncated": the
  server takes them only with truncate_prompt_tokens.

Usage:
    python tools/classifier_reference.py <model-dir> --source <checkpoint-dir> \\
        [--dataset DIR] [--out DIR] [--tokens-fixture PATH]

    <model-dir>       an installed classifier (classifier.toml, tokenizer.json)
    --source          the checkpoint it was converted from (laya: the snapshot
                      directory tools/convert_laya.py read)
    --out             references directory, default <model-dir>/parity

Requires: torch, transformers, tokenizers, safetensors, numpy (arm64-native
Python).
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_reference as pr  # noqa: E402  (corpus hash and materialization, D26)
from sidekick_convert import fp16sim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FORMAT = 1
QTYPES = {"choice": 0, "score": 1, "noul": 2}
# text-classification token fixture: short parity-corpus cases chosen for
# tokenizer coverage (scripts, special-token strings, newlines, code, URLs,
# empty input), none truncated
TEXT_FIXTURE_IDS = {
    "doc-cat", "query-cat", "tiny-empty", "tiny-emoji", "tiny-cjk", "tiny-number",
    "delim-punctuation", "delim-bert-specials", "delim-decoder-specials", "delim-newlines",
    "degenerate-subword", "code-python", "code-json", "numbers-url",
    "ml-german", "ml-japanese", "ml-arabic", "ml-mixed",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


# --- laya ------------------------------------------------------------------------


def laya_question(qtype, labels, instructions):
    """The request's candidate labels as laya's question dict. render_options
    then reproduces each label exactly: choice labels are keys with no
    description, score labels are level descriptions, noul labels are
    'false'/'true' with an optional ': description'."""
    if qtype == "choice":
        return {"t": "choice", "ins": instructions, "crit": {label: "" for label in labels}}
    if qtype == "score":
        return {"t": "score", "ins": instructions, "crit": list(labels)}
    crit = {}
    for want, label in zip(("false", "true"), labels):
        if label == want:
            crit[want] = None
        elif label.startswith(want + ": "):
            crit[want] = label[len(want) + 2:]
        else:
            raise SystemExit(f"noul labels must be false then true, got {labels}")
    return {"t": "noul", "ins": instructions, "crit": crit}


def laya_translate(corpus, where, name, labels, gold):
    """One dataset head in laya's terms: (question type, candidate labels, gold)."""
    if labels == ["yes", "no"]:
        return "noul", ["false", "true"], ["true" if g == "yes" else "false" for g in gold]
    if name in corpus["score"]:
        levels = corpus["score"][name]
        if sorted(levels) != sorted(labels):
            raise SystemExit(f"{where}.{name}: labels {labels} != score levels {levels}")
        return "score", levels, gold
    return "choice", labels, gold


def laya_long_cases(corpus, dataset_dir):
    """[[long]] corpus entries: consecutive rows of one dataset file joined
    into one long input, asked that file's head (translated as for single
    rows). `target_len` asks laya_build to cut the input, at whitespace, to
    the longest prefix whose sequence fits in that many tokens; without it
    the input is as long as its rows make it, truncated past max_len by
    laya's own layout and tagged "at-max-len".
    `candidate_labels_generate` and `instructions` override the head's.
    There is no gold label: the rows' answers differ."""
    cases = []
    for spec in corpus.get("long", []):
        lines = (Path(dataset_dir) / f"{spec['domain']}.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in lines[spec["start"]: spec["start"] + spec["rows"]]]
        head = next(h for h in rows[0]["output"]["classifications"] if h["task"] == spec["head"])
        qtype, cand, _ = laya_translate(corpus, spec["domain"], head["task"], head["labels"], [])
        if "candidate_labels_generate" in spec:
            g = spec["candidate_labels_generate"]
            cand = [g["template"].format(i=i) for i in range(g["count"])]
        cases.append({"id": spec["id"], "tags": ["long", spec["domain"], spec["head"], qtype] + spec.get("tags", []),
                      "input": "\n\n".join(r["input"] for r in rows), "candidate_labels": cand,
                      "question_type": qtype,
                      "instructions": spec.get("instructions", corpus["instructions"][spec["head"]]),
                      "gold": None, "head": f"long:{spec['domain']}.{spec['head']}",
                      "target_len": spec.get("target_len")})
    return cases


def laya_cases(corpus, dataset_dir):
    cases = []
    for path in sorted(Path(dataset_dir).glob("*.jsonl")):
        domain = path.stem
        for row, line in enumerate(path.read_text().splitlines()):
            record = json.loads(line)
            for head in record["output"]["classifications"]:
                if head["multi_label"]:
                    continue
                name = head["task"]
                qtype, cand, gold = laya_translate(corpus, domain, name, head["labels"], head["true_label"])
                cases.append({"id": f"{domain}.{name}.{row:03d}", "tags": [domain, name, qtype],
                              "input": record["input"], "candidate_labels": cand,
                              "question_type": qtype, "instructions": corpus["instructions"][name],
                              "gold": gold, "head": f"{domain}.{name}"})
    for adv in corpus["adversarial"]:
        labels = adv.get("candidate_labels")
        if labels is None:
            g = adv["candidate_labels_generate"]
            labels = [g["template"].format(i=i) for i in range(g["count"])]
        text = adv.get("input")
        if text is None:
            g = adv["input_generate"]
            text = " ".join(g["template"].format(i=i, j=i * 7 % 13) for i in range(g["repeat"]))
        cases.append({"id": adv["id"], "tags": ["adversarial"] + adv.get("tags", []), "input": text,
                      "candidate_labels": labels, "question_type": adv["question_type"],
                      "instructions": adv.get("instructions"), "gold": adv.get("gold"),
                      "head": None})
    return cases + laya_long_cases(corpus, dataset_dir)


def option_token_ids(tok, options, head_max_len):
    """Each option's tokens as laya's build_sequence lays them out ([MASK], the
    option cut to 48 tokens, the even shrink when the budget runs out), for
    the contract's token-level duplicate check."""
    opt_ids = [[tok.mask_token_id] + tok(" " + o.replace(tok.mask_token, " "),
                                         add_special_tokens=False)["input_ids"][:48] for o in options]
    if head_max_len - sum(len(o) for o in opt_ids) < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
    return opt_ids


def laya_build(cases, rl, tok, cfg, defaults, max_labels):
    """build_sequence per case, after the validation the server applies: 2 to
    max_labels labels, no duplicates, none identical at the token level after
    laya's option shrinking."""
    for c in cases:
        labels = c["candidate_labels"]
        k = len(labels)
        if not 2 <= k <= max_labels or len(set(labels)) != k:
            raise SystemExit(f"{c['id']}: {k} labels, or duplicates: the server would reject it")
        ins = c["instructions"] if c["instructions"] is not None else defaults[c["question_type"]]
        q = laya_question(c["question_type"], labels, ins)
        options = rl.render_options(q)
        if c["question_type"] == "choice" and options != list(labels):
            raise SystemExit(f"{c['id']}: rendered options differ from the labels")
        if len({tuple(o) for o in option_token_ids(tok, options, cfg["head_max_len"])}) != k:
            raise SystemExit(f"{c['id']}: labels identical at the token level after shrinking")
        ids, markers = rl.build_sequence(tok, c["input"], q, cfg["max_len"], cfg["head_max_len"])
        if c.get("target_len") and len(ids) > c["target_len"]:
            # the longest whitespace-cut prefix of the input that fits target_len
            text = c["input"]
            cuts = [i for i, ch in enumerate(text) if ch.isspace()]
            lo, hi = 0, len(cuts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                n = len(rl.build_sequence(tok, text[: cuts[mid]], q, cfg["max_len"], cfg["head_max_len"])[0])
                lo, hi = (mid, hi) if n <= c["target_len"] else (lo, mid - 1)
            c["input"] = text[: cuts[lo]]
            ids, markers = rl.build_sequence(tok, c["input"], q, cfg["max_len"], cfg["head_max_len"])
            if not c["target_len"] - 16 <= len(ids) <= c["target_len"]:
                raise SystemExit(f"{c['id']}: cut to {len(ids)} tokens, not within 16 of {c['target_len']}")
        if len(markers) != k:
            raise SystemExit(f"{c['id']}: {len(markers)} markers for {k} labels")
        # laya truncates its own text (the suite's "truncated" tag would ask
        # for truncate_prompt_tokens, which the laya format refuses)
        if "long" in c["tags"] and len(ids) == cfg["max_len"] and "at-max-len" not in c["tags"]:
            c["tags"] = c["tags"] + ["at-max-len"]
        c.update(ids=ids, markers=markers, qtype=QTYPES[c["question_type"]], k=k)
    return cases


# how each oracle runs a model: as published in fp32, or as an ideal fp16 engine
RUNS = {"torch": lambda model, *args, **kwargs: model(*args, **kwargs), "fp16": fp16sim.run}


def oracles(forward, cases, width, log_every=100):
    """Each oracle's logits, [cases, width] and NaN-padded. forward(run, case)
    returns one case's logits, calling the model through run(model, ...). A
    model that overflows fp16 as published has no ceiling: its fp16 oracle is
    left out, and the run says why."""
    out = {}
    for name, run in RUNS.items():
        logits = np.full((len(cases), width), np.nan, dtype=np.float32)
        with torch.no_grad():
            for i, c in enumerate(cases):
                logits[i, : c["k"]] = forward(run, c)
                if log_every and i % log_every == 0:
                    print(f"  {name} {i}/{len(cases)}", flush=True)
        out[name] = logits
    lost = np.isfinite(out["torch"]) & ~np.isfinite(out["fp16"])
    if lost.any():
        bad = [cases[i]["id"] for i in sorted(set(np.nonzero(lost)[0]))]
        print(f"WARNING: ideal fp16 is not finite on {len(bad)} cases ({bad[:5]}): "
              "the model overflows fp16 as published; no fp16 oracle", flush=True)
        del out["fp16"]
    return out


def laya_logits(dm, cases, max_labels):
    def forward(run, c):
        ids = torch.tensor([c["ids"]])
        args = (ids, torch.ones_like(ids), torch.tensor([c["markers"]]),
                torch.ones((1, c["k"]), dtype=torch.bool), torch.tensor([c["qtype"]]))
        logits, _ = run(dm, *args)
        return logits[0].numpy()
    return oracles(forward, cases, max_labels)


def laya_fixture_subset(cases):
    """Every adversarial case, which between them take every branch of
    build_sequence, plus the shortest dataset row of each question type, so
    real dataset text (newlines, quotes, emoji) is pinned too. Kept small."""
    chosen, shortest = [c for c in cases if c["head"] is None], {}
    for c in cases:
        t = c["question_type"]
        if c["head"] is not None and (t not in shortest or len(c["ids"]) < len(shortest[t]["ids"])):
            shortest[t] = c
    return chosen + [shortest[t] for t in ("choice", "score", "noul") if t in shortest]


def run_laya(args, manifest):
    import convert_laya as cl
    from transformers import AutoTokenizer
    src = args.source
    rl = cl.load_rl_common(src, manifest["id"], args.laya_code)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    if sha256(src / "tokenizer" / "tokenizer.json") != sha256(args.model_dir / "tokenizer.json"):
        raise SystemExit("the installed tokenizer.json differs from the checkpoint's")
    tok = AutoTokenizer.from_pretrained(src / "tokenizer")
    corpus_text = (REPO / "fixtures" / "classify" / f"{manifest['id']}.corpus.toml").read_text()
    corpus = tomllib.loads(corpus_text)
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import snapshot_download
        dataset = Path(snapshot_download(corpus["source"]["repo"], repo_type="dataset",
                                         revision=corpus["source"]["revision"], local_files_only=True))
    cls = manifest["classify"]
    cases = laya_build(laya_cases(corpus, dataset), rl, tok, cfg,
                       cls["laya"]["default_instructions"], cls["max_labels"])
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens", flush=True)
    fixture = laya_fixture_subset(cases)
    logits = None
    if not args.fixture_only:
        from sidekick_convert.backbones import modernbert
        modernbert.install_patches()  # laya's own forward, as its converter runs it
        dm = cl.load_decision_model(src, rl, cfg)
        logits = laya_logits(dm, cases, cls["max_labels"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- text-classification -----------------------------------------------------------


def run_text(args, manifest):
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tok.no_padding()
    count = lambda text: len(tok.encode(text, add_special_tokens=True).ids)  # noqa: E731
    corpus_text = pr.CORPUS.read_text()
    corpus = tomllib.loads(corpus_text)
    filler = corpus["filler"]["text"].split()
    buckets, max_len = manifest["buckets"], manifest["max_seq_len"]
    cases = []
    for case in corpus["case"]:
        text = case.get("text")
        if text is None:
            text = pr.materialize(filler, "", pr.target_length(case["length"], buckets), count)
        tags = list(case.get("tags", []))
        enc = tok.encode(text, add_special_tokens=True)
        ids = enc.ids
        if len(ids) > max_len:
            tok.enable_truncation(max_len)
            ids = tok.encode(text, add_special_tokens=True).ids
            tok.no_truncation()
            tags.append("truncated")
        cases.append({"id": case["id"], "tags": tags, "input": text, "ids": ids,
                      "k": len(manifest["classify"]["labels"]), "gold": None})
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens", flush=True)
    fixture = [c for c in cases if c["id"] in TEXT_FIXTURE_IDS]
    logits = None
    if not args.fixture_only:
        model = AutoModelForSequenceClassification.from_pretrained(args.source, dtype=torch.float32).eval()

        def forward(run, c):
            ids = torch.tensor([c["ids"]])
            feed = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            return run(model, **feed).logits[0].numpy()
        logits = oracles(forward, cases, cases[0]["k"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- output --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tokens-fixture", type=Path)
    ap.add_argument("--fixture-only", action="store_true", help="write the token fixture and stop")
    ap.add_argument("--laya-code", type=Path,
                    help="laya-typed-decisions: the laya package's common.py (tools/convert_laya.py)")
    args = ap.parse_args()
    manifest = tomllib.loads((args.model_dir / "classifier.toml").read_text())
    laya = manifest["classify"].get("format") == "laya"
    cases, fixture, logits, corpus_sha = (run_laya if laya else run_text)(args, manifest)
    tokenizer_sha = sha256(args.model_dir / "tokenizer.json")
    source = {"repo": manifest["source"]["repo"], "revision": manifest["source"].get("revision")}

    if args.tokens_fixture:
        fx = {"format": FORMAT, "model": manifest["id"], "source": source,
              "tokenizer_sha256": tokenizer_sha, "max_len": manifest["max_seq_len"], "cases": []}
        for c in fixture:
            rec = {"id": c["id"], "input": c["input"]}
            if laya:
                rec.update(candidate_labels=c["candidate_labels"], question_type=c["question_type"],
                           instructions=c["instructions"])
            rec["ids"] = c["ids"]
            if laya:
                rec.update(markers=c["markers"], qtype=c["qtype"])
            fx["cases"].append(rec)
        args.tokens_fixture.write_text(json.dumps(fx, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"token fixture: {len(fixture)} cases -> {args.tokens_fixture}")
    if args.fixture_only:
        return

    from safetensors.numpy import save_file
    cls = manifest["classify"]
    max_labels = cls.get("max_labels", len(cls.get("labels", [])))
    tensors = {}
    for name, values in logits.items():
        tensors[name] = np.full((len(cases), max_labels), np.nan, dtype=np.float32)
        tensors[name][:, : values.shape[1]] = values
    meta = {
        "format": FORMAT,
        "corpus_sha256": corpus_sha,
        "tokenizer_sha256": tokenizer_sha,
        "model": {"id": manifest["id"], "task": manifest["task"], "format": cls.get("format"),
                  "buckets": manifest["buckets"], "max_seq_len": manifest["max_seq_len"],
                  "max_labels": max_labels, "labels": cls.get("labels", [])},
        "source": source,
        "oracles": list(tensors),
        "versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "numpy")},
        "cases": [],
    }
    for c in cases:
        rec = {"id": c["id"], "tags": c["tags"], "input": c["input"]}
        if laya:
            rec.update(candidate_labels=c["candidate_labels"], question_type=c["question_type"],
                       instructions=c["instructions"])
        rec.update(ids=c["ids"], k=c["k"], gold=c["gold"])
        if laya:
            rec.update(markers=c["markers"], qtype=c["qtype"])
        meta["cases"].append(rec)
    out = (args.out or args.model_dir / "parity") / manifest["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False))
    save_file(tensors, str(out / "reference.safetensors"))
    print(f"wrote {len(cases)} cases x {max_labels} labels, oracles {list(tensors)}, to {out}")


if __name__ == "__main__":
    main()
