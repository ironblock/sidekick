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
- gliner2 format: fixtures/classify/gliner2.5-decide.corpus.toml: every
  fast-decisions head as one single-task request, plus adversarial cases.
  Inputs are laid out by the gliner2 package's own processor (2.0.0); an
  over-length text is cut to its longest prefix of whole words that fits and
  then laid out as is (gliner2 appends the terminal "."), sidekick's
  truncation rule. --source is the GLiNER2 checkpoint.
- laya format: fixtures/classify/<id>.corpus.toml (laya-en's translates
  fastino/fast-decisions mechanically and adds adversarial cases;
  laya-typed-decisions' is the same with one longer case). Its [source]
  revision must be in the local Hugging Face cache (or pass --dataset).
  Inputs are built with the checkpoint's own build_sequence: laya's pinned
  rl_common.py, or for laya-typed-decisions the laya package's common.py
  (--laya-code), checked as tools/convert_laya.py checks it.
- laya format with `option_rendering = "julia"` (Julia-1):
  fixtures/classify/<id>.corpus.toml, the same dataset translation with
  Julia-1's own adversarial cases; heads over max_labels are left out.
  Inputs are built with Julia-1's julia/data.py and options rendered as its
  typed API renders them; the oracles run its JuliaDecisionModel, with RoPE
  read from the encoder config's `rope_parameters` (which transformers 4.x
  ignores). --source is the Julia-1 snapshot; its julia/ code is checked by
  sha256.
- fev format (Lumma-fev): fixtures/classify/<id>.corpus.toml, laya's dataset
  translation with fev's own adversarial and long cases. Inputs are laid out
  by the checkpoint's own modeling_fev.py (pack, encode) and the oracles run
  its FevForDecision, both sha256-pinned and run on transformers 4.x with
  the shims load_fev() documents. --source is the Lumma-fev snapshot.
- agentjev format (AgentJev): fixtures/classify/<id>.corpus.toml, laya's
  dataset translation with AgentJev's own adversarial and long cases. Inputs
  come from AgentJev's own jev_service/contract.py (prepare, encode_paths),
  laid out as the format's tree, and the oracles score each path as its
  service does, through its own head modules (agentjev/model.py), both
  sha256-pinned; the backbone is transformers' Qwen3Model. --source is the
  agent-jev snapshot, --agentjev-code a checkout of its code.
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
GLINER2_CORPUS = REPO / "fixtures" / "classify" / "gliner2.5-decide.corpus.toml"
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
                try:
                    logits[i, : c["k"]] = forward(run, c)
                except fp16sim.Fp16Overflow as e:
                    print(f"WARNING: {c['id']}: {e}; no fp16 oracle", flush=True)
                    break
                if log_every and i % log_every == 0:
                    print(f"  {name} {i}/{len(cases)}", flush=True)
            else:
                out[name] = logits
    if "fp16" not in out:
        return out
    lost = np.isfinite(out["torch"]) & ~np.isfinite(out["fp16"])
    if lost.any():
        bad = [cases[i]["id"] for i in sorted(set(np.nonzero(lost)[0]))]
        print(f"WARNING: ideal fp16 is not finite on {len(bad)} cases ({bad[:5]}): "
              "the model overflows fp16 as published; no fp16 oracle", flush=True)
        del out["fp16"]
    return out


def limited(args, cases):
    """The first `--limit` cases, for a smoke run: a reference of fewer
    cases than the corpus isn't one the suite can grade against."""
    if args.limit is None:
        return cases
    print(f"SMOKE RUN: {args.limit} of {len(cases)} cases; not a usable reference", flush=True)
    return cases[: args.limit]


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
        cases = limited(args, cases)
        logits = laya_logits(dm, cases, cls["max_labels"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- laya format, Julia-1's option rendering ----------------------------------------

# Julia-1's own code at the manifest's revision: its input builder and its model
JULIA_SHA256 = {"data.py": "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
                "model.py": "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81"}


def noul_description(label, key):
    """sidekick's noul label parsing (crates/sidekick-embed/src/laya.rs):
    None if `label` isn't `key`, else its description, '' for none."""
    rest = label.strip()
    if not rest.startswith(key):
        return None
    rest = rest[len(key):]
    if not rest:
        return ""
    return rest[1:].strip() if rest.startswith(":") else None


def julia_options(question_type, labels):
    """The request's labels as Julia-1's option texts, the way its typed API
    (julia/typed.py) builds options from criteria: a choice option is the
    label's description (after its first ": "), or the label itself when it
    has none; a score option is the label; a noul question's options are
    "false"/"true", or both descriptions. An empty description counts as
    none, and a half-described noul question is rejected."""
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
        raise SystemExit(f"noul labels {labels}: false then true, both described or neither")
    return [f, t] if f else ["false", "true"]


def load_julia_code(src):
    """julia/data.py and julia/model.py from the snapshot, after checking
    they're the pinned files."""
    import importlib.util
    mods = {}
    for name, want in JULIA_SHA256.items():
        path = src / "julia" / name
        if sha256(path) != want:
            raise SystemExit(f"{path} is not Julia-1's {name} at the pinned revision (sha256 {sha256(path)})")
        spec = importlib.util.spec_from_file_location(f"julia_{path.stem}", path)
        mods[path.stem] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[path.stem])
    return mods["data"], mods["model"]


def julia_tokenizer(src):
    """Julia-1's tokenizer with the special tokens its tokenizer_config names
    (CLS <bos>, SEP <eos>, MASK <mask>): the ones julia/data.py reads."""
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=str(src / "tokenizer" / "tokenizer.json"))
    config = json.loads((src / "tokenizer" / "tokenizer_config.json").read_text())
    for name in ("cls_token", "sep_token", "mask_token", "pad_token"):
        setattr(tok, name, config[name])
    return tok


def julia_model(src, model_py):
    """JuliaDecisionModel in fp32 with Julia-1's weights. The encoder is built
    from encoder/config.json with RoPE taken from its `rope_parameters`
    block: transformers 4.x ignores that block (it is how transformers 5
    saves ModernBERT) and would run mmBERT's sliding layers at theta 10000
    instead of 160000, a different model (measured: |dp| up to 0.97 on
    laya's gate questions)."""
    from safetensors.torch import load_file
    from transformers import ModernBertConfig, ModernBertModel
    raw = json.loads((src / "encoder" / "config.json").read_text())
    ecfg = ModernBertConfig(**raw)
    rope = raw.get("rope_parameters") or {}
    for layer_type, attr in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
        if layer_type in rope:
            setattr(ecfg, attr, float(rope[layer_type]["rope_theta"]))
    ecfg._attn_implementation = "sdpa"
    jcfg = json.loads((src / "julia_config.json").read_text())
    model = model_py.JuliaDecisionModel(ModernBertModel(ecfg), head_layers=jcfg["head_layers"], n_act=jcfg["n_act"])
    model.load_state_dict(load_file(str(src / "model.safetensors")), strict=True)
    return model.float().eval()


def julia_build(cases, data, tok, manifest):
    """julia/data.py's sequence() per case (non-strict: it truncates, as
    sidekick does), after the validation the server applies: instructions
    present (Julia-1 has no defaults), 2 to max_labels labels, no duplicate
    labels or rendered options, none identical at the token level after
    shrinking. Cases over max_labels are left out, as the corpus says."""
    cls = manifest["classify"]
    max_len, head, max_labels = manifest["max_seq_len"], cls["laya"]["head_max_len"], cls["max_labels"]
    kept, over = [], {}
    for c in cases:
        labels = c["candidate_labels"]
        k = len(labels)
        if k > max_labels:
            over[c["head"] or c["id"]] = over.get(c["head"] or c["id"], 0) + 1
            continue
        if c["instructions"] is None:
            raise SystemExit(f"{c['id']}: no instructions, and Julia-1 has no defaults")
        options = julia_options(c["question_type"], labels)
        if k < 2 or len(set(labels)) != k or len(set(options)) != k or not all(o.strip() for o in options):
            raise SystemExit(f"{c['id']}: {k} labels, or duplicate or empty options: the server would reject it")
        if len({tuple(o) for o in option_token_ids(tok, options, head)}) != k:
            raise SystemExit(f"{c['id']}: options identical at the token level after shrinking")
        row = {"state": c["input"], "question": c["instructions"], "type": c["question_type"], "options": options}
        data.validate_row(row, 1)
        enc = data.sequence(tok, row, max_len, head)
        if len(enc["markers"]) != k:
            raise SystemExit(f"{c['id']}: {len(enc['markers'])} markers for {k} labels")
        c.update(ids=enc["ids"], markers=enc["markers"], qtype=enc["qtype"], k=k)
        if enc["truncated"]:
            # Not "truncated": that tag means the request carries
            # truncate_prompt_tokens, which the laya format refuses because
            # it truncates the state by design.
            c["tags"] = c["tags"] + ["truncated-text"]
        kept.append(c)
    if over:
        print(f"left out {sum(over.values())} cases over max_labels {max_labels}: {over}", flush=True)
    return kept


def julia_logits(model, cases, max_labels):
    def forward(run, c):
        ids = torch.tensor([c["ids"]])
        args = (ids, torch.ones_like(ids), torch.tensor([c["markers"]]),
                torch.ones((1, c["k"]), dtype=torch.bool), torch.tensor([c["qtype"]]))
        return run(model, *args)[0].numpy()
    return oracles(forward, cases, max_labels)


def run_julia(args, manifest):
    """A laya-format model with Julia-1's option rendering: the corpus is
    fixtures/classify/<id>.corpus.toml (laya's dataset translation, Julia-1's
    adversarial cases), inputs come from Julia-1's own julia/data.py, and the
    oracles run its JuliaDecisionModel."""
    src = args.source
    data, model_py = load_julia_code(src)
    if sha256(src / "tokenizer" / "tokenizer.json") != sha256(args.model_dir / "tokenizer.json"):
        raise SystemExit("the installed tokenizer.json differs from the checkpoint's")
    tok = julia_tokenizer(src)
    corpus_path = REPO / "fixtures" / "classify" / f"{manifest['id']}.corpus.toml"
    corpus_text = corpus_path.read_text()
    corpus = tomllib.loads(corpus_text)
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import snapshot_download
        dataset = Path(snapshot_download(corpus["source"]["repo"], repo_type="dataset",
                                         revision=corpus["source"]["revision"], local_files_only=True))
    cases = julia_build(laya_cases(corpus, dataset), data, tok, manifest)
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens", flush=True)
    fixture = laya_fixture_subset(cases)
    logits = None
    if not args.fixture_only:
        cases = limited(args, cases)
        logits = julia_logits(julia_model(src, model_py), cases, manifest["classify"]["max_labels"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- fev (Lumma-fev) -----------------------------------------------------------------

FEV_SHA256 = {"modeling_fev.py": "5d89f8861b9af25eb6f369ffea35659c394025ed70b2aaa1cd44407594e0febc",
              "modeling_nandi.py": "5d724ccf35d65f6e0447f3987d08a4c8588560d41785e45ca8bae96677b415d3",
              "configuration_fev.py": "e887919e9254cf6edc203b179e8b6cbc29fa978431f7b3d35cd914922a9d53a8",
              "configuration_nandi.py": "7949a9e6df044f7c9adb8b7bd59672e2e393b4d89b3ada4b496995347c3822b8",
              "fev_backbone.py": "a35dfec6a6e3fbcf3e108a9a9354d8daa0a3f0304e5f2759a26fd4297b79c676"}


def fev_options(question_type, labels):
    """The request's labels as fev option texts (docs/design/classify.md,
    "The fev format"), the way fev's option_text renders criteria: a choice
    option is the label as given, "key" or "key: description"; a score
    option is the label; a noul question's options are "no" and "yes", each
    with ": description" when its label (false, true) has one. An empty
    description counts as none."""
    if question_type == "choice":
        out = []
        for label in labels:
            key, sep, desc = label.partition(": ")
            out.append(label if sep and desc.strip() else key if sep else label)
        return out
    if question_type == "score":
        return list(labels)
    f, t = noul_description(labels[0], "false"), noul_description(labels[1], "true")
    if f is None or t is None:
        raise SystemExit(f"noul labels {labels}: false then true")
    return ["no: " + f if f else "no", "yes: " + t if t else "yes"]


def fev_question(question_type, labels, instructions):
    """The request as the TypeSafe-shaped question fev's pack() reads, built
    so its option_text reproduces fev_options exactly (checked per case)."""
    if question_type == "choice":
        crit = {}
        for label in labels:
            key, sep, desc = label.partition(": ")
            crit[key if sep else label] = desc if sep and desc.strip() else None
        if len(crit) != len(labels):
            raise SystemExit(f"labels {labels}: two share a key, which fev's criteria can't express")
        return {"type": "choice", "instructions": instructions, "criteria": crit}
    if question_type == "score":
        return {"type": "score", "instructions": instructions, "criteria": list(labels)}
    f, t = noul_description(labels[0], "false"), noul_description(labels[1], "true")
    return {"type": "noul", "instructions": instructions, "criteria": {"false": f or None, "true": t or None}}


def load_fev(src):
    """FevForDecision in fp32 from the checkpoint's own modeling_fev.py and
    modeling_nandi.py (sha256-pinned), and its tokenizer's raw backend.

    That code is written for transformers 5; on the pinned transformers 4.57
    it runs with three shims, none of which touches the computation:
    auto_docstring, merge_with_config_defaults and output_capturing's
    capture_outputs become pass-through decorators (4.57's auto_docstring
    can't parse PEP 604 annotations; the other two don't exist in 4.57),
    and create_causal_mask, called with transformers 5's arguments, returns
    the same causal-and-key-padding additive mask. The layers, RoPE, the
    layer-sharing loop and the pointer head run as written, with eager
    attention."""
    import importlib.util
    import sys as _sys
    import types
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    import transformers.utils as tu
    import transformers.utils.generic as generic
    for name, want in FEV_SHA256.items():
        if sha256(src / name) != want:
            raise SystemExit(f"{src / name} is not the pinned fev code (sha256 {sha256(src / name)})")
    tu.auto_docstring = lambda obj=None, **kw: obj if obj is not None else (lambda o: o)
    if not hasattr(generic, "merge_with_config_defaults"):
        generic.merge_with_config_defaults = lambda f: f
    if "transformers.utils.output_capturing" not in _sys.modules:
        capture = types.ModuleType("transformers.utils.output_capturing")
        capture.capture_outputs = lambda f: f
        _sys.modules["transformers.utils.output_capturing"] = capture
    pkg = types.ModuleType("fev_checkpoint")
    pkg.__path__ = [str(src)]
    _sys.modules["fev_checkpoint"] = pkg
    mods = {}
    for stem in ("configuration_nandi", "modeling_nandi", "configuration_fev", "fev_backbone", "modeling_fev"):
        spec = importlib.util.spec_from_file_location(f"fev_checkpoint.{stem}", src / f"{stem}.py")
        mods[stem] = importlib.util.module_from_spec(spec)
        _sys.modules[spec.name] = mods[stem]
        spec.loader.exec_module(mods[stem])

    def causal_mask(config=None, inputs_embeds=None, attention_mask=None, **kw):
        b, n = inputs_embeds.shape[:2]
        allowed = torch.tril(torch.ones(n, n, dtype=torch.bool))[None]
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, :]
        return torch.zeros(b, 1, n, n, dtype=inputs_embeds.dtype).masked_fill(
            ~allowed[:, None], torch.finfo(inputs_embeds.dtype).min)
    mods["modeling_nandi"].create_causal_mask = causal_mask
    fev = mods["modeling_fev"]
    raw = json.loads((src / "config.json").read_text())
    cfg = fev.FevConfig(**{k: v for k, v in raw.items()
                           if k not in ("architectures", "auto_map", "model_type", "dtype", "transformers_version")})
    model = fev.FevForDecision(cfg)
    model.lm.config._attn_implementation = "eager"
    state = {k: v.float() for k, v in load_file(str(src / "model.safetensors")).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    backend = Tokenizer.from_file(str(src / "tokenizer.json"))
    tok = types.SimpleNamespace(backend_tokenizer=backend, convert_tokens_to_ids=backend.token_to_id, unk_token_id=None)
    return fev, model.float().eval(), tok


def fev_build(cases, fev, model, tok, manifest):
    """fev's own pack() and encode() per case, after the validation the
    server applies: 2 to max_labels labels, no duplicate labels or rendered
    options. A state longer than the state limit is truncated by encode()
    and tagged "truncated-text"; a case whose row then exceeds the window is
    an error here, as it is a 400 for the server. `target_len` cuts the
    input at whitespace to the longest prefix whose row fits in that many
    tokens, as for laya's long cases."""
    cls, fv = manifest["classify"], manifest["classify"]["fev"]
    max_labels = cls["max_labels"]
    if model.limits()[0] != fv["state_max_len"] or model.limits()[1] != manifest["max_seq_len"]:
        raise SystemExit(f"the manifest's limits ({fv['state_max_len']}, {manifest['max_seq_len']}) "
                         f"aren't the checkpoint's {model.limits()}")
    roles = ("state", "question", "option", "option_end", "decide")   # fev's order (configuration_fev.py)
    if [tok.convert_tokens_to_ids(fv["delimiters"][r]) for r in roles] != model.delimiter_ids(tok):
        raise SystemExit("the manifest's delimiters aren't the checkpoint's, role by role")
    if model.config.temperature != 1.0:
        raise SystemExit(f"temperature {model.config.temperature}: the converted graph must fold it in first")

    def encode(text, q):
        S, rows = model.encode(tok, text, fev.pack({"q": q}))
        return S, rows[0]

    def fits(text, q, limit):
        try:
            S, row = encode(text, q)
        except ValueError:   # fev's own over-window error
            return False
        return len(S) + len(row["ids"]) <= limit

    kept, over = [], {}
    for c in cases:
        labels, k = c["candidate_labels"], len(c["candidate_labels"])
        if k > max_labels:
            over[c["head"] or c["id"]] = over.get(c["head"] or c["id"], 0) + 1
            continue
        options = fev_options(c["question_type"], labels)
        if k < 2 or len(set(labels)) != k or len(set(options)) != k:
            raise SystemExit(f"{c['id']}: {k} labels, or duplicate options: the server would reject it")
        q = fev_question(c["question_type"], labels, c["instructions"])
        if fev.pack({"q": q})[0][1] != options:
            raise SystemExit(f"{c['id']}: fev renders {fev.pack({'q': q})[0][1]}, the contract says {options}")
        if c.get("target_len") and not fits(c["input"], q, c["target_len"]):
            text = c["input"]
            cuts = [i for i, ch in enumerate(text) if ch.isspace()]
            lo, hi = 0, len(cuts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if fits(text[: cuts[mid]], q, c["target_len"]) else (lo, mid - 1)
            c["input"] = text[: cuts[lo]]
        S, row = encode(c["input"], q)
        if c.get("target_len"):
            if not c["target_len"] - 16 <= len(S) + len(row["ids"]) <= c["target_len"]:
                raise SystemExit(f"{c['id']}: cut to {len(S) + len(row['ids'])} tokens, not within 16 of {c['target_len']}")
        if len(model.text_ids(tok, fev.render(c["input"]))) + 1 > fv["state_max_len"]:
            c["tags"] = c["tags"] + ["truncated-text"]
        c.update(ids=S + row["ids"], markers=[len(S) + o for o in row["options"]], decide=len(S) + row["decide"],
                 qtype=None, k=k)   # the model takes no question type; only the rendering depends on it
        kept.append(c)
    if over:
        print(f"left out {sum(over.values())} cases over max_labels {max_labels}: {over}", flush=True)
    return kept


class FevLogits(torch.nn.Module):
    """FevForDecision's backbone and pointer head on one unpadded row: the
    logits before fev's softmax. Position ids are left to the backbone
    (0..n-1), so ideal fp16 folds RoPE's tables as constants."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask, markers, decide):
        h = self.model.lm(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[0]
        return self.model.head(h[decide], h[markers])


def fev_logits(model, cases, max_labels):
    wrapped = FevLogits(model).eval()

    def forward(run, c):
        ids = torch.tensor([c["ids"]])
        return run(wrapped, ids, torch.ones_like(ids), torch.tensor(c["markers"]), torch.tensor(c["decide"])).numpy()
    return oracles(forward, cases, max_labels)


def run_fev(args, manifest):
    """A fev-format model (Lumma-fev): the corpus is
    fixtures/classify/<id>.corpus.toml (laya's dataset translation and long
    cases, fev's adversarial cases), and inputs and oracles come from the
    checkpoint's own modeling_fev.py."""
    src = args.source
    fev, model, tok = load_fev(src)
    if sha256(src / "tokenizer.json") != sha256(args.model_dir / "tokenizer.json"):
        raise SystemExit("the installed tokenizer.json differs from the checkpoint's")
    corpus_text = (REPO / "fixtures" / "classify" / f"{manifest['id']}.corpus.toml").read_text()
    corpus = tomllib.loads(corpus_text)
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import snapshot_download
        dataset = Path(snapshot_download(corpus["source"]["repo"], repo_type="dataset",
                                         revision=corpus["source"]["revision"], local_files_only=True))
    cases = fev_build(laya_cases(corpus, dataset), fev, model, tok, manifest)
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens", flush=True)
    fixture = laya_fixture_subset(cases)
    logits = None
    if not args.fixture_only:
        cases = limited(args, cases)
        logits = fev_logits(model, cases, manifest["classify"]["max_labels"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- agentjev (AgentJev) -------------------------------------------------------------

# AgentJev's code (github.com/malevrigns/agent-jev at a965ca8, Apache-2.0): the
# input contract, and the candidate head's modules.
AGENTJEV_SHA256 = {"jev_service/contract.py": "5a28b220a830e816f7b4fe2bf46b314f7aa7d3cffa4d096037bddf7d566353e6",
                   "agentjev/model.py": "91c28187ebbb15c75382d832bb9b75b58e333575d169fd5350ff049aab019566"}


def agentjev_candidates(question_type, labels):
    """The request's labels as AgentJev's candidate texts, in label order
    (docs/design/classify.md, "The agentjev format"): a choice label's
    description, else the label; a score label as given; a noul question's
    descriptions, else FALSE and TRUE."""
    if question_type == "choice":
        out = []
        for label in labels:
            key, sep, desc = label.partition(": ")
            out.append(desc if sep and desc else key if sep else label)
        return out
    if question_type == "score":
        return list(labels)
    f, t = noul_description(labels[0], "false"), noul_description(labels[1], "true")
    if f is None or t is None:
        raise SystemExit(f"noul labels {labels}: false then true")
    return [f or "FALSE", t or "TRUE"]


def agentjev_question(question_type, labels, instructions):
    """The request as the question AgentJev's prepare() reads, and the map
    from its candidate order to label order. A noul question is AgentJev's
    boolean, whose candidates are true then false; a choice's options and a
    score's levels are lists in label order."""
    if question_type == "noul":
        f, t = noul_description(labels[0], "false"), noul_description(labels[1], "true")
        criteria = {k: v for k, v in (("true", t), ("false", f)) if v}
        return {"id": "q", "type": "boolean", "question": instructions, "criteria": criteria}, [1, 0]
    candidates = agentjev_candidates(question_type, labels)
    if question_type == "choice":
        return {"id": "q", "type": "choice", "question": instructions, "options": candidates}, list(range(len(labels)))
    return {"id": "q", "type": "score", "question": instructions, "levels": candidates}, list(range(len(labels)))


def load_agentjev_code(code):
    """AgentJev's contract and model modules from a checkout of its
    repository at the pinned revision (sha256-checked)."""
    for name, want in AGENTJEV_SHA256.items():
        if sha256(code / name) != want:
            raise SystemExit(f"{code / name} is not the pinned AgentJev code (sha256 {sha256(code / name)})")
    sys.path.insert(0, str(code))
    from jev_service import contract
    from agentjev import model
    return contract, model


def agentjev_build(cases, contract, tok, manifest):
    """AgentJev's own prepare() and encode_paths() per case, laid out as the
    tree the format specifies: the prefix, then each candidate's suffix in
    label order, with segments and positions. Every path encode_paths
    returns must be the prefix followed by that suffix. A tree over
    max_seq_len is a 400 for the server, so it is an error here, except that
    `target_len` first cuts the input at whitespace to the longest prefix
    whose tree fits in that many tokens."""
    max_len, max_labels = manifest["max_seq_len"], manifest["classify"]["max_labels"]

    def layout(text, question, order):
        prepared = contract.prepare({"state": text, "questions": [question]})
        paths, _, rows = contract.encode_paths(prepared, tok, max_len)
        state = prepared[0]["state"]
        prefix = tok.encode(state if state.startswith("[STATE]") else "[STATE] " + state, add_special_tokens=False)
        prefix += tok.encode("\n[QUESTION] " + rows[0]["text"], add_special_tokens=False)
        ids, seg, pos, ends = list(prefix), [0] * len(prefix), list(range(len(prefix))), []
        for c, theirs in enumerate(order, 1):
            path = paths[theirs]
            if path[: len(prefix)] != prefix or len(path) == len(prefix):
                raise SystemExit("an AgentJev path isn't its question's prefix and a candidate suffix")
            suffix = path[len(prefix):]
            ids += suffix
            seg += [c] * len(suffix)
            pos += range(len(prefix), len(prefix) + len(suffix))
            ends.append(len(ids) - 1)
        return rows[0], ids, seg, pos, ends

    def size(text, question, order):
        try:
            return len(layout(text, question, order)[1])
        except ValueError:   # AgentJev's own over-limit refusal (one path over max_len)
            return max_len + 1

    kept, over = [], {}
    for c in cases:
        labels, k = c["candidate_labels"], len(c["candidate_labels"])
        if k > max_labels:
            over[c["head"] or c["id"]] = over.get(c["head"] or c["id"], 0) + 1
            continue
        if k < 2 or len(set(labels)) != k:
            raise SystemExit(f"{c['id']}: {k} labels, or duplicates: the server would reject it")
        question, order = agentjev_question(c["question_type"], labels, c["instructions"])
        if c.get("target_len") and size(c["input"], question, order) > c["target_len"]:
            text = c["input"]
            cuts = [i for i, ch in enumerate(text) if ch.isspace()]
            lo, hi = 0, len(cuts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if size(text[: cuts[mid]], question, order) <= c["target_len"] else (lo, mid - 1)
            c["input"] = text[: cuts[lo]]
        row, ids, seg, pos, ends = layout(c["input"], question, order)
        want = agentjev_candidates(c["question_type"], labels)
        if [row["candidates"][i] for i in order] != want:
            raise SystemExit(f"{c['id']}: AgentJev reads {row['candidates']}, the contract says {want}")
        if len(ids) > max_len:
            raise SystemExit(f"{c['id']}: the tree takes {len(ids)} tokens, over {max_len}: the server would "
                             "refuse it, so it can't be in the corpus")
        if c.get("target_len") and not c["target_len"] - 16 <= len(ids) <= c["target_len"]:
            raise SystemExit(f"{c['id']}: cut to {len(ids)} tokens, not within 16 of {c['target_len']}")
        c.update(ids=ids, seg=seg, position_ids=pos, markers=ends, qtype=None, k=k, order=order,
                 question=question)
        kept.append(c)
    if over:
        print(f"left out {sum(over.values())} cases over max_labels {max_labels}: {over}", flush=True)
    return kept


class AgentJevPaths(torch.nn.Module):
    """AgentJev's scoring as its service runs it per path: each candidate's
    path encoded on its own (right-padded, as its PathEncoder batches them),
    the hidden state at the path's last token, then the checkpoint's own
    candidate head (proj_in, CandidateSetEncoder, proj_out, ScalarScorer,
    through AgentJevModel._score). Logits come out in AgentJev's candidate
    order."""

    def __init__(self, backbone, model_module, head_state):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.proj_in = torch.nn.Linear(hidden, 256)
        self.set_encoder = model_module.CandidateSetEncoder(256, 4, 2)
        self.proj_out = torch.nn.Linear(256, hidden)
        self.scorer = model_module.ScalarScorer(hidden, 256)
        missing, unexpected = self.load_state_dict(head_state, strict=False)
        if unexpected or any(not k.startswith("backbone.") for k in missing):
            raise SystemExit(f"the checkpoint's head and AgentJev's modules differ: {unexpected or missing}")
        self._score = lambda v, m: model_module.AgentJevModel._score(self, v, m)

    # Paths encoded per forward: a question's paths are encoded a few at a
    # time, as AgentJev's service batches them, so 32 paths of 2,048 tokens
    # never materialize one [32, heads, 2048, 2048] attention tensor.
    CHUNK = 2

    def forward(self, ids, mask, last):
        vectors = []
        for i in range(0, ids.shape[0], self.CHUNK):
            h = self.backbone(input_ids=ids[i:i + self.CHUNK], attention_mask=mask[i:i + self.CHUNK],
                              use_cache=False).last_hidden_state
            vectors.append(h[torch.arange(h.shape[0]), last[i:i + self.CHUNK]])
        vectors = torch.cat(vectors)[None]
        return self._score(vectors, torch.ones(vectors.shape[:2], dtype=torch.bool))[0]


def load_agentjev(src, model_module):
    """The checkpoint in fp32: transformers' Qwen3Model for the backbone and
    AgentJev's own head modules, each loaded strictly from model.safetensors."""
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3Model
    weights = load_file(str(src / "model.safetensors"))
    cfg = Qwen3Config.from_pretrained(src)
    cfg._attn_implementation = "eager"
    # Built without allocating weights, then given the checkpoint's own
    # tensors, so the model never exists twice (~2.4 GB each in fp32). RoPE's
    # inv_freq isn't in the checkpoint, so the rotary embedding is rebuilt.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    with torch.device("meta"):
        backbone = Qwen3Model(cfg)
    prefix = "path_encoder.backbone."
    backbone.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True,
                             assign=True)
    backbone.rotary_emb = Qwen3RotaryEmbedding(config=cfg)
    head = {k: v for k, v in weights.items() if not k.startswith(prefix)}
    return AgentJevPaths(backbone.float().eval(), model_module, head).eval()


def agentjev_logits(model, cases, max_labels):
    def forward(run, c):
        prefix_len = c["seg"].index(1)
        starts = [prefix_len] + [e + 1 for e in c["markers"][:-1]]
        paths = [c["ids"][:prefix_len] + c["ids"][s: e + 1] for s, e in zip(starts, c["markers"])]
        theirs = [None] * len(paths)
        for label, t in enumerate(c["order"]):
            theirs[t] = paths[label]
        width = max(map(len, theirs))
        ids = torch.zeros((len(theirs), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(theirs):
            ids[i, : len(p)] = torch.tensor(p)
            mask[i, : len(p)] = 1
        last = torch.tensor([len(p) - 1 for p in theirs])
        logits = run(model, ids, mask, last).numpy()
        return logits[c["order"]]
    return oracles(forward, cases, max_labels)


def run_agentjev(args, manifest):
    """An agentjev-format model (AgentJev): the corpus is
    fixtures/classify/<id>.corpus.toml (laya's dataset translation, AgentJev's
    adversarial cases, long cases), inputs come from AgentJev's own
    contract.py, and the oracles score each path as its service does, with
    its own head modules. --source is the agent-jev snapshot; --agentjev-code
    a checkout of its code at the pinned revision."""
    if args.agentjev_code is None:
        raise SystemExit("the agentjev format needs --agentjev-code (a checkout of malevrigns/agent-jev)")
    from transformers import AutoTokenizer
    src = args.source
    contract, model_module = load_agentjev_code(args.agentjev_code)
    if sha256(src / "tokenizer.json") != sha256(args.model_dir / "tokenizer.json"):
        raise SystemExit("the installed tokenizer.json differs from the checkpoint's")
    tok = AutoTokenizer.from_pretrained(src)
    corpus_text = (REPO / "fixtures" / "classify" / f"{manifest['id']}.corpus.toml").read_text()
    corpus = tomllib.loads(corpus_text)
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import snapshot_download
        dataset = Path(snapshot_download(corpus["source"]["repo"], repo_type="dataset",
                                         revision=corpus["source"]["revision"], local_files_only=True))
    cases = agentjev_build(laya_cases(corpus, dataset), contract, tok, manifest)
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens", flush=True)
    fixture = laya_fixture_subset(cases)
    logits = None
    if not args.fixture_only:
        model = load_agentjev(src, model_module)
        cases = limited(args, cases)
        logits = agentjev_logits(model, cases, manifest["classify"]["max_labels"])
    return cases, fixture, logits, pr.corpus_hash(corpus_text)


# --- gliner2 ------------------------------------------------------------------------


def split_label(label):
    """`key` or `key: description`, split at the first ": " (the contract's rule)."""
    key, sep, desc = label.partition(": ")
    key, desc = key.strip(), desc.strip()
    return (key, desc) if sep and desc else (key, None)


def gliner2_cases(corpus, dataset_dir):
    cases = []
    for path in sorted(Path(dataset_dir).glob("*.jsonl")):
        domain = path.stem
        for row, line in enumerate(path.read_text().splitlines()):
            record = json.loads(line)
            for head in record["output"]["classifications"]:
                cases.append({"id": f"{domain}.{head['task']}.{row:03d}", "tags": [domain, head["task"]],
                              "input": record["input"], "candidate_labels": list(head["labels"]),
                              "instructions": head["task"], "multi_label": bool(head["multi_label"]),
                              "gold": head["true_label"], "head": f"{domain}.{head['task']}"})
    for adv in corpus["adversarial"]:
        labels = adv.get("candidate_labels")
        if labels is None:
            g = adv["candidate_labels_generate"]
            labels = [g["template"].format(i=i) for i in range(g["count"])]
        text = adv.get("input")
        if text is None:
            g = adv["input_generate"]
            text = " ".join(g["template"].format(i=i, j=i * 7 % 13) for i in range(g["repeat"])) + g.get("suffix", "")
        cases.append({"id": adv["id"], "tags": ["adversarial"] + adv.get("tags", []), "input": text,
                      "candidate_labels": labels, "instructions": adv.get("instructions"),
                      "multi_label": bool(adv.get("multi_label", False)), "gold": adv.get("gold"), "head": None})
    return cases


def gliner2_layout(proc, text, prompt, labels):
    """gliner2's own processor on one single-task request: (ids, [L] positions)."""
    split = [split_label(l) for l in labels]
    task = {"task": prompt, "labels": [k for k, _ in split], "true_label": ["N/A"], "multi_label": False,
            "cls_threshold": 0.5, "class_act": "auto",
            "label_descriptions": {k: d for k, d in split if d is not None}}
    schema = {"json_structures": [], "classifications": [task], "entities": {}, "relations": [],
              "json_descriptions": {}, "entity_descriptions": {}}
    b = proc.collate_fn_inference([(text, schema)], error_policy="raise")
    return b.input_ids[0].tolist(), [int(p) for p in b.schema_special_indices[0][0]][1:]


def gliner2_fit(proc, text, prompt, labels, max_len):
    """sidekick's truncation: the whole text if it fits once gliner2 lays it
    out, else its longest prefix of whole words (cut at the end of the last
    kept word) that fits; gliner2 then appends the terminal "." itself."""
    ids, markers = gliner2_layout(proc, text, prompt, labels)
    if len(ids) <= max_len:
        return ids, markers, False
    if len(text.encode()) > max_len * 16:
        raise SystemExit("a corpus text exceeds the byte cap; keep generated texts under it")
    ends = [e for _, _, e in proc.word_splitter(text, lower=False)]
    # Every word can be kept even when the whole text doesn't fit: a text
    # ending in "!" and a space gets gliner2's "." appended, one token more
    # than its prefix up to the "!", which needs none.
    lo, hi = 0, len(ends)              # largest n in [lo, hi] whose prefix fits
    while lo < hi:
        n = (lo + hi + 1) // 2
        if len(gliner2_layout(proc, text[: ends[n - 1]], prompt, labels)[0]) <= max_len:
            lo = n
        else:
            hi = n - 1
    prefix = text[: ends[lo - 1]] if lo else ""
    ids, markers = gliner2_layout(proc, prefix, prompt, labels)
    if len(ids) > max_len:
        raise SystemExit("the schema doesn't fit the model: the server would reject it")
    return ids, markers, True


def gliner2_build(cases, proc, tok, default, max_len, max_labels):
    """The layout per case, after the validation the server applies."""
    for c in cases:
        labels = c["candidate_labels"]
        k = len(labels)
        keys = [split_label(l)[0] for l in labels]
        least = 1 if c["multi_label"] else 2   # one label is a multi-label yes/no question
        if not least <= k <= max_labels or len(set(labels)) != k or len(set(keys)) != k or "" in keys:
            raise SystemExit(f"{c['id']}: {k} labels, duplicates or an empty name: the server would reject it")
        if len({tuple(tok.encode(key, add_special_tokens=False).ids) for key in keys}) != k:
            raise SystemExit(f"{c['id']}: labels identical at the token level")
        prompt = c["instructions"] if c["instructions"] is not None else default
        ids, markers, truncated = gliner2_fit(proc, c["input"], prompt, labels, max_len)
        if len(markers) != k:
            raise SystemExit(f"{c['id']}: {len(markers)} markers for {k} labels")
        if truncated:
            # Not "truncated": that tag tells readers the request carries
            # truncate_prompt_tokens, which this format refuses because it
            # truncates the text by design.
            c["tags"] = c["tags"] + ["truncated-text"]
        c.update(ids=ids, markers=markers, qtype=None, k=k)
    return cases


def gliner2_boundary_cases(proc, max_len):
    """Token-fixture cases at the truncation boundary, built for the model's
    max_len: a text ending in terminal punctuation and a space, whose layout
    is one token over the limit only because gliner2 appends "." to it.
    Every word still fits (the prefix up to the "!" or "?" needs no "."), so
    nothing is dropped. They depend on the tokenizer and max_len, so they live
    here, not in the corpus, and the reference doesn't include them."""
    labels, prompt = ["refund", "cancel"], "intent"
    cases = []
    for mark in ("!", "?"):
        def text(n):
            return " ".join(["late"] * n) + f" too late{mark} "
        n = 0
        while len(gliner2_layout(proc, text(n), prompt, labels)[0]) <= max_len:
            n += 1
        if len(gliner2_layout(proc, text(n), prompt, labels)[0]) != max_len + 1:
            raise SystemExit("no boundary text: a filler word isn't one token")
        cases.append({"id": f"boundary-terminal{'-question' if mark == '?' else ''}-then-space",
                      "tags": ["boundary"], "input": text(n), "candidate_labels": labels,
                      "instructions": prompt, "multi_label": False, "gold": None, "head": None})
    return cases


def gliner2_fixture_subset(cases):
    """Every adversarial case, plus the shortest single-label and multi-label
    dataset requests, so real dataset text is pinned too."""
    chosen = [c for c in cases if c["head"] is None]
    for multi in (False, True):
        pool = [c for c in cases if c["head"] is not None and c["multi_label"] == multi]
        if pool:
            chosen.append(min(pool, key=lambda c: len(c["ids"])))
    return chosen


class MarkerScores(torch.nn.Module):
    """GLiNER2's classification logits: the encoder's output at each `[L]`
    marker, scored by the classifier."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, markers):
        h = self.model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
        return self.model.classifier(h[markers]).squeeze(-1)


def run_gliner2(args, manifest):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from gliner2.processor import SchemaTransformer
    src = args.source
    if sha256(src / "tokenizer.json") != sha256(args.model_dir / "tokenizer.json"):
        raise SystemExit("the installed tokenizer.json differs from the checkpoint's")
    if version("gliner2") != "2.0.0":
        raise SystemExit(f"the gliner2 format is pinned to gliner2 2.0.0, not {version('gliner2')}")
    proc = SchemaTransformer(tokenizer=AutoTokenizer.from_pretrained(src))
    proc.change_mode(is_training=False)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    corpus_text = GLINER2_CORPUS.read_text()
    corpus = tomllib.loads(corpus_text)
    dataset = args.dataset
    if dataset is None:
        from huggingface_hub import snapshot_download
        dataset = Path(snapshot_download(corpus["source"]["repo"], repo_type="dataset",
                                         revision=corpus["source"]["revision"], local_files_only=True))
    cls = manifest["classify"]
    cases = gliner2_build(gliner2_cases(corpus, dataset), proc, tok, cls["gliner2"]["default_instructions"],
                          manifest["max_seq_len"], cls["max_labels"])
    print(f"{len(cases)} cases; max {max(len(c['ids']) for c in cases)} tokens; "
          f"{sum('truncated-text' in c['tags'] for c in cases)} truncated", flush=True)
    boundary = gliner2_build(gliner2_boundary_cases(proc, manifest["max_seq_len"]), proc, tok,
                             cls["gliner2"]["default_instructions"], manifest["max_seq_len"], cls["max_labels"])
    fixture = gliner2_fixture_subset(cases) + boundary
    logits = None
    if not args.fixture_only:
        from gliner2.classification import Classifier
        model = Classifier.from_pretrained(str(src), device="cpu", dtype=torch.float32).scorer.model.eval()
        # The encoder and the classifier run as one module, so the fp16
        # oracle rounds the classifier too.
        scorer = MarkerScores(model)
        cases = limited(args, cases)
        logits = oracles(lambda run, c: run(scorer, torch.tensor([c["ids"]]), torch.tensor(c["markers"])).numpy(),
                         cases, cls["max_labels"])
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
        cases = limited(args, cases)
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
    ap.add_argument("--agentjev-code", type=Path,
                    help="agentjev: a checkout of malevrigns/agent-jev at the pinned revision")
    ap.add_argument("--limit", type=int, help="smoke run: only the first N cases (not a usable reference)")
    args = ap.parse_args()
    manifest = tomllib.loads((args.model_dir / "classifier.toml").read_text())
    fmt = manifest["classify"].get("format")
    laya, gliner2, fev, agentjev = fmt == "laya", fmt == "gliner2", fmt == "fev", fmt == "agentjev"
    julia = laya and manifest["classify"]["laya"].get("option_rendering") == "julia"
    run = (run_julia if julia else run_laya if laya else run_gliner2 if gliner2 else run_fev if fev
           else run_agentjev if agentjev else run_text)
    zero_shot = laya or fev or agentjev
    cases, fixture, logits, corpus_sha = run(args, manifest)
    tokenizer_sha = sha256(args.model_dir / "tokenizer.json")
    source = {"repo": manifest["source"]["repo"], "revision": manifest["source"].get("revision")}

    if args.tokens_fixture:
        fx = {"format": FORMAT, "model": manifest["id"], "source": source,
              "tokenizer_sha256": tokenizer_sha, "max_len": manifest["max_seq_len"], "cases": []}
        for c in fixture:
            rec = {"id": c["id"], "input": c["input"]}
            if zero_shot:
                rec.update(candidate_labels=c["candidate_labels"], question_type=c["question_type"],
                           instructions=c["instructions"])
            if gliner2:
                rec.update(candidate_labels=c["candidate_labels"], instructions=c["instructions"])
            rec["ids"] = c["ids"]
            if zero_shot:
                rec.update(markers=c["markers"], qtype=c["qtype"])
            if fev:
                rec["decide"] = c["decide"]
            if agentjev:
                rec.update(seg=c["seg"], position_ids=c["position_ids"])
            if gliner2:
                rec.update(markers=c["markers"], qtype=None)
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
        "versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "numpy")
                     + (("gliner2",) if gliner2 else ())},
        "cases": [],
    }
    if laya:
        # How labels became option texts: a reference made under one
        # rendering is stale for a manifest that names another.
        meta["model"]["option_rendering"] = cls["laya"].get("option_rendering", "laya")
    for c in cases:
        rec = {"id": c["id"], "tags": c["tags"], "input": c["input"]}
        if zero_shot:
            rec.update(candidate_labels=c["candidate_labels"], question_type=c["question_type"],
                       instructions=c["instructions"])
        if gliner2:
            rec.update(candidate_labels=c["candidate_labels"], instructions=c["instructions"])
        rec.update(ids=c["ids"], k=c["k"], gold=c["gold"])
        if gliner2:
            rec["multi_label"] = c["multi_label"]
        if zero_shot or gliner2:
            rec.update(markers=c["markers"], qtype=c["qtype"])
        if fev:
            rec["decide"] = c["decide"]
        if agentjev:
            rec.update(seg=c["seg"], position_ids=c["position_ids"])
        meta["cases"].append(rec)
    out = (args.out or args.model_dir / "parity") / manifest["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False))
    save_file(tensors, str(out / "reference.safetensors"))
    print(f"wrote {len(cases)} cases x {max_labels} labels, oracles {list(tensors)}, to {out}")


if __name__ == "__main__":
    main()
