"""Generate the reference a reranker is graded against (docs/design/rerank.md).

For one installed reranker (a directory with a `text-ranking`
classifier.toml and tokenizer.json), this pairs every query and document of
the committed corpus (fixtures/rerank/corpus.toml), runs the checkpoint as
published in fp32 on the CPU, one unpadded pair at a time, and writes
<out>/<model id>/reference.json + reference.safetensors
(schema: fixtures/classify/reference.schema.json). Each case is one pair:
its `query`, the document as `input`, its `group`, the token and segment
ids, and two oracles, each the raw logit (float32 [cases, 1], before any
activation: the suite applies the manifest's): "torch", the checkpoint as
published in fp32, and "fp16", the same model as an ideal fp16 engine would
run it (sidekick_convert.fp16sim; docs/CONVERTING.md), the ceiling a
converted model is graded against. A model that overflows fp16 as
published has no ceiling: its fp16 oracle is left out, with a warning. Not
committed: it derives from model weights.

Two traps it avoids:
- Pairs are encoded with the install dir's tokenizer.json, the file sidekick
  reads, not with transformers' AutoTokenizer, which can disagree with it
  (transformers 5.x drops jina-reranker's lowercasing). A case whose ids
  differ from AutoTokenizer's is reported, so a disagreement is visible.
- The model runs in fp32, whatever dtype the checkpoint or
  sentence-transformers would pick.

It also checks the manifest's `problem_type` against the activation vLLM
derives from the checkpoint (`get_act_fn`: the config's problem_type, else
sentence-transformers' activation_fn or sbert_ce_default_activation_function,
else sigmoid for one output), so `relevance_score` is on vLLM's scale, and
compares a few pairs with sentence-transformers' CrossEncoder.

A group tagged `truncated` is truncated to the model's maximum
(`longest_first`, special tokens kept), as the suite requests with
truncate_prompt_tokens. Any other pair longer than the model is an error in
the corpus.

Usage:
    python tools/rerank_reference.py <model-dir> --source <checkpoint-dir> [--out DIR]

    <model-dir>   an installed reranker (classifier.toml, tokenizer.json)
    --source      the checkpoint it was converted from (a Hugging Face
                  snapshot directory)
    --out         references directory, default <model-dir>/parity

Requires: torch, transformers, tokenizers, safetensors, numpy;
sentence-transformers for the CrossEncoder comparison (arm64-native Python).
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
import parity_reference as pr  # noqa: E402  (corpus hash and source identity, D26)
from sidekick_convert import fp16sim  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "fixtures" / "rerank" / "corpus.toml"
FORMAT = 1
TRUNCATED = "truncated"


def vllm_problem_type(config):
    """The problem_type matching the activation vLLM applies (get_act_fn in
    vllm/model_executor/layers/pooler/activations.py)."""
    explicit = getattr(config, "problem_type", None)
    if explicit:
        return {"regression": "regression", "single_label_classification": "single_label",
                "multi_label_classification": "multi_label"}[explicit]
    st = getattr(config, "sentence_transformers", None) or {}
    fn = st.get("activation_fn") or getattr(config, "sbert_ce_default_activation_function", None)
    if fn:
        if fn.endswith("Identity"):
            return "regression"
        if fn.endswith("Sigmoid"):
            return "single_label"
        sys.exit(f"activation {fn} has no problem_type equivalent")
    return "single_label"  # one output: sigmoid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    manifest = tomllib.loads((args.model_dir / "classifier.toml").read_text())
    if manifest.get("task") != "text-ranking":
        sys.exit("not a text-ranking model")
    max_len = manifest["max_seq_len"]
    io = manifest.get("classify", {}).get("io", {})
    segments = "token_type_ids" in io

    config = AutoConfig.from_pretrained(args.source)
    derived = vllm_problem_type(config)
    declared = manifest.get("problem_type", "single_label")
    if declared != derived:
        sys.exit(f"manifest problem_type {declared!r}, but vLLM would apply {derived!r} for this checkpoint")
    if config.num_labels != 1:
        sys.exit(f"a reranker has one output; this checkpoint has {config.num_labels}")

    tok_path = args.model_dir / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    tok.no_padding()
    hf_tok = AutoTokenizer.from_pretrained(args.source)
    model = AutoModelForSequenceClassification.from_pretrained(args.source, dtype=torch.float32).eval()
    uses_types = "token_type_ids" in hf_tok.model_input_names

    corpus_text = CORPUS.read_text()
    corpus = tomllib.loads(corpus_text)
    cases, logits, fp16_logits, disagreements = [], [], [], []
    for group in corpus["group"]:
        truncate = TRUNCATED in group.get("tags", [])
        if truncate:
            tok.enable_truncation(max_length=max_len, strategy="longest_first")
        else:
            tok.no_truncation()
        for i, doc in enumerate(group["documents"]):
            doc = " ".join([doc] * group.get("repeat", 1))
            enc = tok.encode(group["query"], doc)
            ids, types = enc.ids, enc.type_ids
            if len(ids) > max_len:
                sys.exit(f"{group['id']}-{i}: {len(ids)} tokens, more than {max_len}, and not tagged {TRUNCATED}")
            # The batch form, as CrossEncoder calls it: a single call with
            # text_pair="" drops the empty document (Python truthiness)
            # and encodes the query alone, which CrossEncoder never does.
            hf = hf_tok([group["query"]], [doc], truncation="longest_first" if truncate else False,
                        max_length=max_len if truncate else None)["input_ids"][0]
            if list(hf) != ids:
                disagreements.append(f"{group['id']}-{i}")
            feed = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            if uses_types:
                feed["token_type_ids"] = torch.tensor([types])
            with torch.no_grad():
                logit = model(**feed).logits[0, 0].item()
                fp16_logit = fp16sim.run(model, **feed).logits[0, 0].item()
            case = {"id": f"{group['id']}-{i}", "tags": group.get("tags", []), "group": group["id"],
                    "query": group["query"], "input": doc, "ids": ids, "k": 1, "qtype": None, "gold": None}
            if segments:
                case["type_ids"] = types
            cases.append(case)
            logits.append(logit)
            fp16_logits.append(fp16_logit)
            print(f"  {case['id']:<20} {len(ids):>4} tokens  logit {logit:+.4f}  fp16 {fp16_logit:+.4f}")

    if disagreements:
        print(f"WARNING: AutoTokenizer's ids differ from tokenizer.json's on {len(disagreements)} pairs "
              f"({', '.join(disagreements[:5])}...): the reference follows tokenizer.json")
    if uses_types and not segments:
        sys.exit("the model takes token_type_ids, but [classify.io] doesn't name them")

    # sentence-transformers' CrossEncoder, on the untruncated pairs: its
    # predict() must equal our logits under the declared activation.
    try:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(args.source, device="cpu", model_kwargs={"dtype": torch.float32})
        sample = [c for c in cases if TRUNCATED not in c["tags"]][:8]
        got = ce.predict([(c["query"], c["input"]) for c in sample], batch_size=1)
        want = np.array([logits[cases.index(c)] for c in sample])
        if derived != "regression":
            want = 1 / (1 + np.exp(-want))
        worst = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - want)))
        print(f"CrossEncoder.predict vs this reference under {derived}: max |Δ| {worst:.2e}")
        if worst > 1e-4:
            sys.exit("CrossEncoder disagrees with this reference; check the activation and tokenizer")
    except ImportError:
        print("sentence-transformers not installed: CrossEncoder comparison skipped")

    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    tensors = {"torch": np.array(logits, dtype=np.float32)[:, None],
               "fp16": np.array(fp16_logits, dtype=np.float32)[:, None]}
    lost = np.isfinite(tensors["torch"]) & ~np.isfinite(tensors["fp16"])
    if lost.any():
        bad = [cases[i]["id"] for i in np.nonzero(lost[:, 0])[0]]
        print(f"WARNING: ideal fp16 is not finite on {len(bad)} pairs ({bad[:5]}): "
              "the model overflows fp16 as published; no fp16 oracle")
        del tensors["fp16"]
    else:
        delta = np.abs(tensors["fp16"] - tensors["torch"])
        print(f"ideal fp16 vs torch, raw logits: max |Δ| {delta.max():.2e}, mean {delta.mean():.2e}")

    source = pr.source_identity(args.source)
    out_dir = args.out / manifest["id"] if args.out else args.model_dir / "parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "format": FORMAT,
        "corpus_sha256": pr.corpus_hash(corpus_text),
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
        "model": {"id": manifest["id"], "task": "text-ranking", "format": None,
                  "buckets": manifest["buckets"], "max_seq_len": max_len, "max_labels": 1,
                  "labels": manifest["classify"]["labels"]},
        "source": {"repo": source["id"], "revision": source["revision"]},
        "oracles": list(tensors),
        "versions": {p: version(p) for p in ("torch", "transformers", "tokenizers", "sentence-transformers")},
        "cases": cases,
    }
    (out_dir / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n")
    from safetensors.numpy import save_file

    save_file(tensors, str(out_dir / "reference.safetensors"))
    print(f"wrote {len(cases)} pairs in {len(corpus['group'])} groups, oracles {list(tensors)}, to {out_dir}")


if __name__ == "__main__":
    main()
