"""Generate the reference vectors the parity suite grades sidekick against.

For one installed model, this materializes the shared corpus
(fixtures/parity/corpus.toml) with the model's tokenizer, prefixes and
buckets, then embeds every case with the model as published:

- **torch**: sentence-transformers in fp32 on the CPU, one text at a time
  (no padding), with the model's own pooling, dense layers and prompts. This
  is the ground truth.
- **onnx** (optional, repeatable): a published ONNX export run with ONNX
  Runtime on the CPU, fed the same token ids. An independent implementation,
  and the vectors that users of ONNX-based embedding libraries already have
  in their indexes. Quantized exports show how much deviation the ecosystem
  already accepts.

It writes reference.json (materialized texts, purposes, tags, the token ids
the reference pipeline used, metadata) and reference.safetensors (one
[cases, dims] float32 tensor per oracle) to <model-dir>/parity, or to
<out>/<model id> with --out. Then run the suite:

    cargo run --release -p sidekick-embed --features coreml --example parity -- \\
        [--refs <out>]

Usage:
    python tools/parity_reference.py <model-dir> --source <hf-id-or-dir> \\
        [--onnx <hf-id-or-dir>:<file.onnx>] ... [--trust-remote-code] [--out DIR]

    <model-dir>   an installed sidekick model directory (manifest.toml,
                  tokenizer.json, artifacts)
    --source      the Hugging Face checkpoint the model was converted from
    --onnx        a published ONNX export, e.g.
                  Alibaba-NLP/gte-modernbert-base:onnx/model_int8.onnx
    --out         a references directory shared by several models (pass
                  the same directory to the suite as --refs)

Requires: torch, sentence-transformers, safetensors, numpy; onnxruntime and
huggingface_hub for --onnx (arm64-native Python).
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "fixtures" / "parity" / "corpus.toml"
FORMAT = 1


def corpus_hash(text):
    """sha256 of the corpus without its full-line comments and blank lines,
    so editing a comment doesn't invalidate every reference. Must match
    corpus_sha256() in the parity suite."""
    # Split exactly as Rust's str::lines() does: on "\n", dropping one "\r".
    lines = [l.removesuffix("\r") for l in text.split("\n")]
    lines = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def source_identity(source):
    """The checkpoint as a Hugging Face id and revision. A local directory is
    recorded only by the id and revision in its hub-cache path, never by the
    path itself."""
    path = Path(source)
    if not path.exists():
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(snapshot_download(source, local_files_only=True))
            return {"id": source, "revision": snapshot.name}
        except Exception:  # noqa: BLE001 - offline or not cached: id only
            return {"id": source, "revision": None}
    parts = path.resolve().parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i >= 1 and parts[i - 1].startswith("models--"):
            return {"id": parts[i - 1].removeprefix("models--").replace("--", "/"),
                    "revision": parts[i + 1] if i + 1 < len(parts) else None}
    return {"id": "local checkpoint", "revision": None}


def target_length(spec, buckets):
    if isinstance(spec, int):
        return spec
    return buckets[spec["bucket"]] + spec.get("offset", 0)


def materialize(filler_words, prefix, target, count):
    """Filler words whose tokenized length, with prefix and special tokens,
    is exactly `target`. Words are taken in order; when the next one would
    overshoot, a one-token word goes in its place and it's retried."""
    words, n = [], count(prefix)
    if n > target:
        sys.exit(f"the prefix alone is {n} tokens, more than the {target}-token target")
    i = 0
    while n < target:
        for word in [filler_words[i % len(filler_words)], "a", "the", "and", "of", "to"]:
            trial = count(prefix + " ".join(words + [word]))
            if trial <= target:
                words.append(word)
                n = trial
                if word is filler_words[i % len(filler_words)]:
                    i += 1
                break
        else:
            sys.exit(f"can't reach exactly {target} tokens (stuck at {n})")
    return " ".join(words)


def st_pooling(model):
    """The pooling mode and whether the model has Dense layers after it."""
    modules = list(model)
    pooling = next(m for m in modules if type(m).__name__ == "Pooling")
    mode = pooling.pooling_mode
    if isinstance(mode, (list, tuple)):
        mode = "+".join(mode)
    has_dense = any(type(m).__name__ == "Dense" for m in modules)
    if not pooling.get_config_dict().get("include_prompt", True):
        # sidekick pools over every real token, prompt included.
        sys.exit("the model excludes prompt tokens from pooling, which sidekick doesn't")
    return mode, has_dense


def pool(hidden, mask, mode):
    if mode == "cls":
        return hidden[0]
    if mode == "mean":
        m = mask[:, None].astype(np.float32)
        return (hidden * m).sum(0) / m.sum()
    if mode == "lasttoken":
        return hidden[int(mask.sum()) - 1]
    sys.exit(f"unsupported pooling mode for ONNX: {mode}")


def run_onnx(spec, cases, mode, has_dense):
    import onnxruntime as ort
    from huggingface_hub import snapshot_download

    source, _, file = spec.rpartition(":")
    if not source or not file.endswith(".onnx"):
        sys.exit(f"--onnx wants <hf-id-or-dir>:<file.onnx>, got {spec!r}")
    root = Path(source) if Path(source).is_dir() else Path(
        snapshot_download(source, allow_patterns=[file, file + "_data", file + ".data"])
    )
    try:
        session = ort.InferenceSession(str(root / file), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - ORT raises its own Fail type
        # Some published exports trip ORT's graph fusions (gte-modernbert's
        # fp16 export on ORT 1.27); they run with optimizations off.
        print(f"{file}: {str(e).splitlines()[0][:160]}; retrying without graph optimizations")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(root / file), opts, providers=["CPUExecutionProvider"])
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    unknown = set(inputs) - {"input_ids", "attention_mask", "token_type_ids"}
    if unknown:
        sys.exit(f"{file}: unsupported ONNX inputs {sorted(unknown)}")
    if "sentence_embedding" in outputs:
        out_name, pooled = "sentence_embedding", True
    elif has_dense:
        sys.exit(f"{file} has no sentence_embedding output, and the model's Dense layers aren't in it")
    else:
        out_name, pooled = ("last_hidden_state" if "last_hidden_state" in outputs else outputs[0]), False

    vectors = []
    for case in cases:
        ids = np.array([case["ids"]], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in inputs}
        out = session.run([out_name], feed)[0][0]
        v = out if pooled else pool(out, feed["attention_mask"][0], mode)
        vectors.append(v / np.linalg.norm(v))
    return np.stack(vectors).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--source", required=True)
    ap.add_argument("--onnx", action="append", default=[])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import torch
    from safetensors.numpy import save_file
    from sentence_transformers import SentenceTransformer

    manifest = tomllib.loads((args.model_dir / "manifest.toml").read_text())
    corpus_bytes = CORPUS.read_bytes()
    corpus = tomllib.loads(corpus_bytes.decode())
    prefixes = manifest.get("prefixes", {})
    buckets = manifest["buckets"]

    model = SentenceTransformer(
        args.source, device="cpu", trust_remote_code=args.trust_remote_code,
        model_kwargs={"dtype": torch.float32},
    )
    model.max_seq_length = manifest["max_seq_len"]
    mode, has_dense = st_pooling(model)
    print(f"{manifest['id']}: pooling {mode}{' + dense' if has_dense else ''}, "
          f"max_seq_len {manifest['max_seq_len']}, buckets {buckets}")

    # The reference uses the prompts the model publishes, so a manifest
    # prefix that disagrees with them fails the suite's token-id check.
    # Where the model publishes none for a purpose (bge-small's query
    # instruction lives only in its model card), the manifest's prefix is
    # used and nothing checks it.
    prompts, prompt_source = {}, {}
    published = model.prompts or {}
    for purpose in ("query", "document"):
        if published.get(purpose):
            prompts[purpose] = published[purpose]
            prompt_source[purpose] = "published"
            if prompts[purpose] != prefixes.get(purpose, ""):
                print(f"WARNING: manifest {purpose} prefix {prefixes.get(purpose, '')!r} "
                      f"!= published prompt {prompts[purpose]!r}; the suite will fail its ids check")
        else:
            prompts[purpose] = prefixes.get(purpose, "")
            prompt_source[purpose] = "manifest"
            if prompts[purpose]:
                print(f"the model publishes no {purpose} prompt; using the manifest's {prompts[purpose]!r}")

    tokenize = model.tokenizer
    def count(text):
        return len(tokenize(text, add_special_tokens=True)["input_ids"])

    filler = corpus["filler"]["text"].split()
    cases, vectors = [], []
    with torch.no_grad():
        for case in corpus["case"]:
            purpose = case.get("purpose", "document")
            prefix = prompts[purpose]
            if "length" in case:
                target = target_length(case["length"], buckets)
                text = materialize(filler, prefix, target, count)
            else:
                text = case["text"]
            # Exactly what SentenceTransformer.encode feeds the model:
            # prompt + text, tokenized and truncated to max_seq_length.
            ids = model.tokenize([prefix + text])["input_ids"][0].tolist()
            v = model.encode([text], prompt=prefix or None, batch_size=1,
                             convert_to_numpy=True, normalize_embeddings=True)[0]
            if not np.all(np.isfinite(v)):
                sys.exit(f"{case['id']}: the reference itself is not finite")
            cases.append({"id": case["id"], "purpose": purpose, "tags": case.get("tags", []),
                          "text": text, "ids": ids})
            vectors.append(v.astype(np.float32))
            print(f"  {case['id']:<28} {len(ids):>4} tokens")

    dims = vectors[0].shape[0]
    if dims != manifest["dims"]:
        sys.exit(f"reference dims {dims} != manifest dims {manifest['dims']}")
    tensors = {"torch": np.stack(vectors)}

    for spec in args.onnx:
        label = "onnx:" + spec.rpartition(":")[2].removeprefix("onnx/").removesuffix(".onnx")
        tensors[label] = run_onnx(spec, cases, mode, has_dense)
        cos = (tensors[label] * tensors["torch"]).sum(1)
        print(f"{label}: worst cosine vs torch {cos.min():.6f}")

    def version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    out_dir = args.out / manifest["id"] if args.out else args.model_dir / "parity"
    tokenizer_sha = hashlib.sha256((args.model_dir / "tokenizer.json").read_bytes()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT,
        "corpus_sha256": corpus_hash(corpus_bytes.decode()),
        "model": {
            "id": manifest["id"],
            "dims": dims,
            "buckets": buckets,
            "max_seq_len": manifest["max_seq_len"],
            "prefixes": {"query": prefixes.get("query", ""), "document": prefixes.get("document", "")},
        },
        "source": source_identity(args.source),
        "tokenizer_sha256": tokenizer_sha,
        "prompt_source": prompt_source,
        "pooling": mode,
        "oracles": list(tensors),
        "versions": {p: version(p) for p in
                     ("torch", "transformers", "sentence-transformers", "onnxruntime")},
        "cases": cases,
    }
    (out_dir / "reference.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    save_file(tensors, str(out_dir / "reference.safetensors"))
    print(f"wrote {len(cases)} cases x {dims} dims, oracles {list(tensors)}, to {out_dir}")


if __name__ == "__main__":
    main()
