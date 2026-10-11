"""Run an ONNX embedder over a model's parity references, for the parity
suite to grade as its `onnx` path.

Until sidekick's ONNX backend can run in the suite's own workers, this is
that worker, in Python: it loads model.onnx with ONNX Runtime on the CPU,
feeds every reference case's token ids (the reference pipeline's, so
tokenization isn't exercised here; the server's own tokenizer is the one
the Core ML paths already grade), pools as the manifest says, normalizes,
and writes the suite's worker result (one JSON object: model, path
"onnx-cpu", cases, repeat_bitwise, load_ms). The suite grades it as the
model's onnx-cpu path (docs/DECISIONS.md D40), against the same references
and gates as every other path:

    cargo run --release -p sidekick-embed --features coreml --example parity -- \\
        --models-dir <dir holding the onnx model> --refs <refs-dir> \\
        --model <id> --external <result.json>

ONNX Runtime has no buckets, so the suite's two invariance checks become:
- batch invariance (reported as bucket invariance): each case run in a
  right-padded batch with the corpus's longest case, against its unpadded
  run;
- pad invariance: the same batch with random pad ids instead of 0.

Usage:
    python tools/onnx_parity.py <model-dir> --refs <refs-dir> --out <result.json>
        [--model-file model.onnx] [--threads N] [--batch-1]

    model-dir: an ONNX model directory: manifest.toml (backend "onnx"),
               tokenizer.json and the model file
    --model-file: another ONNX file to run under the same manifest, e.g. a
               published int8 export (its outputs and pooling must match)
    --batch-1: for a model served one input at a time, unpadded (an int8
               build, D40): skip the batch and pad checks (n = 0), which
               don't apply to it
"""

import argparse
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sidekick_convert.onnx_export import Runner, cosine  # noqa: E402

PAD_ID_RANGE = (1000, 30000)
REPEAT_CASES = 8


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--refs", type=Path, required=True, help="the references directory (<refs>/<model id>/...)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-file", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--batch-1", action="store_true")
    args = p.parse_args()

    manifest = tomllib.loads((args.model_dir / "manifest.toml").read_text())
    if manifest.get("backend") != "onnx":
        sys.exit(f"{args.model_dir}: manifest backend is {manifest.get('backend')!r}, not onnx")
    model_id = manifest["id"]
    ref = json.loads((args.refs / model_id / "reference.json").read_text())
    tok_sha = hashlib.sha256((args.model_dir / manifest["tokenizer"]).read_bytes()).hexdigest()
    if tok_sha != ref["tokenizer_sha256"]:
        sys.exit(f"{model_id}: tokenizer.json differs from the one the reference was generated with")
    pooling = None if manifest.get("pooling", "none") == "none" else manifest["pooling"]
    path = args.model_dir / (args.model_file or manifest["artifact"])

    t = time.perf_counter()
    runner = Runner(path, pooling, threads=args.threads)
    load_ms = (time.perf_counter() - t) * 1000

    rows = [json.loads(c["ids"]) if isinstance(c["ids"], str) else c["ids"] for c in ref["cases"]]
    longest = max(rows, key=len)
    rng = np.random.default_rng(0)
    pads = rng.integers(*PAD_ID_RANGE, size=len(longest))

    def unit(v):
        v = np.asarray(v, np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    cases, first = [], []
    for case, ids in zip(ref["cases"], rows):
        t = time.perf_counter()
        v = runner.run([ids])[0]
        ms = (time.perf_counter() - t) * 1000
        finite = bool(np.isfinite(v).all())
        if not args.batch_1:
            batch = runner.run([ids, longest])[0]
            padded = runner.run([ids, longest], pad_ids=pads)[0]
        vec = unit(np.where(np.isfinite(v), v, 0.0)).astype(np.float32)
        if len(first) < REPEAT_CASES:
            first.append(vec)
        cases.append({
            "id": case["id"], "ids_match": True, "bucket": len(ids), "vector": vec.tolist(), "finite": finite,
            "model_only": None,
            "bucket_invariance": ({"n": 0, "min": None} if args.batch_1
                                  else {"n": 1, "min": _finite(cosine(batch, v))}),
            "pad_invariance": ({"n": 0, "min": None} if args.batch_1
                               else {"n": 1, "min": _finite(cosine(padded, batch))}),
            "ms": ms,
        })
    again = [unit(runner.run([ids])[0]).astype(np.float32) for ids in rows[:REPEAT_CASES]]
    repeat = all(np.array_equal(a, b) for a, b in zip(first, again))
    path_name = "onnx-cpu"
    result = {"model": model_id, "path": path_name, "cases": cases, "repeat_bitwise": repeat, "load_ms": load_ms}
    args.out.write_text(json.dumps(result))
    worst = min(cosine(c["vector"], t) for c, t in zip(cases, _torch(args.refs / model_id)))
    print(f"{model_id} ({path_name}, {path.name}): {len(cases)} cases; worst 1 - cosine vs torch {1 - worst:.1e}; "
          + ("batch-1 only; " if args.batch_1 else
             f"batch {min(c['bucket_invariance']['min'] or 0 for c in cases):.8f}; "
             f"pads {min(c['pad_invariance']['min'] or 0 for c in cases):.8f}; ")
          + f"repeat bitwise {repeat}; "
          f"median {np.median([c['ms'] for c in cases]):.1f} ms; load {load_ms:.0f} ms -> {args.out}")


def _finite(x):
    return x if np.isfinite(x) else None


def _torch(ref_dir):
    from safetensors.numpy import load_file
    return load_file(str(ref_dir / "reference.safetensors"))["torch"]


if __name__ == "__main__":
    main()
