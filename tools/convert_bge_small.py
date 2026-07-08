"""Convert BAAI/bge-small-en-v1.5 into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling baked into the graph, matching examples/manifests/bge-small-en-v1.5.

Usage:
    python tools/convert_bge_small.py <hf-model-dir> <install-dir> [buckets...]

    hf-model-dir: local snapshot of BAAI/bge-small-en-v1.5
                  (hf download BAAI/bge-small-en-v1.5 --local-dir <dir>
                   --include config.json tokenizer.json model.safetensors)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/bge-small-en-v1.5"
    buckets:      default 128 256 512

Requires: torch, transformers, coremltools, numpy (arm64-native Python),
plus Xcode for `xcrun coremlcompiler`.

The recipe encodes four hardware-verified constraints (macOS 26, M-series);
deviate from any of them and the encoder silently falls off the ANE or
produces garbage:

1. STATIC shapes, one artifact per bucket. A single artifact with
   ct.EnumeratedShapes fails ANE plan compilation at load time (E5RT
   "tensor_buffer has known strides while the model has FlexibleShapeInfo")
   and the entire encoder runs on CPU — measured 86ms vs 2.5ms at seq 128.
2. Pooling INSIDE the model (here: CLS + reshape to a literal (1, dims)).
   A raw last_hidden_state output keeps a symbolic seq dim that the
   ANE/CPU (Espresso) path rejects ("Data-dependent shapes were disabled").
   The final .reshape(1, dims) matters: h[:, 0, :] alone leaves a symbolic
   batch dim behind.
3. SDPA attention (torch default), NOT attn_implementation="eager": the
   eager mask path materializes -inf constants that overflow fp16 on the
   ANE and NaN the entire output.
4. Explicit position_ids buffer: without it, coremltools 9.x fails to
   convert the traced graph under static input shapes ("'int' op ...
   only 0-dimensional arrays can be converted to Python scalars").
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

DIMS = 384


class ClsWrapper(torch.nn.Module):
    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.register_buffer("position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0))

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=self.position_ids,
        ).last_hidden_state
        return hidden[:, 0, :].reshape(1, DIMS)


def convert_bucket(model, seq_len, workdir):
    wrapper = ClsWrapper(model, seq_len)
    wrapper.eval()
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0], ids[0, 1] = 101, 102  # [CLS] [SEP]
    mask = torch.zeros((1, seq_len), dtype=torch.int32)
    mask[0, :2] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def compile_to_mlmodelc(pkg, install_dir, seq_len):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True)
        compiled = next(Path(tmp).glob("*.mlmodelc"))
        dest = install_dir / f"model_{seq_len}.mlmodelc"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
    return dest


def parity_check(model, tokenizer, pkg, seq_len):
    """CLS cosine between the Core ML artifact (ANE-eligible path) and torch fp32."""
    text = "Parity check sentence for the converted encoder."
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        ref = model(**enc).last_hidden_state[0, 0].numpy()
    n = enc["input_ids"].shape[1]
    ids = np.zeros((1, seq_len), dtype=np.int32)
    ids[0, :n] = enc["input_ids"][0].numpy()
    mask = np.zeros((1, seq_len), dtype=np.int32)
    mask[0, :n] = 1
    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
    if np.isnan(out).any():
        raise SystemExit(f"seq {seq_len}: NaN output — see recipe constraint 3")
    cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out)))
    if cos < 0.999:
        raise SystemExit(f"seq {seq_len}: parity cosine {cos:.6f} < 0.999")
    return cos


def main():
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            pkg = convert_bucket(model, seq, workdir)
            cos = parity_check(model, tokenizer, pkg, seq)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            print(f"bucket {seq}: parity cos={cos:.6f} -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = Path(__file__).resolve().parent.parent / "examples/manifests/bge-small-en-v1.5/manifest.toml"
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
