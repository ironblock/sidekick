"""Convert BAAI/bge-small-en-v1.5 into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling baked into the graph, matching examples/manifests/bge-small-en-v1.5.

Usage:
    python tools/convert_bge_small.py <hf-model-dir> <install-dir> [buckets...] [--time]
    python tools/convert_bge_small.py --enumerated-shapes <hf-model-dir> <out.mlmodelc>

    hf-model-dir: local snapshot of BAAI/bge-small-en-v1.5
                  (hf download BAAI/bge-small-en-v1.5 --local-dir <dir>
                   --include config.json tokenizer.json model.safetensors)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/bge-small-en-v1.5"
    buckets:      default 128 256 512

Requires: torch, transformers, tokenizers, coremltools, numpy (arm64-native
Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the BERT backbone
with a CLS pooling head. bge-small was the first model on the stack, and its
recipe fixed the rules every later one follows (docs/DECISIONS.md D15):
static shapes, one artifact per bucket; pooling inside the model, ending in
a literal (1, dims) reshape; explicit position_ids.

ATTENTION IS THE FUSED OP, for now. This checkpoint was converted with
transformers' sdpa path, which coremltools lowers to Core ML's fused
attention op. On the ANE that op ignores a mask computed outside its own
procedure (D25), and bge-small's is built on the CPU; its artifact is
correct only because Core ML runs this graph through a fallback, which the
iOS26 opset no longer has. The converter keeps the fused op so the artifact
stays byte-identical to the one graded in docs/MODELS.md. Moving to explicit
attention (the BERT backbone's default) is a separate, measured change. The
gates catch the failure mode either way: pad invariance, and the compute
plan's check for fused attention reading an outside mask.

`--enumerated-shapes` deliberately violates the static-shape rule: it writes
ONE artifact with enumerated sequence lengths 128/256/512. It is a negative
control for `ane_check`, which must reject it (its compute plan puts every
operation on the CPU). Never install it: on macOS 27, predicting with it
under .cpuOnly aborts the process ("E5RT: No memory object bound to port"),
and sidekick refuses to load it there (docs/DECISIONS.md D27).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import bert
from sidekick_convert.heads.pool import Pool

MODEL_ID = "bge-small-en-v1.5"

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # ~400 tokens: exercises long-sequence fp16 accumulation in the buckets
    # it fits (512). Short sentences alone under-test the larger buckets.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]


class FlexibleClsWrapper(torch.nn.Module):
    """CLS pooling without the fixed position_ids buffer, for the
    enumerated-shapes negative control (sequence length varies)."""

    def __init__(self, model, dims):
        super().__init__()
        self.model = model
        self.dims = dims

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        ).last_hidden_state
        return hidden[:, 0, :].reshape(1, self.dims)


def convert_enumerated_negative_control(model, dims, out_path):
    """One flexible-shape artifact: the configuration D15 forbids."""
    ids = torch.zeros((1, 128), dtype=torch.int32)
    ids[0, 0], ids[0, 1] = 101, 102  # [CLS] [SEP]
    mask = torch.zeros((1, 128), dtype=torch.int32)
    mask[0, :2] = 1
    with torch.no_grad():
        traced = torch.jit.trace(FlexibleClsWrapper(model, dims).eval(), (ids, mask))
    shapes = ct.EnumeratedShapes(shapes=[(1, 128), (1, 256), (1, 512)], default=(1, 128))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=shapes, dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=shapes, dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "enumerated.mlpackage"
        mlmodel.save(str(pkg))
        compiled = core.compile_mlmodelc(pkg, Path(tmp) / "compiled")
        shutil.rmtree(out_path, ignore_errors=True)
        shutil.move(str(compiled), out_path)
    print(f"negative control (do not install) -> {out_path}")


def main():
    if sys.argv[1:2] == ["--enumerated-shapes"]:
        src, out = Path(sys.argv[2]).expanduser(), Path(sys.argv[3]).expanduser()
        with tempfile.TemporaryDirectory() as tmp:
            tok = tokenizer.load(tokenizer.prepare(src, Path(tmp) / "tokenizer.json", mode="verbatim"))
            backbone = bert.load(src, tok, attention="fused", token_types="none")
        convert_enumerated_negative_control(backbone.model, backbone.hidden_size, out)
        return

    args = cli.parse(__doc__.split("\n\n")[0])
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    backbone = bert.load(args.src, tok, attention="fused", token_types="none")
    name, value, factor = backbone.check_linear_range(PARITY_SENTENCES, tok)
    print(f"largest linear output {value:.1f} at {name}, {factor:.1f}x under the ANE linear's 2^15")
    job = recipes.embedder(model_id=MODEL_ID, src=args.src, buckets=args.buckets, backbone=backbone,
                           head=Pool("cls"), tok=tok, texts=PARITY_SENTENCES,
                           forbid_ops=frozenset(), timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
