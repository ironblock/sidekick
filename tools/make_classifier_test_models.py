"""Build the tiny Core ML classifiers that sidekick-embed's tests load
(crates/sidekick-embed/tests/fixtures/).

They check the runtime's Core ML plumbing, not accuracy. The vocabulary is
small and the weights are seeded random.

tiny-laya/: the laya format. `input_ids`/`attention_mask` [1, S],
`marker_pos` [1, K] with -1 in unused slots, `qtype` [1], and `logits`
[1, K] with unused slots at -1e4.
- `model_16.mlmodelc`, `model_32.mlmodelc`: K = 4, one per bucket;
- `k5_32.mlmodelc`: bucket 32 with K = 5, a bucket that disagrees with
  the others (and with `max_labels = 4`);
- `expected.json`: fp32 torch logits for a few fixed inputs, which the
  tests compare with Core ML's.

tiny-reranker/: a cross-encoder. `input_ids`/`attention_mask`/
`token_type_ids` [1, S] and `logits` [1, 1].
- `model_16.mlmodelc`, `model_32.mlmodelc`;
- `expected.json`: fp32 torch scores for fixed (ids, type_ids) pairs.

Usage:
    python tools/make_classifier_test_models.py crates/sidekick-embed/tests/fixtures

Requires torch, coremltools and numpy (arm64-native Python 3.12 or
earlier: coremltools' native writer doesn't load on 3.14), plus Xcode for
`xcrun coremlcompiler`.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

VOCAB = 32
D = 8


class TinyLaya(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.qemb = nn.Embedding(3, D)
        self.w = nn.Linear(D, 1)

    def forward(self, input_ids, attention_mask, marker_pos, qtype):
        m = attention_mask.float()[..., None]
        h = self.emb(input_ids.long()) * m
        # Each marker also sees the token after it (its option's first
        # word), so options score differently and their order matters.
        nxt = torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], 1)
        ctx = h.sum(1, keepdim=True) / m.sum(1, keepdim=True)
        h = torch.tanh(h + 2.0 * nxt + ctx + self.qemb(qtype.long())[None, :, :])
        pos = torch.arange(input_ids.shape[1], dtype=torch.int32)
        onehot = (marker_pos[:, :, None] == pos[None, None, :]).float()
        logits = self.w(torch.matmul(onehot, h)).squeeze(-1)
        return torch.where(marker_pos < 0, torch.full_like(logits, -1e4), logits)


class TinyReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.types = nn.Embedding(2, D)
        self.w = nn.Linear(D, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        m = attention_mask.float()[..., None]
        h = torch.tanh(self.emb(input_ids.long()) + self.types(token_type_ids.long())) * m
        return self.w(h.sum(1) / m.sum(1))


LAYA_INPUTS = lambda seq, k: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("marker_pos", (1, k)), ("qtype", (1,))]
PAIR_INPUTS = lambda seq: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("token_type_ids", (1, seq))]


def convert(model, shapes, out_dir, name):
    traced = torch.jit.trace(model, [torch.zeros(s, dtype=torch.int32) for _, s in shapes])
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.int32) for n, s in shapes],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, name + ".mlpackage")
        ml.save(pkg)
        subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, tmp], check=True, capture_output=True)
        target = os.path.join(out_dir, name + ".mlmodelc")
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(os.path.join(tmp, name + ".mlmodelc"), target)
    # Core ML's compile analytics aren't needed to load the model.
    shutil.rmtree(os.path.join(target, "analytics"), ignore_errors=True)


def main():
    root = sys.argv[1]
    out = os.path.join(root, "tiny-laya")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = TinyLaya().eval()
    convert(model, LAYA_INPUTS(16, 4), out, "model_16")
    convert(model, LAYA_INPUTS(32, 4), out, "model_32")
    convert(model, LAYA_INPUTS(32, 5), out, "k5_32")

    # Fixed inputs: (ids, markers, qtype). The ids are arbitrary vocabulary
    # entries; the tests run them directly, without the tokenizer.
    cases = [
        ([1, 10, 11, 2, 3, 12, 3, 13, 2, 14, 15, 2], [4, 6], 0),
        ([1, 16, 2, 3, 17, 3, 18, 3, 19, 2, 20, 21, 22, 2], [3, 5, 7], 1),
        ([1, 10, 2, 3, 23, 3, 24, 2] + [25] * 20 + [2], [3, 5], 2),
    ]
    expected = []
    for ids, markers, qtype in cases:
        mp = markers + [-1] * (4 - len(markers))
        with torch.no_grad():
            logits = model(
                torch.tensor([ids], dtype=torch.int32),
                torch.ones(1, len(ids), dtype=torch.int32),
                torch.tensor([mp], dtype=torch.int32),
                torch.tensor([qtype], dtype=torch.int32),
            )[0][: len(markers)]
        expected.append({"ids": ids, "markers": markers, "qtype": qtype, "logits": [float(x) for x in logits]})
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")

    out = os.path.join(root, "tiny-reranker")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(1)
    reranker = TinyReranker().eval()
    convert(reranker, PAIR_INPUTS(16), out, "model_16")
    convert(reranker, PAIR_INPUTS(32), out, "model_32")
    pairs = [
        ([1, 10, 11, 2, 12, 13, 14, 2], [0, 0, 0, 0, 1, 1, 1, 1]),
        ([1, 10, 11, 2, 10, 11, 2], [0, 0, 0, 0, 1, 1, 1]),
        ([1, 15, 2] + [16] * 22 + [2], [0, 0, 0] + [1] * 23),
    ]
    expected = []
    for ids, types in pairs:
        with torch.no_grad():
            score = reranker(
                torch.tensor([ids], dtype=torch.int32),
                torch.ones(1, len(ids), dtype=torch.int32),
                torch.tensor([types], dtype=torch.int32),
            )[0, 0]
        expected.append({"ids": ids, "type_ids": types, "score": float(score)})
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
