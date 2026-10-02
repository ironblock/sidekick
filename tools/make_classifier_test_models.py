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

tiny-gliner2/: the gliner2 format. `input_ids`/`attention_mask` [1, S] and
`logits` [1, S], one per token, which the runtime reads at the [L] markers.
- `model_16.mlmodelc`, `model_32.mlmodelc`;
- `short_32.mlmodelc`: bucket 32 whose output is only 16 wide, which the
  load check must refuse;
- `expected.json`: fp32 torch logits at the markers for fixed inputs.

tiny-fev/: the fev format, causal. `input_ids`/`attention_mask` [1, S],
`marker_pos` [1, K] (each option's end, -1 in unused slots), `decide_pos`
[1], and `logits` [1, K]: a pointer head between each option's state and
the decide token's, unused slots at -1e4. Each token's state is the mean
of the embeddings up to it, so pads after the row change nothing.
- `model_16.mlmodelc`, `model_32.mlmodelc`: K = 4;
- `expected.json`: fp32 torch logits for fixed (ids, ends, decide).

tiny-agentjev/: the agentjev format, a tree. `input_ids`/`attention_mask`/
`seg`/`position_ids` [1, S], `cand_end` [1, K] (each candidate's last
token, -1 in unused slots), and `logits` [1, K], unused slots at -1e4. One
attention layer masked by sidekick_convert's own masks.tree (siblings never
see each other, pads see only themselves), positions through a one-hot
table, and a head that mixes each candidate's state with the mean of the
real candidates', so the order of candidates doesn't matter. It imports
the mask from tools/sidekick_convert (techniques.masks, torch only), so the
fixture is built with the production mask, not a copy.
- `model_16.mlmodelc`, `model_32.mlmodelc`: K = 4;
- `expected.json`: fp32 torch logits for fixed (ids, seg, positions, ends).

tiny-multishape/: one artifact, `model.mlmodelc`, whose `input_ids` and
`attention_mask` each accept two enumerated shapes, [1, 16] and [1, 32],
and whose `logits` are [1, 1]. Its layout is the one macOS 27 can abort
on, which the loader must refuse there whatever the compute units (D27).

Usage:
    python tools/make_classifier_test_models.py crates/sidekick-embed/tests/fixtures [name ...]

    name    tiny-laya, tiny-reranker, tiny-gliner2, tiny-fev, tiny-multishape,
            tiny-agentjev (default: all but tiny-agentjev, which is built
            only when named). Building
            only the one you changed keeps the others' committed bytes.

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


class TinyGliner2(nn.Module):
    """One logit per token, from the token, the next one and the mean of the
    real tokens, so a marker's logit depends on its label and the context."""

    def __init__(self, seq=None, width=None):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.w = nn.Linear(D, 1)
        self.seq, self.width = seq, width

    def forward(self, input_ids, attention_mask):
        m = attention_mask.float()[..., None]
        h = self.emb(input_ids.long()) * m
        nxt = torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], 1)
        ctx = h.sum(1, keepdim=True) / m.sum(1, keepdim=True)
        logits = self.w(torch.tanh(h + 2.0 * nxt + ctx)).reshape(1, self.seq)
        return logits[:, : self.width] if self.width else logits


LAYA_INPUTS = lambda seq, k: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("marker_pos", (1, k)), ("qtype", (1,))]
PAIR_INPUTS = lambda seq: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("token_type_ids", (1, seq))]


class TinyFev(nn.Module):
    """A causal pointer head: each token's state is the running mean of the
    embeddings so far, and an option scores q(decide) . k(its end)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.q, self.k = nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, input_ids, attention_mask, marker_pos, decide_pos):
        h = torch.tanh(self.emb(input_ids.long()))
        n = torch.arange(1, input_ids.shape[1] + 1, dtype=torch.float32)[None, :, None]
        h = torch.cumsum(h, 1) / n
        pos = torch.arange(input_ids.shape[1], dtype=torch.int32)
        ends = torch.matmul((marker_pos[:, :, None] == pos[None, None, :]).float(), h)
        decide = torch.matmul((decide_pos[:, None, None] == pos[None, None, :]).float(), h)
        logits = (self.k(ends) * self.q(decide)).sum(-1) / D ** 0.5
        return torch.where(marker_pos < 0, torch.full_like(logits, -1e4), logits)


class TinyAgentJev(nn.Module):
    """A tree: one attention layer under masks.tree, positions from a
    one-hot table, and a head over the candidate ends that adds the mean of
    the real candidates' states, so it is permutation-equivariant."""

    def __init__(self, seq):
        super().__init__()
        from sidekick_convert.techniques import masks
        self.masks, self.seq = masks, seq
        self.emb = nn.Embedding(VOCAB, D)
        self.pos = nn.Parameter(torch.randn(32, D) * 0.5)
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.w = nn.Linear(D, 1)
        # per-bucket constants, not weights: kept out of state_dict
        for name, t in masks.tree_buffers(seq).items():
            self.register_buffer(name, t, persistent=False)
        self.register_buffer("positions", torch.arange(seq, dtype=torch.int32).reshape(1, 1, seq), persistent=False)

    def forward(self, input_ids, attention_mask, seg, position_ids, cand_end):
        seq = self.seq
        onehot = (position_ids.reshape(1, seq, 1) == self.positions).float()
        x = torch.tanh(self.emb(input_ids.long()) + onehot @ self.pos[:seq])
        add = self.masks.tree(seg, attention_mask, self.tree_causal, self.tree_eye, seq)
        att = torch.softmax(self.q(x) @ self.k(x).transpose(1, 2) / D ** 0.5 + add[:, 0], -1)
        h = x + att @ self.v(x)
        sel = (cand_end.reshape(1, -1, 1) == self.positions).float()
        valid = (cand_end >= 0).float()
        c = sel @ h
        ctx = (c * valid[..., None]).sum(1, keepdim=True) / valid.sum(1, keepdim=True)[..., None]
        logits = self.w(torch.tanh(c + ctx)).squeeze(-1)
        return torch.where(cand_end < 0, torch.full_like(logits, -1e4), logits)


AGENTJEV_INPUTS = lambda seq, k: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("seg", (1, seq)),
                                  ("position_ids", (1, seq)), ("cand_end", (1, k))]


FEV_INPUTS = lambda seq, k: [("input_ids", (1, seq)), ("attention_mask", (1, seq)), ("marker_pos", (1, k)), ("decide_pos", (1,))]


class TinyPooled(nn.Module):
    """A score from the mean of the real tokens' embeddings: any length."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.w = nn.Linear(D, 1)

    def forward(self, input_ids, attention_mask):
        m = attention_mask.float()[..., None]
        h = torch.tanh(self.emb(input_ids.long())) * m
        return self.w(h.sum(1) / m.sum(1))


def convert(model, shapes, out_dir, name):
    """`shapes`: (name, shape) per input; a shape may be a list of shapes,
    which the input then accepts as enumerated shapes."""
    traced = torch.jit.trace(model, [torch.zeros(s[0] if isinstance(s, list) else s, dtype=torch.int32)
                                     for _, s in shapes])
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=ct.EnumeratedShapes(shapes=s) if isinstance(s, list) else s,
                              dtype=np.int32) for n, s in shapes],
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
    names = sys.argv[2:] or ["tiny-laya", "tiny-reranker", "tiny-gliner2", "tiny-fev", "tiny-multishape"]
    if "tiny-laya" in names:
        laya(root)
    if "tiny-reranker" in names:
        reranker(root)
    if "tiny-gliner2" in names:
        gliner2(root)
    if "tiny-fev" in names:
        fev(root)
    if "tiny-multishape" in names:
        multishape(root)
    if "tiny-agentjev" in names:
        agentjev(root)


def agentjev(root):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "tiny-agentjev")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(5)
    model16 = TinyAgentJev(16).eval()
    with torch.no_grad():   # spread the logits well past fp16 rounding, so a wrong slot or a pad leak shows
        model16.w.weight.mul_(8.0)
        model16.w.bias.mul_(8.0)
    model32 = TinyAgentJev(32).eval()
    model32.load_state_dict(model16.state_dict())     # the same weights in both buckets
    convert(model16, AGENTJEV_INPUTS(16, 4), out, "model_16")
    convert(model32, AGENTJEV_INPUTS(32, 4), out, "model_32")
    # Fixed trees: a prefix (segment 0), then each candidate's branch, its
    # positions continuing from the prefix's end.
    def tree(prefix, branches):
        ids, seg, pos, ends = list(prefix), [0] * len(prefix), list(range(len(prefix))), []
        for c, b in enumerate(branches, 1):
            ids += b; seg += [c] * len(b); pos += range(len(prefix), len(prefix) + len(b)); ends.append(len(ids) - 1)
        return ids, seg, pos, ends
    cases = [tree([5, 6, 7, 8], [[9, 10], [11]]),
             tree([5, 12, 13, 14, 15, 16], [[17, 18], [19], [20, 21, 22]]),
             tree([5] + [23] * 14, [[24, 25], [26, 27], [28], [29, 30]])]
    expected = []
    for ids, seg, pos, ends in cases:
        seq = 16 if len(ids) <= 16 else 32
        model = model16 if seq == 16 else model32
        pad = seq - len(ids)
        t = lambda v, fill: torch.tensor([v + [fill] * pad], dtype=torch.int32)  # noqa: E731
        with torch.no_grad():
            logits = model(t(ids, 0), t([1] * len(ids), 0), t(seg, -1), t(pos, 0),
                           torch.tensor([ends + [-1] * (4 - len(ends))], dtype=torch.int32))[0][: len(ends)]
        expected.append({"ids": ids, "seg": seg, "position_ids": pos, "markers": ends,
                         "logits": [float(x) for x in logits]})
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")


def fev(root):
    out = os.path.join(root, "tiny-fev")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(4)
    model = TinyFev().eval()
    convert(model, FEV_INPUTS(16, 4), out, "model_16")
    convert(model, FEV_INPUTS(32, 4), out, "model_32")
    # Fixed rows laid out as fev's are: <s> state <q> question (<o> option
    # </o>)... <d>, with delimiter ids 0-4 and arbitrary words after them.
    cases = [
        ([0, 10, 11, 1, 12, 2, 13, 3, 2, 14, 3, 4], [7, 10], 11),
        ([0, 15, 16, 17, 1, 2, 18, 19, 3, 2, 20, 3, 2, 21, 3, 4], [8, 11, 14], 15),
        ([0] + [22] * 15 + [1, 2, 23, 3, 2, 24, 3, 4], [19, 22], 23),
    ]
    expected = []
    for ids, ends, decide in cases:
        mp = ends + [-1] * (4 - len(ends))
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.int32), torch.ones(1, len(ids), dtype=torch.int32),
                           torch.tensor([mp], dtype=torch.int32), torch.tensor([decide], dtype=torch.int32))[0][: len(ends)]
        expected.append({"ids": ids, "markers": ends, "decide": decide, "logits": [float(x) for x in logits]})
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")


def multishape(root):
    out = os.path.join(root, "tiny-multishape")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(3)
    shapes = [(1, 16), (1, 32)]
    convert(TinyPooled().eval(), [("input_ids", shapes), ("attention_mask", shapes)], out, "model")


def laya(root):
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

def reranker(root):
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


def gliner2(root):
    out = os.path.join(root, "tiny-gliner2")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(2)
    model = TinyGliner2().eval()
    text_inputs = lambda seq: [("input_ids", (1, seq)), ("attention_mask", (1, seq))]
    for seq, width, name in [(16, None, "model_16"), (32, None, "model_32"), (32, 16, "short_32")]:
        model.seq, model.width = seq, width
        convert(model, text_inputs(seq), out, name)
    model.seq, model.width = None, None
    # Fixed inputs laid out as the gliner2 format lays them out:
    # ( [P] prompt ( [L] label [L] label ) ) [SEP_TEXT] words ., with the ids
    # of the test tokenizer in crates/sidekick-embed/tests/coreml_classifier.rs.
    cases = [
        ([5, 1, 9, 5, 2, 10, 2, 11, 6, 6, 3, 12, 13, 7], [4, 6]),
        ([5, 1, 9, 5, 2, 10, 2, 11, 2, 14, 6, 6, 3, 15, 16, 17, 7], [4, 6, 8]),
        ([5, 1, 9, 5, 2, 12, 2, 13, 6, 6, 3] + [10] * 18 + [7], [4, 6]),
    ]
    expected = []
    for ids, markers in cases:
        model.seq = len(ids)
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.int32), torch.ones(1, len(ids), dtype=torch.int32))[0]
        expected.append({"ids": ids, "markers": markers, "logits": [float(logits[m]) for m in markers]})
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
