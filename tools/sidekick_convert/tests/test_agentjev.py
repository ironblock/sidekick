"""The agentjev tree layout and candidate head: exact against per-path scoring.

AgentJev scores each candidate on its own causal path (prefix + candidate)
and reads the last token. The tree layout runs a question's prefix once with
every candidate as a sibling branch. These tests build a small random Qwen3
and head (no download) and check the converted wrapper against per-path
scoring in fp32, with random pad ids, plus the mask and the head's symmetry.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sidekick_convert.backbones import qwen3
from sidekick_convert.heads.agentjev import AgentJevHead
from sidekick_convert.techniques import masks
from sidekick_convert.wrapper import Wrapper

PREFIX = "path_encoder.backbone."
SET_DIM, HEADS = 16, 2


def tiny_checkpoint(dirname, seed=0):
    """A random Qwen3 backbone and AgentJev-shaped head, saved as AgentJev
    stores them: one model.safetensors with the backbone under PREFIX."""
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=120, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True)
    config.save_pretrained(dirname)
    model = Qwen3Model(config)
    layer = torch.nn.TransformerEncoderLayer(SET_DIM, HEADS, 4 * SET_DIM, 0.0, "gelu", batch_first=True,
                                             norm_first=True)
    state = {PREFIX + k: v.contiguous() for k, v in model.state_dict().items()}
    for name, mod in (("proj_in", torch.nn.Linear(32, SET_DIM)), ("proj_out", torch.nn.Linear(SET_DIM, 32))):
        state.update({f"{name}.{k}": v for k, v in mod.state_dict().items()})
    for i in range(2):
        state.update({f"set_encoder.encoder.layers.{i}.{k}": v.clone() + 0.01 * i
                      for k, v in layer.state_dict().items()})
    fc1, fc2 = torch.nn.Linear(32, SET_DIM), torch.nn.Linear(SET_DIM, 1)
    state.update({"scorer.norm.weight": torch.rand(32) + 0.5,
                  **{f"scorer.fc1.{k}": v for k, v in fc1.state_dict().items()},
                  **{f"scorer.fc2.{k}": v for k, v in fc2.state_dict().items()}})
    save_file(state, str(Path(dirname) / "model.safetensors"))


def load(dirname):
    backbone, rest = qwen3.load_tree(dirname, None, PREFIX)
    head = AgentJevHead.load(rest, backbone.hidden_size, kmax=4, set_dim=SET_DIM, heads=HEADS)
    return backbone, head


def tree_feed(prefix, suffixes, seq, kmax, pad_ids=None):
    ids, seg, pos, ends = list(prefix), [0] * len(prefix), list(range(len(prefix))), []
    for c, s in enumerate(suffixes, 1):
        ids += s
        seg += [c] * len(s)
        pos += range(len(prefix), len(prefix) + len(s))
        ends.append(len(ids) - 1)
    n = len(ids)
    pads = list(pad_ids[: seq - n]) if pad_ids is not None else [0] * (seq - n)
    t = lambda v: torch.tensor([v], dtype=torch.int32)  # noqa: E731
    return {"input_ids": t(ids + pads), "attention_mask": t([1] * n + [0] * (seq - n)),
            "seg": t(seg + [-1] * (seq - n)), "position_ids": t(pos + [0] * (seq - n)),
            "cand_end": t(ends + [-1] * (kmax - len(ends)))}


def per_path(backbone, head, prefix, suffixes):
    """AgentJev's own scoring: each path on its own, its last token's state,
    then the head over the candidates (nn modules, not the traced forms)."""
    with torch.no_grad():
        vecs = torch.stack([backbone.model(input_ids=torch.tensor([prefix + s]),
                                           attention_mask=torch.ones(1, len(prefix + s), dtype=torch.long))
                            .last_hidden_state[0, -1] for s in suffixes])[None]
        z = head.proj_in(vecs)
        for layer in head.set_layers:
            z = layer(z)
        return head.scorer(vecs + head.proj_out(z))[0]


class Tree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tiny_checkpoint(cls.dir.name)
        cls.backbone, cls.head = load(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_tree(self, prefix, suffixes, seq=32, pad_ids=None):
        w = Wrapper(self.backbone, self.head, self.head.ports(), seq)
        with torch.no_grad():
            return w(**tree_feed(prefix, suffixes, seq, self.head.kmax, pad_ids))[0]

    def test_the_tree_scores_every_path_exactly(self):
        prefix, suffixes = [5, 9, 13, 2, 7], [[40, 41], [50], [60, 61, 62]]
        want = per_path(self.backbone, self.head, prefix, suffixes)
        rng = np.random.default_rng(0)
        for pads in (None, rng.integers(1, 120, 32).tolist()):
            got = self.run_tree(prefix, suffixes, pad_ids=pads)
            np.testing.assert_allclose(got[:3].numpy(), want.numpy(), atol=2e-5)
            self.assertTrue(torch.all(got[3:] == -1e4))

    def test_reordering_candidates_permutes_the_logits(self):
        prefix, suffixes = [5, 9, 13], [[40, 41], [50], [60, 61, 62]]
        a = self.run_tree(prefix, suffixes)[:3]
        b = self.run_tree(prefix, [suffixes[2], suffixes[0], suffixes[1]])[:3]
        np.testing.assert_allclose(b.numpy(), a[[2, 0, 1]].numpy(), atol=2e-5)


class TreeMask(unittest.TestCase):
    def test_siblings_never_see_each_other_and_pads_see_themselves(self):
        seq = 7
        seg = torch.tensor([[0, 0, 1, 1, 2, -1, -1]], dtype=torch.int32)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
        b = masks.tree_buffers(seq)
        visible = masks.tree(seg, mask, b["tree_causal"], b["tree_eye"], seq)[0, 0] == 0
        want = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],     # branch 2 sees the prefix, not branch 1
            [0, 0, 0, 0, 0, 1, 0],     # pads see only themselves
            [0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(visible, want), visible.int())


if __name__ == "__main__":
    unittest.main()
