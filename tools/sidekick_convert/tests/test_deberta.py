"""The DeBERTa-v2 backbone and the relative shift: exact against transformers.

Uses small random-weight DeBERTa-v2 models built from a config (no download),
with position buckets small enough that the test lengths reach the
log-bucketed and clamped distances.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sidekick_convert.backbones import deberta_v2
from sidekick_convert.core import text_ports
from sidekick_convert.heads.per_token import PerToken
from sidekick_convert.techniques import relative_shift
from sidekick_convert.wrapper import Wrapper


def tiny_config(**overrides):
    from transformers import DebertaV2Config
    base = dict(vocab_size=120, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                intermediate_size=64, max_position_embeddings=64, relative_attention=True,
                position_buckets=8, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
                share_att_key=True, norm_rel_ebd="layer_norm", position_biased_input=False,
                type_vocab_size=0, hidden_act="gelu", layer_norm_eps=1e-7, pad_token_id=0)
    base.update(overrides)
    return DebertaV2Config(**base)


def tiny_backbone(seed=0, **overrides):
    from transformers import DebertaV2Model
    torch.manual_seed(seed)
    config = tiny_config(**overrides)
    model = DebertaV2Model(config).eval()
    deberta_v2._check_supported(config, model)
    return deberta_v2.DebertaV2Backbone(family="deberta-v2", model=model, config=config, special_ids=(1, 2))


class Hidden:
    """A test head: the backbone's hidden states, all of them."""
    output, task = "hidden", None

    def bind(self, backbone):
        return self

    def register(self, w, seq):
        pass

    def forward(self, w, x, backbone):
        return backbone.call(w, x).last_hidden_state


def padded_forward(backbone, ids, seq, pad_ids=None):
    w = Wrapper(backbone, Hidden(), text_ports(), seq).eval()
    x = torch.zeros((1, seq), dtype=torch.int32)
    if pad_ids is not None:
        x[0] = torch.as_tensor(pad_ids, dtype=torch.int32)
    x[0, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    m = torch.zeros((1, seq), dtype=torch.int32)
    m[0, : len(ids)] = 1
    with torch.no_grad():
        return w(x, m)[0, : len(ids)]


class RelativeShift(unittest.TestCase):
    def gathered(self, x, table, index, seq, key_side):
        """The transformers form: x · T, gathered by index(r - c)."""
        r = torch.arange(seq)
        rows = index(r[:, None] - r[None, :])                       # (L, L)
        full = x @ table.transpose(-1, -2)                          # (..., L, rows of T)
        if key_side:   # score[r, c] = x_c · T[idx(r - c)]
            return torch.gather(full, -1, rows.t().expand(full.shape[:-1] + (seq,))).transpose(-1, -2)
        return torch.gather(full, -1, rows.expand(full.shape[:-1] + (seq,)))

    def check(self, seq, index, rows_in_table):
        torch.manual_seed(seq)
        x = torch.randn(2, 3, seq, 5, dtype=torch.float64)
        table = torch.randn(rows_in_table, 5, dtype=torch.float64)
        q = relative_shift.query_side(x, table[relative_shift.query_side_index(index, seq)], seq)
        k = relative_shift.key_side(x, table[relative_shift.key_side_index(index, seq)], seq)
        torch.testing.assert_close(q, self.gathered(x, table, index, seq, False), rtol=0, atol=1e-12)
        torch.testing.assert_close(k, self.gathered(x, table, index, seq, True), rtol=0, atol=1e-12)

    def test_linear_distances(self):
        for seq in (1, 2, 5, 9):
            self.check(seq, lambda d: d + seq - 1, 2 * seq - 1)

    def test_debertas_log_buckets_and_clamp(self):
        # 4 buckets each way: distances past 2 are log-bucketed, and past the
        # maximum position (8 here) the bucket leaves the table and clamps
        index = lambda d: deberta_v2.relative_index(d, 4, 8, 4)
        self.assertEqual(int(index(torch.tensor(15))), 7)
        self.assertEqual(int(index(torch.tensor(-15))), 0)
        for seq in (3, 7, 16):
            self.check(seq, index, 8)

    def test_skew_reads_column_by_distance(self):
        seq = 4
        d = relative_shift.distances(seq).to(torch.float64)        # column e holds distance d[e]
        out = relative_shift.skew(d.expand(seq, -1), seq)
        r = torch.arange(seq, dtype=torch.float64)
        torch.testing.assert_close(out, r[:, None] - r[None, :], rtol=0, atol=0)


class Backbone(unittest.TestCase):
    ids = [1, 17, 33, 5, 98, 41, 7, 66, 23, 2]

    def check_exact(self, backbone, seqs=(10, 16, 24)):
        with torch.no_grad():
            want = backbone.reference(self.ids).last_hidden_state[0]
        for seq in seqs:
            got = padded_forward(backbone, self.ids, seq)
            torch.testing.assert_close(got, want, rtol=0, atol=2e-5, msg=f"bucket {seq}")

    def test_matches_transformers_padded(self):
        self.check_exact(tiny_backbone())

    def test_one_term_and_separate_position_projections(self):
        self.check_exact(tiny_backbone(pos_att_type=["c2p"]))
        self.check_exact(tiny_backbone(pos_att_type=["p2c"], share_att_key=False))
        self.check_exact(tiny_backbone(share_att_key=False))

    def test_without_position_buckets(self):
        # distances up to 9 between the real tokens, past 6: the clamp is exercised
        self.check_exact(tiny_backbone(position_buckets=-1, max_relative_positions=6))

    def test_pad_content_is_invisible(self):
        b = tiny_backbone()
        a = padded_forward(b, self.ids, 24)
        noisy = padded_forward(b, self.ids, 24, pad_ids=np.random.default_rng(0).integers(3, 120, 24))
        torch.testing.assert_close(a, noisy, rtol=0, atol=0)

    def test_twice_gelu_is_exact(self):
        b = tiny_backbone()
        with torch.no_grad():
            want = b.reference(self.ids).last_hidden_state[0]
        deberta_v2.twice_gelu(b)
        self.assertIn("gelu", b.forbid_ops)
        torch.testing.assert_close(padded_forward(b, self.ids, 16), want, rtol=0, atol=2e-5)

    def test_unsupported_configs_are_refused(self):
        with self.assertRaises(SystemExit):
            tiny_backbone(conv_kernel_size=3)
        with self.assertRaises(SystemExit):
            tiny_backbone(pos_att_type=["p2p"])


class Gliner2(unittest.TestCase):
    def test_checkpoint_layout_and_per_token_head(self):
        from safetensors.torch import save_file
        b = tiny_backbone(seed=3)
        torch.manual_seed(4)
        mlp = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        state = {f"encoder.{k}": v.contiguous() for k, v in b.model.state_dict().items()}
        state.update({f"classifier.{k}": v.contiguous() for k, v in mlp.state_dict().items()})
        state["span_rep.unused.weight"] = torch.zeros(2, 2)      # other GLiNER2 modules are ignored
        tok = SimpleNamespace(encode=lambda text, add_special_tokens=True: SimpleNamespace(ids=[1, 2]))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "encoder_config").mkdir()
            (Path(d) / "encoder_config" / "config.json").write_text(json.dumps(b.config.to_dict()))
            save_file(state, str(Path(d) / "model.safetensors"))
            loaded = deberta_v2.load(d, tok)
            classifier = deberta_v2.gliner2_classifier(d)
        ids = Backbone.ids
        head = PerToken(classifier).bind(loaded)
        ref = head.reference(loaded.reference(ids))
        with torch.no_grad():
            want = mlp(b.reference(ids).last_hidden_state[0]).reshape(-1).double().numpy()
        np.testing.assert_allclose(ref, want, rtol=0, atol=1e-6)
        seq = 16
        w = Wrapper(loaded, head, text_ports(), seq).eval()
        x = torch.zeros((1, seq), dtype=torch.int32)
        x[0, : len(ids)] = torch.tensor(ids)
        m = (x != 0).to(torch.int32)
        with torch.no_grad():
            out = w(x, m)
        self.assertEqual(tuple(out.shape), (1, seq))
        np.testing.assert_allclose(out[0, : len(ids)].double().numpy(), want, rtol=0, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
