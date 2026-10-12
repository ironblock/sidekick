"""Techniques: each must be exact where it claims to be."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from sidekick_convert.techniques import activations, attention, masks, onehot, pooling, precision, reduce, saturation
from sidekick_convert.calibrate import Stat
from sidekick_convert.core import GateFailure


class Activations(unittest.TestCase):
    x = torch.linspace(-8, 8, 4001, dtype=torch.float64)

    def test_tanh_gelu_is_twice_gelu_tanh(self):
        want = 2 * F.gelu(self.x, approximate="tanh")
        torch.testing.assert_close(activations.TanhGelu()(self.x), want, rtol=0, atol=1e-12)

    def test_twice_gelu_is_twice_erf_gelu(self):
        torch.testing.assert_close(activations.TwiceGelu()(self.x), 2 * F.gelu(self.x), rtol=0, atol=1e-12)

    def test_tanh_silu_is_twice_silu(self):
        torch.testing.assert_close(activations.TanhSilu()(self.x), 2 * F.silu(self.x), rtol=0, atol=1e-12)


class Masks(unittest.TestCase):
    def test_key_padding(self):
        m = masks.key_padding(torch.tensor([[1, 1, 0, 0]]))
        self.assertEqual(tuple(m.shape), (1, 1, 1, 4))
        self.assertEqual(m.flatten().tolist(), [0.0, 0.0, masks.MASK_ADD, masks.MASK_ADD])

    def test_mask_is_finite_in_fp16(self):
        self.assertTrue(torch.isfinite(torch.tensor(masks.MASK_ADD, dtype=torch.float16)))

    def test_band_and_causal(self):
        b = masks.band(5, lambda d: d <= 1)[0, 0]
        self.assertEqual(b[2].tolist(), [masks.MASK_ADD, 0, 0, 0, masks.MASK_ADD])
        c = masks.causal(3)[0, 0]
        self.assertEqual(c.tolist(), [[0, masks.MASK_ADD, masks.MASK_ADD], [0, 0, masks.MASK_ADD], [0, 0, 0]])

    def test_self_attending_frees_only_the_diagonal(self):
        full = torch.full((1, 1, 4, 4), masks.MASK_ADD)
        m = masks.self_attending(full)[0, 0]
        self.assertTrue(torch.equal(masks.self_attending(full, 4)[0, 0], m))
        self.assertTrue(torch.equal(torch.diagonal(m), torch.zeros(4)))
        off = m[~torch.eye(4, dtype=torch.bool)]
        self.assertTrue(torch.all(off == masks.MASK_ADD))

    def test_self_attending_is_exact_for_real_rows(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 8, 4, dtype=torch.float64) for _ in range(3))
        am = torch.tensor([[1] * 5 + [0] * 3])
        add = masks.key_padding(am, torch.float64).expand(1, 1, 8, 8)
        a = attention.explicit(q, k, v, add, 0.5)
        b = attention.explicit(q, k, v, masks.self_attending(add), 0.5)
        torch.testing.assert_close(a[..., :5, :], b[..., :5, :], rtol=0, atol=1e-12)


class Attention(unittest.TestCase):
    def test_matmul_softmax_matches_explicit(self):
        torch.manual_seed(1)
        q, k, v = (torch.randn(1, 3, 256, 8, dtype=torch.float64) for _ in range(3))
        am = torch.tensor([[1] * 100 + [0] * 156])
        add = masks.self_attending(masks.key_padding(am, torch.float64).expand(1, 1, 256, 256))
        want = attention.explicit(q, k, v, add, 0.35)
        got = attention.matmul_softmax(q, k, v, add, 0.35, 256)
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


class Reduce(unittest.TestCase):
    def test_blocked_max_is_max(self):
        torch.manual_seed(2)
        for n in (64, 128, 256, 384, 512):
            for x in (torch.randn(2, 3, 5, n), -torch.rand(2, 3, 5, n) - 1.0):  # incl. all-negative rows
                torch.testing.assert_close(reduce.blocked_max(x, n), x.max(dim=-1, keepdim=True).values,
                                           rtol=0, atol=0)


class OneHot(unittest.TestCase):
    def test_positions_onehot_with_pads(self):
        pos = torch.tensor([[2, 0, -1]], dtype=torch.int32)
        oh = onehot.positions_onehot(pos, onehot.positions(4), torch.float32)
        self.assertEqual(oh[0].tolist(), [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])

    def test_index_onehot(self):
        oh = onehot.index_onehot(torch.tensor([2], dtype=torch.int32), onehot.indices(3), torch.float32)
        self.assertEqual(oh.tolist(), [[0, 0, 1]])


class Pooling(unittest.TestCase):
    h = torch.arange(24, dtype=torch.float64).reshape(1, 4, 6)
    am = torch.tensor([[1, 1, 1, 0]])

    def test_masked_mean_ignores_pads(self):
        got = pooling.masked_mean(self.h, self.am, 6)
        torch.testing.assert_close(got, self.h[:, :3].mean(dim=1), rtol=0, atol=1e-12)

    def test_last_token_and_cls(self):
        torch.testing.assert_close(pooling.last_token(self.h, self.am, 6), self.h[:, 2], rtol=0, atol=0)
        torch.testing.assert_close(pooling.cls(self.h, 6), self.h[:, 0], rtol=0, atol=0)

    def test_l2(self):
        y = pooling.l2(torch.tensor([[3.0, 4.0]]))
        torch.testing.assert_close(y, torch.tensor([[0.6, 0.8]]))


class Precision(unittest.TestCase):
    def stat(self, rms, mx):
        s = Stat()
        s.sumsq, s.count, s.max = rms * rms, 1, mx
        return s

    def test_input_scale_is_a_power_of_two_at_least_one(self):
        for rms in (0.004, 0.03, 0.5, 3.0):
            s = precision.input_scale(self.stat(rms, rms * 10))
            self.assertGreaterEqual(s, 1.0)
            self.assertEqual(np.log2(s), round(np.log2(s)))

    def test_input_scale_respects_headroom(self):
        s = precision.input_scale(self.stat(0.01, 100.0))   # rms wants 128, max caps at 2048/100
        self.assertLessEqual(100.0 * s, precision.IN_MAX)


class Saturation(unittest.TestCase):
    def test_k_is_minimal(self):
        self.assertEqual(saturation.choose_k(1000.0, 20000.0)[0], 1)
        self.assertEqual(saturation.choose_k(1000.0, 51500.0)[0], 2)   # ModernBERT's layer 15
        self.assertEqual(saturation.choose_k(1000.0, 90000.0)[0], 4)

    def test_k_fails_when_nothing_fits(self):
        with self.assertRaises(GateFailure):
            saturation.choose_k(30000.0, 1000.0)       # an unscaled output past the limit
        with self.assertRaises(GateFailure):
            saturation.choose_k(1000.0, 1e6)           # beyond K_MAX

    def test_headroom_at(self):
        self.assertAlmostEqual(saturation.headroom_at(1000.0, 51500.0, 2), 32768.0 / 25750.0)

    def test_check(self):
        self.assertEqual(saturation.check({"a": 100.0, "b": 300.0})[0], "b")
        with self.assertRaises(GateFailure):
            saturation.check({"a": 0.9 * saturation.ANE_LINEAR_MAX})


class EmbeddingGemma2Rewrite(unittest.TestCase):
    """The fp16 range rewrite's pieces, exact in fp32."""

    def test_safe_rmsnorm_matches_the_original(self):
        from types import SimpleNamespace
        from sidekick_convert.backbones.embedding_gemma2 import SafeRMSNorm
        torch.manual_seed(0)
        x = torch.randn(3, 64, dtype=torch.float64) * 900.0
        eps, s, gain = 1e-6, 1.0 / 32.0, 2.0
        for weight in (torch.rand(64, dtype=torch.float64) + 0.5, None):
            orig = SimpleNamespace(with_scale=weight is not None, weight=weight)
            want = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
            if weight is not None:
                want = want * weight
            got = SafeRMSNorm(orig, s / gain, eps * s * s)(x * gain)
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
            self.assertLess(float((x * gain * s / gain).abs().max()) ** 2, 65504.0)

    def test_pool_prescale_by_bucket(self):
        from sidekick_convert.heads.pool import Pool
        from types import SimpleNamespace
        h = torch.randn(1, 8, 6)
        am = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
        seen = []
        backbone = SimpleNamespace(call=lambda w, x: SimpleNamespace(last_hidden_state=h), hidden_size=6)
        pool = Pool("mean", prescale=lambda seq: seen.append(seq) or 2.0 ** -seq).bind(backbone)
        got = pool.forward(SimpleNamespace(seq=8), {"attention_mask": am}, backbone)
        self.assertEqual(seen, [8])
        torch.testing.assert_close(got, h[:, :5].mean(dim=1))


class BertAneRewrites(unittest.TestCase):
    """The BERT backbone's ANE rewrites are exact in fp32: the encoder
    written out with the matmul softmax, the erf GELU and the linear-floor
    rescale give a RoBERTa classifier's own logits."""

    def test_written_out_forward_matches(self):
        from types import SimpleNamespace
        from transformers import RobertaConfig, RobertaForSequenceClassification
        from sidekick_convert.backbones import bert
        torch.manual_seed(0)
        cfg = RobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                            intermediate_size=64, max_position_embeddings=40, num_labels=3,
                            attn_implementation="eager")
        model = RobertaForSequenceClassification(cfg).eval()
        for m in {id(model): model, id(model.base_model): model.base_model}.values():
            m.get_extended_attention_mask = bert._finite_extended_mask.__get__(m)
        bb = bert.BertBackbone(family="bert", model=model, config=model.config, special_ids=(0, 2),
                               token_types="zeros", position_offset=cfg.pad_token_id + 1)
        tok = SimpleNamespace(encode=lambda a, b, add_special_tokens=True: SimpleNamespace(
            ids=[0, 5, 6, 7, 2, 2, 9, 10, 2], type_ids=[0] * 9))
        ids = [0, 5, 6, 7, 2, 2, 9, 10, 2]
        with torch.no_grad():
            want = model(input_ids=torch.tensor([ids])).logits
        bert.twice_gelu(bb)
        bert.matmul_softmax_rewrite(bb)
        bert.linear_floor_rewrite([("a", "b")], tok, report=lambda *_: None)(bb)
        seq = 12
        w = SimpleNamespace(model=model, seq=seq, **bb.buffers(seq))
        x = {"input_ids": torch.tensor([ids + [1] * (seq - len(ids))]),
             "attention_mask": torch.tensor([[1] * len(ids) + [0] * (seq - len(ids))])}
        with torch.no_grad():
            got = bb.call(w, x).logits
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
