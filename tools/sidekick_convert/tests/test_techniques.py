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


if __name__ == "__main__":
    unittest.main()
