"""The ideal-fp16 simulation: input-dependent operations stored in fp16,
constants folded exactly and stored once, nothing it touches left changed."""

import math
import unittest

import torch
import torch.nn.functional as F

from sidekick_convert import fp16sim

r = fp16sim.round_fp16


def is_fp16_exact(t):
    return torch.equal(t, t.half().float())


class Fp16Sim(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(16, 8)
        self.x = torch.randn(3, 16)

    def test_linear_rounds_inputs_weights_and_output_once(self):
        want = r(F.linear(r(self.x), r(self.lin.weight), r(self.lin.bias)))
        got = fp16sim.run(self.lin, self.x)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_every_input_dependent_output_is_stored_in_fp16(self):
        seen = []

        class Probe(torch.nn.Module):
            def forward(self, x):
                y = (x @ x.T * 0.3 + 1.0).softmax(dim=-1)
                z = torch.tanh(y) * 2.0
                z += 0.1
                seen.extend([y, z])
                return z

        fp16sim.run(Probe(), self.x)
        self.assertTrue(all(is_fp16_exact(t) for t in seen))

    def test_constants_fold_exactly_and_are_stored_once(self):
        # a RoPE-like table: exact angles, rounded only where the input reads them
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))

        class Rope(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", inv_freq)

            def forward(self, x):
                angles = torch.arange(600).float()[:, None] * self.inv_freq[None, :]
                return x * angles.cos()

        x = torch.ones(600, 4)
        got = fp16sim.run(Rope(), x)
        exact = torch.arange(600).float()[:, None] * inv_freq[None, :]
        torch.testing.assert_close(got, r(r(exact.cos())), rtol=0, atol=0)
        rounded_angles = r(r(torch.arange(600).float())[:, None] * r(inv_freq)[None, :])
        self.assertGreater((r(rounded_angles.cos()) - got).abs().max().item(), 0.01)

    def test_fused_attention_runs_in_explicit_form(self):
        q = torch.randn(1, 2, 6, 8)

        class Attn(torch.nn.Module):
            def forward(self, q):
                return F.scaled_dot_product_attention(q, q, q)

        got = fp16sim.run(Attn(), q)
        x = r(q)
        scores = r(r(x @ x.transpose(-2, -1)) * (1 / math.sqrt(8)))
        want = r(r(scores.softmax(-1)) @ x)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_fused_and_explicit_attention_store_the_same(self):
        q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
        keep = torch.tensor([True] * 4 + [False] * 2).reshape(1, 1, 1, 6)
        add = torch.zeros(1, 1, 1, 6).masked_fill(~keep, -30000.0)

        class Fused(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.3)

        class Explicit(torch.nn.Module):
            def forward(self, q, k, v, mask):
                if mask.dtype == torch.bool:
                    mask = torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))
                return (torch.matmul(q, k.transpose(2, 3)) * 0.3 + mask).softmax(-1) @ v

        for mask in (add, keep):
            torch.testing.assert_close(fp16sim.run(Fused(), q, k, v, mask),
                                       fp16sim.run(Explicit(), q, k, v, mask), rtol=0, atol=0)

    def test_grouped_query_attention(self):
        q, k = torch.randn(1, 4, 6, 8), torch.randn(1, 2, 6, 8)

        class Fused(torch.nn.Module):
            def forward(self, q, k):
                return F.scaled_dot_product_attention(q, k, k, enable_gqa=True)

        class Explicit(torch.nn.Module):
            def forward(self, q, k):
                k = k.repeat_interleave(2, dim=1)
                return (torch.matmul(q, k.transpose(2, 3)) * (1 / math.sqrt(8))).softmax(-1) @ k

        torch.testing.assert_close(fp16sim.run(Fused(), q, k), fp16sim.run(Explicit(), q, k), rtol=0, atol=0)

    def test_transformer_fast_path_is_off_inside_only(self):
        before = torch.backends.mha.get_fastpath_enabled()
        layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True).eval()
        with fp16sim.ideal_fp16() as sim:
            self.assertFalse(torch.backends.mha.get_fastpath_enabled())
            out = layer(sim.input(torch.randn(1, 5, 16)))
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), before)
        self.assertTrue(is_fp16_exact(out))

    def test_module_and_inputs_are_left_unchanged(self):
        before = {k: v.clone() for k, v in self.lin.state_dict().items()}
        x = self.x.clone()
        fp16sim.run(self.lin, x)
        for k, v in self.lin.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertTrue(torch.equal(x, self.x))
        self.assertFalse(is_fp16_exact(self.lin.weight))

    def test_ops_filter_rounds_only_those_outputs(self):
        class Two(torch.nn.Module):
            def forward(self, x):
                return torch.exp(x * 0.1)

        x = torch.randn(4, 4)
        self.assertTrue(is_fp16_exact(fp16sim.run(Two(), x, ops={"exp"})))
        self.assertFalse(is_fp16_exact(fp16sim.run(Two(), x, ops={"mul"})))

    def test_integer_inputs_untouched(self):
        emb = torch.nn.Embedding(70001, 2)
        ids = torch.tensor([[70000, 3]])
        got = fp16sim.run(emb, input=ids)
        torch.testing.assert_close(got, r(emb(ids).detach()), rtol=0, atol=0)


class TinyRMSNorm(torch.nn.Module):
    """transformers' RMSNorm, decomposed: x^2 overflows fp16 for |x| > 256."""

    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + 1e-6))


class Normalizations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 4, 16)
        self.x[0, 1, 3] = 6631.0  # an attention-sink activation

    def test_a_normalization_is_one_operation(self):
        norm = TinyRMSNorm(16)
        got = fp16sim.run(norm, self.x)
        with torch.no_grad():
            want = r(norm(r(self.x)))
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertTrue(bool(got[0, 1].abs().amax() > 0.5))  # the sink token isn't zeroed

    def test_the_same_math_outside_a_norm_raises(self):
        class Unnamed(torch.nn.Module):
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        with self.assertRaises(fp16sim.Fp16Overflow) as caught:
            fp16sim.run(Unnamed(), self.x)
        self.assertIn("mean reads", str(caught.exception))
        self.assertIn("pow stored", str(caught.exception))


class NonFinite(unittest.TestCase):
    def test_a_finfo_min_mask_does_its_job(self):
        q = torch.randn(1, 2, 6, 8)
        keep = torch.tensor([1.0] * 4 + [0.0] * 2)

        class Masked(torch.nn.Module):
            def forward(self, q, keep):
                add = (1.0 - keep) * torch.finfo(torch.float32).min
                return (torch.matmul(q, q.transpose(-1, -2)) + add).softmax(-1) @ q

        out = fp16sim.run(Masked(), q, keep)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_an_overflowed_activation_fails_loudly(self):
        lin = torch.nn.Linear(8, 8)

        class Residual(torch.nn.Module):
            def forward(self, x):
                return torch.tanh(lin(x * 1000.0)) * 0.0  # the output would stay finite

        x = torch.full((1, 8), 100.0)
        with self.assertRaises(fp16sim.Fp16Overflow) as caught:
            fp16sim.run(Residual(), x)
        self.assertIn("mul stored |x| up to 1e+05", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
