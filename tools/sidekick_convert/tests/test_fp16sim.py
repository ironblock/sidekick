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

    def test_fused_attention_runs_decomposed(self):
        q = torch.randn(1, 2, 6, 8)

        class Attn(torch.nn.Module):
            def forward(self, q):
                return F.scaled_dot_product_attention(q, q, q)

        got = fp16sim.run(Attn(), q)
        x = r(q)
        s = math.sqrt(1 / math.sqrt(8))
        scores = r(r(r(x * s) @ r(x.transpose(-2, -1) * s)))
        want = r(r(scores.softmax(-1)) @ x)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)

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


if __name__ == "__main__":
    unittest.main()
