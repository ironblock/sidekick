"""Activations built from ops the ANE computes accurately.

Core ML's native gelu and silu ops are coarse on the ANE: gelu is off by up
to ~6e-3 on [-1, 1], where most gate activations lie, and silu by ~1.5e-2
(docs/DECISIONS.md D17, D19, D20). The forms here are built from tanh or erf
with mul and add. Pair them with a gate that forbids the native op
(Job.forbid_ops), since x * sigmoid(x) is fused back into silu by
coremltools.

Each returns TWICE the activation: the factor 2 saves a multiply. GAIN
records it, and the recipe folds 1/GAIN into the next linear or into a
scale-invariant norm downstream.

Which one: follow the model's own `hidden_act`, because each is exact only
for its own definition.
- TanhGelu for gelu_pytorch_tanh (Gemma): exact to the definition.
- TwiceGelu for erf "gelu" (BERT, ModernBERT): exact in fp32. It is opt-in
  and measured per model: on gte-modernbert an erf GELU didn't help the ANE
  and made the CPU path worse (D25 amendment), while laya measured fewer
  argmax flips with it.
- TanhSilu for silu (LFM2, Qwen3).
- StableSilu for silu where TanhSilu's 1 + tanh(x/2) cancels: for x < 0
  it approaches 1 - 1, losing fp16's precision where silu is small. On
  agent-jev (Qwen3-0.6B, D37), whose first layers run on a tiny residual,
  it cut the ANE's error after layers 0 and 1 to 2.4e-3 and 1.6e-3 (rms,
  relative; native silu 1.0e-2 and 4.0e-3, TanhSilu 2.5e-3 and 2.2e-3).
"""

import math

import torch

GELU_C = (2.0 / math.pi) ** 0.5   # written as the EmbeddingGemma converter wrote it
INV_SQRT2 = 0.7071067811865476    # the literal laya's measured TwiceGelu uses


class TanhGelu(torch.nn.Module):
    """2 * gelu_pytorch_tanh(x) = x * (1 + tanh(x * (c + c*0.044715*x^2))).
    ANE error <= ~9e-4 on [-1, 1] (~2e-4 near 0). x^2 overflows fp16 only
    past |x| ~ 256, where it saturates tanh and still gives 2x or 0."""
    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.tanh(x * (GELU_C + (GELU_C * 0.044715) * (x * x))))


class TwiceGelu(torch.nn.Module):
    """2 * gelu(x) = x * (1 + erf(x / sqrt(2))), the erf GELU (opt-in).
    Never write 0.5 * x * (1 + erf(...)): coremltools fuses that back into
    the native gelu op. Fold the 0.5 into the next linear instead (in a gated
    MLP, into the gate rows). On laya it cut ANE argmax flips from 5 to 1 and
    max |dp| from 0.077 to 0.039; Core ML's CPU erf is coarser than its gelu,
    so the CPU path got slightly worse (laya CPU flips 6 -> 16)."""
    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.erf(x * INV_SQRT2))


class StableSilu(torch.nn.Module):
    """2 * silu(x) = 2x * sigmoid(x), with sigmoid(x) = exp(min(x, 0)) /
    (1 + exp(-|x|)): exp, abs, clip and a divide, no cancellation for
    either sign, and no overflow."""
    GAIN = 2.0

    def forward(self, x):
        return 2.0 * x * torch.exp(torch.clamp(x, max=0.0)) / (1.0 + torch.exp(-torch.abs(x)))


class TanhSilu(torch.nn.Module):
    """2 * silu(x) = x * (1 + tanh(x / 2)). ANE error ~1.6e-3 on [-1, 1]."""
    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.tanh(0.5 * x))
