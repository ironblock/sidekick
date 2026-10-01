"""EmbeddingGemma's sentence-transformers stack, in-graph: masked mean
pooling -> Dense -> Dense -> L2 normalize (docs/DECISIONS.md D17).

It also builds the attention masks, because Gemma3 takes them as
transformers' prepared-mask dict: an additive -30000 key-padding mask for the
full-attention layers, plus a precomputed band for the sliding ones.
transformers halves config.sliding_window for bidirectional models (512
becomes 257), and sliding layers attend iff |q - k| < that window.

fp16 range: channel sums over 512 tokens of |h| <= ~140, and sum(y^2) of the
~1e3-norm Dense output, both pass 65504. The pooling sum runs at 1/32 and is
divided by count/32, so the Dense stack sees the mean at natural scale (its
linears need O(1) inputs on the ANE), and the Dense output is scaled by 1/32
before the L2 sum of squares, which the normalization cancels.

The forward keeps the local names of the EmbeddingGemma converter it came
from, which name the traced values and so the converted program's variables.
"""

import dataclasses

import numpy as np
import torch

from ..techniques import masks

POOL_SCALE = 1.0 / 32.0


@dataclasses.dataclass
class MeanDenseL2:
    dense1_w: torch.Tensor        # 2_Dense weight, (3072, 768)
    dense2_w: torch.Tensor        # 3_Dense weight, (768, 3072)
    window: int = None
    output: str = "embeddings"
    task: str = None
    dims: int = None

    def bind(self, backbone):
        self.window = backbone.window
        self.dims = int(self.dense2_w.shape[0])
        return self

    def register(self, wrapper, seq):
        idx = torch.arange(seq)
        band = ((idx[:, None] - idx[None, :]).abs() >= self.window).to(torch.float32)
        wrapper.register_buffer("band_mask", band.reshape(1, 1, seq, seq) * masks.MASK_ADD)
        wrapper.dense1 = torch.nn.Linear(self.dims, self.dense1_w.shape[0], bias=False)
        wrapper.dense1.weight = torch.nn.Parameter(self.dense1_w)
        wrapper.dense2 = torch.nn.Linear(self.dense2_w.shape[1], self.dims, bias=False)
        wrapper.dense2.weight = torch.nn.Parameter(self.dense2_w)

    def forward(self, wrapper, x, backbone):
        mask_f = x["attention_mask"].to(torch.float32)
        addmask = (1.0 - mask_f).reshape(1, 1, 1, wrapper.seq) * masks.MASK_ADD
        h = getattr(wrapper, backbone.attr)(
            input_ids=x["input_ids"].long(),
            attention_mask={
                "full_attention": addmask,
                "sliding_attention": addmask + wrapper.band_mask,
            },
            position_ids=wrapper.position_ids,
            use_cache=False,
        ).last_hidden_state
        w = (mask_f * POOL_SCALE).unsqueeze(-1)
        summed = (h * w).sum(dim=1)
        count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
        pooled = summed / (count * POOL_SCALE)
        y = wrapper.dense2(wrapper.dense1(pooled)) * POOL_SCALE
        den = y.pow(2).sum(dim=-1, keepdim=True)
        out = y * torch.rsqrt(den + 1e-6)
        return out.reshape(1, self.dims)

    def reference(self, outputs):
        """sentence-transformers' math on an unpadded forward: mean ->
        Dense -> Dense -> L2."""
        h = outputs.last_hidden_state.double()
        pooled = h.mean(dim=1)
        y = pooled @ self.dense1_w.double().T @ self.dense2_w.double().T
        return (y / y.norm())[0].numpy()

    def st_mode(self):
        return "mean"
