"""EmbeddingGemma 2 (google/embeddinggemma-2): the text tower of a Gemma 4
based embedder, loaded text-only (the vision and audio towers dropped).

The architecture, as transformers 5.19 implements it
(models/embedding_gemma2):
- 24 bidirectional layers at hidden size 512. Sliding layers attend where
  |q - k| <= sliding_window (512, inclusive, not halved: 1,025 positions);
  full layers (5, 11, 17, 23) attend everywhere, at head_dim 512 with one
  KV head against the sliding layers' 256 and two. RoPE theta is 10k
  local, 1M global. Attention scaling is 1.0, with q and k RMSNorms.
- Per-layer inputs: a linear projection of the token embeddings to 24 x 512
  (no separate per-layer table), each layer gating its hidden state
  against its slice.
- A 512 -> 768 projection inside the model; the output is [B, S, 768].

What export needs: the masks. transformers builds them with mask
functions that don't trace; the model accepts them prepared, as a mapping
from layer type to an additive 4-D mask, so `onnx_call` builds the
key-padding mask and the sliding band from the input's length, and the
fp32 gate checks them against the checkpoint's own forward.

Requires transformers >= 5.19.
"""

import torch

from . import Backbone

MASK_ADD = -30000.0


class EmbeddingGemma2Backbone(Backbone):
    @property
    def hidden_size(self):
        """The output width: the 768 of the in-model projection, not the
        transformer's 512."""
        return int(self.config.text_config.embedding_dim)

    @property
    def vocab_size(self):
        return int(self.config.text_config.vocab_size)

    @property
    def window(self):
        return int(self.config.text_config.sliding_window)

    def reference(self, ids, token_type_ids=None):
        """The checkpoint's own forward (its own masks), fp32, unpadded."""
        t = torch.tensor([ids])
        with torch.no_grad():
            return self.model(input_ids=t, attention_mask=torch.ones_like(t))

    def onnx_call(self, model, input_ids, attention_mask):
        """The text model on dynamic shapes, with the masks transformers
        would build: key padding everywhere, plus the |q - k| > window band
        on sliding layers."""
        seq = input_ids.shape[1]
        idx = torch.arange(seq, device=input_ids.device)
        pad = (1.0 - attention_mask.to(torch.float32))[:, None, None, :] * MASK_ADD
        band = ((idx[:, None] - idx[None, :]).abs() > self.window).to(torch.float32)[None, None] * MASK_ADD
        return model.language_model(
            input_ids=input_ids,
            attention_mask={"full_attention": pad, "sliding_attention": pad + band},
            position_ids=idx[None, :].expand(input_ids.shape[0], -1),
        ).last_hidden_state


def load(src, tok):
    import transformers
    from transformers import AutoConfig, AutoModel
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 19):
        raise SystemExit(f"EmbeddingGemma 2 needs transformers >= 5.19, not {transformers.__version__}")
    config = AutoConfig.from_pretrained(src, vision_config=None, audio_config=None)
    model = AutoModel.from_pretrained(src, config=config, dtype=torch.float32, attn_implementation="eager").eval()
    from ..tokenizer import special_ids
    return EmbeddingGemma2Backbone(family="embedding_gemma2", model=model, config=model.config,
                                   special_ids=tuple(special_ids(tok)), forbid_ops={"gelu"})
