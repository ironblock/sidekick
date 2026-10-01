"""laya's decision head (docs/DECISIONS.md D28): a question-type embedding
added to the encoder states, a two-layer transformer head, and a scorer read
at each option's [MASK] marker. One logit per marker, padded slots at
`pad_logit`.

The int32 interface builds its selections in-graph, with no data-dependent
gather: `marker_pos [1, K]` (-1 pads unused slots) becomes a [K, S] one-hot
against a position constant, and `qtype [1]` a one-hot row selecting the
question-type embedding (techniques.onehot). The head's attention is written
out (techniques.attention.transformer_encoder_layer), so neither
nn.TransformerEncoderLayer's fast path nor the fused attention op is traced.

Its fp32 references come from the model's own forward, not from the
encoder, so a converter builds laya's evaluation cases itself. The forward
keeps the local names of laya's original converter, which name the traced
values and so the converted program's variables.
"""

import dataclasses

from ..core import Port, sequence_port
from ..techniques import activations, attention, masks
from ..techniques.onehot import index_onehot, indices, positions, positions_onehot


@dataclasses.dataclass
class LayaMarkers:
    head_layers: object              # nn.ModuleList of nn.TransformerEncoderLayer(norm_first)
    scorer: object                   # Sequential(LayerNorm, Linear, GELU, Linear)
    type_weight: object              # the question-type embedding's weight, (3, hidden)
    kmax: int = 32
    pad_logit: float = -1e4
    softmax: str = "native"          # "matmul": techniques.attention.matmul_softmax (opt-in)
    output: str = "logits"
    task: str = None

    def ports(self):
        return [sequence_port("input_ids"), sequence_port("attention_mask"),
                Port("marker_pos", lambda seq: (1, self.kmax)), Port("qtype", lambda seq: (1,))]

    def bind(self, backbone):
        return self

    def register(self, w, seq):
        w.layers = self.head_layers
        w.scorer = self.scorer
        w.type_w = self.type_weight
        w.register_buffer("positions", positions(seq))
        w.register_buffer("qtypes", indices(3))

    def forward(self, w, x, backbone):
        h = backbone.call(w, x).last_hidden_state
        qt = index_onehot(x["qtype"], w.qtypes, h.dtype)                         # [1, 3]
        h = h + (qt @ w.type_w).unsqueeze(1)
        add = (1.0 - x["attention_mask"].to(h.dtype))[:, None, None, :] * masks.MASK_ADD
        for layer in w.layers:
            h = attention.transformer_encoder_layer(layer, h, add, w.seq, self.softmax)
        onehot = positions_onehot(x["marker_pos"], w.positions, h.dtype)        # [1, K, S]
        valid = (x["marker_pos"] >= 0).to(h.dtype)                              # [1, K]
        logits = w.scorer(onehot @ h).squeeze(-1)
        return logits * valid + (valid - 1.0) * -self.pad_logit


def twice_gelu_scorer(head):
    """Opt-in rewrite: the scorer's erf GELU becomes TwiceGelu, with the 0.5
    folded into the following linear's weight (its bias is unchanged)."""
    import torch
    if not (isinstance(head.scorer[2], torch.nn.GELU) and head.scorer[2].approximate == "none"):
        raise SystemExit("twice_gelu_scorer expects the scorer's third module to be an erf GELU")
    head.scorer[2] = activations.TwiceGelu()
    with torch.no_grad():
        head.scorer[3].weight.mul_(1.0 / activations.TwiceGelu.GAIN)
