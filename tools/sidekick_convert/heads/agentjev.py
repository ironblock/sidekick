"""AgentJev's candidate head (the agentjev format, docs/design/classify.md).

The backbone runs a question as a tree, and the head reads the hidden state
at each candidate's last token. AgentJev then scores the candidates as a
set:
- proj_in (hidden -> 256), zeroed on unused slots;
- a two-layer transformer over the candidates, with no positions, so its
  output is permutation-equivariant; unused slots are masked as keys;
- proj_out (256 -> hidden), added back to the candidate states;
- a scorer: RMSNorm -> Linear -> SiLU -> Linear, one logit per candidate.
Unused slots return `pad_logit`.

The int32 interface builds its selection in-graph, with no data-dependent
gather: `cand_end [1, K]` (-1 pads unused slots) becomes a [K, S] one-hot
against a position constant (techniques.onehot). The set transformer's
attention is written out (techniques.attention.transformer_encoder_layer),
so neither nn.TransformerEncoderLayer's fast path nor the fused attention
op is traced.

`load(state)` builds the head from the checkpoint's own tensors (proj_in.*,
set_encoder.encoder.layers.N.*, proj_out.*, scorer.*), with the shapes
AgentJev's agentjev/model.py defines.
"""

import dataclasses

import torch

from ..core import Port, sequence_port
from ..techniques import attention, masks
from ..techniques.onehot import positions, positions_onehot


class _Scorer(torch.nn.Module):
    """AgentJev's ScalarScorer: RMSNorm (computed in fp32) -> fc1 -> SiLU -> fc2."""

    def __init__(self, hidden, inner, eps=1e-6):
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.ones(hidden))
        self.eps = eps
        self.fc1 = torch.nn.Linear(hidden, inner)
        self.fc2 = torch.nn.Linear(inner, 1)

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_weight
        return self.fc2(torch.nn.functional.silu(self.fc1(x))).squeeze(-1)


@dataclasses.dataclass
class AgentJevHead:
    proj_in: object                  # Linear(hidden, set_dim)
    set_layers: object               # nn.ModuleList of nn.TransformerEncoderLayer(norm_first, gelu)
    proj_out: object                 # Linear(set_dim, hidden)
    scorer: object                   # _Scorer
    kmax: int = 32
    pad_logit: float = -1e4
    output: str = "logits"

    @classmethod
    def load(cls, state, hidden, kmax=32, set_dim=256, heads=4, layers=2):
        proj_in, proj_out = torch.nn.Linear(hidden, set_dim), torch.nn.Linear(set_dim, hidden)
        set_layers = torch.nn.ModuleList(
            torch.nn.TransformerEncoderLayer(set_dim, heads, dim_feedforward=4 * set_dim, dropout=0.0,
                                             activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers))
        scorer = _Scorer(hidden, set_dim)
        parts = {"proj_in.": proj_in, "proj_out.": proj_out, "set_encoder.encoder.layers.": set_layers}
        for prefix, module in parts.items():
            module.load_state_dict({k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)},
                                   strict=True)
        scorer.load_state_dict({"norm_weight": state["scorer.norm.weight"].float(),
                                **{k[len("scorer."):]: v.float() for k, v in state.items()
                                   if k.startswith("scorer.fc")}}, strict=True)
        extra = sorted(k for k in state if not k.startswith(("proj_in.", "proj_out.", "set_encoder.", "scorer.")))
        if extra:
            raise ValueError(f"tensors the head doesn't take: {extra}")
        for m in (proj_in, proj_out, set_layers, scorer):
            m.eval().requires_grad_(False)
        return cls(proj_in, set_layers, proj_out, scorer, kmax=kmax)

    def ports(self):
        return [sequence_port("input_ids"), sequence_port("attention_mask"), sequence_port("seg"),
                sequence_port("position_ids"), Port("cand_end", lambda seq: (1, self.kmax))]

    def bind(self, backbone):
        return self

    def register(self, w, seq):
        w.proj_in, w.set_layers, w.proj_out, w.scorer = self.proj_in, self.set_layers, self.proj_out, self.scorer
        w.register_buffer("positions", positions(seq))

    def forward(self, w, x, backbone):
        h = backbone.call(w, x).last_hidden_state                               # [1, S, H]
        onehot = positions_onehot(x["cand_end"], w.positions, h.dtype)          # [1, K, S]
        valid = (x["cand_end"] >= 0).to(h.dtype)                                # [1, K]
        v = onehot @ h                                                          # [1, K, H]
        z = w.proj_in(v) * valid.unsqueeze(-1)
        add = (1.0 - valid)[:, None, None, :] * masks.MASK_ADD                  # unused slots as keys
        for layer in w.set_layers:
            z = attention.transformer_encoder_layer(layer, z, add, self.kmax)
        logits = w.scorer(v + w.proj_out(z))
        return logits * valid + (valid - 1.0) * -self.pad_logit
