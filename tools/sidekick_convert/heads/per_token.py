"""A per-token head: a module applied to every token's hidden state, one
output per token (docs/design/classify.md, the gliner2 format).

GLiNER2's classifier scores the encoder's output at each `[L]` label marker.
Applying it to all S tokens in the graph and letting the runtime read the
positions it placed needs no index inputs and no gather, and costs little:
GLiNER2.5-Decide's MLP (1024 -> 2048 -> 1) adds under 1% to the encoder.

The artifact returns `logits [1, S]`, a literal-shaped reshape. The fp32
reference is the module on an unpadded forward, one value per real token;
the runtime ignores the padded positions, and so do the gates (a case's
reference covers its first n slots).
"""

import dataclasses

import torch


@dataclasses.dataclass
class PerToken:
    module: torch.nn.Module          # hidden (..., H) -> (..., 1)
    attr: str = "token_head"         # the wrapper attribute, which names its converted weights
    output: str = "logits"
    task: str = None                 # the bare encoder

    def bind(self, backbone):
        return self

    def register(self, w, seq):
        w.add_module(self.attr, self.module)

    def forward(self, w, x, backbone):
        hidden = backbone.call(w, x).last_hidden_state
        return getattr(w, self.attr)(hidden).reshape(1, w.seq)

    def reference(self, outputs):
        with torch.no_grad():
            return self.module(outputs.last_hidden_state[0]).reshape(-1).double().numpy()
