"""Pooling heads for embedders: one vector per input, pooled in-graph
(docs/DECISIONS.md D15). sidekick normalizes the output again in f32."""

import dataclasses

import numpy as np

from ..techniques import pooling

MODES = ("cls", "mean", "last_token")


@dataclasses.dataclass
class Pool:
    mode: str = "cls"
    l2: bool = False                 # normalize in-graph (LFM2.5); the server normalizes anyway
    prescale: float = pooling.PRESCALE
    output: str = "embedding"
    task: str = None
    dims: int = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"pooling mode must be one of {MODES}, not {self.mode!r}")

    def bind(self, backbone):
        self.dims = backbone.hidden_size
        return self

    def register(self, w, seq):
        pass

    def forward(self, w, x, backbone):
        hidden = backbone.call(w, x).last_hidden_state
        if self.mode == "cls":
            y = pooling.cls(hidden, self.dims)
        elif self.mode == "mean":
            y = pooling.masked_mean(hidden, x["attention_mask"], self.dims, self.prescale)
        else:
            y = pooling.last_token(hidden, x["attention_mask"], self.dims)
        return pooling.l2(y, self.prescale) if self.l2 else y

    def reference(self, outputs):
        """fp32 from an unpadded forward: every position is real."""
        h = outputs.last_hidden_state[0].double().numpy()
        v = {"cls": h[0], "mean": h.mean(axis=0), "last_token": h[-1]}[self.mode]
        return v / np.linalg.norm(v) if self.l2 else v

    def st_mode(self):
        """The sentence-transformers pooling mode this head reproduces."""
        return {"cls": "cls", "mean": "mean", "last_token": "lasttoken"}[self.mode]
