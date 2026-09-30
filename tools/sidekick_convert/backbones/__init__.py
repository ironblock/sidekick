"""Backbones: an architecture made convertible.

A backbone loads a Hugging Face checkpoint in fp32 and applies everything
its architecture needs to convert faithfully and run on the ANE: attention
implementation and finite masks, traceable helpers, precision and range
rewrites, position offsets. It knows how to call the model inside a wrapper
(`call`), which constant buffers the wrapper needs per bucket (`buffers`),
and how to run the unpadded fp32 reference (`reference`).

It knows nothing about the task: a head (sidekick_convert.heads) decides what
the artifact returns, and ports (sidekick_convert.core) what it takes. When
the head is the checkpoint's own task head (sequence classification), the
backbone loads the task class and patches its base model.

The contract, for a new family:
- `model`: the module the wrapper holds, under the attribute `attr` (its
  weights are named after that path in the converted program);
- `buffers(seq)`: ordered {name: tensor} registered on the wrapper;
- `call(wrapper, x)`: the traced forward; `x` maps port names to inputs;
- `reference(ids, token_type_ids=None)`: fp32, unpadded, the checkpoint's
  own outputs;
- `example(seq, ports)`: an int32 feed to trace with;
- `hidden_size`, `vocab_size`, `config`.
"""

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class Backbone:
    family: str
    model: torch.nn.Module
    config: object
    special_ids: tuple            # ids an empty input encodes to, e.g. [CLS] [SEP]
    attr: str = "model"

    @property
    def hidden_size(self):
        return int(self.config.hidden_size)

    @property
    def vocab_size(self):
        return int(self.config.vocab_size)

    def buffers(self, seq):
        return {"position_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0)}

    def call(self, w, x):
        raise NotImplementedError

    def reference(self, ids, token_type_ids=None):
        raise NotImplementedError

    def example(self, seq, ports):
        """Special ids at the start, the rest padding: an int32 feed to trace with."""
        feed = {}
        n = len(self.special_ids)
        for p in ports:
            a = np.zeros(p(seq), dtype=np.int32)
            if p.name == "input_ids":
                a[0, :n] = self.special_ids
            elif p.name == "attention_mask":
                a[0, :n] = 1
            feed[p.name] = a
        return feed
