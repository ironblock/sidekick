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
- `hidden_size`, `vocab_size`, `config`;
- `forbid_ops`: MIL ops the conversion must not contain, beyond the fused
  attention op (a rewrite that replaces gelu adds "gelu").

A backbone that can be split into chunks at layer boundaries (chunking.py,
docs/DECISIONS.md D37) adds:
- `chunk_ports`: the names of the ports the backbone itself reads;
- `chunk_layers()`: its layers, in order;
- `chunk_parts(lo, hi)`: {attribute: module} a chunk running layers
  [lo, hi) holds (plus the embedding when lo is 0, and the final norm
  when hi is the last layer);
- `chunk_call(chunk, x)`: the chunk's traced forward: the residual stream
  from the embedding (the first chunk) or from `x["hidden_in"]`, through
  the chunk's layers, normed by the last chunk.

Rewrites that change weights or modules (activation swaps, range and
precision rewrites) are functions of a backbone, applied by the recipe AFTER
the evaluation references are computed from the unmodified checkpoint, so
the fp32 gate proves each rewrite exact.
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
    forbid_ops: set = dataclasses.field(default_factory=set)

    @property
    def hidden_size(self):
        return int(self.config.hidden_size)

    @property
    def vocab_size(self):
        return int(self.config.vocab_size)

    def buffers(self, seq):
        return {"position_ids": torch.arange(seq, dtype=torch.long).unsqueeze(0)}

    chunk_ports = None   # set by a backbone that can be chunked

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
