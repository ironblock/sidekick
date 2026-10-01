"""compose(): backbone + head behind int32 ports, one static wrapper per bucket.

The wrapper holds the backbone's model under `backbone.attr` ("model" by
default), so converted weights are named after that path, registers the
backbone's and the head's per-bucket buffers, and traces
`head.forward(wrapper, inputs, backbone)`. Nothing in it reads a tensor's
size: every shape is a Python int fixed per bucket.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, backbone, head, ports, seq):
        super().__init__()
        setattr(self, backbone.attr, backbone.model)
        for name, tensor in backbone.buffers(seq).items():
            self.register_buffer(name, tensor)
        head.register(self, seq)
        self._names = tuple(p.name for p in ports)
        self._backbone = backbone
        self._head = head
        self.seq = seq

    def forward(self, *inputs):
        return self._head.forward(self, dict(zip(self._names, inputs)), self._backbone)


def compose(backbone, head, ports):
    """Returns (make_wrapper(seq), example(seq)) for a core.Job."""
    head.bind(backbone)

    def make_wrapper(seq):
        return Wrapper(backbone, head, ports, seq)

    def example(seq):
        return backbone.example(seq, ports)

    return make_wrapper, example
