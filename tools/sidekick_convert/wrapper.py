"""compose(): backbone + head behind int32 ports, one static wrapper per bucket.

The wrapper holds the backbone's model under `backbone.attr` ("model" by
default), so converted weights are named after that path, registers the
backbone's and the head's per-bucket buffers, and traces
`head.forward(wrapper, inputs, backbone)`. Nothing in it reads a tensor's
size: every shape is a Python int fixed per bucket.

Its forward takes the ports as named parameters (input_ids, attention_mask,
...), as a hand-written wrapper would. jit.trace names graph inputs after
them, and later values after the Python locals they are bound to; with
unnamed inputs, the numbering of every value derived from them shifts, and
the converted program's variable names with it.
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

    def _forward(self, *inputs):
        return self._head.forward(self, dict(zip(self._names, inputs)), self._backbone)


_CLASSES = {}


def _wrapper_class(names):
    """A Wrapper subclass whose forward takes `names` as its parameters."""
    if names not in _CLASSES:
        if not all(n.isidentifier() for n in names):
            raise ValueError(f"port names must be identifiers: {names}")
        params = ", ".join(names)
        namespace = {}
        exec(f"def forward(self, {params}):\n    return self._forward({params})\n", namespace)
        _CLASSES[names] = type("Wrapper", (Wrapper,), {"forward": namespace["forward"]})
    return _CLASSES[names]


def compose(backbone, head, ports):
    """Returns (make_wrapper(seq), example(seq)) for a core.Job."""
    head.bind(backbone)
    cls = _wrapper_class(tuple(p.name for p in ports))

    def make_wrapper(seq):
        return cls(backbone, head, ports, seq)

    def example(seq):
        return backbone.example(seq, ports)

    return make_wrapper, example
