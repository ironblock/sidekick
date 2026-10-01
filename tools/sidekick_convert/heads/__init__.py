"""Heads: what an artifact returns, independent of the architecture.

A head is a plain object with:
- `output`: the Core ML output name ("embedding", "logits", ...);
- `task`: which checkpoint class the backbone loads (None for the bare
  encoder; "sequence-classification" for the checkpoint's own head);
- `bind(backbone)`: read sizes (hidden size, label count) once loaded;
- `register(wrapper, seq)`: buffers or submodules the head needs per bucket;
- `forward(wrapper, x, backbone)`: the traced computation, calling
  `backbone.call(wrapper, x)`;
- `reference(outputs)`: the fp32 reference from the backbone's unpadded
  outputs, as a 1-D numpy vector.
"""
