"""sidekick_convert: Hugging Face checkpoints to ANE-resident Core ML artifacts.

A converter composes three orthogonal layers and hands the result to
core.run():
- a backbone (sidekick_convert.backbones): the architecture made
  convertible, with its attention, masks, positions and precision rewrites;
- a head (sidekick_convert.heads): what the artifact returns, a pooled
  embedding, the checkpoint's own classification logits, or one value per
  token;
- ports and a manifest (core.text_ports, sidekick_convert.manifest): the
  int32 static-shape interface and the committed manifest it must match.

core.run() converts one static-shape artifact per bucket and gates each one
(gates.py) before installing it. The techniques every backbone draws on live
in sidekick_convert.techniques. docs/CONVERTING.md is the catalog: modules,
gotchas, and how to add a family. fp16sim simulates an ideal fp16 engine,
the ceiling the parity suite grades real paths against.

The scripts in tools/ import this package directly (tools/ is on sys.path
when they run). Requires arm64-native Python with torch, transformers,
tokenizers, coremltools and numpy, plus Xcode for `xcrun coremlcompiler`.
"""
