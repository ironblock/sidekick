"""Build the tiny stateful decoder that sidekick-coreml's tests load
(crates/sidekick-coreml/tests/fixtures/tiny-kv-decoder/).

It has the `kv-state-1` program contract of docs/design/open-llm.md ("The
program contract") at toy sizes, so the tests exercise the runtime's Core
ML plumbing (state, the chunk chain, the T-token call, host assets) without
an ANE compile of a real model. The weights are seeded random; the model
says nothing about accuracy.

Core ML places a whole program on the CPU when it judges it too small for
the Neural Engine, and a `kv-state-1` decoder must run there (its state
crashes on the CPU). Measured on an M1 Max under macOS 27: a chunk of one
layer at width 256 and T = 64 went to the CPU, two layers or one with the
head went to the ANE; at width 64 nothing did, whatever T. So the sizes
below are the smallest found with a margin, every chunk has two layers,
and the build reads each chunk's compute plan and fails unless every
operation is on the ANE. The weights sit on a 16-level grid per tensor and
are stored as 4-bit palettes (lossless: the palette is the grid), which
keeps the fixture small without changing what the ANE computes.

A Llama-style decoder: RMSNorm, grouped-query attention with RoPE, a SiLU
MLP, 1x1 convolutions on a channels-first [1, D, 1, T] residual stream,
explicit attention with a finite additive mask, and the KV cache written
at `pos` with a slice update into Core ML state. Four layers in two
chunks (2 + 2); the last chunk holds the final norm and the head.

Files:
- `decoder.0.mlmodelc`, `decoder.1.mlmodelc`: the chunk programs. Inputs
  `hidden_in` [1, D, 1, T], `pos` [1] int32, `mask` [1, 1, T, C], `cos` and
  `sin` [T, head_dim], and (last chunk) `sel` [T]; outputs `hidden_out`
  [1, D, 1, T] or `logits` [1, V]; states `k_cache` and `v_cache`
  [layers, kv_heads, C, head_dim]. Everything fp16 but `pos`.
- `embedding.safetensors`: the host's token-embedding table, `embedding`
  [V, D] fp16.
- `rope.safetensors`: `cos` and `sin` [C, head_dim] fp16.
- `chat.toml`: the manifest's `[decoder]` table for the above.
- `expected.json` and `expected.safetensors`: fixed calls through the
  same T-token contract the runtime uses (a prompt in T-token calls, then
  one token per call, the state carried over), the fp32 torch model's
  logits for each (`logits` [calls, V] fp32), and its greedy
  continuation.

tiny-kv-decoder-cpu/ holds the same contract at width 64 (vocabulary 64,
T = 4, C = 32, one layer per chunk), which Core ML places on the CPU under
`cpu_and_ne`: the build checks it does. The runtime must refuse to load it.

Usage:
    python tools/make_decoder_test_model.py crates/sidekick-coreml/tests/fixtures

Requires torch, coremltools (8 or later, for state), numpy and safetensors
(arm64-native Python 3.12 or earlier), plus Xcode for compiling the
programs.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file

VOCAB = 1024
D = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
FFN = 512
T = 64
C = 256
LAYERS = [2, 2]
THETA = 10000.0
EPS = 1e-5
MASK = -30000.0
SEED = 0


class Norm(nn.Module):
    """RMSNorm over the channels of [1, D, 1, S]."""

    def __init__(self, g):
        super().__init__()
        self.register_buffer("g", g.view(1, -1, 1, 1))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + EPS) * self.g


def conv(gen, cin, cout, scale=1.0):
    """A 1x1 convolution whose weights take 16 values: a symmetric 4-bit
    grid, which the program stores exactly as a palette."""
    c = nn.Conv2d(cin, cout, 1, bias=False)
    w = torch.randn(cout, cin, 1, 1, generator=gen) * scale / math.sqrt(cin)
    step = float(w.abs().max()) / 7
    with torch.no_grad():
        c.weight.copy_(torch.clamp(torch.round(w / step), -8, 7) * step)
    c.requires_grad_(False)
    return c


def rot(x):
    h = HEAD_DIM // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Layer(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.n1 = Norm(1.0 + 0.1 * torch.randn(D, generator=gen))
        self.n2 = Norm(1.0 + 0.1 * torch.randn(D, generator=gen))
        self.q = conv(gen, D, HEADS * HEAD_DIM)
        self.k = conv(gen, D, KV_HEADS * HEAD_DIM)
        self.v = conv(gen, D, KV_HEADS * HEAD_DIM)
        self.o = conv(gen, HEADS * HEAD_DIM, D, 0.5)
        self.gate = conv(gen, D, FFN)
        self.up = conv(gen, D, FFN)
        self.down = conv(gen, FFN, D, 0.5)

    def forward(self, h, li, kc, vc, pos, mask, cos, sin, S):
        rep = HEADS // KV_HEADS
        x = self.n1(h)
        q = self.q(x).view(HEADS, HEAD_DIM, S).transpose(1, 2)
        k = self.k(x).view(KV_HEADS, HEAD_DIM, S).transpose(1, 2)
        v = self.v(x).view(KV_HEADS, HEAD_DIM, S).transpose(1, 2)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        kc[li:li + 1, :, pos:pos + S, :] = k.unsqueeze(0).to(kc.dtype)
        vc[li:li + 1, :, pos:pos + S, :] = v.unsqueeze(0).to(vc.dtype)
        K = kc[li].to(q.dtype)
        V = vc[li].to(q.dtype)
        qg = (q / math.sqrt(HEAD_DIM)).reshape(KV_HEADS, rep * S, HEAD_DIM)
        s = torch.matmul(qg, K.transpose(1, 2)).view(KV_HEADS, rep, S, C) + mask.view(1, 1, S, C)
        p = torch.softmax(s, dim=-1).view(KV_HEADS, rep * S, C)
        o = torch.matmul(p, V).view(HEADS, S, HEAD_DIM).transpose(1, 2).reshape(1, HEADS * HEAD_DIM, 1, S)
        h = h + self.o(o)
        x = self.n2(h)
        return h + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class Chunk(nn.Module):
    """A run of layers owning their KV cache; the last also has the head."""

    def __init__(self, layers, head=None, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.head, self.norm = head, norm
        self.register_buffer("k_cache", torch.zeros(len(layers), KV_HEADS, C, HEAD_DIM, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(len(layers), KV_HEADS, C, HEAD_DIM, dtype=torch.float16))

    def forward(self, hidden_in, pos, mask, cos, sin, sel=None):
        S = T
        h = hidden_in.to(torch.float32)
        mask, cos, sin = mask.to(torch.float32), cos.to(torch.float32), sin.to(torch.float32)
        for i, layer in enumerate(self.layers):
            h = layer(h, i, self.k_cache, self.v_cache, pos, mask, cos, sin, S)
        if self.head is None:
            return h
        h = torch.matmul(h.view(D, S), sel.to(torch.float32).view(S, 1)).view(1, D, 1, 1)
        return self.head(self.norm(h)).view(1, VOCAB)


def rope_tables():
    inv = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    ang = np.arange(C, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], -1)
    return np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)


def mask_for(pos):
    q = pos + np.arange(T)[:, None]
    k = np.arange(C)[None, :]
    return np.where(k <= q, 0.0, MASK).astype(np.float16).reshape(1, 1, T, C)


class Reference:
    """The fp32 torch model run through the contract, as the runtime runs
    the programs: fp16 inputs, one T-token call at a time."""

    def __init__(self, chunks, embed, cos, sin):
        self.chunks, self.embed, self.cos, self.sin = chunks, embed, cos, sin

    def reset(self):
        for c in self.chunks:
            c.k_cache.zero_()
            c.v_cache.zero_()

    def call(self, tokens, pos, sel):
        ids = list(tokens) + [0] * (T - len(tokens))
        h = torch.from_numpy(self.embed[ids].T.reshape(1, D, 1, T).astype(np.float32))
        args = (torch.tensor([pos], dtype=torch.int32), torch.from_numpy(mask_for(pos).astype(np.float32)),
                torch.from_numpy(self.cos[pos:pos + T].astype(np.float32)),
                torch.from_numpy(self.sin[pos:pos + T].astype(np.float32)))
        onehot = torch.zeros(T)
        onehot[sel] = 1.0
        with torch.no_grad():
            for c in self.chunks[:-1]:
                h = c(h, *args)
            return self.chunks[-1](h, *args, onehot)[0].numpy().astype(np.float64)


def build_chunks():
    gen = torch.Generator().manual_seed(SEED)
    embed = (torch.randn(VOCAB, D, generator=gen)).numpy().astype(np.float16)
    layers = [Layer(gen) for _ in range(sum(LAYERS))]
    norm = Norm(1.0 + 0.1 * torch.randn(D, generator=gen))
    head = conv(gen, D, VOCAB, 4.0)
    chunks, lo = [], 0
    for i, n in enumerate(LAYERS):
        last = i == len(LAYERS) - 1
        chunks.append(Chunk(layers[lo:lo + n], head if last else None, norm if last else None).eval())
        lo += n
    return chunks, embed


def convert(chunk, last, out, on_ane=True):
    ex = [torch.zeros(1, D, 1, T), torch.tensor([0], dtype=torch.int32), torch.from_numpy(mask_for(0)).float(),
          torch.zeros(T, HEAD_DIM), torch.zeros(T, HEAD_DIM)]
    inputs = [ct.TensorType("hidden_in", (1, D, 1, T), np.float16), ct.TensorType("pos", (1,), np.int32),
              ct.TensorType("mask", (1, 1, T, C), np.float16), ct.TensorType("cos", (T, HEAD_DIM), np.float16),
              ct.TensorType("sin", (T, HEAD_DIM), np.float16)]
    if last:
        ex.append(torch.zeros(T))
        inputs.append(ct.TensorType("sel", (T,), np.float16))
    traced = torch.jit.trace(chunk, tuple(ex), check_trace=False)
    shape = tuple(chunk.k_cache.shape)
    ml = ct.convert(
        traced, inputs=inputs,
        outputs=[ct.TensorType(name="logits" if last else "hidden_out", dtype=np.float16)],
        states=[ct.StateType(ct.TensorType(shape=shape, dtype=np.float16), name="k_cache"),
                ct.StateType(ct.TensorType(shape=shape, dtype=np.float16), name="v_cache")],
        minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    palettes = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(mode="unique", weight_threshold=1024))
    ml = ct.optimize.coreml.palettize_weights(ml, palettes)
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "chunk.mlpackage"
        ml.save(str(pkg))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(ct.utils.compile_model(str(pkg)), str(out))
    # The compiler's analytics are build-machine telemetry, not part of the program.
    shutil.rmtree(out / "analytics", ignore_errors=True)
    check_placement(out, on_ane)


def check_placement(path, on_ane):
    """Fail unless Core ML's compute plan under CPU_AND_NE puts every
    operation of `path` on the Neural Engine (`on_ane`), or none of them."""
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    ane, off = 0, {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        if isinstance(usage.preferred_compute_device, ct.models.compute_device.MLNeuralEngineComputeDevice):
            ane += 1
        else:
            off[op.operator_name] = off.get(op.operator_name, 0) + 1
    if on_ane and off:
        sys.exit(f"{path.name}: operations off the Neural Engine: {off}")
    if not on_ane and ane:
        sys.exit(f"{path.name}: {ane} operations on the Neural Engine; this fixture must have none")


def manifest(model_id):
    return f"""\
id = "{model_id}"
task = "text-generation"
interface = "kv-state-1"
compute_units = "cpu_and_ne"
tokenizer = "tokenizer.json"

[decoder]
artifact = "decoder.{{chunk}}.mlmodelc"
chunks = {len(LAYERS)}
tokens_per_call = {T}
cache_capacity = {C}
context_length = {C - T + 1}
hidden_size = {D}
head_dim = {HEAD_DIM}
kv_heads = {KV_HEADS}
layers = {LAYERS}
vocab_size = {VOCAB}

[decoder.host]
embedding = "embedding.safetensors"
rope = "rope.safetensors"
"""


def main():
    fixtures = Path(sys.argv[1])
    root = fixtures / "tiny-kv-decoder"
    chunks, embed = write_model(root, "tiny-kv-decoder")
    cos, sin = rope_tables()

    ref = Reference(chunks, embed, cos, sin)
    # A prompt in T-token calls (the last one padded), then greedy decoding
    # one token per call; the state carries over from call to call.
    rng = np.random.default_rng(SEED)
    prompt = [int(x) for x in rng.integers(1, VOCAB, size=100)]
    ref.reset()
    calls, rows = [], []
    pos = 0
    while pos < len(prompt):
        piece = prompt[pos:pos + T]
        logits = ref.call(piece, pos, len(piece) - 1)
        calls.append({"tokens": piece, "pos": pos, "sel": len(piece) - 1})
        rows.append(logits)
        pos += len(piece)
    greedy = []
    while len(greedy) < 24:
        tok = int(np.argmax(rows[-1]))
        greedy.append(tok)
        rows.append(ref.call([tok], pos, 0))
        calls.append({"tokens": [tok], "pos": pos, "sel": 0})
        pos += 1
    expected = {"prompt": prompt, "greedy": greedy, "calls": calls}
    (root / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    save_file({"logits": np.stack(rows).astype(np.float32)}, str(root / "expected.safetensors"))

    for i, c in enumerate(chunks):
        c.k_cache.zero_()
        c.v_cache.zero_()
        convert(c, i == len(chunks) - 1, root / f"decoder.{i}.mlmodelc")
    print(f"wrote {root}: {len(calls)} reference calls, greedy {greedy}")

    # The same contract at width 64, which Core ML runs on the CPU.
    globals().update(VOCAB=64, D=64, HEAD_DIM=16, FFN=128, T=4, C=32, LAYERS=[1, 1])
    small = fixtures / "tiny-kv-decoder-cpu"
    chunks, _ = write_model(small, "tiny-kv-decoder-cpu")
    for i, c in enumerate(chunks):
        convert(c, i == len(chunks) - 1, small / f"decoder.{i}.mlmodelc", on_ane=False)
    print(f"wrote {small}")


def write_model(root, model_id):
    """The seeded model at the current sizes, with its host tables and
    manifest written to `root`."""
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    chunks, embed = build_chunks()
    cos, sin = rope_tables()
    save_file({"embedding": embed}, str(root / "embedding.safetensors"))
    save_file({"cos": cos, "sin": sin}, str(root / "rope.safetensors"))
    (root / "chat.toml").write_text(manifest(model_id))
    return chunks, embed


if __name__ == "__main__":
    main()
