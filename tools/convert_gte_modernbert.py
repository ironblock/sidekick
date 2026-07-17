"""Convert Alibaba-NLP/gte-modernbert-base to Core ML — a DOCUMENTED NEGATIVE
RESULT for the ANE (see OUTCOME below). Kept as the reproduction and as a
correct off-ANE / fp32 converter.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling baked into the graph, matching examples/manifests/gte-modernbert-base.

================================ OUTCOME ================================
ModernBERT does NOT achieve accurate full ANE offload, and this is intrinsic
to the architecture, not a conversion bug. Measured on M-series (macOS 26):

  compute path           worst parity vs fp32     latency
  torch fp16 (oracle)    0.999999                 —
  Core ML fp32           1.000000                 (off-ANE)
  Core ML fp16, full ANE 0.9038                   7.9ms   <- what the ANE gives
  Core ML fp16, any op fp32  ~0.9998              ~41ms   <- falls off the ANE

Root cause: ModernBERT has an outlier feature (dimension 251) that reaches
~40000 in the residual stream (layers.15.mlp), the well-known "massive
activation" phenomenon. That single dim dominates every LayerNorm's variance
(40000^2 ~ 1.6e9), so all other dims are divided by ~1400 and crushed toward
fp16's precision floor. PyTorch fp16 survives because it keeps LayerNorm and
softmax reductions in fp32; the ANE stores fp16 between every op, so the
crushed values cannot be recovered downstream. Forcing the sensitive ops to
fp32 restores parity but relocates the graph off the ANE (~5x slower, no ANE
benefit) — the newer macOS26 ANE compiler behaves identically. A global 1/K
range rewrite (the D17 gemma trick) does not help: cooling 40000 below the
fp16 square limit needs K>=156, at which point the compensated LayerNorm eps
(eps/K^2) underflows fp16 to zero. The degradation is not cosmetic: at 0.90
parity the embedding space compresses (an unrelated pair rose 0.38 -> 0.53),
which hurts retrieval discrimination.

Consequence: the entire ModernBERT embedding family (this model, granite-r2,
nomic-modernbert-embed) is documented ANE-incompatible in docs/MODELS.md. The
converter below is correct (fp32 parity 1.0) and is retained to reproduce the
finding and to build an accurate off-ANE/fp32 artifact if a host ever wants
one. It is NOT installed as a validated ANE model.
=========================================================================

Usage:
    python tools/convert_gte_modernbert.py <hf-model-dir> <install-dir> [buckets...]

    hf-model-dir: local snapshot of Alibaba-NLP/gte-modernbert-base
                  (config.json, tokenizer.json, model.safetensors)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/gte-modernbert-base"
    buckets:      default 128 256 512

Requires: torch, transformers >= 4.48 (native ModernBERT), coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

ModernBERT is the fourth architecture class validated on the stack (after
classic BERT / bge, Gemma3 / embeddinggemma, and the LFM2 hybrid). It is an
encoder-only bidirectional transformer with three features that make it a
NEW conversion path, all handled here without a full re-derivation:

A. ALTERNATING LOCAL/GLOBAL ATTENTION. Every `global_attn_every_n_layers`-th
   layer (here every 3rd) attends globally; the rest use a symmetric sliding
   window of half-width `local_attention // 2` (here 64). transformers builds
   two 4D additive masks in `_update_attention_mask` and the encoder layer
   picks one by layer type. Both masks use `finfo(dtype).min` for masked
   positions — which saturates to -inf in fp16 and NaNs softmax on the ANE
   (same failure class as D15's eager-attention rule). We re-patch
   `_update_attention_mask` to build the identical two masks with the
   fp16-safe MASK_ADD = -30000 additive constant. The band geometry
   (distance <= local_attention // 2) is copied verbatim.

B. RoPE with per-layer-type theta (global 160000 / local 10000). The theta
   split is internal to each attention module and needs nothing from us, but
   stock `rotate_half` slices with `x.shape[-1] // 2`, whose traced Int op
   crashes coremltools under static shapes (D17 constraint 8). Same
   chunk(2)-based replacement as the gemma/LFM recipes.

C. UNPADDING. ModernBERT unpads sequences only on the flash_attention_2
   path; sdpa keeps full static shapes, so loading with
   attn_implementation="sdpa" avoids the data-dependent shapes that would
   push the encoder off the ANE (D15 constraint 1). Explicit position_ids
   are passed for the same static-shape reason as bge (D15 constraint 4).

Pooling: raw CLS (position 0) reshaped to a literal (1, dims), exactly like
bge — no in-graph L2 normalize. The server normalizes pooled vectors in f32
(coreml_embedder.rs), so keeping the CLS unnormalized avoids the fp16
sum-of-squares overflow the L2 would hit (|CLS| ~= 22, 768 dims -> ~3.9e5).
Cosine parity is normalization-invariant, so the gate is unaffected.

fp16 note: no range rewrite. ModernBERT's residual stream is hot (~4e4 peak,
measured) but under the fp16 max (65504), and it uses LayerNorm (numerically
stable) rather than the scale-sensitive RMSNorm the gemma rewrite targeted.
The per-path parity gate is the real check; measured results in docs/MODELS.md.
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
import transformers.models.modernbert.modeling_modernbert as _mb

DIMS = 768
MASK_ADD = -30000.0  # fp16-safe additive mask constant (constraint A)

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # ~440 tokens: exercises the sliding-window band (live for distances > 64)
    # and long-sequence fp16 accumulation. Short sentences never reach the
    # band, so a wrong window would pass every short-text parity check.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]


def _traceable_rotate_half(x):
    # constraint B: identical to stock rotate_half for even head dims, but
    # chunk() keeps shape arithmetic out of the traced graph.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _fp16_safe_update_attention_mask(self, attention_mask, output_attentions=False):
    # constraint A: byte-for-byte the geometry transformers builds in
    # ModernBertModel._update_attention_mask, but with MASK_ADD instead of
    # finfo(dtype).min so masked logits stay fp16-representable. Returns
    # (global_mask, sliding_window_mask), both (bsz, 1, seq, seq) additive.
    seq = attention_mask.shape[-1]
    keypad = (1.0 - attention_mask.to(torch.float32))  # 1 at pad positions
    big = (keypad * MASK_ADD)[:, None, None, :]         # (bsz, 1, 1, seq)
    global_mask = big.expand(attention_mask.shape[0], 1, seq, seq).contiguous()
    rows = torch.arange(seq).unsqueeze(0)
    distance = torch.abs(rows - rows.T)
    window_bad = (distance > self.config.local_attention // 2)[None, None]
    sliding_mask = global_mask.masked_fill(window_bad, MASK_ADD)
    return global_mask, sliding_mask


def install_patches():
    _mb.rotate_half = _traceable_rotate_half
    _mb.ModernBertModel._update_attention_mask = _fp16_safe_update_attention_mask


class ClsWrapper(torch.nn.Module):
    """CLS (position 0) pooling, reshaped to a literal (1, dims). No in-graph
    L2 — the server normalizes in f32 (see module docstring)."""

    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.register_buffer(
            "position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=self.position_ids,
        ).last_hidden_state
        return hidden[:, 0, :].reshape(1, DIMS)


def reference_embeddings(model, tokenizer):
    """fp32 CLS references (unnormalized; cosine is scale-invariant), verified
    equal to SentenceTransformer.encode at cosine 0.99999994."""
    refs = []
    with torch.no_grad():
        for s in PARITY_SENTENCES:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=512)
            refs.append(model(**enc).last_hidden_state[0, 0].numpy())
    return refs


def padded_inputs(tokenizer, text, seq_len):
    ids_list = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids_list) > seq_len:
        raise SystemExit(f"parity text longer than bucket {seq_len}")
    ids = np.zeros((1, seq_len), dtype=np.int32)  # pad id 0, as the server pads
    ids[0, : len(ids_list)] = ids_list
    mask = np.zeros((1, seq_len), dtype=np.int32)
    mask[0, : len(ids_list)] = 1
    return ids, mask


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fitting_pairs(tokenizer, refs, seq_len):
    pairs = []
    for s, ref in zip(PARITY_SENTENCES, refs):
        n = len(tokenizer(s, add_special_tokens=True)["input_ids"])
        if n <= seq_len:
            pairs.append((s, ref))
    return pairs


def convert_bucket(wrapper, seq_len, workdir):
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0], ids[0, 1] = 50281, 50282  # [CLS] [SEP]
    mask = torch.zeros((1, seq_len), dtype=torch.int32)
    mask[0, :2] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs):
    """Cosine vs the fp32 reference on BOTH Espresso compute paths (D17
    constraint 9): CPU_ONLY validates the conversion, CPU_AND_NE validates
    ANE-precision execution."""
    results = {}
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.985),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        worst = 1.0
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, s, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            worst = min(worst, cosine(ref, out))
        if worst < gate:
            raise SystemExit(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        ids, mask = padded_inputs(tokenizer, PARITY_SENTENCES[0], seq_len)
        for _ in range(3):
            m.predict({"input_ids": ids, "attention_mask": mask})
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            m.predict({"input_ids": ids, "attention_mask": mask})
        results[label] = (worst, (time.perf_counter() - t0) / n * 1e3)
    return results


def compile_to_mlmodelc(pkg, install_dir, seq_len):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True)
        compiled = next(Path(tmp).glob("*.mlmodelc"))
        dest = install_dir / f"model_{seq_len}.mlmodelc"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
    return dest


def main():
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    install_patches()

    refs = reference_embeddings(model, tokenizer)

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = ClsWrapper(model, seq).eval()
            pkg = convert_bucket(wrapper, seq, workdir)
            res = parity_check(tokenizer, pkg, seq, refs)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            for label, (cos, ms) in res.items():
                print(f"bucket {seq} [{label}]: parity cos={cos:.6f} {ms:.1f}ms")
            print(f"bucket {seq} -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = (
        Path(__file__).resolve().parent.parent
        / "examples/manifests/gte-modernbert-base/manifest.toml"
    )
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
