"""Convert Alibaba-NLP/gte-modernbert-base into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling baked into the graph, matching examples/manifests/gte-modernbert-base.

============================== KEY CONSTRAINT ==============================
EXPLICIT ATTENTION, NOT THE FUSED SDPA OP. Load the model with
attn_implementation="eager", so attention converts to explicit
matmul -> softmax -> matmul. With "sdpa" it converts to Core ML's fused
scaled_dot_product_attention op, and in this graph that op DROPS ITS MASK on
the Neural Engine (macOS 27, M1 Max): pad tokens are attended and the
sliding window disappears. The ANE output then matches an unmasked fp32
reference at 0.99998 and the intended one at only 0.87-0.975; it changes
with the *content* of the pad positions (pad ids 0 vs random: cosine
0.61-0.94); and the same fused op on the CPU returns NaN whenever fewer than
64 of 128 positions are real (a query whose whole sliding window is masked).
The trigger: transformers builds ModernBERT's masks before the CPU-only
embedding gather, so they are computed on the CPU and reach the ANE's
attention as an input, and the ANE's fused attention ignores a mask it
doesn't compute itself. tools/repro_sdpa_mask.py reproduces this standalone
and checks any compiled model for it (D25).

With explicit attention the whole graph stays on the ANE and parity is
0.9999 at every bucket (docs/MODELS.md). This file used to document
ModernBERT as ANE-incompatible, blaming its massive activation (dim 251,
~48,000 in the residual stream) crushing LayerNorm precision. That was the
mask bug misdiagnosed: the same outlier is harmless once attention is
explicit.

parity_check() therefore also gates PAD INVARIANCE: the same text with
different pad ids must give the same output (cosine >= 0.99999). A correctly
masked model can't see its pads, so this check catches a dropped mask in
seconds.
============================================================================

Usage:
    python tools/convert_gte_modernbert.py <hf-model-dir> <install-dir> [buckets...]
    python tools/convert_gte_modernbert.py --attn sdpa <hf-model-dir> <install-dir> [buckets...]

    --attn sdpa:  NEGATIVE CONTROL. Convert with the fused attention op that
                  drops the mask on the ANE, to prove the parity suite and
                  ane_check catch it. Parity failures are reported, not
                  fatal. Never install the result where the daemon looks.

    hf-model-dir: local snapshot of Alibaba-NLP/gte-modernbert-base
                  (config.json, tokenizer.json, model.safetensors)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/gte-modernbert-base"
    buckets:      default 128 256 512

Requires: torch, transformers >= 4.48 (native ModernBERT), coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

ModernBERT is an encoder-only bidirectional transformer with three features
that make it a new conversion path, all handled here without a full
re-derivation:

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
   path; eager (and sdpa) keep full static shapes, which avoids the
   data-dependent shapes that would push the encoder off the ANE (D15
   constraint 1). Explicit position_ids are passed for the same static-shape
   reason as bge (D15 constraint 4). Use eager: see KEY CONSTRAINT above.

Pooling: raw CLS (position 0) reshaped to a literal (1, dims), exactly like
bge — no in-graph L2 normalize. The server normalizes pooled vectors in f32
(coreml_embedder.rs), so keeping the CLS unnormalized avoids the fp16
sum-of-squares overflow the L2 would hit (|CLS| ~= 22, 768 dims -> ~3.9e5).
Cosine parity is normalization-invariant, so the gate is unaffected.

D. RESIDUAL RANGE REWRITE (docs/DECISIONS.md D25 amendment). The ANE's
   linear op saturates above 2^15 = 32,768, half of fp16's max: an output of
   33,000 comes back inf, while its add, mul and layer_norm handle the full
   fp16 range. ModernBERT's massive activation (dimension 251 on delimiter
   tokens, ~48,000 in the residual) is written by layer 15's MLP output
   projection, at 35,000-51,500 on every input tried. On the ANE the full
   graph then carries -inf in that token's residual. Downstream saturation
   keeps the CLS output finite, but at 0.9994 instead of 0.9999.
   Fix, exact in fp32: run the residual stream at 1/K. The embedding norm's
   weight takes 1/K, layer 0's Wqkv (which reads the embedding directly)
   takes K, both output projections of every layer take 1/K, and every
   LayerNorm (scale-invariant) gets eps / K^2. K is the smallest power of
   two keeping every calibrated linear output under LINEAR_HEADROOM x 2^15.
   That is K = 2 for this checkpoint, with ~1.3x headroom over its largest
   calibrated output. The converter prints the headroom, and fails if no
   K <= 8 fits. Larger K costs precision on both the ANE and the CPU path
   (K = 4: ANE 0.99984, CPU 0.99886), because it shrinks everything else,
   so K is kept minimal. The rescale of small linear inputs from D19 and an
   explicit GELU didn't help this model.

fp16 range: ModernBERT's residual stream peaks at ~48,000 (measured with
tools/probe_activations.py), under the fp16 max (65504), so it needs no
fp16 range rewrite. Constraint D is about the ANE linear's narrower range,
not fp16's.

Gates, per bucket: fp32 exactness of constraint D (>= 0.99999) before
converting; then CPU_ONLY >= 0.999 and CPU_AND_NE >= 0.999, finite output,
and pad invariance. Every gate treats NaN as a failure, and convert_bucket()
rejects the fused attention op outside the --attn sdpa negative control.
Measured results in docs/MODELS.md.
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
ANE_LINEAR_MAX = 32768.0  # the ANE's linear op saturates above 2^15 (constraint D)
LINEAR_HEADROOM = 0.85    # keep calibrated linear outputs under this share of it
K_MAX = 8

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # 442 tokens: exercises the sliding-window band (live for distances > 64)
    # and long-sequence fp16 accumulation. Short sentences never reach the
    # band, so a wrong window would pass every short-text parity check.
    # main() fails if it stops fitting the 512 bucket.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]

# Varied text for constraint D's linear-output maxima. The massive
# activation sits on delimiter tokens and grows with sequence length, so
# include long, punctuated and list-like inputs.
CALIBRATION_TEXTS = PARITY_SENTENCES + [
    "def add(a, b):\n    return a + b  # simple helper\n",
    "Order #48213 shipped 2026-09-14; see https://example.com/track?id=48213&ref=a1b2.",
    "Wait... what?! (No, really — \"that\" isn't it.) [1] {2} <3>",
    "- one\n- two\n- three\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    "Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。",
    " ".join(["buffalo"] * 60),
    "3.14159 2.71828 1.41421 6.02214076e23 299792458",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
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
    """Cosine similarity, NaN when either side is non-finite."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def worst_of(values):
    """min() that keeps NaN: min(worst, nan) returns worst and hides a NaN output."""
    values = list(values)
    return float("nan") if any(np.isnan(v) for v in values) else min(values)


def linear_output_maxima(model, tokenizer):
    """fp32 max |output| of every linear in the encoder, over CALIBRATION_TEXTS
    (unpadded forwards). Keys are (layer, name); constraint D uses them."""
    maxima = {}

    def hook(key):
        def f(mod, args, out):
            maxima[key] = max(maxima.get(key, 0.0), float(out.detach().abs().max()))
        return f

    hooks = []
    for i, layer in enumerate(model.layers):
        for name, lin in (("Wqkv", layer.attn.Wqkv), ("attn.Wo", layer.attn.Wo),
                          ("Wi", layer.mlp.Wi), ("mlp.Wo", layer.mlp.Wo)):
            hooks.append(lin.register_forward_hook(hook((i, name))))
    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            model(**tokenizer(text, return_tensors="pt", truncation=True, max_length=512))
    for h in hooks:
        h.remove()
    return maxima


def choose_k(maxima):
    """Constraint D: the smallest power of two K <= K_MAX keeping every linear
    output under LINEAR_HEADROOM x ANE_LINEAR_MAX. Only the output projections
    (attn.Wo, mlp.Wo) scale with 1/K; Wqkv and Wi read scale-invariant norms."""
    limit = LINEAR_HEADROOM * ANE_LINEAR_MAX
    fixed = max(v for (i, n), v in maxima.items() if not n.endswith("Wo"))
    scaled = max(v for (i, n), v in maxima.items() if n.endswith("Wo"))
    if fixed > limit:
        raise SystemExit(f"a Wqkv/Wi output reaches {fixed:.0f}, past the ANE linear's range")
    k = 1
    while scaled / k > limit:
        k *= 2
        if k > K_MAX:
            raise SystemExit(f"output projections reach {scaled:.0f}; no K <= {K_MAX} fits")
    return k, ANE_LINEAR_MAX / max(fixed, scaled / k)


def range_rewrite(model, k):
    """Constraint D: the residual stream at 1/k, exact in fp32."""
    if k == 1:
        return
    with torch.no_grad():
        def scale(module, factor):
            module.weight.mul_(factor)
            if getattr(module, "bias", None) is not None:
                module.bias.mul_(factor)

        scale(model.embeddings.norm, 1.0 / k)
        for layer in model.layers:
            if isinstance(layer.attn_norm, torch.nn.Identity):
                layer.attn.Wqkv.weight.mul_(k)   # layer 0 reads the embedding directly
            else:
                layer.attn_norm.eps /= k * k
            layer.mlp_norm.eps /= k * k
            scale(layer.attn.Wo, 1.0 / k)
            scale(layer.mlp.Wo, 1.0 / k)
        model.final_norm.eps /= k * k


def fp32_gate(wrapper, tokenizer, refs, seq_len):
    """Constraint D must be ~exact in fp32 before we spend on conversion."""
    cosines = []
    with torch.no_grad():
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, s, seq_len)
            out = wrapper(torch.from_numpy(ids), torch.from_numpy(mask))[0].numpy()
            cosines.append(cosine(ref, out))
    worst = worst_of(cosines)
    if not worst >= 0.99999:  # NaN fails too
        raise SystemExit(f"seq {seq_len}: fp32 rewrite parity {worst:.7f} < 0.99999")
    return worst


def fitting_pairs(tokenizer, refs, seq_len):
    pairs = []
    for s, ref in zip(PARITY_SENTENCES, refs):
        n = len(tokenizer(s, add_special_tokens=True)["input_ids"])
        if n <= seq_len:
            pairs.append((s, ref))
    return pairs


def convert_bucket(wrapper, seq_len, workdir, allow_fused=False):
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
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    if "scaled_dot_product_attention" in ops and not allow_fused:
        raise SystemExit(f"seq {seq_len}: converted graph contains the fused attention "
                         "op — see the KEY CONSTRAINT")
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs, gate_failures=True):
    """Cosine vs the fp32 reference on BOTH Espresso compute paths (D17
    constraint 9): CPU_ONLY validates the conversion, CPU_AND_NE validates
    ANE-precision execution. A negative control reports failures instead of
    stopping on them."""
    def fail(message):
        if gate_failures:
            raise SystemExit(message)
        print(f"negative control, expected: {message}")

    results = {}
    # CPU_ONLY >= 0.999: the conversion is faithful; CPU_AND_NE >= 0.999: the
    # ANE runs it at full precision (constraint D)
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.999),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        cosines = []
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, s, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                fail(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            cosines.append(cosine(ref, out))
        worst = worst_of(cosines)
        if not worst >= gate:  # NaN fails too
            fail(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        # Pad invariance: a correctly masked model can't see its pad
        # positions, so their content must not change the output.
        ids, mask = padded_inputs(tokenizer, PARITY_SENTENCES[0], seq_len)
        noisy = ids.copy()
        pads = mask[0] == 0
        noisy[0, pads] = np.random.default_rng(0).integers(1000, 40000, int(pads.sum()))
        a = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
        b = m.predict({"input_ids": noisy, "attention_mask": mask})["embedding"][0]
        if not cosine(a, b) >= 0.99999:
            fail(f"seq {seq_len} [{label}]: output depends on pad content "
                 f"(cos {cosine(a, b):.6f}); the attention mask is being dropped")
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
    args = sys.argv[1:]
    attn = "eager"
    if args[:1] == ["--attn"]:
        attn, args = args[1], args[2:]
        if attn not in ("eager", "sdpa"):
            raise SystemExit("--attn takes eager or sdpa")
    src = Path(args[0]).expanduser()
    install_dir = Path(args[1]).expanduser()
    buckets = [int(b) for b in args[2:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)
    negative_control = attn != "eager"
    if negative_control:
        print("NEGATIVE CONTROL: fused attention, which drops the mask on the ANE")

    tokenizer = AutoTokenizer.from_pretrained(src)
    # eager, not sdpa: see KEY CONSTRAINT in the module docstring.
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation=attn)
    model.eval()
    install_patches()
    # the long parity text must run in the 512 bucket (see PARITY_SENTENCES)
    longest = max(len(tokenizer(s, add_special_tokens=True)["input_ids"]) for s in PARITY_SENTENCES)
    if not 128 < longest <= 512:
        raise SystemExit(f"the long parity text is {longest} tokens; it must fit the 512 bucket")

    refs = reference_embeddings(model, tokenizer)
    print("calibrating linear output ranges...")
    k, headroom = choose_k(linear_output_maxima(model, tokenizer))
    range_rewrite(model, k)
    print(f"residual scale K={k}; largest calibrated linear output is {headroom:.2f}x "
          f"under the ANE linear's {ANE_LINEAR_MAX:.0f}")

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = ClsWrapper(model, seq).eval()
            print(f"bucket {seq}: fp32 rewrite parity {fp32_gate(wrapper, tokenizer, refs, seq):.7f}")
            pkg = convert_bucket(wrapper, seq, workdir, allow_fused=negative_control)
            res = parity_check(tokenizer, pkg, seq, refs, gate_failures=not negative_control)
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
