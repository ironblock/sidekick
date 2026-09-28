"""Convert google/embeddinggemma-300m into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket with the FULL
sentence-transformers stack baked into the graph: Gemma3 encoder -> mask-aware
mean pooling -> Dense 768->3072 -> Dense 3072->768 -> L2 normalize, output
`embeddings`, statically (1, 768). Matches examples/manifests/embeddinggemma-300m.

Usage:
    python tools/convert_embeddinggemma.py <hf-model-dir> <install-dir> [buckets...]

    hf-model-dir: local snapshot of google/embeddinggemma-300m (the full
                  sentence-transformers repo: config.json, model.safetensors,
                  tokenizer.json, 2_Dense/, 3_Dense/)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/embeddinggemma-300m"
    buckets:      default 128 256 512

Requires: torch, transformers>=4.57 (gemma3_text), coremltools, numpy,
safetensors (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

This inherits the four hardware-verified constraints of the bge-small recipe
(see tools/convert_bge_small.py and docs/DECISIONS.md D15): static shapes per
bucket, pooling inside the graph with a literal (1, dims) reshape, SDPA
attention, explicit position_ids. Gemma's SDPA call passes an explicit
scale, so coremltools lowers it to explicit matmul -> softmax -> matmul
rather than its fused scaled_dot_product_attention op (which dropped
ModernBERT's mask on the ANE, docs/DECISIONS.md D25); convert_bucket()
fails if the fused op ever appears. EmbeddingGemma adds NEW constraints,
verified on hardware (macOS 26 and 27, M-series):

5. fp16 RANGE REWRITE of the residual stream. Gemma3 scales embeddings by
   sqrt(hidden)=27.7 and its residual stream grows to ~1.5e5 by layer 24 —
   past fp16 max (65504), so a straight conversion (ANE is fp16-only)
   produces Inf/NaN or garbage. Worse, RMSNorm materializes x^2, which
   overflows fp16 for |x| > 255. Fix, exact in exact arithmetic because
   RMSNorm is scale-invariant and every factor is a power of two:
     - scale the embedding output and each layer's two residual-branch
       outputs (post_attention/post_feedforward norms) by 1/K (K auto-chosen,
       32 for this checkpoint) so the residual stream stays representable;
     - every RMSNorm gets a power-of-two input pre-scale s (calibrated per
       norm from fp32 activation stats so mean(y^2) ~= 1) with eps
       compensated exactly as eps*s^2, so x*rsqrt(mean(x^2)+eps) is
       reproduced with all fp16 intermediates in range;
     - eps floored at 1e-4 (fp16-representable; raw 1e-6 is fp16-subnormal
       and flushes to zero on the ANE -> rsqrt(0)=Inf -> NaN). The floor
       perturbs the worst-case token norm by <1e-3 relative — measured
       end-to-end parity stays >= 0.999.
   The final model.norm is scale-invariant, so `last_hidden_state` and the
   pooled embedding are unchanged.
6. Attention masks are built IN the wrapper and passed as the prepared-mask
   dict {"full_attention", "sliding_attention"}, bypassing transformers
   masking_utils: additive fp16-safe -30000 on padded keys (-30000, not
   torch.finfo.min: fp32 min becomes fp16 -inf and risks NaN through
   softmax). The sliding mask is a real precomputed band: transformers
   HALVES the checkpoint's sliding_window for bidirectional models
   (config.json says 512; Gemma3TextConfig makes it 512//2+1 = 257) and
   sliding layers attend iff abs(q-k) < 257
   (`_bidirectional_window_overlay` in modeling_gemma3.py). At bucket 512
   the band is live — positions >256 apart don't attend — so the parity
   gates include a 394-token text, and main() fails if no parity text
   reaches the band at a bucket where it is live; short sentences alone
   would pass even with a wrong mask. (An earlier version's "~400-token"
   text was really 527 tokens, so no bucket ever ran it.)
7. Mean pooling and L2 normalization overflow fp16 too: channel sums over
   512 tokens of |h|<=~140 hidden states, and sum(y^2) of the ~1e3-norm
   Dense output, both exceed 65504. The pooling sum runs at 1/32 scale and
   is divided by count/32, so the dense stack sees the mean at natural
   scale (its linears need O(1) inputs, constraint 9); the Dense output is
   scaled by 1/32 before the L2 sum of squares, which the normalize cancels.
8. rotate_half and repeat_kv are monkeypatched to shape-arithmetic-free
   equivalents before tracing (chunk(2) instead of `x[..., : shape//2]`
   slices; expand(-1,..)+flatten instead of reshape(kv_heads * n_rep)).
   The stock versions trace to floor_divide/mul -> Int -> slice/reshape
   chains (144 + 48 sites), and coremltools 9.x's 'int' op handler
   crashes on them under static input shapes ("only 0-dimensional arrays
   can be converted to Python scalars"). Both rewrites are exact.
9. MLP PRECISION REWRITE. Two ANE arithmetic limits that fp16 itself
   doesn't have (the same graph in fp16 on the GPU, or in PyTorch, is
   exact to ~1e-6), measured on macOS 27, M1 Max (docs/DECISIONS.md D17):
     - The ANE's linear op has an ABSOLUTE precision floor on its input:
       its relative error is ~3e-4 / rms(input) (0.04% at rms 1, 2% at
       rms 0.016, 10% at rms 0.004). Gemma's down_proj input,
       gelu(gate) * up, has rms 0.004-0.03 in layers 18-23, so each late
       MLP lost 5-19% of its output. Fix: fold a calibrated power-of-two
       scale m into up_proj (m*GAIN brings the down_proj input to rms ~1;
       the totals span 2-256 across layers) and divide it out in
       post_feedforward_layernorm's pre-scale, which is exact because that
       norm is scale-invariant (the constraint-5 machinery).
     - Core ML's gelu op is coarse on the ANE: ~6e-3 absolute error on
       [-1, 1], where most of Gemma's gate activations lie (GPU: ~7e-5).
       TanhGelu computes it from tanh, mul and add instead (<= 9e-4 on
       [-1, 1], <= 2e-4 near 0); convert_bucket() fails if a gelu op
       survives conversion.
   Before this rewrite the ANE parity was 0.9905 (0.981 on the 394-token
   text at bucket 512) and was documented as intrinsic fp16 accumulation.
   After it, the ANE is more accurate than CPU_ONLY. The rescale alone
   gets to 0.9997 (0.9992 at bucket 512) at no latency cost; the explicit
   gelu gets the rest and costs ~14% at bucket 512 (none at 128). Hence
   the gates: CPU_ONLY >= 0.999 proves the conversion is faithful,
   CPU_AND_NE >= 0.999 that the ANE runs it at full precision.

The parity gate runs the converted artifact under BOTH CPU_AND_NE and
CPU_ONLY (the Espresso paths that reject what .all/GPU tolerates) on real
tokenized sentences and requires cosine >= 0.999 against a float32 reference
computed with the exact sentence-transformers math, finite outputs, and pad
invariance (pad ids 0 vs random must give the same output, D25). A fp32
torch gate (>= 0.9999) validates the range and MLP rewrites before
conversion. Every gate treats NaN as a failure.

Tokenizer note: the snapshot's tokenizer adds BOS(2) ... EOS(1) around the
text (add_special_tokens=True), pad id is 0, and EmbeddingGemma requires task
prefixes ("title: none | text: " for documents) — the server applies
prefixes per manifest [prefixes] and its HF-tokenizers encode(text, true)
matches this recipe's token stream.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
import transformers.models.gemma3.modeling_gemma3 as _gemma3_modeling
import transformers.integrations.sdpa_attention as _sdpa_attention


def _traceable_rotate_half(x):
    # constraint 8: identical to the stock rotate_half for even head dims,
    # but chunk() keeps shape arithmetic out of the traced graph.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _traceable_repeat_kv(hidden_states, n_rep):
    # constraint 8: the stock repeat_kv reshapes with num_kv_heads * n_rep
    # computed from tensor sizes, another Int-op crash. expand(-1, ...) +
    # flatten needs no shape arithmetic and produces the identical layout.
    if n_rep == 1:
        return hidden_states
    return hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)


_gemma3_modeling.rotate_half = _traceable_rotate_half
_gemma3_modeling.repeat_kv = _traceable_repeat_kv
_sdpa_attention.repeat_kv = _traceable_repeat_kv

DIMS = 768
EPS_FLOOR = 1e-4          # smallest fp16-safe rmsnorm eps (1e-6 flushes to 0)
MASK_ADD = -30000.0       # fp16-safe additive attention mask for padded keys
POOL_SCALE = 1.0 / 32.0   # pooling-sum / L2 downscale (constraint 7)
RESIDUAL_MAX_TARGET = 8192.0   # keep |residual| <= this after 1/K scaling
NORM_SQ_MAX = 30000.0     # keep (|x|*s)^2 under this inside every rmsnorm
DOWN_IN_MAX = 2048.0      # cap on |down_proj input| after the MLP rescale (constraint 9)
DOWN_OUT_MAX = 16384.0    # cap on |down_proj output| after the MLP rescale
GELU_C = (2.0 / np.pi) ** 0.5
DOC_PREFIX = "title: none | text: "

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # 394 tokens with the document prefix: exercises the sliding-window band
    # (live for distances > 256), which the short sentences never reach. Do
    # not remove, and keep it between the window and the 512 bucket (main()
    # checks) — a wrong sliding mask passes every short-text parity check.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(30)
    ),
]

CALIBRATION_TEXTS = [DOC_PREFIX + s for s in PARITY_SENTENCES] + [
    DOC_PREFIX + " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
    "task: search result | query: what is the meaning of life, the universe and everything?",
    DOC_PREFIX + "Zahlen wie 3.14159 und Wörter wie Straßenbahn; 東京タワーは高い。",
]


def pow2(x):
    """Nearest power of two (exact in floating point)."""
    return 2.0 ** round(np.log2(x))


class SafeRMSNorm(torch.nn.Module):
    """Gemma3RMSNorm rewritten for fp16 range, numerically identical in fp32.

    Computes x*rsqrt(mean(x^2)+eps)*(1+w) as y*rsqrt(mean(y^2)+eps*s^2)*(1+w)
    with y = x*mult, where s = mult*in_scale is the total power-of-two scale
    relative to the raw (unscaled-model) activation. `out_scale` folds the
    1/K residual-branch scaling into the norms that feed residual adds.
    """

    def __init__(self, orig, mult, eps_eff, out_scale=1.0):
        super().__init__()
        self.register_buffer("weight", orig.weight.detach().clone())
        self.mult = float(mult)
        self.eps_eff = float(eps_eff)
        self.out_scale = float(out_scale)

    def forward(self, x):
        y = x * self.mult
        n = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps_eff)
        return n * ((1.0 + self.weight) * self.out_scale)


class ScaledEmbedding(torch.nn.Module):
    def __init__(self, inner, scale):
        super().__init__()
        self.inner = inner
        self.scale = float(scale)

    def forward(self, input_ids):
        return self.inner(input_ids) * self.scale


class TanhGelu(torch.nn.Module):
    """TWICE gelu_pytorch_tanh, built from tanh, mul and add (constraint 9).
    On the ANE, Core ML's native gelu op is off by up to ~6e-3 on [-1, 1];
    this form by at most ~9e-4 (~2e-4 near 0).

    2*gelu(x) = x * (1 + tanh(x * (c + c*0.044715*x^2))), c = sqrt(2/pi). The
    factor 2 saves the 0.5 multiply; the post-feedforward RMSNorm is
    scale-invariant and absorbs it (GAIN). Nothing overflows fp16 below
    |x| ~ 256, and beyond that x^2 = inf only saturates tanh to +-1, which
    still gives the right answer (2x or 0).
    """

    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.tanh(x * (GELU_C + (GELU_C * 0.044715) * (x * x))))


def calibrate(model, tokenizer):
    """fp32 activation stats: (max_abs, min/max mean-square) at every RMSNorm
    input, and (max_abs, sum of squares, count) at every MLP down_proj input."""
    stats = {}

    def pre_hook(name):
        def f(mod, args):
            t = args[0].detach()
            msq = t.pow(2).mean(-1)
            rec = stats.setdefault(name, [0.0, 0.0, float("inf")])
            rec[0] = max(rec[0], float(t.abs().max()))
            rec[1] = max(rec[1], float(msq.max()))
            rec[2] = min(rec[2], float(msq.min()))
        return f

    def rms_hook(name):
        def f(mod, args):
            t = args[0].detach()
            rec = stats.setdefault(name, [0.0, 0.0, 0])
            rec[0] = max(rec[0], float(t.abs().max()))
            rec[1] += float(t.pow(2).sum())
            rec[2] += t.numel()
        return f

    handles = []
    for i, layer in enumerate(model.layers):
        for attr in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            handles.append(getattr(layer, attr).register_forward_pre_hook(pre_hook(f"L{i}.{attr}")))
        handles.append(layer.self_attn.q_norm.register_forward_pre_hook(pre_hook(f"L{i}.q_norm")))
        handles.append(layer.self_attn.k_norm.register_forward_pre_hook(pre_hook(f"L{i}.k_norm")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(rms_hook(f"L{i}.down_proj")))
    handles.append(model.norm.register_forward_pre_hook(pre_hook("final")))

    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            enc = tokenizer(text, return_tensors="pt")
            model(**enc, use_cache=False)
    for h in handles:
        h.remove()
    return stats


def norm_scale(rec):
    """Power-of-two input scale s for a rmsnorm: mean(y^2)~=1, squares in range."""
    max_abs, max_msq, min_msq = rec
    s = pow2(1.0 / (max(min_msq, 1e-30) * max_msq) ** 0.25)
    while (max_abs * s) ** 2 > NORM_SQ_MAX:
        s /= 2.0
    if min_msq * s * s < 5e-4:
        raise SystemExit(f"rmsnorm dynamic range too wide for fp16: {rec}")
    return s


def mlp_scale(down_in, down_out, gain):
    """Power-of-two up_proj scale m for one layer (constraint 9): brings the
    down_proj input, raw * gain * m, to rms ~1 within fp16 headroom for the
    down_proj input and output. Never below 1."""
    max_abs, sumsq, count = down_in
    m = pow2(1.0 / (gain * (sumsq / count) ** 0.5))
    while m > 1.0 and (max_abs * gain * m > DOWN_IN_MAX or down_out[0] * gain * m > DOWN_OUT_MAX):
        m /= 2.0
    return max(m, 1.0)


def patch_model(model, stats, rms_eps):
    """Apply the 1/K residual rewrite (constraint 5) and the MLP precision
    rewrite (constraint 9). Returns K and each layer's total down_proj input
    scale (GAIN * m)."""
    residual_max = max(rec[0] for name, rec in stats.items()
                       if name.endswith(("input_layernorm", "pre_feedforward_layernorm", "final")))
    k = pow2(1.0)
    while residual_max / k > RESIDUAL_MAX_TARGET:
        k *= 2.0

    def safe(orig, rec, in_scale, out_scale=1.0):
        s = norm_scale(rec)
        eps_eff = max(rms_eps * s * s, EPS_FLOOR)
        return SafeRMSNorm(orig, mult=s / in_scale, eps_eff=eps_eff, out_scale=out_scale)

    mlp_scales = []
    for i, layer in enumerate(model.layers):
        # constraint 9: the MLP branch reaches post_feedforward_layernorm
        # scaled by TanhGelu.GAIN * m, where m is folded into up_proj
        m = mlp_scale(stats[f"L{i}.down_proj"], stats[f"L{i}.post_feedforward_layernorm"],
                      TanhGelu.GAIN)
        mlp_scales.append(TanhGelu.GAIN * m)
        with torch.no_grad():
            layer.mlp.up_proj.weight.mul_(m)
        layer.mlp.act_fn = TanhGelu()
        layer.input_layernorm = safe(layer.input_layernorm, stats[f"L{i}.input_layernorm"], 1 / k)
        layer.pre_feedforward_layernorm = safe(
            layer.pre_feedforward_layernorm, stats[f"L{i}.pre_feedforward_layernorm"], 1 / k)
        # branch-output norms: output folds the 1/K step. The attention branch
        # arrives unscaled, the MLP branch scaled by GAIN * m (constraint 9).
        layer.post_attention_layernorm = safe(
            layer.post_attention_layernorm, stats[f"L{i}.post_attention_layernorm"], 1.0, 1 / k)
        layer.post_feedforward_layernorm = safe(
            layer.post_feedforward_layernorm, stats[f"L{i}.post_feedforward_layernorm"],
            mlp_scales[-1], 1 / k)
        layer.self_attn.q_norm = safe(layer.self_attn.q_norm, stats[f"L{i}.q_norm"], 1.0)
        layer.self_attn.k_norm = safe(layer.self_attn.k_norm, stats[f"L{i}.k_norm"], 1.0)
    model.norm = safe(model.norm, stats["final"], 1 / k)
    model.embed_tokens = ScaledEmbedding(model.embed_tokens, 1 / k)
    return k, mlp_scales


class EmbedWrapper(torch.nn.Module):
    """Encoder + mask-aware mean pooling + dense stack + L2 norm, all in-graph."""

    def __init__(self, model, dense1_w, dense2_w, seq_len, window):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.register_buffer("position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0))
        # constraint 6: precomputed sliding band — attend iff |q-k| < window,
        # where `window` is config.sliding_window AFTER transformers' halving
        # for bidirectional models (257 for this checkpoint).
        idx = torch.arange(seq_len)
        band = ((idx[:, None] - idx[None, :]).abs() >= window).to(torch.float32)
        self.register_buffer("band_mask", band.reshape(1, 1, seq_len, seq_len) * MASK_ADD)
        self.dense1 = torch.nn.Linear(DIMS, dense1_w.shape[0], bias=False)
        self.dense1.weight = torch.nn.Parameter(dense1_w)
        self.dense2 = torch.nn.Linear(dense2_w.shape[1], DIMS, bias=False)
        self.dense2.weight = torch.nn.Parameter(dense2_w)

    def forward(self, input_ids, attention_mask):
        mask_f = attention_mask.to(torch.float32)
        # constraint 6: prepared bidirectional padding masks, fp16-safe constant
        addmask = (1.0 - mask_f).reshape(1, 1, 1, self.seq_len) * MASK_ADD
        h = self.model(
            input_ids=input_ids.long(),
            attention_mask={
                "full_attention": addmask,
                "sliding_attention": addmask + self.band_mask,
            },
            position_ids=self.position_ids,
            use_cache=False,
        ).last_hidden_state
        # constraint 7: mask-aware channel sum at 1/32 scale (fp16 range), then
        # the mean at natural scale — the dense linears need O(1) inputs on
        # the ANE (constraint 9) — and 1/32 again before the L2 sum of squares.
        w = (mask_f * POOL_SCALE).unsqueeze(-1)
        summed = (h * w).sum(dim=1)
        count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)
        pooled = summed / (count * POOL_SCALE)
        y = self.dense2(self.dense1(pooled)) * POOL_SCALE
        den = y.pow(2).sum(dim=-1, keepdim=True)
        out = y * torch.rsqrt(den + 1e-6)
        return out.reshape(1, DIMS)


def st_reference(model, tokenizer, dense1_w, dense2_w):
    """fp32 sentence-transformers math on the UNPATCHED model (mean pool ->
    dense stack -> L2 normalize) for the parity sentences."""
    refs = []
    with torch.no_grad():
        for s in PARITY_SENTENCES:
            enc = tokenizer(DOC_PREFIX + s, return_tensors="pt")
            h = model(**enc, use_cache=False).last_hidden_state
            pooled = h.mean(dim=1)
            y = pooled @ dense1_w.T @ dense2_w.T
            refs.append((y / y.norm())[0].numpy())
    return refs


def padded_inputs(tokenizer, text, seq_len):
    ids_list = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids_list) > seq_len:
        raise SystemExit(f"parity text longer than bucket {seq_len}")
    ids = np.zeros((1, seq_len), dtype=np.int32)   # pad id 0, as the server pads
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


def fitting_pairs(tokenizer, refs, seq_len):
    """(text, ref) pairs whose token count fits the bucket. The long text
    only participates at 512 — that's the bucket with a live sliding band
    (main() checks that it does)."""
    pairs = []
    for s, ref in zip(PARITY_SENTENCES, refs):
        n = len(tokenizer(DOC_PREFIX + s, add_special_tokens=True)["input_ids"])
        if n <= seq_len:
            pairs.append((s, ref))
    return pairs


def fp32_gate(wrapper, tokenizer, refs, seq_len):
    """The range and MLP rewrites must be ~exact in fp32 before we spend on
    conversion."""
    cosines = []
    with torch.no_grad():
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, DOC_PREFIX + s, seq_len)
            out = wrapper(torch.from_numpy(ids), torch.from_numpy(mask))[0].numpy()
            cosines.append(cosine(ref, out))
    worst = worst_of(cosines)
    if not worst >= 0.9999:  # NaN fails too
        raise SystemExit(f"seq {seq_len}: fp32 rewrite parity {worst:.6f} < 0.9999")
    return worst


def convert_bucket(wrapper, seq_len, workdir):
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0], ids[0, 1] = 2, 1  # <bos> <eos>
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
        outputs=[ct.TensorType(name="embeddings")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    # Explicit attention (D25) and the explicit GELU (constraint 9) must
    # survive conversion: coremltools fuses attention into its
    # scaled_dot_product_attention op when the torch call has no explicit
    # scale, and could pattern-match the tanh GELU back into its gelu op.
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    fused = ops & {"scaled_dot_product_attention", "gelu"}
    if fused:
        raise SystemExit(f"seq {seq_len}: converted graph contains {sorted(fused)} — "
                         "see constraint 9 and docs/DECISIONS.md D25")
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs):
    """Cosine vs the fp32 sentence-transformers reference on BOTH Espresso
    compute paths — a pass under .ALL alone hides dynamic-shape/fp16 failures."""
    results = {}
    # per-path gates, see constraint 9: CPU_ONLY validates the conversion,
    # CPU_AND_NE validates hardware execution at the ANE's native precision
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.999),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        cosines = []
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, DOC_PREFIX + s, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embeddings"][0]
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq_len} [{label}]: non-finite output — see constraint 5")
            cosines.append(cosine(ref, out))
        worst = worst_of(cosines)
        if not worst >= gate:
            raise SystemExit(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        # Pad invariance (D25): a correctly masked model can't see its pad
        # positions, so their content must not change the output.
        ids, mask = padded_inputs(tokenizer, DOC_PREFIX + PARITY_SENTENCES[0], seq_len)
        noisy = ids.copy()
        pads = mask[0] == 0
        noisy[0, pads] = np.random.default_rng(0).integers(1000, 40000, int(pads.sum()))
        a = m.predict({"input_ids": ids, "attention_mask": mask})["embeddings"][0]
        b = m.predict({"input_ids": noisy, "attention_mask": mask})["embeddings"][0]
        if not cosine(a, b) >= 0.99999:
            raise SystemExit(f"seq {seq_len} [{label}]: output depends on pad content "
                             f"(cos {cosine(a, b):.6f}); the attention mask is being dropped")
        # latency, as an ANE-residency proxy (D15 measured ratios the same way)
        ids, mask = padded_inputs(tokenizer, DOC_PREFIX + PARITY_SENTENCES[0], seq_len)
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
    assert tokenizer.pad_token_id == 0, "server pads input_ids with 0"
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    model.config.use_cache = False
    # NOTE: model.config.sliding_window is 257 here, not config.json's 512 —
    # transformers halves it (512//2+1) for bidirectional models. It is the
    # exclusive band half-width for the wrapper's sliding mask.
    window = model.config.sliding_window
    # constraint 6: wherever the sliding band is live, a parity text must reach it
    longest = max(len(tokenizer(DOC_PREFIX + s, add_special_tokens=True)["input_ids"])
                  for s in PARITY_SENTENCES)
    for seq in buckets:
        if seq > window and not window < longest <= seq:
            raise SystemExit(f"bucket {seq}: no parity text longer than the sliding window "
                             f"({window}) fits (longest is {longest} tokens)")

    dense1_w = load_file(src / "2_Dense/model.safetensors")["linear.weight"].float()
    dense2_w = load_file(src / "3_Dense/model.safetensors")["linear.weight"].float()

    print("calibrating fp32 activation ranges...")
    stats = calibrate(model, tokenizer)
    refs = st_reference(model, tokenizer, dense1_w, dense2_w)
    k, mlp_scales = patch_model(model, stats, model.config.rms_norm_eps)
    print(f"residual scale K={k:g}; down_proj input scales {[int(m) for m in mlp_scales]}")

    parity = {}
    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = EmbedWrapper(model, dense1_w, dense2_w, seq, window)
            wrapper.eval()
            fp32_cos = fp32_gate(wrapper, tokenizer, refs, seq)
            print(f"bucket {seq}: fp32 rewrite parity {fp32_cos:.6f}; converting...")
            pkg = convert_bucket(wrapper, seq, workdir)
            results = parity_check(tokenizer, pkg, seq, refs)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            parity[seq] = min(v[0] for v in results.values())
            for label, (cos, ms) in results.items():
                print(f"bucket {seq} [{label}]: parity cos={cos:.6f} {ms:.1f}ms")
            print(f"bucket {seq}: -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = Path(__file__).resolve().parent.parent / "examples/manifests/embeddinggemma-300m/manifest.toml"
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")
    print("parity per bucket:", json.dumps(parity))


if __name__ == "__main__":
    main()
