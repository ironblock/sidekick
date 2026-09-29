"""Convert a Qwen3-class causal-decoder embedding model into ANE-resident
Core ML artifacts. Validated on codefuse-ai/F2LLM-v2-160M; the same recipe
covers Qwen3/Qwen3-Embedding-derived last-token embedders (dims read from
config, prefixes read from config_sentence_transformers.json).

Produces one static-shape .mlmodelc per sequence-length bucket, with
LAST-TOKEN pooling baked into the graph, matching the model's manifest.

Usage:
    python tools/convert_qwen3_embedding.py <hf-model-dir> <install-dir> [buckets...]

Requires: torch, transformers >= 4.51 (Qwen3), coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

Qwen3 was the fourth architecture class validated on the stack (after classic
BERT / bge, Gemma3 / embeddinggemma, and LFM2 hybrid; ModernBERT followed, see
docs/MODELS.md). It is the first CAUSAL DECODER used for embeddings here.
Three conversion facts, all handled below:

A. CAUSAL + PADDING MASK, fp16-safe. Qwen3Model builds its mask via
   transformers.masking_utils.create_causal_mask, which materializes
   finfo(dtype).min for disallowed positions — -inf in fp16, NaNs softmax on
   the ANE (D15's eager-mask failure class). We patch create_causal_mask (in
   the qwen3 module namespace) to build the identical lower-triangular +
   key-padding mask with the fp16-safe MASK_ADD = -30000. Right-padding is
   correct for a causal model: the last real token attends only to earlier
   real tokens, so trailing pads never affect it.

B. LAST-TOKEN POOLING, in-graph, no data-dependent index. The embedding is
   the hidden state of the last non-pad token. Rather than gather a computed
   index (a traced Int op that crashes coremltools, D17 constraint 8), we
   select it with the attention mask alone: last_onehot = mask * (1 -
   shift_left(mask)) is 1 exactly at the last real position (for right-
   padding), and a masked sum yields (1, dims). Output stays the (1, dims)
   shape the server expects for pooling = "none"; the server L2-normalizes
   in f32.

C. RoPE + GQA shape arithmetic. Stock rotate_half slices with x.shape[-1]//2,
   and stock repeat_kv reshapes to num_kv_heads * n_rep computed from the
   tensor's shape. Both trace to Int ops that crash coremltools 9 under
   static shapes (D17 constraint 8), so both are replaced with
   shape-arithmetic-free equivalents (chunk(2); expand(-1, ...) + flatten).
   The repeat_kv patch must also reach the sdpa_attention module's copy,
   which is the one the sdpa path calls.

D. PRECISION REWRITE (docs/DECISIONS.md D20 amendment), the two ANE limits
   of the D17/D19 rewrites, to the extent they apply here:
     - Core ML's native silu op is off by up to ~1.5e-2 on [-1, 1] on the
       Neural Engine, and it cost this model its worst cases (0.99966 on a
       run of digits). TanhSilu builds 2*silu(x) = x * (1 + tanh(x / 2))
       from tanh, mul and add; up_proj's weights take the factor 1/2, so
       the MLP's output is unchanged and no op is added. x * sigmoid(x) is
       no alternative: coremltools fuses it back into the native op.
     - The ANE's linear op loses precision on small inputs (relative error
       ~3e-4 / rms). Attention's inputs are small in the early layers,
       whose input norms have weights of ~0.13-0.18: q/k/v at rms
       0.06-0.18, o_proj at 0.02-0.35. That costs long inputs most (0.99995
       at 512 tokens). Fix: power-of-two scales on the input norm's weight
       (q/k RMSNorm eps x s^2) and v_proj bring both to rms ~1, and Descale
       multiplies attention's output by 1/S before the residual add. The
       MLP's inputs (median rms 0.08-0.5) needed no rescale: rescaling the
       down projection changed nothing measurable.
   convert_bucket() fails if a silu, gelu or fused attention op survives
   conversion.

fp16 note: NO range rewrite. Qwen3's q_norm/k_norm (QK-norm) keep activations
tiny (measured max ~420 on F2LLM-v2-160M), so fp16 is simply safe. Verified
before converting by checking that PyTorch-fp16 parity is ~1.0 and no
activation exceeds ~30k (tools/probe_activations.py does this check).

Gates, per bucket: fp32 exactness of constraint D (>= 0.99999) before
converting; then CPU_ONLY >= 0.999 (the conversion is faithful) and
CPU_AND_NE >= 0.999 (the ANE runs it at full precision), finite output, and
pad invariance (pad ids 0 vs random must give the same output, D25). Every
gate treats NaN as a failure.
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
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as _qwen3
import transformers.integrations.sdpa_attention as _sdpa_attention

MASK_ADD = -30000.0
IN_MAX = 2048.0     # cap on |linear input| after a rescale (constraint D)
OUT_MAX = 16384.0   # cap on |linear output| after a rescale
QK_MAX = 150.0      # cap on |q|, |k| entering their RMSNorms, which square them

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]
# (index, is_query) — alternate query/document so both prefix paths are tested
QUERY_FLAGS = [True, False, True, False, False]

# Varied text for constraint D's activation statistics: prose, code, numbers
# and URLs, punctuation, non-English, and degenerate repetition. Documents
# carry no prefix for F2LLM; the query prompt is added where marked.
CALIBRATION_TEXTS = [(s, q) for s, q in zip(PARITY_SENTENCES, QUERY_FLAGS)] + [
    ("def add(a, b):\n    return a + b  # simple helper\n", False),
    ("Order #48213 shipped 2026-09-14; see https://example.com/track?id=48213&ref=a1b2.", False),
    ("Wait... what?! (No, really — \"that\" isn't it.) [1] {2} <3>", False),
    ("Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。", False),
    (" ".join(["buffalo"] * 60), False),
    ("3.14159 2.71828 1.41421 6.02214076e23 299792458", True),
]


def _traceable_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _traceable_repeat_kv(hidden_states, n_rep):
    # expand(-1, ...) + flatten needs no shape arithmetic. Computing
    # kvh * n_rep from the traced shape (as this function once did, and as
    # the stock versions do) traces to an Int op that crashes coremltools
    # 9 under torch 2.13 (D17 constraint 8).
    if n_rep == 1:
        return hidden_states
    return hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)


def _fp16_safe_causal_mask(config=None, input_embeds=None, attention_mask=None, **kw):
    # constraint A: lower-triangular (causal) AND key-not-pad, additive with
    # MASK_ADD. Batch-1 static seq; returns (bsz, 1, seq, seq).
    embeds = input_embeds
    seq = embeds.shape[1]
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.float32))  # 1 where k<=q
    if attention_mask is not None:
        keep = attention_mask.to(torch.float32)                      # (bsz, seq) 1=real
        allowed = causal.unsqueeze(0) * keep[:, None, :]             # (bsz, q, k)
    else:
        allowed = causal.unsqueeze(0)
    add = (1.0 - allowed).unsqueeze(1) * MASK_ADD                    # (bsz, 1, q, k)
    return add.to(embeds.dtype)


class TanhSilu(torch.nn.Module):
    """TWICE silu, from tanh, mul and add (constraint D); up_proj takes the
    compensating 1/2."""

    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.tanh(0.5 * x))


class Descale(torch.nn.Module):
    """inner(x) * inv — undoes constraint D's rescale before the residual add.
    An explicit multiply, not folded into the weights: dividing the weights
    would push small ones into fp16's subnormal range."""

    def __init__(self, inner, inv):
        super().__init__()
        self.inner = inner
        self.inv = float(inv)

    def forward(self, x):
        return self.inner(x) * self.inv


class _Stat:
    """rms over every element seen, plus max |x|."""

    def __init__(self):
        self.sumsq, self.count, self.max = 0.0, 0, 0.0

    def add(self, t):
        t = t.detach().double()
        self.sumsq += float(t.pow(2).sum())
        self.count += t.numel()
        self.max = max(self.max, float(t.abs().max()))

    @property
    def rms(self):
        return (self.sumsq / self.count) ** 0.5


def pow2(x):
    """Nearest power of two (exact in floating point)."""
    return 2.0 ** round(np.log2(x))


def calibrate(model, tokenizer, qprefix, dprefix):
    """fp32 stats (unpadded forwards) of the attention tensors constraint D rescales."""
    stats = {}

    def stat(name):
        return stats.setdefault(name, _Stat())

    hooks = []
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        hooks += [
            a.q_proj.register_forward_pre_hook(lambda m, args, i=i: stat(f"{i}.qkv_in").add(args[0])),
            a.q_proj.register_forward_hook(lambda m, args, out, i=i: stat(f"{i}.q").add(out)),
            a.k_proj.register_forward_hook(lambda m, args, out, i=i: stat(f"{i}.k").add(out)),
            a.o_proj.register_forward_pre_hook(lambda m, args, i=i: stat(f"{i}.o_in").add(args[0])),
            a.o_proj.register_forward_hook(lambda m, args, out, i=i: stat(f"{i}.o_out").add(out)),
        ]
    with torch.no_grad():
        for text, is_q in CALIBRATION_TEXTS:
            model(**tokenizer(full_text(text, is_q, qprefix, dprefix), return_tensors="pt"))
    for h in hooks:
        h.remove()
    return stats


def input_scale(rec, out=None):
    """Power-of-two scale bringing rec to rms ~1 within fp16 headroom for the
    input and, if given, the output. >= 1."""
    s = pow2(1.0 / rec.rms)
    while s > 1.0 and (rec.max * s > IN_MAX or (out is not None and out.max * s > OUT_MAX)):
        s /= 2.0
    return max(s, 1.0)


def precision_rewrite(model, stats):
    """Constraint D. Every factor is a power of two, so the fp32 graph is
    unchanged (fp32_gate checks). Returns the attention scales, for the log."""
    scales = []
    for i, layer in enumerate(model.layers):
        # silu: 2*silu(gate) * (up / 2)
        with torch.no_grad():
            layer.mlp.up_proj.weight.mul_(1.0 / TanhSilu.GAIN)
        layer.mlp.act_fn = TanhSilu()
        # attention: q/k/v via the input norm (q and k are re-normalized per
        # head, so only their RMSNorm eps moves); o_proj's input via v_proj
        a = layer.self_attn
        s_in = input_scale(stats[f"{i}.qkv_in"])
        qk = max(stats[f"{i}.q"].max, stats[f"{i}.k"].max)
        while s_in > 1.0 and qk * s_in > QK_MAX:
            s_in /= 2.0
        s_o = input_scale(stats[f"{i}.o_in"], stats[f"{i}.o_out"])
        with torch.no_grad():
            layer.input_layernorm.weight.mul_(s_in)
            a.v_proj.weight.mul_(s_o / s_in)
        a.q_norm.variance_epsilon *= s_in * s_in
        a.k_norm.variance_epsilon *= s_in * s_in
        a.o_proj = Descale(a.o_proj, 1.0 / s_o)
        scales.append(f"L{i} {s_in:g}/{s_o:g}")
    return scales


def install_patches():
    _qwen3.rotate_half = _traceable_rotate_half
    _qwen3.repeat_kv = _traceable_repeat_kv
    # The sdpa attention path calls repeat_kv from the sdpa_attention module,
    # not qwen3's — patch there too, or the traceable version is inert on the
    # path actually used (matches the LFM converter). It is load-bearing: the
    # stock version's reshape to num_kv_heads * n_rep crashes coremltools 9
    # under torch 2.13 (constraint C).
    _sdpa_attention.repeat_kv = _traceable_repeat_kv
    _qwen3.create_causal_mask = _fp16_safe_causal_mask


class LastTokenWrapper(torch.nn.Module):
    """Last-non-pad-token pooling via the attention mask, reshaped to
    (1, dims). No in-graph L2 — the server normalizes in f32."""

    def __init__(self, model, seq_len, dims):
        super().__init__()
        self.model = model
        self.dims = dims
        self.register_buffer(
            "position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=self.position_ids,
        ).last_hidden_state
        mask_f = attention_mask.to(hidden.dtype)                    # (1, seq)
        shifted = F.pad(mask_f[:, 1:], (0, 1), value=0.0)          # mask[i+1], last=0
        last_onehot = mask_f * (1.0 - shifted)                     # 1 at last real pos
        pooled = (last_onehot.unsqueeze(-1) * hidden).sum(dim=1)   # (1, dims)
        return pooled.reshape(1, self.dims)


def load_prompts(src):
    cfg = json.load(open(Path(src) / "config_sentence_transformers.json"))
    p = cfg.get("prompts", {})
    return p.get("query", ""), p.get("document", "")


def full_text(text, is_query, qprefix, dprefix):
    return (qprefix if is_query else dprefix) + text


def reference_embeddings(model, tokenizer, qprefix, dprefix):
    """fp32 last-token references (unnormalized; cosine is scale-invariant),
    verified equal to SentenceTransformer.encode at cosine 1.0."""
    refs = []
    with torch.no_grad():
        for s, is_q in zip(PARITY_SENTENCES, QUERY_FLAGS):
            enc = tokenizer(full_text(s, is_q, qprefix, dprefix),
                            return_tensors="pt", truncation=True, max_length=MAX_BUCKET)
            refs.append(model(**enc).last_hidden_state[0, -1, :].numpy())
    return refs


MAX_BUCKET = 512  # references are computed at this truncation; the server
# routes any over-length input to the largest bucket, preserving the final
# (EOS) token — see load/truncation in coreml_embedder.rs and constraint B.


def padded_inputs(tokenizer, text, seq_len):
    # Truncate the SAME way the server does: HF right-truncation to MAX_BUCKET
    # keeps [first MAX_BUCKET-1 content, EOS], matching the last-token-safe
    # server truncation. fitting_cases guarantees the result fits seq_len.
    ids_list = tokenizer(text, add_special_tokens=True, truncation=True,
                         max_length=MAX_BUCKET)["input_ids"]
    if len(ids_list) > seq_len:
        raise SystemExit(f"parity text longer than bucket {seq_len}")
    ids = np.zeros((1, seq_len), dtype=np.int32)  # right-pad id 0, as the server pads
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


def fitting_cases(tokenizer, refs, seq_len, qprefix, dprefix):
    # Test each text at buckets >= its server-routed length (MAX_BUCKET-
    # truncated). The long text lands at 512, exercising positions up to 511
    # — the full-bucket / high-RoPE-position path a short-only gate misses.
    out = []
    for s, is_q, ref in zip(PARITY_SENTENCES, QUERY_FLAGS, refs):
        t = full_text(s, is_q, qprefix, dprefix)
        n = len(tokenizer(t, add_special_tokens=True, truncation=True,
                          max_length=MAX_BUCKET)["input_ids"])
        if n <= seq_len:
            out.append((t, ref))
    return out


def fp32_gate(wrapper, tokenizer, refs, seq_len, qprefix, dprefix):
    """Constraint D must be ~exact in fp32 before we spend on conversion."""
    cosines = []
    with torch.no_grad():
        for t, ref in fitting_cases(tokenizer, refs, seq_len, qprefix, dprefix):
            ids, mask = padded_inputs(tokenizer, t, seq_len)
            out = wrapper(torch.from_numpy(ids), torch.from_numpy(mask))[0].numpy()
            cosines.append(cosine(ref, out))
    worst = worst_of(cosines)
    if not worst >= 0.99999:  # NaN fails too
        raise SystemExit(f"seq {seq_len}: fp32 rewrite parity {worst:.7f} < 0.99999")
    return worst


def convert_bucket(wrapper, seq_len, workdir):
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0] = 151643  # any real token; content irrelevant to tracing
    mask = torch.zeros((1, seq_len), dtype=torch.int32)
    mask[0, :1] = 1
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
    # Explicit attention (D25) and the explicit silu (constraint D) must
    # survive conversion: coremltools fuses attention into its
    # scaled_dot_product_attention op when the torch call has no explicit
    # scale, and x * sigmoid(x) comes out as its native silu op.
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    fused = ops & {"scaled_dot_product_attention", "silu", "gelu"}
    if fused:
        raise SystemExit(f"seq {seq_len}: converted graph contains {sorted(fused)} — "
                         "see constraint D and docs/DECISIONS.md D25")
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs, qprefix, dprefix):
    results = {}
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.999),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        cosines = []
        for t, ref in fitting_cases(tokenizer, refs, seq_len, qprefix, dprefix):
            ids, mask = padded_inputs(tokenizer, t, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            cosines.append(cosine(ref, out))
        worst = worst_of(cosines)
        if not worst >= gate:  # NaN fails too
            raise SystemExit(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        t0text = full_text(PARITY_SENTENCES[0], QUERY_FLAGS[0], qprefix, dprefix)
        # Pad invariance (D25): the causal + padding mask hides the right
        # pads from the last real token, so their content can't matter.
        ids, mask = padded_inputs(tokenizer, t0text, seq_len)
        noisy = ids.copy()
        pads = mask[0] == 0
        noisy[0, pads] = np.random.default_rng(0).integers(1000, 40000, int(pads.sum()))
        a = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
        b = m.predict({"input_ids": noisy, "attention_mask": mask})["embedding"][0]
        if not cosine(a, b) >= 0.99999:
            raise SystemExit(f"seq {seq_len} [{label}]: output depends on pad content "
                             f"(cos {cosine(a, b):.6f}); the attention mask is being dropped")
        ids, mask = padded_inputs(tokenizer, t0text, seq_len)
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
    model.config.use_cache = False
    install_patches()
    dims = model.config.hidden_size
    qprefix, dprefix = load_prompts(src)
    print(f"dims={dims} qprefix={qprefix[:40]!r} dprefix={dprefix!r}")

    refs = reference_embeddings(model, tokenizer, qprefix, dprefix)
    scales = precision_rewrite(model, calibrate(model, tokenizer, qprefix, dprefix))
    print("constraint D attention scales (q/k/v in, o_proj in):", ", ".join(scales))

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = LastTokenWrapper(model, seq, dims).eval()
            f32 = fp32_gate(wrapper, tokenizer, refs, seq, qprefix, dprefix)
            print(f"bucket {seq}: fp32 rewrite parity {f32:.7f}")
            pkg = convert_bucket(wrapper, seq, workdir)
            res = parity_check(tokenizer, pkg, seq, refs, qprefix, dprefix)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            for label, (cos, ms) in res.items():
                print(f"bucket {seq} [{label}]: parity cos={cos:.6f} {ms:.1f}ms")
            print(f"bucket {seq} -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = (
        Path(__file__).resolve().parent.parent
        / f"examples/manifests/{install_dir.name}/manifest.toml"
    )
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
