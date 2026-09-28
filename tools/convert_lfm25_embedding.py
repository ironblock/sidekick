"""Convert LiquidAI/LFM2.5-Embedding-350M into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling and L2 normalization baked into the graph, matching
examples/manifests/lfm2.5-embedding-350m.

Usage:
    python tools/convert_lfm25_embedding.py <hf-model-dir> <install-dir> [buckets...]
    python tools/convert_lfm25_embedding.py --no-pad-zeroing <hf-model-dir> <install-dir> [buckets...]

    --no-pad-zeroing:  NEGATIVE CONTROL. Skip constraint D, so the short
                       convs read pad states (0.905 parity when measured),
                       to prove the parity suite catches it. Parity failures
                       are reported, not fatal. Never install the result
                       where the daemon looks.

    hf-model-dir: local snapshot of LiquidAI/LFM2.5-Embedding-350M
                  (the model ships custom code — modeling_lfm2_bidirectional.py —
                   loaded via trust_remote_code; read it before converting)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/lfm2.5-embedding-350m"
    buckets:      default 128 256 512

Requires: torch, transformers >= 4.55 (Lfm2 support), coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

Architecture notes (why this is a third recipe, not a bge/gemma rerun):

LFM2.5 is a hybrid — 10 double-gated short-conv blocks interleaved with
6 full-attention blocks (GQA 16/8, QK-norm). The upstream repo patches the
backbone to be bidirectional: non-causal SDPA plus a symmetric-padding
F.conv1d short-conv forward that is cache-free and traces cleanly. Two
properties once made it look like the EASY class of conversion:

- QK-norm and small norm weights keep fp32 activations tiny (calibrated max
  ~25 across every module) — no D17 range rewrite needed. But tiny is not
  safe on the ANE: see constraint E.
- CLS pooling: no mask-aware mean, no sliding-window band masks.

The recipe still encodes the hardware-verified constraints from
tools/convert_bge_small.py (D15: per-bucket STATIC shapes, pooling inside
the graph, SDPA, explicit position_ids) plus five LFM2.5-specific ones:

A. The upstream bidirectional mask uses -1e9 pad bias, which saturates to
   -inf in fp16 and NaNs softmax on the ANE (same failure class as D15's
   eager-attention rule). We re-patch create_causal_mask with an identical
   mask built at -30000.
B. Stock rotate_half/repeat_kv trace into Int-op chains that crash
   coremltools 9.x under static shapes (D17 constraint 8) — same traceable
   replacements as the gemma recipe.
C. The CLS vector's sum-of-squares can exceed fp16 range inside the
   in-graph L2 normalize (|h| up to ~25, 1024 dims -> ~640k >> 65504), so
   CLS is scaled by 1/32 first; L2 normalization cancels the constant.
D. Pad states must be zeroed before every conv: the symmetric short-conv
   mixes neighbors unconditionally, so right-padding contaminates real
   tokens (measured parity 0.905 without the fix). Zeroing reproduces the
   unpadded forward exactly and makes embeddings bucket-invariant — see
   _traceable_shortconv_forward.
E. PRECISION REWRITE (docs/DECISIONS.md D19 amendment; the same two ANE
   limits as EmbeddingGemma's, D17). Measured on macOS 27, M1 Max:
     - The ANE's linear op has an absolute precision floor on its input,
       relative error ~3e-4 / rms(input). This model's activations are tiny
       everywhere: every output projection's input has rms 0.003-0.07 (the
       MLP down projection, the conv block's out_proj, attention's
       out_proj) and q/k/v's 0.04-0.16. Each sub-block lost 1-13% per
       layer on the ANE. Fix: calibrated power-of-two scales bring each
       input to rms ~1 — the operator norm's weight (q/k/v, in_proj; q/k
       RMSNorm eps x s^2), v_proj, in_proj's B and C rows (the conv's
       input B*x and out_proj's input C*conv(Bx)), and w3. No norm follows
       these branches, only the residual add, so Descale multiplies each
       branch output by 1/S before it.
     - Core ML's silu op is coarse on the ANE (~1.5e-2 on [-1, 1]).
       TanhSilu builds it from tanh instead; x * sigmoid(x) is no
       alternative, since conversion fuses it back into the native op.
       convert_bucket() fails if a silu, gelu or fused attention op
       survives conversion.
   Before: ANE parity 0.987 on prose and 0.954 on a URL (parity suite grade
   D). After: 0.99999 at every bucket, closer to fp32 than CPU_ONLY, at
   13-16% more ANE latency (the explicit silu; the rescales are free).

Parity is gated per compute path: CPU_ONLY >= 0.999 proves the conversion
is faithful, CPU_AND_NE >= 0.999 that the ANE runs it at full precision.
Both are measured on real tokenized sentences, including a 471-token text
in the 512 bucket, against the fp32 reference. The converter also gates
finite output, pad invariance (pad ids 0 vs random must give the same
output, D25) and, before conversion, fp32 exactness of constraint E
(>= 0.99999). Every gate treats NaN as a failure. The parity suite (D26)
grades the result on adversarial inputs.
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
import transformers.models.lfm2.modeling_lfm2 as _lfm2_modeling
import transformers.integrations.sdpa_attention as _sdpa_attention

DIMS = 1024
MASK_ADD = -30000.0     # fp16-safe additive attention mask for padded keys
CLS_SCALE = 1.0 / 32.0  # pre-normalize downscale, cancelled by L2 (constraint C)
IN_MAX = 2048.0         # cap on |linear/conv input| after a rescale (constraint E)
OUT_MAX = 16384.0       # cap on |linear/conv output| after a rescale
QK_MAX = 150.0          # cap on |q|, |k| entering their RMSNorms, which square them
QUERY_PREFIX = "query: "
DOC_PREFIX = "document: "

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # 471 tokens with the document prefix: exercises long sequences and the
    # RoPE position range in the 512 bucket, which short sentences
    # under-test. main() fails if it stops fitting that bucket; an earlier
    # 40-sentence version was 523 tokens, so no bucket ever ran it.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(36)
    ),
]

# Varied text for constraint E's activation statistics: prose, code, numbers
# and URLs, punctuation, non-English, and degenerate repetition.
CALIBRATION_TEXTS = [DOC_PREFIX + s for s in PARITY_SENTENCES] + [
    DOC_PREFIX + "def add(a, b):\n    return a + b  # simple helper\n",
    DOC_PREFIX + "Order #48213 shipped 2026-09-14; see https://example.com/track?id=48213&ref=a1b2.",
    DOC_PREFIX + "Wait... what?! (No, really — \"that\" isn't it.) [1] {2} <3>",
    DOC_PREFIX + "Der schnelle braune Fuchs springt über den faulen Hund. 東京は日本の首都です。",
    DOC_PREFIX + " ".join(["buffalo"] * 60),
    QUERY_PREFIX + "3.14159 2.71828 1.41421 6.02214076e23 299792458",
]


def _traceable_rotate_half(x):
    # constraint B: identical to the stock rotate_half for even head dims,
    # but chunk() keeps shape arithmetic out of the traced graph.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _traceable_repeat_kv(hidden_states, n_rep):
    # constraint B: the stock repeat_kv reshapes with num_kv_heads * n_rep
    # computed from tensor sizes, an Int-op crash. expand + flatten needs
    # no shape arithmetic and produces the identical layout.
    if n_rep == 1:
        return hidden_states
    return hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).flatten(1, 2)


def _make_traceable_shortconv_forward(zero_pads):
    # constraint B, LFM2-specific site: the upstream non-causal short-conv
    # (modeling_lfm2_bidirectional._noncausal_shortconv_forward) computes
    # F.conv1d padding/groups from tensor shapes, which jit.trace turns into
    # traced values that conv1d rejects ("expected padding to be a single
    # integer ... got padding=[]"). Shapes are truly static per bucket, so
    # int() pins them concretely. Math is identical for odd kernels ('same'
    # symmetric padding).
    def forward(
        self, hidden_states, past_key_values=None, cache_position=None,
        attention_mask=None,
    ):
        # constraint D (zero_pads=True): unlike attention (where the mask
        # silences pad KEYS), the symmetric conv mixes neighbors
        # unconditionally — pad-position states contaminate the last real
        # positions, and with 10 stacked conv layers the leak reaches ~10
        # tokens deep, then spreads to CLS via attention. Measured: parity
        # 0.905 vs the unpadded fp32 reference at bucket 128. Zeroing pad
        # states before EVERY conv reproduces the unpadded forward exactly
        # at all real positions (F.conv1d edge-pads with zeros), making
        # embeddings bucket-invariant. The upstream 4D additive mask is 0
        # for real tokens and MASK_ADD for pads, so `== 0` recovers the
        # keep-mask. (Upstream skips this zeroing on the sdpa path to mirror
        # padded-BATCH training; our reference semantics are the unpadded
        # single-text forward, which is what SentenceTransformer.encode
        # computes. ColBERT must NOT zero — its query-expansion tokens are
        # mask=0 but participate in MaxSim; see tools/smoke_lfm25_colbert.py.)
        if zero_pads and attention_mask is not None:
            keep = (attention_mask == 0).to(hidden_states.dtype).reshape(1, -1, 1)
            hidden_states = hidden_states * keep
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x
        k = int(self.conv.weight.shape[-1])
        assert k % 2 == 1, "even conv kernels need an output-length correction"
        conv_out = torch.nn.functional.conv1d(
            Bx, weight=self.conv.weight, bias=self.conv.bias,
            stride=1, padding=k // 2, dilation=1, groups=int(Bx.shape[1]),
        )
        y = C * conv_out
        return self.out_proj(y.transpose(-1, -2).contiguous())

    return forward


def _fp16_safe_bidirectional_mask(config, **kwargs):
    # constraint A: same pad-only additive mask the upstream remote code
    # installs (modeling_lfm2_bidirectional._bidirectional_mask), but built
    # at MASK_ADD instead of -1e9. Cache-free trace: kv_len == q_len, and a
    # (1, 1, 1, S) mask broadcasts over query positions inside SDPA.
    embeds = kwargs.get("inputs_embeds")
    if embeds is None:
        embeds = kwargs.get("input_embeds")
    attention_mask = kwargs.get("attention_mask")
    pad = 1.0 - attention_mask.to(embeds.dtype)
    return pad[:, None, None, :] * MASK_ADD


def install_patches(conv_pad_zeroing=True):
    _lfm2_modeling.rotate_half = _traceable_rotate_half
    _lfm2_modeling.repeat_kv = _traceable_repeat_kv
    _sdpa_attention.repeat_kv = _traceable_repeat_kv
    # Must run AFTER the model is loaded: importing the repo's remote code
    # installs its own create_causal_mask/slow_forward patches, which would
    # overwrite these if the order were reversed. Both are resolved at call
    # time, so the last patch installed wins.
    _lfm2_modeling.create_causal_mask = _fp16_safe_bidirectional_mask
    _lfm2_modeling.Lfm2ShortConv.slow_forward = _make_traceable_shortconv_forward(
        conv_pad_zeroing
    )


def pow2(x):
    """Nearest power of two (exact in floating point)."""
    return 2.0 ** round(np.log2(x))


class TanhSilu(torch.nn.Module):
    """TWICE silu, built from tanh, mul and add (constraint E): Core ML's
    native silu op is off by up to ~1.5e-2 on [-1, 1] on the ANE; tanh by
    ~1.6e-3. 2*silu(x) = x * (1 + tanh(x / 2)); the factor 2 saves the 0.5
    multiply and is divided out with the MLP's rescale (GAIN)."""

    GAIN = 2.0

    def forward(self, x):
        return x * (1.0 + torch.tanh(0.5 * x))


class Descale(torch.nn.Module):
    """inner(x) * inv — undoes a constraint-E rescale before the residual add.
    An explicit multiply, not folded into the weights: dividing the weights
    by up to 256 would push small ones into fp16's subnormal range."""

    def __init__(self, inner, inv):
        super().__init__()
        self.inner = inner
        self.inv = float(inv)

    def forward(self, x):
        return self.inner(x) * self.inv


class ScaledMLP(torch.nn.Module):
    """SwiGLU with the tanh silu, w3 pre-scaled by m and the output descaled."""

    def __init__(self, mlp, m):
        super().__init__()
        self.w1, self.w3, self.w2 = mlp.w1, mlp.w3, mlp.w2
        with torch.no_grad():
            self.w3.weight.mul_(m)
        self.act = TanhSilu()
        self.inv = 1.0 / (TanhSilu.GAIN * m)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x)) * self.inv


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


def calibrate(model, tokenizer):
    """fp32 stats (unpadded forwards) of every tensor constraint E rescales."""
    stats = {}

    def stat(name):
        return stats.setdefault(name, _Stat())

    def pre(name):
        return lambda mod, args: stat(name).add(args[0])

    def post(name):
        return lambda mod, args, out: stat(name).add(out)

    def conv_parts(i):
        def f(mod, args, out):
            B, C, x = out.detach().chunk(3, dim=-1)
            stat(f"{i}.bx").add(B * x)
        return f

    hooks = []
    for i, layer in enumerate(model.layers):
        if layer.is_attention_layer:
            a = layer.self_attn
            hooks += [a.q_proj.register_forward_pre_hook(pre(f"{i}.qkv_in")),
                      a.q_proj.register_forward_hook(post(f"{i}.q")),
                      a.k_proj.register_forward_hook(post(f"{i}.k")),
                      a.out_proj.register_forward_pre_hook(pre(f"{i}.o_in")),
                      a.out_proj.register_forward_hook(post(f"{i}.o_out"))]
        else:
            c = layer.conv
            hooks += [c.in_proj.register_forward_pre_hook(pre(f"{i}.in")),
                      c.in_proj.register_forward_hook(conv_parts(i)),
                      c.out_proj.register_forward_pre_hook(pre(f"{i}.y")),
                      c.out_proj.register_forward_hook(post(f"{i}.y_out"))]
        f = layer.feed_forward
        hooks += [f.w2.register_forward_pre_hook(pre(f"{i}.d_in")),
                  f.w2.register_forward_hook(post(f"{i}.d_out"))]
    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            model(**tokenizer(text, return_tensors="pt"))
    for h in hooks:
        h.remove()
    return stats


def input_scale(rec, out=None, gain=1.0):
    """Power-of-two scale bringing rec (arriving scaled by gain) to rms ~1,
    within fp16 headroom for the input and, if given, the output. >= 1."""
    s = pow2(1.0 / (gain * rec.rms))
    while s > 1.0 and (rec.max * gain * s > IN_MAX
                       or (out is not None and out.max * gain * s > OUT_MAX)):
        s /= 2.0
    return max(s, 1.0)


def precision_rewrite(model, stats):
    """Constraint E. Every factor is a power of two, so the fp32 graph is
    unchanged (fp32_gate checks). Returns the scales, for the log."""
    scales = []
    for i, layer in enumerate(model.layers):
        if layer.is_attention_layer:
            a = layer.self_attn
            # q/k/v inputs: scale the norm feeding them. q and k are
            # re-normalized per head, so only their RMSNorm eps moves (x s^2).
            s_in = input_scale(stats[f"{i}.qkv_in"])
            qk = max(stats[f"{i}.q"].max, stats[f"{i}.k"].max)
            while s_in > 1.0 and qk * s_in > QK_MAX:
                s_in /= 2.0
            # out_proj input: v arrives scaled by s_in; v_proj adds the rest
            s_o = input_scale(stats[f"{i}.o_in"], stats[f"{i}.o_out"])
            with torch.no_grad():
                layer.operator_norm.weight.mul_(s_in)
                a.v_proj.weight.mul_(s_o / s_in)
            a.q_layernorm.variance_epsilon *= s_in * s_in
            a.k_layernorm.variance_epsilon *= s_in * s_in
            a.out_proj = Descale(a.out_proj, 1.0 / s_o)
            scales.append(f"L{i} attn {s_in:g}/{s_o:g}")
        else:
            c = layer.conv
            h = c.out_proj.weight.shape[0]
            # in_proj input via the norm; B*x (the conv's input) via B's rows;
            # y = C * conv(Bx) (out_proj's input) via C's rows. A row factor
            # below 1 is still an exact power-of-two weight scale.
            s_in = input_scale(stats[f"{i}.in"])
            s_bx = input_scale(stats[f"{i}.bx"])
            s_y = input_scale(stats[f"{i}.y"], stats[f"{i}.y_out"])
            with torch.no_grad():
                layer.operator_norm.weight.mul_(s_in)
                c.in_proj.weight[:h].mul_(s_bx / (s_in * s_in))
                c.in_proj.weight[h:2 * h].mul_(s_y / (s_in * s_bx))
            c.out_proj = Descale(c.out_proj, 1.0 / s_y)
            scales.append(f"L{i} conv {s_in:g}/{s_bx:g}/{s_y:g}")
        m = input_scale(stats[f"{i}.d_in"], stats[f"{i}.d_out"], TanhSilu.GAIN)
        layer.feed_forward = ScaledMLP(layer.feed_forward, m)
        scales[-1] += f" mlp {TanhSilu.GAIN * m:g}"
    return scales


class ClsWrapper(torch.nn.Module):
    """CLS (= BOS, position 0) pooling + L2 normalize, inside the graph."""

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
        cls = hidden[:, 0, :].reshape(1, DIMS) * CLS_SCALE
        return cls / torch.linalg.vector_norm(cls, dim=-1, keepdim=True)


def reference_embeddings(model, tokenizer):
    """fp32 CLS+L2 references, verified equal to SentenceTransformer.encode
    (cosine 1.000000) — same tokenizer.json postprocessor adds BOS, so the
    daemon's Rust tokenization matches this pipeline exactly."""
    refs = []
    with torch.no_grad():
        for s in PARITY_SENTENCES:
            enc = tokenizer(DOC_PREFIX + s, return_tensors="pt")
            h = model(**enc).last_hidden_state[:, 0, :]
            refs.append(torch.nn.functional.normalize(h, dim=-1)[0].numpy())
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


def fitting_pairs(tokenizer, refs, seq_len):
    """(text, ref) pairs whose token count fits the bucket — the long text
    only participates in buckets it fits (512)."""
    pairs = []
    for s, ref in zip(PARITY_SENTENCES, refs):
        n = len(tokenizer(DOC_PREFIX + s, add_special_tokens=True)["input_ids"])
        if n <= seq_len:
            pairs.append((s, ref))
    return pairs


def fp32_gate(wrapper, tokenizer, refs, seq_len, gate_failures=True):
    """Constraint E must be ~exact in fp32 before we spend on conversion. The
    negative control leaks pads into the padded forward, so it only reports."""
    cosines = []
    with torch.no_grad():
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, DOC_PREFIX + s, seq_len)
            out = wrapper(torch.from_numpy(ids), torch.from_numpy(mask))[0].numpy()
            cosines.append(cosine(ref, out))
    worst = worst_of(cosines)
    if not worst >= 0.99999:  # NaN fails too
        message = f"seq {seq_len}: fp32 rewrite parity {worst:.7f} < 0.99999"
        if gate_failures:
            raise SystemExit(message)
        print(f"negative control, expected: {message}")
    return worst


def convert_bucket(wrapper, seq_len, workdir):
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0] = 1  # <|startoftext|>
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
    # Explicit attention (D25) and the explicit silu (constraint E) must
    # survive conversion: coremltools fuses attention into its
    # scaled_dot_product_attention op when the torch call has no explicit
    # scale, and x * sigmoid(x) comes out as its native silu op.
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    fused = ops & {"scaled_dot_product_attention", "silu", "gelu"}
    if fused:
        raise SystemExit(f"seq {seq_len}: converted graph contains {sorted(fused)} — "
                         "see constraint E and docs/DECISIONS.md D25")
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs, gate_failures=True):
    """Cosine vs the fp32 reference on BOTH Espresso compute paths — a pass
    under .ALL alone hides fp16/plan-compilation failures (D17 constraint 9).
    A negative control reports failures instead of stopping on them."""
    def fail(message):
        if gate_failures:
            raise SystemExit(message)
        print(f"negative control, expected: {message}")

    results = {}
    # CPU_ONLY >= 0.999 proves the conversion is faithful; CPU_AND_NE >= 0.999
    # that the ANE runs it at full precision (constraint E)
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.999),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        cosines = []
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, DOC_PREFIX + s, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                fail(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            cosines.append(cosine(ref, out))
        worst = worst_of(cosines)
        if not worst >= gate:  # NaN fails too
            fail(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        # Pad invariance (D25): with constraint D, pad content can't reach the
        # real tokens, so it must not change the output.
        ids, mask = padded_inputs(tokenizer, DOC_PREFIX + PARITY_SENTENCES[0], seq_len)
        noisy = ids.copy()
        pads = mask[0] == 0
        noisy[0, pads] = np.random.default_rng(0).integers(1000, 40000, int(pads.sum()))
        a = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
        b = m.predict({"input_ids": noisy, "attention_mask": mask})["embedding"][0]
        if not cosine(a, b) >= 0.99999:
            fail(f"seq {seq_len} [{label}]: output depends on pad content "
                 f"(cos {cosine(a, b):.6f}) — see constraint D")
        # latency, as an ANE-residency proxy (the real gate is ane_check)
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
    args = sys.argv[1:]
    negative_control = args[:1] == ["--no-pad-zeroing"]
    if negative_control:
        args = args[1:]
        print("NEGATIVE CONTROL: convs read pad states (constraint D skipped)")
    src = Path(args[0]).expanduser()
    install_dir = Path(args[1]).expanduser()
    buckets = [int(b) for b in args[2:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    assert tokenizer.pad_token_id == 0, "server pads input_ids with 0"
    # trust_remote_code: modeling_lfm2_bidirectional.py — read it first.
    model = AutoModel.from_pretrained(
        src, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    model.config.use_cache = False
    # constraint D's exactness needs bias-free in_proj (zeroed pad states
    # must map to zero, matching F.conv1d's zero edge padding); upstream
    # keys the in_proj/conv biases off config.conv_bias. False for this
    # checkpoint — assert in case a future LFM variant flips it.
    assert not getattr(model.config, "conv_bias", False), \
        "conv_bias=true would break constraint D's pad-zeroing exactness"
    # after load — see install_patches() ordering note
    install_patches(conv_pad_zeroing=not negative_control)
    # the long parity text must run in the 512 bucket (see PARITY_SENTENCES)
    longest = max(len(tokenizer(DOC_PREFIX + s, add_special_tokens=True)["input_ids"])
                  for s in PARITY_SENTENCES)
    if not 256 < longest <= 512:
        raise SystemExit(f"the long parity text is {longest} tokens; it must fit the 512 bucket")

    refs = reference_embeddings(model, tokenizer)
    print("calibrating fp32 activation ranges...")
    scales = precision_rewrite(model, calibrate(model, tokenizer))
    print("constraint E scales (attn in/out, conv in/Bx/y, mlp):")
    for line in scales:
        print(f"  {line}")

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = ClsWrapper(model, seq).eval()
            f32 = fp32_gate(wrapper, tokenizer, refs, seq, gate_failures=not negative_control)
            print(f"bucket {seq}: fp32 rewrite parity {f32:.7f}")
            pkg = convert_bucket(wrapper, seq, workdir)
            res = parity_check(tokenizer, pkg, seq, refs, gate_failures=not negative_control)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            for label, (cos, ms) in res.items():
                print(f"bucket {seq} [{label}]: parity cos={cos:.6f} {ms:.1f}ms")
            print(f"bucket {seq} -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = (
        Path(__file__).resolve().parent.parent
        / "examples/manifests/lfm2.5-embedding-350m/manifest.toml"
    )
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
