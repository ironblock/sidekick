"""Convert LiquidAI/LFM2.5-Embedding-350M into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling and L2 normalization baked into the graph, matching
examples/manifests/lfm2.5-embedding-350m.

Usage:
    python tools/convert_lfm25_embedding.py <hf-model-dir> <install-dir> [buckets...]

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
properties make it the EASY class of conversion (bge-class, not gemma-class):

- QK-norm keeps fp32 activations tiny (calibrated max ~25 across every
  module on the calibration set) — no D17 range rewrite needed, fp16 is
  simply safe.
- CLS pooling: no mask-aware mean, no sliding-window band masks.

The recipe still encodes the hardware-verified constraints from
tools/convert_bge_small.py (D15: per-bucket STATIC shapes, pooling inside
the graph, SDPA, explicit position_ids) plus four LFM2.5-specific ones:

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

Parity is gated per compute path (D17 constraint 9): CPU_ONLY >= 0.999
proves the conversion is faithful; CPU_AND_NE >= 0.985 is what ANE fp16
accumulation delivers on deep models. Measured on M-series hardware:
CPU_ONLY 0.9999 (varies in the 5th decimal per bucket), CPU_AND_NE
0.987010 with the worst-case ANE cosine agreeing to all six printed
decimals across buckets — constraint D's bucket-invariance holding on
the path where the padding leak was measured (see docs/MODELS.md).
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
QUERY_PREFIX = "query: "
DOC_PREFIX = "document: "

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    # ~480 tokens: exercises long-sequence fp16 accumulation and the full
    # RoPE position range. Short sentences alone under-test the 512 bucket.
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fitting_pairs(tokenizer, refs, seq_len):
    """(text, ref) pairs whose token count fits the bucket — the long text
    only participates in buckets it fits (512)."""
    pairs = []
    for s, ref in zip(PARITY_SENTENCES, refs):
        n = len(tokenizer(DOC_PREFIX + s, add_special_tokens=True)["input_ids"])
        if n <= seq_len:
            pairs.append((s, ref))
    return pairs


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
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs):
    """Cosine vs the fp32 reference on BOTH Espresso compute paths — a pass
    under .ALL alone hides fp16/plan-compilation failures (D17 constraint 9)."""
    results = {}
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.985),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        worst = 1.0
        for s, ref in fitting_pairs(tokenizer, refs, seq_len):
            ids, mask = padded_inputs(tokenizer, DOC_PREFIX + s, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            worst = min(worst, cosine(ref, out))
        if worst < gate:
            raise SystemExit(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
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
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
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
    install_patches()  # after load — see install_patches() ordering note

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
        / "examples/manifests/lfm2.5-embedding-350m/manifest.toml"
    )
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
