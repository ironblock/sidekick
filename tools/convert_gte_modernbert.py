"""Convert Alibaba-NLP/gte-modernbert-base into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling baked into the graph, matching examples/manifests/gte-modernbert-base.

Usage:
    python tools/convert_gte_modernbert.py <hf-model-dir> <install-dir> [buckets...] [--time]
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

Requires: torch, transformers >= 4.48 (native ModernBERT), tokenizers,
coremltools, numpy (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the ModernBERT
backbone with a CLS pooling head. What it handles, and why:

A. EXPLICIT ATTENTION, NOT THE FUSED SDPA OP (docs/DECISIONS.md D25). With
   sdpa, Core ML's fused attention op drops this model's mask on the Neural
   Engine: transformers builds ModernBERT's masks before the CPU-only
   embedding gather, so they are computed on the CPU and reach the ANE's
   attention as an input, which the ANE's fused attention ignores. Pads are
   then attended and the sliding window vanishes (parity 0.87-0.975,
   matching an unmasked reference at 0.99998), the output depends on pad
   content (0.61-0.94), and the same op on the CPU returns NaN whenever fewer
   than 64 of 128 positions are real. This file once documented ModernBERT as
   ANE-incompatible, blaming its massive activation; that was this bug
   misdiagnosed. tools/repro_sdpa_mask.py reproduces it standalone.
B. FINITE MASKS. Both of transformers' masks (global, and the +-64 sliding
   window) are rebuilt with -30000 instead of finfo.min, the same geometry.
C. TRACEABLE RoPE (per-layer-type theta: global 160000, local 10000); stock
   rotate_half's shape arithmetic crashes coremltools under static shapes.
   Eager attention keeps full static shapes; position_ids is a buffer.
D. RESIDUAL RANGE REWRITE (D25 amendment). The ANE's linear op saturates
   above 2^15 = 32,768. ModernBERT's massive activation (dimension 251 on
   delimiter tokens, ~48,000 in the residual) is written by layer 15's MLP
   output projection at up to ~50,000, so the residual stream runs at 1/K
   (exact in fp32). K is the smallest power of two keeping every calibrated
   linear output under 0.85 x 2^15: K = 2, with ~1.3x headroom over the
   largest calibrated output. Larger K costs precision on both the ANE and
   the CPU path (K = 4: ANE 0.99984, CPU 0.99886). Calibration uses the
   texts below minus any the graded parity corpus holds; K is the same
   either way.

Pooling: raw CLS reshaped to a literal (1, dims), no in-graph L2 (the server
normalizes in f32; |CLS| ~ 22 over 768 dims would overflow an fp16 sum of
squares). fp16 range: the residual stream peaks at ~48,000, under fp16's
65,504, so no fp16 range rewrite; constraint D is about the ANE linear's
narrower range.

Gates, per bucket: fp32 exactness of the rewritten wrapper (cosine >=
0.99999); no fused attention op; compute plan; parity >= 0.999 on
CPU_AND_NE and CPU_ONLY; finite output; pad invariance. Every gate treats
NaN as a failure. Measured results in docs/MODELS.md.
"""

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import modernbert
from sidekick_convert.heads.pool import Pool
from sidekick_convert.techniques import saturation

# tools/convert_laya.py imports these from this module; they go when laya's
# converter moves onto the library.
install_patches = modernbert.install_patches
range_rewrite = modernbert.residual_rewrite_encoder
ANE_LINEAR_MAX = saturation.ANE_LINEAR_MAX
LINEAR_HEADROOM = saturation.HEADROOM

MODEL_ID = "gte-modernbert-base"

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


def main():
    args = cli.parse(__doc__.split("\n\n")[0], flags=[
        ("--attn", {"choices": ["eager", "sdpa"], "default": "eager",
                    "help": "sdpa builds the fused-attention negative control"}),
    ])
    negative_control = args.attn == "sdpa"
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    longest = max(len(tokenizer.encode(tok, s)) for s in PARITY_SENTENCES)
    if not 128 < longest <= 512:
        raise SystemExit(f"the long parity text is {longest} tokens; it must fit the 512 bucket")
    backbone = modernbert.load(args.src, tok, attention="fused" if negative_control else "explicit")
    calibration = core.Calibration.without_graded(CALIBRATION_TEXTS)
    job = recipes.embedder(
        model_id=MODEL_ID, src=args.src, buckets=args.buckets, backbone=backbone, head=Pool("cls"), tok=tok,
        texts=PARITY_SENTENCES, calibration=calibration,
        rewrites=[modernbert.residual_k(calibration, tok)],
        forbid_ops=frozenset() if negative_control else frozenset({core.FUSED_ATTENTION}),
        negative_control=negative_control, timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
