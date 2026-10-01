"""Convert google/embeddinggemma-300m into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket with the FULL
sentence-transformers stack baked into the graph: Gemma3 encoder -> mask-aware
mean pooling -> Dense 768->3072 -> Dense 3072->768 -> L2 normalize, output
`embeddings`, statically (1, 768). Matches examples/manifests/embeddinggemma-300m.

Usage:
    python tools/convert_embeddinggemma.py <hf-model-dir> <install-dir> [buckets...] [--time]

    hf-model-dir: local snapshot of google/embeddinggemma-300m (the full
                  sentence-transformers repo: config.json, model.safetensors,
                  tokenizer.json, 2_Dense/, 3_Dense/)
    install-dir:  model directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/embeddinggemma-300m"
    buckets:      default 128 256 512

Requires: torch, transformers>=4.57 (gemma3_text), tokenizers, coremltools,
numpy, safetensors (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the Gemma3
backbone (backbones/gemma3.py: constraints 5, 8 and 9) with the
sentence-transformers head (heads/gemma_st.py: constraints 6 and 7).
Calibration keeps one graded parity-corpus text, by a logged exemption (see
main()).

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

from safetensors.torch import load_file

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import gemma3
from sidekick_convert.gates import EmbeddingGates
from sidekick_convert.heads.gemma_st import MeanDenseL2

MODEL_ID = "embeddinggemma-300m"
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


def main():
    args = cli.parse(__doc__.split("\n\n")[0])
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    from transformers import AutoTokenizer
    if AutoTokenizer.from_pretrained(args.src).pad_token_id != 0:
        raise SystemExit("the server pads input_ids with 0, and this tokenizer's pad id isn't 0")
    backbone = gemma3.load(args.src, tok)
    # constraint 6: wherever the sliding band is live, a parity text must reach it
    window = backbone.window
    longest = max(len(tokenizer.encode(tok, DOC_PREFIX + s)) for s in PARITY_SENTENCES)
    for seq in args.buckets:
        if seq > window and not window < longest <= seq:
            raise SystemExit(f"bucket {seq}: no parity text longer than the sliding window "
                             f"({window}) fits (longest is {longest} tokens)")
    dense1_w = load_file(args.src / "2_Dense/model.safetensors")["linear.weight"].float()
    dense2_w = load_file(args.src / "3_Dense/model.safetensors")["linear.weight"].float()
    # Calibration keeps one graded parity-corpus text ("A cat sat on the mat.",
    # behind the document prefix): without it, two power-of-two scales move
    # (layer 14's MLP input scale 32 -> 16, layer 8's pre-feedforward norm
    # 0.5 -> 1.0) and the graded artifact would change. Dropping it is a
    # separate, measured change with a parity re-grade.
    calibration = core.Calibration(CALIBRATION_TEXTS, legacy_graded=(
        "EmbeddingGemma's graded artifact was calibrated with it; dropping it moves two scales"))
    job = recipes.embedder(
        model_id=MODEL_ID, src=args.src, buckets=args.buckets, backbone=backbone,
        head=MeanDenseL2(dense1_w, dense2_w), tok=tok, texts=[DOC_PREFIX + s for s in PARITY_SENTENCES],
        calibration=calibration, rewrites=[gemma3.fp16_range_rewrite(calibration, tok)],
        # the eps floor (constraint 5) perturbs exactness slightly, by design
        gates=EmbeddingGates(fp32_min_cos=0.9999, pad_id_range=(1000, 40000)),
        strict_max_seq_len=False, timing=args.time, **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
