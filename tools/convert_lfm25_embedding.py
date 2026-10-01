"""Convert LiquidAI/LFM2.5-Embedding-350M into ANE-resident Core ML artifacts.

Produces one static-shape .mlmodelc per sequence-length bucket, with CLS
pooling and L2 normalization baked into the graph, matching
examples/manifests/lfm2.5-embedding-350m.

Usage:
    python tools/convert_lfm25_embedding.py <hf-model-dir> <install-dir> [buckets...] [--time]
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

Requires: torch, transformers >= 4.55 (Lfm2 support), tokenizers,
coremltools, numpy (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The recipe (tools/sidekick_convert; docs/CONVERTING.md) is the LFM2 backbone
(backbones/lfm2.py, which carries constraints A, B, D and E below) with a
CLS pooling head that L2-normalizes in-graph (constraint C). Calibration for
constraint E uses the texts below minus any the graded parity corpus holds.

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

from sidekick_convert import cli, core, recipes, tokenizer
from sidekick_convert.backbones import lfm2
from sidekick_convert.heads.pool import Pool

MODEL_ID = "lfm2.5-embedding-350m"
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


def main():
    args = cli.parse(__doc__.split("\n\n")[0], flags=[
        ("--no-pad-zeroing", {"action": "store_true",
                              "help": "negative control: skip constraint D (convs read pad states)"}),
    ])
    negative_control = args.no_pad_zeroing
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="verbatim"))
    from transformers import AutoTokenizer
    if AutoTokenizer.from_pretrained(args.src).pad_token_id != 0:
        raise SystemExit("the server pads input_ids with 0, and this tokenizer's pad id isn't 0")
    # the long parity text must run in the 512 bucket (see PARITY_SENTENCES)
    longest = max(len(tokenizer.encode(tok, DOC_PREFIX + s)) for s in PARITY_SENTENCES)
    if not 256 < longest <= 512:
        raise SystemExit(f"the long parity text is {longest} tokens; it must fit the 512 bucket")
    backbone = lfm2.load(args.src, tok, conv_pad_zeroing=not negative_control)
    calibration = core.Calibration.without_graded(CALIBRATION_TEXTS)
    job = recipes.embedder(
        model_id=MODEL_ID, src=args.src, buckets=args.buckets, backbone=backbone, head=Pool("cls", l2=True),
        tok=tok, texts=[DOC_PREFIX + s for s in PARITY_SENTENCES], calibration=calibration,
        rewrites=[lfm2.precision_rewrite(calibration, tok)], strict_max_seq_len=False,
        negative_control=negative_control, timing=args.time, **cli.job_options(args))
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
