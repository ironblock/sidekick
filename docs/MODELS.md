# Model compatibility registry

What runs through sidekick's two embedding backends, what doesn't, and how
to tell before spending an afternoon finding out: Core ML on the ANE, GPU
and CPU (Apple Silicon), and ONNX Runtime on the CPU of any supported
platform ([onnx-cpu](#onnx-runtime-on-the-cpu-onnx-cpu)). Every entry here
was measured on real hardware: the Core ML entries on Apple Silicon (macOS
26; ANE eligibility re-checked on macOS 27), each onnx-cpu entry on the
machines its section names. Nothing is extrapolated from model cards.

How confident to be in each model on each compute path, graded on inputs
chosen to break it, is in [Confidence grades](#confidence-grades-the-parity-suite)
below. That is the number to trust; the converter parity in the first table
is measured on short prose and can flatter a model (it did for LFM2.5 before
its precision rewrite).

A sortable, filterable view of these results, by model, path, precision,
chip and runtime, is at
[ironblock.github.io/sidekick](https://ironblock.github.io/sidekick/),
built from [docs/data/models.json](data/models.json) by
`tools/models_table.py`. The same script writes this document's data
tables (marked `generated` in the source) from that file, and CI checks
that the two agree: to correct a number, correct its input in docs/data
and rerun the script.

Method, for every validated entry:
- **parity** — worst-case cosine between the Core ML artifact and the fp32
  torch/sentence-transformers reference over the converter's parity set
  (short pairs + a ~400-token text), reported per compute path (D17):
  `CPU_ONLY` proves the conversion is faithful (gate ≥ 0.999), `CPU_AND_NE`
  is what the ANE delivers on those inputs (gate ≥ 0.985; the
  embeddinggemma converter gates ≥ 0.999 since its MLP precision rewrite,
  D17). Check that the long text really fits the bucket: an over-length
  parity text is silently skipped.
- **grade** — the parity suite (D26): every compute path, through sidekick's
  own product code, on a 51-input adversarial corpus.
- **ANE eligibility** (`cargo run -p sidekick-coreml --example ane_check`),
  the verdict: Core ML's compute plan for `.cpuAndNeuralEngine`, i.e. which
  device each operation is assigned to. It never runs the model and is
  unaffected by machine load. A model passes when every compute-heavy
  operation (matmul, linear, conv, attention) is on the ANE and at least 80%
  of assigned operations are. The table's "ANE ops" column is that share.
  The plan is the compiler's intent, so it can't see failures that only
  happen at run time. That is what the ratio below is for.
- **pad invariance** (same tool), a gate: a half-full input is run with pad
  ids 0 and with random pad ids; the attention mask hides the pads, so the
  output must be identical (cosine ≥ 0.99999) and finite on both the ANE and
  CPU paths. It catches a dropped attention mask, which neither the compute
  plan nor a parity set of short texts reliably exposes. Core ML's fused
  attention op dropped ModernBERT's mask on the ANE while every op sat on
  the ANE (see gte-modernbert-base below).
- **residency ratio**, runtime evidence, reported by the same tool: the
  median-latency ratio of `.cpuOnly` over `.cpuAndNeuralEngine` per bucket.
  A ratio near 1.0 on an eligible model points at a runtime fallback, so
  `ane_check` warns below 1.1x. It is not a pass/fail gate, for two reasons:
  - It moves with things other than residency. Concurrent GPU and
    memory-bandwidth load depresses and destabilizes it: the same bge
    artifact under an active MLX workload swung 1.4x–2.8x between runs.
  - A faster CPU path shrinks it on a model that is fully on the ANE. On
    macOS 27, bge-small's 512 bucket measures ~1.25x (it was 1.75x on 26.5)
    with a compute plan identical to its 3x buckets. The old 1.5x ratio gate
    called that bucket "not resident".
  Parity is insensitive to load. Judge ratios only on a quiet machine, and
  judge eligibility by the plan.

## Validated on ANE

| model | dims | pooling | conversion | parity CPU_ONLY | parity ANE | ANE ops (macOS 27) | residency ratio, macOS 26.5 (128/256/512) |
|---|---|---|---|---|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | CLS | [convert_bge_small.py](../tools/convert_bge_small.py) | 0.999972 | 0.999984 | 229/245 (93.5%) | 3.4x / 2.4x / 1.75x |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 768 (MRL 512/256/128) | mean | [convert_embeddinggemma.py](../tools/convert_embeddinggemma.py) | 0.99993 | 0.99999 | 2161/2170 (99.6%) | 3.4x / 3.1x / 2.9x (before the MLP rewrite) |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | 768 (MRL 512/256/128) | mean | [convert_embeddinggemma2.py](../tools/convert_embeddinggemma2.py) | 0.99993 | 0.999995 | 2756/2765 (99.7%) | not measured yet (buckets 128 to 2048) |
| [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | 1024 | CLS | [convert_lfm25_embedding.py](../tools/convert_lfm25_embedding.py) | 0.99991 | 0.99999 | 773/778 (99.4%) | 2.49x / 1.91x / 1.66x (before the precision rewrite) |
| [codefuse-ai/F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 640 | last-token | [convert_qwen3_embedding.py](../tools/convert_qwen3_embedding.py) | 0.99992 | 0.99998 | 648/653 (99.2%) | 2.02x / 1.77x / 1.59x (before the precision rewrite) |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 768 | CLS | [convert_gte_modernbert.py](../tools/convert_gte_modernbert.py) | 0.99992 | 0.99998 | 794/805 (98.6%) | validated on macOS 27 (below) |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | mean | [convert_bert_embedder.py](../tools/convert_bert_embedder.py) | 0.999982 | 0.999971 | 146/168 (86.9%) | not measured yet (buckets 64/128/256) |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | 384 | mean | [convert_bert_embedder.py](../tools/convert_bert_embedder.py) | 0.999974 | 0.999963 | 290/312 (92.9%) | not measured yet |

ANE ops are identical at every bucket. On every model, the operations off
the ANE are mask and cast plumbing plus the embedding `gather` (e.g. bge:
add, cast, expand_dims, gather, greater_equal, layer_norm, select, sub,
tile). Nothing compute-heavy is placed off the ANE; bge's fused attention
ops get no device at all and run correctly through a Core ML fallback (D25).
Ratios re-measured on macOS 27
(M1 Max, not a quiet machine):

| model | 128 | 256 | 512 |
|---|---|---|---|
| bge-small | 3.0x | 2.0x | 1.26x |
| embeddinggemma | 2.6x | 2.4x | 1.7x |
| LFM2.5 | 2.3x | 1.9x | 1.6x |
| F2LLM | 2.7x | 2.0x | 1.5x |
| gte-modernbert | 2.9x | 2.0x | 1.6x |

Notes per model:

- **bge-small-en-v1.5** — the reference "easy" conversion: BERT-class,
  attention-only, small activations. The recipe (D15) is the template for
  MiniLM/gte/e5-class encoders. Parity re-measured per-path over the shared
  parity set (July 2026, artifacts regenerated); residency from the D15
  quiet-machine measurement of the same recipe. Loads may print one
  `ANECCompile() FAILED` line on stderr while the encoder still resides on
  the ANE (a small ineligible segment). Judge by the ane_check compute plan,
  not stderr. `convert_bge_small.py --enumerated-shapes` builds the
  flexible-shape artifact that D15 rules out, as a negative control:
  `ane_check` must reject it, and does (0 of 362 operations on the ANE).
  On macOS 27, `CoremlModel::load` refuses it too (D27).
  Its fused attention is correct only through Core ML's fallback: the
  compute plan gives the 12 attention ops no device, and at the iOS26 opset
  the same graph fails to load on CPU_AND_NE (D25). Keep the macOS15 target.
- **embeddinggemma-300m** — the "hard" conversion: needed a calibrated fp16
  range rewrite, hand-built sliding-window band masks, traceable
  rotate_half/repeat_kv, and an MLP precision rewrite (D17). Its ANE parity
  was 0.9905 (0.981 on a 394-token text) until September 2026, documented
  as intrinsic fp16 accumulation. It wasn't. The ANE's `linear` op loses
  precision on small inputs (Gemma's late-layer down projections see rms
  ~0.005), and Core ML's native `gelu` is coarse on the ANE. The
  converter now rescales each down_proj input to rms ~1 with a
  power-of-two scale that the post-feedforward RMSNorm cancels exactly,
  and builds GELU from tanh. The ANE path is now more accurate than
  CPU_ONLY. The explicit GELU costs ~14% latency at bucket 512 and nothing
  at 128. Ratios in the table's last column predate the rewrite (re-measured
  on macOS 27 above). Live `/v1/embeddings` worst parity 0.999982 over 13
  inputs, including a 527-token document truncated to 512. The parity
  suite grades it A on the ANE (below; it was D before the rewrite).
  ~590 MB per bucket.
- **embeddinggemma-2** — the text tower of a Gemma 4 based embedder, with
  per-layer inputs and buckets to 2,048 tokens. Its residual stream stays
  under ~200, but a written-out RMSNorm squares its input, and several
  norm inputs pass 256 (the per-layer-input norm reaches ~1,600). Converted
  straight, the GPU and ANE paths were unharmed (0.999999 and 0.99994) and
  Core ML's CPU lost 5% (0.947). Power-of-two pre-scales on every RMSNorm,
  GELU from tanh, and a per-bucket pooling scale fix it: the parity suite
  grades the ANE and GPU A. The CPU path is as accurate (0.99993) but fails
  bucket invariance, because Core ML's CPU softmax rounds differently past
  512 keys; the matmul softmax that fixes it broke the ANE at bucket 256.
  Converted with transformers 5.19 (docs/CONVERTING.md). ~540 MB per bucket.
- **LFM2.5-Embedding-350M** — the first hybrid (10 short-conv + 6
  full-attention blocks) and the model that motivated conversion
  constraint D: symmetric convs mix neighbors regardless of attention
  mask, so pad states must be zeroed before every conv or right-padding
  contaminates real tokens (measured 0.905 parity without the fix, 0.987
  with it — the worst-case ANE cosine agrees to all six printed decimals
  across buckets, the fix's bucket-invariance holding on the path where
  the leak was measured; CPU parity varies in the 5th decimal).
  QK-norm keeps activations tiny (max ~25),
  so no range rewrite is needed despite the model being deeper than bge.
  Ships custom code (`modeling_lfm2_bidirectional.py`, ~140 benign lines —
  read before trusting). ~670 MB per bucket, 2.0 GB installed.
  **Precision rewrite (D19 amendment).** Its ANE parity was 0.987 on prose
  and 0.954 on a 39-token URL (grade D), with similarity scores moving by
  up to 0.19. The cause was the same as EmbeddingGemma's, spread over every
  sub-block. Tiny activations meant every output projection's input had
  rms 0.003–0.07, below the ANE `linear`'s precision floor, and Core ML's
  native silu is coarse on the ANE. The converter now rescales each of
  those inputs to rms ~1 with power-of-two scales. Because no norm follows
  the conv, attention or MLP branch, each is undone by an explicit multiply
  before the residual add. It also builds SiLU from tanh. The ANE now
  grades A (0.99999, drift 0.003, no rank flips) and tracks fp32 more
  closely than CPU_ONLY does, at 13–16% more ANE latency. Live
  `/v1/embeddings` worst parity is 0.999988 over the suite's 51 inputs.
  Ratios in the table's last column predate the rewrite (re-measured on
  macOS 27 above).
- **F2LLM-v2-160M** — the first **causal decoder** and first **last-token
  pooling** on the stack. A Qwen3 decoder; its QK-norm keeps activations
  tiny (max ~420), so it converts as cleanly as bge (ANE parity 0.99985).
  Last-token pooling is baked in-graph via the attention mask (no
  data-dependent index): `last_onehot = mask · (1 − shift_left(mask))`, then
  a masked sum. Validating it surfaced and fixed a real server bug: naive
  `take(max)` truncation dropped the trailing EOS that last-token pooling
  reads, collapsing over-length-doc parity to 0.36 — the server now
  preserves the final token on truncation (harmless for CLS/mean). About 920
  MB installed (three buckets of about 305 MB each, and the tokenizer) for a
  640-dim decoder. **Precision rewrite (D20 amendment).** It graded B on the
  ANE (0.99966 on a run of digits), mostly from Core ML's coarse native
  SiLU. Its activations are about ten times LFM2.5's, so only attention's
  inputs in the early layers fell below the ANE `linear`'s precision floor,
  and they mattered only for long texts. The converter builds SiLU from tanh
  and rescales attention's inputs. The ANE now grades A (0.99997, drift
  0.002, no rank flips) at unchanged latency. A second step (D39) builds
  SiLU from exp instead (StableSilu, which doesn't cancel in fp16 for
  negative inputs) and rescales down_proj's input too: the ANE's worst case
  moves to 0.999993 and its drift to 0.0008, a quarter of the distance from
  fp32, for about 0.7 ms more per input (4.6 → 5.3 ms). Artifacts converted
  before it keep working; reconvert for the closer ANE output. The CPU path
  stays B (0.99988), which is its own error. The converter had also stopped
  running under torch 2.13: its repeat_kv traced to an Int op coremltools
  can't convert, now replaced.

- **gte-modernbert-base** — validated September 2026, on macOS 27, after
  being documented ANE-incompatible (D20, D25). ModernBERT alternates
  sliding-window and global attention and has per-layer-type RoPE (see the
  converter). **Convert attention explicitly** (`attn_implementation="eager"`,
  i.e. matmul → softmax → matmul). With `sdpa`, Core ML's fused
  `scaled_dot_product_attention` op drops the attention mask on the ANE,
  because ModernBERT's masks are built on the CPU (see the checklist rule):
  - pads are attended and the sliding window vanishes (parity 0.87–0.975,
    matching an *unmasked* reference at 0.99998);
  - the output changes with pad content (pad ids 0 vs random: 0.61–0.94);
  - the same op on the CPU returns NaN below 64 of 128 real tokens. A pad
    query's whole ±64 window is then padding, and -30000 × √64 overflows
    fp16.

  The earlier diagnosis blamed ModernBERT's massive activation (dimension
  251, ~48,000 on delimiter tokens) crushing fp16 precision. It doesn't:
  with explicit attention the same activations convert at 0.9998. They
  leave a small per-token effect: the lowest per-token cosines, 0.98–0.99
  on the ANE, sit on those tokens' own output vectors. Pooled CLS isn't
  affected.

  **Range rewrite (D25 amendment).** The massive activation mattered in
  another way. Layer 15's MLP output projection writes it at
  35,000–51,500, and the ANE's `linear` saturates above 2^15 = 32,768 (an
  output of 33,000 comes back inf). The artifact before the fix carried
  -inf in those tokens' residual inside the ANE graph, and graded B
  (0.99940) only because downstream ops saturate. The converter now runs
  the residual stream at 1/K, with K = 2 chosen from calibration to keep
  every linear output under 0.85 × 2^15 (1.31x headroom). That is exact in
  fp32, because the LayerNorms are scale-invariant. The ANE now grades A
  (0.99992, drift 0.003), the CPU path improves to 0.99951, and latency is
  unchanged. Neither D17/D19 limit applied: GELU and small linear inputs
  made no measurable difference.

  Results:
  - parity is bucket-invariant (CPU_ONLY 0.999919, CPU_AND_NE 0.999793) and
    pad invariance is 1.0000000 on both paths;
  - live `/v1/embeddings` worst parity is 0.99896 over nine texts, including
    a 722-token input truncated to 512;
  - similarity structure matches fp32 (unrelated pair 0.417 vs 0.416);
  - ~7.8 ms warm for a short text, including HTTP.

  ~285 MB per bucket, 0.86 GB installed.
- **all-MiniLM-L6-v2**: a 6-layer BERT (22.7M parameters, Apache-2.0),
  validated September 2026 on macOS 27. It was the first model converted by
  the conversion library's BERT recipe
  ([convert_bert_embedder.py](../tools/convert_bert_embedder.py)), with
  explicit attention and mask-aware mean pooling in the graph.
  - Its buckets stop at 256, sentence-transformers' `max_seq_length` for
    it, which is what the model was trained and published at. Truncated at
    512 instead, the embedding of a ~450-token text is only 0.973 cosine to
    the one at 256.
  - The published tokenizer.json pads every input to 128 and truncates at
    128. Both are removed at install, and sidekick never pads a single
    input.
  - The mean-pooling tail (`reduce_sum`, `real_div`, `clip`) runs on the
    CPU after the encoder. That is one extra hand-off, and it is why its
    ANE share (86.9%) is below bge-small's; the encoder's heavy operations
    are all on the ANE.
  - ~43 MB per bucket.
- **e5-small-v2**: a 12-layer BERT (33.4M parameters, MIT), validated with
  the same recipe. The `query: ` / `passage: ` prefixes its model card
  requires are in the manifest, and the server applies them; the
  checkpoint publishes no prompts. ~64 MB per bucket.

## Confidence grades: the parity suite

The converter parity numbers above come from short prose. The parity suite
(`crates/sidekick-embed/examples/parity`, D26) grades every validated model
on every compute path against a deliberately adversarial corpus,
[fixtures/parity/corpus.toml](../fixtures/parity/corpus.toml). Its 51
inputs are each tagged with the failure they exist to catch:

- mostly-padding inputs;
- exact bucket boundaries and over-length truncation;
- delimiter and special-token floods, and repeated tokens;
- digits, URLs and code;
- four scripts.

It runs them through sidekick's product path: the same prefixes,
tokenizer, truncation, bucketing, padding and pooling the daemon uses.
Each run happens on Core ML's `.cpuOnly`, `.cpuAndGPU` and
`.cpuAndNeuralEngine`, and, for a model with an ONNX build, on ONNX
Runtime's CPU (`onnx-cpu`); the output is compared with the model as
published: sentence-transformers in fp32, one input at a time.

A grade is the worst-case cosine over the corpus, with repeated-token
stress inputs graded separately:

| grade | worst-case cosine | meaning |
|---|---|---|
| **A** | ≥ 0.9999 | indistinguishable from fp32 |
| **B** | ≥ 0.999 | |
| **C** | ≥ 0.985 | the converters' ANE acceptance gate |
| **D** | below 0.985 | |
| **F** | — | a hard gate failed |

Grades are a best-effort statement about these inputs on this hardware,
not a bound on every input. How much accuracy the ANE loses depends on the
content: before their precision rewrites, LFM2.5 lost most on a URL,
EmbeddingGemma and F2LLM on a run of digits, and gte-modernbert on
delimiter lists.

M1 Max, macOS 27.0, September 2026. sidekick serves the ANE column; ms is
the median per input, ANE (CPU).

<!-- generated: coreml-grades -->
| model | CPU | GPU | ANE: worst case | ANE: similarity drift (bias) | ANE: rank flips | ms |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | A 0.99992 | A 0.999998 | **A** 0.99992 (empty input) | 0.003 (0.000) | 0 | not measured |
| bge-small-en-v1.5 | A 0.99993 | A 0.999999 | **A** 0.99997 (empty input) | 0.002 (+0.001) | 0 | 1.9 (7.0) |
| e5-small-v2 | A 0.99995 | A 0.999997 | **A** 0.99995 (a 255-token boundary case) | 0.002 (−0.001) | 0 | not measured |
| gte-modernbert-base | B 0.99951 | A 0.99999 | **A** 0.99992 (a delimiter flood) | 0.003 (0.000) | 0 | 6.4 (18.3) |
| embeddinggemma-300m | B 0.99989 | A 0.999998 | **A** 0.99999 (an over-length query) | 0.001 (0.000) | 0 | 8.1 (20.3) |
| embeddinggemma-2 | F 0.99993 | A 0.9999993 | **A** 0.999995 (numbers-order) | 0.001 | 0 | 10.1 (28.0) |
| F2LLM-v2-160M | B 0.99986 | A 0.999999 | **A** 0.99999 (a run of digits) | 0.001 (0.000) | 0 | 5.3 (13.0) |
| LFM2.5-Embedding-350M | B 0.99986 | A 0.999999 | **A** 0.99999 (a delimiter flood) | 0.003 (0.000) | 0 | 14.9 (33.8) |
<!-- /generated -->

- **Similarity drift** is the largest change in any pairwise similarity
  score against fp32. **Bias** is the mean signed change: LFM2.5's ANE
  scores run systematically low.
- **Rank flips** counts the triples (anchor, two candidates) whose
  candidates the reference separates by at least 0.02 and the ANE orders
  the other way. Only one of the two orders of a pair can qualify, so
  about 62,000 comparisons count.
- EmbeddingGemma's Matryoshka dimensions grade the same on the ANE:
  0.99999 at 512, 256 and 128.
- EmbeddingGemma graded D before its MLP precision rewrite (D17): 0.975 on
  a run of digits, drift 0.042 with scores biased +0.009 high, and 122 rank
  flips.
- LFM2.5 graded D before its precision rewrite (D19 amendment): 0.954 on a
  URL, drift 0.187 with scores biased −0.010 low, and 1,399 rank flips.
- F2LLM graded B before its precision rewrite (D20 amendment): 0.99966 on
  a run of digits, drift 0.006; with the rewrite's first form (TanhSilu,
  attention rescales only), A at 0.99997, drift 0.002. The table's CPU
  milliseconds predate StableSilu.
- gte-modernbert graded B before its range rewrite (D25 amendment): 0.99940
  on a Markdown list, drift 0.010. The rewrite costs its GPU path a little
  on one repeated-subword stress case (0.99998 → 0.99995).

All seven pass every hard gate on every path:
- token ids identical to the reference pipeline's;
- finite output;
- exact pad invariance;
- bucket invariance of at least 0.99995 on the ANE (it was 0.9977 for
  EmbeddingGemma and 0.9965 for LFM2.5 before their rewrites) and exact on
  the CPU;
- bit-identical ANE output across two processes.

**What changed.** Two models graded D on the ANE while their CPU and GPU
paths stayed at 0.9999. LFM2.5 fell to 0.954 on URLs, delimiters and
repeated tokens, with drift up to 0.19; EmbeddingGemma fell to 0.975 on
digits. Both were the ANE's `linear` precision floor plus its coarse
native GELU/SiLU, and both now grade A after precision rewrites (D17 and
D19 amendments). F2LLM's B had the milder version: mostly the SiLU, plus
small attention inputs on long texts (D20 amendment). gte-modernbert's B
was a third limit: the ANE's `linear` saturates at 2^15, which its massive
activation crossed (D25 amendment). All five now grade A on the ANE. An
ANE-only loss is worth diagnosing before it's accepted.

**Reading a report.** For each model the report prints its lowest cases
with the CPU, GPU and ANE cosines side by side. The pattern points at a
layer:
- every path low: the conversion (the graph isn't the model);
- the CPU fine but the GPU and ANE low: reduced-precision arithmetic;
- the CPU and GPU fine but the ANE low: the ANE's execution of a faithful
  graph;
- a failed pad- or bucket-invariance gate: masking or padding.

On this hardware the GPU path runs at close to fp32 accuracy (A on every
model), so "GPU fine, ANE low" isolates the ANE, not fp16 arithmetic in
general. No rule is automatic. The fused-attention ModernBERT control below
shows three things at once:
- GPU 0.99998: the graph is faithful;
- ANE 0.71, with pad invariance 0.61: the dropped mask;
- CPU NaN: a separate CPU bug in the same fused op.

"The CPU fails, so it's the conversion" would have misread it, as the
original diagnosis did (D25).

**Negative controls.** The suite was accepted only after it failed each
known-bad artifact on the gate that targets it:

| artifact | built with | what fails |
|---|---|---|
| flexible-shape bge-small | `convert_bge_small.py --enumerated-shapes` | compute plan: 0 of 362 ops on the ANE. Nothing predicts, so nothing aborts. |
| fused-attention gte-modernbert | `convert_gte_modernbert.py --attn sdpa` | CPU output non-finite on 43 of 51 inputs; ANE pad invariance 0.61, bucket invariance 0.40, worst case 0.71 |
| LFM2.5 without pad zeroing | `convert_lfm25_embedding.py --no-pad-zeroing` | pad invariance 0.40–0.41 on every path (0.33–0.39 before the precision rewrite); worst case 0.54–0.55 |
| naive truncation, on F2LLM | the daemon with `take(max)` truncation | token ids differ on both over-length inputs; cosine 0.23 |

**ONNX oracles.** Where a model's publisher, or onnx-community, ships ONNX
exports, the reference generator also runs them in ONNX Runtime on the same
token ids. The report shows them as report-only columns, never gated. They
measure the ecosystem, not sidekick:
- bge-small's and gte-modernbert's fp32 exports match torch at 1.000000,
  and gte-modernbert's fp16 export at 0.999994.
- gte-modernbert's published int8 export reaches only 0.892, on a
  repeated-syllable input. sidekick's ANE path is 0.9994.
- onnx-community's EmbeddingGemma exports, fp32 and fp16 alike, diverge
  from sentence-transformers past about 257 tokens (worst 0.9936, at 512),
  because they don't halve the sliding window as transformers does for
  bidirectional models (D17): running the checkpoint with the unhalved
  512-token window reproduces them to 0.99998. sidekick's own ONNX export
  matches at every length. The export's q8 variant reaches 0.972.

**Running it.**

```sh
# Once per converted model: reference vectors (Python; ONNX exports optional).
python tools/parity_reference.py "<data dir>/models/gte-modernbert-base" \
    --source Alibaba-NLP/gte-modernbert-base \
    --onnx Alibaba-NLP/gte-modernbert-base:onnx/model_int8.onnx
# The suite on macOS: every model that has a reference, Core ML paths
# (about 10 minutes for five), plus onnx-cpu with the onnx feature.
cargo run --release -p sidekick-embed --features coreml,onnx --example parity
# onnx-cpu only, on any OS (Linux included):
ORT_DYLIB_PATH=<dir>/libonnxruntime.so \
cargo run --release -p sidekick-embed --features onnx --example parity -- \
    --models-dir <dir> --refs <refs> [--fixtures <repo>/fixtures]
```

- The fixtures (corpora, expectations) come from `--fixtures`,
  `$SIDEKICK_FIXTURES`, `./fixtures`, or the checkout the suite was built
  from. `--external <result.json>` grades onnx-cpu results produced outside
  the suite (`tools/onnx_parity.py`) beside the native ones.
- Each model runs on each Core ML path in its own process, after its compute
  plan is checked, so an artifact that aborts inside Core ML costs one cell
  of the report, not the run. (The timing above is for the Core ML paths.)
- [fixtures/parity/expectations.toml](../fixtures/parity/expectations.toml)
  holds the hard gates and, per chip, floors set at 1.5x the measured error.
  Floors turn this table into a regression test; `--suggest-floors` prints
  them from a run.
- On a chip with no floors, the suite enforces the gates and reports
  accuracy without judging it. Other Apple Silicon generations have their
  own ANE and will measure differently.
- For Core ML, re-run after every macOS update, because the ANE's numerics
  come with the OS.
- The suite builds with `--features onnx` alone on Linux, where it grades
  onnx models on `onnx-cpu` and names the chip by its CPU model.
- `onnx-cpu` (D40) has floors only for the Apple M1 Max (ONNX Runtime
  1.28.3) and the Intel Xeon W-3235 (1.23.2), set from sidekick's own fp32
  exports; other chips are reported, not judged. An int8 graph of a model
  with fp32 floors fails them by design: int8 is graded like any other
  build, never assumed equivalent.

### Confidence by architecture family

For a model that isn't validated yet, the closest validated relative is the
best available prior. Most families have one validated model so far, so
treat its grade as a starting expectation, not a promise, and run the
suite.

| family | validated | ANE grade | risks seen | inputs that find them |
|---|---|---|---|---|
| BERT (bge, MiniLM, e5) | bge-small-en-v1.5, all-MiniLM-L6-v2, e5-small-v2 | A | none | — |
| ModernBERT | gte-modernbert-base; laya-en (ModernBERT-large classifier) | A (gte); C (laya: one flip caps it; p99 1.93× its fp16 ceiling) | fused attention drops the mask on the ANE (convert eager); the massive activation's output projection crosses the ANE linear's 2^15 limit without the range rewrite (graded B); the vectors of delimiter tokens are a little less accurate, and laya's head reads single-token vectors, where that loss isn't averaged away | pad and bucket invariance; delimiters |
| Qwen3 decoder, last-token pooling | F2LLM-v2-160M | A | truncation must keep the final token; without the precision rewrite, the native SiLU costs ~0.03% on the ANE (graded B) | over-length; the ids gate; numbers |
| Gemma3, bidirectional | embeddinggemma-300m | A | fp16 overflow without the range rewrite; without the MLP precision rewrite, 1–2.5% ANE loss on digits, long and repeated-token inputs (graded D) | numbers, long, degenerate |
| Gemma 4 text tower, bidirectional | embeddinggemma-2 | A | fp16 overflow in RMSNorm's squares (Core ML CPU 0.947 without the pre-scales); Core ML's CPU softmax isn't bucket-invariant past 512 keys | long, over-length |
| LFM2 hybrid (conv + attention) | LFM2.5-Embedding-350M | A | convolutions read pad states unless they're zeroed; without the precision rewrite, up to 4.6% ANE loss on URLs and delimiters (graded D) | pad invariance; delimiters, numbers |

## ONNX Runtime on the CPU (`onnx-cpu`)

sidekick's `onnx` backend (D40) serves the same embedders through ONNX
Runtime on the CPU of any supported platform: one fp32 `model.onnx` per
model, exported by the conversion library (`--format onnx`, see
docs/CONVERTING.md) from the same checkpoint and gated against the same
fp32 references, or fp16 weights computed in fp32 (`--onnx-weights fp16`).
It serves embedders only: classifiers and rerankers are Core ML only. At
scan time the daemon checks the graph's inputs and outputs against the
manifest's `[io]`, marks a quantized graph (run one input at a time), and
keeps external weight data inside the model's directory. The parity suite
grades it as the `onnx-cpu` path.

### Accuracy

Graded natively on the shipped runtimes, 51-case corpus: ONNX Runtime
1.28.3 on an Apple M1 Max (macOS, arm64), an AMD Ryzen 7 5800X and an AMD
Threadripper 3970X (Linux, x86-64), and 1.23.2 on an Intel Xeon W-3235
(macOS, x86-64), there with ONNX Runtime's AVX-512 kernels (D42).
onnx-cpu cells give 1 − the worst-case cosine; the
Threadripper's grades match the Ryzen's bit for bit, so they share a
column. The Core ML columns are the Apple paths from the table above.

<!-- generated: onnx-accuracy -->
| model | onnx-cpu, M1 Max | onnx-cpu, Ryzen 7 5800X and Threadripper 3970X | onnx-cpu, Xeon W-3235 | Core ML CPU | Core ML GPU | Core ML ANE |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | **A** 1 − 2e-12 | **A** 1 − 2e-12 | **A** 1 − 2e-12 | A 0.99992 | A 0.999998 | A 0.99992 |
| bge-small-en-v1.5 | **A** 1 − 2e-12 | **A** 1 − 2e-12 | **A** 1 − 2e-12 | A 0.99993 | A 0.999999 | A 0.99997 |
| gte-modernbert-base | **A** 1 − 2e-11 | **A** 1 − 2e-11 | **A** 1 − 2e-11 | B 0.99951 | A 0.99999 | A 0.99992 |
| embeddinggemma-300m | **A** 1 − 3e-12 | **A** 1 − 3e-12 | **A** 1 − 3e-12 | B 0.99989 | A 0.999998 | A 0.99999 |
| embeddinggemma-2 | **A** 1 − 2e-12 | | **A** 1 − 1e-12 | F 0.99993 | A 0.9999993 | A 0.999995 |
| F2LLM-v2-160M | **A** 1 − 5e-12 | **A** 1 − 4e-12 | **A** 1 − 4e-12 | B 0.99986 | A 0.999999 | A 0.99999 |
| Qwen3-Embedding-0.6B | **A** 1 − 2e-11 | **A** 1 − 2e-11 | **A** 1 − 2e-11 | not converted | not converted | not converted |
| Qwen3-Embedding-0.6B, fp16 weights | **A** 1 − 2e-11 | | **A** 1 − 2e-11 | | | |
| Qwen3-Embedding-0.6B, int8 | **D** 0.406 | | **D** 0.404 | | | |
<!-- /generated -->

- onnx-cpu is fp32, so it reproduces the reference to rounding: no rank
  flips on any model or machine, and similarity drift at most 2.3e-6
  (Qwen3-Embedding; at most 1.3e-6 for the others) in the native grades. It
  is the most accurate CPU path on Apple Silicon, where Core ML's CPU path
  computes in fp16 and grades B for three of these five models.
- At a given thread count results are bit-identical run to run. Across 1,
  2, 4 and 8 threads they are bit-identical for four of the five models
  measured; EmbeddingGemma moves by at most 1.4e-7 per component at 4 and
  8 threads. Padding or batching moves results by at most 2.6e-7 per
  component (cosine 1.000000000).
- fp16 weights (`--onnx-weights fp16`) grade A on Qwen3-Embedding (1 −
  cosine at most 2.1e-11 on both architectures measured). They save disk,
  not memory (see Memory).
- **int8 isn't offered.** ONNX Runtime's int8 kernels quantize activations
  with one scale per tensor, computed over the whole batch. On
  Qwen3-Embedding a dynamic int8 export and onnx-community's published one
  both grade D one input at a time (median cosine 0.86, worst 0.406 on the
  empty input on the M1 Max; 7,500–8,700 rank flips), and batched they also
  fail the batch and pad gates, because an input's vector depends on what it
  is batched with. The same holds on x86-64 (0.404 on the Xeon). The backend
  runs any quantized graph one input at a time.

Published ONNX exports, graded the same way: bge-small-en-v1.5's and
gte-modernbert-base's fp32 exports match their checkpoints exactly
(gte's fp16 export: 0.999994, and ONNX Runtime loads it only with graph
optimizations off; its int8 export: 0.892). onnx-community's
embeddinggemma-300m export is wrong on inputs longer than about 257
tokens (0.9936 worst at 512): it applies the configuration's 512-token
sliding window where transformers halves it to 257 for this bidirectional
model. Running the checkpoint with a 512 window reproduces the export to
0.99998. sidekick's own export is exact at every length.

### Latency

M1 Max (8 performance and 2 efficiency cores), macOS 27.0, from Python (ONNX
Runtime 1.27, coremltools 9; the daemon ships 1.28.3, and the thread table
below was re-measured on it through sidekickd): the median of 20 forward
passes after warm-up, one input at a time, real tokens cut to each length,
with the load average recorded per run: 5–10 at rest, rising to about 20
during the longest onnx-cpu runs from their own threads. Core ML runs a
length in the smallest bucket that fits it, padded. ms per input:

<!-- generated: latency -->
| model | tokens | onnx-cpu (fp32) | Core ML CPU (fp16) | Core ML GPU | Core ML ANE |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 16 / 64 / 256 | 2.0 / 5.4 / 19.3 | 1.4 / 1.4 / 5.3 | 2.0 / 2.0 / 2.7 | 0.8 / 0.9 / 2.9 |
| bge-small-en-v1.5 | 16 / 64 / 256 | 4.2 / 10.5 / 37.8 | 5.4 / 5.3 / 12.8 | 3.5 / 3.5 / 4.4 | 2.8 / 2.5 / 6.5 |
| gte-modernbert-base | 16 / 64 / 256 | 15.2 / 45.4 / 178 | 18.7 / 18.1 / 32.7 | 9.7 / 9.6 / 13.6 | 6.7 / 7.9 / 17.0 |
| embeddinggemma-300m | 16 / 64 / 256 | 16.1 / 43.7 / 161 | 19.2 / 18.7 / 36.8 | 12.0 / 11.9 / 16.1 | 7.2 / 7.2 / 15.2 |
| F2LLM-v2-160M | 16 / 64 / 256 | 8.9 / 25.3 / 103 | 13.5 / 13.8 / 24.7 | 5.6 / 5.7 / 8.3 | 5.9 / 5.9 / 13.0 |
| Qwen3-Embedding-0.6B | 16 / 64 / 256 | 48.4 / 158 / 620 | | | |
| Qwen3-Embedding-0.6B, fp16 weights | 16 / 64 / 256 | 53.5 / 179 / 617 | | | |
| Qwen3-Embedding-0.6B, int8 (graded D) | 16 / 64 / 256 | 22.1 / 62.5 / 255 | | | |
<!-- /generated -->

- On Apple Silicon, onnx-cpu is the accurate CPU path, not the fast one.
  Core ML's fp16 CPU path is 3–5× faster at 256 tokens. At 16 tokens
  onnx-cpu is faster for four of the five models, because Core ML pads a
  short input to its smallest bucket (64 or 128 tokens). The ANE and GPU
  are faster than either.
- **Threads.** ONNX Runtime's own default, one thread per core, is the
  wrong one on Apple Silicon: the efficiency cores hold back threads that
  must finish together. Through sidekickd on the M1 Max, 10 threads are
  about twice as slow as 8 (Qwen3-Embedding 638 against 358 ms at ~290
  tokens); from Python in the latency runs above, 8 threads beat the
  default by 16–27% at 256 tokens. sidekickd defaults to the fastest
  physical cores (the P-cores on Apple silicon and hybrid Intel CPUs,
  physical cores elsewhere, never SMT siblings), capped at 8 unless a model
  asks for more (D40).
  Fewer threads leave room for a primary model on the same machine, at a
  cost. Median ms for one long input by intra-op threads, on four
  machines:

  <!-- generated: threads -->
  | machine, model | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 24 | 32 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | M1 Max (8P + 2E), gte-modernbert-base | 580 | 320 | 169 | 119 | 103 | 239 | | | | |
  | M1 Max, embeddinggemma-300m | 547 | 291 | 158 | 113 | 104 | 235 | | | | |
  | M1 Max, Qwen3-Embedding-0.6B | 2,125 | 1,114 | 584 | 427 | 358 | 638 | | | | |
  | Ryzen 7 5800X (8 cores, 16 threads), gte-modernbert-base | 514 | 267 | 144 | 104 | 85 | | 101 | 87 | | |
  | Ryzen 7 5800X, embeddinggemma-300m | 476 | 256 | 140 | 99 | 83 | | 100 | 86 | | |
  | Ryzen 7 5800X, Qwen3-Embedding-0.6B | 1,861 | 947 | 501 | 358 | 277 | | 370 | 298 | | |
  | Threadripper 3970X (32 cores), gte-modernbert-base | 451 | 236 | 133 | | 117 | | 103 | 104 | 103 | 118 |
  | Threadripper 3970X, embeddinggemma-300m | 417 | 223 | 129 | | 114 | | 120 | 112 | 106 | 137 |
  | Threadripper 3970X, Qwen3-Embedding-0.6B | 1,627 | 833 | 460 | | 335 | | 303 | 277 | 294 | 339 |
  | Xeon W-3235 (12 cores, 24 threads), gte-modernbert-base | 362 | 189 | 110 | 85 | 77 | | 83 | 78 | 207 | |
  | Xeon W-3235, embeddinggemma-300m | 352 | 185 | 108 | 84 | 74 | | 81 | 78 | 254 | |
  | Xeon W-3235, Qwen3-Embedding-0.6B | 1,563 | 772 | 432 | 328 | 266 | | 307 | 277 | 664 | |
  <!-- /generated -->

  Every row is sidekickd's onnx-cpu path through the HTTP API
  (`/v1/embeddings`, one request at a time, the median of 15) at about 290
  tokens. The M1 Max (macOS 27.0) and the Ryzen 5800X and Threadripper
  3970X (Linux, load average under 6) ran ONNX Runtime 1.28.3; the Xeon
  W-3235 (macOS, otherwise idle) ran 1.23.2 with its AVX-512 kernels
  (D42). Without them, the Xeon was 1.2–1.7× slower at every setting up
  to 12 threads. On the M1 Max the load average
  read 20–70 throughout, inflated by another user session's waiting threads
  while the CPU stayed about 80% idle; re-running 6, 8 and 10 threads
  reproduced the numbers to within 7%.

  Latency roughly halves with each doubling up to 4–8 threads, then
  flattens, even with 32 cores. Simultaneous multithreading adds nothing,
  and more threads than fast cores make it worse: on the M1 Max, 10 threads
  (adding its two efficiency cores, ONNX Runtime's own default) is twice as
  slow as 8, and the Xeon at 24 threads is 2.5–3.4× slower. Only a model as large as
  Qwen3-Embedding keeps gaining past 8 threads on a machine with more fast
  cores (Threadripper: 277 ms at 16 threads against 335 at 8).
- Batching helps little on the CPU: a batch of 8 costs 0.9–1.0× per input
  at 256 tokens and 0.6–0.8× at 16.
- x86-64 latency is from the thread table only.

### Memory

Measured per run in a fresh process (`tools/bench_memory.py`): the
process's own footprint (`phys_footprint`, what `footprint` and Activity
Monitor report) before loading, after loading and after one prediction per
length, and the artifact's size on disk. Where a model was measured more
than once, footprints repeated to within 7 MB across three runs, except
EmbeddingGemma's on onnx-cpu (2.7–3.4 GB), which the table gives as a
range.

<!-- generated: memory -->
| model | onnx-cpu: disk | onnx-cpu: footprint | Core ML: disk (all buckets) | Core ML CPU | Core ML GPU | Core ML ANE (process) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 86 MB | +170 MB | 129 MB | +6 MB | +45 MB | +9 MB |
| bge-small-en-v1.5 | 127 MB | +221 MB | 191 MB | +8 MB | +52 MB | +22 MB |
| gte-modernbert-base | 569 MB | +893 MB | 856 MB | +17 MB | +150 MB | +22 MB |
| embeddinggemma-300m | 1,174 MB | +2.7–3.4 GB | 1,764 MB | +0 MB | +471 MB | +0 MB |
| F2LLM-v2-160M | 608 MB | +1,337 MB | 913 MB | +13 MB | +36 MB | +16 MB |
| Qwen3-Embedding-0.6B | 2,274 MB | +1,724 MB | | | | |
| Qwen3-Embedding-0.6B, fp16 weights | 1,138 MB | +2,685 MB | | | | |
| Qwen3-Embedding-0.6B, int8 | 570 MB | +472 MB | | | | |
<!-- /generated -->

- onnx-cpu holds its weights in the process's own memory, often more than
  their size on disk (EmbeddingGemma's 262k-token vocabulary in fp32, plus
  ONNX Runtime's prepacked copies). Its footprint is what it costs.
- Core ML maps its weights from the file, so the process's footprint
  undercounts it on every path. On the ANE the compiled program also lives
  in system services: the whole machine's memory in use rose by about
  1.0–1.3 GB for EmbeddingGemma (three runs) and 1.0 GB for F2LLM (one run) while
  sidekick's process grew by under 40 MB. The machine-wide count swings
  by several hundred MB between runs on a desktop with other applications,
  so only those two are clear of the noise.
- fp16 weights halve the file, not the memory: ONNX Runtime upcasts them
  at load, so Qwen3-Embedding's fp16-weight build holds more (+2.7 GB) than
  its fp32 one.

### Recommendations

- **On Apple Silicon,** keep the ANE (or the GPU) for throughput, and use
  onnx-cpu where fp32 accuracy matters more than speed or the ANE and GPU
  are busy. It is the most accurate CPU path: fp32, A on every model, against
  Core ML CPU's B on gte-modernbert, EmbeddingGemma and F2LLM, at 3–5× its
  latency on long inputs.
- **On x86-64 macOS and Linux,** onnx-cpu is graded at the same grades as
  on arm64 (shipped runtime 1.23.2 on Intel macOS, 1.28.3 elsewhere;
  graded on a Xeon W-3235, a Ryzen 7 5800X and a Threadripper 3970X). On
  Linux it is the only path. On an Intel Mac whose CPU has AVX-512 (a Mac
  Pro or iMac Pro Xeon), sidekick makes ONNX Runtime use its AVX-512
  kernels, which macOS would otherwise hide from it (D42).
- **On an Intel Mac,** Core ML runs on the CPU and GPU (no Neural Engine;
  the parity suite skips the ANE path). Graded on one Mac Pro (Xeon
  W-3235, AMD GPU, macOS 26.6.2), with four hours per path:

  | model | Core ML CPU | Core ML GPU |
  |---|---|---|
  | bge-small-en-v1.5 | A, worst cosine 0.999998 | A, 0.999999 |
  | nlptown-sentiment | A, worst Δp 7.35e-4 (bucket drift 1.9e-4, under the Intel gate) | A, 4.78e-4 |
  | ms-marco-minilm-l6-v2 | B, 1.92e-3 | A, 7.44e-4 |
  | lumma-fev-0.1b | not graded: didn't finish in four hours | A |
  | gliner2.5-decide | not graded: didn't finish | F: bucket drift Δp 4.7e-3, over its 3.5e-3 gate |
  | laya-typed-decisions | not graded: didn't finish | F: bucket drift Δp 2.7e-2, over its 2.5e-2 gate |
  | agent-jev | not graded: didn't finish | not graded: didn't finish |
  | nli-minilm2-l6-h768 | not run (GPU only) | F: bucket drift Δp 2.3e-3, over its 2.1e-3 ceiling; worst Δp 1.7e-3 (A level), 11.5 ms; MultiNLI predictions match fp32 on every pair |
  | nli-deberta-v3-small | not run | F: bucket drift Δp 1.1e-2, over its 3.0e-3 ceiling; worst Δp 6.1e-3 (A level), 33 ms; MultiNLI predictions match fp32 on every pair |
  | nli-minilm2-l6-h768-ane (its ANE build) | not run | F: bucket drift Δp 2.7e-3; worst Δp 3.6e-3 (B level), 18 ms |
  | deberta-v3-base-zeroshot-v2.0-c | not run | F: bucket drift Δp 4.1e-3, over its 2.7e-3 ceiling; worst Δp 2.7e-3 (B level), 64 ms; MultiNLI predictions match fp32 on every pair |
  | bge-m3-zeroshot-v2.0-c | not run | **B**, passes every gate: worst Δp 1.3e-2 (p99 1.40× its ceiling), bucket drift 5.6e-3 within its 8.0e-3 ceiling; 65 ms; 234 ms per pair at 1,024 tokens and 425 ms at 2,048; MultiNLI predictions match fp32 on every pair |
  | roberta-large-mnli | not run | F: **bucket drift Δp 0.08**, over its 0.012 ceiling; worst Δp 2.1e-2 within a bucket (B level), 66 ms; MultiNLI predictions match fp32 on every pair |
  | nli-deberta-v3-base | not run | F: bucket drift Δp 3.3e-3, over its 3.0e-3 ceiling; worst Δp 2.3e-3 (A level), 64 ms; MultiNLI predictions match fp32 on every pair |
  | modernbert-large-zeroshot-v2.0 (licence note above) | not run | F: bucket drift Δp 6.0e-3, over its 4.7e-3 ceiling; worst Δp 6.2e-3 (B level), 48 ms; MultiNLI predictions match fp32 on every pair |

  The NLI models, graded on its GPU only (ms per hypothesis, the median
  one-label request), run 2.5–10× slower than on an M1 Max's GPU, at 11–66
  ms, so a 16-label request takes about 0.2–1 s. Their accuracy within a
  bucket is at their fp16 ceilings, and MultiNLI predictions match fp32 on
  every pair. Every one but bge-m3-zeroshot-v2.0-c fails bucket
  invariance, most by a little. **roberta-large-mnli by a lot: a request's
  probabilities can move by up to 0.08 between buckets, so its scores
  depend on input length there; prefer another NLI model on an Intel
  Mac.** bge-m3-zeroshot-v2.0-c passes every gate and is the long-input
  option: 234 ms per pair at 1,024 tokens and 425 ms at 2,048 (208 ms at
  2,048 on an M1 Max), each bucket's first load about 10 s.

  On this Mac the GPU is 6–13× faster than the CPU on the small models
  (nlptown 300 → 23 ms per case, ms-marco 42 → 7.5 ms) and as accurate or
  better, but it is no more bucket-invariant. Core ML logs that MPSGraph
  doesn't support padded tensors on Intel Macs, and the GPU's outputs move
  between buckets at every position. The CPU's move only in a few
  attention scores of inputs that nearly fill their bucket
  (`tools/repro_cpu_matmul_tail.py`). gliner2.5-decide and
  laya-typed-decisions, which pass bucket invariance on Apple silicon's GPU,
  fail it on this one, and their CPU paths are impractically slow, so they
  are **unsupported on Intel Macs**. So, by family and not graded on
  Intel, are laya-en (ModernBERT-large, like laya-typed-decisions) and
  julia-1 (mmBERT-small, a ModernBERT, with laya's head).
  agent-jev's paths and Lumma-fev's CPU path are ungraded. Compute units
  stay per model: the default runs a Core ML model on the CPU on an Intel
  Mac, and `[models."<id>"] compute_units = "cpu_and_gpu"` selects the
  faster GPU path. The parity suite's CPU bucket gate is looser on Intel
  (cosine 0.99999 for embedders, Δp 5e-4 for classifiers, the latter
  provisional, since two classifiers measure it), D38's amendment.
- **Threads:** the default, the fastest physical cores capped at 8, needs no
  setting. To leave cores to a primary model, lower it with `[models."<id>"]
  intra_op_threads` or, for every model, `onnx_intra_op_threads` in the
  daemon config: latency roughly halves per doubling up to 4–8 threads, then
  flattens. Qwen3-Embedding's manifest raises its cap to 16 (`[onnx]
  default_intra_op_threads`), which pays only on machines with more than 8
  fast cores (Threadripper 3970X: 277 ms at 16 threads against 335 at 8).
- **Qwen3-Embedding-0.6B:** serve fp32 (or fp16 weights to halve the disk).
  Don't use int8 on ONNX Runtime.

## Classifiers (`/v1/classify`)

Classifiers run as Core ML only; the ONNX backend serves embedders.

Classifiers are served by `POST /v1/classify` (D28), from a
`classifier.toml` manifest ([examples/classifiers/](../examples/classifiers/)).
The parity suite grades them in probability space against the model's own
fp32 forward: A for a worst-case |Δp| ≤ 1e-3, B ≤ 5e-3, C ≤ 2e-2, D
above; a flipped decision whose fp32 top-2 margin is at least 0.05 logits
caps the grade at C; F is a failed hard gate. Unlike the embedders, the
classifiers have no per-chip floors yet, so the suite reports their
accuracy without regression-testing it.

Some models resolve decisions finer than fp16 can represent, so a second
grade compares each path with the model's **ideal-fp16 ceiling**: what an
ideal fp16 engine would lose, simulated by `sidekick_convert.fp16sim` and
recorded in the reference as the `fp16` oracle. The ratio grade is the
path's p99 |Δp| over the ceiling's p99: A ≤ 1.25×, B ≤ 2×, C ≤ 4×, D
beyond. The grade that counts is the better of the two, so a model is
credited for being practically exact or for being as good as fp16 allows.
On the GPU and ANE, variation between buckets within the ceiling passes
the bucket gate (D28 amendment). Models whose references don't yet carry
the `fp16` oracle are graded on the absolute scale only.

**Supported or preview.** A classifier is supported when it passes every
hard gate on every path, its conversion is exact in fp32 with no known
unfixed defect, and it grades A on at least one path. Otherwise it's a
preview. When the A path isn't the one a model is served on by default,
its notes say so, and a manifest's `compute_units` can serve it there.
sidekick serves on the ANE by default (D14), even where the GPU grades
higher, to keep background work off the GPU.

M1 Max, macOS 27.0, September 2026. ms is the median per input on that
path.

| model | task | conversion | CPU | GPU | ANE | ANE ops | ANE ms |
|---|---|---|---|---|---|---|---|
| [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) as `nlptown-sentiment` | text-classification, 5 labels | [convert_bert_classifier.py](../tools/convert_bert_classifier.py) | B 3.4e-3 | A 6.6e-4 | **B** 2.6e-3 | 294/304 | 4.2 |
| [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) as `julia-1` (**preview**) | zero-shot, laya's format with Julia-1's option rendering, 1,024 tokens | [convert_julia.py](../tools/convert_julia.py) | D 0.39 (39 flips; 5.3× ceiling) | **C** 0.053 (2 flips cap it; 1.07× ceiling, A level) | **C** 0.116 (7 flips cap it; 1.84× ceiling, B level) | 1647/1665 | 9.0 |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) as `laya-en` | zero-shot, laya's format | [convert_laya.py](../tools/convert_laya.py) | D 0.16 (13 flips; 6.1× ceiling) | **A** 0.037 (0.93× ceiling) | **C** 0.043 (1 flip caps it; 1.93× ceiling, B level) | 1701/1719 | 40 |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) as `laya-typed-decisions` | zero-shot, laya's format, 1,024 tokens | [convert_laya.py](../tools/convert_laya.py) `--model laya-typed-decisions` | D 0.099 (1 flip; 5.4× ceiling) | **A** 0.034 (1.02× ceiling) | **C** 0.025 (2.14× ceiling) | 2061/2079 | 40 |
| [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) as `gliner2.5-decide` (served on the GPU) | zero-shot, gliner2's format | [convert_gliner2.py](../tools/convert_gliner2.py) | D 0.024 | **A** 3.7e-3 | C 0.019 (512 bucket only) | 947/956 | 1,280 at 512 |
| [FrontiersMind/Lumma-fev-0.1b](https://huggingface.co/FrontiersMind/Lumma-fev-0.1b) as `lumma-fev-0.1b` | zero-shot, fev format, 2,048 tokens | [convert_fev.py](../tools/convert_fev.py) | D 0.030 (5.7× ceiling; 1 flip); over 1,024 tokens, the D33 limit (buckets differ by up to 0.021) | **A** 4.3e-3 (0.92× ceiling) | **B** 5.7e-3 (1.43× ceiling) | 3858/3875 | 37 |
| [aimeigaoshou/agent-jev](https://huggingface.co/aimeigaoshou/agent-jev) as `agent-jev` (served on the GPU) | zero-shot, agentjev's format, 2,048 tokens | [convert_agentjev.py](../tools/convert_agentjev.py) | D 0.021 (5.9× ceiling; 1 flip); over 1,024 tokens, the D33 limit (buckets differ by up to 0.009) | **A** 3.7e-3 (0.84× ceiling) | B 5.6e-3 (1.44× ceiling; no flips), chunked (D37), an operator's option (D38) | 2,710–3,742 (of 2,767–3,775) | 152 |
| [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) as `nli-minilm2-l6-h768` (served on the GPU) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) | C 0.0148 | **A** 0.0019 (1.16× ceiling) | F 0.0186: bucket invariance 5.9e-3, over its 2.1e-3 ceiling | 150/160 | 2.4 |
| [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) as `nli-minilm2-l6-h768-ane` (its ANE build) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) with the BERT backbone's ANE rewrites | C 0.0179 | **A** 0.0027 (1.20× ceiling) | **B** 0.0049 (bucket invariance exact) | 233/243 | 2.4 |
| [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) as `nli-deberta-v3-xsmall` (served on the GPU) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) | C 0.0189 | **B** 0.004 (1.27× ceiling) | F 0.0276: bucket invariance 5.1e-3, over its 4.4e-3 ceiling | 479/488 | 15 |
| [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) as `nli-deberta-v3-small` (served on the GPU) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) | D 0.0295 | **A** 0.005 (1.03× ceiling) | F 0.0254: bucket invariance 7.4e-3, over its 3.0e-3 ceiling | 245/254 | 16 |
| [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) as `nli-deberta-v3-base` (served on the GPU) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) | C 0.013 | **B** 0.0037 (1.29× ceiling) | C 0.0101 (4.6× ceiling) | 479/488 | 33 |
| [FacebookAI/roberta-large-mnli](https://huggingface.co/FacebookAI/roberta-large-mnli) as `roberta-large-mnli` (served on the GPU) | zero-shot, nli format | [convert_nli.py](../tools/convert_nli.py) | D 0.1629 | **A** 0.1191 (0.92× ceiling; one fp16-sensitive MultiNLI pair) | F 0.0428: bucket invariance 1.7e-2, over its 1.2e-2 ceiling | 582/591 | 14 |
| [MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c) as `deberta-v3-base-zeroshot-v2.0-c` (served on the GPU) | zero-shot, nli format, two-way | [convert_nli.py](../tools/convert_nli.py) | D 0.0281 | **A** 0.0024 (1.15× ceiling) | F 0.0178: bucket invariance 3.9e-3, over its 2.7e-3 ceiling | 479/488 | 33 |
| [MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c) as `deberta-v3-large-zeroshot-v2.0-c` (served on the GPU) | zero-shot, nli format, two-way | [convert_nli.py](../tools/convert_nli.py) | C 0.0167 | **A** 0.0038 (1.19× ceiling) | F 0.0131: bucket invariance 3.7e-3, over its 2.1e-3 ceiling | 947/956 | 100 |
| [MoritzLaurer/roberta-large-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/roberta-large-zeroshot-v2.0-c) as `roberta-large-zeroshot-v2.0-c` (served on the GPU) | zero-shot, nli format, two-way | [convert_nli.py](../tools/convert_nli.py) | D 0.033 | **B** 0.0033 (1.63× ceiling) | F 0.0179: bucket invariance 3.9e-3, over its 2.4e-3 ceiling | 582/592 | 15 |
| [MoritzLaurer/bge-m3-zeroshot-v2.0-c](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c) as `bge-m3-zeroshot-v2.0-c` (served on the GPU) | zero-shot, nli format, two-way, multilingual, 2,048 tokens | [convert_nli.py](../tools/convert_nli.py) | F 0.0667 (bucket invariance; past 1,024 tokens the D33 limit) | **A** 0.0026 (0.93× ceiling) | F 0.0667: not served on the ANE (1.04 GiB of weights, past the 1 GiB limit, D32) | 0/592 | 41 |
| [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) as `modernbert-large-zeroshot-v2.0` (served on the GPU; **licence note**: trained partly on ANLI, see below) | zero-shot, nli format, two-way, 2,048 tokens | [convert_nli.py](../tools/convert_nli.py) | F 0.05 (bucket invariance; past 1,024 tokens the D33 limit) | **B** 0.01 (1.92× ceiling) | F 0.0171: bucket invariance 7.9e-3, over its 4.7e-3 ceiling; the head runs on the CPU | 1009/1031 | 20 |

**nlptown-sentiment** passes every gate on every path, on D26's 51-input
corpus: no decision changes, pad invariance exact, bucket invariance
about 5e-4 on the ANE. Its conversion uses eager attention with a finite
mask, since the SDPA path emits the fused op that drops masks on the ANE
(D25).

**The NLI cross-encoders** (nli format, D41) serve zero-shot classification
as transformers' zero-shot pipeline does: one (premise, hypothesis) pair
per candidate label, scored by the model's entailment logits. They share
one corpus (fixtures/classify/nli.corpus.toml), 339 requests and 419 pairs:
308 MultiNLI validation pairs as one-label requests (half matched, half
mismatched), zero-shot topic, sentiment and intent requests with 1 to 16
labels, agent-like texts (errors, stack traces, test runs, reasoning that
runs into a limitation), and premises past 512 tokens truncated on the right
and, for logs and traces, on the left (one past 2,048 tokens). Each
reference is checked against the pipeline's own scores (agreement within
1e-7).
- **Grades:** every model grades A or B on the GPU, where all five are
  served, at 4.0 (MiniLM2), 4.2 (DeBERTa-v3-xsmall), 4.0 (small), 6.7
  (base) and 15.6 ms (RoBERTa-large) per request. On the ANE four of the
  five fail bucket invariance beyond their fp16 ceiling, and the DeBERTa
  models run 3.6–5× slower than on the GPU, as GLiNER2.5-Decide does.
- **Task accuracy is unchanged by conversion.** On the 308 MultiNLI pairs,
  3-way accuracy is the same on every Core ML path as in fp32 (MiniLM2
  0.877 matched / 0.864 mismatched, DeBERTa-v3-xsmall 0.844 / 0.851, small
  0.857 / 0.857, base 0.896 / 0.883, RoBERTa-large 0.916 / 0.877), and
  predictions agree with fp32 on 100% of pairs, except MiniLM2's CPU and
  ANE paths (99.7%: one pair each).
- **An ANE build:** for a companion that should leave the GPU to a
  primary model, `nli-minilm2-l6-h768-ane` is MiniLM2 converted with the
  BERT backbone's three ANE rewrites (backbones/bert.py): the softmax
  denominator from the value matmul makes it exactly bucket-invariant (F
  to C), the erf GELU and the linear-floor rescale take its worst |Δp|
  from 0.013 to 4.9e-3 (B), at the same 2.4 ms per request. Its MNLI
  accuracy is 0.883 / 0.870 (fp32 0.877 / 0.864; 99.4% agreement). The
  same rewrites make roberta-large-mnli's ANE path exactly bucket-invariant
  too, but it grades C (p99 2.01× its fp16 ceiling), so it has no ANE build.
- **Modern equivalents, the zeroshot-v2.0 "-c" line:** two-way
  (entailment, not_entailment) models trained only on commercially-friendly
  data: Mixtral-generated synthetic data plus MultiNLI and FEVER-NLI, no
  ANLI. All four grade A or B on the GPU, where they are served: DeBERTa-v3
  base 6.7 ms and large 17 ms, RoBERTa-large 15.6 ms, and bge-m3 15.5 ms
  per request on this corpus, whose requests are mostly short (bge-m3 takes
  208 ms per pair at 2,048 tokens). bge-m3 is the long-context option: it
  is multilingual, and its 2,048-token buckets keep the end of a long log
  or trace where the 512-token models drop it. Its tokenizer config says
  512, so transformers' pipeline would truncate there by default; the
  reference truncates at the manifest's 2,048. MultiNLI accuracy,
  entailment against not, is the same on every Core ML path as in fp32:
  DeBERTa-v3-base 0.942 matched / 0.916 mismatched, large 0.948 / 0.935,
  RoBERTa-large 0.942 / 0.942, bge-m3 0.942 / 0.922.
- **ModernBERT-large-zeroshot-v2.0** is supported but not recommended
  by default. **Licence note: trained partly on ANLI (CC BY-NC 4.0);
  whether that restriction applies to the weights is unsettled; treat as
  non-commercial unless you've cleared it.** The "-c" models above are the
  recommended set. It is the most accurate on MultiNLI of any model here
  (entailment against not: 0.961 matched / 0.955 mismatched, unchanged on
  every Core ML path) and grades B on the GPU, to 2,048 tokens, at 19 ms
  per request (249 ms per pair at 2,048). On the ANE, tried with each of
  ModernBERT's rewrites (measured at buckets 128 to 1,024):
  - as converted for the GPU, Core ML runs the classification head on the
    CPU, after the mean pool's reduce_sum; written as a matmul, the pool
    and head stay on the ANE (98.9% of operations) at the GPU's accuracy
    up to 256 tokens (|Δp| 1.8e-3), but bucket invariance fails (7.3e-3,
    over its 4.7e-3 ceiling), and at 2,048 tokens its error passes the
    converter's gate (0.024);
  - laya's matmul softmax makes it exactly bucket-invariant, and every
    gate passes, but it grades C (|Δp| 0.015, 5× its ceiling), and the GPU
    path drops to C too;
  - laya's erf GELU costs it precision (|Δp| 0.031 at 128).

  So it has no ANE build; tools/convert_nli.py builds the invariant C
  variant for a manifest that serves it on the ANE.
- **roberta-large-mnli's GPU worst case** (|Δp| 0.12) is one MultiNLI pair
  whose entailment and contradiction logits nearly cancel, so its
  one-label score sits on the sigmoid's steep part; an ideal fp16 engine
  moves it by 0.012, and the GPU's p99 is within the ceiling (0.92×).
- **Training data:** MultiNLI and SNLI for the cross-encoder models
  (Apache-2.0), MultiNLI for RoBERTa-large (MIT); none of them trains on
  ANLI, whose licence is non-commercial.

**laya-en** is supported: it passes every gate and grades A on the GPU
(served on the ANE by default; `compute_units = "cpu_and_gpu"` serves the
A path at about the same latency). Measured on 2,612 cases: fastino/fast-decisions
translated into laya's three question types, plus adversarial cases.
- **Grades:** C on the ANE, A on the GPU, D on the CPU, graded against
  laya's ideal-fp16 ceiling (D28 amendment): an ideal fp16 engine moves
  laya's probabilities by up to 0.017 (p99 0.0081). The ANE's p99 is 1.93×
  that, a B by the ratio alone; its one graded flip caps it at C. The GPU
  is at the ceiling (0.93×).
- **Decisions:** 99.96% agree with fp32 on the ANE and 100% on the GPU.
  The one ANE flip had an fp32 margin of 0.055 logits, so read `probs`,
  not just `label`, when a decision is close. |Δp| on the ANE is 0.043 at
  most, 0.016 at p99.
- **Where the loss was:** Core ML's native `gelu` op, which is coarse on
  the ANE, in the first layers of the ModernBERT-large encoder. The
  converter builds GELU from erf (constraint E, D28 amendment). That took
  the ANE from 5 flips and |Δp| 0.077 to 1 flip and 0.039, at 1–7% more
  latency.
- **Bucket invariance:** exact on the ANE. Core ML's softmax sums in an
  order that depends on the compiled shape, so the converter builds every
  softmax from exp and one matmul instead (constraint F), and the same
  input gives bit-identical output in every bucket (it moved by up to
  0.027 before). That costs a few inputs: |Δp| max 0.039 → 0.043, with p99
  and mean unchanged. On the GPU an input moves by up to 0.0072 between
  buckets, inside the gate, which allows what fp16 storage alone produces
  (the ceiling's 0.017).
- **CPU:** Core ML's fp16 CPU backend is laya's least accurate path (13
  flips, |Δp| up to 0.16). Its erf is a little coarser than its native
  gelu. The conversion is exact in fp32 (|Δlogit| ≤ 1.2e-4 against laya's
  own forward), so the loss is the backend's fp16 arithmetic. The
  converter gates accuracy on the ANE, the served path, and only reports
  the CPU.
- **Gold accuracy** is 52.8% on the ANE and in fp32 (reported, not
  graded). That measures the dataset's mechanical translation, 28-way
  intents with bare label keys, as much as laya.
- Its token layout is a port of laya's own code and reproduces laya's
  Python token for token on 15 cases that take every branch.

**laya-typed-decisions** is supported: it passes every gate and grades A
on the GPU (served on the ANE by default, as laya-en). It is laya's format with a
1,024-token input and a 256-token head, converted the same way as laya-en
(erf GELU, the matmul softmax) in four buckets, 128 to 1,024 tokens. Its
inputs are built by the laya package's code (`--laya-code`, checked by
sha256). Measured on 2,641 cases: laya-en's corpus plus 29 inputs between
587 and 1,024 tokens. The long inputs join consecutive fast-decisions rows
of one domain and cover every question type, 32 long options, inputs that
laya truncates at 1,024 tokens, and inputs within 16 tokens of the limit
(fixtures/classify/laya-typed-decisions.corpus.toml).
- **Grades:** C on the ANE, A on the GPU, D on the CPU. The ideal-fp16
  ceiling is 0.025 at most (p99 0.0038). The ANE's worst |Δp| equals that
  maximum (0.025), and its p99 is 2.14× the ceiling's, just short of B. It
  changes no decision; three ties under 0.05 logits move.
- **Long inputs** are as accurate as short ones. The worst |Δp| over the
  inputs above 512 tokens is 0.0084 on the ANE, and 0.0024 for those laya
  truncated.
- **Bucket invariance:** exact on the ANE, as for laya-en; up to 0.0031 on
  the GPU, inside the gate.
- **CPU:** 1 flip (margin 0.17 logits), |Δp| up to 0.099: Core ML's fp16
  CPU backend again, while the conversion is exact in fp32 (|Δlogit| ≤
  1.6e-5 against the checkpoint in every bucket).
- **Gold accuracy** is 55.1% in fp32 and on every path, within one input
  (reported, not graded). The checkpoint was fine-tuned on its own
  typed-decisions data, so this corpus checks parity, not the model's
  accuracy.

The suite passes both laya models on every path. laya-en's CPU path takes about 20 minutes for its 2,612 cases on an M1 Max, which is
the suite's default per-worker limit, so run it with `--timeout 3600`.

**julia-1 is a preview**: it passes every gate, but no path grades A,
since decision flips cap both its fp16 paths at C. Julia-1 is an mmBERT-small encoder (a
multilingual ModernBERT) with laya's decision head, so it runs on the laya
format; only its option texts are rendered differently
(`option_rendering = "julia"`). Measured on 2,510 cases: fastino/fast-decisions
translated as for laya (heads of up to 20 labels, Julia-1's limit), plus
adversarial cases in Julia-1's terms.
- **Grades:** C on the ANE and on the GPU, both capped by flips; by their
  p99 ratios alone they are B (ANE, 1.84× the ceiling) and A (GPU, 1.07×).
  D on the CPU (39 flips).
- **The ceiling is high.** An ideal fp16 engine moves Julia-1's
  probabilities by up to 0.055 (p99 0.028), several times laya's, and
  flips two of its decisions. About half of that is the weights alone:
  rounded to fp16 with every activation exact, they reach p99 0.0135 and
  flip one of the two. The embedding table is a small part of it; the
  encoder's and head's other weights are most of it. Julia-1's decisions
  resolve finer than its weights do in fp16.
- **The ANE's 7 flips** have fp32 margins of 0.05 to 0.35 logits, and most
  are borderline in fp16 already: one is also a flip of the ideal-fp16
  engine (`ticket_route.contains_pii.077`), and four more are among the
  101 inputs fp16 storage moves most. Read `probs`, not just `label`, when
  a decision is close.
- **Bucket invariance** is exact on the ANE (the converter builds every
  softmax from exp and one matmul, as for laya); up to 0.049 on the GPU,
  inside the ceiling. The ANE takes 9 ms per input.
- **Gold accuracy** is 45% in fp32 and on every path (reported, not
  graded). Julia-1's training data isn't published, so this corpus
  measures conversion parity, not the model's accuracy.
- The conversion is exact in fp32 (|Δlogit| ≤ 9.7e-5 against Julia-1's own
  forward in every bucket). mmBERT's config gives RoPE in transformers 5's
  `rope_parameters` block, which transformers 4.57 ignores; the converter
  reads it and checks every layer's rotary frequencies.

**GLiNER2.5-Decide is served on the GPU**, not the ANE: its manifest sets
`compute_units = "cpu_and_gpu"`. It is fastino's DeBERTa-v3-large
(486M parameters, Apache-2.0) with GLiNER2's per-token classifier, which
scores an `[L]` marker placed before each candidate label. The converter
replaces DeBERTa's relative-position gathers with a relative shift
(per-bucket tables over the 2L−1 distances, then a reshape-and-slice
skew), which is exact in fp32 and puts 947 of its 956 operations (99.1%,
every bucket) on the ANE.
Measured on 2,916 cases: every classification head of
fastino/fast-decisions sent as a gliner2 request, plus 16 adversarial
cases.
- **Graded against its fp16 ceiling**, as this section's introduction
  describes. Its ideal-fp16 engine reaches |Δp| 3.5e-3 at most and
  1.75e-3 at p99 on this corpus, so the absolute letters alone would cap
  every fp16 path at B.
- **GPU: A.** p99 at 1.14× the ceiling, worst |Δp| 3.7e-3, no decision
  flips, 34 ms median. Bucket invariance is 3.3e-3, within the ceiling's
  own 3.5e-3.
- **First requests are slow.** Each bucket loads onto the GPU when it's
  first used: in a fresh `sidekickd`, the first request took about 15 s,
  and the first use of each other bucket 2.4 s and 4.6 s. Warm requests
  took 29–135 ms.
- **ANE: C, and slow.** p99 at 3.3× the ceiling, worst |Δp| 0.019, one
  flip on a near tie (fp32 margin 0.003 logits). A prediction takes about
  0.11 s at 128 tokens, 0.34 s at 256 and 1.28 s at 512: 10–40× the
  GPU. The relative-shift path compiles onto the ANE, but runs far slower
  there than the attention of the other encoders here. That is not
  diagnosed yet. The suite's worker completed all 2,916 cases. This grade
  was computed from its output against the same reference and ceiling,
  because the run outlasted the suite's time limit.
- **CPU: D.** p99 at 5.5× the ceiling, worst |Δp| 0.024, no flips beyond
  5 near ties, 131 ms median. As with laya, Core ML's fp16 CPU backend is
  the least accurate path.
- **An open observation on the ANE.** With all three buckets
  (128/256/512) loaded in one process, the ANE path failed after 84 and
  after 322 cases in two runs: Core ML returned "Unable to compute the
  prediction using ML Program" on cases that predict correctly when run
  again. One
  bucket alone ran all 2,916 cases cleanly, so the ANE grade above comes
  from a 512-only manifest (padding is exact, so every case can run
  there). Other multi-bucket models, including ModernBERT-large ones, run
  on the ANE without it. The cause is unknown. GPU serving avoids it.
- **Parity only.** fast-decisions is fastino's own benchmark, so its gold
  labels measure nothing independent. Agreement with them is the same on
  every path (68.7% fp32, 68.8% CPU, 68.7% GPU) and isn't graded.
- Its token layout is a port of gliner2's own code, checked token for
  token against gliner2's Python on the adversarial cases and on a
  single-label and a multi-label dataset request.

The CPU path takes 25–30 minutes for the 2,916 cases on an M1 Max, past
the suite's default per-worker limit, so run it with `--timeout 3600`. The
ANE path at 512 needs about two hours, past even that.

**lumma-fev-0.1b** is supported: it passes every gate and grades A on the
GPU and B on the ANE, where it is served. Its CPU path is subject to the
documented CPU limit over 1,024 tokens (D33, below). It is FrontiersMind's Lumma-fev-0.1b (154M parameters, Apache-2.0): a
causal decoder (Nandi, Llama-style, with each of its 16 layers applied
twice) and a pointer head that compares the hidden state at a final
`<decide>` token with the hidden state at the end of each option, on the
fev format (docs/design/classify.md). Measured on 2,630 cases:
fastino/fast-decisions translated as for laya, rendered in fev's terms,
plus fev's adversarial cases (delimiter strings in every field, an empty
state, no instructions, 32 long options) and 15 long cases up to the
2,048-token window, two of them landing at 2,040 and 2,048 tokens.
- **Grades,** against the ideal-fp16 ceiling (|Δp| at most 0.0059, p99
  0.0028; no decision flips): GPU A (p99 0.92× the ceiling, worst |Δp|
  0.0043), ANE B (p99 1.43×, worst 0.0057, under the ceiling's own
  worst). Neither changes a decision. The ANE's silu is StableSilu (D39):
  with TanhSilu, artifacts converted before it graded B at 1.57× with a
  worst |Δp| of 0.010. Those keep working; reconvert for the smaller ANE
  error.
  Long inputs are as accurate as short ones on both (worst |Δp| 0.0031 on
  the ANE over the long cases).
- **Bucket invariance** is exact on the ANE and 0.0034 on the GPU, inside
  the ceiling. On the CPU every bucket up to 1,024 is bit-identical, but
  inputs over 512 tokens move by up to 0.021 between the 1,024 and 2,048
  buckets. Core ML's fp16 CPU matmul sums a contraction over 1,024 in a
  different order from a shorter one (accurately, within its own rounding;
  tools/repro_cpu_matmul_accumulation.py), and the 32 layer applications
  amplify the difference. Slicing the contraction makes the buckets agree
  but costs far more accuracy, so the converter keeps the single matmul.
  This is the documented CPU limit over 1,024 tokens (D33): a model served
  on the CPU is capped at 1,024 tokens by default, and the suite reports
  the CPU's bucket check above 1,024 as that limit rather than failing it.
- **CPU:** D on accuracy (p99 5.7× the ceiling, worst |Δp| 0.030, one flip
  at a 0.061 margin). As with laya, Core ML's fp16 CPU backend is the
  least accurate path.
- **Latency** grows with the bucket: 17, 36, 115, 305 and 1,308 ms on the
  ANE at 128, 256, 512, 1,024 and 2,048 tokens, timed by the converter.
  Most fast-decisions inputs land at 256. The ANE takes a 37 ms median on
  this corpus (33 ms with TanhSilu: StableSilu's exp, abs and divide cost
  about 12%), the GPU 31 ms; it is served on the ANE, the default.
- **Size:** about 455 MB per bucket. The checkpoint's factorized embedding
  (a 131k × 196 table and a projection) is folded into one 131k × 832
  table: Core ML keeps the first linear after the CPU gather on the CPU
  however it is written, and every linear of the graph runs on the ANE
  only without it.
- **The fp16 range.** The residual stream reaches ~870, and each RMSNorm
  squares its input, past fp16's 65504. The converter pre-scales those
  norms by a power of two (exact). The published graph run as an ideal
  fp16 engine overflows the same way, which is why the ceiling's
  simulation runs a normalization as one operation.
- **Gold accuracy** is 30.7% in fp32 and on every path (reported, not
  graded). The checkpoint's training data isn't published, so this corpus
  measures conversion parity. FrontiersMind reports 0.49 on its own
  typed-decisions benchmark, 0.89 on AG News, 0.68 on DAIR Emotion and
  0.47 on Banking77 for this size: whether that is enough is the
  consumer's call.
- The conversion is exact in fp32 (|Δlogit| ≤ 1.6e-5 against the
  checkpoint's own forward in every bucket). The checkpoint's code needs
  transformers 5; the reference generator runs it on transformers 4.57
  with shims that leave its computation untouched, and the converter's
  backbone is a plain-torch port (sidekick_convert/backbones/nandi.py).

The suite samples bucket invariance for buckets over 512 tokens; even so,
Lumma-fev's CPU path took about four hours on a heavily loaded M1 Max, so run
it with `--timeout 14400`. agent-jev's CPU path takes a little over an hour, past
the default timeout: run it with `--timeout 14400` too, or a worker reported
as timed out isn't a regression.

**agent-jev is served on the GPU** (`compute_units = "cpu_and_gpu"`). It is
AgentJev-0.6B (Apache-2.0): Qwen3-0.6B without its language-model head, and
a candidate head that scores a question's candidates as a set (a two-layer
transformer over the candidates, with no positions, then a scorer).
AgentJev's service scores each candidate on its own causal path: the
state, the question, then that candidate. sidekick lays a question out
once, as a tree: the state and question as a shared prefix, then every
candidate as its own branch. A branch sees only the prefix and its own
earlier tokens, and its positions continue from the prefix's end, so one
pass scores every candidate exactly as AgentJev would (docs/design/classify.md,
"The agentjev format"). In fp32 the converted tree matches AgentJev's
per-path scoring within |Δp| 5e-7, and the input layout reproduces
AgentJev's own `encode_paths` on all 2,627 corpus cases.

Measured on 2,627 cases: laya's translation of fastino/fast-decisions,
plus adversarial and long cases in AgentJev's terms, up to a tree of
exactly 2,048 tokens.
- **Why the GPU.** It is fastest and most accurate there, with nothing
  else running. Its fp16 weights (1.2 GB, 1.12 GiB) exceed Core ML's
  ~1 GiB limit for placing a program on the ANE (see the size note under
  "Quick triage"), so the converter splits each bucket into two programs
  at a layer boundary (D37): the embedding and layers 0–8 (0.57 GiB),
  then layers 9–27 and the head (0.58 GiB). On the GPU the chain gives
  the unchunked program's output bit for bit, at under 2% more latency,
  so the same artifact serves both placements, and an operator who wants
  the GPU left to another model can serve it on the ANE (D38). Artifacts
  from sidekick 0.6 are one program per bucket and serve the GPU only.
- **Supported.** It passes every hard gate on every path, its conversion
  is exact in fp32, and its served GPU path grades A.
- **GPU: A.** p99 |Δp| at 0.84× the ideal-fp16 ceiling's p99 (the ceiling:
  4.3e-3 at most, 2.1e-3 at p99), worst 3.7e-3 (0.86× the ceiling's
  worst), mean 5.1e-4, no decision flips, 8 near ties. Bucket invariance
  is exact, and so is pad invariance. 88 ms median. Its agreement with
  the gold labels is within one case of fp32's (1432/2600 against 1433).
- **A bucket-invariant softmax.** Every attention computes its softmax
  as exp(w − rowmax) and one matmul against [V | 1], as lumma-fev's does.
  Converted through transformers' sdpa path, the attention used Core ML's
  softmax, and on the CPU Core ML's softmax after the in-graph score
  matmul rounds differently for different key lengths, below 1,024 too
  (tools/repro_cpu_softmax_length.py). That first build's CPU moved by up
  to |Δp| 0.019 between its 512 and 1,024 buckets and failed the CPU's
  exact gate. The matmul form is exact in fp32 and bit-identical across
  buckets; on the GPU it moved p99 from 0.70× to 0.85× the ceiling and
  the worst case from 4.2e-3 to 3.2e-3, and made the GPU's buckets exact
  too (they were 2.5e-3 apart). An artifact converted before this fix (as
  in sidekick 0.6.0) keeps working at its GPU grade; reconvert it for
  exact buckets.
- **CPU: D, on accuracy.** p99 |Δp| at 5.9× the ceiling, worst 0.021,
  one flip at a 0.078 fp32 margin, 32 near ties, 273 ms median. Bucket
  invariance is exact up to 1,024 tokens. Past 1,024 the difference
  (8.9e-3 over 282 comparisons) is D33's documented limit, reported, not
  gated. agent-jev isn't CPU-served.
- **ANE (chunked): B, an option.** 97.9–99.1% of operations on the ANE.
  p99 |Δp| at 1.44× the ceiling, worst 5.6e-3, no decision flips, 6 near
  ties; agreement with the gold labels 1429/2600 against fp32's 1433. Bucket
  invariance is exact, and two processes give bit-identical results. It was
  D (4.2×, worst 0.022) until the ANE precision rewrite below (D39).
  152 ms per case against the GPU's 88; through sidekickd, a 1,024-token
  request took about 370 ms against 227 on the GPU with nothing else
  running (measured before the rewrite, which added about 20%). The first load compiles each chunk: about 40 s for the
  smallest bucket, 244 s for the 2,048-token one. What it saves a model
  sharing the GPU is measured with tools/companion_bench.py, below.
- **On the ANE, chunk it; don't quantize its embedding.** The other way
  under the ANE's weight limit, one program per bucket with the token
  embedding stored in int8 (`--int8-embedding --chunks 1`), was graded with
  the same precision rewrite on the same 2,627 cases:

  | | chunked (the default) | int8 embedding, one program |
  |---|---|---|
  | GPU | **A** (0.84×, worst 3.7e-3) | B (1.26×, worst 8.1e-3) |
  | ANE | B (1.44×, worst 5.6e-3) | B (1.74×, worst 8.5e-3) |
  | CPU | D (5.9×, 1 flip) | D (6.1×, 2 flips) |
  | weights per program | about 0.57 GiB | 0.971–0.987 GiB |
  | ANE latency, 256 / 1,024 / 2,048 tokens | 151 / 387 / 1,409 ms | 152 / 375 / 1,308 ms |
  | GPU latency, same lengths | 88 / 206 / 565 ms | 86 / 203 / 552 ms |

  The int8 build is a few percent faster at the long buckets, but it costs
  the GPU its A, and its programs sit 1–3% under a limit measured on one
  chip: the 2,048-token bucket, at 0.987 GiB, is above the largest program
  measured on the ANE (0.964 GiB), and Core ML produced no compute plan for
  it. Latencies are medians through sidekick's own Core ML path
  (chain_timing), one input at a time.
- **As a companion beside a larger model** (tools/companion_bench.py; M1
  Max with 32 GB, macOS 27.0). The primary was a 4-bit 27B Qwen3.8 with
  speculative decoding, served by oMLX on the GPU, which kept about 25 GB
  wired. agent-jev answered 1,024-token requests at one per second, from
  one install, its compute units switched by the daemon config (D38):
  - On the GPU, Core ML maps the companion's weights into sidekickd: 1,195
    MB resident for its one 1,024-token bucket. (Measured with an install of
    that bucket only: sidekick then also kept a model's smallest bucket
    resident, which doubled it. Since 0.7.0 a full install holds only the
    buckets its requests use.) Within seconds of it
    loading, oMLX's own memory-pressure policy unloaded the 27B and
    refused its requests ("process memory pressure requested this model
    to unload"): none of five generations completed, in two runs. That is
    how oMLX handles memory pressure, not a general fact about the GPU.
  - On the ANE, sidekickd holds 58 MB: the weights are read through the
    file cache, which macOS can reclaim. All five generations completed,
    at 19.50 tokens/s against 19.96 with the primary alone (about 2%
    slower; first token 0.37 s against 0.34). The companion answered all
    85 requests, 367 ms at the median and 873 ms at p99. The phase's swap
    grew by 462 MB while oMLX reloaded the 27B after the GPU phase had
    evicted it, which is why the baseline is the primary-alone phase
    measured after that reload (oMLX decoded at about 17.1 tokens/s
    before the reload and about 20 after).
  - With the primary idle, the companion took 226 ms at the median on the
    GPU and 377 ms on the ANE, and drew 7.36 W of GPU power on the GPU
    against 0.77 W of GPU and 1.10 W of ANE power on the ANE.
  - Its answers on the two paths differed by at most |Δp| 7.1e-3, and never
    within a phase.
- **An ANE precision rewrite (D39).** The ANE's native silu is coarse, and
  agent-jev's first layers run on a residual of rms 0.03, so that error
  dominated its ANE grade: layer 0 alone added 10× the GPU's error. The
  converter replaces silu with StableSilu, 2x·exp(min(x, 0)) / (1 +
  exp(−|x|)), which doesn't cancel in fp16 for either sign as TanhSilu's
  1 + tanh(x/2) does; rescales the inputs of q/k/v, o_proj and down_proj by
  powers of two, since they sit under the ANE linear's precision floor
  (rms 0.06–0.2); and replaces the head's own silu (StableSilu) and erf gelu
  (TwiceGelu). It is exact in fp32 and calibrated on the converter's gate
  states. The ANE went from D to B; the GPU kept its A.
- **No RMSNorm rewrite.** Its residual stream reaches about 6,800 on
  one dimension of the first token (the attention sink, which every
  candidate's path shares). Squared in fp16, as RMSNorm squares its input,
  that would overflow, but the converted model shows no sign of it: the GPU
  is within its ceiling, and a 2^-6 pre-scale on the norms measured worse
  on every path.
- **Quality, as its authors report it.** On its own coding-completion
  benchmark, accuracy is 57.8% and AUROC 0.589, up from 0.516 before its
  final training stage (0.5 is chance). Its service calls its probabilities "model distribution;
  domain calibration is not guaranteed".
- **Parity only.** fast-decisions isn't AgentJev's domain. Agreement with
  its gold labels is reported, not graded: 55.1% on the GPU, as in fp32,
  and 55.0% on the CPU.
- AgentJev's per-type temperatures are in the manifest; `calibration:
  model` applies them.

## Rerankers (`/v1/rerank`)

Rerankers run as Core ML only; the ONNX backend serves embedders.

Rerankers are cross-encoders served by `/v1/rerank`, `/rerank` and
`/v2/rerank` (D29), from a `classifier.toml` with `task = "text-ranking"`
([examples/classifiers/](../examples/classifiers/)). The parity suite
grades them on [fixtures/rerank/corpus.toml](../fixtures/rerank/corpus.toml),
51 (query, document) pairs in 13 groups, against `CrossEncoder` in fp32:
- **Score fidelity** is graded as a probability, |Δ sigmoid(logit)|, with
  the classifiers' letters (A ≤ 1e-3, B ≤ 5e-3, C ≤ 2e-2). Cross-encoders
  are trained with a sigmoid objective. A model that serves raw logits
  reports them unsquashed, and the sigmoid compresses errors at large
  logits, so the table also gives the largest raw |Δlogit|.
- **Ranking** is graded on raw logits: a flip is two documents of one
  query in the opposite order, where fp32 separates them by at least 0.05
  logits. Any flip caps the grade at C.

M1 Max, macOS 27.0, September 2026.

| model | conversion | CPU | GPU | ANE | rank flips | ANE ops | ANE ms |
|---|---|---|---|---|---|---|---|
| [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) as `ms-marco-minilm-l6-v2` | [convert_bert_classifier.py](../tools/convert_bert_classifier.py) `--twice-gelu` | B 3.2e-3 (Δlogit 0.082) | B 1.3e-3 (Δlogit 0.0091) | **B** 3.4e-3 (Δlogit 0.024) | 0 on every path | 168/183 | not measured |

**ms-marco-MiniLM-L6-v2**: a 6-layer BERT cross-encoder (22.7M
parameters, Apache-2.0), converted by the conversion library's BERT recipe.
It uses explicit attention, the checkpoint's own classification head, and
segment ids as a third int32 input (`token_type_ids`: 0 for the query, 1
for the document).
- Its score is the raw logit: the checkpoint pins sentence-transformers'
  Identity activation, and vLLM serves the same.
- **Converted with `--twice-gelu`.** BERT's erf GELU is built as
  `x * (1 + erf(x/√2))`, with the 0.5 folded into each layer's output
  projection, instead of Core ML's native `gelu`, which is coarse on the
  ANE. Compared with the native op, on the ANE:
  - mean Δp halves (2.9e-4 → 1.4e-4), and the worst Δlogit drops from 0.030
    to 0.024;
  - bucket invariance improves from 8.3e-4 to 1.4e-4, against the 1e-3
    gate;
  - 50 of the 51 pairs come within A (≤ 8.2e-4).
  The CPU path gets a little worse (2.3e-3 → 3.2e-3; Core ML's CPU `erf` is
  coarser than its native `gelu`), but sidekick serves the ANE.
- **One pair keeps it at B.** `hardware-0` (fp32 logit −1.03) is at
  Δp 3.4e-3 on the ANE with either GELU. Part of that is where it sits:
  near logit 0 the sigmoid is steepest, so its logit error, about 0.018,
  becomes a large Δp, though that error is within the ANE's range over the
  corpus (worst 0.024). Part isn't fp16's: an ideal fp16 engine is off by
  only 2.0e-4 on it (the corpus's fp16 ceiling is 9.6e-4 at most, mean
  5.8e-5), so the ANE's own arithmetic costs it about 17x what fp16
  storage does. That excess hasn't been diagnosed. The checkpoint's linear
  inputs are not small (smallest rms 0.107), so the ANE `linear` precision
  floor is an unlikely cause.
- It passes every hard gate on every path:
  - ids and segment ids equal `CrossEncoder`'s for all 51 pairs,
    including a pair truncated to 512 tokens and an empty document;
  - pad invariance is exact;
  - ANE output is bit-identical across processes.
- It ranks exactly as fp32 does, with no flips and no near-ties. That
  includes five near-duplicate documents with fp32 logits from 7.1 to 9.7,
  the closest two 0.24 apart.

## Incompatible / not integrated

| model | class | why |
|---|---|---|
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | late-interaction (multi-vector) | Emits one 128-d vector **per token**, scored with MaxSim — there is no single vector to return through `/v1/embeddings` or `sk_embed`. The encoder itself converts and offloads fine (smoke-tested at seq 256: per-token parity 0.9995 CPU / 0.9919 ANE, 2.0x ANE speedup, MaxSim ranking preserved — [smoke_lfm25_colbert.py](../tools/smoke_lfm25_colbert.py)), so a future late-interaction API could host it; nothing in today's API can. Its padded-batch conv semantics (expansion tokens must NOT be zeroed) also make real-token embeddings bucket-dependent under Core ML's static buckets. Re-run on macOS 27: unchanged (per-token ANE parity 0.9919, 2.1x over CPU, MaxSim ranking preserved). |
| Apple NLContextualEmbedding | OS-provided contextual | Mean-pooled MLM states, strongly anisotropic (unrelated-pair cosine ~0.75) — unusable for similarity thresholds without post-hoc calibration sidekick doesn't own (D16). Re-measured on macOS 27: same model revision, same cosines (0.96/0.89 vs 0.75), faster (~11 ms). |
| Apple NLEmbedding.sentenceEmbedding | OS-provided static-ish | 2020-era quality, measurably weaker than bge-small on the same pairs; no prefixes, no control over dims (D16). Unchanged on macOS 27 (revision 1; 0.74/0.44 vs 0.14). |
| Apple FoundationModels | LLM | Has **no embedding API at all** (verified against macOS 26 SDK docs/headers, D16) — chat only. Still none in the macOS 27 SDK; 27's Spotlight integration is a search tool for sessions, not vectors. |

## Quick triage: is a model worth converting?

The triage, the checklist and the gates below are for Core ML. The ANE's
weight cap, fp16 range and precision limits don't apply to onnx-cpu, which
computes in fp32 with dynamic shapes, so an embedder rejected only by those
may still serve there (Qwen3-Embedding-0.6B is one: past the ANE's cap
unchunked, A on onnx-cpu). Its gate is the ONNX export's own (fp32 parity
against the checkpoint, batch and pad invariance; docs/CONVERTING.md, "ONNX
export"), then the parity suite's `onnx-cpu` path.

Most rejections are visible long before a conversion. Cheapest first:

1. **`config.json` (seconds).**
   - `model_type` / `architectures` is the strongest signal. Validated:
     `bert`, `gemma3_text` (with a range rewrite), `lfm2`, `qwen3`,
     `modernbert` (with explicit attention). A checkpoint can wrap one of
     these, as laya's `encoder/config.json` does.
   - Also check:
     - **QK-norm** (`q_norm`/`k_norm` in the modeling code): keeps
       activations small (LFM2.5 ~25, F2LLM ~420), so no range rewrite.
     - **bf16 training**: nothing kept its activations in fp16 range.
     - **Size**: each bucket stores the whole model at ~2 bytes/param, and
       Core ML runs an ML program on the ANE only if its weights fit in about
       1 GiB: about 500M parameters in fp16, embedding table included.
       Measured on an M1 Max with macOS 27 (October 2026) by cutting
       agent-jev's Qwen3-0.6B at different depths: 0.964 GiB of weights put
       99.4% of operations on the ANE, and 1.022 GiB put 0%. Over the cap,
       `cpu_and_ne` silently runs the whole model on the CPU, with no error,
       so a compute plan of 0 ANE operations is the symptom. Splitting each
       bucket into programs at layer boundaries (`--chunks`, D37) keeps every
       program well under the cap at no cost in accuracy. Storing only the
       embedding table in int8 also brought agent-jev (1.12 GiB) to 0.97 GiB
       and 98% on the ANE, but with almost no headroom and a measurable cost
       in accuracy (agent-jev's notes compare the two). This is Core ML's
       documented limit: Apple's coremltools guide ("MLModel Utilities",
       Bisect Model) states it as 1 GB for the Neural Engine on iPhone, and
       its `ct.models.utils.bisect_model` splits a model into two chunks
       under it. It may differ on other chips and OS releases; measure it
       there. GPU serving has no such cap (GLiNER2.5-Decide and agent-jev).
     - **Needed sequence length**: buckets are ≤512, and the ANE's advantage
       shrinks with length.
     - **Output shape**: one vector per input fits `/v1/embeddings`, and a
       sequence-classification head (or laya's format) fits `/v1/classify`
       (D28). Rerankers and multi-vector models need a new API.
2. **The modeling code (minutes).** See the checklist below: data-dependent
   shapes, attention that can't run as SDPA, and non-attention token mixers.
3. **`tools/probe_activations.py` (minutes, PyTorch on the CPU, no Core
   ML).** It hooks every normalization layer's input and reports peak
   activation, which dimension carries it, how many times the median
   dimension it is, and which norm type reads it. Its one calibrated verdict
   is fp16 range: over 65504 needs a range rewrite (EmbeddingGemma), and
   under ~30,000 is safe to convert directly.

   | model | norms | peak (dim) | × median | probe says | measured on ANE |
   |---|---|---|---|---|---|
   | bge-small-en-v1.5 | LayerNorm | 338 (99) | 146× | no range issue | 0.99998 |
   | LFM2.5-Embedding-350M | RMSNorm | 1.9 | 17× | no range issue | 0.99999 after the precision rewrite |
   | embeddinggemma-300m | RMSNorm | 152,485 (731) | 313× | range rewrite | 0.99999 after range + MLP rewrites |
   | gte-modernbert-base | LayerNorm | 47,973 (251) | 502× | no range issue | 0.99998 (explicit attention + 2^15 range rewrite) |
   | laya (ModernBERT-large) | LayerNorm | 27,296 (379) | 556× | no range issue | CLS ≥ 0.997 (explicit attention) |

   The verdict is about the residual stream. A written-out RMSNorm squares
   its input, so any norm input past 256 overflows in fp16:
   EmbeddingGemma 2's peak at ~1,600 cost Core ML's CPU path 5% until the
   norms were pre-scaled.

   Massive activations, one dimension hundreds of times the rest on a few
   tokens, are common, and they are not by themselves an ANE problem. The
   ModernBERT failure once blamed on them was a Core ML attention bug. Expect
   those tokens' own output vectors to be slightly less accurate, which
   matters only when you read individual token vectors.
4. **Convert with explicit attention, then gate (hours):** the gates at the
   end of the checklist. Pad invariance in `ane_check` catches a dropped
   attention mask in seconds.

## Will a new model convert? A checklist

Read the model's `modeling_*.py` before anything else. The recipe survives:

- **Encoder-style, single-vector output** — bidirectional attention (or a
  published bidirectional patch), CLS or mean pooling; or a causal decoder
  with last-token pooling (F2LLM). Multi-vector, and rerank heads don't fit
  the API; generative models have no pooled vector at all.
- **Last-token pooling has two traps** — (1) select the last real token
  in-graph via the mask (`mask · (1 − shift_left(mask))`, masked sum), never
  a data-dependent gather index; (2) the embedding IS the trailing EOS
  token, so truncation must preserve it — the server keeps `[first max-1,
  last]` for exactly this reason. A short-input-only parity gate misses
  both; test an over-length input that fills the largest bucket.
- **Explicit attention with fp16-safe masks.** For Core ML, masks must use a
  finite constant such as -30000, not `-inf`/`finfo.min`, which NaN in fp16
  (D15). The converters patch the mask builders to do this. Then use
  attention that converts to explicit matmul → softmax → matmul (gemma,
  F2LLM, ModernBERT via `attn_implementation="eager"`). Passing `scale=` to
  `F.scaled_dot_product_attention` works too: coremltools then emits
  explicit ops instead of the fused one. Whatever the form, `ane_check`'s
  pad-invariance gate must pass.
- **Fused SDPA drops the mask on the ANE when the mask is an input of the
  ANE procedure that runs the attention.** That is, when no op inside that
  procedure computes the mask. It may be a model input, the output of an op
  on the CPU, or the output of an earlier ANE procedure. The output then
  equals unmasked attention exactly, for every mask shape, dtype and fill
  value tried and at both opsets (D25). The compute plan shows it:
  - `scaled_dot_product_attention` is on the NeuralEngine, and
  - its `attn_mask` comes from a model input or a CPU op, or from an ANE op
    with a CPU op between it and the attention.

  ModernBERT builds its masks before the CPU-only embedding gather, so the
  mask plumbing lands on the CPU. `python tools/repro_sdpa_mask.py --check
  <model>` applies the rule to a compiled model. A fused attention whose plan
  shows no device (bge-small) is correct only through a Core ML fallback that
  doesn't run it natively. At the iOS26 opset that fallback is gone and bge
  fails to load on CPU_AND_NE, so keep the macOS15 target for fused-attention
  models.
- **Fused SDPA on the CPU (`CPU_ONLY`) has two more defects**
  (reproduced by the same tool):
  - A query row whose keys are all masked returns NaN when
    |fill| × √head_dim > 65504, e.g. -30000 at head dim 64. Explicit
    attention returns a finite row.
  - q/k/v split straight off a packed projection with
    `.transpose(3, 1).unbind(2)` give wrong output. `.permute(2, 0, 3, 1, 4)`
    is correct.

  CPU_ONLY parity catches both.
- **Static-shape-friendly graph** (for Core ML; an ONNX export takes dynamic
  axes by design) — no data-dependent shapes. Stock
  `rotate_half`/`repeat_kv` and any `F.conv1d(padding=shape-derived)`
  need the traceable rewrites (D17 constraint 8, LFM2.5 constraint B).
- **fp16-safe activations** — calibrate first (forward hooks, max |activation|
  on a mixed corpus). Under ~30k: convert directly (bge, LFM2.5). Over:
  apply the D17 power-of-two range rewrite (gemma). For Core ML, watch for
  `-1e9` mask constants (rewrite at -30000) and rmsnorm eps below ~1e-4.
- **Keep every linear-layer input near rms 1.** The ANE's `linear` op has
  an absolute precision floor on its input, not a relative one: its
  relative error is ~3e-4 / rms(input) (0.04% at rms 1, 2% at 0.016, 10% at
  0.004; the GPU is flat at 0.04%). Scaling the weights doesn't help; only
  the input's magnitude matters. Measure the rms of every linear input in
  fp32 with forward pre-hooks. Likely offenders are MLP down projections
  after a gated product (act(gate)·up is a product of two small numbers),
  in-graph heads that pool at a reduced scale, and any model whose
  activations are tiny everywhere (LFM2.5: every output projection at rms
  0.003–0.07). A small activation range spares a range rewrite, but it
  isn't safe on the ANE. Fix with a power-of-two scale folded into the
  weights upstream. Cancel it with a scale-invariant norm or the final L2
  normalize (EmbeddingGemma, D17). Where only a residual add follows,
  cancel it with an explicit multiply rather than weights divided into
  fp16's subnormal range (LFM2.5, D19).
- **Keep every linear-layer output under 2^15 on the ANE.** The ANE's
  `linear` op saturates at 32,768, half of fp16's 65,504: an output of
  33,000 comes back inf. Its add, mul and layer_norm handle the full fp16
  range, so fp16 headroom alone doesn't protect a projection. Measure the
  largest output of every linear in fp32 with forward hooks. Keep it at or
  below 0.85 × 2^15 by running the residual stream at a power-of-two 1/K
  that the norms cancel (gte-modernbert: K = 2, D25). Massive activations
  are the usual cause: ModernBERT writes one at 35,000–51,500.
- **Don't use Core ML's native GELU or SiLU ops on the ANE.** Measured
  absolute error on [-1, 1]: gelu ~6e-3, silu ~1.5e-2, versus tanh 1.6e-3,
  sigmoid 3e-3, and exact mul/relu. Build GELU as
  `x * (1 + tanh(x * (c + c·0.044715·x²)))` with the factor 2 absorbed
  downstream, and SiLU as `x * (1 + tanh(x / 2))` (LFM2.5, F2LLM). Check the
  converted program for surviving `gelu`/`silu` ops: coremltools has passes
  that fuse such patterns, and `x * sigmoid(x)` converts straight back to
  the native silu.
- **Massive activations are not disqualifying.** A few feature dimensions
  reaching tens of thousands on delimiter/[SEP] tokens (ModernBERT ~48,000,
  EmbeddingGemma ~152,000) convert fine once attention is explicit and values
  fit fp16. Their cost is a small per-token effect on those tokens. An
  earlier version of this checklist called them an "ANE killer" under
  LayerNorm; that was the fused-attention mask bug misdiagnosed (D25).
- **Token mixing other than attention** (convs, SSMs): decide the padding
  semantics explicitly. Attention masks silence pad *keys*, but anything
  convolutional reads pad *states* — zero them per layer if the reference
  is the unpadded forward (LFM2.5 constraint D).
- **Per-bucket artifact size is the whole model** — weights duplicate per
  bucket until multifunction mlprograms land. 350M params ≈ 700 MB × 3
  buckets. Fine on disk, but mind the install footprint.

Gates to pass, in order: fp32 rewrite parity ≥ 0.9999 (only if rewriting),
`CPU_ONLY` ≥ 0.999, `CPU_AND_NE` ≥ 0.985, `ane_check` per bucket (compute
plan eligible and pad invariance on both paths; its ratio should be clearly
above 1.0 on a quiet machine). Then generate a reference with
`tools/parity_reference.py` and run the parity suite. It repeats those gates
through the product code on every path and adds the adversarial corpus. Its
grade is the one to publish, and `--suggest-floors` records it for the
chip. For an embedder's ONNX build: the export's fp32 gate, then the
suite's `onnx-cpu` path.

For Core ML, flexible input shapes are ruled out (D15), and on macOS 27 one
kind became dangerous. Under `.cpuOnly`, an artifact whose inputs have several
enumerated shapes (`ct.EnumeratedShapes`) aborts the process at its first
prediction ("E5RT: No memory object bound to port"). Under
`.cpuAndNeuralEngine` and `.all` it still falls back to the CPU: 85 ms per
bge-small embed at seq 128, where a static bucket takes 2.2 ms on the ANE.
The abort is an Objective-C exception Rust can't catch. It would take down
`sidekickd`, or the host app linking `libsidekick.dylib`.

So on macOS 27, `CoremlModel::load` refuses such a model, whatever the
compute units, with an error that names the input. Two cases only get a
warning: range-shaped inputs (`ct.RangeDim`), which run on the CPU and never
aborted, and any flexible layout on earlier macOS (D27). Always ship one
static-shape artifact per bucket, and run `ane_check` on each: it reads the
compute plan without predicting, so it rejects any flexible artifact without
running it.

Make every parity metric NaN-safe. `min(worst, cos)` returns the old value
when `cos` is NaN, which hid a NaN-producing CPU path behind "parity
1.000000" during the ModernBERT investigation; fail on non-finite output
instead, as the converters' parity checks do.

Three hard-won measurement gotchas: run residency checks on a quiet machine
(see the method note — concurrent GPU load makes ratios swing 2x), and
treat `E5RT ... ANECCompile() FAILED` stderr lines as *possibly transient
service state*, not proof of a bad artifact — the same file measured 1.48x
with failures and 2.63x clean forty minutes apart. Re-measure before
re-converting. Likewise, a compute plan with *every* operation unassigned
("no operations are assigned to any compute device"), or "internal
failure", can be a broken entry in Core ML's compiled-bundle cache for that
path (`~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`, D24).
An embeddinggemma artifact read that way repeatedly while predicting at ANE
speed, and the same file copied to another path read normally (2015/2024).
Those caches grow large: the parity suite's reached 40 GB and ane_check's
25 GB. They're safe to delete.
