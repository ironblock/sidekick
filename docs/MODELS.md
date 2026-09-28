# Model compatibility registry

What runs on the ANE through sidekick's Core ML encoder path, what doesn't,
and how to tell before spending an afternoon finding out. Every entry here
was measured on real hardware (Apple Silicon, macOS 26; ANE eligibility
re-checked on macOS 27); nothing is extrapolated from model cards.

How confident to be in each model on each compute path, graded on inputs
chosen to break it, is in [Confidence grades](#confidence-grades-the-parity-suite)
below. That is the number to trust; the converter parity in the first table
is measured on short prose and can flatter a model (LFM2.5).

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
| [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | 1024 | CLS | [convert_lfm25_embedding.py](../tools/convert_lfm25_embedding.py) | 0.9999 | 0.9870 | 693/698 (99.3%) | 2.49x / 1.91x / 1.66x |
| [codefuse-ai/F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 640 | last-token | [convert_qwen3_embedding.py](../tools/convert_qwen3_embedding.py) | 0.9999 | 0.99985 | 612/617 (99.2%) | 2.02x / 1.77x / 1.59x |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 768 | CLS | [convert_gte_modernbert.py](../tools/convert_gte_modernbert.py) | 0.999919 | 0.999793 | 794/805 (98.6%) | validated on macOS 27 (below) |

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
| LFM2.5 | 2.4x | 2.0x | 1.7x |
| F2LLM | 2.3x | 2.0x | 1.5x |
| gte-modernbert | 2.9x | 2.0x | 1.55x |

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
  read before trusting). Live `/v1/embeddings` worst parity 0.9856 (a
  483-token text); ~670 MB per bucket, 2.0 GB installed. The parity suite
  finds worse on the ANE: 0.954 on a 39-token URL, and similarity scores
  moving by up to 0.19 (grade D, below). CPU and GPU stay at 0.9999.
- **F2LLM-v2-160M** — the first **causal decoder** and first **last-token
  pooling** on the stack. A Qwen3 decoder; its QK-norm keeps activations
  tiny (max ~420), so it converts as cleanly as bge (ANE parity 0.99985). Last-token pooling is baked in-graph via the attention mask
  (no data-dependent index): `last_onehot = mask · (1 − shift_left(mask))`,
  then a masked sum. Validating it surfaced and fixed a real server bug:
  naive `take(max)` truncation dropped the trailing EOS that last-token
  pooling reads, collapsing over-length-doc parity to 0.36 — the server now
  preserves the final token on truncation (harmless for CLS/mean).
  ~950 MB installed; a 640-dim decoder for ~0.95 GB.

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

  Results:
  - parity is bucket-invariant (CPU_ONLY 0.999919, CPU_AND_NE 0.999793) and
    pad invariance is 1.0000000 on both paths;
  - live `/v1/embeddings` worst parity is 0.99896 over nine texts, including
    a 722-token input truncated to 512;
  - similarity structure matches fp32 (unrelated pair 0.417 vs 0.416);
  - ~7.8 ms warm for a short text, including HTTP.

  ~285 MB per bucket, 0.86 GB installed.

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
Each run happens on `.cpuOnly`, `.cpuAndGPU` and `.cpuAndNeuralEngine`, and
the output is compared with the model as published: sentence-transformers
in fp32, one input at a time.

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
content: LFM2.5 loses most on a URL, EmbeddingGemma on a run of digits.

M1 Max, macOS 27.0, September 2026. sidekick serves the ANE column; ms is
the median per input, ANE (CPU).

| model | CPU | GPU | ANE: worst case | ANE: similarity drift (bias) | ANE: rank flips | ms |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | A 0.99993 | A 0.999999 | **A** 0.99997 (empty input) | 0.002 (+0.001) | 0 | 1.9 (7.0) |
| gte-modernbert-base | B 0.99926 | A 0.99998 | **B** 0.99940 (a Markdown list) | 0.010 (+0.003) | 0 | 6.9 (18.6) |
| F2LLM-v2-160M | B 0.99987 | A 0.999999 | **B** 0.99966 (a run of digits) | 0.006 (0.000) | 0 | 5.3 (13.4) |
| embeddinggemma-300m | B 0.99989 | A 0.999998 | **A** 0.99999 (an over-length query) | 0.001 (0.000) | 0 | 8.1 (20.3) |
| LFM2.5-Embedding-350M | B 0.99990 | A 0.999999 | **D** 0.9536 (a URL) | 0.187 (−0.010) | 1,399 | 13.0 (34.5) |

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

All five pass every hard gate on every path:
- token ids identical to the reference pipeline's;
- finite output;
- exact pad invariance;
- bucket invariance of at least 0.9965 on the ANE (LFM2.5; 0.99998 for
  EmbeddingGemma since its rewrite) and exact on the CPU;
- bit-identical ANE output across two processes.

**What changes.** LFM2.5's 0.987 in the table above holds for prose. On
URLs, delimiters and repeated tokens its ANE path falls to 0.954, and a
similarity threshold calibrated on fp32 vectors can move by 0.19. Its CPU
and GPU paths stay at 0.9999, so the loss happens on the ANE.
EmbeddingGemma showed the same pattern until its MLP precision rewrite
(D17), which recovered it from D to A, so LFM2.5's SwiGLU MLP is the first
suspect. Until it's fixed, if inputs like these matter, prefer
bge-small, EmbeddingGemma, gte-modernbert or F2LLM, which are all B or
better on the ANE.

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
| LFM2.5 without pad zeroing | `convert_lfm25_embedding.py --no-pad-zeroing` | pad invariance 0.33–0.39 on every path; worst case 0.46–0.54 |
| naive truncation, on F2LLM | the daemon with `take(max)` truncation | token ids differ on both over-length inputs; cosine 0.23 |

**ONNX oracles.** Where a model's publisher, or onnx-community, ships ONNX
exports, the reference generator also runs them in ONNX Runtime on the same
token ids. The report shows them as report-only columns, never gated. They
measure the ecosystem, not sidekick:
- bge-small's and gte-modernbert's fp32 exports match torch at 1.000000,
  and gte-modernbert's fp16 export at 0.999994.
- gte-modernbert's published int8 export reaches only 0.892, on a
  repeated-syllable input. sidekick's ANE path is 0.9994.
- onnx-community's EmbeddingGemma exports, fp32 and fp16 alike, differ from
  sentence-transformers at 0.9936 on inputs of 400 tokens or more, and match
  below about 200. That fits the export applying Gemma's full 512-token
  window, which leaves no band within 512 tokens. transformers halves the
  window for bidirectional models (D17), and in fp32, no band against the
  halved window measures 0.997 on a 394-token text. It's a convention
  difference, not an error in either. sidekick follows
  sentence-transformers: its CPU path scores 0.99989 on the same inputs.
  The export's q8 variant reaches 0.972.

**Running it.**

```sh
# Once per converted model: reference vectors (Python; ONNX exports optional).
python tools/parity_reference.py "<data dir>/models/gte-modernbert-base" \
    --source Alibaba-NLP/gte-modernbert-base \
    --onnx Alibaba-NLP/gte-modernbert-base:onnx/model_int8.onnx
# The suite: every model that has a reference (about 10 minutes for five).
cargo run --release -p sidekick-embed --features coreml --example parity
```

- Each model runs on each path in its own process, after its compute plan
  is checked, so an artifact that aborts inside Core ML costs one cell of
  the report, not the run.
- [fixtures/parity/expectations.toml](../fixtures/parity/expectations.toml)
  holds the hard gates and, per chip, floors set at 1.5x the measured error.
  Floors turn this table into a regression test; `--suggest-floors` prints
  them from a run.
- On a chip with no floors, the suite enforces the gates and reports
  accuracy without judging it. Other Apple Silicon generations have their
  own ANE and will measure differently.
- Re-run after every macOS update, because the ANE's numerics come with the
  OS.

### Confidence by architecture family

For a model that isn't validated yet, the closest validated relative is the
best available prior. Each family has one validated model so far, so treat
its grade as a starting expectation, not a promise, and run the suite.

| family | validated | ANE grade | risks seen | inputs that find them |
|---|---|---|---|---|
| BERT (bge, MiniLM, e5) | bge-small-en-v1.5 | A | none | — |
| ModernBERT | gte-modernbert-base; laya's encoder (CLS ≥ 0.997) | B | fused attention drops the mask on the ANE (convert eager); the vectors of delimiter tokens are a little less accurate | pad and bucket invariance; delimiters |
| Qwen3 decoder, last-token pooling | F2LLM-v2-160M | B | truncation must keep the final token | over-length; the ids gate |
| Gemma3, bidirectional | embeddinggemma-300m | A | fp16 overflow without the range rewrite; without the MLP precision rewrite, 1–2.5% ANE loss on digits, long and repeated-token inputs (graded D) | numbers, long, degenerate |
| LFM2 hybrid (conv + attention) | LFM2.5-Embedding-350M | D | convolutions read pad states unless they're zeroed; up to 4.6% ANE loss on URLs and delimiters (suspected: SwiGLU MLP precision, D17) | pad invariance; delimiters, numbers |

## Incompatible / not integrated

| model | class | why |
|---|---|---|
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | late-interaction (multi-vector) | Emits one 128-d vector **per token**, scored with MaxSim — there is no single vector to return through `/v1/embeddings` or `sk_embed`. The encoder itself converts and offloads fine (smoke-tested at seq 256: per-token parity 0.9995 CPU / 0.9919 ANE, 2.0x ANE speedup, MaxSim ranking preserved — [smoke_lfm25_colbert.py](../tools/smoke_lfm25_colbert.py)), so a future late-interaction API could host it; nothing in today's API can. Its padded-batch conv semantics (expansion tokens must NOT be zeroed) also make real-token embeddings bucket-dependent under static shapes. Re-run on macOS 27: unchanged (per-token ANE parity 0.9919, 2.1x over CPU, MaxSim ranking preserved). |
| Apple NLContextualEmbedding | OS-provided contextual | Mean-pooled MLM states, strongly anisotropic (unrelated-pair cosine ~0.75) — unusable for similarity thresholds without post-hoc calibration sidekick doesn't own (D16). Re-measured on macOS 27: same model revision, same cosines (0.96/0.89 vs 0.75), faster (~11 ms). |
| Apple NLEmbedding.sentenceEmbedding | OS-provided static-ish | 2020-era quality, measurably weaker than bge-small on the same pairs; no prefixes, no control over dims (D16). Unchanged on macOS 27 (revision 1; 0.74/0.44 vs 0.14). |
| Apple FoundationModels | LLM | Has **no embedding API at all** (verified against macOS 26 SDK docs/headers, D16) — chat only. Still none in the macOS 27 SDK; 27's Spotlight integration is a search tool for sessions, not vectors. |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large encoder + decision head (classifier) | **Not an embedding model:** it scores options through a head that reads [MASK] marker tokens and CLS, so it would need its own endpoint. Its encoder converts like gte-modernbert (explicit attention; pad invariance 1.0). Seq 128, M1 Max, macOS 27: CLS ≥ 0.997 and per-token mean ≥ 0.983. Individual token vectors on short inputs can deviate on the ANE (worst 0.80–0.90 at 12 tokens; 0.16 on one test input), and that matters here because the head reads individual tokens. Validate end-to-end decisions before running it on the ANE; the GPU path is essentially exact (CLS 0.99999, 19 ms). An earlier CLS 0.07 on the ANE was the fused-attention mask bug. |

## Quick triage: is a model worth converting?

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
     - **Size**: validated up to 350M parameters, and each bucket stores the
       whole model at ~2 bytes/param.
     - **Needed sequence length**: buckets are ≤512, and the ANE's advantage
       shrinks with length.
     - **Output shape**: one vector per input fits `/v1/embeddings`;
       classifiers, rerankers and multi-vector models need a new API.
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
   | LFM2.5-Embedding-350M | RMSNorm | 1.9 | 17× | no range issue | 0.987 |
   | embeddinggemma-300m | RMSNorm | 152,485 (731) | 313× | range rewrite | 0.99999 after range + MLP rewrites |
   | gte-modernbert-base | LayerNorm | 47,973 (251) | 502× | no range issue | 0.9998 (explicit attention) |
   | laya (ModernBERT-large) | LayerNorm | 27,296 (379) | 556× | no range issue | CLS ≥ 0.997 (explicit attention) |

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
- **Explicit attention with fp16-safe masks.** Masks must use a finite
  constant such as -30000, not `-inf`/`finfo.min`, which NaN in fp16 (D15).
  The converters patch the mask builders to do this. Then use attention that
  converts to explicit matmul → softmax → matmul (gemma, F2LLM, ModernBERT
  via `attn_implementation="eager"`). Passing `scale=` to
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
- **Static-shape-friendly graph** — no data-dependent shapes. Stock
  `rotate_half`/`repeat_kv` and any `F.conv1d(padding=shape-derived)`
  need the traceable rewrites (D17 constraint 8, LFM2.5 constraint B).
- **fp16-safe activations** — calibrate first (forward hooks, max |activation|
  on a mixed corpus). Under ~30k: convert directly (bge, LFM2.5). Over:
  apply the D17 power-of-two range rewrite (gemma). Watch for `-1e9` mask
  constants (rewrite at -30000) and rmsnorm eps below ~1e-4.
- **Keep every linear-layer input near rms 1.** The ANE's `linear` op has
  an absolute precision floor on its input, not a relative one: its
  relative error is ~3e-4 / rms(input) (0.04% at rms 1, 2% at 0.016, 10% at
  0.004; the GPU is flat at 0.04%). Scaling the weights doesn't help; only
  the input's magnitude matters. Measure the rms of every linear input in
  fp32 with forward pre-hooks. Likely offenders are MLP down projections
  after a gated product (act(gate)·up is a product of two small numbers)
  and in-graph heads that pool at a reduced scale. Fix with a power-of-two
  scale folded into the weights upstream and cancelled by a
  scale-invariant norm or the final L2 normalize (EmbeddingGemma, D17).
- **Don't use Core ML's native GELU or SiLU ops on the ANE.** Measured
  absolute error on [-1, 1]: gelu ~6e-3, silu ~1.5e-2, versus tanh 1.6e-3,
  sigmoid 3e-3, and exact mul/relu. Build GELU as
  `x * (1 + tanh(x * (c + c·0.044715·x²)))` with the factor 2 absorbed
  downstream. Check the converted program for surviving `gelu` ops, since
  coremltools has passes that fuse such patterns. The SiLU replacement
  hasn't been validated yet.
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
chip.

Flexible input shapes are ruled out (D15), and on macOS 27 one kind became
dangerous. Under `.cpuOnly`, an artifact whose inputs have several
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
