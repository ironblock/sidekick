# Decision log

Autonomous judgment calls made during initial implementation, so review can
target the decisions rather than reverse-engineer them. Newest last.

## D1 — Daemon is OpenAI-compatible; the tier router lives behind it, not beside it
Per discussion: `sidekickd` mimics llama.cpp/MLX servers in miniature. The
"library" consumer shape still exists (crates are cleanly layered), but no
separate library facade was built yet — YAGNI until a second consumer shows up.

## D2 — macOS 26 (Tahoe) is the API baseline
The WWDC 2026 / macOS 27 `LanguageModel` provider protocol is deliberately
not used anywhere: corporate fleets lag, and there's no realistic test
vehicle. The Swift shim targets `macosx26.0` and only uses macOS 26 APIs
(`SystemLanguageModel.availability`, `LanguageModelSession`,
`DynamicGenerationSchema`). Revisit when 27 is deployable.

*Amended by D21:* the runtime floor is still macOS 26.0, but a build with
the macOS 27 SDK now uses selected 27 APIs behind compile- and run-time
gates. The `LanguageModel` provider protocol is still unused.

## D3 — Chat 503s when Foundation Models is unavailable; no generation fallback tier
The daemon returns an honest OpenAI-shaped `503 backend_unavailable` (with the
specific reason: AI toggled off, model downloading, ineligible hardware) and
`/health` exposes the same. Heuristic degradation ("title = first line") is a
*client* policy, and an Anemll-style local LLM tier is deferred until there's
evidence the FM-unavailable population matters (design doc §3, open question 1).

## D4 — Fake streaming in v1
`stream: true` is wire-compatible (SSE chunks: role → content → finish →
optional usage → `[DONE]`) but emits the whole completion as one content
delta. Sidekick-sized outputs finish in well under a second, so buying real
token streaming (callback across the C ABI) wasn't worth the FFI complexity
for v1. The shim upgrade path is noted in `bridge.swift`.

*Superseded for plain text by D23* (real streaming). Schema-constrained
replies are still sent as one delta.

## D5 — Session TTL = Foundation Models session reuse keyed by conversation prefix
OpenAI requests are stateless; FM sessions are stateful. After each response
the live session is filed under `sha256(instructions + full transcript incl.
our reply)`; a follow-up whose history hash-matches takes the session back and
sends only the new user message. Anything else cold-starts with a labeled
history replay in the prompt. TTL (default 300s) and an LRU cap of 8 bound
memory. This lives *inside* the FM backend (`ConversationCache`), keeping the
`ChatBackend` trait stateless.

## D6 — Multi-turn cold starts replay history as prompt text
macOS 26 has no public "init session from transcript" that fits the shim's C
ABI budget, so a cache-miss on a multi-turn conversation replays prior turns
as `User:`/`Assistant:` labeled text in a single prompt. Correct-but-slower
path; single-turn requests (the dominant sidekick workload) pass through
untouched.

## D7 — Token usage is estimated at ~4 chars/token
The macOS 26 Foundation Models API doesn't report token usage (the `usage`
property arrived with the 27 SDK). Clients get plausible numbers rather than
zeros; revisit under D2's review.

*Superseded on macOS 27 by D21* (real per-response usage). macOS 26 keeps
the estimate, now rounded up so a non-empty reply is never zero tokens.

## D8 — `response_format` mapping
`json_schema` → guided generation (the shim converts a JSON Schema subset —
object/string/integer/number/boolean/enum/array/nested objects/required — to
`DynamicGenerationSchema`). `json_object` → prompt nudge only, since there's
no schema to constrain against. Unsupported schema keywords fail loudly in
the shim rather than being silently dropped. D21 extends the fail-loudly rule to
errors (typed classification, real HTTP statuses) and to `stop`.

## D9 — Embedding purpose via non-standard `input_type`
OpenAI's embeddings API has no query/document distinction, but EmbeddingGemma
and bge-family models want prompt prefixes. Added the Cohere-style optional
`input_type: "query" | "document"` field (default `document`); prefixes come
from the model manifest, so standard OpenAI clients work unchanged.

## D10 — `dimensions` only honors manifest-declared Matryoshka dims
Truncating a non-Matryoshka embedding silently degrades quality, so a
`dimensions` value not in the manifest's `matryoshka` list is a 400, not a
best-effort truncation.

## D11 — Core ML loader requires/prefers precompiled `.mlmodelc`
The bindings' synchronous `compileModelAtURL` is deprecated but retained as a
convenience path for `.mlpackage`; docs steer users to
`xcrun coremlcompiler compile` at install time. No implicit compile cache was
built (Core ML itself caches ANE specialization per model+OS).

## D12 — `tokenizers` uses the pure-Rust `fancy-regex` backend
The default `onig` backend needs a C toolchain per target and broke the
aarch64-apple-darwin cross-check. Pure Rust keeps CI and cross-compiles
trivial; per-embed cost difference is irrelevant at sidekick batch sizes.

## D13 — Default bind `127.0.0.1:8790`, `/health` unauthenticated
Loopback by default because this fronts on-device models. When an `api_key`
is configured it guards `/v1/*` only; `/health` stays open for probes
(launchd, uptime checks) and leaks nothing beyond availability states.

## D14 — Compute units default to `.cpuAndNeuralEngine`
Not `.all`: keeping background work off the GPU is the project's thesis. The
wrapper exposes the choice; measurement can override. (D31 lets a manifest
name another choice, for a model the ANE runs badly; the default stands.)

## D15 — Per-bucket static artifacts, pooling baked into the model
The design doc (§5) assumed enumerated shapes keep an encoder on the ANE.
Hardware disagreed on both counts:
- A single `ct.EnumeratedShapes` artifact fails ANE plan compilation at load
  ("tensor_buffer has known strides while the model has FlexibleShapeInfo")
  and the whole encoder silently runs on CPU — 86 ms vs 2.4 ms/embed for
  bge-small at seq 128.
- A raw `last_hidden_state` output keeps a symbolic seq dim (coremltools
  can't unify the shape symbols of two enumerated inputs) which the
  ANE/CPU Espresso path rejects outright ("Data-dependent shapes were
  disabled") while `.all` (GPU) tolerates it — an especially nasty trap
  given D14.

*(On macOS 27 the flexible-shape artifact still falls back to the CPU under
`.cpuAndNeuralEngine`, but aborts the process at predict time under
`.cpuOnly`; `CoremlModel::load` refuses it there. See D27.)*

So: `artifact` supports a `{seq}` placeholder, one static-shape `.mlmodelc`
per bucket, loaded lazily and kept resident; pooling happens inside the
converted graph (statically-shaped `(1, dims)` output, manifest
`pooling = "none"`). Measured residency ratios for bge-small (M-series,
macOS 26.5): 3.4x/2.4x/1.75x at 128/256/512. Full recipe with the other
two traps (SDPA-not-eager fp16 NaNs, explicit position_ids for the
coremltools static-shape bug) in `tools/convert_bge_small.py`.

## D16 — No Apple OS-embedding tier (NLContextualEmbedding / NLEmbedding)
Evaluated as a candidate zero-download tier (July 2026) and declined.
FoundationModels exposes no embedding API at all (confirmed: framework
symbol index, Apple engineers at the WWDC25 group lab — "consider using
Core ML for your embedding model" — and WWDC26 answering RAG demand with a
Spotlight search tool instead of vectors). The NaturalLanguage options,
measured on-device against the same sentence set as the bge parity check:
- NLContextualEmbedding (512-d multilingual BERT, mean-pooled DIY): related
  pairs 0.96/0.89 vs unrelated 0.75 — rank order survives but the
  anisotropic baseline makes raw-cosine thresholds useless; ~17-22 ms warm;
  it's an MLM feature extractor, not a retrieval model.
- NLEmbedding.sentenceEmbedding (512-d, 2020-era): clean separation
  (0.74/0.44 vs 0.14) at ~5-7 ms — but bge-small on the ANE is stronger
  (0.78/0.81 vs 0.40 with retrieval-tuned training), faster (~2.4 ms), and
  already shipped. The static floor tier covers the no-download niche.
Not worth a third backend; revisit only if Apple ships a retrieval-tuned
embedding API.

## D17 — EmbeddingGemma ships ANE-default (amended: full parity after an MLP precision rewrite)
Gemma3's 300m encoder needed real conversion engineering
(tools/convert_embeddinggemma.py): a calibrated power-of-two fp16 range
rewrite (the residual stream reaches ~1.5e5, past fp16 max — scale-invariant
RMSNorm rewrites make it exact; fp32 parity gates at 1.000000), hand-built
attention masks (transformers halves config.json's sliding_window to 257
for bidirectional models; the 512 bucket has a live band, so the parity
gates include a ~400-token text — which, the amendment found, never ran),
and shape-arithmetic-free rotate_half / repeat_kv rewrites for
coremltools' static-shape 'int' op crash.

*The next paragraph is superseded by the amendment below.* After all of
that, the ANE itself costs ~1% cosine — intrinsic fp16
accumulation across 24 layers, insensitive to residual scale and not
attributable to softmax (measured; see the script docstring). The matrix at
bucket 128, worst-of-parity vs fp32 sentence-transformers reference:
CPU_AND_NE 0.9905 at 7.9ms; CPU_ONLY 0.9999 at 25.1ms; ALL/GPU 0.999999 at
11.9ms. We keep the D14 no-GPU default and take 3x latency for 1% cosine:
rank order in similarity tests is preserved with wide margins, and callers
needing exact parity can use bge-small (0.99998 on ANE) or load with
CpuOnly. Conversion gates are per-path: CPU_ONLY >= 0.999 (conversion is
faithful), CPU_AND_NE >= 0.985 (what the ANE delivers). Artifact cost:
~600MB per bucket; multifunction weight sharing is a possible future
optimization.

**Amendment (September 2026, macOS 27.0, M1 Max): the ~1% was not
intrinsic.** It came from two ANE arithmetic limits inside the MLP, and a
converter rewrite removes it. Re-examined after D25 found ModernBERT's
"intrinsic" loss to be a Core ML bug.

Ruled out first:
- *fp16 itself.* The same range-rewritten graph in fp16 scores 0.999998 in
  PyTorch (CPU and MPS) and 0.999999 on Core ML's GPU path.
- *A slightly different model.* The ANE output isn't closer to any
  variant: sliding window 256 instead of 257, no window, exact-erf GELU,
  pooling over pads, or attended pads. It scores 0.9905 against every
  plausible one, and the pad variants are far worse. Pad invariance is
  exact. The attention is explicit (Gemma's SDPA call passes a scale, so
  coremltools doesn't fuse it), so D25's mask bug doesn't apply.

Located:
- *By layer, teacher-forced* (each layer converted alone and fed the exact
  fp32 residual): attention adds at most 0.3% local error on the ANE, less
  than the CPU path's 0.4–0.9%. The MLP adds up to 19% (most layers 1–5%),
  worst in layers 19–21. The loss is injected by one sub-block, not
  accumulated evenly.
- *By op class kept in fp32* (which moves it to the CPU): attention 0.9905
  (no change), the MLP 0.99996, GELU alone 0.9913.

The two causes, measured:
- **The ANE's `linear` op has an absolute precision floor on its input.**
  Relative error ≈ 3e-4 / rms(input): on a synthetic 1152→768 projection,
  0.04% at rms 1, 0.4% at 0.06, 2% at 0.016 and 10% at 0.004; the GPU is
  0.036% at every scale. Scaling the weights, i.e. the output, changes
  nothing. Gemma's down_proj input, gelu(gate)·up, has rms 0.004–0.03 in
  layers 18–23. Teacher-forced, layer 20's down projection alone loses
  11.5% at natural scale and 0.16% with its input scaled ×64.
- **Core ML's `gelu` op is coarse on the ANE:** ~6e-3 absolute error on
  [-1, 1] (GPU ~7e-5), which is where most gate activations lie. SiLU is
  worse (~1.5e-2); tanh (1.6e-3), sigmoid (3e-3) and exp (within a few
  ulp) are better; mul and relu are exact.

Each effect hides the other: fixing GELU alone gives 0.9913, fixing the
magnitude alone 0.9997 (0.9992 at bucket 512). This also explains the
original observations. The residual scale K never reaches the normalized
MLP interior, and fp32 softmax couldn't help because attention was never
the problem.

**The fix** (tools/convert_embeddinggemma.py, constraint 9):
- fold a calibrated power-of-two scale into each up_proj so the down_proj
  input has rms ~1 (totals 2–256 across layers). It is divided out exactly
  in the scale-invariant post-feedforward RMSNorm, the constraint-5
  machinery;
- build GELU from tanh, mul and add (≤ 9e-4 on [-1, 1] on the ANE);
- pool at natural scale before the dense head, whose inputs had been at
  1/32 scale (rms 0.015–0.05). This is a small gain, worst 0.999982 →
  0.999989 at bucket 512 and none at 128.

**Results**, buckets 128/256/512, worst over the parity set:
- CPU_AND_NE 0.999996 / 0.999996 / 0.999989 (was 0.9905, and 0.981 on the
  long text at 512);
- CPU_ONLY 0.999940 / 0.999940 / 0.999933, so the ANE is now the more
  accurate path; GPU 0.999999;
- the fp32 rewrite gate is exact (1.000000);
- 2161/2170 operations on the ANE (was 2015/2024), and pad invariance
  1.0000000 on both paths;
- live `/v1/embeddings` worst parity 0.999982 over 13 inputs, including
  the long text and a 527-token document truncated to 512;
- on the parity suite's 51-input adversarial corpus (D26: runs of digits, URLs,
  repeated tokens, code, 512-token and over-length inputs), run through
  sidekick's own embedding path: the ANE's worst case went from 0.975 to
  0.99999, pairwise-similarity drift from 0.042 to 0.001, and rank flips at
  a 0.02 margin from 122 to 0.

The rescale is free. The explicit GELU costs nothing at bucket 128 and
~14% at 512 (34.8 → 39.6 ms, interleaved on the same loaded machine);
`ane_check` ratios are 2.6x/2.4x/1.7x.

**The parity gate had a hole.** The "~400-token" text was 527 tokens with
the document prefix, so `fitting_pairs()` dropped it at every bucket and
the sliding band was never gated. Against it, the old 512 artifact scored
0.981, below its own 0.985 gate. The text is now 394 tokens. The converter
fails if no parity text reaches the band where the band is live. It also
gates pad invariance, rejects fused attention and native `gelu` ops, treats
NaN as failure, and gates the ANE path at 0.999.

**Consequences.**
- For this model the D14 trade-off is gone: CpuOnly is no longer the
  route to exact parity.
- Two conversion rules, now in docs/MODELS.md: keep every linear-layer
  input near rms 1 on the ANE, and don't use Core ML's native GELU/SiLU
  ops there.

**Not done:** the other validated models haven't been checked against
these rules. LFM2.5 (SwiGLU MLPs, ANE parity 0.987) is the obvious
candidate, and its converter has the same gate hole: its "~480-token"
text is 523 tokens with the `document: ` prefix, so it never ran.

## D18 — Embeddings get a C ABI dylib; chat stays daemon-only
Amends D1 with the use case it was waiting for: a host app that wants
bge-on-ANE embeddings opportunistically, across "daemon running / installed
but not running / not installed". `sidekick-embed-ffi` builds
`libsidekick.dylib` (~5 MB): `sk_pool_open/close/models`, `sk_embed_dims`,
`sk_embed` — panic-safe, thread-safe, no tokio, no FoundationModels
linkage, same models directory as the daemon. Hosts probe: daemon `/health`
(short timeout) → `dlopen` → their own fallback (docs/INTEGRATING.md).
Chat is deliberately excluded: it would drag the Swift shim (and the
Xcode 26 build requirement) into every host, and sessions want a daemon
lifetime. The daemon remains the primary interface — it shares resident
models across clients; the dylib trades that for zero service management.

## D19 — Model registry doc + LFM2.5: flexibility validated, ColBERT declined (amended: full ANE parity after a precision rewrite)
Two LiquidAI models were run through the stack to test whether the recipe
generalizes beyond BERT-class and Gemma-class encoders — it does, with one
new constraint. LFM2.5-Embedding-350M (hybrid: 10 short-conv + 6
full-attention blocks, custom bidirectional remote code) converted and
shipped (tools/convert_lfm25_embedding.py); the new lesson is constraint D:
token mixers other than attention (convs, SSMs) read pad STATES, not pad
keys, so pads must be zeroed before every conv layer or right-padding
contaminates the tail and embeddings become bucket-dependent (measured
0.905 → 0.987 parity). LFM2.5-ColBERT-350M is documented as not integrated:
late-interaction emits per-token vectors scored with MaxSim — no single
vector exists for /v1/embeddings or sk_embed to return — and its
query-expansion tokens (mask=0 but scored) require the padded-batch conv
semantics the embedding model must avoid. The encoder itself converts and
offloads (tools/smoke_lfm25_colbert.py), so a late-interaction API surface
remains possible if wanted. Measured results and a
"will a new model convert?" checklist live in docs/MODELS.md, which is now
the registry of validated/incompatible models.
*Qualified by D26:* 0.987 is LFM2.5's ANE parity on prose. On URLs and
delimiters the ANE reaches 0.954, and similarity scores move by up to 0.19.
*Superseded by the amendment below.*

**Amendment (September 2026, macOS 27.0, M1 Max): LFM2.5's ANE loss was
the same two limits as EmbeddingGemma's (D17 amendment), spread over the
whole block.** A converter rewrite removes it.

Ruled out first: the graph is faithful and fp16 itself is fine. The parity
suite (D26) grades the CPU path B (0.9999) and Core ML's GPU path A
(0.999999). Only the ANE is low: D, 0.954 on a URL.

Located:
- *By layer, teacher-forced* (each layer converted alone and fed the exact
  fp32 residual): on the ANE, the conv or attention branch adds 1–8% local
  error and the MLP 3–13%, in every layer. The CPU adds 0.3–0.9%, the GPU
  0.03%. Unlike EmbeddingGemma, where only the MLP was lossy, every
  sub-block loses.
- *By op class kept in fp32*, moving it to the CPU: the conv op 0.9539 and
  attention's matmul/softmax 0.9460, so neither is the cause. SiLU alone
  gives 0.9777, the MLP down projections alone 0.9788, and both 0.9974. Kept
  as a whole class, the linears or the MLP moved the entire graph off the
  ANE, which isolates nothing.
- *By input magnitude* (fp32, over the suite's inputs): the rms of every
  output projection's input is far below 1. It is 0.003–0.036 for the MLP
  down projection, 0.003–0.06 for the conv block's out_proj, 0.007–0.07 for
  attention's out_proj, and 0.04–0.16 for q/k/v. At the ANE linear's
  ~3e-4 / rms error, that predicts 0.5–10% per projection. QK-norm and
  small norm weights (attention layers' operator norms average 0.06–0.13)
  keep this model's activations tiny. That spares a range rewrite, which is
  why the converter called LFM2.5 the "easy class", and it is exactly what
  the ANE's `linear` punishes.

**The fix** (tools/convert_lfm25_embedding.py, constraint E): calibrated
power-of-two scales bring each of those inputs to rms ~1, per layer.
- Attention: the operator norm's weight ×8–16 scales q/k/v. q and k are
  re-normalized per head, so only their RMSNorm eps moves (×s²). v_proj
  takes attention's output to ×16–64.
- Conv: the operator norm ×2–4, B's rows of in_proj so the conv input B·x
  has rms ~1 (×2–8 total), and C's rows so out_proj's input C·conv(B·x)
  does (×32–128 total).
- MLP: w3 ×m, so the down projection's input is ×32–128.
- Nothing follows these branches but the residual add, so there is no norm
  to absorb a scale, as there was in Gemma. Each branch's output is
  multiplied by 1/S before the add. The multiply is explicit, not folded
  into the weights, because dividing weights by up to 256 would push small
  ones into fp16's subnormal range.
- SiLU is built as x·(1 + tanh(x/2)), and its factor 2 is folded into the
  MLP's 1/S. Core ML's native silu is off by up to ~1.5e-2 on [-1, 1] on
  the ANE. `x * sigmoid(x)` isn't an alternative: conversion fuses it back
  into the native op, with identical output and op count.

At bucket 128 on the suite's inputs, the ANE's worst case:
- 0.9536 before;
- 0.9917 with the explicit SiLU alone;
- 0.9758 with every rescale but the native SiLU;
- 0.9974 with SiLU and the MLP rescale;
- 0.9990 adding attention;
- 0.99999 adding the conv block.

**Results:**
- converter parity CPU_AND_NE 0.999992 at every bucket (was 0.987010) and
  CPU_ONLY 0.999935 / 0.999935 / 0.999912; the fp32 rewrite gate is exact
  (1.0000000);
- 773/778 operations on the ANE (was 693/698), and pad invariance
  1.0000000 on both paths;
- parity suite ANE grade **A**: worst 0.999988 (was 0.953594), mean
  0.999994, similarity drift 0.0033 (was 0.187), bias 0.0000 (was −0.010),
  0 rank flips (was 1,399), and bucket invariance 0.999995 (was 0.9965).
  The ANE path now tracks fp32 more closely than the CPU path does (drift
  0.0076);
- live `/v1/embeddings` over all 51 suite inputs: worst 0.999988.

The rescales are free. The explicit SiLU costs 13–16% of ANE latency at
every bucket (13.4 → 15.5, 28.8 → 33.0, 59.1 → 66.8 ms, interleaved on
the same loaded machine). `ane_check` ratios are 2.3x/1.9x/1.6x.

**The same gate hole as EmbeddingGemma's.** The "~480-token" parity text
was 523 tokens with the `document: ` prefix, so no bucket ever ran it. It
is now 471 tokens, and the converter fails if it stops fitting the 512
bucket. The converter also gates pad invariance, rejects fused attention
and native silu/gelu ops, treats NaN as failure, and gates the ANE path at
0.999.

## D20 — Two more architecture classes: ModernBERT rejected, Qwen3 decoder validated (amended: F2LLM precision rewrite)
Triaged the MTEB/CoIR leaderboards and validated the two families sidekick
hadn't covered, one of each verdict:

- **ModernBERT (gte-modernbert-base) is ANE-incompatible** — a documented
  negative result. It converts faithfully (Core ML fp32 parity 1.0) and is
  perfect in PyTorch fp16 (0.999999), but Core ML's ANE fp16 delivers only
  0.9038. Root cause: a massive-activation outlier (dim 251 ~40000 in the
  residual) dominates every LayerNorm variance and crushes the other dims
  below the ANE's fp16 between-op storage precision. PyTorch survives via
  fp32-internal reductions; the ANE can't. Forcing sensitive ops to fp32
  restores parity but relocates off the ANE (~5x, no benefit); the macOS26
  compiler is identical; a D17 range rewrite underflows the compensated eps.
  Affects the whole ModernBERT family. New checklist rule: compare
  CPU_AND_NE vs PyTorch-fp16 (not just fp32) — a gap is the outlier
  signature. QK-norm models avoid this by construction.

  *Superseded by D25:* the root cause was Core ML's fused attention op
  dropping ModernBERT's attention mask on the ANE, not the outlier. With
  explicit attention, gte-modernbert-base converts at 0.9998 and is
  validated. The two checklist rules above are withdrawn.

- **Qwen3 causal decoder (F2LLM-v2-160M) validated** — the first decoder and
  first last-token pooling on the stack (tools/convert_qwen3_embedding.py).
  QK-norm keeps activations tiny (max ~420), so ANE parity is 0.99985 — as
  clean as bge. (That this "vindicated the ModernBERT lesson" is withdrawn:
  see D25.) Last-token pooling is
  baked in-graph via the attention mask (last_onehot = mask·(1−shift_left(
  mask)), masked sum → (1, dims)); no server pooling change. It did require
  one server fix: naive take(max) truncation dropped the trailing EOS that
  last-token pooling reads (over-length-doc parity 0.36 → 0.99985), so the
  server now preserves the final token on truncation (harmless for
  CLS/mean; unit-tested). docs/MODELS.md carries both results and the
  hardened last-token checklist.

**Amendment (September 2026, macOS 27.0, M1 Max): F2LLM's remaining ANE
loss was mostly the native SiLU.** The parity suite (D26) graded its ANE
path B: 0.99966 on a run of digits, drift 0.006. Its CPU path grades B
(0.99987) and its GPU path A. The D17/D19 method applied only in part:
- *SiLU is the main cause.* Keeping only the silu ops in fp32 (on the CPU)
  gives 0.99998. Building SiLU from tanh on the ANE gives 0.999965 at
  bucket 128 with no latency cost.
- *Small linear inputs mostly aren't.* F2LLM's activations are about ten
  times LFM2.5's: median rms 0.08–0.5 into the MLP down projection, and
  keeping the down projections in fp32 changes nothing (0.99967). Only
  attention in the early layers is small. Those layers' input norms have
  weights of ~0.13–0.18, so q/k/v arrive at rms 0.06–0.18 and o_proj at
  0.02–0.35. Rescaling them lifts long inputs, from 0.99995 to 0.99998 on
  a 512-token query.
- *By layer, teacher-forced:* on the ANE, attention adds 0.1–0.4% local
  error (the CPU path 0.6–0.8%) and the MLP 0.5–1.9% (the CPU path
  0.5–1.0%). Per layer, the ANE is close to the CPU; the SiLU is the
  difference.

The converter (constraint D) builds 2·SiLU from tanh, with up_proj's
weights taking the 1/2, so no op is added. It also rescales attention's
inputs the way D19 does: the input norm's weight ×1–8 (q/k RMSNorm eps
×s²), v_proj, and an explicit ×1/S before the residual add. The MLP
rescale isn't used, since it changed nothing measurable.

**Results:**
- converter parity CPU_AND_NE 0.999979 at every bucket (was 0.99985) and
  CPU_ONLY 0.999924; the fp32 rewrite gate is 0.9999999;
- parity suite ANE grade **A**: worst 0.999972 (was 0.99966), mean
  0.999986, drift 0.0019 (was 0.0057), 0 rank flips, bucket invariance
  0.999991;
- 648/653 operations on the ANE (was 612/617), pad invariance 1.0000000;
- live `/v1/embeddings` over all 51 suite inputs: worst 0.999972;
- latency unchanged (−1% to +4%); `ane_check` ratios 2.7x/2.0x/1.5x.

The CPU path stays B (0.99988). Its error is the CPU path's own and
doesn't move with the rewrite.

**The converter had stopped running.** Its "traceable" repeat_kv computed
num_kv_heads·n_rep from the traced shape. That traces to the Int op that
crashes coremltools 9 under torch 2.13 (D17 constraint 8), and it had been
on the SDPA path since this decision's review. It now uses the
shape-arithmetic-free `expand(-1, …).flatten(1, 2)` of the Gemma and
LFM2.5 converters. The converter also gained the D19 gate hardening: pad
invariance, a rejection of surviving silu/gelu/fused-attention ops,
NaN-safe metrics, an fp32 exactness gate, and an ANE gate of 0.999. Its
long parity text is truncated to 512 tokens on purpose, so it already ran.

## D21 — macOS 27: real usage and model facts, typed errors; the shim stays
macOS 27 reworked Foundation Models. This entry records what sidekick
adopts, what it declined, and the measurements behind both (M1 Max,
macOS 27.0, Xcode 27.0).

**Build.** The shim keeps its macOS 26.0 runtime floor. APIs that exist only
in the 27 SDK are compiled when build.rs finds SDK ≥ 27 (`SK_SDK_27`) and run
under `#available(macOS 27, *)`. SDKs older than 26.4 are a build error.
Release binaries must be built with the 27 SDK — built with a 26.x SDK, a
binary silently behaves like macOS 26 even on 27 — so the release workflow
pins Xcode 27, fails on an older SDK, and proves the result still loads on
macOS 26. `sidekickd --version` and `/health` report the SDK in use.

**Token usage** comes from the per-response `usage` on macOS 27. Prompt
counts include instructions and chat framing and matched Apple's own
`fm serve` exactly on single-turn requests (62, 63 and 382 tokens in a spot
check). `prompt_tokens_details.cached_tokens` reports what the reused session
(D5) served from cache: a follow-up measured 69 of 87 prompt tokens cached.
`completion_tokens` can exceed `max_tokens` because output counts include
framing.

**`finish_reason: "length"`** is inferred; Foundation Models reports no
finish reason. For plain text, a reply is truncated when its re-counted
tokens, net of the counter's constant overhead (1 token on macOS 27), reach
`max_tokens`. Measured replies cut at 5, 12 and 30 re-counted to exactly the
limit; natural ones stayed well below. The re-count (~45 ms) runs only near
the limit: fewer UTF-8 bytes than the limit, or on macOS 27 an output count
under it, rules truncation out. Constrained output can't be re-counted
(structure tokens), and a truncated constrained reply still reports itself
complete while missing required properties. There the macOS 27 output count
is the only signal, and clients must check `finish_reason`.

**`stop`** is honored; it used to be silently ignored. The reply ends before
the earliest match. Up to four sequences are accepted. `stop` can't be
combined with `json_schema`, since cutting constrained output breaks the
schema. A session whose reply was cut by `stop` isn't cached, because its
transcript holds text the client never saw. A `max_tokens` cut is cached.

**Model facts.** `/health` reports the model variant (display name plus a
stable id), its real context size, and what the model supports.
`SystemLanguageModel.variant` is read-only: the OS decides. Per Apple, AFM 3
Core Advanced needs a Mac with M3 or later and 12 GB+ of memory; an M1 Max
gets AFM 3 Core (4096-token context, no reasoning).

**Typed errors.** On macOS 27 an over-long prompt returned HTTP 500 instead
of 400: overflow had been recognized by matching macOS 26's error text, and
27 throws a new type, `LanguageModelError.contextSizeExceeded`. The shim now
classifies errors by type, macOS 27's and macOS 26's, into kinds the server
maps to statuses:

| Kind | Status |
|---|---|
| context overflow | 400, with real counts on 27 |
| guardrails | 400 `content_filter` |
| rate limit | 429 with Retry-After |
| transient | one retry on a fresh session, then 503 |
| assets unavailable | 503 |
| bad schema | 400 |
| unsupported language | 400 |

The only string matching left is a fallback that recognizes 27's overflow
message, for binaries built without the 27 SDK. A shim self-test checks the
classification in CI on both SDKs, without a model.

**Not adopted.**
- *`PrivateCloudComputeLanguageModel`.* It reports available on an eligible
  Mac, but `respond` fails from an unentitled binary (ModelManagerError
  1046). Apple documents a managed entitlement for PCC development. It would
  also move inference off-device, which is contrary to this project's
  premise.
- *Apple's `fm serve`*, the macOS 27 CLI's OpenAI-compatible server, as a
  replacement for the shim.
  - It streams when `stream` is omitted.
  - It ignores legacy `max_tokens`: 399 tokens came back for a limit of 5.
  - It rejects `json_object` and `stop`.
  - It reports no cached tokens, since it uses a fresh session per request.
  - It needs a machine-wide license acceptance and a second process to
    supervise.
  - It is useful as a conformance reference, and the token counts above were
    checked against it.

## D22 — Reject request parameters the daemon can't honor
Extends D8's fail-loudly rule from `response_format` to the rest of the chat
request. Silently answering as if a parameter weren't there gives the client
a wrong answer that looks right. So a request that depends on something
sidekick can't do now gets a 400:
- `n` > 1
- non-empty `tools` / `functions`
- a `tool_choice` / `function_call` that requires a tool or names one
- `logprobs: true`, or `top_logprobs` > 0

The harmless forms that OpenAI SDKs send by default are still accepted:
`n: 1`, empty tools, `tool_choice` `"auto"`/`"none"`/null, `logprobs:
false`. Everything else unknown is still ignored on purpose, including
`seed`, `top_p`, penalties and `user`. Foundation Models' sampling options
could map `seed`/`top_p` in the future. This is a behavior change for
clients that relied on those parameters being dropped, so it warrants a
minor version bump when released.

**Amendment (September 2026): the same rule for `/v1/classify`, and
malformed bodies are a 400 everywhere.** `/v1/classify` (D28) applies this
decision to every field vLLM defines for classification. Each one is
honored or rejected:
- `add_special_tokens: false` is a 400;
- chat-form `messages` and token-id input are 400s;
- `truncation_side: "left"` is a 400 on a model whose format truncates by
  design (laya).

sidekick's own extension fields are held to a stricter rule than unknown
fields. Sending one to a model whose task or format doesn't take it is a
400, not ignored: `candidate_labels` on a fixed-label model, or
`question_type` on a model that isn't in laya's format. A client that sends
an extension is asking for it, so dropping it would give a wrong answer
that looks right. Unknown fields outside the extension set, `user`
included, are still ignored.

Every JSON body on every route now goes through one extractor. Bad
syntax, a missing field, a wrong type, or a missing JSON content type is an
`ApiError` 400 in the API's usual error shape. Before, axum answered these
with a plain-text 422 or 415. A body over the size limit is still a 413.

## D23 — Real streaming for plain text
Supersedes D4 for plain text. The shim's `sk_fm_respond_stream` iterates
`streamResponse` and hands each snapshot's cumulative text to a C callback.
The callback can stop generation by returning nonzero; the shim then leaves
the stream loop, which ends generation promptly on macOS 27. Rust turns
snapshots into deltas with `StreamShaper`, the incremental form of the
reply post-processing (speaker-label strip, stop sequences, no trailing
U+FFFD). It holds back only text a later snapshot could still change, so
streamed and non-streamed replies are identical.

Rules, each backed by a measurement or a test:
- **Never reuse an interrupted session.** On macOS 27, calling `respond` on
  a session whose stream was left early, by break or by task cancellation,
  traps the process (EXC_BREAKPOINT, uncatchable). Stopped, errored and
  diverged streams drop their session. A session is cached only when the
  stream finished on its own and the client holds exactly the final reply.
- **No retry after the first delta.** The client would see the reply start
  over. Retries on a fresh session still happen before any text is sent.
- **Stop sequences end generation**, rather than trimming a finished reply.
  Plain-text requests with `stop` use the streaming path internally even
  when `stream` is false. Measured: a count stopped at "5" returned in
  0.76 s, against 3.5 s uncut.
- **Client disconnects stop generation.** The SSE body owns the channel
  receiver, so when a client leaves, the next snapshot's send fails and
  generation stops (verified live). The same happens at the request
  timeout. A stall before the first snapshot can't be interrupted, because
  cancellation is checked per snapshot.
- **Errors keep their HTTP status until text is sent.** The response is
  committed on the first delta or on completion. After that, an error
  becomes an `{"error": …}` event, and a guardrail stop becomes a
  `content_filter` finish.
- **Constrained output is sent whole.** Partial JSON snapshots aren't
  prefix-stable.

Snapshots are expected to extend each other. One that rewrites
already-sent text is skipped, and emission resumes if a later snapshot
agrees again. If the final reply contradicts what was sent, the stream ends
with an error instead of silently sending different text. Apple's own
`fm serve` carries a warning for this case. All snapshots observed on
macOS 27 extended their predecessors.

## D24 — ANE eligibility is judged by Core ML's compute plan; the latency ratio is evidence
`ane_check`'s pass/fail used to be a latency ratio: median `.cpuOnly` over
`.cpuAndNeuralEngine`, gated at 1.5x. On macOS 27 that gate flagged
bge-small's 512 bucket (1.23–1.26x over five runs) as "not resident". Its
compute plan is identical to the buckets measuring 2–3x, with 229 of 245
operations on the ANE. The ratio moves with things other than residency:
machine load, and a faster CPU path.

`ane_check` now reads Core ML's compute plan first (`MLComputePlan`, macOS
14.4+, exposed as `sidekick_coreml::compute_plan`). The plan reports the
device each operation of the `main` function is assigned to, without
running the model. The model passes when:
- every compute-heavy operation (matmul, linear, conv, einsum, attention)
  is on the ANE, and
- at least 80% of assigned operations are on the ANE.

Core ML reports no per-operation costs (`estimatedCost` is empty on
macOS 27), so operation counts are unweighted; the heavy-operation rule
covers that gap. Measured on macOS 27:
- The four validated encoders pass at 93.5–99.6%, identically at every
  bucket. What stays on the CPU is mask and cast plumbing plus the
  embedding gather.
- A flexible-shape bge artifact (the configuration D15 rules out,
  reproducible with `convert_bge_small.py --enumerated-shapes`) scores 0%.

A failing plan ends the run before any prediction. That matters on
macOS 27, where predicting with that flexible-shape artifact under
`.cpuOnly` aborts the process with an Objective-C exception. (This entry
first said "whatever the compute units"; D27 measured otherwise.)

A plan can come back empty (every operation unassigned) or fail with
"internal failure" while the artifact is fine. Core ML caches compiled
bundles per executable (`~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`),
keyed by artifact path. A broken entry makes every plan read for that path
fail the same way, until the entry is gone. A copy of the artifact at
another path reads normally, and so does the same path read by a
differently named executable. On macOS 27 this hit 4 of 15 buckets, in a
cache that had grown to 40 GB. `verdict` fails an empty plan, so don't read
it as "ineligible". The parity suite (D26) re-reads an unavailable plan from
an APFS clone before failing the model.

The latency ratio is still measured and reported, as runtime evidence.
The plan is the compiler's intent and can't see a runtime ANE compile
failure; a ratio near 1.0 would. `ane_check` warns below 1.1x but doesn't
fail on the ratio. MODELS.md records plan shares alongside the ratios.

A load-time guard in `CoremlModel::load` doesn't read the plan: that costs
1.3–18 s cold per bucket. D27 adds one that checks input shape constraints
instead, and refuses only on macOS 27, so macOS 26, where such models still
run (slowly) on the CPU, doesn't regress.

## D25 — ModernBERT validated: its rejection was a Core ML attention-mask bug (amended: range rewrite for the ANE linear's 2^15 limit)
Supersedes D20's ModernBERT verdict. Found by an adversarial review of the
macOS 27 re-investigation (September 2026, M1 Max, macOS 27.0), then
reproduced independently.

**What happens.** Converted with PyTorch `sdpa` attention, ModernBERT lowers
to Core ML's fused `scaled_dot_product_attention` op. In this graph, on the
ANE, that op ignores its attention mask: pads are attended and the sliding
window is dropped. Evidence:
- The ANE output matches an *unmasked* fp32 reference at 0.99998, but the
  intended one at only 0.87–0.975.
- The output depends on the content of the masked pad positions (pad ids
  0 vs random: cosine 0.61–0.94). A correctly masked model can't see its
  pads.
- The same op on the CPU returns NaN whenever fewer than 64 of 128
  positions are real (a query whose whole sliding window is masked).

At first the trigger wasn't isolated. A single synthetic fused-SDPA layer
honoured its mask, while ModernBERT cut down to one layer didn't, and
bge-small's fused SDPA passed the same pad-invariance check. The amendment
below explains all three.

**The fix** is to convert with `attn_implementation="eager"`, so attention
becomes explicit matmul → softmax → matmul:
- the graph stays on the ANE (794/805 ops);
- parity is CPU_AND_NE 0.999793 / CPU_ONLY 0.999919 at every bucket, and
  pad invariance is exact;
- live `/v1/embeddings` worst parity is 0.99896;
- the ANE is 2.9x/2.0x/1.55x over the CPU.

gte-modernbert-base is the fifth validated ANE model. laya's
ModernBERT-large encoder also converts accurately (CLS ≥ 0.997); its earlier
CLS 0.07 on the ANE was the same bug.

**What was wrong before.** D20 blamed the massive activation (dimension 251,
~48,000 on delimiter tokens) for crushing fp16 LayerNorm precision. The
macOS 27 follow-ups narrowed the loss to attention, but still read it as
fp16 precision:
- a range rewrite at 1/8–1/256 gave at most 0.93;
- fp32 LayerNorm changed nothing;
- fp32 attention gave 0.9998.

The "CPU parity 1.000000" controls behind those readings were NaN outputs
hidden by `min(worst, nan)`, and every short-text parity input was mostly
padding. The same outlier is harmless with explicit attention. Its only cost
is a small per-token effect on those tokens' own output vectors, while
pooled outputs are unaffected.

**Consequences.**
- `ane_check` gates pad invariance on the ANE and CPU paths. It rejects the
  fused artifact (cosine 0.27) and passes every validated model. The
  ModernBERT converter gates it too.
- Parity metrics must fail on non-finite output.
- `tools/probe_activations.py` no longer calls massive activations hostile;
  fp16 range is its only calibrated verdict.
- D20's rules "a CPU_AND_NE vs PyTorch-fp16 gap is the outlier signature"
  and "QK-norm models avoid this by construction" are withdrawn. QK-norm
  does keep activations small, which spares a range rewrite.

**Amendment: the trigger, isolated** (September 2026, M1 Max, macOS 27.0,
coremltools 9). The ANE's native fused attention ignores its `attn_mask`
when the mask is an input of the ANE procedure that runs the attention,
i.e. when no op inside that procedure computes it. The output then equals
unmasked attention exactly. `tools/repro_sdpa_mask.py` reproduces it with
one random-weight layer in seconds.

How it was found. The bisection ran from ModernBERT cut to one layer
(fails) toward a synthetic layer (passes), one factor at a time. It used a
NaN-safe drop fraction: distance to the masked fp32 reference over the
distance between masked and unmasked references, 0 when the mask is
honoured and 1 when it's ignored. Pad invariance was measured alongside.
- A reimplementation of that layer, bit-exact with transformers in fp32,
  passed. Its MIL program had the same ops as the failing one, only in a
  different order.
- transformers builds the masks in `_update_attention_mask`, before the
  embedding gather. The partitioner puts that plumbing on the CPU with the
  CPU-only gather, so the mask reaches the ANE as an input. Built after the
  gather, the same ops run on the ANE and the mask holds, through the
  fallback described below.
- With the mask fed as a model input, the fully synthetic layer fails
  too: drop 1.000. So do these variants of it:
  - every q/k/v layout tried (transpose + unbind, permute, separate linears);
  - with and without RoPE, and with and without biases;
  - mask shapes (1,1,S,S), (1,H,S,S) and (1,1,1,S), in fp16 or fp32;
  - fills -inf, -30000, -10000 and -100;
  - head dims 32 and 64, 6 and 12 heads, sequence lengths 128 to 512;
  - the macOS15 and iOS26 opsets.

  The pass pipeline doesn't matter. Removing `topological_reorder` changes
  nothing. An empty pipeline or fp32 compute "passes" only because the
  attention then runs on the CPU.
- A mask built entirely by ANE ops still fails when a CPU gather between it
  and the attention splits the ANE work into two procedures. The E5 program
  shows the mask leaving one `AneInference` and entering the next.
- Core ML's E5 program for the failing model is a `BnnsCpuInference` that
  emits the mask, then an `AneInference` that runs the attention. The
  earlier synthetic layer passed because it built its mask by ANE ops in the
  attention's own procedure. That case works natively.

On the full 22-layer model, the compute plan puts all 22 fused attention
ops on the ANE. Their masks come from `tile` (8 global layers) and `select`
(14 sliding-window layers), all on the CPU. The rule predicts every result
measured.

Every other correct fused-attention graph measured is correct through a
fallback. For these graphs Core ML evidently builds no native ANE plan: no
ANE bundle is cached, and the compute plan reports the attention op with no
device. They include:
- bge-small, whose mask is built on the CPU too;
- ModernBERT cut to one layer, with its mask built after the gather;
- ModernBERT with its mask passed through one clamp before each attention
  (full model: parity 0.999793, pad invariance 1.0).

What triggers the fallback isn't pinned down. At the iOS26 opset there is
no fallback, and those same graphs fail to load on CPU_AND_NE ("Failed to
build the model execution plan", error -14). bge is correct at macOS15 by
that fallback alone.

The CPU NaN has its own rule. On the CPU, the fused op returns NaN for a
query row whose keys are all masked when |fill| × √head_dim > 65504,
measured at head dims 32, 64 and 128. A second CPU defect: q/k/v split off a
packed projection by `.transpose(3, 1).unbind(2)` and fed straight to the op
give wrong output (cosine 0.59 against fp32; `.permute` is correct).
ModernBERT's RoPE sits between the split and the attention, so it escapes
that one.

Consequences:
- The MODELS.md checklist states the rule, and `repro_sdpa_mask.py --check`
  applies it to any compiled model by reading its compute plan.
  - It flags all 22 attention ops of a fused ModernBERT artifact.
  - It passes bge's (attention ops with no device).
  - The other validated models have no fused attention op.
- Explicit attention stays the fix. Passing `scale=` to
  `F.scaled_dot_product_attention` is an equivalent one-line alternative:
  coremltools then emits explicit ops. On full ModernBERT it gives ANE
  parity 0.999787, pad invariance 1.0, and CPU output that stays finite.
- Fused-attention models keep the macOS15 target, and pad invariance stays
  the run-time gate.

**Amendment (September 2026, macOS 27.0, M1 Max): the ANE's `linear` op
saturates at 2^15, and ModernBERT's massive activation crossed it.** The
parity suite (D26) graded gte-modernbert's ANE path B: 0.99940 on a
Markdown list, drift 0.010. Its CPU path graded B (0.99926) and its GPU
path A.

It was neither of the D17/D19 limits:
- Keeping GELU in fp32 changes nothing (0.99939).
- An explicit erf GELU doesn't help the ANE and makes the CPU path worse.
- Rescaling small linear inputs changes nothing (0.99940).

Keeping only the MLP output projections in fp32 gives 0.99984, which
located it. Teacher-forced, layer 15's MLP returns inf on the ANE for
delimiter tokens.

**The limit.** On a synthetic 1152→768 projection, the ANE's `linear`
returns an output of 32,000 exactly and 33,000 as inf, whether the output
comes from one term or 1,152. The GPU is exact to 60,000, and the ANE's add,
mul and layer_norm handle 40,000–60,000. The massive activation (dimension
251 on delimiter tokens, ~48,000 in the residual) is written by layer 15's
MLP output projection, at 35,000–51,500 on every input tried. A search of
~80 inputs, including floods of 17 delimiter types and lists of up to 500
items, found nothing higher. Layer 11's reaches ~15,700.

**The artifact before the fix produced -inf internally.** Probed through
the full ANE graph, the residual after layer 15 held -inf at dimension 251
of those tokens in five of six cases (the sixth read -65,504; fp32 has
-38,600 to -50,600). Downstream saturation kept the CLS output finite, at
0.9994–0.9998. Every input with a delimiter ran past the limit and relied
on that undocumented saturation.

**The fix** (tools/convert_gte_modernbert.py, constraint D) runs the
residual stream at 1/K, which is exact in fp32:
- the embedding norm's weight takes 1/K;
- layer 0's Wqkv, which reads the embedding directly, takes K;
- both output projections of every layer take 1/K;
- every LayerNorm, being scale-invariant, only needs eps / K².

K is the smallest power of two that keeps every calibrated linear output
at most 0.85 × 2^15. That is K = 2, leaving 1.31x of headroom over the
largest calibrated output and 1.27x over the largest input found. An input
past it would reproduce the old behaviour, not something worse. The
converter prints the headroom, and fails if no K ≤ 8 fits.

Larger K costs precision on both paths, because it shrinks everything
else: K = 4 gives ANE 0.99984 and CPU 0.99886. Shrinking output-projection
weights hurts the CPU path in particular: layer 15's alone at ÷8 gives CPU
0.9975. Undoing the scale with a multiply after the projection doesn't
work, because the projection itself still overflows (0.9976).

**Results:**
- converter parity CPU_AND_NE 0.999981 at every bucket (was 0.999793) and
  CPU_ONLY 0.99992; the fp32 rewrite gate is 0.9999998;
- parity suite ANE grade **A**: worst 0.999915 (a delimiter flood), stress
  0.999833, mean 0.99998, drift 0.0032 (was 0.010), 0 rank flips, bucket
  invariance 0.999954;
- CPU B 0.99951 (was 0.99926), and GPU A 0.999985, though its
  repeated-subword stress case moved from 0.99998 to 0.99995;
- 794/805 operations on the ANE, and pad invariance 1.0000000;
- the residual after layer 15 now matches fp32 at dimension 251 (-44,896
  against -44,903);
- live `/v1/embeddings` over all 51 suite inputs: worst 0.999833 (the
  repeated-word stress case), all others ≥ 0.99991;
- latency unchanged (±1%); `ane_check` ratios 2.9x/2.0x/1.6x.

The `--attn sdpa` negative control still builds, and still fails pad
invariance (0.74) and parity (0.904). The converter also gained an fp32
exactness gate, NaN-safe metrics, a rejection of the fused attention op
outside the negative control, a check that its 442-token parity text fits
the 512 bucket, and an ANE gate of 0.999.

**Consequences.** A new MODELS.md checklist rule: keep calibrated linear
outputs at or below 0.85 × 2^15 on the ANE. fp16's own 65,504 isn't the
limit that matters there.

**Not done:** tools/probe_activations.py doesn't yet report each linear's
largest output against 32,768, so triage can't catch this class before
converting.

## D26 — A parity suite grades every model on every compute path
Before this, a model's accuracy was checked by its converter, on short
prose, through coremltools rather than sidekick's own code. Two bugs got
through that way:
- ModernBERT's dropped attention mask (D25) hid behind mostly-padding
  parity inputs and a NaN swallowed by `min`.
- F2LLM's truncation bug lived in the server, not in the model.

**What.** `crates/sidekick-embed/examples/parity`, with a shared corpus
(`fixtures/parity/corpus.toml`: 51 inputs, each tagged with the failure it
targets) and references from `tools/parity_reference.py`.
- **Reference.** sentence-transformers in fp32, one input at a time, using
  the model's own prompts. The manifest's prefix is used only where the
  model publishes none (bge-small).
- **Product path.** Every path goes through `CoremlEmbedder`: prefix,
  tokenizer, truncation, bucketing, padding with id 0, pooling,
  normalization. Its `prepare`/`run` split is public but `doc(hidden)`, so
  the suite can check token ids and re-run the same ids in larger buckets.
- **Gates.** The suite fails a model on:
  - a failing compute plan, read on every bucket before anything predicts;
  - non-finite output, including a zero vector;
  - token ids that differ from the reference pipeline's;
  - bucket invariance below 0.999999 on the CPU, or 0.9999 on the GPU or
    the ANE (each bucket is compiled separately there). The ANE gate was
    0.995 until the D17 and D19 precision rewrites: before them, precision
    lost in EmbeddingGemma's and LFM2.5's MLPs varied by bucket (0.9977,
    0.9965);
  - pad invariance below 0.99999, with random pad ids from the vocabulary;
  - output that changes when re-run, or that differs between two ANE
    processes.
- **Floors.** Accuracy is gated only where a floor is recorded for the chip
  (`fixtures/parity/expectations.toml`): 1.5x the measured error, and at
  least 1e-5 below the measurement. Each ANE generation rounds differently,
  so a floor measured on an M1 Max would give contributors on other chips
  false failures.
- **Grades** (A ≥ 0.9999, B ≥ 0.999, C ≥ 0.985, D below, F for a failed
  gate) are documentation vocabulary for MODELS.md and are never gated.
  Every aggregate is NaN-poisoning, and a non-finite value fails instead of
  being folded away.

**Engineering choices.**
- **An example binary, not an ignored test.** Each compute plan is read in
  its own child process. One Core ML can't produce is re-read from an APFS
  clone of the artifact (D24). If that fails too, the model fails, unless
  `--allow-unverified-plans` is given. In that case the model still fails
  if that bucket's ANE output is bit-identical to CPU_ONLY, which is a CPU
  fallback. Each (model, path) runs in its own worker process, with a
  timeout. An Objective-C exception in Core ML
  (on macOS 27 a flexible-shape artifact aborts at predict under
  `.cpuOnly`) or a stuck ANE compile then costs one cell of the report. The pure logic is unit-tested
  by `cargo test` on every platform (`[[example]] test = true`).
- **ONNX is report-only.** Published exports run at reference time. They
  measure the ecosystem (gte-modernbert's int8 export: 0.892), and one
  disagreement with sentence-transformers turned out to be a convention
  difference, not an error (EmbeddingGemma past about 257 tokens). Neither
  says anything about sidekick's correctness.
- **No automatic attribution.** Rules such as "CPU fails → conversion"
  would have misread the fused-attention ModernBERT artifact, as D25's first
  diagnosis did. The report prints each low case's CPU, GPU and ANE results
  side by side instead, and MODELS.md explains how to read them.
- **References aren't committed.** They derive from weights users convert
  themselves. They record the corpus hash (comments excluded), the
  tokenizer.json hash, and the checkpoint's Hugging Face id and revision,
  never a local path. The suite refuses a stale reference.

**Acceptance.** An adversarial review required the suite to fail known-bad
artifacts on the gates that target them, and it does:
- a flexible-shape bge: compute plan, before any prediction;
- fused-attention gte-modernbert: CPU NaN, ANE pad invariance 0.61;
- LFM2.5 without pad zeroing: pad invariance 0.33–0.39 on every path;
- naive truncation: token ids on both over-length inputs.

The converters gained `--attn sdpa` and `--no-pad-zeroing` to rebuild the
last two on demand.

**Findings (M1 Max, macOS 27.0).**
- On the ANE, bge-small and EmbeddingGemma graded A, and gte-modernbert and
  F2LLM B. Both B models now grade A:
  - F2LLM 0.99997, after the D20 amendment's rewrite. Of the D17/D19
    causes, only the coarse SiLU mattered for it, plus small attention
    inputs on long texts.
  - gte-modernbert 0.99992, after the D25 amendment's range rewrite. Its
    loss was a third ANE limit: the `linear` op saturates at 2^15, which
    ModernBERT's massive activation crossed.
- LFM2.5 graded **D**: 0.954 on a URL, pairwise similarity drifting by up
  to 0.187, and 1,399 rank flips at a 0.02 margin. Its CPU and GPU paths
  were 0.9999, so the graph was faithful and the loss happened on the ANE.
- EmbeddingGemma graded D too (0.975 on a run of digits, drift 0.042, 122
  flips). Both had the same two limits of ANE arithmetic: the `linear` op
  loses precision on small inputs, and the native GELU/SiLU are coarse.
  Precision rewrites fixed both (the D17 and D19 amendments), so an
  ANE-only loss can be a conversion problem to fix, not just something to
  measure. Both now grade A:
  - EmbeddingGemma 0.99999, drift 0.001, no flips; ANE bucket invariance
    0.9977 → 0.99998.
  - LFM2.5 0.99999, drift 0.003, no flips; ANE bucket invariance
    0.9965 → 0.999995.
- With both rewrites, every model's ANE bucket invariance is at least
  0.99998 (bge-small 0.999992, EmbeddingGemma 0.999983, F2LLM 0.999982,
  LFM2.5 0.999995, gte-modernbert 0.999984), so the ANE gate was raised
  from 0.995 to 0.9999, the GPU's value. gte-modernbert measures 0.999954
  after its later range rewrite.
- The GPU path measures at fp32-like accuracy on every model (A). "GPU fine,
  ANE low" therefore isolates the ANE, not fp16 arithmetic in general.

**The 0.985 gate.** It stays the converters' acceptance gate on their own
parity sets; the EmbeddingGemma, LFM2.5, F2LLM and gte-modernbert converters
now gate at 0.999.
A D on the adversarial corpus doesn't remove a model: the grade is
published, and this chip's floor makes it a regression test.

**Not done.**
- Per-token grading for models whose token vectors are the product (laya,
  ColBERT). Every registry model pools inside its graph, so the suite can't
  see per-token vectors.
- Parity through the HTTP layer.
- A corpus case built to maximize ModernBERT's massive activation, so the
  suite exercises the ANE linear's 2^15 limit (D25 amendment). Adding it
  means regenerating every reference.

**Amendment (October 2026): bucket invariance is sampled for buckets over
512 tokens.** Re-running every case in every larger bucket doesn't scale to
2,048-token buckets. For Lumma-fev (2,630 cases, ~1.24 s per ANE prediction
at 2,048) it alone took 2–3 hours, past the suite's time limit. With
`--bucket-invariance auto` (the default), a model whose largest bucket is
over 512 re-runs every case in its next larger bucket, and the adversarial
cases plus every 10th case in every larger bucket; `full` restores the old
behavior. That keeps the gate's two purposes:
- A bucket-specific defect shows on any input in that bucket, and every
  bucket still sees every case just below it plus the sample.
- A mask or position defect shows in any larger bucket, and every case is
  still checked one bucket up.
What it gives up is narrow: a defect that appears only two or more buckets
up, only for inputs outside the sample and the adversarial cases, and never
in the next bucket. Every model graded before this keeps full coverage and
an unchanged report, and the report prints the coverage it used.

## D27 — Refuse multi-shape Core ML models at load on macOS 27
D24 said that on macOS 27 a flexible-shape artifact aborts the process at
its first prediction "whatever the compute units". Measured again on an M1 Max
(macOS 27.0, September 2026), that holds only for `.cpuOnly`. The test
artifacts were bge-small negative controls: `convert_bge_small.py
--enumerated-shapes`, built with torch 2.7 and with 2.13 (coremltools 9.0),
plus a `ct.RangeDim(1, 512)` variant. Each was predicted from Rust
(`CoremlModel`) and from Swift, in a fresh process per run:
- **Several enumerated shapes** (`ct.EnumeratedShapes`, 128/256/512).
  `.cpuOnly` aborted every time with an uncatchable `NSGenericException`
  ("Failed to add operation to E5 stream. E5RT: No memory object bound to
  port."). `.cpuAndNeuralEngine` and `.all` never aborted, including all
  three shapes predicted in one process. They ran on the CPU, the fallback
  D15 measured: 85 ms per call at seq 128 (a static bucket takes 2.2 ms on
  the ANE), and up to 1 s at 512.
- **A range** (`ct.RangeDim`) never aborted under any compute units. Its
  compute plan puts 0 of 375 operations on the ANE.

The `ane_check` that first hit the abort predicted under `.cpuOnly` before
`.cpuAndNeuralEngine`, which fits these results. `sidekickd` and
`libsidekick.dylib` load with `.cpuAndNeuralEngine` (D14), so on this
machine such an artifact runs slowly there rather than aborting. Any caller
choosing `.cpuOnly` can still hit the abort. Whether the ANE preference's
CPU fallback always avoids it is Core ML's internal behavior, observed on one
machine.

**Decision.** `CoremlModel::load` reads each input's shape constraint from
the model description once the model has loaded:
- On macOS 27 and later (`available!(macos = 27.0)`), an input with more
  than one enumerated shape is refused with `Error::Inference`, whatever the
  compute units. The error names the input and points to the per-bucket
  recipe. An abort that no one can catch, inside a host app, outweighs
  refusing an artifact D15 already rules out and that gets nothing from the
  ANE.
- A range input that admits more than one size is only warned about
  (`tracing::warn!`): it never aborted, and it runs on the CPU.
- Before macOS 27, every flexible layout only warns. Those models still run,
  so refusing them would be a regression.

The discriminator is the number of shapes, not the constraint type. All
twelve installed per-bucket artifacts (four models, three buckets each)
report `.enumerated` with exactly one shape, and the control reports three.
Range constraints read as `(location, length)` per dimension, and a fixed
dimension of 128 reads `(128, 1)`.

The check costs microseconds; the compute plan stays in `ane_check`.
`available!` reads `kern.osproductversion`, and that reported 27.0 even in
binaries whose SDK version was rewritten (with `vtool`) to 10.15, 11, 15 or
26. So the check holds inside a host app built against an older SDK.

The rule is `sidekick_coreml::shape_verdict`: platform-neutral and
unit-tested on every CI platform. `sidekick_coreml::input_shapes` reads a
model's constraints with a CPU-only load that never predicts, for tools that
want the verdict without loading the model for inference.

Verified on the M1 Max, macOS 27.0:
- The enumerated control is refused under `.cpuOnly`, `.cpuAndNeuralEngine`
  and `.all`, with no abort.
- The range control loads with a warning and predicts.
- The twelve installed buckets load with no warning and predict.

## D28 — Classification: `POST /v1/classify`, vLLM's contract exactly
sidekick served embeddings and chat. Classification — sentiment, intent,
routing, "which of these options fits" — is the next most common local
inference job, and the encoders that do it are the same size and shape as
the embedders D26 already grades A on the ANE. The design is in
`docs/design/classify.md`. This entry records the decisions and why.

**The contract is vLLM's.** OpenAI has no classification endpoint. vLLM's
`/classify` and SGLang's `/v1/classify` share one request and response
shape, with the same conventions as `/v1/embeddings`, and clients already
exist for it. sidekick follows it field for field: `{model, input}` in,
`{id, object: "list", created, model, data[{index, label, probs,
num_classes}], usage}` out. Hugging Face's pipeline shape was the
alternative. It fits a Python library call, not an OpenAI-style server, and
nothing that speaks OpenAI-family APIs sends it.
- `probs` uses transformers' text-classification activation: none for
  regression, sigmoid for multi-label or a single output, otherwise
  softmax. `use_activation: false` returns the raw values.
- vLLM's other fields are honored or rejected, never dropped (D22 and its
  amendment), each checked against vLLM's own handling: `request_id` sets
  the response id, `priority`, `padding`, `cache_salt` and
  `mm_processor_kwargs` accept their no-op forms, and `normalize` is a 400.
  `truncate_prompt_tokens` truncates, with −1 meaning the model's maximum
  as in vLLM. Without it, an over-length input is a 400.
- Deliberate deviations, listed in the design doc: `model` is required,
  since sidekick serves several models where a vLLM process serves one,
  and truncation keeps `[CLS]` and `[SEP]` where vLLM's slice drops one.
- A request is validated in full before any input runs, so a bad input in
  a batch costs no inference.
- Non-finite model output is a 500, never a response with nulls in it.

**Extensions, only where no standard exists.** Zero-shot classification
takes its labels per request as `candidate_labels`, the name Hugging Face's
zero-shot task uses. `calibration: "model"` applies the manifest's declared
temperature; the default is `none`, so `probs` is the model's own output
unless a client asks. `question_type` and `instructions` serve laya's
format (below). Each extension is a 400 on a model that doesn't take it.

**Manifests: a separate file.** Classifiers use `classifier.toml`, not
`manifest.toml`. Daemons and `libsidekick.dylib` builds from before this
release never read the new file, so installing a classifier can't break
them. From this release, the registry skips and warns on a manifest that
doesn't parse or validate, instead of failing the whole scan, and so does
`sk_pool_open` in the C ABI, which used to fail. `/health` and the new
`sk_pool_skipped` list the skipped manifests with the reason, by paths
relative to the models directory, since `/health` needs no API key.
Loading a classifier checks every bucket's artifact interface before any
runs, so a bad bucket fails the load rather than the first long request. A classifier whose id an
embedder already uses is skipped, so adding one never breaks a working
embedder. `max_batch` defaults to 32. Calibration is rejected on
fixed-label models, which have no question type to key it on.

**Routes know their models' tasks.** `/v1/models` reports each model's
`task`, using Hugging Face's pipeline names (`feature-extraction`,
`text-classification`, `zero-shot-classification`, `text-generation`).
Classifiers also report their labels or `max_labels`, `max_batch`, the
extensions they take, and their calibration table. A model sent to the
wrong route is a 400 naming its task and the route that serves it. Before,
an embedder's id on the chat route was a 404. This is a behavior change.
The C ABI (`sk_pool_models`, `sk_model_info`) still lists embedders only.
The embedder pool becomes a generic `ModelPool`, with the same lazy
loading and idle eviction for both kinds.

**Provenance headers** on every inference route, in the style of OpenAI's
`openai-model` and `openai-version`, and on successful responses only:
- `sidekick-version`;
- `sidekick-model`: `<id>@<revision>` when the manifest names a source
  revision, otherwise `<id>`; for chat, the Foundation Models variant,
  read in the background so no request waits on it;
- `sidekick-compute-units`: `cpu_and_ne` for Core ML models, `cpu` for
  static ones; chat omits it. This is the configuration the model was
  loaded with. Core ML doesn't report which device ran a prediction, and
  reading the compute plan at load (D24) takes too long.

**laya's format.** [laya](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0) is a decision model: a ModernBERT-large encoder plus a head
that scores a `[MASK]` marker placed before each option. It answers three
question types: `choice`, `score` (ordered levels) and `noul`
(false/true). Its accuracy depends on its exact training layout, so
`crates/sidekick-embed/src/laya.rs` ports laya's own `build_sequence`
(`THIRD_PARTY_NOTICES.md`) rather than re-deriving it. The special tokens
come from the tokenizer, so English and multilingual checkpoints both
resolve. Token-id fixtures generated by laya's Python pin the port. It
matches all 15 laya cases, which take every branch of `build_sequence`,
and all 18 cases for the BERT classifier.

The Core ML interface is int32 only, which the runtime already feeds:
`marker_pos [1,32]` (−1 pads unused slots) and `qtype [1]`. The graph
builds the one-hot selections itself. The ANE `linear` saturation rewrite
(D25 amendment) is pinned at K = 2. laya's layer-19 MLP output projection
writes up to about 27,500 on the converter's calibration inputs. K = 1 is
the smallest that calibration allows, but it would put that output within
2% of the rule's 0.85 × 2^15 target and 19% under the limit itself; an
input a little further out than the calibration set would saturate. K = 2
halves it. laya's act (escalate) head isn't served.

**Validation.** The parity suite (D26) grades classifiers in probability
space:
- **Gates:** finite output; ids, markers and qtype equal to the
  reference's; bucket invariance (max |Δp| 1e-5 on the CPU, 1e-3 on the
  GPU and ANE); pad invariance (1e-4); determinism across ANE processes.
- **Graded:** raw Δp and Δlogit against the model's own fp32 forward. An
  argmax flip counts only where the reference's top-2 logit margin is at
  least 0.05; closer ones are near-ties, reported but not graded. A for
  max Δp ≤ 1e-3, B ≤ 5e-3, C ≤ 2e-2, D above; a graded flip caps the grade
  at C.
- **Reported only:** calibrated Δp, and accuracy against gold labels. Gold
  accuracy measures the corpus translation as much as the model.

laya's corpus is [fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions)
(Apache-2.0), translated mechanically into laya's three question types by
a committed table, plus adversarial cases. Fixed-label models use D26's
51-input corpus.

**Measured** (M1 Max, macOS 27.0, `tools/measure_classifier.py`, against
fp32):

| model | path | argmax agreement | graded flips | raw Δp max / p99 | ms @128/256/512 |
|---|---|---|---|---|---|
| laya-en (2,612 cases) | ANE | 99.81% | 5 (margins 0.055–0.12) | 0.077 / 0.030 | 20 / 38 / 107 |
| | GPU | 100% | 0 | 0.030 / 0.008 | 20 / 32 / 58 |
| | CPU | 99.77% | 6 (0.06–0.23) | 0.211 / 0.048 | 47 / 84 / 163 |
| nlptown-sentiment (51) | ANE | 100% | 0 | 0.0026 | 4.3 / 10.4 / 27.3 |
| | GPU | 100% | 0 | 0.0007 | |
| | CPU | 100% | 0 | 0.0034 | 19 / 32 / 62 |

Pad invariance is exact on every path. laya's bucket invariance on the
ANE is 0.045 in logits; on the CPU and GPU it is 0.

The parity suite, through sidekick's own product path, measures the same
accuracy and grades nlptown-sentiment B on the ANE (A on the GPU). It
grades laya-en F on the GPU and the ANE: an input's probabilities move by
up to 0.018 (GPU) or 0.038 (ANE) in the next larger bucket, against the
1e-3 bucket-invariance gate. The CPU path is exact across buckets, pads
change nothing on any path, and two ANE processes agree exactly, so the
graph is invariant and the failure is fp16 rounding that differs per
compiled shape. Localized, the loss is all in laya's ModernBERT-large
encoder, not its head.

**laya-en ships as a preview.** The gate is not loosened for it; the
suite keeps reporting the failure, and docs/MODELS.md states it. As a
decision model it is usable now: one case in 500 flips a decision, and
only decisions whose fp32 margin is under 0.12 logits. Clients deciding
on close calls should read `probs`, not just `label`. The GPU is laya's
most accurate path, and Core ML's fp16 CPU backend its least. On one
512-token `noul` item the CPU backend moves both logits by about 0.46 and
flips a 0.74 margin, where the ANE moves them by 0.04. The conversion is
exact in fp32 (max |Δlogit| 1.2e-4 against laya's own forward), so the
loss is the CPU backend's fp16 arithmetic. laya's converter therefore
gates accuracy on `CPU_AND_NE`, the served path, and only reports
`CPU_ONLY`. gte-modernbert showed the same CPU-path sensitivity (D25
amendment).

**Not in this release:** choosing compute units per request; zero-shot
classification by NLI; multi-label zero-shot; text pairs; laya's act head;
a Hugging Face-shaped route.

**Amendment (September 2026, macOS 27.0, M1 Max): laya's ANE loss was
Core ML's native gelu, and fp16 itself bounds its grade.**

Localized by stage, then by layer:
- The encoder alone on the ANE, with laya's head and scorer in fp32,
  accounts for the whole loss (304-case sample: |Δp| max 0.077, 5 flips).
  The head and scorer alone, fed the fp32 encoder's output, lose 0.0014.
- Teacher-forced layer by layer (one layer on the ANE, fed the exact fp32
  input, everything else in fp32), the damage comes from the first ~8 of
  the 28 layers, whose local error was ~5× the GPU's. Split further, their
  MLP branches were 9–15× less accurate than on the GPU.
- The cause: Core ML's `gelu` op on the ANE is off by up to 6e-3 on
  [-1, 1] (the GPU: 3e-4), where most of the MLP inputs lie. Rescaling small
  linear inputs (D17, D19 amendments), LayerNorm and the softmax form made
  no measurable difference.

The fix is constraint E of `tools/convert_laya.py`. Every exact GELU (the
encoder's MLPs and the scorer) is written as x·(1 + erf(x/√2)), twice the
GELU, with the 0.5 folded into the gate rows of each MLP input projection
and into the scorer's output linear. It is exact in fp32 and 9× closer to
GELU than the native op on [-1, 1]. Written with the 0.5, coremltools
fuses the pattern back into the native op, so the converter fails if a
`gelu` op survives. EmbeddingGemma's tanh-built GELU (D17) remains the
right form for tanh-approximate activations. For an erf GELU model the tanh
form changes the function: on laya it moves Δp by up to 0.008 in fp32
alone.

Measured on the 2,612 cases, before → after:

| path | graded flips | raw Δp max / p99 / mean | bucket invariance (max Δp) | ms @128/256/512 |
|---|---|---|---|---|
| ANE | 5 → 1 (margin 0.057) | 0.077 / 0.030 / 0.0036 → 0.039 / 0.016 / 0.0019 | 0.038 → 0.027 | 19.6 / 37.9 / 106 → 20.8 / 40.0 / 107 |
| GPU | 0 → 0 | 0.030 / 0.008 / 0.0009 → 0.025 / 0.008 / 0.0010 | 0.018 → 0.011 | unchanged (within 2%) |
| CPU | 6 → 16 | 0.211 / 0.048 / 0.0055 → 0.172 / 0.053 / 0.0057 | 0 → 0 | |

Timings were measured interleaved, before and after on the same inputs in
the same session (load average 3–5). The CPU gains flips because Core ML's CPU erf
is a little coarser than its native gelu, and the CPU stays laya's least
accurate path. The suite's verdicts don't change: ANE and GPU fail bucket
invariance, and the CPU grades D.

**The floor.** An ideal fp16 engine was simulated in PyTorch on the same
corpus: fp32 arithmetic, with every stored tensor rounded to fp16. It
reaches raw Δp max 0.026, p99 0.0074, mean 0.00083, with no flips.
Rounding only the embedding output, one fp16 rounding at 3e-4 relative,
moves Δp by up to 0.0058; rounding only the final logits moves it by up to
0.0006. For Δp ≤ 1e-3, laya's logits (median largest |logit| 4) must be
accurate to about 0.004, roughly one fp16 ulp. Its early layers amplify
ulp-level perturbations into [MASK]-token errors of up to 2%, even on the
GPU. No fp16 path can therefore grade laya above D by max Δp. The ANE now
sits 1.5× above that floor at the max and 2× at p99. The same ANE encoder
error, read as an embedding, has a median cosine of 0.99999 at the marker
tokens, which D26 would grade A.

**Bucket invariance, bisected.** On the ANE, `linear`, `layer_norm`,
`matmul`, `exp` and `reduce_max` give bit-identical results for the same
real tokens at any sequence length. `reduce_sum` does not: 63% of its
outputs are identical, the rest one ulp off. Core ML's softmax is built on
it. Write the softmax as exp(w − rowmax) followed by one matmul against
[V | 1], so the numerator and denominator come from the same matmul, and
laya on the ANE becomes exactly bucket-invariant. Measured over all 2,359
case/bucket pairs, the max Δp is 0, and outputs also match an unpadded run
exactly. It wasn't adopted at first (it is now: see the amendment below):
- On the ANE it keeps the corpus p99 and mean but costs some inputs. A
  36-token `noul` case goes from Δp 0.001 to 0.032, and the corpus max from
  0.039 to 0.043.
- The cost isn't the softmax's arithmetic. Layer 7's MLP, where the massive
  activation forms, is 10× less accurate inside that graph than the same
  MLP compiled alone and fed the same inputs.
- The regression isn't tied to a bucket: its worst case sits in bucket
  128. Choosing the form per bucket wouldn't avoid it, and mixing forms
  would break invariance at the boundary between them.
- On the GPU it costs 34% latency at 512 tokens.

It exposed two CPU traps:
- On the CPU, Core ML's `reduce_max` (and `reduce_min`) over 256 or more
  elements returns max(x, 0), which is wrong for rows whose max is
  negative. `tools/repro_cpu_reduce_max.py` reproduces it, and `--check`
  lists a model's affected reductions. Taking the max in blocks of 128 is
  exact on every compute unit.
- In a sliding-window layer, a pad query whose whole window is padding
  has every key masked. With the softmax written out, the CPU turns that
  row into NaN, which spreads through the next layer. Letting every query
  attend to itself, by clearing the mask's diagonal, prevents it and is
  exact for real tokens.

**Amendment (September 2026): classifiers are also graded against their
ideal-fp16 ceiling.** The absolute scale (A for a worst-case |Δp| ≤ 1e-3)
can't tell a perfect fp16 conversion of laya from a poor one. An ideal fp16
engine moves laya's probabilities by up to ~0.017, so no fp16 path could
grade laya above D however good its conversion. The 1e-3 bucket gate is
unreachable for the same reason: laya's GPU path varies by 0.011 between
buckets. Grading still has to work for models like nlptown, whose ideal
fp16 error is tiny (5.4e-4) and whose absolute B is honest.

- **The ceiling.** Each classifier reference may carry a second oracle,
  `fp16`, beside `torch`: the model as published, run through
  `sidekick_convert.fp16sim`. That simulates an ideal fp16 engine with
  fp32 arithmetic inside each op and every input-dependent op output
  stored in fp16. Constants (weights, buffers, RoPE tables) are stored in
  fp16 once, and attention runs in the explicit form converted programs
  use (docs/CONVERTING.md defines it). The ceiling is that oracle's |Δp|
  against fp32 over the graded cases, and every model's is computed by the
  same function.
- **The ratio grade** is the path's p99 |Δp| divided by the ceiling's p99:
  A ≤ 1.25×, B ≤ 2×, C ≤ 4×, D beyond. It is anchored on p99, not the
  worst case. Two valid implementations of the same simulation agreed on
  laya's mean and p99 within 15% but differed 1.7× on the worst case,
  which depends on exactly which tensors are rounded. The worst-case ratio
  is reported, not graded. With fewer than 100 cases the p99 is the
  maximum, and the report says so.
- **The grade that counts** is the better of the absolute grade and the
  ratio grade. A model is credited either for being practically exact or
  for being as good as fp16 allows, and a low ceiling never demotes an
  honest absolute grade (nlptown's B). A graded decision flip still caps
  the grade at C, and a failed gate is still F.
- **Bucket invariance on fp16 paths** (GPU and ANE) passes at |Δp| ≤
  max(1e-3, ceiling worst case). Variation between buckets that's within
  what fp16 storage alone produces isn't a defect. The CPU path's exact
  invariance gate is unchanged.
- **Multi-label cases** (gliner2's `multi_label`) are graded on per-label
  sigmoids. **Rerankers** use the same rules in sigmoid space (D29).
- A reference without the `fp16` oracle grades exactly as before, so
  models move to this scale as their references are regenerated.

**Amendment (September 2026): laya adopts the matmul softmax.** Under the
ceiling grading above, a few inputs costing a little more accuracy is a
better trade than failing the bucket gate, so tools/convert_laya.py now
builds every softmax, in the encoder and in the head, from exp and one
matmul against [V | 1] (its constraint F), with the row max in 128-wide
blocks and self-attending pad queries. Graded by the parity suite on all
2,612 cases, against laya's ceiling (|Δp| max 0.0169, p99 0.0081):

| path | grade | p99 ratio | Δp max / p99 / mean | flips | buckets (max Δp) |
|---|---|---|---|---|---|
| ANE | C | 1.93× (B) | 0.043 / 0.016 / 0.0019 | 1 | 0 (was 0.027) |
| GPU | A | 0.93× | 0.037 / 0.0076 / 0.00095 | 0 | 0.0072 |
| CPU | D | 6.1× | 0.162 / 0.049 / 0.0058 | 13 | 0 |

The ANE's one graded flip (margin 0.055, one of the five v0.3.0 flipped) caps
it at C; its p99 ratio alone would be a B. The bucket gate passes on both
fp16 paths: exactly on the ANE, and within the ceiling's 0.017 on the GPU.
Against the erf-only conversion, the ANE's worst case moves from 0.039 to
0.043, with p99 and mean unchanged, and latency is unchanged (19.8 / 39.9 /
106.6 ms at 128 / 256 / 512). One long noul gate item reaches |Δp| 0.065,
so the converter's sanity gate on the ANE moves from 0.05 to 0.08, with
flips still gated at zero. Accuracy is graded by the suite, not by that
gate.

laya-typed-decisions, laya's format at 1,024 tokens, is converted the same
way (`convert_laya.py --model laya-typed-decisions`, buckets 128 to 1,024).
On its 2,641 cases, which add 29 inputs of 587 to 1,024 tokens to laya-en's
corpus, the ceiling is |Δp| max 0.0252, p99 0.0038. The ANE grades C (p99
2.14×, worst case 0.025, no flips, bucket invariance exact), the GPU A
(1.02×), the CPU D (1 flip). The inputs above 512 tokens stay within
0.0084 on the ANE, so the 1,024 bucket needs nothing beyond constraints E
and F.

**Amendment (October 2026): when a classifier is a preview.** laya-en
shipped as a preview because it failed a hard gate. The rule from now on:
a classifier is supported when it passes every hard gate on every path,
its conversion is exact in fp32 with no known unfixed defect, and it
grades A on at least one path; otherwise it is a preview. When the A path
isn't the default served path, docs/MODELS.md says so, and a manifest's
`compute_units` (D31) can serve it there. Under it, laya-en and
laya-typed-decisions (A on the GPU, served on the ANE) and
GLiNER2.5-Decide (A on the GPU, served there) are supported. Julia-1,
whose fp16 paths are both capped at C by decision flips, stays a preview.

## D29 — Reranking: vLLM's and Cohere's contracts, a reranker is a classifier
Reranking (scoring documents against a query) is how retrieval pipelines
use cross-encoders, and it has the strongest API convention of anything
sidekick didn't serve. The design is `docs/design/rerank.md`. This entry
records the decisions and why.

**The contracts.** As with classification (D28), sidekick follows the
standard field for field and extends only where none exists:
- `POST /v1/rerank` and `POST /rerank` are vLLM's `RerankRequest` and
  `RerankResponse`, the Jina shape that vLLM, llama.cpp, LocalAI and
  Infinity serve. Results are sorted by score, highest first, with ties in
  request order. The response id has vLLM's `score-` prefix.
- `POST /v2/rerank` is Cohere's v2 shape. Its response is a superset that
  both clients parse: Cohere's `id`, `results[{index, relevance_score}]`
  and `meta`, plus the `model`, `usage` and per-result `document` that
  vLLM's response model requires. That was checked against both: vLLM's
  `RerankResult.document` is required, and Cohere's SDK tolerates extra
  fields.
- `POST /v2/embed` is Cohere's v2 embed shape over the existing embedders,
  as vLLM serves it: `input_type` maps to the manifest's prefixes, and
  `embedding_types` covers `float`, `base64`, `binary` and `ubinary` (int8
  and uint8 are 400s, as in vLLM). Cohere requires `input_type`. Without
  it sidekick uses the document prefix, as `/v1/embeddings` does, so the
  two embed routes agree for a given model.
- The one extension is `return_documents` on `/v1/rerank`, from Jina and
  Cohere v1. `/v2/rerank` ignores it, because vLLM requires `document`
  there.
- One shape per route. TEI's `/rerank` and SGLang's `/v1/rerank` return a
  bare list. Serving them too would mean sniffing request bodies, so
  sidekick serves vLLM's shape only.

Every field these standards define is honored, accepted in the form that
changes nothing, or rejected with a 400 (D22). The new routes sit behind
the API key with the rest of the API.

**Truncation follows vLLM, with three deliberate deviations.**
`max_tokens_per_query` and `max_tokens_per_doc` cut each text first, as
vLLM does. On `/v1/rerank`, a pair still too long is a 400 unless
`truncate_prompt_tokens` is set; it is then truncated `longest_first`, or
in the document only when `max_tokens_per_doc` is also set. On
`/v2/rerank`, documents are truncated and the query kept whole, which is
Cohere's contract. Each deviation keeps the model from seeing a sequence
unlike its training data:
- Special tokens are always kept. vLLM's sliced truncation can drop
  `[CLS]` or the final `[SEP]`. This is classify's rule.
- An empty document is paired, as `CrossEncoder` pairs it (`[CLS] q [SEP]
  [SEP]`). A single-pair tokenizer call would read it as no pair.
- `/v2/embed`'s `START` truncation keeps the model's prompt prefix (e5's
  `query: `) whole, where vLLM's left slice would cut it off.

**A reranker is a classifier.** Its manifest is a `classifier.toml` with
`task = "text-ranking"` (Hugging Face's pipeline name) and one output.
`[classify.io]` gains an optional int32 `token_type_ids`, for BERT's
segment ids; XLM-R rerankers have none. The activation follows D28's
`problem_type` rule, and the converter derives `problem_type` exactly as
vLLM's `get_act_fn` does, so `relevance_score` is on the scale vLLM
reports for the same model. Every pair runs as its own prediction, in the
smallest bucket that fits, because each bucket is a static `[1, S]`
artifact (D15). Each manifest sets `max_batch` so that a full request fits
within the request timeout.

**Compatibility.** A v0.3 daemon or library given a `text-ranking`
manifest fails to parse its task and skips it with a warning (D28), so
nothing breaks. But v0.3 ignores an unknown `[classify.io]` key, so a
text-classification manifest naming `token_type_ids` would load there and
fail every prediction. Two rules close that:
- the converters write `token_type_ids` only for text-ranking models;
- from this release, loading a classifier refuses an artifact that has an
  input its manifest doesn't name. The check runs at classifier load, sees
  multi-array inputs, and also refuses inputs marked optional. Embedders
  aren't checked.
Every installed artifact was verified to name all of its inputs.

**Validation.** The parity suite grades rerankers with the classifier
grader at k = 1, on a committed corpus of 51 pairs in 13 query groups
(`fixtures/rerank/corpus.toml`), including adversarial pairs: an empty
document, an over-length pair, literal special tokens, multilingual text
and code. References come from `tools/rerank_reference.py`: `CrossEncoder`
in fp32, with pairs encoded from the installed `tokenizer.json`.
- **Graded in sigmoid space**, the score as a probability, whatever the
  manifest serves. Cross-encoders are trained with a sigmoid (BCE)
  objective. ms-marco serves raw logits of magnitude ~10, and against
  thresholds meant for probabilities even an exact fp16 conversion would
  grade D. Sigmoid compresses errors at saturated logits, so the largest
  raw |Δlogit| is published beside each grade.
- **Rank flips** count within each query's documents, on raw logits,
  where the reference's gap is at least the flip margin. A gap that
  sidekick collapses to an exact tie counts as a flip.

**Measured** (M1 Max, macOS 27.0), converted with the conversion library
(`tools/sidekick_convert`, `docs/CONVERTING.md`):
- `cross-encoder/ms-marco-MiniLM-L6-v2` grades B on every path, with no
  rank flips. On the ANE its worst |Δp| is 3.4e-3 (|Δlogit| 0.024). It is
  converted with GELU built from erf (D28 amendment), which halves its
  mean ANE error and leaves a single pair at B level.
- The same release adds two embedders on the library's BERT recipe, both
  grade A on every path: `sentence-transformers/all-MiniLM-L6-v2` (ANE
  worst cosine 0.99992) and `intfloat/e5-small-v2` (0.99995).
  all-MiniLM's largest bucket is 256, the sentence-transformers
  `max_seq_length` it was trained with. Run at 512, its output moves to
  cosine 0.973 on a ~450-token text.

**Not in this release:** vLLM's `/score` and `/v1/score` (`/v1/score`
collides with SGLang's unrelated route); late interaction (ColBERT,
MaxSim); LLM-based rerankers, which need decoder support and chat
templates.

## D30 — Two more zero-shot formats: Julia-1's rendering and gliner2
D28 served zero-shot classification in one format, laya's. Two more decision
models are worth serving, and both fit `/v1/classify`'s zero-shot contract
(`candidate_labels`, one label per input) without changing it. The contract
is in `docs/design/classify.md`.

**Julia-1 is laya's format with another rendering.** Julia-1
(SupersonicLabs/Julia-1, an mmBERT-small encoder with laya's decision head)
builds laya's exact sequence but renders option text differently. Rather
than a new format, `[classify.laya] option_rendering = "julia"` picks its
rendering, so one input builder serves both. Its token fixture comes from
Julia-1's own `julia/data.py`. Julia-1 takes 2 to 20 labels and has no
default question, so a request without `instructions` is a 400.

**gliner2 is a new format.** GLiNER2 models (GLiNER2.5-Decide, a
DeBERTa-v3-large span extractor) classify from a schema written before the
text, with an `[L]` marker per label. sidekick ports gliner2 2.0.0's input
builder to Rust: word-level text splitting and lowercasing, per-item
tokenization, word-level truncation that keeps the schema whole, and
marker positions taken from the layout. The port matches gliner2's own
processor on its fixture and on about 8,000 random requests, and its word
splitter matches Python's `re` on all of Unicode 15.0. The graph emits one
logit per token, read at the markers, so it needs no index inputs and no
gather.

**`multi_label`**, the name Hugging Face's zero-shot pipeline uses, asks a
gliner2 model for independent per-label sigmoids instead of a softmax.
`label` stays the argmax, because vLLM's response has one label per
input; clients threshold `probs`. `multi_label: false` is accepted on
every model as the default form (D22), and `true` is a 400 on formats
other than gliner2. One task per request: scoring a task alone or jointly
with others agreed on 83 of 85 fast-decisions tasks.

**Discoverable requirements.** `/v1/models` lists each classifier's
`required` request fields: `candidate_labels` on zero-shot models,
`question_type` on laya-format models, and `instructions` where the
manifest has no default. A client learns the fail-loud contract from the
listing, not from a 400.

**Compatibility.** A v0.4 daemon skips a gliner2 manifest (unknown format).
It also skips the committed Julia-1 manifest, which has no
`default_instructions`. A Julia manifest that does set them would load on
v0.4 with laya's rendering, so they should stay unset while v0.4 daemons
may read the directory.

Both models are graded for parity, never accuracy: fast-decisions is
fastino's own benchmark, and Julia-1's training data is private. Grades,
against each model's ideal-fp16 ceiling (D28 amendment), are in
docs/MODELS.md: GLiNER2.5-Decide is A on the GPU, where it's served
(D31), and Julia-1 is C on its fp16 paths, a preview.

## D31 — A manifest may name its compute units
D14 loads every Core ML model with `.cpuAndNeuralEngine`. GLiNER2.5-Decide
is the first model the ANE runs badly: its DeBERTa relative-position
rewrite compiles onto the ANE (99.1% of operations) but runs 10–40× slower
there than on the GPU (0.11, 0.34 and 1.28 s per prediction at 128, 256
and 512 tokens, against ~34 ms on the GPU), and grades lower (C against
the GPU's A). With several of its buckets loaded, the ANE also failed
predictions after tens to hundreds of cases, for reasons not yet
diagnosed. On the GPU it does neither.

**Decision.** An optional top-level `compute_units` in `classifier.toml`
and in an embedder's `manifest.toml`: `cpu_and_ne` (the default, D14
unchanged), `cpu_and_gpu`, `cpu_only` or `all`. The daemon and
`libsidekick.dylib` load every bucket with it. `sidekick-compute-units` and
`/v1/models` report it. An unknown value skips the manifest with a reason
(D28), and a static embedder rejects the key. D27's shape guard reads the
model description, which doesn't depend on compute units, so it refuses a
multi-shape artifact on macOS 27 whatever the manifest asks for.
Converters gate the path the manifest serves and report the others.
GLiNER2.5-Decide's manifest sets `cpu_and_gpu`.

The default stays the ANE, even for laya-en and laya-typed-decisions,
which grade A on the GPU and C on the ANE at about the same latency:
keeping background work off the GPU is still the project's thesis, and a
manifest that wants the GPU can say so. A v0.4 daemon ignores the key and
loads on the ANE.

## D32 — Refuse ANE-served models over the Neural Engine's weight limit
Core ML places an ML program on the Neural Engine only if its weights are
under about 1 GiB. Above that, a model loaded with `.cpuAndNeuralEngine`
runs entirely on the CPU, with no error and no log line. Measured on an
M1 Max with macOS 27.0, a Qwen3 backbone truncated to 23 layers
(0.964 GiB of fp16 weights) put 99.4% of its operations on the ANE, and
at 25 layers (1.022 GiB) none. agent-jev (Qwen3-0.6B, 1.12 GiB) found it.
Apple's coremltools guide states the same 1 GB Neural Engine limit for
iPhone, and ships `bisect_model` to split larger models. The limit is per
program: two 0.85 GB laya buckets run together on the ANE without trouble.
Other chips and OS versions may draw it elsewhere.

**Decision.** `MAX_ANE_PROGRAM_WEIGHT_BYTES` (1 GiB), measured from an
artifact's weight files, is a hard limit wherever a model would be served
on the ANE, with a bypass for trying anyway:
- **Converters** fail an ANE-served artifact over it, naming the limit and
  the fixes: serve it on the GPU (`compute_units = "cpu_and_gpu"`, D31),
  convert with `--int8-embedding`, or pass `--ignore-ane-weight-cap`. The
  bypass warns per bucket and is recorded in the report and in a comment in
  the installed manifest. The compute-plan gate still judges the actual
  placement.
- **sidekickd and `libsidekick.dylib`** skip a bucket over it when the
  model's compute units are `cpu_and_ne` or `all` (D28's skip-and-warn),
  with the reason and the fixes in `/health` and `sk_pool_skipped`. A
  manifest can opt out with `ane_weight_limit = "ignore"`, and the daemon
  with `--ignore-ane-weight-cap` (or `ignore_ane_weight_cap` in its
  config). GPU- and CPU-served models aren't checked. The parity suite
  scans with the limit off, so it can still grade an over-limit model's
  CPU and GPU paths.

A model that silently runs on the CPU is slower than its owner expects and
holds the same memory, so failing loudly is the right default, in line with
D22. `--int8-embedding` is the first weight quantization: it stores only the
token table (a CPU-side lookup) as per-row int8, and is graded against the
model's ideal-fp16 ceiling like any other rewrite. Chunked variants, which
lose no precision, are planned.

## D33 — Cap CPU-served models at 1,024 tokens
On the CPU, Core ML's fp16 matmul is accurate at every contraction length,
but past 1,024 it sums in a different order. Measured on an M1 Max with
macOS 27.0, on attention's value matmul with the same 754 real keys
padded to 1,024 and to 2,048: the two results differ by up to 2e-3, while
each is within about half an fp16 step of the exact (float64) product.
Up to 1,024 the results are bit-identical. Attention contracts over the
bucket length, so a model with buckets past 1,024 isn't exactly
bucket-invariant on the CPU. Lumma-fev-0.1b's probabilities differ by up
to 0.021 between its 1,024 and 2,048 buckets, agent-jev's by up to 0.018,
on inputs over 512 tokens. `tools/repro_cpu_matmul_accumulation.py`
reproduces it.

No conversion fixes it. Fixed 512-key slices summed in order make lengths
agree, but Core ML runs a sliced matmul about 13× less accurately, and
that would break the exact agreement short inputs have today. The ANE is
exactly bucket-invariant once the softmax is written as a matmul (D28
amendment), and the GPU stays within its fp16 gate.

**Decision.** As with the ANE's weight limit (D32), a known hardware limit
becomes a default cap with an explicit opt-out, not a grading failure:
- **The runtime caps a model served `cpu_only` at `MAX_CPU_INVARIANT_SEQ`
  (1,024).** Buckets above it aren't loaded, and longer inputs follow the
  model's own over-length rules. That means a 400, or truncation where its
  format truncates; fev's state limit follows fev's own formula on the
  capped window. A model with no bucket at or under 1,024 is skipped with
  the reason and the fixes. `/v1/models` and `/health` report a cap in
  effect.
- **Opt-out:** `cpu_seq_limit = "ignore"` in a manifest, or
  `--ignore-cpu-seq-cap` (`ignore_cpu_seq_cap`) for the daemon. It reads
  as "I accept small bucket-dependent differences above 1,024". Models
  served on the ANE or GPU aren't capped. Neither is `all`, which lets
  Core ML choose, so the cap would hide what it does.
- **The parity suite** reports the CPU path's bucket check past 1,024 as
  this limit, with the measured variation, rather than failing it. Up to
  1,024 the CPU gate stays exact, and the GPU and ANE gates are unchanged.
  The worker records each bucket pair's delta, so a run can be re-rendered
  under a changed rule.

With the CPU's past-1,024 check reported rather than failed, Lumma-fev-0.1b
passes every gate. Served on the ANE (B, against its fp16 ceiling) and
graded A on the GPU, it is supported under the preview rule (D28
amendment).

**Also in this change:** the parity suite treats a compute plan as a gate
only for a model served on the ANE (`cpu_and_ne` or `all`), as the
converters do. For a GPU- or CPU-served model, an ineligible plan is
reported, and every path is still graded. Placement isn't correctness: the
other hard gates (bucket and pad invariance, determinism, finite output)
still apply on every path, because they catch defects in the artifact
itself whichever device runs it.

## D34 — The fev format: Lumma-fev decision models
Lumma-fev (FrontiersMind/Lumma-fev, Apache-2.0) is a family of decision
models on a causal decoder backbone. Lumma-fev-0.1b (154M parameters) reads
the request's state and one question in a single row, and scores each option
by comparing the hidden state at the end of the option with the hidden state
at a final decide token. That is one static pass per question, so it fits
`/v1/classify`'s zero-shot contract (D28) as a third format beside laya and
gliner2 (D30). The contract is in `docs/design/classify.md`.

- **The row** is the state, the question's instructions, each option, then a
  decide token, separated by the five delimiter tokens the manifest names by
  role. There are no other special tokens. sidekick ports the checkpoint's
  own `modeling_fev.py` input builder to Rust and checks it against a token
  fixture the checkpoint's code generated. `<|name|>` in any text is
  rewritten to `<¦name¦>`, so no request can forge a delimiter.
- **Options** render as fev renders them: choice as given, score as given,
  noul's `false`/`true` labels as `no`/`yes`, each optionally described.
  `question_type` is required. The model never sees it, but the rendering
  depends on it. Instructions are optional, because the model has no
  default question.
- **Lengths** follow the checkpoint: the state is truncated to 1,408 tokens
  (`min(8192, window − 640)` for its 2,048-token window), and a row still
  over 2,048 tokens is a 400. When the model is served on the CPU, D33's cap
  shrinks the window to 1,024, and the state limit follows the same formula
  (384). That is sidekick's choice; the checkpoint itself keeps 1,408.
- **The Core ML interface** adds an int32 `decide_pos` input beside laya's
  `marker_pos`, and the graph selects the option and decide states with
  one-hot matmuls. The checkpoint's temperature is folded into the graph.
- **The converter** (`tools/convert_fev.py`, on the conversion library's new
  Nandi backbone) folds the factorized embedding's projection into the
  token table, so that no compute-heavy linear stays on the CPU. It writes
  the softmax as a matmul, builds SiLU from tanh, and pre-scales the norms
  where their squares would overflow fp16.

**Measured** (M1 Max, macOS 27.0, 2,630 cases, against the ideal-fp16
ceiling): B on the ANE, where it's served (1.57× the ceiling, no flips,
exactly bucket-invariant), A on the GPU (0.95×), and D on accuracy on the
CPU, with the D33 limit past 1,024 tokens. It's supported under the preview
rule. Latency is ~33 ms at 256 tokens, but ~1.2 s at 2,048 on the ANE.
Gold accuracy on fast-decisions is 30.7%, the same as fp32, and is
reported, not graded, beside FrontiersMind's own published numbers. The
larger Lumma-fev models (0.6b and up) exceed the ANE's 1 GiB weight limit
(D32).

## D35 — Report where Core ML placed each bucket, and which bucket answered
The `sidekick-compute-units` header reports the configured compute units
(D28): Core ML doesn't say which device ran a given prediction. For a client
that records what produced each answer (a regression harness, an
evaluation), configuration alone isn't enough. A model can be configured for
the ANE and still run some operations elsewhere: laya-en's embedding lookup
runs on the CPU, so 1,701 of its 1,719 operations are on the ANE. Before D32,
a model over the weight limit could run entirely on the CPU.

**Decision.** Two additions, both measured facts reported beside the
configuration:
- **Placement per bucket.** `/v1/models` reports, for each bucket, Core ML's
  compute-plan op counts by device (`ane`, `gpu`, `cpu`), and `/health` adds
  the unassigned ops, the off-ANE op names and where the counts were
  measured. Counting matches `ane_check`, the parity suite and
  docs/MODELS.md.
- **The bucket each input used.** A `sidekick-buckets` response header lists
  one bucket per input, in input order, on the classify, embeddings and
  rerank routes, so a response carries which compiled program answered it.
  Chat and static embedders omit it.

**Where the counts come from.** A live compute-plan read in the daemon is a
second full compile: measured on an M1 Max with macOS 27.0, about 50 s for a
1,024-token bucket and nearly 4 minutes for a 2,048-token one, about the
weights' size again on disk in Core ML's cache, and up to ~0.4 GB of
footprint kept afterwards. So by default sidekick reports the plan the
converter already read on the same machine. Converters record it per bucket
in the installed manifest's `[placement]` table, stamped with the chip, the
macOS build and the date. The daemon reports it as `source: "conversion"`,
and marks it `stale` when the running chip or macOS build differs (a macOS
update can move operations). A live read (`report_compute_plans = true`,
off by default) reports `source: "live"`, one bucket at a time, after the
bucket's first load, cached on disk. It's for an operator who needs the
current OS's plan and accepts the cost. Artifacts converted before this
release report no placement until they're reconverted.

It remains the compiled plan, not the device a particular prediction ran
on, which Core ML doesn't expose; the docs say so.

## D36 — The agentjev format: candidates scored in one tree-shaped pass
agent-jev (aimeigaoshou/agent-jev, Apache-2.0) is a 598M-parameter
done-detector on a Qwen3-0.6B backbone: given a task's state, a question and
candidate answers, it returns how likely each is. Its own code encodes each
candidate as a separate causal sequence (state + question + candidate) and
reads the last token. A permutation-equivariant set head then compares the
candidates, followed by a per-question-type temperature. sidekick serves it
as a fourth zero-shot format on `/v1/classify` (D28, D30, D34). The contract
is in `docs/design/classify.md`.

**One static pass, exact.** A static graph can't reuse a KV-cache prefix per
candidate, so sidekick lays a question out as a tree: the shared prefix, then
every candidate's suffix. A segment-id mask, built in the graph, lets each
candidate see the prefix and its own earlier tokens but never a sibling, and
position ids restart at the prefix length for each candidate. In fp32 this
reproduces the authors' per-path scoring to |Δp| 5e-7. The tree is also much
shorter than the separate paths: 63–76 tokens where they needed 122–291.
- The Rust port of the authors' `contract.py` (pinned by hash) builds the
  tree, and a token fixture from their own code checks it. It matches on
  all 2,627 corpus cases.
- Requests are refused, never truncated, when the tree exceeds the largest
  bucket, as the authors' service refuses an over-long path. sidekick's
  limit counts the prefix plus every suffix, which is stricter. At most 32
  candidates (`max_labels`).
- The authors' per-type temperatures are the manifest's calibration table.

**Served on the GPU.** At 1.12 GiB of fp16 weights per bucket, agent-jev is
over the ANE's weight limit (D32), so Core ML would run it on the CPU. An int8
token table brings it under the limit, but on the ANE it is then slower than
the GPU at every length (and slower than the CPU at 2,048 tokens) and less
accurate. Its manifest sets `compute_units = "cpu_and_gpu"` (D31). Graded
against its ideal-fp16 ceiling: A on the GPU (p99 at 0.70× the ceiling, no
flips, 77 ms per case). It stays a **preview**: its CPU path fails the exact
bucket-invariance gate below 1,024 tokens, where D33's limit doesn't apply
(Δp up to 7e-3 between its 512 and 1,024 buckets on the same input, with
similar accuracy against fp32 in each). Its ANE path runs on the CPU and
fails the same way. Unlike Lumma-fev, whose CPU path is exact up to 1,024,
agent-jev's CPU arithmetic depends on the bucket below it too. That's
observed and not yet isolated.

Its authors report modest quality: AUROC 0.589 for coding-task completion,
where 0.5 is chance. Their service says its calibration isn't guaranteed.
sidekick grades fidelity only, so those numbers are the consumer's call and
are in docs/MODELS.md.

**Amendment (October 2026, macOS 27.0, M1 Max): the CPU's bucket
dependence below 1,024, isolated and fixed; agent-jev is supported.** The
cause was the softmax. Converted through transformers' sdpa path, every
attention became matmul → softmax → matmul with Core ML's own softmax op.
On the CPU, that softmax, reading the score matmul's output in the same
program, rounds differently for different key lengths below 1,024: with
the same real scores and the rest masked, 512 and 1,024 keys differ by up
to 1.5e-3 in a probability. The same softmax fed the scores as an input
is bit-identical across lengths. tools/repro_cpu_softmax_length.py
reproduces it standalone and exits 1 while the CPU behaves this way.
Bisecting one input through agent-jev's own graph agreed: the embedding,
tree mask, RoPE tables, norms, projections and MLP were bit-identical
between the buckets, the attention scores too, and the softmax was the
first step to differ.
- **The fix** is the softmax lumma-fev already used (D34, its converter's
  constraint D): exp(w − rowmax) and one matmul against [V | 1], whose
  matmul doesn't depend on the length up to 1,024. The library's
  `qwen3.matmul_softmax` rewrite routes every layer's attention through
  it, and the converter applies it after its fp32 references, so the
  fp32 gate proves it exact.
- **Graded on 2,627 cases.** The GPU stays A (p99 at 0.85× the ceiling,
  worst 3.2e-3, no flips) and its buckets become exact too. The CPU is
  exact in every bucket up to 1,024 (it moved by up to 0.019 before) and
  D on accuracy; past 1,024 its 5.0e-3 is D33's limit, reported. The ANE
  path, which the runtime doesn't serve because the model is over the
  weight limit (D32), is reported, not graded: the parity suite now
  reports, rather than grades, a path the runtime refuses to serve that
  way.
- Every hard gate passes on every path, the conversion is exact in fp32,
  and the GPU grades A, so agent-jev is supported (D28 amendment). It is
  still served on the GPU.
- Artifacts converted with an earlier `convert_agentjev.py` keep working
  and keep their GPU grade, but their buckets agree exactly only once
  reconverted.

## D37 — Chunked buckets: a chain of programs under the ANE's weight limit
D32 refuses to serve a bucket on the ANE when its program carries more
than 1 GiB of weights, because Core ML would silently run it on the CPU.
Until now a model over the limit had two ways onto sidekick: the GPU, or
`--int8-embedding`. The limit is per program, so a bucket split into
several programs, each under it, can run on the ANE with no change to its
arithmetic.

The reason to want that is choice, more than speed. sidekick is meant to
run a small companion model beside a larger primary one, a local LLM say,
that the GPU serves. A companion on the GPU competes with the primary for
it; on the ANE it leaves the primary's GPU time alone, even where it
answers more slowly. Which matters more is the operator's call (D38), so
an artifact should be able to serve either way.

**What was measured** (agent-jev, Qwen3-0.6B with 1.115 GiB of fp16
weights; M1 Max, macOS 27.0; two chunks):
- coremltools' `bisect_model` splits the converted program in two at its
  weight midpoint. On agent-jev that point falls inside an MLP, so six
  fp32 tensors cross the boundary: the attention mask, RoPE's cos and sin,
  two hidden states and an MLP intermediate. It makes two chunks only, and
  it sets the second chunk's opset to iOS 16.
- Splitting in torch, before tracing, at layer boundaries (the embedding
  and layers 0–9, then layers 10–27, the final norm and the head; 0.587
  and 0.536 GiB) passes only the residual stream: one fp16 tensor
  [1, S, 1024]. Each chunk rebuilds the tree mask and RoPE from the int32
  inputs it takes, keeps every sidekick rewrite and the macOS 15 opset,
  and puts 96–99% of its operations on the ANE (98.6% and 99.3% at 2,048
  tokens).
- The boundary loses nothing. On the GPU the chain's logits are
  bit-identical to the unchunked model's. On the ANE, the layer-boundary
  chain and `bisect_model`'s chain, cut in different places, give
  bit-identical logits. What separates the ANE from the GPU is the ANE's
  own arithmetic, not the chunking.
- The boundary costs under 2% on the GPU. Through sidekick's own Core ML
  path (`chain_timing`), median per prediction, unchunked → chunked
  (cut after layer 8, 0.570 and 0.578 GiB at 2,048 tokens): 85.4 →
  87.0 ms at 256 tokens (the 512 bucket), 204.4 → 207.2 ms at 1,024,
  550.8 → 559.6 ms at 2,048.
- On the ANE, chunked agent-jev answers more slowly than on the GPU when
  nothing else runs: 128.2, 331.3 and 1,234.5 ms at the same lengths. Its
  first load compiles each chunk, about 40 s for the first bucket and
  244 s for the 2,048-token one, after which Core ML's cache serves it.
  What it costs or saves the GPU's other work is measured separately
  (tools/companion_bench.py).
- Graded by the parity suite on its 2,627 cases, the chunked artifact
  keeps the unchunked one's grades on the GPU (A, 0.85× the ideal-fp16
  ceiling) and the CPU (D), and grades D on the ANE (p99 at 4.2× the
  ceiling, worst |Δp| 0.022, no decision flips, buckets exact, the same
  output in two processes). An ANE precision rewrite later took its ANE
  path to B (1.44×; D39), the GPU staying A.
- Beside a 4-bit 27B Qwen3.8 that oMLX served on the GPU of a 32 GB M1
  Max (tools/companion_bench.py, docs/MODELS.md), the same install served
  either way by the daemon config (D38): on the GPU, the companion held
  1,195 MB in sidekickd, and oMLX's memory-pressure policy unloaded the
  27B within seconds, so none of its generations completed; on the ANE it
  held 58 MB, every generation completed at about 2% below the primary
  alone, and the companion drew about 1.9 W against 7.4 W on the GPU.
  That is the case D38 exists for.

**Decision.** A bucket may be an ordered chain of programs.

- **One artifact serves either placement.** On the GPU a chain gives the
  unchunked program's output bit for bit at under 2% more latency, so a
  converter whose backbone can be chunked chunks a model over the budget
  by default (`--chunks auto`), whatever its manifest serves it on.
  `--chunks N` or `--chunks 10,20` (the layers at which each chunk after
  the first begins) choose the split, and `--chunks 1` keeps one program
  per bucket. A model under the budget stays one program. A backbone that
  can't be chunked converts as before, and an ANE-served bucket of one
  over the limit fails as D32 says, naming chunking as the first fix.
- **Where it runs is a choice, not a default.** The manifest's
  `compute_units` (D31), or the operator's (D38), decides; neither follows
  from chunking. agent-jev stays GPU-served, and its chunked ANE path is an
  option, graded in docs/MODELS.md.
- **Cuts fall at layer boundaries.** `auto` packs whole layers into as few
  chunks as keep each chunk's fp16 weights under
  `CHUNK_WEIGHT_BUDGET_BYTES`, 0.9 GiB, then balances them. The limit was
  measured on one chip and one OS (D32), between 0.964 and 1.022 GiB, so
  the budget leaves a margin below the largest program measured on the
  ANE. The compiled chunks are measured against D32's limit like any
  program.
- **Only the residual stream crosses.** Every chunk but the last outputs
  `hidden_out`, an fp16 [1, S, H] multi-array, and every chunk but the
  first takes it as `hidden_in`. Each chunk takes the model's int32 inputs
  that it reads, by their usual names (the first chunk `input_ids`; every
  chunk the mask, segment and position inputs; the last chunk any input
  its head reads), and the last chunk produces the manifest's output. The
  runtime passes the output feature value of one chunk to the next as it
  is: sidekick makes no copy, though Core ML may copy internally (the
  boundary's measured cost includes that).
- **The manifest.** `artifact` gains a `{chunk}` placeholder, numbered
  from 0, beside `{seq}`: `model_{seq}.{chunk}.mlmodelc`. A `[chunking]`
  table says how many chunks each bucket has, and records the budget the
  converter split under:

  ```toml
  artifact = "model_{seq}.{chunk}.mlmodelc"

  [chunking]
  chunks = 2
  weight_budget_bytes = 966367641
  ```

  A model without `[chunking]` has one program per bucket; inside the
  table, `chunks` is required. `{chunk}` without a `[chunking]` table, or
  a `[chunking]` table without `{chunk}`, skips the manifest with a reason
  (D28). At load a classifier checks every chunk's declared inputs and
  outputs against these rules. Chunking is for classifiers for now: no
  converter makes a chunked embedder, and an embedder's manifest with
  `[chunking]` is skipped with that reason. Buckets, `sidekick-buckets` and every other
  per-bucket behavior are unchanged: a bucket is still one entry, whatever
  the number of programs that answer it.
- **A chain is served one way.** Every chunk of every bucket loads with
  the manifest's `compute_units`. Placing chunks on different devices
  (chunk 0 on the ANE, chunk 1 on the GPU) is possible later, not now.
- **The limits apply as to one program.** D32's weight limit is checked
  per chunk, and `/health` and `sk_pool_skipped` name the chunk that is
  over (`model_1024.1.mlmodelc`). D33's CPU cap applies to the chain's
  buckets as to any model's. `/health` reports each chunk's weights, the
  limit and the conversion's budget.
- **Placement** (D35) keeps one count per bucket, the sum over its chunks,
  with each chunk's counts under it in `[placement.buckets.<seq>]`'s
  `chunks` list. The converter also records the plan for the other of
  `cpu_and_ne` and `cpu_and_gpu`, under
  `[placement.alternatives.<units>]`, for every model, and the daemon
  reports the record for the units the model runs on, so a model the
  operator moves keeps its placement report.
- **Gates.** The torch gate checks the composed chunk wrappers against
  the unchunked wrapper in fp32 in every bucket, as well as the unchunked
  wrapper against the checkpoint. The usual Core ML gates run on the
  chain. In the smallest bucket, a Core ML gate also converts the bucket
  unchunked and requires the chain's output on the GPU to be bit-identical
  to it: an exact check of the chain's plumbing, which is the same in
  every bucket (`--chunk-identity-all` runs it in all of them, at the cost
  of converting every bucket twice). The parity suite grades a chained
  model like any other, reading each bucket's compute plan from all of
  its chunks.

A daemon older than this release ignores `[chunking]`, and its `{chunk}`
path names no file. It lists the model, finds no weights at that path to
check, and fails every request to it with an error naming the missing
artifact, so it never serves part of a chain. Chunked models need
sidekick 0.7.0.

## D38 — The operator may choose a model's compute units
D31 lets a manifest name its compute units, and the converter that wrote
the manifest chose them from measurements of the model alone. Where a model
should run also depends on what else the machine runs. A small model served
beside a larger one that the GPU serves (a companion to a local LLM, say)
competes with it for the GPU. On the ANE it may answer more slowly, but it
leaves the larger model's GPU time alone. That is the operator's trade-off
to make, not the converter's, and until now making it meant editing an
installed manifest.

**Decision.** The daemon config may set a Core ML model's compute units,
by model id:

```toml
[models."agent-jev"]
compute_units = "cpu_and_ne"
```

- The registry applies it when it scans, before D32's weight limit and
  D33's CPU cap, which judge it exactly as they would the manifest's own
  choice. An operator who moves an over-limit model onto the ANE has it
  skipped, with D32's reason, and one who moves a long model onto the CPU
  has it capped at 1,024 tokens.
- `/v1/models` reports each Core ML model's `compute_units_source`,
  `manifest` or `operator`, beside its `compute_units`. `/health` lists
  the overrides applied, and any that named no Core ML model (a static
  embedder has no compute units, and a typo shouldn't pass silently). The
  `sidekick-compute-units` header keeps its meaning: the units the model
  runs on, whoever chose them.
- An unknown key under `[models."<id>"]`, or an unknown value, is a
  config error, as for the rest of the config.
- The recorded placement (D35) is reported for the units the model runs
  on, when the converter recorded one for them.

`libsidekick.dylib` reads no daemon config, so it keeps the manifests'
choices.

## D39 — The ANE's SiLU, and a cancellation-free form
Chunked agent-jev (Qwen3-0.6B, D37) graded D on the ANE: p99 |Δp| at 4.23×
its ideal-fp16 ceiling, worst 0.022, though no decision flipped. Its GPU
graded A with the same artifact, so the ANE's arithmetic was at fault.

**What was measured** (M1 Max, macOS 27.0; the 12 worst ANE cases at 512
tokens, the model converted one layer per program and each fed its fp32
input, so the error a layer adds on its own is measured apart from what
reaches it):
- Running each half of the chunked chain on the ANE and the other on the
  GPU put twice as much error in layers 0–8 as in layers 9–27. Per layer,
  layer 0 (with the embedding) added 1.0e-2 relative rms error on the ANE
  against 9.5e-4 on the GPU, layer 1 4.0e-3 against 5.6e-4, and layers 2–27
  about 1.5–2× the GPU's.
- agent-jev's residual stream starts tiny (the embedding's rms is 0.03) and
  every layer's o_proj and down_proj inputs sit under the ANE linear's
  precision floor (rms 0.06–0.2; D17). Power-of-two rescales of q/k/v,
  o_proj and down_proj alone changed almost nothing (layer 0: 9.9e-3).
- Replacing silu did: layer 0 / layer 1 local error was 1.0e-2 / 4.0e-3 with
  Core ML's native silu, 2.5e-3 / 2.2e-3 with TanhSilu, 3.8e-3 / 5.5e-3 with
  x·sigmoid(x), and 2.4e-3 / 1.6e-3 with a cancellation-free form,
  StableSilu: 2x·exp(min(x, 0)) / (1 + exp(−|x|)). TanhSilu's 1 + tanh(x/2)
  cancels in fp16 as x grows negative; the stable form has no cancellation
  for either sign and can't overflow. The GPU's per-layer error was the
  same with every form.
- A one-op program doesn't show this: Core ML doesn't place a lone `silu`
  on the ANE, so every form measured alike there. Activations have to be
  compared inside the model.
- Through all 28 layers and the head, the ANE's |Δp| on those cases fell
  from 1.27e-2 to 1.81e-3 with StableSilu, and to 1.52e-3 with the rescales
  as well. The rest was the head, whose scorer has its own silu and whose
  set layers use the erf gelu.

**Decision.** `activations.StableSilu` joins the ANE-safe activations, and
`qwen3.precision_rewrite` takes the silu form and an optional down_proj
input rescale (`silu=`, `mlp_down=`; the defaults, TanhSilu without it,
leave F2LLM unchanged). agent-jev's converter applies StableSilu, the
q/k/v, o_proj and down_proj rescales, and the head's own rewrite (its
scorer silu as StableSilu, its set layers' gelu as TwiceGelu), after the
fp32 references and calibrated on its gate states (D26), and it forbids
the native `silu` and `gelu` ops. Every factor is a power of two, so the
fp32 graph is exact (max |Δlogit| 8e-6 to 1.2e-5 against the checkpoint),
and on the GPU the chain stays bit-identical to the unchunked program.

Graded on its 2,627 cases, agent-jev's ANE path moves from D to B (p99 at
1.44× the ceiling, worst 5.6e-3, no flips, buckets exact, the same output
in two processes; 152 ms per case against 128), the GPU keeps its A
(0.84×, worst 3.7e-3), and the CPU its D (5.9×). The claim is general: in a SiLU MLP, the ANE's
native silu can be the dominant error. Whether StableSilu also lifts the
other SiLU models (Lumma-fev, LFM2.5) is measured separately.

**Amendment: the other SiLU models.** The three other SiLU models already
used TanhSilu. Each was reconverted with StableSilu and graded by the full
parity suite against its fp32 reference (M1 Max, macOS 27.0, each ANE run
alone on the machine):
- **Lumma-fev-0.1b** (Nandi decoder, no QK-norm, no rescales before). On
  its 12 worst ANE cases at 512 tokens, StableSilu alone halved the mean
  |Δp| (6.2e-3 → 3.0e-3). Adding power-of-two rescales of q/k/v, o_proj
  and down_proj changed nothing measurable: the calibrated scales came
  out at 1 for q/k/v and at most 2 for o_proj and 4 for down_proj, since
  this model's linear inputs aren't small. Over all 2,630 cases the ANE stays B, but p99 moves from
  1.57× to 1.43× the ceiling and the worst |Δp| from 0.010 to 0.0057,
  under the ceiling's own worst (0.0059). The GPU stays A (0.92×), the
  CPU D (5.7×); buckets stay exact on the ANE. The ANE's median rises
  from 33 to 37 ms per case.
- **F2LLM-v2-160M** (Qwen3, attention already rescaled). Its ANE grade
  was A. StableSilu cut the ANE's worst 1 − cosine from 2.8e-5 to
  1.8e-5; the down_proj rescale (scales up to 16 in the early layers),
  which TanhSilu's error had masked, cut it to 0.7e-5, with similarity
  drift from 0.0019 to 0.0008. The GPU is unchanged (worst 1e-6), the
  CPU stays B. The ANE's median rises from 4.6 to 5.3 ms per input.
- **LFM2.5-Embedding-350M** (hybrid conv/attention; its MLP input was
  already rescaled to rms ~1). No measurable change: worst 1 − cosine
  1.2e-5 → 1.4e-5, drift 0.0033 → 0.0035, mean 6e-6 → 5e-6. Its rescale
  keeps the silu's inputs where TanhSilu's 1 + tanh(x/2) doesn't cancel
  in fp16 (that needs large negative inputs), so the form doesn't matter.

So the claim holds with a bound: StableSilu helps where a SiLU sees large
negative inputs in fp16, and is neutral where an earlier rescale keeps
them small. Rescales help only where linear inputs sit under the ANE
linear's floor (agent-jev, F2LLM's down_proj), and do nothing elsewhere
(Lumma-fev). The explicit exp, abs and divide cost 12–15% of ANE time.

Lumma-fev's converter (`nandi.stable_silu`) and F2LLM's
(`qwen3.precision_rewrite(silu=StableSilu, mlp_down=True)`) adopt it by
default; LFM2.5's keeps TanhSilu. Every factor is a power of two, so each
fp32 gate stays exact. Artifacts converted earlier keep working; a
reconversion gives the smaller ANE error.

## Hardware verification status

Verified on Apple Silicon (macOS 26.5.1, Xcode 26.6, July 2026), via
`cargo run -p sidekick-server --bin smoke-test` and live `sidekickd` runs:
- `swift/bridge.swift` compiles against the real macOS 26 SDK and behaves:
  availability probe, plain completion, session reuse (~4x faster warm than
  cold), and `DynamicGenerationSchema` constrained decoding returning valid
  schema-conforming JSON.
- Static embedding tier end-to-end over HTTP with a real model2vec artifact
  (potion-base-8M): float + base64 encodings, sane cosine structure.
- One runtime lesson encoded in code: binaries linking the Swift shim need
  `-rpath /usr/lib/swift` or they abort at dyld load (see sidekick-fm and
  sidekick-server build.rs), and cold-replay transcripts can make the model
  emit a leading `Assistant:` label (stripped in the backend).
- Core ML encoder path end-to-end with a locally converted bge-small
  (tools/convert_bge_small.py): server `/v1/embeddings` parity vs torch
  fp32 at worst cosine 0.99998; query-prefix, bucket selection (incl. lazy
  per-bucket load), matryoshka rejection, and residency reporting all
  exercised over HTTP. ANE residency measured via
  `cargo run -p sidekick-coreml --example ane_check`: 3.4x/2.4x/1.75x over
  CPU at seq 128/256/512 (see D15 for the conversion constraints this
  required). Re-measured per compute path over the shared parity set when
  the artifacts were regenerated for docs/MODELS.md (D19): CPU_ONLY
  0.999972 / CPU_AND_NE 0.999984 worst-case. That re-measurement also
  established two operational lessons now in MODELS.md: ANE residency
  ratios destabilize under concurrent GPU load, and ANECCompile stderr
  failures can be transient service state rather than artifact defects. Objc exceptions from Core ML (e.g. its E5RT/IOSurface
  failures) abort the process — Rust cannot catch them; the fix is
  converting models that don't provoke them (D15), not catching.

- EmbeddingGemma-300m end-to-end (July 2026): conversion via
  tools/convert_embeddinggemma.py (D17), ANE residency 3.4x/3.1x/2.9x at
  buckets 128/256/512 via ane_check, server /v1/embeddings parity 0.9905
  vs fp32 sentence-transformers (matching the ANE gate exactly — the Rust
  tokenizer path is token-identical), matryoshka dimensions 512/256/128
  with unit norms and 400 on undeclared dims, query/document prefixes,
  and a 831-token input through the 512 bucket (47ms warm). Re-converted
  in September 2026 with the MLP precision rewrite (D17 amendment, macOS
  27): parity CPU_AND_NE 0.999996/0.999996/0.999989, CPU_ONLY 0.99994,
  2161/2170 operations on the ANE, pad invariance 1.0000000, ane_check
  ratios 2.6x/2.4x/1.7x, live /v1/embeddings worst parity 0.999982 over 13
  inputs (394-token text, 527-token doc truncated to 512, query prefix).

- LFM2.5-Embedding-350M end-to-end (July 2026, D19): conversion via
  tools/convert_lfm25_embedding.py, ANE residency 2.49x/1.91x/1.66x at
  buckets 128/256/512, conversion parity CPU_ONLY 0.9999 / CPU_AND_NE
  0.987010 (the ANE figure identical to six decimals across buckets —
  constraint D's bucket-invariance), live /v1/embeddings worst parity 0.9856
  over the reference set incl. a 483-token text, unit norms, prefixes,
  and preserved similarity structure. Re-converted in September 2026 with
  the precision rewrite (D19 amendment, macOS 27): parity CPU_AND_NE
  0.999992 at every bucket, CPU_ONLY 0.99991–0.99994, 773/778 operations
  on the ANE, pad invariance 1.0000000, ane_check ratios 2.3x/1.9x/1.6x,
  parity suite ANE grade A (0.999988), and live /v1/embeddings worst parity
  0.999988 over all 51 suite inputs. LFM2.5-ColBERT-350M encoder
  smoke-tested on ANE (2.0x, per-token parity 0.9919, MaxSim ranking
  preserved) but not integrated — see docs/MODELS.md.

- F2LLM-v2-160M end-to-end (July 2026, D20): first causal-decoder embedder,
  conversion via tools/convert_qwen3_embedding.py, ANE residency
  2.02x/1.77x/1.59x at buckets 128/256/512, conversion parity CPU_ONLY
  0.9999 / CPU_AND_NE 0.99985 (bucket-invariant), live /v1/embeddings worst
  parity 0.99850 over the reference set including a 512-token doc (the case
  that exposed the EOS-truncation bug), unit norms, asymmetric query/document
  prefixes, and preserved similarity structure. gte-modernbert-base was
  tested and rejected at the time; that verdict was a Core ML
  attention-mask bug and is superseded (D25). Re-converted in September
  2026 with the precision rewrite (D20 amendment, macOS 27): parity
  CPU_AND_NE 0.999979 at every bucket, CPU_ONLY 0.999924, 648/653
  operations on the ANE, pad invariance 1.0000000, ane_check ratios
  2.7x/2.0x/1.5x, parity suite ANE grade A (0.999972), and live
  /v1/embeddings worst parity 0.999972 over all 51 suite inputs.

- macOS 27 (September 2026, D21): M1 Max, macOS 27.0, Xcode 27.0.
  Verified with the smoke test and a live `sidekickd`:
  - the shim builds against the 27 SDK and against a real 26.5 SDK
  - model info: AFM 3 Core, 4096 context
  - real usage, with cached tokens on session reuse
  - `finish_reason` length/stop, and `stop` sequences
  - over-long prompts → ContextOverflow with real counts → HTTP 400
  - single-turn prompt token counts identical to Apple's `fm serve`
  - real streaming (D23): 25 deltas for a 184-character reply, and deltas
    identical to the returned content
  - a mid-stream stop ends generation (0.76 s vs 3.5 s), and the follow-up
    after it survives
  - a client disconnect stops generation, seen in the daemon log

- ANE eligibility on macOS 27 (September 2026, D24):
  - compute plans for all four validated encoders at every bucket
  - a flexible-shape negative control rejected before any prediction
  - ratios re-measured (MODELS.md)

- gte-modernbert-base end-to-end (September 2026, macOS 27, D25). Conversion
  via tools/convert_gte_modernbert.py with explicit attention:
  - parity CPU_ONLY 0.999919 / CPU_AND_NE 0.999793 at every bucket
  - 794/805 ops on the ANE
  - pad invariance 1.0000000 on both paths (the old fused artifact: 0.27)
  - ANE ratios 2.9x/2.0x/1.55x
  - live /v1/embeddings worst parity 0.99896 over nine texts, including a
    722-token input
  - unit norms, and similarity structure matching fp32
  - re-converted with the range rewrite (D25 amendment): parity CPU_AND_NE
    0.999981 at every bucket, CPU_ONLY 0.99992, 794/805 operations on the
    ANE, pad invariance 1.0000000, ane_check ratios 2.9x/2.0x/1.6x, parity
    suite ANE grade A (0.999915), live /v1/embeddings worst 0.999833 over
    all 51 suite inputs

- Flexible-shape load guard (September 2026, macOS 27, D27):
  - the enumerated-shapes control aborts only under `.cpuOnly`, and is now
    refused at load under every compute unit
  - the range-shaped control loads with a warning and predicts
  - all twelve installed buckets load clean and predict

Still open:
- An automated ANE gate in a self-hosted CI job. `ane_check` now exits
  non-zero on an ineligible plan, so it is ready to be wired in, but nothing
  runs it automatically yet.
- Multifunction mlprogram weight sharing to collapse the 3x ~600MB
  per-bucket artifact duplication for large encoders (D17).
