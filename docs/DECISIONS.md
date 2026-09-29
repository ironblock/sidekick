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
wrapper exposes the choice; measurement can override.

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

## D20 — Two more architecture classes: ModernBERT rejected, Qwen3 decoder validated
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

## D25 — ModernBERT validated: its rejection was a Core ML attention-mask bug
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
- On the ANE, bge-small and EmbeddingGemma grade A, and gte-modernbert and
  F2LLM grade B.
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
  from 0.995 to 0.9999, the GPU's value.
- The GPU path measures at fp32-like accuracy on every model (A). "GPU fine,
  ANE low" therefore isolates the ANE, not fp16 arithmetic in general.

**The 0.985 gate.** It stays the converters' acceptance gate on their own
parity sets; the EmbeddingGemma and LFM2.5 converters now gate at 0.999.
A D on the adversarial corpus doesn't remove a model: the grade is
published, and this chip's floor makes it a regression test.

**Not done.**
- Per-token grading for models whose token vectors are the product (laya,
  ColBERT). Every registry model pools inside its graph, so the suite can't
  see per-token vectors.
- Parity through the HTTP layer.
- F2LLM (SwiGLU) and gte-modernbert (GeGLU) grade B on the ANE. Neither has
  been checked against the D17/D19 rules yet.

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
  attention-mask bug and is superseded (D25).

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
