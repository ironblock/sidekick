# Decision log

Why sidekick works the way it does: each architectural decision, the
evidence it rests on and the alternatives it turned down, numbered in the
order they were made (newest last). An entry is amended in place when a
later measurement or decision changes it, and the amendment says what
changed; the code and the other docs cite entries by number (D1, D2, …).

## D1 — Daemon is OpenAI-compatible; the tier router lives behind it, not beside it
`sidekickd` mimics llama.cpp/MLX servers in miniature. The
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

*Amended by D53:* the chat route is to serve open models that a client
names, beside Foundation Models. `apple-fm` still answers 503 when
Foundation Models is unavailable, and nothing falls back to another model.

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

*Amended by D53:* the shim does drop some unsupported forms silently
(#179). D53 moves validation into a JSON Schema layer in `sidekick-core`,
which checks one subset for every chat backend and refuses the rest with a
400 `unsupported_schema`.

## D9 — Embedding purpose via non-standard `input_type` (amended: `none` turns the prompt off)
OpenAI's embeddings API has no query/document distinction, but EmbeddingGemma
and bge-family models want prompt prefixes. Added the Cohere-style optional
`input_type: "query" | "document"` field (default `document`); prefixes come
from the model manifest, so standard OpenAI clients work unchanged.

**Amendment (October 2026): `input_type: "none"` embeds the text as sent,
and the prompts are disclosed.** The prefix had no opt-out, and nothing
showed it to a client. A client that brought its own prompt got two: on
EmbeddingGemma, `task: classification | query: …` became `title: none |
text: task: classification | query: …`, a prompt the model never saw. So
MTEB's classification, clustering and similarity scores for EmbeddingGemma
couldn't be reproduced through sidekick, and neither could e5-small-v2's
non-retrieval scores, which MTEB measured with no prefix.
- **`none`.** `input_type: "none"` puts no prefix before the text: the
  tokenizer still adds its special tokens, and nothing else is added. It
  is accepted on `/v1/embeddings`, on `/v2/embed` beside Cohere's names,
  and on `/tokenize`, through the same parsers, so a count and `fits` are
  the route's. The library's `EmbedPurpose::Raw` is the same, and so is
  `sk_embed`'s purpose 2.
- **The default stays.** No `input_type`, or `document`, still applies the
  document prefix, and `query` the query prefix, so no existing request's
  vectors change; tests pin them. `/v2/embed` still refuses Cohere's
  `classification` and `clustering`. No manifest declares a prompt for
  them, and mapping them to `none` would silently change what those
  names have meant here. A client with its own prompt for them sends
  `none`.
- **Disclosure.** `/v1/models` lists each embedder's prefixes as `prompts`,
  `{"query", "document"}`, beside the nli format's `hypothesis_template`,
  with `""` for none. `sk_model_info` reports the same. docs/MODELS.md
  has a table of every build's prompts, generated from the committed
  manifests by `tools/models_table.py`. The README says the document
  prompt is the default and when to send `none`.

Named prompts (a model's task-specific prompts, chosen by name) are a
separate question, answered by D45.

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

**Amendment (October 2026, v0.8.0): what `/health` reports has grown.**
Beyond availability, it now gives the chat model's variant, context
limit and the SDK the shim was built with (D21); the models resident;
each skipped manifest with its reason and its path relative to the
models directory; seq caps (D33), compute-unit overrides (D38),
compute plans (D35) and chunked models (D37); and, for ONNX Runtime
(D40), the runtime's version, where it came from and its file name
(never a path), each onnx model's threads and their source, the
detected cores, and which models are quantized. It still carries no
credentials.

**Amendment (October 2026): the key guards every route but `/health`.**
"`/v1/*` only" stopped being exact when routes outside `/v1/` arrived:
`/rerank` and Cohere's `/v2/rerank` and `/v2/embed` (D29), and vLLM's
`/tokenize` and `/detokenize` (D44). Each of them serves a model as the
`/v1/` routes do, so the key guards them too: with an `api_key` set,
every route needs `Authorization: Bearer <key>` except `/health`, which
stays open for probes. A missing or wrong key is a 401 with code
`invalid_api_key`. A client that probes `/v1/models` must send the key.

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
optimization. *(Measured and declined in favour of one weights file shared
by every bucket, D49.)*

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

**Amendment (October 2026, v0.8.0): no ONNX in the dylib.**
`libsidekick.dylib` is built without the `onnx` feature (D40). It skips
onnx manifests, with the reason in `sk_pool_skipped`, and never loads
ONNX Runtime, so a host gets no new library dependency. Releases ship it
only in the Apple silicon archive.

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

**Amendment (October 2026, v0.8.0): the 27 SDK rule is for the Apple
silicon archive.** Only `aarch64-apple-darwin` links the Foundation
Models shim, so only that archive must be built with the 27 SDK. The
`x86_64-apple-darwin` archive builds the shim as a stub (chat answers
503, as on Linux); it declares macOS 13.4 and is built and tested on
macOS 26. The Linux archives use no Apple SDK.

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
penalties and `user` (`seed` and `top_p` were too, until the D53
amendment below; [API.md](API.md#sampling) has how they are honored). This is a behavior change for
clients that relied on those parameters being dropped, so it warrants a
minor version bump when released.

*Amended by D53:* `seed` and `top_p` are to be honored on the chat route
by both backends: open models sample with them, and Foundation Models
receives them through its random sampling mode.

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
now gate at 0.999. (Since superseded: every embedder converter gates at
0.999; see D26's October 2026 amendment.)
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

**Amendment (October 2026, v0.8.0): ONNX is a graded path.** With D40's
backend, three points above no longer hold as written:
- `onnx-cpu` is graded, not report-only. It runs natively through
  `OnnxEmbedder`, the daemon's own code path, in a build with the `onnx`
  feature (on macOS or Linux), and has floors for the M1 Max (ONNX
  Runtime 1.28.3) and the Xeon W-3235 (1.23.2). `--external` grades
  vectors produced elsewhere as a cross-check. Published exports still
  run at reference time, as before.
- Not every path goes through `CoremlEmbedder`: an onnx model goes
  through `OnnxEmbedder`, with the same prefix, tokenizer, truncation,
  pooling and normalization, and no buckets or padding.
- Not every registry model pools inside its graph: an ONNX graph with
  token-level output is pooled in Rust. The suite still sees only the
  pooled vector.

The Core ML gates (compute plans, bucket invariance, ANE determinism)
don't apply to an onnx model, which has no plan and no buckets.

**Amendment (October 2026): one definition of the letters.** The grade
scales had been written out separately in the suite's code, in MODELS.md,
on the comparison page and in this log. The published letters were stored
by hand beside the numbers that should determine them, so they could drift
apart, and they had: the data and MODELS.md's Intel table graded one model
differently. Now:
- **The scales are `[grades]` in `fixtures/parity/expectations.toml`:** the
  embedders' cosine scale, the classifiers' |Δp| scale, and the ratio to
  the fp16 ceiling. The suite reads them, and so does
  `tools/models_table.py`, which generates MODELS.md's legends and puts the
  scales in `docs/data/models.json` for the comparison page's legend. The
  thresholds quoted elsewhere in this log are history, not the definition.
- **No published letter is stored by hand.** The generator computes each
  row's letter from its numbers: worst error, ratio, graded flips and
  failed gates. A report's own letter must agree, and CI fails on any row
  whose stored letter doesn't. `docs/data/carried.toml` holds numbers only.
- **Errors are published on the same side of every bound.** Rounding for
  publication keeps adding digits until the rounded value earns the same
  letter. One formatter writes each error's text for MODELS.md and the
  page alike.
- **0.985 is no longer a converter gate.** Every embedder converter gates
  `CPU_AND_NE` at 0.999, the default in `EmbeddingGates`. 0.985 remains only
  the C bound.

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

**Amendment (October 2026, v0.8.0): provenance headers for ONNX models.**
`sidekick-compute-units` reads `cpu` for static embedders (the "static
ones" above) and for onnx models (D40); a Core ML model's value is its
configured units, which D31 and D38 let differ from `cpu_and_ne`. An
onnx model, like a static one, sends no `sidekick-buckets`: it runs each
input at its own length.

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

**Amendment (October 2026, v0.8.0): `token_type_ids` in ONNX embedder
manifests.** The rule that converters write `token_type_ids` only for
text-ranking models is about classifier manifests (`[classify.io]`). An
ONNX embedder's `[io]` (D40) names `token_type_ids` whenever its graph
takes it, as BERT-family exports do, and the daemon feeds it zeros. That
reopens no hazard: a daemon before v0.8.0 can't parse `backend = "onnx"`
and skips the whole manifest.

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

**Amendment (October 2026, v0.8.0): onnx manifests reject the key too.**
An onnx embedder (D40) always runs on ONNX Runtime's CPU, so, like a
static embedder, it has no compute units to choose: a manifest naming
`compute_units` is skipped ("only valid for the coreml backend").

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

**Amendment (October 2026): the limit depends on the chip or OS.** On an M4
Max under macOS 26.5.2, Qwen3-Embedding-0.6B's unchunked 128 bucket, with
1.110 GiB of fp16 weights, runs on the ANE.
- **The compute plan** put 2,124 of its 2,129 ops there. The five off it
  are mask setup.
- **Latency agreed:** 15.4 ms per input under `cpu_and_ne`, against 43.2 ms
  under `cpu_only`, with a resident set of 322 MB against 1.17 GB.
- **The same program on the M1 Max under macOS 27.0** gets 0 of 2,129 ops
  on the ANE.

So 1 GiB is the M1 Max's limit under macOS 27.0, measured between 0.964
and 1.022 GiB. On the M4
Max under macOS 26.5.2 it is at least 1.110 GiB. The two machines differ in
both chip and OS, so which of them moved the limit is open. The constant
stays, as the conservative choice. Chunking (D37) keeps a chunkable model's
programs under it, losslessly and at under 2% on the GPU. Deciding at load
from the machine's own compute plan, rather than from the constant, is
tracked as #102.

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

**Amendment (October 2026, v0.8.0): Core ML models only.** The cap is
Core ML's CPU behavior. An onnx model (D40) reports `compute_units:
"cpu"` but runs each input unpadded at its own length, so it has no cap
and no `seq_cap`.

**Amendment (October 2026): the ANE has a cap of its own (D48).** Models
served on the ANE (`cpu_and_ne`, and `all`) are now capped at 2,048
tokens, or their manifest's `ane_max_seq_len`, for the ANE's own reasons.
This record's cap is unchanged, and the GPU is still uncapped.

**Amendment (October 2026): the parity suite grades the cpu path under
this cap.** The suite grades what sidekick serves on each path. It already
applied D48's cap to its ane path; it now applies this record's cap to
its cpu path in the same way, by default. For a Core ML model whose
`max_seq_len` passes 1,024 and whose manifest doesn't set `cpu_seq_limit =
"ignore"`, the cpu worker loads only the buckets up to 1,024, as the
runtime serves a `cpu_only` model. It runs only the cases whose reference
ids fit, and lists the longer ones as `capped` in the cpu path's grade, not
failed. The grade covers the rest and is printed in a table of its own.
`--ignore-cpu-seq-cap` grades the cpu path past the cap, for measurement, as
the daemon's `ignore_cpu_seq_cap` serves it.
- **Why.** The bullet above, reporting the CPU's bucket check past 1,024
  rather than failing it, was implemented for classifiers only. An
  embedder's cpu path still gated every bucket pair and graded every case.
  So Qwen3-Embedding-0.6B's 2,048-token build graded F on the CPU for
  bucket invariance of 0.999894 against the 0.999999 gate (D49). That
  difference comes from buckets the runtime never serves on the CPU.
  Grading past the cap also mixed two questions into one letter: how
  accurate the CPU path is, and how much Core ML's CPU varies past 1,024.
  The second is this record's measurement, and `--ignore-cpu-seq-cap`
  still takes it.
- **Other models' CPU grades may change at their next regrade.** The cap
  affects every Core ML model with buckets past 1,024 that is graded on the
  cpu path: today Lumma-fev-0.1b, agent-jev, bge-m3-zeroshot-v2.0-c,
  ModernBERT-large-zeroshot-v2.0 and EmbeddingGemma 2, each at 2,048
  tokens. Their published CPU grades (D or F, with the D33 limit noted)
  came from runs that graded every bucket, and they stand until those
  models are graded again. A regrade under the cap leaves out the cases
  longer than 1,024 tokens and the bucket checks into 2,048. Where those
  decided the letter (bucket invariance past 1,024; Δp on long inputs), it
  may rise. Where the shorter cases decide it, it won't: EmbeddingGemma 2's
  CPU softmax already varies between buckets past 512 keys. Such a change
  reflects what the cpu path serves, not a change to the model.

**Amendment (October 2026, M1 Max, macOS 27.0): a manifest may lower the
CPU cap (`cpu_max_seq_len`).** 1,024 tokens isn't where every model's
buckets start to differ on the CPU. Qwen3-Embedding-0.6B's five-bucket
Core ML build (128 to 2,048 tokens) was graded on the cpu path under the
1,024 cap above. Its accuracy was B level (worst cosine 0.999684, as its
512-token build), but its bucket invariance was 0.999802 against the
0.999999 gate. A corpus of inputs at the bucket boundaries placed the
difference: a 511- and a 512-token input, run in the 512 bucket and again
in the 1,024 bucket, gave cosines of 0.999914 and 0.999917 between the two.
The 128-, 256- and 512-token buckets agree exactly on the CPU (1 − 2e-16
in the 512-token build's report), so the difference enters with the 1,024
bucket. That is below the 1,024 keys D33 measured on one matmul;
which operation sums differently there isn't isolated.
- **Decision.** A top-level manifest key, `cpu_max_seq_len` (either
  manifest file, coreml backend only, 1 to 1,024), lowers this record's cap
  for one model. It mirrors D48's `ane_max_seq_len` but only lowers: a
  value past 1,024 is refused, since lifting the cap is `cpu_seq_limit =
  "ignore"`'s. It applies wherever this record's cap does (`cpu_only`, and
  `cpu_and_ne` on a Mac without a Neural Engine). `cpu_seq_limit =
  "ignore"` and the daemon's `ignore_cpu_seq_cap` still lift the cap
  entirely, and `/v1/models`' `seq_cap.reason` says when the manifest set
  it. The parity suite grades the cpu path under it, as above.
- **Qwen3-Embedding-0.6B's manifest sets 512.** Its CPU path serves the
  128, 256 and 512 buckets, as its earlier 512-token build did; the ANE
  serves all five, and the GPU too. Graded again under it, the same build's
  cpu path is B (worst cosine 0.999684), with bucket invariance of 1.0
  over its 88 comparisons, and the run passes every gate.

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
  outputs against these rules. (Embedders were first left out: an
  embedder's manifest with `[chunking]` was skipped. The amendment below
  admits Core ML embedders.) Buckets, `sidekick-buckets` and every other
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

**Amendment (October 2026): chunked embedders.** A Core ML embedder may be
chunked, by the same rules: `{chunk}` and `[chunking]` come together, and
static and ONNX embedders still reject both. Its load checks each bucket's
chain as a classifier's does (the shared check: the inputs the manifest's
`[io]` names, `hidden_out` handed to `hidden_in` at [1, S, H], and the
last chunk's `embedding` output). The embedder converter's Qwen3 backbone
(F2LLM, Qwen3-Embedding) implements chunking. Each chunk rebuilds the
causal and padding mask from `attention_mask` and RoPE from the position
buffer, and its unchunked path is unchanged: F2LLM-v2-160M reconverts
byte-identically.

Measured on Qwen3-Embedding-0.6B (1.11 GiB of fp16 weights per bucket;
M1 Max, macOS 27.0), converted with `--chunks auto --chunk-identity-all`:
- **The chain.** It has two chunks per bucket: the embedding and layers
  0–8, then layers 9–27, the final norm and the pooling. Each has
  566–571 MB of compiled weights. Every bucket (128, 256, 512) passed
  the chain checks:
  - in fp32, the composed chunks equal the unchunked wrapper exactly
    (max |diff| 0);
  - on the GPU, the chain is bit-identical to the unchunked program;
  - the compute plan puts 2,129 of 2,135 operations on the ANE (99.7%),
    the rest being mask and pooling setup.
- **Which gate covered which chunk.** `ane_check` covered chunk 0: plan
  eligible, and pad invariance 1.0000000 on `.cpuAndNeuralEngine` and
  `.cpuOnly`. It feeds only int32 inputs, so it can't drive chunk 1,
  which takes `hidden_in`. The converter covered chunk 1, with its
  per-chunk compute plan and weight check, and with the chain's pad gate
  (1.0000000).
- **Grades.** The parity suite's 51 cases grade it A on the ANE
  (worst cosine 0.999986, mean 0.999993) and on the GPU (0.999997), and
  B on the CPU (0.999684). Every path had no rank flips and exact pads.
  Buckets reached 0.999988 or better, and the ANE gave identical output
  in two processes. The machine was busy (load about 6), so its timings
  (ANE 33.3, GPU 25.6, CPU 69.0 ms per case) are provisional.
- **Unlike agent-jev.** Chunked agent-jev, the same backbone size, graded
  D on the ANE until D39's StableSilu. The embedder recipe already
  carries StableSilu and the attention and `down_proj` input rescales
  (D20 amendment), so Qwen3-Embedding is A on the ANE as converted.

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

**Amendment (October 2026, v0.8.0): threads for onnx models.** The same
table takes `intra_op_threads` for an onnx model (D40), over the global
`onnx_intra_op_threads` and the default. `compute_units` set for an onnx
id isn't applied: like one naming a static embedder, it is logged and
listed among the overrides that named no Core ML model. Likewise,
`intra_op_threads` naming no onnx model is listed in `/health`'s
`onnx.unmatched_thread_overrides`.

**Amendment (October 2026): Core ML on Intel Macs.** The default compute
units stay per model on `x86_64-apple-darwin`; there is no
platform-wide `cpu_and_gpu` default. Measured on one Mac Pro (Xeon W-3235,
AMD GPU, macOS 26.6.2) with the parity suite:
- Core ML's CPU there isn't bucket-invariant. Its attention-score matmul
  rounds the last, partial tile of output columns differently by bucket:
  one fp16 ulp in a few scores of inputs that nearly fill their bucket.
  It's deterministic, and it costs no accuracy (`tools/repro_cpu_matmul_tail.py`).
  nlptown-sentiment's probabilities move by up to 1.9e-4, and
  bge-small-en-v1.5's cosine by 9e-7. ms-marco-minilm-l6-v2 doesn't move.
- The GPU there was 6–13× faster on the small classifiers (nlptown 300 →
  23 ms per case, ms-marco 42 → 7.5 ms) and as accurate or better, but no
  more bucket-invariant. It moves outputs at every position, logging that
  MPSGraph doesn't support padded tensors on Intel Macs.
  gliner2.5-decide's and laya-typed-decisions' GPU drift (Δp 4.7e-3 and
  2.7e-2) exceeds the gates they pass on Apple silicon's GPU.
- The larger classifiers' CPU paths didn't finish grading in four hours
  per path (gliner2.5-decide, laya-typed-decisions, lumma-fev-0.1b,
  agent-jev; agent-jev's GPU path neither).

So switching an Intel Mac to `cpu_and_gpu` buys speed, not invariance, and
the models it would matter most for fail on the GPU there. The operator
chooses with `[models."<id>"] compute_units`, as before.
gliner2.5-decide and laya-typed-decisions are unsupported on Intel Macs,
as are laya-en and julia-1, by family.

The parity suite's CPU bucket-invariance gate is per architecture. Apple
silicon stays exact (cosine 0.999999, Δp 1e-5). On x86_64 macOS it is
cosine 0.99999 for embedders, and Δp 5e-4 for classifiers and rerankers.
The classifier value is provisional: only nlptown and ms-marco measure it,
on one Mac. The GPU gates are unchanged. Under it, nlptown's Intel CPU run
(worst Δp 7.35e-4, drift 1.9e-4) grades A. The run was graded F under the
exact gate, and the suite doesn't re-judge a stored report.

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

## D40 — An ONNX Runtime CPU backend for embeddings
Until v0.7.1, sidekick served transformer encoders only through Core ML,
so only on macOS, and only Apple Silicon releases existed. A client
developed on an Intel Mac or a Linux machine had no local way to call the
same API. A third embedding backend, `onnx`, runs the same encoders on the
CPU through ONNX Runtime (the `ort` crate), on every platform sidekick
builds for. On Apple Silicon it is also a CPU baseline to compare
Core ML's paths with. Classifiers, rerankers and GPU execution providers
are out of scope: the backend is CPU-only by design, so it never
competes with a model the GPU serves.

**Manifest.** `manifest.toml` takes `backend = "onnx"`:

```toml
id = "bge-small-en-v1.5"
backend = "onnx"
artifact = "model.onnx"            # one graph; an external-data weights file beside it is fine
tokenizer = "tokenizer.json"
dims = 384
pooling = "none"                   # none | mean | cls | last_token
max_seq_len = 512

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"  # optional
token_type_ids = "token_type_ids"  # optional; fed zeros
position_ids = "position_ids"      # optional; fed 0..n per input
output = "embedding"               # [batch, dims] with pooling = "none", else [batch, seq, hidden]
```

A graph's output is one of two kinds, decided by the model's head, not by
who exported it:
- **Token-level `last_hidden_state`**, `[batch, seq, hidden]`, pooled in
  Rust with `mean`, `cls`, or `last_token` (the last real token, new, for
  decoder embedders such as F2LLM and Qwen3-Embedding). sidekick's own
  exports and published ones share this interface, so either can replace
  the other through one pooling path. BERT-family exports take
  `token_type_ids`, fed zeros.
- **A pooled `embedding`**, `[batch, dims]`, with `pooling = "none"`, for
  a head with layers beyond pooling. EmbeddingGemma's (mean, then a
  768→3072→768 dense stack) stays in the graph, as onnx-community's export
  keeps it, so Rust gains no dense layers.

Every export is gated against the same fp32 references as the model's
Core ML build. sidekick's exports come from the converters' `--format
onnx`, which exports the model's backbone with the same head and gate
texts as its Core ML build. `--onnx-weights fp16` stores the weights in
fp16 and still computes in fp32; it halves the file but saves no memory,
since the weights are upcast when loaded. There is no int8 export (see
below). sidekick's exports of bge-small-en-v1.5 and all-MiniLM-L6-v2
match the references at 1 − cos ≈ 2e-12 (graded from Python's
onnxruntime 1.27 when the exports were written).

The inputs' element type follows what the graph declares, int64 (ONNX's
convention) or int32. The graph's batch and sequence axes are dynamic,
so there are no `buckets`: each input runs at its own length, up to
`max_seq_len`. Truncation, prefixes, `max_tokens` and `truncate` work as
for Core ML models, because the tokenizing code is shared
(`sidekick_embed::embed_input`). Normalization and Matryoshka truncation
stay in Rust, as does pooling, shared with the Core ML embedder
(`sidekick_embed::pool`). An onnx manifest refuses Core ML's keys:
`{seq}`, `buckets`, `compute_units`, `ane_weight_limit`,
`cpu_seq_limit`, `[chunking]` and `[placement]`. D32's weight limit
and D33's CPU cap are Core ML limits and don't apply.

An ONNX build keeps the model's id, so a client addresses the same model
on every platform, whichever backend serves it there. A models directory
holds one manifest per id, as now. To compare the two builds on Apple
Silicon, put them in separate models directories.

**Interface checks fail loudly** (D22's rule, applied at load). At scan,
the registry reads the graph's declared inputs and outputs from the
`.onnx` protobuf (`sidekick_core::onnx_graph`). It needs no runtime
library, and it seeks past every length-delimited field it doesn't
need, so the weights are never read. That matters because exporters
serialize the initializers before the inputs and outputs. On a 600 MB
model stored that way, the read took 155 µs, and the whole process
peaked at 2.4 MiB resident (M1 Max). A graph whose weights sit in an
external-data file reads the same way. A manifest is skipped, with the
reason in `/health`'s `skipped_models`, when:
- `[io]` names an input the graph lacks, or the graph takes an input
  `[io]` doesn't name;
- an input isn't an int32 or int64 tensor of rank 2;
- the output's rank doesn't fit `pooling`;
- the output's last dimension, when the graph fixes it, isn't `dims`.

sidekick checks external-data locations at scan time, because ONNX Runtime
before 1.24 doesn't, and Intel macOS runs 1.23.2. A crafted graph's
external data could otherwise make the runtime read any file. Every
`location` an initializer in the main graph names must be relative,
without `..`, exist, and resolve (symlinks included) inside the graph's
directory, or the manifest is skipped with the reason. A model directory
installed as one symlink is fine, but an external-data file symlinked out
of it isn't. Subgraphs' (If, Loop) initializers aren't walked; no
exporter puts weights there, and none of sidekick's graphs has them.

When ONNX Runtime itself won't load, or the build has no `onnx` feature,
every `onnx` manifest is skipped with that reason
(`ModelRegistry::skip_backend`), and the rest of the daemon serves as
usual.

**Threads.** `intra_op_threads` bounds the CPU threads a model's
inference uses, so a CPU encoder doesn't starve a larger model's host
threads. It is the operator's choice, like D38's compute units, not the
converter's:

```toml
onnx_intra_op_threads = 8          # every onnx model (default: computed, below)

[models."bge-small-en-v1.5"]
intra_op_threads = 2
```

- The default is `min(max(8, the model's default), the machine's fast
  physical cores)`:
  - **fast physical cores:** one per physical core of the machine's
    fastest tier, so the threads never share a core (SMT) or wait on an
    efficiency core. On macOS that's `hw.perflevel0.physicalcpu` (Apple
    silicon's P-cores), or `hw.physicalcpu` where there are no performance
    levels. On Linux it's a hybrid Intel CPU's P-cores
    (`/sys/devices/cpu_core/cpus`), else every physical core: unique
    (package, core) pairs in sysfs topology, so SMT siblings count once.
    If neither can be read, it's 4, or one less than the logical CPUs if
    that's fewer. It's never the logical CPU count.
  - **8** (`MAX_DEFAULT_INTRA_OP_THREADS`) caps it: past 8 threads,
    latency stopped improving for every model but the largest.
  - **the model's default:** `[onnx] default_intra_op_threads` in its
    manifest can raise the cap, for a model that gains from more cores,
    never lower it. A value under 8 is accepted and has no effect, and
    nothing exceeds the fast cores. A base M1, with 4 P-cores, gets 4
    whatever the manifest says. 0 is a scan error.

  The operator's setting always wins: `[models."<id>"] intra_op_threads`
  first, then `onnx_intra_op_threads` for every model, then the default.
  `/health` reports each onnx model's `intra_op_threads`, its `source`
  (`config (model)`, `config` or `default`) and its `model_default`. It
  also reports `default_intra_op_threads`, the machine's capped value,
  and what it was `detected` from (`{source, cores}`).
  `inter_op_threads` is 1: the graphs are sequential.
- Measured (#64), on four machines, with ONNX Runtime 1.28.3 (1.23.2 on
  the Intel Mac):
  - **M1 Max** (8 P-cores, 2 E-cores), two ways. From Python's
    onnxruntime at 256 tokens (#64's latency runs), 8 threads beat ONNX
    Runtime's own all-cores default by 16–27%, scaling almost linearly
    from 1 to 8. Through sidekickd's HTTP API at about 290 tokens (the
    method of the table below; docs/MODELS.md has its M1 Max rows), 10
    threads, one per core including the efficiency cores, were about
    twice as slow as 8 (Qwen3-0.6B 638 ms against 358).
  - **The three x86-64 machines**, median ms for one 290-token input (best
    per row in bold; the Xeon ran under a background load of 4–16, so its
    row is indicative):

    | machine | model | 4 | 8 | 12 | 16 | 24 | 32 |
    |---|---|---|---|---|---|---|---|
    | Intel Xeon W-3235 (12 cores, SMT 2×) | bge-small | 37 | **24** | 30 | 28 | 69 | – |
    | Intel Xeon W-3235 (12 cores, SMT 2×) | gte-modernbert | 161 | **98** | 119 | 99 | 241 | – |
    | Intel Xeon W-3235 (12 cores, SMT 2×) | EmbeddingGemma | 161 | **97** | 112 | 108 | 187 | – |
    | Intel Xeon W-3235 (12 cores, SMT 2×) | Qwen3-0.6B | 572 | 337 | 385 | **316** | 451 | – |
    | AMD Ryzen 7 5800X (8 cores, SMT 2×) | bge-small | 33 | **20** | 24 | 23 | – | – |
    | AMD Ryzen 7 5800X (8 cores, SMT 2×) | gte-modernbert | 143 | **85** | 101 | 87 | – | – |
    | AMD Ryzen 7 5800X (8 cores, SMT 2×) | EmbeddingGemma | 140 | **83** | 100 | 86 | – | – |
    | AMD Ryzen 7 5800X (8 cores, SMT 2×) | Qwen3-0.6B | 501 | **277** | 370 | 298 | – | – |
    | AMD Threadripper 3970X (32 cores, SMT off) | bge-small | **31** | 35 | 32 | 33 | 34 | 41 |
    | AMD Threadripper 3970X (32 cores, SMT off) | gte-modernbert | 133 | 117 | 103 | 104 | **103** | 118 |
    | AMD Threadripper 3970X (32 cores, SMT off) | EmbeddingGemma | 129 | 113 | 120 | 112 | **106** | 137 |
    | AMD Threadripper 3970X (32 cores, SMT off) | Qwen3-0.6B | 460 | 335 | 303 | **277** | 294 | 339 |

  At about 290 tokens, 8 threads is within 21% of each machine's best for
  every model (the widest gap: the Threadripper's Qwen3-0.6B, 335 ms
  against 277), and is the fastest on the M1 Max and the 5800X. More
  threads help short inputs least: on the Threadripper, 32 threads ran a
  16-token bge-small input in 21 ms against 5.7 ms on 8, and is at best
  level with 8 at 290 tokens. More threads than physical cores (SMT
  siblings) never clearly helped: the one row where they came out ahead,
  the Xeon's Qwen3-0.6B at 16 threads (316 ms against 337 at 8), is
  within that loaded machine's noise. Only Qwen3-0.6B kept gaining past
  8: on the Threadripper, 277 ms at 16 against 335 at 8. That's why a
  model's manifest can raise its default (Qwen3's sets 16). On the Xeon
  the rule then gives Qwen3 its 12 physical cores, measured at 385 ms.
- Each model has one session. Concurrent requests to it take turns,
  since ONNX Runtime's `Run` holds the session (in ort 2.0.0-rc.13), and
  each uses all of the model's threads, so a model never uses more than
  its `intra_op_threads`.
- Zero threads is a config error: ONNX Runtime reads it as every core.
- `/health` reports each onnx model's effective threads, the default,
  and any thread setting that named no onnx model, beside the runtime.

**Batching.** Each input runs on its own, at its own length, unpadded.
The invariances that gives were measured with `examples/onnx_invariance`
on ONNX Runtime 1.28.3's CPU. The machine was an M1 Max. The models were
fp32 exports of bge-small-en-v1.5, all-MiniLM-L6-v2, F2LLM-v2-160M,
EmbeddingGemma-300m and Qwen3-Embedding-0.6B, on the parity corpus's
50–51 cases of 1 to 512 tokens:
- **Re-runs: bit-exact.** The same input in the same session gave the
  same bits, in every case, for all five.
- **Thread count: bit-exact for four of five.** Every vector was
  bit-identical on 1, 2, 4 and 8 intra-op threads for every model but
  EmbeddingGemma. On 4 and 8 threads, 4 of its 51 vectors changed, by at
  most 1.43e-7 per component, at cosine 1.000000000. So an embedding is
  bit-exact for a given `intra_op_threads`, and changing the setting
  moves it only within rounding.
- **Padding and batching: not bit-exact, within 2.6e-7.** The same input
  padded by 1, 7 or 64 tokens, or to `max_seq_len`, or run in a batch of
  2 or 8 padded to the longest, was bit-identical in only 3–41 of the
  cases. The largest component difference was 2.6e-7 (1–2 fp32 ulp), and
  the cosine was 1.000000000 to nine digits. MLAS orders its sums by
  sequence length.

So the CPU path claims what running unpadded guarantees: for a given
thread setting, an embedding is a function of its own text, bit for
bit, whatever else the request holds. Batching would trade that for
throughput, within 2.6e-7. It stays off, and if it is ever enabled it
applies to fp32 graphs only.

Graded natively on the same machine (4 threads, every corpus case), all
six of sidekick's exports grade A against fp32 torch, with token ids
matching the reference pipeline's and re-runs bit-identical:

| model | 1 − cos, worst | median per input |
|---|---|---|
| all-MiniLM-L6-v2 | 1.97e-12 | 2.6 ms |
| bge-small-en-v1.5 | 1.91e-12 | 5.8 ms |
| F2LLM-v2-160M | 4.52e-12 | 14.5 ms |
| gte-modernbert-base | 1.94e-11 | 23.1 ms |
| EmbeddingGemma-300m | 2.86e-12 | 24.9 ms |
| Qwen3-Embedding-0.6B | 1.71e-11 | 84.6 ms |

The timings aren't a benchmark; #64 measures latency. Qwen3's export with
fp16 weights also grades A (2.14e-11). An int8 graph quantized from
sidekick's fp32 export with `quantize_dynamic` grades D: worst cosine
0.406 on the M1 Max (8,355 rank flips, similarity drift 0.31) and 0.404
on the Xeon W-3235 (ONNX Runtime 1.23.2). onnx-community's published
int8 export also grades D (docs/MODELS.md). `--format onnx` writes no
int8 graph.

The same six exports also grade A on the Xeon W-3235 (macOS, ONNX
Runtime 1.23.2), the Ryzen 7 5800X and the Threadripper 3970X (Linux,
1.28.3), and the two Linux machines produce bit-identical vectors. For
comparison, Core ML's fp16 CPU path grades B on gte-modernbert,
EmbeddingGemma and F2LLM. Among published exports, onnx-community's
EmbeddingGemma fp32 diverges past about 257 tokens (worst cosine 0.9936
at 512): it applies the configuration's 512-token sliding window where
transformers halves it for this bidirectional model. sidekick's export
matches transformers at every length.

**Quantized graphs always run alone.** ONNX Runtime's CPU quantizes a
dynamic int8 MatMul's activations at run time with one scale for the
whole batch, padding included. Its default optimizations fuse a
weight-only DequantizeLinear + MatMul into that same kernel. So an int8
graph's output depends on its batch-mates and its pad values. Measured
from Python's onnxruntime 1.27 while sidekick's exports were written, on
F2LLM-v2-160M (Qwen3), 51 cases:
- fp32 graded A, exact;
- dynamic int8 (`quantize_dynamic`, how most published int8 exports are
  made) graded F, worst cosine 0.875, batch invariance 0.925 and pad
  invariance 0.918;
- weight-only int8 graded F on the batch gate (0.998 worst, 0.9998 batch).

Qwen3's activation outliers make the per-tensor scale coarse even for one
input. The registry marks a graph quantized when its main graph has a
quantizing operator (DynamicQuantizeLinear, QuantizeLinear,
DequantizeLinear, MatMulInteger, ConvInteger, MatMulNBits,
DynamicQuantizeMatMul or any QLinear*). Such a model always runs one
input at a time, unpadded, and `/health` reports it as `quantized`. An
int8 artifact is graded like any other, never assumed equivalent, and is
no model's default. The published int8 exports people reach for,
Qwen3-Embedding-0.6B's included, use this scheme.

A published decoder export may also expect a KV cache:
onnx-community's Qwen3-Embedding-0.6B int8 graph takes 59 inputs
(`past_key_values.*` among them) and returns 57 outputs. It doesn't fit
`[io]`, so the registry skips it with the input it can't feed. sidekick's
own export of the model takes only ids, mask and positions.

**Linking.** ort's `load-dynamic`: sidekick links no ONNX Runtime at build
time. Each target ships one archive with its runtime bundled; there's no
separate slim build.
- The runtime loads only when the scan finds an `onnx` manifest, and then
  at startup. A missing runtime skips the onnx models with its reason in
  `/health` right away, and everything else serves. With no onnx model,
  it's never loaded, and `/health` reports
  `"runtime": {"state": "not loaded", "reason": "no onnx models"}`.
- It looks for the library at the daemon config's `onnx_runtime`, else
  `ORT_DYLIB_PATH`, else beside the `sidekickd` binary
  (`libonnxruntime.dylib`, `.so`), where release archives ship it. There
  is no search of Homebrew, `/usr/local/lib` or the system library path:
  naming the file is the opt-in to an existing install. `onnx_runtime` is
  the way to use one under launchd, which doesn't pass a shell's
  environment.
- The ONNX Runtime version is pinned per target: 1.28.3, and 1.23.2 on
  Intel macOS (`x86_64-apple-darwin`). The Intel archive declares macOS
  13.4, ONNX Runtime 1.23.2's own minimum, and is built and tested on
  macOS 26; the Linux archives are built on Ubuntu 22.04 and need glibc
  2.35 or later. Microsoft's last osx-x86_64 build is 1.23.2
  (`onnxruntime-osx-x86_64-1.23.2.tgz`). From 1.24 it publishes arm64
  only, as do PyPI and ort's own binaries. sidekick builds ort 2.0.0-rc.13
  with `api-23`, so one binary accepts any runtime from 1.23 on. The whole
  workspace builds at that API level, and passes its tests on 1.28.3. The
  Intel archive's sha256 is `d10359e1…485a6`, matched by a download on an
  Intel Mac. Its `libonnxruntime.1.23.2.dylib` (sha256 `8c9c78de…4c054`)
  is x86_64 Mach-O linking only system frameworks and libc++, so it is
  self-contained in a release archive like the arm64 one. Intel results
  are graded on Intel, and vectors aren't bit-identical across
  architectures or runtime versions anyway. Building 1.28.3 from source
  for Intel macOS, as an optional download, is a follow-up.
- `sidekickd --version` names the pinned version for its target.
  `/health` names the version actually loaded, and where it came from
  (`config`, `ORT_DYLIB_PATH` or `beside sidekickd`) and its file name. It doesn't give
  the full path, just as a skipped manifest is named relative to the
  models directory. `tools/fetch_onnxruntime.sh` holds each pinned
  archive's sha256, from the GitHub release's asset digest; the Apple
  silicon one, `c436bd9f…0a1d32`, matches a local download.
- A loaded library of another 1.x version than the pin is used, with a
  warning in the log and in `/health`, since `onnx_runtime` and
  `ORT_DYLIB_PATH` exist to allow it. One older than 1.23, which this build of ort can't drive, is
  refused with the reason. The library's version is read through `OrtGetApiBase` before
  ort loads it, so a wrong file fails with a reason, not a panic.
- Everything is behind an `onnx` cargo feature. A build without it is
  unchanged, and skips `onnx` manifests as unsupported in that build, as
  a build without `coreml` skips classifiers.
- `libsidekick.dylib` (D18) is built without `onnx` in this release.

**Reporting.** `/v1/models` gains `backend` for every embedder (`coreml`,
`static` or `onnx`). An `onnx` model's `compute_units`, and its
`sidekick-compute-units` header, are `cpu`. It reports no `placement`
(D35) and no `seq_cap`. It sends no `sidekick-buckets`: one graph
serves every length, so there is no compiled program per bucket to name,
as for a static model.

**The parity suite** grades an `onnx` model on one path, `onnx-cpu`,
against the same fp32 references and gates as its Core ML build. Its
bucket-invariance gate is the CPU's unless `expectations.toml` sets
`onnx-cpu`. A parity build with the `onnx` feature runs the path natively:
a worker runs the product path (prefix, tokenizer, truncation, ONNX
Runtime, pooling, normalization) and writes the suite's own worker result.
Each input runs alone and unpadded, as the daemon runs it, so there are no
bucket or pad comparisons. If batching is ever enabled, the bucket check
becomes a length check, the same input run padded to longer lengths. A
copy of the references and a parity build grade any machine, an Intel Mac
included, with no Python or torch there. A build with the `onnx` feature
alone runs on any OS, Linux included. It grades onnx models on `onnx-cpu`,
which is then its only path, lists Core ML models as skipped, and names
the chip by `/proc/cpuinfo`'s model name, as floors (below) do. CI grades
the tiny test graph that way on x86-64 and arm64 Linux, against a
reference over the whole corpus that is built with numpy and committed
beside it. A copied parity binary finds its fixtures (corpora and
expectations) at `--fixtures DIR`, else `$SIDEKICK_FIXTURES`, else
`./fixtures`, else the checkout it was built from.

Floors are the suite's per-chip regression thresholds, in
`fixtures/parity/expectations.toml`: each is a measured value less a
margin, so a later run that does worse fails, and on a chip with none the
suite reports accuracy without judging it. `onnx-cpu` floors are recorded
for the M1 Max (ONNX Runtime 1.28.3) and the Xeon W-3235 (1.23.2). An int8
build fails them by design.

`tools/onnx_parity.py`, the Python grader written with sidekick's exports,
writes the same result from Python's onnxruntime, and `parity --external
FILE` grades it beside the native run as `onnx-cpu:external`, as a
cross-check. The file is a worker result in the suite's JSON shape:
`model`, `path: "onnx-cpu"`, `cases`, `repeat_bitwise`, `load_ms`. Each
case gives its `id`, `ids_match`, its token count as `bucket`, `vector`,
`finite`, `model_only`, and `bucket_invariance` and `pad_invariance` as
`{n: 0}` when not measured, and `ms`. It must hold every reference case in
reference order, each with a vector of the model's width. On ONNX Runtime
1.28.3 (M1 Max) the native path grades sidekick's exports of
bge-small-en-v1.5 and all-MiniLM-L6-v2 A, at 1 − cos 1.91e-12 and 1.97e-12
worst against fp32 torch. Token ids match the reference pipeline's, and
re-runs are bit-identical.

**Amendment (October 2026): ONNX Runtime in its own crate, and the
backends a build serves.** The ONNX Runtime code moved out of
`sidekick-embed` into `sidekick-onnx`, beside `sidekick-coreml`, so the
two backends are laid out the same way:
- `sidekick-onnx` finds and loads `libonnxruntime`, checks its version,
  runs a graph in a session (`OnnxModel`, fed the integer types its
  inputs declare), and holds the thread rule and core detection. `ort`
  and `libloading` are its dependencies alone, behind its `runtime`
  feature. Without that feature the crate is the pinned version and the
  thread rule only, which the daemon reports in every build, so
  `sidekick-embed` depends on it unconditionally and its `onnx` feature
  turns `runtime` on. CI checks that a build without `onnx` has no `ort`
  in its tree, for `sidekickd` and for `libsidekick.dylib`.
- `OnnxEmbedder` stays in `sidekick-embed`, as `CoremlEmbedder` does: it
  tokenizes and pools with the code the Core ML embedder shares, so
  moving it would mean moving that code too, or a dependency cycle.
- The scan-time checks (interface, quantization, external data) stay in
  `sidekick_core::onnx_graph`: the registry runs them without a runtime,
  in builds without `onnx` too.

`sidekick_embed::onnx` re-exports what moved, so the Rust API, the C ABI,
the config keys and `/health`'s `onnx` object are unchanged. A build now
says which model backends it serves: `sidekickd --version` ends with
`(backends: coreml, onnx, static)`, or the subset compiled in, and
`/health` has the same list as a top-level `backends` array. `static` is
always there; `coreml` needs the `coreml` feature on macOS, and `onnx`
the `onnx` feature.

**Amendment (October 2026): an ONNX build's own length limit, and a
long-input corpus.** Qwen3-Embedding-0.6B takes 32k tokens, but both of
its builds stopped at 512. That was the converter's default bucket list,
which the ONNX export inherited as its `max_seq_len`; nobody had chosen it
for the ONNX build, and no gate tested past it.
- **Its own setting.** An `onnx.toml` beside the committed Core ML
  manifest may set a top-level `max_seq_len` for the ONNX build. The export
  checks it against the checkpoint as it checks the Core ML one (at most
  the position embeddings, less any offset), cuts its gate texts at it,
  and writes it into the installed manifest with the comment above it.
  No other top-level key is accepted. The Core ML bucket list doesn't
  change: a bucket is a program with its own copy of the weights (about
  1.1 GiB for this model in fp16), and attention at 4,096 tokens would
  hold about 0.5 GiB of fp16 scores per layer. Core ML buckets past 512
  stay out of scope until there's demand.
- **Qwen3-Embedding-0.6B's ONNX build takes 4,096.** Its graph's
  sequence axis is dynamic and each input runs alone, unpadded, so a
  short input costs what it did at 512; only a long one pays.
- **A long-input corpus,** `fixtures/parity/corpus_long.toml`: about 600,
  1,500 (a document and a query), 2,400 and 4,096 tokens of ordinary text
  in varied registers, and one input 188 tokens past `max_seq_len`, which
  truncation must cut keeping its final token. It has its own hash and its
  own references (`reference-long.*`, from `tools/parity_reference.py
  --corpus long`), so adding it changed neither `corpus.toml`'s hash nor
  any model's existing references. A reference holds only the cases that
  fit the model's `max_seq_len`. The parity suite grades it for a model
  whose `max_seq_len` reaches its shortest case, on the same paths as the
  main corpus, and reports it as the model's `long` section, absent for
  every other model, so their reports are unchanged. Its gate failures are
  the model's; it has no floors yet. A missing long reference fails a
  model named with `--model` whose `max_seq_len` is past the main corpus's
  longest case, as a missing main reference does, since nothing else
  grades its long inputs; it is a warning where the main corpus already
  reaches the limit (EmbeddingGemma-2's 2,048, which its over-length cases
  fill) or the model wasn't named. A run with only `--external` results
  says the long corpus went ungraded. `tools/models_table.py` turns it
  into a note on the model's row, whose grade stays the main corpus's.
- **An ONNX build with a longer limit needs its own main-corpus
  reference.** The suite checks that the reference's `max_seq_len` equals
  the build's, and at 4,096 corpus.toml's over-length cases (700 tokens)
  are no longer truncated. `tools/parity_reference.py` now reads an onnx
  install directly, taking corpus.toml's bucket-relative lengths from the
  committed Core ML manifest, so both builds are graded on the same texts.

- **The arena gives back a long run's memory.** ONNX Runtime's CPU arena
  keeps every block a run grows it by, so one long input left the process
  at its peak for as long as the model stayed loaded. `OnnxModel` now runs
  an input of 1,024 tokens or more (`SHRINK_FROM_TOKENS`, rows × length)
  with the run option `memory.enable_memory_arena_shrinkage = "cpu:0"`,
  which frees the blocks that run grew the arena by when it ends; shorter
  runs are unchanged. It applies to every onnx model; with every other
  embedder's limit at 512 or below, it changes nothing for them.
  Shrinking after every run, or no arena at all, cost short inputs nothing
  measurable either, but neither returned memory any better, and both
  change the path every short input takes. On macOS the system allocator
  keeps the freed pages and returns them as later allocations run, not
  while the process is idle; `malloc_zone_pressure_relief` released
  nothing on macOS 27.0, so sidekick doesn't call it.

Measured on an M1 Max, macOS 27.0, ONNX Runtime 1.28.3:
- **Accuracy** (the parity suite's native worker, 4 threads as for the
  other M1 Max rows; 8 gives bit-identical vectors): A on both corpora.
  Worst 1 − cos 1.7e-11 on corpus.toml, unchanged from the 512-token
  build, and 7.6e-12 at 600 tokens to 1.2e-11 at 4,096 on the long
  corpus, the truncated input included, with token ids equal to the
  reference pipeline's.
- **Time** (8 threads, the machine's default): a median 0.85–0.90 s per
  input at 512 tokens and 12.9–14.2 s at 4,096 on sidekick's own path
  (`examples/onnx_memory`, 8 runs, load average 4–11), and 6.8–7.1 s at
  2,400 from Python (`tools/bench_onnx.py`, onnxruntime 1.27, 2 runs on a
  busier machine, load average 5.6–7.9 rising to about 27 during each). The
  export's attention is a MatMul, a Softmax and a MatMul per layer, which
  materializes the fp32 scores, so time grows faster than length. A fused
  attention kernel would cut time and memory; that is a follow-up.
- **Memory** (`examples/onnx_memory`, the footprint above the pre-load
  one): +1,777 MB loaded, +1,825 MB after a 512-token input, +4,152 MB
  after a 4,096-token one (+2,633 MB at 2,400, `tools/bench_memory.py`).
  Before the change, 69 more inputs of up to 512 tokens and 30 s idle
  left it at +4,146–4,156 MB (six runs). After it, the same inputs
  brought it back to +1,838–1,883 MB in all six runs; a seventh, of a
  variant that also called `malloc_zone_pressure_relief`, stayed at the
  peak. Short inputs' latency was the
  same with and without the change (16 / 256 / 512 tokens: 37–40 /
  401–423 / 851–896 ms).
- **Timeouts.** A request's inputs run one after another under one
  `request_timeout_secs` (default 60), so four 4,096-token inputs in one
  request can pass it, and five do. The 504 abandons the wait, not the
  work: the remaining inputs still run, each holding the model's session
  for 13–17 s, and other requests to the model wait behind them.
  docs/MODELS.md advises one or two long inputs per request, or a higher
  timeout.
- **Vectors change for long text.** Inputs past 512 tokens were cut at
  512 by this build up to 0.12.0 and now embed whole, and the Core ML
  build of the same id still stops at 512, so the two builds' vectors of
  long text differ. Long documents need re-embedding, and the two builds'
  long-text vectors don't belong in one index. A client that wants the old
  vectors sends `max_tokens: 512` to `/v2/embed`; `truncate` on
  `/v1/embeddings` and `/v2/embed` picks the end that is cut, or refuses
  an over-long input.

## D41 — The nli format: NLI cross-encoders as zero-shot classifiers
A natural-language-inference cross-encoder reads a (premise, hypothesis)
pair and says whether the premise entails, contradicts or is neutral
toward the hypothesis. Hugging Face's `zero-shot-classification` pipeline
turns one into a zero-shot classifier: each candidate label becomes a
hypothesis ("This example is {label}."), and the labels are ranked by
entailment. Small MNLI-trained models (`cross-encoder/nli-deberta-v3-*`,
`nli-MiniLM2-L6-H768`, `roberta-large-mnli`, and the newer zeroshot-v2.0
and ModernBERT ones) are the established tool for questions a host asks
of short texts at high volume, such as whether a step in an agent's
transcript shows the agent surprised or mistaken. sidekick's zero-shot
formats so far (D28, D30, D34, D36) each serve one model family's own
layout; none serves these.

**Decision.** A fifth zero-shot format, `format = "nli"`, reproduces the
pipeline's scores, so a client written against `pipeline(...)` gets the
same numbers from `/v1/classify`:
- The manifest's `[classify.nli]` names the `entailment` and
  `contradiction` output indices, `n_classes`, and a default
  `hypothesis_template` (the pipeline's, `This example is {}.`). The
  indices are the pipeline's own: entailment is the first `label2id`
  label starting with "entail"; contradiction is index 0, or the last
  index when entailment is 0. That is the contradiction or not_entailment
  output of every MNLI and zeroshot model checked.
- For each label, the tokenizer's pair post-processor lays out (text,
  template with the label), as the pipeline's fast tokenizer does, so
  BERT's `[CLS] a [SEP] b [SEP]` and RoBERTa's `<s> a </s></s> b </s>`
  both come out right. Segment ids are fed exactly when `[classify.io]`
  names `token_type_ids`. The tokenizer can't decide that: DeBERTa-v3's
  gives pairs segment ids that its model, without token-type embeddings,
  never reads. The converter checks the naming against the checkpoint's
  config instead.
- Only the text is truncated, on `truncation_side`, until the pair fits:
  the pipeline's `truncation="only_first"`, which always applies. `left`
  keeps the end of a long input, which is what tool output and stack
  traces need. The hypothesis is never cut; one too long for any text is
  a 400, where the pipeline would retry untruncated and overflow the
  model's positions.
- Scores: with several labels, the softmax across labels of their
  entailment logits. With `multi_label: true`, or a single label (the
  pipeline's rule), each label's softmax over [contradiction,
  entailment], taking entailment. The classifier returns the entailment
  logit, or entailment − contradiction, whose sigmoid is exactly that
  probability, so the server's existing activations produce the scores
  and `use_activation: false` returns those values. `probs` keeps
  `candidate_labels` order, as on every format.
- A request may send its own `hypothesis_template`, as the pipeline
  takes one per call. It must hold exactly one `{}` and no other brace,
  so a plain substitution and Python's `str.format` can't disagree.

**Each pair is its own prediction**, in the smallest bucket that fits it,
as rerank pairs are (D29). A static `[1, S]` artifact per bucket keeps
the Core ML interface every other classifier has: `input_ids`,
`attention_mask`, optional `token_type_ids`, and `logits [1,
n_classes]`, checked at load against `n_classes`. Batching a request's
labels into a `[K, S]` artifact was the alternative. It would fix K per
bucket, pad every request to it, and multiply the variants to convert and
check; the per-pair cost it saves is launch overhead, small beside the
encoder. A request therefore costs k predictions per input, which
`max_labels` and `max_batch` bound. `usage.prompt_tokens` counts every
pair's tokens, and `sidekick-buckets` reports each input's largest
bucket. `Prepared` carries the pairs (`Prepared::nli`).

**Compatibility.** A daemon or library before this release can't parse
`format = "nli"` and skips the manifest with a warning (D28). Nothing else
changes for existing formats. `multi_label` and `hypothesis_template` are
400s on formats that don't take them.

**Validation.** Platform-neutral tests check the pair layout for BERT- and
RoBERTa-style tokenizers, segment ids only when named, truncation on both
sides, the template rule, and the scores against hand-computed values.
Two tiny Core ML cross-encoders (`tests/fixtures/tiny-nli`: 3-way with
segment ids, 2-way without) match their fp32 torch logits in every
bucket, and their activated values match the pipeline's own
`postprocess`, run in Python on those logits, with and without
`multi_label`. The parity suite grades nli models with the classifier
grader: a reference carries each label's pair and the per-label values
above, so the existing softmax and multi-label grading apply unchanged.
The converted models' grades are recorded in docs/MODELS.md as they
land.

**Amendment (October 2026): the classes themselves.** The scores above
drop neutral, so they can't tell a contradicted label from an
unmentioned one. D47 adds an opt-in `nli_probs`, each pair's softmax
over the model's own classes, named by a new optional `[classify.nli]
classes` (sidekickd 0.15.0 or later). The scores and every default are
unchanged.

## D42 — AVX-512 for ONNX Runtime on Intel Macs
ONNX Runtime's CPU kernels (MLAS) are chosen once per process, on the
first thread that needs them. The AVX-512 kernels are taken only when the
CPU reports AVX-512F and the XCR0 register says the operating system saves
AVX-512 state. Linux and Windows set those XCR0 bits for every thread.
macOS sets them lazily, per thread, the first time that thread executes an
AVX-512 instruction. Until then XCR0 reads 0x7 (x87, SSE and AVX state
only), even on a CPU that has AVX-512 and an OS that supports it
(`hw.optional.avx512f` = 1). On a Xeon W-3235 Mac Pro, XCR0 read 0x7 at
process start and 0xe7 after one `vpxord zmm` instruction. So on every
Intel Mac with AVX-512 (Mac Pro 2019, iMac Pro), ONNX Runtime ran its
AVX2/FMA3 kernels. The check is the same in 1.23.2 (sidekick's Intel
runtime), 1.28.3 and ONNX Runtime's main branch, so a newer or
source-built runtime wouldn't change it.

**Decision.** On Intel macOS, when `hw.optional.avx512f` reports AVX-512F,
sidekick executes one AVX-512 instruction on the thread that loads ONNX
Runtime and on each thread that creates a session, before ONNX Runtime
reads XCR0 there. ONNX Runtime then selects its AVX-512F kernels (the
fp32 GEMM, convolution, exp and reduction kernels; the int8 AVX-512 Core
and VNNI kernels are selected too, though sidekick ships no int8 model).
The instruction zeroes `zmm0`, followed by `vzeroupper`; no other state
changes. `SIDEKICK_ONNX_AVX512=0` in the environment turns this off, for
comparison or if a machine misbehaves. It does nothing on Apple silicon,
on Linux, or on an Intel Mac without AVX-512. BF16 and AMX aren't
affected: the CPUs Apple shipped with AVX-512 have neither, and ONNX
Runtime doesn't run fp32 models in bf16 on x86 anyway.

**Verification.** On the Xeon W-3235 (12 cores, macOS 26.6.2, ONNX Runtime
1.23.2):
- A profile of a running daemon shows `MlasGemmFloatKernelFma3` without
  the change and `MlasGemmFloatKernelAvx512F` with it.
- A test reads XCR0 on a fresh thread after the instruction and checks
  that the AVX-512 bits are set exactly when sidekick enables them; on
  this machine they are.
- Latency through `/v1/embeddings` at about 290 tokens, one request at a
  time, with each thread count run in both modes back to back on an
  otherwise idle machine. Without the AVX-512 kernels, each model took
  this many times as long as with them:

  | threads | bge-small | gte-modernbert | embeddinggemma-300m | Qwen3-0.6B |
  |---|---|---|---|---|
  | 1 | 1.51 | 1.57 | 1.60 | 1.39 |
  | 4 | 1.37 | 1.46 | 1.49 | 1.32 |
  | 8 (the default) | 1.17 | 1.28 | 1.29 | 1.19 |
  | 12 | 1.23 | 1.26 | 1.73 | 1.63 |

  At 16 tokens the gain is smaller: 2–11% at 8 threads, up to 35% on one
  thread. AVX-512 lowers this Xeon's clock, but no setting ran slower. A
  full sweep with the AVX-512 kernels (docs/MODELS.md, threads table)
  keeps 8 threads as the fastest setting, so the thread default (D40)
  stands.
- Accuracy: every exported embedder, EmbeddingGemma 2 included, grades A
  on onnx-cpu with the AVX-512 kernels (worst 1 − cos ≤ 2e-11), as it did
  with the AVX2 ones. The two kernel sets round differently, so vectors
  from an AVX-512 Mac and an AVX2 machine agree to that level, not bit
  for bit.

## D43 — Over-length /v1/embeddings input is refused by default
Through 0.12, `/v1/embeddings` truncated an input longer than the model's
limit, keeping its start, and returned a normal vector with nothing in
the response to say the text was cut. No decision chose that: it was the
default of the embedders' limits (`EmbedLimits::default()`, `END`) that
the route passed through. A client that embeds long documents gets
vectors for their openings and can't tell which. Issue #111 records one
consumer's measurement: embedding a 2,355-document corpus through
Qwen3-Embedding-0.6B at 512 tokens (its ONNX build now takes 4,096,
D40's length-limit amendment), 36% of the documents were over the limit,
and their vectors agreed with full-length ones only down to cosine 0.855.
Every gate graded A: their references truncate the same way, and no case
went past about 520 tokens.

That is the harm D10 already refuses for the output: a `dimensions`
value a model wasn't trained to truncate to is a 400, not a best-effort
truncation. Truncating the input silently is the same harm, and D22's
rule is the same: a wrong answer that looks right is worse than a 400.
The APIs clients compare sidekick with refuse too: OpenAI's
`/v1/embeddings` is a 400 for an over-length input, and Hugging Face Text
Embeddings Inference returns an error unless the request sends
`truncate: true`. Within sidekick, `/v1/rerank` and text-classification
models on `/v1/classify` are 400s unless the request sends
`truncate_prompt_tokens`. Other routes and formats truncate by design and
are unchanged: laya, gliner2, fev and nli classifiers cut the text
(docs/design/classify.md), and `/v2/rerank` truncates documents as
Cohere's contract does.

**Decision.** On `/v1/embeddings`, a request without `truncate` (or with
`null`) is `NONE`. An over-length input is a 400 `invalid_request_error`
naming the input's index in the batch, its token count, the model and
its limit, and saying how to truncate instead:

```
input 1: 600 tokens, over model `bge-small-en-v1.5`'s limit of 512 tokens, and truncate is NONE (send truncate: END or START to truncate it)
```

The whole request is refused before any inference runs: each backend
prepares every input before it runs any. A client that wants truncation
sends `truncate: "END"` (or TEI's `true`), which gives exactly the
vectors the old default did, or `"START"` to keep the end of the text.
`truncate` was added in 0.12.1 with the old default, so clients could
opt in before the default changed in 0.13.0.

**What still truncates by default.** The library's default,
`EmbedLimits::default()`, stays `END`; only the `/v1/embeddings` route
changed.
- `/v2/embed` follows Cohere's v2 contract, whose `truncate` defaults to
  `END`. Its clients expect that, so it is unchanged.
- `libsidekick.dylib`'s `sk_embed` has no `truncate` parameter and always
  truncates (`END`), as it always has. `sk_model_info` reports
  `max_seq_len` for hosts that check.
- The Rust library API, `Embedder::embed`, has no limits parameter and
  truncates (`END`); `embed_with` takes explicit limits.
- The parity suite embeds with the library default (`END`), as its
  references were made.

So the daemon and the dylib now differ for an over-length input. A host
that follows INTEGRATING.md's probe order (the daemon, else the dylib)
should send `truncate: "END"` to the daemon to get the same vectors on
both paths, or handle both behaviours.

**A tokenizer's own truncation.** LFM2.5-Embedding-350M's
`tokenizer.json` truncates at 512 tokens itself, before sidekick sees the
ids, so `NONE` couldn't fire for it. `NONE` and `START` encode with a
copy of the tokenizer with its truncation turned off and apply the lower
of its limit and the model's `max_seq_len` themselves. So under the new
default, LFM2.5 refuses an input past 512 tokens like any other model.
`END` still uses the tokenizer as shipped.

## D44 — /tokenize and /detokenize count as the embedding path does
A client deciding whether a document fits a model's limit counted tokens
with its own tokenizer or estimated them from characters, and either can
disagree with sidekick. A Hugging Face tokenizer's count usually misses
the model's prompt prefix, which sidekick applies per `input_type`, and an
estimate can be off in either direction. Since D43 refuses an over-length
input, the client needs sidekick's own verdict.

**Decision.** `POST /tokenize` and `POST /detokenize` serve vLLM's shapes
(`{model, prompt, add_special_tokens, return_token_strs}` →
`{count, max_model_len, tokens, token_strs?}`, and `{model, tokens}` →
`{prompt}`) for embedders and text-classification models, with one
response extension, `fits`, and two request extensions, `input_type` and
`max_tokens`. They are not mounted under `/v1/`: vLLM and SGLang serve
them at the root, and their clients call them there.

- **`fits` is the route's verdict, by the route's check.** It says
  whether the model's route takes the text whole with its default
  options. For an embedder that is `/v1/embeddings` under
  `truncate: NONE`, and for a text classifier `/v1/classify` without
  `truncate_prompt_tokens`. It is computed by the one function each route
  runs: `EmbedInput::check_whole` for every embedding backend,
  `InputBuilder::check_text` for `/v1/classify`. So it can't drift from
  the route. `count <= max_model_len` is not the same test and clients
  shouldn't use it in its place. Both routes also refuse a text by its
  size in bytes before tokenizing it: an embedder past 64 bytes per token
  of the limit, a text classifier past 16. Real text can be far over that
  ratio. A WordPiece vocabulary maps a run of more than 100 characters
  without spaces to one `[UNK]`, so ordinary Thai, Lao, Khmer or Burmese
  prose is more than 64 bytes per token. Measured on bge-small-en-v1.5,
  42,599 bytes of Thai prose counted 122 of 512 tokens, and
  `/v1/embeddings` refuses it by size. A text of spaces counts no tokens at
  all, however long. An empty text is refused by a Core ML or ONNX model
  that adds no special tokens and has no prefix, though it counts 0; a
  static model embeds it. `fits` covers all of these.
- **The count is the embedding path's.** For an embedder, the count is
  the encoding that check produces, or, when the text doesn't fit, the
  same encoding without the check (`EmbedInput::tokenize`). It includes
  the prompt prefix, the text, and the special tokens the backend adds, by
  the tokenizer without its own truncation. The static backend had its own
  copy of the `NONE` check and now shares the one above. `input_type`
  takes `/v1/embeddings`' names and `/v2/embed`'s, mapped as those routes
  map them, with their default, the document prefix.
  `add_special_tokens: false` leaves the special tokens out of `count` and
  `tokens` but not out of `fits`. The prefix stays, since it is part of
  every input the model sees. A static model's inputs carry no special
  tokens, so for one it changes nothing.
- **`max_tokens`** is `/v2/embed`'s, validated as `/v2/embed` validates
  it. With it, `fits` is `/v2/embed`'s verdict under `truncate: "NONE"`:
  the limit is `max_tokens` (when lower), and the byte bound scales with
  it. `max_model_len` stays the model's limit. Without it, `fits` is
  `/v1/embeddings`'.
- **Nothing is truncated.** A text past the limit is counted in full, so a
  client can see by how much. That includes a model whose tokenizer.json
  truncates on its own (LFM2.5-Embedding-350M's, at 512). This departs
  from issue #4, which asked for "truncation and byte-cap behavior
  identical to the embedding path". Truncating would hide the count the
  route compares with the limit, and the routes' byte bounds would refuse
  the long texts a client most needs counted. The verdict those bounds
  give is in `fits` instead. `/tokenize`'s own bound on the text is the
  request body limit, 2 MiB.
- **`max_model_len` is the limit the count is compared with:** the
  model's `max_seq_len` as `/v1/models` lists it, after any cap (D33).
  For a Core ML model that is its largest bucket, which is what issue #4
  proposed ("bucket seems right"). The one case where it differs from
  `/v1/models` is an embedder whose tokenizer.json truncates at fewer
  tokens than `max_seq_len`. `/v1/embeddings` enforces the tokenizer's
  limit, but `/v1/models` doesn't read tokenizers to list models. There,
  `max_model_len` reports the tokenizer's limit, because that is what the
  count is compared with. No shipped model has this case: LFM2.5's
  tokenizer limit and `max_seq_len` are both 512.
- **Text classifiers count as `/v1/classify` does** without
  `truncate_prompt_tokens`, by the same function, and `max_model_len` is
  their `max_seq_len`. Rerankers and the zero-shot formats (laya, gliner2,
  fev, agentjev, nli) are a 400. A reranker's input is a pair, and a
  zero-shot format lays the text out with the request's labels and
  instructions, then cuts the text to fit by design. A count of the text
  alone wouldn't be compared with any limit, so returning one would
  mislead. `/detokenize` takes any embedder or classifier, since decoding
  doesn't depend on layout. `input_type` and `max_tokens` are 400s for a
  classifier.
- **The chat model and `messages` are 400s.** Foundation Models reports
  token counts but not token ids, and counting for chat belongs with the
  Responses and Messages token-count routes. `messages`, vLLM's chat form,
  needs a chat template, which encoders don't have.
- **Cost is bounded by a byte budget.** Tokenizing takes memory in
  proportion to the text: a 2 MiB text of random alphanumerics is 1.6
  million tokens to LFM2.5-Embedding's tokenizer. At most 4 MiB of text
  is tokenized at once, across requests. A request takes one 64 KiB
  permit per 64 KiB of its text, at least one, and waits until its text
  fits. One of the largest size the 2 MiB body limit allows always fits
  alone. The wait is bounded by the request timeout and the tokenizer's
  load by the load timeout, as on the other routes. A request that times
  out waiting never runs. One whose work outlasts the timeout gets its
  504, but the work runs to the end and holds its share of the budget
  until it does, as `api::deadline` describes for every route. Token
  strings are built only when `return_token_strs` asks for them.
  Measured on an M1 Max, 24 concurrent 2 MiB requests on LFM2.5's
  tokenizer peaked at 0.45 GB physical footprint with macOS's cache of
  freed large blocks turned off (`MallocLargeCache=0`), which is the
  memory the budget bounds. With the default allocator they peaked at
  2.1 GB, against 4.1 GB with the earlier one-tokenization-per-core
  limit. The difference is libmalloc's large-block cache: freed buffers
  over about 1 MB that it keeps dirty for reuse. It grows to about 1.5 GB
  whatever the concurrency (eight 2 MiB requests one at a time reach it),
  and `malloc_zone_pressure_relief` releases none of it. An operator who
  wants the lower footprint can set `MallocLargeCache=0` in the daemon's
  environment.
- **One tokenizer per model.** `/tokenize` loads a model's tokenizer
  without its weights, into a pool dropped after the same idle time as
  models. The tokenizer is shared: a model's backend and `/tokenize` get
  the same copy through a process-wide cache (`EmbedInput::shared`,
  `InputBuilder::shared`), which holds it weakly. Concurrent first loads of
  one model wait for one load instead of each loading a copy. A model
  loaded while `/tokenize` holds its tokenizer reuses it, and the other
  way round. The cache is keyed by the model's directory and a
  fingerprint of its manifest as resolved, so a manifest that reads
  differently (other prefixes, a capped `max_seq_len`) never gets another's
  copy. `/detokenize` keeps special tokens, as vLLM's does, and an id
  outside the vocabulary is a 400.

## D45 — Named prompts, chosen by `prompt_name`
D9 gives each embedder two prompts, chosen by `input_type`, but many
checkpoints publish more. EmbeddingGemma's
`config_sentence_transformers.json` names prompts for classification,
clustering, sentence similarity, code retrieval and others. Before this, a
client that wanted EmbeddingGemma's classification prompt had to copy its
text and send it with `input_type: "none"`, and under `truncate: "START"`
that copy could be cut, since sidekick can't tell it from the text. Hugging
Face Text Embeddings Inference (TEI) already has a convention for this: its
`/embed` takes `prompt_name`, a key of the model's sentence-transformers
`prompts`, as sentence-transformers' `encode(prompt_name=…)` does.

**Decision.** Requests may name one of the model's own prompts, as TEI's
do, and the prompts are copied from the checkpoint into the manifest.

- **Request.** `/v1/embeddings`, `/v2/embed` and `/tokenize` take an
  optional `prompt_name`, read by the one parser behind all three, so
  `/tokenize`'s count and `fits` follow it. It replaces the `input_type`
  prompt. Sending it with one of sidekick's `input_type` values (`query`,
  `document`, `passage`, `none`) is a 400 on every route: each asks for a
  prompt of its own (`none` for no prompt at all), so honoring either
  field would silently drop the other. Cohere's names are different.
  Cohere's SDK always sends `input_type`, so on `/v2/embed` a
  `prompt_name` overrides `search_query`, `search_document`,
  `classification` and `clustering`. A name is the more specific request,
  and `classification` and `clustering` name no prompt of a model's own
  anyway. `/tokenize` follows the same rule, so a `/v2/embed` client can
  pass its fields through and get `/v2/embed`'s count. `/v1/embeddings`
  doesn't take Cohere's names, so there any `input_type` with a name is a
  400. A request without `prompt_name` is unchanged, and tests pin its
  vectors.
- **Names match exactly, case included.** TEI and sentence-transformers
  look the name up as a dictionary key, and EmbeddingGemma 2's
  checkpoint has both `document` and `Document`. A name the model
  doesn't have is a 400 that lists the names it has, refused from the
  manifest before the model loads. A model with none says so.
- **A named prompt is sidekick's prompt.** It goes before the text as
  an `input_type` prefix does, so `truncate: "START"` keeps it whole.
  Under `START`, a `max_tokens` that leaves no room for text after it is
  a 400 before any input is embedded. Under `END` (`/v2/embed`'s
  default), the tokenizer keeps the first `max_tokens` tokens, so such a
  `max_tokens` embeds a cut prompt and no text, with a 200. The default
  prefixes on `END` have always done the same, and this change leaves
  `END` as it was.
- **Manifest.** A `[prompts]` table maps name to text. `[prefixes]`
  stays as it was: it holds the `input_type` prompts, which sidekick
  chose. A manifest without `[prompts]` reads as before, and older
  daemons ignore the table. The committed manifests of
  embeddinggemma-300m, embeddinggemma-2, F2LLM-v2-160M,
  LFM2.5-Embedding-350M and Qwen3-Embedding-0.6B carry their
  checkpoints' `prompts`, every name, copied exactly at the revision each
  build pins. EmbeddingGemma-300m's repository is gated; its prompts
  were read from a snapshot of that revision. The other committed
  embedders' checkpoints publish none. `query` and `document` are names
  too, since the checkpoints list them and TEI accepts them.
- **The converter copies them.** Converters copy the committed manifest
  and never generate one, so `manifest.check_embedder()` stops a
  conversion whose `[prompts]` differ from the checkpoint's, printing the
  table to paste, and `python -m sidekick_convert.manifest prompts <dir>`
  prints it. A checkpoint that publishes no prompts leaves the table to
  the manifest, so prompts a model card gives only in prose can be
  written there by hand, with the source cited in a comment.
- **Disclosure.** `/v1/models` lists each embedder's `named_prompts`,
  name to text, `{}` for none, beside D9's `prompts`. `sk_model_info`
  reports the same. docs/MODELS.md has a table of every build's named
  prompts, generated from the committed manifests.
- **C ABI.** `sk_embed_named` is `sk_embed` with a prompt name in place
  of the purpose, added without changing `sk_embed` or
  `SK_ABI_VERSION`, as `sk_pool_skipped` was. The library's
  `EmbedPurpose` gains `Named(&str)`, and so it gains a lifetime
  parameter. Since that breaks code that names the type, it is also
  marked `#[non_exhaustive]` now, so a later purpose won't break
  matches on it again.
- **Instruction models.** Qwen3-Embedding and F2LLM take a free-text
  instruction per task (`Instruct: {task}\nQuery:`), not a fixed set,
  and their checkpoints name only `query` and `document`. An
  `instruction` field that fills the model's template was considered and
  left out for now. It would add a template to the manifest, a field to
  three routes and another C function, for two models, and the template
  it would fill is one line. The cost of leaving it out is truncation.
  On retrieval the instruction goes on queries only, which are short.
  On symmetric tasks (classification, clustering, similarity), MTEB puts
  it on every input, and those inputs can be long. An instruction typed
  into the text with `input_type: "none"` is text to sidekick, so
  `truncate: "START"` cuts it. Measured on EmbeddingGemma, an
  over-length input with its prompt typed into the text gave cosine 0.79
  under `START` against the same prompt sent by name. These clients send the
  whole instruction with `input_type: "none"` and truncate with `END`, or
  keep inputs within the limit (`/tokenize`'s `fits`). docs/MODELS.md and
  docs/INTEGRATING.md give each model's format, trailing space included,
  and say so. If a client needs `START` with an instruction, or more
  instruction models arrive, an `instruction` field can be added beside
  `prompt_name` without changing it.

  The same holds for the model's own document prompt. Measured on a
  900-word text, that prompt typed into the text under `none` against the
  default request, both truncated the same way: under `START`, cosine
  0.877 for EmbeddingGemma and 0.893 for LFM2.5-Embedding-350M; under
  `END`, identical vectors.

## D46 — How far each build goes toward its model's context
Most builds stop short of their model's trained context, because the
converters' default bucket list stops at 512, not because anyone chose
that for each model. Qwen3-Embedding-0.6B takes 32,768 tokens,
EmbeddingGemma-300m 2,048 and gte-modernbert-base 8,192. Since D43 an
over-length input is a 400 on `/v1/embeddings`, so a build's cap is
visible, but a short cap still costs whoever has long documents. Issue
#117 asked how far each build should go, on each backend, and what that
costs. A consumer's report there gives the scale of the demand: of 2,355
documents it embedded with Qwen3-Embedding-0.6B, 35% were over 512
tokens, 5.7% over 1,024 and 0.2% over 2,048, the longest 2,408.

**What was measured.** Qwen3-Embedding-0.6B's chunked Core ML build
(D37), converted with buckets of 1,024, 2,048 and 4,096 tokens
(`convert_qwen3_embedding.py`, the committed recipe, with those buckets
and `max_seq_len` 4,096 in a local copy of its manifest). M1 Max (32 GB),
macOS 27.0. Sizes are in GiB and MiB.

Latency and memory come from `examples/embed_timing`: sidekick's own
Core ML path, one input at a time, on real token ids (the long corpus's
4,096-token case, cut keeping its final token). Each bucket and compute
units ran in two processes: one with an empty Core ML cache, timing the
median of 3 predictions after two warm-ups, then one with the cache it
left, timing the median of 10. The ranges below span both. Every figure
is from one complete run, made from fresh caches with no gaps in its log;
an earlier run agrees within 4%. Load average was 2.2–3.6 throughout. The
grades come from the parity suite on both corpora, with references
generated for each bucket list.

- **Every bucket converts.** The converter's gates pass at all three
  sizes: fp32 exactness, the chunks equal to the unchunked wrapper, the
  GPU chain bit-identical to the unchunked program (bucket 1,024), pad
  invariance 1.0000000, and 2,129 of 2,135 operations on the ANE in every
  bucket. (The GPU plan it records puts all 2,135 on the GPU.) Each
  bucket's two chunks hold 0.555–0.590 GiB of weights, under D32's limit
  at every size. On disk a bucket is 1.12, 1.13 and 1.18 GiB at 1,024,
  2,048 and 4,096 tokens, and the same artifact serves either device.
- **Latency is the bucket's, not the input's.** A 1,024-token input in
  the 4,096 bucket costs what a 4,096-token one does.

  | bucket | GPU | ANE | Core ML CPU | onnx-cpu (D40) |
  |---|---|---|---|---|
  | 1,024 | 152 ms | 377–384 ms | 557–558 ms | |
  | 2,048 | 349–350 ms | 1,227–1,250 ms | capped (D33) | 6.8–7.1 s at 2,400 (Python, busier machine) |
  | 4,096 | 938–943 ms | 5.0–5.6 s, when it completes (below) | capped (D33) | 12.9–14.2 s |

  The ANE takes 2.5× the GPU's time at 1,024, 3.5–3.6× at 2,048 and
  5.3–5.9× at 4,096; agent-jev, the same backbone, measured 1.9× and
  2.5× at 1,024 and 2,048 (docs/MODELS.md). At 4,096 the GPU is
  13.7–15.1× faster than onnx-cpu's figure, which came from another
  harness (`examples/onnx_memory`, load average 4–11; the Python bench
  gave 15–17 s), so the ratio is approximate.
- **First load.** With an empty Core ML cache, each bucket's first
  prediction compiles it: 2.3–3.0 s on the GPU at every size, and on the
  ANE 55 s at 1,024, 151 s at 2,048 and 1,759 s (29 minutes) at 4,096.
  From the cache: 0.7–1.5 s on the GPU, 0.5, 1.4 and 5.2 s on the ANE,
  though in the parity suite, which shares one cache between processes,
  each new process compiled the 4,096 bucket again.
- **Disk beyond the artifacts.** Core ML's compiled cache held 3.7 GiB
  per bucket on the ANE (3.4 GiB on the CPU, 4.5–4.6 GiB on the GPU),
  3–4 times the bucket's weights. Adding the 1,024 and 2,048 buckets to
  an ANE-served install costs about 2.2 GiB of artifacts and 7.4 GiB of
  cache. *(Not reproduced, and the ANE's programs are elsewhere: see the
  amendment below.)*
- **Memory.** After a bucket's predictions, the process's resident size
  was 1.30–1.72 GiB on the GPU, which maps the weights into the process
  (its `phys_footprint`, which leaves out clean mapped pages, 0.31–0.89
  GiB), and 121–234 MiB on the ANE, which reads them through the file
  cache. Core ML's CPU path: 1.08–1.10 GiB at 1,024. onnx-cpu's
  footprint grows by 4.05 GiB after a 4,096-token input (D40).
- **The ANE fails in the 4,096 bucket.** Some of its predictions fail
  with Core ML's "Unable to compute the prediction using ML Program": 7
  of 51 timed attempts, 3 of 15 in the process that compiled it and 4 of
  36 in the next, from the cache. By input length that is 5 of 17 at
  4,096 tokens, 2 of 17 at 1,024 and none of 17 at 2,048, so a short
  input fails too; what the failures track is the 4,096 bucket. An
  earlier pair of runs failed too, one of them on its first prediction.
  The parity suite's main-corpus ANE worker stopped on one. The 1,024
  and 2,048 buckets failed none of their predictions on the ANE, and no
  other path failed any.
- **Two of sidekick's limits are shorter than the ANE's compile at
  4,096.**
  - The compute-plan read (`compute_plan`, 300 s) compiles the bucket
    for the ANE. The parity suite reads it before anything predicts; it
    timed out, so the suite marked the bucket ineligible and skipped the
    model. Every grade below, the 2,048 install's included, was taken
    with that limit raised to 7,200 s in a local build. That is harmless
    at 2,048, whose compile (151 s) is under 300 s. The daemon reads the
    same plan when `report_compute_plans` is on (`placement.rs`), so it
    would report a 4,096 ANE bucket's placement as failed.
  - The daemon's `load_timeout_secs` (900 s by default) would give the
    first request that needs the bucket a 504 (`load_timeout`), though
    the load goes on and a retry finds it resident (docs/INTEGRATING.md).
- **Grades.**
  - **GPU: A** on both corpora: worst cosine 0.999997 on each, with
    buckets within 0.999996 and exact pads. On the long corpus, 1 − cos
    is 1.3e-6 at 600 tokens, 1.7–2.7e-6 at 1,500, 1.6e-6 at 2,400 and
    2.1e-6 at 4,096.
  - **ANE, to 4,096: fails the suite**, because its main-corpus worker
    stopped on one of the prediction failures above. The long corpus ran
    all six cases, both 4,096-token ones included, and grades A: 1 − cos
    6.4e-6 at 600 tokens, 1.0–1.4e-5 at 1,500, 8.6e-6 at 2,400 and
    8.0–9.0e-6 at 4,096.
  - **ANE, to 2,048: A, every gate passed.** The same 1,024 and 2,048
    artifacts, installed alone with `max_seq_len` 2,048 and references of
    their own: worst cosine 0.999986 on both corpora, drift 0.0017, no
    rank flips, buckets within 0.999985, exact pads, and identical output
    in two processes. The GPU grades the same install A (0.999997).
  - **Core ML's CPU** wasn't graded: D33 caps it at 1,024 tokens, and its
    1,024 bucket is the only one it would serve.

**Decision.**

- **Targets per model and backend.** A build goes to its model's trained
  context or 2,048 tokens, whichever is shorter, unless a measurement
  shows that's not worth it; past 2,048, only where a backend has been
  measured to serve it well, which so far is 4,096 on Qwen3-Embedding's
  ONNX build and Core ML GPU path.

  | model | trained context | today | target |
  |---|---|---|---|
  | all-MiniLM-L6-v2 | 256 | 256 | 256: done |
  | bge-small-en-v1.5, e5-small-v2 | 512 | 512 | 512: done |
  | LFM2.5-Embedding-350M | 512 (its sentence-transformers config) | 512 | 512: done |
  | embeddinggemma-2 | 8,192 | 2,048 | 2,048, unless there's demand |
  | EmbeddingGemma-300m | 2,048 | 512 | 2,048, all paths; to be measured |
  | gte-modernbert-base | 8,192 | 512 | 2,048, when there's demand; to be measured |
  | F2LLM-v2-160M | not stated (a 40,960-position table) | 512 | as Qwen3-Embedding, when there's demand |
  | Qwen3-Embedding-0.6B | 32,768 | ONNX 4,096; Core ML 512 | ONNX 4,096 (unchanged); Core ML 2,048 on the ANE and GPU, 4,096 on the GPU once an ANE cap exists, 1,024 on the CPU (D33) |

  Classifiers keep the lengths their formats set (docs/design/classify.md).
- **Qwen3-Embedding's Core ML build: 2,048 next, 4,096 after an ANE
  cap.** This record changes no manifest; publishing is its own change,
  with release notes, because it changes the build users install.
  - **2,048 is ready to grade as the published bucket list.** The 1,024
    and 2,048 buckets grade A on the ANE and the GPU in an install of
    their own. The published list would put them beside 128, 256 and
    512, and that five-bucket install, including inputs that switch
    between the 512 and 1,024 buckets, must be graded on every path
    before it is published. It would add about 2.2 GiB on disk, plus
    about 7.4 GiB of Core ML cache once both buckets have run on the ANE,
    and cover the reported corpus but for 0.2% of its documents (which
    `/v2/embed`'s `max_tokens` with `truncate: "END"` can cut). The first
    request that needs the 2,048 bucket on the ANE waits about 2.5
    minutes for its compile, inside `load_timeout_secs`. Its CPU path
    stays capped at 1,024 (D33).
  - **4,096 waits for an ANE cap.** One artifact serves every placement
    (D37), and the model is ANE-served by default, so a published 4,096
    bucket would compile for half an hour on its first long request and
    then fail some of its predictions (7 of 51 here). The runtime needs
    an ANE cap, as D33 caps the CPU: buckets past 2,048 aren't loaded for
    a model served on the ANE unless the manifest or the daemon opts out,
    and `/v1/models` and `/health` report the cap. With it, the published
    build can add 4,096 (about 1.2 GiB more) for the GPU.
    `compute_plan`'s limit also needs to cover a long bucket's compile,
    or the parity suite can't grade one and the daemon can't report its
    placement.
  - The ONNX build keeps 4,096, the only path that takes it everywhere,
    Linux included.
- **Declined: anything past 4,096 tokens.** Not Qwen3-Embedding's 32,768,
  gte-modernbert's 8,192 or F2LLM's position table:
  - Attention's cost grows with the square of the length. onnx-cpu
    already takes 13–14 s and 4 GiB at 4,096, from fp32 scores that
    would be 64 times larger at 32,768. A Core ML bucket at 8,192 would
    hold 2 GiB of fp16 scores per layer (16 heads × 8,192² × 2 bytes),
    and the ANE's compile already takes 29 minutes at 4,096.
  - Nothing grades it: the long corpus stops at 4,096, and references
    past that length are torch runs of their own.
  - sidekick runs a small model beside a large one. A client that splits
    a long document into passages pays neither the seconds nor the
    gigabytes, and retrieval usually wants passages anyway.

  A fused or memory-efficient attention export for ONNX Runtime would
  change the first point; this is revisited then, or on demand.
- **How the cap shows.** A build's grades cover inputs up to its
  `max_seq_len`, and no further.
  - The published table gives each build's `max_seq_len` ("Max tokens")
    beside the context its benchmark scores were measured at, and
    `/v1/models` gives `max_seq_len` and `buckets` (#109).
  - A build whose `max_seq_len` passes 512 is graded on the long corpus,
    and its row says so (D40's amendment). Asked to grade a model whose
    `max_seq_len` passes the main corpus, the suite fails it without a
    long reference, so a longer cap isn't published ungraded.
  - A cap that differs by compute path, as D33's does and the ANE cap
    above would, is reported by the runtime, and the docs give each
    path's limit beside its grade.

**Amendment (October 2026): the ANE cap is D48.** The runtime now caps a
model served on the ANE at 2,048 tokens, or its manifest's
`ane_max_seq_len`, and the parity suite grades the ane path under the
same cap without reading a compute plan past it. Publishing a 4,096-token
bucket for Qwen3-Embedding's GPU path remains its own change.

**Amendment (October 2026, M1 Max, macOS 27.0): the cache sizes, the
compile and the failures, revisited.** An audit of Core ML's compile
cache (docs/design/coreml-cache.md, which has the method and the data),
on synthetic MIL encoders and copies of bge-small-en-v1.5's
programs, each run under its own `CFFIXED_USER_HOME`, read hits and misses
from Core ML's own cache-lookup log lines and the ANE compiler's runs
rather than from timing. It corrects four points above. The decisions
stand.
- **Disk beyond the artifacts.** "3–4 times the bucket's weights" and
  "3.7 GiB per bucket on the ANE" were not reproduced. On the GPU and the
  CPU, an entry per bucket and compute path holds about 1.0× the weights
  (a `.mpsgraphpackage` or a `bnns_program.bnnsir`), and a compute-plan
  read, an entry of its own, adds another 1.0×. An ANE compile leaves
  about 12 kB in the user's cache; its program, about 1.0× the weights,
  goes to the system's ANE cache, kept by Apple's `aned` service on the
  system volume, which deleting the user's cache doesn't free. The
  figures above came from the user's cache, so the ANE's describes the
  wrong place. They are to be re-measured with a breakdown that includes
  the system volume; until then, the 7.4 GiB of cache quoted for adding
  the 1,024 and 2,048 buckets is not a reliable estimate ("Transplanting a
  cache", "GPU and CPU specialization").
- **The 4,096 bucket's compile.** Its slope comes from masked attention
  at 4,096 tokens in one program, not from length alone. On synthetic
  encoders at Qwen3's width, with a mask, one program compiled in:

  | layers per program | compile |
  |---|---|
  | 1 | 7.4 s |
  | 2 | 16.9 s |
  | 4 | 42.6 s |
  | 8 | 126 s |

  That is superlinear in layers per program. A mask passed as an input
  costs the same as a constant one; with no mask, eight layers compile in
  28 s but predict 2.5× slower. Eight one-layer programs compiled in
  about 59 s, against 126 s for one eight-layer program, and predicted in
  1,290 ms against 1,228 ms. More, smaller chunks (D37) are therefore a
  candidate fix for the 29-minute compile; #164 measures Qwen3 at 4,096
  tokens in more chunks. At bge-small's shape and 2,048 tokens, splitting
  the queries into blocks (26.0–27.2 s) and the fused SDPA op (34.1 s)
  compiled slower than plain attention (13.8 s). ("Compile time against
  length".)
- **The 4,096 bucket's failures.** In the audit, one ANE prediction (of a
  synthetic program, not a sidekick build) hung for about 100 s, failed
  with "Request cancelled", and left every ANE prediction on the machine
  failing or hanging, probably until a reboot. The intermittent failures
  above may have been that 100 s request timeout; that is inferred, not
  shown ("A hung ANE request").
- **The parity suite's recompiles** of the 4,096 bucket in each worker
  (#145) have two candidate causes: ANE compiles are serialized across
  the whole system, every process's queued behind any other's (a 2.2 s
  compile took 41.9 s behind a 43 s one), and a compute-plan read, which
  the suite makes before predicting, is a cache entry of its own with a
  full compile ("Serialization", "The cache key").

## D47 — An NLI model's class probabilities, on request (`nli_probs`)
D41 scores an NLI model's labels as Hugging Face's zero-shot pipeline
does, and both of its scores reduce a pair's classes to one number:
- **Several labels:** a softmax across labels of their entailment logits.
- **One label, or `multi_label`:** the softmax over [contradiction,
  entailment], which is sigmoid(entailment − contradiction).

Neither reports neutral. A low score can mean the text contradicts the
claim or never mentions it, and a claim the text ignores (neutral high,
entailment and contradiction low and close) scores near 0.5 alone, as a
weakly supported one does. To check a claim against evidence, the
distinction is the point: a contradicted claim is worse than an
unsupported one (#134). The verification corpus
(#133) scores its CONTRADICTED and NOT MENTIONED variants apart, and
needs that from sidekick.

**Decision.** An opt-in request field, `nli_probs: true`, on the nli
format:
- **Response.** Each result gains `nli_probs`, parallel to `probs`: per
  candidate label, an object from each of the model's classes to that
  pair's softmax over its logits, in output order:
  `{"contradiction": 0.97, "entailment": 0.01, "neutral": 0.02}`. One per
  (input, label) pair, for any number of labels, with or without
  `multi_label`, which changes `probs` only. With `use_activation: false`
  the values are the class logits, as `probs` then carries logits.
  Truncation and batching are unchanged: the pairs are the ones `probs`
  is computed from.
- **Defaults unchanged.** Without the field, or with `false`, a response
  is byte for byte what it was. With it, `probs` and `label` are too:
  `Classifier::run_nli` returns each pair's class logits, and the server
  derives the per-label values from them with the same `nli_value` the
  classifier's `run` uses, so one prediction per pair serves both fields.
- **The name.** sidekick's `/v1/classify` follows vLLM's, whose `probs`
  are a classifier's activated class values; `nli_probs` is the same for
  an nli pair's own classes, and its prefix scopes it to the format, as
  the response field it adds is scoped. `nli_scores`, the issue's working
  name, was set aside because Hugging Face's pipeline calls its reduced
  per-label values `scores`, which `probs` already carries. The
  text-classification pipeline's `top_k=None` (formerly
  `return_all_scores`) returns every class of one classifier, but would
  read as changing `probs` itself. Like `multi_label: false`,
  `nli_probs: false` is accepted on every model; `true` elsewhere is a
  400.
- **The classes come from the manifest.** `[classify.nli] classes` lists
  config.json's `id2label` in output order, lowercased, so
  roberta-large-mnli's `ENTAILMENT` and the cross-encoders' `entailment`
  key alike. A `neutral` index beside `entailment` and `contradiction`
  was the alternative. A two-way model has no neutral, and its second
  output isn't a contradiction but not_entailment; naming every output
  says what each one is without inferring it. The registry checks the
  list against `n_classes`, that its names are distinct and nonempty,
  that `entailment` points at an entailment class, and that
  `contradiction` points at `contradiction` or `not_entailment`, so a
  list in the wrong order is refused. The converter and the reference generator
  check it against the checkpoint's config. `/v1/models` lists it as
  `nli_classes`, and lists `nli_probs` in `extensions` exactly when it is
  present.

The example manifests' classes, from each checkpoint's config.json at
its pinned revision:

| model | classes, in output order |
|---|---|
| nli-deberta-v3-xsmall, -small, -base; nli-minilm2-l6-h768 (and its ANE build) | contradiction, entailment, neutral |
| roberta-large-mnli | contradiction, neutral, entailment |
| deberta-v3-base-zeroshot-v2.0-c, deberta-v3-large-zeroshot-v2.0-c, roberta-large-zeroshot-v2.0-c, bge-m3-zeroshot-v2.0-c, modernbert-large-zeroshot-v2.0 | entailment, not_entailment |

The first six are three-way. The zeroshot-v2.0 models are two-way and
can't separate contradicted from unmentioned: their `nli_probs` says so
by its keys.

**Compatibility.** `classes` needs sidekickd 0.15.0 or later; older
daemons skip the model. `[classify.nli]` refuses unknown keys, so a
daemon before 0.15.0 given a manifest with `classes` drops the whole
model (D28): it is missing from `/v1/models`, and the only signal is a
warning in the log. Nothing writes `classes` into an installed manifest
except a conversion: converters install the committed example manifest,
so a conversion from a checkout with this change needs a daemon of
0.15.0 or later to serve it (docs/CONVERTING.md says so).

The other direction is safe. `classes` is optional, so a manifest
installed before 0.15.0 still loads on it. Such a model doesn't list
`nli_probs`, and a request for it is a 400 naming the missing key. To
enable it, add the model's single `classes = [...]` line under
`[classify.nli]` in the installed manifest (each model's line is in
docs/design/classify.md). Don't copy the example manifest over it: an
installed manifest carries the `[placement]` table its conversion
recorded, which the example lacks. The artifacts don't change, and
parity references stay current (they record only the indices, count
and template).

*Proposed, not implemented here:* an unknown key that only adds
information, as `classes` does, should produce a warning rather than
skip the model, so that a new manifest degrades on an older daemon
instead of disappearing. That needs a rule for which keys are additive
and is left to its own issue.

**Validation.** Unit tests check the softmax per pair (sums to 1, equal
to the softmax of the logits, keys in output order), the per-label
values derived from the same logits, and the manifest rules. Server
tests, on mock three-way and two-way models, pin the existing responses
with and without the field, check each label's distribution and the
class it favors (entailed, contradicted, unmentioned), `use_activation:
false`, truncation, batching, the 400s and the listing. On the tiny Core
ML cross-encoders, the class logits match fp32 torch and the values
`run` returns are exactly those derived from them. Live, through the
daemon, on nli-minilm2-l6-h768, deberta-v3-base-zeroshot-v2.0-c and
roberta-large-mnli (on the GPU, as served; 9 pairs each, requested with
and without `multi_label`): `probs` identical with and without the
field, `nli_probs` within 1.8e-3 of fp32 torch's softmax, and the same
argmax on every pair. A side-by-side review on four installed nli
models found 1,296 requests identical to the previous release's once
`nli_probs` was removed.

## D48 — Cap ANE-served models at 2,048 tokens
One Core ML artifact serves every compute unit (D37), and most embedders
are served on the ANE by default. So a long bucket that the GPU serves
well can't be published while the ANE can't: everyone on the default would
get it. D46 measured exactly that on Qwen3-Embedding-0.6B's chunked build
(M1 Max, macOS 27.0). Its 4,096-token bucket took 29 minutes to compile for
the ANE on first load, then took 5.0–5.6 s per prediction, and 7 of 51
predictions failed ("Unable to compute the prediction using ML Program"),
short inputs in that bucket included. The GPU served the same artifact in
under a second, graded A. The 1,024- and 2,048-token buckets compiled for
the ANE in 55 s and 151 s, failed none of their predictions, and graded A
there. D33 caps the CPU for a different reason (summation order); this is
the ANE's own limit, with its own default and override, so it is a record
of its own rather than an amendment to D33.

**Decision.** As D33 does for the CPU, the registry caps the length of a
Core ML model served on the ANE:
- **`DEFAULT_ANE_MAX_SEQ` (2,048).** For `compute_units` `cpu_and_ne` or
  `all`, a manifest's `max_seq_len` past the cap drops the buckets above it,
  and the largest bucket kept becomes the effective `max_seq_len`. Those
  buckets are never loaded, compiled or planned on the ANE. A model with no
  bucket within the cap is skipped, with the reason and the fixes. The
  capped manifest is validated again (laya's head budget must still fit),
  and a fev model's state limit follows fev's formula for the capped
  window, as under D33. 2,048 is the longest length measured to work on
  the ANE, and no published build goes past it, so the default changes no
  published model.
- **`all` is capped too, unlike under D33, by reasoning rather than
  measurement.** D46 measured `cpu_and_ne` only. Under `all`, Core ML
  places ANE-eligible operations on the ANE, so it compiles the bucket for
  the ANE, and the long compile and the failures are expected to come with
  it. D33's reasoning runs the other way: under `all` Core ML may not use
  the CPU at all, so a CPU cap could hide what `all` does, whereas the
  ANE is exactly where `all` puts an ANE-eligible model. A measurement of
  a long bucket under `all` would confirm or revise this.
- **`cpu_and_gpu` is unaffected,** and `cpu_only` keeps D33's 1,024. The
  two caps never stack: `seq_len_cap` chooses at most one per model, by
  the compute units Core ML will actually use (`capped_as`). On a Mac with
  a Neural Engine that is the units' name. On a Mac without one (an Intel
  Mac), Core ML runs `cpu_and_ne` on the CPU, so it gets D33's 1,024 and
  honours `cpu_seq_limit`, and the ANE cap and `ane_max_seq_len` don't
  apply; `all` runs on the CPU and GPU there, which D33 leaves uncapped.
  The daemon and `libsidekick.dylib` ask Core ML whether a Neural Engine
  is present (`ScanOptions::no_neural_engine`).
- **Per model: `ane_max_seq_len`.** A top-level manifest key (either
  manifest file, beside `cpu_seq_limit` and `ane_weight_limit`; coreml
  backend only; at least 1) replaces the default. It can lower the cap,
  for a model that fails sooner, or raise it to the model's `max_seq_len`
  or past it, which lifts the cap, for a model measured to serve its long
  buckets on the ANE. How long a bucket the ANE serves well is a property
  of the artifact, measured when it is converted and graded, so the value
  belongs in the manifest the converter writes.
- **The daemon's opt-out:** `ignore_ane_seq_cap = true`, or
  `sidekickd --ignore-ane-seq-cap`, lifts the cap for every model. It is
  for measurement, as D32's and D33's opt-outs are. There's no per-model
  length in the daemon config: the operator's per-model choice is where a
  model runs (`[models."<id>"] compute_units`, D38), which the cap then
  judges as it would the manifest's. An operator who wants a model's long
  inputs serves it on the GPU. A per-model length would let the daemon
  config re-enable buckets measured to fail; a model measured to work past
  the default on the ANE says so in its manifest instead.
- **One limit everywhere.** The effective `max_seq_len` is the one every
  route already uses, so nothing else changes: `/v1/embeddings` refuses a
  longer input with a 400 naming the effective limit under `truncate:
  NONE` (D43) and truncates at it otherwise, `/tokenize` reports it as
  `max_model_len` and judges `fits` by it (D44), and a classifier's format
  truncates or refuses by its usual rules. `/v1/models` lists the
  effective `max_seq_len` and `buckets` with `seq_cap` (`limit`,
  `manifest_max_seq_len`, and a `reason` that names D48 and says whether
  the manifest set the cap), `/health` lists every capped model under
  `seq_caps`, and the daemon logs each cap at startup. `libsidekick.dylib`
  scans as the daemon does, so it applies the cap too. The caps apply
  before D32's weight limit, so a bucket the cap drops is never read.
- **The parity suite** grades the ane path under the cap the runtime
  applies. Its ane worker loads only the buckets within it, and runs only
  the cases whose reference ids fit it. The cases past it are listed as
  `capped` in the ane path's grade, not failed, and the grade covers the
  rest, printed in a table of its own. For a model served on the ANE, no
  compute plan is read for a bucket past the cap (the report records it
  as not read): the runtime never serves it there, and reading one
  compiles the bucket for the ANE, which timed out the plan read at 4,096
  tokens (D46). The cpu and gpu paths still run every bucket.
  `--ignore-ane-seq-cap` grades the ane path past the cap, as D46's
  measurement did. *(The cpu path now runs under D33's cap the same way:
  see D33's amendment.)*

With the cap in place, Qwen3-Embedding's Core ML build can add a
4,096-token bucket for the GPU (D46).

**Amendment (October 2026, M1 Max, macOS 27.0): 2,048 is a measured
length, not a limit of the ANE.** The cap and its default stay. Its
rationale is narrower than "the ANE's own limit": an audit of Core ML's
compiles (docs/design/coreml-cache.md, "Compile time against length")
found that the 4,096 bucket's steep compile
comes from masked attention at 4,096 tokens in one program, and grows
superlinearly with the number of layers in that program (7.4 s for one
layer at Qwen3's width, 126 s for eight), while eight one-layer programs
compiled in about 59 s and predicted within 5% of one eight-layer
program. D46's build split the model into two chunks. More chunks are a
candidate fix, and #164 measures Qwen3 at 4,096 tokens in more of them,
with a timeout on every prediction, since a hung ANE request in the audit
probably needed a reboot to clear. A build measured to compile and grade
there lifts the cap for itself with `ane_max_seq_len`, as above; the
default stays 2,048 until such a measurement.

## D49 — One weights file per chunk, shared by every bucket
D15 makes each bucket its own static-shape program, and D37 splits a
large bucket into chunks, each its own program. Every program carried its
own `weights/weight.bin`, so a model stored its weights once per bucket:
three times for most models, which also triples what publishing them
(#137) would ask users to download. The tensors are the same in every
bucket, but the files weren't. coremltools writes constants in program
order, and the few that depend on the bucket shift every tensor after
them: the folded position and token-type tables, the RoPE tables and the
causal mask, under 1 MB per bucket at 512 tokens.

A spike (#142, M1 Max, macOS 27.0, coremltools 9.0) measured two ways to
store the weights once. One was Core ML multifunction programs, one
package with a function per bucket. The other was separate programs that
share one byte-identical weights file. Both stored the weights once (bge-small-en-v1.5
199.5 → 67.4 MB, F2LLM-v2-160M 956 → 320 MB, chunked
Qwen3-Embedding-0.6B 3.60 → 1.22 GB), and both gave outputs bit-identical
to the separate packages on CPU, GPU and ANE. Neither changed first-load
compile time, Core ML's cache or memory. Multifunction saves no more disk,
needs loader changes (and the weight check and build fingerprints would
have to understand a function path), and bge-small, which keeps the fused
attention fallback (D25), refuses `functionName` on the ANE.

**Decision.** A Core ML conversion with more than one bucket stores each
chunk's weights once, in one file every bucket's program shares
(`tools/sidekick_convert/shared_weights.py`):
- **The converter re-exports, after the gates.** Once every bucket has
  passed its gates as a separate build, each bucket's saved `.mlpackage`
  is loaded back into MIL and written again, with no graph passes, against
  one union weights file per chunk: each distinct tensor (by dtype, shape
  and bytes) once, the shared tensors first and each bucket's own after.
  It is the default; `--no-shared-weights` gives each bucket its own file.
- **It is gated against the separate build,** and a failure stops the
  conversion. Every compiled weights file in the chunk must be the union
  file, byte for byte; every compiled `model.mil` must equal the separate
  build's with weight offsets set aside; and on CPU_ONLY, CPU_AND_GPU and
  CPU_AND_NE every program must give the separate build's outputs on the
  gate cases, bit for bit. Each of those comparisons runs in a child
  process with its own Core ML cache, deleted when it exits.
- **Per chunk, never across chunks.** The spike found the ANE's weight
  limit (D32) judged on the whole file a program references: a file
  spanning both of Qwen3's chunks (about 1.2 GB) put every chunk on the
  CPU. So chunk k of every bucket shares one file. The union is the shared
  weights plus every bucket's own tables, slightly larger than one
  bucket's file. For a model served on the ANE, a union over
  `MAX_ANE_PROGRAM_WEIGHT_BYTES` keeps that chunk's files separate, with a
  note, rather than failing.
- **Installed as hard links.** The converter installs each union file once,
  with a hard link per bucket. Where the filesystem refuses a link (another
  volume, or no hard links), it keeps a copy. `python -m
  sidekick_convert.shared_weights link <model-dir>` does the same for a
  model directory copied or downloaded without its links. Hard links rather
  than APFS clones: they work on any POSIX filesystem, and the sharing is
  visible (one inode) to the tools that hash or size a model.
- **Nothing else changes.** Each bucket is still its own program at its own
  path, so the daemon, the manifest and the weight-limit check (which sums
  each program's own files) are unchanged, and any one bucket directory is
  still a complete program.
- **What isn't shared.** Core ML's compile cache keeps a specialized copy
  of the weights per bucket, per compute path, per executable and per path
  (Qwen3: about 3.6 GB on the GPU or CPU for three buckets). *(On the GPU
  and the CPU; the ANE's copy is in a system cache, and the key has more
  parts: see the amendment below.)* Each loaded
  bucket maps its own copy from it, so memory doesn't shrink, and every
  bucket still compiles on its first load. Pre-warming that cache is
  separate work (#145).
- **Build fingerprints move.** A shared-weights build's `weight.bin` is a
  different file from a separate build's, so its sha256 differs, and so do
  the fingerprints computed from it (the parity report's weights hash,
  `tools/mteb_transfer.py --build`). The outputs are bit-identical, so the
  grades and scores carry over, but a ledger entry or a fingerprint
  recorded for a separate build doesn't match a shared one.

**Publishing (#137): one copy of each distinct file, verified by hash, and
hard-linked by fetch.** Not a tar with hard links. The published tree is
the install tree, with every path present: the Hub stores files by content
(Git LFS objects by sha256, Xet by chunk), so a file at three paths is
stored once there. Beside it, the build's file list gives each distinct
sha256 once, with its size and every path that holds it. Fetch downloads
each distinct hash once, to a temporary file on the destination's
filesystem; verifies it before anything is placed; moves it to the
first path; and hard-links the others, copying where a link is refused.
That the Hub stores a repeated file once is to be confirmed at the first
upload. If it doesn't, the build uploads each distinct file once, at its
first path, and fetch, which already follows the list, is unchanged. Why
not a tar:
- **Verification per file, before placing it,** against the same hashes
  the ledger (#136) and the model card publish. A tar is verified as a
  whole, after all of it has downloaded, and what it holds only once
  extracted.
- **Partial fetches.** A user who needs fewer buckets (D48 caps the ANE at
  2,048 tokens) or one compute path's build fetches only those files; a
  tar is all or nothing.
- **Resumable, parallel downloads** of gigabyte files, which the Hub's
  per-file transfer already gives; a tar is one object.
- **No transient double:** a downloaded tar and its extracted tree take
  twice the disk until the tar is deleted.
- **Plain tools still work.** `hf download` into the Hugging Face cache
  stores each content hash once; into a directory it writes copies, which
  `shared_weights link` turns back into links. Hard links inside a tar
  survive only extractors that support them.

No fetch command exists yet, so its code is #137's.

*Amended by D51:* fetch downloads into the Hugging Face cache, which holds
each distinct file once as a blob, verifies each before it is placed, and
installs every path as a hard link to its blob; a symlink wouldn't do, as
Core ML can run a program whose files are symlinks out of its directory
off the ANE (D51, as amended).

**The checksum ledger (#136)** records each distinct file once: its sha256,
its size, and the list of installed paths that hold it, for example
`model_1024.0.mlmodelc/weights/weight.bin` and
`model_2048.0.mlmodelc/weights/weight.bin` under one hash. A file only one
path holds has a one-path list. Fetch reads the same list, and a verify run
reports a path whose hash moved, or a shared file that split, as a change
to the weights layer. *(Implemented as docs/data/ledger.json and
tools/ledger.py, D50.)*

**Measured** (M1 Max, macOS 27.0, coremltools 9.0, converted from the same
checkpoints): weight files on disk, and the parity suite's grades, for
separate and shared builds. The converter found every program
bit-identical to its separate build on CPU, GPU and ANE, and the suite's
grades, worst cases and per-case cosines were identical between the two
builds on every path.

| model | buckets | weight files, separate → shared | grades, CPU / GPU / ANE |
|---|---|---|---|
| bge-small-en-v1.5 | 128, 256, 512 | 199.5 → 67.4 MB (34%) | A / A / A |
| F2LLM-v2-160M | 128, 256, 512 | 956.3 → 319.5 MB (33%) | B / A / A |
| Qwen3-Embedding-0.6B, chunked | 1,024, 2,048 | 2,407.3 → 1,215.7 MB (51%) | F / A / A, both corpora |

With two buckets the files halve, and with three they fall to a third.
Qwen3's separate build is the 2,048-token release candidate (D46, D48).
Its CPU F is that build's own: bucket invariance of 0.999894 against a
0.999999 gate, which D33's CPU cap covers.

**Amendment (October 2026, M1 Max, macOS 27.0): what Core ML's cache is
keyed by, and where the ANE's programs are.** The spike's (#142) account
of the cache was partly wrong. An audit measured it from Core ML's own
cache-lookup log lines and the ANE compiler's runs
(docs/design/coreml-cache.md, "The cache key" and "Transplanting a
cache"). The decision is unaffected: each bucket is still its own
program at its own path, so sharing a weights file changes none of this.
- **The key.** The cache directory is named for the process as invoked
  (or for the bundle identifier of an Info.plist embedded in the binary),
  under the user's home, then for the macOS build. An entry is keyed by
  the program's path, its parsed MIL, the identity of its `weight.bin`
  (inode and birth time, not its contents or modification time), the
  compute units and, on the ANE, the loading binary's code-signing
  identifier. A copy of the program at another path misses, hard-linked
  or cloned ones included, as does another executable name and, on the
  ANE, another signing identifier. A binary moved to another directory,
  or rebuilt, with the same name and identifier hits: the spike's "the
  same one after a move recompiled" doesn't hold. A compute-plan read is
  an entry of its own, with a full compile and another copy of the
  weights.
- **A copy per bucket per path holds for the GPU and the CPU only.** Their
  entries hold about 1.0× the weights in the user's cache. An ANE entry
  there is about 12 kB; the program, about 1.0× the weights, is in the
  system's ANE cache, kept by Apple's `aned` service on the system
  volume. It survives deleting the user's cache, can't be deleted by the
  user (System Integrity Protection), and each recompile adds another
  copy.
- **Upgrades recompiled every ANE bucket.** sidekickd's signing
  identifier was the linker's, cargo's name for the build
  (`sidekickd-<hash>`), and the hash changes with the package version.
  Release archives now sign sidekickd ad hoc with a fixed identifier,
  `io.github.ironblock.sidekickd` (`tools/sign_sidekickd.sh`, which
  source builds run too). Measured with bge-small-en-v1.5's 128-token
  bucket on the ANE, two builds a version apart, the second loading at
  the same model path and home as the first: with both signed that way,
  the second's first request took 0.87 s, and the ANE compiler ran only
  for the failed attempt every bge-small load makes (D25's fused
  attention, about 0.17 s); with the linker's signatures it took 3.76 s,
  and the ANE compiler ran again for 2.4 s, as on the first load (4.3 s).
  The outputs were identical.

## D50 — Publishing: compiled programs, a checksum ledger, and gated consent

Publishing graded builds on Hugging Face (#137) needs three things settled
before the first upload: what may be published, how a download is
verified, and in which form Core ML builds ship. docs/PUBLISHING.md has the
detail.

**Decision.**
- **A licence tier per model, recomputed from its evidence.**
  docs/data/licences.toml records each model's licence, base model and
  training datasets (each under its original source's terms, pinned), and
  a tier: 1 publish; 2a publish with a notice (share-alike, undisclosed or
  unnamed-generator data); 2b publish behind consent (non-commercial data,
  or terms that travel with the weights); 3 don't publish. A base model's
  pretraining data doesn't count. `tools/licences.py` recomputes the lowest
  tier the recorded data allows and refuses a verdict that differs. Today:
  none at 1, 18 at 2a, 9 at 2b, and LFM2.5-Embedding-350M at 3 (commercial
  use licensed only below a revenue threshold).
- **A checksum ledger, docs/data/ledger.json (#136),** for every build,
  published or not: identity, what produced it, every distinct file once by
  sha256 with its size and paths (so a D49 shared weights file is one
  record with several paths), the grades of the parity reports that graded
  those exact files (matched by each bucket's weights hash), and each
  upload. JSON because tools write it and the standard library can't write
  TOML. `tools/ledger.py verify` checks a directory against it and names
  the layer that moved.
- **Ship compiled `.mlmodelc`, not `.mlpackage`.** The compiled programs
  are what was graded and what the ledger hashes; a package compiled on the
  user's Mac would be a program nobody graded, whose hash can't be known
  in advance. They travel: eight ledger builds were graded from the same
  compiled files on an Intel Mac under macOS 26.6.2, another architecture
  and an older macOS than they were graded on first. Compiling at install
  needs Xcode or new daemon code, and doesn't remove the per-device
  first-load compile (#142), which happens either way.
- **Tier 2b is a gated repository with automatic approval.** The card's
  `extra_gated_*` front matter states terms generated from the audit:
  the non-commercial datasets and that the user decides whether their use
  is permitted; the travelling terms (Gemma) the user receives the files
  under and passes on; that the files are an as-is conversion under the
  original licence; and that Hugging Face shares the user's username and
  email with the repository's owner. One checkbox, nothing else collected.
  The same text ships as TERMS.md for fetch to show and record.
- **Repositories:** `sidekickd/<model id>-coreml` and `-onnx`, under the
  `sidekickd` Hugging Face organization, one directory
  per variant holding the install tree as is; cards generated by
  `tools/model_card.py` from committed data only.

**Amendment (October 2026, macOS 27.0): Core AI's ahead-of-time path.**
Shipping `.mlmodelc` stays right for Core ML, which offers nothing that
would remove the first-load compile: `compileModel` produces the same
device-independent program, the specialized program can't be exported,
and an ANE cache copied to another home or executable recompiles (D49's
amendment). macOS 27's Core AI framework does support ahead-of-time
compilation: `xcrun coreai-build compile` builds one `.aimodelc` per
device architecture from a Core AI model (`.aimodel`, converted with
Apple's coreai-torch tools), and `AIModel.specialize` and `AIModelCache`
manage the specialized form. An audit of Core ML's cache
(docs/design/coreml-cache.md, "Supported APIs") found this in Core AI's
documentation but couldn't run it (it needs the Metal
Toolchain). #163 evaluates Core AI as a backend beside Core ML, which
would stay for macOS 26 and Intel Macs.

**Amendment (October 2026): a gate only for terms that must travel.**
Tier 2b, the gated repository, is now only for terms that must be passed
on to every recipient of the files: today the Gemma Terms of Use and
Prohibited Use Policy (EmbeddingGemma-300m, and EmbeddingGemma 2, whose
card requires the policy and whose coverage by the Gemma terms is
unconfirmed). Training data with non-commercial terms moves to 2a, a notice
without a gate. The seven models it affects (all-MiniLM-L6-v2,
e5-small-v2, gte-modernbert-base, F2LLM-v2-160M, Qwen3-Embedding-0.6B,
ms-marco-MiniLM-L6-v2, ModernBERT-large-zeroshot-v2.0) are now 2a: 25 at
2a, 2 at 2b, LFM2.5-Embedding-350M at 3.
- **Why.** A gate costs every user a Hugging Face account, a click through
  the terms, and their username and email address shared with the
  publisher. That is worth it when the terms legally must reach every
  recipient, as the Gemma terms require of a redistributor. No licence asks
  a redistributor to collect consent to a training set's non-commercial
  terms, and whether such terms bind a trained model at all is unsettled,
  so a gate bought no legal protection for that cost.
- **What replaces it.** The card's notice keeps the gate's substance and
  its prominence, at the top of the card: which datasets, with their
  terms; that whether such terms bind a model trained on the data is
  legally unsettled; that the user decides whether their use is permitted,
  with no assurance from the publisher; and that the model's own licence
  is unchanged. Nothing is accepted, nothing is shared, no TERMS.md ships
  and fetch asks for no `--accept-terms`.
- `tools/licences.py` recomputes the tiers by the new rule, and the cards'
  `datasets` front matter now lists every training set with a Hub id
  (MS MARCO as `microsoft/ms_marco`), not only those whose cited source is
  a Hub page.

## D51 — `sidekickd fetch`: the Hugging Face cache, verified, installed as hard links
Publishing graded builds on Hugging Face (#137) needs a client: download a
build, check it is the build sidekick published, and install it where the
daemon serves models. Users install sidekick from a release archive and may
have no Python, so the client is part of sidekickd.

**Decision.**
- **A `sidekickd fetch` subcommand,** not a separate binary: the archive
  already ships sidekickd, and fetch needs the daemon's own code, its
  config and models directory, its manifest types (to write the manifest
  for a subset of buckets) and its registry (to report whether the daemon
  will serve what it installed). `sidekickd fetch <model> [--backend
  coreml|onnx] [--variant V] [--buckets 128,512] [--accept-terms <model>]`,
  with `--repo` and `--revision` to name a build the ledger doesn't.
  Bare `sidekickd` still serves.
- **The user's `hf` CLI downloads when it's installed** (1.0 or later):
  fetch runs `hf download <repo> --revision <rev> --include …`, so Xet
  transfers, the user's token, its rate limits and its settings (`HF_HOME`,
  `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `HF_ENDPOINT`) apply as they would to
  any download. sidekickd never reads the token.
- **Without `hf`, sidekickd downloads into the same cache, in the same
  layout:** the root huggingface_hub would use (`$HF_HUB_CACHE`, else
  `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
  `~/.cache/huggingface/hub`), `blobs/<etag>`, `snapshots/<commit>/<path>`
  as relative links into `blobs/`, `refs/<revision>`, huggingface_hub's
  `trees/<commit>.json` listing and its per-blob lock. A blob downloads to
  `<etag>.incomplete`, resumes, and is verified before it is renamed into
  place. Measured with hf 1.27 against a public repository: `hf` finds
  sidekickd's cache complete offline and downloads nothing online, and the
  reverse; the two caches are identical, file for file. The downloader is
  anonymous, through the system's `curl`: the workspace has no HTTP client,
  and one would bring a TLS stack (with rustls, C code to cross-compile)
  into a daemon that serves loopback HTTP. A gated or private repository
  needs `hf`, and fetch says so.
- **Both paths select files with the same `--include` patterns.** The
  first pass fetches every variant's manifest and TERMS.md; the chosen
  variant's buckets map to `<variant>/<manifest>`, `<variant>/<tokenizer>`
  and `<variant>/<program>/*` for each program (both chunks of a chunked
  bucket, D37), and an ONNX graph's external weights beside it. sidekickd
  matches them with huggingface_hub's `fnmatchcase` semantics, so `hf` and
  the fallback fetch the same files.
- **Every file is verified twice.** Against the Hub: a Git LFS or Xet
  file's etag is its sha256, any other file's its Git blob id. Against
  sidekick's checksum ledger (docs/data/ledger.json, D50), built into
  sidekickd: the files must be one of the builds the ledger records for the
  model, backend and variant, every installed path with its sha256 and
  size, and every recorded path of the buckets fetched present. A mismatch
  installs nothing. With no build recorded (a repository named with
  `--repo`), fetch verifies against the Hub's hashes and says so. The
  ledger also gives the repository and revision a build was published at.
- **Terms first.** A repository with a TERMS.md (a tier-2b model's, D50)
  shows it, and stops before downloading the build, until the user names
  the model with `--accept-terms`; `fetch.toml` records the sha256 of the
  terms accepted. A gated repository's own consent is given on the Hub,
  which fetch can't do for the user; a refused download says where.
- **The install is a real directory of hard links to the cache's blobs,**
  with the manifest written for exactly the buckets fetched (the published
  one, unchanged, when all of them are) and `fetch.toml`, the provenance:
  repository, revision, commit, variant, buckets, ledger build, and every
  file's sha256. It is built beside the models directory's entries and
  moved into place; fetch replaces a directory only when its own
  `fetch.toml` is there, or with `--force`. Where the filesystem refuses a
  link (the cache on another volume), the file is copied, and fetch says
  how much.

**Why hard links, not symlinks into the snapshot.** Measured on an M1 Max,
macOS 27.0, with bge-small-en-v1.5's three programs: a compiled program
with **any** file inside it a symbolic link (the weights alone, the other
files alone, or all of them, relative or absolute) is planned entirely off
the ANE, with no error: its compute plan puts 0 of 245 operations on the
ANE (229 normally), a prediction takes 28 ms instead of 2, and its outputs
differ from the ANE's by up to 0.016. On the CPU and the GPU the same
programs give bit-identical outputs. A program whose own directory is a
symlink to a real directory, and a program of hard links, stay on the ANE
and are bit-identical to the original. A Hugging Face snapshot holds every
file as a link into `blobs/`, so serving one in place, or linking a
model's programs to it, would quietly leave the ANE, the failure D32 and
D48 exist to prevent. The registry now refuses a model served on the ANE
with a symlinked file in any program, naming it, rather than serving it
off the ANE. Hard links also keep the ONNX external-data check (D40)
unchanged, which refuses weights that resolve outside the graph's
directory. The weight-limit check (D32) sums each program's own files by
their metadata, which reads a link's target, so it sees the same sizes
either way (unit-tested).

**Amendment (October 2026, M1 Max, macOS 27.0): which symlinks leave the
ANE.** Hard links and the registry's refusal stay; their reason is
narrower than "any symlinked file". An audit of Core ML's cache repeated
the measurement (docs/design/coreml-cache.md, "Symlinks and the fallback
loader"):
- **Synthetic programs,** which Core ML's current loader compiles for the
  ANE, stayed on the ANE, bit-identical, under every symlink variant.
- **bge-small-en-v1.5 leaves the ANE because it falls back.** Its ANE
  compile on the current loader fails (D25's fused attention), and Core
  ML falls back to its older loader. The ANE service then can't open a
  link whose target is outside the program directory (`Code=2`), so the
  program runs on the CPU: 28.8 ms against 3.2 ms, with different
  outputs. A relative link inside the program directory stays on the
  ANE, as does a link for `coremldata.bin` pointing outside. The likely
  mechanism, the scope of the ANE service's sandbox extension, is
  inferred.

So the failure is Core ML's fallback loader with links that leave the
program directory, which every link of a Hugging Face snapshot does. Any
program that falls back would leave the ANE silently, so installs stay
hard links and the refusal stays. That programs the current loader takes
are unaffected is to be confirmed on a sidekick build with explicit
attention. Reconverting bge-small without the fused attention would also
remove a failed ANE compile attempt, about 0.17 s, from every load.

**Measured end to end** (a local stand-in for the Hub serving
bge-small-en-v1.5's installed build in the published layout, through both
paths): a fetch of buckets 128 and 256 installs 11 hard-linked files and a
manifest for those two; served by sidekickd on the ANE, its embeddings and
those of a full fetch are bit-identical to the original install's, and the
two-bucket install refuses a 400-token input (D43). A second fetch into
the same cache downloads nothing. A changed ledger hash, a corrupted blob,
an unknown bucket, a wrong backend and a directory fetch didn't install
are each refused before anything is installed.

## D52 — Warm Core ML's compile cache before requests need it
The first time a process loads a Core ML bucket on a compute path, Core ML
specializes it for the device. On the ANE that is a full compile: seconds
for a small model, about a minute for Qwen3-Embedding-0.6B's 1,024-token
bucket and 2.5 minutes for its 2,048 (D46, D49). sidekickd loads buckets
lazily, so the first request that needed a bucket paid its compile, after
every install, model move or macOS update. No request should wait on a
compile sidekickd could have done in advance.

**What the cache is keyed by, measured.** The cache's behaviour is
recorded in docs/design/coreml-cache.md, from Core ML's own cache-lookup
logs and the ANE compiler's runs on an M1 Max, macOS 27.0 (its "Method"
section). What this decision rests on, with its own timed loads on the
same machine (bge-small-en-v1.5 on the ANE at 128/256/512,
nli-minilm2-l6-h768 on the GPU and its ANE build, under a temporary
home, with fresh copies of the models at new paths for cold starts):
- **An entry is keyed by** the executable's name (the cache directory is
  `~/Library/Caches/<name>/com.apple.e5rt.e5bundlecache/<build>/`, per
  macOS build), the home, the program's path and parsed MIL, its weights
  file's inode and birth time (not its content or modification time),
  the compute units, and on the ANE the loading binary's code-signing
  identifier (docs/design/coreml-cache.md, "The cache key"). Timed here:
  the same sidekickd binary copied to another directory hit
  (nli-minilm's ANE build 0.02–0.06 s); a copy under another name
  compiled again (1.83, 2.62 and 3.64 s); a copy of bge-small at a new
  path compiled from scratch (3.47 s against 0.85 s warm).
- **A plain `cargo build` signs sidekickd ad hoc as
  `sidekickd-<hash>`,** an identifier that changes from build to build,
  so a rebuilt or upgraded sidekickd finds every ANE bucket cold unless
  it is re-signed with a fixed identifier, as `tools/sign_sidekickd.sh`
  and the release archives do (docs/design/coreml-cache.md, "The cache
  key").
- **The compiled ANE program lives in the system's ANE cache, not the
  home;** the home keeps a small record per ANE program and a
  weights-sized copy per GPU or CPU entry. Deleting the user's cache
  doesn't remove the ANE program, yet a load under an empty home misses
  and compiles again, adding another copy to the system cache
  (docs/design/coreml-cache.md, "Transplanting a cache"). Timed here,
  with the home emptied, nli-minilm's ANE build loaded in 1.77 s (cold
  2.0 s, warm 0.02 s). bge-small loaded in 0.85 s even so: its ANE
  compile fails over its fused attention (D25) and Core ML falls back to
  an older loader, which reads the system's ANE cache directly
  (docs/design/coreml-cache.md, "Symlinks and the fallback loader").
- **ANE compiles run one at a time across the whole system,** whichever
  process asks: a 2.2 s compile from another process took 41.9 s, queued
  behind a 43 s one (docs/design/coreml-cache.md, "Serialization").
- **Within one process, even a cached load waits for an ANE compile in
  progress; a cached load in another process doesn't.** With the warm-up
  in the daemon's own process, a request for an already cached bge-small
  bucket took 3.2 s to load instead of 0.85 s, because the warm-up was
  compiling another bucket, and a classifier's model load (its buckets'
  CPU-only description reads) waited 1.6 s for a compile to finish. With
  the compile in another process, a cached load took 0.06 s. A load that
  needs a compile queues behind the other process's compile either way.
- **A macOS update** starts a new cache directory, named for the build;
  that it invalidates the old entries is inferred, not observed
  (docs/design/coreml-cache.md, "OS updates").

**Decision.**
- **Warm by loading, in sidekickd.** A warm-up loads a bucket exactly as a
  request does (`CoremlChain::load` with the same `MLModelConfiguration`
  and compute units, the same program paths) and releases it: what remains
  is the cache entry, and nothing stays resident that no request asked
  for, so idle unloading (`model_idle_ttl_secs`) is unaffected. Keeping
  buckets resident from startup (#58) is separate.
- **Three triggers:**
  - **At startup, in the background,** on by default (`warm_on_start`,
    `--no-warm`). The daemon runs `sidekickd warm` as a child process: its
    own executable file, so the same name and signing identifier, with the
    same home and environment, so the same cache entries. It hands the child
    exactly the buckets and compute units its own requests would load. A
    separate process is what keeps a warm-up compile from holding up
    requests' cached loads of other buckets. A request whose bucket needs a
    compile still queues behind the compile running, as it would behind
    any process's.
  - **On demand,** `sidekickd warm [<model>…] [--buckets …]
    [--budget-secs N] [--list]`, run as the same user with the same
    sidekickd binary the daemon runs: another name misses the cache, and
    another signing identifier misses it on the ANE. A build from source
    warms only what that same build, signed the same way, will load
    (`tools/sign_sidekickd.sh`).
  - **After `sidekickd fetch`** installs a Core ML build the daemon will
    serve, for the buckets fetched (`--no-warm` skips it).
- **Only what will be served, where it will be served.** The buckets come
  from the registry, after the length caps (D33, D48) and the operator's
  compute units (D38), so a 4,096-token bucket past the ANE cap is never
  warmed, and a model moved to the GPU warms on the GPU.
  `[models."<id>"] warm = false` leaves a model out of the automatic
  warm-ups.
- **One compile at a time,** since the machine runs only one anyway, cold
  buckets first, then by bucket size, smallest first,
  since short inputs are the most common. Warm buckets are
  loaded too, at cache-hit cost, since only a load proves the cache still
  holds them.
- **A budget.** `warm_budget_secs` (default 1,800; 0 for none) stops the
  startup warm-up from starting another load after that long. A load in
  progress finishes, since Core ML can't cancel one; the buckets left
  compile on first use, as before.
- **Requests and the warm-up.** Every bucket load in a process goes through
  one place, one load of a bucket at a time, and takes a lock file per
  bucket beside the cache, so a request for a bucket the warm-up process is
  compiling waits for that compile and then loads from the cache instead
  of compiling it a second time. Requests for other buckets don't wait.
  The warm-up process exits when the daemon does (its stdin closes), even
  mid-compile.
- **Cold, warming, warm, failed.** sidekickd keeps a ledger beside Core
  ML's cache (`~/Library/Caches/sidekick/coreml-warm.json`) of the buckets
  it has loaded, keyed by what Core ML keys an entry by: the executable's
  name and signing identifier, the home, the macOS build, the compute
  units, and each program's path with its files' sizes, inodes and birth
  times (never their modification times, which the cache ignores). A key
  that counts more than Core ML's only makes a warm bucket look cold, which
  costs a cached load; one that counted less could call a bucket warm
  whose next load misses. The ledger is sidekick's own record; Core ML's
  private layout isn't read at all. `/v1/models` reports each Core ML model's buckets as
  `compile_cache` (`warm`, `warming`, `cold`, `failed`), and `/health`'s
  `warmup` the startup warm-up's progress and each bucket's load times. A
  bucket the ledger records but this daemon hasn't loaded is reported warm.
  If the user's Core ML cache was deleted since, its next load misses and
  compiles again (on the ANE, the system cache's program survives the
  deletion but isn't reused), and the startup warm-up's load finds that
  out first.

**Measured** (same machine, the release build, warm-up in a child
process). bge-small-en-v1.5 on the ANE, first request per bucket after a
restart: 4.03, 4.33 and 5.38 s cold (no warm-up) against 1.27, 1.14 and
0.92 s once warmed; bucket loads 3.4–5.3 s cold, 0.85–1.25 s warm.
nli-minilm2-l6-h768 on the ANE: bucket loads 2.1–3.8 s cold, 23–25 ms
warm; on the GPU 0.36–0.47 s cold, 0.03–0.04 s warm. During the startup
warm-up, a request for a cached bucket of another model answered in
0.34 s, and one for the bucket being compiled waited for that compile
(about 3.4 s) and loaded it from the cache in 0.87 s, without a second
compile.

**Not covered.** A compute-plan read (`report_compute_plans`) is a cache
entry of its own, a full compile (docs/design/coreml-cache.md, "The
cache key"), so warming doesn't warm it (#145).

## D53 — Open LLMs on the chat route: a stateful ANE decoder beside Foundation Models
D3 deferred a local LLM tier until there was evidence it mattered. There is
now. The chat benchmark (docs/CHAT-BENCHMARK.md, #174) found MiniCPM5-2B
level with Foundation Models on titles, fixed-vocabulary tags and
instructions, well ahead on extraction, and refusing nothing, while
Foundation Models' guided generation drops optional fields (#177) and its
guardrails refuse guided output on ordinary text (#178). The spike in #173
ran that model on the Neural Engine: four chunk programs of one 16-token
function, the KV cache in Core ML state, weights rotated and quantised by
GPTQ onto 8-bit palettes, grade B, beside a GPU model whose decode it slowed by about
6%. Slower end to end than Foundation Models or the GPU, but off the GPU,
which is what sidekick is for. The design is docs/design/open-llm.md;
this records what it settles. None of it is implemented yet.

**Decision.**
- **Open causal LMs serve `/v1/chat/completions` beside `apple-fm`,**
  chosen by the request's `model`, exactly; no default and no alias. The
  first is MiniCPM5-2B.
- **A third manifest, `chat.toml`** (`task = "text-generation"`), with a
  versioned program contract (`interface = "kv-state-1"`): N chunk programs
  at layer boundaries (D37), each one static function taking T tokens per
  call, for prompt and generation alike, and owning its layers' KV cache as
  state; a cache capacity C under the ANE length cap (D48), and a usable
  context of C − T + 1. The embedding table and RoPE tables ship as fp16
  files the host reads, with the tokenizer and the checkpoint's chat
  template. Quantisation and rotation are folded into the weights and
  recorded for provenance only.
- **`cpu_and_ne` only.** `cpu_only` with state segfaults (#173), and the
  GPU with state is ungraded, so an operator's D38 override is refused for
  this task, and a Mac without a Neural Engine skips the model.
- **The runtime is Rust:** a stateful chunk chain in `sidekick-coreml`
  (`objc2-core-ml` binds `MLState`, stateful prediction and output
  backings) and the loop in a new platform-neutral crate, `sidekick-llm`.
  Every call runs in an autorelease pool, with inputs allocated once: the
  spike's loop without one grew to a 19.7 GB footprint. One decoder thread
  per model, one sequence at a time, a bounded queue (503 `backend_busy`
  past it), one KV state reused across requests and replaced after a
  failure, a per-call watchdog for a hung ANE request. One open model is
  resident at a time, and the registry reports the ANE-resident weights of
  everything loaded, since #173 found a budget of roughly 2–3 GiB.
- **Tokenizer and template:** the `tokenizers` crate, as the embedders use
  it; the template rendered by `minijinja` with Python compatibility, gated
  against transformers' `apply_chat_template` on committed fixtures; the
  text of special tokens rewritten out of message content, as fev does.
- **Sampling and the wire:** greedy at temperature 0, temperature and
  `top_p` honored, `seed` honored; the checkpoint's sampling defaults when
  a request omits them, not greedy. Foundation Models honors `seed` and
  `top_p` too, through its random sampling mode, so both backends read the
  same fields. `request_timeout_secs` is set per model: the manifest names
  a model's default (120 s for the 2B build, whose long replies pass the
  daemon's 60 s) and a `[models."<id>"]` table overrides it. A prompt plus `max_tokens` past the context is a
  400 `context_length_exceeded`. Exact usage counts; the stop-sequence
  shaper shared with the AFM path; the route's existing streaming events,
  guided output included.
- **One JSON Schema layer for both backends,** in `sidekick-core`: it
  validates `response_format`'s schema against one subset and refuses the
  rest with a 400 `unsupported_schema` naming the keyword (#179), keeps
  `properties` in the order sent (#180; `serde_json`'s `preserve_order`),
  and keeps JSON Schema's optional-property meaning (#177's AFM mapping is
  decided by measurement). A keyword enters the subset only when both
  backends honor it. AFM's bridge receives properties as an ordered list,
  and its reply is re-serialized in schema order.
- **Constrained decoding on the open path is llguidance** (MIT, Rust),
  masking each step's logits on the host, concurrently with the next ANE
  call; the grammar's forced tokens ride in the call's padding slots.
- **No fallback between backends.** A guardrail refusal stays a 400
  `content_filter` (#178), saying whether output was guided; `apple-fm`
  still answers 503 when Foundation Models is unavailable (D3).
  `/v1/models` lists each chat model's backend, context, honored fields and
  guardrails, so a client chooses.
- **Conversion and release:** the spike's converter moves into
  `tools/sidekick_convert`; calibration is a committed, seeded fixture of
  this repository's text and revision-pinned Wikipedia leads, disjoint from
  what is graded; recipe variants are screened in PyTorch
  before any ANE compile. A build is graded teacher-forced against fp32,
  anchored on the first-percentile logit cosine and top-1 agreement rather
  than the worst position, which moved with the calibration draw, and a
  second calibration draw must grade alike. The release gate runs a slice
  of the benchmark's tasks, guided and prompt-only, and checks replies at
  temperature 0 are byte-identical. The build publishes under its D50 tier
  (Apache-2.0 model and listed training sets).

**Rejected.**
- **A multifunction package** with prefill and decode functions: its
  functions share weights on disk but not on the ANE, so both are resident
  and every switch stalls (#173).
- **A Swift runtime,** like the AFM bridge: a vocabulary-sized FFI crossing
  every step, and a second copy of the Core ML handling the encoders have
  in Rust.
- **Falling back to another model on a refusal,** silently or by
  configuration: the answer would come from a model the client didn't name,
  and D3 already left degradation to the client.
- **Grading on the worst position alone:** the same recipe went from B to C
  on another calibration draw, while the first percentile held.

**Deferred, not precluded:** speculative decoding (verifying 16 drafted
tokens costs about 1.12× a step, #173), prefix caching, `logprobs`, tool
calling and thinking, other families, a GPU path, Intel Macs, and a grade
on macOS 26 (the first build ships graded on macOS 27, and `/v1/models`
says so). Whether the ANE budget is per process or system-wide, and whether
Foundation Models shares it, is to be measured before the registry
enforces it.

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
