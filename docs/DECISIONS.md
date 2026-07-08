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

## D8 — `response_format` mapping
`json_schema` → guided generation (the shim converts a JSON Schema subset —
object/string/integer/number/boolean/enum/array/nested objects/required — to
`DynamicGenerationSchema`). `json_object` → prompt nudge only, since there's
no schema to constrain against. Unsupported schema keywords fail loudly in
the shim rather than being silently dropped.

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

## D17 — EmbeddingGemma ships ANE-default at ~0.990 parity
Gemma3's 300m encoder needed real conversion engineering
(tools/convert_embeddinggemma.py): a calibrated power-of-two fp16 range
rewrite (the residual stream reaches ~1.5e5, past fp16 max — scale-invariant
RMSNorm rewrites make it exact; fp32 parity gates at 1.000000), hand-built
attention masks (transformers halves config.json's sliding_window to 257
for bidirectional models; the 512 bucket has a live band, so the parity
gates include a ~400-token text), and shape-arithmetic-free rotate_half /
repeat_kv rewrites for coremltools' static-shape 'int' op crash.

After all of that, the ANE itself costs ~1% cosine — intrinsic fp16
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
  required). Objc exceptions from Core ML (e.g. its E5RT/IOSurface
  failures) abort the process — Rust cannot catch them; the fix is
  converting models that don't provoke them (D15), not catching.

- EmbeddingGemma-300m end-to-end (July 2026): conversion via
  tools/convert_embeddinggemma.py (D17), ANE residency 3.4x/3.1x/2.9x at
  buckets 128/256/512 via ane_check, server /v1/embeddings parity 0.9905
  vs fp32 sentence-transformers (matching the ANE gate exactly — the Rust
  tokenizer path is token-identical), matryoshka dimensions 512/256/128
  with unit norms and 400 on undeclared dims, query/document prefixes,
  and a 831-token input through the 512 bucket (47ms warm).

Still open:
- An automated ANE-residency gate in a self-hosted CI job (the example
  exists; nothing runs it automatically).
- Multifunction mlprogram weight sharing to collapse the 3x ~600MB
  per-bucket artifact duplication for large encoders (D17).
