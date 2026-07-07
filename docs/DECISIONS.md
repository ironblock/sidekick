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

## Needs hardware verification (blocking v0.2)
- `swift/bridge.swift` compiles and behaves against the real macOS 26 SDK
  (exact `DynamicGenerationSchema`/`GenerationOptions` initializer shapes are
  the risk; everything else is plain Foundation).
- `sidekick-coreml` runtime behavior (compiles clean against `objc2-core-ml`
  0.3.2 via cross-check; predictions untested).
- ANE residency check for a converted EmbeddingGemma/bge artifact
  (`.cpuOnly` vs `.cpuAndNeuralEngine` latency ratio, per design doc §5).
