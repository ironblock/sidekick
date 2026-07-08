# Rust + Apple Neural Engine inference — a first-principles design review

*Status: research/design, July 2026. No code in this repo depends on it yet.*

This document reconsiders, from scratch, the idea of a Rust-based system for running
"very small" asynchronous inference tasks (session-title generation, tagging, embeddings,
entity/structure extraction) on Apple Silicon — preferring the Apple Neural Engine (ANE),
and falling back to Apple's Foundation Models when Apple Intelligence is available.

It answers three questions:

1. **Are we duplicating effort? Should we leverage existing crates?**
2. **Are Apple's Foundation Models the "best" CoreML/ANE-compatible option for general use?**
3. **Which embedding and extraction models actually fit ANE/CoreML constraints?**

---

## 0. TL;DR

- **The ANE is only reachable through Core ML.** MLX, llama.cpp, candle-metal, and burn all
  run on the GPU; the ANE sits idle under every one of them. Any Rust design that wants the
  ANE goes through Core ML — the only choices are *which bindings* and *which models*.
- **Don't write bindings; write orchestration.** `objc2-core-ml` (Core ML), a small Swift
  C-ABI shim (Foundation Models — it's a Swift-only framework, so ObjC bindings can't reach
  it), and `tokenizers` cover the plumbing. What doesn't exist as a crate is the layer this
  project should be: task→tier routing, availability probing, a model registry with shape
  buckets, and an async background queue.
- **Foundation Models is the right *generation* tier, not a general engine.** It's the best
  free option for exactly your task shape (short constrained text tasks, structured output
  via guided generation), but it's gated on Apple Intelligence, capped at 4096 combined
  tokens, Swift-only, and **exposes no embedding API**. Since WWDC 2026 the framework is
  also a pluggable front-end for any LLM provider, which strengthens the case for putting
  your abstraction *behind* a Foundation-Models-shaped interface rather than beside it.
- **Encoders are the ANE's sweet spot — and your embedding/extraction interest lands
  exactly there.** EmbeddingGemma-300m and the small BERT-family encoders (bge-small,
  MiniLM, gte-small) convert cleanly and run near-fully on ANE with fixed shapes.
  Zero-shot extraction (GLiNER) is shape-dynamic and belongs on CPU via ONNX Runtime, or —
  better when available — on Foundation Models guided generation.

---

## 1. First principles: what the ANE actually is (and isn't)

Facts that constrain every other decision:

- **Access path.** There is no public direct ANE API. The only supported route is Core ML
  with `MLModelConfiguration.computeUnits = .all` / `.cpuAndNeuralEngine`. Private
  frameworks (ANEServices/Espresso) are not viable for a distributable tool.
- **What the ANE likes.** Fixed (or enumerated) input shapes, fp16, batch size 1,
  encoder-style transformers, the conv-friendly layouts described in Apple's
  [ane_transformers](https://github.com/apple/ml-ane-transformers) work (up to ~10× faster,
  ~14× lower peak memory vs naive Core ML on CPU/GPU for transformer encoders).
- **What silently breaks it.** Dynamic shapes, int64 tensors, and unsupported ops cause
  Core ML (or the ONNX Runtime Core ML EP) to *partition* the graph and bounce
  CPU↔ANE per partition — the model still "works" but slower than pure CPU
  ([example: 14 round-trips from one op](https://github.com/microsoft/onnxruntime/issues/28022)).
  ANE residency must be *verified* (Xcode performance report, `powermetrics`), never assumed.
- **Autoregressive decode is possible but not the ANE's strength.** Projects like
  [Anemll](https://github.com/Anemll/Anemll) (MIT, beta 0.3.x) run Llama 3.2 1B/8B,
  Qwen 3 0.6B–8B, Gemma 3 270M–4B on ANE, and Core ML stateful models (macOS 15+) fixed the
  KV-cache story. The trade is consistent: roughly **half the decode tok/s of GPU at 3–8×
  lower peak memory and much lower power**. For a *background* sidekick that must not steal
  GPU/CPU from the foreground, that trade is actually favorable — but Apple's own 3B model
  via Foundation Models makes a hand-rolled ANE LLM mostly unnecessary (see §3).

**Design consequence:** the natural split is *encoders on ANE via Core ML* (embeddings,
classification, fixed-schema NER — single fixed-shape forward passes, the thing ANE is
genuinely great at) and *generation via Foundation Models* (Apple already did the
ANE-optimization work on their 3B model; you can't beat it at its own game with public APIs).

---

## 2. Ecosystem map: are we duplicating effort?

### 2.1 Rust → Core ML

| Route | What it is | Assessment |
|---|---|---|
| [`objc2-core-ml`](https://crates.io/crates/objc2-core-ml) | Mechanically generated bindings from the [madsmtm/objc2](https://github.com/madsmtm/objc2) project; covers `MLModel`, `MLMultiArray`, compute-device enumeration | **Recommended base.** Maintained as part of the broadest Apple-bindings effort in Rust; unsafe-ish but complete. Write a small safe wrapper over the ~6 types you need. |
| [`cidre`](https://github.com/yury/cidre) | Hand-crafted zero-cost bindings to many Apple frameworks; battle-tested in the author's shipping apps | Viable alternative with nicer ergonomics, but a personal project — bus-factor risk for a foundation layer. |
| [`coreml-rs`](https://github.com/swarnimarun/coreml-rs) | Small independent Core ML wrapper | Too thin/young to build on; useful as reference. |
| [`candle-coreml`](https://crates.io/crates/candle-coreml) | Core ML execution for candle tensors | Early-stage; interesting if you're already in candle-land, but it adds candle as a dependency for what is ultimately "call MLModel.prediction". |
| [`ort`](https://docs.rs/ort) + [Core ML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) | ONNX Runtime with Core ML delegation | **Escape hatch, not foundation.** Great model portability and the same crate runs on Linux/CPU, but the EP partitions on unsupported ops and needs `OnlyAllowStaticInputShapes` + fixed-shape ONNX exports to reliably hit ANE. Keep it for models that resist Core ML conversion (GLiNER). |

### 2.2 Rust → Foundation Models

Foundation Models is **Swift-only** (no ObjC headers), so `objc2` can't generate bindings.
Every existing approach — [`fm-bindings`](https://github.com/remdalm/fm-bindings),
[`rusty_foundationmodels`](https://github.com/undivisible/RUSTY_FOUNDATIONMODELS) — uses the
same pattern: a `bridge.swift` compiled by `build.rs` via `xcrun swiftc`, exposing a small
C ABI that Rust calls through `extern "C"`.

**Assessment:** the pattern is validated; the crates are young and thin. The shim is a few
hundred lines, and you will want control over its surface — especially **guided generation**
(passing a JSON-schema-like structure down to `@Generable`-equivalent dynamic schemas,
`GenerationSchema` at runtime) and **availability probing**
(`SystemLanguageModel.default.availability`, which returns *why* the model is unavailable:
device ineligible, Apple Intelligence off, model still downloading). Recommendation: **own
the shim**, borrow the build.rs pattern from `fm-bindings`. Pin the C ABI, keep the Swift
side dumb (no logic beyond marshaling).

### 2.3 What does NOT exist (the actual gap)

No crate today provides:

- a **task-tier router** ("this is an `embed` task; ANE encoder is loaded and healthy → use
  it; else static-embedding CPU floor"),
- **availability probing unified across tiers** (Apple Intelligence state + ANE presence via
  `MLComputeDevice` enumeration + model-asset download state),
- a **model registry** pairing `.mlmodelc` artifacts with their HF `tokenizers` config,
  enumerated shape buckets, pooling strategy, and expected output layout,
- a **background execution queue** with QoS semantics appropriate for "async, invisible"
  work (the ANE serializes requests anyway; coalesce, batch=1, never block a caller).

That orchestration layer is the project. **Verdict on duplication: you are not duplicating
anyone if you build that and only that.**

---

## 3. Is Foundation Models the "best" general CoreML/ANE option?

### What you get (macOS 26 "Tahoe", expanded at WWDC 2026)

- Apple's on-device **~3B parameter** model, aggressively quantized (~2-bit QAT), running on
  the ANE with **OS-managed memory** — it doesn't count against your process the way a
  self-hosted model does, and it's shared across all apps.
- **Guided generation**: constrained decoding against a schema. For extraction-shaped tasks
  this is the killer feature — a 3B model with constrained decoding beats a much larger
  model that has to be parsed-and-prayed. This is the right default for "generate a session
  title", "pick a category", "extract fields".
- **Tool calling**, **adapters** (LoRA-style, trainable with Apple's toolkit), **image
  input** (2026), and since WWDC 2026 a **pluggable provider protocol**
  ([`LanguageModel` / `LanguageModelExecutor`](https://developer.apple.com/videos/play/wwdc2026/339/))
  that lets *any* local or remote LLM sit behind the same API, plus session/response
  **token-usage reporting**.

### What you don't get

- **Availability is gated**: Apple Silicon + Apple Intelligence enabled + macOS 26+. On any
  Mac where the toggle is off, you have nothing — this is why the fallback tier is a
  requirement, not an optimization.
- **4096-token combined context** (input + output), explicitly stated as a durable limit
  for the on-device model ([Apple forums](https://developer.apple.com/forums/thread/806542)).
  Fine for titles and snippets; disqualifying for long-document work (route those to
  Private Cloud Compute — 32K context — or your own stack).
- **No embeddings.** The framework exposes generation only.
- **~3B-class quality**: excellent at constrained tasks (extract a date, pick a category,
  rewrite a sentence), poor at open-ended generation and world knowledge.
- **Swift-only API** — hence the shim tax.

### Alternatives for the generation tier, compared

| Option | Quality | Power/memory | Maintenance | Availability |
|---|---|---|---|---|
| Foundation Models (on-device) | Good for constrained tasks; guided gen | Best (OS-managed, ANE) | Zero | Gated on Apple Intelligence |
| Anemll-style ANE LLM (Llama 3.2 1B / Qwen 3 0.6–1.7B / Gemma 3 270M–1B) | Below FM's 3B; 512–1024 ctx recommended | Very good (ANE) | You own conversion + updates | Any Apple Silicon |
| llama.cpp / MLX small model (GPU) | Same models, faster decode | Worst for a background task (GPU contention, power) | Moderate | Any Apple Silicon |

**Answer: Foundation Models is the best *generation* tier when available — use it first,
always.** It is not a general engine: it can't embed, it can't do long context on-device,
and it can be absent. "Best for general use" therefore means *best first tier in a tiered
system*, which is exactly the architecture you were already circling. The WWDC 2026
provider protocol also suggests Apple expects apps to treat it as the front-end for a
pluggable stack — your Rust layer can mirror that shape.

For the generation *fallback* (Apple Intelligence off), the honest options are: (a) an
Anemll/Core-ML-converted Qwen 3 0.6B or Gemma 3 270M/1B on ANE — keeps the "invisible
background work" property; (b) skip local generation and degrade gracefully (e.g., a
heuristic title from the first user message); (c) defer to whatever `diet-inference`
provides remotely. Recommendation: ship (b) first, add (a) only if telemetry says the
FM-unavailable population matters.

---

## 4. Embedding and extraction models that fit ANE/CoreML constraints

**The constraint recap:** encoder-only, fixed or enumerated sequence lengths (bucket at
128/256/512), fp16, batch 1, standard attention (learned or absolute positions convert most
cleanly; rotary usually converts fine now but verify op coverage), tokenizer handled outside
Core ML (Rust `tokenizers` crate).

### 4.1 Embeddings — ranked fit

1. **EmbeddingGemma-300m** ([Google](https://developers.googleblog.com/introducing-embeddinggemma/),
   [HF](https://huggingface.co/google/embeddinggemma-300m)) — current best fit. Demonstrated
   **~99.8% ANE utilization** in Core ML conversion
   ([CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)), ~295 MB, multilingual, and
   **Matryoshka dims (768/512/256/128)** so you can truncate stored vectors without
   re-embedding. Best quality-per-watt of anything convertible today.
2. **bge-small-en-v1.5 / gte-small / all-MiniLM-L6-v2** — classic BERT-family encoders,
   convert to Core ML trivially (community conversions exist, e.g.
   [bge-small-en-coreml-v1.5](https://huggingface.co/michaeljelly/bge-small-en-coreml-v1.5));
   30–130 MB in fp16. MiniLM-L6 is the latency/size floor for "good enough" English
   retrieval; bge-small is the quality pick in this class.
3. **snowflake-arctic-embed-s / nomic-embed-text** — fine models; verify rotary/op coverage
   in conversion before committing. No advantage over (1) for this project.
4. **Static embeddings (model2vec-style)** — not ANE models at all: microsecond CPU lookup,
   ~30 MB. Keep one as the **unconditional floor tier** — it works on every machine
   including Intel, and for near-duplicate detection / coarse clustering it's often enough.
5. **Apple's built-in `NLContextualEmbedding`** (NaturalLanguage framework, macOS 14+) —
   multilingual BERT-style *token* embeddings with OS-managed asset download, runs on ANE,
   zero deployment cost. Retrieval quality (mean-pooled) is clearly below modern sentence
   encoders; treat it as a curiosity/fallback, not a primary. Likewise `NLEmbedding`
   (word-level) — skip.

### 4.2 Extraction — two philosophies

**(a) Generative extraction — Foundation Models guided generation.** Schema-constrained
decoding, zero model management, handles arbitrary/novel schemas. This should be the
*default* extraction path when Apple Intelligence is available. The 4096-token limit is the
main constraint — chunk or pre-trim inputs.

**(b) Discriminative encoders:**

- **GLiNER** (~205M, [urchade/GLiNER](https://github.com/urchade/GLiNER)) — zero-shot NER
  with arbitrary entity types; **GLiNER2** ([Fastino](https://pioneer.ai/blog/gliner-modern-named-entity-recognition))
  adds classification + structured extraction in one small model. The span-enumeration
  architecture is shape-dynamic, which fights the ANE; the pragmatic deployment is **ONNX on
  CPU via `ort`** (it's a 205M encoder — CPU is fast enough for background work). Don't
  burn time forcing it onto the ANE.
- **Fixed-label BERT NER fine-tunes** — if a hot path has a *stable* schema, a fine-tuned
  small encoder converts to Core ML as cleanly as an embedding model and runs fully on ANE.
  Only worth it when volume justifies the per-schema model.
- **Apple `NLTagger`** built-in NER — free, instant, but only person/place/org. Floor tier.

**Extraction recommendation:** FM guided generation → GLiNER(2) via `ort` on CPU →
`NLTagger`. Add an ANE-resident fixed-schema encoder only when a specific high-volume task
earns it.

---

## 5. Proposed architecture

```
sidekick-core        # traits + router: Task, Tier, Availability, ModelRegistry
sidekick-coreml      # safe wrapper over objc2-core-ml: load/compile cache,
                     #   shape buckets, MLMultiArray<->ndarray, ANE-residency check
sidekick-fm          # Swift C-ABI shim (build.rs + xcrun swiftc):
                     #   availability, sessions, guided generation (runtime schema)
sidekick-onnx        # optional: ort wrapper for GLiNER-class models (CPU EP)
sidekick-models      # manifest format: model id -> artifact, tokenizer.json,
                     #   shape buckets, pooling, output layout, checksums
```

**Routing table (initial):**

| Task | Tier 1 | Tier 2 | Floor |
|---|---|---|---|
| `generate.title` / `summarize.short` | Foundation Models | — (heuristic degrade) | first-line heuristic |
| `extract.structured` | FM guided generation | GLiNER2 via ort (CPU) | `NLTagger` / none |
| `embed.text` | EmbeddingGemma-300m (ANE) | bge-small / MiniLM (ANE) | static embeddings (CPU) |
| `classify.zero_shot` | FM guided (pick-from-enum) | GLiNER2 | embedding + centroid |

**Engineering notes that will bite if skipped:**

- **Compile-on-first-load:** Core ML compiles `.mlmodelc` → ANE binary on first load
  (seconds); cache is keyed by model + OS. Warm at install/startup, not first request.
- **Keep models resident.** `MLModel` load is 100 ms–1 s; a background daemon should hold
  hot models and drop them under memory pressure, not load per request.
- **Enumerated shapes, not flexible shapes.** Flexible ranges push work off the ANE; use
  `MLMultiArrayShapeConstraint` enumerated buckets (128/256/512) and pad.
- **Verify residency in CI on a Mac runner**: run once with `.cpuOnly` vs `.all` and assert
  a latency ratio, or parse the Core ML performance report. Silent CPU fallback is the
  failure mode of this entire design.
- **Availability is a state machine, not a boolean**: Apple Intelligence can be toggled,
  models download lazily, ANE presence varies. Probe at startup, subscribe where possible,
  and re-route per request.

---

### Addendum (post-review, July 2026)

Two directives were settled after review and now bind the implementation:

1. **The daemon is an OpenAI-compatible server** (`sidekickd`): chat
   completions + embeddings + `/v1/models` + health, with TTL'd session reuse
   — llama.cpp/MLX-server in miniature, so consumers like OpenCode work
   unmodified. The library/daemon split stands; the daemon is the first
   consumer.
2. **macOS 26 (Tahoe) is the API baseline.** The macOS 27 provider-protocol
   surface is deliberately unused until corporate deployment makes it
   testable. See docs/DECISIONS.md for the full judgment-call log.

## 6. Open questions

1. **Generation fallback**: is "no local generation, heuristic degrade" acceptable when
   Apple Intelligence is off, or does the `diet-inference` remote path cover that
   population? (Determines whether an Anemll-style tier is ever built.)
2. **Process model**: in-process library vs. a small daemon (XPC-ish) owning resident
   models. Daemon wins on model residency and memory accounting; library wins on simplicity.
3. **WWDC 2026 provider protocol**: worth tracking whether Apple's `LanguageModel`
   protocol becomes the de-facto abstraction — if the host app is Swift-adjacent, the Rust
   layer could *implement* a provider rather than wrap the framework.
4. **Embedding version pinning**: embeddings are only comparable within one model+revision;
   the registry must stamp vectors with model id + dims (Matryoshka truncation included).

---

## Appendix: sources

- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) · [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers)
- [On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) (stateful KV cache)
- [Apple Foundation Models 2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates) · [Tech report (arXiv 2507.13575)](https://arxiv.org/pdf/2507.13575)
- [WWDC26: What's new in Foundation Models](https://developer.apple.com/videos/play/wwdc2026/241/) · [WWDC26: Bring an LLM provider to Foundation Models](https://developer.apple.com/videos/play/wwdc2026/339/)
- [4096-token limit discussion (Apple Developer Forums)](https://developer.apple.com/forums/thread/806542)
- [objc2 / objc2-core-ml](https://github.com/madsmtm/objc2) · [cidre](https://github.com/yury/cidre) · [coreml-rs](https://github.com/swarnimarun/coreml-rs) · [candle-coreml](https://crates.io/crates/candle-coreml)
- [fm-bindings](https://github.com/remdalm/fm-bindings) · [rusty_foundationmodels](https://github.com/undivisible/RUSTY_FOUNDATIONMODELS)
- [ONNX Runtime Core ML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) · [partition round-trip issue #28022](https://github.com/microsoft/onnxruntime/issues/28022)
- [Anemll](https://github.com/Anemll/Anemll) · [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)
- [EmbeddingGemma announcement](https://developers.googleblog.com/introducing-embeddinggemma/) · [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
- [bge-small-en-coreml-v1.5](https://huggingface.co/michaeljelly/bge-small-en-coreml-v1.5)
- [GLiNER](https://github.com/urchade/GLiNER) · [GLiNER2 (Fastino)](https://pioneer.ai/blog/gliner-modern-named-entity-recognition)
