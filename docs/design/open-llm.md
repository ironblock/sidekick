# Open LLMs on the chat route, beside Foundation Models

How sidekickd serves a small open causal language model from
[`POST /v1/chat/completions`](../API.md#chat), next to Apple's Foundation
Models (AFM): the model format and its manifest, the runtime, tokenization
and chat templates, sampling and the wire contract, structured output,
routing between the two backends, and how a build is converted, graded and
published. [D53](../DECISIONS.md#d53--open-llms-on-the-chat-route-a-stateful-ane-decoder-beside-foundation-models)
records the decision; this note has the detail.

Implemented so far: the decoder chain (`sidekick_coreml::KvDecoder`,
#183), the loop and the chat route's open-model backend (`sidekick-llm`,
#185): template, tokenizer, sampling, the wire contract, the ANE chat lane
and `/health`'s accounting; constrained decoding on the open path with
llguidance (#186); and the converter (#184,
[CONVERTING.md](../CONVERTING.md#causal-language-models)). Where this note
states a number, the measurement is linked, not repeated:
- [#173](https://github.com/ironblock/sidekick/issues/173): the spike that
  ran a 1B and then a 2B decoder on the Neural Engine (ANE), with its KV
  cache in Core ML state. Its comments hold the speeds, grades and Core ML
  hazards this note relies on. The spike's converter and reference runtime
  are on the branch `spike/ane-decoder-2b`, in `tools/spike_ane_decoder/`.
- [CHAT-BENCHMARK.md](../CHAT-BENCHMARK.md): Foundation Models against
  small open models on titles, tags, extraction and instructions. It is
  why the first model is MiniCPM5-2B.
- [#177](https://github.com/ironblock/sidekick/issues/177)–[#180](https://github.com/ironblock/sidekick/issues/180):
  the chat route's guided-generation issues that the benchmark found.

## Contents

- [Why, and the principles](#why-and-the-principles)
- [Decisions at a glance](#decisions-at-a-glance)
- [The model format](#the-model-format)
- [The runtime](#the-runtime)
- [Tokenizer and chat template](#tokenizer-and-chat-template)
- [Sampling and the wire contract](#sampling-and-the-wire-contract)
- [Structured output](#structured-output)
- [Routing](#routing)
- [Conversion and release](#conversion-and-release)
- [Deferred, not precluded](#deferred-not-precluded)
- [Open questions](#open-questions)

## Why, and the principles

sidekick runs a small model off the GPU, beside a large model that owns the
GPU. Its chat route has served only Foundation Models, which is absent on a
Mac without Apple Intelligence (D3), and which the benchmark found weak at
extraction: guided generation drops optional fields (#177) and its
guardrails refuse ordinary text when output is guided (#178). MiniCPM5-2B
matched it on titles, fixed-vocabulary tags and instructions, and beat it
on extraction ([Recommendation](../CHAT-BENCHMARK.md#recommendation)). On
the ANE the 2B build is slower end to end than AFM or the GPU, but it
leaves a GPU model's
decode almost untouched (#173). That trade is the product: choices here are
judged by what they cost the GPU's model and the machine's power first, and
by the small model's own latency second.

- **One route, one contract.** Both backends answer the same OpenAI
  request and return the same shapes. A difference a client can observe is
  either removed or reported in `/v1/models`.
- **Fail loudly (D22).** A request field, schema keyword or length a model
  can't honor is a 400, on both backends alike.
- **No silent substitution.** The model a client names answers, or the
  request fails with the reason.
- **Reuse what the encoders proved.** Static shapes (D15), chunks under the
  ANE weight limit (D32, D37), the ANE length cap (D48), compute units
  (D38), warm-up (D52), the parity suite's grading (D26), the ledger and
  fetch (D50, D51).

## Decisions at a glance

| Area | Decision | Rejected |
|---|---|---|
| Format | A `chat.toml` manifest with `task = "text-generation"` and a versioned program contract: N chunk programs, each one function taking T tokens per call, with its layers' KV cache as Core ML state | A multifunction package with prefill and decode functions (doubles resident weights on the ANE, stalls on every switch); one program (over the 1 GiB limit) |
| Host assets | Embedding table and RoPE tables shipped as fp16 safetensors, mapped, not computed; tokenizer.json; the checkpoint's chat template | An in-program embedding (vocabulary-sized gather, weights on the ANE); RoPE computed at load (libm differences) |
| Compute units | `cpu_and_ne` only; Intel Macs and Macs without an ANE skip the model | `cpu_only` (MLState crash), `all` and `cpu_and_gpu` (ungraded with state) |
| Runtime | Rust, in `sidekick-coreml` (the decoder) and a new platform-neutral `sidekick-llm` crate (the loop) | A Swift shim like `sidekick-fm`'s: an FFI crossing per token, and a second Core ML loader |
| Memory | An autorelease pool around every call; preallocated inputs; output backings if measured equal | Relying on the thread's pool (the spike's loop reached a 19.7 GB footprint) |
| Concurrency | One decoder thread per model, a bounded FIFO queue shared with `apple-fm` (one ANE chat lane), one KV state reused across requests | A state per request; batching; evicting models to fit the ANE budget |
| Tokenizer | The `tokenizers` crate, as the embedders use it | — |
| Template | `minijinja` with Python compatibility, gated against transformers | Hand-written templates per family |
| Sampling | Greedy at temperature 0; temperature and top-p; `seed` honored | Penalties, `n` > 1, `logprobs` (deferred) |
| Structured output | One JSON Schema layer for both backends; llguidance masks each step's logits on the open path | Per-backend schema handling (today's drift: #179, #180) |
| Routing | By the request's `model`, exactly; no fallback between backends | Silent or configured fallback on a guardrail refusal |
| Grading | Teacher-forced KL divergence from fp32 (mean and p99) with top-1 as a floor, the worst evaluation family's letter, both calibration draws in one band, and a task check against fp32 (D54); a #174 task slice in the release gate | The logit cosine and wide-flip cap the encoders use (a few low-confidence positions decided it); grading on the worst position alone |

## The model format

### The program contract

A build is N chunk programs over contiguous runs of layers (D37). Every
chunk is a single-function ML program (`main`) with static shapes, called
with T tokens at a time, for the prompt and for generation alike. The
contract is named in the manifest (`interface = "kv-state-1"`), so a later
contract can change it and a daemon that doesn't know it skips the model.

| Name | Shape (dtype) | Chunks | Meaning |
|---|---|---|---|
| `hidden_in` | `[1, D, 1, T]` (fp16) | all | The residual stream, channels first. For chunk 0 it is the host's embedding lookup; after that, the previous chunk's `hidden_out`, passed without a copy (D37) |
| `pos` | `[1]` (int32) | all | The cache position of the call's first token |
| `mask` | `[1, 1, T, C]` (fp16) | all | Additive causal mask over the whole cache: 0 where key ≤ `pos` + query index, −30,000 elsewhere (finite, D25) |
| `cos`, `sin` | `[T, head_dim]` (fp16) | all | RoPE rows for positions `pos` … `pos` + T − 1 |
| `sel` | `[T]` (fp16) | last | One-hot: which of the T positions' logits to return |
| `hidden_out` | `[1, D, 1, T]` (fp16) | all but last | The residual stream after this chunk's layers |
| `logits` | `[1, V]` (fp16) | last | One row, chosen by `sel`, after the final norm and the head |
| state `k_cache`, `v_cache` | `[layers in chunk, kv_heads, C, head_dim]` (fp16) | all | This chunk's KV cache, written with `slice_update` at `pos` |

**Why one T-token function, T = 16.** #173 found that a multifunction
package shares weights on disk but not on the ANE: loading a prefill and a
decode function doubles the resident weights, and every switch between them
stalls. One function serves both. A call costs only a little more as T
grows, so one function at T = 16 gives a prompt rate about 16 times the
per-token rate, for a modest cost per generated token (#173's third
comment has the measurement). The same width serves speculative
verification and llguidance's forced tokens later at no extra cost. T is a
manifest value, so a later build may choose another.

**Why the cache is static.** The ANE runs static shapes (D15). The cache
capacity C is the attention width every call reads, so a call's cost
depends on C, not on the position (#173). C is judged by D48's cap as a
bucket's length is: a manifest whose C exceeds the model's ANE cap
(`DEFAULT_ANE_MAX_SEQ`, 2,048, or `ane_max_seq_len`) is skipped with the
reason. MiniCPM5-2B's build uses C = 2,048.

**The usable context is C − T + 1.** A call writes T cache positions,
`pos` … `pos` + T − 1, so its first token can sit no later than C − T. The
manifest states the result as `context_length` (2,033 for C = 2,048,
T = 16), and the runtime enforces it.

### The manifest

`chat.toml`, beside `manifest.toml` (embedders) and `classifier.toml`
(classifiers). As with the other manifests, the converter copies a
reviewed, committed file and fills in what the build measured.

```toml
id = "minicpm5-2b"
task = "text-generation"
source = { repo = "openbmb/MiniCPM5-2B", revision = "<sha>" }
interface = "kv-state-1"
compute_units = "cpu_and_ne"         # the only value this interface accepts
tokenizer = "tokenizer.json"

[decoder]
artifact = "decoder.{chunk}.mlmodelc"
chunks = 4
tokens_per_call = 16                 # T
cache_capacity = 2048                # C: at most the ANE cap (D48)
context_length = 2033                # C − T + 1: prompt plus completion
hidden_size = 2048
head_dim = 128
kv_heads = 2
layers = [11, 11, 10, 10]            # per chunk: the state's first dimension
vocab_size = 130560

[decoder.host]
embedding = "embedding.safetensors"  # one tensor, [vocab, hidden] fp16
rope = "rope.safetensors"            # cos and sin, [C, head_dim] fp16

[chat]
template = "chat_template.jinja"
template_kwargs = { enable_thinking = false }
bos_token = "<from tokenizer_config.json>"   # for the template
eos_token = "<from tokenizer_config.json>"
stop_token_ids = [1, 130073]         # generation_config.json's eos_token_id

[sampling]                           # generation_config.json: used when a request omits them
temperature = 1.0
top_p = 0.95

[serving]
request_timeout_secs = 120           # this model's default; [models."<id>"] overrides it

[build]                              # written by the converter; provenance, not read to serve
weights = "palette8"                 # GPTQ onto 8-bit palettes, one per group of 16 output channels
rotation = { r1 = true, r2 = true, seed = 0 }
calibration = { set = "<fixture>", sha256 = "<sha>", tokens = 40960, seed = 0 }
residual_scale = 8                   # K: the residual stream runs at 1/K, folded into the embedding table
fp16_projections = []
```

The values above are illustrative where the converter decides them (the
layer split, the residual scale). Validation refuses a manifest whose
programs' declared inputs, outputs or state shapes don't match the table
above, as `classifier.toml`'s check does for classifiers, and checks each
chunk against the ANE weight limit (D32). A Core ML install gains the
`[placement]` record (D35) as other models do.

### Host assets

- **The embedding table** is a file of its own, `[vocab, hidden]` fp16 in
  safetensors, memory-mapped. One row is one 4 KiB page at MiniCPM5-2B's
  width, so a lookup touches one page per token and the table's resident
  share stays small. The program has no gather and no vocabulary-sized
  constant beyond the head. The host table is never the head's weights,
  even for a model with tied embeddings: the rotation folds the final
  norm's gain into the head, the table carries the residual stream's
  1/K, and the head is quantised in the program. (MiniCPM5-2B's
  embeddings aren't tied in any case: its config sets
  `tie_word_embeddings: false`.)
- **RoPE tables** are shipped, not computed at load: 1 MiB at C = 2,048. A
  table computed on the Mac would round fp64 cosines from a different libm
  than the converter's, and a one-ulp difference can move an fp16 value.
  Shipped, the graded and the served tables are the same bytes, hashed by
  the ledger. `rope_scaling` other than none is refused by the converter in
  this interface.
- **The mask** is built per call on the host (64 KiB at T = 16, C = 2,048):
  it is a function of `pos` alone.
- **The tokenizer and chat template** are the checkpoint's own files (see
  [Tokenizer and chat template](#tokenizer-and-chat-template)).
- **Quantisation and rotation** need nothing at serve time: both are folded
  into the weights. The `[build]` table records them for reproduction and
  the model card.

### Compute units

The interface accepts `cpu_and_ne` only, and refuses an operator's D38
override to anything else for this task, with the reason:
- **`cpu_only` crashes.** #173 found that programs sharing an `MLState`
  segfault within 10–130 decode steps on the CPU, reproducibly, in pure
  Swift.
- **`cpu_and_gpu` and `all` are ungraded with state.** A GPU path is
  deferred ([below](#deferred-not-precluded)); serving one on the GPU is
  also against the point of the model.
- **A Mac without a Neural Engine** (an Intel Mac) runs `cpu_and_ne` on the
  CPU, so the registry skips the model there, with the reason, as it
  already detects the ANE (`ScanOptions::no_neural_engine`, D48).

The converter's compute-plan gate (D24) must find every operation of every
chunk on the ANE, as #173's builds did, since an operation Core ML puts on
the CPU under `cpu_and_ne` would run the state there too.

## The runtime

### Where the decode loop lives: Rust

`sidekick-coreml` already calls Core ML from Rust through `objc2` and
`objc2-core-ml`, which bind everything a stateful decoder needs:
`MLModel.newState`, `predictionFromFeatures:usingState:options:error:`,
`MLPredictionOptions.outputBackings`, and `objc2::rc::autoreleasepool`.
- **`sidekick-coreml` gains the decoder:** load a model's chunks with the
  same loader, cache ledger and lock files the encoders use (D52), allocate
  one `MLState` per chunk, and run one T-token call through the chain,
  returning one logits row. It also gains fp16 inputs, which it doesn't
  have today.
- **A new crate, `sidekick-llm`, holds the loop** and is platform-neutral:
  template, tokenizer, sampler, constraint, stop sequences, the queue, and
  a `ChatBackend` implementation over a `Decoder` trait. Its tests run
  everywhere against a mock decoder, as `sidekick-fm`'s run against a mock
  engine.

Swift, as the spike's `skrun.swift` and the `sidekick-fm` bridge use, was
rejected. The host work per step (sampling, the llguidance mask, stop
matching, detokenizing) is Rust, so a Swift loop would cross the FFI with
a vocabulary-sized row every step, or move all of it into Swift. And the
encoders' Core ML handling (warm-up, the cache ledger, the symlink refusal,
the weight limit) is Rust and would be duplicated.

**macOS floor.** `MLState` needs macOS 15, under D2's baseline of macOS 26.
The first build is graded on macOS 27 only. It is served on macOS 26 once
graded there; until then `/v1/models` reports the OS it was graded on, as
models.json does for every build.

### One call, T tokens

- **The prompt** goes in T-token calls at positions 0, T, 2T, …; the last
  call is padded. Only the last call's logits are used, at `sel` = the last
  real token's index. The earlier calls' logits rows are computed and
  discarded: the head is in the last chunk, and skipping it would need a
  second function.
- **Generation** is one call per token: the new token at index 0, the
  other T − 1 slots padded, `sel` = 0. Llguidance's forced tokens, when
  there are any, fill those slots instead (see
  [Structured output](#structured-output)), and `sel` picks the last one.
- **Padding** is token 0's embedding at the following positions. Its keys
  and values land in the cache at positions no real query attends to: the
  mask hides every key past the query's own position, and the next call's
  real tokens overwrite them. The spike's reference runtime does exactly
  this (`MULTI=T`).
- **Inputs are allocated once** per decoder and rewritten in place before
  each call; the call is synchronous, so they outlive it. #173 found the
  ANE path stable only while each call's inputs were kept alive.

### Releasing Core ML's memory

The spike's long decode loop reached a 19.7 GB footprint against a 3.2 GB
resident set: Core ML's per-call outputs were autoreleased into a pool that
never drained. So:
- **Every call runs inside `autoreleasepool`**, and the logits row is copied
  to a host buffer before the pool drains. Rust threads have no pool of
  their own that drains between calls, so this is not optional. It belongs
  in the shared prediction path, where the encoders benefit too.
- **Output backings** (`MLPredictionOptions.outputBackings`): preallocated
  `hidden_out` and `logits` arrays that Core ML writes into. They remove the
  per-call allocations, and are adopted if the runtime, once built,
  measures outputs bit-identical and a step no slower.
- **The acceptance test** is a flat footprint (`phys_footprint`) over 10,000
  steps.
- **Loads need a pool too.** Core ML autoreleases the model while it loads;
  left to a Rust thread's pool, a model its owner dropped stayed loaded,
  its programs still on the ANE, until the thread exited. `sidekick-coreml`
  drains a pool around every load and every prediction, the encoders'
  included.

Measured with the runtime (#183; its pull request has the numbers): the
2B build's footprint held flat over 10,000 calls. Leaving the pool out
grew even a toy decoder's footprint, by 440 MB over 5,000 calls without
output backings (about 88 KB a call) and by 1.8 MB with them (about 360
bytes a call). Output backings
gave bit-identical logits at the same speed, so the decoder uses them: no
call allocates an output or a feature provider. Core ML may decline a
backing, so the decoder checks each chunk's returned output against its
backing and, the first time it differs, carries the returned array and
stops using backings.

### Concurrency and state

- **One decoder thread per loaded open model** owns its programs, states
  and buffers; requests reach it over a channel. A dedicated thread, not
  tokio's blocking pool: `MLState` isn't documented as thread-safe, and the
  thread's pool and buffers stay with it.
- **One sequence at a time.** Batching would multiply the cache and change
  the program's static shapes, and a companion model's requests are short
  and few.
- **A bounded FIFO queue,** `[models."<id>"] queue_depth` (default 4;
  the lane shared with `apple-fm`, below, takes the largest set for an
  installed open model).
  A request past it is a 503 `backend_busy` with `Retry-After`, as AFM's
  busy state is. Time in the queue counts against the request's timeout.
- **A timeout per model.** `request_timeout_secs` bounds a request's whole
  work, streamed or not, as it does for Foundation Models. At the 2B
  build's rate a reply near `context_length` passes the daemon's 60 s
  default, so the manifest's `[serving] request_timeout_secs` sets the
  model's default (120 s for the 2B build) and a `[models."<id>"]` table
  overrides it, as it overrides `compute_units`. The global setting stays
  the default for models whose manifest names none.
- **One KV state, reused across requests, not reset.** A request writes its
  prompt from position 0 and the mask hides everything past its own
  position, so a previous request's keys never reach it. The state is
  replaced (`newState`) after any failed call or a non-finite logit, since
  a non-finite value in a hidden key could poison a later request through
  its zero attention weight. On the ANE that is a guard rather than a path
  taken: #183 fed it an infinite and a NaN residual stream, and the logits
  came back as zeros, never non-finite. A fresh state costs little: on the
  2B build, `newState` took under 0.5 ms and the first call on it 63 ms
  against 54 ms (#183).
- **No prefix caching in the first release** ([deferred](#deferred-not-precluded)).
  Reusing the state makes it a runtime change: the decoder keeps the token
  ids its state holds, and a request that shares a prefix starts at the
  first position that differs.
- **Cancellation.** A client that disconnects, or a request past its
  timeout, stops at the next step. A call can't be cancelled, so a step
  finishes first.
- **A hung call.** An ANE request can hang for about 100 s and leave every
  ANE prediction on the machine failing (docs/design/coreml-cache.md, "A
  hung ANE request"). A per-call watchdog (60 s, as the spike's runtime
  used) marks the decoder failed: queued and later requests get a 503 with
  the reason, and `/health` reports it. The thread can't be killed, so
  recovery is a restart.

### Memory budget

For MiniCPM5-2B at C = 2,048 (the formulas carry to other builds):

| Item | Size | Where |
|---|---|---|
| Weights, 4 chunks | 2.12 GiB (#173) | the ANE's resident set; mapped by Core ML |
| Embedding table | 510 MiB on disk (130,560 × 2,048 × fp16) | mapped; one page per distinct token touched |
| KV state | 84 MiB (42 layers × 2 KV heads × 128 × K and V × fp16 × C) | Core ML state |
| Logits row | 255 KiB fp16, 510 KiB as f32 on the host | per call |
| Inputs | under 200 KiB (mask 64 KiB, hidden 64 KiB, RoPE rows, `sel`) | allocated once |
| Tokenizer, token trie, grammars | tens of MB, to be measured | host |

**The ANE's budget is shared by the whole machine.** #187 measured it on
an M1 Max (macOS 27) beside the 2B build, with `tools/ane_budget_probe.py`
(re-measure with it on other hardware or builds):
- **It is system-wide.** A second process running the same build slowed
  both decoders about 21×, as #173 saw from 4.2 GiB in one process.
- **Foundation Models counts against it.** Its model takes about 2.3 GiB.
  With `apple-fm` requests running, a decoder call took 1.2 s instead of
  54 ms, and a 37-token AFM reply 32–39 s instead of 1.3 s. Any overlap
  thrashes both for its whole duration.
- **Only programs in active use count.** Up to 2.79 GiB of active programs
  the decoder only shares the ANE's time (54 ms per call became 56–106 ms
  beside busy embedders); from 2.88 GiB programs reload on nearly every
  call, about 20× slower on both sides. Loaded but idle programs cost
  nothing: the system evicts the least recently used one on the next
  load, and it returns in 0.1–1.2 s.
- **A program's ANE size is its weights minus the token-embedding table,**
  which runs on the CPU (the 2B decoder: 2,185 MiB). A process can read
  its own total (`ri_neural_footprint` from `proc_pid_rusage`).

So the daemon schedules rather than evicts. Evicting its own models gains
nothing while idle ones are free, and can't reach Foundation Models or
other apps.
- **One ANE chat lane.** `apple-fm` and the open chat models share one
  queue: a Foundation Models request waits while an open model generates,
  and the reverse. Each would otherwise slow the other about 25×. The lane
  has the open model's `queue_depth`. When no open model is installed,
  `apple-fm` runs as it does today.
- **One open chat model is resident at a time.** Two decoders pass the
  knee. Several may be installed; a request for another waits for the
  resident one to unload.
- **The registry accounts for it.** `/health` reports the daemon's ANE
  total as the process reads it, and each loaded Core ML model's share
  (estimated before a load as its weights minus its embedding table). The
  daemon logs a warning when a load would take the total past
  `ane_budget_mib` (default 2,816, 2.75 GiB, under the 2.79 GiB measured
  safe; measured on an M1 Max only).
- **Embedders and classifiers aren't held.** An occasional request that
  takes the active total past the knee costs a stall of up to about 1.7 s
  on each side; a continuous load past it thrashes. An operator who serves
  large ANE embedders beside an open model can move them to the GPU
  (D38's `compute_units`). Holding them while a generation runs is
  deferred until a deployment needs it.
- **Contention from outside is reported, not prevented.** Other apps'
  Foundation Models or ANE use slows a generation the same way, and the
  daemon can't stop it. The decoder counts calls over five times its
  warm median as stalls; `/health` reports recent stalls, and the request
  timeout bounds the worst case.

### Loading, warm-up and where it doesn't run

- **Warm-up (D52)** loads the chunks exactly as a request does, in the child
  process, like any other Core ML model; the decoder is one "bucket" of N
  chunks. The spike's first loads, compiles included, took tens of seconds
  (#173), well inside the default budget.
- **A load ends with one call**, and then reads the process's ANE
  footprint (`ri_neural_footprint`). Core ML runs a program it judges too
  small for the ANE on the CPU, with no error; #183 measured that for one
  layer at width 256 and anything at width 64. The decoder is refused unless
  the footprint grew by at least half its programs' weights. The footprint
  is the whole process's, and a load can evict the process's other idle ANE
  programs, so only a process holding nothing else on the ANE reads it
  conclusively: the warm-up child (D52) is where the check holds, and the
  serving process can load with the lenient check, which refuses only a
  process holding no ANE program at all. The growth is the decoder's share
  of the ANE budget. A strict refusal is recorded in the warm-up ledger for
  that build, and the daemon then refuses the build too, with the reason;
  the daemon itself loads with the strict check whenever it holds nothing
  on the ANE, as after a model switch, which waits for the old decoder's
  thread to exit.
- **Idle unloading** (`model_idle_ttl_secs`) applies, which also frees the
  ANE's resident budget for other models.
- **Where it doesn't run** — an Intel Mac, a Mac without an ANE, a manifest
  past the ANE cap — the registry skips the model with the reason, and
  `/v1/models` doesn't list it. `apple-fm` is unaffected.

## Tokenizer and chat template

- **Tokenizer:** the `tokenizers` crate (0.21, pure-Rust regex, D12), from
  the checkpoint's `tokenizer.json`, as the embedders and classifiers use it.
- **Template:** the checkpoint's own Jinja template, rendered by
  [`minijinja`](https://github.com/mitsuhiko/minijinja) (Apache-2.0) with
  `minijinja-contrib`'s Python compatibility (`pycompat`): Hugging Face
  templates call Python string methods (`startswith`, `split`, `strip`,
  `replace`), which MiniCPM5's template does. Hand-written templates per
  family were rejected: each would be a code change, and would drift from
  the checkpoint's.
- **Gated against transformers.** Fixture conversations (system and
  multi-turn ones included) are committed with the text and token ids
  `apply_chat_template` gives for them (`fixtures/chat-templates/`, written
  by `tools/make_chat_template_fixture.py`), and a Rust test
  (`crates/sidekick-llm/tests/chat_templates.rs`) renders and tokenizes
  each and compares: the text always, the ids when the checkpoint's
  tokenizer is in the Hugging Face cache. A template the renderer can't
  reproduce fails there, not at the request. MiniCPM5-2B's template,
  SmolLM2's (a second family) and a hand-written Llama 3 style template
  (`tools/chat_template_fixtures/`: `tojson`'s arguments and
  `{% generation %}`) reproduce exactly. A model whose tokenizer or template
  can't be used is skipped at startup with the reason.
- **What the template gets:** the messages, `add_generation_prompt = true`,
  the manifest's `bos_token` and `eos_token`, and its `template_kwargs`.
  MiniCPM5 renders with `enable_thinking = false`, as the benchmark ran it.
  No `tools`: tool calling stays a 400 (D22).
- **Special tokens in message text.** The rendered prompt is tokenized with
  special tokens parsed, so a message containing `<|im_start|>` could forge
  a turn. Before rendering, the text of every special token the tokenizer
  defines is rewritten in message content so it no longer matches, as the
  fev format does for its delimiters (`<|name|>` → `<¦name¦>`; any other
  special token gets a word joiner after its first character, and a
  tokenizer with a one-character special token is refused). Ordinary text
  tokenizes exactly as transformers tokenizes it. As a second line, a
  prompt holding any special token more or fewer times than the template
  alone writes it is a 400: text a normalizer turns into a special token,
  or one formed across a message's edge.
- **What transformers adds to jinja2** is reproduced: `raise_exception`,
  `strftime_now` (local time), a `tojson` that writes Python's
  `json.dumps` with its arguments and no HTML escaping, and the
  `{% generation %}` block, which renders its body. Mappings keep their
  keys' order, as Python's dicts do.
- **Roles:** `system` and `developer` are the system role, as on the AFM
  path; text-only content parts, as today.

## Sampling and the wire contract

The request and response shapes are the chat route's
([API.md](../API.md#chat)); the open path fills them as follows.

| Field | Open models | AFM, for comparison |
|---|---|---|
| `temperature` | 0 is greedy argmax, ties to the lower token id; otherwise softmax at that temperature. Omitted: the manifest's `[sampling]` (the checkpoint's `generation_config.json`) | passed to Foundation Models; omitted, its own default |
| `top_p` | honored (nucleus over the masked distribution) | honored: Foundation Models' random sampling mode with that probability threshold |
| `seed` | honored: a seeded ChaCha generator makes sampling at temperature > 0 repeatable | honored: the sampling mode's seed |
| `max_tokens`, `max_completion_tokens` | a limit on generated tokens; omitted, the rest of `context_length` | as today |
| prompt + `max_tokens` past `context_length` | 400 `context_length_exceeded`, as vLLM and OpenAI answer | 400 for an over-long prompt (D21) |
| `stop` | up to four strings, matched on the detokenized text; the reply ends before the earliest | as today |
| end of reply | a `stop_token_ids` token: `finish_reason: "stop"`; `max_tokens` or the context's end: `"length"` | as today |
| `stream` | text as generated, guided JSON included | guided JSON arrives whole |
| `usage` | exact: `prompt_tokens` counts the rendered template, `completion_tokens` every generated token, the stop token included; `cached_tokens` 0 | real on macOS 27 (D21) |
| `n` > 1, `tools`, `logprobs`, `top_logprobs` | 400 (D22) | 400 (D22) |
| penalties, `user` | ignored (D22) | ignored (D22) |

- **Streaming** uses the route's existing events: a role delta, content
  deltas, a finish chunk with `constrained`, a usage chunk when
  `stream_options.include_usage` is set, and `[DONE]`. Text is emitted only
  as complete UTF-8, and text that could still become a stop sequence is
  held back. `sidekick-fm`'s stop-sequence shaper (`shaping.rs`) moves to
  `sidekick-core`, so both backends cut text the same way.
- **Defaults are the model's.** A request that omits `temperature` or
  `top_p` samples at the checkpoint's own defaults, as OpenAI's API and
  Foundation Models do; greedy is not the default. A client that wants
  repeatable output sends `temperature: 0`, as the benchmark does.
- **Determinism.** At temperature 0 a request gives the same bytes on the
  same build, Mac and macOS, guided or not; the release gate checks it.
  With a `seed`, so does sampling. Prefix caching, when it comes, must keep
  this or say where it doesn't: a prompt split into different calls can
  round differently.
- **Thinking.** The template renders with thinking off, so
  `reasoning_tokens` is absent. A thinking mode, if wanted, is a later
  extension.

## Structured output

### One schema layer for both backends

Today each backend reads the request's JSON Schema its own way, and the
Swift bridge's reading is where #179 (unsupported forms answered as an
empty object) and #180 (keys in another order each run) come from: it reads
`type` as a string with `"object"` as the fallback, ignores keywords it
doesn't know, and sorts property names. D8 meant unsupported keywords to
fail loudly; this layer is how they will.

**A schema module in `sidekick-core`** parses `response_format.json_schema`
into an ordered, validated representation of one subset, or refuses it with
a 400 `unsupported_schema` naming the keyword and its JSON pointer. Both
backends consume that representation; neither reads the request's schema
directly.

| Keyword | First release | AFM mapping | Open-model mapping |
|---|---|---|---|
| `type`: `object`, `array`, `string`, `integer`, `number`, `boolean` | yes | `DynamicGenerationSchema` | llguidance |
| `properties`, in the order sent | yes | properties in that order | properties in that order (llguidance's rule) |
| `required` | yes | `isOptional` for the rest | the rest may be omitted |
| `additionalProperties` | absent or `false`; anything else 400 | closed objects | closed objects (`false` set explicitly) |
| `enum` of strings; `const` string | yes | `anyOf` strings | llguidance |
| nullable: `type: [T, "null"]`, or `anyOf` of one schema and `{"type": "null"}` | yes | an optional property; the layer writes `null` where it is absent | llguidance |
| `items`, `minItems`, `maxItems` | yes | `arrayOf` with bounds | llguidance |
| `description`, `title`, `$schema`, `examples`, `default` | accepted; `description` is passed to AFM | a guide | written with the schema into the system message (#186; the measurement is under "Constrained decoding on the open path") |
| `anyOf` (other), `oneOf`, `allOf`, `$ref`/`$defs`, `enum` of non-strings | 400 | — | — |
| `minimum`, `maximum`, `pattern`, `format`, `minLength`, `maxLength`, `patternProperties` | 400 | — | — |

The subset is what both backends can honor. A keyword joins it when both
do: llguidance supports the last two rows' keywords (with the coverage its
documentation states), and AFM's generation guides may cover some (number
bounds, a pattern), which is to be confirmed against the SDK and its regex
dialect. Until then a
keyword one backend would honor and the other ignore is a 400 on both,
not a difference in behavior by model. `response_format.json_schema.strict`
is accepted either way; output is always constrained.

**Key order (#180).** The representation keeps `properties` in the order the
request sent them, which needs `serde_json`'s `preserve_order` feature in
the workspace. (llguidance turns it on anyway, by feature unification, so
the workspace enables it explicitly; every `serde_json::Map` the daemon
builds then keeps insertion order instead of sorting, and its JSON outputs
are checked for that.)
- The bridge receives the properties as an ordered list, since a Swift
  dictionary from `JSONSerialization` doesn't keep order, and builds the
  generation schema in that order, so fields are also generated in the
  client's order. The Rust side re-serializes AFM's reply in schema order.
- llguidance generates properties in schema order by construction.
- Both are then byte-identical across runs at temperature 0.

**Optional fields (#177).** The layer keeps JSON Schema's meaning: a
property not in `required` may be left out. It doesn't rewrite optional
properties as required. How AFM's mapping should treat them (#177's
options, such as required-but-nullable) is decided by measurement on the
benchmark's extraction set, and changes only AFM's mapping. On the open
path, the benchmark found llguidance made MiniCPM5-1B fill in fields the
text doesn't state; whether the 2B does is measured before guided
extraction is recommended on it. A first look (#186), on 24 of the
extraction items: guided without the schema in the system message held
the same values as prompt-only on every item (13 stated optional fields
left out by both). With it, guided left out 9 with the benchmark's
prompt and 6 without, so guidance with the schema shown drops fewer
fields, not more ("Constrained decoding on the open path" has the
table). The full set, guided and prompt-only, is the release gate's to
run.

### Constrained decoding on the open path

[llguidance](https://github.com/guidance-ai/llguidance) (MIT; Rust; 1.9.1 is
the version the benchmark used) computes, per step, the set of tokens the
grammar allows; the sampler sets every other logit to −∞.
- **API.** One `ParserFactory` per model, built from its tokenizer
  through `toktrie_hf_tokenizers` (which depends on `tokenizers` 0.21 with
  the pure-Rust regex, as the workspace does), with the model's stop tokens
  as its end-of-sequence tokens. It is built on the model's first guided
  request (about 0.4 s for MiniCPM5-2B's 130k vocabulary), not at startup,
  so a model never asked for JSON costs nothing. Per request, the schema
  layer's representation is written back out as JSON Schema with the
  `x-guidance` options (`sidekick_llm::guided::grammar_schema`) and
  compiled with `TopLevelGrammar::from_json_schema` (under 1 ms for the
  benchmark's extraction schemas); a `Matcher` then gives `compute_mask`,
  `consume_token` and `compute_ff_tokens` each step. A model keeps its 32
  most recently used grammars, keyed by the schema written out, and a
  request starts from a clone, which shares the entry's lexer state; a
  grammar that errors during a request is evicted, so the next request
  compiles it afresh, and two requests compiling one schema at once keep
  one entry. `/health` reports the cache (`chat.models.<id>.guided`). A
  schema the layer accepts but llguidance can't compile is a 400
  `unsupported_schema`. A tokenizer llguidance can't read (one that isn't
  byte-level or byte-fallback BPE) is found at startup: the model's
  listing leaves `response_format` out of its `honored_fields`, and guided
  requests to it are a 400.
- **The schema is in the prompt too.** The grammar only forces what may
  come next: it can't tell the model which fields exist, which are
  optional, what an enum offers or what a description asks, so a model
  that met the schema only through the grammar would meet each field name
  as it is forced. Foundation Models, by default, puts the schema in its
  prompt as well. So a guided request's schema, written out as JSON Schema
  with its descriptions (`guided::prompt_schema`, on one line as
  `json.dumps` writes it), goes into the system message under a
  `# Response format` heading, after any system text the request sent and
  a blank line, or as a system message of its own. That is where
  MiniCPM5's template puts tool definitions, in the same one-line JSON. It
  is written before the prompt is tokenized, so it counts in
  `prompt_tokens` and against the context. `[models."<id>"]
  schema_in_prompt = false` turns it off for a model.

  Measured on MiniCPM5-2B (#186's PR), with 24 of the chat benchmark's
  extraction items (3 per schema), greedy. Scores are from the
  benchmark's scorer, with a stand-in schema validator and no stemming,
  so they compare modes rather than match CHAT-BENCHMARK.md:

  | Prompt | Schema in the system message | Field accuracy | Exact | Stated optional fields left out | Prompt tokens |
  |---|---|---|---|---|---|
  | No schema in the user turn | no | 0.680 | 0.042 | 27 | 3,624 |
  | No schema in the user turn | yes | 0.951 | 0.625 | 6 | 7,314 |
  | The benchmark's (names the schema) | no | 0.902 | 0.583 | 13 | 7,074 |
  | The benchmark's (names the schema) | yes | 0.934 | 0.625 | 9 | 10,764 |
  | The benchmark's, prompt-only (not guided) | no | 0.902 | 0.583 | 13 | 7,074 |

  Every guided reply was valid. Without the schema, a client that puts
  it only in `response_format` gets a model that guesses at the fields.
  With it, it does better than the benchmark's own prompt. Writing the
  schema twice (the client's prompt and the system message) didn't hurt
  either: 0.934 against 0.902. It costs prefill: the benchmark's
  requests took 25% longer end to end (78.8 s against 63.1 s for the 24),
  for 52% more prompt tokens. On the hand-written meeting request ("Let's
  meet Thursday…", every field required and nullable), the schema in the
  system message turned `"day": null` into `"day": "Thursday"`.
- **Cost, measured: the mask doesn't need hiding.** The design first had
  each mask computed on another thread while the next ANE call ran. Over
  the chat benchmark's 80 extraction answers walked through MiniCPM5-2B's
  tokenizer (`cargo run -p sidekick-llm --example guided_masks`), a mask
  took 12 µs on average (p50 5 µs, p99 55 µs, at most 266 µs), against a
  call's 53 ms; handing it to a scoped thread and back costs 27–41 µs,
  more than the mask. Served end to end (#186's PR), a reply's masks
  averaged 23–214 µs, and the slowest single mask was 8.5 ms, the first
  of a newly compiled grammar. So the mask is computed after each call,
  on the decoder thread. Each reply's log line carries its masks' maximum,
  so a schema that makes them slow shows.
- **Forced tokens are free.** Where the grammar allows exactly one
  continuation (`{"name": "`, a closing brace), `compute_ff_tokens`
  returns it, and those tokens are fed in the same call as the sampled
  token before them, in slots that call would otherwise pad: a call costs
  about the same for 1 as for 16 tokens. A reply's forced opening follows
  the prompt in its calls, and a reply the grammar forces whole (a `const`)
  needs no call. Over the extraction answers, 34% of the tokens were forced
  and a reply took 1.57 tokens a call; served, guided replies decoded at
  25–83 tokens/s against 18.5 unguided.
- **One line, spelled as models write it.** The grammar allows no
  whitespace between JSON tokens except in its separators, `, ` and `: `
  (`whitespace_flexible: false`), rather than bounding whitespace with a
  pattern. The reply is one line with its keys in schema order
  ([API.md](../API.md#structured-output), "Key order"), the same bytes
  streamed or not, and a model can't fill its budget with blank lines.
  The separators are the ones MiniCPM5-2B writes unprompted. Compact `,`
  and `:` split its usual tokens (`": "`), and it wrote
  `{"name":", "}` for a signature from which, with spaces or unguided, it
  extracts all five fields. llguidance's default, free whitespace, gave
  the same values as the spaced separators on #186's requests, at
  19–25 tokens/s against 25–83, since less of the reply is forced.
  Strings and numbers are as the model writes them, within JSON's
  grammar.
- **The end.** When the JSON is complete and nothing may follow it, the
  reply ends with `finish_reason: "stop"` and no stop token is generated;
  where the JSON could go on (a number at the root), the stop tokens are
  allowed once it is complete. `max_tokens` before the end is `"length"`
  with the JSON cut off, as `apple-fm` reports it. A grammar that allows no
  token of the vocabulary, or an llguidance error mid-reply, fails the
  request (500), and is logged.
- **`constrained` is true** on every guided open-model reply, and each
  reply's log line carries its forced tokens and its masks' mean and
  maximum time.

### What the benchmark implies

On MiniCPM5-1B, constrained decoding made every reply schema-valid but left
tag F1 and extraction accuracy where they were
([Findings](../CHAT-BENCHMARK.md#findings)): its errors are wrong choices,
not malformed JSON. So:
- guided output is sold as **validity**, never as accuracy; which model
  answers matters more than how its output is constrained;
- on the open path guided output costs nothing measurable and removes the
  parsing step, so it is the recommended way to ask an open model for JSON;
  on AFM it costs fields (#177) and refusals (#178), so the
  recommendation is per backend, and the docs say so;
- the release gate and the benchmark score open models guided and
  prompt-only, so a change in either shows.

## Routing

- **By `model`, exactly.** `apple-fm` is Foundation Models; an open model is
  its manifest's `id` (`minicpm5-2b`). A name served by another route is
  the existing wrong-route 400; an unknown one is a 404. The server's single
  chat backend becomes a map from id to backend.
- **No default and no alias** in the first release: OpenAI requires
  `model`, and every client already names one.
- **`/v1/models`** lists every chat model with `task: "text-generation"`,
  its backend, `context_length`, the request fields it honors (`seed`,
  `top_p`), whether guided output streams, whether it has guardrails, and,
  for an open model, its compute units and graded OS. `/health` reports
  each chat model's availability and reason, and the open model's queue,
  state and ANE accounting.
- **Provenance headers** name the model that answered (`sidekick-model`,
  with its revision), and `sidekick-compute-units` for an open model.
- **AFM's unavailability is unchanged.** `apple-fm` still answers 503 with
  the reason when Foundation Models is unavailable (D3); an open model is a
  model the client chooses, not a fallback.
- **A guardrail refusal doesn't fall back (#178).** It stays a 400
  `content_filter` (or a `content_filter` finish when streaming), and its
  message says whether output was guided, so a client can retry prompt-only
  or with another model. A silent fallback would answer with a different
  model than the one named, with different quality, against its provenance
  and repeatability; a configured fallback moves a client policy into the
  daemon, which D3 declined for the same reason. `/v1/models` says which
  chat models have guardrails, so a client can choose up front.
- **AFM-only behavior is reported, not emulated:** session reuse and
  `cached_tokens`, guardrails, `unsupported_language`. Open-only behavior
  likewise: streamed guided output. `seed` and `top_p` are honored by both
  backends (amending D22).

## Conversion and release

### The converter

The spike's converter (`llama_ane.py`, `convert.py`, `gptq.py`) moves into
`tools/sidekick_convert` as the converters' layers expect
([CONVERTING.md](../CONVERTING.md#layers)); as built, it is described in
[CONVERTING.md](../CONVERTING.md#causal-language-models):
- **Backbone** `backbones.llama_causal`: LlamaForCausalLM in the channels-first
  form, explicit attention (D25), StableSilu (D39), RMSNorm pre-scales, the
  residual stream at 1/K, the R1 and R2 rotations folded into the weights.
- **Head:** a causal-LM head with `sel`. **Interface:** the contract above.
- **Weights:** GPTQ onto 8-bit palettes, stored with the palettizer's
  `unique` mode so the values survive exactly. Per-channel grids only: the
  ANE runs per-block int8 on the CPU without saying so, and per-block int4
  doesn't load (#173). 4-bit is ruled out for this model.
- **Chunks** at layer boundaries under the 1 GiB limit (D32, D37).
- **Gates:** the fp32 rewrite gate (the rewritten, rotated model in fp32
  reproduces the checkpoint's logits), the forbidden-op check, the compute
  plan (every operation on the ANE), finite output, and the grading below.
- **A PyTorch screen before any ANE compile.** #173 found GPTQ's palettes
  score almost the same in PyTorch as on the ANE, so a calibration or
  recipe variant is screened in PyTorch first, and only a candidate that
  passes there is compiled. Every ANE compile leaves about one copy of the
  weights in a system cache nobody can delete (below).

### Calibration

- **Committed, not read live.** The spike calibrated on spans of this
  repository's own files, which change with every commit. The calibration
  set becomes a fixture: text and the token ids it encodes to, its
  sources and licences recorded as the benchmark's items are, and its
  sha256 in the build's `[build]` table.
- **Broader than this repository.** The spike's set was mostly code and
  docs, which may be why its grade depended on the draw. The fixture
  mixes this repository's text and prompts written for it (MIT) with
  revision-pinned Wikipedia lead sections (CC BY-SA 4.0), recorded with
  their attribution as the benchmark records its own.
- **Disjoint from what is graded:** the parity texts and the benchmark's
  task set (no Wikipedia article the benchmark draws on), under `core.Calibration`'s existing guard.
- **Seeded:** the rotation's Hadamard seed, GPTQ's order and the palettes'
  k-means initialisation are recorded, so a rerun reproduces the build's
  bytes (CONVERTING.md, "Reproducing a published build").

### Grading a causal LM

D54 replaced this note's first rubric (a first-percentile logit cosine with
a wide-flip cap, as classifiers are graded); the rubric, its bands and the
anchors they come from are in
[CONVERTING.md](../CONVERTING.md#grading). In short:
- **Teacher-forced** over a fixed evaluation set against the fp32
  checkpoint: per position the KL divergence of the build's next-token
  distribution from fp32's, and top-1 agreement; plus greedy continuations
  of fixed prompts compared with fp32's.
- **Graded on KL** (mean and p99) with top-1 as a floor, per evaluation
  family, the build earning its worst family's letter. The logits' cosine
  and wide flips are reported, not graded: a chat model samples a
  distribution, and a flip where fp32 itself is unsure is noise.
- **A stability check:** a second, seeded calibration draw (screened in
  PyTorch, not compiled) must land in the same band.
- **A task check:** the build served by sidekickd against fp32 on a seeded
  sample of the chat benchmark, within per-family tolerances.

### The release gate

`tools/release_gate.py` gains a text-generation entry, served over HTTP like
every other model:
- **a slice of the benchmark's task set** (the first 12 items of each
  family, as #173's check ran), scored by `tools/chat_bench`'s own scorers,
  guided and prompt-only, each score compared with the baseline under an
  absolute tolerance;
- **repeatability:** every reply at temperature 0 byte-identical between
  two runs, and the share differing from the baseline run's under the
  `flips` tolerance;
- **exact counts:** schema-invalid guided replies (0), refused requests,
  and token-count mismatches between `usage` and the tokenizer.

Teacher-forced fidelity stays in the converter and the parity suite, which
can read logits; the gate checks what a client sees.

### Publishing

- **Licence.** MiniCPM5-2B is Apache-2.0 on its model card, and the
  training sets the card lists are Apache-2.0. `tools/licences.py` computes
  its tier (D50); with no terms that must travel, there's no gate either
  way.
- **What ships:** the chunks' compiled programs (D50), the embedding and
  RoPE files, tokenizer.json, the template, `chat.toml`, in
  `sidekickd/minicpm5-2b-coreml`, recorded in the ledger and installed by
  `sidekickd fetch` as hard links (D51).
- **The grade on the card.** The build is published at its D54 grade
  (KL divergence and top-1 per text family, with the task check), which
  replaced this note's percentile-cosine anchor; the card states each
  family's numbers and the chip and macOS they were measured on.

### The ANE cache cost

Each compile of a build for the ANE leaves about one copy of its weights in
the system's ANE cache, which the user can't delete
(docs/design/coreml-cache.md, "Transplanting a cache"): about 2.1 GiB for
this build. A compute-plan read is another. Rebuilding the loading binary
with another signing identifier recompiles everything (sidekickd is signed
with a fixed one), and a macOS update presumably does too.
- **Converting:** one plan read and one load per chunk per build, in a
  persistent conversion home, after the PyTorch screen; the build record
  states the cost.
- **Users:** one compile per install per macOS version, about the size of
  the weights, described in BACKENDS.md and measured on the model card.

## Deferred, not precluded

- **Speculative decoding.** Verifying 16 drafted tokens costs about 1.12×
  one step (#173), and the T-token function already takes them, so this is
  a runtime change, not a new build. Prompt-lookup drafting needs no draft
  model and suits extraction, which copies spans of the input.
- **Prefix caching:** the reused state and its token record are the
  groundwork ([Concurrency and state](#concurrency-and-state)).
- **`logprobs`:** the host has every step's logits.
- **Tool calling and thinking:** the template supports both; the route
  rejects tools (D22) and renders without thinking.
- **Other model families:** a new backbone in the converter and the same
  interface, where the architecture fits it (RoPE without scaling, grouped
  attention, RMSNorm).
- **A GPU path** (MLX, or Core ML on the GPU with state): against the
  product's thesis by default, but it would serve an operator who wants
  speed (D38's reasoning).
- **Intel Macs:** no Neural Engine, and the CPU path with state crashes.
- **Responses and Messages APIs** (#6) would reuse the same backends.

## Open questions

None remain. Settled while this note was reviewed:

- **The ANE's budget** is system-wide and shared with Foundation Models
  (#187): the daemon serializes its ANE chat models and warns, rather
  than evicting ([Memory budget](#memory-budget)).

- **Grading** is by KL divergence and a task check, with the stability
  check (D54; [Grading a causal LM](#grading-a-causal-lm)).
- **Default sampling** is the checkpoint's
  ([Sampling and the wire contract](#sampling-and-the-wire-contract)).
- **macOS 26.** The first build ships graded on macOS 27, and
  `/v1/models` says so ([Where the decode loop lives](#where-the-decode-loop-lives-rust));
  a macOS 26 grade is added when a Mac running it is free.
- **`seed` and `top_p` on AFM** map to Foundation Models' sampling mode.
- **`request_timeout_secs`** is set per model
  ([Concurrency and state](#concurrency-and-state)).
- **Calibration** adds Wikipedia leads to this repository's text
  ([Calibration](#calibration)).
