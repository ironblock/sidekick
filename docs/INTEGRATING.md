# Integrating sidekick into a host application

For an app that wants on-device embeddings or classification when they're
available, without hard-depending on sidekick. For embeddings there are
three situations and one probe chain (classification is daemon-only; see
[Classification](#classification)):

1. **sidekickd is running** → talk HTTP.
2. **sidekick is installed but not running** → load `libsidekick.dylib`
   in-process.
3. **no sidekick** → your own fallback (skip the feature, or a bundled
   lexical/static method).

Probe in that order at startup (and optionally re-probe on failure):

```
try:  GET http://127.0.0.1:8790/health          (timeout ~150ms)
      -> use POST /v1/embeddings (OpenAI-compatible)
else: dlopen("libsidekick.dylib")               (see search paths below)
      -> sk_pool_open(NULL) -> sk_embed(...)
else: fallback
```

Prefer the daemon when both are available: it shares resident models across
every client on the machine, owns idle eviction, and its API is stable JSON
over HTTP. The dylib is the zero-daemon path — right when you can't manage
a service, want process-lifetime control, or are embedding sidekick into a
sandboxed host.

Both paths read the same models directory
(`~/Library/Application Support/sidekick/models`), so Core ML models
converted once (e.g. with `tools/convert_bge_small.py`) serve both. The
dylib serves Core ML and static models only: it skips ONNX manifests,
and `sk_pool_skipped` gives the reason. If the models dir is empty, the
daemon 404s the model and `sk_pool_models` returns `[]` — treat either as
"fall back".

**On an Intel Mac or Linux,** the daemon can serve embedders as ONNX
models on the CPU, under the same ids and the same API; what each platform
serves is in [INSTALL.md](INSTALL.md#platforms). Intel Macs serve Core ML
classifiers and embedders on the CPU and GPU (there is no Neural Engine):
the default units (`cpu_and_ne`) run a Core ML model on the CPU there, and
`[models."<id>"] compute_units = "cpu_and_gpu"` in the daemon config
selects the faster GPU. Four classifiers are unsupported on Intel Macs;
[MODELS.md](MODELS.md#recommendations) lists them, with the grades
measured there. Linux serves ONNX embedders only.
The dylib ships only in the Apple silicon archive.

So the same client code runs everywhere, as long as it checks what's
there: `/health`'s `backends` lists the model backends the build serves
(`coreml`, `onnx`, `static`; `static` always), `/v1/models` lists each
embedder's `backend`, and `sidekick-compute-units` reads `cpu` for ONNX
models. An ONNX model's vectors come from a different runtime than its
Core ML build's, and aren't guaranteed bit-identical across runtimes,
architectures or thread counts
([BACKENDS.md](BACKENDS.md#onnx-runtime-on-the-cpu)): don't mix vectors
stored from the two builds in one index without checking their agreement.

For Core ML, install only artifacts built with the repository's converters
(one static-shape `.mlmodelc` per bucket). An ONNX model can come from a
converter's `--format onnx` or from a published export; grade it with the
parity suite before relying on it. On macOS 27, a Core ML artifact whose
inputs accept several enumerated shapes can abort the process at its first
prediction, with an exception nothing can catch. With the dylib, that
process is your host app. So on macOS 27 sidekick refuses to load such a
model: `sk_embed` returns NULL with an error naming the input, and the
daemon answers with an error. Other flexible-shape artifacts load with a
warning and run on the CPU, off the ANE. `ane_check` rejects all of them
without running them (docs/MODELS.md, [D27](DECISIONS.md#d27--refuse-multi-shape-core-ml-models-at-load-on-macos-27)).

## Path 1: the daemon

**Authentication.** A daemon configured with an `api_key` requires
`Authorization: Bearer <key>` on every route except `/health` ([D13](DECISIONS.md#d13--default-bind-1270018790-health-unauthenticated)), so a
probe that reads `/v1/models` without the key gets a 401 with code
`invalid_api_key`. `/health` stays open: probe it first, and treat a 401
from the routes after it as "configured, but not for this client".

Every route, field and error is in [API.md](API.md): chat, embeddings
(`input_type`, `prompt_name`, `dimensions`, `encoding_format`,
`truncate`) and classification ([below](#classification)). What a host
needs beyond that:

**Prompts.** sidekick puts the model's own prompt before every input
([API.md](API.md#prompts)). A host that brings its own (an MTEB task
prompt, or a Qwen3-Embedding or F2LLM instruction for a symmetric task, in
the format [MODELS.md](MODELS.md#named-prompts-prompt_name) gives) sends
`input_type: "none"`, or a `prompt_name` the model lists in `/v1/models`
([API.md](API.md#named-prompts)). With `none`, truncate with `END` or keep
inputs within the limit: `START` can cut a prompt sidekick can't tell from
the text. Over the C ABI, `sk_embed`'s purpose 2 is `none`,
`sk_embed_named` takes a name where `sk_embed` takes a purpose, and
`sk_model_info` reports `prompts` and `named_prompts`.

**Over-length input.** `/v1/embeddings` refuses an input over the model's
limit with a 400, and refuses the whole request: split or shorten that
input and send the batch again
([API.md](API.md#input-length-and-truncation)). To know before
embedding, send `POST /tokenize` with the `input_type` you will embed
with: its `fits` is the route's own check ([API.md](API.md#tokenize)).
Use it rather than a tokenizer or a character estimate of your own, which
can disagree at the edge. A client that truncates some documents on
purpose can send `/tokenize` first, log the ones that don't fit, with
their counts, and send those with `truncate: "END"`.

**The two paths differ here.** The library's own default stays `END`:
`sk_embed` has no `truncate` parameter and always truncates, as does the
Rust `Embedder::embed`. `sk_model_info` reports each model's
`max_seq_len`. A host that follows the probe order above (daemon, else
dylib) should send `truncate: "END"` to the daemon to get the same vectors
on both paths, or handle both behaviours: a 400 from the daemon, and a
silently truncated vector from the dylib.

To make "installed but not running" disappear entirely, install the
LaunchAgent ([INSTALL.md](INSTALL.md#running-as-a-service); `KeepAlive`
keeps it running); then path 2 only matters for machines where the user
skipped that step. On Linux, run sidekickd under your service manager; no
unit file ships.

## Path 2: `libsidekick.dylib`

C ABI defined in
[`crates/sidekick-embed-ffi/include/sidekick.h`](../crates/sidekick-embed-ffi/include/sidekick.h);
build with `cargo build --release -p sidekick-embed-ffi` → 
`target/release/libsidekick.dylib` (~5 MB, embeddings only, no
FoundationModels linkage — it runs Core ML and static models on any macOS
they run on). Releases ship it only in the Apple silicon archive.

**Compiled models are cached under your app's identity, not sidekickd's.**
The dylib loads models in your process, so Core ML caches what it
compiles for that process: under your executable's name (or your bundle
identifier), your user's home and, on the Neural Engine, your binary's
code-signing identifier. Two consequences:
- **The daemon's compiles don't help you, and yours don't help it.** A
  model sidekickd has already compiled compiles again on its first load
  in your app, and the reverse; each takes seconds for a small model and
  minutes for a long bucket of a large one.
- **Keep your code-signing identifier stable.** A signed app's identifier
  is its bundle identifier, which doesn't change. An unbundled tool built
  with cargo is signed by the linker as `<name>-<hash>`, and the hash
  changes with the version, so every upgrade compiles every ANE model
  again. Sign such a host ad hoc with a fixed identifier after each build
  (`codesign --force --sign - --identifier <your id> <binary>`), as
  `tools/sign_sidekickd.sh` does for sidekickd.

That the host's identity, not the dylib's, is the one that counts follows
from Core ML keying its cache to the loading process; it was measured
with standalone executables (docs/design/coreml-cache.md, "The cache
key"), not with
the dylib.

Suggested `dlopen` search order for hosts:

1. `$SIDEKICK_DYLIB` (explicit override)
2. next to your app's own binary / inside your app bundle (if you ship it)
3. `/opt/homebrew/lib/libsidekick.dylib`, `/usr/local/lib/libsidekick.dylib`

Minimal usage (C; every language with FFI maps 1:1 — the symbols are plain
C, no callbacks, no structs by value):

```c
if (sk_abi_version() != SK_ABI_VERSION_EXPECTED) goto fallback;
char *err = NULL;
sk_pool *pool = sk_pool_open(NULL, &err);          /* default models dir */
if (!pool) goto fallback;
char *models = sk_pool_models(pool, &err);         /* JSON id array */
size_t dims = 0;
const char *texts[] = {"the quick brown fox"};
float *v = sk_embed(pool, "bge-small-en-v1.5", texts, 1,
                    /*purpose: 0=document, 1=query, 2=none*/ 0,
                    /*requested_dims: 0=native, or a matryoshka value*/ 0,
                    &dims, &err);
/* ... use v[0..dims) ... */
sk_floats_free(v, 1 * dims);
sk_string_free(models);
sk_pool_close(pool);
```

Notes:
- Rows come back unit-normalized; cosine similarity is a plain dot product.
- `requested_dims` has the daemon's `dimensions` semantics (matryoshka
  truncate + renormalize), so vectors indexed via one path stay compatible
  with the other. Discover a model's valid values with `sk_model_info`
  (JSON: `{"id","backend","dims","matryoshka","max_seq_len","prompts",
  "named_prompts"}`).
- `sk_embed_named` is `sk_embed` with one of the model's `named_prompts`
  in place of `purpose`, as the daemon's `prompt_name`. It was added in
  0.15.0 without an ABI version bump, so resolve it with `dlsym` if the host
  must also run against older libraries.
- The first `sk_embed` per model loads it, and the first input to each
  bucket compiles that bucket unless Core ML's cache already holds it for
  your process: seconds for a small model, minutes for a long bucket of a
  large one ([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)).
  The model stays resident until `sk_pool_close`.
  Loads don't block calls on other, already-loaded models.
- Thread-safe; calls from multiple threads are fine. On Apple silicon the
  ANE serializes predictions anyway, so client-side batching beats
  client-side threading.
- Chat is deliberately not in the dylib: it would drag the FoundationModels
  Swift shim into every host, and conversational sessions want a daemon
  lifetime. If you need chat too, run sidekickd.

## Classification

Classifiers (`POST /v1/classify`) are served by the daemon only. The C ABI
serves embeddings, and `sk_pool_models` and `sk_model_info` list embedding
models only, even when the models directory holds classifiers. For a host,
the probe chain is therefore two steps: the daemon, or your fallback.

```
try:  GET http://127.0.0.1:8790/v1/models        (timeout ~150ms)
      -> a model with "task": "text-classification" or
         "zero-shot-classification" -> POST /v1/classify
else: fallback
```

The request and response are vLLM's `/classify`; a client written for vLLM
or SGLang works unchanged. What a host needs to know beyond that:

- **Discover before you send.** `/v1/models` gives each classifier's
  `task`, and either its fixed `labels` (text-classification) or its
  `max_labels` (zero-shot). It also gives `max_batch`, the extension fields
  it accepts (`candidate_labels`, `question_type`, `instructions`,
  `calibration`, `multi_label`), the ones every request must send
  (`required`), its calibration temperatures, and the compute units it runs
  on. Nothing is loaded to answer.
- **Batch up to `max_batch`.** One request with several inputs beats
  several requests: on Apple silicon the ANE serializes predictions
  anyway.
- **Zero-shot labels are the answer space.** `probs` follows
  `candidate_labels`, and `label` is the most probable one. laya's
  `noul` questions take exactly `["false", "true"]`, each optionally with a
  description (`"true: the customer wants a refund"`). A model rendering
  options as Julia-1 does (`docs/design/classify.md`, "Option rendering")
  shows the model a choice label's description (the text after its first
  `": "`) rather than the whole label, takes noul descriptions on both or
  neither, accepts at most 20 labels, and needs `instructions` on every
  request when its listing's `required` says so. The response's `label` is
  always your label as sent.
- **Multi-label (gliner2 and nli models).** `multi_label: true` scores
  each label on its own: `probs` are independent sigmoids, not a
  distribution, so threshold them rather than reading `label` (still the
  argmax). One label is then allowed, as a yes/no question. Other models
  answer `true` with a 400 and accept `false`, the default.
- **NLI models** (those whose `extensions` include `hypothesis_template`)
  are NLI cross-encoders scored as Hugging Face's zero-shot pipeline
  scores them ([D41](DECISIONS.md#d41--the-nli-format-nli-cross-encoders-as-zero-shot-classifiers)): each label goes into a hypothesis (the listing's
  `hypothesis_template`, or your own with one `{}`), and each (text,
  hypothesis) pair is its own prediction, so latency grows with the
  number of labels. One label is a yes/no question, scored against
  contradiction, with or without `multi_label`. The text is truncated to
  fit; send `truncation_side: left` to keep its end, as tool output and
  stack traces want. That score can't tell a claim the text contradicts
  from one it never mentions. To check a claim against evidence, send
  `nli_probs: true` (where `extensions` lists it): each result gains
  `nli_probs`, per label, the model's probabilities over its own classes,
  keyed by the listing's `nli_classes`. Three-way models give
  `entailment`, `neutral` and `contradiction`; the two-way zeroshot-v2.0
  models give `entailment` and `not_entailment`, and can't make that
  distinction ([D47](DECISIONS.md#d47--an-nli-models-class-probabilities-on-request-nli_probs)). It needs sidekickd 0.15.0 or later, and a manifest
  that names the model's classes.
- **Over-length input.** A text-classification input longer than the
  model's maximum is a 400 unless you send `truncate_prompt_tokens` (`-1`
  truncates to the model's maximum). laya-format models (laya, Julia-1),
  gliner2 models and fev models (Lumma-fev) truncate the text themselves,
  keeping its start, so there it's never a 400, and `truncate_prompt_tokens`
  and `truncation_side: left` are. A fev model never cuts its question or
  options: a request whose instructions and labels don't fit the window
  beside a state is a 400. gliner2 drops whole words from the end and
  then ends the text with `.` as its training inputs did; its labels and
  instructions must fit the model whole, or the request is a 400.
  agentjev models (AgentJev) never truncate, as AgentJev's own service
  doesn't: a request whose input, instructions and labels don't fit the
  model together is a 400, and so are `truncate_prompt_tokens` and
  `truncation_side: left`. Send a shorter input. nli models truncate only
  the text, on either side, and never the hypothesis: one too long to
  leave room for any text is a 400.
- **Errors are data.** Every error, malformed JSON included, is the
  OpenAI shape `{"error": {"message", "type", "code"}}`. A 400 names the
  field it rejected. Sending a classifier to `/v1/embeddings`, or an
  embedder to `/v1/classify`, is a 400 naming the model's task.
- **Timeouts.** A bucket's first request can wait for Core ML to compile
  it: seconds, up to minutes for a 2,048-token bucket. The daemon compiles
  every bucket in the background at startup, so the wait is rare once that
  is done, and `/v1/models` shows each bucket's `compile_cache` state
  ([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)). Give the
  first request a generous client timeout, and on a 504 `load_timeout`
  retry after its `Retry-After`: the load continues
  ([API.md](API.md#errors)).
- **Provenance.** Record the `sidekick-model` response header
  (`<id>@<revision>`) with any stored label or score, so results from
  different model revisions don't mix silently. The other headers, and
  each bucket's placement in `/v1/models`, are in
  [API.md](API.md#provenance-headers).

## Path 3: your fallback

`sk_pool_models` returning `[]`, `sk_pool_open` failing, or the daemon
404ing your model id all mean the same thing: sidekick is present but has
no usable model. Treat it identically to "no sidekick".

A broken manifest doesn't fail either path: a manifest that doesn't
parse or validate, or repeats another model's id, is skipped with a
warning, and every other model still loads. So a model you expect can be
missing from `sk_pool_models` (or `/v1/models`) while the rest work.
`sk_pool_skipped` (and `skipped_models` in the daemon's `/health`) lists
each skipped manifest with the reason, its path relative to the models
directory: check it when your model id isn't listed. That includes a model
served on the ANE whose compiled weights exceed Core ML's 1 GiB limit
([BACKENDS.md](BACKENDS.md#the-ane-weight-cap)).

A Core ML model's longest input can also be capped below its manifest's
`max_seq_len`: at 1,024 tokens on the CPU and 2,048 on the ANE
([BACKENDS.md](BACKENDS.md#the-cpu-sequence-cap),
[the ANE sequence cap](BACKENDS.md#the-ane-sequence-cap)). Its
`/v1/models` entry then shows the effective maximum under `seq_cap`, and
`/tokenize`'s `max_model_len` and `fits` use it. For longer inputs, serve
the model on the GPU (`[models."<id>"] compute_units = "cpu_and_gpu"` in
the daemon config). An ONNX model has no such cap.
