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
(`~/Library/Application Support/sidekick/models`), so models converted once
(e.g. with `tools/convert_bge_small.py`) serve both. If the models dir is
empty, the daemon 404s the model and `sk_pool_models` returns `[]` — treat
either as "fall back".

Install only artifacts built with the repository's converters (one
static-shape `.mlmodelc` per bucket). On macOS 27, a Core ML artifact
whose inputs accept several enumerated shapes can abort the process at its
first prediction, with an exception nothing can catch. With the dylib, that
process is your host app. So on macOS 27 sidekick refuses to load such a
model: `sk_embed` returns NULL with an error naming the input, and the daemon
answers with an error. Other flexible-shape artifacts load with a warning and
run on the CPU, off the ANE. `ane_check` rejects all of them without running
them (docs/MODELS.md, D27).

## Path 1: the daemon

`POST /v1/chat/completions`-style OpenAI compatibility, documented in the
README. Embeddings: `POST /v1/embeddings` with optional `input_type:
"query"`, `dimensions` (matryoshka models), `encoding_format: "base64"`.
Classification: `POST /v1/classify`; see [Classification](#classification)
below.

To make "installed but not running" disappear entirely, install the
LaunchAgent from the README (`KeepAlive` keeps it warm); then path 2 only
matters for machines where the user skipped that step.

## Path 2: `libsidekick.dylib`

C ABI defined in
[`crates/sidekick-embed-ffi/include/sidekick.h`](../crates/sidekick-embed-ffi/include/sidekick.h);
build with `cargo build --release -p sidekick-embed-ffi` → 
`target/release/libsidekick.dylib` (~5 MB, embeddings only, no
FoundationModels linkage — it runs on any macOS the models run on).

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
                    /*purpose: 0=document, 1=query*/ 0,
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
  (JSON: `{"id","backend","dims","matryoshka","max_seq_len"}`).
- The first `sk_embed` per model loads it (Core ML: ~1s for bge-small,
  seconds for large encoders) and it stays resident until `sk_pool_close`.
  Loads don't block calls on other, already-loaded models.
- Thread-safe; calls from multiple threads are fine. The ANE serializes
  predictions anyway, so client-side batching beats client-side threading.
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
  several requests: the ANE serializes predictions anyway.
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
- **Multi-label (gliner2 models).** `multi_label: true` scores each label
  on its own: `probs` are independent sigmoids, not a distribution, so
  threshold them rather than reading `label` (still the argmax). One label
  is then allowed, as a yes/no question. Other models answer `true` with a
  400 and accept `false`, the default.
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
- **Errors are data.** Every error, malformed JSON included, is the
  OpenAI shape `{"error": {"message", "type", "code"}}`. A 400 names the
  field it rejected. Sending a classifier to `/v1/embeddings`, or an
  embedder to `/v1/classify`, is a 400 naming the model's task.
- **Provenance.** Record the `sidekick-model` response header
  (`<id>@<revision>`) with any stored label or score, so results from
  different model revisions don't mix silently. The daemon also sends
  `sidekick-version` and `sidekick-compute-units` (the compute units the
  model is configured for, also in its `/v1/models` entry) and
  `sidekick-buckets` (the sequence-length bucket each input ran in, in
  input order, e.g. `256,256,512`). The embeddings and rerank routes send
  the same headers; a static embedder sends no `sidekick-buckets`. A
  model's `/v1/models` entry reports, under `placement`, how many of each
  bucket's operations Core ML runs on the ANE, the GPU and the CPU
  (`docs/design/classify.md`, "Where Core ML places operations").

## Path 3: your fallback

`sk_pool_models` returning `[]`, `sk_pool_open` failing, or the daemon
404ing your model id all mean the same thing: sidekick is present but has
no usable model. Treat it identically to "no sidekick".

A broken manifest doesn't fail either path. Since 0.3.0, a manifest that
doesn't parse or validate, or repeats another model's id, is skipped with
a warning, and every other model still loads; before, `sk_pool_open`
failed on it. So a model you expect can be missing from
`sk_pool_models` (or `/v1/models`) while the rest work. `sk_pool_skipped`
(and `skipped_models` in the daemon's `/health`) lists each skipped
manifest with the reason, its path relative to the models directory:
check it when your model id isn't listed. That includes a model served
on the ANE whose compiled weights exceed Core ML's 1 GiB limit, which
Core ML would otherwise run off the ANE without telling you; the reason
names the fixes (`docs/design/classify.md`, "The ANE weight cap").

A model served on the CPU (`compute_units = "cpu_only"`) runs no longer
than 1,024 tokens, past which Core ML's CPU gives slightly different
results in different buckets. Its `/v1/models` entry then shows the
effective maximum under `seq_cap`, beside the manifest's own; send longer
inputs to a model served on the GPU or the ANE (`docs/design/classify.md`,
"The CPU sequence cap").
