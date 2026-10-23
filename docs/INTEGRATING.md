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
models on the CPU (`backend = "onnx"`, D40), under the same ids and the
same API. What else it serves depends on the platform:
- **Intel macOS:** Core ML models too, embedders and classifiers alike,
  on the CPU and GPU (there is no Neural Engine). Chat answers 503 (no
  Foundation Models). A Core ML model's default units (`cpu_and_ne`) run
  it on the CPU there, and `[models."<id>"] compute_units = "cpu_and_gpu"`
  in the daemon config selects the GPU. On the one Intel Mac graded (a
  Xeon W-3235 Mac Pro), the GPU was 6–13× faster on small classifiers and
  as accurate or better, but no more bucket-invariant than the CPU: an
  input's probabilities can move by up to about 2e-4 between buckets on
  the CPU, and by more on the GPU, where Apple silicon's CPU is exact.
  gliner2.5-decide, laya-typed-decisions, laya-en and julia-1 are
  unsupported on Intel Macs. docs/MODELS.md lists what was graded there.
- **Linux:** ONNX embedders only. There is no Core ML, so no
  classification or reranking, and chat answers 503.
- **The dylib** ships only in the Apple silicon archive.

So the same client code runs everywhere, as long as it checks what's
there: `/health`'s `backends` lists the model backends the build serves
(`coreml`, `onnx`, `static`; `static` always), `/v1/models` lists each
embedder's `backend`, and `sidekick-compute-units` reads `cpu` for ONNX
models. An ONNX model's vectors come from a different runtime than its
Core ML build's. Both are graded against the same fp32 reference, but
don't mix vectors stored from the two in one index without checking their
agreement in docs/MODELS.md. The daemon loads ONNX Runtime only when it
has an onnx model. It uses the library bundled beside `sidekickd`, or one
the config's `onnx_runtime` or `ORT_DYLIB_PATH` names, and `/health`'s
`onnx.runtime` says which loaded, or why none did. Intel Macs run ONNX
Runtime 1.23.2, the last release Microsoft built for them, where every
other platform runs 1.28.3. Each platform's results are graded on that
platform. On an Intel Mac with AVX-512, ONNX Runtime runs its AVX-512
kernels (D42); `SIDEKICK_ONNX_AVX512=0` in the daemon's environment turns
them off.

How far ONNX vectors are reproducible: at a given thread count, re-runs
are bit-identical, and so were two Linux x86-64 machines with different
CPUs on the same runtime and thread count. Across runtime versions,
architectures or thread counts, vectors agree to within rounding but
aren't guaranteed bit-identical (EmbeddingGemma moved by up to 1.43e-7
per component between 4 and 8 threads). Padding and batching would move
them by up to 2.6e-7, which is why sidekick runs each input alone.

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
without running them (docs/MODELS.md, D27).

## Path 1: the daemon

`POST /v1/chat/completions`-style OpenAI compatibility, documented in the
README. Embeddings: `POST /v1/embeddings` with optional `input_type:
"query"` or `"none"` (or `prompt_name`), `dimensions` (matryoshka models), `encoding_format:
"base64"`, and `truncate`. Classification: `POST /v1/classify`; see
[Classification](#classification) below.

**Prompts.** sidekick puts the model's own prompt before every input:
its document prompt by default, its query prompt for `input_type:
"query"` (each model's are in `/v1/models` as `prompts`, and in
docs/MODELS.md). A host that brings its own prompts must send
`input_type: "none"`, which embeds the text exactly as sent. Examples are
MTEB-style task prompts (EmbeddingGemma's `task: classification | query:
`), an instruction for a Qwen3-Embedding or F2LLM symmetric task such as
clustering or similarity, or no prompt where a model's evaluation used
none (e5-small-v2 outside retrieval). Without `none`, sidekick's prompt
goes before the host's. Truncating with `truncate: "START"` keeps the end
of the text and sidekick's own prompt whole, but sidekick can't tell a
host's prompt from its text, so `START` may cut it: with `none`, truncate
with `END`, or keep inputs within the limit (`/tokenize`'s `fits`). `/tokenize` takes the same `input_type`, so its
count and `fits` match. Over the C ABI, `sk_embed`'s purpose 2 is the
same, and `sk_model_info` reports the prompts.

**Named prompts.** Where the checkpoint names prompts for other tasks
(EmbeddingGemma's `Classification`, `Clustering`, `STS` and others), a
host can send `prompt_name` instead of `input_type`, as with Hugging
Face Text Embeddings Inference, rather than copying the prompt's text
and sending `none`. The names are each model's `named_prompts` in
`/v1/models` (name to prompt; `{}` for none), match exactly, case
included, and are listed in docs/MODELS.md. A named prompt is sidekick's
own, so `truncate: "START"` keeps it whole, which a host's prompt sent
with `none` doesn't get. Sending `prompt_name` with any `input_type`,
`none` included, is a 400, and so is a name the model doesn't have; the
message lists the model's names. `/v2/embed` and `/tokenize` take it too.
There, because Cohere's SDK always sends `input_type`, a `prompt_name`
overrides Cohere's names (`search_query`, `search_document`,
`classification`, `clustering`); with sidekick's own (`query`,
`document`, `passage`, `none`) it is still a 400. Over the C ABI,
`sk_embed_named` takes a name where `sk_embed` takes a purpose, and
`sk_model_info` lists `named_prompts`. Qwen3-Embedding and F2LLM take a
free-text instruction per task instead of a fixed set, so their only
names are `query` and `document`. For another task, send `Instruct:
<task>\nQuery:` (F2LLM: `Instruct: <task>\nQuery: `, with a trailing
space) before the text with `input_type: "none"`, and truncate with
`END` or keep inputs within the limit: on symmetric tasks the
instruction goes on every input, and `START` would cut it (D45).

**Over-length input.** `/v1/embeddings` refuses an input longer than the
model's limit with a 400 `invalid_request_error`, as OpenAI's embeddings
API does. The message names the input's index in the batch, its token
count, the model and the limit, and the whole request is refused, so split
or shorten that input and send the batch again. The limit is `max_seq_len`
in the model's `/v1/models` entry, in tokens, counting special tokens and
the model's prompt prefix (for a Core ML model, its largest bucket; its
`buckets` are listed too). A text over 64 bytes per token of the limit is
refused without being tokenized, and its message gives the size in bytes
instead. A model whose own `tokenizer.json` truncates
(LFM2.5-Embedding-350M's does, at 512 tokens, its `max_seq_len`) is
refused past that point too.

**Checking fit before embedding.** `POST /tokenize` with `{model,
prompt}` (and the `input_type` you will embed with) answers with `fits`:
whether `/v1/embeddings` will take the text whole under its default
`truncate: NONE`, decided by the same check the route runs. Use `fits`
rather than re-deriving it. The route refuses a text by its size in bytes
as well as its tokens, and the two disagree for some text: a run of Thai
or Khmer without spaces can be one unknown token yet far too many bytes,
and a text of spaces counts no tokens at all. The response also has
sidekick's own `count`, never truncated, with the prompt prefix and
special tokens included, and `max_model_len`, the limit it is compared
with. A tokenizer of your own, or an estimate from characters, can
disagree with that count at the edge: it may miss the prompt prefix or the
special tokens, or tokenize differently. A client that truncates some
documents on purpose can send `/tokenize` first, log the ones that don't
fit, with their counts, and send those with `truncate: "END"`. At most
4 MiB of text is tokenized at once across requests, and a request waits
for room within the request timeout. The request body limit (2 MiB)
bounds the text. See the README's Tokenize
section for the full shape (D44).

To truncate instead, send `truncate: "END"` (or TEI's `truncate: true`;
OpenAI's Python SDK sends it through `extra_body`): the input is cut to
the limit and returns a normal vector, with nothing in the response to say
so, so a long document embeds only its start. `truncate: "START"` keeps the end
instead. Until 0.13.0, `END` was the default (D43). `/v2/embed` keeps
Cohere's default, `END`.

**The two paths differ here.** The library's own default stays `END`:
`sk_embed` has no `truncate` parameter and always truncates, as does the
Rust `Embedder::embed`. `sk_model_info` reports each model's
`max_seq_len`. A host that follows the probe order above (daemon, else
dylib) should send `truncate: "END"` to the daemon to get the same vectors
on both paths, or handle both behaviours: a 400 from the daemon, and a
silently truncated vector from the dylib.

To make "installed but not running" disappear entirely, install the
LaunchAgent from the README (`KeepAlive` keeps it warm); then path 2 only
matters for machines where the user skipped that step. On Linux, run
sidekickd under your service manager; no unit file ships.

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
- The first `sk_embed` per model loads it (Core ML: ~1s for bge-small,
  seconds for large encoders) and it stays resident until `sk_pool_close`.
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
  scores them (D41): each label goes into a hypothesis (the listing's
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
  distinction (D47). It needs sidekickd 0.15.0 or later, and a manifest
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
- **Timeouts.** A model's first request after it loads, and the first to
  reach a larger bucket, waits while Core ML loads that bucket, and, if
  its compiled form isn't cached yet, compiles it: seconds, up to minutes
  for a 2,048-token bucket. The daemon compiles every bucket it serves in
  the background at startup (D52), so that wait is rare once the startup
  warm-up is done; `/v1/models` shows each model's buckets under
  `compile_cache` (`warm`, `warming`, `cold`, `failed`), and a request
  for a `warming` bucket waits for that compile. After installing models
  or a macOS update, `sidekickd warm` compiles them ahead of time. The
  wait doesn't count toward `request_timeout_secs`; the prediction after
  it does. Give the first request a generous client timeout. A 504 with code `timeout` is a
  prediction that ran too long; one with code `load_timeout` (and a
  `Retry-After` header) means the load outlasted `load_timeout_secs` and
  continues, so retry later.
- **Provenance.** Record the `sidekick-model` response header
  (`<id>@<revision>`) with any stored label or score, so results from
  different model revisions don't mix silently. The daemon also sends
  `sidekick-version` and `sidekick-compute-units` (the compute units the
  model is configured for, also in its `/v1/models` entry) and
  `sidekick-buckets` (the sequence-length bucket each input ran in, in
  input order, e.g. `256,256,512`). The embeddings and rerank routes send
  the same headers. A static or ONNX embedder sends no `sidekick-buckets`,
  and an ONNX model's `sidekick-compute-units` is `cpu`. A
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

A Core ML model served on the CPU (`compute_units = "cpu_only"`) runs no
longer than 1,024 tokens, past which Core ML's CPU gives slightly different
results in different buckets. Its `/v1/models` entry then shows the
effective maximum under `seq_cap`, beside the manifest's own (its
`max_seq_len` and `buckets` are already the capped ones); send longer
inputs to a model served on the GPU or the ANE (`docs/design/classify.md`,
"The CPU sequence cap"). The cap is Core ML's: an ONNX model reports
`compute_units: "cpu"` but has no cap and no `seq_cap`.

Likewise, a Core ML model served on the ANE (`cpu_and_ne`, the default, or
`all`) runs no longer than 2,048 tokens, or its manifest's
`ane_max_seq_len`: past that, a bucket can take half an hour to compile
for the ANE and then fail some predictions. The same artifact served on
the GPU (`cpu_and_gpu`) has no cap. On an Intel Mac, which has no Neural
Engine, `cpu_and_ne` runs on the CPU and gets the CPU's 1,024-token cap
instead. Its `/v1/models` entry shows
`seq_cap` the same way, `/tokenize`'s `max_model_len` and `fits` use the
capped limit, and a longer input to `/v1/embeddings` is refused (or
truncated) at it. For longer inputs, serve the model on the GPU, in its
manifest or with the daemon config's `[models."<id>"] compute_units`
(`docs/design/classify.md`, "The ANE sequence cap").
