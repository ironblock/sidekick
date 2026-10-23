# sidekick

On-device inference for very small, asynchronous tasks — session titles,
tags, embeddings, classification, structured extraction. On Apple silicon it
uses every part of the silicon: Apple's Foundation Models (via Apple
Intelligence) for generation, and Core ML encoders on the Apple Neural Engine
for embeddings and classification. On Intel Macs and Linux, and as a CPU
cross-check on Apple silicon, embedding models run on ONNX Runtime's CPU
backend under the same API (D40). A pure-CPU static-embedding floor tier
works anywhere.

The daemon, **`sidekickd`**, speaks the OpenAI API, so anything that can point
at an OpenAI-compatible base URL (OpenCode, editors, scripts) can use it:

```
POST /v1/chat/completions   Apple Foundation Models (Apple silicon, macOS 26+, Apple Intelligence)
POST /v1/embeddings         Core ML encoders (ANE on Apple silicon), ONNX Runtime CPU encoders,
                            and static floor models
POST /v1/classify           Core ML classifiers, on macOS (vLLM's /classify contract)
POST /v1/rerank, /rerank    Core ML cross-encoder rerankers, on macOS (vLLM's / Jina's shape)
POST /v2/rerank             the same, in Cohere's v2 shape
POST /v2/embed              the embedders, in Cohere's v2 embed shape
POST /tokenize, /detokenize an embedder's or text classifier's tokenizer (vLLM's shape)
GET  /v1/models             what this machine can serve: each model's backend and
                            longest input (max_seq_len)
GET  /health                availability per tier, and why when unavailable
```

Design rationale lives in
[docs/design/ane-inference-first-principles.md](docs/design/ane-inference-first-principles.md);
autonomous implementation decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

## Status

The Apple silicon inference paths are hardware-verified (macOS 26.5.1): the Foundation Models Swift shim (including constrained
decoding), the static embedding tier, and the Core ML encoder tier with a
locally converted bge-small running ANE-resident (3.4x over CPU at seq 128,
verified via `cargo run -p sidekick-coreml --example ane_check`, which judges
ANE eligibility from Core ML's compute plan). See
"Hardware verification status" in [docs/DECISIONS.md](docs/DECISIONS.md),
`cargo run -p sidekick-server --bin smoke-test`, and
[tools/convert_bge_small.py](tools/convert_bge_small.py) for the conversion
recipe (the constraints it encodes are hardware-verified; deviating from
them silently pushes the encoder off the ANE).

Chat streams for real: `stream: true` sends text as the model generates it
(schema-constrained replies arrive whole). On macOS 27 (built with Xcode 27)
chat reports real token usage, including
cached tokens when a follow-up reuses its session. `/health` reports which
Foundation Models variant the Mac has and its context size. Over-long
prompts are a proper 400. See D21 in [docs/DECISIONS.md](docs/DECISIONS.md).

Which encoders are known to convert cleanly and achieve full ANE offload —
and which don't, and why — is tracked with measured numbers in
[docs/MODELS.md](docs/MODELS.md), along with a "will a new model convert?"
checklist. Validated so far: bge-small-en-v1.5, all-MiniLM-L6-v2,
e5-small-v2, EmbeddingGemma-300m, EmbeddingGemma 2, LFM2.5-Embedding-350M,
F2LLM-v2-160M, gte-modernbert-base.
`tools/probe_activations.py` triages a candidate in minutes before any
conversion. A parity suite (D26) grades every model through sidekick's own
code, on inputs chosen to break it: Core ML models on the CPU, GPU and ANE
paths, ONNX models on `onnx-cpu`. On the ANE, all eight validated
embedders grade A. Seven embedding models' ONNX exports grade A on
`onnx-cpu` on an M1 Max and on an Intel Xeon W-3235 (macOS); six of them
also on two x86-64 Linux machines, with EmbeddingGemma 2's Linux grades to
follow. See the
confidence grades in [docs/MODELS.md](docs/MODELS.md), or browse every graded
result, by model, path, chip and runtime, at
[ironblock.github.io/sidekick](https://ironblock.github.io/sidekick/).

Embeddings are also available in-process via `libsidekick.dylib`
(`cargo build --release -p sidekick-embed-ffi`) for hosts that can't or
don't want to run the daemon — see
[docs/INTEGRATING.md](docs/INTEGRATING.md) for the daemon → dylib →
fallback probe chain. It serves Core ML and static models; it's built
without ONNX Runtime, so it skips ONNX models, and it ships in the Apple
silicon archive.

## Supported platforms

| Platform | Chat (Foundation Models) | Embeddings | Classify, rerank | Release archive |
|---|---|---|---|---|
| macOS 26+, Apple silicon (`aarch64-apple-darwin`) | yes | Core ML (ANE, GPU, CPU), ONNX on the CPU, static | Core ML | yes |
| macOS 13.4+, Intel (`x86_64-apple-darwin`); built and tested on macOS 26 | no (503) | Core ML (GPU, CPU), ONNX on the CPU (ONNX Runtime 1.23.2), static | Core ML | yes |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | no (503) | ONNX on the CPU, static | no | yes, glibc 2.35+ |
| Linux arm64 (`aarch64-unknown-linux-gnu`) | no (503) | ONNX on the CPU, static | no | yes, glibc 2.35+ |

ONNX models need ONNX Runtime (D40): 1.28.3, or 1.23.2 on Intel macOS, the
last release Microsoft built for it. Release archives ship it beside
`sidekickd`. An ONNX build of a model keeps its id, so a client calls the
same model the same way on every platform.

## Quickstart

```sh
cargo build --release -p sidekick-server --features coreml,onnx   # on a Mac
tools/sign_sidekickd.sh                                           # on a Mac; see below
./target/release/sidekickd --models-dir ~/Library/Application\ Support/sidekick/models
curl -s localhost:8790/health | jq
```

`tools/sign_sidekickd.sh` re-signs the binary ad hoc with a fixed
code-signing identifier, `io.github.ironblock.sidekickd`, as the release
archives' sidekickd is signed. Core ML keys each model it compiles for the
Neural Engine to the identifier of the process that loads it, and the one
the linker gives a Rust binary is cargo's name for the build
(`sidekickd-<hash>`), which changes with the version, features and
toolchain. Without the step, a rebuild that changes it compiles every ANE
bucket again on its first load. Measured on an M1 Max, macOS 27.0, with
bge-small-en-v1.5's 128-token bucket on the ANE: after a version bump, the
first request took 0.87 s with both builds signed this way, and 3.76 s with
the linker's signatures (4.3 s with no cache at all), because the ANE
compiler ran again.

On Linux, build with `--features onnx` only. A source build serving ONNX
models needs the pinned ONNX Runtime: `tools/fetch_onnxruntime.sh <dir>`
fetches and checks it, and `onnx_runtime` in the config (or
`ORT_DYLIB_PATH`) points at it. Release archives ship `sidekickd` with the
runtime beside it, for all four targets.

Chat requires an Apple silicon Mac on macOS 26+ with Apple Intelligence
enabled; Intel Macs and Linux get a 503 from `/v1/chat/completions`.
`/health` tells you where you stand. Embeddings require at least one model in the models
directory (see below). Requests degrade honestly: a missing tier is a 503
with a reason, never a hang.

### Running as a daemon

For always-on use, run under launchd as a user agent. Save as
`~/Library/LaunchAgents/dev.sidekick.sidekickd.plist` (adjust the binary
path), then `launchctl load ~/Library/LaunchAgents/dev.sidekick.sidekickd.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.sidekick.sidekickd</string>
  <key>ProgramArguments</key>
  <array><string>/usr/local/bin/sidekickd</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/sidekickd.log</string>
</dict></plist>
```

Foundation Models requires a logged-in user session (it is Apple
Intelligence), so a LaunchAgent — not a LaunchDaemon — is the right unit.
To serve ONNX models, keep `libonnxruntime.dylib` beside the binary (copy
both from the release archive) or set `onnx_runtime` in the config:
launchd doesn't pass a shell's `ORT_DYLIB_PATH`. On Linux, run `sidekickd`
under your service manager.

### Example: OpenCode session titles

Point a provider at `http://127.0.0.1:8790/v1` with model `apple-fm`.
Constrained output works via the standard `response_format`:

```json
{
  "model": "apple-fm",
  "messages": [{"role": "user", "content": "Title this session: ..."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {"name": "title", "schema": {
      "type": "object",
      "properties": {"title": {"type": "string"}},
      "required": ["title"]
    }}
  }
}
```

On-device guided generation makes the 3B model reliable at exactly this kind
of task — the schema is enforced by constrained decoding, not by hoping.
Check `finish_reason`: a reply cut by `max_tokens` reports `"length"`, and
for constrained output that means the JSON may be missing properties.

`stop` (one string or up to four) is supported, except together with
`json_schema`.

### Embeddings

`POST /v1/embeddings` is OpenAI's embeddings route: `input` (one string or
up to 256), `dimensions` for Matryoshka models, and `encoding_format:
"base64"`.

Most embedders were trained with a prompt before each input, so sidekick
puts the model's own prompt before every input, chosen by `input_type`:
- **No `input_type`, `"document"` or `"passage"`:** the model's document
  prompt, such as e5-small-v2's `passage: ` or EmbeddingGemma's `title:
  none | text: `. This is the default.
- **`"query"`:** the model's query prompt, for the search side of
  retrieval.
- **`"none"`:** no prompt. The text is embedded exactly as sent, with the
  tokenizer's special tokens. Send it when your text already carries a
  prompt, such as an MTEB task prompt or a Qwen3 or F2LLM instruction for
  a symmetric task. Otherwise sidekick's prompt goes before yours, and the
  model sees a prompt it was never trained on. Under `truncate: "START"`,
  sidekick keeps its own prompt whole but can't tell yours from the text,
  so it may cut yours: with `none`, use `END`, or keep inputs within the
  limit.

Some models have no prompt for one side or both, and then that
`input_type` is the same as `"none"`. Each model's prompts are in
[docs/MODELS.md](docs/MODELS.md#prompts-input_type), and `/v1/models`
lists each installed model's as `prompts` (`{"query": "…", "document":
"…"}`, `""` for none). D9 records the choice.

Some models also name prompts for other tasks, such as EmbeddingGemma's
`Classification`, `Clustering` and `STS`. Send `prompt_name` instead of
`input_type` to use one, as with Hugging Face Text Embeddings Inference:

```sh
curl -s localhost:8790/v1/embeddings -H 'content-type: application/json' -d '{
  "model": "embeddinggemma-300m", "input": "I loved this film", "prompt_name": "Classification"
}'
```

The name is one of the model's `named_prompts` in `/v1/models` (name to
prompt, `{}` for a model with none), copied from the checkpoint's
sentence-transformers configuration, and matches exactly, case included.
Another name is a 400 that lists the model's, and so is a request that
sends `prompt_name` with any `input_type`, `"none"` included. On
`/v2/embed`, whose Cohere clients always send `input_type`, a
`prompt_name` overrides Cohere's names (`search_query`,
`search_document`, `classification`, `clustering`), and `/tokenize`
does the same, so a `/v2/embed` request's fields count as it embeds.
`/v1/embeddings` doesn't take Cohere's names. A named prompt is
sidekick's, so `truncate: "START"` keeps it whole.
Qwen3-Embedding and F2LLM take a free-text instruction per task instead:
send it with `input_type: "none"`, in the model's format
([docs/MODELS.md](docs/MODELS.md#named-prompts-prompt_name)), and
truncate with `END`, since `START` would cut an instruction it can't
tell from the text. D45 records the choice.

An input longer than the model's limit is a 400 `invalid_request_error`.
OpenAI's `/v1/embeddings` refuses it too, and Hugging Face TEI returns an
error unless the request asks it to truncate. Here, `/v1/rerank` and
text-classification models on `/v1/classify` do the same unless the
request sends `truncate_prompt_tokens`. The message names the input's
index in the batch, its token count, the model and the limit, and nothing
in the request is embedded, so split or shorten that input and send the
batch again. The limit is the
model's `max_seq_len` in `/v1/models`, in tokens, counting special tokens
and the model's prompt prefix. For a Core ML model it is the largest of
its `buckets`, also listed there. A text longer than 64 bytes per token of
the limit is refused without being tokenized, and the message gives its
size in bytes.

Send `truncate: "END"` to truncate instead: the vector then embeds the
start of the text, and nothing in the response says it was cut. `truncate`
takes `/v2/embed`'s values in any case: `NONE` (the default) refuses,
`END` keeps the start of the text, and `START` keeps its end. Hugging Face
TEI's boolean works too: `true` is `END`, `false` is `NONE`. Anything
else is a 400, and `null` is the default. Before 0.13.0, `END` was the
default (D43). OpenAI's Python SDK sends it through `extra_body`, as in
`client.embeddings.create(..., extra_body={"truncate": "END"})`.

`/v2/embed` keeps Cohere's default, `END`, and so do `libsidekick.dylib`'s
`sk_embed` and the Rust library's `Embedder::embed`, which have no
`truncate` parameter. A host that falls back from the daemon to the dylib
gets the same vectors on both paths only if it sends `truncate: "END"` to
the daemon.

A model whose own `tokenizer.json` truncates is refused past that point
too. Under `END` it stops there, and if that were fewer tokens than
`max_seq_len`, `/v1/models` wouldn't show it. LFM2.5-Embedding-350M's
truncates at 512, which is also its `max_seq_len`.

### Tokenize

`POST /tokenize` and `POST /detokenize` are vLLM's (and SGLang's), for
embedders and text-classification models. They let a client count a
text's tokens with the model's own tokenizer, and ask whether the model
will take it, before it embeds it:

```sh
curl -s localhost:8790/tokenize -H 'content-type: application/json' -d '{
  "model": "bge-small-en-v1.5", "prompt": "How do I reset my password?"
}'
```

```json
{"count": 9, "max_model_len": 512, "tokens": [101, 2129, 2079, 1045, 25141, 2026, 20786, 1029, 102], "fits": true}
```

`fits`, an extension, is the answer to "will `/v1/embeddings` take this
text whole?": the route's own check, run on the text, under the default
`truncate: NONE`. Use it rather than comparing `count` with
`max_model_len`. The route also refuses a text by its size in bytes (see
above), and some text is far more bytes per token than English. A run
of Thai, Lao, Khmer or Burmese with no spaces can be a single unknown
token to a WordPiece vocabulary, so it fits by count and not by size.
A text of nothing but spaces counts no tokens at all. An empty text
is refused by a Core ML or ONNX model that adds no special tokens.

`count` is the number `/v1/embeddings` compares with the limit. It
includes the special tokens and the prompt prefix, which `input_type`
chooses by `/v1/embeddings`' names or `/v2/embed`'s (`query` or
`search_query`; `document`, the default, `passage` or `search_document`;
or `none`, no prompt), or `prompt_name` chooses by name.
The text is never truncated, so a text over the limit is counted in full.
`max_model_len` is that limit: the model's `max_seq_len` from
`/v1/models`, or lower when the model's own `tokenizer.json` truncates
sooner (D44). `max_tokens`, `/v2/embed`'s, makes `fits` measure against
it instead, as `/v2/embed` with `truncate: "NONE"` does.
`add_special_tokens: false` leaves the special tokens out of `count` and
`tokens` (not out of `fits`), and `return_token_strs: true` adds each
token's vocabulary entry as `token_strs`. `/detokenize` takes `{model,
tokens}` and returns `{prompt}`, special tokens kept.

For a text-classification model, `count` and `fits` are `/v1/classify`'s,
without `truncate_prompt_tokens`, which refuses a text over 16 bytes per
token of the limit by its size. Rerankers and zero-shot models are a 400:
their inputs are pairs, or a text laid out with the request's labels, not
one text on its own. So is `messages`, vLLM's chat form, since these
models have no chat template. The chat model is a 400 too, because
Foundation Models reports token counts but not token ids. The routes
aren't mounted under `/v1/`, as in vLLM and SGLang (D44).

### Classification

`POST /v1/classify` is vLLM's `/classify` (and SGLang's `/v1/classify`),
field for field, so vLLM and SGLang clients work unchanged:

```sh
curl -s localhost:8790/v1/classify -H 'content-type: application/json' -d '{
  "model": "nlptown-sentiment",
  "input": ["The battery lasts for days.", "It broke on day two."]
}'
```

The response (values illustrative):

```json
{"id": "classify-…", "object": "list", "created": 1790663954, "model": "nlptown-sentiment",
 "data": [{"index": 0, "label": "5 stars", "probs": [0.01, 0.01, 0.04, 0.22, 0.72], "num_classes": 5},
          {"index": 1, "label": "1 star", "probs": [0.81, 0.14, 0.03, 0.01, 0.01], "num_classes": 5}],
 "usage": {"prompt_tokens": 19, "completion_tokens": 0, "total_tokens": 19}}
```

`probs` follows the model's label order. vLLM's `use_activation: false`
returns raw logits, and `truncate_prompt_tokens` (with `truncation_side`)
truncates. Without it, an input longer than the model's maximum is a 400,
as in vLLM. Every request field vLLM defines is honored or rejected with a
400, never silently dropped.

Zero-shot models take their labels per request, as `candidate_labels`
(Hugging Face's name). laya, the first zero-shot model, also needs a
`question_type`: `choice`, `score` (ordered levels) or `noul` (does a
statement hold: labels `false`, `true`). `probs` then follows
`candidate_labels`:

```json
{"model": "laya-en", "input": "Please reset my password, I can't log in.",
 "question_type": "choice", "instructions": "Which team should handle this ticket?",
 "candidate_labels": ["billing: invoices and refunds", "account: login and passwords", "other"]}
```

NLI cross-encoders (`nli-deberta-v3-*`, `roberta-large-mnli`, the
zeroshot-v2.0 models) put each label into a hypothesis and score it as
Hugging Face's zero-shot pipeline does. That score reduces the model's
classes to one number, so a claim the text contradicts and one it never
mentions can look alike. `nli_probs: true` adds each label's
probabilities over the model's own classes:

```json
{"model": "nli-deberta-v3-base", "input": "The config sets port 8080 and enables TLS.",
 "candidate_labels": ["the port is 9090", "the server logs to a file"],
 "hypothesis_template": "{}.", "nli_probs": true}
```

Measured on nli-deberta-v3-base: `probs` is `[0.55, 0.45]`, which
can't tell the two labels apart, but `nli_probs` can:

```json
"nli_probs": [{"contradiction": 0.9999, "entailment": 0.0000, "neutral": 0.0001},
              {"contradiction": 0.0010, "entailment": 0.0001, "neutral": 0.9990}]
```

One object per label, beside the unchanged `probs`. The three-way (MNLI)
models report `entailment`, `neutral` and `contradiction`. The two-way
zeroshot-v2.0 models report `entailment` and `not_entailment`, so they
can't separate contradicted from unmentioned (deberta-v3-base-zeroshot-v2.0-c
gives both labels not_entailment 0.9996). `/v1/models` lists each one's
`nli_classes`. `nli_probs` needs sidekickd 0.15.0 or later, and so do
the nli manifests from this release on: older daemons skip the model.

`calibration: "model"` applies the temperature the model declares for the
question type and label count; `/v1/models` lists each classifier's
temperatures, labels or `max_labels`, `max_batch`, and the extension fields
it accepts. The full contract is in
[docs/design/classify.md](docs/design/classify.md).

### Rerank

`POST /v1/rerank` (also `/rerank`) is vLLM's rerank, the Jina shape that
vLLM, llama.cpp, LocalAI and Infinity serve. It scores each document
against the query with a cross-encoder and returns them best first:

```sh
curl -s localhost:8790/v1/rerank -H 'content-type: application/json' -d '{
  "model": "ms-marco-minilm-l6-v2", "query": "Will it rain tomorrow?",
  "documents": ["It rained all day yesterday.", "Tomorrow brings heavy showers."],
  "top_n": 1
}'
```

```json
{"id": "score-…", "model": "ms-marco-minilm-l6-v2",
 "usage": {"prompt_tokens": 31, "total_tokens": 31},
 "results": [{"index": 1, "document": {"text": "Tomorrow brings heavy showers."}, "relevance_score": 2.28}]}
```

(Values illustrative.) `relevance_score` is on the scale vLLM reports for
the model: a raw logit for the ms-marco cross-encoders, a sigmoid
probability for most others. vLLM's `truncate_prompt_tokens`,
`max_tokens_per_query` and `max_tokens_per_doc` truncate. Without them, a
pair longer than the model is a 400, as in vLLM. `return_documents: false`
leaves the texts out.

`POST /v2/rerank` takes Cohere's v2 request and truncates long documents,
as Cohere does, keeping the query whole. Its response parses as both
Cohere's and vLLM's. `POST /v2/embed` serves the embedders in Cohere's v2
shape: `input_type` (`search_query` / `search_document`, or `none`) or
`prompt_name`,
`embedding_types` (`float`, `base64`, `binary`, `ubinary`),
`output_dimension`, and `truncate`. The full contract is in
[docs/design/rerank.md](docs/design/rerank.md).

Inference responses (every route above) carry provenance headers:
`sidekick-version`, `sidekick-model` (`<id>@<revision>` when the manifest
records the source revision; for chat, the Foundation Models variant) and
`sidekick-compute-units` (a Core ML model's configured compute units,
`cpu_and_ne` by default; `cpu` for static and ONNX models). Core ML
responses also
carry `sidekick-buckets`: the sequence-length bucket, so the compiled
program, each input ran in, in input order (`256,256,512`; a rerank pair
is one input). Each bucket's placement on the ANE, GPU and CPU is in the
model's `/v1/models` entry once the bucket has loaded.

## Models directory

Each embedding model is a directory with a `manifest.toml`, and each
classifier a directory with a `classifier.toml`:

```
~/Library/Application Support/sidekick/models/
  embeddinggemma-300m/
    manifest.toml
    model.mlmodelc/        # precompile: xcrun coremlcompiler compile model.mlpackage .
    tokenizer.json
```

See [examples/manifests/](examples/manifests/) for annotated manifests
(EmbeddingGemma-300m on ANE, bge-small, a static floor model). Manifest rules
that matter: Core ML models must declare enumerated sequence-length `buckets`
(fixed shapes are what keep the model on the ANE), and `matryoshka` declares
which `dimensions` values the OpenAI API may request. A Core ML model loads
for the ANE (`.cpuAndNeuralEngine`; on an Intel Mac, which has none, Core ML
runs it on the CPU) unless its manifest names other
`compute_units` (`cpu_and_gpu`, `cpu_only` or `all`), for a model the ANE
runs badly; `/v1/models` lists each model's (see "Compute units" in
[docs/design/classify.md](docs/design/classify.md)). A manifest that doesn't
parse or validate is skipped with a warning, and listed with the reason under
`skipped_models` in `/health`; the rest of the directory still loads.

**ONNX models on the CPU.** An embedder may instead be an ONNX graph run by
ONNX Runtime on the CPU (`backend = "onnx"`, D40), on any platform sidekick
builds for. Whether it's sidekick's own export (`--format onnx`) or a
published one, the graph outputs token-level `last_hidden_state`, which
sidekick pools (`mean`, `cls` or `last_token`); a head that does more than
pooling, such as EmbeddingGemma's, stays in the graph, and its manifest says
`pooling = "none"`. An ONNX build keeps the model's id, so clients address
the same model everywhere. The daemon needs the `onnx` cargo feature and
ONNX Runtime (1.28.3, or 1.23.2 on Intel macOS). Release archives ship
`libonnxruntime` beside `sidekickd`. To use an existing install, name
the library with `onnx_runtime` in the config (the way under launchd),
or `ORT_DYLIB_PATH`. The daemon checks `onnx_runtime`, then
`ORT_DYLIB_PATH`, then the library beside itself, and never searches system
paths. The runtime loads at startup only when there's an
onnx model. If it's missing, the onnx models are skipped with the reason
and everything else serves. Each manifest's
graph is checked against its `[io]` when the directory is scanned.
`/health`'s `onnx` section reports the runtime and each model's threads.

```toml
id = "bge-small-en-v1.5"
backend = "onnx"
artifact = "model.onnx"
tokenizer = "tokenizer.json"
dims = 384
pooling = "cls"
max_seq_len = 512

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"  # when the graph takes it: fed zeros
output = "last_hidden_state"

[onnx]
default_intra_op_threads = 16      # optional: let this model use more than 8 threads
```

An onnx model runs on the machine's fast physical cores, at most 8 unless
its manifest's `[onnx] default_intra_op_threads` raises that limit. A value
under 8 has no effect. The config's `onnx_intra_op_threads`, or a model's
`intra_op_threads` under `[models."<id>"]`, sets threads outright.

Classifier manifests are in [examples/classifiers/](examples/classifiers/):
laya-en, from `tools/convert_laya.py`, and nlptown-sentiment, from
`tools/convert_bert_classifier.py`. Each converter writes the artifacts and
installs the manifest. [docs/MODELS.md](docs/MODELS.md) has their measured
accuracy on each compute path.

### Fetching a published build

`sidekickd fetch` downloads a build
sidekick publishes on Hugging Face ([docs/PUBLISHING.md](docs/PUBLISHING.md)),
verifies every file, and installs it in the models directory (D51):

```sh
sidekickd fetch bge-small-en-v1.5                  # the Core ML build (on Linux: the ONNX one)
sidekickd fetch qwen3-embedding-0.6b --buckets 128,512,2048
sidekickd fetch bge-small-en-v1.5 --backend onnx
sidekickd fetch <model> --accept-terms <model>     # a model whose repository publishes terms
```

The ledger records where and at which revision each build was published,
so the model id is enough; `--repo` and `--revision` name another
repository or revision (a build not published yet needs `--repo`).

- **Downloads go into the Hugging Face cache,** through the `hf` CLI when
  it's installed (its token, Xet transfers and settings, such as `HF_HOME`,
  `HF_HUB_CACHE` and `HF_HUB_OFFLINE`), else by sidekickd itself, anonymously
  through `curl`, into the same cache in the same layout. Each sees the
  other's downloads as cached. A gated repository needs `hf`, signed in to
  an account that accepted its terms on the Hub.
- **Every file is verified** against the Hub's hash for it and against
  sidekick's checksum ledger, which is built into sidekickd. A file that
  matches neither is refused.
- **The model directory is hard links to the cached files,** so the model
  takes no more disk than the cache does, and deleting either one leaves
  the other whole. A program whose files are symlinks out of its own
  directory, as a Hugging Face snapshot's are, can run off the ANE without
  an error (bge-small-en-v1.5 does, D51), so a symlinked install wouldn't
  do; where the cache is on another volume, the files are copied.
- **`--buckets` fetches part of a Core ML build,** and writes the manifest
  for exactly those buckets; `--variant` picks a build (`fp16`,
  `fp16-max2048`) when a repository has several. `fetch.toml` in the model
  directory records the repository, the commit and every file's sha256.
- **A Core ML build is compiled once installed:** fetch loads each bucket
  it installed, on the compute units the daemon will serve it on, so that
  the first requests don't wait on Core ML's compile (D52). `--no-warm`
  skips it, and so does `warm = false` under the model's
  `[models."<id>"]` in the config.

### Installing a published build by hand

Without `sidekickd fetch`, download one build's directory from its
repository (`sidekickd/<model>-coreml` or `-onnx` on Hugging Face; the model card names its
builds), check it against its `<build>.sha256`, and move it into the models
directory under the model's id:

```sh
hf download sidekickd/bge-small-en-v1.5-coreml --include 'fp16/*' --include 'fp16.sha256' \
    --local-dir ./bge-small-download
(cd ./bge-small-download/fp16 && shasum -a 256 -c ../fp16.sha256)
mv ./bge-small-download/fp16 "$HOME/Library/Application Support/sidekick/models/bge-small-en-v1.5"
```

- **Use `--local-dir`.** Don't point the models directory at a Hugging
  Face cache snapshot (`hf download` without it): every file there is a
  symlink out of its program's directory, and Core ML can run such a
  program off the Neural Engine without an error (D51), so sidekick
  refuses to serve it there.
  `--local-dir` writes real files.
- **Shared weights arrive as copies.** A build whose buckets share one
  weights file (D49) is downloaded this way as one copy per bucket, so it
  takes more disk than fetch installs. `python3 -m
  sidekick_convert.shared_weights link <model dir>`, run from sidekick's
  `tools/`, turns the copies back into hard links.
- **A gated repository** (a model whose card asks you to accept terms)
  needs `hf auth login` with an account that accepted them on the Hub.
- **On Linux,** or with another `models_dir` in the config, move the
  directory there instead.

## Configuration

`config.toml` in sidekick's config directory: `~/Library/Application
Support/sidekick/config.toml` on macOS, `~/.config/sidekick/config.toml` on
Linux, or any file passed with `--config`. sidekickd's startup log names
the file it read. Every key is optional (defaults shown):

```toml
addr = "127.0.0.1:8790"        # loopback only by default
# models_dir = "..."           # default: <data dir>/sidekick/models
# api_key = "..."              # require Authorization: Bearer <key> on /v1
session_ttl_secs = 300         # Foundation Models session reuse window
model_idle_ttl_secs = 900      # model residency (embedders, classifiers) after last use
request_timeout_secs = 60      # a request's work: generation, or a prediction after any load it waited on
load_timeout_secs = 900        # how long a request waits on its model's and buckets' first loads
ignore_ane_weight_cap = false  # load ANE-served models past Core ML's 1 GiB weight limit
ignore_cpu_seq_cap = false     # serve cpu_only models past 1,024 tokens (results vary by bucket)
ignore_ane_seq_cap = false     # serve ANE models past 2,048 tokens or their ane_max_seq_len (slow compiles, failures)
report_compute_plans = false   # read each loaded bucket's Core ML placement live (costs a second compile)
warm_on_start = true           # compile every Core ML bucket at startup, in the background (D52)
warm_budget_secs = 1800        # start no warm-up load after this long; 0 for no limit
# onnx_intra_op_threads = 8    # inference threads for every onnx model (default: fast physical cores, at most 8)
# onnx_runtime = "..."         # libonnxruntime to load instead of the bundled one (before ORT_DYLIB_PATH)

# Per model, by id: serve it on other compute units than its manifest names,
# e.g. keep a companion model off the GPU a larger model uses, or serve a long
# model's longest buckets on the GPU. Core ML models only; the ANE weight limit
# and the CPU and ANE length caps apply as to the manifest's.
# [models."agent-jev"]
# compute_units = "cpu_and_ne"   # cpu_and_ne, cpu_and_gpu, cpu_only or all
# [models."bge-small-en-v1.5"]
# intra_op_threads = 2           # an onnx model's inference threads
# [models."qwen3-embedding-0.6b"]
# warm = false                   # leave a Core ML model out of startup and fetch warm-ups
```

CLI flags override the file: `sidekickd --addr ... --models-dir ... --api-key ...
--ignore-ane-weight-cap --ignore-cpu-seq-cap --no-warm --warm-budget-secs N`.

## Warming Core ML's compile cache

The first time sidekickd loads a Core ML bucket, Core ML compiles it for
the device: seconds for a small model on the ANE, minutes for a long
bucket of a large one. Core ML keeps the result on disk, for this
executable and user, the model's path and the macOS build, so later loads
take a fraction of that. sidekickd compiles ahead of requests (D52):

- **At startup, in the background,** in a `sidekickd warm` child process,
  every bucket the daemon serves, on the compute units it serves it on,
  cold ones first. After a macOS update, a model move or a fresh install,
  that's where the compiles happen, instead of in the first requests. A
  request for a bucket being compiled waits for that compile and then
  loads from the cache; requests for other buckets don't wait. Nothing
  stays loaded that no request asked for. `--no-warm` or `warm_on_start =
  false` turns it off; `warm_budget_secs` (default 30 minutes) stops it
  starting new loads after that long.
- **On demand:** `sidekickd warm [<model>…] [--buckets 128,512]
  [--budget-secs N]`, or `--list` to see each bucket's state without
  loading anything. Run it as the user, and with the `sidekickd`, that the
  daemon runs as: Core ML's cache is per user and per executable name.
- **After `sidekickd fetch`,** for the buckets fetched; `--no-warm` skips
  it.

`/v1/models` lists each Core ML model's buckets as `compile_cache`
(`warm`, `warming`, `cold` or `failed`), and `/health`'s `warmup` shows
the startup warm-up's progress and each bucket's load times. Measured on
an M1 Max: bge-small-en-v1.5's first request per bucket took 4.0–5.4 s
cold and 0.9–1.3 s warmed.

## Memory

What sidekick holds while a model is loaded, measured on an M1 Max under
macOS 27.0 (numbers are approximate: system-wide counters move with
everything else the machine runs):

- **Models load lazily.** A model loads on its first request, and each of
  a Core ML model's sequence-length buckets, its own compiled program, loads
  the first
  time an input needs it. Loading the model checks every bucket's artifact
  from its description, and keeps none of them resident. A short-input
  workload never loads the large buckets, and a long-input one never loads
  the small ones. (Before 0.7.0 the smallest bucket also loaded with the
  model and stayed: on the GPU that cost chunked agent-jev, serving
  1,024-token requests, 2.4 GB of resident memory where it now uses
  1.3 GB.)
- **On the ANE (the default), little of it is counted against
  sidekickd.** Its own footprint stays at tens to a couple of hundred MB:
  57 MB with bge-small's three buckets, 55 MB with nlptown-sentiment's
  three (1 GB of weights), about 200 MB with laya-typed-decisions' four
  (3.2 GB) or Lumma-fev's five (2.3 GB). The rest shows up elsewhere in
  Activity Monitor, in two parts:
  - most of it under **Cached Files**: the ANE reads its compiled program,
    about the size of the bucket's weights, through macOS's file cache,
    which macOS can reclaim when another app needs the memory;
  - a smaller share under **Wired Memory**, which the ANE pins while the
    model is resident and macOS can't reclaim: about 0.8 GB with
    laya-typed-decisions' four buckets. It is released when the model
    unloads.
- **On the GPU (`compute_units = "cpu_and_gpu"`), the weights are mapped
  into sidekickd.** Each loaded bucket adds about its weight size to its
  memory: nlptown-sentiment's resident size grows from 0.6 to 1.3 GB as
  its three 320 MB buckets load, about 0.2 GB of it counted as footprint.
- **The first load of a bucket costs more, once.** It compiles the bucket:
  Apple's ANE compiler service grew to 1.3 GB, for four minutes, compiling
  Lumma-fev's 2,048-token bucket. The compiled programs stay on disk, about
  the weights' size again per bucket and compute path (3.5 GB for
  laya-typed-decisions), so later loads take seconds. Where they're kept
  depends on the path:
  - **GPU and CPU:** in sidekickd's own cache,
    `~/Library/Caches/sidekickd/com.apple.e5rt.e5bundlecache/<macOS build>/`.
  - **ANE:** in the system's ANE cache, kept by Apple's `aned` service on
    the system volume. A user can't delete it (System Integrity Protection
    protects it), and deleting `~/Library/Caches/sidekickd` doesn't free
    it: that holds only about 12 kB per ANE bucket.

  Core ML reuses a compiled program for the same model files at the same
  path (a copy of the model elsewhere compiles again), loaded by a process
  of the same name and, on the ANE, with the same code-signing
  identifier, which is why release archives sign sidekickd with a fixed
  one (see Quickstart). The cache directory is named for the macOS build,
  and Apple's Core AI documentation says an OS update invalidates
  specialized models, so every bucket is expected to compile again after
  one (not yet measured across an update; docs/design/coreml-cache.md has
  the measurements).

  A request that has to wait for a compile doesn't count it against
  `request_timeout_secs` (default 60): its deadline
  starts once its buckets are loaded. Loading is bounded instead by
  `load_timeout_secs` (default 900, well past the four minutes measured);
  a load that outlasts it is a 504 with code `load_timeout`, the load
  continues, and a retry finds the bucket ready.
- **Releasing memory.** A model unloads `model_idle_ttl_secs` (default
  900) after its last use, whether or not anything else reaches the
  daemon. Lower it to give memory back sooner, at the cost of a reload
  (seconds, from the compiled cache) the next time the model is used.
  sidekickd's footprint shrinks over the following seconds rather than at
  once, as Core ML releases what it held.
- **ONNX models hold their weights in sidekickd's own memory.** They have
  no buckets, and where Core ML maps weights from the file, ONNX Runtime
  loads them: about +0.9 GB for gte-modernbert and +1.7 GB for
  Qwen3-Embedding-0.6B in fp32. An fp16-weight export halves the file but
  holds more memory than fp32 (+2.7 GB for Qwen3-Embedding), since its
  weights are upcast when loaded. See the
  `onnx-cpu` section of [docs/MODELS.md](docs/MODELS.md).

## Workspace layout

| Crate | What it is |
|---|---|
| `sidekick-core` | Backend-neutral traits and types: `ChatBackend`, `Embedder`, `Classifier`, availability states, model manifests and registry. No Apple dependencies. |
| `sidekick-coreml` | Small safe wrapper over `objc2-core-ml`: load, compute units, int32-in/float-out predictions. macOS only; empty stub elsewhere. |
| `sidekick-onnx` | ONNX Runtime on the CPU: finds and loads `libonnxruntime` at run time, runs a graph in a session, and sets inference threads from the machine's fast cores. `ort` comes in only with its `runtime` feature, which `sidekick-embed`'s `onnx` feature turns on. |
| `sidekick-fm` | Foundation Models backend: Swift C-ABI shim built by `build.rs` (macOS 26.4+ SDK; 27 SDK for the macOS 27 features), guided generation from JSON Schema, TTL'd session reuse keyed by conversation prefix. Stub elsewhere. |
| `sidekick-embed` | Encoder pipelines: Core ML/ANE embedders and classifiers (feature `coreml`, over `sidekick-coreml`), ONNX Runtime CPU embedders (feature `onnx`, over `sidekick-onnx`), laya's input format, and the static floor tier. |
| `sidekick-server` | `sidekickd`, the OpenAI-compatible daemon. |
| `sidekick-embed-ffi` | `libsidekick.dylib`, the C ABI for in-process embeddings (Core ML and static models; built without ONNX support). |

## Development

```sh
cargo test --workspace                 # runs anywhere, including Linux
cargo check --workspace --target aarch64-apple-darwin --features sidekick-embed/coreml
# On a Mac with converted models and references (docs/MODELS.md):
cargo run --release -p sidekick-embed --features coreml,onnx --example parity
# ONNX models only, on any OS (needs ORT_DYLIB_PATH):
cargo run --release -p sidekick-embed --features onnx --example parity
```

The second command is the cross-check CI uses to keep the macOS-only code
honest from non-Mac machines. On a Mac, `cargo test --workspace --features
sidekick-server/coreml` additionally builds the Swift shim. That needs Xcode
26.4 or later; older SDKs are a build error (set `SIDEKICK_FM_STUB=1` to build
the stub backend instead). Build with **Xcode 27** to get the macOS 27
Foundation Models features (real token usage, model variant, typed errors):
they are compiled in only against the 27 SDK. `sidekickd --version` reports
which SDK a binary was built with; after switching Xcode with
`xcode-select`, run `cargo clean -p sidekick-fm` so the shim is rebuilt.

### Contributing

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, …).
- **Branching** is light, trunk-based: `main` is the trunk and is kept
  releasable; feature and chore work happens on short-lived branches
  (`feat/…`, `chore/…`, `fix/…`) that merge back into `main`.
- **Releases** are annotated `vX.Y.Z` tags cut from `main`
  ([semver](https://semver.org/)). Each has release notes in
  [`docs/releases/`](docs/releases/), written before tagging. The tag drives
  [`release.yml`](.github/workflows/release.yml), which publishes the notes
  with the build artifacts on the Releases page.

See [AGENTS.md](AGENTS.md) for the full contributor guidelines, including
content-hygiene rules for this public repository.
