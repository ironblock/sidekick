# The HTTP API

Every route `sidekickd` serves, the fields each takes, and its limits and
errors. The daemon speaks OpenAI's API, and vLLM's and Cohere's shapes for
routes OpenAI doesn't have, so a client written for one of those works
unchanged; this document gives what sidekick adds or does differently.
Two design documents hold the full contracts of classification
([design/classify.md](design/classify.md)) and reranking
([design/rerank.md](design/rerank.md)).

## Contents

- [Routes](#routes)
- [Authentication](#authentication)
- [Errors](#errors)
- [Provenance headers](#provenance-headers)
- [Chat](#chat)
- [Embeddings](#embeddings)
- [Tokenize](#tokenize)
- [Classification](#classification)
- [Rerank](#rerank)
- [Listing models and health](#listing-models-and-health)

## Routes

| Route | What it serves |
|---|---|
| `POST /v1/chat/completions` | Chat models, by `model`: Apple Foundation Models (`apple-fm`) on Apple silicon, macOS 26+, Apple Intelligence on ([BACKENDS.md](BACKENDS.md#foundation-models)), and open models from the models directory on the Neural Engine ([below](#open-models)) |
| `POST /v1/embeddings` | Embedders, OpenAI's shape |
| `POST /v2/embed` | The embedders, in Cohere's v2 shape |
| `POST /v1/classify` | Classifiers, including zero-shot and NLI: vLLM's `/classify` (Core ML, macOS) |
| `POST /v1/rerank`, `POST /rerank` | Cross-encoder rerankers: vLLM's (Jina's) shape (Core ML, macOS) |
| `POST /v2/rerank` | The same, in Cohere's v2 shape |
| `POST /tokenize`, `POST /detokenize` | An embedder's or text classifier's tokenizer: vLLM's shape |
| `GET /v1/models` | What this machine serves: each model's task, backend, limits and prompts |
| `GET /health` | Availability per backend, and why anything is unavailable |

The daemon listens on `127.0.0.1:8790` by default ([CONFIG.md](CONFIG.md)).
A request body is at most 2 MiB.

## Authentication

With no `api_key` configured (the default), no route needs a key, and an
OpenAI client may send any. With one
([CONFIG.md](CONFIG.md#the-config-file)), every route except `/health`
needs `Authorization: Bearer <key>`, `/v1/models` included; a missing or
wrong key is a 401 with code `invalid_api_key`. `/health` stays open for
probes, and carries no credentials
([D13](DECISIONS.md#d13--default-bind-1270018790-health-unauthenticated)).

## Errors

Every error, malformed JSON included, is OpenAI's shape:
`{"error": {"message", "type", "code"}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request_error` | A field the route rejects, named in the message: an unknown option, an input over the model's limit, or a model sent to a route that doesn't serve its task (the message names the route that does) |
| 400 | `context_length_exceeded`, `content_filter`, `unsupported_language`, `unsupported_schema` | Chat: a prompt over the model's context, a guardrail refusal ([below](#guardrail-refusals)), a language the model doesn't support, or a `response_format` schema outside the [supported subset](#structured-output) |
| 401 | `invalid_api_key` | A key is configured, and the request's is missing or wrong |
| 404 | `model_not_found` | No model by that id is served |
| 413 | | A body over 2 MiB |
| 429 | `rate_limit_exceeded` | Foundation Models is rate-limiting; `Retry-After` says when to retry |
| 503 | `backend_unavailable` | A backend this machine or build doesn't have, or an open model whose decoder has failed (a hung Neural Engine call; a restart recovers), with the reason, which `/health` reports too. Never a hang. |
| 503 | `backend_busy` | Foundation Models is busy, or the ANE chat lane's queue is full ([Open models](#open-models)); retry, after `Retry-After` when it is set |
| 504 | `timeout` | The request's work ran past its model's request timeout ([CONFIG.md](CONFIG.md#per-model-settings)) |
| 504 | `load_timeout` | Its model's or buckets' first load ran past `load_timeout_secs`. `Retry-After` is set: the load continues, so a retry finds it ready |

## Provenance headers

Every inference response carries provenance headers. They follow OpenAI's
`openai-model` / `openai-version` style:
- `sidekick-version`.
- `sidekick-model`: `<id>@<revision>` when the manifest has a `source`,
  otherwise `<id>`. For `apple-fm` it's the Foundation Models variant id,
  read in the background and cached; until it's known, `apple-fm`.
- `sidekick-compute-units`: the configuration the serving instance was
  loaded with: a Core ML model's `compute_units` (`cpu_and_ne` unless its
  manifest says otherwise; see [Compute units](BACKENDS.md#compute-units)), `cpu` for static and ONNX
  models ([D40](DECISIONS.md#d40--an-onnx-runtime-cpu-backend-for-embeddings)).
  `apple-fm` omits it. It reports configuration, not the executing device, which
  Core ML doesn't expose.
- `sidekick-buckets`: the sequence-length bucket each input ran in, in
  input order and comma-separated (`256,256,512`; one input gives `256`).
  ONNX models have no buckets and don't send it.
  Each bucket is its own compiled program ([D15](DECISIONS.md#d15--per-bucket-static-artifacts-pooling-baked-into-the-model)), so this names the
  program that answered each input, the one whose placement `/v1/models`
  reports per bucket (see [Where Core ML places operations](BACKENDS.md#where-core-ml-places-operations)). On
  `/v1/classify`, `/v1/embeddings`, `/v1/rerank`, `/rerank`, `/v2/rerank`
  and `/v2/embed`. A rerank pair is one input, in request order rather
  than the sorted results' order. Chat and static embedders omit it.

## Chat

`POST /v1/chat/completions` serves each chat model `/v1/models` lists with
`task: "text-generation"`: Apple's Foundation Models as `apple-fm`, and
each open model installed in the models directory, by its id
([Open models](#open-models)). A request goes to the model its `model` names, exactly: there
is no default, no alias and no fallback from one model to another. An id
another route serves (an embedder, a classifier) is a 400 naming the route
that serves it; any other id is a 404 `model_not_found`.

Point an OpenAI provider (OpenCode, an editor, a script) at
`http://127.0.0.1:8790/v1` with that model. Constrained output works via
the standard `response_format`; for example, a session title:

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

The schema is enforced by the model's constrained decoding (Foundation
Models' guided generation, or llguidance on an [open model](#open-models)),
and both backends also show it to the model: Foundation Models puts it in
its prompt, and an open model's system message carries it. It must be in
the [supported subset](#structured-output). Check `finish_reason`: a reply cut
by `max_tokens` reports `"length"`, and for constrained output that means
the JSON is incomplete: on `apple-fm` it may be missing properties, and on
an open model it stops mid-value and won't parse.

`stop` (one string or up to four) is supported, except together with
`json_schema`.

Chat streams for real: `stream: true` sends text as the model generates it.
Schema-constrained replies arrive whole on `apple-fm`, and as they are
generated on an open model (`guided_streaming` in `/v1/models`). On macOS 27 (built with Xcode 27)
chat reports real token usage, including
cached tokens when a follow-up reuses its session. `/health` reports which
Foundation Models variant the Mac has and its context size. Over-long
prompts are a proper 400 ([D21](DECISIONS.md#d21--macos-27-real-usage-and-model-facts-typed-errors-the-shim-stays)).

### Open models

An open causal language model (the first is MiniCPM5-2B, as `minicpm5-2b`)
answers the same request and returns the same shapes as `apple-fm`. It
runs on the Neural Engine, beside a large model that owns the GPU
([D53](DECISIONS.md#d53--open-llms-on-the-chat-route-a-stateful-ane-decoder-beside-foundation-models);
[open-llm.md](design/open-llm.md) has the design). Where it differs:

- **The prompt** is the checkpoint's own chat template, rendered as
  transformers renders it, with thinking off. `system` and `developer`
  messages are the system role. The text of the model's special tokens is
  rewritten in message content (`<|im_start|>` becomes `<¦im_start¦>`, and
  others get a zero-width word joiner after their first character), so a
  message can't forge a turn. Text that still tokenizes as a special token
  (one the tokenizer's normalizer forms, such as full-width brackets) is a
  400.
- **The context** (`context_length` in `/v1/models`) is the prompt plus
  the completion. A prompt plus `max_tokens` (or `max_completion_tokens`)
  past it is a 400 `context_length_exceeded`; `max_tokens` omitted means
  the rest of the context.
- **`usage` is exact:** `prompt_tokens` counts the rendered template,
  `completion_tokens` every generated token, the stop token included;
  `cached_tokens` is 0.
- **The reply** ends at one of the model's stop tokens
  (`finish_reason: "stop"`), at a `stop` sequence (the reply ends before
  it; `"stop"`), or at `max_tokens` or the end of the context
  (`"length"`). It is the model's text as generated: the leading
  `Assistant:` label that `apple-fm`'s replies are cleared of isn't
  removed.
- **Streaming** sends text as it is generated, as whole UTF-8 characters,
  holding back only text that could still become a `stop` sequence.
- **`response_format` `json_schema`** is enforced with
  [llguidance](https://github.com/guidance-ai/llguidance): each token is
  sampled from the tokens the schema's grammar allows, so a reply that
  ends with `finish_reason: "stop"` is JSON that matches the schema, on
  one line, with `, ` and `: ` between values and its keys in schema
  order (`{"name": "Ada", "age": 36}`). The reply ends when the JSON is
  complete, without a stop token, and streams as it is generated. That is
  a guarantee of valid JSON, not of correct values: constraining an open
  model fixed its malformed replies but not its wrong answers (the chat
  benchmark's [findings](CHAT-BENCHMARK.md#findings)). The grammar only
  forces what may come next, so the model also reads the schema: it is
  written, with its `description`s, required properties and enum
  choices, into the system message (after any system text the request
  sent, under a `# Response format` heading), where the model's template
  puts tool definitions, and counts in `prompt_tokens` and against the
  context. `[models."<id>"] schema_in_prompt = false`
  ([CONFIG.md](CONFIG.md#per-model-settings)) leaves the prompt as sent,
  for a client whose prompt already states the schema. Integers and
  numbers are as long as the model writes them: JSON allows exponents
  past what a double holds. The first guided request a model serves after
  sidekickd starts
  builds its tokenizer's index for the grammar engine first (about 0.4 s
  for MiniCPM5-2B). (`json_object` asks for JSON in the prompt, as on
  `apple-fm`.)
- **No guardrails:** an open model never answers `content_filter`.
- **One request at a time, shared with `apple-fm`.** Open models and
  `apple-fm` take turns on the Neural Engine in arrival order, since a
  Foundation Models request overlapping a generation slows both about
  25×. One open model is loaded at a time: a request for another waits
  its turn, then unloads it. Requests past the queue (`queue_depth`,
  default 4, [CONFIG.md](CONFIG.md#per-model-settings)) are a 503
  `backend_busy` with `Retry-After: 5`, for `apple-fm` too. Time in the
  queue counts against the request's timeout (`request_timeout_secs`;
  120 s for MiniCPM5-2B, from its manifest), including time spent behind
  another request's model load. A load the request itself waits for, once
  its turn has come, doesn't count, and is bounded by
  `load_timeout_secs`. A request whose time ran out in the queue doesn't
  start. A streamed request whose client leaves while it waits gives up its
  place.
- **Provenance:** `sidekick-model` is `<id>@<revision>` and
  `sidekick-compute-units` is `cpu_and_ne`.

**What changes for `apple-fm` when an open model is installed.** Without
one, `apple-fm` runs as before. With one, `apple-fm` shares the open
models' queue:
- its requests run one at a time, in arrival order with the open models';
- past the queue, a request is a 503 `backend_busy` with `Retry-After: 5`;
- its timeout counts time spent in the queue;
- a request that timed out keeps its turn until Foundation Models has
  actually stopped generating, so the next request doesn't overlap it.

### Guardrail refusals

A model with guardrails (`guardrails: true` in `/v1/models`;
Foundation Models has them) can refuse a request. A refusal before any
text is a 400 with code `content_filter`; one after text has streamed
ends the stream with `finish_reason: "content_filter"`. The daemon
never retries a refusal or answers it with another model. The 400's
message names the model and says whether the output was guided. When it
was, the error also carries `param: "response_format"`, because guided
output can be refused where the same prompt as plain text isn't
([#178](https://github.com/ironblock/sidekick/issues/178)), so a client
can retry without `response_format`, or with another model:

```json
{"error": {
  "message": "`apple-fm`'s guardrails refused this request, whose output was guided by response_format json_schema: May contain sensitive content. Guided output can be refused where plain text isn't; a retry without response_format, or with another model, may succeed",
  "type": "invalid_request_error",
  "param": "response_format",
  "code": "content_filter"
}}
```

### Sampling

| Field | Meaning |
|---|---|
| `temperature` | 0 to 2 (outside is a 400). 0 is greedy decoding: the most likely token every step, so a request repeats byte for byte on the same Mac and macOS. Above 0, sampling at that temperature. Omitted, the model's default. On an open model, ties go to the lower token id, and the default is the checkpoint's (`generation_config.json`; MiniCPM5-2B: 1.0), so greedy isn't the default. |
| `top_p` | Nucleus sampling, 0 to 1: Foundation Models' random sampling mode with that probability threshold; on an open model, sampling from the most likely tokens whose probabilities first add up to `top_p`. 0 is greedy. Outside 0–1 is a 400. Omitted on an open model, the checkpoint's (MiniCPM5-2B: 0.95). |
| `seed` | A non-negative integer that makes sampling at a temperature above 0 repeatable. With `seed` and no `top_p`, `apple-fm` samples the whole distribution (`top_p` 1). |

A request with neither `top_p` nor `seed` samples as Foundation Models
does by default. Other sampling fields (penalties, `logit_bias`) and
`user` are ignored; fields that change what the reply is (`n` > 1, tools,
`logprobs`) are a 400 ([D22](DECISIONS.md#d22--reject-request-parameters-the-daemon-cant-honor)).

### Structured output

`response_format: {"type": "json_schema", ...}` constrains the reply to a
JSON Schema. Every chat backend reads the schema through one layer, which
accepts the subset below and refuses everything else with a 400
`unsupported_schema` whose message names the keyword and its JSON pointer
(``unsupported keyword `pattern` at `/properties/sku` ``). A keyword is in
the subset only when every chat backend can honor it, so no schema is
answered as if part of it weren't there. `json_schema.strict` is accepted
either way: output is always constrained.

| Keyword | Supported |
|---|---|
| `type` | `object`, `array`, `string`, `integer`, `number`, `boolean`; one type, optionally with `"null"` (see nullable below). Inferred as `object` from `properties`, `array` from `items`, `string` from `enum`/`const`; otherwise required |
| `properties` | Generated, and returned, in the order sent |
| `required` | Properties not listed may be left out (JSON Schema's meaning); every name listed must be in `properties` |
| `additionalProperties` | Absent or `false`: objects are closed |
| `enum`, `const` | Strings only (and `null` in a nullable enum) |
| `items`, `minItems`, `maxItems` | `items` is required on an array, and is one schema |
| `anyOf` | Only as a nullable union: one schema and `{"type": "null"}` |
| `description` | Shown to the model with the schema: in Foundation Models' prompt, and in an open model's system message |
| `title`, `$schema`, `examples`, `default` | Accepted and ignored |

Refused, among others: `anyOf` of anything else, `oneOf`, `allOf`, `not`,
`$ref` and `$defs` (write the schema inline), a union of types
(`["string", "integer"]`), `enum` of non-strings, `minimum`, `maximum`,
`multipleOf`, `pattern`, `format`, `minLength`, `maxLength`,
`patternProperties`, `uniqueItems`, `prefixItems`, and any keyword not
listed above.

**Nullable properties.** `{"type": ["string", "null"]}` and
`{"anyOf": [{"type": "string"}, {"type": "null"}]}` are supported as an
object property's value, and nowhere else (not as the schema root or an
array's items). Foundation Models generates such a property as one it may
leave out, and the reply carries `null` where it did, when the property is
`required`; an optional nullable property it leaves out stays absent. So a
`required` nullable property is the way to ask for every field, with
`null` for what the text doesn't state. An open model writes a nullable
property's value or `null`: a `required` one is always there, and an
optional one may be left out.

**Key order.** A guided reply, on either backend, is JSON on one line with
its keys in the order the schema lists them, so identical requests at
temperature 0 give identical bytes. `apple-fm`'s is compact
(`{"name":"Ada","age":36}`); an open model's has `, ` and `: ` between
values (`{"name": "Ada", "age": 36}`), the spelling models write
unprompted, since forcing compact JSON on one degrades what it writes
([open-llm.md](design/open-llm.md#constrained-decoding-on-the-open-path)). The model also generates the properties in that order,
which can change what it writes: put the fields it should decide first
first.

## Embeddings

`POST /v1/embeddings` is OpenAI's embeddings route: `input` (one string or
up to 256), `dimensions` for Matryoshka models, and `encoding_format:
"base64"`.

### Prompts

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
  a symmetric task, or for no prompt where a model's evaluation used none
  (e5-small-v2 outside retrieval). Otherwise sidekick's prompt goes before yours, and the
  model sees a prompt it was never trained on. Under `truncate: "START"`,
  sidekick keeps its own prompt whole but can't tell yours from the text,
  so it may cut yours: with `none`, use `END`, or keep inputs within the
  limit.

Some models have no prompt for one side or both, and then that
`input_type` is the same as `"none"`. Each model's prompts are in
[MODELS.md](MODELS.md#prompts-input_type), and `/v1/models`
lists each installed model's as `prompts` (`{"query": "…", "document":
"…"}`, `""` for none). [D9](DECISIONS.md#d9--embedding-purpose-via-non-standard-input_type-amended-none-turns-the-prompt-off) records the choice, and [D45](DECISIONS.md#d45--named-prompts-chosen-by-prompt_name) measures
what `START` costs a prompt typed into the text.

### Named prompts

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
([MODELS.md](MODELS.md#named-prompts-prompt_name)), and
truncate with `END`, since `START` would cut an instruction it can't
tell from the text. [D45](DECISIONS.md#d45--named-prompts-chosen-by-prompt_name) records the choice.

### Input length and truncation

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
else is a 400, and `null` is the default ([D43](DECISIONS.md#d43--over-length-v1embeddings-input-is-refused-by-default)). OpenAI's Python SDK sends it through `extra_body`, as in
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

## Tokenize

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
sooner ([D44](DECISIONS.md#d44--tokenize-and-detokenize-count-as-the-embedding-path-does)). `max_tokens`, `/v2/embed`'s, makes `fits` measure against
it instead, as `/v2/embed` with `truncate: "NONE"` does.
At most 4 MiB of text is tokenized at once, across requests; a request
waits for room within the request timeout.
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
aren't mounted under `/v1/`, as in vLLM and SGLang ([D44](DECISIONS.md#d44--tokenize-and-detokenize-count-as-the-embedding-path-does)).

## Classification

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
`nli_classes`. `nli_probs`, and a manifest that names the model's
classes, need sidekickd 0.15.0 or later: an older daemon skips such a
manifest.

`calibration: "model"` applies the temperature the model declares for the
question type and label count; `/v1/models` lists each classifier's
temperatures, labels or `max_labels`, `max_batch`, and the extension fields
it accepts. The full contract is in
[design/classify.md](design/classify.md).

## Rerank

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
[design/rerank.md](design/rerank.md).

## Listing models and health

`GET /v1/models` lists what this machine serves, OpenAI's shape with
sidekick's fields beside it: for every model its task and backend; for a
chat model its `context_length` (prompt and completion together), the
optional request fields it honors (`honored_fields`), whether guided
output streams as it is generated (`guided_streaming`), whether it
has guardrails (`guardrails`), and for an open model its `compute_units`,
the macOS its build was graded on (`graded_os`, from its `chat.toml`; absent
when the build doesn't record one) and its decoder's `compile_cache` state;
for an embedder its `max_seq_len`, `buckets`, `dims`, `matryoshka`, `prompts`
and `named_prompts`; for a classifier its labels or `max_labels`,
`max_batch`, `extensions`, `required` fields and calibration; for a Core
ML model its `compute_units`, any `seq_cap`, and each bucket's
`placement` ([BACKENDS.md](BACKENDS.md#where-core-ml-places-operations)).
Nothing is loaded to answer.

`GET /health` reports availability per backend and why anything is
unavailable: each chat model's state and the reason it is unavailable,
variant, context size and request timeout, by id (`chat.models`), and
for an open model its state (`unloaded`, `loading`, `ready`, `failed`
with the reason: its decoder hung, and a restart recovers it; or
`load_failed`, its last load's error in `load_error`, which the next request
retries), its own `queue_depth` and the lane's (`lane_queue_depth`), loads,
its decoder's calls, stalls (calls past
five times the warm median: other work on the Neural Engine) and warm
median, and its Neural Engine share (`ane`: measured at its load, and
estimated from its weights before); the ANE chat lane (`chat.lane`: the
request running, those waiting, the resident open model, and the
process's Neural Engine footprint against `ane_budget_mib`); the
backends the build serves (`backends`: `coreml`, `onnx`, `static`); the
models resident; each manifest skipped, with the reason (`skipped_models`);
length caps, compute-unit overrides and compute plans; and ONNX Runtime's
state (`onnx`). It never needs the API key and carries no credentials.
