# Rerank: `POST /v1/rerank`, `/v2/rerank` and `/v2/embed`

The contract, manifest, model interface and validation for sidekick's
rerankers, plus Cohere's `/v2/embed` over the existing embedders. It builds
on the classifier runtime (docs/design/classify.md, D28): a reranker is a
classifier whose input is a (query, document) pair and whose output is one
relevance score.

Principles, as for classification:
- **Follow the standard exactly where one exists.** `/v1/rerank` and
  `/rerank` are vLLM's `RerankRequest`/`RerankResponse`, the Jina shape that
  vLLM, llama.cpp, LocalAI and Infinity serve. `/v2/rerank` and `/v2/embed`
  are Cohere's v2 shapes.
- **Extend only where no standard exists.** The one extension is Jina's and
  Cohere v1's `return_documents`, which vLLM doesn't define.
- **Fail loudly (D22).** Every field these standards define is honored,
  accepted in the form that changes nothing, or rejected with a 400.
- **Serve one shape per route.** TEI's `/rerank` (`texts` in, a bare list
  out) and SGLang's `/v1/rerank` (a bare list out) collide with vLLM's
  routes. Serving both would mean sniffing request bodies, so sidekick
  serves vLLM's shape only.

## `POST /v1/rerank` and `POST /rerank`

```json
{"model": "<id>", "query": "text", "documents": ["text", ...], "top_n": 3}
```

```json
{"id": "score-<uuid>", "model": "<id>",
 "usage": {"prompt_tokens": 412, "total_tokens": 412},
 "results": [{"index": 2, "document": {"text": "..."}, "relevance_score": 0.93}, ...]}
```

| field | source | sidekick |
|---|---|---|
| `model`, `query` (str), `documents` ([str]) | vLLM, Jina | required; `documents` is capped by the model's `max_batch` |
| `top_n` (int, default 0) | vLLM, Jina | the best `top_n` results; 0 (or ≥ the document count) returns all |
| `use_activation` (bool, default true) | vLLM | `false` returns the raw logit as `relevance_score` |
| `truncate_prompt_tokens` (int, −1 = max) | vLLM | truncate each pair to this many tokens (see *Over-length*) |
| `truncation_side` | vLLM | honored, as for classify |
| `max_tokens_per_query`, `max_tokens_per_doc` (int, 0 = off) | vLLM | truncate the query or each document to this many tokens before pairing |
| `return_documents` (bool, default true) | extension (Jina; Cohere v1) | `false` omits `document` from each result |
| `instruction`, `chat_template_kwargs` | vLLM | 400: they feed a chat template, and cross-encoders have none |
| `request_id`, `priority`, `cache_salt`, `mm_processor_kwargs`, `padding`, `normalize`, `task`, `user` | vLLM | exactly as for `/v1/classify` |

- `results` is sorted by `relevance_score`, highest first. `index` is the
  document's position in the request. vLLM sorts the same way.
- The response `id` is `score-<X-Request-Id or request_id or random>`: vLLM
  serves rerank from its scoring handler, whose prefix is `score`.
- `documents` holds strings only. vLLM's multimodal documents and Jina's
  `{"text": ...}` objects are 400s.
- `usage.prompt_tokens` counts every pair's real tokens.

## `POST /v2/rerank`

Cohere's v2 shape: `{model, query, documents: [str], top_n,
max_tokens_per_doc, priority}` in, `{id, results: [{index,
relevance_score}], meta: {api_version: {version: "2"}, billed_units:
{input_tokens}}}` out, sorted the same way. There's no `document` in a
result, and no `return_documents`, as in Cohere v2.

vLLM serves its v1 shape on `/v2/rerank` too. A client written for either
parses sidekick's Cohere-shaped response: vLLM's adds fields, it removes
none. `max_tokens_per_doc` defaults to 4096 as in Cohere, so long documents
are truncated there instead of rejected (see *Over-length*).

## `POST /v2/embed`

Cohere's v2 embed shape over the existing embedders, as vLLM serves it:

| field | sidekick |
|---|---|
| `model`, `texts` ([str]) | required; capped as `/v1/embeddings` is |
| `input_type` | `search_query` / `query` → the manifest's query prefix; `search_document` / `document` → the document prefix. `classification` and `clustering` are 400s: no embedder declares a prompt for them. Absent: the document prefix, as `/v1/embeddings` does. |
| `embedding_types` (default `["float"]`) | `float`, `base64` (little-endian f32), `binary` and `ubinary` (sign bits packed MSB-first, signed or not; dims must be a multiple of 8). `int8` and `uint8` are 400s, as in vLLM: they need calibration ranges. |
| `output_dimension` | a Matryoshka dimension, as `dimensions` on `/v1/embeddings` |
| `truncate` (`END` default, `START`, `NONE`) and `max_tokens` | `END` truncates to `max_tokens` or the model's maximum, which is what embedders already do. `NONE` is a 400 for an over-length input. `START` keeps the end. |
| `images`, `inputs` | 400: text only |
| `priority` | as for `/v1/classify` |

The response is `{id: "embd-<uuid>", embeddings: {<type>: [...]}, texts,
meta: {api_version: {version: "2"}, billed_units: {input_tokens}},
response_type: "embeddings_by_type"}`, as vLLM returns it.

## Over-length input

A pair is tokenized as the model's tokenizer pairs text: `[CLS] q [SEP] d
[SEP]` with `token_type_ids` 0 then 1 for BERT, and `<s> q </s></s> d </s>`
for XLM-R. That's also what `CrossEncoder` and vLLM feed the model.
- `max_tokens_per_query` / `max_tokens_per_doc` cut each text first, as
  vLLM does.
- Then, on `/v1/rerank`, a pair longer than the model's maximum is a 400
  unless `truncate_prompt_tokens` is set, as in vLLM. Truncation is
  tokenizers' `longest_first` over the pair, keeping the special tokens (as
  classify does).
- On `/v2/rerank`, documents are cut to `max_tokens_per_doc` (default
  4096), then pairs still too long are truncated `only_second`, keeping the
  query whole. That matches Cohere's "documents are truncated" contract.
  A query that alone fills the model is a 400.

## Manifest

A reranker is a classifier (`classifier.toml`) with a new task, Hugging
Face's pipeline name for rerankers:

```toml
id = "ms-marco-minilm-l6"
task = "text-ranking"
source = { repo = "cross-encoder/ms-marco-MiniLM-L6-v2", revision = "<sha>" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 128                 # documents per request
problem_type = "regression"     # the score's activation (see below)

[classify]
labels = ["score"]              # one output

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"   # new, optional: BERT pairs need it; XLM-R has none
output = "logits"
```

- `problem_type` picks the activation, with D28's rule. The converter
  derives it from the checkpoint exactly as vLLM's `get_act_fn` does:
  - the config's `problem_type`, if set;
  - otherwise `sentence_transformers.activation_fn`, or the older
    `sbert_ce_default_activation_function`: `Identity` → `regression`
    (raw logits), `Sigmoid` → `single_label`;
  - otherwise the single-output default, `single_label` → sigmoid.
  So `relevance_score` is on the same scale vLLM reports for that model.
- `token_type_ids` is a new optional `[classify.io]` feature, int32 like
  every input. It's valid on any task: nlptown's BERT could take it too.
  Validation requires it for `text-ranking` models whose tokenizer emits
  segment ids.
- `text-ranking` models serve `/v1/rerank`, `/rerank` and `/v2/rerank`
  only. `/v1/classify` on one, or rerank on another task, is a 400 naming
  the task, as D28 does for embedders and classifiers.
- /v1/models lists a reranker with `task: "text-ranking"` and `max_batch`.

## Core ML interface

| input | shape | notes |
|---|---|---|
| `input_ids`, `attention_mask` | `[1, S]` int32 | as classify |
| `token_type_ids` | `[1, S]` int32 | when the manifest names it; pads are 0 |
| output `logits` | `[1, 1]` | the raw score |

Load-time checks are the classifier's (every bucket, D28), plus
`token_type_ids`' shape. Batch execution is one pair per prediction, as
for classify, so `max_batch` bounds a request's ANE time.

## Validation

The parity suite grades rerankers in score space, reusing the classifier
grader with k = 1:
- **Reference:** `CrossEncoder` in fp32, the published activation, on
  (query, documents) groups: 51 pairs from the corpus's queries and
  documents, plus adversarial pairs (empty document, over-length document
  truncated, literal `[SEP]` in the query).
- **Gates:** finite output, ids and `token_type_ids` equal to the
  reference's, bucket and pad invariance (|Δscore| after the activation).
- **Graded:** worst |Δ relevance_score|, and rank flips within each query's
  documents where the reference's score gap is at least a margin (the
  classify flip rule, applied per group).
- **First model:** `cross-encoder/ms-marco-MiniLM-L6-v2` (BERT, 22.7M,
  Apache-2.0), on the same recipe as the small embedders (#8). An XLM-R
  reranker (bge-reranker-base, 278M) follows, after the XLM-R position-id
  offset is validated.

## Out of scope

- vLLM's `/score` and `/v1/score` (pair scores without ranking): `/v1/score`
  collides with SGLang's unrelated route.
- Late interaction (ColBERT through rerank, MaxSim server-side): it needs a
  multi-vector embedder path first.
- LLM-based rerankers (Qwen3-Reranker, chat-template scoring): the decoder
  family's, with `instruction` support then.

## Open questions

1. `/v2/rerank`'s default truncation: Cohere's (truncate documents), as
   proposed, or vLLM's (400 unless asked)? Proposed: Cohere's, since that
   route is Cohere's contract.
2. `return_documents` default `true` (vLLM's behavior, Jina's default) or
   `false` (Cohere v1's)? Proposed: `true`.
3. The response `id` prefix: vLLM's `score-`, or `rerank-`? Proposed:
   vLLM's.
4. `/v2/embed` `input_type` absent: vLLM's (no prefix) or `/v1/embeddings`'
   (document prefix)? Proposed: the document prefix, so the two embed
   routes agree for a given model.
