# Classification: `POST /v1/classify`

The contract, manifest, model interface and validation for sidekick's
classifiers. The decision record is D28 in docs/DECISIONS.md. The Rust
interface is `sidekick_core::classify`.

Principles:
- **Follow the standard exactly where one exists.** `/v1/classify` is vLLM's
  `/classify` and SGLang's `/v1/classify`, field for field, with the same
  conventions as `/v1/embeddings`.
- **Extend only where no OpenAI-family standard exists.** Zero-shot labels
  and laya's question format are extensions, and each is additive.
- **Fail loudly (D22).** Every request field vLLM defines is honored or
  rejected with a 400, never silently dropped.

## Request

```json
POST /v1/classify
{"model": "<id>", "input": "text" | ["text", ...]}
```

| field | source | sidekick |
|---|---|---|
| `model`, `input` (string or array) | vLLM, SGLang | required; the batch is capped by the model's `max_batch` |
| `use_activation` (bool, default true) | vLLM | `false` returns pre-activation values in `probs` |
| `truncate_prompt_tokens` (int) | vLLM | truncate to this many tokens |
| `truncation_side` (`right` \| `left`) | vLLM | honored for text-classification; `left` is a 400 on the laya format |
| `add_special_tokens` | vLLM | `true` is accepted; `false` is a 400 |
| `messages` (chat-form input) | vLLM | 400 |
| `request_id` (str) | vLLM | the response `id` is `classify-<request_id>`; an `X-Request-Id` header takes precedence, as in vLLM |
| `priority` (int) | vLLM | `0` or null; any other value is a 400 (vLLM rejects it without priority scheduling) |
| `padding` | vLLM | null or `"do_not_pad"`; `"max_length"` would add attended pads, so it and anything else are 400s |
| `cache_salt` (str) | vLLM | validated as vLLM does, then accepted; sidekick has no prefix cache, so it changes nothing |
| `mm_processor_kwargs` | vLLM | null or `{}`; anything else is a 400 |
| `normalize` | vLLM (removed) | a 400 whenever present, with vLLM's message |
| `task` | vLLM | `"score"` and `"encode"` are 400s with vLLM's messages; other values are ignored, as in vLLM |
| `user` | SGLang, OpenAI | accepted and ignored |
| `candidate_labels` ([str]) | extension (HF zero-shot's name) | required on zero-shot models; 400 on fixed-label models |
| `calibration` (`none` \| `model`) | extension | only on models that declare temperatures (every value, `none` included, is a 400 elsewhere); default `none`; `model` applies the manifest's temperature, and with `use_activation: false` needs none |
| `question_type` (`choice` \| `score` \| `noul`) | extension, laya format | required on laya |
| `instructions` (str) | extension, laya format | optional; the manifest's per-type default when absent |

Any other top-level field is ignored, as D22 already does. Extension fields
unsupported by the model's task are a 400.

A request is validated in full before any input runs: every field, the
manifest's calibration table and laya's noul labels are checked before the
model loads, and every input is prepared before the first one runs. A
batch input that fails is a 400 prefixed `input N: `, and nothing runs.

Over-length input:
- **text-classification:** without `truncate_prompt_tokens`, an input longer
  than the largest bucket is a 400, as in vLLM.
- **laya format:** the state is truncated by design, keeping its start, and
  `truncate_prompt_tokens` is a 400.

Other 400s:
- an empty batch, or more inputs than `max_batch`;
- fewer than 2 candidate labels, or more than `max_labels`;
- duplicate labels, including labels identical at the token level after
  laya's option shrinking;
- a laya `noul` question whose labels aren't `false` / `true` in that order,
  optionally with descriptions (`"false: …"`, `"true: …"`);
- `calibration: model` where the model declares no temperature for that
  question type and label count;
- malformed JSON, on every route, in the API's usual error shape (D22
  amendment).

Where sidekick deliberately differs from vLLM:
- `model` is required. vLLM and SGLang serve one model per process, so
  they can default it; sidekick serves several.
- `truncate_prompt_tokens` keeps the special tokens (`[CLS]`, `[SEP]`) and
  truncates the text between them, on either side. vLLM slices the token
  list including them, so its `left` drops `[CLS]` and its `right` drops
  `[SEP]`, a sequence the model never saw in training.
- Before tokenizing, an input over `max_seq_len × 16` bytes is a 400, or,
  with truncation, is cut to that many bytes first. This bounds the
  tokenizer's work. vLLM has a similar characters-per-token check.
- `calibration: model` is a 400 when the manifest has no temperature for
  the request's question type and label count. laya's own API falls back
  to a per-type temperature there; sidekick declares only what laya fitted,
  and fails loudly otherwise.

## Response

```json
{"id": "classify-<uuid>", "object": "list", "created": 1745383065, "model": "<id>",
 "data": [{"index": 0, "label": "<argmax label>", "probs": [..], "num_classes": 5}],
 "usage": {"prompt_tokens": 10, "total_tokens": 10, "completion_tokens": 0}}
```

- `probs` follows the label order: the manifest's `labels` for fixed-label
  models, the request's `candidate_labels` for zero-shot models. `label` is
  the argmax.
- The activation follows transformers' text-classification pipeline:
  regression → none; multi-label or a single output → sigmoid; otherwise
  softmax, with temperature 1 unless `calibration: model`.
- `usage.prompt_tokens` counts the real tokens of every input.

**Provenance headers**, on every inference route (`/v1/classify`,
`/v1/embeddings`, `/v1/chat/completions`). They follow OpenAI's
`openai-model` / `openai-version` style:
- `sidekick-version`.
- `sidekick-model`: `<id>@<revision>` when the manifest has a `source`,
  otherwise `<id>`. For chat it's the Foundation Models variant id, read
  in the background and cached; until it's known, the chat model's id.
- `sidekick-compute-units`: the configuration the serving instance was
  loaded with: `cpu_and_ne` for Core ML models, `cpu` for static models.
  Chat omits it. It reports configuration, not the executing device, which
  Core ML doesn't expose.

## Manifest: `classifier.toml`

Classifiers use their own filename, so daemons and `libsidekick.dylib`
builds that predate them never parse one. The registry scans both
filenames. From this release on, it skips and warns on a bad manifest
instead of failing the whole scan.

```toml
id = "laya-en"
task = "zero-shot-classification"        # or "text-classification"
source = { repo = "convaiinnovations/laya", revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 32
problem_type = "single_label"             # single_label | multi_label | regression

[classify]
format = "laya"                           # zero-shot formats: "laya"
max_labels = 32                           # must equal the artifact's marker_pos width (checked at load)
labels = []                               # text-classification: output order (id2label)

[classify.laya]
head_max_len = 192
default_instructions = { choice = "Which option fits the text best?", score = "Which level fits the text best?", noul = "Does the statement hold for the text?" }

[classify.calibration]                    # opt-in; "<question_type>:<k bucket>", laya's temp_bucket keys
"choice:2" = 1.906
"choice:3-5" = 1.760
"choice:6-10" = 1.000
"score:3-5" = 1.251
"noul:2" = 1.983
# "choice:11+" deliberately absent: laya's single fit (T = 0.10) is too thin

[classify.io]                             # Core ML feature names
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
qtype = "qtype"
output = "logits"
```

A text-classification manifest sets `task = "text-classification"`,
`[classify] labels = [...]` and `problem_type`. Its `[classify.io]` has
`input_ids`, `attention_mask` and `output` only.

Registry validation rejects:
- `labels` on a zero-shot model, or `format` on a fixed-label model;
- a missing `[classify.io]` feature for the format.

Loading a classifier checks every bucket's artifact before any runs, from
its model description (a CPU-only load that never predicts, about 2 s cold
for a large bucket, read in parallel): `input_ids` and `attention_mask`
are `[1, bucket]`, `marker_pos` is `[1, max_labels]`, `qtype` is `[1]`,
the output has one slot per label where it declares a shape, and D27's
shape guard passes. A bad bucket fails the load, not a later request.

Task-aware listings:
- `/v1/models` gains `task`, `labels` or `max_labels`, the accepted
  extension fields, and the calibration table.
- `/health` lists classifiers separately, with `classifiers.supported`,
  and the manifests the registry skipped, with paths relative to the
  models directory.
- Builds without Core ML hide classifiers from `/v1/models`, as chat is
  hidden without Foundation Models; `/v1/classify` is a 503 there.
- `sk_pool_models` and `sk_model_info` in the C ABI list embedders only.
  `sk_pool_open` skips bad or duplicate manifests too, and the new
  `sk_pool_skipped` lists them.
- `/v1/embeddings` on a classifier, and `/v1/classify` on an embedder, are a
  400 naming the model's task.

## Core ML interface

Every input is int32, which is what the runtime's `predict_int32` feeds.

| format | inputs | output |
|---|---|---|
| text-classification | `input_ids [1,S]`, `attention_mask [1,S]` | `logits [1,N]` |
| laya | `input_ids [1,S]`, `attention_mask [1,S]`, `marker_pos [1,KMAX]` (−1 pads unused slots), `qtype [1]` (rank 1) | `logits [1,KMAX]`, padded slots at −1e4 |

laya's graph builds the one-hot marker selection and the question-type
embedding from these inputs itself. It pins the residual range rewrite at
K = 2. The calibration rule would allow K = 1, but laya's largest measured
linear output (about 27,500) would then sit within 2% of the rule's
0.85 × 2^15 target (D25). The load-time check reads KMAX
from `marker_pos`'s shape in the model description.

## The laya format

laya is a decision model: a ModernBERT-large encoder plus a head that scores
a `[MASK]` marker before each option. The Rust port of its `build_sequence`
(rl_common.py at the pinned revision) must copy these exactly:
- The sequence is
  `[CLS] "<type> question: <instructions>" [SEP] [MASK] " "+opt0 … [SEP] <state> [SEP]`,
  with every fragment tokenized with `add_special_tokens=False`.
- Literal `[MASK]` in text is replaced by a space before tokenizing.
- Each option's text is cut to 48 tokens before its mask token is prepended.
- If the option budget (`head_max_len` minus the options' length) falls
  under 16, every option is cut to `max(4, (head_max_len − 16) / k)`.
- The question text is cut to `max(8, budget)` tokens.
- The state is truncated to fit, keeping its start.
- The final sequence is `ids[:max_len]`.
- Options are rendered as laya renders them:
  - choice: `"key"` or `"key: description"`;
  - score: `"level i: description"`;
  - noul: `"false: …"` then `"true: …"`, with laya's default texts when a
    label has no description.
- laya's act (escalate) head isn't served.

## Fixtures and references (frozen formats)

- **Token-id fixture**, `fixtures/classify/<model id>.tokens.json`,
  generated by `tools/classifier_reference.py` with the model's own Python
  (laya's `rl_common.py` for laya). `crates/sidekick-embed/tests/classify_tokens.rs`
  asserts that the Rust input builder reproduces it. The test needs the
  model's tokenizer installed; it skips otherwise, and fails instead under
  `SIDEKICK_REQUIRE_CLASSIFY_FIXTURES=1`.
  Schema: `fixtures/classify/tokens.schema.json`.
- **Reference file**, `<refs>/<model id>/reference.json` plus
  `reference.safetensors`. It's written by the classifier reference
  generator and not committed, like D26's. It carries the per-case ids,
  markers, qtype, labels, fp32 logits and gold labels, plus D26's staleness
  keys. Schema: `fixtures/classify/reference.schema.json`.

## Validation

The parity suite gains classifiers. Per path:
- **Hard gates:** finite output; ids equal to the reference's; bucket and
  pad invariance.
- **Graded:** raw Δp and Δlogit against the fp32 reference. Argmax flips
  count only where the reference's top-2 logit margin is ≥ 0.05; near-ties
  are reported, not graded. Calibrated Δp is reported separately.
- **Reported, never graded:** accuracy against gold labels. It measures the
  corpus translation as much as the model.

Corpora:
- **Fixed-label models:** the existing 51-input parity corpus.
- **laya:** fastino/fast-decisions at revision `1a33070` (Apache-2.0),
  translated mechanically: ordinal heads → `score`, yes/no heads → `noul`,
  everything else → `choice`. Multi-label heads are skipped. Instructions
  come from a fixed template per head name. The translation table is
  committed.
- **laya adversarial additions:**
  - k = 32 with long options;
  - custom noul texts;
  - default instructions;
  - a massive-activation case;
  - literal `[SEP]` and `[CLS]` in the text.

First models:
- `laya-en` (zero-shot, laya format);
- `nlptown/bert-base-multilingual-uncased-sentiment` (text-classification;
  5 labels; BERT). If its checkpoint has no `tokenizer.json`, the converter
  generates one, and the Python reference tokenizes with that same file.
