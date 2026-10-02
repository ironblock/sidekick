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
| `instructions` (str) | extension, laya and gliner2 formats | laya: the question text, with the manifest's per-type default when absent; required by a model whose manifest has none (Julia-1). gliner2: the task prompt (below), with the manifest's default when absent |
| `multi_label` (bool, default false) | extension (HF zero-shot's name), gliner2 format | `true` (gliner2 only; a 400 elsewhere) scores each label independently: `probs` are per-label sigmoids instead of a softmax, and one candidate label is allowed. `false`, the default, is accepted on every model, as Hugging Face clients send it |

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
- **gliner2 format:** the text is truncated at the word level, keeping its
  start, until the sequence fits the largest bucket, and the `.` is then
  appended as for any text. The schema is never truncated: a schema that
  doesn't fit on its own is a 400.
  `truncate_prompt_tokens` and `truncation_side: left` are 400s.

Other 400s:
- an empty batch, or more inputs than `max_batch`;
- fewer than 2 candidate labels (1 with gliner2's `multi_label: true`),
  or more than `max_labels`;
- duplicate labels, including labels identical at the token level after
  laya's option shrinking;
- a laya `noul` question whose labels aren't `false` / `true` in that order,
  optionally with descriptions (`"false: …"`, `"true: …"`);
- with Julia-1's option rendering: a noul question that describes only
  one of `false` and `true`, a label that renders as an empty option, or
  two labels that render alike (`"b"` and `"x: b"`);
- no `instructions` for a model whose manifest has no default;
- `multi_label: true` on any format but gliner2;
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
  softmax, with temperature 1 unless `calibration: model`. On a
  gliner2-format model, `multi_label: true` makes the request multi-label;
  on any other model it is a 400.
- For gliner2 these are the probabilities its own API reports at its
  default activation and temperature 1: a softmax over the task's label
  logits for a single-label task, and each label's sigmoid for a
  multi-label one. (Its per-label training loss is binary cross-entropy,
  but its single-label confidence is the softmax.)
- A multi-label request still returns one `label`, the argmax, because
  vLLM's response shape has one label per input. Clients choose the
  labels that apply by thresholding `probs`. gliner2's own API returns
  every label at or above its `cls_threshold` (0.4 to 0.5 in its
  examples), falling back to the argmax so the answer is never empty.
- `usage.prompt_tokens` counts the real tokens of every input.

**Provenance headers**, on every inference route (`/v1/classify`,
`/v1/embeddings`, `/v1/chat/completions`). They follow OpenAI's
`openai-model` / `openai-version` style:
- `sidekick-version`.
- `sidekick-model`: `<id>@<revision>` when the manifest has a `source`,
  otherwise `<id>`. For chat it's the Foundation Models variant id, read
  in the background and cached; until it's known, the chat model's id.
- `sidekick-compute-units`: the configuration the serving instance was
  loaded with: a Core ML model's `compute_units` (`cpu_and_ne` unless its
  manifest says otherwise; see "Compute units"), `cpu` for static models.
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
compute_units = "cpu_and_ne"              # cpu_and_ne (default) | cpu_and_gpu | cpu_only | all

[classify]
format = "laya"                           # zero-shot formats: "laya", "gliner2"
max_labels = 32                           # laya: must equal the artifact's marker_pos width (checked at load)
labels = []                               # text-classification: output order (id2label)

[classify.laya]
head_max_len = 192
option_rendering = "laya"                 # laya (default) | julia: see "Option rendering"
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

A gliner2 manifest is zero-shot with `format = "gliner2"`. Its
`max_labels` is a policy cap on labels per request, since the artifact
has no marker width. `problem_type` is `single_label`; a request makes
itself multi-label with `multi_label`. Its `[classify.io]` has
`input_ids`, `attention_mask` and `output`:

```toml
id = "gliner2.5-decide"
task = "zero-shot-classification"
source = { repo = "fastino/GLiNER2.5-Decide", revision = "5a7adf72a23b4d311abae6ce050d7f0012bb3416" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 32
problem_type = "single_label"

[classify]
format = "gliner2"
max_labels = 32

[classify.gliner2]
default_instructions = "label"            # the task prompt when a request sends none

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"                         # one logit per token, [1, bucket]
```

Registry validation rejects:
- `labels` on a zero-shot model, or `format` on a fixed-label model;
- a missing `[classify.io]` feature for the format.

Loading a classifier checks every bucket's artifact before any runs, from
its model description (a CPU-only load that never predicts, about 2 s cold
for a large bucket, read in parallel): `input_ids` and `attention_mask`
are `[1, bucket]`, `marker_pos` is `[1, max_labels]`, `qtype` is `[1]`,
the output has one slot per label where it declares a shape (one per
token, `[1, bucket]`, for gliner2), and D27's shape guard passes. A bad
bucket fails the load, not a later request.

Task-aware listings:
- `/v1/models` gains `task`, `labels` or `max_labels`, the accepted
  extension fields (`extensions`), the ones every request must send
  (`required`: `candidate_labels` on zero-shot models, `question_type` on
  laya-format ones, and `instructions` where the manifest has no default),
  and the calibration table.
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
| gliner2 | `input_ids [1,S]`, `attention_mask [1,S]` | `logits [1,S]`, one per token |

laya's graph builds the one-hot marker selection and the question-type
embedding from these inputs itself. It pins the residual range rewrite at
K = 2. The calibration rule would allow K = 1, but laya's largest measured
linear output (about 27,500) would then sit within 2% of the rule's
0.85 × 2^15 target (D25). The load-time check reads KMAX
from `marker_pos`'s shape in the model description.

gliner2's graph applies the checkpoint's classifier MLP to every token, and
the runtime reads the logits at the `[L]` positions it placed while
building the sequence. That needs no index inputs and no gather, and costs
under 1% of the encoder's compute. The graph holds the encoder and that
MLP only: GLiNER2's span, count and count-prediction modules serve
extraction, not classification, and stay out of the artifact. The
encoder is DeBERTa-v3, whose relative-position attention is rewritten to
convert (`tools/probe_deberta.py`).

## Compute units

A Core ML model loads with `.cpuAndNeuralEngine` unless its manifest asks
otherwise (D14): the ANE keeps background work off the GPU. Some models run
badly there. GLiNER2.5-Decide grades A on the GPU at about 34 ms per input,
but C on the ANE, where a prediction takes about 0.11, 0.34 and 1.28 s at
128, 256 and 512 tokens, because its DeBERTa relative-position rewrite is
slow on the ANE (docs/MODELS.md). Such a model
names its compute units with an optional top-level key, in
`classifier.toml` and in an embedder's `manifest.toml` alike:

| `compute_units` | Core ML | |
|---|---|---|
| `cpu_and_ne` | `.cpuAndNeuralEngine` | the default |
| `cpu_and_gpu` | `.cpuAndGPU` | for a model the ANE runs badly |
| `cpu_only` | `.cpuOnly` | |
| `all` | `.all` | Core ML chooses; for measuring |

- The daemon and `libsidekick.dylib` load the model with it, and every
  bucket shares it. Tests and the parity suite can still load any path
  explicitly (`load_with`).
- `sidekick-compute-units` and the model's `/v1/models` entry
  (`compute_units`) report it. A static embedder reports `cpu`, and setting
  the key on one is a validation error.
- An unknown value fails validation, so the registry skips the manifest
  with the reason (in `/health`'s `skipped_models`) and loads the rest.
- D27's shape guard runs for every choice: it reads the model description,
  which doesn't depend on compute units, so a multi-shape artifact is
  refused on macOS 27 whatever the manifest asks for.
- The parity suite grades every path regardless. The configured one is the
  path a model is served on, so it is the grade that matters for that
  model.

### The ANE weight cap

Core ML runs an ML program on the ANE only while its weights stay under
about 1 GiB. Past that it runs the whole program elsewhere, with no error
and nothing in the log: measured on an M1 Max under macOS 27.0, a program
with 0.964 GiB of weights ran on the ANE and one with 1.022 GiB didn't, and
coremltools documents a 1 GB Neural Engine limit. A model served on the
ANE (`compute_units` `cpu_and_ne` or `all`) whose compiled weights exceed
`MAX_ANE_PROGRAM_WEIGHT_BYTES` (1 GiB) would look ANE-served and not be,
so the registry skips it, as D28 skips any bad manifest:
- Each bucket is its own program, and each is checked. The weights are
  the files under the artifact's `weights/` directory (an `.mlmodelc`, or
  `Data/com.apple.CoreML/weights/` in an `.mlpackage`), measured as the
  converters measure them (`tools/sidekick_convert/plan.py`), from file
  metadata, without reading them.
- The reason, in `/health`'s `skipped_models` and in `sk_pool_skipped`,
  names the bucket, its weights and the fixes: serve it on the GPU
  (`compute_units = "cpu_and_gpu"`), convert a quantized or chunked
  variant, or load it anyway.
- Loading it anyway, for experimentation: `ane_weight_limit = "ignore"`
  in the manifest (top level, either manifest file), or `sidekickd
  --ignore-ane-weight-cap` (`ignore_ane_weight_cap = true` in the config)
  for every model.
- Models served on `cpu_and_gpu` or `cpu_only` aren't checked. Neither is
  the parity suite's registry: it measures every compute path itself, and
  its ANE plan check reports a model Core ML doesn't place on the ANE.

### The CPU sequence cap

Core ML's fp16 CPU matmul sums in an order that depends on the contraction
length past 1,024 keys. Each result is accurate within its own rounding,
but a model then gives slightly different answers for the same input in
different buckets: up to 0.021 in probability between Lumma-fev's 1,024-
and 2,048-token buckets, 0.018 for agent-jev. Running the matmul in slices
restores invariance only at about 13 times the error, so no conversion
fixes it (D33 records the measurement and its reproduction).
Up to 1,024 tokens the CPU is bucket-invariant, and the GPU and the ANE
are unaffected.

So a model served on the CPU runs no longer than
`MAX_CPU_INVARIANT_SEQ` (1,024 tokens):
- For `compute_units = "cpu_only"` and a `max_seq_len` over 1,024, the
  registry drops the buckets above it, and the largest bucket kept becomes
  the model's effective `max_seq_len`. Longer inputs then follow the
  model's usual over-length rules: a 400, or truncation where the format
  truncates its own text. A model with no bucket of at most 1,024 tokens
  is skipped, with the fixes as the reason.
- The capped manifest is validated again (laya's head budget, for
  example, must still fit), and a model that no longer validates is
  skipped with the reason.
- The model's `/v1/models` entry gains `seq_cap` (`limit`,
  `manifest_max_seq_len` and `reason`), and `/health` lists every capped
  model under `seq_caps`.
- `all` isn't capped: Core ML chooses the device per operation, may not
  use the CPU at all, and `all` exists for measuring what it does.
- To serve every bucket anyway, accepting differences between buckets past
  1,024 tokens: `cpu_seq_limit = "ignore"` (top level, either manifest
  file, coreml backend only), or `sidekickd --ignore-cpu-seq-cap`
  (`ignore_cpu_seq_cap = true` in the config) for every model.
- The parity suite serves every bucket. On its CPU path it reports a
  bucket comparison into a bucket past 1,024 tokens as this documented
  limit, with the variation it measured, instead of gating it; up to 1,024
  tokens the CPU's bucket gate stays exact, and the GPU and ANE paths are
  gated as before. Its worker output records each bucket comparison, so a
  saved run can be graded again under a changed rule.

Compatibility: sidekick 0.4 and earlier ignore unknown keys in both
manifest files. A 0.4 daemon given a manifest with `compute_units` loads it
with `.cpuAndNeuralEngine` and reports `cpu_and_ne`; it doesn't fail. (A
gliner2-format manifest is skipped by 0.4 anyway, since it predates that
format.)

The same holds for `[classify.laya] option_rendering`, with a sharper edge:
0.4 ignores the key and would render a Julia-1 manifest's options as laya
does, which Julia-1 wasn't trained on. The committed Julia-1 manifest is
safe there because it has no `default_instructions`, which 0.4 requires,
so 0.4 skips it with that reason. Don't add `default_instructions` to a
manifest with `option_rendering = "julia"` while 0.4 daemons may read the
models directory.

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
- Options are rendered as the manifest's `option_rendering` says (below).
- laya's act (escalate) head isn't served.

### Option rendering

Models trained on laya's sequence differ in how a label becomes option
text. `[classify.laya] option_rendering` picks one; the sequence around
the options is the same. A label's description is the text after its
first `": "`; an empty description counts as none. The response's `label`
is always the request's label as sent.

| question | `laya` (default; laya's `render_options`) | `julia` (Julia-1's typed API) |
|---|---|---|
| choice | the label as given: `"key"` or `"key: description"` | the description, else the key |
| score | `"level i: <label>"` | the label as given |
| noul | `"false: …"` then `"true: …"`, laya's default text for a missing description | `"false"`, `"true"` when neither is described; the two descriptions when both are; one alone is a 400 |

Julia-1's rendering also needs:
- `max_labels` at most 20, Julia-1's most options per question;
- no empty option and no two labels rendering alike (both 400s);
- `instructions` on every request when the manifest has no
  `default_instructions`, which Julia-1's API doesn't have. laya's
  rendering requires them in the manifest.

An over-long state is truncated, keeping its start, under both: sidekick
doesn't implement Julia-1's strict mode, which rejects one.

## The gliner2 format

[GLiNER2](https://github.com/fastino-ai/GLiNER2) (Apache-2.0) models such as
GLiNER2.5-Decide classify from a schema placed before the text. Each label
gets an `[L]` marker token, and the classifier scores the encoder's output
at each marker. The Rust port of the gliner2 package's input builder
(`processor.py` in gliner2 2.0.0) must copy these exactly:
- The sequence is the task's schema, `[SEP_TEXT]`, then the text:
  `( [P] <prompt> ( [L] label0 [L] label1 … ) ) [SEP_TEXT] <text words>`.
  It has no `[CLS]` or `[SEP]`.
- Each item (each parenthesis, marker, the prompt, each label, each text
  word) is tokenized on its own with the model's tokenizer, and the pieces
  are concatenated.
- `<prompt>` is the request's `instructions`, or the manifest's
  `default_instructions`. gliner2's own API builds it as `task` or
  `task: instruction` (for example `intent`, or `intent: what the
  customer wants`); the string is tokenized as one item either way.
  The default, `label`, matched the dataset's own task names (`intent`,
  `sentiment`, …) on every argmax in an fp32 check of 45 single-task
  fast-decisions rows from 9 domains.
- A label sent as `"key: description"` puts `key` in its `[L]` slot and
  appends ` [DESCRIPTION] key: description` to the prompt, in label order.
- The text is split into words by gliner2's whitespace splitter regex,
  and each word is lowercased. Labels, the prompt and descriptions keep
  their case. The splitter's character classes are spelled out to match
  Python 3.12's `re`, measured equal on all of Unicode 15.0. The runtime's
  Unicode tables come from its Rust crates (`regex`, and the standard
  library's lowercasing), which may be newer: a character added after
  Unicode 15.0 can split or lowercase differently from gliner2 on Python
  3.12.
- An over-length text loses words from its end until the sequence fits the
  largest bucket. Then a `.` is appended, as a word of its own, unless the
  kept text ends in `.`, `!` or `?` (an empty text becomes `.`). gliner2
  appends first and never truncates by default; truncating first is
  sidekick's choice, so a truncated text still ends the way every training
  input did. The reference runs gliner2 on the kept prefix of the text, cut
  at the end of its last kept word, which reproduces this.
- The `[P]` and `[L]` positions come from the layout, the first piece of
  each marker's slot, never from searching for their token ids. A marker
  string inside a label or the prompt tokenizes to its marker id but isn't
  scored, as in gliner2. In the text, the word splitter breaks it up
  (`[L]` becomes `[`, `l`, `]`), so it never becomes a marker.

Scope, per request:
- **One task.** gliner2 can score several tasks in one pass, with their
  schemas joined by `[SEP_STRUCT]`, but vLLM's response carries one label
  per input. Tasks attend to each other, so a task scored alone isn't
  identical to the same task scored jointly. Measured in fp32 on 85
  single-label tasks from fast-decisions rows that have several tasks:
  alone vs joint agree on 83 argmaxes, the logits move by at most 2.4, and
  gold accuracy is 54 vs 56 of 85. A multi-task extension would save passes
  but isn't needed for accuracy.
- **No few-shot examples** (`[EXAMPLE]` … `[OUTPUT]`) in this version.
- **No long-text chunking.** gliner2's `classify_long` repeats the schema
  over word windows and merges the logits; sidekick truncates instead.

## Fixtures and references (frozen formats)

- **Token-id fixture**, `fixtures/classify/<model id>.tokens.json`,
  generated by `tools/classifier_reference.py` with the model's own Python:
  laya's `rl_common.py` for laya-en, the laya package's `common.py` for
  laya-typed-decisions, Julia-1's `julia/data.py` for Julia-1, and the
  gliner2 package's processor for gliner2. gliner2's fixture also has two
  cases the generator builds at the truncation boundary (a text ending in
  `!` or `?` and a space, one token over only because of the appended
  `.`), which aren't in the corpus. `crates/sidekick-embed/tests/classify_tokens.rs`
  asserts that the Rust input builder reproduces it. The test needs the
  model's tokenizer installed; it skips otherwise, and fails instead under
  `SIDEKICK_REQUIRE_CLASSIFY_FIXTURES=1`.
  Schema: `fixtures/classify/tokens.schema.json`.
- **Reference file**, `<refs>/<model id>/reference.json` plus
  `reference.safetensors`. It's written by the classifier reference
  generator and not committed, like D26's. It carries the per-case ids,
  markers, qtype, labels, fp32 logits and gold labels, plus D26's staleness
  keys. It also records the option rendering of a laya-format model and the
  gliner2 release of a gliner2 one: a reference whose rendering differs
  from the manifest's, or whose gliner2 release isn't the one the runtime
  ports, is stale. Schema: `fixtures/classify/reference.schema.json`.

## Validation

The parity suite gains classifiers. Per path:
- **Hard gates:** finite output; ids equal to the reference's; bucket and
  pad invariance.
- **Graded:** raw Δp and Δlogit against the fp32 reference. Argmax flips
  count only where the reference's top-2 logit margin is ≥ 0.05; near-ties
  are reported, not graded. Calibrated Δp is reported separately.
- **Against the ideal-fp16 ceiling**, when the reference carries the
  `fp16` oracle: a ratio grade from p99 |Δp| over the ceiling's p99, and
  the better of it and the absolute grade counts. On the GPU and ANE the
  bucket gate passes within the ceiling's worst case (D28 amendment).
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
- **Julia-1** (`fixtures/classify/julia-1.corpus.toml`): the laya
  translation of fast-decisions, without heads over its 20-label limit,
  plus adversarial cases in Julia-1's terms (20 long options, an option cut
  to 48 tokens, descriptions containing `": "`, empty descriptions and bare
  colons, described noul labels, a state truncated at 1,024 tokens, and
  literal `<bos>`, `<eos>` and `<mask>`). Inputs are built by Julia-1's own
  `julia/data.py` with options rendered as its typed API renders them, and
  the oracles run its `JuliaDecisionModel`.
- **gliner2:** fastino/fast-decisions at revision `1a33070`, each task
  sent as its own request, with multi-label tasks as `multi_label: true`.
  It is fastino's own benchmark, so it grades parity only; gold accuracy
  isn't reported for gliner2. Adversarial additions:
  - 32 labels, and labels with descriptions;
  - marker strings (`[L]`, `[P]`, `[SEP_TEXT]`) inside the text, a label
    and a description;
  - non-ASCII text and emoji (the tokenizer is a Unigram model without
    byte fallback, so a character outside its vocabulary becomes `[UNK]`);
  - text long enough to be truncated at the largest bucket, including one
    whose last kept word is followed by terminal punctuation that
    truncation removes, and one whose kept part already ends in `.`.

First models:
- `laya-en` (zero-shot, laya format);
- `nlptown/bert-base-multilingual-uncased-sentiment` (text-classification;
  5 labels; BERT). If its checkpoint has no `tokenizer.json`, the converter
  generates one, and the Python reference tokenizes with that same file.
- `SupersonicLabs/Julia-1` (zero-shot, laya format with Julia-1's option
  rendering; mmBERT-small).
- `fastino/GLiNER2.5-Decide` (zero-shot, gliner2 format; DeBERTa-v3-large),
  served on the GPU (`compute_units = "cpu_and_gpu"`): it passes the gates
  there (grade A), while the ANE grades C and is 10–40× slower
  (docs/MODELS.md).
