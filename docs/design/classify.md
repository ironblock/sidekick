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
| `truncation_side` (`right` \| `left`) | vLLM | honored for text-classification and the nli format; `left` is a 400 on the laya, gliner2, fev and agentjev formats |
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
| `question_type` (`choice` \| `score` \| `noul`) | extension, laya, fev and agentjev formats | required on all three |
| `instructions` (str) | extension, laya, gliner2, fev and agentjev formats | laya: the question text, with the manifest's per-type default when absent; required by a model whose manifest has none (Julia-1). gliner2: the task prompt (below), with the manifest's default when absent. fev: the question text; absent, the question is empty, as fev's own API sends it. agentjev: the question text, required and nonempty, as AgentJev's API requires |
| `multi_label` (bool, default false) | extension (HF zero-shot's name), gliner2 and nli formats | `true` (gliner2 and nli only; a 400 elsewhere) scores each label independently: `probs` are per-label sigmoids instead of a softmax, and one candidate label is allowed. `false`, the default, is accepted on every model, as Hugging Face clients send it |
| `hypothesis_template` (str) | extension (HF zero-shot's name), nli format | the hypothesis each label is put into, with exactly one `{}` for the label and no other brace; the manifest's when absent (Hugging Face's default, `This example is {}.`, unless the manifest says otherwise) |
| `nli_probs` (bool, default false) | extension, nli format (sidekickd 0.15.0 or later) | `true` adds `nli_probs` to each result: per candidate label, the pair's softmax over the model's own classes (`entailment`, `neutral`, `contradiction`, or a two-way model's `entailment`, `not_entailment`), keyed by class name; see [The nli format](#the-nli-format). `probs` and `label` are unchanged. A 400 on other formats, and on an nli model whose manifest names no `classes`; `false` is accepted on every model |

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
- **fev format:** the state is truncated to `state_max_len` tokens,
  keeping its start. The question and options are never cut: a row that
  still exceeds the window is a 400 naming their token count.
  `truncate_prompt_tokens` and `truncation_side: left` are 400s.
- **agentjev format:** nothing is truncated, as AgentJev refuses an
  over-long path: a tree longer than the largest bucket is a 400 naming
  its token count, and so are `truncate_prompt_tokens` and
  `truncation_side: left`.
- **nli format:** as Hugging Face's pipeline does (`only_first`), only
  the text is truncated, on `truncation_side`, until each label's pair
  fits `max_seq_len`, or `truncate_prompt_tokens` when it's lower. The
  hypothesis is never cut: one that leaves no room for the text is a 400.

Other 400s:
- an empty batch, or more inputs than `max_batch`;
- fewer than 2 candidate labels (1 with gliner2's `multi_label: true`,
  and always 1 on the nli format), or more than `max_labels`;
- duplicate labels, including labels identical at the token level after
  laya's option shrinking;
- a laya `noul` question whose labels aren't `false` / `true` in that order,
  optionally with descriptions (`"false: …"`, `"true: …"`);
- with Julia-1's option rendering: a noul question that describes only
  one of `false` and `true`, a label that renders as an empty option, or
  two labels that render alike (`"b"` and `"x: b"`);
- with the fev format: two labels that render alike once control tokens
  are rewritten (`"<|x|>"` and `"<¦x¦>"`), or a label that renders as an
  empty option;
- with the agentjev format: no `instructions`; an empty input or
  instructions (whitespace by Python's `str.isspace`); two labels that give
  the same candidate text (`"x: b"` and `"b"`), or one that gives an empty
  one;
- no `instructions` for a model whose manifest has no default;
- `multi_label: true` on any format but gliner2 and nli;
- a `hypothesis_template` on any format but nli, or one without exactly
  one `{}` or with any other brace;
- `nli_probs: true` on any format but nli, or on an nli model whose
  manifest doesn't name its `classes`;
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
- An nli request with `nli_probs: true` adds one field to each result,
  `nli_probs`: per candidate label, in `probs` order, an object of the
  model's classes and their probabilities (see [The nli
  format](#the-nli-format)). Without it, the result has vLLM's four fields
  and nothing else.

**Provenance headers**, on every inference route (`/v1/classify`,
`/v1/embeddings`, `/v1/chat/completions`). They follow OpenAI's
`openai-model` / `openai-version` style:
- `sidekick-version`.
- `sidekick-model`: `<id>@<revision>` when the manifest has a `source`,
  otherwise `<id>`. For chat it's the Foundation Models variant id, read
  in the background and cached; until it's known, the chat model's id.
- `sidekick-compute-units`: the configuration the serving instance was
  loaded with: a Core ML model's `compute_units` (`cpu_and_ne` unless its
  manifest says otherwise; see [Compute units](../BACKENDS.md#compute-units)), `cpu` for static and ONNX
  models (D40).
  Chat omits it. It reports configuration, not the executing device, which
  Core ML doesn't expose.
- `sidekick-buckets`: the sequence-length bucket each input ran in, in
  input order and comma-separated (`256,256,512`; one input gives `256`).
  ONNX models have no buckets and don't send it.
  Each bucket is its own compiled program (D15), so this names the
  program that answered each input, the one whose placement `/v1/models`
  reports per bucket (see [Where Core ML places operations](../BACKENDS.md#where-core-ml-places-operations)). On
  `/v1/classify`, `/v1/embeddings`, `/v1/rerank`, `/rerank`, `/v2/rerank`
  and `/v2/embed`. A rerank pair is one input, in request order rather
  than the sorted results' order. Chat and static embedders omit it.

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
format = "laya"                           # zero-shot formats: "laya", "gliner2", "fev", "agentjev", "nli"
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

An nli manifest is zero-shot with `format = "nli"` and an `[classify.nli]`
section (see [The nli format](#the-nli-format)). Its `max_labels` is a
policy cap, since each label is its own prediction. `[classify.io]` names
`token_type_ids` exactly when the model reads segment ids:

```toml
id = "nli-deberta-v3-base"
task = "zero-shot-classification"
source = { repo = "cross-encoder/nli-deberta-v3-base" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 8
problem_type = "single_label"

[classify]
format = "nli"
max_labels = 16

[classify.nli]
entailment = 1                            # output indices, as Hugging Face's pipeline resolves them
contradiction = 0
n_classes = 3
classes = ["contradiction", "entailment", "neutral"]   # config.json's id2label, lowercased; names nli_probs
hypothesis_template = "This example is {}."   # optional; this is the default

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"                         # [1, n_classes] per pair
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
  the calibration table, and an nli model's default `hypothesis_template`
  and its classes in output order (`nli_classes`, which key `nli_probs`).
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
| fev | `input_ids [1,S]`, `attention_mask [1,S]`, `marker_pos [1,KMAX]` (each option's `<option_end>` position, −1 pads unused slots), `decide_pos [1]` (rank 1) | `logits [1,KMAX]`, padded slots at −1e4 |
| agentjev | `input_ids [1,S]`, `attention_mask [1,S]`, `seg [1,S]` (−1 for pads), `position_ids [1,S]` (0 for pads), `marker_pos [1,KMAX]` (each candidate's last token, −1 pads unused slots) | `logits [1,KMAX]`, padded slots at −1e4 |
| nli | `input_ids [1,S]`, `attention_mask [1,S]`, and `token_type_ids [1,S]` when the manifest names it; one (text, hypothesis) pair per prediction | `logits [1,n_classes]` |

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

Compute units, the ANE weight cap, the CPU and ANE sequence caps, and
where Core ML places operations apply to every Core ML model, embedders
included, and are in [BACKENDS.md](../BACKENDS.md#core-ml).

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

`[classify.laya] option_rendering` needs care with older daemons:
0.4 ignores the key and would render a Julia-1 manifest's options as laya
does, which Julia-1 wasn't trained on. The committed Julia-1 manifest is
safe there because it has no `default_instructions`, which 0.4 requires,
so 0.4 skips it with that reason. Don't add `default_instructions` to a
manifest with `option_rendering = "julia"` while 0.4 daemons may read the
models directory.

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

## The fev format

Lumma-fev decision models (FrontiersMind/Lumma-fev, Apache-2.0) are causal
decoders that read a request's state and one question in a single row, and
compare the hidden state at a final decide token with the hidden state at
the end of each option. The Rust port of their input builder
(`modeling_fev.py` at the pinned revision: `option_text`, `pack`,
`text_ids`, `encode`) must copy these exactly:
- The row is `<state> state <question> instructions (<option> option
  <option_end>)… <decide>`, where the five delimiters are the tokens the
  manifest's `[classify.fev] delimiters` names by role (`<|reserved_0|>` to
  `<|reserved_4|>` in Lumma-fev-0.1b). There is no BOS, EOS or other
  special token.
- Each fragment (the state, the instructions, each option's text) is
  tokenized on its own with the model's tokenizer and no special tokens,
  and the ids are concatenated with the delimiters between them.
- In every fragment, `<|name|>` (name: ASCII letters, digits and `_`) is
  rewritten to `<¦name¦>` before tokenizing, so no request text, label
  included, can forge a delimiter or another control token. The response's
  `label` is the request's label as sent.
- The instructions are the request's `instructions`, or empty when there
  are none: the model has no default question text, and an empty question
  is what its own API sends.
- Options are rendered from the request's labels as fev's `option_text`
  renders criteria. A label's description is the text after its first
  `": "`; an empty description counts as none:
  - choice: the label as given, or the key alone when its description is
    empty;
  - score: the label as given;
  - noul: labels `false` then `true`, each optionally described, rendered
    `"no"` and `"yes"`, or `"no: <description>"` and `"yes: <description>"`.
    Each side is rendered on its own, so describing only one is allowed.
- The state keeps its first `state_max_len` tokens, its delimiter
  included: 1,408 for Lumma-fev-0.1b, which is `min(8192, window − 640)`
  for the 2,048-token window its config gives. (Its card says 1,024; its
  code, which is what the model was served with, says 1,408.)
- Served `cpu_only`, the window is capped at 1,024 tokens (see "The CPU
  sequence cap"), and the state keeps `min(state_max_len, 1,024 − 640)`:
  384 for Lumma-fev-0.1b. That is sidekick's choice, not the model's
  limit. It applies fev's formula to the capped window, so the question
  and options keep their 640 tokens. The checkpoint itself never sees a
  1,024-token window: its own code keeps a 1,408-token state and refuses
  rows over 2,048. The references and the token fixture use 1,408.
- A row longer than the window (`max_seq_len`) after that is a 400: fev
  never shrinks options or the question, and neither does sidekick.
- The logits are read at each option's `<option_end>` and at the final
  `<decide>`. Their positions come from the layout, never from searching
  for delimiter ids (which the rewrite keeps out of the text anyway).
- Each fragment is capped at 16 bytes per token of its budget before
  tokenizing (the state's own, the window for the rest), which bounds the
  tokenizer's work and never changes a row that fits.

The probabilities are a softmax over the option logits, which is what fev's
`decide()` reports at its checkpoint temperature. The converter folds that
temperature into the graph (it is 1.0 for Lumma-fev-0.1b), so the manifest
has neither a temperature nor a calibration table. The graph selects the
option and decide states with one-hot matmuls against a position constant,
as laya's does, and applies the pointer head, `logits_i = k(h[end_i]) ·
q(h[decide]) / sqrt(head_dim)`. A causal model never attends forward, so
pads never reach a real token; the pad-invariance gate checks it.

Scope, per request:
- **One question.** fev scores several questions per request, each on its
  own row with the shared state, so one question never sees another. A
  sidekick request is one question, which is the same computation.
- **String states only.** fev also renders JSON objects and arrays as
  indented text; a sidekick input is a string. An array of strings is a
  batch, and each input is its own row with the request's question, as on
  the laya format.
- **`question_type` is required:** the model never sees the type, but noul
  labels render differently. `required` in `/v1/models` lists
  `candidate_labels` and `question_type`.
- **Labels:** 2 to `max_labels` per request, as on every zero-shot format
  (fev allows 1 to 255).

```toml
[classify]
format = "fev"
max_labels = 32
labels = []

[classify.fev]
state_max_len = 1408          # the state's tokens at most, its delimiter included
delimiters = { state = "<|reserved_0|>", question = "<|reserved_1|>", option = "<|reserved_2|>", option_end = "<|reserved_3|>", decide = "<|reserved_4|>" }

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"     # each option's <option_end> position
decide_pos = "decide_pos"
output = "logits"
```

Validation for `format = "fev"`: the `[classify.fev]` section, with five
different delimiters, each one token of the tokenizer (checked at load);
`1 ≤ state_max_len < max_seq_len`; `marker_pos` and `decide_pos` in
`[classify.io]`, and no `qtype` or `token_type_ids`; `single_label`; no
calibration table. `decide_pos` is an explicit input although it is always
the row's last real token, so the token-id fixture pins it like the option
positions.

## The agentjev format

AgentJev (aimeigaoshou/agent-jev; code at malevrigns/agent-jev,
Apache-2.0) is a Qwen3-0.6B decision model with a candidate head. Its
service scores each candidate on its own causal path,

```text
[STATE] state \n[QUESTION] question \n[CANDIDATE] candidate
```

reads the hidden state at the path's last token, and runs a
permutation-equivariant set transformer and a scorer over the candidates'
vectors. Every path of a question shares the state and question, so
sidekick lays the question out once as a tree, and the graph scores every
candidate in one pass. The Rust port of AgentJev's input contract
(`jev_service/contract.py` at the pinned revision: `semantic`, `prepare`'s
candidate rules, `encode_paths`) must copy these exactly:
- Each fragment is tokenized on its own with the model's tokenizer and no
  special tokens, and the ids are concatenated, with no BOS or EOS:
  - the state, as `"[STATE] " + state`, or as sent when it already starts
    with `[STATE]`;
  - the question, as `"\n[QUESTION] " + instructions`;
  - each candidate, as `"\n[CANDIDATE] " + candidate`.
  The tags are plain text, not added tokens. Added tokens in the text are
  matched as the tokenizer matches them, so `<|endoftext|>` in a state is
  that token, as in AgentJev's own service.
- The tokenizer is the checkpoint's `tokenizer.json`, which is
  Qwen3-0.6B-Base's, as AgentJev's service loads. Qwen3-0.6B's (the
  instruct model's) has extra added tokens and isn't interchangeable.
- The state and the instructions must be nonempty once Python's
  whitespace (`str.isspace`, which includes U+001C to U+001F) is
  stripped. Text is otherwise kept as sent.
- Candidates come from the request's labels in label order. A label's
  description is the text after its first `": "`; an empty description
  counts as none:
  - choice: the description, else the label, as AgentJev reads an
    option's value;
  - score: the label as given, as AgentJev reads a level's description;
  - noul: labels `false` then `true`, each optionally described. The
    candidates are the descriptions, else `FALSE` and `TRUE`, AgentJev's
    defaults for undescribed boolean criteria. AgentJev orders boolean
    candidates true first. The head is permutation-equivariant and the
    tree's branches never see each other, so label order changes only the
    order of `probs`.
  Candidates must be nonempty and distinct, as AgentJev requires; either
  failure is a 400.
- The tree is the prefix (the state's and question's ids), then each
  candidate's ids in label order. Per token:
  - `seg`: 0 for the prefix, `c` for candidate `c` (1-based);
  - `position_ids`: `0…P−1` over the prefix, then `P, P+1, …` within each
    candidate, restarting at `P`.
  Each candidate's last token is its `marker_pos` entry. Positions come
  from the layout, never from searching for token ids.
- Nothing is truncated (see "Over-length input"). AgentJev refuses a path
  over 2,048 tokens. sidekick's limit is stricter: the whole tree,
  the prefix plus every candidate, must fit the largest bucket.

The graph's attention lets token `i` attend token `j` when
`attention_mask_j = 1`, `j ≤ i`, `seg_j ≥ 0`, and `seg_j` is 0 or
`seg_i`. A pad (segment −1) attends only itself, so no row is fully
masked and random pad ids change nothing. RoPE reads `position_ids`. Each
candidate therefore sees exactly its own path at its own positions. Against
AgentJev's per-path encoding, through the checkpoint's own head, the tree
reproduced every probability within 2.1e-7 in fp32 (a boolean, a 4-option
choice and a 5-level score). The head reads the hidden state at each
`marker_pos` with one-hot matmuls. It runs the set transformer over the
KMAX slots, with unused slots masked out of its attention and at −1e4 in
the output.

The probabilities are a softmax over the candidates' logits. AgentJev's
service divides the logits by a per-type temperature (`temperatures.json`:
1.0718 for boolean and score, 1.0353 for choice) when it is started with
one, as its card does. sidekick's calibration is opt-in, as on laya:
`calibration: model` applies the manifest's temperatures. AgentJev's are
per type, whatever the candidate count, so the manifest declares each
type's temperature under every one of laya's label-count keys the type can
reach (`noul:2`; `choice:2` … `choice:11+`; `score:2` … `score:6-10`, as
AgentJev takes 2 to 10 levels). A missing bucket would make
`calibration: model` a 400 at that label count. The default is
temperature 1.

Scope, per request:
- **One question.** AgentJev's service takes several questions per state.
  Each is scored on its own paths, so a sidekick request with one question
  is the same computation.
- **String states only.** AgentJev also takes a JSON object or array as a
  state and renders it as compact JSON with sorted keys
  (`separators=(',', ':')`, `ensure_ascii=False`). A sidekick input is a
  string, used as sent: a client reproduces AgentJev's rendering by
  sending that JSON text.
- **`question_type` and `instructions` are required:** the model has no
  default question, and noul labels render differently. `required` in
  `/v1/models` lists `candidate_labels`, `question_type` and
  `instructions`.
- **Labels:** 2 to `max_labels` per request (AgentJev allows 2 to 255
  choices and 2 to 10 levels).
- **The response** is vLLM's: `probs` in label order and the argmax
  `label`. AgentJev's own answers derive from the same distribution: a
  boolean's probability is `p(true)`, and a score's expected level is
  `Σ i·pᵢ`.

```toml
[classify]
format = "agentjev"
max_labels = 32
labels = []

[classify.calibration]                    # AgentJev's per-type temperatures, at every
"noul:2" = 1.0718                         # label-count bucket the type reaches
"choice:2" = 1.0353
"choice:3-5" = 1.0353
"choice:6-10" = 1.0353
"choice:11+" = 1.0353
"score:2" = 1.0718
"score:3-5" = 1.0718
"score:6-10" = 1.0718

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
seg = "seg"
position_ids = "position_ids"
marker_pos = "cand_end"       # each candidate's last token
output = "logits"
```

Validation for `format = "agentjev"`: `marker_pos`, `seg` and
`position_ids` in `[classify.io]`, and no `qtype` or `token_type_ids`;
`seg` and `position_ids` on no other format; `single_label`. Loading
checks `seg` and `position_ids` are `[1, bucket]`, like `input_ids`.

## The nli format

An NLI cross-encoder (MNLI-style: `cross-encoder/nli-deberta-v3-*`,
`roberta-large-mnli`, the zeroshot-v2.0 family) classifies the way
Hugging Face's `zero-shot-classification` pipeline uses it (D41). For an
input and its k candidate labels:
- **One pair per label.** The pair is (text, the hypothesis template with
  the label in place of `{}`), laid out by tokenizer.json's pair
  post-processor exactly as the pipeline's fast tokenizer does: `[CLS]
  text [SEP] hypothesis [SEP]` for BERT and DeBERTa, `<s> text </s></s>
  hypothesis </s>` for RoBERTa. Segment ids are fed when `[classify.io]`
  names `token_type_ids`. The manifest decides, not the tokenizer:
  DeBERTa-v3's tokenizer gives pairs segment ids that its model, with no
  token-type embeddings, never reads.
- **Truncation:** the pipeline's `only_first`. Only the text is cut, on
  `truncation_side`, so every pair fits the model; the hypothesis never is.
- **Each pair is its own prediction,** in the smallest bucket that fits
  it, as a reranker's pairs are. A request costs k predictions per input,
  which `max_labels` and `max_batch` bound. `usage.prompt_tokens` counts
  every pair's tokens, and `sidekick-buckets` names each input's largest
  bucket.
- **Scoring**, from each pair's logits, with the manifest's `entailment`
  and `contradiction` indices:
  - several labels, `multi_label: false`: a softmax across the labels of
    their entailment logits;
  - `multi_label: true`, or a single label (the pipeline's rule): per
    label, the softmax over [contradiction, entailment], taking
    entailment. The classifier returns entailment − contradiction, whose
    sigmoid is that probability, so the server's per-label sigmoid gives
    it, and `use_activation: false` returns those differences.

The indices are the pipeline's, which the converter resolves from
config.json: `entailment` is the first label in `label2id` that starts
with "entail" (ignoring case), and `contradiction` is index 0, or the last
index when entailment is 0. For every MNLI and zeroshot model those are
the contradiction (or not_entailment) output.

`probs` follows `candidate_labels`, and `label` is the argmax, as on every
format; the pipeline instead sorts labels by score.

### Class probabilities (`nli_probs`)

Both scores above reduce a pair's classes to one number, and the
per-label one drops neutral: a claim the text contradicts and one it
never mentions can score alike. A request that sends `nli_probs: true`
also gets each pair's full distribution (D47):

```json
{"model": "nli-deberta-v3-base", "input": "The config sets port 8080 and enables TLS.",
 "candidate_labels": ["the port is 8080", "the port is 9090", "the server logs to a file"],
 "hypothesis_template": "{}.", "nli_probs": true}
```

Measured on nli-deberta-v3-base, rounded:

```json
{"index": 0, "label": "the port is 8080", "probs": [0.9995, 0.0003, 0.0002], "num_classes": 3,
 "nli_probs": [{"contradiction": 0.0001, "entailment": 0.9968, "neutral": 0.0031},
               {"contradiction": 0.9999, "entailment": 0.0000, "neutral": 0.0001},
               {"contradiction": 0.0010, "entailment": 0.0001, "neutral": 0.9990}]}
```

nli-minilm2-l6-h768 and roberta-large-mnli favor the same class for each
label. Asked about only the last two labels, the default `probs` is
`[0.55, 0.45]`: it can't tell the contradicted label from the
unmentioned one.
- One object per candidate label, in `probs` order; one result per input,
  as always. Each object is the softmax over that pair's logits, keyed by
  the manifest's `classes` in output order, so its values sum to 1.
- The same for any number of labels, with or without `multi_label`,
  which change `probs` only.
- Three-way (MNLI) models report `entailment`, `neutral` and
  `contradiction`. The zeroshot-v2.0 models are two-way and report
  `entailment` and `not_entailment`: they can't tell contradicted from
  unmentioned. `/v1/models` lists each nli model's `nli_classes`; of the
  example manifests, the `cross-encoder/nli-*` models and
  `roberta-large-mnli` are three-way.
- `use_activation: false` returns the class logits instead, as `probs`
  then holds logits.
- It costs nothing extra: the pairs run once, and `probs` comes from the
  same logits, so it is unchanged by the field.

`classes` is config.json's `id2label`, lowercased, which the converter
checks against the checkpoint. The registry also checks that
`entailment` points at an entailment class and `contradiction` at
`contradiction` or `not_entailment`.

**Versions.** `classes` needs sidekickd 0.15.0 or later. `[classify.nli]`
refuses unknown keys, so an older daemon skips a manifest that has it:
the model is missing from `/v1/models`, with only a warning in the log.
The key is optional, so manifests installed before it still load on
0.15.0. Such a model doesn't list `nli_probs` among its `extensions` and
answers it with a 400.

**Enabling `nli_probs` on an installed model.** Add the model's one line
under `[classify.nli]` in its installed `classifier.toml`; the artifacts
don't change and nothing needs reconverting. Don't copy the example
manifest over the installed one: the installed file carries a
`[placement]` table the converter recorded on your machine, which the
example lacks.

| model | line to add under `[classify.nli]` |
|---|---|
| nli-deberta-v3-xsmall, nli-deberta-v3-small, nli-deberta-v3-base, nli-minilm2-l6-h768, nli-minilm2-l6-h768-ane | `classes = ["contradiction", "entailment", "neutral"]` |
| roberta-large-mnli | `classes = ["contradiction", "neutral", "entailment"]` |
| deberta-v3-base-zeroshot-v2.0-c, deberta-v3-large-zeroshot-v2.0-c, roberta-large-zeroshot-v2.0-c, bge-m3-zeroshot-v2.0-c, modernbert-large-zeroshot-v2.0 | `classes = ["entailment", "not_entailment"]` |

## Fixtures and references (frozen formats)

- **Token-id fixture**, `fixtures/classify/<model id>.tokens.json`,
  generated by `tools/classifier_reference.py` with the model's own Python:
  laya's `rl_common.py` for laya-en, the laya package's `common.py` for
  laya-typed-decisions, Julia-1's `julia/data.py` for Julia-1, the
  gliner2 package's processor for gliner2, Lumma-fev's
  `modeling_fev.py` (`encode`) for fev, with each case's `decide`
  position, and AgentJev's `contract.py` (`encode_paths`) for agentjev,
  laid out as a tree with each case's `seg` and `position_ids`. Every
  path `encode_paths` returns must equal the case's prefix followed by
  that candidate's suffix. gliner2's fixture also has two
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
  ports, is stale. An nli reference records the manifest's `[classify.nli]`
  (stale if it changes) and, per case, each label's pair (`pair_ids`,
  `pair_type_ids`), the request's template and `truncation_side`. Its
  per-label values are the entailment logits, or entailment −
  contradiction for a multi-label or single-label case, so the usual
  grading applies. Schema: `fixtures/classify/reference.schema.json`.

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
- **fev** (`fixtures/classify/lumma-fev-0.1b.corpus.toml`): the laya
  translation of fast-decisions with fev's rendering, plus adversarial
  cases: delimiter and other control strings in the state, the
  instructions and an option; the rewritten form itself; an empty state; no
  instructions; half-described noul labels; empty descriptions; 32 long
  and 32 short options; states truncated at 1,408 tokens; and rows that end
  exactly at the 2,048-token window and just under it. Lumma-fev's training
  data isn't published, so it measures conversion parity only.
- **nli** (`fixtures/classify/nli.corpus.toml`, shared by every nli
  model, so they are graded alike): MNLI validation pairs (matched and
  mismatched, a fixed stride of rows, read from the dataset when the
  reference is generated) as one-label cases whose hypothesis is the
  label itself (template `{}`), plus written zero-shot cases: topics,
  sentiment and intents with 2 to 16 labels; agent transcripts (failures,
  stack traces, test runs, discovered limitations); one-label and
  multi-label versions of the same request; custom and non-English
  templates; and premises over 512 tokens truncated on either side.
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
- `FrontiersMind/Lumma-fev-0.1b` (zero-shot, fev format; a causal decoder
  with a pointer head).
- `fastino/GLiNER2.5-Decide` (zero-shot, gliner2 format; DeBERTa-v3-large),
  served on the GPU (`compute_units = "cpu_and_gpu"`): it passes the gates
  there (grade A), while the ANE grades C and is 10–40× slower
  (docs/MODELS.md).
