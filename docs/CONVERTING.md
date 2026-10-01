# Converting models for sidekick

sidekick runs embedders and classifiers as Core ML models on the Apple
Neural Engine (ANE). Getting a Hugging Face checkpoint there faithfully takes
more than `coremltools.convert`: the ANE computes in fp16 with its own
arithmetic limits, Core ML has op-level bugs, and a model that converts can
still be wrong in ways only a careful gate catches. This document catalogs
what the conversion library knows, and how to use and extend it.

The library is `tools/sidekick_convert`. The converters in `tools/` are thin
scripts on top of it. docs/MODELS.md lists the validated models and their
measured accuracy; docs/DECISIONS.md records why each rule exists (entries
cited as Dnn).

## Contents

- [Layers](#layers)
- [The conversion run and its gates](#the-conversion-run-and-its-gates)
- [Techniques](#techniques)
- [Calibration and evaluation](#calibration-and-evaluation)
- [Ideal fp16](#ideal-fp16)
- [Tokenizers](#tokenizers)
- [Manifests](#manifests)
- [Converters](#converters)
- [Adding a family](#adding-a-family)
- [Refactoring a converter: the acceptance method](#refactoring-a-converter-the-acceptance-method)
- [Gotchas](#gotchas)
- [Tests](#tests)

## Layers

A converter composes three orthogonal layers, so that architecture and task
don't multiply into one script per combination:

- **Backbone** (`sidekick_convert.backbones.<family>`): the architecture
  made convertible. It loads the checkpoint in fp32 and applies what that
  architecture needs: the attention implementation and finite masks,
  traceable helpers, position offsets, and the precision and range rewrites.
  It knows how to call the model inside the traced wrapper (`call`), which
  constant buffers the wrapper needs per bucket (`buffers`), and how to run
  the unpadded fp32 reference (`reference`).
- **Head** (`sidekick_convert.heads`): what the artifact returns.
  `pool.Pool` pools an embedding in-graph (CLS, mean or last token, optionally
  L2-normalized). `sequence.SequenceClassification` is the checkpoint's own
  classification head, run through its task class, so each family's head
  semantics are exactly the checkpoint's; a reranker is this head with one
  label and a token_type_ids input.
- **Interface** (`core.Port`, `sidekick_convert.manifest`): the int32
  static-shape inputs, the output name, and the committed manifest the
  artifact must match.

`wrapper.compose(backbone, head, ports)` builds one static wrapper per
bucket; `recipes.embedder()` and `recipes.classifier()` turn the three into
a `core.Job`; `core.run(job, install_dir)` converts and installs it.

A backbone's contract, for a new family: `model` (the module the wrapper
holds, under the attribute `attr`, which names the converted weights),
`buffers(seq)`, `call(wrapper, inputs)`, `reference(ids, token_type_ids)`,
`example(seq, ports)`, `hidden_size`, `vocab_size`, `config`, and
`forbid_ops`. Rewrites (activation swaps, range and precision rewrites) are
functions of a backbone, applied by the recipe after the fp32 references
are computed from the unmodified checkpoint, so the fp32 gate proves each
rewrite exact.

## The conversion run and its gates

For every bucket, `core.run()`:

1. builds the static-shape wrapper, `Job.make_wrapper(seq)`, so a recipe may
   differ per bucket (if one does, bucket invariance across the boundary
   still has to hold);
2. runs the **torch gate**: the padded, static-shape, rewritten wrapper must
   reproduce the checkpoint's own unpadded fp32 forward (embedders: cosine
   ≥ 0.99999; classifiers: max |Δlogit| ≤ 1e-3);
3. traces it and converts with coremltools to an ML program (macOS 15 opset,
   int32 inputs with static shapes);
4. refuses the graph if it contains a **forbidden op**: Core ML's fused
   `scaled_dot_product_attention` always (D25), plus whatever native op a
   rewrite replaced (`gelu`, `silu`);
5. compiles it with `xcrun coremlcompiler`, and gates the compiled artifact,
   the exact bytes that get installed:
   - **compute plan** (D24): every linear, matmul and conv on the ANE, at
     least 80% of assigned ops on the ANE, and no fused attention reading a
     mask built outside its ANE procedure (D25). An unreadable plan, or one
     with every op unassigned, fails, after one re-read from an APFS clone
     (see the stale-cache gotcha);
   - **accuracy per compute path**: CPU_AND_NE (what sidekick serves) and
     CPU_ONLY (an independent execution of the same graph). Embedders:
     cosine ≥ 0.999 against the fp32 reference. Classifiers: argmax
     agreement wherever the fp32 top-2 margin is ≥ 0.05, and max |Δp| after
     the manifest's activation;
   - **pad invariance**: the same input with random pad ids must give the
     same output. It catches a dropped attention mask (D25) and a mixer that
     reads pad states (D19) in seconds;
   - **finite output**;
6. installs `model_{seq}.mlmodelc`, then the tokenizer and manifest.

Every metric is NaN-safe (`metrics.py`): Python's `min(worst, nan)` returns
`worst`, which once hid a NaN-producing CPU path behind "parity 1.000000"
(D25). A negative-control flag turns gate failures into reports. Latency is
measured only with `--time`: it moves with machine load, and accuracy
doesn't, so time only on a quiet machine.

## Techniques

Each module in `sidekick_convert.techniques` states the limit it works
around and where it was measured.

| module | what it handles | when to use it | measured in |
|---|---|---|---|
| `masks` | finite additive masks (`MASK_ADD = -30000`; `finfo.min` is -inf in fp16 and NaNs softmax), bands, causal masks, `self_attending()` so no query row is fully masked | every model; `self_attending` wherever a row can be fully masked (sliding windows, pairwise masks) with a softmax variant or op that NaNs on it | D15, D25 |
| `attention` | `explicit()` matmul → softmax → matmul, never the fused op; `matmul_softmax()` opt-in: the denominator comes from the value matmul, because the ANE's `reduce_sum` is the one op not bit-identical across compiled buckets | `explicit` always (or pass `scale=` to `F.scaled_dot_product_attention`, which makes coremltools emit explicit ops); `matmul_softmax` only as a measured per-model change | D25 |
| `activations` | `TanhGelu`, `TwiceGelu` (erf), `TanhSilu`: 2·f(x) from tanh or erf, since the native `gelu` (≤ 6e-3 on [-1, 1]) and `silu` (≤ 1.5e-2) are coarse on the ANE; the factor 2 is folded downstream | follow the model's own `hidden_act`: TanhGelu for `gelu_pytorch_tanh`, TwiceGelu for erf `gelu`, TanhSilu for `silu`. Opt-in and measured per model: an erf GELU didn't help gte-modernbert and made its CPU path worse, and Core ML's CPU erf is coarser than its gelu | D17, D19, D20, D25 |
| `precision` | the ANE linear's small-input floor (relative error ~3e-4 / rms(input)): power-of-two input scales folded upstream, `Descale` after | a linear whose calibrated input rms is small (≲ 0.02–0.03) | D17, D19, D20 |
| `saturation` | the ANE linear saturates above 2^15: `check()` the rule (calibrated outputs ≤ 0.85 × 2^15), `choose_k()` / `headroom_at()` for a residual stream run at 1/K | check on every model; residual K where a massive activation crosses it (ModernBERT) | D25 amendment |
| `pooling` | CLS, masked mean (by the attention_mask input, never by token id; summed at 1/32 for fp16 range), last token without a gather, L2 squared at 1/32; outputs end in a literal `(1, dims)` reshape | every pooled head | D15, D17, D20 |
| `reduce` | `blocked_max()`: max over 128-wide slices, exact everywhere (macOS 27's CPU `reduce_max` over ≥ 256 elements returns max(x, 0)) | any explicit max over a long axis | laya |
| `traceable` | `rotate_half`, `repeat_kv` without shape arithmetic; `install()` patches a transformers module | RoPE and grouped-query attention | D17 |
| `onehot` | selections from int32 inputs by comparison with a position constant, instead of data-dependent gathers | inputs that index positions (laya's markers and question type) | D28 |

The range rewrite for activations past fp16's own maximum (EmbeddingGemma's
residual stream at ~1.5e5, D17) and pad zeroing for convolutional mixers
(D19) live in their converters for now and move into the library with those
families.

## Calibration and evaluation

Two kinds of inputs, kept apart by type:

- **`core.Calibration`** decides rewrites: the residual K, input rescales,
  anything that changes the artifact. It refuses texts from the graded parity
  corpus (`fixtures/parity/corpus.toml`), because calibrating on what is
  graded would flatter the grades. A converter's own texts, or a committed
  calibration set, are fine.
- **`core.Evaluation`** holds the gate cases and their fp32 references. It
  judges and decides nothing. Checks that decide nothing, such as the BERT
  2^15 range check, may run on evaluation texts.

The guard is not hypothetical. bge-small's parity sentences are in the graded
corpus, which was seeded from them.

## Ideal fp16

`sidekick_convert.fp16sim` simulates an ideal fp16 engine: one that loses
nothing to fp16 beyond storing its tensors in it. Its error against the fp32
reference is the model's ceiling, the best any fp16 path can do. Classifier
and reranker references carry it as the `fp16` oracle beside `torch`
(`tools/classifier_reference.py`; `fixtures/classify/reference.schema.json`),
and the parity suite grades each path as a ratio to it. The definition lives
in one function so that every model's ceiling means the same thing.

**The definition: fp32 arithmetic inside each operation, every
input-dependent tensor stored in fp16.**

1. **Every operation that depends on the input rounds its float output to
   fp16**, at operation granularity. Rounding module outputs instead (a hook
   on each `nn.Linear` and `LayerNorm`) misses everything a module computes
   inside its forward: attention scores and probabilities, residual adds, an
   MLP's activation times its gate. In ModernBERT and DeBERTa a whole
   attention block is one module. A fused kernel (layer norm, GELU, softmax,
   a linear with its bias) is one operation, rounded once. Fused attention
   runs as its math decomposition, so its scores and probabilities are
   stored like any other tensor.
2. **Everything that does not depend on the input is a constant**: computed
   exactly, as a converter folds it in fp32, and stored in fp16 once. That
   covers weights, buffers, and tables built from them. RoPE is the case that
   matters. Rounding positions x inv_freq op by op puts the angle at
   position 500 off by up to 0.25 radians, a loss no converted program has.
   On gte-modernbert's 442-token gate text it costs 6.5x in 1 - cos
   (5.8e-6 against 8.9e-7).
3. **Python scalars** (an epsilon, a `1/sqrt(d)` written as a float) are
   applied exactly.

The rounding happens at PyTorch's dispatcher, below Python, so it reaches an
operation however the model calls it (`F.linear`, `@`, `torch.bmm`,
`Tensor.softmax`, `+`). The split between constants and input-dependent
tensors is found by tracking which tensors derive from the inputs. Neither
needs a per-architecture list.

**It runs the model the `torch` oracle runs**: the published checkpoint in
fp32, one unpadded input at a time. It does not run the converted wrapper.
The ceiling belongs to the model, so a converter's rewrites (TwiceGelu's tanh
form, the residual 1/K) are graded against it, not folded into it. To
separate rewrite error from fp16 storage, a converter can run the same
function on its wrapper; that is a diagnosis, not a ceiling.

    from sidekick_convert import fp16sim
    logits = fp16sim.run(model, input_ids=ids, attention_mask=mask).logits

`fp16sim.run` treats every tensor argument as an input. `fp16sim.ideal_fp16()`
is the same as a context manager, for a call that mixes inputs and constants.
`ops=` rounds only some operations' outputs, which is useful for diagnosis
and never for grading.

Caveats:

- **A real engine can beat the ceiling slightly.** Core ML fuses some runs
  of operations into one (a decomposed layer norm, a GELU pattern) and rounds
  once where the simulation rounds each step. A ratio a little under 1 is
  not an error.
- **A model whose activations pass fp16's 65,504 has no ceiling as
  published.** EmbeddingGemma's residual stream is one example. The
  reference generator leaves out a non-finite fp16 oracle, and says why,
  rather than write it.
- **The single worst case is sensitive to the exact rounding points.** On
  laya at bucket 128, two implementations of this definition, one with
  hand-placed rounding points and one generic, agreed on mean and p99 Δp to
  within 15% but differed 1.7x on the maximum. That sensitivity is why the
  definition lives in one function. Grades that divide by the ceiling's
  maximum should report the p99 ratio too. On a corpus of under 100 cases
  the nearest-rank p99 is the maximum, so there a p99 ratio carries the
  same sensitivity.

## Tokenizers

sidekick tokenizes with the Rust `tokenizers` crate from `tokenizer.json`,
and truncates and pads by itself. So:

- the installed `tokenizer.json` enables neither padding nor truncation.
  `tokenizer.prepare(mode="clean")` copies the checkpoint's file byte for
  byte unless it enables either, and otherwise loads it, calls
  `no_padding()` and `no_truncation()`, and saves it. A checkpoint with only
  `vocab.txt` gets the fast tokenizer transformers builds from it, saved the
  same way. The result is deterministic: all-MiniLM-L6-v2's cleaned file is
  byte-identical to e5-small-v2's and bge-small's upstream ones.
  `mode="verbatim"` copies byte for byte, and existing models use it so their
  pinned `tokenizer_sha256` never changes. `expected_sha256` fails the run
  on a mismatch;
- gate cases and parity references encode with that file through Python's
  `tokenizers` (`tokenizer.encode`, `tokenizer.encode_pair`), never with
  AutoTokenizer (see the gotchas).

## Manifests

Manifests carry reviewed decisions (buckets, `max_batch`, prefixes,
`problem_type`), and parity references check them, so converters copy the
committed file and never generate one: `examples/manifests/<id>/manifest.toml`
for embedders, `examples/classifiers/<id>/classifier.toml` for classifiers.
The install directory's name is the model id.

Before converting, `manifest.check_embedder()` / `check_classifier()` stop the
run on any mismatch with the checkpoint, where the field exists:
- `max_seq_len` equals sentence-transformers' `max_seq_length` (at most, for
  existing models that cap it at 512), and the largest bucket equals it;
- `max_seq_len` plus the position offset fits the position embeddings;
- `dims` equals the head's output size, and the head's pooling equals the
  checkpoint's sentence-transformers pooling;
- classifiers: labels in `id2label` order, and `problem_type` by
  transformers' activation rule (D28). A reranker (`task = "text-ranking"`)
  has one label, `"score"`, and the `problem_type` vLLM derives from the
  checkpoint;
- `token_type_ids` is named only by text-ranking manifests, exactly when
  pairs carry segment ids (older daemons would feed a text-classification
  model no segment ids);
- `source.revision` equals the snapshot's revision.

## Converters

| converter | backbone | head | notes |
|---|---|---|---|
| `convert_bge_small.py` | BERT, fused attention | CLS pool | fused attention kept for byte identity with the graded artifact; see its docstring |
| `convert_bert_embedder.py` | BERT, explicit | CLS or mean pool (from the checkpoint) | MiniLM, e5 and other BERT-family sentence-transformers |
| `convert_bert_classifier.py` | BERT, explicit | sequence classification | classifiers and rerankers; `--twice-gelu` opt-in |
| `convert_gte_modernbert.py`, `convert_laya.py` | standalone | | move to a ModernBERT backbone next |
| `convert_embeddinggemma.py`, `convert_lfm25_embedding.py`, `convert_qwen3_embedding.py` | standalone | | move to Gemma3, LFM2 and Qwen3 backbones after that |

Every converter takes `<hf-model-dir> <install-dir> [buckets...]`, plus
`--time` and its own flags. The library needs arm64-native Python with
torch, transformers 4.x (the BERT backbone proves at load that its finite
mask is on the forward path), tokenizers, coremltools and numpy, and Xcode
for `xcrun coremlcompiler`.

## Adding a family

1. **Triage** (docs/MODELS.md, "Quick triage"): read `config.json` and the
   modeling code; run `tools/probe_activations.py` for fp16 range.
2. **Write the backbone**, `backbones/<family>.py`, to the contract above:
   - load in fp32 with explicit (eager) attention, and patch every mask
     builder to a finite additive mask of the same geometry;
   - replace any traced shape arithmetic (`techniques.traceable`, literal
     sizes);
   - `buffers()` for positions and anything else constant per bucket;
   - rewrites as functions of the backbone, applied after the references.
3. **Pick a head**, or add one to `heads/` if the output is new (a
   per-token head, a marker head).
4. **Commit the manifest** under `examples/`, reviewed like code.
5. **Write the converter**: gate texts of its own (short, long, multilingual,
   code, delimiters; at least one in each bucket for classifiers), a
   calibration set if any rewrite needs one (never the graded corpus), and a
   `recipes.*` call.
6. **Test** new techniques in `tools/sidekick_convert/tests` (torch only),
   and prove every rewrite exact with the fp32 gate.
7. **Grade** with the parity suite (D26) against sentence-transformers or the
   checkpoint's own fp32 forward, and record the model in docs/MODELS.md; a
   new constraint gets a DECISIONS entry.

## Refactoring a converter: the acceptance method

A converter moved onto the library must produce the same artifact. Build
with the old converter and the new one into separate directories, then
compare SHA-256 per bucket of:
- `model.mil` and `weights/weight.bin`;
- `metadata.json`, without coremltools' `conversion_date`;
- `tokenizer.json` and the manifest.

`coremldata.bin` (and `analytics/coremldata.bin`) are excluded: two builds by
the same, unchanged converter differ there, per compile. When the hashes are
identical, the compute plan and every grade are too. Where a change is
intended (an opt-in technique turned on), acceptance is the same compute-plan
op counts and parity-suite grades within the recorded floors instead
(`fixtures/parity/expectations.toml`).

The BERT converters passed this way: bge-small-en-v1.5 and
nlptown-sentiment are byte-identical to their pre-library builds at every
bucket.

## Gotchas

**Core ML and the ANE**
- **The fused attention op ignores its mask on the ANE** when the mask is an
  input of the ANE procedure that runs it: a model input, a CPU op's output,
  or an earlier procedure's (D25). Graphs where it seems to work (bge-small)
  work through a fallback that the iOS26 opset doesn't have: those graphs
  fail to load on CPU_AND_NE there ("error code: -14").
  `tools/repro_sdpa_mask.py --check` applies the rule to a compiled model.
- **Fully masked query rows NaN on the CPU.** Core ML's fused attention
  returns NaN for a row whose keys are all masked when
  |fill| × √head_dim > 65504 (so -30000 at head dim 64), and explicit
  softmax rewrites that divide by a row sum do too. The NaN then reaches
  every token through the next layer's value products. Use
  `masks.self_attending()` wherever a row can be fully masked.
- **The fused attention op crashed the process on the CPU at head dim 16**
  (SIGBUS/SIGSEGV, intermittently; macOS 27.0, M1 Max, coremltools 9). This
  was observed, not investigated. The library never emits the fused op.
- **CPU `reduce_max` over ≥ 256 elements returns max(x, 0)**, and
  `reduce_min` min(x, 0) (macOS 27). A small reduce can land on the CPU even
  under CPU_AND_NE. Use `reduce.blocked_max()` for any explicit max.
- **The ANE `linear` saturates above 2^15** (32,768 exact, 33,000 inf), and
  loses precision on small inputs (~3e-4 / rms). Its add, mul and layer_norm
  cover fp16's full range; fp16's own 65,504 isn't the limit that matters.
- **Multi-shape artifacts abort under `.cpuOnly`** on macOS 27 (D27). One
  static artifact per bucket, always.
- **An empty compute plan can come from a stale cache.** Core ML's cache of
  compiled bundles (`~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`)
  can hold a broken entry for an artifact's path; plans for that path then
  come back with every op unassigned, or fail with "internal failure", until
  the entry is gone. A copy at a new path (`cp -c -R`) reads normally.
- **The cache grows fast**, by tens of GB a day during conversion work, and is
  per executable name. Setting `CFFIXED_USER_HOME` to a private directory
  gives a process a private cache. Clear a cache only when no process is
  using it: rename it, then delete it.
- **The mean-pooling tail runs on the CPU** (reduce_sum, real_div, clip):
  one extra hand-off after the encoder, well within the compute-plan gate.

**coremltools and transformers**
- **Shape arithmetic in traced code crashes coremltools 9** under static
  shapes ("only 0-dimensional arrays can be converted to Python scalars").
  Never compute with `x.size()` or `x.shape`; use Python ints and literal
  reshapes, including the pooled output's `reshape(1, dims)`.
- **`0.5 * x * (1 + erf(...))` is fused back into native gelu**, and
  `x * sigmoid(x)` into silu. Write 2·f(x) and fold the 0.5 elsewhere.
- **transformers 5 builds BERT's mask in `masking_utils`**, bypassing the
  finite-mask patch; the BERT backbone refuses to load if a padded forward
  doesn't go through its patch. The library is validated on transformers
  4.57.
- **transformers fills masks with `finfo.min`**, and ModernBERT builds its
  masks before the CPU-only embedding gather, which is what put its mask on
  the CPU (D25).

**References and tokenizers**
- **AutoTokenizer isn't the tokenizer sidekick runs.** Its Python-side
  configuration can differ from `tokenizer.json` (lowercasing was dropped in
  one case found while building references). Encode from `tokenizer.json`.
- **sentence-transformers can load a checkpoint in fp16** when its config
  says so (gte-modernbert's `torch_dtype` is float16). References must force
  fp32.
- **sentence-transformers' `max_seq_length` isn't always 512**:
  all-MiniLM-L6-v2 uses 256, EmbeddingGemma 2048. A new model's manifest
  follows it, so truncation matches the reference; existing models cap at 512.
- **Gate units**: a reranker's raw logit is |x| ~ 10, so its error is gated
  in sigmoid space (cross-encoders train with a binary cross-entropy), with
  |Δlogit| reported.

## Tests

The logic that decides artifacts has torch-only unit tests (no Core ML), in
`tools/sidekick_convert/tests`: the calibration guard, the tokenizer rule,
masks, one-hots, the blocked max, the K choice, activations, pooling, the
manifest rules, and the ideal-fp16 simulation. From the repository root:

    python -m pytest tools/sidekick_convert/tests

or, without pytest:

    python -m unittest discover -s tools/sidekick_convert/tests -t tools

CI runs the Rust workspace only; run these before changing the library.
