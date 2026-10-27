# Converting models for sidekick

sidekick runs embedders either as Core ML models on the Apple Neural Engine
(ANE), GPU or CPU, or as ONNX models on ONNX Runtime's CPU backend (D40);
classifiers and rerankers run as Core ML only. Getting a Hugging Face
checkpoint onto the ANE faithfully takes more than `coremltools.convert`:
the ANE computes in fp16 with its own arithmetic limits, Core ML has
op-level bugs, and a model that converts can still be wrong in ways only a
careful gate catches. Most of this document covers that Core ML path; the
ONNX export (`--format onnx`, [ONNX export](#onnx-export)) reuses the same
converters and references, in fp32. This document catalogs what the
conversion library knows, and how to use and extend it.

The library is `tools/sidekick_convert`. The converters in `tools/` are thin
scripts on top of it; an embedder converter writes Core ML artifacts by
default, or one ONNX model with `--format onnx`. docs/MODELS.md lists the
validated models and their measured accuracy; docs/DECISIONS.md records why
each rule exists (entries cited as Dnn).

## Contents

- [Layers](#layers)
- [The conversion run and its gates](#the-conversion-run-and-its-gates)
- [Techniques](#techniques)
- [Calibration and evaluation](#calibration-and-evaluation)
- [Ideal fp16](#ideal-fp16)
- [Tokenizers](#tokenizers)
- [Manifests](#manifests)
- [Converters](#converters)
- [ONNX export](#onnx-export)
- [Causal language models](#causal-language-models)
- [Shared weights across buckets](#shared-weights-across-buckets)
- [Recording the build](#recording-the-build)
- [Reproducing a published build](#reproducing-a-published-build)
- [Adding a family](#adding-a-family)
- [Refactoring a converter: the acceptance method](#refactoring-a-converter-the-acceptance-method)
- [Gotchas](#gotchas)
- [Quick triage: is a model worth converting?](#quick-triage-is-a-model-worth-converting)
- [Will a new model convert? A checklist](#will-a-new-model-convert-a-checklist)
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
  `pool.Pool` pools an embedding in-graph for Core ML (CLS, mean or last
  token, optionally L2-normalized); an ONNX export leaves that pooling to the
  server. `sequence.SequenceClassification` is the checkpoint's own
  classification head, run through its task class, so each family's head
  semantics are exactly the checkpoint's; a reranker is this head with one
  label and a token_type_ids input. `per_token.PerToken` applies a module to
  every token and returns one value per token, `(1, S)`; the runtime reads
  the positions it placed (GLiNER2's `[L]` label markers).
- **Interface** (`core.Port`, `sidekick_convert.manifest`): the int32
  static-shape inputs, the output name, and the committed manifest the
  artifact must match.

`wrapper.compose(backbone, head, ports)` builds one static wrapper per
bucket; `recipes.embedder()` and `recipes.classifier()` turn the three into
a `core.Job`; `core.run(job, install_dir)` converts and installs it (an
ONNX job goes to `onnx_export.run()` instead, without buckets).

A backbone's contract, for a new family: `model` (the module the wrapper
holds, under the attribute `attr`, which names the converted weights),
`buffers(seq)`, `call(wrapper, inputs)`, `reference(ids, token_type_ids)`,
`example(seq, ports)`, `hidden_size`, `vocab_size`, `config`, and
`forbid_ops`. Rewrites (activation swaps, range and precision rewrites) are
functions of a backbone, applied by the recipe after the fp32 references
are computed from the unmodified checkpoint, so the fp32 gate proves each
rewrite exact.

## The conversion run and its gates

For a Core ML job, for every bucket, `core.run()`:

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
     cosine ≥ 0.999 against the fp32 reference. Classifiers: the served
     decision agrees with fp32 (the argmax wherever the fp32 top-2 margin is
     ≥ 0.05; for a multi-label case, each label's sign wherever its fp32 logit
     is ≥ 0.05 from 0), and max |Δp| after the activation;
   - **pad invariance**: the same input with random pad ids must give the
     same output. It catches a dropped attention mask (D25) and a mixer that
     reads pad states (D19) in seconds;
   - **finite output**;
6. with more than one bucket, re-exports every bucket against one shared
   weights file, gated ([Shared weights across buckets](#shared-weights-across-buckets));
7. installs `model_{seq}.mlmodelc`, hard-linking identical weight files,
   then the tokenizer and manifest. Nothing is installed until every bucket
   has passed every gate.

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
| `activations` | `TanhGelu`, `TwiceGelu` (erf), `TanhSilu`, `StableSilu`: 2·f(x) from tanh, erf or exp, since the native `gelu` (≤ 6e-3 on [-1, 1]) and `silu` (≤ 1.5e-2) are coarse on the ANE; the factor 2 is folded downstream | follow the model's own `hidden_act`: TanhGelu for `gelu_pytorch_tanh`, TwiceGelu for erf `gelu`, TanhSilu or StableSilu for `silu` (StableSilu where TanhSilu's 1 + tanh(x/2) cancels in fp16 for negative x: agent-jev, Lumma-fev and F2LLM; LFM2.5's rescaled MLP input makes the two alike, D39). Opt-in and measured per model: an erf GELU didn't help gte-modernbert and made its CPU path worse, and Core ML's CPU erf is coarser than its gelu | D17, D19, D20, D25, D39 |
| `precision` | the ANE linear's small-input floor (relative error ~3e-4 / rms(input)): power-of-two input scales folded upstream, `Descale` after | a linear whose calibrated input rms is small (≲ 0.02–0.03) | D17, D19, D20 |
| `saturation` | the ANE linear saturates above 2^15: `check()` the rule (calibrated outputs ≤ 0.85 × 2^15), `choose_k()` / `headroom_at()` for a residual stream run at 1/K | check on every model; residual K where a massive activation crosses it (ModernBERT) | D25 amendment |
| `pooling` | CLS, masked mean (by the attention_mask input, never by token id; summed at 1/32 for fp16 range), last token without a gather, L2 squared at 1/32; outputs end in a literal `(1, dims)` reshape | every pooled head | D15, D17, D20 |
| `reduce` | `blocked_max()`: max over 128-wide slices, exact everywhere (macOS 27's CPU `reduce_max` over ≥ 256 elements returns max(x, 0)) | any explicit max over a long axis | laya |
| `traceable` | `rotate_half`, `repeat_kv` without shape arithmetic; `install()` patches a transformers module | RoPE and grouped-query attention | D17 |
| `attention.transformer_encoder_layer` | `nn.TransformerEncoderLayer(norm_first)` with its attention written out, so its fast path is never traced | task heads built from PyTorch encoder layers (laya) | D28 |
| `onehot` | selections from int32 inputs by comparison with a position constant, instead of data-dependent gathers | inputs that index positions (laya's markers and question type) | D28 |
| `relative_shift` | relative-position terms read by query-key distance, `score[q, k] += x · T[idx(q − k)]`, without gathers: the table expanded over the bucket's 2L − 1 distances as a weight-only constant, then one matmul and a reshape/slice skew | any attention bias that gathers by q − k (DeBERTa-v2/v3's c2p and p2c); transformers' gathers run on the CPU and take their matmuls along | `tools/probe_deberta.py` |

Two rewrites are specific to one family so far and live in its backbone:
the range rewrite for activations past fp16's own maximum (EmbeddingGemma's
residual stream at ~1.5e5, D17; `backbones.gemma3.fp16_range_rewrite`), and
pad zeroing before every convolution of a convolutional mixer (D19;
`backbones.lfm2`, on by default, `--no-pad-zeroing` for the negative
control).

## Calibration and evaluation

Two kinds of inputs, kept apart by type:

- **`core.Calibration`** decides rewrites: the residual K, input rescales,
  anything that changes the artifact. It refuses texts from the graded parity
  corpus (`fixtures/parity/corpus.toml`), because calibrating on what is
  graded would flatter the grades. A converter's own texts, or a committed
  calibration set, are fine. A text that ends with a corpus text counts as
  the corpus text (a model's prompt prefix doesn't make it new);
  `Calibration.without_graded()` drops such texts and says how many.
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
   a linear with its bias) is one operation, rounded once.
   **Attention runs in the explicit form every converted program uses**
   (D25): matmul, scale, mask, softmax, matmul, each output stored. That
   holds whatever attention implementation the model loaded with: a
   checkpoint loaded with `sdpa` and the same checkpoint loaded with `eager`
   give bit-identical results. PyTorch's own decomposition of fused attention
   scales q and k separately before the matmul. Those are two roundings no
   converted program has, and on laya they inflated the ceiling 1.5x at the
   mean and 2.8x at the maximum. `nn.TransformerEncoderLayer`'s fast path
   runs a whole layer as one fused operation, so it is turned off.
   **A normalization is one operation too**: fp32 inside, its output stored
   in fp16. Most checkpoints write RMSNorm out by hand (`x.pow(2).mean()`,
   `rsqrt`, a multiply). Rounding those steps would store RMSNorm's x² in
   fp16, which overflows past |x| 256. On agent-jev, whose attention-sink
   activation reaches 6,800, that zeroed the sink token and left the last
   token at cosine 0.53 to fp32 on a 16-token input. A converter computes a
   norm whole or pre-scales it, so no converted program stores that
   intermediate. With the norm as one operation, the same token is at
   0.999999. Normalizations are found by class name: any module whose name
   ends in `RMSNorm` or `LayerNorm` (Qwen3's, Gemma's, Llama's, Nandi's, and
   PyTorch's).
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
function on its wrapper; that is a diagnosis, not a ceiling. Even a
power-of-two rewrite is not free in fp16. On laya's converted wrapper at
bucket 128, the residual 1/K raises the ideal-fp16 error by 30% at the mean
and p99 with K = 2, against K = 1 (which matches the published model).

    from sidekick_convert import fp16sim
    logits = fp16sim.run(model, input_ids=ids, attention_mask=mask).logits

`fp16sim.run` treats every tensor argument as an input. `fp16sim.ideal_fp16()`
is the same as a context manager, for a call that mixes inputs and constants;
pass it the model (`module=`) so it finds the normalizations. `ops=` rounds
only some operations' outputs, which is useful for diagnosis and never for
grading.

**A ceiling exists only if fp16 storage keeps every value meaningful.** An
infinity or NaN may flow only through operations that handle it exactly:
masking arithmetic, softmax, comparisons and data movement. That is how a
`finfo.min` mask fill, -inf in fp16, does its job. Reaching anything else (a
reduction, a matmul, a square root), it would corrupt the result, possibly
silently: rsqrt of an overflowed sum is 0, and the output stays finite. So
the run raises `fp16sim.Fp16Overflow`, naming the operation and where the
value came from, e.g. "mean reads an infinity (pow stored |x| up to 4.4e+07
in fp16, past 65,504)". The reference generators then leave out the `fp16`
oracle and print why. The check covers whatever the normalization rule
misses.

Caveats:

- **A real engine can beat the ceiling slightly.** Core ML fuses some runs
  of operations into one (a decomposed layer norm, a GELU pattern) and rounds
  once where the simulation rounds each step. A ratio a little under 1 is
  not an error.
- **A model whose activations pass fp16's 65,504 has no ceiling as
  published.** EmbeddingGemma's residual stream is one example. The run
  raises `Fp16Overflow` (above), and the reference generator leaves out the
  fp16 oracle rather than write a meaningless one.
- **The single worst case is sensitive to the exact rounding points.** On
  laya at bucket 128, a hand-placed set of rounding points agreed with this
  definition on mean and p99 Δp to within 15% but differed 1.7x on the
  maximum. Two independent implementations of the definition itself agree
  exactly. That sensitivity is why the definition lives in one function.
  Grades that divide by the ceiling's maximum should report the p99 ratio
  too. On a corpus of under 100 cases the nearest-rank p99 is the maximum,
  so there a p99 ratio carries the same sensitivity.

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
- classifiers: `token_type_ids` is named only by text-ranking manifests,
  exactly when pairs carry segment ids (older daemons would feed a
  text-classification model no segment ids). An ONNX embedder's manifest
  names it when its graph takes it (BERT-family published exports);
- embedders: `[prompts]` equals the checkpoint's sentence-transformers
  `prompts` (`config_sentence_transformers.json`), every name and text
  exactly, so the named prompts a request chooses by `prompt_name` are
  the checkpoint's (D45). On a mismatch the message gives the table to
  paste into the committed manifest, and `python -m
  sidekick_convert.manifest prompts <hf-model-dir>`, run from `tools/`,
  prints it for a new one. A checkpoint that publishes no prompts leaves
  the table to the manifest: prompts its model card gives only in prose
  go there by hand, with the source cited in a comment;
- `source.revision` equals the snapshot's revision.

Buckets and `[placement]` are Core ML's; an ONNX install has neither (see
[ONNX export](#onnx-export)). A Core ML install gains one thing the
committed file doesn't have: a `[placement]` table recording, per bucket,
the compute plan the gates read on this machine. A live plan read in the
daemon is a second compile, up to minutes and hundreds of MB for a large
bucket, so the daemon reports this record instead.

    [placement]
    compute_units = "cpu_and_ne"   # the units the model is served with
    chip = "Apple M1 Max"          # sysctl machdep.cpu.brand_string
    macos_build = "26A428"         # sw_vers -buildVersion
    date = "2026-10-01"

    [placement.buckets.128]
    ane = 229
    gpu = 0
    cpu = 16
    unassigned = 363
    total = 608
    off_ane_ops = { add = 3, cast = 4, gather = 1 }   # assigned off the ANE, by operator

The counts are `sidekick_coreml::compute_plan`'s, so the daemon's own reads
match them. That means the `main` function with nested blocks, constants
counted as unassigned, and total = ane + gpu + cpu + unassigned (bge-small's
128 bucket reads 229 / 0 / 16 / 363 either way). The plan is read for the
manifest's `compute_units`, not always for the ANE. A bucket whose plan was
unreadable is left out. Daemons up to 0.5 ignore the table: their manifest
structs don't deny unknown fields.

**Convert for a matching daemon.** The installed manifest is the
checkout's committed one, so it can carry keys only the matching daemon
knows. Most manifest sections refuse unknown keys, and a daemon that
meets one skips the whole model, with only a warning in its log: the
model is missing from `/v1/models`. For example, the nli manifests'
`[classify.nli] classes` (D47) needs sidekickd 0.15.0 or later. Convert
with a checkout no newer than the daemon that will serve the model, or
upgrade the daemon first. To enable a newer key on a model already
installed, add the key to the installed manifest rather than copying the
committed one over it, which would drop its `[placement]` table.

## Converters

| converter | backbone | head | notes |
|---|---|---|---|
| `convert_bge_small.py` | BERT, fused attention | CLS pool | fused attention kept for byte identity with the graded artifact; see its docstring |
| `convert_bert_embedder.py` | BERT, explicit | CLS or mean pool (from the checkpoint) | MiniLM, e5 and other BERT-family sentence-transformers |
| `convert_bert_classifier.py` | BERT, explicit | sequence classification | classifiers and rerankers; `--twice-gelu` opt-in |
| `convert_gte_modernbert.py` | ModernBERT, explicit, residual K calibrated (2) | CLS pool | `--attn sdpa` builds the fused-attention negative control |
| `convert_qwen3_embedding.py` | Qwen3 decoder, precision rewrite | last-token pool | F2LLM; any Qwen3 last-token embedder (manifest by install-dir name) |
| `convert_laya.py` | ModernBERT, explicit, residual K pinned (2), TwiceGelu | laya's marker head (`heads/laya.py`) | the decision model's own forward is the fp32 reference; gated on CPU_AND_NE, CPU_ONLY reported |
| `convert_embeddinggemma.py` | Gemma3, fp16 range rewrite (residual K = 32) and per-layer precision scales | mean pool, the two sentence-transformers dense layers, L2 (`heads/gemma_st.py`) | its calibration keeps one graded-corpus text under a logged legacy exemption, because the graded artifact was calibrated with it |
| `convert_embeddinggemma2.py` | EmbeddingGemma 2 (Gemma 4 text tower, per-layer inputs), RMSNorm pre-scales, TanhGelu | mean pool, scaled per bucket | needs transformers 5.19; see below |
| `convert_lfm25_embedding.py` | LFM2 (short convolutions with pad zeroing, attention), per-layer precision scales | CLS pool, L2 in graph | `--no-pad-zeroing` builds the negative control |
| `convert_nli.py` | BERT/RoBERTa/XLM-R (explicit), DeBERTa-v3 (relative shift) or ModernBERT (residual K), by model_type | the checkpoint's sequence-classification head, one (premise, hypothesis) pair per prediction | zero-shot `format = "nli"`; gates the manifest's served path, reports the others |
| `convert_julia.py` | mmBERT-small (ModernBERT), explicit | laya's marker head, Julia-1's option rendering | Julia-1's own code, pinned by sha256, is the fp32 reference; laya's interface |
| `convert_gliner2.py` | DeBERTa-v3, relative shift | GLiNER2's per-token classifier, read at the label markers | zero-shot `format = "gliner2"`; needs the gliner2 package for the gate inputs |
| `convert_fev.py` | Nandi causal decoder | fev's pointer head | zero-shot `format = "fev"`; the checkpoint's own code is the fp32 reference |
| `convert_agentjev.py` | Qwen3 decoder, tree mask, bucket-invariant softmax | AgentJev's set transformer and scorer | zero-shot `format = "agentjev"`; chunked by default (D37) |
| `convert_causal_lm.py` | LlamaForCausalLM, channels first, rotated, GPTQ onto 8-bit palettes (`llama_causal`) | the final norm and head on the last chunk, one row chosen by `sel` | a `kv-state-1` decoder for the chat route; its own pipeline ([Causal language models](#causal-language-models)) |

Every converter takes `<hf-model-dir> <install-dir> [buckets...]`, plus
`--time` and its own flags; the embedder converters also take `--format
onnx`. Requirements:
- every converter: Python with torch, transformers 4.x (the BERT backbone
  proves at load that its finite mask is on the forward path), tokenizers
  and numpy. The library is validated on transformers 4.57.6;
- except `convert_embeddinggemma2.py`, whose model code first shipped in
  transformers 5.19. Run it from a separate environment with transformers
  >= 5.19 and sentence-transformers >= 6.1 (for its references, which
  also need Pillow and torchvision: the checkpoint's processor loads them
  even text-only), and keep the other converters on 4.57.6;
- the Core ML path (the default): arm64-native Python on macOS, coremltools,
  and Xcode for `xcrun coremlcompiler`. Use a Python version coremltools
  publishes wheels for (3.13 or earlier for coremltools 9.0; the converters
  are run on 3.12). On a newer Python, pip installs coremltools without its
  native libraries, so it imports but can't write a model; `core.run`
  checks for them before it converts anything;
- `--format onnx`: `onnx` and `onnxruntime` instead, with neither
  coremltools nor Xcode. Every export so far was made on Apple silicon; it
  hasn't been tested elsewhere. Any recent `onnxruntime` wheel for your
  Python will do: the export's gate only runs the graph on the CPU, to
  check it against the checkpoint. It needn't match the ONNX Runtime the
  daemon bundles (D40), which is what the published onnx-cpu grades
  measure; ONNX Runtime 1.28.3, for one, has no Python 3.14 wheel.

## ONNX export

`--format onnx` on an embedder converter exports the same model for ONNX
Runtime's CPU backend (docs/DECISIONS.md D40) instead of Core ML: the same
backbone, head and gate texts, gated against the same fp32 references
(`sidekick_convert/onnx_export.py`).

- **One graph, dynamic shapes.** `model.onnx` takes int64 `input_ids` and
  `attention_mask` of any batch and length; there are no buckets. Weights
  past protobuf's 2 GB go to one `model.onnx_data` beside it.
- **Pooling.** The graph outputs `last_hidden_state` and the server pools it
  (`mean`, `cls` or `last_token`), as it does a published export. A head that
  is more than pooling stays in the graph: EmbeddingGemma's mean, dense stack
  and L2 output `embedding`, with `pooling = "none"`.
- **Buckets and the length limit.** The installed manifest drops the
  buckets, and the bucket arguments are ignored. The ONNX build's
  `max_seq_len` is the Core ML manifest's unless the model's `onnx.toml`
  sets its own (below); the gate texts are cut at it, and it is checked
  against the checkpoint like the Core ML one (at most the position
  embeddings, less any offset). Only embedder converters take `--format
  onnx`.
- **Bucket lists and ONNX caps are per-model decisions.** Nothing derives
  them: each is chosen for its model and recorded, with the reason, in the
  committed manifest or `onnx.toml`. A Core ML bucket is a program that
  maps its own copy of the weights when loaded (its file on disk is shared,
  D49), and its attention scores grow with the square of its length, so the bucket list is a cost the model's size sets (most
  models stop at 512; EmbeddingGemma-2's reaches 2,048). The ONNX graph
  runs each input at its own length, so its cap costs nothing until a long
  input arrives, and can be the length the model was trained for, once it
  is graded there. Qwen3-Embedding-0.6B's ONNX build takes 4,096, graded
  on the long-input parity corpus (`fixtures/parity/corpus_long.toml`;
  its Core ML builds are in
  [MODELS.md](MODELS.md#qwen3-embedding-06bs-core-ml-builds)). A cap past 512
  should be graded on that corpus before it ships, with its time and
  memory at the cap measured (`examples/onnx_memory`,
  `tools/bench_onnx.py`). Raising a cap changes the vectors of inputs
  past the old one, which were cut there before, and makes the model's
  two builds disagree on them: say so in the release notes.
- **fp32, no rewrites.** A recipe's `rewrites=` don't run for an ONNX
  export: the precision and range rewrites exist for the ANE's fp16
  arithmetic, and the activation swaps for its coarse native ops (so
  `--twice-gelu` has no effect). All are exact in fp32 and would only add
  work on the CPU.
- **Gates,** through ONNX Runtime's Python API: every gate case against the
  checkpoint's own output (cosine at least 0.99999), the cases in one
  right-padded batch against their unpadded runs, and random pad ids. The
  daemon ships a pinned ONNX Runtime
  ([INSTALL.md](INSTALL.md#onnx-runtime)), fetched by
  `tools/fetch_onnxruntime.sh`; the parity suite grades the export on
  that runtime.
- **Manifest.** The installed manifest is derived from the committed Core ML
  one (same id, tokenizer, dims, prefixes, Matryoshka), with backend
  `onnx`, artifact `model.onnx`, the pooling, and every extra graph input
  named in `[io]`. Settings for the ONNX build alone live in an `onnx.toml`
  beside the committed manifest: an `[onnx]` table (such as
  `default_intra_op_threads`, which a Core ML manifest refuses), appended
  to the installed manifest, and a top-level `max_seq_len`, which replaces
  the Core ML manifest's, with the comment above it. No other top-level
  key is accepted. Model ids are the same across backends, so compare the
  two side by side from separate models directories.
- **`--onnx-weights fp16`** stores the weights in fp16 behind casts to fp32:
  half the file, the same fp32 arithmetic. It saves disk, not memory: ONNX
  Runtime upcasts at load, and Qwen3-Embedding-0.6B's fp16-weight build
  holds more than its fp32 one (a footprint of 2.7 GB against 1.7 GB, M1
  Max). Measured: Qwen3-Embedding-0.6B, a bf16 checkpoint, grades A at
  1 − cosine 2e-11, and all-MiniLM-L6-v2, an fp32 one, passes the export
  gate at 0.9999998.
- **No int8.** ONNX Runtime's int8 kernels quantize activations at run time
  with one scale per tensor, computed over the whole batch, padding
  included. A model's output then depends on what it is batched with, which
  breaks sidekick's batch and pad invariance, and one scale per tensor
  collapses on activation outliers. On Qwen3-Embedding-0.6B, both a dynamic
  int8 export and the published one grade D even one input at a time
  (median cosine 0.86, worst 0.406 on an M1 Max), and fail the batch and
  pad gates when batched. A weight-only int8 file
  doesn't escape it: ONNX Runtime's optimizer fuses its dequantize into the
  same dynamic kernel.

Grade an export with the parity suite's `onnx-cpu` path, natively: build
the suite with `--features onnx` (any OS, Linux included; add `coreml` on
macOS to grade the Core ML paths too), point `ORT_DYLIB_PATH` at the
runtime library, and run it on a models directory holding the ONNX install:

    ORT_DYLIB_PATH=<dir>/libonnxruntime.dylib \
    cargo run --release -p sidekick-embed --features onnx --example parity -- \
        --models-dir <dir> --refs <refs> --model <id> \
        [--fixtures <repo>/fixtures]

The fixtures (corpora, expectations) come from `--fixtures`,
`$SIDEKICK_FIXTURES`, `./fixtures` or the checkout the suite was built
from. A model whose `max_seq_len` reaches the long-input corpus
(`fixtures/parity/corpus_long.toml`, 600 tokens and up) is graded on it
too, once it has that reference (`tools/parity_reference.py --corpus
long`); the suite reports it as a section of its own, and warns when the
reference is missing. `tools/onnx_parity.py` is a cross-check from Python: it writes the
suite's worker result for the same model, which `parity --external
result.json` grades beside the native run. An int8 build fails the floors
by design. `tools/bench_onnx.py` measures load time, resident memory and
latency per length, batch size and thread count.

## Causal language models

An open chat model (D53; docs/design/open-llm.md) converts to a
`kv-state-1` decoder: chunk programs over contiguous runs of layers, each
one static function taking T tokens per call and owning its layers' KV
cache as Core ML state, plus the host's embedding and RoPE tables. Its
pipeline is its own (`sidekick_convert/causal_lm.py`), since it has no
buckets, its inputs are fp16 and its weights are quantised, but it uses the
library's layers: a backbone (`backbones.llama_causal`), a technique
(`techniques.gptq`), the cosine scale, the compute plan, `core.Calibration`'s
guard and the tokenizer rule.

    python tools/convert_causal_lm.py <hf-model-dir> <models dir>/minicpm5-2b --work <work dir>
    python tools/grade_causal_lm.py <models dir>/minicpm5-2b --work <work dir> --decoder <kv_decoder binary>

The manifest is `examples/chat/<id>/chat.toml`, reviewed like the others:
the converter checks it against the checkpoint (shapes, the layer split
summing to the model's layers, context = C − T + 1, C under the ANE length
cap, the stop tokens and sampling defaults from `generation_config.json`,
the template's bos and eos from `tokenizer_config.json`) and installs it
with two tables added: `[build]`, the recipe and the measured residual
scale, and `[placement]`, each chunk's compute plan.

**The recipe** (`recipes.causal_lm`; #173 found it):
- the channels-first rewrite of the backbone: 1x1 convolutions on a
  [1, D, 1, T] residual stream, explicit attention under a finite mask
  (D25), StableSilu (D39), RMSNorm inputs pre-scaled by a power of two so
  their squares stay in fp16 range, and the residual stream at 1/K when a
  projection's output would cross the ANE linear's 2^15 (K = 1 for
  MiniCPM5-2B, whose rotated outputs stay under 200);
- R1 and R2 Hadamard rotations folded into the weights, which spread the
  massive activations over every channel: the largest RMSNorm input falls
  from about 5,500 to about 226, and the weights' outliers to what Gaussian
  weights give;
- GPTQ with act-order onto 8-bit palettes, one palette per 16 output
  channels, fixed beforehand by k-means as coremltools computes it, stored
  with the palettizer's `unique` mode so GPTQ's values survive exactly.
  Per-channel grids only: the ANE runs per-block int8 on the CPU without
  saying so, and per-block int4 doesn't load. 4-bit is ruled out for this
  model (87% top-1 at best);
- chunks at layer boundaries, as the manifest's `layers` gives them (each
  under the 1 GiB per-program limit, D32), with the head split into four
  convolutions on the last.

**Steps.** Each heavy step runs in its own process and leaves its result
in the work directory, where a rerun reuses it: the fp32 references (the
checkpoint's own teacher-forced logits, its greedy continuations, the
ideal-fp16 ceiling), the range calibration and the rewrite gate per
calibration draw, the palettes, GPTQ per draw, the PyTorch screen per draw,
then one conversion per chunk. Only the references hold the whole model in
fp32 (about 10 GB for a 2B model); every other step reads the checkpoint
tensor by tensor and holds one layer in fp32.

### Calibration and evaluation sets

Both are committed under `fixtures/causal-lm/`, with the token ids each
model's tokenizer gives them in `<model id>.tokens.json`. The converter
checks the text files' sha256 and the tokenizer's against that file and
re-encodes every document, so a changed text, tokenizer or template fails
the run instead of changing the build.
`tools/make_causal_lm_fixture.py` built them, and builds the token file for
a new model.

- **Calibration** (`calibration.jsonl`, about 115,000 tokens for
  MiniCPM5-2B): spans of this repository's documents and code at a pinned
  commit, conversations written for the fixture (both MIT), and the lead
  sections of 149 Wikipedia articles, each pinned to a revision (CC BY-SA
  4.0, attributed; THIRD_PARTY_NOTICES.md). Each document records its
  source and licence.
- **Disjoint from what is graded**, checked at every run: no parity-corpus
  text (`core.Calibration`), no run of 32 tokens shared with an evaluation
  text (shorter runs are shared link anchors and signatures, not passages),
  and no Wikipedia article the chat benchmark draws on.
- **A draw** is what one conversion calibrates on: the documents in a
  seeded order (numpy's `default_rng(seed).permutation`), packed into
  sequences of exactly 2,048 tokens, each starting at a document, until
  40,960 tokens. Draws 0 and 1 share about a third of their text.
- **Evaluation** (`evaluation.jsonl`): #173's set, so grades stay
  comparable with it: eight texts (this repository's prose, Rust and TOML,
  a run of numbers), teacher-forced over their first 256 tokens and graded
  on the logits at positions 127–255 (1,032 positions), and six chat
  prompts whose greedy continuations are compared with fp32's.

### Grading

Every grade compares logits rows with the checkpoint's fp32 forward at the
same positions: per position the cosine of the two rows, and whether their
top-1 tokens agree. The PyTorch screen runs the quantised model in fp32;
the ANE grade runs the installed build through `sidekick_coreml::KvDecoder`,
one token per call (`tools/grade_causal_lm.py`).

| | A | B | C | D |
|---|---|---|---|---|
| first-percentile cosine, at least (the parity suite's cosine scale) | 0.9999 | 0.999 | 0.985 | below |
| top-1 agreement, at least | 99.5% | 98.0% | 95.0% | below |

- **The grade is the worse of the two letters**, anchored on a percentile
  as D26 anchors classifiers on p99: #173 found the worst position moves
  with the calibration draw while the first percentile holds.
- **A wide flip caps it at C:** a position where the build's top-1 differs
  from fp32's and fp32 rates its own top token at least twice as likely as
  the runner-up (a log-probability margin of ln 2 or more), as a decision
  flip caps a classifier. Flips between near-ties are counted, not capped.
- **The worst position is reported, not graded**, and so are the mean
  cosine and the ideal-fp16 ceiling (`fp16sim`) on the same positions:
  for MiniCPM5-2B, p1 0.99999, top-1 99.90%, worst 0.99998 (A).
- **Non-finite logits are F.**
- **The stability check:** the converter also quantises with a second
  calibration draw (seed 1), screened in PyTorch and never compiled. The
  two draws' screens must grade alike; a grade that depends on the draw is
  reported as the lower one.
- **The screen gates compiling:** below the recipe's floor (B) nothing is
  compiled.
- **Gates** before install: the rewrite gate (the rotated, rescaled model in
  fp32 against the checkpoint, worst cosine at least 0.99999), no forbidden
  op (the fused attention op, the native `silu`), each chunk's weights under
  1 GiB, and every operation of every chunk on the ANE in its compute plan:
  a decoder's state crashes off it.

### The ANE cache

Each compile of a build for the ANE leaves about a copy of its weights in a
system cache that can't be deleted (docs/design/coreml-cache.md): about
2.2 GB for MiniCPM5-2B. A build costs two: the compute-plan gate's read of
each chunk (`--no-plan` skips it) and the first load by the grader. Screen
recipe changes in PyTorch first, keep the build at one path, and run the
grader from one `kv_decoder` binary signed with a fixed identifier
(`codesign -s - -f -i <identifier>`), which a rebuild of the example then
reuses.

### Reproducing a build

The `[build]` table records everything a rerun needs: the rotation seed,
the calibration set's sha256, its token count and the draw's seed, the
palettes' group and k-means seed, GPTQ's order, damping and thread count,
the residual scale and the head split. K-means runs single-threaded, GPTQ
in float64 in a fixed order, and every step at the recorded torch thread
count, so a rerun into a fresh work directory on the same machine and
library versions gives the same program bytes (`model.mil`,
`weights/weight.bin` and `metadata.json` of every chunk), host tables and
chat.toml; `coremldata.bin` differs per compile, as it does for every
converter ([the acceptance method](#refactoring-a-converter-the-acceptance-method)).
Two MiniCPM5-2B builds from empty work directories matched this way (#184).

### Requirements

The converters' environment ([Converters](#converters)), plus
scikit-learn and threadpoolctl (the palettes' k-means) and safetensors.
transformers 4.57.6 reads MiniCPM5-2B, a plain Llama, although the
checkpoint was saved with 5.6.

## Shared weights across buckets

Each Core ML bucket is its own program, and each program (each chunk of a
chunked bucket) reads its constants from `weights/weight.bin`. The weights
are the same in every bucket, but separately converted files aren't:
coremltools writes constants in program order, and the few that depend on
the bucket (folded position and token-type tables, RoPE tables, the causal
mask, all under 1 MB at 512 tokens) shift every tensor after them. So a
model with three buckets stored its weights three times. Since D49 the
converter stores them once per chunk (`shared_weights.py`):

- **The re-export.** After every bucket has passed its gates, the
  converter loads each bucket's saved `.mlpackage` back into MIL and writes
  it again, with no graph passes, against one union weights file per chunk:
  each distinct tensor (by dtype, shape and bytes) is written once, in
  bucket order, so the tensors every bucket holds come first and each
  bucket's own follow. Every bucket's program then points into the same
  file, and is compiled with `xcrun coremlcompiler` as before.
- **Gates.** The union build replaces the separate one only if:
  - every compiled `weights/weight.bin` in the chunk is the union file,
    byte for byte;
  - every compiled `model.mil` equals the separate build's once weight
    offsets are set aside: the same program, with its constants elsewhere;
  - on CPU_ONLY, CPU_AND_GPU and CPU_AND_NE, every program gives the
    separate build's outputs on the gate cases, bit for bit (each chunk fed
    the separate build's input, so each program is judged alone). Each
    comparison runs in a child process with a Core ML cache of its own,
    deleted when it exits, so the check leaves no compiled copies behind.

  A failure stops the conversion; `--no-shared-weights` converts with one
  file per bucket, as before D49.
- **Keeping the packages.** The converter deletes each `.mlpackage` once
  it is compiled. `--keep-packages DIR` (an empty or new directory) keeps
  them: each bucket's package as converted in `DIR/separate/`, and the
  re-exported packages the installed programs were compiled from in
  `DIR/shared/`, their identical weight files hard-linked. Keep them for a
  build you publish: the published files are the compiled programs, and
  the packages let them be recompiled, or compared before compilation.
- **Per chunk, never across chunks.** The ANE's weight limit (D32) is
  judged on the whole weights file a program references, not on the
  tensors it reads: a file spanning both of Qwen3-Embedding-0.6B's chunks
  (about 1.2 GB) put every chunk on the CPU. So chunk k of every bucket
  shares one file, and no file spans chunks. For a model served on the ANE,
  a union file over the limit keeps that chunk's files separate, and the
  converter says so; the union is the shared weights plus every bucket's
  own tables, so it is a little larger than any one bucket's file.
- **Install.** The union files are installed as one file with a hard link
  per bucket. A move within one filesystem keeps the links; after a move
  across filesystems, or a copy that doesn't preserve hard links (`cp -R`),
  `python -m sidekick_convert.shared_weights link <model-dir>` links
  identical weight files again. It keeps a copy wherever the filesystem
  refuses a link. `ditto` and `rsync -aH` preserve hard links; `cp -Rc`
  (APFS clones) saves the disk too, with a separate file per bucket.
- **What is shared:** the files on disk, and so the download (D49 records
  the publishing format). The daemon needs no change: each bucket is still
  its own program at its own path, and the weight-limit check sums each
  program's own files.
- **What isn't:**
  - **Core ML's compile cache** keeps a specialized copy of the weights
    per bucket and per compute path, as it did (for Qwen3-Embedding-0.6B,
    about 3.6 GB on the GPU or CPU for three buckets). On the GPU and the
    CPU that copy is in the loading process's cache under its home; an
    ANE compile leaves only about 12 kB there and its program, about the
    weights' size, in the system's ANE cache (see "Core ML's cache"
    below).
  - **Memory.** Each loaded bucket maps its own copy from that cache, so
    two loaded buckets map the weights twice, as before.
  - **Compile time.** Every bucket is compiled for each compute path on
    its first load, as before.
- **Measured** (M1 Max, macOS 27.0, coremltools 9.0): weight files on disk,
  separate buckets → shared, with every output bit-identical on CPU, GPU
  and ANE and every parity grade unchanged:

  | model | buckets | separate | shared |
  |---|---|---|---|
  | bge-small-en-v1.5 | 128, 256, 512 | 199.5 MB | 67.4 MB |
  | F2LLM-v2-160M | 128, 256, 512 | 956.3 MB | 319.5 MB |
  | Qwen3-Embedding-0.6B, chunked | 1,024, 2,048 | 2,407.3 MB | 1,215.7 MB |

## Recording the build

Every build gets an entry in the checksum ledger, docs/data/ledger.json,
whether or not it is ever published (#136; docs/PUBLISHING.md). Record a
build once it has passed its gates and been graded:

```sh
python3 tools/ledger.py add <models dir>/<id> --reports <parity reports dir> \
    --converter-commit "$(git rev-parse --short HEAD)" --os "<macOS version (build)>" --chip "<chip>"
```

The entry holds each file's sha256 and size, each shared weights file
once with all its paths, and the grades of the reports that graded those
exact files (matched by weights hash). A later reconversion is compared with it by
`python3 tools/ledger.py verify <models dir>/<id>`, which names the layer
that moved: the weights, the compiled program, the tokenizer or the
manifest.

## Reproducing a published build

Every build sidekick publishes on Hugging Face can be rebuilt from its
original checkpoint. The model card names the checkpoint and its revision,
the converter, and the build's ledger id; its "Checksums" section lists the
converter's commit and the tool versions the build was made with.

1. Check out sidekick at the converter's commit (or a later one), and set
   up the converter's environment ([Converters](#converters): Python,
   coremltools and Xcode for Core ML, `onnx` and `onnxruntime` for
   `--format onnx`).
2. Download the checkpoint at the card's revision, with
   `hf download <repo> --revision <revision> --local-dir <checkpoint dir>`.
3. Run the converter into a fresh directory, with the build's options:
   the committed manifest's header (examples/manifests/,
   examples/classifiers/) gives the command, and an ONNX build adds
   `--format onnx`:

   ```sh
   python tools/convert_bge_small.py <checkpoint dir> <models dir>/bge-small-en-v1.5
   ```

4. Compare the result with the published build's entry:

   ```sh
   python3 tools/ledger.py verify <models dir>/bge-small-en-v1.5 --id <ledger id from the card>
   ```

   It names each file that differs and its layer: the weights, the
   compiled program, the tokenizer or the manifest. Core ML compiles on
   another macOS or Xcode version can legitimately change the compiled
   program while the weights stay identical; the parity suite (docs/MODELS.md)
   then grades the rebuild, which is what decides whether it is as good.

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
3. **Pick a head**, or add one to `heads/` if the output is new (a marker
   head, say; `per_token` covers one output per token).
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
8. **For an embedder, export it with `--format onnx`** and grade the
   `onnx-cpu` path too. A head that is more than pooling needs an
   `onnx_module(backbone)` with dynamic shapes (see `heads/gemma_st.py`).

## Refactoring a converter: the acceptance method

A converter moved onto the library must produce the same artifact. Build
with the old converter and the new one into separate directories, then
compare SHA-256 per bucket of:
- `model.mil` and `weights/weight.bin`;
- `metadata.json`, without coremltools' `conversion_date`;
- `tokenizer.json` and the manifest.

`coremldata.bin` (and `analytics/coremldata.bin`) are excluded: two builds by
the same, unchanged converter differ there, per compile.

Identical math isn't enough for identical bytes. `torch.jit.trace` names each
value after the Python local it is bound to (rebinding a name makes
`name.1`), and those names become the converted program's variable names. A
refactor keeps the old code's binding structure, and the wrapper's attribute
names, which name the weights. When the hashes are
identical, the compute plan and every grade are too. Where a change is
intended (an opt-in technique turned on), acceptance is the same compute-plan
op counts and parity-suite grades within the recorded floors instead
(`fixtures/parity/expectations.toml`).

The converters passed this way: bge-small-en-v1.5, nlptown-sentiment,
gte-modernbert-base and F2LLM-v2-160M are byte-identical to their
pre-library builds at every bucket.

## Gotchas

Matthijs Hollemans' [neural-engine](https://github.com/hollance/neural-engine)
notes are the long-standing community reference on the ANE: supported
layers, fp16 behaviour, device generations and diagnostics. Some of it
predates ML programs and `MLComputePlan`, and sidekick's measurements
refine two points there:
- Core ML's CPU path can run an fp16 ML program in fp16, as the notes
  allow. Whether that is safe depends on the machine: Qwen3-Embedding-0.6B
  grades B on an M1 Max's CPU, but collapses to exactly zero on an M4 Max
  under macOS 26.5 ([MODELS.md](MODELS.md)).
- The notes warn about activations above about 1e2 or below about 1e-4.
  The limits measured here are fp16's own range (65504; D17's range
  rewrite), a precision floor on small `linear` inputs (D17), and the
  `linear` op saturating at 2^15 (D25 amendment). Large activations below
  2^15 are not an ANE problem by themselves.

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
- **The fused attention op crashes the process on the CPU at head dim 16**:
  SIGBUS inside libBNNS, Core ML's CPU kernels, mostly on the first
  prediction. With 12 heads over 128 tokens (CPU_ONLY; macOS 27.0, M1 Max,
  coremltools 9), it crashed 10 of 13 processes at head dim 16, whether or
  not the prediction inputs were kept referenced. It crashed none of 10 at
  head dim 64, and none of 6 at head dim 16 with the same attention written
  out explicitly. It is not the input-lifetime crash below. The library
  never emits the fused op.
- **coremltools crashes the process about a second after a prediction
  whose inputs Python has freed.** Core ML keeps a prediction's inputs bound
  to its execution stream and releases them, on a queue of its own, once the
  stream has been idle for about a second. coremltools backs them with the
  NumPy arrays, so a freed array is a use-after-free there (EXC_BAD_ACCESS on
  `MLE5ExecutionStream`'s reset queue). Any model and any compute unit,
  CPU_ONLY included (macOS 27.0, coremltools 9): `m.predict({"x":
  np.ones(...)})` followed by a second of other work, or of predictions on
  another model, crashes; a tight loop of predictions on one model never
  does, which makes it look intermittent. Keep every input referenced until
  the model is released. The gates' models do (`gates._Model`); a script
  that calls `predict()` directly must too.
- **CPU `reduce_max` over ≥ 256 elements returns max(x, 0)**, and
  `reduce_min` min(x, 0) (macOS 27). A small reduce can land on the CPU even
  under CPU_AND_NE. Use `reduce.blocked_max()` for any explicit max.
- **The ANE `linear` saturates above 2^15** (32,768 exact, 33,000 inf), and
  loses precision on small inputs (~3e-4 / rms). Its add, mul and layer_norm
  cover fp16's full range; fp16's own 65,504 isn't the limit that matters.
- **The ANE's native `silu` and `gelu` are coarse** (D17, D20, D39), and in a
  SiLU MLP whose residual is small the silu can dominate a model's whole ANE
  error: agent-jev's went from D to B by replacing it (D39). Replace them
  with `activations` forms and forbid the native ops. Measure in the model,
  layer by layer, not with a one-op program: Core ML doesn't place a lone
  `silu` on the ANE, so a micro-benchmark shows every form alike.
- **Multi-shape artifacts abort under `.cpuOnly`** on macOS 27 (D27). One
  static artifact per bucket, always.
- **A program whose weights pass about 1 GiB runs entirely off the ANE**,
  with no error and nothing in the log. On an M1 Max (macOS 27.0), a Qwen3
  backbone with 0.964 GiB of weights ran 1,526 of its 1,535 operations on
  the ANE, and with 1.022 GiB none; other chips or OS versions may draw the
  line elsewhere. A model served on the ANE whose compiled weights pass
  `plan.MAX_ANE_PROGRAM_WEIGHT_BYTES` (1 GiB) is refused after compiling,
  with its options:
  - chunk it (D37): a backbone that can be chunked (one with `chunk_ports`,
    `chunk_layers`, `chunk_parts` and `chunk_call`; Qwen3TreeBackbone does)
    is chunked by default whenever a bucket's fp16 weights pass
    `chunking.CHUNK_WEIGHT_BUDGET_BYTES` (0.9 GiB, a margin under the
    measured limit). See "Chunked buckets" below;
  - serve it on the GPU (`compute_units = "cpu_and_gpu"`);
  - convert with `--int8-embedding` (`Job(int8_embedding=True)`), which
    stores the token-embedding table, the largest constant a gather reads,
    in int8, linear symmetric with one scale per row. On agent-jev that
    halves a 151,936 x 1,024 table from 311 to 155 MB, which brings 1.198 GB
    of weights to 1.043 GB and the plan to 98.3% on the ANE. Its cost is
    within fp16's own (GPU max |Δp| 4.5e-4 with it, 4.8e-4 without, against
    an ideal-fp16 ceiling of 7.3e-4 on 8 cases). The lookup is a gather, off
    the ANE either way;
  - or pass `--ignore-ane-weight-cap` to try anyway. That turns the error
    into a warning, and the compute-plan gate still judges where Core ML
    places the program. A bypass is recorded in the run's output, in its
    report (`ane_weight_cap`), and as a comment in the installed manifest,
    so a bypassed artifact stays visible.
- **Chunked buckets** (D37, `chunking.py`). Each bucket is split in torch,
  before tracing, at layer boundaries, and each chunk is its own program:
  - `--chunks auto` (the default for a backbone that can be chunked) uses
    the fewest chunks whose fp16 weights fit the budget, balanced; `--chunks
    N` makes N; `--chunks 10,20` starts chunks at those layers; `--chunks 1`
    keeps one program per bucket. The cuts are planned at the largest bucket
    and used for all.
  - Only the residual stream crosses a boundary: every chunk but the last
    outputs `hidden_out`, fp16 [1, S, H], and every chunk but the first
    takes it as `hidden_in`. Each chunk takes the int32 ports its backbone
    reads and rebuilds the mask and positions from them; the first takes
    `input_ids`, the last the head's own ports, and the last produces the
    output.
  - Gates: the composed chunks against the unchunked wrapper in fp32, in
    every bucket; in the smallest bucket, the chain's GPU output against
    the unchunked program's, bit for bit (`--chunk-identity-all`: every
    bucket, converting each twice); each chunk against the weight limit;
    then the usual gates on the chain.
  - It installs `model_{seq}.{chunk}.mlmodelc`, rewrites the installed
    manifest's `artifact` to `"model_{seq}.{chunk}.mlmodelc"`, and adds
    `[chunking]` (`chunks`, and `weight_budget_bytes` for `auto`). A
    manifest whose artifact isn't `model_{seq}.mlmodelc` fails before
    converting. Installing either layout removes the other's artifacts.
  - `[placement]` records each bucket's counts as its chunks' sums, with
    each chunk's under `chunks`, and every conversion also records the plan
    for the other of `cpu_and_ne` and `cpu_and_gpu` under
    `[placement.alternatives.<units>]`.
  - Chunked models need sidekick 0.7.0.
- **Conversions keep no Core ML cache.** `cli.parse` re-runs the converter
  with `CFFIXED_USER_HOME` set to a temporary home, deleted when it exits:
  the conversion loads each bucket from a temporary directory and then
  moves it, so every cache entry it would leave behind is orphaned (about
  11 GB for agent-jev). The conversion also empties that cache as it goes,
  once each compute path's gate is done with a program
  (`core.purge_coreml_cache`): it never loads the same path again, and
  each compute path's entry holds about the program's weights again (a
  compute-plan read adds an entry of its own), so the cache otherwise
  peaked at 3.9 GB for F2LLM-v2-160M's three buckets and would not have fit
  a chunked 0.6B model on a nearly full disk. Set `CFFIXED_USER_HOME`
  yourself to keep a cache; it is then never emptied. Neither step frees
  what the gates' ANE compiles leave in the system's ANE cache (below):
  each one adds about the program's weights to the system volume, from a
  throwaway home as from any other.
- **An empty compute plan can come from a stale cache.** Core ML's cache of
  compiled bundles (`~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`)
  can hold a broken entry for an artifact's path; plans for that path then
  come back with every op unassigned, or fail with "internal failure", until
  the entry is gone. A copy at a new path (`cp -c -R`) reads normally.
- **Core ML's cache: what a load reuses.** Measured on an M1 Max, macOS
  27.0, from Core ML's own cache-lookup log lines and the ANE compiler's
  runs (docs/design/coreml-cache.md has the method and the data):
  - **The directory** is
    `~/Library/Caches/<name>/com.apple.e5rt.e5bundlecache/<macOS build>/`,
    where `<name>` is the process's name as invoked, or the bundle
    identifier of an Info.plist embedded in the binary. Setting
    `CFFIXED_USER_HOME` to a private directory gives a process a private
    cache.
  - **An entry** is keyed by the program's path, its parsed MIL, the
    identity of its `weight.bin` (inode and birth time, not its contents
    or modification time), the compute units and, on the ANE, the loading
    binary's code-signing identifier.
  - **So a load misses** for a copy of the program at another path (a
    copy, a hard-linked tree, an APFS clone, a rename), for new files at
    the same path (new inodes: measured with new weights on every path,
    and on the ANE with the original bytes copied back), under another
    executable name (a copy, symlink or hard link to the binary) and, on
    the ANE, under another code-signing identifier. It hits for a binary
    moved to another directory, or rebuilt, with the same name and
    identifier, and for a program directory moved away and back.
  - **A compute-plan read** (`MLComputePlan`, as `ane_check`, the
    converters' gates and the parity suite use) is its own entry: a full
    compile, and another copy of the weights.
  - **Rewriting `weight.bin` in place serves stale output on the GPU and
    the CPU.** New weights written into the same file (same inode and
    birth time) still hit there, and the program runs with the old
    compiled weights; the ANE recompiles. Replace a program's files with
    new ones (a new directory, then a move), as the converters do, never
    by writing into them.
- **Where compiled programs live.** On the GPU and the CPU, in that
  directory, about the program's weights per entry (a `.mpsgraphpackage`
  or `bnns_program.bnnsir`). An ANE entry there is about 12 kB; the
  program itself, about the weights' size per compile, goes to the
  system's ANE cache kept by `aned` on the system volume, which System
  Integrity Protection keeps users from deleting. A copied home or cache
  directory hits on the GPU and the CPU, but the ANE always recompiles.
- **The cache grows fast**, by tens of GB a day during conversion work.
  Deleting a cache frees its GPU and CPU programs only: ANE compiles
  accumulate in the system's ANE cache, from throwaway conversion homes
  too (an audit of this cache left about 5–8 GB there). Clear a cache only
  when no process is using it: rename it, then delete it. Directories
  named for an older macOS build are orphans, safe to delete.
- **The mean-pooling tail runs on the CPU** (reduce_sum, real_div, clip):
  one extra hand-off after the encoder, well within the compute-plan gate.
- **Core ML's diagnostics go to the unified log, not to stderr.** An
  error code such as -14 says little on its own; the reason is often in
  the log. On macOS 27, Core ML logs under `com.apple.coreml`,
  `com.apple.e5rt` and `com.apple.ane`; Hollemans' notes also name
  `com.apple.espresso`, from older releases. Use `/usr/bin/log`, because
  `log` alone is a zsh builtin:

  ```sh
  /usr/bin/log stream --info --debug --predicate \
    'subsystem IN {"com.apple.coreml","com.apple.e5rt","com.apple.ane","com.apple.espresso"}'
  ```

  ANE compile failures come from the system's `ANECompilerService` and
  `aned` processes, not from sidekickd or the converter, so filter by
  subsystem rather than by `--process`. Model and layer names appear as
  `<private>` unless private data logging is enabled with a logging
  configuration profile.
- **sidekick doesn't set `allowLowPrecisionAccumulationOnGPU`.** It
  leaves Core ML's default (off), under which the GPU accumulates in
  higher precision than fp16, according to the reference.
  Turning it on could change GPU grades, so it would need regrading.

**coremltools and transformers**
- **Shape arithmetic in traced code crashes coremltools 9** under static
  shapes ("only 0-dimensional arrays can be converted to Python scalars").
  Never compute with `x.size()` or `x.shape`; use Python ints and literal
  reshapes, including the pooled output's `reshape(1, dims)`. That includes
  `int(t.shape[-1])` on a tensor computed from an input (pass `seq` to
  `masks.self_attending`), and keeping leading dims by reading them: use
  `torch.flatten`/`torch.unflatten` on the trailing dims instead.
- **transformers' DeBERTa-v2 attention doesn't trace**, through its
  TorchScript helpers, and made traceable its relative-position gathers
  land on the CPU with their matmuls. `backbones.deberta_v2` writes the
  forward out with `relative_shift` and leaves transformers' own forward
  as the fp32 reference.
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
  It matters: embedding a ~400–480-token text truncated at the model's own
  length instead of at 512 gives cosine 0.973 for all-MiniLM-L6-v2 (256),
  0.896 for all-MiniLM-L12-v2 (128) and 0.681 for
  paraphrase-multilingual-MiniLM-L12-v2 (128).
- **Configs saved by transformers 5 can be misread by transformers 4.57.**
  ModernBERT checkpoints carry RoPE theta in a `rope_parameters` block,
  which 4.57 ignores, falling back to 160000 / 10000; mmBERT's are
  160000 / 160000. A reference built under the same misreading can't catch
  it. The ModernBERT backbone applies the block and checks every layer's
  rotary frequencies and attention type against config.json.
- **Gate units**: a reranker's raw logit is |x| ~ 10, so its error is gated
  in sigmoid space (cross-encoders train with a binary cross-entropy), with
  |Δlogit| reported.

## Quick triage: is a model worth converting?

The triage, the checklist and the gates below are for Core ML. The ANE's
weight cap, fp16 range and precision limits don't apply to onnx-cpu, which
computes in fp32 with dynamic shapes, so an embedder rejected only by those
may still serve there (Qwen3-Embedding-0.6B was one, until chunking (D37)
gave it an ANE build). Its gate is the ONNX export's own (fp32 parity
against the checkpoint, batch and pad invariance; "[ONNX export](#onnx-export)"), then the parity suite's `onnx-cpu` path.

Most rejections are visible long before a conversion. Cheapest first:

1. **`config.json` (seconds).**
   - `model_type` / `architectures` is the strongest signal. Validated:
     `bert`, `gemma3_text` (with a range rewrite), `lfm2`, `qwen3`,
     `modernbert` (with explicit attention). A checkpoint can wrap one of
     these, as laya's `encoder/config.json` does.
   - Also check:
     - **QK-norm** (`q_norm`/`k_norm` in the modeling code): keeps
       activations small (LFM2.5 ~25, F2LLM ~420), so no range rewrite.
     - **bf16 training**: nothing kept its activations in fp16 range.
     - **Size**: each bucket stores the whole model at ~2 bytes/param, and
       Core ML runs an ML program on the ANE only if its weights fit in about
       1 GiB: about 500M parameters in fp16, embedding table included.
       Measured on an M1 Max with macOS 27 (October 2026) by cutting
       agent-jev's Qwen3-0.6B at different depths: 0.964 GiB of weights put
       99.4% of operations on the ANE, and 1.022 GiB put 0%. Over the cap,
       `cpu_and_ne` silently runs the whole model on the CPU, with no error,
       so a compute plan of 0 ANE operations is the symptom. Splitting each
       bucket into programs at layer boundaries (`--chunks`, D37) keeps every
       program well under the cap at no cost in accuracy. Storing only the
       embedding table in int8 also brought agent-jev (1.12 GiB) to 0.97 GiB
       and 98% on the ANE, but with almost no headroom and a measurable cost
       in accuracy (agent-jev's notes compare the two). This is Core ML's
       documented limit: Apple's coremltools guide ("MLModel Utilities",
       Bisect Model) states it as 1 GB for the Neural Engine on iPhone, and
       its `ct.models.utils.bisect_model` splits a model into two chunks
       under it. It may differ on other chips and OS releases; measure it
       there. GPU serving has no such cap (GLiNER2.5-Decide and agent-jev).
     - **Needed sequence length**: buckets are ≤512, and the ANE's advantage
       shrinks with length.
     - **Output shape**: one vector per input fits `/v1/embeddings`, and a
       sequence-classification head (or laya's format) fits `/v1/classify`
       (D28). Rerankers and multi-vector models need a new API.
2. **The modeling code (minutes).** See the checklist below: data-dependent
   shapes, attention that can't run as SDPA, and non-attention token mixers.
3. **`tools/probe_activations.py` (minutes, PyTorch on the CPU, no Core
   ML).** It hooks every normalization layer's input and reports peak
   activation, which dimension carries it, how many times the median
   dimension it is, and which norm type reads it. Its one calibrated verdict
   is fp16 range: over 65504 needs a range rewrite (EmbeddingGemma), and
   under ~30,000 is safe to convert directly.

   | model | norms | peak (dim) | × median | probe says | measured on ANE |
   |---|---|---|---|---|---|
   | bge-small-en-v1.5 | LayerNorm | 338 (99) | 146× | no range issue | 0.99998 |
   | LFM2.5-Embedding-350M | RMSNorm | 1.9 | 17× | no range issue | 0.99999 after the precision rewrite |
   | embeddinggemma-300m | RMSNorm | 152,485 (731) | 313× | range rewrite | 0.99999 after range + MLP rewrites |
   | gte-modernbert-base | LayerNorm | 47,973 (251) | 502× | no range issue | 0.99998 (explicit attention + 2^15 range rewrite) |
   | laya (ModernBERT-large) | LayerNorm | 27,296 (379) | 556× | no range issue | CLS ≥ 0.997 (explicit attention) |

   The verdict is about the residual stream. A written-out RMSNorm squares
   its input, so any norm input past 256 overflows in fp16:
   EmbeddingGemma 2's peak at ~1,600 cost Core ML's CPU path 5% until the
   norms were pre-scaled.

   Massive activations, one dimension hundreds of times the rest on a few
   tokens, are common, and they are not by themselves an ANE problem. The
   ModernBERT failure once blamed on them was a Core ML attention bug. Expect
   those tokens' own output vectors to be slightly less accurate, which
   matters only when you read individual token vectors.
4. **Convert with explicit attention, then gate (hours):** the gates at the
   end of the checklist. Pad invariance in `ane_check` catches a dropped
   attention mask in seconds.

## Will a new model convert? A checklist

Read the model's `modeling_*.py` before anything else. The recipe survives:

- **Encoder-style, single-vector output** — bidirectional attention (or a
  published bidirectional patch), CLS or mean pooling; or a causal decoder
  with last-token pooling (F2LLM). Multi-vector, and rerank heads don't fit
  the API; generative models have no pooled vector at all.
- **Last-token pooling has two traps** — (1) select the last real token
  in-graph via the mask (`mask · (1 − shift_left(mask))`, masked sum), never
  a data-dependent gather index; (2) the embedding IS the trailing EOS
  token, so truncation must preserve it — the server keeps `[first max-1,
  last]` for exactly this reason. A short-input-only parity gate misses
  both; test an over-length input that fills the largest bucket.
- **Explicit attention with fp16-safe masks.** For Core ML, masks must use a
  finite constant such as -30000, not `-inf`/`finfo.min`, which NaN in fp16
  (D15). The converters patch the mask builders to do this. Then use
  attention that converts to explicit matmul → softmax → matmul (gemma,
  F2LLM, ModernBERT via `attn_implementation="eager"`). Passing `scale=` to
  `F.scaled_dot_product_attention` works too: coremltools then emits
  explicit ops instead of the fused one. Whatever the form, `ane_check`'s
  pad-invariance gate must pass.
- **Fused SDPA drops the mask on the ANE when the mask is an input of the
  ANE procedure that runs the attention.** That is, when no op inside that
  procedure computes the mask. It may be a model input, the output of an op
  on the CPU, or the output of an earlier ANE procedure. The output then
  equals unmasked attention exactly, for every mask shape, dtype and fill
  value tried and at both opsets (D25). The compute plan shows it:
  - `scaled_dot_product_attention` is on the NeuralEngine, and
  - its `attn_mask` comes from a model input or a CPU op, or from an ANE op
    with a CPU op between it and the attention.

  ModernBERT builds its masks before the CPU-only embedding gather, so the
  mask plumbing lands on the CPU. `python tools/repro_sdpa_mask.py --check
  <model>` applies the rule to a compiled model. A fused attention whose plan
  shows no device (bge-small) is correct only through a Core ML fallback that
  doesn't run it natively. At the iOS26 opset that fallback is gone and bge
  fails to load on CPU_AND_NE, so keep the macOS15 target for fused-attention
  models.
- **Fused SDPA on the CPU (`CPU_ONLY`) has two more defects**
  (reproduced by the same tool):
  - A query row whose keys are all masked returns NaN when
    |fill| × √head_dim > 65504, e.g. -30000 at head dim 64. Explicit
    attention returns a finite row.
  - q/k/v split straight off a packed projection with
    `.transpose(3, 1).unbind(2)` give wrong output. `.permute(2, 0, 3, 1, 4)`
    is correct.

  CPU_ONLY parity catches both.
- **Static-shape-friendly graph** (for Core ML; an ONNX export takes dynamic
  axes by design) — no data-dependent shapes. Stock
  `rotate_half`/`repeat_kv` and any `F.conv1d(padding=shape-derived)`
  need the traceable rewrites (D17 constraint 8, LFM2.5 constraint B).
- **fp16-safe activations** — calibrate first (forward hooks, max |activation|
  on a mixed corpus). Under ~30k: convert directly (bge, LFM2.5). Over:
  apply the D17 power-of-two range rewrite (gemma). For Core ML, watch for
  `-1e9` mask constants (rewrite at -30000) and rmsnorm eps below ~1e-4.
- **Keep every linear-layer input near rms 1.** The ANE's `linear` op has
  an absolute precision floor on its input, not a relative one: its
  relative error is ~3e-4 / rms(input) (0.04% at rms 1, 2% at 0.016, 10% at
  0.004; the GPU is flat at 0.04%). Scaling the weights doesn't help; only
  the input's magnitude matters. Measure the rms of every linear input in
  fp32 with forward pre-hooks. Likely offenders are MLP down projections
  after a gated product (act(gate)·up is a product of two small numbers),
  in-graph heads that pool at a reduced scale, and any model whose
  activations are tiny everywhere (LFM2.5: every output projection at rms
  0.003–0.07). A small activation range spares a range rewrite, but it
  isn't safe on the ANE. Fix with a power-of-two scale folded into the
  weights upstream. Cancel it with a scale-invariant norm or the final L2
  normalize (EmbeddingGemma, D17). Where only a residual add follows,
  cancel it with an explicit multiply rather than weights divided into
  fp16's subnormal range (LFM2.5, D19).
- **Keep every linear-layer output under 2^15 on the ANE.** The ANE's
  `linear` op saturates at 32,768, half of fp16's 65,504: an output of
  33,000 comes back inf. Its add, mul and layer_norm handle the full fp16
  range, so fp16 headroom alone doesn't protect a projection. Measure the
  largest output of every linear in fp32 with forward hooks. Keep it at or
  below 0.85 × 2^15 by running the residual stream at a power-of-two 1/K
  that the norms cancel (gte-modernbert: K = 2, D25). Massive activations
  are the usual cause: ModernBERT writes one at 35,000–51,500.
- **Don't use Core ML's native GELU or SiLU ops on the ANE.** Measured
  absolute error on [-1, 1]: gelu ~6e-3, silu ~1.5e-2, versus tanh 1.6e-3,
  sigmoid 3e-3, and exact mul/relu. Build GELU as
  `x * (1 + tanh(x * (c + c·0.044715·x²)))` with the factor 2 absorbed
  downstream, and SiLU as `x * (1 + tanh(x / 2))` (LFM2.5, F2LLM). Check the
  converted program for surviving `gelu`/`silu` ops: coremltools has passes
  that fuse such patterns, and `x * sigmoid(x)` converts straight back to
  the native silu.
- **Massive activations are not disqualifying.** A few feature dimensions
  reaching tens of thousands on delimiter/[SEP] tokens (ModernBERT ~48,000,
  EmbeddingGemma ~152,000) convert fine once attention is explicit and values
  fit fp16. Their cost is a small per-token effect on those tokens. An
  earlier version of this checklist called them an "ANE killer" under
  LayerNorm; that was the fused-attention mask bug misdiagnosed (D25).
- **Token mixing other than attention** (convs, SSMs): decide the padding
  semantics explicitly. Attention masks silence pad *keys*, but anything
  convolutional reads pad *states* — zero them per layer if the reference
  is the unpadded forward (LFM2.5 constraint D).
- **Each loaded bucket maps the whole model.** On disk, every bucket of a
  chunk shares one weights file (D49), so 350M params ≈ 700 MB installed
  for any number of buckets; in memory each loaded bucket still holds its
  own ≈ 700 MB, and Core ML's compiled cache keeps one per bucket and
  compute path. On the
  GPU and the CPU that copy is in the user's cache; an ANE compile puts it
  in the system's ANE cache instead, on the system volume, where deleting
  the user's cache doesn't reach it (docs/design/coreml-cache.md).

Gates to pass, in order: fp32 rewrite parity ≥ 0.9999 (only if rewriting),
`CPU_ONLY` ≥ 0.999, `CPU_AND_NE` ≥ 0.999, `ane_check` per bucket (compute
plan eligible and pad invariance on both paths; its ratio should be clearly
above 1.0 on a quiet machine). Then generate a reference with
`tools/parity_reference.py` and run the parity suite. It repeats those gates
through the product code on every path and adds the adversarial corpus. Its
grade is the one to publish, and `--suggest-floors` records it for the
chip. For an embedder's ONNX build: the export's fp32 gate, then the
suite's `onnx-cpu` path.

For Core ML, flexible input shapes are ruled out (D15), and on macOS 27 one
kind became dangerous. Under `.cpuOnly`, an artifact whose inputs have several
enumerated shapes (`ct.EnumeratedShapes`) aborts the process at its first
prediction ("E5RT: No memory object bound to port"). Under
`.cpuAndNeuralEngine` and `.all` it still falls back to the CPU: 85 ms per
bge-small embed at seq 128, where a static bucket takes 2.2 ms on the ANE.
The abort is an Objective-C exception Rust can't catch. It would take down
`sidekickd`, or the host app linking `libsidekick.dylib`.

So on macOS 27, `CoremlModel::load` refuses such a model, whatever the
compute units, with an error that names the input. Two cases only get a
warning: range-shaped inputs (`ct.RangeDim`), which run on the CPU and never
aborted, and any flexible layout on earlier macOS (D27). Always ship one
static-shape artifact per bucket, and run `ane_check` on each: it reads the
compute plan without predicting, so it rejects any flexible artifact without
running it.

Make every parity metric NaN-safe. `min(worst, cos)` returns the old value
when `cos` is NaN, which hid a NaN-producing CPU path behind "parity
1.000000" during the ModernBERT investigation; fail on non-finite output
instead, as the converters' parity checks do.

Three hard-won measurement gotchas: run residency checks on a quiet machine
(see [the method note](MODELS.md) — concurrent GPU load makes ratios swing 2x), and
treat `E5RT ... ANECCompile() FAILED` stderr lines as *possibly transient
service state*, not proof of a bad artifact — the same file measured 1.48x
with failures and 2.63x clean forty minutes apart. Re-measure before
re-converting. Likewise, a compute plan with *every* operation unassigned
("no operations are assigned to any compute device"), or "internal
failure", can be a broken entry in Core ML's compiled-bundle cache for that
path (`~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache`, D24).
An embeddinggemma artifact read that way repeatedly while predicting at ANE
speed, and the same file copied to another path read normally (2015/2024).
Those caches grow large: the parity suite's reached 40 GB and ane_check's
25 GB. They're safe to delete, but that frees only the GPU and CPU
programs. Each ANE compile also leaves about its program's weights in the
system's ANE cache, on the system volume, which users can't delete
(System Integrity Protection); those copies accumulate with every
recompile, from throwaway homes too (docs/design/coreml-cache.md).

## Tests

The logic that decides artifacts has torch-only unit tests (no Core ML), in
`tools/sidekick_convert/tests`: the calibration guard, the tokenizer rule,
masks, one-hots, the blocked max, the K choice, activations, pooling, the
manifest rules, the ideal-fp16 simulation, the gates' prediction inputs,
and the causal-LM converter (on a tiny random Llama: the rewrites against
transformers, the chunk contract called T tokens at a time, GPTQ, the
draws, the grades and the committed fixtures). The shared-weights tests also re-export two tiny programs built
with coremltools' MIL builder, and are skipped without coremltools and
Xcode's `coremlcompiler`. From the repository root:

    python -m pytest tools/sidekick_convert/tests

or, without pytest:

    python -m unittest discover -s tools/sidekick_convert/tests -t tools

CI runs the Rust workspace only; run these before changing the library.
