# Backends

What runs each kind of model, and the limits of each. A model's manifest
names its backend (`backend = "coreml"`, the default for an embedder,
`"onnx"` or `"static"`); classifiers and rerankers are Core ML models;
chat is Apple's Foundation Models.

| Backend | Serves | Where |
|---|---|---|
| [Core ML](#core-ml) | embedders, classifiers, rerankers | macOS: the Neural Engine, GPU or CPU on Apple silicon; the GPU or CPU on Intel |
| [ONNX Runtime](#onnx-runtime-on-the-cpu) | embedders | the CPU, on macOS and Linux |
| [Static](#the-static-tier) | embedders | the CPU, anywhere |
| [Foundation Models](#foundation-models) | chat | Apple silicon, macOS 26+, Apple Intelligence on |

Which platform has which backend is in
[INSTALL.md](INSTALL.md#platforms); each model's grade on each compute
path is in [MODELS.md](MODELS.md).

## Contents

- [Core ML](#core-ml)
  - [Compute units](#compute-units)
  - [The ANE weight cap](#the-ane-weight-cap)
  - [The CPU sequence cap](#the-cpu-sequence-cap)
  - [The ANE sequence cap](#the-ane-sequence-cap)
  - [Where Core ML places operations](#where-core-ml-places-operations)
- [ONNX Runtime on the CPU](#onnx-runtime-on-the-cpu)
  - [Threads](#threads)
- [The static tier](#the-static-tier)
- [Foundation Models](#foundation-models)

## Core ML

Each Core ML model is one static-shape compiled program (`.mlmodelc`) per
sequence-length bucket ([D15](DECISIONS.md#d15--per-bucket-static-artifacts-pooling-baked-into-the-model)),
or a chain of programs per bucket for a model too large for one
([D37](DECISIONS.md#d37--chunked-buckets-a-chain-of-programs-under-the-anes-weight-limit)).
An input runs in the smallest bucket that fits it. The first load of each
bucket compiles it for the device; what that costs, and how sidekickd
compiles ahead of requests, is in
[RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache).

### Compute units

A Core ML model loads with `.cpuAndNeuralEngine` unless its manifest asks
otherwise (D14): the ANE keeps background work off the GPU. Some models run
badly there. GLiNER2.5-Decide grades A on the GPU at about 34 ms per input,
but C on the ANE, where a prediction takes about 0.11, 0.34 and 1.28 s at
128, 256 and 512 tokens, because its DeBERTa relative-position rewrite is
slow on the ANE ([MODELS.md](MODELS.md)). Such a model
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
  names the program (a bucket, or one chunk of a chunked bucket, D37), its
  weights and the fixes: convert it in chunks under the limit (`--chunks
  auto`), serve it on the GPU (`compute_units = "cpu_and_gpu"`), convert a
  quantized variant (`--int8-embedding`), or load it anyway. When the
  daemon config's `[models."<id>"]` chose the ANE (D38), the reason says
  so and names removing that override first.
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
fixes it (D33 records the measurement; `tools/repro_cpu_matmul_accumulation.py`
reproduces it standalone).
Up to 1,024 tokens the CPU is bucket-invariant, and the GPU and the ANE
are unaffected by it (the ANE has a cap of its own, below).

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
- A fev model's state limit becomes `min(state_max_len, limit − 640)`:
  fev's own formula for a window, applied to the capped one (see "The fev
  format").
- The model's `/v1/models` entry gains `seq_cap` (`limit`,
  `manifest_max_seq_len` and `reason`, plus `state_max_len` when the cap
  lowered a fev model's), and `/health` lists every capped
  model under `seq_caps`.
- `all` isn't capped by it: Core ML chooses the device per operation, may not
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

### The ANE sequence cap

Long buckets compile slowly for the ANE, and past some length they fail.
Measured on Qwen3-Embedding-0.6B (M1 Max, macOS 27.0; D46): its 2,048-token
bucket compiled for the ANE in 151 s and graded A there, while its
4,096-token bucket took 29 minutes to compile and then failed 7 of 51
predictions ("Unable to compute the prediction using ML Program"). The GPU
served the same artifact at 4,096 tokens in under a second, graded A.

So a model served on the ANE runs no longer than its ANE cap (D48):
`ane_max_seq_len` from its manifest, else `DEFAULT_ANE_MAX_SEQ` (2,048
tokens):
- For `compute_units = "cpu_and_ne"` (the default) or `"all"` and a
  `max_seq_len` over the cap, the registry drops the buckets above it, as
  the CPU cap does: the largest bucket kept becomes the effective
  `max_seq_len`, longer inputs follow the model's over-length rules, the
  capped manifest is validated again, fev's state limit follows the
  capped window, and a model with no bucket within the cap is skipped with
  the fixes as the reason. The buckets past it are never loaded or
  compiled for the ANE.
- `all` is capped too, by reasoning rather than measurement (D46 measured
  `cpu_and_ne` only): Core ML compiles a bucket for the ANE whenever it
  places operations there, and the long compile is expected to come with
  it. The CPU cap reasons the other way, leaving `all` alone because Core
  ML may not use the CPU there at all. `cpu_and_gpu` isn't capped, and
  `cpu_only` has the CPU cap above. A model has at most one of the two
  caps.
- On a Mac without a Neural Engine (an Intel Mac), Core ML runs
  `cpu_and_ne` on the CPU, so it gets the CPU cap instead (1,024 tokens,
  `cpu_seq_limit` honoured, `ane_max_seq_len` ignored), and `all`, which
  runs on the CPU and GPU there, isn't capped.
- `/v1/models` shows it as `seq_cap`, whose `reason` names D48 and says
  whether the manifest set the cap; `/health` lists it under `seq_caps`,
  and the daemon logs it at startup.
- `ane_max_seq_len` (top level, either manifest file, coreml backend only,
  at least 1) is the cap measured for that artifact: lower for a model
  that fails sooner, or at least its `max_seq_len` for one measured to
  serve every bucket on the ANE.
- To serve a model's long inputs, serve it on the GPU (`compute_units =
  "cpu_and_gpu"` in its manifest, or `[models."<id>"] compute_units` in the
  daemon config, D38). For measurement, `sidekickd --ignore-ane-seq-cap`
  (`ignore_ane_seq_cap = true` in the config) lifts the cap for every
  model.
- The parity suite grades the ane path under the same cap: its ane worker
  loads only the buckets within it and runs only the cases that fit it,
  and the cases past it are listed as `capped` in the ane path's grade, not
  failed. For a model served on the ANE it reads no compute plan for a
  bucket past the cap, since that compiles the bucket for the ANE, and
  records the plan as not read. The cpu and gpu paths run every bucket;
  `--ignore-ane-seq-cap` grades the ane path past the cap.

### Where Core ML places operations

`compute_units` is a preference. Core ML decides per operation which
device runs it, and an operation the ANE can't run moves to the CPU or the
GPU without an error. The parity suite gates that before a model ships;
the daemon also reports it, per bucket, for the models it serves. Each
bucket is its own compiled program (D15), and `sidekick-buckets` names the
one that answered each input, so a response can be matched to its
program's placement.

A model's `/v1/models` entry lists, under `placement`, each bucket's
operation counts on the ANE, the GPU and the CPU, from one of two
sources:
- `conversion`, by default: the plan the converter read when it built the
  artifact, recorded in the manifest. It costs nothing at runtime. It is
  marked `stale` (`"measured on Apple M1 Max, macOS 25A354"`) when this
  machine's chip or macOS build differs, since another compiler can place
  operations differently. The converter records a plan for the units the
  manifest serves with and, as an alternative, for the other of
  `cpu_and_ne` and `cpu_and_gpu`; the one for the units the model runs on
  is reported, so a model the operator moves (D38) keeps its report. A
  plan recorded only for other units isn't reported.
- `live`, with `report_compute_plans = true`: the daemon reads the plan
  itself after a bucket's first load. A live plan takes precedence over a
  recorded one; while the read is pending, or if it fails, the recorded
  plan is reported.

`/health` adds, under `compute_plans`, each bucket's unassigned operations
(constants and other bookkeeping), the operators off the ANE, where a
recorded plan was measured, why a live read failed, and, for a chunked
bucket (D37), each chunk's recorded counts under `chunks` (the bucket's
are their sums).

`/health`'s `chunked_models` lists each chunked model by id: its
`chunks` per bucket, `ane_limit_bytes` (the ANE's per-program weight
limit), the `weight_budget_bytes` the converter split under (null when it
was given explicit cuts), and per bucket each program's `artifact` and
`weight_bytes`. `compute_unit_overrides` lists the compute units the
daemon config chose (`applied`, by model id) and overrides naming no Core
ML model (`unmatched`); `/v1/models` gives every Core ML model's
`compute_units_source`, `manifest` or `operator` (D38).

The manifest records the plan in an optional table, written by the
conversion library:

```toml
[placement]
compute_units = "cpu_and_ne"   # the units the plan was read for
chip = "Apple M1 Max"          # sysctl machdep.cpu.brand_string
macos_build = "25A354"         # sw_vers -buildVersion
date = "2026-10-01"            # when it was read (a string)

[placement.buckets.128]        # one table per bucket, by its length
ane = 294
gpu = 0
cpu = 10
unassigned = 410
total = 714                    # ane + gpu + cpu + unassigned
off_ane_ops = { gather = 2, cast = 8 }   # { } when every operation is on the ANE

[[placement.buckets.128.chunks]]   # a chunked bucket (D37): one per chunk, in order,
ane = 150                          # whose counts sum to the bucket's
gpu = 0
cpu = 6
unassigned = 200
total = 356
off_ane_ops = { gather = 2, cast = 4 }

# ... the next chunk ...

[placement.alternatives.cpu_and_gpu.buckets.128]   # the plan for other units (D38)
ane = 0
gpu = 704
cpu = 0
unassigned = 410
total = 1114
off_ane_ops = { ... }
```

A bucket missing from the table is reported without a placement (a
conversion of some buckets only records those). A record that doesn't fit
the model, with a key that isn't one of its buckets, counts that don't add
up to `total`, chunks that aren't the model's number or don't sum to their
bucket's counts, or on a static embedder, is ignored with a warning in
the daemon's log: it only describes the model, so it never takes the
model offline. A bad alternative is dropped on its own, keeping the main
plan. Daemons up to 0.5 ignore the table, and daemons before 0.7 ignore
`chunks` and `alternatives`.

Live reads are off by default because a plan read is a second compile.
Measured on an M1 Max under macOS 27.0, Core ML doesn't share a plan's
compile with the model's load, in either order:

| Bucket | Load (cold) | Plan read (cold) | Plan read (warm) |
|---|---|---|---|
| nlptown-sentiment, 128 tokens, ANE | 2.9 s | 3.1 s | 0.07 s |
| laya-typed-decisions, 256, ANE | 20.6 s | 20.6 s | |
| laya-typed-decisions, 1,024, ANE | 46.8 s | 49.5 s | 0.7 s |
| Lumma-fev-0.1b, 2,048, ANE | 218 s | 222 s | 1.2 s |
| nlptown-sentiment, 512, GPU | 1.0 s | 1.1 s | 0.17 s |

The plan's compile also gets its own entry in Core ML's cache, about the
bucket's weight size (0.9 GB for laya-typed-decisions' 256-token bucket),
and the read held up to 0.4 GB of footprint (Lumma-fev's 2,048-token
bucket) after it returned. With live reads on:
- a bucket's plan is read after its first load, never for a bucket no
  request has used;
- one background thread reads every plan, one at a time;
- results are cached under `~/Library/Caches/sidekick/compute-plans`,
  keyed by the artifact (its program's content, its weights' sizes and
  modification times), the compute units and the macOS build, so a
  restart doesn't read them again.

Compatibility: sidekick 0.4 and earlier ignore unknown keys in both
manifest files. A 0.4 daemon given a manifest with `compute_units` loads it
with `.cpuAndNeuralEngine` and reports `cpu_and_ne`; it doesn't fail. (A
gliner2-format manifest is skipped by 0.4 anyway, since it predates that
format.)

## ONNX Runtime on the CPU

An embedder may instead be an ONNX graph run by
ONNX Runtime on the CPU (`backend = "onnx"`, D40), on any platform sidekick
builds for. Whether it's sidekick's own export (`--format onnx`) or a
published one, the graph outputs token-level `last_hidden_state`, which
sidekick pools (`mean`, `cls` or `last_token`); a head that does more than
pooling, such as EmbeddingGemma's, stays in the graph, and its manifest says
`pooling = "none"`. An ONNX build keeps the model's id, so clients address
the same model everywhere. The daemon needs the `onnx` cargo feature and
ONNX Runtime: which version ([INSTALL.md](INSTALL.md#onnx-runtime)) and
where the daemon finds it ([CONFIG.md](CONFIG.md#where-onnx-runtime-is-loaded-from)).
Each manifest's graph is checked against its `[io]` when the directory is
scanned. `/health`'s `onnx` section reports the runtime and each model's
threads.

An ONNX model's vectors come from a different runtime than its Core ML
build's. Both are graded against the same fp32 reference, but don't mix
vectors stored from the two in one index without checking their agreement
in [MODELS.md](MODELS.md#onnx-runtime-on-the-cpu-onnx-cpu). How far ONNX
vectors are reproducible: at a given thread count, re-runs are
bit-identical, and so were two Linux x86-64 machines with different CPUs on
the same runtime and thread count. Across runtime versions, architectures
or thread counts, vectors agree to within rounding but aren't guaranteed
bit-identical (EmbeddingGemma moved by up to 1.43e-7 per component between
4 and 8 threads). Padding and batching would move them by up to 2.6e-7,
which is why sidekick runs each input alone. On an Intel Mac with
AVX-512, ONNX Runtime runs its AVX-512 kernels (D42);
`SIDEKICK_ONNX_AVX512=0` in the daemon's environment turns them off.

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

### Threads

An onnx model runs on the machine's fast physical cores, at most 8 unless
its manifest's `[onnx] default_intra_op_threads` raises that limit. A value
under 8 has no effect. The config's `onnx_intra_op_threads`, or a model's
`intra_op_threads` under `[models."<id>"]`, sets threads outright.

## The static tier

A static embedder (`backend = "static"`) is a model2vec-style table of
token embeddings: a safetensors file holding one `embeddings` tensor of
shape `[vocab, dims]` (f32 or f16), and a `tokenizer.json`. Embedding a
text tokenizes it, averages its tokens' rows and normalizes the result:
no context and no attention, so it runs on any CPU in microseconds, Intel
Macs and Linux included, at a quality well below a contextual encoder.
Export one from a model2vec distillation (for example
`minishlab/potion-base-8M`); [examples/manifests/static-floor](../examples/manifests/static-floor/manifest.toml)
is an annotated manifest. Every build serves static models, and
`libsidekick.dylib` too.

## Foundation Models

`/v1/chat/completions` runs Apple's on-device Foundation Models: an Apple
silicon Mac on macOS 26 or later with Apple Intelligence turned on. Intel
Macs and Linux, and a Mac where Apple Intelligence is off or still
downloading its model, answer it with a 503 and the reason, as `/health`
does ([D3](DECISIONS.md#d3--chat-503s-when-foundation-models-is-unavailable-no-generation-fallback-tier)).
Foundation Models needs a logged-in user session, which is why sidekickd
runs as a LaunchAgent rather than a LaunchDaemon
([INSTALL.md](INSTALL.md#running-as-a-service)).

A build with Xcode 27's SDK adds the macOS 27 features (real token usage,
the model's variant and context size, typed errors; D21) and still runs on
macOS 26 ([DEVELOPMENT.md](DEVELOPMENT.md)).

A conversation's follow-ups reuse its Foundation Models session for
`session_ttl_secs` (default 300) after its last request
([CONFIG.md](CONFIG.md#the-config-file)). The request and response
shapes are in [API.md](API.md#chat).
