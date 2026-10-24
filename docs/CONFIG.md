# Configuring sidekickd

Every setting `sidekickd` reads: the config file's keys, the command-line
flags and the environment variables, with their defaults on each OS. Every
setting is optional; a daemon started with none serves the default models
directory on `127.0.0.1:8790`.

A test (`config_md_documents_every_setting`, in
`crates/sidekick-server/src/main.rs`) checks that every config key, every
flag and every environment variable the daemon reads appears here.

## Contents

- [Where things live](#where-things-live)
- [The config file](#the-config-file)
- [Per-model settings](#per-model-settings)
- [Command-line flags](#command-line-flags)
- [Environment variables](#environment-variables)
- [Where ONNX Runtime is loaded from](#where-onnx-runtime-is-loaded-from)

## Where things live

| What | macOS | Linux |
|---|---|---|
| Config file | `~/Library/Application Support/sidekick/config.toml` | `$XDG_CONFIG_HOME/sidekick/config.toml` (`~/.config/sidekick/config.toml`) |
| Models directory | `~/Library/Application Support/sidekick/models` | `$XDG_DATA_HOME/sidekick/models` (`~/.local/share/sidekick/models`) |
| Compute-plan cache (`report_compute_plans`) | `~/Library/Caches/sidekick/compute-plans` | not used (no Core ML) |
| Core ML's compile cache | `~/Library/Caches/sidekickd/` and the system's ANE cache ([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)) | not used |
| Hugging Face cache (`sidekickd fetch`) | `~/.cache/huggingface/hub`, or as the environment below says | the same |

`--config` names another config file, and `--models-dir` (or `models_dir`
in the file) another models directory. sidekickd's startup log names the
config file it read, or says it used the defaults.

## The config file

TOML. Every key is optional; the values below are the defaults. A key the
daemon doesn't know is an error, so a typo stops the daemon rather than
being ignored.

```toml
addr = "127.0.0.1:8790"        # loopback only by default
# models_dir = "..."           # default: see "Where things live"
# api_key = "..."              # require Authorization: Bearer <key> on every route but /health
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
```

| Key | Default | What it does |
|---|---|---|
| `addr` | `127.0.0.1:8790` | The listen address. Loopback by default: the daemon fronts on-device models and has no business on the network unless you say so ([D13](DECISIONS.md#d13--default-bind-1270018790-health-unauthenticated)). |
| `models_dir` | per OS, above | The directory scanned for models ([RUNNING.md](RUNNING.md#the-models-directory)). |
| `api_key` | none | When set, every route except `/health` needs `Authorization: Bearer <key>`; a missing or wrong key is a 401 ([API.md](API.md#authentication)). |
| `session_ttl_secs` | 300 | How long a Foundation Models session is kept for a conversation's follow-ups. |
| `model_idle_ttl_secs` | 900 | How long a loaded embedder or classifier stays resident after its last use ([RUNNING.md](RUNNING.md#memory)). |
| `request_timeout_secs` | 60 | Bounds a request's work: a generation, or the predictions after any load the request waited on. A request past it is a 504 with code `timeout`. |
| `load_timeout_secs` | 900 | How long a request waits for its model and buckets to load, compiles included. A load past it is a 504 with code `load_timeout` and continues in the background, so a retry finds it ready. |
| `ignore_ane_weight_cap` | false | Load models served on the ANE even past Core ML's 1 GiB weight limit ([BACKENDS.md](BACKENDS.md#the-ane-weight-cap)). For experimentation. |
| `ignore_cpu_seq_cap` | false | Serve `cpu_only` models past 1,024 tokens, or past their manifest's lower `cpu_max_seq_len` ([BACKENDS.md](BACKENDS.md#the-cpu-sequence-cap)). |
| `ignore_ane_seq_cap` | false | Serve ANE models past their ANE cap ([BACKENDS.md](BACKENDS.md#the-ane-sequence-cap)). For measurement. |
| `report_compute_plans` | false | Read where Core ML places each loaded bucket's operations on this machine, once per bucket, in the background ([BACKENDS.md](BACKENDS.md#where-core-ml-places-operations)). Each read is a second compile. |
| `warm_on_start` | true | Compile every Core ML bucket the daemon serves at startup, in the background ([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)). |
| `warm_budget_secs` | 1800 | Start no startup warm-up load after this many seconds; 0 for no limit. |
| `onnx_intra_op_threads` | the machine's fast physical cores, at most 8 | Inference threads for every onnx model ([BACKENDS.md](BACKENDS.md#threads)). At least 1. |
| `onnx_runtime` | none | The ONNX Runtime library to load ([below](#where-onnx-runtime-is-loaded-from)). |
| `models` | none | Per-model settings, below. |

## Per-model settings

A `[models."<id>"]` table, by model id, overrides what a model's manifest
says, for one machine:

```toml
# Serve a Core ML model on other compute units than its manifest names:
# keep a companion model off the GPU a larger model uses, or serve a long
# model's longest buckets on the GPU.
[models."agent-jev"]
compute_units = "cpu_and_ne"   # cpu_and_ne, cpu_and_gpu, cpu_only or all

[models."bge-small-en-v1.5"]
intra_op_threads = 2           # an onnx model's inference threads

[models."qwen3-embedding-0.6b"]
warm = false                   # leave a Core ML model out of startup and fetch warm-ups
```

| Key | What it does |
|---|---|
| `compute_units` | A Core ML model's compute units ([D38](DECISIONS.md#d38--the-operator-may-choose-a-models-compute-units)). The ANE weight limit and the CPU and ANE length caps judge it as they would the manifest's choice, and `/v1/models` and `/health` report that the operator chose ([BACKENDS.md](BACKENDS.md#compute-units)). |
| `intra_op_threads` | An onnx model's inference threads, over `onnx_intra_op_threads`. At least 1. |
| `warm` | `false` leaves a Core ML model out of the startup warm-up and fetch's; `sidekickd warm <id>` still warms it. |

## Command-line flags

A flag overrides the config file.

**Serving** (`sidekickd` with no command):

| Flag | Overrides | |
|---|---|---|
| `--config <file>` | | The config file to read, instead of the default path. Also taken by `fetch` and `warm`. |
| `--models-dir <dir>` | `models_dir` | Also taken by `fetch` and `warm`. |
| `--addr <addr>` | `addr` | |
| `--api-key <key>` | `api_key` | Or `SIDEKICK_API_KEY`. |
| `--request-timeout-secs <secs>` | `request_timeout_secs` | Or `SIDEKICK_TIMEOUT_SECS`. |
| `--ignore-ane-weight-cap` | `ignore_ane_weight_cap` | |
| `--ignore-cpu-seq-cap` | `ignore_cpu_seq_cap` | |
| `--ignore-ane-seq-cap` | `ignore_ane_seq_cap` | |
| `--no-warm` | `warm_on_start` | |
| `--warm-budget-secs <secs>` | `warm_budget_secs` | |
| `--version` | | The version, the macOS SDK the Foundation Models shim was built with, the ONNX Runtime it expects and the backends compiled in. |

**`sidekickd fetch <model>`** downloads a published build and installs it
([RUNNING.md](RUNNING.md#fetching-a-published-build)):

| Flag | |
|---|---|
| `--backend coreml\|onnx` | The backend whose build to fetch. Default: `coreml` on macOS, `onnx` elsewhere. |
| `--variant <name>` | The build to fetch (`fp16`, `fp16-max2048`): its directory in the repository. Default: the one most recently published, or the only one. |
| `--buckets <list>` | Only these sequence-length buckets of a Core ML build, comma-separated. Default: all of them. |
| `--accept-terms <model>` | Accept the model's terms (its repository's TERMS.md), naming the model. |
| `--repo <owner/name>` | Another Hugging Face repository. Default: where the checksum ledger records the build as published. |
| `--revision <rev>` | A branch, tag or commit. Default: the published revision the ledger records, else `main`. |
| `--no-hf` | Download with sidekickd's own downloader even when the `hf` CLI is installed. |
| `--ledger <file>` | Verify against this checksum ledger instead of the one built into sidekickd. |
| `--force` | Replace a model directory that `sidekickd fetch` didn't install. |
| `--no-warm` | Don't compile the installed Core ML buckets. |

**`sidekickd warm [<model>…]`** compiles Core ML buckets ahead of requests
([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)). With no
model, every Core ML model the daemon serves but those with `warm = false`:

| Flag | |
|---|---|
| `--buckets <list>` | Only these sequence-length buckets, comma-separated. |
| `--budget-secs <secs>` | Start no load after this many seconds. Default: no limit. |
| `--list` | List each bucket's state without loading anything. |

The serving flags that change what the daemon serves (the caps above)
apply to `fetch`'s and `warm`'s compiles too, written before the command:
`sidekickd --ignore-ane-seq-cap warm`.

## Environment variables

**The daemon:**

| Variable | |
|---|---|
| `SIDEKICK_API_KEY` | `--api-key`. |
| `SIDEKICK_TIMEOUT_SECS` | `--request-timeout-secs`. |
| `ORT_DYLIB_PATH` | The ONNX Runtime library, after the config's `onnx_runtime` ([below](#where-onnx-runtime-is-loaded-from)). |
| `SIDEKICK_ONNX_AVX512` | `0` turns off ONNX Runtime's AVX-512 kernels on an Intel Mac whose CPU has them ([D42](DECISIONS.md#d42--avx-512-for-onnx-runtime-on-intel-macs)). |
| `RUST_LOG` | The log filter, in `tracing`'s syntax. Default: `sidekick=info,sidekickd=info,sidekick_server=info`. |

launchd doesn't pass a shell's environment to a LaunchAgent: set these in
the agent's plist (`EnvironmentVariables`), or use the config file.

**`sidekickd fetch`** reads the Hugging Face cache's location and settings
as `huggingface_hub` does, so it shares the cache with `hf` and Python:

| Variable | |
|---|---|
| `HF_HUB_CACHE` (or the older `HUGGINGFACE_HUB_CACHE`) | The cache directory. |
| `HF_HOME` | The cache is `$HF_HOME/hub`, when `HF_HUB_CACHE` isn't set. |
| `XDG_CACHE_HOME` | The cache is `$XDG_CACHE_HOME/huggingface/hub`, when neither is set. Else `~/.cache/huggingface/hub`. |
| `HF_HUB_OFFLINE` (or `TRANSFORMERS_OFFLINE`) | `1` uses the cache only and never contacts the Hub. |
| `HF_ENDPOINT` | The Hub's address, for a mirror. |
| `PATH` | Where fetch looks for the `hf` CLI, which brings its own token and settings (`HF_TOKEN`, Xet transfers). |

**Building from source:**

| Variable | |
|---|---|
| `SIDEKICK_FM_STUB` | `1` builds the Foundation Models stub instead of the Swift shim ([DEVELOPMENT.md](DEVELOPMENT.md#the-foundation-models-shim-and-the-sdk)). |

## Where ONNX Runtime is loaded from

The daemon loads ONNX Runtime at startup, and only when the models
directory has an onnx model ([D40](DECISIONS.md#d40--an-onnx-runtime-cpu-backend-for-embeddings)).
It takes the first of:

1. `onnx_runtime` in the config file;
2. `ORT_DYLIB_PATH` in the environment;
3. `libonnxruntime.dylib` (macOS) or `libonnxruntime.so` (Linux) beside
   `sidekickd`, as the release archives ship it.

It never searches system paths. If none is there, or the library doesn't
load, the onnx models are skipped with the reason and everything else
serves; `/health`'s `onnx.runtime` says which library loaded, or why none
did. A runtime of another minor version than the one sidekick is tested
with ([INSTALL.md](INSTALL.md#onnx-runtime)) loads with a warning.
