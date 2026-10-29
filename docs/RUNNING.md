# Running sidekickd

What a running daemon needs and does: the models directory, installing
models, first loads and Core ML's compile cache, memory and disk, logs, and
what to check when something is missing. Installing sidekickd itself is in
[INSTALL.md](INSTALL.md), and every setting in [CONFIG.md](CONFIG.md).

## Contents

- [The models directory](#the-models-directory)
- [Installing models](#installing-models)
  - [Which models are published](#which-models-are-published)
  - [Fetching a published build](#fetching-a-published-build)
  - [Installing a published build by hand](#installing-a-published-build-by-hand)
  - [Converting a model yourself](#converting-a-model-yourself)
  - [Restart after installing](#restart-after-installing)
- [First loads and the compile cache](#first-loads-and-the-compile-cache)
- [Memory](#memory)
- [Disk](#disk)
- [Logs](#logs)
- [Troubleshooting](#troubleshooting)

## The models directory

The daemon serves the models in one directory: by default
`~/Library/Application Support/sidekick/models` on macOS and
`~/.local/share/sidekick/models` on Linux, or `models_dir` in the config
([CONFIG.md](CONFIG.md#where-things-live)).

Each embedding model is a directory with a `manifest.toml`, each
classifier a directory with a `classifier.toml`, and each open chat model
a directory with a `chat.toml` (its decoder's chunk programs, embedding
and RoPE tables, tokenizer and chat template; [D53](DECISIONS.md#d53--open-llms-on-the-chat-route-a-stateful-ane-decoder-beside-foundation-models)):

```
~/Library/Application Support/sidekick/models/
  embeddinggemma-300m/
    manifest.toml          # artifact = "model_{seq}.mlmodelc", buckets = [128, 256, 512]
    model_128.mlmodelc/    # one compiled program per sequence-length bucket
    model_256.mlmodelc/
    model_512.mlmodelc/
    tokenizer.json
  nlptown-sentiment/
    classifier.toml
    model_128.mlmodelc/
    ...
```

See [examples/manifests/](../examples/manifests/) for annotated manifests
(EmbeddingGemma-300m on ANE, bge-small, a static floor model). Manifest rules
that matter: Core ML models must declare enumerated sequence-length `buckets`
(fixed shapes are what keep the model on the ANE), and `matryoshka` declares
which `dimensions` values the OpenAI API may request. A Core ML model loads
for the ANE (`.cpuAndNeuralEngine`; on an Intel Mac, which has none, Core ML
runs it on the CPU) unless its manifest names other
`compute_units` (`cpu_and_gpu`, `cpu_only` or `all`), for a model the ANE
runs badly; `/v1/models` lists each model's
([BACKENDS.md](BACKENDS.md#compute-units)). A manifest that doesn't
parse or validate is skipped with a warning, and listed with the reason under
`skipped_models` in `/health`; the rest of the directory still loads.

Classifier and reranker manifests are in
[examples/classifiers/](../examples/classifiers/), one per model. Each
converter writes the artifacts and installs the manifest.
[MODELS.md](MODELS.md) has every model's measured accuracy on each compute
path.

## Installing models

A model is a directory in the models directory, named for the model's id.
There are three ways to get one: fetch a build sidekick publishes, download
one by hand, or convert the model yourself. Whichever you choose, restart
the daemon afterwards.

### Which models are published

`sidekickd fetch` installs only builds that sidekick has published on
Hugging Face and recorded in its checksum ledger. Not every model is: a
model's licence tier decides whether it may be published
([PUBLISHING.md](PUBLISHING.md#what-is-published-licence-tiers)), and
builds are published one release at a time. Each backend's build of a
model is published on its own, in its own repository
(`sidekickd/<model>-coreml`, `sidekickd/<model>-onnx`), so a model can
have a Core ML build to fetch, an ONNX build, or both; `--backend` picks
one (the default is Core ML on macOS and ONNX on Linux). The list of
published builds, each with its backend, is generated from the ledger in
[MODELS.md](MODELS.md#published-builds). A model or backend not listed
there is installed by converting it ([CONVERTING.md](CONVERTING.md)).

### Fetching a published build

`sidekickd fetch` downloads a build
sidekick publishes on Hugging Face ([PUBLISHING.md](PUBLISHING.md)),
verifies every file, and installs it in the models directory ([D51](DECISIONS.md#d51--sidekickd-fetch-the-hugging-face-cache-verified-installed-as-hard-links)). It
fetches the builds in [the published list](MODELS.md#published-builds):

```sh
sidekickd fetch bge-small-en-v1.5                     # its Core ML build on macOS; its ONNX build on Linux
sidekickd fetch bge-small-en-v1.5 --buckets 128,256   # only the buckets for inputs up to 256 tokens
sidekickd fetch bge-small-en-v1.5 --backend onnx      # its ONNX build, on any OS (an Intel Mac included)
sidekickd fetch <model> --accept-terms <model>        # a gated build (the Gemma models): accept its terms
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
  the other whole. Not symlinks, as a Hugging Face snapshot's are: a
  program whose files link out of its directory can run off the ANE
  without an error
  ([design/coreml-cache.md](design/coreml-cache.md#symlinks-and-the-fallback-loader)).
  Where the cache is on another volume, the files are copied.
- **`--buckets` fetches part of a Core ML build,** and writes the manifest
  for exactly those buckets; `--variant` picks a build (`fp16`,
  `fp16-max2048`) when a repository has several. `fetch.toml` in the model
  directory records the repository, the commit and every file's sha256.
- **A Core ML build is compiled once installed:** fetch loads each bucket
  it installed, on the compute units the daemon will serve it on, so that
  the first requests don't wait on Core ML's compile ([D52](DECISIONS.md#d52--warm-core-mls-compile-cache-before-requests-need-it)). `--no-warm`
  skips it, and so does `warm = false` under the model's
  `[models."<id>"]` in the config.
- **An open chat model's build comes whole:** its decoder's chunk
  programs, embedding and RoPE tables, tokenizer, template and
  `chat.toml`; it has no buckets, so `--buckets` doesn't apply. fetch
  warms its decoder like any Core ML build, and that compile leaves a
  Neural Engine cache about the size of its weights, which you can't
  delete ([BACKENDS.md](BACKENDS.md#open-chat-models-on-the-neural-engine)).

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
  symlink out of its program's directory, which sidekick refuses to serve
  on the ANE (see the fetch bullets above). `--local-dir` writes real
  files.
- **Shared weights arrive as copies.** A build whose buckets share one
  weights file ([D49](DECISIONS.md#d49--one-weights-file-per-chunk-shared-by-every-bucket)) is downloaded this way as one copy per bucket, so it
  takes more disk than fetch installs. `python3 -m
  sidekick_convert.shared_weights link <model dir>`, run from sidekick's
  `tools/`, turns the copies back into hard links.
- **A gated repository** (a model whose card asks you to accept terms)
  needs `hf auth login` with an account that accepted them on the Hub.
- **On Linux,** or with another `models_dir` in the config, move the
  directory there instead.

### Converting a model yourself

Every model in [MODELS.md](MODELS.md), published or not, can be converted
from its original checkpoint with the converter MODELS.md names. The
converters write the compiled programs and install the manifest;
[CONVERTING.md](CONVERTING.md) has the requirements and the gates.

### Restart after installing

The daemon reads the models directory once, when it starts. A model
installed, replaced or removed while it runs takes effect at its next
start, as `sidekickd fetch` says when it finishes ("sidekickd serves it
from its next start"). Restart it: stop and start it in a terminal, or,
under launchd, `launchctl kickstart -k gui/$(id -u)/dev.sidekick.sidekickd`
([INSTALL.md](INSTALL.md#running-as-a-service)). Then check that the model
is listed in `/v1/models`, or, if it isn't, why in `/health`'s
`skipped_models`.

## First loads and the compile cache

The first time sidekickd loads a Core ML bucket, Core ML compiles it for
the device: seconds for a small model on the ANE, minutes for a long
bucket of a large one. Later loads find the result in Core ML's cache and
take seconds or less. The measurements behind this section are in
[design/coreml-cache.md](design/coreml-cache.md).

**Where the compiled programs are kept.** About the weights' size per
bucket and compute path (3.5 GB for laya-typed-decisions):
- **GPU and CPU:** in sidekickd's own cache,
  `~/Library/Caches/sidekickd/com.apple.e5rt.e5bundlecache/<macOS build>/`.
  Deleting it is safe; the buckets compile again on their next load.
- **ANE:** in the system's ANE cache, kept by Apple's `aned` service on
  the system volume. A user can't delete it (System Integrity Protection
  protects it), and deleting `~/Library/Caches/sidekickd` doesn't free
  it: that holds only about 12 kB per ANE bucket. Each recompile adds
  another copy.

**What makes Core ML compile again.** A load reuses a compiled program
only for the same model files at the same path, the same compute units,
an executable of the same name run by the same user and, on the ANE, the
same code-signing identifier. So:
- a copy of a model at another path compiles again, and so does a model
  reinstalled with new files;
- another executable name compiles again, so keep the binary's name when
  you move or upgrade it;
- on the ANE, another signing identifier compiles again, which is why
  release archives sign sidekickd with a fixed one
  ([INSTALL.md](INSTALL.md#signing));
- a macOS update is expected to compile everything again: the cache
  directory is named for the macOS build, and Apple's Core AI
  documentation says an OS update invalidates specialized models. That is
  inferred, not yet measured across an update
  ([design/coreml-cache.md](design/coreml-cache.md#os-updates)).

Replace a model's files with new ones (fetch, a converter); never rewrite
a `weight.bin` in place. On the GPU and the CPU, Core ML's cache then
still hits and serves the old weights' output
([design/coreml-cache.md](design/coreml-cache.md#the-cache-key)).

**Timeouts.** A request that has to wait for a compile doesn't count it
against `request_timeout_secs` (default 60): its deadline starts once its
buckets are loaded. Loading is bounded instead by `load_timeout_secs`
(default 900, well past the four minutes measured); a load that outlasts
it is a 504 with code `load_timeout`, the load continues, and a retry
finds the bucket ready.

**Compiling ahead of requests.** sidekickd compiles ahead
of requests ([D52](DECISIONS.md#d52--warm-core-mls-compile-cache-before-requests-need-it)):

- **At startup, in the background,** in a `sidekickd warm` child process,
  every bucket the daemon serves, on the compute units it serves it on,
  cold ones first. After a model move, a fresh install or a macOS update,
  that's where the compiles happen, instead of in the first requests. A
  request for a bucket being compiled waits for that compile and then
  loads from the cache, and requests for cached buckets don't wait. The
  machine runs one ANE compile at a time, so a request whose bucket needs
  a compile of its own waits for the one in progress. Nothing
  stays loaded that no request asked for. `--no-warm` or `warm_on_start =
  false` turns it off; `warm_budget_secs` (default 30 minutes) stops it
  starting new loads after that long.
- **On demand:** `sidekickd warm [<model>…] [--buckets 128,512]
  [--budget-secs N]`, or `--list` to see each bucket's state without
  loading anything. Run it as the user, and with the `sidekickd` binary,
  that the daemon runs as: Core ML's cache is per user, executable name
  and, on the ANE, code-signing identifier. Warming only helps a build
  signed the way the daemon's is: release archives are, and a build from
  source is once `tools/sign_sidekickd.sh` has run on it
  ([INSTALL.md](INSTALL.md#signing)). Without that, every rebuild gets a new identifier and
  recompiles its ANE buckets, which the startup warm-up then does in the
  background.
- **After `sidekickd fetch`,** for the buckets fetched; `--no-warm` skips
  it.

`/v1/models` lists each Core ML model's buckets as `compile_cache`
(`warm`, `warming`, `cold` or `failed`), and `/health`'s `warmup` shows
the startup warm-up's progress and each bucket's load times. Measured on
an M1 Max, macOS 27.0: bge-small-en-v1.5's first request per bucket on the
ANE took 4.0–5.4 s cold and 0.9–1.3 s warmed.

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
  the small ones.
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
  Lumma-fev's 2,048-token bucket
  ([First loads](#first-loads-and-the-compile-cache)).
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
  `onnx-cpu` section of [MODELS.md](MODELS.md#memory).

## Disk

- **A model's files.** Each Core ML bucket is its own compiled program.
  A build converted with shared weights, the converters' default
  ([D49](DECISIONS.md#d49--one-weights-file-per-chunk-shared-by-every-bucket)),
  keeps one weights file for all its buckets, so it takes about its
  weights' size on disk however many buckets it has; an older build keeps
  one per bucket. A model installed by `sidekickd fetch` is hard links into the
  Hugging Face cache, so the two together take that space once.
- **Core ML's compile cache.** Each bucket compiled on a compute path adds
  about its weights' size again, in a cache that is safe to delete for the
  GPU and the CPU, and that users can't delete for the ANE
  ([First loads](#first-loads-and-the-compile-cache)).

## Logs

sidekickd logs to standard output: the config file it read, the models it
found and skipped, and each model's loads. Under launchd, the plist in
[INSTALL.md](INSTALL.md#running-as-a-service) writes them to
`~/Library/Logs/sidekickd.log`. `RUST_LOG` sets the filter, in `tracing`'s
syntax (`RUST_LOG=sidekick=debug,sidekick_server=debug`).

## Troubleshooting

`curl -s localhost:8790/health` is the first stop: it says what is
available and why anything isn't.

- **A model isn't in `/v1/models`.** Restart the daemon if you installed it
  while it ran ([above](#restart-after-installing)). If it still isn't
  listed, `/health`'s `skipped_models` names its manifest and the reason:
  a manifest that doesn't validate, a duplicate id, a model over the ANE's
  weight limit, or an onnx model with no ONNX Runtime to run it.
- **Chat answers 503.** `/health` gives the reason: no Foundation Models on
  this platform, Apple Intelligence turned off, or its model still
  downloading.
- **Every request answers 401.** The daemon has an `api_key`
  ([API.md](API.md#authentication)).
- **A first request is slow, or answers 504 `load_timeout`.** Core ML is
  compiling its bucket ([above](#first-loads-and-the-compile-cache)); the
  load continues, and a retry finds it ready.
- **macOS won't run sidekickd or load its libraries.** A browser download is
  quarantined ([INSTALL.md](INSTALL.md#release-archives)).
