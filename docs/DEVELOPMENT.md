# Developing sidekick

The workspace, building and testing, and the checks CI runs. The rules for
commits, branches and releases, and for keeping this public repository
clean, are in [AGENTS.md](../AGENTS.md). Converting models is in
[CONVERTING.md](CONVERTING.md).

## Contents

- [The workspace](#the-workspace)
- [Building and testing](#building-and-testing)
- [The Foundation Models shim and the SDK](#the-foundation-models-shim-and-the-sdk)
- [Checks on the hardware](#checks-on-the-hardware)
- [The docs and the data](#the-docs-and-the-data)
- [Test-only environment variables](#test-only-environment-variables)

## The workspace

| Crate | What it is |
|---|---|
| `sidekick-core` | Backend-neutral traits and types: `ChatBackend`, `Embedder`, `Classifier`, availability states, model manifests and registry; the chat route's JSON Schema layer and stop-sequence shaper, shared by every chat backend. No Apple dependencies. |
| `sidekick-coreml` | Small safe wrapper over `objc2-core-ml`: load, compute units, int32-in/float-out predictions, and the stateful decoder chain open chat models run on (`KvDecoder`, [D53](DECISIONS.md#d53--open-llms-on-the-chat-route-a-stateful-ane-decoder-beside-foundation-models)). macOS only; empty stub elsewhere. |
| `sidekick-onnx` | ONNX Runtime on the CPU: finds and loads `libonnxruntime` at run time, runs a graph in a session, and sets inference threads from the machine's fast cores. `ort` comes in only with its `runtime` feature, which `sidekick-embed`'s `onnx` feature turns on. |
| `sidekick-fm` | Foundation Models backend: Swift C-ABI shim built by `build.rs` (macOS 26.4+ SDK; 27 SDK for the macOS 27 features), guided generation from the schema layer's validated schema, TTL'd session reuse keyed by conversation prefix. Stub elsewhere. |
| `sidekick-embed` | Encoder pipelines: Core ML/ANE embedders and classifiers (feature `coreml`, over `sidekick-coreml`), ONNX Runtime CPU embedders (feature `onnx`, over `sidekick-onnx`), laya's input format, and the static floor tier. |
| `sidekick-server` | `sidekickd`, the OpenAI-compatible daemon. |
| `sidekick-embed-ffi` | `libsidekick.dylib`, the C ABI for in-process embeddings (Core ML and static models; built without ONNX support). |

## Building and testing

```sh
cargo test --workspace                 # runs anywhere, including Linux
cargo check --workspace --target aarch64-apple-darwin --features sidekick-embed/coreml
# On a Mac with converted models and references (MODELS.md):
cargo run --release -p sidekick-embed --features coreml,onnx --example parity
# ONNX models only, on any OS (needs ORT_DYLIB_PATH):
cargo run --release -p sidekick-embed --features onnx --example parity
```

The second command is the cross-check CI uses to keep the macOS-only code
honest from non-Mac machines. On a Mac, `cargo test --workspace --features
sidekick-server/coreml` additionally builds the Swift shim (below).

## The Foundation Models shim and the SDK

The shim needs Xcode
26.4 or later; older SDKs are a build error (set `SIDEKICK_FM_STUB=1` to build
the stub backend instead). Build with **Xcode 27** to get the macOS 27
Foundation Models features (real token usage, model variant, typed errors):
they are compiled in only against the 27 SDK, and a 27-SDK build still
runs on macOS 26. `sidekickd --version` reports
which SDK a binary was built with; after switching Xcode with
`xcode-select`, run `cargo clean -p sidekick-fm` so the shim is rebuilt.

## Checks on the hardware

- `cargo run -p sidekick-server --bin smoke-test`, on a Mac with Apple
  Intelligence on, exercises the Foundation Models shim end to end:
  availability, completion, session reuse, constrained generation, finish
  reasons, streaming and context overflow. It exits non-zero on the first
  failure.
- `cargo run -p sidekick-coreml --example ane_check -- <bucket.mlmodelc>
  <seq> <inputs…> <output>` judges one compiled bucket's ANE eligibility
  from Core ML's compute plan, without running it, then checks pad
  invariance on the ANE and CPU paths; its header documents the arguments
  ([MODELS.md](MODELS.md) has the method).
- `cargo run --release -p sidekick-coreml --example kv_decoder -- <model
  dir> <job.json> <out prefix>` drives a `chat.toml` decoder through a job
  of measurements (call speed, greedy continuations, teacher-forced logits,
  a long run's memory); its header documents the job. Sign the binary with
  a fixed identifier after each build (`codesign -s - -f -i <identifier>`),
  or a large build's ANE compile is redone and stored again whenever the
  linker's identifier changes ([coreml-cache.md](design/coreml-cache.md#the-cache-key)).
- `SIDEKICK_ANE_TESTS=1 cargo test -p sidekick-coreml --test kv_decoder`
  runs a small `kv-state-1` decoder, built by
  `tools/make_decoder_test_model.py`, on the Neural Engine. Without the
  variable, and on a Mac whose Core ML lists no Neural Engine (a CI virtual
  machine), the tests skip: they compile programs for the ANE and want it
  to themselves.
- The parity suite (`crates/sidekick-embed/examples/parity`,
  [D26](DECISIONS.md#d26--a-parity-suite-grades-every-model-on-every-compute-path))
  grades every model on every compute path against its reference;
  [MODELS.md](MODELS.md#confidence-grades-the-parity-suite) has the method.

## The docs and the data

CI checks that the docs agree with their data and with the code:

- `python3 tools/models_table.py --docs --check`: MODELS.md's generated
  tables against [data/models.json](data/models.json), and the publishing
  data (the licence audit and the checksum ledger).
- `python3 tools/model_card.py --check`: every model card the ledger
  allows renders, and its links into these docs resolve.
- `python3 tools/doclinks.py check`: every relative link and anchor in the
  repository's Markdown resolves.
- `python3 -m unittest discover -s tools/tests -t tools`: the tools' own
  tests.
- `cargo test -p sidekick-server`: includes a check that
  [CONFIG.md](CONFIG.md) documents every config key, flag and environment
  variable the daemon reads.

## Test-only environment variables

Read by tests and examples, never by the daemon:

| Variable | |
|---|---|
| `SIDEKICK_ANE_TESTS` | `1` runs `sidekick-coreml`'s decoder tests on the Neural Engine (`tests/kv_decoder.rs`); they skip otherwise. |
| `SIDEKICK_REQUIRE_ONNX` | `1` fails the ONNX tests instead of skipping them when `ORT_DYLIB_PATH` isn't set. |
| `SIDEKICK_CLASSIFY_FIXTURES` | The classifier token fixtures (default: `fixtures/classify`). |
| `SIDEKICK_MODELS_DIR` | The models directory the classifier fixture test reads. |
| `SIDEKICK_REQUIRE_CLASSIFY_FIXTURES` | Fails the classifier fixture test instead of skipping it when a model is missing. |
| `SIDEKICK_FIXTURES` | The parity suite's fixtures directory (or `--fixtures`). |
| `SIDEKICK_ONNX_BIG_FILE` | A large ONNX file for a by-hand test of reading a graph without its weights. |
