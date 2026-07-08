# sidekick

On-device inference for very small, asynchronous tasks on Apple Silicon —
session titles, tags, embeddings, structured extraction — using every part of
the silicon: Apple's Foundation Models (ANE, via Apple Intelligence) for
generation, Core ML encoders on the Apple Neural Engine for embeddings, and a
pure-CPU static-embedding floor tier that works anywhere.

The daemon, **`sidekickd`**, speaks the OpenAI API, so anything that can point
at an OpenAI-compatible base URL (OpenCode, editors, scripts) can use it:

```
POST /v1/chat/completions   Apple Foundation Models (macOS 26+, Apple Intelligence)
POST /v1/embeddings         Core ML / ANE encoders + static floor models
GET  /v1/models             what this machine can serve
GET  /health                availability per tier, and why when unavailable
```

Design rationale lives in
[docs/design/ane-inference-first-principles.md](docs/design/ane-inference-first-principles.md);
autonomous implementation decisions are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

## Status

All three inference paths are hardware-verified on Apple Silicon
(macOS 26.5.1): the Foundation Models Swift shim (including constrained
decoding), the static embedding tier, and the Core ML encoder tier with a
locally converted bge-small running ANE-resident (3.4x over CPU at seq 128,
verified via `cargo run -p sidekick-coreml --example ane_check`). See
"Hardware verification status" in [docs/DECISIONS.md](docs/DECISIONS.md),
`cargo run -p sidekick-server --bin smoke-test`, and
[tools/convert_bge_small.py](tools/convert_bge_small.py) for the conversion
recipe (the constraints it encodes are hardware-verified; deviating from
them silently pushes the encoder off the ANE).

## Quickstart

```sh
cargo build --release -p sidekick-server --features coreml   # on a Mac
./target/release/sidekickd --models-dir ~/Library/Application\ Support/sidekick/models
curl -s localhost:8790/health | jq
```

Chat requires macOS 26+ with Apple Intelligence enabled; `/health` tells you
where you stand. Embeddings require at least one model in the models
directory (see below). Requests degrade honestly: a missing tier is a 503
with a reason, never a hang.

### Running as a daemon

For always-on use, run under launchd as a user agent. Save as
`~/Library/LaunchAgents/dev.sidekick.sidekickd.plist` (adjust the binary
path), then `launchctl load ~/Library/LaunchAgents/dev.sidekick.sidekickd.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.sidekick.sidekickd</string>
  <key>ProgramArguments</key>
  <array><string>/usr/local/bin/sidekickd</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/sidekickd.log</string>
</dict></plist>
```

Foundation Models requires a logged-in user session (it is Apple
Intelligence), so a LaunchAgent — not a LaunchDaemon — is the right unit.

### Example: OpenCode session titles

Point a provider at `http://127.0.0.1:8790/v1` with model `apple-fm`.
Constrained output works via the standard `response_format`:

```json
{
  "model": "apple-fm",
  "messages": [{"role": "user", "content": "Title this session: ..."}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {"name": "title", "schema": {
      "type": "object",
      "properties": {"title": {"type": "string"}},
      "required": ["title"]
    }}
  }
}
```

On-device guided generation makes the 3B model reliable at exactly this kind
of task — the schema is enforced by constrained decoding, not by hoping.

## Models directory

Each embedding model is a directory with a `manifest.toml`:

```
~/Library/Application Support/sidekick/models/
  embeddinggemma-300m/
    manifest.toml
    model.mlmodelc/        # precompile: xcrun coremlcompiler compile model.mlpackage .
    tokenizer.json
```

See [examples/manifests/](examples/manifests/) for annotated manifests
(EmbeddingGemma-300m on ANE, bge-small, a static floor model). Manifest rules
that matter: Core ML models must declare enumerated sequence-length `buckets`
(fixed shapes are what keep the model on the ANE), and `matryoshka` declares
which `dimensions` values the OpenAI API may request.

## Configuration

`~/.config/sidekick/config.toml`, all optional (defaults shown):

```toml
addr = "127.0.0.1:8790"        # loopback only by default
# models_dir = "..."           # default: <data dir>/sidekick/models
# api_key = "..."              # require Authorization: Bearer <key> on /v1
session_ttl_secs = 300         # Foundation Models session reuse window
model_idle_ttl_secs = 900      # embedding model residency after last use
```

CLI flags override the file: `sidekickd --addr ... --models-dir ... --api-key ...`.

## Workspace layout

| Crate | What it is |
|---|---|
| `sidekick-core` | Backend-neutral traits and types: `ChatBackend`, `Embedder`, availability states, model manifest/registry. No Apple dependencies. |
| `sidekick-coreml` | Small safe wrapper over `objc2-core-ml`: load, compute units, int32-in/float-out predictions. macOS only; empty stub elsewhere. |
| `sidekick-fm` | Foundation Models backend: Swift C-ABI shim built by `build.rs` (macOS 26 SDK), guided generation from JSON Schema, TTL'd session reuse keyed by conversation prefix. Stub elsewhere. |
| `sidekick-embed` | Embedding pipelines: Core ML/ANE encoder (feature `coreml`) and the static floor tier. |
| `sidekick-server` | `sidekickd`, the OpenAI-compatible daemon. |

## Development

```sh
cargo test --workspace                 # runs anywhere, including Linux
cargo check --workspace --target aarch64-apple-darwin --features sidekick-embed/coreml
```

The second command is the cross-check CI uses to keep the macOS-only code
honest from non-Mac machines. On a Mac, `cargo test --workspace --features
sidekick-server/coreml` additionally builds the Swift shim (Xcode 26 needed
for the real Foundation Models backend; anything older falls back to the
stub with a build warning).
