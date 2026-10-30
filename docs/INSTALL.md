# Installing sidekick

How to install `sidekickd` from a release archive or from source, keep it
running, and upgrade it. Installing models is in
[RUNNING.md](RUNNING.md#installing-models).

## Contents

- [Platforms](#platforms)
- [Release archives](#release-archives)
- [ONNX Runtime](#onnx-runtime)
- [Building from source](#building-from-source)
- [Running as a service](#running-as-a-service)
- [Upgrading](#upgrading)

## Platforms

| Platform | Chat (Foundation Models) | Embeddings | Classify, rerank | Release archive |
|---|---|---|---|---|
| macOS 26+, Apple silicon (`aarch64-apple-darwin`) | yes | Core ML (ANE, GPU, CPU), ONNX on the CPU, static | Core ML | yes |
| macOS 13.4+, Intel (`x86_64-apple-darwin`); built and tested on macOS 26 | no (503) | Core ML (GPU, CPU), ONNX on the CPU, static | Core ML | yes |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | no (503) | ONNX on the CPU, static | no | yes, glibc 2.35+ |
| Linux arm64 (`aarch64-unknown-linux-gnu`) | no (503) | ONNX on the CPU, static | no | yes, glibc 2.35+ |

Chat requires an Apple silicon Mac on macOS 26+ with Apple Intelligence
enabled; Intel Macs and Linux get a 503 from `/v1/chat/completions`.
`/health` tells you where you stand. An ONNX build of a model keeps its id,
so a client calls the same model the same way on every platform. What each
backend does, and its limits, is in [BACKENDS.md](BACKENDS.md).

## Release archives

Each [release](https://github.com/ironblock/sidekick/releases) has one
archive per platform, `sidekick-<version>-<target>.tar.gz`, with its
`.sha256` beside it. On an Apple silicon Mac:

```sh
VERSION=v0.17.0
NAME=sidekick-$VERSION-aarch64-apple-darwin
curl -LO https://github.com/ironblock/sidekick/releases/download/$VERSION/$NAME.tar.gz
curl -LO https://github.com/ironblock/sidekick/releases/download/$VERSION/$NAME.tar.gz.sha256
shasum -a 256 -c $NAME.tar.gz.sha256
tar -xzf $NAME.tar.gz && cd $NAME
```

For another platform, set `NAME` to its target from the table above; on
Linux, check the archive with `sha256sum -c`.

Every archive holds:
- `sidekickd`, the daemon;
- ONNX Runtime (`libonnxruntime.dylib` or `.so`), with its licence and
  notices;
- `README.md`, `INTEGRATING.md` and `RELEASE_NOTES.md`, whose links point
  at the release's tag on GitHub;
- `LICENSE` and `THIRD_PARTY_NOTICES.md`.

The Apple silicon archive also holds `libsidekick.dylib` and its header,
`sidekick.h`, for in-process embeddings ([INTEGRATING.md](INTEGRATING.md)).

Keep the files together: `sidekickd` loads the ONNX Runtime beside it.
Run it from the unpacked directory, or move the directory somewhere
permanent (for example `~/.local/share/sidekick/bin`) and run it from
there.

**macOS quarantine.** An archive downloaded with `curl`, as above, isn't
quarantined. One downloaded with a web browser is: macOS marks its files
with the `com.apple.quarantine` attribute, and since sidekickd is signed
ad hoc rather than notarized, Gatekeeper may refuse to run it or to load
its libraries. Clear the attribute from the unpacked directory before the
first run:

```sh
xattr -dr com.apple.quarantine sidekick-v0.17.0-aarch64-apple-darwin
```

## ONNX Runtime

ONNX models need ONNX Runtime ([D40](DECISIONS.md#d40--an-onnx-runtime-cpu-backend-for-embeddings)):
1.28.3, or 1.23.2 on Intel macOS, the last release Microsoft built for it.
Release archives ship it beside `sidekickd`, for all four targets. How the
daemon finds it, and how to point it at another install, is in
[CONFIG.md](CONFIG.md#where-onnx-runtime-is-loaded-from).

The daemon loads ONNX Runtime only when the models directory has an onnx
model, so a Mac serving only Core ML models never loads it.

## Building from source

```sh
cargo build --release -p sidekick-server --features coreml,onnx   # on a Mac
tools/sign_sidekickd.sh                                           # on a Mac; see below
./target/release/sidekickd --models-dir ~/Library/Application\ Support/sidekick/models
curl -s localhost:8790/health | jq
```

On Linux, build with `--features onnx` only. A source build serving ONNX
models needs the pinned ONNX Runtime: `tools/fetch_onnxruntime.sh <dir>`
fetches and checks it, and `onnx_runtime` in the config (or
`ORT_DYLIB_PATH`) points at it.

Building the Foundation Models shim needs Xcode 26.4 or later, and Xcode
27 for the macOS 27 features; [DEVELOPMENT.md](DEVELOPMENT.md) has the
details.

### Signing

`tools/sign_sidekickd.sh` re-signs the binary ad hoc with a fixed
code-signing identifier, `io.github.ironblock.sidekickd`, as the release
archives' sidekickd is signed. Core ML keys each model it compiles for the
Neural Engine to the identifier of the process that loads it, and the one
the linker gives a Rust binary is cargo's name for the build
(`sidekickd-<hash>`), which changes with the version, features and
toolchain. Without the step, a rebuild that changes it compiles every ANE
bucket again on its first load: after a version bump, bge-small-en-v1.5's
first request took 3.76 s instead of 0.87 s
([design/coreml-cache.md](design/coreml-cache.md#the-cache-key) has the
measurement).

## Running as a service

For always-on use, run under launchd as a user agent. Save as
`~/Library/LaunchAgents/dev.sidekick.sidekickd.plist`, with the path to
your `sidekickd` and your home directory in place of the examples (launchd
doesn't expand `~`), then load it:

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.sidekick.sidekickd.plist
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.sidekick.sidekickd</string>
  <key>ProgramArguments</key>
  <array><string>/Users/you/.local/share/sidekick/bin/sidekickd</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/you/Library/Logs/sidekickd.log</string>
  <key>StandardErrorPath</key><string>/Users/you/Library/Logs/sidekickd.log</string>
</dict></plist>
```

sidekickd logs to standard output, so the plist sends both streams to one
file under `~/Library/Logs`, where Console finds it.

- **Restart it** after installing or removing a model, or changing the
  config, since the daemon reads both when it starts:
  `launchctl kickstart -k gui/$(id -u)/dev.sidekick.sidekickd`.
- **Stop it:** `launchctl bootout gui/$(id -u)/dev.sidekick.sidekickd`.

Foundation Models requires a logged-in user session (it is Apple
Intelligence), so a LaunchAgent — not a LaunchDaemon — is the right unit.
To serve ONNX models, keep `libonnxruntime.dylib` beside the binary (copy
both from the release archive) or set `onnx_runtime` in the config:
launchd doesn't pass a shell's `ORT_DYLIB_PATH`. On Linux, run `sidekickd`
under your service manager.

## Upgrading

1. Read the release notes, `RELEASE_NOTES.md` in the archive or
   [docs/releases/](releases/), for changes that need action.
2. Replace `sidekickd` and the libraries beside it with the new archive's,
   at the same path. Keep the same file name: Core ML's compile cache is
   keyed to the executable's name ([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)).
3. Restart the daemon.

The config file and the models directory are untouched. From 0.16.0 on,
every release archive signs sidekickd with the same identifier, so an
upgrade keeps the Neural Engine programs Core ML has already compiled
(the upgrade from 0.15.0 compiles them once more). A macOS update is
expected to make Core ML compile them again
([RUNNING.md](RUNNING.md#first-loads-and-the-compile-cache)).
