#!/bin/sh
# Give a macOS sidekickd a stable code-signing identifier, so Core ML's ANE
# cache survives rebuilds and upgrades.
#
#   tools/sign_sidekickd.sh [path to sidekickd]   (default: target/release/sidekickd)
#
# Core ML keys each compiled ANE program to the code-signing identifier of
# the process that loads it. The linker signs a Rust binary ad hoc as
# `sidekickd-<hash>`, cargo's name for the build, and the hash changes with
# the package version, features and toolchain. Every such change made every
# ANE bucket compile again on its first load: seconds for a small model,
# minutes for a long bucket of a large one. Re-signing ad hoc with a fixed
# identifier keeps the key the same from one build to the next. The
# identifier is reverse-DNS under the project's GitHub namespace
# (ironblock.github.io), which the project controls.
#
# The release workflow runs this on every macOS archive's sidekickd. After
# building from source, run it once per build:
#
#   cargo build --release -p sidekick-server --features coreml,onnx
#   tools/sign_sidekickd.sh
#
# The signature is ad hoc (no certificate), as the linker's was. Anything
# that re-signs the binary with an identity of its own decides the
# identifier from then on.
set -eu

IDENTIFIER=io.github.ironblock.sidekickd

bin=${1:-target/release/sidekickd}
[ "$(uname -s)" = Darwin ] || { echo "sign_sidekickd.sh: macOS only (Core ML's cache is macOS's)" >&2; exit 1; }
[ -f "$bin" ] || { echo "sign_sidekickd.sh: no binary at $bin; build it first" >&2; exit 1; }

codesign --force --sign - --identifier "$IDENTIFIER" "$bin"
codesign --verify --strict "$bin"
got=$(codesign --display --verbose=1 "$bin" 2>&1 | sed -n 's/^Identifier=//p')
[ "$got" = "$IDENTIFIER" ] || { echo "sign_sidekickd.sh: $bin is signed as '$got', not $IDENTIFIER" >&2; exit 1; }
echo "$bin: signed ad hoc as $IDENTIFIER"
