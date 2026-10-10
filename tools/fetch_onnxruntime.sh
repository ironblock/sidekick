#!/bin/sh
# Fetch the ONNX Runtime the onnx backend is pinned to (D40), verify it, and
# print the path of its shared library: set ORT_DYLIB_PATH to it, or copy it
# beside sidekickd.
#
#   tools/fetch_onnxruntime.sh <dest dir> [platform]
#
# platform: osx-arm64, osx-x86_64, linux-x64 or linux-aarch64 (default: this
# machine's). Intel macOS is pinned to 1.23.2, the last release Microsoft
# built for it; the others to 1.28.3. The archive is downloaded once into
# <dest dir> and checked against its pinned sha256 (the GitHub release's
# asset digest) before it is extracted.
set -eu

dest=${1:?usage: fetch_onnxruntime.sh <dest dir> [platform]}
platform=${2:-}
if [ -z "$platform" ]; then
    case "$(uname -s)/$(uname -m)" in
        Darwin/arm64) platform=osx-arm64 ;;
        Darwin/x86_64) platform=osx-x86_64 ;;
        Linux/x86_64) platform=linux-x64 ;;
        Linux/aarch64) platform=linux-aarch64 ;;
        *) echo "no pinned ONNX Runtime build for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
    esac
fi
case "$platform" in
    osx-arm64) VERSION=1.28.3 sha=c436bd9f47dbce6f6311ccc21de97f829b3c862c09d24aa321958cb72f0a1d32 ;;
    osx-x86_64) VERSION=1.23.2 sha=d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6 ;;
    linux-x64) VERSION=1.28.3 sha=db14e4863bd37893fc59729d986ab2a0d043d10b7d44da1913c4982b7e3d009c ;;
    linux-aarch64) VERSION=1.28.3 sha=6c6b1ae96d7b0be9f555092857c6a4f2b0b5587a5298d732d9e52e8253038ce7 ;;
    *) echo "unknown platform $platform" >&2; exit 1 ;;
esac
case "$platform" in
    osx-*) lib=libonnxruntime.$VERSION.dylib ;;
    *) lib=libonnxruntime.so.$VERSION ;;
esac

name=onnxruntime-$platform-$VERSION
archive=$dest/$name.tgz
mkdir -p "$dest"
if [ ! -f "$archive" ]; then
    curl -fsSL -o "$archive.part" "https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$name.tgz"
    mv "$archive.part" "$archive"
fi
if command -v sha256sum >/dev/null; then
    got=$(sha256sum "$archive" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$archive" | cut -d' ' -f1)
fi
if [ "$got" != "$sha" ]; then
    echo "$archive: sha256 $got, expected $sha" >&2
    exit 1
fi
[ -d "$dest/$name" ] || tar -xzf "$archive" -C "$dest"
echo "$(cd "$dest/$name/lib" && pwd)/$lib"
