#!/bin/sh
# Smoke-launch a release's sidekickd: start it on an empty models directory,
# wait for /health, and require the ONNX Runtime beside it to have loaded
# at the pinned version (D40).
#
#   tools/smoke_launch.sh <path to sidekickd>
#
# The expected version is the one the binary names in --version: the pin
# for its target.
set -eu

daemon=${1:?usage: smoke_launch.sh <sidekickd>}
want=$("$daemon" --version | sed -n 's/.*(ONNX Runtime \([0-9.]*\) expected).*/\1/p')
[ -n "$want" ] || { echo "$daemon --version names no ONNX Runtime: built without the onnx feature?" >&2; exit 1; }
work=$(mktemp -d)
trap 'kill "$pid" 2>/dev/null || true; rm -rf "$work"' EXIT
mkdir -p "$work/models"
port=18$(( $$ % 900 + 100 ))
printf 'addr = "127.0.0.1:%s"\nmodels_dir = "%s/models"\n' "$port" "$work" > "$work/config.toml"
# Let the runtime come from beside the binary, as installed.
env -u ORT_DYLIB_PATH "$daemon" --config "$work/config.toml" > "$work/log" 2>&1 &
pid=$!
health=
for _ in $(seq 1 100); do
    if health=$(curl -fsS "http://127.0.0.1:$port/health" 2>/dev/null); then
        break
    fi
    sleep 0.1
done
if [ -z "$health" ]; then
    echo "sidekickd didn't answer /health" >&2
    cat "$work/log" >&2
    exit 1
fi
echo "$health" | grep -q "\"version\":\"$want\"" || { echo "ONNX Runtime $want didn't load: $health" >&2; exit 1; }
echo "$health" | grep -q '"source":"beside sidekickd"' || { echo "ONNX Runtime didn't load from beside sidekickd: $health" >&2; exit 1; }
echo "sidekickd answered /health with ONNX Runtime $want loaded from beside it"
