#!/bin/sh
# Smoke-launch a release's sidekickd: start it on an empty models directory,
# wait for /health, and require the ONNX Runtime beside it to have loaded
# at the pinned version (D40).
#
#   tools/smoke_launch.sh <path to sidekickd> [expected ONNX Runtime version]
set -eu

daemon=${1:?usage: smoke_launch.sh <sidekickd> [onnx runtime version]}
want=${2:-1.28.3}
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
