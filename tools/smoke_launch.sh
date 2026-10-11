#!/bin/sh
# Smoke-launch a release's sidekickd twice (D40):
# 1. with the tiny test ONNX model installed: ONNX Runtime must load at
#    startup from beside the binary, at the pinned version for its target
#    (the one `sidekickd --version` names), and the model must embed;
# 2. with an empty models directory: ONNX Runtime must not load at all.
#
#   tools/smoke_launch.sh <path to sidekickd>
#
# Run from a checkout: the test model comes from
# crates/sidekick-embed/tests/fixtures/tiny-onnx.
set -eu

daemon=${1:?usage: smoke_launch.sh <sidekickd>}
fixtures="$(cd "$(dirname "$0")/.." && pwd)/crates/sidekick-embed/tests/fixtures/tiny-onnx"
want=$("$daemon" --version | sed -n 's/.*(ONNX Runtime \([0-9.]*\) expected).*/\1/p')
[ -n "$want" ] || { echo "$daemon --version names no ONNX Runtime: built without the onnx feature?" >&2; exit 1; }

work=$(mktemp -d)
pid=
trap 'if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi; rm -rf "$work"' EXIT

# Start the daemon on <models dir>, wait for /health, print it, stop it.
launch() {
    port=18$(( ($$ + $2) % 900 + 100 ))
    printf 'addr = "127.0.0.1:%s"\nmodels_dir = "%s"\n' "$port" "$1" > "$work/config.toml"
    # The runtime must come from beside the binary, as installed.
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
    if [ "$2" = 1 ]; then
        embedding=$(curl -fsS "http://127.0.0.1:$port/v1/embeddings" -H 'content-type: application/json' \
            -d '{"model": "tiny", "input": "[CLS] b c [SEP]"}' || true)
    fi
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    pid=
}

# 1. An onnx model installed: the runtime loads, and the model embeds.
mkdir -p "$work/models/tiny"
cp "$fixtures/hidden.onnx" "$fixtures/tokenizer.json" "$work/models/tiny/"
cat > "$work/models/tiny/manifest.toml" <<'EOF'
id = "tiny"
backend = "onnx"
artifact = "hidden.onnx"
tokenizer = "tokenizer.json"
dims = 8
pooling = "cls"
max_seq_len = 16

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"
position_ids = "position_ids"
output = "last_hidden_state"
EOF
launch "$work/models" 1
echo "$health" | grep -q '"state":"loaded"' || { echo "ONNX Runtime didn't load: $health" >&2; exit 1; }
echo "$health" | grep -q "\"version\":\"$want\"" || { echo "ONNX Runtime $want didn't load: $health" >&2; exit 1; }
echo "$health" | grep -q '"source":"beside sidekickd"' || { echo "ONNX Runtime didn't load from beside sidekickd: $health" >&2; exit 1; }
echo "$embedding" | grep -q '"embedding"' || { echo "the test model didn't embed: $embedding" >&2; exit 1; }
echo "with an onnx model: ONNX Runtime $want loaded from beside sidekickd, and the model embeds"

# 2. No onnx model: the runtime never loads.
mkdir -p "$work/empty"
launch "$work/empty" 2
echo "$health" | grep -q '"state":"not loaded"' || { echo "ONNX Runtime loaded with no onnx model: $health" >&2; exit 1; }
echo "without one: ONNX Runtime not loaded"
