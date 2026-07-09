# C smoke client

Hardware verification for the C ABI, meant to run against a real models dir:

```sh
cargo build --release -p sidekick-embed-ffi
clang -O2 -o /tmp/sk_smoke examples/smoke.c \
  -I include -L ../../target/release -lsidekick \
  -Wl,-rpath,"$PWD/../../target/release"
/tmp/sk_smoke
```

Expects bge-small-en-v1.5 and embeddinggemma-300m in the default models
directory (see tools/ for conversion scripts).
