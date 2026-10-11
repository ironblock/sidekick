# C smoke client

Hardware verification for the C ABI, meant to run against a real models dir:

```sh
cargo build --release -p sidekick-embed-ffi
clang -O2 -o sk_smoke examples/smoke.c \
  -I include -L ../../target/release -lsidekick \
  -Wl,-rpath,"$PWD/../../target/release"
./sk_smoke
```

Run from this crate's directory, on macOS. Expects Core ML builds of
bge-small-en-v1.5 and embeddinggemma-300m in the default models directory
(see tools/ for conversion scripts): the library skips ONNX manifests.
