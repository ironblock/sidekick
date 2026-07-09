# Integrating sidekick embeddings into a host application

For an app that wants on-device embeddings when they're available without
hard-depending on sidekick. Three situations, one probe chain:

1. **sidekickd is running** → talk HTTP.
2. **sidekick is installed but not running** → load `libsidekick.dylib`
   in-process.
3. **no sidekick** → your own fallback (skip the feature, or a bundled
   lexical/static method).

Probe in that order at startup (and optionally re-probe on failure):

```
try:  GET http://127.0.0.1:8790/health          (timeout ~150ms)
      -> use POST /v1/embeddings (OpenAI-compatible)
else: dlopen("libsidekick.dylib")               (see search paths below)
      -> sk_pool_open(NULL) -> sk_embed(...)
else: fallback
```

Prefer the daemon when both are available: it shares resident models across
every client on the machine, owns idle eviction, and its API is stable JSON
over HTTP. The dylib is the zero-daemon path — right when you can't manage
a service, want process-lifetime control, or are embedding sidekick into a
sandboxed host.

Both paths read the same models directory
(`~/Library/Application Support/sidekick/models`), so models converted once
(e.g. with `tools/convert_bge_small.py`) serve both. If the models dir is
empty, the daemon 404s the model and `sk_pool_models` returns `[]` — treat
either as "fall back".

## Path 1: the daemon

`POST /v1/chat/completions`-style OpenAI compatibility, documented in the
README. Embeddings: `POST /v1/embeddings` with optional `input_type:
"query"`, `dimensions` (matryoshka models), `encoding_format: "base64"`.

To make "installed but not running" disappear entirely, install the
LaunchAgent from the README (`KeepAlive` keeps it warm); then path 2 only
matters for machines where the user skipped that step.

## Path 2: `libsidekick.dylib`

C ABI defined in
[`crates/sidekick-embed-ffi/include/sidekick.h`](../crates/sidekick-embed-ffi/include/sidekick.h);
build with `cargo build --release -p sidekick-embed-ffi` → 
`target/release/libsidekick.dylib` (~5 MB, embeddings only, no
FoundationModels linkage — it runs on any macOS the models run on).

Suggested `dlopen` search order for hosts:

1. `$SIDEKICK_DYLIB` (explicit override)
2. next to your app's own binary / inside your app bundle (if you ship it)
3. `/opt/homebrew/lib/libsidekick.dylib`, `/usr/local/lib/libsidekick.dylib`

Minimal usage (C; every language with FFI maps 1:1 — the symbols are plain
C, no callbacks, no structs by value):

```c
if (sk_abi_version() != SK_ABI_VERSION_EXPECTED) goto fallback;
char *err = NULL;
sk_pool *pool = sk_pool_open(NULL, &err);          /* default models dir */
if (!pool) goto fallback;
char *models = sk_pool_models(pool, &err);         /* JSON id array */
size_t dims = 0;
const char *texts[] = {"the quick brown fox"};
float *v = sk_embed(pool, "bge-small-en-v1.5", texts, 1,
                    /*purpose: 0=document, 1=query*/ 0,
                    /*requested_dims: 0=native, or a matryoshka value*/ 0,
                    &dims, &err);
/* ... use v[0..dims) ... */
sk_floats_free(v, 1 * dims);
sk_string_free(models);
sk_pool_close(pool);
```

Notes:
- Rows come back unit-normalized; cosine similarity is a plain dot product.
- `requested_dims` has the daemon's `dimensions` semantics (matryoshka
  truncate + renormalize), so vectors indexed via one path stay compatible
  with the other. Discover a model's valid values with `sk_model_info`
  (JSON: `{"id","backend","dims","matryoshka","max_seq_len"}`).
- The first `sk_embed` per model loads it (Core ML: ~1s for bge-small,
  seconds for large encoders) and it stays resident until `sk_pool_close`.
  Loads don't block calls on other, already-loaded models.
- Thread-safe; calls from multiple threads are fine. The ANE serializes
  predictions anyway, so client-side batching beats client-side threading.
- Chat is deliberately not in the dylib: it would drag the FoundationModels
  Swift shim into every host, and conversational sessions want a daemon
  lifetime. If you need chat too, run sidekickd.

## Path 3: your fallback

`sk_pool_models` returning `[]`, `sk_pool_open` failing, or the daemon
404ing your model id all mean the same thing: sidekick is present but has
no usable model. Treat it identically to "no sidekick".
