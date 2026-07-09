/* sidekick.h — C ABI for sidekick's on-device embedding tiers.
 *
 * In-process alternative to sidekickd's /v1/embeddings for hosts that can't
 * or don't want to run the daemon. Both share the same models directory
 * (default: ~/Library/Application Support/sidekick/models).
 *
 * Every function is thread-safe and panic-safe. Strings are UTF-8.
 * Out-strings are freed with sk_string_free, embedding buffers with
 * sk_floats_free. Recommended probe order for hosts: sidekickd /health
 * (short timeout) -> dlopen this library -> your own fallback.
 */
#ifndef SIDEKICK_H
#define SIDEKICK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any breaking ABI change. Check before anything else. */
#define SK_ABI_VERSION_EXPECTED 1
uint32_t sk_abi_version(void);

typedef struct sk_pool sk_pool;

/* Open a pool over a models directory (NULL -> the daemon's default).
 * Returns NULL on failure with *err set (free with sk_string_free).
 * An empty/missing directory is not an error: sk_pool_models returns "[]". */
sk_pool *sk_pool_open(const char *models_dir, char **err);

/* Drop every resident model. pool may be NULL. */
void sk_pool_close(sk_pool *pool);

/* JSON array of model ids, e.g. ["bge-small-en-v1.5","static-floor"].
 * Free with sk_string_free. Returns NULL on failure with *err set;
 * an empty models directory is "[]", not NULL. */
char *sk_pool_models(const sk_pool *pool, char **err);

/* JSON description of one model from its manifest (not loaded):
 *   {"id","backend","dims","matryoshka","max_seq_len"}
 * "matryoshka" lists the dims values sk_embed accepts as requested_dims;
 * empty means native dims only. Free with sk_string_free. Returns NULL on
 * failure with *err set. */
char *sk_model_info(const sk_pool *pool, const char *model_id, char **err);

/* Native output dimensionality from the model's manifest (does not load
 * the model). Returns 0 for an unknown id, with *err set. */
size_t sk_embed_dims(const sk_pool *pool, const char *model_id, char **err);

/* Embed n_texts strings. purpose: 0 = document, 1 = query (applies the
 * model's query prefix). requested_dims: 0 for native dims, or one of the
 * model's matryoshka dims for truncated + renormalized vectors — the same
 * semantics as the daemon's "dimensions" parameter. On success returns
 * n_texts * (*dims_out) floats, row-major, each row unit-normalized; free
 * with sk_floats_free(ptr, n_texts * dims). Returns NULL on failure with
 * *err set. The first call for a model loads it (Core ML: seconds for
 * large encoders) without blocking calls on other, already-loaded models;
 * it stays resident until sk_pool_close. */
float *sk_embed(const sk_pool *pool, const char *model_id,
                const char *const *texts, size_t n_texts, int purpose,
                size_t requested_dims, size_t *dims_out, char **err);

void sk_floats_free(float *ptr, size_t n_floats);
void sk_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* SIDEKICK_H */
