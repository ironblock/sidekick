/* C smoke client for libsidekick.dylib: bge + gemma on the ANE, in-process. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sidekick.h"

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0;
    for (size_t i = 0; i < n; i++) dot += (double)a[i] * b[i];
    return dot; /* rows are unit-normalized */
}

int main(void) {
    if (sk_abi_version() != SK_ABI_VERSION_EXPECTED) {
        fprintf(stderr, "FAIL: abi version %u\n", sk_abi_version());
        return 1;
    }

    char *err = NULL;
    sk_pool *pool = sk_pool_open(NULL, &err); /* default models dir */
    if (!pool) {
        fprintf(stderr, "FAIL: open: %s\n", err ? err : "?");
        return 1;
    }

    char *models = sk_pool_models(pool, &err);
    if (!models) { fprintf(stderr, "FAIL: models: %s\n", err ? err : "?"); return 1; }
    printf("models: %s\n", models);
    sk_string_free(models);

    char *info = sk_model_info(pool, "embeddinggemma-300m", &err);
    if (!info) { fprintf(stderr, "FAIL: info: %s\n", err ? err : "?"); return 1; }
    printf("gemma info: %s\n", info);
    sk_string_free(info);

    const char *texts[3] = {
        "A cat sat on the mat.",
        "A kitten rested on the rug.",
        "Quarterly financial earnings exceeded expectations.",
    };
    size_t d = 0;
    float *v = sk_embed(pool, "bge-small-en-v1.5", texts, 3, 0, 0, &d, &err);
    if (!v) { fprintf(stderr, "FAIL: bge embed: %s\n", err ? err : "?"); return 1; }
    printf("bge  (%zu d): cat~kitten %.3f  cat~earnings %.3f\n",
           d, cosine(v, v + d, d), cosine(v, v + 2 * d, d));

    /* matryoshka: gemma at requested_dims=256, daemon-compatible semantics */
    size_t dg = 0;
    float *vg = sk_embed(pool, "embeddinggemma-300m", texts, 3, 0, 256, &dg, &err);
    if (!vg) { fprintf(stderr, "FAIL: gemma embed: %s\n", err ? err : "?"); return 1; }
    double norm = sqrt(cosine(vg, vg, dg));
    printf("gemma(%zu d): cat~kitten %.3f  cat~earnings %.3f  |row0|=%.4f\n",
           dg, cosine(vg, vg + dg, dg), cosine(vg, vg + 2 * dg, dg), norm);
    if (dg != 256 || fabs(norm - 1.0) > 1e-4) {
        fprintf(stderr, "FAIL: truncation dims/norm wrong\n");
        return 1;
    }

    /* undeclared dims must error, not truncate silently */
    char *err2 = NULL;
    size_t dz = 0;
    float *bad = sk_embed(pool, "embeddinggemma-300m", texts, 1, 0, 200, &dz, &err2);
    if (bad || !err2) { fprintf(stderr, "FAIL: expected dims error\n"); return 1; }
    printf("dims=200 err: %s\n", err2);
    sk_string_free(err2);

    sk_floats_free(v, 3 * d);
    sk_floats_free(vg, 3 * dg);
    sk_pool_close(pool);
    printf("FFI SMOKE PASSED\n");
    return 0;
}
