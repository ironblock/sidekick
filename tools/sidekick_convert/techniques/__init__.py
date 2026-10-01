"""Techniques: small, independent pieces of conversion know-how.

Each module states the limit it works around and where it was measured
(docs/DECISIONS.md), and docs/CONVERTING.md catalogs them:
- masks: finite additive masks, bands, causal, self-attending pad queries;
- attention: explicit attention (never the fused op), matmul softmax (opt-in);
- activations: GELU/SiLU from tanh or erf (the native ops are coarse on the ANE);
- precision: the ANE linear's small-input floor, power-of-two rescales;
- saturation: the ANE linear's 2^15 output limit, residual K;
- pooling: in-graph pooling within fp16 range;
- reduce: blocked max (macOS 27 CPU reduce_max bug);
- traceable: rotate_half/repeat_kv without shape arithmetic;
- onehot: selections from int32 inputs without gathers;
- relative_shift: relative-position terms by query-key distance without gathers.
"""
