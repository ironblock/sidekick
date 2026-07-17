# Model compatibility registry

What runs on the ANE through sidekick's Core ML encoder path, what doesn't,
and how to tell before spending an afternoon finding out. Every entry here
was measured on real hardware (Apple Silicon, macOS 26); nothing is
extrapolated from model cards.

Method, for every validated entry:
- **parity** — worst-case cosine between the Core ML artifact and the fp32
  torch/sentence-transformers reference over the parity set (short pairs +
  a ~400-token text), reported per compute path (D17): `CPU_ONLY` proves the
  conversion is faithful (gate ≥ 0.999), `CPU_AND_NE` is what the ANE's
  fp16 arithmetic actually delivers (gate ≥ 0.985).
- **residency** — `cargo run -p sidekick-coreml --example ane_check`:
  median-latency ratio of `.cpuOnly` over `.cpuAndNeuralEngine` per bucket.
  A ratio near 1.0 means the plan silently fell back to CPU — the failure
  mode this whole table exists to catch. Ratios are from a quiet machine:
  concurrent GPU/memory-bandwidth load (a local LLM, overnight media
  indexing) measurably depresses and destabilizes them — re-measuring the
  same bge artifact under an active MLX workload swung 1.4x–2.8x between
  runs. Parity is insensitive to load; judge residency only when quiet.

## Validated on ANE

| model | dims | pooling | conversion | parity CPU_ONLY | parity ANE | residency (128/256/512) |
|---|---|---|---|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | CLS | [convert_bge_small.py](../tools/convert_bge_small.py) | 0.999972 | 0.999984 | 3.4x / 2.4x / 1.75x |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 768 (MRL 512/256/128) | mean | [convert_embeddinggemma.py](../tools/convert_embeddinggemma.py) | 0.9999 | 0.9905 | 3.4x / 3.1x / 2.9x |
| [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | 1024 | CLS | [convert_lfm25_embedding.py](../tools/convert_lfm25_embedding.py) | 0.9999 | 0.9870 | 2.49x / 1.91x / 1.66x |
| [codefuse-ai/F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 640 | last-token | [convert_qwen3_embedding.py](../tools/convert_qwen3_embedding.py) | 0.9999 | 0.99985 | 2.02x / 1.77x / 1.59x |

Notes per model:

- **bge-small-en-v1.5** — the reference "easy" conversion: BERT-class,
  attention-only, small activations. The recipe (D15) is the template for
  MiniLM/gte/e5-class encoders. Parity re-measured per-path over the shared
  parity set (July 2026, artifacts regenerated); residency from the D15
  quiet-machine measurement of the same recipe. Loads may print one
  `ANECCompile() FAILED` line on stderr while the encoder still resides on
  the ANE (a small ineligible segment) — judge by the ane_check ratio, not
  stderr.
- **embeddinggemma-300m** — the "hard" conversion: needed a calibrated fp16
  range rewrite, hand-built sliding-window band masks, and traceable
  rotate_half/repeat_kv (D17). The ~1% ANE parity cost is intrinsic fp16
  accumulation across 24 layers, not a conversion defect; rank order in
  similarity tests is preserved with wide margins. ~600 MB per bucket.
- **LFM2.5-Embedding-350M** — the first hybrid (10 short-conv + 6
  full-attention blocks) and the model that motivated conversion
  constraint D: symmetric convs mix neighbors regardless of attention
  mask, so pad states must be zeroed before every conv or right-padding
  contaminates real tokens (measured 0.905 parity without the fix, 0.987
  with it — the worst-case ANE cosine agrees to all six printed decimals
  across buckets, the fix's bucket-invariance holding on the path where
  the leak was measured; CPU parity varies in the 5th decimal).
  QK-norm keeps activations tiny (max ~25),
  so no range rewrite is needed despite the model being deeper than bge.
  Ships custom code (`modeling_lfm2_bidirectional.py`, ~140 benign lines —
  read before trusting). Live `/v1/embeddings` worst parity 0.9856 (a
  483-token text); ~670 MB per bucket, 2.0 GB installed.
- **F2LLM-v2-160M** — the first **causal decoder** and first **last-token
  pooling** on the stack. A Qwen3 decoder; its QK-norm keeps activations
  tiny (max ~420), so it converts as cleanly as bge (ANE parity 0.99985) —
  the exact opposite of ModernBERT, and the reason we chose it after that
  failure. Last-token pooling is baked in-graph via the attention mask
  (no data-dependent index): `last_onehot = mask · (1 − shift_left(mask))`,
  then a masked sum. Validating it surfaced and fixed a real server bug:
  naive `take(max)` truncation dropped the trailing EOS that last-token
  pooling reads, collapsing over-length-doc parity to 0.36 — the server now
  preserves the final token on truncation (harmless for CLS/mean).
  ~950 MB installed; a 640-dim decoder for ~0.95 GB.

## Incompatible / not integrated

| model | class | why |
|---|---|---|
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | late-interaction (multi-vector) | Emits one 128-d vector **per token**, scored with MaxSim — there is no single vector to return through `/v1/embeddings` or `sk_embed`. The encoder itself converts and offloads fine (smoke-tested at seq 256: per-token parity 0.9995 CPU / 0.9919 ANE, 2.0x ANE speedup, MaxSim ranking preserved — [smoke_lfm25_colbert.py](../tools/smoke_lfm25_colbert.py)), so a future late-interaction API could host it; nothing in today's API can. Its padded-batch conv semantics (expansion tokens must NOT be zeroed) also make real-token embeddings bucket-dependent under static shapes. |
| Apple NLContextualEmbedding | OS-provided contextual | Mean-pooled MLM states, strongly anisotropic (unrelated-pair cosine ~0.75) — unusable for similarity thresholds without post-hoc calibration sidekick doesn't own (D16). |
| Apple NLEmbedding.sentenceEmbedding | OS-provided static-ish | 2020-era quality, measurably weaker than bge-small on the same pairs; no prefixes, no control over dims (D16). |
| Apple FoundationModels | LLM | Has **no embedding API at all** (verified against macOS 26 SDK docs/headers, D16) — chat only. |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) **and the ModernBERT family** (incl. granite-embedding-r2, nomic-modernbert-embed) | ModernBERT encoder | Converts faithfully (Core ML **fp32 parity 1.000000**) and **PyTorch fp16 is perfect (0.999999)** — but **Core ML's ANE fp16 gives only 0.9038**. Root cause: a **massive-activation outlier** (dim 251 reaches ~40000 in the residual stream) dominates every LayerNorm's variance (40000² ≈ 1.6e9), dividing all other dims by ~1400 and crushing them below fp16's between-op storage precision *on the ANE*. PyTorch survives via fp32-internal reductions; the ANE stores fp16 between every op and can't recover them. Forcing sensitive ops to fp32 restores 0.9998 but relocates the graph off the ANE (~41ms, ~5× slower, no ANE benefit); macOS26's newer ANE compiler is identical; a D17 global 1/K range rewrite can't win (K≥156 needed to bound the square, at which point the compensated eps/K² underflows fp16). At 0.90 the space compresses (an unrelated pair rose 0.38→0.53), hurting retrieval. Full diagnosis + reproduction in [convert_gte_modernbert.py](../tools/convert_gte_modernbert.py). |

## Will a new model convert? A checklist

Read the model's `modeling_*.py` before anything else. The recipe survives:

- **Encoder-style, single-vector output** — bidirectional attention (or a
  published bidirectional patch), CLS or mean pooling; or a causal decoder
  with last-token pooling (F2LLM). Multi-vector, and rerank heads don't fit
  the API; generative models have no pooled vector at all.
- **Last-token pooling has two traps** — (1) select the last real token
  in-graph via the mask (`mask · (1 − shift_left(mask))`, masked sum), never
  a data-dependent gather index; (2) the embedding IS the trailing EOS
  token, so truncation must preserve it — the server keeps `[first max-1,
  last]` for exactly this reason. A short-input-only parity gate misses
  both; test an over-length input that fills the largest bucket.
- **SDPA-capable attention** — the conversion forces `sdpa`; eager-only
  mask code tends to materialize -inf constants that NaN in fp16 (D15).
- **Static-shape-friendly graph** — no data-dependent shapes. Stock
  `rotate_half`/`repeat_kv` and any `F.conv1d(padding=shape-derived)`
  need the traceable rewrites (D17 constraint 8, LFM2.5 constraint B).
- **fp16-safe activations** — calibrate first (forward hooks, max |activation|
  on a mixed corpus). Under ~30k: convert directly (bge, LFM2.5). Over:
  apply the D17 power-of-two range rewrite (gemma). Watch for `-1e9` mask
  constants (rewrite at -30000) and rmsnorm eps below ~1e-4.
- **Massive-activation outliers are an ANE killer, and calibration alone
  won't warn you** — a *single* feature dimension in the tens of thousands
  (common in models trained without QK-norm; ModernBERT's dim 251 hits 40k)
  passes an fp16 convert, matches in fp32, and is even perfect in *PyTorch*
  fp16 — then lands at ~0.90 on the ANE, because that dim dominates every
  LayerNorm/RMSNorm variance and crushes the rest below the ANE's fp16
  between-op storage. A global range rewrite can't fix it (it's the outlier's
  *ratio* to other dims, not the absolute scale). QK-norm models (LFM2.5,
  Qwen3) avoid it by construction; LayerNorm-only models (ModernBERT) are the
  risk. Test this specifically: compare **CPU_AND_NE vs PyTorch-fp16** parity,
  not just vs fp32 — a gap there is the outlier signature.
- **Token mixing other than attention** (convs, SSMs): decide the padding
  semantics explicitly. Attention masks silence pad *keys*, but anything
  convolutional reads pad *states* — zero them per layer if the reference
  is the unpadded forward (LFM2.5 constraint D).
- **Per-bucket artifact size is the whole model** — weights duplicate per
  bucket until multifunction mlprograms land. 350M params ≈ 700 MB × 3
  buckets. Fine on disk, but mind the install footprint.

Gates to pass, in order: fp32 rewrite parity ≥ 0.9999 (only if rewriting),
`CPU_ONLY` ≥ 0.999, `CPU_AND_NE` ≥ 0.985, `ane_check` ratio comfortably
above 1.5x per bucket, then a live `/v1/embeddings` parity check.

Two hard-won measurement gotchas: run residency checks on a quiet machine
(see the method note — concurrent GPU load makes ratios swing 2x), and
treat `E5RT ... ANECCompile() FAILED` stderr lines as *possibly transient
service state*, not proof of a bad artifact — the same file measured 1.48x
with failures and 2.63x clean forty minutes apart. Re-measure before
re-converting.
