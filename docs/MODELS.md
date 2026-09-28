# Model compatibility registry

What runs on the ANE through sidekick's Core ML encoder path, what doesn't,
and how to tell before spending an afternoon finding out. Every entry here
was measured on real hardware (Apple Silicon, macOS 26; ANE eligibility
re-checked on macOS 27); nothing is extrapolated from model cards.

Method, for every validated entry:
- **parity** — worst-case cosine between the Core ML artifact and the fp32
  torch/sentence-transformers reference over the parity set (short pairs +
  a ~400-token text), reported per compute path (D17): `CPU_ONLY` proves the
  conversion is faithful (gate ≥ 0.999), `CPU_AND_NE` is what the ANE's
  fp16 arithmetic actually delivers (gate ≥ 0.985).
- **ANE eligibility** (`cargo run -p sidekick-coreml --example ane_check`),
  the verdict: Core ML's compute plan for `.cpuAndNeuralEngine`, i.e. which
  device each operation is assigned to. It never runs the model and is
  unaffected by machine load. A model passes when every compute-heavy
  operation (matmul, linear, conv, attention) is on the ANE and at least 80%
  of assigned operations are. The table's "ANE ops" column is that share.
  The plan is the compiler's intent, so it can't see failures that only
  happen at run time. That is what the ratio below is for.
- **residency ratio**, runtime evidence, reported by the same tool: the
  median-latency ratio of `.cpuOnly` over `.cpuAndNeuralEngine` per bucket.
  A ratio near 1.0 on an eligible model points at a runtime fallback, so
  `ane_check` warns below 1.1x. It is not a pass/fail gate, for two reasons:
  - It moves with things other than residency. Concurrent GPU and
    memory-bandwidth load depresses and destabilizes it: the same bge
    artifact under an active MLX workload swung 1.4x–2.8x between runs.
  - A faster CPU path shrinks it on a model that is fully on the ANE. On
    macOS 27, bge-small's 512 bucket measures ~1.25x (it was 1.75x on 26.5)
    with a compute plan identical to its 3x buckets. The old 1.5x ratio gate
    called that bucket "not resident".
  Parity is insensitive to load. Judge ratios only on a quiet machine, and
  judge eligibility by the plan.

## Validated on ANE

| model | dims | pooling | conversion | parity CPU_ONLY | parity ANE | ANE ops (macOS 27) | residency ratio, macOS 26.5 (128/256/512) |
|---|---|---|---|---|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | CLS | [convert_bge_small.py](../tools/convert_bge_small.py) | 0.999972 | 0.999984 | 229/245 (93.5%) | 3.4x / 2.4x / 1.75x |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 768 (MRL 512/256/128) | mean | [convert_embeddinggemma.py](../tools/convert_embeddinggemma.py) | 0.9999 | 0.9905 | 2015/2024 (99.6%) | 3.4x / 3.1x / 2.9x |
| [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) | 1024 | CLS | [convert_lfm25_embedding.py](../tools/convert_lfm25_embedding.py) | 0.9999 | 0.9870 | 693/698 (99.3%) | 2.49x / 1.91x / 1.66x |
| [codefuse-ai/F2LLM-v2-160M](https://huggingface.co/codefuse-ai/F2LLM-v2-160M) | 640 | last-token | [convert_qwen3_embedding.py](../tools/convert_qwen3_embedding.py) | 0.9999 | 0.99985 | 612/617 (99.2%) | 2.02x / 1.77x / 1.59x |

ANE ops are identical at every bucket. On every model, the operations off
the ANE are mask and cast plumbing plus the embedding `gather` (e.g. bge:
add, cast, expand_dims, gather, greater_equal, layer_norm, select, sub,
tile). Nothing compute-heavy is off the ANE. Ratios re-measured on macOS 27
(M1 Max, not a quiet machine):

| model | 128 | 256 | 512 |
|---|---|---|---|
| bge-small | 3.0x | 2.0x | 1.26x |
| embeddinggemma | 2.5x | 2.5x | 1.9x |
| LFM2.5 | 2.4x | 2.0x | 1.7x |
| F2LLM | 2.3x | 2.0x | 1.5x |

Notes per model:

- **bge-small-en-v1.5** — the reference "easy" conversion: BERT-class,
  attention-only, small activations. The recipe (D15) is the template for
  MiniLM/gte/e5-class encoders. Parity re-measured per-path over the shared
  parity set (July 2026, artifacts regenerated); residency from the D15
  quiet-machine measurement of the same recipe. Loads may print one
  `ANECCompile() FAILED` line on stderr while the encoder still resides on
  the ANE (a small ineligible segment). Judge by the ane_check compute plan,
  not stderr. `convert_bge_small.py --enumerated-shapes` builds the
  flexible-shape artifact that D15 rules out, as a negative control:
  `ane_check` must reject it, and does (0 of 362 operations on the ANE).
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
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | late-interaction (multi-vector) | Emits one 128-d vector **per token**, scored with MaxSim — there is no single vector to return through `/v1/embeddings` or `sk_embed`. The encoder itself converts and offloads fine (smoke-tested at seq 256: per-token parity 0.9995 CPU / 0.9919 ANE, 2.0x ANE speedup, MaxSim ranking preserved — [smoke_lfm25_colbert.py](../tools/smoke_lfm25_colbert.py)), so a future late-interaction API could host it; nothing in today's API can. Its padded-batch conv semantics (expansion tokens must NOT be zeroed) also make real-token embeddings bucket-dependent under static shapes. Re-run on macOS 27: unchanged (per-token ANE parity 0.9919, 2.1x over CPU, MaxSim ranking preserved). |
| Apple NLContextualEmbedding | OS-provided contextual | Mean-pooled MLM states, strongly anisotropic (unrelated-pair cosine ~0.75) — unusable for similarity thresholds without post-hoc calibration sidekick doesn't own (D16). Re-measured on macOS 27: same model revision, same cosines (0.96/0.89 vs 0.75), faster (~11 ms). |
| Apple NLEmbedding.sentenceEmbedding | OS-provided static-ish | 2020-era quality, measurably weaker than bge-small on the same pairs; no prefixes, no control over dims (D16). Unchanged on macOS 27 (revision 1; 0.74/0.44 vs 0.14). |
| Apple FoundationModels | LLM | Has **no embedding API at all** (verified against macOS 26 SDK docs/headers, D16) — chat only. Still none in the macOS 27 SDK; 27's Spotlight integration is a search tool for sessions, not vectors. |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) **and the ModernBERT family** (incl. granite-embedding-r2, nomic-modernbert-embed) | ModernBERT encoder | Converts faithfully (Core ML **fp32 parity 1.000000**) and **PyTorch fp16 is perfect (0.999999)** — but **Core ML's ANE fp16 gives only 0.9038**. Root cause: a **massive-activation outlier** (dim 251 reaches ~40000 in the residual stream) dominates every LayerNorm's variance (40000² ≈ 1.6e9), dividing all other dims by ~1400 and crushing them below fp16's between-op storage precision *on the ANE*. PyTorch survives via fp32-internal reductions; the ANE stores fp16 between every op and can't recover them. Forcing sensitive ops to fp32 restores 0.9998 but relocates the graph off the ANE (~41ms, ~5× slower, no ANE benefit); macOS26's newer ANE compiler is identical; a D17 global 1/K range rewrite can't win (K≥156 needed to bound the square, at which point the compensated eps/K² underflows fp16). At 0.90 the space compresses (an unrelated pair rose 0.38→0.53), hurting retrieval. Full diagnosis + reproduction in [convert_gte_modernbert.py](../tools/convert_gte_modernbert.py). **Re-tested on macOS 27 (M1 Max): unchanged.** ANE parity 0.903807 with 98.5% of operations on the ANE (3.2x over CPU): placement is fine, the ANE's fp16 arithmetic is not. Keeping LayerNorm and reductions in fp32 moves all 45 LayerNorms to the CPU; the ANE↔CPU hand-offs make the ANE path slower than CPU-only (0.75x), and parity doesn't improve (0.9039) because the damage is in the fp16 residual stream. The macOS 26 opset behaves identically. Follow-ups measured on macOS 27. A power-of-two range rewrite is exact (CPU parity 1.0) but gives only 0.913/0.929/0.921 at scales 1/8, 1/64 and 1/256. Bisection: keeping only attention in fp32 restores 0.9998, but as a Core ML CPU/ANE hybrid that is slower than CPU-only (0.82x, from 22 hand-offs per inference). |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large encoder + decision head (classifier) | Same signature: dim 379 reaches ~27,300 (556× the median) under 57 LayerNorms. Its encoder on the ANE (seq 128, M1 Max, macOS 27): CLS cosine **0.07**, per-token mean 0.77. The GPU is essentially exact (CLS 0.99999, 19 ms); CPU-only 0.99996 at 47 ms. Also not an embedding model: it scores options through a decision head, so it would need its own endpoint. |

## Quick triage: is a model worth converting?

Most rejections are visible long before a conversion. Cheapest first:

1. **`config.json` (seconds).**
   - `model_type` / `architectures` is the strongest signal. Validated:
     `bert`, `gemma3_text` (with a range rewrite), `lfm2`, `qwen3`. Known
     ANE-hostile: `modernbert`, including checkpoints that wrap it (laya's
     `encoder/config.json`).
   - Also check:
     - **QK-norm** (`q_norm`/`k_norm` in the modeling code): a good sign,
       not a guarantee.
     - **bf16 training**: nothing kept its activations in fp16 range.
     - **Size**: validated up to 350M parameters, and each bucket stores the
       whole model at ~2 bytes/param.
     - **Needed sequence length**: buckets are ≤512, and the ANE's advantage
       shrinks with length.
     - **Output shape**: one vector per input fits `/v1/embeddings`;
       classifiers, rerankers and multi-vector models need a new API.
2. **The modeling code (minutes).** See the checklist below: data-dependent
   shapes, attention that can't run as SDPA, and non-attention token mixers.
3. **`tools/probe_activations.py` (minutes, PyTorch on the CPU, no Core
   ML).** It hooks every normalization layer's input and reports peak
   activation, which dimension carries it, how many times the median
   dimension it is, and which norm type reads it. Calibrated verdicts:

   | model | norms | peak (dim) | × median | probe says | measured on ANE |
   |---|---|---|---|---|---|
   | bge-small-en-v1.5 | LayerNorm | 338 (99) | 146× | compatible | 0.99998 |
   | LFM2.5-Embedding-350M | RMSNorm | 1.9 | 17× | compatible | 0.987 |
   | embeddinggemma-300m | RMSNorm | 152,485 (731) | 313× | range rewrite | 0.9905 after rewrite |
   | gte-modernbert-base | LayerNorm | 47,973 (251) | 502× | hostile | 0.904 |
   | laya (ModernBERT-large) | LayerNorm | 27,296 (379) | 556× | hostile | CLS 0.07 |

   An outlier alone doesn't decide it: EmbeddingGemma's is bigger than
   ModernBERT's and converts fine under RMSNorm. The failing combination is
   a massive activation read by mean-subtracting LayerNorms. With one
   confirmed failure family, LayerNorm peaks between ~1,000 and ~10,000 are
   unknown territory: convert and measure.
4. **Convert and gate (hours):** the gates at the end of the checklist.

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
- **Massive activations under LayerNorm are an ANE killer; under RMSNorm
  they're survivable.** A few feature dimensions reach tens of thousands on a
  few tokens ([SEP], delimiters). ModernBERT's dimension 251 reaches ~48,000.
  Such a model converts faithfully (fp32 parity 1.0) and is perfect in
  PyTorch fp16, yet lands at ~0.90 on the ANE. What's been ruled out:
  - **Weights:** they are fine.
  - **Magnitude:** a power-of-two range rewrite shrinks the peak to 163 and
    still gives at most 0.93.
  - **LayerNorm's own precision:** fp32 LayerNorm changes nothing.

  Keeping only the attention ops in fp32 restores 0.9998, so the loss is
  inside attention on the ANE; the exact mechanism isn't established.
  EmbeddingGemma carries an even larger outlier (~152,000) under RMSNorm and
  converts fine with the range rewrite, so the norm type matters. Test this
  before converting with `tools/probe_activations.py`. After converting,
  compare **CPU_AND_NE against PyTorch fp16** as well as fp32: a gap there
  is the signature.
- **Token mixing other than attention** (convs, SSMs): decide the padding
  semantics explicitly. Attention masks silence pad *keys*, but anything
  convolutional reads pad *states* — zero them per layer if the reference
  is the unpadded forward (LFM2.5 constraint D).
- **Per-bucket artifact size is the whole model** — weights duplicate per
  bucket until multifunction mlprograms land. 350M params ≈ 700 MB × 3
  buckets. Fine on disk, but mind the install footprint.

Gates to pass, in order: fp32 rewrite parity ≥ 0.9999 (only if rewriting),
`CPU_ONLY` ≥ 0.999, `CPU_AND_NE` ≥ 0.985, `ane_check` eligibility OK per
bucket (its ratio should be clearly above 1.0 on a quiet machine), then a
live `/v1/embeddings` parity check.

Flexible input shapes are ruled out (D15), and on macOS 27 they became
dangerous. A single enumerated-shapes artifact used to run slowly on the
CPU; it now aborts the process at the first prediction ("E5RT: No memory
object bound to port"), whatever the compute units. That abort is an
Objective-C exception Rust can't catch. It takes down `sidekickd` or the
host app linking `libsidekick.dylib`. Always ship one static-shape artifact
per bucket, and run `ane_check` on each: it reads the compute plan without
predicting, so it rejects such an artifact instead of crashing.

Two hard-won measurement gotchas: run residency checks on a quiet machine
(see the method note — concurrent GPU load makes ratios swing 2x), and
treat `E5RT ... ANECCompile() FAILED` stderr lines as *possibly transient
service state*, not proof of a bad artifact — the same file measured 1.48x
with failures and 2.63x clean forty minutes apart. Re-measure before
re-converting.
