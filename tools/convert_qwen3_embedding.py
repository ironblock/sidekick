"""Convert a Qwen3-class causal-decoder embedding model into ANE-resident
Core ML artifacts. Validated on codefuse-ai/F2LLM-v2-160M; the same recipe
covers Qwen3/Qwen3-Embedding-derived last-token embedders (dims read from
config, prefixes read from config_sentence_transformers.json).

Produces one static-shape .mlmodelc per sequence-length bucket, with
LAST-TOKEN pooling baked into the graph, matching the model's manifest.

Usage:
    python tools/convert_qwen3_embedding.py <hf-model-dir> <install-dir> [buckets...]

Requires: torch, transformers >= 4.51 (Qwen3), coremltools, numpy
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

Qwen3 is the fifth architecture class validated on the stack (after classic
BERT / bge, Gemma3 / embeddinggemma, LFM2 hybrid, and — as a negative result
— ModernBERT). It is the first CAUSAL DECODER used for embeddings here. Three
conversion facts, all handled below:

A. CAUSAL + PADDING MASK, fp16-safe. Qwen3Model builds its mask via
   transformers.masking_utils.create_causal_mask, which materializes
   finfo(dtype).min for disallowed positions — -inf in fp16, NaNs softmax on
   the ANE (D15's eager-mask failure class). We patch create_causal_mask (in
   the qwen3 module namespace) to build the identical lower-triangular +
   key-padding mask with the fp16-safe MASK_ADD = -30000. Right-padding is
   correct for a causal model: the last real token attends only to earlier
   real tokens, so trailing pads never affect it.

B. LAST-TOKEN POOLING, in-graph, no data-dependent index. The embedding is
   the hidden state of the last non-pad token. Rather than gather a computed
   index (a traced Int op that crashes coremltools, D17 constraint 8), we
   select it with the attention mask alone: last_onehot = mask * (1 -
   shift_left(mask)) is 1 exactly at the last real position (for right-
   padding), and a masked sum yields (1, dims). Output stays the (1, dims)
   shape the server expects for pooling = "none"; the server L2-normalizes
   in f32.

C. RoPE + GQA shape arithmetic. Stock rotate_half slices with x.shape[-1]//2
   and repeat_kv reshapes from tensor sizes — both trace to Int ops that
   crash coremltools under static shapes. Same chunk(2) / expand+flatten
   replacements as the gemma/LFM recipes.

fp16 note: NO range rewrite. Qwen3's q_norm/k_norm (QK-norm) keep activations
tiny (measured max ~420 on F2LLM-v2-160M), so fp16 is simply safe — the
opposite of ModernBERT, whose LayerNorm-only design let an outlier reach
~40000 and lose the ANE (see docs/MODELS.md). Verified before converting by
checking that PyTorch-fp16 parity is ~1.0 and no activation exceeds ~30k.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModel, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as _qwen3

MASK_ADD = -30000.0

PARITY_SENTENCES = [
    "A cat sat on the mat.",
    "A kitten rested on the rug.",
    "Quarterly financial earnings exceeded expectations.",
    "The company reported strong revenue growth this quarter.",
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]
# (index, is_query) — alternate query/document so both prefix paths are tested
QUERY_FLAGS = [True, False, True, False, False]


def _traceable_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _traceable_repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    b, kvh, s, d = hidden_states.shape
    return hidden_states.unsqueeze(2).expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


def _fp16_safe_causal_mask(config=None, input_embeds=None, attention_mask=None, **kw):
    # constraint A: lower-triangular (causal) AND key-not-pad, additive with
    # MASK_ADD. Batch-1 static seq; returns (bsz, 1, seq, seq).
    embeds = input_embeds
    seq = embeds.shape[1]
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.float32))  # 1 where k<=q
    if attention_mask is not None:
        keep = attention_mask.to(torch.float32)                      # (bsz, seq) 1=real
        allowed = causal.unsqueeze(0) * keep[:, None, :]             # (bsz, q, k)
    else:
        allowed = causal.unsqueeze(0)
    add = (1.0 - allowed).unsqueeze(1) * MASK_ADD                    # (bsz, 1, q, k)
    return add.to(embeds.dtype)


def install_patches():
    _qwen3.rotate_half = _traceable_rotate_half
    _qwen3.repeat_kv = _traceable_repeat_kv
    _qwen3.create_causal_mask = _fp16_safe_causal_mask


class LastTokenWrapper(torch.nn.Module):
    """Last-non-pad-token pooling via the attention mask, reshaped to
    (1, dims). No in-graph L2 — the server normalizes in f32."""

    def __init__(self, model, seq_len, dims):
        super().__init__()
        self.model = model
        self.dims = dims
        self.register_buffer(
            "position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=self.position_ids,
        ).last_hidden_state
        mask_f = attention_mask.to(hidden.dtype)                    # (1, seq)
        shifted = F.pad(mask_f[:, 1:], (0, 1), value=0.0)          # mask[i+1], last=0
        last_onehot = mask_f * (1.0 - shifted)                     # 1 at last real pos
        pooled = (last_onehot.unsqueeze(-1) * hidden).sum(dim=1)   # (1, dims)
        return pooled.reshape(1, self.dims)


def load_prompts(src):
    cfg = json.load(open(Path(src) / "config_sentence_transformers.json"))
    p = cfg.get("prompts", {})
    return p.get("query", ""), p.get("document", "")


def full_text(text, is_query, qprefix, dprefix):
    return (qprefix if is_query else dprefix) + text


def reference_embeddings(model, tokenizer, qprefix, dprefix):
    """fp32 last-token references (unnormalized; cosine is scale-invariant),
    verified equal to SentenceTransformer.encode at cosine 1.0."""
    refs = []
    with torch.no_grad():
        for s, is_q in zip(PARITY_SENTENCES, QUERY_FLAGS):
            enc = tokenizer(full_text(s, is_q, qprefix, dprefix),
                            return_tensors="pt", truncation=True, max_length=MAX_BUCKET)
            refs.append(model(**enc).last_hidden_state[0, -1, :].numpy())
    return refs


MAX_BUCKET = 512  # references are computed at this truncation; the server
# routes any over-length input to the largest bucket, preserving the final
# (EOS) token — see load/truncation in coreml_embedder.rs and constraint B.


def padded_inputs(tokenizer, text, seq_len):
    # Truncate the SAME way the server does: HF right-truncation to MAX_BUCKET
    # keeps [first MAX_BUCKET-1 content, EOS], matching the last-token-safe
    # server truncation. fitting_cases guarantees the result fits seq_len.
    ids_list = tokenizer(text, add_special_tokens=True, truncation=True,
                         max_length=MAX_BUCKET)["input_ids"]
    if len(ids_list) > seq_len:
        raise SystemExit(f"parity text longer than bucket {seq_len}")
    ids = np.zeros((1, seq_len), dtype=np.int32)  # right-pad id 0, as the server pads
    ids[0, : len(ids_list)] = ids_list
    mask = np.zeros((1, seq_len), dtype=np.int32)
    mask[0, : len(ids_list)] = 1
    return ids, mask


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fitting_cases(tokenizer, refs, seq_len, qprefix, dprefix):
    # Test each text at buckets >= its server-routed length (MAX_BUCKET-
    # truncated). The long text lands at 512, exercising positions up to 511
    # — the full-bucket / high-RoPE-position path a short-only gate misses.
    out = []
    for s, is_q, ref in zip(PARITY_SENTENCES, QUERY_FLAGS, refs):
        t = full_text(s, is_q, qprefix, dprefix)
        n = len(tokenizer(t, add_special_tokens=True, truncation=True,
                          max_length=MAX_BUCKET)["input_ids"])
        if n <= seq_len:
            out.append((t, ref))
    return out


def convert_bucket(wrapper, seq_len, workdir):
    ids = torch.zeros((1, seq_len), dtype=torch.int32)
    ids[0, 0] = 151643  # any real token; content irrelevant to tracing
    mask = torch.zeros((1, seq_len), dtype=torch.int32)
    mask[0, :1] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    pkg = Path(workdir) / f"model_{seq_len}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(tokenizer, pkg, seq_len, refs, qprefix, dprefix):
    results = {}
    for label, cu, gate in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE, 0.985),
                            ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY, 0.999)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        worst = 1.0
        for t, ref in fitting_cases(tokenizer, refs, seq_len, qprefix, dprefix):
            ids, mask = padded_inputs(tokenizer, t, seq_len)
            out = m.predict({"input_ids": ids, "attention_mask": mask})["embedding"][0]
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq_len} [{label}]: non-finite output — see constraint A")
            worst = min(worst, cosine(ref, out))
        if worst < gate:
            raise SystemExit(f"seq {seq_len} [{label}]: parity cosine {worst:.6f} < {gate}")
        t0text = full_text(PARITY_SENTENCES[0], QUERY_FLAGS[0], qprefix, dprefix)
        ids, mask = padded_inputs(tokenizer, t0text, seq_len)
        for _ in range(3):
            m.predict({"input_ids": ids, "attention_mask": mask})
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            m.predict({"input_ids": ids, "attention_mask": mask})
        results[label] = (worst, (time.perf_counter() - t0) / n * 1e3)
    return results


def compile_to_mlmodelc(pkg, install_dir, seq_len):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True)
        compiled = next(Path(tmp).glob("*.mlmodelc"))
        dest = install_dir / f"model_{seq_len}.mlmodelc"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
    return dest


def main():
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    model.config.use_cache = False
    install_patches()
    dims = model.config.hidden_size
    qprefix, dprefix = load_prompts(src)
    print(f"dims={dims} qprefix={qprefix[:40]!r} dprefix={dprefix!r}")

    refs = reference_embeddings(model, tokenizer, qprefix, dprefix)

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            wrapper = LastTokenWrapper(model, seq, dims).eval()
            pkg = convert_bucket(wrapper, seq, workdir)
            res = parity_check(tokenizer, pkg, seq, refs, qprefix, dprefix)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            for label, (cos, ms) in res.items():
                print(f"bucket {seq} [{label}]: parity cos={cos:.6f} {ms:.1f}ms")
            print(f"bucket {seq} -> {dest}")

    shutil.copy(src / "tokenizer.json", install_dir / "tokenizer.json")
    repo_manifest = (
        Path(__file__).resolve().parent.parent
        / f"examples/manifests/{install_dir.name}/manifest.toml"
    )
    shutil.copy(repo_manifest, install_dir / "manifest.toml")
    print(f"installed manifest + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
