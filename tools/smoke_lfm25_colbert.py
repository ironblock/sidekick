"""Smoke-test: can LiquidAI/LFM2.5-ColBERT-350M's encoder run on the ANE?

NOT a conversion-and-install script. LFM2.5-ColBERT is a late-interaction
retriever: it emits one 128-d vector PER TOKEN and scores query/document
pairs with MaxSim. That multi-vector output fundamentally does not fit
sidekick's single-vector embeddings API (OpenAI /v1/embeddings returns one
vector per input), so the model is not integrated — see docs/MODELS.md.

What this script answers, for the record:
  1. Does the encoder (LFM2 bidirectional backbone + Dense 1024->128 +
     per-token L2) convert to a static-shape Core ML mlprogram at all?
  2. Is the per-token output faithful (cosine per real token vs torch fp32)?
  3. Does it achieve ANE offload (CPU_ONLY vs CPU_AND_NE latency ratio)?
  4. Does MaxSim ranking survive the conversion (demo, not a benchmark)?

Usage:
    python tools/smoke_lfm25_colbert.py <hf-model-dir> [seq]   # default 256

Shares the LFM2.5 conversion constraints with tools/convert_lfm25_embedding.py
(fp16-safe mask, traceable rotate_half/repeat_kv, explicit position_ids,
static shapes). The per-token output is reshaped to a LITERAL (1, seq, 128) —
static, so D15's "no symbolic dims in outputs" rule is satisfied; whether
Espresso still plan-compiles a rank-3 output for the ANE is exactly what
this smoke test measures.

Unlike the embedding conversion, the conv layers do NOT zero pad states
(install_patches(conv_pad_zeroing=False)): ColBERT's query-expansion tokens
are attention_mask=0 yet must flow through the convs and score in MaxSim —
upstream's modeling file warns that zeroing them "hurts ColBERT MaxSim".
The torch reference here is computed through the same padded wrapper, so
parity below is self-consistent; but note the semantic consequence for any
real static-bucket deployment: without zeroing, real-token embeddings vary
with pad count, i.e. depend on the bucket you picked. That is inherent to
the upstream padded-batch semantics, and one more reason this model is
documented as not integrated (docs/MODELS.md).

Host-side ColBERT semantics (PyLate query expansion to query_length=32 with
mask tokens, punctuation skiplist filtering, per-token masking before
MaxSim) are out of scope here — they live above the encoder either way.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_lfm25_embedding import install_patches  # noqa: E402

TOKEN_DIMS = 128
HIDDEN = 1024

QUERY = "[Q] which animal rested on the rug?"
DOCS = [
    "[D] A kitten rested on the rug all afternoon.",
    "[D] Quarterly financial earnings exceeded expectations.",
]


class ColbertWrapper(torch.nn.Module):
    """Backbone + Dense(1024->128, no bias) + per-token L2, per-token output."""

    def __init__(self, model, dense_w, seq_len):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.register_buffer("dense_w", dense_w)  # (128, 1024)
        self.register_buffer(
            "position_ids", torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
            position_ids=self.position_ids,
        ).last_hidden_state
        tok = torch.nn.functional.linear(hidden, self.dense_w)
        tok = tok.reshape(1, self.seq_len, TOKEN_DIMS)
        # per-token L2; token vectors are small (QK-normed backbone), no
        # overflow-guard scaling needed at 128 dims
        return tok / torch.linalg.vector_norm(tok, dim=-1, keepdim=True)


def encode_np(tokenizer, text, seq_len):
    ids_list = tokenizer(text, add_special_tokens=True)["input_ids"]
    ids = np.zeros((1, seq_len), dtype=np.int32)
    ids[0, : len(ids_list)] = ids_list
    mask = np.zeros((1, seq_len), dtype=np.int32)
    mask[0, : len(ids_list)] = 1
    return ids, mask, len(ids_list)


def maxsim(q_tok, d_tok):
    """MaxSim over real tokens only: sum over query tokens of the max
    similarity against any document token."""
    return float((q_tok @ d_tok.T).max(axis=1).sum())


def main():
    src = Path(sys.argv[1]).expanduser()
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    tokenizer = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(
        src, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()
    model.config.use_cache = False
    install_patches(conv_pad_zeroing=False)  # see docstring: expansion tokens

    dense_w = load_file(src / "1_Dense/model.safetensors")["linear.weight"].float()
    assert dense_w.shape == (TOKEN_DIMS, HIDDEN)
    wrapper = ColbertWrapper(model, dense_w, seq).eval()

    texts = [QUERY] + DOCS
    encs = [encode_np(tokenizer, t, seq) for t in texts]

    # fp32 torch reference (same wrapper, so this tests conversion fidelity)
    refs = []
    with torch.no_grad():
        for ids, mask, n in encs:
            out = wrapper(torch.from_numpy(ids), torch.from_numpy(mask))
            refs.append(out[0, :n].numpy())

    ids0 = torch.zeros((1, seq), dtype=torch.int32)
    ids0[0, 0] = 1
    mask0 = torch.zeros((1, seq), dtype=torch.int32)
    mask0[0, :1] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids0, mask0))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="token_embeddings")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    with tempfile.TemporaryDirectory() as workdir:
        pkg = Path(workdir) / f"colbert_{seq}.mlpackage"
        mlmodel.save(str(pkg))

        for label, cu in (("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                          ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)):
            m = ct.models.MLModel(str(pkg), compute_units=cu)
            outs, worst = [], 1.0
            for (ids, mask, n), ref in zip(encs, refs):
                out = m.predict({"input_ids": ids, "attention_mask": mask})
                tok = out["token_embeddings"][0, :n]
                if not np.isfinite(tok).all():
                    raise SystemExit(f"[{label}] non-finite token embeddings")
                for i in range(n):
                    worst = min(worst, float(np.dot(ref[i], tok[i])))
                outs.append(tok)

            q, d0, d1 = outs
            rq, rd0, rd1 = refs
            t0 = time.perf_counter()
            for _ in range(10):
                m.predict({"input_ids": encs[0][0], "attention_mask": encs[0][1]})
            ms = (time.perf_counter() - t0) / 10 * 1e3
            print(f"[{label}] worst per-token cosine: {worst:.6f}  latency: {ms:.1f}ms")
            print(f"[{label}] MaxSim  relevant doc: {maxsim(q, d0):.3f} (fp32 {maxsim(rq, rd0):.3f})"
                  f"  unrelated doc: {maxsim(q, d1):.3f} (fp32 {maxsim(rq, rd1):.3f})")
            if maxsim(q, d0) <= maxsim(q, d1):
                raise SystemExit(f"[{label}] MaxSim ranking inverted after conversion")


if __name__ == "__main__":
    main()
