"""autoround_export.py <hf_src> <out_dir> [--bits 4] [--iters 50] [--nsamples 128] [--seqlen 512] [--threads 12]

Intel AutoRound (SignRound) weight-only quantisation, symmetric, one scale
per output channel (group_size -1), tuned on CPU, exported in gptq.py's
qweights format so convert.py and probe_quant.py take it unchanged:
each projection becomes a palette per output channel (group 1) holding that
channel's grid values, codes uint8. A per-channel symmetric grid has at most
2^bits values, so the palettizer's "unique" mode stores it exactly on any
bit width, without re-deriving AutoRound's scales (its tuned clipping means
a channel's max code need not reach the grid's end, which would break a
linear re-quantisation).

Calibration: WikiText-2 raw (train split, CC BY-SA), which shares nothing
with the fidelity texts (spans of this repository). The head is quantised
too (quant_lm_head). Weights are taken as the checkpoint's [out, in], with the converter's own
whole-matrix power-of-two rewrites applied (up_proj's 1/2 for StableSilu;
the residual 1/K is 1 for MiniCPM5-2B); norm pre-scales act on inputs at
run time, so the grid survives them."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--nsamples", type=int, default=128)
ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()
torch.set_num_threads(a.threads)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(torch, "xpu"):
    # Intel Macs top out at torch 2.2, which predates torch.xpu; AutoRound only asks whether one exists
    class _NoXpu:
        def __getattr__(self, name):          # is_available, device_count, _is_in_bad_fork, ...: all "no"
            return lambda *args, **kwargs: 0

    torch.xpu = _NoXpu()
from auto_round import AutoRound

tok = AutoTokenizer.from_pretrained(a.src)
model = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.float32).eval()
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in ds["text"] if len(t) > 400][:4 * a.nsamples]
t0 = time.time()
ar = AutoRound(model, tok, bits=a.bits, group_size=-1, sym=True, iters=a.iters, nsamples=a.nsamples,
               seqlen=a.seqlen, dataset=texts, device_map="cpu", quant_lm_head=True,
               batch_size=1, gradient_accumulate_steps=8, low_gpu_mem_usage=True)
ar.quantize()
tune_s = time.time() - t0
print(f"tuning took {tune_s:.0f} s", flush=True)
qm = ar.model

# fake-quantised weights: recover each channel's grid values (<= 2^bits) as a per-channel palette
PROJ = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
K = 2 ** a.bits


def to_lut(W):
    W = W.detach().float().half().float()            # the values Core ML will hold (fp16)
    vals, codes = [], []
    for r in range(W.shape[0]):
        u, inv = torch.unique(W[r], return_inverse=True)
        if len(u) > K:
            raise SystemExit(f"a channel holds {len(u)} values > {K}: not a {a.bits}-bit grid")
        lut = torch.cat([u, u[-1:].expand(K - len(u))])
        vals.append(lut)
        codes.append(inv.to(torch.uint8))
    return torch.stack(codes), torch.stack(vals)


def weight_of(mod):
    # AutoRound leaves fake-quantised Linear weights in place (or a qdq'd copy in .weight)
    return mod.weight


layers = qm.model.layers
for i, layer in enumerate(layers):
    res = {}
    for p, path in PROJ.items():
        mod = layer.get_submodule(path)
        w = weight_of(mod)
        # llama_ane's StableSilu returns 2 * silu, so up_proj carries 1/2 (a power of two: the grid survives)
        res[p] = to_lut(w * 0.5 if p == "up" else w)
    torch.save(res, out / f"layer_{i}.pt")
torch.save({"head": to_lut(weight_of(qm.lm_head))}, out / "head.pt")
json.dump({"block": 0, "grid": f"autoround-w{a.bits}", "group": 1, "head": True, "tokens": a.nsamples * a.seqlen,
           "iters": a.iters, "tuning_s": round(tune_s), "calibration": "wikitext-2-raw-v1 train"},
          open(out / "meta.json", "w"), indent=1)
print("done", flush=True)
