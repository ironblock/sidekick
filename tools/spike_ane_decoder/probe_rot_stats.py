"""probe_rot_stats.py <hf_src> <out.json> [--seed 0]

What does incoherence processing (llama_ane.rotate_model: R1 on the
residual stream, R2 per head on v/o) do to the weights' outliers? For every
projection class, the distribution over output channels of max|w| / rms(w)
(the quantity that sets a per-channel grid's step: 4-bit round-to-nearest
error ~ (max/rms) / (7 * sqrt(12))), before and after rotation, with the
same RMSNorm folds in both so only the rotation differs. Also the head."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_grad_enabled(False)
PROJ = ["q", "k", "v", "o", "gate", "up", "down"]


def ratio(W):
    W = W.view(W.shape[0], -1).float()
    return W.abs().amax(1) / W.pow(2).mean(1).sqrt()


def summary(rs):
    r = torch.cat(rs)
    return {"median": round(float(r.median()), 2), "p99": round(float(torch.quantile(r, 0.99)), 2),
            "max": round(float(r.max()), 2)}


hf = LA.load_hf(a.src, dtype=torch.bfloat16)
L = hf.config.num_hidden_layers
before = {p: [] for p in PROJ}
for i in range(L):
    layer = LA.build_layer(hf, i)
    with torch.no_grad():               # fold the norm weights only, so the comparison isolates the rotation
        for norm, readers in ((layer.n1, ("q", "k", "v")), (layer.n2, ("gate", "up"))):
            g = norm.weight.view(1, -1, 1, 1)
            for p in readers:
                getattr(layer, p).weight.mul_(g)
    for p in PROJ:
        before[p].append(ratio(getattr(layer, p).weight))
head_before = ratio(hf.lm_head.weight.float() * hf.model.norm.weight.float())
LA.rotate_model(hf, a.seed)
after = {p: [] for p in PROJ}
for i in range(L):
    layer = LA.build_layer(hf, i)
    for p in PROJ:
        after[p].append(ratio(getattr(layer, p).weight))
res = {p: {"before": summary(before[p]), "after": summary(after[p])} for p in PROJ}
res["head"] = {"before": summary([head_before]), "after": summary([ratio(hf.lm_head.weight)])}
for k, v in res.items():
    print("RESULT", k, json.dumps(v))
json.dump(res, open(a.out, "w"), indent=1)
