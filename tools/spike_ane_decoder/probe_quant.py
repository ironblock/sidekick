"""probe_quant.py <hf_src> <refs> <out.json> [--quant int8pc] [--only q,k,v,o,gate,up,down,head]
                 [--layers 0-41] [--local] [--rescale <rescale.json>]

Where does weight quantisation lose fidelity? Simulated in PyTorch (fp32
arithmetic, the checkpoint in bf16, one layer at a time: llama_ane.stream),
so it separates the quantisation's own loss from the ANE's arithmetic and
needs no Core ML compile.

- End to end: the rewritten decoder (reference.py's calibration) with the
  chosen projections' weights quantised, teacher-forced over the eval texts,
  scored as spike.py scores a build: top-1 agreement and logit cosine
  against the fp32 checkpoint.
- --local: per layer and projection, the rms of the projection's input (the
  ANE linear's small-input floor, D17/D19) and the relative error the
  quantised weight adds to its output, given the exact input; nothing is
  propagated.

Quantisers (weights [out, in], as coremltools sees a 1x1 conv's weight):
  int8pc        linear symmetric int8, one fp16 scale per output channel
  int8b<N>      the same per block of N input channels
  pal8g<N>      8-bit k-means palette per group of N output channels
  int8pc-mse    per output channel, clipping chosen to minimise the squared
                error (a scale coremltools can't pick: shows the headroom)"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("refs")
ap.add_argument("out")
ap.add_argument("--quant", default="int8pc")
ap.add_argument("--only", default="q,k,v,o,gate,up,down,head")
ap.add_argument("--layers", default=None)
ap.add_argument("--local", action="store_true")
ap.add_argument("--rescale", default=None)
ap.add_argument("--texts", type=int, default=8)
ap.add_argument("--sat", type=float, default=0.0,
                help="simulate an accumulator that saturates at +-SAT in units of the int8 scale (y / scale)")
ap.add_argument("--keep", default="", help="with --qweights: layer.proj left exact (fp16-keep)")
ap.add_argument("--qweights", default=None, help="gptq.py output: use its weights (then --quant none)")
a = ap.parse_args()
torch.set_grad_enabled(False)

from probe_quant_lib import PROJ, f16, quantizer  # noqa: E402

Q = quantizer(a.quant)
only = set(a.only.split(","))
lsel = None
if a.layers:
    lsel = set()
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-")
        lsel |= set(range(int(lo), int(hi or lo) + 1))

refs = Path(a.refs)
meta = json.load(open(refs / "meta.json"))
cal = json.load(open(refs / "calibration.json"))
ids = json.load(open(refs / "eval_ids.json"))
names = meta["texts"][:a.texts]
p, n = meta["p"], meta["n"]
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
rescale = json.load(open(a.rescale)) if a.rescale else None


class Probe(nn.Module):
    """A 1x1 conv that records its input rms and the quantised weight's
    relative output error; returns the exact or the quantised output."""

    def __init__(self, conv, qw, rec, propagate):
        super().__init__()
        self.conv, self.rec, self.propagate = conv, rec, propagate
        self.qw = qw

    def forward(self, x):
        y = self.conv(x)
        self.rec["rms_in"].append(float(x.pow(2).mean().sqrt()))
        if self.qw is None:
            return y
        yq = torch.nn.functional.conv2d(x, self.qw)
        if a.sat:
            inv = (127.0 / self.qw.view(self.qw.shape[0], -1).abs().amax(1).clamp(min=1e-30)).view(1, -1, 1, 1)
            yq = torch.clamp(yq * inv, -a.sat, a.sat) / inv
        self.rec["rel_err"].append(float((yq - y).norm() / y.norm()))
        return yq if self.propagate else y


local = {}


qmeta = json.load(open(Path(a.qweights) / "meta.json")) if a.qweights else None
keep = set(a.keep.split(",")) - {""}


def on_layer(i, layer):
    if qmeta is not None:
        import gptq
        qw = torch.load(Path(a.qweights) / f"layer_{i}.pt")
        for name, (q, s) in qw.items():
            if name not in only or f"{i}.{name}" in keep:
                continue
            conv = getattr(layer, name)
            conv.weight.data = gptq.dequant(q, s, qmeta["block"]).view_as(conv.weight)
    if rescale is not None:
        LA.apply_rescales(layer, rescale.get(str(i), {}))
    for name in PROJ:
        conv = getattr(layer, name)
        doq = Q is not None and name in only and (lsel is None or i in lsel)
        qw = None
        if doq or a.local:
            w = conv.weight.view(conv.weight.shape[0], -1)
            qw = Q(w).view_as(conv.weight)
        rec = local.setdefault(f"{i}.{name}", {"rms_in": [], "rel_err": []})
        setattr(layer, name, Probe(conv, qw, rec, propagate=doq and not a.local))


t0 = time.time()
texts = [ids[nm] for nm in names]
if qmeta is not None and qmeta["head"] and "head" in only:
    import gptq
    q, s = torch.load(Path(a.qweights) / "head.pt")["head"]
    hf.lm_head.weight.data = gptq.dequant(q, s, qmeta["block"])
elif Q is not None and "head" in only and not a.local:
    # fp32 storage: bf16 would round the quantised head again
    hf.lm_head.weight.data = Q(hf.lm_head.weight.data.float())
logits = LA.stream(hf, texts, cal=cal, on_layer=on_layer, rows=(p - 1, n))
agree, tot, coss = 0, 0, []
per_text = {}
for nm, lg in zip(names, logits):
    ref = torch.from_numpy(np.load(refs / f"eval_{nm}.npy"))
    cos = (lg * ref).sum(1) / lg.norm(dim=1) / ref.norm(dim=1)
    top = (lg.argmax(1) == ref.argmax(1))
    agree += int(top.sum())
    tot += len(top)
    coss += cos.tolist()
    per_text[nm] = {"top1": float(top.float().mean()), "cos_min": float(cos.min())}
c = np.array(coss)
res = {"keep": a.keep, "sat": a.sat, "qweights": a.qweights, "quant": a.quant, "only": sorted(only), "layers": a.layers, "rescale": a.rescale, "local": a.local,
       "top1": agree / tot, "positions": tot, "cos_mean": float(c.mean()), "cos_min": float(c.min()),
       "cos_p1": float(np.percentile(c, 1)), "texts": per_text, "seconds": round(time.time() - t0)}
if a.local:
    res["per_proj"] = {k: {"rms_in_min": min(v["rms_in"]), "rms_in_mean": float(np.mean(v["rms_in"])),
                           "rel_err_mean": float(np.mean(v["rel_err"])) if v["rel_err"] else None,
                           "rel_err_max": max(v["rel_err"]) if v["rel_err"] else None}
                       for k, v in local.items()}
json.dump(res, open(a.out, "w"), indent=1)
print("RESULT", json.dumps({k: res[k] for k in ("qweights", "quant", "only", "layers", "top1", "cos_min", "cos_p1", "seconds")}))
