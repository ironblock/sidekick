"""probe_intacc.py <hf_src> <refs> <out.json> [--rescale <rescale.json>]

Hypothesis check for int8 builds that fail on some inputs only: if the ANE
convolves with the int8 codes and applies the per-channel scale after, the
accumulator holds y / scale, not y. Records, per eval text and per
projection, max |y / scale| and max (|x| * |w|) / scale, the accumulator's
bound whatever the order of summation (fp32, the rewritten decoder, int8 per-channel
scales from the weights coremltools would see), to compare with the ANE
linear's 2^15 limit (D25) and fp16's 65504."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("refs")
ap.add_argument("out")
ap.add_argument("--rescale", default=None)
ap.add_argument("--rot", type=int, default=None, help="rotate the model (llama_ane.rotate_model) with this seed")
ap.add_argument("--calib", action="store_true", help="the calibration texts instead of the eval texts")
a = ap.parse_args()
torch.set_grad_enabled(False)
refs = Path(a.refs)
meta = json.load(open(refs / "meta.json"))
cal = json.load(open(refs / "calibration.json"))
ids = json.load(open(refs / "eval_ids.json"))
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
if a.rot is not None:
    LA.rotate_model(hf, a.rot)
    cal = json.load(open(refs / f"calibration_rot{a.rot}.json"))
rescale = json.load(open(a.rescale)) if a.rescale else None
PROJ = ["q", "k", "v", "o", "gate", "up", "down"]
stat = {}          # (text, layer, proj) -> max |y/scale|
cur = {"t": 0}


class Rec(nn.Module):
    def __init__(self, conv, key):
        super().__init__()
        self.conv, self.key = conv, key
        w = conv.weight.view(conv.weight.shape[0], -1)
        self.inv = (127.0 / w.abs().amax(1).clamp(min=1e-30)).view(1, -1, 1, 1)

    def forward(self, x):
        y = self.conv(x)
        k = (cur["t"],) + self.key
        l1 = torch.nn.functional.conv2d(x.abs(), self.conv.weight.abs()) * self.inv
        old = stat.get(k, (0.0, 0.0))
        stat[k] = (max(old[0], float((y * self.inv).abs().max())), max(old[1], float(l1.max())))
        return y


def on_layer(i, layer):
    if rescale is not None:
        LA.apply_rescales(layer, rescale.get(str(i), {}))
    for p in PROJ:
        setattr(layer, p, Rec(getattr(layer, p), (i, p)))


def on_hidden(i, t, h):
    cur["t"] = (t + 1) % len(meta["texts"])


if a.calib:
    import texts as T
    from transformers import AutoTokenizer
    cids = T.calib_texts(AutoTokenizer.from_pretrained(a.src)) + T.gptq_texts(AutoTokenizer.from_pretrained(a.src), 8192)
    meta["texts"] = [f"calib{i}" for i in range(len(cids))]
    ids = dict(zip(meta["texts"], cids))
LA.stream(hf, [ids[n] for n in meta["texts"]], cal=cal, on_layer=on_layer, on_hidden=on_hidden, head=False)
out = {}
for ti, name in enumerate(meta["texts"]):
    per = {f"{i}.{p}": v for (t, i, p), v in stat.items() if t == ti}
    worst = sorted(per.items(), key=lambda kv: -kv[1][1])[:6]
    out[name] = {"max_y": max(v[0] for v in per.values()), "max_l1": max(v[1] for v in per.values()),
                 "l1_over_65504": sum(v[1] > 65504 for v in per.values()), "worst_l1": worst, "per": per}
    print("RESULT", name, json.dumps({k: v for k, v in out[name].items() if k != "per"}))
json.dump(out, open(a.out, "w"), indent=1)
