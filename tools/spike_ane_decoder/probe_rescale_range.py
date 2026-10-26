"""probe_rescale_range.py <hf_src> <refs> <rescale.json> <out.json> [--rot 0] [--limit 16384]

Before applying D17/D19 input rescales (llama_ane.apply_rescales), check
the ranges they create: per layer, max |input| of o_proj and down_proj and
max |output| of v_proj and up_proj, times the planned scale, over the eval
and calibration texts. Lowers any scale whose scaled maximum would pass
--limit (headroom under the ANE linear's 2^15) and writes the safe spec."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA
import texts as T

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("refs")
ap.add_argument("rescale")
ap.add_argument("out")
ap.add_argument("--rot", type=int, default=None)
ap.add_argument("--limit", type=float, default=16384.0)
a = ap.parse_args()
torch.set_grad_enabled(False)
refs = Path(a.refs)
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
cal = json.load(open(refs / "calibration.json"))
if a.rot is not None:
    LA.rotate_model(hf, a.rot)
    cal = json.load(open(refs / f"calibration_rot{a.rot}.json"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.src)
ids = json.load(open(refs / "eval_ids.json"))
texts = list(ids.values()) + T.calib_texts(tok)
spec = json.load(open(a.rescale))
mx = {}


class Rec(nn.Module):
    def __init__(self, conv, key, side):
        super().__init__()
        self.conv, self.key, self.side = conv, key, side

    def forward(self, x):
        y = self.conv(x)
        v = float((x if self.side == "in" else y).abs().max())
        mx[self.key] = max(mx.get(self.key, 0.0), v)
        return y


def on_layer(i, layer):
    for p, side in (("o", "in"), ("down", "in"), ("v", "out"), ("up", "out")):
        setattr(layer, p, Rec(getattr(layer, p), f"{i}.{p}", side))


LA.stream(hf, texts, cal=cal, on_layer=on_layer, head=False)
safe, report = {}, {}
for i in range(hf.config.num_hidden_layers):
    s = dict(spec.get(str(i), {}))
    for p, other in (("o", "v"), ("down", "up")):
        if p not in s:
            continue
        while s[p] > 1 and max(mx[f"{i}.{p}"], mx[f"{i}.{other}"]) * s[p] > a.limit:
            s[p] //= 2
        if s[p] == 1:
            del s[p]
    report[i] = {k: round(v, 1) for k, v in mx.items() if k.startswith(f"{i}.")}
    if s:
        safe[str(i)] = s
json.dump(safe, open(a.out, "w"))
lowered = {k: (spec.get(k), safe.get(k)) for k in spec if spec.get(k) != safe.get(k)}
print("RESULT", json.dumps({"layers": len(safe), "lowered": lowered}))
