"""lut_qweights.py <hf_src> <calibration.json> <out_dir> --layers 0-14 [--bits 8] [--group 16]

Round-to-nearest palette weights (gptq.lloyd_lut, min/max pinned) in
gptq.py's qweights format, for chosen layers only and in seconds: a build
with the same layout and byte count as a GPTQ palette build, for timing
probes where fidelity doesn't matter."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gptq as G
import llama_ane as LA

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("cal")
ap.add_argument("out")
ap.add_argument("--layers", required=True)
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--group", type=int, default=16)
a = ap.parse_args()
torch.set_grad_enabled(False)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
cal = json.load(open(a.cal))
lo, hi = map(int, a.layers.split("-"))
for i in range(lo, hi):
    layer = LA.build_layer(hf, i, cal)
    res = {}
    for p in G.PROJ:
        W = getattr(layer, p).weight.view(getattr(layer, p).weight.shape[0], -1)
        grid = G.Lut(W, G.lloyd_lut(W, a.group, 2 ** a.bits, pin=True), a.group)
        for c in range(W.shape[1]):
            grid.round(W[:, c].double(), c)
        res[p] = grid.result()
    torch.save(res, out / f"layer_{i}.pt")
    print(f"layer {i}", flush=True)
json.dump({"block": 0, "grid": f"lut{a.bits}-rtn", "group": a.group, "head": False}, open(out / "meta.json", "w"))
