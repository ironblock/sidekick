"""chunk_trace.py <hf_src> <refs> <model_dir> <skrun_prefix> <text> <n_tokens> <out.json> [--quant int8pc]

Where does a converted build leave the PyTorch model? skrun's "trace" task
decodes the first n_tokens of an eval text one token per call and dumps
every chunk's output per position (<skrun_prefix>_trace<i>.f32). This
compares them with the same layers in PyTorch: the rewritten decoder in fp32,
with exact weights and with the build's quantiser simulated
(probe_quant_lib), and prints each chunk's per-position cosine, so the first
chunk and position that go wrong stand out."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("refs")
ap.add_argument("model")
ap.add_argument("prefix")
ap.add_argument("text")
ap.add_argument("n", type=int)
ap.add_argument("out")
ap.add_argument("--quant", default="int8pc")
a = ap.parse_args()
torch.set_grad_enabled(False)
refs, mdir = Path(a.refs), Path(a.model)
meta = json.load(open(mdir / "meta.json"))
cal = json.load(open(refs / "calibration.json"))
ids = json.load(open(refs / "eval_ids.json"))[a.text][:a.n]
D = meta["hidden"]
got = [np.fromfile(f"{a.prefix}_trace{i}.f32", dtype=np.float32).reshape(a.n, -1) for i in range(len(meta["chunks"]))]

# PyTorch: the same layers, exact and with simulated quantised weights
import probe_quant_lib as PQ
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
bounds = {c["hi"] - 1: ci for ci, c in enumerate(meta["chunks"]) if c["hi"] > c["lo"]}
res = {}
for mode in ("exact", a.quant):
    Q = PQ.quantizer(mode) if mode != "exact" else None
    hid = {}

    def on_layer(i, layer):
        if Q is not None:
            for p in PQ.PROJ:
                cv = getattr(layer, p)
                cv.weight.data = Q(cv.weight.view(cv.weight.shape[0], -1)).view_as(cv.weight)

    def on_hidden(i, t, h):
        if i in bounds:
            hid[bounds[i]] = h.view(D, -1).T.numpy()

    hs = LA.stream(hf, [ids], cal=cal, on_layer=on_layer, on_hidden=on_hidden, head=False)
    W = hf.lm_head.weight.float()
    if Q is not None:
        hf.lm_head.weight.data = Q(W)
    lg = LA.logits_of(hf, cal, hs[0]).numpy()
    hf.lm_head.weight.data = W
    for ci in range(len(got)):
        g = got[ci]
        r = hid.get(ci, lg) if ci < len(got) - 1 else lg
        cosv = (g * r).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(r, axis=1)
        res[f"{mode}.chunk{ci}"] = [round(float(x), 4) for x in cosv]
        print(f"RESULT {mode} chunk {ci}: min {cosv.min():.4f} at {int(cosv.argmin())}; first < 0.99: "
              f"{next((i for i, x in enumerate(cosv) if x < 0.99), None)}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
