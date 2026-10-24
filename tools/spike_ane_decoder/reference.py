"""reference.py <hf_src> <out_dir> [--n 256] [--p 128] [--new 96]

PyTorch references for the spike, per eval text (texts.py):
- fp32 teacher-forced logits at positions p-1 .. n-1 (eval_<name>.npy, fp32)
- the same from the model in fp16 (CPU), the ideal-fp16 ceiling (ceil_<name>.npy)
- greedy continuations of the chat prompts, fp32 (greedy.json)
- an exactness check of the rewritten module (llama_ane, fp32 cache, after its
  rewrites) against the checkpoint (exact.json)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA
import texts as T

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--p", type=int, default=128)
ap.add_argument("--new", type=int, default=96)
ap.add_argument("--skip-ceiling", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.src)
hf = LA.load_hf(a.src)
evals = T.eval_texts(tok, a.n)
meta = {"n": a.n, "p": a.p, "texts": [name for name, _ in evals]}

for name, ids in evals:
    t = time.time()
    lg = hf(torch.tensor([ids])).logits[0, a.p - 1:a.n].float().numpy()
    np.save(out / f"eval_{name}.npy", lg)
    print(f"fp32 {name} {time.time() - t:.1f}s", flush=True)
json.dump({name: ids for name, ids in evals}, open(out / "eval_ids.json", "w"))

# greedy continuations (fp32)
greedy = []
for user in T.prompts():
    ids = T.chat_ids(tok, user)
    g = hf.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    do_sample=False, max_new_tokens=a.new, pad_token_id=tok.eos_token_id)
    new = g[0, len(ids):].tolist()
    greedy.append({"prompt": user, "ids": ids, "new": new, "text": tok.decode(new)})
    print("greedy:", repr(tok.decode(new)[:120]), flush=True)
json.dump(greedy, open(out / "greedy.json", "w"), indent=1)

# rewritten module, fp32 cache, exactness against the checkpoint
chunks, layers, norm = LA.build(hf, C=1024, cuts=[(0, hf.config.num_hidden_layers)], cache_dtype=torch.float32)
embed = hf.model.embed_tokens.weight.detach()
cal = LA.calibrate(chunks, layers, norm, hf.config, embed, T.calib_texts(tok))
json.dump(cal, open(out / "calibration.json", "w"))
worst = 0.0
for name, ids in evals[:3]:
    for c in chunks:
        c.reset()
    mine = LA.run_eager(chunks, hf.config, embed, ids)[a.p - 1:a.n].numpy()
    ref = np.load(out / f"eval_{name}.npy")
    worst = max(worst, float(np.abs(mine - ref).max()))
meta["rewrite_max_abs_logit_diff"] = worst
print("rewritten module vs checkpoint, fp32: max |dlogit|", worst, flush=True)

if not a.skip_ceiling:
    hf16 = hf.half()
    for name, ids in evals:
        t = time.time()
        lg = hf16(torch.tensor([ids])).logits[0, a.p - 1:a.n].float().numpy()
        np.save(out / f"ceil_{name}.npy", lg)
        print(f"fp16 {name} {time.time() - t:.1f}s", flush=True)
json.dump(meta, open(out / "meta.json", "w"), indent=1)
