"""grade_trace.py <refs> <prefix_dir> <model_dir>

Teacher-forced grade from skrun "trace" dumps: <prefix_dir>/<text>_trace<last>.f32
holds the last chunk's logits for every position of an eval text, each
computed by the build's decode path. Scores rows p-1..n-1 as spike.py does
(top-1 agreement, worst/1st-percentile logit cosine, D26 grade) beside the
fp16 ceiling."""
import json
import sys
from pathlib import Path

import numpy as np

refs, pdir, mdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m = json.load(open(refs / "meta.json"))
last = len(json.load(open(mdir / "meta.json"))["chunks"]) - 1
p, n = m["p"], m["n"]


def grade(x):
    return "A" if x >= 0.9999 else "B" if x >= 0.999 else "C" if x >= 0.985 else "D"


cs, agree, tot, texts = [], 0, 0, {}
for name in m["texts"]:
    ref = np.load(refs / f"eval_{name}.npy")
    got = np.fromfile(pdir / f"{name}_trace{last}.f32", dtype=np.float32).reshape(n, -1)[p - 1:n]
    cos = (got * ref).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(ref, axis=1)
    top = got.argmax(1) == ref.argmax(1)
    agree += int(top.sum())
    tot += len(top)
    cs += cos.tolist()
    texts[name] = {"top1": round(float(top.mean()), 4), "cos_min": round(float(cos.min()), 6)}
c = np.array(cs)
s = {"positions": tot, "top1": agree / tot, "cos_mean": float(c.mean()), "cos_min": float(c.min()),
     "cos_p1": float(np.percentile(c, 1)), "grade_worst": grade(float(c.min())), "grade_p1": grade(float(np.percentile(c, 1))),
     "texts": texts}
json.dump(s, open(pdir / "grade.json", "w"), indent=1)
print("PARITY", json.dumps({k: v for k, v in s.items() if k != "texts"}))
print("TEXTS", json.dumps(texts))
