"""spike.py job <refs> <job.json> [--tasks teacher,greedy,bench,gen] [--positions 128,512,1024,2016] [--gen-n 256]
   spike.py analyze <refs> <out_prefix>     (skrun's outputs) -> <out_prefix>_summary.json

Builds skrun jobs from the PyTorch references (reference.py) and grades its
outputs: teacher-forced top-1 agreement and per-position logit cosine against
fp32, beside the ideal-fp16 ceiling (the checkpoint run in fp16), graded with
D26's vocabulary on the worst position (A >= 0.9999, B >= 0.999,
C >= 0.985, D below); greedy continuations' first divergence from fp32."""
import argparse
import json
from pathlib import Path

import numpy as np


def grade(x):
    return "A" if x >= 0.9999 else "B" if x >= 0.999 else "C" if x >= 0.985 else "D"


def job(a):
    refs = Path(a.refs)
    m = json.load(open(refs / "meta.json"))
    ids = json.load(open(refs / "eval_ids.json"))
    g = json.load(open(refs / "greedy.json"))
    tasks = []
    want = a.tasks.split(",")
    if "bench" in want:
        tasks.append({"type": "bench", "positions": [int(p) for p in a.positions.split(",")], "steps": 32, "reps": 5})
    if "teacher" in want:
        for name in m["texts"]:
            tasks.append({"type": "teacher", "name": name, "ids": ids[name], "p": m["p"], "bucket": m["p"]})
    if "greedy" in want:
        for i, rec in enumerate(g):
            tasks.append({"type": "greedy", "name": f"greedy{i}", "ids": rec["ids"], "n": len(rec["new"])})
    if "gen" in want:
        tasks.append({"type": "gen", "name": "gen", "ids": g[0]["ids"], "n": a.gen_n})
    json.dump({"tasks": tasks}, open(a.out, "w"))


def analyze(a):
    refs, pre = Path(a.refs), a.prefix
    m = json.load(open(refs / "meta.json"))
    g = json.load(open(refs / "greedy.json"))
    res = json.load(open(pre + ".json"))
    out = {"units": res["units"], "load_total_s": res["load_total_s"], "chunk_median_ms": res["chunk_median_ms"]}
    allc, allcc, agree, agree_c, tot, texts = [], [], 0, 0, 0, {}
    greedy = []
    for t in res["tasks"]:
        if t["type"] == "teacher":
            name = t["name"]
            got = np.fromfile(f"{pre}_{name}.f32", dtype=np.float32).reshape(t["rows"], t["vocab"])
            ref = np.load(refs / f"eval_{name}.npy")
            cos = (got * ref).sum(1) / np.linalg.norm(got, axis=1) / np.linalg.norm(ref, axis=1)
            top = got.argmax(1) == ref.argmax(1)
            rec = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "top1": float(top.mean()),
                   "max_abs_dlogit": float(np.abs(got - ref).max()), "prefill_pos_cos": float(cos[0])}
            ceil_p = refs / f"ceil_{name}.npy"
            if ceil_p.exists():
                ceil = np.load(ceil_p)
                cc = (ceil * ref).sum(1) / np.linalg.norm(ceil, axis=1) / np.linalg.norm(ref, axis=1)
                tc = ceil.argmax(1) == ref.argmax(1)
                rec.update(ceil_cos_mean=float(cc.mean()), ceil_cos_min=float(cc.min()), ceil_top1=float(tc.mean()))
                allcc += cc.tolist()
                agree_c += int(tc.sum())
            allc += cos.tolist()
            agree += int(top.sum())
            tot += len(top)
            texts[name] = rec
        elif t["type"] == "greedy":
            i = int(t["name"][6:])
            ref = g[i]["new"]
            div = next((j for j, (x, y) in enumerate(zip(t["tokens"], ref)) if x != y), None)
            greedy.append({"prompt": g[i]["prompt"][:50], "n": len(ref), "first_divergence": div,
                           "identical": div is None})
        elif t["type"] in ("bench", "gen"):
            out[t["type"]] = {k: v for k, v in t.items() if k not in ("type", "name")}
    if tot:
        c = np.array(allc)
        s = {"positions": tot, "top1": agree / tot, "cos_mean": float(c.mean()), "cos_min": float(c.min()),
             "cos_p1": float(np.percentile(c, 1)), "grade_worst": grade(float(c.min())), "grade_p1": grade(float(np.percentile(c, 1)))}
        if allcc:
            cc = np.array(allcc)
            s.update(ceil_top1=agree_c / tot, ceil_cos_mean=float(cc.mean()), ceil_cos_min=float(cc.min()),
                     ceil_cos_p1=float(np.percentile(cc, 1)), ceil_grade_worst=grade(float(cc.min())))
        out["parity"] = {"summary": s, "texts": texts}
    if greedy:
        n_ident = sum(x["identical"] for x in greedy)
        divs = [x["first_divergence"] for x in greedy if x["first_divergence"] is not None]
        out["greedy"] = {"identical": f"{n_ident}/{len(greedy)}", "divergences": divs, "prompts": greedy}
    json.dump(out, open(pre + "_summary.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k in ("parity", "greedy", "bench", "gen")}, indent=1)[:6000])


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
j = sub.add_parser("job")
j.add_argument("refs")
j.add_argument("out")
j.add_argument("--tasks", default="bench,teacher,greedy,gen")
j.add_argument("--positions", default="128,512,1024,2016")
j.add_argument("--gen-n", type=int, default=256)
an = sub.add_parser("analyze")
an.add_argument("refs")
an.add_argument("prefix")
a = ap.parse_args()
job(a) if a.cmd == "job" else analyze(a)
