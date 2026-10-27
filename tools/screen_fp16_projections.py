"""Which projections of a causal-LM build carry its quantisation error:
fp16 ablations on the PyTorch screen (docs/CONVERTING.md, "Causal language
models"). Nothing is compiled.

    python tools/screen_fp16_projections.py --work <converter work dir> --draw 0 \
        [--search | --sets sets.json] --out results.jsonl

Runs the converter's screen (the GPTQ build of one calibration draw,
teacher-forced in fp32 over the evaluation set, graded with
causal_lm.grade) with chosen projections put back to the checkpoint's own
weights, rotated and rounded to fp16, as `--fp16-projections` would store
them. The other weights are the draw's GPTQ weights unchanged: GPTQ of the
later layers isn't rerun around the restored projections, so a set found
here is confirmed by a conversion that keeps it in fp16
(`convert_causal_lm.py --fp16-projections`), which recalibrates every
later layer.

- `--sets`: a JSON list of sets to grade, each a list of "layer.proj"
  names ("3.down"; "3.*" for a whole layer).
- `--search`: a ranked search over `--layers` (default 0-20). It grades
  every layer alone with all seven projections in fp16, ranks the layers
  by how far each lifts the prose family (wide flips first, then the
  first-percentile cosine), adds the best layers one at a time until the
  set grades B (pooled and in the prose family) or passes `--budget-mib`,
  then tries to drop each projection type from that set while it still
  grades B.

The model is held in memory once (the GPTQ weights of every layer, in
fp32, and the fp16 originals of the searched layers' projections: about
12 GB for MiniCPM5-2B), and the residual stream entering each searched
layer is cached from the baseline run, so a set only reruns the layers from
its first one on. Each graded set is one line of `--out`: the set, its
extra ANE weight over the 8-bit palettes, the grade overall and per family
(p1 cosine, top-1, wide flips, KL), and the seconds it took.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidekick_convert import causal_lm as L  # noqa: E402
from sidekick_convert.techniques import gptq as G  # noqa: E402

PROJ = ("q", "k", "v", "o", "gate", "up", "down")


def extra_mib(ck, names, group=16, entries=256):
    """ANE weight added by storing `names` in fp16 rather than as 8-bit
    palettes (one byte a weight and a 256-entry fp16 palette per `group`
    output channels)."""
    shape = {"q": (ck.heads * ck.head_dim, ck.hidden), "k": (ck.kv_heads * ck.head_dim, ck.hidden),
             "v": (ck.kv_heads * ck.head_dim, ck.hidden), "o": (ck.hidden, ck.heads * ck.head_dim),
             "gate": (int(ck.config["intermediate_size"]), ck.hidden),
             "up": (int(ck.config["intermediate_size"]), ck.hidden),
             "down": (ck.hidden, int(ck.config["intermediate_size"]))}
    total = 0
    for name in names:
        o, i = shape[name.split(".")[1]]
        total += 2 * o * i - (o * i + o // group * entries * 2)
    return total / 2**20


def expand(names):
    out = []
    for n in names:
        layer, proj = n.split(".")
        out += [f"{layer}.{p}" for p in PROJ] if proj == "*" else [n]
    return sorted(set(out), key=lambda n: (int(n.split(".")[0]), PROJ.index(n.split(".")[1])))


class Screen:
    def __init__(self, job, seed, layers):
        torch.set_grad_enabled(False)
        torch.set_num_threads(job.recipe.threads)
        self.LC, _, self.fixture = L._setup(job)
        LC, r = self.LC, job.recipe
        self.job, self.seed = job, seed
        self.ck = LC.Checkpoint(job.src)
        self.rot = LC.Rotation(r.rotation_seed, r.r2).bind(self.ck)
        self.ranges = L._ranges(LC, job, seed)
        self.texts = self.fixture.teacher_forced
        self.refs = L.load_refs(job)
        self.families = L.families(self.fixture)
        self.norm = LC.final_norm(self.ck, self.rot, self.ranges)
        self.head = L.quantized_head(G, job, seed)
        t0 = time.time()
        self.layers = [L.quantized_layer(LC, G, job, self.ck, self.rot, self.ranges, seed, i)
                       for i in range(self.ck.layers)]
        self.original = {}
        for i in layers:
            plain = LC.build_layer(self.ck, i, self.rot, self.ranges)
            for p in PROJ:
                self.original[f"{i}.{p}"] = LC.weight_of(getattr(plain, p)).half().clone()
            del plain
        print(f"loaded draw {seed}: {time.time() - t0:.0f} s", flush=True)
        table = LC.embedding_table(self.ck, self.rot, self.ranges.K)
        self.cached = {}                       # residual stream entering each searched layer, baseline run
        first = min(layers)
        hs = [LC.embed(table, t.ids) for t in self.texts]
        for i in range(self.ck.layers):
            if i in layers or i == first:
                self.cached[i] = [h.clone() for h in hs]
            if i >= max(layers):
                break
            hs = [LC.run_layer(self.layers[i], h, self.ck) for h in hs]

    def grade(self, names):
        names = expand(names)
        start = min((int(n.split(".")[0]) for n in names), default=min(self.cached))
        saved = []
        for n in names:
            i, p = n.split(".")
            c = getattr(self.layers[int(i)], p)
            saved.append((c, c.weight.detach().clone()))
            c.weight.copy_(self.original[n].float().view_as(c.weight))
        t0 = time.time()
        try:
            hs = [h for h in self.cached[start]]
            for i in range(start, self.ck.layers):
                hs = [self.LC.run_layer(self.layers[i], h, self.ck) for h in hs]
            got = {t.name: self.LC.logits(self.norm, self.head, h, t.rows).numpy() for t, h in zip(self.texts, hs)}
        finally:
            for c, w in saved:
                c.weight.copy_(w)
        g = L.grade(got, self.refs, self.families)
        keep = ("grade", "cos_p1", "top1", "wide_flips", "cos_min", "kl_mean", "kl_p99", "kl_max")
        return {"draw": self.seed, "set": names, "extra_mib": round(extra_mib(self.ck, names), 1),
                "grade": g["grade"], **{k: g[k] for k in keep if k != "grade"},
                "families": {f: {k: v[k] for k in keep} for f, v in g["families"].items()},
                "flips_wide": [(f["text"], f["row"], round(f["fp32_margin"], 3)) for f in g["flips"]
                               if f["fp32_margin"] >= L.WIDE_MARGIN],
                "seconds": round(time.time() - t0)}


def good(res):
    return res["grade"] in ("A", "B") and res["families"]["prose"]["grade"] in ("A", "B")


def score(res):
    prose = res["families"]["prose"]
    return (res["wide_flips"], -prose["cos_p1"], prose["kl_mean"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--layers", default="0-20", help="the layers searched, as lo-hi")
    ap.add_argument("--sets", type=Path, help="a JSON list of sets to grade")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--budget-mib", type=float, default=300.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    job = L.Job.load(a.work / "job.json")
    screen = Screen(job, a.draw, layers)
    out = open(a.out, "a")

    def run(names, label):
        res = screen.grade(names)
        res["label"] = label
        out.write(json.dumps(res) + "\n")
        out.flush()
        prose = res["families"]["prose"]
        print(f"{label:28s} {res['extra_mib']:7.1f} MiB  {res['grade']} p1 {res['cos_p1']:.5f}  prose "
              f"{prose['grade']} p1 {prose['cos_p1']:.5f} top-1 {prose['top1']:.4f} wide {prose['wide_flips']} "
              f"KL {prose['kl_mean']:.2e}  ({res['seconds']} s)", flush=True)
        return res

    base = run([], "baseline")
    if a.sets:
        for s in json.loads(a.sets.read_text()):
            run(s, "+".join(s) if len(s) < 4 else f"{len(s)} projections")
    if not a.search:
        return
    singles = {i: run([f"{i}.*"], f"layer {i}") for i in layers}
    ranked = sorted(layers, key=lambda i: score(singles[i]))
    print("layers ranked:", ranked, flush=True)
    chosen, res = [], base
    for i in ranked:
        if good(res) or extra_mib(screen.ck, expand(chosen + [f"{i}.*"])) > a.budget_mib:
            break
        chosen.append(f"{i}.*")
        res = run(chosen, "layers " + ",".join(c.split(".")[0] for c in chosen))
    if not good(res):
        print(f"no set of whole layers under {a.budget_mib:.0f} MiB grades B", flush=True)
        return
    names = expand(chosen)
    for p in sorted(PROJ, key=lambda p: -extra_mib(screen.ck, [f"0.{p}"])):    # drop the costliest first
        trial = [n for n in names if n.split(".")[1] != p]
        if trial and trial != names:
            r = run(trial, f"without {p}")
            if good(r):
                names, res = trial, r
    print("smallest set found:", names, f"{res['extra_mib']} MiB", flush=True)
    out.write(json.dumps({"draw": a.draw, "smallest": names, "extra_mib": res["extra_mib"]}) + "\n")


if __name__ == "__main__":
    main()
