"""Measure an installed classifier's Core ML artifacts against its fp32
reference, on each compute path, by the rules of docs/design/classify.md.

Reads the reference tools/classifier_reference.py wrote and feeds every case's
exact ids (and, for laya, markers and qtype) to the artifact of the smallest
bucket that fits. Per path it reports:

- hard checks: finite logits (laya: padded slots at -1e4), pad invariance
  (pad ids 0 vs random, max |dlogit| on the first cases of each bucket) and
  bucket invariance (every case in its own bucket vs each larger one, max
  |dp|, as the parity suite gates it);
- graded: argmax agreement with fp32, counting a disagreement as a flip only
  where fp32's top-2 logit margin is >= 0.05 (smaller margins are near-ties,
  reported separately); raw |dp| (softmax at temperature 1) and |dlogit|;
- calibrated |dp| (laya: softmax at the manifest's temperature for the case's
  question type and label count, where the manifest has one);
- reported, never graded: accuracy against the corpus's gold labels, for fp32
  and for the path;
- latency: median and p90 per bucket.

This is the Python counterpart of the parity suite's classifier checks, and
reads the same reference files.

Usage:
    python tools/measure_classifier.py <model-dir> --refs DIR [--paths ane,cpu,gpu] [--json FILE]

Requires: coremltools, numpy, safetensors (arm64-native Python).
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import coremltools as ct
from safetensors.numpy import load_file

MARGIN = 0.05
PAD_LOGIT = -1e4
UNITS = {"ane": ct.ComputeUnit.CPU_AND_NE, "cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU}
QTYPE_NAMES = {0: "choice", 1: "score", 2: "noul"}


def softmax(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_key(qtype, k):
    """laya's temp_bucket key (rl_common.temp_bucket)."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[qtype]}:{size}"


def inputs_for(case, seq, laya, kmax, pad_ids=None):
    n = len(case["ids"])
    ids = np.zeros((1, seq), dtype=np.int32)
    ids[0, :n] = case["ids"]
    if pad_ids is not None:
        ids[0, n:] = pad_ids[: seq - n]
    mask = np.zeros((1, seq), dtype=np.int32)
    mask[0, :n] = 1
    x = {"input_ids": ids, "attention_mask": mask}
    if laya:
        mp = np.full((1, kmax), -1, dtype=np.int32)
        mp[0, : case["k"]] = case["markers"]
        x.update(marker_pos=mp, qtype=np.array([case["qtype"]], dtype=np.int32))
    return x


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None
    return {"max": float(v.max()), "p99": float(np.percentile(v, 99)), "mean": float(v.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--paths", default="ane,cpu,gpu")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    man = tomllib.loads((args.model_dir / "classifier.toml").read_text())
    laya = man["classify"].get("format") == "laya"
    calibration = man["classify"].get("calibration", {})
    ref_dir = args.refs / man["id"]
    meta = json.loads((ref_dir / "reference.json").read_text())
    ref = load_file(str(ref_dir / "reference.safetensors"))["torch"].astype(np.float64)
    cases = meta["cases"]
    kmax = meta["model"]["max_labels"]
    buckets = man["buckets"]
    labels = man["classify"].get("labels", [])
    bucket_of = lambda n: next(b for b in buckets if n <= b)  # noqa: E731

    report = {"model": man["id"], "cases": len(cases), "paths": {}}
    for path in args.paths.split(","):
        models = {b: ct.models.CompiledMLModel(str(args.model_dir / man["artifact"].format(seq=b)),
                                              compute_units=UNITS[path]) for b in buckets}
        rows, ms = [], {b: [] for b in buckets}
        warmed = set()
        keep = []  # see run() below
        own = {}   # each case's logits in its own bucket
        for i, c in enumerate(cases):
            b = bucket_of(len(c["ids"]))
            if b not in warmed:
                for _ in range(3):
                    models[b].predict(inputs_for(c, b, laya, kmax))
                warmed.add(b)
            x = inputs_for(c, b, laya, kmax)
            keep.append(x)
            t0 = time.perf_counter()
            out = models[b].predict(x)[man["classify"]["io"]["output"]][0]
            ms[b].append((time.perf_counter() - t0) * 1e3)
            out = out.astype(np.float64)
            k = c["k"]
            own[i] = out[:k]
            got, r = out[:k], ref[i, :k]
            ok = bool(np.isfinite(out).all() and (not laya or np.all(out[k:] == PAD_LOGIT)))
            row = {"id": c["id"], "bucket": b, "ok": ok}
            if ok:
                top2 = np.sort(r)[-2:]
                row.update(
                    margin=float(top2[1] - top2[0]),
                    agree=bool(np.argmax(got) == np.argmax(r)),
                    dlogit=float(np.abs(got - r).max()),
                    dp=float(np.abs(softmax(got) - softmax(r)).max()),
                    pred=int(np.argmax(got)), ref_pred=int(np.argmax(r)))
                if laya:
                    t = calibration.get(temp_key(c["qtype"], k))
                    if t is not None:
                        row["dp_cal"] = float(np.abs(softmax(got, t) - softmax(r, t)).max())
                gold = c.get("gold")
                if gold:
                    names = c.get("candidate_labels") or labels
                    row["gold_ok_path"] = names[row["pred"]] in gold
                    row["gold_ok_ref"] = names[row["ref_pred"]] in gold
            rows.append(row)

        # pad invariance on a sample (the first cases of each bucket), as max
        # |dlogit|; bucket invariance on every case, as the parity suite
        # measures it: max |dp| between a case's own bucket and each larger one.
        # Inputs stay referenced until the end: Core ML can release a finished
        # prediction's input buffers after predict() returns, and freeing them
        # first can crash the process when predictions alternate between
        # models; rapid alternation between large models has also aborted ANE
        # requests, so the bucket check runs one model at a time.
        key = man["classify"]["io"]["output"]

        def run(b, c, pad_ids=None):
            x = inputs_for(c, b, laya, kmax, pad_ids)
            keep.append(x)
            return np.asarray(models[b].predict(x)[key][0, : c["k"]], dtype=np.float64)

        pad_d, bucket_dp = 0.0, 0.0
        rng = np.random.default_rng(0)
        for b in buckets:
            for c in [c for c in cases if bucket_of(len(c["ids"])) == b][:8]:
                if len(c["ids"]) < b:
                    a, p = run(b, c), run(b, c, rng.integers(1000, 40000, b))
                    pad_d = max(pad_d, float(np.abs(a - p).max()))
        # one bucket's model at a time, against each case's own-bucket output
        for up in buckets:
            for i, c in enumerate(cases):
                if bucket_of(len(c["ids"])) < up:
                    bucket_dp = max(bucket_dp, float(np.abs(softmax(run(up, c)) - softmax(own[i])).max()))

        good = [r for r in rows if r["ok"]]
        graded = [r for r in good if r["margin"] >= MARGIN]
        flips = [r for r in graded if not r["agree"]]
        gold_rows = [r for r in good if "gold_ok_path" in r]
        s = {
            "non_finite_or_unpadded": len(rows) - len(good),
            "graded": len(graded), "near_ties": len(good) - len(graded),
            "argmax_agreement": 1.0 - len(flips) / max(1, len(graded)),
            "flips": [{"id": r["id"], "margin": round(r["margin"], 4), "dlogit": round(r["dlogit"], 4)} for r in flips],
            "near_tie_disagreements": sum(1 for r in good if r["margin"] < MARGIN and not r["agree"]),
            "dp_raw": stats([r["dp"] for r in good]),
            "dlogit": stats([r["dlogit"] for r in good]),
            "dp_cal": stats([r["dp_cal"] for r in good if "dp_cal" in r]),
            "pad_invariance_max_dlogit": pad_d,
            "bucket_invariance_max_dp": bucket_dp,
            "gold_accuracy": (None if not gold_rows else {
                "path": sum(r["gold_ok_path"] for r in gold_rows) / len(gold_rows),
                "fp32": sum(r["gold_ok_ref"] for r in gold_rows) / len(gold_rows),
                "n": len(gold_rows)}),
            "latency_ms": {b: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                           for b, v in ms.items() if v},
        }
        report["paths"][path] = s
        lat = ", ".join(f"{b}: {v['median']:.1f} ms" for b, v in s["latency_ms"].items())
        g = s["gold_accuracy"]
        print(f"[{path}] finite/padded fails {s['non_finite_or_unpadded']}, argmax agreement "
              f"{s['argmax_agreement']:.4f} over {s['graded']} graded ({len(flips)} flips, "
              f"{s['near_ties']} near-ties, {s['near_tie_disagreements']} of them disagree)")
        print(f"   raw |dp| max {s['dp_raw']['max']:.4f} p99 {s['dp_raw']['p99']:.4f} mean {s['dp_raw']['mean']:.5f}; "
              f"|dlogit| max {s['dlogit']['max']:.3f}"
              + (f"; calibrated |dp| max {s['dp_cal']['max']:.4f} p99 {s['dp_cal']['p99']:.4f}" if s["dp_cal"] else ""))
        print(f"   pad invariance {pad_d:.2e} (dlogit), bucket invariance {bucket_dp:.2e} (dp); latency {lat}"
              + (f"; gold accuracy {g['path']:.4f} (fp32 {g['fp32']:.4f}, n={g['n']})" if g else ""))
        if flips:
            print(f"   flips: {s['flips'][:10]}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
