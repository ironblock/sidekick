"""Grade a converted causal LM on the Neural Engine, through sidekick's own
decoder (docs/CONVERTING.md, "Causal language models").

    python tools/grade_causal_lm.py <model-dir> --work <converter work dir> --decoder <kv_decoder binary>

Runs the installed build with `sidekick_coreml::KvDecoder` (the
`kv_decoder` example of sidekick-coreml) over the converter's evaluation
set: each teacher-forced text one token per call from position 0, and each
greedy prompt as the server runs a prompt (T-token calls, then one token
per call). Grades the logits rows against the converter's fp32 references
with sidekick_convert.causal_lm.grade, compares the greedy continuations
with fp32's, and writes <work>/ane-grade.json. The build's grade is the
ANE's, or lower when the converter's two calibration draws screened
differently (the stability check).

Build the example once and sign it with a fixed identifier, so later builds
of it reuse the ANE compile (the cache key includes the code signature, and
each compile of a 2B build adds about 2.2 GB to a system cache that can't be
deleted; docs/design/coreml-cache.md):

    cargo build --release -p sidekick-coreml --example kv_decoder
    codesign -s - -f -i io.github.ironblock.sidekick.kv-decoder target/release/examples/kv_decoder

An ANE request can hang (docs/design/coreml-cache.md, "A hung ANE
request"): the decoder fails a call past 60 s, and this tool kills the run
past `--timeout` seconds.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidekick_convert import causal_lm as L  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=lambda s: Path(s).expanduser())
    p.add_argument("--work", type=lambda s: Path(s).expanduser(), required=True)
    p.add_argument("--decoder", type=lambda s: Path(s).expanduser(), required=True,
                   help="the kv_decoder example binary, signed with a fixed identifier")
    p.add_argument("--timeout", type=float, default=1800.0, help="kill the run after this many seconds")
    p.add_argument("--speed", action="store_true", help="also time one call at three positions and a generation")
    args = p.parse_args()
    job = L.Job.load(args.work / "job.json")
    refs_dir = args.work / "references"
    meta = json.loads((refs_dir / "meta.json").read_text())
    fixture = L.load_fixture(job.src, job.model_id)
    tasks = [{"type": "trace", "name": name, "ids": ids} for name, ids in fixture.teacher_forced]
    greedy = json.loads((refs_dir / "greedy.json").read_text())
    tasks += [{"type": "greedy", "name": g["name"], "ids": g["ids"], "n": L.GREEDY_TOKENS} for g in greedy]
    if args.speed:
        tasks += [{"type": "tbench", "positions": [128, 1024, 1900], "reps": 20},
                  {"type": "gen", "name": "gen", "ids": greedy[0]["ids"], "n": 128}]
    out = args.work / "ane"
    out.mkdir(exist_ok=True)
    (out / "job.json").write_text(json.dumps({"tasks": tasks}))
    try:
        subprocess.run([str(args.decoder), str(args.model_dir), str(out / "job.json"), str(out / "run")],
                       check=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"the decoder run passed {args.timeout:.0f} s: a possible ANE hang; stopping")
    run = json.loads((out / "run.json").read_text())
    if run["health"]["failed"]:
        raise SystemExit(f"the decoder failed: {run['health']['failed']}")
    first = meta["first_graded"]
    got, refs = {}, {}
    for name, ids in fixture.teacher_forced:
        rows = np.fromfile(out / f"run_{name}.f32", dtype=np.float32).reshape(len(ids), -1)
        got[name] = rows[first - 1:len(ids)]
        refs[name] = np.load(refs_dir / f"fp32_{name}.npy")
    g = L.grade(got, refs)
    results = {t["name"]: t for t in run["tasks"]}
    same = []
    for ref in greedy:
        mine = results[ref["name"]]["tokens"][:len(ref["new"])]
        n = next((i for i, (a, b) in enumerate(zip(mine, ref["new"])) if a != b), len(ref["new"]))
        same.append({"name": ref["name"], "identical": n == len(ref["new"]), "first_difference": n,
                     "tokens": len(ref["new"])})
    report = json.loads((args.work / "report.json").read_text())
    final = g["grade"] if report.get("stable", True) else L.worse(g["grade"], report["screen_grade"])
    summary = {"ane": {k: v for k, v in g.items()}, "greedy": same, "stable": report.get("stable"),
               "screens": report.get("screens"), "grade": final, "ceiling": meta.get("ceiling"),
               "decoder": {k: run[k] for k in ("load_ms", "first_call_ms", "ane_bytes", "health")},
               "speed": [t for t in run["tasks"] if t["type"] in ("tbench", "gen")]}
    (args.work / "ane-grade.json").write_text(json.dumps(summary, indent=1))
    print(f"ANE: {L.describe(g)}")
    print(f"greedy: {sum(s['identical'] for s in same)}/{len(same)} continuations identical to fp32")
    print(f"build grade: {final}" + ("" if report.get("stable", True) else " (the draws screened differently)"))


if __name__ == "__main__":
    main()
