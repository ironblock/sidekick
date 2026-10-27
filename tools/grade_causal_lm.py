"""Grade a converted causal LM on the Neural Engine, through sidekick's own
decoder (docs/CONVERTING.md, "Causal language models").

    python tools/grade_causal_lm.py <model-dir> --work <converter work dir> --decoder <kv_decoder binary>

Runs the installed build with `sidekick_coreml::KvDecoder` (the
`kv_decoder` example of sidekick-coreml) over the converter's evaluation
set, along both of the runtime's paths:
- **generation**: each teacher-forced text one token per call from
  position 0, as replies are generated;
- **prompt**: each text T tokens per call, as prompts are fed, each graded
  position read with `sel` at it;
and each greedy prompt as the server runs one (T-token calls, then one
token per call), whose prompt row is compared with fp32's last-position
logits and whose continuation with fp32's.

Every path is graded at the same positions against the converter's fp32
references with sidekick_convert.causal_lm.grade, overall and per text
family (the repository's texts, #173's set, against held-out prose). The
ANE grade is the worse of the two paths' grades; the build's grade is the
ANE's, or lower when the converter's two calibration draws screened
differently (the stability check). Writes <work>/ane-grade.json.

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

PATHS = {"generation": "token", "prompt": "tokens"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model_dir", type=lambda s: Path(s).expanduser())
    p.add_argument("--work", type=lambda s: Path(s).expanduser(), required=True)
    p.add_argument("--decoder", type=lambda s: Path(s).expanduser(), required=True,
                   help="the kv_decoder example binary, signed with a fixed identifier")
    p.add_argument("--timeout", type=float, default=2400.0, help="kill the run after this many seconds")
    p.add_argument("--speed", action="store_true", help="also time one call at three positions and a generation")
    p.add_argument("--keep-rows", action="store_true",
                   help="keep the logits rows the decoder wrote (<work>/ane/run_*.f32, about 0.5 MB a position)")
    args = p.parse_args()
    job = L.Job.load(args.work / "job.json")
    refs_dir = args.work / "references"
    meta = json.loads((refs_dir / "meta.json").read_text())
    fixture = L.load_fixture(job.src, job.model_id)
    texts = fixture.teacher_forced
    if [t.name for t in texts] != meta["texts"] or any(t.rows != meta["rows"][t.name] for t in texts):
        raise SystemExit("the references were made for another evaluation set: rerun the converter")
    tasks = [{"type": "trace", "name": f"{path}.{t.name}", "ids": t.ids, "rows": t.rows, "calls": calls}
             for path, calls in PATHS.items() for t in texts]
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
    refs = {t.name: np.load(refs_dir / f"fp32_{t.name}.npy") for t in texts}
    grades = {}
    for path in PATHS:
        got = {t.name: np.fromfile(out / f"run_{path}.{t.name}.f32", dtype=np.float32).reshape(len(t.rows), -1)
               for t in texts}
        grades[path] = L.grade(got, refs, L.families(fixture))
    results = {t["name"]: t for t in run["tasks"]}
    prefill_ref = np.load(refs_dir / "fp32_prefill.npy").astype(np.float64)
    same, prompt_rows = [], []
    for j, ref in enumerate(greedy):
        mine = results[ref["name"]]["tokens"][:len(ref["new"])]
        n = next((i for i, (a, b) in enumerate(zip(mine, ref["new"])) if a != b), len(ref["new"]))
        same.append({"name": ref["name"], "identical": n == len(ref["new"]), "first_difference": n,
                     "tokens": len(ref["new"])})
        row = np.fromfile(out / f"run_{ref['name']}_prefill.f32", dtype=np.float32).astype(np.float64)
        b = prefill_ref[j]
        prompt_rows.append({"name": ref["name"], "cosine": float(row @ b / np.linalg.norm(row) / np.linalg.norm(b)),
                            "top1": bool(row.argmax() == b.argmax())})
    report = json.loads((args.work / "report.json").read_text())
    ane = L.worse(*(g["grade"] for g in grades.values()))
    final = ane if report.get("stable", True) else L.worse(ane, report["screen_grade"])
    summary = {"paths": grades, "ane_grade": ane, "prompt_rows": prompt_rows, "greedy": same,
               "stable": report.get("stable"), "screens": report.get("screens"), "grade": final,
               "ceiling": meta.get("ceiling"),
               "decoder": {k: run[k] for k in ("load_ms", "first_call_ms", "ane_bytes", "health")},
               "speed": [t for t in run["tasks"] if t["type"] in ("tbench", "gen")]}
    (args.work / "ane-grade.json").write_text(json.dumps(summary, indent=1))
    for path, g in grades.items():
        print(f"ANE, {path} path: {L.describe(g)}")
        for fam, sub in g["families"].items():
            print(f"  {fam}: {L.describe(sub)}")
    print("prompt rows: worst cosine {:.5f}, top-1 {}/{}".format(
        min(r["cosine"] for r in prompt_rows), sum(r["top1"] for r in prompt_rows), len(prompt_rows)))
    print(f"greedy: {sum(s['identical'] for s in same)}/{len(same)} continuations identical to fp32")
    print(f"build grade: {final}" + ("" if report.get("stable", True) else " (the draws screened differently)"))
    if not args.keep_rows:
        for f in out.glob("run_*.f32"):      # the rows are graded; they're large
            f.unlink()


if __name__ == "__main__":
    main()
