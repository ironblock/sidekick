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
differently (the stability check). Writes <work>/ane-grade.json, the
grade report: the grades, the rubric (D54), the chip and macOS, and the
sha256 of every file of the build it graded but `chat.toml`, so the
checksum ledger can attach the grade to exactly these files
(tools/ledger.py). `--report-dir DIR` also writes it to DIR under a name of
its own, for docs/data/runs.toml to describe; `--record-os` writes the
macOS it was graded on into the build's `chat.toml` as `graded_os`, which
`/v1/models` lists.

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
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidekick_convert import causal_lm as L  # noqa: E402

PATHS = {"generation": "token", "prompt": "tokens"}
MANIFEST = "chat.toml"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def graded_files(model_dir):
    """[{path, sha256}] for every file of the build but its manifest, which
    `--record-os` edits after grading."""
    root = Path(model_dir)
    return [{"path": str(f.relative_to(root)), "sha256": sha256(f)}
            for f in sorted(root.rglob("*")) if f.is_file() and f.name != ".DS_Store"
            and str(f.relative_to(root)) != MANIFEST]


def machine():
    """(chip, "macOS <version> (<build>)", "macOS <version>")."""
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    version = platform.mac_ver()[0]
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return chip, f"macOS {version} ({build})", f"macOS {version}"


def record_os(manifest, graded_os):
    """Set `graded_os` in a chat.toml, as a top-level key before its first table."""
    text = Path(manifest).read_text()
    line = f'graded_os = "{graded_os}"\n'
    if re.search(r"(?m)^graded_os = ", text):
        text = re.sub(r'(?m)^graded_os = .*\n', line, text)
    else:
        first = re.search(r"(?m)^\[", text)
        at = first.start() if first else len(text)
        text = text[:at] + line + "\n" + text[at:]
    Path(manifest).write_text(text)


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
    p.add_argument("--report-dir", type=lambda s: Path(s).expanduser(),
                   help="also write the grade report here as causal-lm-<model>-<date>.json")
    p.add_argument("--record-os", action="store_true",
                   help="write the macOS it was graded on into the build's chat.toml as graded_os")
    args = p.parse_args()
    files_before = graded_files(args.model_dir)
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
    if graded_files(args.model_dir) != files_before:
        raise SystemExit("the build's files changed while it was graded")
    chip, macos, graded_os = machine()
    summary = {"rubric": "D54", "model": job.model_id, "chip": chip, "macos": macos,
               "date": datetime.date.today().isoformat(), "files": files_before,
               "paths": grades, "ane_grade": ane, "prompt_rows": prompt_rows, "greedy": same,
               "stable": report.get("stable"), "screens": report.get("screens"), "grade": final,
               "ceiling": meta.get("ceiling"),
               "decoder": {k: run[k] for k in ("load_ms", "first_call_ms", "ane_bytes", "health")},
               "speed": [t for t in run["tasks"] if t["type"] in ("tbench", "gen")]}
    (args.work / "ane-grade.json").write_text(json.dumps(summary, indent=1))
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        named = args.report_dir / f"causal-lm-{job.model_id}-{summary['date']}.json"
        named.write_text(json.dumps(summary, indent=1))
        print(f"grade report: {named}")
    if args.record_os:
        record_os(args.model_dir / MANIFEST, graded_os)
        print(f"recorded graded_os = {graded_os!r} in {args.model_dir / MANIFEST}")
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
