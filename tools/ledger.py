"""The conversion checksum ledger (docs/data/ledger.json, #136).

Every build sidekick converts gets an entry here, published or not: which
model and revision it came from, the backend, build and quantization, what
produced it, the sha256 and size of every file, and the parity grades it
was graded with. A reconversion, a fetched download or a release candidate
is checked against its entry with `verify`, which says which layer moved:
the weights, the compiled program, the tokenizer or the manifest.

A file held at several paths (D49: one weights file per chunk, shared by
every bucket) is recorded once, by hash, with every path that holds it.
That list is also what fetch follows: it downloads each distinct hash once,
verifies it, and places it at every path (docs/PUBLISHING.md).

Usage:
    python3 tools/ledger.py add <model dir> [--variant NAME] [--reports DIR ...]
        [--converter-commit SHA] [--os "macOS 27.0 (26A428)"] [--chip "Apple M1 Max"]
        [--manifest FILE] [--exclude GLOB ...] [--note TEXT] [--dry-run]
    python3 tools/ledger.py add --sums SHA256SUMS --sums-prefix <dir in it>/ --model ID ...
    python3 tools/ledger.py verify <model dir> [--id BUILD] [--partial]
    python3 tools/ledger.py check
    python3 tools/ledger.py list [MODEL]
    python3 tools/ledger.py published BUILD --repo OWNER/NAME --revision SHA --sidekick X.Y.Z

`add` hashes a built model directory (hard-linked files once) and writes
or replaces its entry. With `--reports`, it reads parity reports (`parity
--json`) and records the grades of every report whose per-bucket weight
hashes match this build's files: the grades that build was graded with
(by default only from reports docs/data/runs.toml describes, whose grades
are published; `--any-report` also takes a release candidate's own).
With `--sums`, it records an archived build from a `shasum -a 256` listing
instead of hashing files; sizes are then unknown unless `--sizes` gives a
`stat`-style "size path" listing (it may be the same file, the sizes
following the checksums), which must then cover every file.

`verify` exits non-zero unless every file matches the entry. `--partial`
accepts missing paths (a fetch of fewer buckets).

`check` validates the ledger's shape and its references (models.toml, the
grade letters, and each build's digest against its file list). CI runs it
through tools/models_table.py --docs --check.

Standard library only.
"""

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "data" / "ledger.json"
MODELS = REPO / "docs" / "data" / "models.toml"

SCHEMA = 1
BACKENDS = ("coreml", "onnx")
QUANTIZATIONS = ("fp32", "fp16", "fp16 weights", "int8")
PATHS = ("coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu")
LETTERS = ("A", "B", "C", "D", "F")
EVIDENCE = ("hashed", "sha256sums")
REPORT_PATH = {"ane": "coreml-ane", "gpu": "coreml-gpu", "cpu": "coreml-cpu", "onnx-cpu": "onnx-cpu"}
IGNORED = (".DS_Store",)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BUCKET_DIR = re.compile(r"^model_(\d+)(?:\.(\d+))?\.mlmodelc$")

ENTRY_FIELDS = {
    "id": str, "model": str, "source": str, "backend": str, "variant": str, "quantization": str,
    "buckets": list, "chunks": int, "max_seq_len": (int, type(None)), "layout": str,
    "digest": str, "fingerprint": str, "produced": dict, "files": list, "grades": list,
    "published": list, "recorded": str, "evidence": str, "note": (str, type(None)),
}
PRODUCED_FIELDS = ("converter", "converter_commit", "coremltools", "torch", "python", "os", "xcode", "chip", "date")


# --- hashing -----------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def walk(root, excludes=()):
    """Every file under `root` as (relative posix path, absolute path),
    sorted, skipping Finder litter and excluded globs."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames.sort()
        for name in filenames:
            if name in IGNORED:
                continue
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, g) for g in excludes):
                continue
            out.append((rel, p))
    return sorted(out)


def hash_tree(root, excludes=(), manifest=None):
    """{relative path: (sha256, size)} for a model directory. A hard-linked
    file is read once. `manifest` replaces the directory's manifest file
    (a build installed with another manifest, such as a capped variant)."""
    seen = {}
    files = {}
    for rel, p in walk(root, excludes):
        if manifest is not None and rel in ("manifest.toml", "classifier.toml"):
            p = Path(manifest)
        st = p.stat()
        key = (st.st_dev, st.st_ino)
        if key not in seen:
            seen[key] = (sha256_file(p), st.st_size)
        files[rel] = seen[key]
    return files


def group(files):
    """The ledger's file list: each distinct sha256 once, with its size and
    every path that holds it, ordered by first path."""
    by_hash = {}
    for rel, (sha, size) in sorted(files.items()):
        e = by_hash.setdefault(sha, {"sha256": sha, "size": size, "paths": []})
        e["paths"].append(rel)
    return sorted(by_hash.values(), key=lambda e: e["paths"][0])


def flatten(entries):
    """{path: (sha256, size)} from a ledger file list."""
    return {p: (e["sha256"], e["size"]) for e in entries for p in e["paths"]}


def digest(files):
    """sha256 over every (path, sha256) pair in path order: the build's
    identity, independent of how its files are stored."""
    h = hashlib.sha256()
    for rel, (sha, _) in sorted(files.items()):
        h.update(f"{rel}\0{sha}\n".encode())
    return h.hexdigest()


def fingerprint(files):
    """tools/mteb_transfer.py's build fingerprint, from the same hashes:
    (path, sha256) over the weights, programs, tokenizer and manifest."""
    h = hashlib.sha256()
    for rel, (sha, _) in sorted(files.items()):
        name = rel.rsplit("/", 1)[-1]
        if name in ("weight.bin", "tokenizer.json", "manifest.toml", "model.mil") or name.endswith(
                (".onnx", ".onnx_data", ".onnx.data")):
            h.update(f"{rel}\0{sha}\n".encode())
    return h.hexdigest()[:12]


def layer(rel):
    """Which part of a build a path belongs to, for `verify`'s report."""
    name = rel.rsplit("/", 1)[-1]
    if name == "weight.bin" or name.endswith((".onnx_data", ".onnx.data")):
        return "weights"
    if name.endswith(".onnx"):
        return "graph"
    if "/analytics/" in rel:
        return "compile analytics"
    if ".mlmodelc/" in rel:
        return "compiled program"
    if name in ("manifest.toml", "classifier.toml"):
        return "manifest"
    if name.startswith("tokenizer") or name in ("vocab.txt", "special_tokens_map.json"):
        return "tokenizer"
    return "other"


# --- reading a build ---------------------------------------------------------

def read_manifest(text):
    return tomllib.loads(text)


def bucket_layout(paths):
    """(sorted buckets, chunk count) from compiled program directory names."""
    buckets, chunks = set(), set()
    for rel in paths:
        top = rel.split("/", 1)[0]
        m = BUCKET_DIR.match(top)
        if m:
            buckets.add(int(m.group(1)))
            chunks.add(int(m.group(2)) if m.group(2) is not None else 0)
    return sorted(buckets), max(len(chunks), 1)


def bucket_weights(files):
    """{bucket: "sha12[+sha12...]"} in chunk order, as parity reports
    record each bucket's weights."""
    per = {}
    for rel, (sha, _) in files.items():
        top, _, rest = rel.partition("/")
        m = BUCKET_DIR.match(top)
        if m and rest == "weights/weight.bin":
            per.setdefault(int(m.group(1)), []).append((int(m.group(2) or 0), sha[:12]))
    return {b: "+".join(s for _, s in sorted(v)) for b, v in per.items()}


def coreml_metadata(root, rels):
    """coremltools and torch versions, conversion date and storage
    precision, from the compiled programs' metadata.json (all must agree)."""
    found = {}
    for rel in rels:
        if not rel.endswith(".mlmodelc/metadata.json") or rel.count("/") != 1:
            continue
        try:
            meta = json.loads((Path(root) / rel).read_text())[0]
        except (OSError, ValueError, IndexError, KeyError):
            continue
        user = meta.get("userDefinedMetadata", {})
        vals = {
            "coremltools": user.get("com.github.apple.coremltools.version"),
            "torch": (user.get("com.github.apple.coremltools.source") or "").removeprefix("torch==") or None,
            "date": user.get("com.github.apple.coremltools.conversion_date"),
            "precision": meta.get("storagePrecision"),
        }
        for k, v in vals.items():
            if v is not None:
                found.setdefault(k, set()).add(v)
    return {k: (sorted(v)[0] if len(v) == 1 else ", ".join(sorted(v))) for k, v in found.items()}


def converter_of(manifest_text):
    """The converter a committed manifest's header names."""
    m = re.search(r"tools/(convert_[a-z0-9_]+\.py)", manifest_text)
    return f"tools/{m.group(1)}" if m else None


def committed_manifest(model):
    for p in (REPO / "examples" / "manifests" / model / "manifest.toml",
              REPO / "examples" / "classifiers" / model / "classifier.toml"):
        if p.exists():
            return p.read_text()
    return ""


def load_models():
    return tomllib.loads(MODELS.read_text())["models"]


# --- grades from parity reports ---------------------------------------------

def report_records(dirs):
    """(file name, report, model record) for every parity report under
    `dirs` whose records carry per-bucket weight hashes."""
    for d in dirs:
        for p in sorted(Path(d).rglob("*.json")):
            try:
                rep = json.loads(p.read_text())
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if not isinstance(rep, dict) or "chip" not in rep:
                continue
            for rec in (rep.get("models") or []) + (rep.get("classifiers") or []):
                if isinstance(rec, dict) and rec.get("buckets"):
                    yield p.name, rep, rec


def registered_reports():
    """The parity reports docs/data/runs.toml describes: the ones whose
    grades are published."""
    return set(tomllib.loads((REPO / "docs" / "data" / "runs.toml").read_text()).get("runs", {}))


def matching_grades(model, files, dirs, any_report=False):
    """The grades of every report whose buckets are exactly this build's,
    with the same weights: by default only reports runs.toml describes
    (published grades), with `any_report` every report found."""
    mine = bucket_weights(files)
    registered = registered_reports()
    out = []
    for name, rep, rec in report_records(dirs):
        if rec.get("id") != model or not (any_report or name in registered):
            continue
        theirs = {b["bucket"]: b.get("weights") for b in rec["buckets"]}
        if not mine or theirs != mine:
            continue
        for p in rec.get("paths", []):
            path = REPORT_PATH.get(p.get("path"))
            if path and p.get("letter") in LETTERS:
                out.append({"path": path, "chip": rep["chip"], "os": rep.get("macos"), "grade": p["letter"],
                            "report": name})
    uniq = {json.dumps(g, sort_keys=True): g for g in out}
    return sorted(uniq.values(), key=lambda g: (g["chip"], PATHS.index(g["path"]), g["report"]))


# --- the ledger file ---------------------------------------------------------

def load(path=LEDGER):
    if not Path(path).exists():
        return {"schema": SCHEMA, "builds": []}
    return json.loads(Path(path).read_text())


def save(ledger, path=LEDGER):
    ledger["builds"].sort(key=lambda b: (b["model"], b["backend"], b["variant"], b["id"]))
    Path(path).write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n")


def parse_sums(path, prefix):
    """{relative path: sha256} from a `shasum -a 256` listing, for the
    paths under `prefix`."""
    out = {}
    prefix = prefix.removeprefix("./")
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(.+)$", line.strip())
        if not m:
            continue
        rel = m.group(2).removeprefix("./")
        if rel.startswith(prefix):
            out[rel[len(prefix):]] = m.group(1)
    return out


def parse_sizes(path, prefix):
    """{relative path: size} from "size path" lines (`find . -type f -exec
    stat -f '%z %N' {} +`)."""
    out = {}
    prefix = prefix.removeprefix("./")
    for line in Path(path).read_text().splitlines():
        size, _, rel = line.strip().partition(" ")
        rel = rel.removeprefix("./")
        if size.isdigit() and rel.startswith(prefix):
            out[rel[len(prefix):]] = int(size)
    return out


def cmd_add(args):
    models = load_models()
    excludes = args.exclude or []
    if args.sums:
        if not args.model:
            raise SystemExit("--sums needs --model")
        sums = parse_sums(args.sums, args.sums_prefix or "")
        sizes = parse_sizes(args.sizes, args.sums_prefix or "") if args.sizes else {}
        files = {rel: (sha, sizes.get(rel)) for rel, sha in sums.items()
                 if rel.rsplit("/", 1)[-1] not in IGNORED and not any(fnmatch.fnmatch(rel, g) for g in excludes)}
        if not files:
            raise SystemExit(f"no files under {args.sums_prefix!r} in {args.sums}")
        if args.sizes and (unsized := sorted(r for r, (_, size) in files.items() if size is None)):
            raise SystemExit(f"--sizes gives no size for: {', '.join(unsized)}")
        manifest_text = ""
        meta = {}
        evidence = "sha256sums"
    else:
        root = Path(args.dir)
        files = hash_tree(root, excludes, args.manifest)
        mpath = Path(args.manifest) if args.manifest else next(
            (root / n for n in ("manifest.toml", "classifier.toml") if (root / n).exists()), None)
        manifest_text = mpath.read_text() if mpath else ""
        meta = coreml_metadata(root, files)
        evidence = "hashed"
    manifest = read_manifest(manifest_text) if manifest_text else {}
    model = args.model or manifest.get("id")
    if model not in models:
        raise SystemExit(f"{model!r} isn't in docs/data/models.toml")
    backend = args.backend or manifest.get("backend") or (
        "onnx" if any(r.endswith(".onnx") for r in files) else "coreml")
    src = manifest.get("source")
    source = models[model]["source"]
    if isinstance(src, dict) and src.get("repo") and src.get("revision"):
        source = f"{src['repo']}@{src['revision']}"
    if args.source:
        source = args.source
    precision = {"Float16": "fp16", "Float32": "fp32"}.get(meta.get("precision"))
    quant = args.quantization or precision or ("fp32" if backend == "onnx" else "fp16")
    buckets, chunks = bucket_layout(files)
    if backend == "onnx":
        buckets, chunks = [], 1
    shared = any(len(e["paths"]) > 1 and layer(e["paths"][0]) == "weights" for e in group(files))
    dg = digest(files)
    variant = args.variant or quant.replace(" ", "-")
    entry = {
        "id": f"{model}/{backend}/{variant}/{dg[:12]}",
        "model": model,
        "source": source,
        "backend": backend,
        "variant": variant,
        "quantization": quant,
        "buckets": buckets,
        "chunks": chunks,
        "max_seq_len": manifest.get("max_seq_len"),
        "layout": "shared" if shared else "separate",
        "digest": dg,
        "fingerprint": fingerprint(files),
        "produced": {
            "converter": args.converter or converter_of(committed_manifest(model) or manifest_text),
            "converter_commit": args.converter_commit,
            "coremltools": meta.get("coremltools"),
            "torch": meta.get("torch"),
            "python": args.python,
            "os": args.os,
            "xcode": args.xcode,
            "chip": args.chip,
            "date": args.date or meta.get("date"),
        },
        "files": group(files),
        "grades": matching_grades(model, files, args.reports or [], args.any_report) + [parse_grade(g) for g in args.grade or []],
        "published": [],
        "recorded": args.recorded or datetime.date.today().isoformat(),
        "evidence": evidence,
        "note": args.note,
    }
    if args.dry_run:
        print(json.dumps(entry, indent=1))
        return
    ledger = load(args.ledger)
    old = next((b for b in ledger["builds"] if b["id"] == entry["id"]), None)
    if old:
        entry["published"] = old["published"]
        entry["grades"] = entry["grades"] or old["grades"]
        ledger["builds"].remove(old)
    ledger["builds"].append(entry)
    save(ledger, args.ledger)
    n = sum(len(e["paths"]) for e in entry["files"])
    print(f"{'replaced' if old else 'added'} {entry['id']}: {n} paths, {len(entry['files'])} distinct files, "
          f"{len(entry['grades'])} grades")


def parse_grade(text):
    """"path:chip:letter[:report]" from the command line."""
    parts = text.split(":")
    if len(parts) < 3 or parts[0] not in PATHS or parts[2] not in LETTERS:
        raise SystemExit(f"--grade {text!r}: expected path:chip:letter[:report]")
    return {"path": parts[0], "chip": parts[1], "os": None, "grade": parts[2],
            "report": parts[3] if len(parts) > 3 else None}


def compare(entry, files, partial=False):
    """Problems with `files` against a ledger entry, each naming the layer
    that moved."""
    want = flatten(entry["files"])
    out = []
    for rel in sorted(set(want) | set(files)):
        if rel not in files:
            if not partial:
                out.append(f"{layer(rel)}: missing {rel}")
        elif rel not in want:
            out.append(f"{layer(rel)}: not in the ledger: {rel}")
        elif files[rel][0] != want[rel][0]:
            out.append(f"{layer(rel)}: sha256 differs: {rel}")
        elif want[rel][1] is not None and files[rel][1] != want[rel][1]:
            out.append(f"{layer(rel)}: size differs: {rel}")
    for e in entry["files"]:
        held = {files[p][0] for p in e["paths"] if p in files}
        if len(held) > 1:
            out.append(f"{layer(e['paths'][0])}: a shared file split: {', '.join(e['paths'])}")
    return out


def cmd_verify(args):
    ledger = load(args.ledger)
    root = Path(args.dir)
    files = hash_tree(root)
    if args.id:
        cands = [b for b in ledger["builds"] if b["id"] == args.id]
        if not cands:
            raise SystemExit(f"no build {args.id!r} in the ledger")
    else:
        dg = digest(files)
        exact = [b for b in ledger["builds"] if b["digest"] == dg]
        if exact:
            print(f"{root}: matches {exact[0]['id']}")
            return
        mpath = next((root / n for n in ("manifest.toml", "classifier.toml") if (root / n).exists()), None)
        model = read_manifest(mpath.read_text()).get("id") if mpath else None
        cands = [b for b in ledger["builds"] if b["model"] == model] if model else []
        if not cands:
            raise SystemExit(f"{root}: no ledger build matches, and none is recorded for model {model!r}")
    best = min(cands, key=lambda b: len(compare(b, files, args.partial)))
    problems = compare(best, files, args.partial)
    if problems:
        raise SystemExit(f"{root} differs from {best['id']}:\n  " + "\n  ".join(problems))
    print(f"{root}: matches {best['id']}" + (" (the files present)" if args.partial else ""))


def cmd_published(args):
    ledger = load(args.ledger)
    b = next((b for b in ledger["builds"] if b["id"] == args.id), None)
    if b is None:
        raise SystemExit(f"no build {args.id!r} in the ledger")
    record = {"repo": args.repo, "revision": args.revision, "date": args.date or datetime.date.today().isoformat(),
              "sidekick": args.sidekick}
    if record not in b["published"]:
        b["published"].append(record)
    save(ledger, args.ledger)
    print(f"{b['id']}: published at {args.repo}@{args.revision}")


def cmd_list(args):
    for b in load(args.ledger)["builds"]:
        if args.model and b["model"] != args.model:
            continue
        size = sum(e["size"] or 0 for e in b["files"])
        grades = " ".join(sorted({f"{g['path'].split('-', 1)[1]}:{g['grade']}" for g in b["grades"]})) or "-"
        print(f"{b['id']:70} {size / 1e6:9.1f} MB  {b['layout']:8} {grades}")


# --- validation --------------------------------------------------------------

def problems(ledger, models):
    """Everything wrong with the ledger's shape or references."""
    out = []
    if ledger.get("schema") != SCHEMA:
        out.append(f"ledger: schema {ledger.get('schema')!r}, expected {SCHEMA}")
    ids = set()
    for i, b in enumerate(ledger.get("builds", [])):
        w = f"ledger build {b.get('id', i)!r}"
        for k, t in ENTRY_FIELDS.items():
            if k not in b:
                out.append(f"{w}: no {k}")
            elif not isinstance(b[k], t):
                out.append(f"{w}: {k} has the wrong type")
        if extra := set(b) - set(ENTRY_FIELDS):
            out.append(f"{w}: unknown fields {sorted(extra)}")
        if any(k not in b or not isinstance(b[k], ENTRY_FIELDS[k]) for k in ENTRY_FIELDS):
            continue
        if b["id"] in ids:
            out.append(f"{w}: duplicate id")
        ids.add(b["id"])
        if b["model"] not in models:
            out.append(f"{w}: model {b['model']!r} isn't in models.toml")
        repo, _, rev = b["source"].partition("@")
        if not repo or not HEX40.match(rev):
            out.append(f"{w}: source must be repo@<40-hex revision>")
        if b["backend"] not in BACKENDS:
            out.append(f"{w}: backend {b['backend']!r}")
        if b["quantization"] not in QUANTIZATIONS:
            out.append(f"{w}: quantization {b['quantization']!r}")
        if b["layout"] not in ("separate", "shared"):
            out.append(f"{w}: layout {b['layout']!r}")
        if b["evidence"] not in EVIDENCE:
            out.append(f"{w}: evidence {b['evidence']!r}")
        if set(b["produced"]) != set(PRODUCED_FIELDS):
            out.append(f"{w}: produced must have exactly {', '.join(PRODUCED_FIELDS)}")
        paths = [p for e in b["files"] for p in e.get("paths", [])]
        if len(paths) != len(set(paths)):
            out.append(f"{w}: a path is listed twice")
        for e in b["files"]:
            if not HEX64.match(e.get("sha256", "")):
                out.append(f"{w}: bad sha256 {e.get('sha256')!r}")
            if not e.get("paths"):
                out.append(f"{w}: a file with no paths")
            size = e.get("size")
            if not (isinstance(size, int) or (size is None and b["evidence"] == "sha256sums")):
                out.append(f"{w}: {e.get('paths')}: size must be a byte count")
        if not any(o.startswith(w + ":") for o in out):
            files = flatten(b["files"])
            dg = digest(files)
            if b["digest"] != dg:
                out.append(f"{w}: digest doesn't match its files")
            if not b["id"] == f"{b['model']}/{b['backend']}/{b['variant']}/{dg[:12]}":
                out.append(f"{w}: id must be model/backend/variant/<digest[:12]>")
            if b["fingerprint"] != fingerprint(files):
                out.append(f"{w}: fingerprint doesn't match its files")
            if b["backend"] == "coreml":
                buckets, chunks = bucket_layout(files)
                if (buckets, chunks) != (b["buckets"], b["chunks"]):
                    out.append(f"{w}: buckets/chunks don't match its programs")
        for g in b["grades"]:
            if g.get("path") not in PATHS or g.get("grade") not in LETTERS or not g.get("chip"):
                out.append(f"{w}: bad grade {g}")
        for p in b["published"]:
            if not (p.get("repo") and p.get("revision") and p.get("date") and p.get("sidekick")):
                out.append(f"{w}: a publication needs repo, revision, date and sidekick")
    return out


def cmd_check(args):
    ledger = load(args.ledger)
    found = problems(ledger, load_models())
    if found:
        raise SystemExit("docs/data/ledger.json:\n  " + "\n  ".join(found))
    print(f"ledger: {len(ledger['builds'])} builds, valid")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ledger", type=Path, default=LEDGER)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="record a build")
    a.add_argument("dir", nargs="?", help="the built model directory")
    a.add_argument("--model", help="model id (default: the manifest's)")
    a.add_argument("--backend", choices=BACKENDS)
    a.add_argument("--variant", help="the build's name in its repository (default: its quantization)")
    a.add_argument("--quantization", choices=QUANTIZATIONS)
    a.add_argument("--source", help="repo@revision (default: the manifest's, else models.toml's)")
    a.add_argument("--manifest", help="hash this file as the build's manifest")
    a.add_argument("--exclude", action="append", help="a glob of paths to leave out")
    a.add_argument("--reports", action="append", help="parity reports to take matching grades from")
    a.add_argument("--any-report", action="store_true",
                   help="take grades from reports runs.toml doesn't describe too (a release candidate's own)")
    a.add_argument("--grade", action="append", help="path:chip:letter[:report], when no report matches by hash")
    a.add_argument("--converter")
    a.add_argument("--converter-commit")
    a.add_argument("--python")
    a.add_argument("--os")
    a.add_argument("--xcode")
    a.add_argument("--chip")
    a.add_argument("--date", help="conversion date (default: the programs' metadata)")
    a.add_argument("--recorded", help="date recorded (default: today)")
    a.add_argument("--note")
    a.add_argument("--sums", help="a SHA256SUMS listing to record instead of hashing")
    a.add_argument("--sums-prefix", help="the build's directory inside the listing, with a trailing /")
    a.add_argument("--sizes", help="'size path' lines for the --sums listing")
    a.add_argument("--dry-run", action="store_true")
    v = sub.add_parser("verify", help="check a directory against its ledger entry")
    v.add_argument("dir")
    v.add_argument("--id")
    v.add_argument("--partial", action="store_true", help="accept missing paths")
    sub.add_parser("check", help="validate the ledger")
    li = sub.add_parser("list", help="list builds")
    li.add_argument("model", nargs="?")
    pb = sub.add_parser("published", help="record that a build was uploaded")
    pb.add_argument("id")
    pb.add_argument("--repo", required=True, help="the Hugging Face repository, owner/name")
    pb.add_argument("--revision", required=True, help="the repository commit that holds the build")
    pb.add_argument("--sidekick", required=True, help="the sidekick release it was published with")
    pb.add_argument("--date")
    args = p.parse_args()
    if args.cmd == "add" and not args.dir and not args.sums:
        p.error("add needs a model directory or --sums")
    {"add": cmd_add, "verify": cmd_verify, "check": cmd_check, "list": cmd_list,
     "published": cmd_published}[args.cmd](args)


if __name__ == "__main__":
    main()
