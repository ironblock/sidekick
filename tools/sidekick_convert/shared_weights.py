"""One weights file per chunk, shared by every bucket (docs/DECISIONS.md D49).

Each Core ML bucket is its own program with its own `weights/weight.bin`.
The weights are the same in every bucket, but the files differ: coremltools
writes constants in program order, and the few that depend on the bucket
(position and token-type tables, RoPE tables, the causal mask) shift every
tensor after them. A model with three buckets stores its weights three
times.

`share()` re-exports every bucket's program for one chunk against a single
union weights file: each distinct tensor is written once, at one offset,
and every bucket's program points into the same file. The re-export loads
each saved .mlpackage back into MIL and writes it again with no graph
passes, so the programs are unchanged except for the offsets of their
constants. Once compiled, every bucket's `weights/weight.bin` is the same
file, byte for byte, and is installed as one file with a hard link per
bucket.

The step is gated, and a bucket set that fails a gate is not installed:
- every compiled weights file in the chunk must be byte-identical to the
  union file;
- each compiled program (`model.mil`) must equal the separate build's once
  weight offsets are set aside;
- on CPU_ONLY, CPU_AND_GPU and CPU_AND_NE, every program must give the
  separate build's outputs bit for bit on the gate cases. Each comparison
  runs in a child process with a Core ML cache of its own, deleted when it
  exits, so the check doesn't leave a compiled copy of every bucket behind.
A union file over the Neural Engine's per-program limit (D32) isn't an
error: the ANE judges the whole file, so for a model served on the ANE that
chunk keeps one file per bucket, and the converter says so.

What is shared and what isn't: the files on disk, and so the download, are.
Core ML's compile cache still keeps a specialized copy per bucket and per
compute path, and each loaded bucket maps its own copy, so neither the
cache nor memory shrinks. Sharing is per chunk only: the ANE limit applies
to the whole file, so a file spanning chunks would push each over it.

`link_identical()` is the install side: it hard-links every set of
identical weight files under a model directory to one file, falling back to
the copies where the filesystem refuses a link. `python -m
sidekick_convert.shared_weights link <model-dir>` runs it on a directory
installed by other means (copied without preserving hard links, or
downloaded).
"""

import dataclasses
import filecmp
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE")
"""The compute paths the re-exported programs must match the separate build
on, bit for bit."""

WEIGHT_FILE = Path("weights") / "weight.bin"
_TOOLS = Path(__file__).resolve().parents[1]


class ShareFailure(Exception):
    """A shared-weights gate failed. core.run turns it into a GateFailure."""


@dataclasses.dataclass
class Program:
    """One compiled program of a bucket (the whole bucket, or one chunk),
    waiting to be installed. `package` is the saved .mlpackage it was
    compiled from, kept for the re-export."""
    seq: int
    chunk: object          # None for an unchunked bucket
    compiled: Path
    package: Path
    inputs: tuple
    output: str

    @property
    def name(self):
        return f"model_{self.seq}.mlmodelc" if self.chunk is None else f"model_{self.seq}.{self.chunk}.mlmodelc"


def _content_key(var):
    val = np.ascontiguousarray(var.val)
    return str(var.dtype), tuple(val.shape), hashlib.sha256(val.tobytes()).hexdigest()


def reexport(packages, out_dir):
    """Re-export the programs in `packages` ({seq: .mlpackage}, one chunk's
    buckets) against one union weights file, `out_dir`/weight.bin. Returns
    ({seq: Model proto}, the union file). Each distinct constant (by dtype,
    shape and bytes) is written once, in bucket order, so the tensors every
    bucket holds come first and each bucket's own follow."""
    from coremltools import proto
    from coremltools.converters.mil.backend.mil import helper
    from coremltools.converters.mil.backend.mil import load as backend
    from coremltools.converters.mil.frontend.milproto import load as milproto
    from coremltools.models.utils import _try_get_weights_dir_path, load_spec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = out_dir / "weight.bin"
    writer = backend.BlobWriter(str(blob))
    written = {}

    class UnionExporter(backend.MILProtoExporter):
        def create_file_value(self, var):
            key = _content_key(var)
            if key not in written:
                offset = helper._get_offset_by_writing_data(var, writer)
                prim = var.sym_type.get_primitive() if hasattr(var.sym_type, "get_primitive") else var.sym_type
                written[key] = backend.create_file_value_tensor(
                    file_name="@model_path/weights/weight.bin", offset=offset, dim=var.val.shape,
                    data_type=backend.types_to_proto_primitive(prim))
            return written[key]

    models = {}
    for seq in sorted(packages):
        pkg = str(packages[seq])
        spec = load_spec(pkg)
        prog = milproto.load(spec, spec.specificationVersion, _try_get_weights_dir_path(pkg))
        mil = UnionExporter(prog, str(out_dir), spec.specificationVersion).export()
        model = proto.Model_pb2.Model()
        model.CopyFrom(spec)
        for name, fn in mil.functions.items():
            model.mlProgram.functions[name].CopyFrom(fn)
        models[seq] = model
        del prog
    del writer      # flushes and closes the union file
    return models, blob


def _package(model, blob, path):
    """An .mlpackage of `model` whose weights are `blob`."""
    from coremltools.models.utils import _create_mlpackage
    with tempfile.TemporaryDirectory(dir=Path(path).parent) as weights:
        os.link(blob, Path(weights) / "weight.bin")
        _create_mlpackage(model, weights, str(path))
    return Path(path)


_OFFSET = re.compile(r"(BLOBFILE\(path = string\(\"[^\"]*\"\), offset = uint64\()\d+\)")


def program_text(compiled):
    """The compiled program's text with every weight offset set aside: equal
    for two builds that differ only in where their constants sit."""
    return _OFFSET.sub(r"\1_)", (Path(compiled) / "model.mil").read_text())


def _weight_files(compiled):
    return sorted(f for f in (Path(compiled) / "weights").rglob("*") if f.is_file())


def files_bytes(paths):
    """The bytes the files take on disk, a file hard-linked more than once
    counted once."""
    seen, total = set(), 0
    for p in paths:
        st = Path(p).stat()
        if (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def build_union(group, work, compile_mlmodelc, log=print, keep=None):
    """Re-export, package and compile one chunk's programs (`group`, one
    Program per bucket) against a union weights file in `work`. Returns
    ({seq: compiled union program}, union bytes). Raises ShareFailure if a
    compiled weights file isn't the union file byte for byte. The separate
    packages are deleted once re-exported; every compiled weights file ends
    as a hard link to the union file. With `keep` (a directory), each
    re-exported package is copied to `keep`/shared/ before it is deleted."""
    work = Path(work)
    models, blob = reexport({p.seq: p.package for p in group}, work / "blob")
    for p in group:
        shutil.rmtree(p.package, ignore_errors=True)
    out = {}
    for p in group:
        pkg = _package(models.pop(p.seq), blob, work / f"model_{p.seq}.mlpackage")
        keep_package(pkg, keep, "shared", _package_name(p))
        compiled = compile_mlmodelc(pkg, work / f"compiled_{p.seq}")
        shutil.rmtree(pkg)
        files = _weight_files(compiled)
        if [f.relative_to(compiled) for f in files] != [WEIGHT_FILE]:
            raise ShareFailure(f"bucket {p.seq}: the compiled program has weight files "
                               f"{[str(f.relative_to(compiled)) for f in files]}, expected only {WEIGHT_FILE}")
        if not filecmp.cmp(blob, compiled / WEIGHT_FILE, shallow=False):
            raise ShareFailure(f"bucket {p.seq}: coremlcompiler's {WEIGHT_FILE} differs from the union weights file")
        os.unlink(compiled / WEIGHT_FILE)
        os.link(blob, compiled / WEIGHT_FILE)
        if program_text(compiled) != program_text(p.compiled):
            raise ShareFailure(f"{p.name}: the re-exported program differs from the separate build's beyond "
                               "its weight offsets")
        out[p.seq] = compiled
    size = blob.stat().st_size
    os.unlink(blob)
    return out, size


def compare(separate, union, feeds, units, log=print):
    """Whether two builds of one bucket give the same outputs bit for bit on
    `units`: `separate` and `union` are lists of Programs, one per chunk, in
    order, and `feeds` a list of {port: int32 array}. Each chunk of the
    union build gets the separate build's input, so every program is judged
    on its own. Runs in a child process with a private Core ML cache.
    Returns the child's report: {"identical": bool, "cases": n, "outputs": n,
    "first_difference": ...}."""
    with tempfile.TemporaryDirectory(prefix="sidekick-share-") as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "feeds.npz", **{f"{i}/{k}": v for i, f in enumerate(feeds) for k, v in f.items()})
        spec = {"units": units, "cases": len(feeds), "feeds": str(tmp / "feeds.npz"),
                "separate": [str(p.compiled) for p in separate], "union": [str(p.compiled) for p in union],
                "inputs": [list(p.inputs) for p in separate], "outputs": [p.output for p in separate]}
        (tmp / "spec.json").write_text(json.dumps(spec))
        home = tmp / "home"
        home.mkdir()
        env = {**os.environ, "CFFIXED_USER_HOME": str(home), "SIDEKICK_CONVERT_PRIVATE_CACHE": str(home),
               "PYTHONPATH": os.pathsep.join([str(_TOOLS)] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
        child = subprocess.run([sys.executable, "-m", "sidekick_convert.shared_weights", "_compare",
                                str(tmp / "spec.json"), str(tmp / "result.json")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if child.returncode != 0 or not (tmp / "result.json").exists():
            tail = "\n".join(child.stdout.strip().splitlines()[-20:])
            raise ShareFailure(f"the {units} comparison failed (exit {child.returncode}):\n{tail}")
        return json.loads((tmp / "result.json").read_text())


def _compare_child(spec_path, result_path):
    """The child side of compare(): load both builds for one compute path
    and compare every program's output on every case, bit for bit."""
    import coremltools as ct
    from .core import purge_coreml_cache
    spec = json.loads(Path(spec_path).read_text())
    units = getattr(ct.ComputeUnit, spec["units"])
    keep = []     # Core ML releases a prediction's inputs later, on its own queue (gates._INPUTS)
    feeds = np.load(spec["feeds"])
    result = {"units": spec["units"], "cases": spec["cases"], "outputs": 0, "identical": True}
    hidden = [None] * spec["cases"]
    # one chunk at a time, both builds of it, so the cache holds two programs at most
    for k, (sep, uni, names, output) in enumerate(zip(spec["separate"], spec["union"], spec["inputs"],
                                                      spec["outputs"])):
        a = ct.models.CompiledMLModel(sep, compute_units=units)
        b = ct.models.CompiledMLModel(uni, compute_units=units)
        for i in range(spec["cases"]):
            feed = {n: (hidden[i] if n == "hidden_in" else np.ascontiguousarray(feeds[f"{i}/{n}"])) for n in names}
            keep.append(feed)
            x = np.asarray(a.predict(feed)[output])
            y = np.asarray(b.predict(feed)[output])
            result["outputs"] += 1
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                result["identical"] = False
                diff = np.abs(x.astype(np.float64) - y.astype(np.float64)) if x.shape == y.shape else None
                result.setdefault("first_difference", {
                    "case": i, "program": k, "dtypes": [str(x.dtype), str(y.dtype)],
                    "max_abs": None if diff is None else float(np.nanmax(diff))})
            hidden[i] = x
        del a, b
        purge_coreml_cache()
    Path(result_path).write_text(json.dumps(result))


def share(job, programs, cases, work, compile_mlmodelc, ane_limited, log=print):
    """The shared-weights step for a whole conversion. `programs` is
    {seq: [Program per chunk]}, `cases` {seq: gate cases}. Rewrites
    `programs` in place to the union builds where every gate passed, deletes
    the separate builds it replaces, and returns a report. `ane_limited`:
    the model is served on the ANE, so a union file must stay under
    plan.MAX_ANE_PROGRAM_WEIGHT_BYTES. Raises ShareFailure."""
    from .plan import MAX_ANE_PROGRAM_WEIGHT_BYTES
    seqs = sorted(programs)
    chunks = len(programs[seqs[0]])
    work = Path(work)
    report = {"chunks": {}}
    unions = {seq: [None] * chunks for seq in seqs}
    for k in range(chunks):
        group = [programs[seq][k] for seq in seqs]
        separate_bytes = sum(files_bytes(_weight_files(p.compiled)) for p in group)
        keep = getattr(job, "keep_packages", None)
        built, size = build_union(group, work / f"union_{k}", compile_mlmodelc, log, keep)
        what = "" if chunks == 1 else f"chunk {k}: "
        r = report["chunks"][k] = {"separate_bytes": separate_bytes, "union_bytes": size}
        if ane_limited and size > MAX_ANE_PROGRAM_WEIGHT_BYTES:
            r["shared"] = False
            log(f"shared weights: {what}the union weights file would be {size / 2**30:.3f} GiB, over the Neural "
                "Engine's per-program limit (D32), so this chunk keeps one weights file per bucket")
            for compiled in built.values():
                shutil.rmtree(compiled.parent, ignore_errors=True)
            if keep:
                for p in group:
                    shutil.rmtree(Path(keep) / "shared" / _package_name(p), ignore_errors=True)
            continue
        r["shared"] = True
        for seq in seqs:
            unions[seq][k] = built[seq]
        log(f"shared weights: {what}{len(seqs)} buckets, {separate_bytes / 1e6:.1f} MB as separate files, "
            f"{size / 1e6:.1f} MB shared; every compiled weights file is the union file, byte for byte")
    if not any(r["shared"] for r in report["chunks"].values()):
        return report
    report["identical"] = {}
    for seq in seqs:
        separate = programs[seq]
        union = [dataclasses.replace(p, compiled=unions[seq][k] or p.compiled) for k, p in enumerate(separate)]
        feeds = [c.feed(seq, job.ports) for c in cases[seq]]
        for units in UNITS:
            res = compare(separate, union, feeds, units, log)
            report["identical"].setdefault(seq, {})[units] = res["identical"]
            if not res["identical"]:
                raise ShareFailure(f"bucket {seq} [{units}]: the shared-weights build differs from the separate "
                                   f"build ({res.get('first_difference')}); it must be bit-identical. Convert "
                                   "with --no-shared-weights to install separate files.")
            log(f"bucket {seq} [{units}]: shared-weights build bit-identical to the separate build "
                f"({res['outputs']} outputs, {res['cases']} cases)")
        for k, p in enumerate(separate):
            if unions[seq][k] is not None:
                shutil.rmtree(p.compiled, ignore_errors=True)
                p.compiled = unions[seq][k]
    return report


def _package_name(program):
    return program.name.removesuffix(".mlmodelc") + ".mlpackage"


def keep_package(pkg, keep, kind, name=None):
    """Copy the .mlpackage `pkg` to `keep`/`kind`/ (as `name`, else its own
    name), when a directory to keep packages in was given (`--keep-packages`):
    the converter otherwise deletes every package once it is compiled."""
    if not keep:
        return
    dest = Path(keep) / kind / (name or Path(pkg).name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(pkg, dest)


def link_kept(keep, log=print):
    """Hard-link identical weight files among the kept packages (a
    shared-weights build's re-exported packages all hold the union file),
    and say what was kept."""
    pkgs = sorted(Path(keep).glob("*/*.mlpackage"))
    data = [p / "Data" / "com.apple.CoreML" for p in pkgs]
    linked, _ = link_identical(data, log)
    files = [f for d in data for f in _weight_files(d)]
    log(f"kept {len(pkgs)} .mlpackage source(s) in {keep}: weight files {files_bytes(files) / 1e6:.1f} MB on disk"
        + (f", {linked} hard-linked" if linked else ""))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def link_identical(artifacts, log=print):
    """Hard-link identical weight files across `artifacts` (compiled
    .mlmodelc directories) to one file each. A file the filesystem won't
    link (another volume, or no hard links) keeps its copy. Returns
    (files linked, bytes saved)."""
    by_size = {}
    for a in artifacts:
        for f in _weight_files(a):
            by_size.setdefault(f.stat().st_size, []).append(f)
    linked = saved = 0
    for size, files in by_size.items():
        if len(files) < 2:
            continue
        by_hash, inode_hash = {}, {}
        for f in files:
            st = f.stat()
            node = (st.st_dev, st.st_ino)
            if node not in inode_hash:
                inode_hash[node] = _sha256(f)
            by_hash.setdefault(inode_hash[node], []).append(f)
        for same in by_hash.values():
            first = same[0]
            for f in same[1:]:
                if os.path.samefile(first, f):
                    continue
                tmp = f.with_name(f".{f.name}.link")
                try:
                    if tmp.exists():
                        tmp.unlink()
                    os.link(first, tmp)
                except OSError as e:
                    log(f"can't hard-link {f} to {first} ({e.strerror}); keeping the copy")
                    continue
                os.replace(tmp, f)
                linked += 1
                saved += size
    return linked, saved


def model_artifacts(model_dir):
    """The compiled programs in an installed model directory."""
    return sorted(p for p in Path(model_dir).glob("*.mlmodelc") if p.is_dir())


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m sidekick_convert.shared_weights",
                                description="Hard-link identical Core ML weight files in an installed model "
                                            "directory (docs/DECISIONS.md D49).")
    sub = p.add_subparsers(dest="cmd", required=True)
    link = sub.add_parser("link", help="hard-link identical weight files across a model's buckets")
    link.add_argument("model_dir", type=lambda s: Path(s).expanduser())
    child = sub.add_parser("_compare", help=argparse.SUPPRESS)
    child.add_argument("spec")
    child.add_argument("result")
    args = p.parse_args(argv)
    if args.cmd == "_compare":
        return _compare_child(args.spec, args.result)
    artifacts = model_artifacts(args.model_dir)
    if not artifacts:
        raise SystemExit(f"no .mlmodelc under {args.model_dir}")
    files = [f for a in artifacts for f in _weight_files(a)]
    before = files_bytes(files)
    linked, saved = link_identical(artifacts)
    print(f"{args.model_dir}: {len(artifacts)} programs, weight files {before / 1e6:.1f} MB -> "
          f"{files_bytes(files) / 1e6:.1f} MB ({linked} hard-linked)")


if __name__ == "__main__":
    sys.exit(main())
