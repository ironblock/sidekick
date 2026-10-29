"""The pre-tag release gate: run each published model's task sample through
the release candidate's sidekickd over HTTP, compare the numbers with the
committed baseline (docs/data/release-gate.json), and fail if any of them
drifts beyond its stated tolerance.

Why. A build's fidelity is checked when it is converted: the parity suite
(D26) compares it with fp32 on every path, and task reproduction checks it
against its published numbers. CI covers the runtime only with tiny fixture
models. This gate re-runs real models through the daemon about to be
released, so a runtime change that shows only with a full model (how
`nli_probs` maps classes, how an nli pair is encoded, a prompt or tokenizer
change) is caught before the tag rather than after.

What it does:
1. Builds sidekickd (`cargo build --release -p sidekick-server --features
   coreml,onnx`) and signs it with tools/sign_sidekickd.sh, or takes
   `--daemon <binary>`, such as the release archive's.
2. Finds each published model's builds (docs/data/models.toml) in the
   models directories given, and identifies each build in the checksum
   ledger (docs/data/ledger.json, the same hashes `tools/ledger.py verify`
   compares), so the baseline records exactly which build it graded.
3. For each model, one at a time, on its default path (models.toml's
   `served`, the fp16 Core ML build) and on any other paths asked for with
   `--paths` (every (path, precision) docs/data/models.json grades on this
   chip, such as onnx-cpu's fp32, fp16-weights and int8 builds): serves
   the build from sidekickd on 127.0.0.1, on a free port outside 8790-8799,
   in the gate's own home (below), with a models directory holding only
   that build. A path other than the default is served through the
   config's `[models."<id>"] compute_units` (D38); the default must be
   what the build's own manifest serves. A Core ML build is warmed first
   (`sidekickd warm`, D52), so compiles happen before any request is timed.
4. Runs the model's task sample through the HTTP API:
   - nli models: the MultiNLI pairs of the nli parity corpus (308, every
     64th validation row), each sent as one label with the template "{}":
     three-way accuracy from `nli_probs`, two-way (entailment or not) from
     `probs`, and agreement with fp32 for both; and, for the models the
     verification harness scores, the textbook MNLI triple through its
     exact client path and through the stock request
     (`tools/tasks/scoring_run.py sanity`).
   - other classifiers (laya, julia, gliner2, fev, agentjev and
     text-classification formats): the parity corpus's labelled cases,
     batched as a client would: gold accuracy and agreement with fp32.
   - rerankers: the rerank corpus (fixtures/rerank/corpus.toml): the
     share of groups whose top document is one a person marked relevant,
     and agreement with fp32 (top document, largest score difference).
   - embedders: a fixed MTEB slice through tools/mteb_transfer.py's client
     path, and the parity corpus's vectors' cosine with fp32's.
   - chat models (D53): the first 12 items of each chat-benchmark family
     (fixtures/tasks/chat/), prompt-only and, for the families with a
     schema, guided (`response_format`), at temperature 0, run twice:
     every reply must be byte-identical between the two runs; each
     family's headline score (tools/chat_bench's scorers), the guided
     replies that don't satisfy their schema, and the requests whose
     `usage` differs from the build's tokenizer and template (a guided
     request's prompt count, which carries the daemon's schema instruction,
     exactly) are compared with the baseline (tools/chat_bench/gate_score.py, in the chat
     benchmark's environment: `--chat-python`). The first request loads
     the build; the free disk before and after it is recorded as the Neural
     Engine compile cache the load left, when it compiled.
5. Compares every number with the baseline entry for the same model, path
   and build variant, under the baseline's tolerances, and ends with a
   one-line summary.

The gate's home. Core ML keys a compiled program by, among other things,
the home directory, the program's path and its weights file's inode, and
on the ANE the loading binary's signing identifier; the ANE's compiled
programs live in a system cache nobody can delete, about one copy of the
weights per compile (docs/design/coreml-cache.md). A fresh home therefore
recompiles every ANE program and leaks that much disk on every run. So the
gate keeps one home, `--home` (default
~/Library/Caches/sidekick-release-gate/), across runs:
- `home/` is sidekickd's HOME and CFFIXED_USER_HOME: Core ML's and
  sidekick's caches;
- `models/<model>/<build>/<model>/` serves each build in place: a
  directory of hard links to the build's own files (the same inodes, so
  the same cache keys and the same ledger hashes) at a path that doesn't
  change between runs. Only a manifest the gate must amend (below) is a
  copy. A build is never modified;
- `hashes.json` caches file hashes by inode, size and modification time.
After each model, the gate deletes the GPU and CPU programs (about one copy
of the weights each, recompiled in a second or two) from its home, unless
`--keep-gpu-cache`, and keeps every ANE entry: its compiled program is in
the system cache, and the entry (with the CPU segments the ANE program
hands off to) is what lets the next run find it. `--fresh-home` uses a temporary home instead,
deleted afterwards, for when a cold measurement is the point; every ANE
program then compiles again.

Safety. Each record notes the free disk before and after it and how many
programs Apple's ANE compiler finished meanwhile (its unified log). The
gate stops before a model when free disk is under `--min-free-gib`
(default 25). A request served on the ANE that takes longer than 90 s (a
hung ANE request fails after about 100 s and can leave the ANE unusable
until a reboot) stops all further ANE work in the run.

Inputs that aren't committed: the builds, and the parity references
(`<refs dir>/<model id>/reference.json` and `reference.safetensors`, from
tools/parity_reference.py and tools/classifier_reference.py), which hold
each sample's requests and the fp32 outputs. Where several references of a
model are found, the gate takes the first whose tokenizer and length cap
match the build's. The baseline records each reference's sha256 and each
build's ledger id and digest, so a different sample or build is reported,
not compared silently.

Which build. The models directories are searched in order, and for each
(model, backend, variant) the first build found is used, unless the
baseline names a ledger build that one of them is: put the published
build's directory first. A published model with no build here is listed as
not available, never fetched. A baselined (model, path, variant) whose
build is missing fails the gate.

An nli build converted before 0.15.0 has no `classes` in its manifest, and
`nli_probs` needs them (D47). The gate serves such a build with the one
`classes` line from its committed example manifest added to a copy of the
manifest, and records that it did.

Usage (macOS). The gate itself needs only the standard library, so any
Python 3.11 or later runs it. Two of its steps run in a subprocess of a
second interpreter, which needs packages: the embedders' MTEB slice
(tools/mteb_transfer.py: mteb==2.26.14 and sentence-transformers) and the
textbook MNLI triple (tools/tasks/scoring_run.py sanity: `tokenizers`).
Build that environment once from the pinned
tools/release_gate.requirements.txt (its header has the commands), and
name its interpreter with `--python` or SIDEKICK_GATE_PYTHON:

    python3 tools/release_gate.py --python <venv>/bin/python \\
        --models-dir <dir> [--models-dir <dir> ...] \\
        --refs <dir> [--refs <dir> ...] [--paths default|all|<path> ...] [--models <id> ...] \\
        [--daemon <sidekickd>] [--home <dir> | --fresh-home] [--json <file>] [--update-baseline]

    --python       the interpreter for those two steps; default
                   $SIDEKICK_GATE_PYTHON, else the one running the gate.
                   It is checked before any model runs.
    --models-dir   a directory of builds (one subdirectory per build),
                   searched in order
    --refs         a directory of parity references, by model id
    --paths        any of `default` (the default path; the default),
                   `all` (every graded path), and path names (coreml-ane,
                   coreml-gpu, coreml-cpu, onnx-cpu): every graded
                   (path, precision) asked for, e.g. `default onnx-cpu`
    --home         the gate's persistent home (above)
    --fresh-home   a temporary home instead, deleted afterwards
    --update-baseline
                   write this run's numbers into the baseline, replacing
                   the entries for the models, paths and variants it ran.
                   Only for a deliberate change: the PR that commits it
                   says why.
    --list         show each published model's builds and references
                   here, and exit

An onnx-cpu build needs ONNX Runtime: `--onnx-runtime <libonnxruntime>`,
ORT_DYLIB_PATH, or the library beside the binary.

For the MTEB slice, mteb reads the tasks' datasets, and each checkpoint's
configuration at its pinned revision (for its prompts), through the
Hugging Face cache, fetching them on first use.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import http.client
import json
import math
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"
BASELINE = DATA / "release-gate.json"
RERANK_CORPUS = REPO / "fixtures" / "rerank" / "corpus.toml"
SCORING = REPO / "fixtures" / "tasks" / "verification" / "scoring.toml"
DEFAULT_HOME = Path.home() / "Library" / "Caches" / "sidekick-release-gate"
SIGNING_IDENTIFIER = "io.github.ironblock.sidekickd"   # tools/sign_sidekickd.sh

sys.path.insert(0, str(REPO / "tools"))
import ledger as ledger_tool  # noqa: E402  (tools/ledger.py: the ledger's hashing, standard library only)

# The compute units each path serves on (D38's config values).
PATH_UNITS = {"coreml-ane": "cpu_and_ne", "coreml-gpu": "cpu_and_gpu", "coreml-cpu": "cpu_only"}
PATH_ORDER = ["coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
RESERVED_PORTS = range(8790, 8800)   # sidekickd's default and the ports maintainers run it on
# docs/data/models.json's precision -> the ledger's variant name.
PRECISION_VARIANT = {"fp16": "fp16", "fp32": "fp32", "fp16 weights": "fp16-weights", "int8": "int8"}
DEFAULT_VARIANT = {"coreml": "fp16", "onnx": "fp32"}
VARIANT_ORDER = ["fp16", "fp32", "fp16-weights", "int8", "palette8"]
MANIFESTS = ("manifest.toml", "classifier.toml", "chat.toml")
# A chat model's published build, by its weights (the ledger's variant).
CHAT_VARIANT = "palette8"
# The chat sample: the first CHAT_ITEMS items of each family, prompt-only, and
# guided for the families whose requests carry a schema.
CHAT_ITEMS = 12
CHAT_FAMILIES = ["titles", "tags_vocab", "tags_free", "extraction", "instructions"]

# A request on the ANE taking longer than this is taken for a hang: a hung
# ANE request fails after about 100 s (docs/design/coreml-cache.md, "A hung
# ANE request"). Requests elsewhere get the daemon's usual generous limit.
ANE_REQUEST_TIMEOUT_SECS = 90
REQUEST_TIMEOUT_SECS = 900
WARM_TIMEOUT_SECS = 3600

# The MTEB slice: an STS task (symmetric prompts) and a reranking task
# (query prompts against documents). Both are deterministic, and short on
# every embedder (seconds to a few minutes on an M1 Max).
MTEB_TASKS = ["STSBenchmark", "AskUbuntuDupQuestions"]
# The parity corpus's stress inputs, which the parity suite grades apart
# from its worst case (crates/sidekick-embed/examples/parity/grade.rs).
STRESS_TAG = "degenerate"

# What counts as drift, written into the baseline when it is updated; a run
# is judged by the baseline's own copy. `abs`: |run - baseline| <= value.
# `flips`: the share of items whose prediction differs from the baseline's.
# `error`: a run's error (1 - worst cosine, or the largest score difference
# from fp32) may grow to max(factor x baseline, baseline + floor).
TOLERANCES = {
    "accuracy": {"abs": 0.01, "about": "task accuracy (MNLI, gold labels, relevant top document)"},
    "agreement": {"abs": 0.01, "about": "share of items whose prediction matches fp32's"},
    "flips": {"max_share": 0.01, "about": "share of items whose prediction differs from the baseline run's"},
    "mean_prob": {"abs": 0.002, "about": "mean probability of the predicted label (catches drift without flips)"},
    "mteb": {"abs": 0.002, "about": "MTEB main score"},
    "error": {"factor": 2.0, "floor": 1e-5,
              "about": "1 - worst cosine with fp32 (embedders), largest |score - fp32| (rerankers)"},
    "sanity": {"abs": 0.01, "about": "each class probability in the textbook MNLI triple"},
    "chat_score": {"abs": 0.02, "about": "a chat-benchmark family's headline score (mode:family), over 12 items"},
    "exact": {"about": "counts that must not change: refused requests, token-count mismatches, "
                       "sanity verdicts, probs/nli_probs disagreements; a chat model's replies that differ "
                       "between two runs at temperature 0, and its guided replies that fail their schema"},
}


class GateError(Exception):
    """A model that couldn't be run: reported, and the gate fails."""


class Hang(GateError):
    """A request that outlasted its timeout: on the ANE, all ANE work stops."""


# ---------------------------------------------------------------- inputs


def models_toml() -> dict:
    return tomllib.loads((DATA / "models.toml").read_text())["models"]


def chip() -> str:
    out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return out.stdout.strip() or platform.processor()


def macos() -> tuple[str, str]:
    def sw(flag):
        return subprocess.run(["sw_vers", flag], capture_output=True, text=True).stdout.strip()
    return sw("-productVersion"), sw("-buildVersion")


def graded_targets(model: str, this_chip: str) -> list[tuple[str, str]]:
    """The (path, variant) pairs docs/data/models.json grades `model` on,
    on this chip."""
    rows = json.loads((DATA / "models.json").read_text())["rows"]
    got = {(r["path"], PRECISION_VARIANT.get(r["precision"], r["precision"]))
           for r in rows if r["model"] == model and r["chip"] == this_chip}
    return sorted(got, key=lambda t: (PATH_ORDER.index(t[0]),
                                      VARIANT_ORDER.index(t[1]) if t[1] in VARIANT_ORDER else 99))


class Build:
    """A build directory: its manifest, and (once hashed) its ledger entry."""

    def __init__(self, path: Path):
        self.dir = path
        self.manifest_name = next(n for n in MANIFESTS if (path / n).is_file())
        m = tomllib.loads((path / self.manifest_name).read_text())
        self.model = m.get("id", path.name)
        self.backend = m.get("backend", "coreml")
        self.chat = self.manifest_name == "chat.toml"
        self.max_seq_len = m["decoder"]["context_length"] if self.chat else m.get("max_seq_len")
        self.files = None
        self.ledger = None
        self.nearest = None

    def identify(self, hashes: "HashCache", ledger: dict):
        """Hash the build, and find it in the ledger."""
        if self.files is not None:
            return
        self.files = hashes.tree(self.dir)
        self.digest = ledger_tool.digest(self.files)
        # sha256 over `<sha256>  <relative path>` lines: the verification
        # runs' model_sha256, and the first baseline's build fingerprint.
        lines = "".join(f"{sha}  {rel}\n" for rel, (sha, _) in sorted(self.files.items()))
        self.tree_sha256 = hashlib.sha256(lines.encode()).hexdigest()
        self.ledger = next((b for b in ledger["builds"] if b["digest"] == self.digest), None)
        if self.ledger is None:
            cands = [b for b in ledger["builds"] if b["model"] == self.model and b["backend"] == self.backend]
            if cands:
                best = min(cands, key=lambda b: len(ledger_tool.compare(b, self.files)))
                self.nearest = (best["id"], ledger_tool.compare(best, self.files)[:3])

    @property
    def variant(self) -> str:
        if self.ledger:
            return self.ledger["variant"]
        return CHAT_VARIANT if self.chat else DEFAULT_VARIANT.get(self.backend, self.backend)

    def record(self) -> dict:
        out = {"ledger": self.ledger["id"] if self.ledger else None, "digest": self.digest,
               "fingerprint": self.tree_sha256, "variant": self.variant, "max_seq_len": self.max_seq_len}
        if self.ledger is None:
            out["ledger_note"] = ("not in the ledger" + (f"; nearest {self.nearest[0]}: " + "; ".join(self.nearest[1])
                                                         if self.nearest else ""))
        return out


class HashCache:
    """sha256 of files, cached by (device, inode, size, modification time)."""

    def __init__(self, path: Path | None):
        self.path = path
        self.cache = {}
        if path and path.exists():
            try:
                self.cache = json.loads(path.read_text())
            except ValueError:
                self.cache = {}

    def tree(self, root: Path) -> dict:
        files = {}
        for rel, p in ledger_tool.walk(root):
            st = p.stat()
            key = f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
            if key not in self.cache:
                self.cache[key] = ledger_tool.sha256_file(p)
            files[rel] = (self.cache[key], st.st_size)
        if self.path:
            self.path.write_text(json.dumps(self.cache))
        return files


def find_builds(dirs: list[Path]) -> dict[tuple[str, str], list[Build]]:
    """{(model id, backend): [builds, in search order]}."""
    out: dict[tuple[str, str], list[Build]] = {}
    for root in dirs:
        if not root.is_dir():
            raise SystemExit(f"--models-dir {root}: not a directory")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if any((d / n).is_file() for n in MANIFESTS):
                b = Build(d)
                out.setdefault((b.model, b.backend), []).append(b)
    return out


def find_reference(dirs: list[Path], model: str, build: Build) -> Path | None:
    """The first reference of `model` made with the build's tokenizer and
    length cap; else the first with its tokenizer; else the first."""
    found = [root / model for root in dirs
             if (root / model / "reference.json").is_file() and (root / model / "reference.safetensors").is_file()]
    if not found:
        return None
    tok = sha256_file(build.dir / "tokenizer.json") if (build.dir / "tokenizer.json").is_file() else None

    def info(d):
        j = json.loads((d / "reference.json").read_text())
        return j.get("tokenizer_sha256"), j["model"].get("max_seq_len")
    infos = [(d, *info(d)) for d in found]
    for d, t, cap in infos:
        if t in (None, tok) and cap == build.max_seq_len:
            return d
    for d, t, _cap in infos:
        if t in (None, tok):
            return d
    return found[0]


def read_tensors(path: Path, names: list[str]) -> dict[str, tuple[list[int], array]]:
    """float32 tensors from a safetensors file, with the standard library:
    {name: (shape, flat array)}."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        for name in names:
            info = header.get(name)
            if info is None:
                raise GateError(f"{path.name}: no tensor {name!r}")
            if info["dtype"] != "F32":
                raise GateError(f"{path.name}: {name} is {info['dtype']}, not F32")
            start, end = info["data_offsets"]
            f.seek(8 + n + start)
            a = array("f")
            a.frombytes(f.read(end - start))
            if sys.byteorder != "little":
                a.byteswap()
            out[name] = (info["shape"], a)
    return out


def sha256_file(path: Path) -> str:
    return ledger_tool.sha256_file(path)


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def ane_compiles(start: float, end: float) -> int | None:
    """Programs Apple's ANE compiler finished between two times, from the
    unified log; None if the log can't be read."""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        p = subprocess.run(["/usr/bin/log", "show", "--style", "compact",
                            "--start", time.strftime(fmt, time.localtime(start - 1)),
                            "--end", time.strftime(fmt, time.localtime(end + 1)),
                            "--predicate", 'process == "ANECompilerService" AND eventMessage BEGINSWITH '
                                           '"End of compilation of network"'],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return sum("End of compilation of network" in line for line in p.stdout.splitlines())


# ---------------------------------------------------------------- the gate's home


class Home:
    """The gate's home: persistent (--home) or temporary (--fresh-home)."""

    def __init__(self, root: Path | None):
        self.temporary = root is None
        self.root = Path(tempfile.mkdtemp(prefix="sidekick-release-gate-")) if root is None else root
        self.home = self.root / "home"
        (self.home / "tmp").mkdir(parents=True, exist_ok=True)
        self.hashes = HashCache(None if self.temporary else self.root / "hashes.json")

    def close(self):
        if self.temporary:
            shutil.rmtree(self.root, ignore_errors=True)

    def serve_dir(self, build: Build) -> tuple[Path, str | None]:
        """The build in place at a stable path: a directory of hard links
        to its files, with an amended manifest where one is needed.
        Returns the models directory to serve, and the amendment's note."""
        mroot = self.root / "models" / build.model / build.digest[:16]
        dest = mroot / build.model
        text, note = served_manifest(build.dir, build.manifest_name, build.model)
        want = set()
        for rel, src in ledger_tool.walk(build.dir):
            want.add(rel)
            d = dest / rel
            if rel == build.manifest_name and note:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists() and os.path.samefile(d, src):
                continue
            if d.exists() or d.is_symlink():
                d.unlink()
            try:
                os.link(src, d)
            except OSError as e:
                raise GateError(f"can't hard-link {src} into the gate's home ({e}); put --home on the same "
                                f"volume as the builds")
        if note:
            m = dest / build.manifest_name
            if m.is_file() and m.stat().st_nlink > 1:
                m.unlink()
            if not m.is_file() or m.read_text() != text:
                m.write_text(text)
        for rel, p in ledger_tool.walk(dest):
            if rel not in want:
                p.unlink()
        return mroot, note

    def prune_gpu_cpu(self) -> int:
        """Delete the GPU and CPU programs in Core ML's caches in the home
        (each about the weights' size, recompiled in a second or two), and
        keep every ANE program's. An ANE entry's `<key>.bundle` holds a
        `model.anehash` (under `main/main_ane/`, or under
        `ane_compile/multiprocedure/` for a program split across the ANE and
        the CPU) and stays whole: its CPU segments (`main_bnns`,
        `main_classic_cpu*`) are part of it, and deleting any of it makes the
        ANE compile again. A bundle without one goes. Returns the bytes
        freed."""
        freed = 0
        caches = self.home / "Library" / "Caches"
        for bundle in caches.glob("*/com.apple.e5rt.e5bundlecache/*/*/*.bundle"):
            if not bundle.is_dir() or next(bundle.rglob("*.anehash"), None) is not None:
                continue
            freed += sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
            shutil.rmtree(bundle, ignore_errors=True)
        return freed


# ---------------------------------------------------------------- daemon


def free_port() -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port
    raise SystemExit("no free port outside 8790-8799")


def served_manifest(build: Path, manifest_name: str, model: str) -> tuple[str, str | None]:
    """The manifest text to serve, and a note when it differs from the
    build's: an nli manifest without `classes` gains its committed example
    manifest's line (D47), as docs/DECISIONS.md tells an operator to."""
    text = (build / manifest_name).read_text()
    m = tomllib.loads(text)
    nli = m.get("classify", {}).get("nli")
    if nli is None or "classes" in nli:
        return text, None
    example = REPO / "examples" / "classifiers" / model / "classifier.toml"
    classes = tomllib.loads(example.read_text())["classify"]["nli"].get("classes") if example.is_file() else None
    if not classes:
        return text, None
    if len(classes) != nli.get("n_classes"):
        raise GateError(f"{model}: the example manifest's classes {classes} don't match n_classes {nli.get('n_classes')}")
    line = "classes = [" + ", ".join(json.dumps(c) for c in classes) + "]\n"
    patched, n = re.subn(r"(?m)^\[classify\.nli\]\n", lambda mt: mt.group(0) + line, text, count=1)
    if n != 1:
        raise GateError(f"{model}: no [classify.nli] header to add classes under")
    return patched, (f"served with `{line.strip()}` added under [classify.nli], from "
                     f"examples/classifiers/{model}/classifier.toml (D47); the build is unchanged")


class Daemon:
    """One sidekickd serving one build from the gate's home."""

    def __init__(self, binary: Path, home: Home, models_dir: Path, model: str, units: str | None, ane: bool,
                 onnx_runtime: str | None):
        self.binary, self.home, self.models_dir, self.model, self.units = binary, home, models_dir, model, units
        self.ane = ane
        self.onnx_runtime = onnx_runtime
        self.run = home.root / "run"
        self.proc = None
        self.port = None
        self.log = None

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def env(self) -> dict:
        # Core ML's compile caches and sidekick's own follow the home.
        env = dict(os.environ, HOME=str(self.home.home), CFFIXED_USER_HOME=str(self.home.home),
                   TMPDIR=str(self.home.home / "tmp"))
        env.pop("SIDEKICK_API_KEY", None)
        return env

    def write_config(self):
        shutil.rmtree(self.run, ignore_errors=True)
        self.run.mkdir(parents=True)
        self.port = free_port()
        timeout = ANE_REQUEST_TIMEOUT_SECS if self.ane else REQUEST_TIMEOUT_SECS
        cfg = [f'addr = "127.0.0.1:{self.port}"', f"models_dir = {json.dumps(str(self.models_dir))}",
               f"request_timeout_secs = {timeout}", "load_timeout_secs = 3600", "model_idle_ttl_secs = 3600",
               "warm_on_start = false"]
        if self.onnx_runtime:
            cfg.append(f"onnx_runtime = {json.dumps(self.onnx_runtime)}")
        if self.units:
            cfg += [f'[models.{json.dumps(self.model)}]', f'compute_units = "{self.units}"']
        (self.run / "config.toml").write_text("\n".join(cfg) + "\n")

    def warm(self) -> float:
        """Compile the build's buckets for the path it's served on before
        any request: `sidekickd warm` (D52), the same binary, home and config."""
        t = time.monotonic()
        try:
            p = subprocess.run([str(self.binary), "--config", str(self.run / "config.toml"), "warm", self.model],
                               env=self.env(), capture_output=True, text=True, timeout=WARM_TIMEOUT_SECS,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise Hang(f"{self.model}: sidekickd warm didn't finish within {WARM_TIMEOUT_SECS} s")
        (self.run / "warm.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise GateError(f"{self.model}: sidekickd warm failed ({p.returncode}):\n{(p.stdout + p.stderr)[-1500:]}")
        return time.monotonic() - t

    def start(self):
        if self.port is None:       # not written yet for a warm-up
            self.write_config()
        self.log = open(self.run / "sidekickd.log", "wb")
        self.proc = subprocess.Popen([str(self.binary), "--config", str(self.run / "config.toml")], env=self.env(),
                                     stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise GateError(f"sidekickd exited ({self.proc.returncode}) at startup:\n{self.log_tail()}")
            try:
                st, _h, health = Client(self.url).call("GET", "/health")
                if st == 200:
                    self.health = health
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise GateError(f"sidekickd didn't answer /health within 120 s:\n{self.log_tail()}")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def log_tail(self, n=30) -> str:
        try:
            return "\n".join((self.run / "sidekickd.log").read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()       # by PID
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self.log:
            self.log.close()


class Client:
    """JSON over one keep-alive connection. A 504 (the daemon's request
    timeout) or a socket timeout is a Hang."""

    def __init__(self, url: str, timeout: float = 3600):
        host, port = url.removeprefix("http://").split(":")
        self.host, self.port = host, int(port)
        self.timeout = timeout
        self.conn = None
        self.version = None

    def call(self, method: str, route: str, body=None):
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                data = None if body is None else json.dumps(body).encode()
                self.conn.request(method, route, body=data, headers={"content-type": "application/json"})
                r = self.conn.getresponse()
                raw = r.read()
            except (TimeoutError, socket.timeout):
                self.conn.close()
                self.conn = None
                raise Hang(f"{method} {route}: no response within {self.timeout:.0f} s")
            except (http.client.HTTPException, ConnectionError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
                continue
            v = r.getheader("sidekick-version")
            if v:
                if self.version not in (None, v):
                    raise GateError(f"the daemon's version changed during the run: {self.version}, {v}")
                self.version = v
            try:
                payload = json.loads(raw) if raw else None
            except ValueError:
                payload = {"raw": raw.decode("utf-8", "replace")}
            if r.status == 504:
                raise Hang(f"{method} {route}: HTTP 504 (the daemon's request timeout): {json.dumps(payload)[:300]}")
            return r.status, {k.lower(): v for k, v in r.getheaders()}, payload

    def post(self, route: str, body: dict):
        st, h, payload = self.call("POST", route, body)
        if st != 200:
            raise GateError(f"POST {route}: HTTP {st}: {json.dumps(payload)[:400]}")
        return payload

    def model_info(self, model: str, health: dict) -> dict:
        st, _h, models = self.call("GET", "/v1/models")
        for m in (models or {}).get("data", []):
            if m["id"] == model:
                return m
        skipped = [s for s in health.get("skipped_models", []) if model in json.dumps(s)]
        raise GateError(f"{model} isn't served: {json.dumps(skipped)[:400] if skipped else 'not listed'}")


# ---------------------------------------------------------------- task samples


def argmax(xs) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def load_reference(ref: Path) -> tuple[dict, str]:
    raw = (ref / "reference.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check_reference(ref_json: dict, build: Path, model: str):
    want = ref_json.get("tokenizer_sha256")
    have = sha256_file(build / "tokenizer.json") if (build / "tokenizer.json").is_file() else None
    if want and have and want != have:
        raise GateError(f"{model}: the reference was made with another tokenizer.json ({want[:12]}, the build "
                        f"has {have[:12]})")


def run_nli(client: Client, model: str, info: dict, ref: Path) -> dict:
    """The nli parity corpus's MultiNLI pairs, one label each, template "{}"."""
    r, ref_sha = load_reference(ref)
    classes = info.get("nli_classes")
    if not classes:
        raise GateError(f"{model}: /v1/models lists no nli_classes, so nli_probs can't be requested")
    three = "neutral" in classes
    e_i = classes.index("entailment")
    c_i = classes.index("contradiction") if "contradiction" in classes else classes.index("not_entailment")
    cases = [(i, c) for i, c in enumerate(r["cases"]) if "mnli" in c.get("tags", [])]
    if not cases:
        raise GateError(f"{model}: the reference has no MultiNLI cases")
    shape, pairs = read_tensors(ref / "reference.safetensors", ["torch_pairs"])["torch_pairs"]
    if shape[2] != len(classes):
        raise GateError(f"{model}: the reference has {shape[2]} classes, the daemon lists {classes}")
    stride = shape[1] * shape[2]
    sample = hashlib.sha256()
    pred3, pred2, f3, f2, gold3, split, tops = [], [], [], [], [], [], []
    disagree = 0
    for i, c in cases:
        gold = next(t for t in c["tags"] if t.startswith("mnli-")).removeprefix("mnli-")
        hyp = c["candidate_labels"][0]
        sample.update(json.dumps([c["input"], hyp, gold]).encode() + b"\n")
        res = client.post("/v1/classify", {"model": model, "input": c["input"], "candidate_labels": [hyp],
                                           "hypothesis_template": "{}", "nli_probs": True})["data"][0]
        p = res["nli_probs"][0]
        if abs(sum(p.values()) - 1) > 1e-4:
            raise GateError(f"{model}: nli_probs sum to {sum(p.values())}")
        probs = res["probs"][0]
        top3 = max(classes, key=lambda k: p[k])
        ent2 = probs >= 0.5
        if not three and (top3 == "entailment") != ent2:
            disagree += 1
        logits = pairs[i * stride: i * stride + shape[2]]
        pred3.append(top3)
        pred2.append(ent2)
        f3.append(classes[argmax(logits)])
        f2.append(logits[e_i] >= logits[c_i])
        gold3.append(gold)
        split.append("matched" if "validation_matched" in c["tags"] else "mismatched")
        tops.append(p[top3] if three else (probs if ent2 else 1 - probs))
    n = len(cases)
    gold2 = [g == "entailment" for g in gold3]

    def acc(pred, gold, which=None):
        idx = [j for j in range(n) if which is None or split[j] == which]
        return sum(pred[j] == gold[j] for j in idx) / len(idx)

    m = {"n": n,
         "mnli2_accuracy": acc(pred2, gold2), "mnli2_matched": acc(pred2, gold2, "matched"),
         "mnli2_mismatched": acc(pred2, gold2, "mismatched"),
         "agree_fp32_2": sum(a == b for a, b in zip(pred2, f2)) / n,
         "mean_prob": sum(tops) / n}
    if three:
        m.update({"mnli3_accuracy": acc(pred3, gold3), "mnli3_matched": acc(pred3, gold3, "matched"),
                  "mnli3_mismatched": acc(pred3, gold3, "mismatched"),
                  "agree_fp32_3": sum(a == b for a, b in zip(pred3, f3)) / n})
    else:
        m["probs_vs_nli_probs_disagreements"] = disagree
    preds = {"probs": "".join("e" if x else "x" for x in pred2)}
    if three:
        preds["nli_probs"] = "".join(x[0] for x in pred3)
    return {"sample": {"name": "nli parity corpus, MultiNLI pairs", "n": n, "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": m, "predictions": preds}


def scoring_models() -> set[str]:
    return {m["id"] for m in tomllib.loads(SCORING.read_text()).get("nli", [])}


def run_sanity(python: str, url: str, model: str, build: Path) -> dict:
    """tools/tasks/scoring_run.py sanity: the textbook MNLI triple through
    the verification harness's client path and through D47's stock request."""
    p = subprocess.run([python, str(REPO / "tools" / "tasks" / "scoring_run.py"), "sanity", "--model", model,
                        "--model-dir", str(build), "--base-url", url], capture_output=True, text=True,
                       env=dict(os.environ, SIDEKICK_API_KEY=""))
    try:
        out = json.loads(p.stdout)
    except ValueError:
        raise GateError(f"{model}: scoring_run.py sanity failed ({p.returncode}):\n{p.stderr[-1500:]}")
    rows = [{"hypothesis": r["hypothesis"], "expected": r["expected"], "right": r["right"],
             "scores": {k: round(v, 6) for k, v in r["scores"].items()},
             "stock_max_abs_difference": r["max_abs_difference"]} for r in out["rows"]]
    return {"passed": out["passed"], "rows": rows}


def request_fields(c: dict) -> dict:
    """A parity case's request, as a client sends it."""
    body = {}
    for k in ("candidate_labels", "question_type", "instructions", "hypothesis_template"):
        if c.get(k) is not None:
            body[k] = c[k]
    if c.get("multi_label"):
        body["multi_label"] = True
    if c.get("truncation_side") == "left":
        body["truncation_side"] = "left"
    return body


def run_classify(client: Client, model: str, info: dict, ref: Path) -> dict:
    """The parity corpus's labelled cases, batched by request shape."""
    r, ref_sha = load_reference(ref)
    max_labels = r["model"]["max_labels"]
    shape, logits = read_tensors(ref / "reference.safetensors", ["torch"])["torch"]
    # The labelled cases; a corpus with none (text-classification's parity
    # corpus) runs whole, for agreement with fp32 only.
    labelled = [(i, c) for i, c in enumerate(r["cases"]) if c.get("gold")]
    cases = labelled or list(enumerate(r["cases"]))
    fixed = r["model"].get("labels") or []
    groups: dict[str, list] = {}
    for i, c in cases:
        groups.setdefault(json.dumps(request_fields(c), sort_keys=True), []).append((i, c))
    batch = int(info.get("max_batch") or 8)
    sample = hashlib.sha256()
    got: dict[int, tuple[int, float] | None] = {}
    for key, items in groups.items():
        fields = json.loads(key)
        for s in range(0, len(items), batch):
            chunk = items[s:s + batch]
            body = {"model": model, "input": [c["input"] for _, c in chunk], **fields}
            st, _h, resp = client.call("POST", "/v1/classify", body)
            if st == 400 and len(chunk) > 1:
                # One input refused fails the batch: send them one by one so
                # only the refused case counts.
                for i, c in chunk:
                    st1, _h1, resp1 = client.call("POST", "/v1/classify", {**body, "input": [c["input"]]})
                    got[i] = None if st1 == 400 else _top(resp1["data"][0]) if st1 == 200 else _fail(model, st1, resp1)
                continue
            if st == 400:
                got[chunk[0][0]] = None
                continue
            if st != 200:
                _fail(model, st, resp)
            for (i, _c), d in zip(chunk, sorted(resp["data"], key=lambda d: d["index"])):
                got[i] = _top(d)
    n = len(cases)
    right = agree = refused = 0
    tops, preds = [], []
    for i, c in cases:
        labels = c.get("candidate_labels") or fixed
        sample.update(json.dumps([c["id"], c["input"], labels, c.get("question_type"), c.get("instructions"),
                                  c["gold"]]).encode() + b"\n")
        g = got[i]
        if g is None:
            refused += 1
            preds.append("-")
            continue
        idx, top = g
        k = len(labels)
        fp32 = argmax(logits[i * max_labels: i * max_labels + k])
        right += bool(c.get("gold")) and labels[idx] in c["gold"]
        agree += idx == fp32
        tops.append(top)
        preds.append(_symbol(idx))
    answered = n - refused
    return {"sample": {"name": "parity corpus, " + ("labelled cases" if labelled else "every case"), "n": n, "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": {"n": n, "refused": refused,
                        **({"gold_accuracy": right / answered if answered else 0.0} if labelled else {}),
                        "agree_fp32": agree / answered if answered else 0.0,
                        "mean_prob": sum(tops) / len(tops) if tops else 0.0},
            "predictions": {"label": "".join(preds)}}


def _top(d: dict) -> tuple[int, float]:
    probs = d["probs"]
    i = argmax(probs)
    return i, probs[i]


def _fail(model, st, resp):
    raise GateError(f"{model}: /v1/classify HTTP {st}: {json.dumps(resp)[:400]}")


def _symbol(i: int) -> str:
    """A label index as one character (0-9, then a-z, A-Z)."""
    return "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] if i < 62 else "?"


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def run_rerank(client: Client, model: str, info: dict, ref: Path | None) -> dict:
    """The rerank corpus: each group as one request."""
    corpus = tomllib.loads(RERANK_CORPUS.read_text())["group"]
    fp32 = {}
    ref_sha = None
    if ref is not None:
        r, ref_sha = load_reference(ref)
        shape, t = read_tensors(ref / "reference.safetensors", ["torch"])["torch"]
        for i, c in enumerate(r["cases"]):
            fp32[c["id"]] = t[i * shape[1]]
    sample = hashlib.sha256()
    tops, relevant_hits, relevant_groups, agree, worst, worst_p = [], 0, 0, 0, 0.0, 0.0
    scored = 0
    for g in corpus:
        docs = [" ".join([d] * g.get("repeat", 1)) for d in g["documents"]]
        sample.update(json.dumps([g["id"], g["query"], docs, g.get("relevant")]).encode() + b"\n")
        resp = client.post("/v1/rerank", {"model": model, "query": g["query"], "documents": docs,
                                          "truncate_prompt_tokens": -1, "return_documents": False})
        scores = [0.0] * len(docs)
        for x in resp["results"]:
            scores[x["index"]] = x["relevance_score"]
        top = argmax(scores)
        tops.append(top)
        if g.get("relevant"):
            relevant_groups += 1
            relevant_hits += top in g["relevant"]
        ref_scores = [fp32.get(f"{g['id']}-{j}") for j in range(len(docs))]
        if fp32 and None not in ref_scores:
            scored += 1
            agree += top == argmax(ref_scores)
            worst = max(worst, max(abs(a - b) for a, b in zip(scores, ref_scores)))
            worst_p = max(worst_p, max(abs(sigmoid(a) - sigmoid(b)) for a, b in zip(scores, ref_scores)))
    m = {"n": len(corpus), "top1_relevant": relevant_hits / relevant_groups if relevant_groups else 0.0}
    if fp32:
        if scored != len(corpus):
            raise GateError(f"{model}: the reference doesn't cover every group of the rerank corpus")
        # The score on its own scale (a logit for the ms-marco cross-encoders),
        # and as its sigmoid, the scale the parity suite grades |dp| on.
        m.update({"agree_fp32_top1": agree / scored, "max_abs_diff_fp32": worst, "max_abs_dp_fp32": worst_p})
    return {"sample": {"name": "rerank corpus", "n": len(corpus), "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": m, "predictions": {"top": "".join(_symbol(t) for t in tops)}}


def run_embed_parity(client: Client, model: str, info: dict, ref: Path) -> dict:
    """The parity corpus's texts, each with the prompt its reference used
    put before it (input_type "none"), against the reference's fp32 vectors."""
    r, ref_sha = load_reference(ref)
    prefixes = r["model"].get("prefixes") or {}
    shape, t = read_tensors(ref / "reference.safetensors", ["torch"])["torch"]
    dims = shape[1]
    cases = r["cases"]
    texts = [prefixes.get(c.get("purpose"), "") + c["text"] for c in cases]
    cap = info["max_seq_len"]
    sample = hashlib.sha256()
    vecs = []
    for s in range(0, len(texts), 16):
        resp = client.post("/v1/embeddings", {"model": model, "input": texts[s:s + 16], "input_type": "none",
                                              "truncate": "END", "encoding_format": "base64"})
        for d in sorted(resp["data"], key=lambda d: d["index"]):
            a = array("f")
            a.frombytes(base64.b64decode(d["embedding"]))
            vecs.append(a)
    cos, mismatched = [], 0
    for i, (c, text, v) in enumerate(zip(cases, texts, vecs)):
        sample.update(json.dumps([c["id"], text]).encode() + b"\n")
        count = client.post("/tokenize", {"model": model, "prompt": text, "input_type": "none"})["count"]
        if min(count, cap) != len(c["ids"]):
            mismatched += 1
        w = t[i * dims:(i + 1) * dims]
        if len(v) != dims:
            raise GateError(f"{model}: {len(v)} dimensions, the reference has {dims}")
        dot = sum(a * b for a, b in zip(v, w))
        cos.append(dot / (math.sqrt(sum(a * a for a in v)) * math.sqrt(sum(b * b for b in w))))
    # As the parity suite grades: the worst over the corpus's ordinary
    # cases, and its stress inputs (tagged "degenerate") apart.
    stress = [STRESS_TAG in c.get("tags", []) for c in cases]
    core = [x for x, st in zip(cos, stress) if not st]
    hard = [x for x, st in zip(cos, stress) if st]
    m = {"n": len(cases), "worst_cosine_fp32": min(core), "mean_cosine_fp32": sum(cos) / len(cos),
         "cosine_error_fp32": 1 - min(core), "token_count_mismatches": mismatched}
    if hard:
        m.update({"stress_cosine_fp32": min(hard), "stress_error_fp32": 1 - min(hard)})
    return {"sample": {"name": "parity corpus", "n": len(cases), "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": m}



def run_mteb(python: str, url: str, model: str, path: str, build: Path, work: Path) -> dict:
    """tools/mteb_transfer.py run on the MTEB slice. Not under HF_HUB_OFFLINE:
    with it, mteb 2.26.14 loads AskUbuntuDupQuestions without its candidate
    lists and scores bge-small-en-v1.5 0.384 instead of 0.618, silently."""
    out = work / "mteb"
    shutil.rmtree(out, ignore_errors=True)
    p = subprocess.run([python, str(REPO / "tools" / "mteb_transfer.py"), "run", "--url", url, "--model", model,
                        "--path", path, "--out", str(out), "--build", str(build), "--tasks", *MTEB_TASKS],
                       capture_output=True, text=True)
    if p.returncode != 0:
        err = f"{model}: mteb_transfer.py run failed ({p.returncode}):\n{p.stderr[-2000:]}"
        raise (Hang if "HTTP 504" in p.stderr or "timed out" in p.stderr else GateError)(err)
    published = json.loads((DATA / "mteb_transfer.json").read_text())
    ref = {(s["model"], s["task"]): s["score"] for s in published["scores"] if s["path"] == "reference"}
    tasks, m = {}, {}
    for task in MTEB_TASKS:
        rec = json.loads((out / f"{model}__{path}__{task}.json").read_text())
        tasks[task] = {"score": rec["score"], "inputs": rec["inputs"], "over_cap": rec["over_cap"],
                       "seconds": rec["seconds"]}
        m[f"mteb:{task}"] = rec["score"]
        if (model, task) in ref:
            m[f"mteb:{task}:minus_fp32"] = rec["score"] - ref[(model, task)]
    return {"metrics": m, "mteb": {"version": rec["mteb_version"], "benchmark": rec["benchmark"], "tasks": tasks}}


# ---------------------------------------------------------------- one model


def chat_requests() -> list[dict]:
    """The chat sample: the first CHAT_ITEMS items of each family, in the
    task set's order (tools/chat_bench/tasks.py, standard library only)."""
    from chat_bench import tasks
    reqs = tasks.requests()
    out = []
    for fam in CHAT_FAMILIES:
        out += [r for r in reqs if r["family"] == fam][:CHAT_ITEMS]
    return out


def chat_reply(client: Client, model: str, r: dict, guided: bool) -> dict:
    body = {"model": model, "messages": r["messages"], "temperature": 0, "max_tokens": r["max_tokens"]}
    if guided:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": r["family"], "schema": r["schema"]}}
    t = time.monotonic()
    st, _h, payload = client.call("POST", "/v1/chat/completions", body)
    total = time.monotonic() - t
    rec = {"family": r["family"], "item": r["item"], "messages": r["messages"],
           "schema": r["schema"] if guided else None, "status": st, "total_s": total, "ttft_s": total,
           "ttft_content_s": total, "error": None, "content": "", "reasoning": "", "usage": None,
           "finish_reason": None}
    if st != 200:
        rec["error"] = (payload or {}).get("error") if isinstance(payload, dict) else {"message": str(payload)}
        return rec
    choice = payload["choices"][0]
    rec.update(content=choice["message"].get("content") or "", finish_reason=choice.get("finish_reason"),
               usage=payload.get("usage"))
    return rec


def run_chat(args, client: Client, model: str, build: Build, run_dir: Path) -> dict:
    """The chat sample twice, prompt-only and guided; the replies must be
    byte-identical between the runs, and the first run is scored."""
    reqs = chat_requests()
    modes = {"prompt": [(r, False) for r in reqs], "guided": [(r, True) for r in reqs if r["schema"] is not None]}
    runs = []
    for attempt in range(2):
        runs.append({m: {"records": [chat_reply(client, model, r, g) for r, g in todo]} for m, todo in modes.items()})
        print(f"[{model}] chat sample, run {attempt + 1} of 2: "
              f"{sum(len(v['records']) for v in runs[-1].values())} requests", flush=True)
    first, second = runs
    differ = sum(a["content"] != b["content"] or a["usage"] != b["usage"]
                 for m in modes for a, b in zip(first[m]["records"], second[m]["records"]))
    path = run_dir / "chat-run.json"
    path.write_text(json.dumps({"modes": first}, ensure_ascii=False))
    try:
        p = subprocess.run([args.chat_python, str(REPO / "tools" / "chat_bench" / "gate_score.py"), str(path),
                            str(build.dir)], capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        raise GateError(f"{model}: scoring the chat sample took over 30 min")
    if p.returncode != 0:
        raise GateError(f"{model}: tools/chat_bench/gate_score.py failed:\n{p.stderr[-1500:]}")
    scored = json.loads(p.stdout)
    recs = [r for m in modes for r in first[m]["records"]]
    metrics = {"n": len(recs), "refused": sum(r["status"] != 200 for r in recs), "nondeterministic_replies": differ,
               "schema_invalid": scored["schema_invalid"],
               "prompt_token_mismatches": scored["prompt_token_mismatches"],
               "completion_token_mismatches": scored["completion_token_mismatches"],
               # every guided request's prompt_tokens, exactly: the daemon's schema instruction is in them
               "guided_prompt_tokens": hashlib.sha256(json.dumps(scored["guided_prompt_tokens"]).encode())
               .hexdigest()[:16]}
    metrics.update({f"chat:{k}": v for k, v in scored["scores"].items()})
    sample = hashlib.sha256(json.dumps([[m, r["family"], r["item"], r["messages"], r["max_tokens"]]
                                        for m, todo in modes.items() for r, _g in todo]).encode()).hexdigest()
    predictions = {m: "".join(_symbol(int(hashlib.sha256(r["content"].encode()).hexdigest(), 16) % 62)
                              for r in first[m]["records"]) for m in modes}
    return {"metrics": metrics, "predictions": predictions,
            "sample": {"source": f"fixtures/tasks/chat (first {CHAT_ITEMS} per family)", "sha256": sample}}


def kind_of(meta: dict, build: Build) -> str:
    if meta["task"] in ("embed", "rerank", "chat"):
        return meta["task"]
    fmt = tomllib.loads((build.dir / build.manifest_name).read_text()).get("classify", {}).get("format")
    return "nli" if fmt == "nli" else "classify"


def run_one(args, binary: Path, home: Home, t: "Target") -> dict:
    t0 = time.monotonic()
    model, path, build, ref = t.model, t.path, t.build, t.ref
    kind = kind_of(t.meta, build)
    if kind in ("nli", "classify", "embed") and ref is None:
        raise GateError(f"{model}: no parity reference in --refs")
    units = None if t.default else PATH_UNITS.get(path)
    if kind == "chat":
        return run_one_chat(args, binary, home, t)
    rec = {"model": model, "path": path, "variant": t.variant, "kind": kind, "default": t.default,
           "build": build.record()}
    if ref is not None:
        check_reference(load_reference(ref)[0], build.dir, model)
    models_dir, note = home.serve_dir(build)
    if note:
        rec["build"]["served_manifest"] = note
    free0, w0 = free_gib(home.root), time.time()
    cache = {"free_gib_before": round(free0, 2)}
    ane = path == "coreml-ane"
    d = Daemon(binary, home, models_dir, model, units, ane, args.onnx_runtime)
    try:
        if path.startswith("coreml-"):
            d.write_config()
            print(f"[{t.label}] warming (sidekickd warm)", flush=True)
            cache["warm_seconds"] = round(d.warm(), 1)
        with d:
            client = Client(d.url, timeout=ANE_REQUEST_TIMEOUT_SECS + 60 if ane else 3600)
            info = client.model_info(model, d.health)
            if path.startswith("coreml-"):
                if info.get("backend", "coreml") != "coreml" or info.get("compute_units") != PATH_UNITS[path]:
                    raise GateError(f"{model}: served as {info.get('backend')}/{info.get('compute_units')}, not "
                                    f"{path}" + ("" if units else
                                                 " (the build's manifest doesn't serve its published default)"))
            elif path == "onnx-cpu" and info.get("backend") != "onnx":
                raise GateError(f"{model}: served by {info.get('backend')}, not onnx")
            print(f"[{t.label}] serving on {d.url}; running the {kind} sample", flush=True)
            if kind == "nli":
                rec.update(run_nli(client, model, info, ref))
                if model in scoring_models():
                    rec["sanity"] = run_sanity(args.python, d.url, model, build.dir)
            elif kind == "classify":
                rec.update(run_classify(client, model, info, ref))
            elif kind == "rerank":
                rec.update(run_rerank(client, model, info, ref))
            else:
                part = run_embed_parity(client, model, info, ref)
                mt = run_mteb(args.python, d.url, model, path, build.dir, d.run)
                rec.update(part)
                rec["metrics"].update(mt["metrics"])
                rec["mteb"] = mt["mteb"]
            rec["sidekick"] = client.version
    finally:
        if not args.keep_gpu_cache:
            home.prune_gpu_cpu()
        cache["free_gib_after"] = round(free_gib(home.root), 2)
        if path.startswith("coreml-"):
            cache["ane_compiles"] = ane_compiles(w0, time.time())
        rec["cache"] = cache
        t.cache = cache
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


def df_free_kib() -> dict:
    """Free KiB of the root and the Data volume, from `df -k` (APFS's
    purgeable space and other processes move these: measure with nothing
    else heavy running)."""
    out = {}
    for vol in ("/", "/System/Volumes/Data"):
        p = subprocess.run(["df", "-k", vol], capture_output=True, text=True)
        lines = p.stdout.strip().splitlines()
        if p.returncode == 0 and len(lines) >= 2:
            out[vol] = int(lines[1].split()[3])
    return out


def run_one_chat(args, binary: Path, home: Home, t: "Target") -> dict:
    """A chat model on its default path, the Neural Engine: one request to
    load it (and the disk its compile leaves, when it compiles), then the
    chat sample."""
    t0 = time.monotonic()
    model, build = t.model, t.build
    if not t.default or t.path != "coreml-ane":
        raise GateError(f"{model}: a chat model is gated on its default path, the Neural Engine, only")
    rec = {"model": model, "path": t.path, "variant": t.variant, "kind": "chat", "default": True,
           "build": build.record()}
    models_dir, _note = home.serve_dir(build)
    d = Daemon(binary, home, models_dir, model, None, True, None)
    cache = {}
    weights = sum(f.stat().st_size for f in build.dir.glob("decoder.*.mlmodelc/weights/weight.bin"))
    cache["weights_bytes"] = weights
    try:
        with d:
            client = Client(d.url, timeout=WARM_TIMEOUT_SECS)
            info = client.model_info(model, d.health)
            if info.get("backend") != "coreml" or info.get("compute_units") != PATH_UNITS["coreml-ane"]:
                raise GateError(f"{model}: served as {info.get('backend')}/{info.get('compute_units')}, not coreml-ane")
            print(f"[{t.label}] serving on {d.url}; loading the build", flush=True)
            df0, w0 = df_free_kib(), time.time()
            cache["free_gib_before"] = round(free_gib(home.root), 2)
            st, _h, payload = client.call("POST", "/v1/chat/completions", {
                "model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "temperature": 0})
            if st != 200:
                raise GateError(f"{model}: the loading request failed: HTTP {st}: {json.dumps(payload)[:300]}")
            w1 = time.time()
            df1 = df_free_kib()
            compiles = ane_compiles(w0, w1)
            cache.update(load_seconds=round(w1 - w0, 1), load_compiles=compiles,
                         load_df_free_kib={"before": df0, "after": df1})
            if compiles:
                cache["load_disk_bytes"] = {vol: (df0[vol] - df1[vol]) * 1024 for vol in df0 if vol in df1}
            client.timeout = ANE_REQUEST_TIMEOUT_SECS + 60
            rec.update(run_chat(args, client, model, build, d.run))
            rec["sidekick"] = client.version
    finally:
        cache["free_gib_after"] = round(free_gib(home.root), 2)
        if "load_seconds" in cache:
            cache["ane_compiles"] = ane_compiles(w0, time.time())
        rec["cache"] = cache
        t.cache = cache
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


# ---------------------------------------------------------------- comparison


def metric_rule(name: str) -> str:
    if name in ("n", "refused", "token_count_mismatches", "probs_vs_nli_probs_disagreements",
                "nondeterministic_replies", "schema_invalid", "prompt_token_mismatches",
                "completion_token_mismatches", "guided_prompt_tokens"):
        return "exact"
    if name.startswith("chat:"):
        return "chat_score"
    if name.startswith("mteb:"):
        return "mteb"
    if name in ("cosine_error_fp32", "stress_error_fp32", "max_abs_diff_fp32", "max_abs_dp_fp32"):
        return "error"
    if name in ("worst_cosine_fp32", "mean_cosine_fp32", "stress_cosine_fp32"):
        return "derived"        # judged through cosine_error_fp32 and stress_error_fp32
    if name == "mean_prob":
        return "mean_prob"
    if name.startswith("agree"):
        return "agreement"
    return "accuracy"


def same_build(rec: dict, base: dict) -> bool:
    """By the ledger digest where the baseline has one; else by the first
    baseline's fingerprint (the same tree hash)."""
    b, r = base["build"], rec["build"]
    if "digest" in b:
        return r["digest"] == b["digest"]
    return r["fingerprint"] == b["fingerprint"]


def compare(rec: dict, base: dict, tol: dict) -> list[str]:
    """What drifted beyond tolerance: one line each."""
    out = []
    if not same_build(rec, base):
        out.append(f"build differs from the baseline's ({rec['build'].get('ledger') or rec['build']['digest'][:12]} "
                   f"vs {base['build'].get('ledger') or base['build'].get('digest', base['build']['fingerprint'])[:12]}); "
                   f"a deliberate change re-baselines")
    if rec.get("sample", {}).get("sha256") != base.get("sample", {}).get("sha256"):
        out.append("the task sample differs from the baseline's (another reference or corpus)")
    for name, b in base.get("metrics", {}).items():
        v = rec.get("metrics", {}).get(name)
        if v is None:
            out.append(f"{name}: missing (baseline {b})")
            continue
        rule = metric_rule(name)
        if rule == "exact":
            if v != b:
                out.append(f"{name}: {v} (baseline {b}, must not change)")
        elif rule == "error":
            limit = max(tol["error"]["factor"] * b, b + tol["error"]["floor"])
            if v > limit:
                out.append(f"{name}: {v:.3g} > {limit:.3g} (baseline {b:.3g})")
        elif rule in tol and "abs" in tol[rule]:
            if abs(v - b) > tol[rule]["abs"] + 1e-12:
                out.append(f"{name}: {v:.6g} (baseline {b:.6g}, tolerance ±{tol[rule]['abs']})")
    for name, b in base.get("predictions", {}).items():
        v = rec.get("predictions", {}).get(name, "")
        if len(v) != len(b):
            out.append(f"predictions[{name}]: {len(v)} items, baseline {len(b)}")
            continue
        flips = sum(x != y for x, y in zip(v, b))
        if flips > tol["flips"]["max_share"] * len(b):
            out.append(f"predictions[{name}]: {flips} of {len(b)} differ from the baseline run's "
                       f"(tolerance {tol['flips']['max_share']:.0%})")
    if "sanity" in base:
        s = rec.get("sanity")
        if s is None:
            out.append("sanity: not run")
        else:
            if s["passed"] != base["sanity"]["passed"]:
                out.append(f"sanity: passed={s['passed']} (baseline {base['sanity']['passed']})")
            for r, br in zip(s["rows"], base["sanity"]["rows"]):
                if r["right"] != br["right"]:
                    out.append(f"sanity {r['hypothesis']!r}: right={r['right']} (baseline {br['right']})")
                for k, bv in br["scores"].items():
                    if abs(r["scores"].get(k, math.inf) - bv) > tol["sanity"]["abs"]:
                        out.append(f"sanity {r['hypothesis']!r} {k}: {r['scores'].get(k)} (baseline {bv})")
    return out


def flips(rec: dict, base: dict | None) -> int | None:
    if not base:
        return None
    total = 0
    for name, b in base.get("predictions", {}).items():
        v = rec.get("predictions", {}).get(name, "")
        if len(v) == len(b):
            total += sum(x != y for x, y in zip(v, b))
    return total


def headline(rec: dict) -> str:
    m = rec.get("metrics", {})
    k = rec["kind"]
    if k == "nli":
        acc = m.get("mnli3_accuracy", m.get("mnli2_accuracy"))
        agree = m.get("agree_fp32_3", m.get("agree_fp32_2"))
        ways = "3-way" if "mnli3_accuracy" in m else "2-way"
        s = f"MNLI {ways} {acc:.3f}, fp32 agreement {agree:.3f}"
        if "sanity" in rec:
            s += f", triple {'ok' if rec['sanity']['passed'] else 'FAILED'}"
        return s
    if k == "classify":
        gold = f"gold {m['gold_accuracy']:.3f}, " if "gold_accuracy" in m else "no gold labels, "
        return f"{gold}fp32 agreement {m['agree_fp32']:.4f}, refused {m['refused']}"
    if k == "rerank":
        s = f"relevant top-1 {m['top1_relevant']:.3f}"
        if "agree_fp32_top1" in m:
            s += f", fp32 top-1 agreement {m['agree_fp32_top1']:.3f}, max |Δp| {m['max_abs_dp_fp32']:.3g}"
        return s
    if k == "chat":
        return (f"{m['n']} requests, {m['nondeterministic_replies']} not repeatable, {m['schema_invalid']} guided "
                f"invalid, usage mismatches {m['prompt_token_mismatches']} prompt / "
                f"{m['completion_token_mismatches']} completion, refused {m['refused']}")
    parts = [f"{t} {m[f'mteb:{t}']:.4f}" for t in MTEB_TASKS if f"mteb:{t}" in m]
    return ", ".join(parts) + f"; worst cos fp32 {m['worst_cosine_fp32']:.6f}"


# ---------------------------------------------------------------- main


def provenance(this_chip: str, versions: set, signing: str | None) -> dict:
    os_version, os_build = macos()
    # Uncommitted changes to tracked files, the baseline this gate writes excepted.
    status = git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 f":(exclude){BASELINE.relative_to(REPO).as_posix()}")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status),
            "sidekick": sorted(v for v in versions if v)[0] if any(versions) else None,
            "signing_identifier": signing,
            "chip": this_chip, "os": f"macOS {os_version}", "os_build": os_build,
            "date": datetime.date.today().isoformat()}


def signing_identifier(binary: Path) -> str | None:
    p = subprocess.run(["codesign", "--display", "--verbose=1", str(binary)], capture_output=True, text=True)
    return next((line.split("=", 1)[1] for line in (p.stdout + p.stderr).splitlines()
                 if line.startswith("Identifier=")), None)


def build_daemon(args) -> Path:
    """The binary to gate, signed with the stable identifier Core ML's ANE
    cache is keyed by (tools/sign_sidekickd.sh): a built one is signed
    here; a given one is checked and used as it is."""
    if args.daemon:
        binary = Path(args.daemon).resolve()
    else:
        print("building sidekickd (release, coreml + onnx)", flush=True)
        subprocess.run(["cargo", "build", "--release", "-p", "sidekick-server", "--features", "coreml,onnx",
                        "--bin", "sidekickd"], cwd=REPO, check=True)
        meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=REPO,
                                         capture_output=True, text=True, check=True).stdout)
        binary = Path(meta["target_directory"]) / "release" / "sidekickd"
        subprocess.run([str(REPO / "tools" / "sign_sidekickd.sh"), str(binary)], check=True)
    ident = signing_identifier(binary)
    if ident != SIGNING_IDENTIFIER:
        print(f"note: {binary} is signed as {ident!r}, not {SIGNING_IDENTIFIER}: its ANE programs won't hit "
              f"the cache other builds left (tools/sign_sidekickd.sh)", flush=True)
    return binary


def check_python(python: str, todo: list):
    """Fail before any model runs if `--python` lacks what this run needs:
    mteb (at the pinned version) for an embedder, `tokenizers` for an nli
    model the verification harness scores."""
    kinds = {t.meta["task"] for t in todo if t.build is not None} - {"chat"}
    sanity = any(t.build is not None and t.model in scoring_models() for t in todo)
    probe = []
    if "embed" in kinds:
        probe.append("import mteb, sentence_transformers; assert mteb.__version__ == '2.26.14', mteb.__version__")
    if sanity:
        probe.append("import tokenizers")
    if not probe:
        return
    p = subprocess.run([python, "-c", "; ".join(probe)], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"--python {python} can't run the MTEB slice or the MNLI triple "
                         f"({(p.stderr.strip().splitlines() or ['not runnable'])[-1]}). Build an environment from "
                         f"tools/release_gate.requirements.txt and pass its interpreter with --python or "
                         f"SIDEKICK_GATE_PYTHON.")


class Target:
    """One (model, path, variant) to run, with its build and reference."""

    def __init__(self, model, meta, path, variant, default):
        self.model, self.meta, self.path, self.variant, self.default = model, meta, path, variant, default
        self.build: Build | None = None
        self.ref: Path | None = None
        self.cache = None

    @property
    def key(self):
        return (self.model, self.path, self.variant)

    @property
    def label(self):
        v = "" if self.variant == DEFAULT_VARIANT["onnx" if self.path == "onnx-cpu" else "coreml"] else f" {self.variant}"
        return f"{self.model} @ {self.path}{v}"


def entry_key(e: dict) -> tuple:
    return (e["model"], e["path"], e.get("variant") or DEFAULT_VARIANT["onnx" if e["path"] == "onnx-cpu" else "coreml"])


def _chat_check_python(python: str, todo: list):
    """Fail before any model runs if `--chat-python` can't score the chat sample."""
    if not any(t.build is not None and t.meta["task"] == "chat" for t in todo):
        return
    p = subprocess.run([python, "-c", "import rouge_score, jsonschema, numpy, transformers"], capture_output=True,
                       text=True)
    if p.returncode != 0:
        raise SystemExit(f"--chat-python {python} can't score the chat sample "
                         f"({(p.stderr.strip().splitlines() or ['not runnable'])[-1]}): use the chat benchmark's "
                         f"environment (tools/chat_bench/requirements.txt)")


def plan(args, this_chip: str, by_key: dict, home: Home) -> list[Target]:
    models = models_toml()
    builds = find_builds([Path(d) for d in args.models_dir])
    refs = [Path(d) for d in args.refs]
    ledger = ledger_tool.load()
    wanted = args.models or list(models)
    unknown = [m for m in wanted if m not in models]
    if unknown:
        raise SystemExit(f"not published (docs/data/models.toml): {', '.join(unknown)}")
    named = {p for p in args.paths if p in PATH_ORDER}
    bad = [p for p in args.paths if p not in PATH_ORDER and p not in ("default", "all")]
    if bad:
        raise SystemExit(f"--paths: unknown {', '.join(bad)}")
    out = []
    for model in wanted:
        meta = models[model]
        default = (meta["served"], "fp16")
        targets = [default] if ("default" in args.paths or "all" in args.paths) else []
        for path, variant in graded_targets(model, this_chip):
            if (path, variant) != default and ("all" in args.paths or path in named):
                targets.append((path, variant))
        for path, variant in targets:
            if meta["task"] == "chat":
                variant = CHAT_VARIANT
            t = Target(model, meta, path, variant, path == meta["served"] and variant in ("fp16", CHAT_VARIANT))
            backend = "onnx" if path == "onnx-cpu" else "coreml"
            want = by_key.get(t.key, {}).get("build", {}).get("ledger")
            cands = builds.get((model, backend), [])
            for b in cands:      # the baseline's build, if it is here
                if want:
                    b.identify(home.hashes, ledger)
                    if b.ledger and b.ledger["id"] == want:
                        t.build = b
                        break
            if t.build is None:  # else the first of the variant
                for b in cands:
                    b.identify(home.hashes, ledger)
                    if b.variant == variant:
                        t.build = b
                        break
            if t.build is not None:
                t.ref = find_reference(refs, model, t.build)
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", action="append", default=[], required=True,
                    help="a directory of builds (repeatable, searched in order)")
    ap.add_argument("--refs", action="append", default=[], help="a directory of parity references (repeatable)")
    ap.add_argument("--paths", nargs="+", default=["default"],
                    help="default, all, or path names: the graded (path, precision) pairs to run")
    ap.add_argument("--models", nargs="+", help="only these published models")
    ap.add_argument("--daemon", help="a built sidekickd to gate instead of building one")
    ap.add_argument("--onnx-runtime", help="libonnxruntime for onnx-cpu builds (else ORT_DYLIB_PATH or beside "
                                            "the binary)")
    ap.add_argument("--python", default=os.environ.get("SIDEKICK_GATE_PYTHON") or sys.executable,
                    help="the interpreter for tools/mteb_transfer.py and scoring_run.py sanity, built from "
                         "tools/release_gate.requirements.txt (default: $SIDEKICK_GATE_PYTHON, else this one)")
    ap.add_argument("--chat-python", default=os.environ.get("SIDEKICK_CHAT_BENCH_PYTHON") or sys.executable,
                    help="the interpreter for tools/chat_bench/gate_score.py, the chat benchmark's environment "
                         "(tools/chat_bench/requirements.txt; default: $SIDEKICK_CHAT_BENCH_PYTHON, else this one)")
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME,
                    help=f"the gate's persistent home (default {DEFAULT_HOME})")
    ap.add_argument("--fresh-home", action="store_true", help="a temporary home instead, deleted afterwards")
    ap.add_argument("--keep-gpu-cache", action="store_true",
                    help="keep Core ML's GPU and CPU entries in the home after each model")
    ap.add_argument("--min-free-gib", type=float, default=25.0,
                    help="stop before a model when free disk is below this (default 25)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--json", type=Path, help="write this run's full record here")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    this_chip = chip()
    base = json.loads(args.baseline.read_text()) if args.baseline.exists() else {"entries": []}
    tol = {**TOLERANCES, **base.get("tolerances", {})}
    by_key = {entry_key(e): e for e in base["entries"]}
    home = Home(None if args.fresh_home else args.home.expanduser().resolve())
    try:
        return gate(args, this_chip, base, tol, by_key, home)
    finally:
        home.close()


def gate(args, this_chip, base, tol, by_key, home) -> int:
    todo = plan(args, this_chip, by_key, home)
    if args.list:
        for t in todo:
            b = t.build
            print(f"{t.label:48} " + (f"{(b.ledger or {}).get('id') or 'not in the ledger':58} {b.dir}"
                                      if b else "not available here")
                  + (f"\n{'':48} reference: {t.ref}" if b else ""))
        return 0
    check_python(args.python, todo)
    _chat_check_python(args.chat_python, todo)
    binary = build_daemon(args)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    print(version, flush=True)
    print(f"home: {home.root}" + (" (temporary)" if home.temporary else ""), flush=True)
    t_all = time.monotonic()
    results, errors, unavailable, not_run = [], {}, [], []
    ane_stopped = None
    stop = None
    for t in todo:
        if t.build is None:
            unavailable.append(t.label)
            print(f"[{t.label}] not available here", flush=True)
            continue
        if stop:
            not_run.append(t.label)
            continue
        free = free_gib(home.root)
        if free < args.min_free_gib:
            stop = f"free disk {free:.1f} GiB, under --min-free-gib {args.min_free_gib:g}"
            print(f"[{t.label}] STOP: {stop}; nothing further runs", flush=True)
            not_run.append(t.label)
            continue
        if ane_stopped and t.path == "coreml-ane":
            errors[t.key] = f"not run: ANE work stopped after a hang ({ane_stopped})"
            print(f"[{t.label}] {errors[t.key]}", flush=True)
            continue
        try:
            rec = run_one(args, binary, home, t)
        except GateError as e:
            errors[t.key] = str(e)
            print(f"[{t.label}] ERROR: {e}", flush=True)
            if isinstance(e, Hang) and t.path == "coreml-ane":
                ane_stopped = t.label
                print(f"[{t.label}] an ANE request hung: no further ANE work in this run", flush=True)
            continue
        b = by_key.get(t.key)
        rec["drift"] = compare(rec, b, tol) if b else None
        results.append(rec)
        f = flips(rec, b)
        state = ("no baseline" if b is None else "ok" if not rec["drift"] else "DRIFT")
        c = rec.get("cache", {})
        print(f"[{t.label}] {state} in {rec['seconds']} s: {headline(rec)}"
              + (f"; {f} prediction(s) differ from the baseline run" if f else "")
              + (f" [ANE compiles {c['ane_compiles']}, free {c['free_gib_before']} -> {c['free_gib_after']} GiB]"
                 if "ane_compiles" in c else ""), flush=True)
        for line in rec["drift"] or []:
            print(f"    {line}", flush=True)
    total = round(time.monotonic() - t_all, 1)

    # Baselined targets this run couldn't serve: the gate can't vouch for them.
    ran = {(r["model"], r["path"], r["variant"]) for r in results} | set(errors)
    missing = [k for k in by_key if k not in ran and (not args.models or k[0] in args.models)
               and any(t.key == k for t in todo)]
    drifted = [r for r in results if r["drift"]]
    unbaselined = [r for r in results if r["drift"] is None]
    prov = provenance(this_chip, {r.get("sidekick") for r in results}, signing_identifier(binary))
    prov["daemon"] = version

    print()
    print(f"{'model @ path':52} {'seconds':>8}  result")
    for r in results:
        label = f"{r['model']} @ {r['path']}" + (f" {r['variant']}" if r["variant"] not in DEFAULT_VARIANT.values()
                                                 else "")
        print(f"{label:52} {r['seconds']:8.1f}  {headline(r)}")
    for (m, p, v), e in errors.items():
        print(f"{m + ' @ ' + p + ' ' + v:52} {'':>8}  ERROR: {e.splitlines()[0]}")
    if unavailable:
        print(f"not available here ({len(unavailable)}): " + ", ".join(unavailable))
    if not_run:
        print(f"not run ({stop}): " + ", ".join(not_run))

    if args.json:
        args.json.write_text(json.dumps({"provenance": prov, "seconds": total, "results": results,
                                         "errors": {" ".join(k): e for k, e in errors.items()},
                                         "not_available": unavailable, "not_run": not_run}, indent=1) + "\n")

    if args.update_baseline:
        if errors or not_run:
            print("refusing to update the baseline from a run with errors or models not run", file=sys.stderr)
            return 1
        write_baseline(args.baseline, base, results, prov)
        print(f"baseline updated: {len(results)} entries written to {args.baseline}")

    ok = not errors and not drifted and not missing and not not_run
    sha = (prov["commit"] or "unknown")[:7] + ("+dirty" if prov["dirty"] else "")
    print(f"release gate: {'PASS' if ok else 'FAIL'} — {len(results) - len(drifted) - len(unbaselined)} within "
          f"tolerance, {len(drifted)} drifted, {len(errors)} errors, {len(missing)} baselined but missing, "
          f"{len(not_run)} not run, {len(unbaselined)} without a baseline, {len(unavailable)} not available here — "
          f"{prov['sidekick'] or version} ({sha}), {prov['chip']}, {prov['os']}, {total / 60:.1f} min")
    return 0 if ok else 1


def write_baseline(path: Path, base: dict, results: list[dict], prov: dict):
    keep = {entry_key(e): e for e in base.get("entries", [])}
    for r in results:
        e = {k: r[k] for k in ("model", "path", "variant", "kind", "default", "build", "sample", "metrics",
                               "predictions", "sanity", "mteb", "seconds") if k in r}
        e["provenance"] = {**{k: prov[k] for k in ("commit", "dirty", "chip", "os", "os_build", "date")},
                           "sidekick": r.get("sidekick")}
        keep[entry_key(e)] = e
    order = list(models_toml())
    entries = sorted(keep.values(), key=lambda e: (order.index(e["model"]) if e["model"] in order else 999,
                                                   PATH_ORDER.index(e["path"]),
                                                   VARIANT_ORDER.index(entry_key(e)[2])
                                                   if entry_key(e)[2] in VARIANT_ORDER else 99))
    gated = {e["model"] for e in entries}
    out = {
        "about": ("The release gate's baseline: each published model's task sample, run through sidekickd over "
                  "HTTP, on the paths and build variants listed. Written by tools/release_gate.py "
                  "--update-baseline; do not edit by hand. A change to it is deliberate, and its PR says why."),
        "tolerances": {**TOLERANCES, **base.get("tolerances", {})},
        "mteb_tasks": MTEB_TASKS,
        "updated": {k: prov[k] for k in ("commit", "dirty", "sidekick", "chip", "os", "os_build", "date")},
        "entries": entries,
        "not_gated": sorted((m for m in models_toml() if m not in gated), key=order.index),
    }
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
