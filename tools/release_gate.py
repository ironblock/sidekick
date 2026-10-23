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
   coreml,onnx`), or takes `--daemon <binary>`, such as the release
   archive's.
2. For each published model (docs/data/models.toml) with a build on this
   machine, one at a time, on its default path (models.toml's `served`;
   `--paths all` adds every other path docs/data/models.json grades on this
   chip): starts sidekickd on 127.0.0.1, on a free port outside 8790-8799,
   with a temporary home (so Core ML's compile caches and sidekick's own
   land there and are deleted afterwards) and a models directory holding
   only that model. The build's files are linked, never copied or changed.
   A path other than the default is served through the config's
   `[models."<id>"] compute_units` (D38); the default must be what the
   build's own manifest serves.
3. Runs the model's task sample through the HTTP API:
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
4. Compares every number with the baseline entry for the same model and
   path, under the baseline's tolerances, and ends with a one-line summary.

Inputs that aren't committed: the builds, and the parity references
(`<refs dir>/<model id>/reference.json` and `reference.safetensors`, from
tools/parity_reference.py and tools/classifier_reference.py), which hold
each sample's requests and the fp32 outputs. The baseline records each
reference's sha256 and each build's fingerprint (a sha256 over its files),
so a different sample or build is reported, not compared silently.

A published model with no build here is listed as not available, never
fetched. A model in the baseline whose build is missing fails the gate.

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
        --refs <dir> [--refs <dir> ...] [--paths default|all] [--models <id> ...] \\
        [--daemon <sidekickd>] [--json <file>] [--update-baseline]

    --python       the interpreter for those two steps; default
                   $SIDEKICK_GATE_PYTHON, else the one running the gate.
                   It is checked before any model runs.
    --models-dir   a directory of installed builds (one subdirectory per
                   model); the first holding a model wins
    --refs         a directory of parity references, by model id
    --update-baseline
                   write this run's numbers into the baseline, replacing
                   the entries for the models and paths it ran. Only for
                   a deliberate change: the PR that commits it says why.
    --list         show which published models have a build and a
                   reference here, and exit

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

# The compute units each path serves on (D38's config values).
PATH_UNITS = {"coreml-ane": "cpu_and_ne", "coreml-gpu": "cpu_and_gpu", "coreml-cpu": "cpu_only"}
PATH_ORDER = ["coreml-ane", "coreml-gpu", "coreml-cpu", "onnx-cpu"]
RESERVED_PORTS = range(8790, 8800)   # sidekickd's default and the ports maintainers run it on

# The MTEB slice: an STS task (symmetric prompts) and a reranking task
# (query prompts against documents). Both are deterministic, and short on
# every embedder (seconds to a few minutes on an M1 Max).
MTEB_TASKS = ["STSBenchmark", "AskUbuntuDupQuestions"]

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
    "exact": {"about": "counts that must not change: refused requests, token-count mismatches, "
                       "sanity verdicts, probs/nli_probs disagreements"},
}


class GateError(Exception):
    """A model that couldn't be run: reported, and the gate fails."""


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


def graded_paths(model: str, this_chip: str) -> list[str]:
    """The paths docs/data/models.json grades `model` on, on this chip."""
    rows = json.loads((DATA / "models.json").read_text())["rows"]
    paths = {r["path"] for r in rows if r["model"] == model and r["chip"] == this_chip}
    return [p for p in PATH_ORDER if p in paths]


def find_builds(dirs: list[Path]) -> dict[tuple[str, str], Path]:
    """{(model id, backend): build dir} across the models directories; the
    first directory holding a build wins."""
    out: dict[tuple[str, str], Path] = {}
    for root in dirs:
        if not root.is_dir():
            raise SystemExit(f"--models-dir {root}: not a directory")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / "manifest.toml").is_file():
                m = tomllib.loads((d / "manifest.toml").read_text())
                key = (m.get("id", d.name), m.get("backend", "coreml"))
            elif (d / "classifier.toml").is_file():
                m = tomllib.loads((d / "classifier.toml").read_text())
                key = (m.get("id", d.name), "coreml")
            else:
                continue
            out.setdefault(key, d)
    return out


def find_reference(dirs: list[Path], model: str) -> Path | None:
    for root in dirs:
        d = root / model
        if (d / "reference.json").is_file() and (d / "reference.safetensors").is_file():
            return d
    return None


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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(build: Path) -> str:
    """sha256 over a build's files: one `<sha256>  <relative path>` line
    per file, sorted by path (as the verification runs' model_sha256)."""
    lines = []
    for p in sorted(build.rglob("*")):
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {p.relative_to(build).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
    """One sidekickd serving one model, in a temporary home of its own."""

    def __init__(self, binary: Path, build: Path, model: str, units: str | None, onnx_runtime: str | None):
        self.binary, self.build, self.model, self.units = binary, build, model, units
        self.onnx_runtime = onnx_runtime
        self.root = Path(tempfile.mkdtemp(prefix="sidekick-release-gate-"))
        self.proc = None
        self.port = None
        self.manifest_note = None

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        home, models = self.root / "home", self.root / "models"
        (home / "tmp").mkdir(parents=True)
        mdir = models / self.model
        mdir.mkdir(parents=True)
        manifest_name = "manifest.toml" if (self.build / "manifest.toml").is_file() else "classifier.toml"
        for p in self.build.iterdir():
            if p.name != manifest_name:
                (mdir / p.name).symlink_to(p.resolve())
        text, self.manifest_note = served_manifest(self.build, manifest_name, self.model)
        (mdir / manifest_name).write_text(text)
        self.port = free_port()
        cfg = [f'addr = "127.0.0.1:{self.port}"', f"models_dir = {json.dumps(str(models))}",
               "request_timeout_secs = 900", "load_timeout_secs = 3600", "model_idle_ttl_secs = 3600"]
        if self.onnx_runtime:
            cfg.append(f"onnx_runtime = {json.dumps(self.onnx_runtime)}")
        if self.units:
            cfg += [f'[models.{json.dumps(self.model)}]', f'compute_units = "{self.units}"']
        (self.root / "config.toml").write_text("\n".join(cfg) + "\n")
        # Core ML's compile caches and sidekick's own follow the home
        # directory: they land in the temporary one and go with it.
        env = dict(os.environ, HOME=str(home), CFFIXED_USER_HOME=str(home), TMPDIR=str(home / "tmp"))
        env.pop("SIDEKICK_API_KEY", None)
        self.log = open(self.root / "sidekickd.log", "wb")
        self.proc = subprocess.Popen([str(self.binary), "--config", str(self.root / "config.toml")], env=env,
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
            return "\n".join((self.root / "sidekickd.log").read_text(errors="replace").splitlines()[-n:])
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
        if getattr(self, "log", None):
            self.log.close()
        shutil.rmtree(self.root, ignore_errors=True)


class Client:
    """JSON over one keep-alive connection."""

    def __init__(self, url: str):
        host, port = url.removeprefix("http://").split(":")
        self.host, self.port = host, int(port)
        self.conn = None
        self.version = None

    def call(self, method: str, route: str, body=None):
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=3600)
            try:
                data = None if body is None else json.dumps(body).encode()
                self.conn.request(method, route, body=data, headers={"content-type": "application/json"})
                r = self.conn.getresponse()
                raw = r.read()
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
    cases = [(i, c) for i, c in enumerate(r["cases"]) if c.get("gold")]
    if not cases:
        raise GateError(f"{model}: the reference has no labelled cases")
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
        right += labels[idx] in c["gold"]
        agree += idx == fp32
        tops.append(top)
        preds.append(_symbol(idx))
    answered = n - refused
    return {"sample": {"name": "parity corpus, labelled cases", "n": n, "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": {"n": n, "refused": refused, "gold_accuracy": right / answered if answered else 0.0,
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
    tops, relevant_hits, relevant_groups, agree, worst = [], 0, 0, 0, 0.0
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
    m = {"n": len(corpus), "top1_relevant": relevant_hits / relevant_groups if relevant_groups else 0.0}
    if fp32:
        if scored != len(corpus):
            raise GateError(f"{model}: the reference doesn't cover every group of the rerank corpus")
        m.update({"agree_fp32_top1": agree / scored, "max_abs_diff_fp32": worst})
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
    return {"sample": {"name": "parity corpus", "n": len(cases), "sha256": sample.hexdigest(),
                       "reference_sha256": ref_sha},
            "metrics": {"n": len(cases), "worst_cosine_fp32": min(cos), "mean_cosine_fp32": sum(cos) / len(cos),
                        "cosine_error_fp32": 1 - min(cos), "token_count_mismatches": mismatched}}


def run_mteb(python: str, url: str, model: str, path: str, build: Path, work: Path) -> dict:
    """tools/mteb_transfer.py run on the MTEB slice. Not under HF_HUB_OFFLINE:
    with it, mteb 2.26.14 loads AskUbuntuDupQuestions without its candidate
    lists and scores bge-small-en-v1.5 0.384 instead of 0.618, silently."""
    out = work / "mteb"
    p = subprocess.run([python, str(REPO / "tools" / "mteb_transfer.py"), "run", "--url", url, "--model", model,
                        "--path", path, "--out", str(out), "--build", str(build), "--tasks", *MTEB_TASKS],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise GateError(f"{model}: mteb_transfer.py run failed ({p.returncode}):\n{p.stderr[-2000:]}")
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


def kind_of(model: str, meta: dict, build: Path) -> str:
    if meta["task"] in ("embed", "rerank"):
        return meta["task"]
    fmt = tomllib.loads((build / "classifier.toml").read_text()).get("classify", {}).get("format")
    return "nli" if fmt == "nli" else "classify"


def run_one(args, binary: Path, model: str, meta: dict, path: str, default: str, build: Path,
            ref: Path | None) -> dict:
    t0 = time.monotonic()
    kind = kind_of(model, meta, build)
    if kind in ("nli", "classify", "embed") and ref is None:
        raise GateError(f"{model}: no parity reference in --refs")
    units = None if path == default else PATH_UNITS.get(path)
    rec = {"model": model, "path": path, "kind": kind, "default": path == default}
    print(f"[{model} @ {path}] fingerprinting the build", flush=True)
    rec["build"] = {"fingerprint": fingerprint(build)}
    if ref is not None:
        check_reference(load_reference(ref)[0], build, model)
    with Daemon(binary, build, model, units, args.onnx_runtime) as d:
        if d.manifest_note:
            rec["build"]["served_manifest"] = d.manifest_note
        client = Client(d.url)
        info = client.model_info(model, d.health)
        if path.startswith("coreml-"):
            if info.get("backend", "coreml") != "coreml" or info.get("compute_units") != PATH_UNITS[path]:
                raise GateError(f"{model}: served as {info.get('backend')}/{info.get('compute_units')}, not {path}"
                                + ("" if units else " (the build's manifest doesn't serve its published default)"))
        elif path == "onnx-cpu" and info.get("backend") != "onnx":
            raise GateError(f"{model}: served by {info.get('backend')}, not onnx")
        print(f"[{model} @ {path}] serving on {d.url}; running the {kind} sample", flush=True)
        if kind == "nli":
            rec.update(run_nli(client, model, info, ref))
            if model in scoring_models():
                rec["sanity"] = run_sanity(args.python, d.url, model, build)
        elif kind == "classify":
            rec.update(run_classify(client, model, info, ref))
        elif kind == "rerank":
            rec.update(run_rerank(client, model, info, ref))
        else:
            part = run_embed_parity(client, model, info, ref)
            mt = run_mteb(args.python, d.url, model, path, build, d.root)
            rec.update(part)
            rec["metrics"].update(mt["metrics"])
            rec["mteb"] = mt["mteb"]
        rec["sidekick"] = client.version
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


# ---------------------------------------------------------------- comparison


def metric_rule(name: str) -> str:
    if name in ("n", "refused", "token_count_mismatches", "probs_vs_nli_probs_disagreements"):
        return "exact"
    if name.startswith("mteb:"):
        return "mteb"
    if name in ("cosine_error_fp32", "max_abs_diff_fp32"):
        return "error"
    if name in ("worst_cosine_fp32", "mean_cosine_fp32"):
        return "derived"        # judged through cosine_error_fp32
    if name == "mean_prob":
        return "mean_prob"
    if name.startswith("agree"):
        return "agreement"
    return "accuracy"


def compare(rec: dict, base: dict, tol: dict) -> list[str]:
    """What drifted beyond tolerance: one line each."""
    out = []
    if rec["build"]["fingerprint"] != base["build"]["fingerprint"]:
        out.append(f"build differs from the baseline's ({rec['build']['fingerprint'][:12]} vs "
                   f"{base['build']['fingerprint'][:12]}); a deliberate change re-baselines")
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
        return f"gold {m['gold_accuracy']:.3f}, fp32 agreement {m['agree_fp32']:.3f}, refused {m['refused']}"
    if k == "rerank":
        s = f"relevant top-1 {m['top1_relevant']:.3f}"
        if "agree_fp32_top1" in m:
            s += f", fp32 top-1 agreement {m['agree_fp32_top1']:.3f}, max |Δ| {m['max_abs_diff_fp32']:.3g}"
        return s
    parts = [f"{t} {m[f'mteb:{t}']:.4f}" for t in MTEB_TASKS if f"mteb:{t}" in m]
    return ", ".join(parts) + f"; worst cos fp32 {m['worst_cosine_fp32']:.6f}"


# ---------------------------------------------------------------- main


def provenance(this_chip: str, versions: set) -> dict:
    os_version, os_build = macos()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "sidekick": sorted(v for v in versions if v)[0] if any(versions) else None,
            "chip": this_chip, "os": f"macOS {os_version}", "os_build": os_build,
            "date": datetime.date.today().isoformat()}


def build_daemon(args) -> Path:
    if args.daemon:
        return Path(args.daemon).resolve()
    print("building sidekickd (release, coreml + onnx)", flush=True)
    subprocess.run(["cargo", "build", "--release", "-p", "sidekick-server", "--features", "coreml,onnx",
                    "--bin", "sidekickd"], cwd=REPO, check=True)
    meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout)
    return Path(meta["target_directory"]) / "release" / "sidekickd"


def check_python(python: str, todo: list):
    """Fail before any model runs if `--python` lacks what this run needs:
    mteb (at the pinned version) for an embedder, `tokenizers` for an nli
    model the verification harness scores."""
    kinds = {meta["task"] for _m, meta, _p, _d, build, _r in todo if build is not None}
    sanity = any(build is not None and m in scoring_models() for m, _meta, _p, _d, build, _r in todo)
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


def plan(args, this_chip: str):
    """[(model, meta, path, default path, build or None, reference or None)]."""
    models = models_toml()
    builds = find_builds([Path(d) for d in args.models_dir])
    refs = [Path(d) for d in args.refs]
    wanted = args.models or list(models)
    unknown = [m for m in wanted if m not in models]
    if unknown:
        raise SystemExit(f"not published (docs/data/models.toml): {', '.join(unknown)}")
    out = []
    for model in wanted:
        meta = models[model]
        default = meta["served"]
        paths = [default]
        if args.paths == "all":
            paths += [p for p in graded_paths(model, this_chip) if p != default]
        for path in paths:
            backend = "onnx" if path == "onnx-cpu" else "coreml"
            out.append((model, meta, path, default, builds.get((model, backend)), find_reference(refs, model)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", action="append", default=[], required=True,
                    help="a directory of installed builds (repeatable; the first holding a model wins)")
    ap.add_argument("--refs", action="append", default=[], help="a directory of parity references (repeatable)")
    ap.add_argument("--paths", choices=("default", "all"), default="default")
    ap.add_argument("--models", nargs="+", help="only these published models")
    ap.add_argument("--daemon", help="a built sidekickd to gate instead of building one")
    ap.add_argument("--onnx-runtime", help="libonnxruntime for onnx-cpu builds (else ORT_DYLIB_PATH or beside "
                                            "the binary)")
    ap.add_argument("--python", default=os.environ.get("SIDEKICK_GATE_PYTHON") or sys.executable,
                    help="the interpreter for tools/mteb_transfer.py and scoring_run.py sanity, built from "
                         "tools/release_gate.requirements.txt (default: $SIDEKICK_GATE_PYTHON, else this one)")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--json", type=Path, help="write this run's full record here")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    this_chip = chip()
    todo = plan(args, this_chip)
    if args.list:
        for model, meta, path, default, build, ref in todo:
            print(f"{model:36} {path:11} build: {'yes' if build else 'not available here':20} "
                  f"reference: {'yes' if ref else 'none'}")
        return 0
    base = json.loads(args.baseline.read_text()) if args.baseline.exists() else {"entries": []}
    tol = base.get("tolerances", TOLERANCES)
    by_key = {(e["model"], e["path"]): e for e in base["entries"]}

    check_python(args.python, todo)
    binary = build_daemon(args)
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    print(version, flush=True)
    t_all = time.monotonic()
    results, errors, unavailable = [], {}, []
    for model, meta, path, default, build, ref in todo:
        if build is None:
            unavailable.append((model, path))
            print(f"[{model} @ {path}] not available here", flush=True)
            continue
        try:
            rec = run_one(args, binary, model, meta, path, default, build, ref)
        except GateError as e:
            errors[(model, path)] = str(e)
            print(f"[{model} @ {path}] ERROR: {e}", flush=True)
            continue
        b = by_key.get((model, path))
        rec["drift"] = compare(rec, b, tol) if b else None
        results.append(rec)
        f = flips(rec, b)
        state = ("no baseline" if b is None else "ok" if not rec["drift"] else "DRIFT")
        print(f"[{model} @ {path}] {state} in {rec['seconds']} s: {headline(rec)}"
              + (f"; {f} prediction(s) differ from the baseline run" if f else ""), flush=True)
        for line in rec["drift"] or []:
            print(f"    {line}", flush=True)
    total = round(time.monotonic() - t_all, 1)

    # Baselined models this run couldn't serve: the gate can't vouch for them.
    missing = [(m, p) for (m, p) in unavailable if (m, p) in by_key]
    drifted = [r for r in results if r["drift"]]
    unbaselined = [r for r in results if r["drift"] is None]
    prov = provenance(this_chip, {r.get("sidekick") for r in results})
    prov["daemon"] = version

    print()
    print(f"{'model':36} {'path':11} {'seconds':>8}  result")
    for r in results:
        print(f"{r['model']:36} {r['path']:11} {r['seconds']:8.1f}  {headline(r)}")
    for (m, p), e in errors.items():
        print(f"{m:36} {p:11} {'':>8}  ERROR: {e.splitlines()[0]}")
    if unavailable:
        print(f"not available here ({len(unavailable)}): " + ", ".join(f"{m} @ {p}" for m, p in unavailable))

    if args.json:
        args.json.write_text(json.dumps({"provenance": prov, "seconds": total, "results": results,
                                         "errors": {f"{m} @ {p}": e for (m, p), e in errors.items()},
                                         "not_available": [f"{m} @ {p}" for m, p in unavailable]},
                                        indent=1) + "\n")

    if args.update_baseline:
        if errors:
            print("refusing to update the baseline from a run with errors", file=sys.stderr)
            return 1
        write_baseline(args.baseline, base, results, prov)
        print(f"baseline updated: {len(results)} entries written to {args.baseline}")

    ok = not errors and not drifted and not missing
    sha = (prov["commit"] or "unknown")[:7] + ("+dirty" if prov["dirty"] else "")
    print(f"release gate: {'PASS' if ok else 'FAIL'} — {len(results) - len(drifted) - len(unbaselined)} within "
          f"tolerance, {len(drifted)} drifted, {len(errors)} errors, {len(missing)} baselined but missing, "
          f"{len(unbaselined)} without a baseline, {len(unavailable)} not available here — "
          f"{prov['sidekick'] or version} ({sha}), {prov['chip']}, {prov['os']}, {total / 60:.1f} min")
    return 0 if ok else 1


def write_baseline(path: Path, base: dict, results: list[dict], prov: dict):
    keep = {(e["model"], e["path"]): e for e in base.get("entries", [])}
    for r in results:
        e = {k: r[k] for k in ("model", "path", "kind", "default", "build", "sample", "metrics", "predictions",
                               "sanity", "mteb", "seconds") if k in r}
        e["provenance"] = {**{k: prov[k] for k in ("commit", "dirty", "chip", "os", "os_build", "date")},
                           "sidekick": r.get("sidekick")}
        keep[(r["model"], r["path"])] = e
    order = list(models_toml())
    entries = sorted(keep.values(), key=lambda e: (order.index(e["model"]) if e["model"] in order else 999,
                                                   PATH_ORDER.index(e["path"])))
    gated = {e["model"] for e in entries}
    out = {
        "about": ("The release gate's baseline: each published model's task sample, run through sidekickd over "
                  "HTTP, on the paths listed. Written by tools/release_gate.py --update-baseline; do not edit "
                  "by hand. A change to it is deliberate, and its PR says why."),
        "tolerances": base.get("tolerances", TOLERANCES),
        "mteb_tasks": MTEB_TASKS,
        "updated": {k: prov[k] for k in ("commit", "dirty", "sidekick", "chip", "os", "os_build", "date")},
        "entries": entries,
        "not_gated": sorted((m for m in models_toml() if m not in gated), key=order.index),
    }
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
