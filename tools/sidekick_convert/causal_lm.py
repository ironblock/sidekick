"""Causal language models: a LlamaForCausalLM checkpoint to a `kv-state-1`
decoder for the chat route (docs/DECISIONS.md D53; docs/design/open-llm.md,
"Conversion and release"; docs/CONVERTING.md, "Causal language models").

`run(job, install_dir)` takes a checkpoint through:
1. **Checks**: the committed chat.toml (examples/chat/<id>/chat.toml)
   against the checkpoint, and the committed text fixtures
   (fixtures/causal-lm/) against their recorded hashes and token ids, with
   the calibration disjoint from what is graded.
2. **References**: the checkpoint's own fp32 forward, teacher-forced over
   the evaluation texts, its greedy continuations of the evaluation
   prompts, and the ideal-fp16 ceiling (fp16sim) of the same texts.
3. **Ranges**, per calibration draw: the R1/R2 rotation, then the residual
   stream's K and the RMSNorm pre-scales from the draw's activations
   (backbones.llama_causal); then the **rewrite gate**: the rotated,
   rewritten model in fp32 must reproduce the checkpoint's logits.
4. **Palettes**: k-means per group of output channels, once per rotation.
5. **GPTQ** onto those palettes, per draw (techniques.gptq), layer by
   layer, each layer calibrated on the outputs of the layers already
   quantised.
6. **The PyTorch screen**, per draw: the quantised model teacher-forced in
   fp32 over the evaluation texts, graded as an ANE build is graded. Only a
   recipe whose screen passes is compiled, since every ANE compile leaves
   about a copy of the weights in a system cache nobody can delete. With a
   second (stability) draw, both draws must grade alike.
7. **Conversion** of the first draw, one chunk per process: trace,
   coremltools with the KV cache as state, the weights stored as 8-bit
   palettes (`unique` mode, so GPTQ's values survive exactly), compiled.
8. **Gates**: no forbidden op (the fused attention op, D25; the native
   silu, which StableSilu replaced), each chunk under the ANE weight limit
   (D32), and, unless skipped, the compute plan with every operation of
   every chunk on the ANE (one plan read per chunk, which compiles it).
9. **Install**: the chunks, the host's embedding and RoPE tables, the
   tokenizer, the chat template, and chat.toml with its [build] and
   [placement] tables.

Each heavy step runs in its own process (`python -m sidekick_convert.causal_lm
step ...`), so its memory is returned when it ends, and leaves its result in
the job's work directory, where a later run finds it: the references, the
ranges, the palettes and each draw's GPTQ weights are computed once. The
ANE grade itself is tools/grade_causal_lm.py's: it runs the installed build
through sidekick's own decoder (sidekick_coreml::KvDecoder) and grades it
with `grade()` below.
"""

import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from . import core
from . import tokenizer as _tok

REPO = core.REPO
FIXTURES = REPO / "fixtures" / "causal-lm"
CALIBRATION = FIXTURES / "calibration.jsonl"
EVALUATION = FIXTURES / "evaluation.jsonl"
BENCHMARK = REPO / "fixtures" / "tasks" / "chat"
EXPECTATIONS = REPO / "fixtures" / "parity" / "expectations.toml"

FIRST_GRADED = 128
"""Teacher-forced grading reads the logits at positions FIRST_GRADED - 1 to
n - 1 of each evaluation text (the predictions of tokens FIRST_GRADED to n),
as #173 graded: every graded position has at least 127 tokens of context."""

GREEDY_TOKENS = 96

TOP1 = {"A": 0.995, "B": 0.98, "C": 0.95}
"""Top-1 agreement with fp32 for each letter, at least. The ideal-fp16
ceiling of MiniCPM5-2B agrees at 99.7% (#173), so A allows about that."""

WIDE_MARGIN = math.log(2.0)
"""A top-1 flip caps the grade at C where fp32 prefers its top token by at
least this much in log-probability (twice as likely as the runner-up), as
a classifier's decision flip caps its grade (D26). Flips between
near-ties are counted, not capped."""

ANE_SEQ_CAP = 2048
"""DEFAULT_ANE_MAX_SEQ (D48): the cache capacity's limit unless the manifest
sets ane_max_seq_len."""

FORBID_OPS = frozenset({core.FUSED_ATTENTION, "silu"})


def cosine_scale():
    """The cosine letter scale, from the one definition the parity suite
    reads (fixtures/parity/expectations.toml, [grades].cosine)."""
    return tomllib.loads(EXPECTATIONS.read_text())["grades"]["cosine"]


def letter(value, scale, higher=True):
    if not np.isfinite(value):
        return "F"
    for g in ("A", "B", "C"):
        if (value >= scale[g]) if higher else (value <= scale[g]):
            return g
    return "D"


def worse(*letters):
    return max(letters)        # "A" < "B" < "C" < "D" < "F"


# ---------------------------------------------------------------- recipe and job

@dataclasses.dataclass
class Recipe:
    """The choices a build is made with, recorded in its [build] table. The
    shapes (T, C, the layer split) are the committed chat.toml's."""
    rotation_seed: int = 0
    r2: bool = True
    draw_seed: int = 0
    stability_seed: int = 1          # None: no stability draw
    tokens: int = 40960              # calibration tokens per draw
    sequence: int = 2048             # tokens per calibration sequence
    K: int = None                    # the residual scale; None: calibrated
    palette_group: int = 16
    palette_seed: int = 0
    act_order: bool = True
    damp: float = 0.01
    head_split: int = 4              # the head as this many convolutions, concatenated
    fp16_projections: tuple = ()     # "layer.proj" names left unquantised
    min_grade: str = "B"             # the PyTorch screen's floor for compiling
    threads: int = 8                 # torch threads in every step (GPTQ's bytes may depend on it)

    def draws(self):
        return [self.draw_seed] + ([] if self.stability_seed is None else [self.stability_seed])


@dataclasses.dataclass
class Job:
    model_id: str
    src: str
    work: str
    recipe: Recipe
    plan: bool = True                # read the compute plans (one ANE compile per chunk)
    screen_only: bool = False
    keep_packages: bool = False

    def path(self, *parts):
        return Path(self.work).joinpath(*parts)

    def save(self):
        self.path().mkdir(parents=True, exist_ok=True)
        d = dataclasses.asdict(self)
        d["recipe"]["fp16_projections"] = list(self.recipe.fp16_projections)
        self.path("job.json").write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d["recipe"] = Recipe(**{**d["recipe"], "fp16_projections": tuple(d["recipe"]["fp16_projections"])})
        return cls(**d)


def manifest_path(model_id):
    return REPO / "examples" / "chat" / model_id / "chat.toml"


# ---------------------------------------------------------------- tokens and fixtures

class Tokens:
    """tokenizer.json (Python `tokenizers`, as sidekick's Rust side loads it)
    for text, and the checkpoint's chat template with the manifest's
    template arguments for conversations."""

    def __init__(self, src, template_kwargs):
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        self.tok = Tokenizer.from_file(str(Path(src) / "tokenizer.json"))
        self.hf = AutoTokenizer.from_pretrained(str(src))
        self.template_kwargs = dict(template_kwargs)

    @classmethod
    def load(cls, src, model_id):
        m = tomllib.loads(manifest_path(model_id).read_text())
        return cls(src, m.get("chat", {}).get("template_kwargs", {}))

    def text(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def chat(self, messages, generation_prompt):
        ids = self.hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt,
                                          **self.template_kwargs)
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def encode_entry(self, e):
        if "messages" in e:
            return self.chat(e["messages"], e["messages"][-1]["role"] == "user")
        ids = self.text(e["text"])
        return ids[:e["tokens"]] if "tokens" in e else ids


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass
class Fixture:
    calibration: list          # entries
    evaluation: list
    ids: dict                  # {"calibration": {id: ids}, "evaluation": {id: ids}}
    calibration_sha256: str
    tokens_file: Path

    @property
    def teacher_forced(self):
        return [(e["id"], self.ids["evaluation"][e["id"]]) for e in self.evaluation if e["kind"] == "teacher-forced"]

    @property
    def greedy(self):
        return [(e["id"], self.ids["evaluation"][e["id"]]) for e in self.evaluation if e["kind"] == "greedy"]


SHARED_RUN = 32
"""Calibration and evaluation may share no run of this many tokens: long
enough that a shared passage counts and a shared link anchor or function
signature doesn't (the repository's documents repeat both)."""


def ngrams(ids, n=SHARED_RUN):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def load_fixture(src, model_id, tokens=None, report=print):
    """The committed fixtures with their token ids for `model_id`, checked:
    the text files' and the tokenizer's sha256 as recorded, every document
    re-encoded to the recorded ids (when `tokens` is given), and the
    calibration disjoint from what is graded: no parity-corpus text
    (core.Calibration's guard), no SHARED_RUN-token run shared with an
    evaluation text, and no Wikipedia article the chat benchmark uses."""
    path = FIXTURES / f"{model_id}.tokens.json"
    rec = json.loads(path.read_text())
    problems = []
    for key, f in (("calibration_sha256", CALIBRATION), ("evaluation_sha256", EVALUATION)):
        if rec[key] != sha256(f):
            problems.append(f"{f.name} changed since {path.name} was written (rerun make_causal_lm_fixture.py tokens)")
    if rec["tokenizer_sha256"] != sha256(Path(src) / "tokenizer.json"):
        problems.append("the checkpoint's tokenizer.json isn't the one the token ids were made with")
    if problems:
        raise core.GateFailure("; ".join(problems))
    cal, ev = read_jsonl(CALIBRATION), read_jsonl(EVALUATION)
    if tokens is not None:
        if tokens.template_kwargs != rec["template_kwargs"]:
            raise core.GateFailure(f"template arguments {tokens.template_kwargs} != the fixture's "
                                   f"{rec['template_kwargs']}")
        for kind, entries in (("calibration", cal), ("evaluation", ev)):
            bad = [e["id"] for e in entries if tokens.encode_entry(e) != rec[kind][e["id"]]]
            if bad:
                raise core.GateFailure(f"{kind} documents encode differently than recorded: {bad[:5]}")
    core.Calibration([e["text"] for e in cal if "text" in e]
                     + [m["content"] for e in cal for m in e.get("messages", ())])
    graded = set()
    for _, ids in [(e["id"], rec["evaluation"][e["id"]]) for e in ev]:
        graded |= ngrams(ids)
    shared = [e["id"] for e in cal if ngrams(rec["calibration"][e["id"]]) & graded]
    if shared:
        raise core.GateFailure(f"calibration documents share {SHARED_RUN}-token runs with the evaluation set: "
                               f"{shared[:5]}")
    bench = set()
    import urllib.parse
    for f in BENCHMARK.glob("*.jsonl"):
        for e in read_jsonl(f):
            if "wikipedia.org" in e.get("source", ""):
                bench.add(urllib.parse.parse_qs(urllib.parse.urlparse(e["source"]).query)["title"][0])
    clash = [e["id"] for e in cal if "wikipedia.org" in e["source"]
             and urllib.parse.parse_qs(urllib.parse.urlparse(e["source"]).query)["title"][0] in bench]
    if clash:
        raise core.GateFailure(f"calibration articles the chat benchmark uses: {clash}")
    n = sum(map(len, rec["calibration"].values()))
    report(f"fixture: {len(cal)} calibration documents ({n} tokens), {len(ev)} evaluation entries; disjoint")
    return Fixture(cal, ev, {"calibration": rec["calibration"], "evaluation": rec["evaluation"]},
                   rec["calibration_sha256"], path)


def draw(fixture, seed, tokens, sequence, separator):
    """A calibration draw: the documents in a seeded order (numpy's
    default_rng(seed).permutation), packed into sequences of exactly
    `sequence` tokens, each starting at a document: documents are appended,
    `separator` between them, until the sequence is full, and the rest of
    the last one is dropped. `tokens / sequence` sequences."""
    if tokens % sequence:
        raise ValueError(f"{tokens} calibration tokens aren't a whole number of {sequence}-token sequences")
    docs = [fixture.ids["calibration"][e["id"]] for e in fixture.calibration]
    order = np.random.default_rng(seed).permutation(len(docs))
    out, cur = [], []
    for j in order:
        cur = cur + (separator if cur else []) + docs[j]
        if len(cur) >= sequence:
            out.append(cur[:sequence])
            cur = []
            if len(out) == tokens // sequence:
                return out
    raise ValueError(f"the fixture holds too few tokens for {tokens // sequence} sequences of {sequence}")


# ---------------------------------------------------------------- manifest

def check_manifest(m, src, model_id, report=print):
    """The committed chat.toml against the checkpoint and the contract."""
    cfg = json.loads((Path(src) / "config.json").read_text())
    gen = json.loads((Path(src) / "generation_config.json").read_text()) if (
        Path(src) / "generation_config.json").exists() else {}
    tcfg = json.loads((Path(src) / "tokenizer_config.json").read_text())
    d, errors = m["decoder"], []

    def check(ok, message):
        if not ok:
            errors.append(message)

    check(m["id"] == model_id, f"id {m['id']!r} != {model_id!r} (name the install directory after the model id)")
    check(m["task"] == "text-generation" and m["interface"] == "kv-state-1", "task/interface")
    check(m.get("compute_units", "cpu_and_ne") == "cpu_and_ne", "compute_units must be cpu_and_ne")
    snap = _tok.snapshot_revision(src)
    check(snap is None or m["source"]["revision"] == snap,
          f"source.revision {m['source']['revision']} != snapshot {snap}")
    for key, want in (("hidden_size", cfg["hidden_size"]),
                      ("head_dim", cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]),
                      ("kv_heads", cfg.get("num_key_value_heads") or cfg["num_attention_heads"]),
                      ("vocab_size", cfg["vocab_size"])):
        check(d[key] == want, f"[decoder] {key} {d[key]} != the checkpoint's {want}")
    check(sum(d["layers"]) == cfg["num_hidden_layers"], f"layers {d['layers']} don't sum to {cfg['num_hidden_layers']}")
    check(d["chunks"] == len(d["layers"]), "chunks != len(layers)")
    C, T = d["cache_capacity"], d["tokens_per_call"]
    check(d["context_length"] == C - T + 1, f"context_length {d['context_length']} != C - T + 1 = {C - T + 1}")
    check(C <= m.get("ane_max_seq_len", ANE_SEQ_CAP), f"cache_capacity {C} is past the ANE length cap (D48)")
    check(C <= cfg.get("max_position_embeddings", C), "cache_capacity past max_position_embeddings")
    chat = m.get("chat", {})
    check(chat.get("bos_token") == tcfg.get("bos_token"), f"bos_token != tokenizer_config's {tcfg.get('bos_token')!r}")
    check(chat.get("eos_token") == tcfg.get("eos_token"), f"eos_token != tokenizer_config's {tcfg.get('eos_token')!r}")
    eos = gen.get("eos_token_id", cfg.get("eos_token_id"))
    eos = eos if isinstance(eos, list) else [eos]
    check(chat.get("stop_token_ids") == eos,
          f"stop_token_ids {chat.get('stop_token_ids')} != generation_config's {eos}")
    check((Path(src) / chat.get("template", "chat_template.jinja")).exists(), "the chat template file is missing")
    for key in ("temperature", "top_p"):
        if key in gen:
            check(m.get("sampling", {}).get(key) == gen[key], f"[sampling] {key} != generation_config's {gen[key]}")
    if errors:
        raise core.GateFailure("chat.toml: " + "; ".join(errors))
    report(f"chat.toml: {d['chunks']} chunks {d['layers']}, T = {T}, C = {C}, context {d['context_length']}")


def cuts_of(layers):
    out, lo = [], 0
    for n in layers:
        out.append((lo, lo + n))
        lo += n
    return out


def build_toml(job, ranges, fixture, placement=None):
    r = job.recipe
    fp16 = ", ".join(f'"{p}"' for p in r.fp16_projections)
    lines = ["", "[build]",
             'weights = "palette8"',
             f"palettes = {{ group = {r.palette_group}, kmeans_seed = {r.palette_seed} }}",
             f"gptq = {{ act_order = {str(r.act_order).lower()}, damp = {r.damp}, threads = {r.threads} }}",
             f"rotation = {{ r1 = true, r2 = {str(r.r2).lower()}, seed = {r.rotation_seed} }}",
             f'calibration = {{ set = "{CALIBRATION.relative_to(REPO)}", sha256 = "{fixture.calibration_sha256}", '
             f"tokens = {r.tokens}, sequence = {r.sequence}, seed = {r.draw_seed} }}",
             f"residual_scale = {ranges['K']}",
             f"head_split = {r.head_split}",
             f"fp16_projections = [{fp16}]"]
    return "\n".join(lines) + "\n" + (placement or "")


def placement_toml(summaries):
    from . import plan
    chip, build = plan.machine()
    lines = ["", "[placement]", 'compute_units = "cpu_and_ne"', f'chip = "{chip}"', f'macos_build = "{build}"',
             f'date = "{datetime.date.today().isoformat()}"']
    for i, s in enumerate(summaries):
        lines += ["", f"[placement.chunks.{i}]", *plan._counts(s)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- grading

def softmax_margin(ref):
    """fp32's log-probability margin between its top two tokens, per row."""
    top2 = np.partition(ref, -2, axis=1)[:, -2:]
    return np.abs(top2[:, 1] - top2[:, 0])         # a logit difference is a log-probability difference


def grade(got, refs):
    """Teacher-forced grade of `got` ({text: logits rows}) against the fp32
    `refs` (the same rows): per position the logits' cosine and whether the
    top-1 token agrees. The letter is the worse of the first-percentile
    cosine's (on the parity suite's cosine scale) and top-1 agreement's
    (TOP1), capped at C by any flip where fp32's top-two margin is at least
    WIDE_MARGIN. The worst position is reported, not graded."""
    cos, agree, flips, texts = [], [], [], {}
    for name in refs:
        a = np.asarray(got[name], dtype=np.float64)
        b = np.asarray(refs[name], dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"{name}: rows {a.shape} != reference {b.shape}")
        finite = np.isfinite(a).all(1)
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        c[~finite] = np.nan
        t = a.argmax(1) == b.argmax(1)
        margin = softmax_margin(b)
        for j in np.nonzero(~t)[0]:
            flips.append({"text": name, "row": int(j), "fp32_margin": round(float(margin[j]), 4)})
        cos += c.tolist()
        agree += t.tolist()
        texts[name] = {"top1": round(float(t.mean()), 4), "cos_min": round(float(np.min(c)), 6)}
    c = np.array(cos)
    nonfinite = not np.isfinite(c).all()
    top1 = float(np.mean(agree))
    p1 = float("nan") if nonfinite else float(np.percentile(c, 1))
    wide = [f for f in flips if f["fp32_margin"] >= WIDE_MARGIN]
    by_cos, by_top1 = letter(p1, cosine_scale()), letter(top1, TOP1)
    g = "F" if nonfinite else worse(by_cos, by_top1)
    if wide and g in ("A", "B"):
        g = "C"
    return {"positions": len(c), "top1": round(top1, 6), "cos_mean": float(np.nanmean(c)),
            "cos_p1": p1, "cos_min": float(np.min(c)) if not nonfinite else float("nan"),
            "grade": g, "grade_cos_p1": by_cos, "grade_top1": by_top1, "wide_flips": len(wide),
            "flips": flips, "texts": texts}


def describe(g):
    return (f"grade {g['grade']} (p1 cosine {g['cos_p1']:.5f} {g['grade_cos_p1']}, top-1 {g['top1']:.2%} "
            f"{g['grade_top1']}, {g['wide_flips']} wide flips); mean {g['cos_mean']:.6f}, worst {g['cos_min']:.5f} "
            f"over {g['positions']} positions")


def load_refs(job, kind="fp32"):
    d = job.path("references")
    meta = json.loads((d / "meta.json").read_text())
    return {name: np.load(d / f"{kind}_{name}.npy") for name in meta["texts"]}


# ---------------------------------------------------------------- the run

def _child(job, step, *args):
    """Run one step in its own process (its memory is returned when it ends)."""
    t0 = time.time()
    tools = str(REPO / "tools")
    env = {**os.environ, "PYTHONPATH": tools + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, "-m", "sidekick_convert.causal_lm", "step", step, str(job.path("job.json")),
                    *map(str, args)], check=True, cwd=tools, env=env)
    print(f"[{step}{' ' + ' '.join(map(str, args)) if args else ''}: {time.time() - t0:.0f} s]", flush=True)


def run(job, install_dir):
    """Convert, screen, gate and install (see the module docstring)."""
    install_dir = Path(install_dir).expanduser()
    if install_dir.name != job.model_id:
        raise SystemExit(f"name the install directory after the model id: {job.model_id}")
    m = tomllib.loads(manifest_path(job.model_id).read_text())
    check_manifest(m, job.src, job.model_id)
    tokens = Tokens(job.src, m.get("chat", {}).get("template_kwargs", {}))
    fixture = load_fixture(job.src, job.model_id, tokens)
    if job.path("job.json").exists():
        # the intermediates are keyed by draw seed alone, so one work directory holds one recipe
        old = Job.load(job.path("job.json"))
        same = ("draw_seed", "stability_seed", "min_grade")
        differ = [k for k, v in dataclasses.asdict(job.recipe).items()
                  if k not in same and getattr(old.recipe, k) != v]
        if differ or old.src != job.src or old.model_id != job.model_id:
            raise core.GateFailure(f"{job.work} holds another recipe's intermediates ({differ or 'checkpoint'}); "
                                   "give each recipe its own work directory")
    job.save()
    report = {"recipe": dataclasses.asdict(job.recipe)}
    if not job.path("references", "meta.json").exists():
        _child(job, "references")
    for seed in job.recipe.draws():
        if not job.path(f"ranges-{seed}.json").exists():
            _child(job, "ranges", seed)
        report[f"rewrite_gate_{seed}"] = json.loads(job.path(f"ranges-{seed}.json").read_text())["gate"]
    K = {json.loads(job.path(f"ranges-{s}.json").read_text())["ranges"]["K"] for s in job.recipe.draws()}
    for k in sorted(K):
        if not job.path(f"palettes-{job.recipe.rotation_seed}-K{k}.pt").exists():
            _child(job, "palettes", k)
    screens = {}
    for seed in job.recipe.draws():
        if not job.path(f"gptq-{seed}", "meta.json").exists():
            _child(job, "gptq", seed)
        if not job.path(f"screen-{seed}.json").exists():
            _child(job, "screen", seed)
        screens[seed] = json.loads(job.path(f"screen-{seed}.json").read_text())
        print(f"screen, draw {seed}: {describe(screens[seed])}", flush=True)
    report["screens"] = {str(s): {k: v for k, v in g.items() if k != "flips"} for s, g in screens.items()}
    letters = {s: g["grade"] for s, g in screens.items()}
    stable = len(set(letters.values())) == 1
    report["stable"] = stable
    report["screen_grade"] = worse(*letters.values())
    if not stable:
        print(f"the draws grade differently: {letters}; the build's grade is the lower, {report['screen_grade']}")
    job.path("report.json").write_text(json.dumps(report, indent=1))
    if worse(report["screen_grade"], job.recipe.min_grade) != job.recipe.min_grade:
        raise core.GateFailure(f"the PyTorch screen grades {report['screen_grade']}, below the recipe's "
                               f"{job.recipe.min_grade}: not compiling")
    if job.screen_only:
        return report
    core.require_coremltools()
    stage = job.path("stage")
    for i in range(m["decoder"]["chunks"]):
        if not (stage / f"decoder.{i}.mlmodelc").exists():
            _child(job, "convert", i)
    from . import plan
    chunks, summaries = [], []
    for i in range(m["decoder"]["chunks"]):
        c = stage / f"decoder.{i}.mlmodelc"
        info = json.loads((stage / f"decoder.{i}.json").read_text())
        bad = sorted(set(info["ops"]) & FORBID_OPS)
        if bad:
            raise core.GateFailure(f"chunk {i} contains forbidden ops {bad}")
        size = plan.weights_bytes(c)
        if size > plan.MAX_ANE_PROGRAM_WEIGHT_BYTES:
            raise core.GateFailure(plan.over_cap(job.model_id, "decoder", size, i, True))
        chunks.append({"weight_bytes": size, **info})
    report["chunks"] = chunks
    install_dir.mkdir(parents=True, exist_ok=True)
    for i in range(m["decoder"]["chunks"]):
        dest = install_dir / f"decoder.{i}.mlmodelc"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(stage / f"decoder.{i}.mlmodelc", dest)
    if job.plan:
        for i in range(m["decoder"]["chunks"]):
            s = plan.summarize(plan.read(install_dir / f"decoder.{i}.mlmodelc"))
            s["units"] = "CPU_AND_NE"
            if s["cpu"] or s["gpu"]:
                raise core.GateFailure(f"compute plan: chunk {i} has operations off the ANE: {s['off']} "
                                       "(a kv-state-1 decoder's state must stay on the ANE)")
            print(f"chunk {i}: compute plan {s['ane']}/{s['assigned']} operations on the ANE", flush=True)
            summaries.append(s)
        report["plans"] = [{k: s[k] for k in ("ane", "gpu", "cpu", "unassigned", "total")} for s in summaries]
    install_host(job, m, install_dir, report, fixture, summaries)
    job.path("report.json").write_text(json.dumps(report, indent=1))
    print(f"installed {install_dir}", flush=True)
    return report


def install_host(job, m, install_dir, report, fixture, summaries):
    from safetensors.numpy import save_file
    from .backbones import llama_causal as LC
    d = m["decoder"]
    ranges = json.loads(job.path(f"ranges-{job.recipe.draw_seed}.json").read_text())["ranges"]
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(job.recipe.rotation_seed, job.recipe.r2).bind(ck)
    table = LC.embedding_table(ck, rot, ranges["K"]).half().numpy()
    save_file({"embedding": table}, str(install_dir / m["decoder"]["host"]["embedding"]))
    cos, sin = LC.rope_tables(ck.theta, d["head_dim"], range(d["cache_capacity"]))
    save_file({"cos": cos.astype(np.float16), "sin": sin.astype(np.float16)},
              str(install_dir / m["decoder"]["host"]["rope"]))
    for name in (m["decoder"]["host"]["embedding"], m["decoder"]["host"]["rope"]):
        os.chmod(install_dir / name, 0o644)        # safetensors writes owner-only files
    _tok.prepare(job.src, install_dir / m["tokenizer"], mode="clean")
    template = m["chat"]["template"]
    shutil.copy(Path(job.src) / template, install_dir / template)
    text = manifest_path(job.model_id).read_text()
    placement = placement_toml(summaries) if summaries else None
    (install_dir / "chat.toml").write_text(text.rstrip("\n") + "\n" + build_toml(job, ranges, fixture, placement))


# ---------------------------------------------------------------- steps (child processes)

def _setup(job):
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(job.recipe.threads)
    from .backbones import llama_causal as LC
    m = tomllib.loads(manifest_path(job.model_id).read_text())
    fixture = load_fixture(job.src, job.model_id, report=lambda *_: None)
    return LC, m, fixture


def _separator(job):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(job.src) / "tokenizer.json")).encode("\n\n", add_special_tokens=False).ids


def step_references(job):
    """fp32 teacher-forced logits, greedy continuations, the fp16 ceiling."""
    import torch
    from transformers import AutoModelForCausalLM
    from . import fp16sim
    _, _, fixture = _setup(job)
    out = job.path("references")
    out.mkdir(parents=True, exist_ok=True)
    hf = AutoModelForCausalLM.from_pretrained(job.src, dtype=torch.float32, attn_implementation="eager").eval()
    texts = fixture.teacher_forced
    for name, ids in texts:
        t0 = time.time()
        lg = hf(torch.tensor([ids])).logits[0, FIRST_GRADED - 1:len(ids)].float().numpy()
        np.save(out / f"fp32_{name}.npy", lg)
        print(f"fp32 {name}: {time.time() - t0:.1f} s", flush=True)
    meta = {"texts": [n for n, _ in texts], "first_graded": FIRST_GRADED, "revision": _tok.snapshot_revision(job.src)}
    greedy = []
    for name, ids in fixture.greedy:
        g = hf.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        do_sample=False, max_new_tokens=GREEDY_TOKENS, pad_token_id=hf.config.pad_token_id)
        greedy.append({"name": name, "ids": ids, "new": g[0, len(ids):].tolist()})
        print(f"greedy {name}: {len(greedy[-1]['new'])} tokens", flush=True)
    (out / "greedy.json").write_text(json.dumps(greedy))
    for name, ids in texts:
        t0 = time.time()
        try:
            lg = fp16sim.run(hf, torch.tensor([ids])).logits[0, FIRST_GRADED - 1:len(ids)].float().numpy()
        except fp16sim.Fp16Overflow as e:
            print(f"no fp16 ceiling: {e}", flush=True)
            meta["ceiling"] = str(e)
            break
        np.save(out / f"fp16_{name}.npy", lg)
        print(f"fp16 ceiling {name}: {time.time() - t0:.1f} s", flush=True)
    else:
        meta["ceiling"] = grade(load_refs_from(out, meta, "fp16"), load_refs_from(out, meta, "fp32"))
        print(f"fp16 ceiling: {describe(meta['ceiling'])}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_refs_from(d, meta, kind):
    return {name: np.load(d / f"{kind}_{name}.npy") for name in meta["texts"]}


def step_ranges(job, seed):
    """The draw's range calibration, and the rewrite gate."""
    LC, m, fixture = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    seqs = draw(fixture, seed, r.tokens, r.sequence, _separator(job))
    ranges = LC.calibrate_ranges(ck, seqs, rot, K=r.K)
    # the rewrite gate: the rotated, rescaled model in fp32 against the checkpoint's own forward
    texts = fixture.teacher_forced
    hs = LC.stream(ck, [ids for _, ids in texts], lambda i: LC.build_layer(ck, i, rot, ranges),
                   LC.embedding_table(ck, rot, ranges.K))
    norm, head = LC.final_norm(ck, rot, ranges), LC.head_weight(ck, rot)
    refs = load_refs(job)
    got = {name: LC.logits(norm, head, h, (FIRST_GRADED - 1, len(ids))).numpy() for (name, ids), h in zip(texts, hs)}
    g = grade(got, refs)
    gate = {"cos_min": g["cos_min"], "top1": g["top1"],
            "max_abs_logit_diff": max(float(np.abs(got[n] - refs[n]).max()) for n in refs)}
    print(f"rewrite gate, draw {seed}: worst cosine {g['cos_min']:.7f}, top-1 {g['top1']:.2%}, max |dlogit| "
          f"{gate['max_abs_logit_diff']:.2e}", flush=True)
    if not g["cos_min"] >= 0.99999:
        raise core.GateFailure(f"the rewritten model doesn't reproduce the checkpoint: worst cosine {g['cos_min']}")
    job.path(f"ranges-{seed}.json").write_text(json.dumps({"ranges": ranges.to_json(), "gate": gate,
                                                           "sequences": len(seqs)}, indent=1))


def _ranges(LC, job, seed):
    return LC.Ranges.from_json(json.loads(job.path(f"ranges-{seed}.json").read_text())["ranges"])


def step_palettes(job, K):
    """k-means palettes for every projection and the head, at residual scale K."""
    import torch
    from .techniques import gptq as G
    LC, _, _ = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    ranges = LC.Ranges(K=int(K), norm_scale=[1.0] * (2 * ck.layers + 1), norm_max=[], linear_out_max=[])
    out = {}
    for i in range(ck.layers):
        t0 = time.time()
        layer = LC.build_layer(ck, i, rot, ranges)
        for p in LC.PROJECTIONS:
            out[f"{i}.{p}"] = G.kmeans_palettes(LC.weight_of(getattr(layer, p)), r.palette_group, seed=r.palette_seed)
        print(f"palettes, layer {i}: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    out["head"] = G.kmeans_palettes(LC.head_weight(ck, rot), r.palette_group, seed=r.palette_seed)
    print(f"palettes, head: {time.time() - t0:.0f} s", flush=True)
    torch.save(out, job.path(f"palettes-{r.rotation_seed}-K{K}.pt"))


def _keep_fp16(job, i, p):
    return f"{i}.{p}" in job.recipe.fp16_projections


def step_gptq(job, seed):
    """GPTQ of every projection and the head onto the palettes, on the draw."""
    import torch
    from .techniques import gptq as G
    LC, _, fixture = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    ranges = _ranges(LC, job, seed)
    palettes = torch.load(job.path(f"palettes-{r.rotation_seed}-K{ranges.K}.pt"))
    out = job.path(f"gptq-{seed}")
    out.mkdir(parents=True, exist_ok=True)
    seqs = draw(fixture, seed, r.tokens, r.sequence, _separator(job))
    table = LC.embedding_table(ck, rot, ranges.K)
    hs = [LC.embed(table, ids) for ids in seqs]
    del table
    inputs = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}
    stats = {}
    for i in range(ck.layers):
        t0 = time.time()
        layer = LC.build_layer(ck, i, rot, ranges)
        hess = {k: G.Hessian() for k in set(inputs.values())}
        convs = {p: getattr(layer, p) for p in LC.PROJECTIONS}
        for p in ("q", "o", "gate", "down"):              # one recorder per distinct input
            setattr(layer, p, G.Recorder(convs[p], hess[inputs[p]]))
        for h in hs:
            LC.run_layer(layer, h, ck)
        for p in LC.PROJECTIONS:
            setattr(layer, p, convs[p])
        res, st = {}, {}
        for p in LC.PROJECTIONS:
            W = LC.weight_of(convs[p])
            if _keep_fp16(job, i, p):
                convs[p].weight.copy_(G.f16(W).view_as(convs[p].weight))
                continue
            H = hess[inputs[p]].mean()
            grid = G.Palette(W, palettes[f"{i}.{p}"], r.palette_group)
            codes, pal = G.gptq(W, H, grid, r.damp, act_order=r.act_order)
            Wq = G.dequantize(codes, pal, r.palette_group)
            st[p] = {"gptq": G.output_error(Wq, W, H), "rtn": G.output_error(grid.round_to_nearest(W), W, H)}
            res[p] = (codes, pal)
            convs[p].weight.copy_(Wq.view_as(convs[p].weight))
        torch.save(res, out / f"layer_{i}.pt")
        hs = [LC.run_layer(layer, h, ck) for h in hs]
        stats[i] = st
        print(f"gptq, layer {i}: {time.time() - t0:.0f} s " + " ".join(
            f"{p} {v['gptq']:.4f}/{v['rtn']:.4f}" for p, v in st.items()), flush=True)
        del layer, convs, hess
    t0 = time.time()
    norm = LC.final_norm(ck, rot, ranges)
    H = G.Hessian()
    for h in hs:
        H.add(norm(h))
    W = LC.head_weight(ck, rot)
    grid = G.Palette(W, palettes["head"], r.palette_group)
    codes, pal = G.gptq(W, H.mean(), grid, r.damp, act_order=r.act_order)
    torch.save({"head": (codes, pal)}, out / "head.pt")
    print(f"gptq, head: {time.time() - t0:.0f} s", flush=True)
    (out / "meta.json").write_text(json.dumps({"seed": seed, "tokens": sum(map(len, seqs)), "stats": stats},
                                              indent=1))


def quantized_layer(LC, G, job, ck, rot, ranges, seed, i, weights=None):
    """Layer i with its GPTQ weights."""
    import torch
    layer = LC.build_layer(ck, i, rot, ranges)
    weights = weights if weights is not None else torch.load(job.path(f"gptq-{seed}", f"layer_{i}.pt"))
    for p in LC.PROJECTIONS:
        c = getattr(layer, p)
        if p in weights:
            c.weight.copy_(G.dequantize(*weights[p], job.recipe.palette_group).view_as(c.weight))
        else:                                  # an fp16 projection
            c.weight.copy_(G.f16(c.weight))
    return layer


def quantized_head(G, job, seed):
    import torch
    codes, pal = torch.load(job.path(f"gptq-{seed}", "head.pt"))["head"]
    return G.dequantize(codes, pal, job.recipe.palette_group)


def step_screen(job, seed):
    """The quantised model, fp32, teacher-forced over the evaluation texts."""
    from .techniques import gptq as G
    LC, _, fixture = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    ranges = _ranges(LC, job, seed)
    texts = fixture.teacher_forced
    hs = LC.stream(ck, [ids for _, ids in texts],
                   lambda i: quantized_layer(LC, G, job, ck, rot, ranges, seed, i),
                   LC.embedding_table(ck, rot, ranges.K))
    norm, head = LC.final_norm(ck, rot, ranges), quantized_head(G, job, seed)
    got = {name: LC.logits(norm, head, h, (FIRST_GRADED - 1, len(ids))).numpy() for (name, ids), h in zip(texts, hs)}
    g = grade(got, load_refs(job))
    print(f"screen, draw {seed}: {describe(g)}", flush=True)
    job.path(f"screen-{seed}.json").write_text(json.dumps(g, indent=1))


def step_convert(job, index):
    """Chunk `index` of the first draw: trace, convert with state, palettize,
    compile into the stage directory."""
    import coremltools as ct
    import coremltools.optimize as cto
    import torch
    from .techniques import gptq as G
    LC, m, _ = _setup(job)
    r, d = job.recipe, m["decoder"]
    seed = r.draw_seed
    index = int(index)
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    ranges = _ranges(LC, job, seed)
    lo, hi = cuts_of(d["layers"])[index]
    last = hi == ck.layers
    T, C, D, Dh = d["tokens_per_call"], d["cache_capacity"], ck.hidden, ck.head_dim
    layers = [quantized_layer(LC, G, job, ck, rot, ranges, seed, i) for i in range(lo, hi)]
    chunk = LC.Chunk(layers, ck, T, C, LC.final_norm(ck, rot, ranges) if last else None,
                     quantized_head(G, job, seed) if last else None, r.head_split).eval()
    t0 = time.time()
    example = [torch.zeros(1, D, 1, T), torch.tensor([0], dtype=torch.int32), LC.mask_for(0, T, C),
               torch.zeros(T, Dh), torch.zeros(T, Dh)]
    inputs = [ct.TensorType("hidden_in", (1, D, 1, T), np.float16), ct.TensorType("pos", (1,), np.int32),
              ct.TensorType("mask", (1, 1, T, C), np.float16), ct.TensorType("cos", (T, Dh), np.float16),
              ct.TensorType("sin", (T, Dh), np.float16)]
    if last:
        example.append(torch.zeros(T))
        inputs.append(ct.TensorType("sel", (T,), np.float16))
    traced = torch.jit.trace(chunk, tuple(example), check_trace=False)
    shape = tuple(chunk.k_cache.shape)
    ml = ct.convert(traced, inputs=inputs,
                    outputs=[ct.TensorType(name="logits" if last else "hidden_out", dtype=np.float16)],
                    states=[ct.StateType(ct.TensorType(shape=shape, dtype=np.float16), name="k_cache"),
                            ct.StateType(ct.TensorType(shape=shape, dtype=np.float16), name="v_cache")],
                    minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
    del traced, chunk, layers
    # the const names the trace gives each projection: layers_<index in chunk>_<proj>_weight_to_fp16
    consts = {op.name for fn in ml._mil_program.functions.values() for op in fn.operations if op.op_type == "const"}
    skip = {}
    for name in r.fp16_projections:
        i, p = name.split(".")
        if lo <= int(i) < hi:
            const = f"layers_{int(i) - lo}_{p}_weight_to_fp16"
            if const not in consts:
                raise core.GateFailure(f"fp16 projection {name}: no const {const} in chunk {index}")
            skip[const] = None
    config = cto.coreml.OpPalettizerConfig(mode="unique", granularity="per_grouped_channel",
                                           group_size=r.palette_group, weight_threshold=2048)
    ml = cto.coreml.palettize_weights(ml, config=cto.coreml.OptimizationConfig(global_config=config,
                                                                               op_name_configs=skip))
    # the palettizer skips a weight it can't store with a warning only (more than 256 values in a
    # group): every projection and head weight must have become a palette but those kept in fp16
    plain = {op.name for fn in ml._mil_program.functions.values() for op in fn.operations if op.op_type == "const"}
    projection = re.compile(r"(layers_\d+_(q|k|v|o|gate|up|down)|heads_\d+)_weight_to_fp16")
    weights = {c for c in consts if projection.fullmatch(c)}
    if len(weights) != 7 * (hi - lo) + (r.head_split if last else 0):
        raise core.GateFailure(f"chunk {index}: expected every projection's weight as a const, found {len(weights)}")
    left = sorted((weights & plain) - set(skip))
    if left:
        raise core.GateFailure(f"chunk {index}: weights not stored as palettes: {left[:5]}")
    ops = sorted(core.mil_op_types(ml))
    stage = job.path("stage")
    stage.mkdir(parents=True, exist_ok=True)
    pkg = stage / "packages" / f"decoder.{index}.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(pkg))
    del ml
    tmp = stage / f"compile.{index}"
    if tmp.exists():
        shutil.rmtree(tmp)
    compiled = core.compile_mlmodelc(pkg, tmp)
    dest = stage / f"decoder.{index}.mlmodelc"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(compiled), str(dest))
    shutil.rmtree(tmp)
    if not job.keep_packages:
        shutil.rmtree(pkg)
    (stage / f"decoder.{index}.json").write_text(json.dumps({"layers": [lo, hi], "head": last, "ops": ops,
                                                             "seconds": round(time.time() - t0)}, indent=1))
    print(f"chunk {index} (layers {lo}-{hi}{', head' if last else ''}): {time.time() - t0:.0f} s", flush=True)


STEPS = {"references": step_references, "ranges": step_ranges, "palettes": step_palettes, "gptq": step_gptq,
         "screen": step_screen, "convert": step_convert}


def main(argv):
    if len(argv) < 3 or argv[0] != "step" or argv[1] not in STEPS:
        raise SystemExit(f"usage: python -m sidekick_convert.causal_lm step {{{','.join(STEPS)}}} <job.json> [arg]")
    job = Job.load(argv[2])
    args = [int(a) for a in argv[3:]]
    STEPS[argv[1]](job, *args)


if __name__ == "__main__":
    main(sys.argv[1:])
