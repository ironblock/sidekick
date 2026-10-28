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
   (backbones.llama_causal). The stability draw reads only documents the
   build's draw didn't.
4. **Palettes**: k-means per group of output channels, once per rotation.
5. **GPTQ** onto those palettes, per draw (techniques.gptq), layer by
   layer, each layer calibrated on the outputs of the layers already
   quantised.
6. **The rewrite gate and the PyTorch screen**, per draw: the rotated,
   rewritten model in fp32 must reproduce the checkpoint's logits, and the
   quantised model, teacher-forced in
   fp32 over the evaluation texts, graded as an ANE build is graded. Only a
   recipe whose screen passes is compiled, since every ANE compile leaves
   about a copy of the weights in a system cache nobody can delete. With a
   second (stability) draw, both draws must grade alike.
7. **Conversion** of the build's draw, one chunk per process: trace,
   coremltools with the KV cache as state, the weights stored as 8-bit
   palettes (`unique` mode, so GPTQ's values survive exactly), compiled.
8. **Gates**: no forbidden op (the fused attention op, D25; the native
   silu, which StableSilu replaced), each chunk under the ANE weight limit
   (D32), and, unless skipped, the compute plan with every operation of
   every chunk on the ANE (one plan read per new program, which compiles
   it; recorded by the program's hash).
9. **Install**, atomically: the chunks (unchanged ones moved, not copied,
   to keep their ANE compile), the host's embedding and RoPE tables, the
   tokenizer, the chat template, and chat.toml with its [build] and
   [placement] tables.

Each heavy step runs in its own process (`python -m sidekick_convert.causal_lm
step ...`), so its memory is returned when it ends, and leaves its result in
the job's work directory with the key of the inputs it was made from
(`keys()`); a later run reuses a result only when its key still matches. The
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

KL_MEAN = {"A": 1e-4, "B": 2.5e-3, "C": 2e-2}
"""Mean KL(fp32 || build) per position, at most, for each letter (D54;
docs/CONVERTING.md, "Grading"). Applied to every evaluation family."""

KL_P99 = {"A": 1e-3, "B": 3e-2, "C": 0.25}
"""The 99th percentile of KL(fp32 || build) over a family's positions, at most."""

TOP1 = {"A": 0.995, "B": 0.98, "C": 0.95}
"""Top-1 agreement with fp32 for each letter, at least: a floor beside the
KL bands. The ideal-fp16 ceiling of MiniCPM5-2B agrees at 99.7-99.9%, so A
allows about that."""

WIDE_MARGIN = math.log(2.0)
"""A top-1 flip where fp32 prefers its top token by at least this much in
log-probability (twice as likely as the runner-up). Reported, with the
first-percentile logit cosine, as the encoders' and classifiers' rubric
(D26) would judge them; for chat models they don't decide the grade (D54):
a chat model samples a distribution, and a flip at a position fp32 itself
is unsure of moves with the calibration draw."""

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
    chat_sequences: int = 8          # of a draw's sequences, how many are long chat requests (draw())
    K: int = None                    # the residual scale; None: calibrated
    palette_group: int = 16
    palette_seed: int = 0
    palette_bits: int = 8            # entries per palette: 2 ** palette_bits
    rounding: str = "gptq"           # "gptq", or "nearest" (each weight to its nearest entry; a reference only)
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
    plan: bool = True                # read the compute plans (one ANE compile per new chunk program)
    screen_only: bool = False
    keep_packages: bool = False
    inputs: dict = None              # what the intermediates depend on (inputs()): hashes and versions

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
                                          return_dict=False, **self.template_kwargs)
        ids = list(ids)
        if not all(isinstance(i, int) for i in ids):
            raise TypeError(f"apply_chat_template returned {type(ids[0]).__name__}s, not token ids")
        return ids

    SENTINEL = "\u2063SIDEKICK-DOCUMENTS\u2063"

    def frame(self, e):
        """A request frame's token ids around the documents a draw puts in
        it: (prefix, tail). The prefix is the template's rendering up to the
        end of the instruction and a blank line, where the documents go;
        the tail closes the user turn and opens the assistant's, as a
        served request ends."""
        messages = [dict(m) for m in e["messages"]]
        messages[-1]["content"] += "\n\n" + self.SENTINEL
        text = self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           **self.template_kwargs)
        head, tail = text.split(self.SENTINEL)
        return {"prefix": self.text(head), "tail": self.text(tail)}

    def encode_entry(self, e):
        if e.get("kind") == "frame":
            return self.frame(e)
        if "messages" in e:
            return self.chat(e["messages"], e["messages"][-1]["role"] == "user")
        ids = self.text(e["text"])
        return ids[:e["tokens"]] if "tokens" in e else ids


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass
class Text:
    name: str
    ids: list
    rows: list                 # the positions graded
    family: str                # "repository" or "prose": grades are also reported per family


def wikipedia_title(source):
    """The normalized article title of a Wikipedia permalink, or None."""
    import urllib.parse
    if "wikipedia.org" not in source:
        return None
    title = urllib.parse.parse_qs(urllib.parse.urlparse(source).query)["title"][0]
    return urllib.parse.unquote(title).replace("_", " ").strip().casefold()


@dataclasses.dataclass
class Fixture:
    calibration: list          # entries
    evaluation: list
    ids: dict                  # {"calibration": {id: ids}, "evaluation": {id: ids}}
    calibration_sha256: str
    tokens_file: Path

    @property
    def documents(self):
        """The calibration entries a draw packs (every kind but frames)."""
        return [e for e in self.calibration if e["kind"] != "frame"]

    @property
    def frames(self):
        """The request frames, as {"prefix": ids, "tail": ids}, in fixture order."""
        return [self.ids["calibration"][e["id"]] for e in self.calibration if e["kind"] == "frame"]

    @property
    def teacher_forced(self):
        """Every teacher-forced text: its name, ids, graded positions
        (FIRST_GRADED - 1 to the end, every `stride`-th) and family."""
        out = []
        for e in self.evaluation:
            if e["kind"] == "teacher-forced":
                ids = self.ids["evaluation"][e["id"]]
                out.append(Text(e["id"], ids, list(range(FIRST_GRADED - 1, len(ids), e.get("stride", 1))),
                                e.get("family", "repository")))
        return out

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
    def calibration_ids(e):
        ids = rec["calibration"][e["id"]]
        return ids["prefix"] + ids["tail"] if e["kind"] == "frame" else ids
    shared = [e["id"] for e in cal if ngrams(calibration_ids(e)) & graded]
    if shared:
        raise core.GateFailure(f"calibration documents share {SHARED_RUN}-token runs with the evaluation set: "
                               f"{shared[:5]}")
    bench = {wikipedia_title(e.get("source", "")) for f in BENCHMARK.glob("*.jsonl") for e in read_jsonl(f)}
    bench.discard(None)
    calibrated = {wikipedia_title(e["source"]) for e in cal} - {None}
    clash = [e["id"] for e in cal if wikipedia_title(e["source"]) in bench]
    if clash:
        raise core.GateFailure(f"calibration articles the chat benchmark uses: {clash}")
    clash = [e["id"] for e in ev if wikipedia_title(e["source"]) in bench | calibrated]
    if clash:
        raise core.GateFailure(f"evaluation articles the calibration or the chat benchmark uses: {clash}")
    docs = [e for e in cal if e["kind"] != "frame"]
    n = sum(len(rec["calibration"][e["id"]]) for e in docs)
    report(f"fixture: {len(docs)} calibration documents ({n} tokens), {len(cal) - len(docs)} request frames, "
           f"{len(ev)} evaluation entries; disjoint")
    return Fixture(cal, ev, {"calibration": rec["calibration"], "evaluation": rec["evaluation"]},
                   rec["calibration_sha256"], path)


def framed(n, k):
    """Which of n sequences are chat requests: k of them, spread evenly."""
    if not 0 <= k <= n:
        raise ValueError(f"{k} chat sequences of {n}")
    return sorted({i * n // k for i in range(k)}) if k else []


def draw(fixture, seed, tokens, sequence, separator, exclude=frozenset(), chat_sequences=0):
    """A calibration draw: the documents in a seeded order (numpy's
    default_rng(seed).permutation), less those in `exclude`, packed into
    `tokens / sequence` sequences of exactly `sequence` tokens, each
    starting at a document: documents are appended, `separator` between
    them, until the sequence is full, and the rest of the last one is
    dropped.

    `chat_sequences` of them (framed(), spread evenly) are long chat
    requests instead: a request frame's prefix (a system prompt and an
    instruction, in the chat template), then the draw's next documents up
    to the length that leaves room for the frame's tail, which closes the
    user turn and opens the assistant's. Frames are taken in a seeded order
    (default_rng([seed, 1])), and a chat-request document is never put
    inside one: it waits for the next plain sequence. Returns the sequences
    and the ids of the documents they read."""
    if tokens % sequence:
        raise ValueError(f"{tokens} calibration tokens aren't a whole number of {sequence}-token sequences")
    docs = fixture.documents
    kind = {e["id"]: e["kind"] for e in docs}
    names = [e["id"] for e in docs]
    pending = [names[j] for j in np.random.default_rng(seed).permutation(len(names)) if names[j] not in exclude]
    n = tokens // sequence
    chat = set(framed(n, chat_sequences))
    frames = fixture.frames
    if chat and not frames:
        raise ValueError("chat sequences asked for, and the fixture has no request frames")
    frame_order = list(np.random.default_rng([seed, 1]).permutation(len(frames))) if frames else []
    out, used = [], set()
    for i in range(n):
        frame = frames[frame_order[len([c for c in chat if c < i]) % len(frames)]] if i in chat else None
        budget = sequence - (len(frame["prefix"]) + len(frame["tail"]) if frame else 0)
        body, kept = [], []
        while len(body) < budget:
            if not pending:
                raise ValueError(f"the fixture holds too few tokens for {n} sequences of {sequence}"
                                 + (f" outside the {len(exclude)} documents excluded" if exclude else ""))
            d = pending.pop(0)
            if frame and kind[d] == "chat":
                kept.append(d)
                continue
            body = body + (separator if body else []) + fixture.ids["calibration"][d]
            used.add(d)
        pending = kept + pending
        out.append(frame["prefix"] + body[:budget] + frame["tail"] if frame else body[:sequence])
    return out, used


def job_draw(job, fixture, seed):
    """The draw a job calibrates on with `seed`: the build's draw, or the
    stability draw, which reads only documents the build's draw didn't."""
    r, sep = job.recipe, _separator(job)
    if seed == r.draw_seed:
        return draw(fixture, seed, r.tokens, r.sequence, sep, chat_sequences=r.chat_sequences)[0]
    _, used = draw(fixture, r.draw_seed, r.tokens, r.sequence, sep, chat_sequences=r.chat_sequences)
    return draw(fixture, seed, r.tokens, r.sequence, sep, exclude=used, chat_sequences=r.chat_sequences)[0]


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
             f'weights = "palette{r.palette_bits}"',
             f"palettes = {{ group = {r.palette_group}, bits = {r.palette_bits}, kmeans_seed = {r.palette_seed} }}",
             f"gptq = {{ act_order = {str(r.act_order).lower()}, damp = {r.damp}, threads = {r.threads} }}",
             f"rotation = {{ r1 = true, r2 = {str(r.r2).lower()}, seed = {r.rotation_seed} }}",
             f'calibration = {{ set = "{CALIBRATION.relative_to(REPO)}", sha256 = "{fixture.calibration_sha256}", '
             f"tokens = {r.tokens}, sequence = {r.sequence}, chat_sequences = {r.chat_sequences}, "
             f"seed = {r.draw_seed} }}",
             f"residual_scale = {ranges['K']}",
             f"head_split = {r.head_split}",
             f"fp16_projections = [{fp16}]",
             "versions = { " + ", ".join(f'{k} = "{v}"' for k, v in sorted(job.inputs["versions"].items())) + " }"]
    return "\n".join(lines) + "\n" + (placement or "")


def versions():
    """The library versions a build's bytes depend on, recorded in [build]."""
    import importlib.metadata as md
    out = {}
    for name, dist in (("torch", "torch"), ("transformers", "transformers"), ("scikit_learn", "scikit-learn"),
                       ("coremltools", "coremltools"), ("numpy", "numpy"), ("safetensors", "safetensors")):
        try:
            out[name] = md.version(dist)
        except md.PackageNotFoundError:
            out[name] = "absent"
    return out


def inputs(job, m):
    """What the intermediates depend on besides the recipe: the checkpoint,
    the fixtures, the manifest and the library versions."""
    src = Path(job.src)
    return {"revision": _tok.snapshot_revision(src), "config_sha256": sha256(src / "config.json"),
            "tokenizer_sha256": sha256(src / "tokenizer.json"), "calibration_sha256": sha256(CALIBRATION),
            "evaluation_sha256": sha256(EVALUATION), "manifest_sha256": sha256(manifest_path(job.model_id)),
            "versions": versions()}


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


def grade(got, refs, families=None):
    """Teacher-forced grade of `got` ({text: logits rows}) against the fp32
    `refs` (the same rows), by the chat rubric (D54): per position
    KL(fp32 || got), and whether the top-1 token agrees. A group of
    positions earns the worst of its mean KL's letter (KL_MEAN), its p99
    KL's (KL_P99) and its top-1 agreement's (TOP1); with `families`
    ({text: family}) each family is graded on its own texts, under
    "families", and the grade is the worst family's, so a weak family
    can't hide in the pooled numbers. Non-finite logits are F.

    Reported beside it, not deciding: the logits' cosine per position (its
    first percentile, mean and worst), the wide flips, and the letter the
    encoders' rubric would give (`cosine_grade`: the first-percentile
    cosine on the parity suite's scale and top-1, capped at C by a wide
    flip)."""
    g = _grade(got, refs)
    if families:
        g["families"] = {}
        for fam in sorted(set(families.values())):
            names = [n for n in refs if families.get(n) == fam]
            sub = _grade({n: got[n] for n in names}, {n: refs[n] for n in names})
            g["families"][fam] = {k: v for k, v in sub.items() if k not in ("flips", "texts")}
        g["pooled_grade"] = g["grade"]
        g["grade"] = worse(*(f["grade"] for f in g["families"].values()))
    return g


def _grade(got, refs):
    cos, agree, flips, texts, kls = [], [], [], {}, []
    for name in refs:
        a = np.asarray(got[name], dtype=np.float64)
        b = np.asarray(refs[name], dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"{name}: rows {a.shape} != reference {b.shape}")
        finite = np.isfinite(a).all(1)
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        c[~finite] = np.nan
        t = a.argmax(1) == b.argmax(1)
        lb = b - np.logaddexp.reduce(b, axis=1, keepdims=True)
        la = a - np.logaddexp.reduce(a, axis=1, keepdims=True)
        kl_text = (np.exp(lb) * (lb - la)).sum(1)
        kls += kl_text.tolist()
        margin = softmax_margin(b)
        for j in np.nonzero(~t)[0]:
            flips.append({"text": name, "row": int(j), "fp32_margin": float(margin[j])})
        cos += c.tolist()
        agree += t.tolist()
        texts[name] = {"top1": round(float(t.mean()), 4), "cos_min": round(float(np.min(c)), 6),
                       "kl_mean": float(np.mean(kl_text)), "kl_p99": float(np.percentile(kl_text, 99)),
                       "kl_max": float(np.max(kl_text))}
    c = np.array(cos)
    nonfinite = not np.isfinite(c).all()
    top1 = float(np.mean(agree))
    p1 = float("nan") if nonfinite else float(np.percentile(c, 1))
    wide = [f for f in flips if f["fp32_margin"] >= WIDE_MARGIN]
    by_cos, by_top1 = letter(p1, cosine_scale()), letter(top1, TOP1)
    cosine_grade = "F" if nonfinite else worse(by_cos, by_top1)
    if wide and cosine_grade in ("A", "B"):
        cosine_grade = "C"
    kl = np.array(kls)
    kl_mean = float("nan") if nonfinite else float(np.mean(kl))
    kl_p99 = float("nan") if nonfinite else float(np.percentile(kl, 99))
    by_mean, by_p99 = letter(kl_mean, KL_MEAN, higher=False), letter(kl_p99, KL_P99, higher=False)
    g = "F" if nonfinite else worse(by_mean, by_p99, by_top1)
    return {"positions": len(c), "top1": round(top1, 6), "kl_mean": kl_mean, "kl_p99": kl_p99,
            "kl_max": float(np.max(kl)) if not nonfinite else float("nan"),
            "grade": g, "grade_kl_mean": by_mean, "grade_kl_p99": by_p99, "grade_top1": by_top1,
            "cos_mean": float(np.nanmean(c)), "cos_p1": p1,
            "cos_min": float(np.min(c)) if not nonfinite else float("nan"),
            "cosine_grade": cosine_grade, "grade_cos_p1": by_cos, "wide_flips": len(wide),
            "flips": flips, "texts": texts}


def describe(g):
    pooled = f", pooled {g['pooled_grade']}" if "pooled_grade" in g else ""
    return (f"grade {g['grade']}{pooled} (KL mean {g['kl_mean']:.2e} {g['grade_kl_mean']}, p99 {g['kl_p99']:.2e} "
            f"{g['grade_kl_p99']}, top-1 {g['top1']:.2%} {g['grade_top1']}; KL max {g['kl_max']:.3f}) over "
            f"{g['positions']} positions; reported: cosine p1 {g['cos_p1']:.5f}, worst {g['cos_min']:.5f}, "
            f"{g['wide_flips']} wide flips (the encoders' rubric: {g['cosine_grade']})")


def load_refs(job, kind="fp32"):
    d = job.path("references")
    meta = json.loads((d / "meta.json").read_text())
    return {name: np.load(d / f"{kind}_{name}.npy") for name in meta["texts"]}


def families(fixture):
    return {t.name: t.family for t in fixture.teacher_forced}


# ---------------------------------------------------------------- the work directory's cache

def _key_path(path):
    return Path(str(path) + ".key.json")


def fresh(path, key):
    """Whether the intermediate at `path` exists and was made from `key`."""
    k = _key_path(path)
    return Path(path).exists() and k.exists() and json.loads(k.read_text()) == key


def _discard(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    _key_path(path).unlink(missing_ok=True)


def _stamp(path, key):
    _key_path(path).write_text(json.dumps(key, indent=1, sort_keys=True))


def keys(job, seed=None):
    """What each intermediate depends on. An intermediate is reused only
    when its recorded key equals the current one, so a changed fixture,
    tokenizer, checkpoint, manifest, library or recipe recomputes what it
    touches and nothing else."""
    r, i = job.recipe, job.inputs
    ck = {"revision": i["revision"], "config_sha256": i["config_sha256"]}
    references = {**ck, "evaluation_sha256": i["evaluation_sha256"], "tokenizer_sha256": i["tokenizer_sha256"],
                  "first_graded": FIRST_GRADED, "greedy_tokens": GREEDY_TOKENS,
                  "torch": i["versions"]["torch"], "transformers": i["versions"]["transformers"]}
    rotation = {**ck, "rotation_seed": r.rotation_seed, "r2": r.r2, "torch": i["versions"]["torch"],
                "threads": r.threads}
    out = {"references": references}
    if seed is not None:
        draw_key = {**rotation, "calibration_sha256": i["calibration_sha256"],
                    "tokenizer_sha256": i["tokenizer_sha256"], "tokens": r.tokens, "sequence": r.sequence,
                    "chat_sequences": r.chat_sequences,
                    "seed": seed, "K": r.K, "disjoint_from": None if seed == r.draw_seed else r.draw_seed}
        out["ranges"] = draw_key
        out["gptq"] = {**draw_key, "palette_group": r.palette_group, "palette_seed": r.palette_seed,
                       "palette_bits": r.palette_bits, "rounding": r.rounding,
                       "scikit_learn": i["versions"]["scikit_learn"], "act_order": r.act_order, "damp": r.damp,
                       "fp16_projections": list(r.fp16_projections)}
        out["screen"] = {"gptq": out["gptq"], "references": references}
        out["stage"] = {"gptq": out["gptq"], "manifest_sha256": i["manifest_sha256"], "head_split": r.head_split,
                        "coremltools": i["versions"]["coremltools"]}
    out["palettes"] = lambda K: {**rotation, "K": K, "palette_group": r.palette_group, "palette_bits": r.palette_bits,
                                 "palette_seed": r.palette_seed, "scikit_learn": i["versions"]["scikit_learn"]}
    return out


def _cached(job, path, key, step, *args):
    """Run `step` unless `path` is fresh for `key`; then record the key."""
    if fresh(path, key):
        return False
    _discard(path)
    _child(job, step, *args)
    _stamp(path, key)
    return True


def program_hash(compiled):
    """sha256 over a compiled program's model.mil and weight files: what its
    compute plan depends on (coremldata.bin changes with every compile)."""
    h = hashlib.sha256()
    for f in sorted(Path(compiled).rglob("*")):
        if f.is_file() and (f.name == "model.mil" or "weights" in f.relative_to(compiled).parts):
            h.update(str(f.relative_to(compiled)).encode())
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
    return h.hexdigest()


def same_tree(a, b):
    """Whether two directories hold the same files, byte for byte."""
    import filecmp
    fa = sorted(p.relative_to(a) for p in Path(a).rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in Path(b).rglob("*") if p.is_file())
    return fa == fb and all(filecmp.cmp(Path(a) / p, Path(b) / p, shallow=False) for p in fa)


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
    d, r = m["decoder"], job.recipe
    if d["vocab_size"] % (r.head_split * r.palette_group):
        raise core.GateFailure(f"the head's {d['vocab_size']} rows don't split into {r.head_split} convolutions of "
                               f"whole {r.palette_group}-row palette groups")
    long = [t.name for t in fixture.teacher_forced if len(t.ids) > d["context_length"]]
    if long:
        raise core.GateFailure(f"evaluation texts longer than the context ({d['context_length']}): {long}")
    job.inputs = inputs(job, m)
    job.save()
    report = {"recipe": dataclasses.asdict(r), "inputs": job.inputs}
    _cached(job, job.path("references"), keys(job)["references"], "references")
    screens = {}
    for seed in r.draws():
        k = keys(job, seed)
        _cached(job, job.path(f"ranges-{seed}.json"), k["ranges"], "ranges", seed)
        K = json.loads(job.path(f"ranges-{seed}.json").read_text())["ranges"]["K"]
        _cached(job, palettes_path(job, K), keys(job)["palettes"](K), "palettes", K)
        _cached(job, job.path(f"gptq-{seed}"), k["gptq"], "gptq", seed)
        _cached(job, job.path(f"screen-{seed}.json"), k["screen"], "screen", seed)
        screens[seed] = json.loads(job.path(f"screen-{seed}.json").read_text())
        report[f"rewrite_gate_{seed}"] = screens[seed]["rewrite_gate"]
        print(f"screen, draw {seed}: {describe(screens[seed])}", flush=True)
    report["screens"] = {str(s): {k: v for k, v in g.items() if k != "flips"} for s, g in screens.items()}
    letters = {s: g["grade"] for s, g in screens.items()}
    stable = len(set(letters.values())) == 1
    report["stable"] = stable
    report["screen_grade"] = worse(*letters.values())
    if not stable:
        print(f"the draws grade differently: {letters}; the build's grade is the lower, {report['screen_grade']}")
    job.path("report.json").write_text(json.dumps(report, indent=1))
    if worse(report["screen_grade"], r.min_grade) != r.min_grade:
        raise core.GateFailure(f"the PyTorch screen grades {report['screen_grade']}, below the recipe's "
                               f"{r.min_grade}: not compiling")
    if job.screen_only:
        return report
    core.require_coremltools()
    stage = job.path(f"stage-{r.draw_seed}")
    if not fresh(stage, keys(job, r.draw_seed)["stage"]):
        _discard(stage)
        stage.mkdir(parents=True)
        _stamp(stage, keys(job, r.draw_seed)["stage"])
    for i in range(d["chunks"]):
        if not (stage / f"decoder.{i}.mlmodelc").exists():
            _child(job, "convert", i)
    from . import plan
    chunks, summaries = [], []
    for i in range(d["chunks"]):
        c = stage / f"decoder.{i}.mlmodelc"
        info = json.loads((stage / f"decoder.{i}.json").read_text())
        if info["draw"] != r.draw_seed:
            raise core.GateFailure(f"staged chunk {i} is draw {info['draw']}'s, not {r.draw_seed}'s")
        bad = sorted(set(info["ops"]) & FORBID_OPS)
        if bad:
            raise core.GateFailure(f"chunk {i} contains forbidden ops {bad}")
        size = plan.weights_bytes(c)
        if size > plan.MAX_ANE_PROGRAM_WEIGHT_BYTES:
            raise core.GateFailure(plan.over_cap(job.model_id, "decoder", size, i, True))
        chunks.append({"weight_bytes": size, **info})
    report["chunks"] = chunks
    if job.plan:
        plans = job.path("plans")
        plans.mkdir(exist_ok=True)
        for i in range(d["chunks"]):
            c = stage / f"decoder.{i}.mlmodelc"
            cached = plans / f"{program_hash(c)}.json"
            if cached.exists():          # these exact program bytes were read before: no second ANE compile
                s = json.loads(cached.read_text())
            else:
                s = plan.summarize(plan.read(c))
                s["units"] = "CPU_AND_NE"
                cached.write_text(json.dumps(s))
            if s["cpu"] or s["gpu"]:
                raise core.GateFailure(f"compute plan: chunk {i} has operations off the ANE: {s['off']} "
                                       "(a kv-state-1 decoder's state must stay on the ANE)")
            print(f"chunk {i}: compute plan {s['ane']}/{s['assigned']} operations on the ANE", flush=True)
            summaries.append(s)
        report["plans"] = [{k: s[k] for k in ("ane", "gpu", "cpu", "unassigned", "total")} for s in summaries]
    install(job, m, stage, install_dir, fixture, summaries)
    job.path("report.json").write_text(json.dumps(report, indent=1))
    print(f"installed {install_dir}", flush=True)
    return report


def install(job, m, stage, install_dir, fixture, summaries):
    """Install atomically: the build is assembled in a sibling directory and
    renamed into place, so a failed run never leaves new programs beside an
    old manifest. A chunk whose installed bytes already equal the staged
    ones is moved over, not copied: the ANE's compile of a program is
    cached by its path and file identity, so a copy would cost a second
    compile, and another copy of its weights in a cache nobody can delete
    (docs/design/coreml-cache.md)."""
    tmp = install_dir.parent / f".{install_dir.name}.installing"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    keep = []
    for i in range(m["decoder"]["chunks"]):
        name = f"decoder.{i}.mlmodelc"
        if (install_dir / name).exists() and same_tree(install_dir / name, stage / name):
            keep.append(name)
        else:
            shutil.copytree(stage / name, tmp / name)
    install_host(job, m, tmp, fixture, summaries)
    moved = []
    try:
        for name in keep:
            os.rename(install_dir / name, tmp / name)
            moved.append(name)
    except OSError:
        for name in moved:
            os.rename(tmp / name, install_dir / name)
        raise
    old = install_dir.parent / f".{install_dir.name}.replaced"
    if old.exists():
        shutil.rmtree(old)
    if install_dir.exists():
        os.rename(install_dir, old)
    os.rename(tmp, install_dir)
    if old.exists():
        shutil.rmtree(old)
    if keep:
        print(f"kept {len(keep)} installed chunk(s) whose bytes were unchanged: {', '.join(keep)}", flush=True)


def install_host(job, m, install_dir, fixture, summaries):
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
    """fp32 teacher-forced logits at each text's graded positions, fp32's
    greedy continuations and its last prompt row for each greedy prompt,
    and the fp16 ceiling at the graded positions."""
    import torch
    from transformers import AutoModelForCausalLM
    from . import fp16sim
    _, _, fixture = _setup(job)
    out = job.path("references")
    out.mkdir(parents=True, exist_ok=True)
    hf = AutoModelForCausalLM.from_pretrained(job.src, dtype=torch.float32, attn_implementation="eager").eval()
    texts = fixture.teacher_forced
    for t in texts:
        t0 = time.time()
        keep = torch.tensor(t.rows)          # only the graded rows: a 2,000-row logits tensor is 1 GB
        lg = hf(torch.tensor([t.ids]), logits_to_keep=keep).logits[0].float().numpy()
        np.save(out / f"fp32_{t.name}.npy", lg)
        print(f"fp32 {t.name}: {len(t.ids)} tokens, {len(t.rows)} graded, {time.time() - t0:.1f} s", flush=True)
    meta = {"texts": [t.name for t in texts], "rows": {t.name: t.rows for t in texts},
            "families": {t.name: t.family for t in texts}, "first_graded": FIRST_GRADED,
            "revision": _tok.snapshot_revision(job.src)}
    greedy, prefill = [], []
    for name, ids in fixture.greedy:
        prefill.append(hf(torch.tensor([ids])).logits[0, -1].float().numpy())
        g = hf.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        do_sample=False, max_new_tokens=GREEDY_TOKENS, pad_token_id=hf.config.pad_token_id)
        greedy.append({"name": name, "ids": ids, "new": g[0, len(ids):].tolist()})
        print(f"greedy {name}: {len(greedy[-1]['new'])} tokens", flush=True)
    (out / "greedy.json").write_text(json.dumps(greedy))
    np.save(out / "fp32_prefill.npy", np.stack(prefill))
    for t in texts:
        t0 = time.time()
        try:
            lg = fp16sim.run(hf, torch.tensor([t.ids]), logits_to_keep=torch.tensor(t.rows)).logits[0]
            lg = lg.float().numpy()
        except fp16sim.Fp16Overflow as e:
            print(f"no fp16 ceiling: {e}", flush=True)
            meta["ceiling"] = str(e)
            break
        np.save(out / f"fp16_{t.name}.npy", lg)
        print(f"fp16 ceiling {t.name}: {time.time() - t0:.1f} s", flush=True)
    else:
        meta["ceiling"] = grade(load_refs_from(out, meta, "fp16"), load_refs_from(out, meta, "fp32"),
                                meta["families"])
        print(f"fp16 ceiling: {describe(meta['ceiling'])}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_refs_from(d, meta, kind):
    return {name: np.load(d / f"{kind}_{name}.npy") for name in meta["texts"]}


def step_ranges(job, seed):
    """The draw's range calibration: the residual scale and the norms'
    pre-scales."""
    LC, m, fixture = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    seqs = job_draw(job, fixture, seed)
    ranges = LC.calibrate_ranges(ck, seqs, rot, K=r.K)
    job.path(f"ranges-{seed}.json").write_text(json.dumps({"ranges": ranges.to_json(), "sequences": len(seqs)},
                                                          indent=1))


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
            out[f"{i}.{p}"] = G.kmeans_palettes(LC.weight_of(getattr(layer, p)), r.palette_group,
                                                2 ** r.palette_bits, seed=r.palette_seed)
        print(f"palettes, layer {i}: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    out["head"] = G.kmeans_palettes(LC.head_weight(ck, rot), r.palette_group, 2 ** r.palette_bits,
                                    seed=r.palette_seed)
    print(f"palettes, head: {time.time() - t0:.0f} s", flush=True)
    torch.save(out, palettes_path(job, K))


def palettes_path(job, K):
    """The palettes of one rotation, residual scale and palette size (the
    8-bit ones keep their original name)."""
    r = job.recipe
    bits = "" if r.palette_bits == 8 else f"-{r.palette_bits}bit"
    return job.path(f"palettes-{r.rotation_seed}-K{K}{bits}.pt")


def round_weights(r, W, H, grid):
    """W onto its palettes: GPTQ, or each weight to its nearest entry (a
    deliberately worse reference build)."""
    import torch
    from .techniques import gptq as G
    if r.rounding == "gptq":
        return G.gptq(W, H, grid, r.damp, act_order=r.act_order)
    if r.rounding == "nearest":
        return grid.nearest(W.double()).to(torch.uint8), grid.palettes.float()
    raise ValueError(f"unknown rounding {r.rounding!r}")


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
    palettes = torch.load(palettes_path(job, ranges.K))
    out = job.path(f"gptq-{seed}")
    out.mkdir(parents=True, exist_ok=True)
    seqs = job_draw(job, fixture, seed)
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
            codes, pal = round_weights(r, W, H, grid)
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
    codes, pal = round_weights(r, W, H.mean(), grid)
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
    """The rewrite gate, then the PyTorch screen: the rotated, rescaled model
    in fp32 must reproduce the checkpoint's logits at every graded position
    (worst cosine 0.99999), and the quantised model, teacher-forced in fp32
    over the evaluation texts, is graded as an ANE build is."""
    from .techniques import gptq as G
    LC, _, fixture = _setup(job)
    r = job.recipe
    ck = LC.Checkpoint(job.src)
    rot = LC.Rotation(r.rotation_seed, r.r2).bind(ck)
    ranges = _ranges(LC, job, seed)
    texts = fixture.teacher_forced
    refs = load_refs(job)
    table = LC.embedding_table(ck, rot, ranges.K)
    norm = LC.final_norm(ck, rot, ranges)
    hs = LC.stream(ck, [t.ids for t in texts], lambda i: LC.build_layer(ck, i, rot, ranges), table)
    head = LC.head_weight(ck, rot)
    got = {t.name: LC.logits(norm, head, h, t.rows).numpy() for t, h in zip(texts, hs)}
    g = grade(got, refs)
    gate = {"cos_min": g["cos_min"], "top1": g["top1"],
            "max_abs_logit_diff": max(float(np.abs(got[n] - refs[n]).max()) for n in refs)}
    print(f"rewrite gate, draw {seed}: worst cosine {g['cos_min']:.7f}, top-1 {g['top1']:.2%}, max |dlogit| "
          f"{gate['max_abs_logit_diff']:.2e}", flush=True)
    if not g["cos_min"] >= 0.99999:
        raise core.GateFailure(f"the rewritten model doesn't reproduce the checkpoint: worst cosine {g['cos_min']}")
    del hs, head, got
    hs = LC.stream(ck, [t.ids for t in texts], lambda i: quantized_layer(LC, G, job, ck, rot, ranges, seed, i),
                   table)
    head = quantized_head(G, job, seed)
    got = {t.name: LC.logits(norm, head, h, t.rows).numpy() for t, h in zip(texts, hs)}
    g = grade(got, refs, families(fixture))
    g["rewrite_gate"] = gate
    print(f"screen, draw {seed}: {describe(g)}", flush=True)
    for fam, sub in g["families"].items():
        print(f"  {fam}: {describe(sub)}", flush=True)
    job.path(f"screen-{seed}.json").write_text(json.dumps(g, indent=1))


def step_convert(job, index):
    """Chunk `index` of the build's draw: trace, convert with state,
    palettize, compile into the draw's stage directory."""
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
    stage = job.path(f"stage-{seed}")
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
    (stage / f"decoder.{index}.json").write_text(json.dumps({"draw": seed, "layers": [lo, hi], "head": last, "ops": ops,
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
