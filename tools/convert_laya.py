"""Convert laya-format checkpoints into ANE-resident Core ML classifier artifacts
for sidekick's `POST /v1/classify` (docs/design/classify.md):
convaiinnovations/laya (English, `laya-en`) and its fine-tune
convaiinnovations/laya-typed-decisions (`laya-typed-decisions`).

laya is a decision model: a ModernBERT-large encoder, then a two-layer
transformer head and a scorer that reads a [MASK] marker placed before each
candidate option. One artifact per sequence-length bucket runs the whole
model and returns one logit per marker.

Usage:
    python tools/convert_laya.py <laya-dir> <install-dir> [buckets...] [--time]
    python tools/convert_laya.py --model laya-typed-decisions --laya-code <common.py> \\
        <laya-typed-dir> <install-dir> [buckets...] [--time]

    laya-dir:     local snapshot of convaiinnovations/laya at revision
                  55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851: model.safetensors,
                  rl_common.py, rl_agent_config.json, encoder/config.json,
                  tokenizer/tokenizer.json
    laya-typed-dir: local snapshot of convaiinnovations/laya-typed-decisions
                  at revision 1a793eb568e6718f15941d08f85432581df534e3 (the
                  same files, without rl_common.py)
    --laya-code:  laya-typed-decisions' code: common.py from the `laya`
                  package 0.3.22 (its sdist, unpacked), checked by sha256
    install-dir:  classifier directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/laya-en"
    buckets:      default: the manifest's (laya-en 128 256 512,
                  laya-typed-decisions 128 256 512 1024)

Requires: torch, transformers >= 4.48 (ModernBERT), coremltools, numpy,
safetensors (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

laya's own code (rl_common.py, Apache-2.0, convaiinnovations) is imported
from the snapshot rather than copied; its git blob hash is checked against
the pinned revision, because the token-id fixtures and the Rust port of its
build_sequence must agree with exactly that version. laya-typed-decisions
ships no code: its model and build_sequence are the `laya` package's
common.py (Apache-2.0, convaiinnovations), pinned by sha256. For sidekick's
requests (string labels, no custom noul names) that build_sequence produces
the same tokens as rl_common.py's, and its DecisionModel is the same
computation.

Core ML interface (all int32, docs/design/classify.md):
    input_ids      [1, S]
    attention_mask [1, S]
    marker_pos     [1, KMAX]   position of each option's [MASK]; -1 pads
    qtype          [1]         0 choice, 1 score, 2 noul (rank 1)
    logits         [1, KMAX]   one per option; padded slots are -1e4

Conversion constraints:

A. The encoder is the library's ModernBERT backbone (tools/sidekick_convert,
   docs/CONVERTING.md), as for gte-modernbert: explicit (eager) attention
   (D25), fp16-safe masks, traceable rotate_half, and a check that the built
   encoder matches encoder/config.json (rope theta per layer type).
B. RESIDUAL RANGE REWRITE PINNED AT K = 2 (D25's amendment). The ANE's
   linear op saturates above 2^15. laya's encoder writes up to ~27,500 in
   layer 19's MLP output projection, so K = 1 would leave only ~19%
   headroom under 32,768; the calibration rule would pick K = 1, and the
   design pins K = 2 instead. The converter measures the maxima on its
   gate set and fails if K = 2 leaves less than 1/0.85 headroom.
C. THE HEAD'S ATTENTION IS WRITTEN OUT (the library's laya head,
   heads/laya.py). nn.TransformerEncoderLayer's fast path and the fused
   attention op are never traced: q/k/v projections, matmul -> softmax ->
   matmul, with a finite additive mask for pad keys.
D. INPUTS BUILT IN-GRAPH. marker_pos becomes a [KMAX, S] one-hot by
   comparison with a position constant (a -1 pad matches nothing), and
   qtype becomes a one-hot row that selects laya's question-type
   embedding. No data-dependent gather, so the graph stays static.
E. EXPLICIT GELU (docs/DECISIONS.md D28 amendment). Core ML's native gelu
   op is coarse on the ANE: up to 6e-3 off on [-1, 1] (the GPU: 3e-4), where
   most of the encoder's MLP inputs lie. laya's decisions amplify it. Its
   encoder's first layers are the sensitive ones, and there the native gelu
   made the MLP branch 9-15x less accurate than on the GPU. TwiceGelu
   (techniques.activations) computes x * (1 + erf(x / sqrt 2)) from erf,
   mul and add, 9x closer on [-1, 1]. The factor 2 goes into the weights that consume it: each
   encoder MLP's gate rows and the scorer's output linear. Exact in fp32.
   Keeping the 0.5 out of the graph matters: 0.5 * x * (1 + erf(x / sqrt 2))
   is fused back into the native op. The conversion fails if a gelu op
   survives.
F. BUCKET-INVARIANT SOFTMAX (docs/DECISIONS.md D28 amendment). On the ANE,
   linear, layer_norm, matmul, exp and max give bit-identical results for
   the same real tokens at any sequence length; reduce_sum does not, and
   Core ML's softmax is built on it, so the same input ran differently in
   each bucket (max |dp| 0.027 between buckets). Every attention, the
   encoder's and the head's, computes its softmax as exp(w - rowmax) and
   one matmul against [V | 1] (techniques.attention.matmul_softmax): the
   numerator and denominator come from the same matmul, and the ANE output
   is the same in every bucket, bit for bit. Two CPU traps come with it:
   Core ML's CPU reduce_max over 256 or more elements returns max(x, 0)
   (tools/repro_cpu_reduce_max.py), so the row max is taken in 128-wide
   blocks; and a pad query whose whole sliding window is padding would have
   every key masked, which the CPU turns into NaN, so every query may attend
   to itself (self_attending_pads, exact for real tokens). The cost is
   accuracy on a few inputs (see Measured): Core ML compiles layer 7's MLP,
   where the massive activation forms, less accurately inside this graph.

Gates, per bucket:
- fp32: the wrapper (rewrite, explicit attention, one-hots) reproduces
  laya's own forward, max |dlogit| <= FP32_TOL;
- converted graph: no fused attention op and no native gelu;
- on CPU_AND_NE, the path sidekick serves: argmax agreement with fp32
  wherever fp32's top-2 margin is >= MARGIN, and max raw |dp| <= DP_GATE.
  These gates catch a broken conversion; accuracy is graded by the parity
  suite against laya's ideal-fp16 ceiling. DP_GATE is 0.08 because
  constraint F takes one long noul gate item to 0.065;
- on CPU_ONLY the same numbers are reported, not gated. Core ML's fp16 CPU
  backend is laya's weakest path: on one 512-token noul gate item it moves
  both logits by ~0.46 and flips a 0.74 margin, where the ANE moves them by
  0.04 and the GPU by 0.02. The fp32 gate, not the CPU path, is what proves
  the conversion faithful;
- on both paths: finite logits, padded slots at -1e4, and pad invariance
  (pad ids 0 vs random);
- compute plan: every linear/matmul on the ANE and >= 80% of ops.
Every gate treats NaN as a failure.

Measured (M1 Max, macOS 27.0) with tools/classifier_reference.py and
tools/measure_classifier.py on laya's 2,612-case corpus
(fixtures/classify/laya-en.corpus.toml), against laya's fp32 forward,
before -> after constraint E:
- ANE: argmax agreement 99.81% -> 99.96% over the 2,583 cases with a top-2
  margin of at least 0.05 logits (flips 5 -> 1; the one left is at a 0.057
  margin); raw |dp| max 0.077 -> 0.039, p99 0.030 -> 0.016, mean 0.0036 ->
  0.0019; bucket invariance (max |dp|, every case in its own bucket vs each
  larger one) 0.038 -> 0.027; 19.6 / 37.9 / 106 ms -> 20.8 / 40.0 / 107 ms
  at buckets 128 / 256 / 512, timed interleaved (load average 3-5);
- GPU: 100% both (0 flips); raw |dp| max 0.030 -> 0.025; bucket invariance
  0.018 -> 0.011; latency unchanged (interleaved, within 2%);
- CPU_ONLY: flips 6 -> 16; raw |dp| max 0.211 -> 0.172, p99 0.048 -> 0.053.
  Core ML's CPU erf is a little coarser than its native gelu, and this path
  is laya's least accurate either way;
- the floor: an ideal fp16 engine (fp32 arithmetic, every stored tensor
  rounded to fp16) reaches raw |dp| max 0.026, p99 0.0074 on the same
  corpus, and rounding only the embedding output to fp16 moves |dp| by up
  to 0.0058. laya's decisions resolve finer than fp16 does, so no fp16 path
  holds its max |dp| under 1e-3. The ANE is within 1.5x of that floor at
  max and 2x at p99;
- pad invariance exact on every path; 1,161 of 1,177 operations on the ANE;
- accuracy against the corpus's gold labels (reported, not graded) is the
  same on every path: 52.8% on the ANE and in fp32.

With constraint F, graded by the parity suite against laya's ideal-fp16
ceiling (raw |dp| max 0.0169, p99 0.0081; the D28 amendment):
- ANE: bucket invariance exact (max |dp| 0 between buckets, was 0.027);
  raw |dp| max 0.043, p99 0.016, mean 0.0019; 1 flip (the same 0.055-margin
  case), which caps the grade at C; p99 1.93x the ceiling's (a B by itself);
  19.8 / 39.9 / 106.6 ms at buckets 128 / 256 / 512, unchanged;
- GPU: grade A, p99 0.93x the ceiling's; raw |dp| max 0.037; bucket
  invariance 0.0072, inside the gate (the ceiling's max);
- CPU_ONLY: grade D, p99 6.1x the ceiling's; 13 flips; bucket invariance
  exact;
- the cost of constraint F is a few inputs: raw |dp| max 0.039 -> 0.043 on
  the ANE, with p99 and mean unchanged (0.0156, 0.0019); 1,701 of 1,719
  operations on the ANE at bucket 512.
"""

import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from sidekick_convert import cli, core, tokenizer
from sidekick_convert.backbones import modernbert
from sidekick_convert.calibrate import linear_maxima
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.heads.laya import LayaMarkers, twice_gelu_scorer
from sidekick_convert.techniques import saturation
from sidekick_convert.wrapper import compose

REPO = Path(__file__).resolve().parent.parent
# each laya-format checkpoint: its manifest, pinned revision and code
MODELS = {
    "laya-en": {
        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "code": ("rl_common.py", "git-blob", "d90d564964bcdc77586a257b0acccb5b7b19d6cf"),
    },
    "laya-typed-decisions": {
        "revision": "1a793eb568e6718f15941d08f85432581df534e3",
        "code": ("--laya-code", "sha256", "cb77c34b3b5abfc1f59eb1a73357ad80238df397ffdddcfaf634c01949f89b3f"),
    },
}
KMAX = 32
K_RESIDUAL = 2
PAD_LOGIT = -1e4
FP32_TOL = 2e-3
MARGIN = 0.05       # argmax flips count only above this fp32 top-2 logit margin
DP_GATE = 0.08      # max raw |dp| vs fp32 on CPU_AND_NE (CPU_ONLY is reported); see the gates below

# A compact gate set: states of varied length and content x questions that
# exercise every question type and option count.
_LONG = " ".join(f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
                 for i in range(36))
GATE_STATES = [
    "The quarterly report shows revenue up 12% and margins improving.",
    "Hi, my package never arrived and the tracking page has not updated in a week. Can you help?",
    "def add(a, b):\n    return a + b  # simple helper",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    " ".join(["The quick brown fox jumps over the lazy dog."] * 12),
    _LONG,
    _LONG + " " + _LONG,
]
GATE_QUESTIONS = [
    {"t": "noul", "ins": "The text discusses money, markets or finance.", "crit": None},
    {"t": "noul", "ins": "The text contains source code.",
     "crit": {"true": "yes, there is code in it", "false": "no, there is no code"}},
    {"t": "choice", "ins": "What is the main topic of the text?",
     "crit": {"finance": "money, companies, markets", "shipping": "deliveries and parcels",
              "software": "code and computers", "other": ""}},
    {"t": "score", "ins": "How formal is the writing?",
     "crit": ["very informal", "informal", "neutral", "formal", "very formal"]},
    {"t": "choice", "ins": "Which label fits best?",
     "crit": {f"label_{i}": f"description number {i}" for i in range(KMAX)}},
]


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def manifest_path(model):
    return REPO / "examples" / "classifiers" / model / "classifier.toml"


def load_rl_common(src, model="laya-en", code_path=None):
    """The checkpoint's own code: laya's rl_common.py from the snapshot, or,
    for laya-typed-decisions, the laya package's common.py given by path.
    Either is checked against its pin before it is imported."""
    where, kind, want = MODELS[model]["code"]
    path = src / where if code_path is None else Path(code_path)
    if where.startswith("--") and code_path is None:
        raise SystemExit(f"{model} needs {where} (the laya package's common.py)")
    got = git_blob_sha1(path) if kind == "git-blob" else hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{path} is not {model}'s code at {MODELS[model]['revision'][:7]} "
                         f"({kind} {got}, expected {want})")
    spec = importlib.util.spec_from_file_location("laya_rl_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_decision_model(src, rl, cfg):
    dm = rl.build_model(cfg, encoder_dir=str(src / "encoder"))
    state = {k: v.float() for k, v in load_file(src / "model.safetensors").items()}
    missing, unexpected = dm.load_state_dict(state, strict=False)
    if unexpected or any(m != "temperature" for m in missing):
        raise SystemExit(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    return dm.float().eval()


def gate_items(rl, tok, cfg):
    items = []
    for si, state in enumerate(GATE_STATES):
        for qi, q in enumerate(GATE_QUESTIONS):
            ids, markers = rl.build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            k = len(rl.render_options(q))
            if len(markers) != k:
                raise SystemExit(f"gate item {si}/{qi}: {len(markers)} markers for {k} options")
            items.append({"state": si, "q": qi, "t": q["t"], "k": k, "ids": ids, "markers": markers,
                          "qtype": rl.QTYPES[q["t"]]})
    return items


def reference_logits(dm, items):
    """laya's own forward in fp32, one unpadded sequence at a time."""
    out = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([it["ids"]])
            logits, _ = dm(ids, torch.ones_like(ids), torch.tensor([it["markers"]]),
                           torch.ones((1, it["k"]), dtype=torch.bool), torch.tensor([it["qtype"]]))
            out.append(logits[0].numpy().astype(np.float64))
    return out


def check_manifest(cfg, model):
    """classifier.toml must agree with the checkpoint's config: head_max_len,
    max_len, KMAX and the calibration table (its keys are laya's temp_bucket
    keys). Returns the manifest."""
    path = manifest_path(model)
    man = tomllib.loads(path.read_text())
    if man["classify"]["max_labels"] != KMAX:
        raise SystemExit(f"{path}: max_labels {man['classify']['max_labels']} != KMAX {KMAX}")
    if man["classify"]["laya"]["head_max_len"] != cfg["head_max_len"] or man["max_seq_len"] != cfg["max_len"]:
        raise SystemExit(f"{path}: head_max_len/max_seq_len disagree with rl_agent_config.json")
    if man["source"]["revision"] != MODELS[model]["revision"]:
        raise SystemExit(f"{path}: source revision is not {MODELS[model]['revision']}")
    for key, t in man["classify"]["calibration"].items():
        fitted = cfg["temperature_by_options"].get(key)
        if fitted is None or abs(fitted - t) > 5e-4:
            raise SystemExit(f"{path}: calibration {key} = {t}, the checkpoint fitted {fitted}")
    return man


def cases_for(items, refs):
    """Evaluation cases: laya's own fp32 logits, with the marker and
    question-type inputs of the int32 interface (constraint D)."""
    out = []
    for i, (it, ref) in enumerate(zip(items, refs)):
        marker_pos = np.full((1, KMAX), -1, dtype=np.int32)
        marker_pos[0, : it["k"]] = it["markers"]
        out.append(core.Case(ids=it["ids"], ref=ref, label=f"{it['state']}/{it['q']}",
                             extra={"marker_pos": marker_pos, "qtype": np.array([it["qtype"]], dtype=np.int32)}))
    return core.Evaluation(out)


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=(
        ("--model", {"default": "laya-en", "choices": sorted(MODELS), "help": "which laya-format checkpoint"}),
        ("--laya-code", {"type": Path, "help": "laya-typed-decisions: the laya package's common.py"})))
    src, install_dir = args.src, args.install_dir
    install_dir.mkdir(parents=True, exist_ok=True)

    rl = load_rl_common(src, args.model, args.laya_code)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    man = check_manifest(cfg, args.model)
    buckets = args.buckets or man["buckets"]
    tok = AutoTokenizer.from_pretrained(src / "tokenizer")
    items = gate_items(rl, tok, cfg)
    modernbert.install_patches()   # finite masks and traceable RoPE, for laya's own forward too
    dm = load_decision_model(src, rl, cfg)
    print(f"gate set: {len(items)} items; fp32 reference...", flush=True)
    refs = reference_logits(dm, items)

    # constraint B: K is pinned; check its headroom on the gate items, with
    # laya's own forward (a check, not a calibration: it decides nothing)
    maxima = linear_maxima(lambda: reference_logits(dm, items), modernbert.linears(dm.encoder))
    fixed, scaled = modernbert.split_maxima(maxima)
    headroom = saturation.headroom_at(fixed, scaled, K_RESIDUAL)
    at = max(maxima, key=maxima.get)
    print(f"residual scale K={K_RESIDUAL}: largest calibrated linear output (at layer {at[0]} {at[1]}) "
          f"is {headroom:.2f}x under the ANE linear's {saturation.ANE_LINEAR_MAX:.0f} "
          f"({saturation.ANE_LINEAR_MAX / max(maxima.values()):.2f}x at K=1)", flush=True)
    if not headroom >= 1.0 / saturation.HEADROOM:
        raise SystemExit(f"K={K_RESIDUAL} leaves only {headroom:.2f}x headroom")

    # explicit attention for conversion (D25); laya's own build uses sdpa
    tj = tokenizer.load(tokenizer.prepare(src / "tokenizer", install_dir / "tokenizer.json", mode="verbatim"))
    backbone = modernbert.load(None, tj, model=dm.encoder, config_dir=src / "encoder",
                               self_attending_pads=True)   # constraint F
    modernbert.matmul_softmax(backbone)
    backbone.attr = "enc"
    modernbert.residual_rewrite(backbone, K_RESIDUAL)
    head = LayaMarkers(dm.head.layers, dm.scorer, dm.type_emb.weight, kmax=KMAX, pad_logit=PAD_LOGIT,
                       softmax="matmul")                  # constraint F
    modernbert.twice_gelu(backbone)    # constraint E
    twice_gelu_scorer(head)
    ports = head.ports()
    make_wrapper, example = compose(backbone, head, ports)
    job = core.Job(
        name=args.model, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
        example=example, evaluation=cases_for(items, refs),
        gates=ClassifierGates(fp32_tol=FP32_TOL, margin=MARGIN, dp_gate=DP_GATE, gated_paths=("CPU_AND_NE",),
                              report_paths=("CPU_ONLY",), pad_value=PAD_LOGIT, pad_id_range=(1000, 40000)),
        install_files=[(manifest_path(args.model), "classifier.toml")], landing_required=True, gate_cases="landing",
        timing=args.time)
    core.run(job, install_dir)


if __name__ == "__main__":
    main()
