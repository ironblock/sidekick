"""Convert convaiinnovations/laya (English) into ANE-resident Core ML classifier
artifacts for sidekick's `POST /v1/classify` (docs/design/classify.md).

laya is a decision model: a ModernBERT-large encoder, then a two-layer
transformer head and a scorer that reads a [MASK] marker placed before each
candidate option. One artifact per sequence-length bucket runs the whole
model and returns one logit per marker.

Usage:
    python tools/convert_laya.py <laya-dir> <install-dir> [buckets...] [--time]

    laya-dir:     local snapshot of convaiinnovations/laya at revision
                  55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851: model.safetensors,
                  rl_common.py, rl_agent_config.json, encoder/config.json,
                  tokenizer/tokenizer.json
    install-dir:  classifier directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/laya-en"
    buckets:      default 128 256 512

Requires: torch, transformers >= 4.48 (ModernBERT), coremltools, numpy,
safetensors (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

laya's own code (rl_common.py, Apache-2.0, convaiinnovations) is imported
from the snapshot rather than copied; its git blob hash is checked against
the pinned revision, because the token-id fixtures and the Rust port of its
build_sequence must agree with exactly that version.

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

Gates, per bucket:
- fp32: the wrapper (rewrite, explicit attention, one-hots) reproduces
  laya's own forward, max |dlogit| <= FP32_TOL;
- converted graph: no fused attention op and no native gelu;
- on CPU_AND_NE, the path sidekick serves: argmax agreement with fp32
  wherever fp32's top-2 margin is >= MARGIN, and max raw |dp| <= DP_GATE;
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
MANIFEST = REPO / "examples" / "classifiers" / "laya-en" / "classifier.toml"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
RL_COMMON_BLOB = "d90d564964bcdc77586a257b0acccb5b7b19d6cf"  # git blob of rl_common.py at LAYA_REVISION
KMAX = 32
K_RESIDUAL = 2
PAD_LOGIT = -1e4
FP32_TOL = 2e-3
MARGIN = 0.05       # argmax flips count only above this fp32 top-2 logit margin
DP_GATE = 0.05      # max raw |dp| vs fp32 on CPU_AND_NE (CPU_ONLY is reported)

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


def load_rl_common(src):
    path = src / "rl_common.py"
    blob = git_blob_sha1(path)
    if blob != RL_COMMON_BLOB:
        raise SystemExit(f"{path} is not laya's rl_common.py at {LAYA_REVISION[:7]} "
                         f"(git blob {blob}, expected {RL_COMMON_BLOB})")
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


def check_manifest(cfg):
    """classifier.toml must agree with laya's config: head_max_len, KMAX and
    the calibration table (its keys are laya's temp_bucket keys)."""
    man = tomllib.loads(MANIFEST.read_text())
    if man["classify"]["max_labels"] != KMAX:
        raise SystemExit(f"{MANIFEST}: max_labels {man['classify']['max_labels']} != KMAX {KMAX}")
    if man["classify"]["laya"]["head_max_len"] != cfg["head_max_len"] or man["max_seq_len"] != cfg["max_len"]:
        raise SystemExit(f"{MANIFEST}: head_max_len/max_seq_len disagree with rl_agent_config.json")
    if man["source"]["revision"] != LAYA_REVISION:
        raise SystemExit(f"{MANIFEST}: source revision is not {LAYA_REVISION}")
    for key, t in man["classify"]["calibration"].items():
        fitted = cfg["temperature_by_options"].get(key)
        if fitted is None or abs(fitted - t) > 5e-4:
            raise SystemExit(f"{MANIFEST}: calibration {key} = {t}, laya fitted {fitted}")


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
    args = cli.parse(__doc__.split("\n\n")[0])
    src, install_dir = args.src, args.install_dir
    install_dir.mkdir(parents=True, exist_ok=True)

    rl = load_rl_common(src)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    check_manifest(cfg)
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
    backbone = modernbert.load(None, tj, model=dm.encoder, config_dir=src / "encoder")
    backbone.attr = "enc"
    modernbert.residual_rewrite(backbone, K_RESIDUAL)
    head = LayaMarkers(dm.head.layers, dm.scorer, dm.type_emb.weight, kmax=KMAX, pad_logit=PAD_LOGIT)
    modernbert.twice_gelu(backbone)    # constraint E
    twice_gelu_scorer(head)
    ports = head.ports()
    make_wrapper, example = compose(backbone, head, ports)
    job = core.Job(
        name="laya-en", buckets=args.buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
        example=example, evaluation=cases_for(items, refs),
        gates=ClassifierGates(fp32_tol=FP32_TOL, margin=MARGIN, dp_gate=DP_GATE, gated_paths=("CPU_AND_NE",),
                              report_paths=("CPU_ONLY",), pad_value=PAD_LOGIT, pad_id_range=(1000, 40000)),
        install_files=[(MANIFEST, "classifier.toml")], landing_required=True, gate_cases="landing",
        timing=args.time)
    core.run(job, install_dir)


if __name__ == "__main__":
    main()
