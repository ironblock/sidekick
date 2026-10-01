"""Convert SupersonicLabs/Julia-1 into ANE-resident Core ML classifier
artifacts for sidekick's `POST /v1/classify` (docs/design/classify.md).

Julia-1 is a decision model in laya's format: an mmBERT-small encoder (a
multilingual ModernBERT with a 256k vocabulary), then laya's two-layer
transformer head and a scorer that reads a <mask> marker placed before each
candidate option. Its option texts are rendered differently from laya's
(`option_rendering = "julia"` in its manifest), which changes the tokens
sidekick builds, not the graph. One artifact per sequence-length bucket runs
the whole model and returns one logit per marker, with laya's interface.

Usage:
    python tools/convert_julia.py <julia-dir> <install-dir> [buckets...] [--time]

    julia-dir:    local snapshot of SupersonicLabs/Julia-1 at revision
                  a85b127321d580d65176c89ced8273f305745d85: model.safetensors,
                  julia_config.json, julia/data.py, julia/model.py,
                  encoder/config.json, tokenizer/
    install-dir:  classifier directory the daemon scans, e.g.
                  "~/Library/Application Support/sidekick/models/julia-1"
    buckets:      default: the manifest's (128 256 512 1024)

Requires: torch, transformers >= 4.48 (ModernBERT), coremltools, numpy,
safetensors (arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

Julia-1's own code (julia/data.py and julia/model.py, from its snapshot) is
imported rather than copied, after a sha256 check against the pinned
revision: its JuliaDecisionModel is the fp32 reference, and its sequence()
builds the gate set's inputs, as tools/classifier_reference.py does for the
parity suite.

Core ML interface (all int32), laya's (tools/convert_laya.py):
    input_ids      [1, S]
    attention_mask [1, S]
    marker_pos     [1, KMAX]   position of each option's <mask>; -1 pads
    qtype          [1]         0 choice, 1 score, 2 noul (rank 1)
    logits         [1, KMAX]   one per option; padded slots are -1e4

KMAX is 20, Julia-1's option limit (the manifest's max_labels).

The conversion is convert_laya.py's, constraints A and C to F, on Julia's
modules:
- A: the library's ModernBERT backbone with explicit attention. mmBERT's
  config.json gives RoPE theta in transformers 5's `rope_parameters` block,
  which transformers 4.57 ignores: it would run the sliding layers at theta
  10000 instead of 160000, a different model. The encoder is built with
  modernbert.resolve_config, and verify_config checks every layer.
- B differs: the residual range rewrite's K is chosen by calibration on the
  gate set (saturation.choose_k), not pinned. mmBERT-small's largest linear
  output leaves enough headroom at K = 1.
- C, D: laya's head (heads/laya.py), with its attention written out and its
  inputs built in-graph.
- E: GELU from erf (TwiceGelu) in the encoder's MLPs and the scorer.
- F: every softmax from exp and one matmul against [V | 1], with the row max
  in 128-wide blocks and self-attending pad queries, so the ANE's output is
  the same in every bucket.

Gates, per bucket, as convert_laya.py's: the fp32 wrapper reproduces
Julia-1's own forward (max |dlogit| <= FP32_TOL); no fused attention or
native gelu in the program; on CPU_AND_NE, no argmax flip where fp32's top-2
margin is >= MARGIN, and max raw |dp| <= DP_GATE; CPU_ONLY reported only;
finite logits, padded slots at -1e4, pad invariance; the compute plan.
DP_GATE is 0.10 because Julia-1's ideal-fp16 ceiling alone reaches 0.055:
fp16 storage moves its probabilities more than laya's. These gates catch a
broken conversion; accuracy is graded by the parity suite against that
ceiling.

Measured (M1 Max, macOS 27.0) by the parity suite on Julia-1's 2,510-case
corpus (fixtures/classify/julia-1.corpus.toml), against Julia-1's fp32
forward and its ideal-fp16 ceiling (|dp| max 0.055, p99 0.028, mean 0.0040,
and 2 decisions flipped by fp16 storage alone):
- ANE: grade C, capped by 7 flips. Its p99 is 1.84x the ceiling's, a B by
  itself. Worst |dp| 0.116, mean 0.0072; 9.0 ms median; bucket invariance
  exact; 1,647 of 1,665 operations on the ANE at bucket 1024. The flips
  have fp32 margins of 0.05-0.35 logits, and most are fp16-borderline
  inputs: one is also a flip of the ceiling itself, and four more rank
  among the 101 inputs fp16 storage moves most.
- GPU: grade C, capped by 2 flips (both also CPU flips; one is the
  ceiling's). Its p99 is 1.07x the ceiling's, an A by itself.
- CPU_ONLY: grade D, 39 flips, p99 5.3x the ceiling's.
- Where the ceiling comes from: rounding only the weights to fp16, every
  activation exact (fp16sim with ops=set()), already reaches |dp| max
  0.052, p99 0.0135 and mean 0.0020, about half the ceiling at p99, and
  flips one of its two decisions. The 256k-token embedding table is a small
  part of that (p99 0.0024, alone); the encoder's and head's other weights
  are the rest (p99 0.0119, and the flip). Julia-1's decisions resolve
  finer than its weights do in fp16.
- Gold accuracy (reported, not graded) is 45% in fp32 and on every path.
  Julia-1's training data isn't published, so fast-decisions measures
  conversion parity, not the model's accuracy.
"""

import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from convert_laya import GATE_STATES
from sidekick_convert import cli, core, tokenizer
from sidekick_convert.backbones import modernbert
from sidekick_convert.calibrate import linear_maxima
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.heads.laya import LayaMarkers, twice_gelu_scorer
from sidekick_convert.techniques import saturation
from sidekick_convert.wrapper import compose

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "examples" / "classifiers" / "julia-1" / "classifier.toml"
REVISION = "a85b127321d580d65176c89ced8273f305745d85"
CODE_SHA256 = {"data.py": "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
               "model.py": "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81"}
KMAX = 20
PAD_LOGIT = -1e4
FP32_TOL = 2e-3
MARGIN = 0.05       # argmax flips count only above this fp32 top-2 logit margin
DP_GATE = 0.10      # max raw |dp| vs fp32 on CPU_AND_NE (CPU_ONLY is reported); see the gates above

# convert_laya.py's gate questions in Julia-1's terms: a choice option is its
# description (the label when it has none), a score option its label, a noul
# question's options "false"/"true" or both descriptions
GATE_QUESTIONS = [
    {"type": "noul", "question": "The text discusses money, markets or finance.", "options": ["false", "true"]},
    {"type": "noul", "question": "The text contains source code.",
     "options": ["no, there is no code", "yes, there is code in it"]},
    {"type": "choice", "question": "What is the main topic of the text?",
     "options": ["money, companies, markets", "deliveries and parcels", "code and computers", "other"]},
    {"type": "score", "question": "How formal is the writing?",
     "options": ["very informal", "informal", "neutral", "formal", "very formal"]},
    {"type": "choice", "question": "Which label fits best?",
     "options": [f"description number {i}" for i in range(KMAX)]},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_julia_code(src):
    """julia/data.py and julia/model.py from the snapshot, after checking
    they are the pinned files."""
    mods = {}
    for name, want in CODE_SHA256.items():
        path = src / "julia" / name
        if sha256(path) != want:
            raise SystemExit(f"{path} is not Julia-1's {name} at {REVISION[:7]} (sha256 {sha256(path)})")
        spec = importlib.util.spec_from_file_location(f"julia_{path.stem}", path)
        mods[path.stem] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[path.stem])
    return mods["data"], mods["model"]


def julia_tokenizer(src):
    """Julia-1's tokenizer with the special tokens its tokenizer_config names
    (CLS <bos>, SEP <eos>, MASK <mask>): the ones julia/data.py reads."""
    tok = PreTrainedTokenizerFast(tokenizer_file=str(src / "tokenizer" / "tokenizer.json"))
    config = json.loads((src / "tokenizer" / "tokenizer_config.json").read_text())
    for name in ("cls_token", "sep_token", "mask_token", "pad_token"):
        setattr(tok, name, config[name])
    return tok


def load_decision_model(src, model_py):
    """JuliaDecisionModel in fp32 with Julia-1's weights, its encoder built
    from encoder/config.json with RoPE from its rope_parameters block
    (constraint A)."""
    raw = modernbert.file_config(src / "encoder")
    ecfg = modernbert.resolve_config(ModernBertConfig(**raw), raw)
    ecfg._attn_implementation = "sdpa"
    jcfg = json.loads((src / "julia_config.json").read_text())
    dm = model_py.JuliaDecisionModel(ModernBertModel(ecfg), head_layers=jcfg["head_layers"], n_act=jcfg["n_act"])
    dm.load_state_dict({k: v.float() for k, v in load_file(src / "model.safetensors").items()}, strict=True)
    return dm.float().eval()


def gate_items(data, tok, man):
    max_len, head = man["max_seq_len"], man["classify"]["laya"]["head_max_len"]
    items = []
    for si, state in enumerate(GATE_STATES):
        for qi, q in enumerate(GATE_QUESTIONS):
            row = dict(q, state=state)
            data.validate_row(row, 1)
            enc = data.sequence(tok, row, max_len, head)
            k = len(q["options"])
            if len(enc["markers"]) != k:
                raise SystemExit(f"gate item {si}/{qi}: {len(enc['markers'])} markers for {k} options")
            items.append({"state": si, "q": qi, "k": k, "ids": enc["ids"], "markers": enc["markers"],
                          "qtype": enc["qtype"]})
    return items


def reference_logits(dm, items):
    """Julia-1's own forward in fp32, one unpadded sequence at a time."""
    out = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([it["ids"]])
            logits = dm(ids, torch.ones_like(ids), torch.tensor([it["markers"]]),
                        torch.ones((1, it["k"]), dtype=torch.bool), torch.tensor([it["qtype"]]))
            out.append(logits[0].numpy().astype(np.float64))
    return out


def check_manifest():
    man = tomllib.loads(MANIFEST.read_text())
    if man["classify"]["max_labels"] != KMAX:
        raise SystemExit(f"{MANIFEST}: max_labels {man['classify']['max_labels']} != KMAX {KMAX}")
    if man["source"]["revision"] != REVISION:
        raise SystemExit(f"{MANIFEST}: source revision is not {REVISION}")
    if man["classify"]["laya"].get("option_rendering") != "julia":
        raise SystemExit(f"{MANIFEST}: option_rendering must be \"julia\"")
    return man


def cases_for(items, refs):
    """Evaluation cases: Julia-1's own fp32 logits, with the marker and
    question-type inputs of the int32 interface."""
    out = []
    for it, ref in zip(items, refs):
        marker_pos = np.full((1, KMAX), -1, dtype=np.int32)
        marker_pos[0, : it["k"]] = it["markers"]
        out.append(core.Case(ids=it["ids"], ref=ref, label=f"{it['state']}/{it['q']}",
                             extra={"marker_pos": marker_pos, "qtype": np.array([it["qtype"]], dtype=np.int32)}))
    return core.Evaluation(out)


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None)
    src, install_dir = args.src, args.install_dir
    install_dir.mkdir(parents=True, exist_ok=True)

    data, model_py = load_julia_code(src)
    man = check_manifest()
    buckets = args.buckets or man["buckets"]
    tok = julia_tokenizer(src)
    items = gate_items(data, tok, man)
    modernbert.install_patches()   # finite masks and traceable RoPE, for Julia-1's own forward too
    dm = load_decision_model(src, model_py)
    print(f"gate set: {len(items)} items, {min(len(i['ids']) for i in items)}-"
          f"{max(len(i['ids']) for i in items)} tokens; fp32 reference...", flush=True)
    refs = reference_logits(dm, items)

    maxima = linear_maxima(lambda: reference_logits(dm, items), modernbert.linears(dm.encoder))
    fixed, scaled = modernbert.split_maxima(maxima)
    k_residual, headroom = saturation.choose_k(fixed, scaled)
    at = max(maxima, key=maxima.get)
    print(f"residual scale K={k_residual}: largest calibrated linear output (at layer {at[0]} {at[1]}) "
          f"is {headroom:.2f}x under the ANE linear's {saturation.ANE_LINEAR_MAX:.0f}", flush=True)

    tj = tokenizer.load(tokenizer.prepare(src / "tokenizer", install_dir / "tokenizer.json", mode="verbatim"))
    backbone = modernbert.load(None, tj, model=dm.encoder, config_dir=src / "encoder",
                               self_attending_pads=True)   # constraint F
    modernbert.matmul_softmax(backbone)
    backbone.attr = "enc"
    modernbert.residual_rewrite(backbone, k_residual)
    head = LayaMarkers(dm.head.layers, dm.scorer, dm.type_emb.weight, kmax=KMAX, pad_logit=PAD_LOGIT,
                       softmax="matmul")                  # constraint F
    modernbert.twice_gelu(backbone)    # constraint E
    twice_gelu_scorer(head)
    ports = head.ports()
    make_wrapper, example = compose(backbone, head, ports)
    job = core.Job(
        name="julia-1", buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
        example=example, evaluation=cases_for(items, refs),
        gates=ClassifierGates(fp32_tol=FP32_TOL, margin=MARGIN, dp_gate=DP_GATE, gated_paths=("CPU_AND_NE",),
                              report_paths=("CPU_ONLY",), pad_value=PAD_LOGIT, pad_id_range=(1000, 40000)),
        install_files=[(MANIFEST, "classifier.toml")], landing_required=True, gate_cases="landing",
        timing=args.time)
    core.run(job, install_dir)


if __name__ == "__main__":
    main()
