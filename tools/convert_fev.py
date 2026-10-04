"""Convert fev-format decision models (FrontiersMind/Lumma-fev-0.1b) into
ANE-resident Core ML classifier artifacts for sidekick's `POST /v1/classify`
(docs/design/classify.md, "The fev format").

Lumma-fev is a causal decoder (Nandi, sidekick_convert.backbones.nandi) read
by a pointer head: one row holds the state and one question,
`<state> text <question> instructions (<option> text </option>)... <decide>`,
and each option's logit is k(h[</option>]) . q(h[<decide>]) / sqrt(d). One
artifact per sequence-length bucket runs the whole model.

Usage:
    python tools/convert_fev.py <lumma-fev-dir> <install-dir> [buckets...] [--time]

    lumma-fev-dir: local snapshot of FrontiersMind/Lumma-fev-0.1b at revision
                   085f4705aa860a6404d6cc3ff17de8a2969ac0f4: config.json,
                   model.safetensors, tokenizer.json, modeling_fev.py,
                   modeling_nandi.py and their configuration modules
    install-dir:   classifier directory the daemon scans, e.g.
                   "~/Library/Application Support/sidekick/models/lumma-fev-0.1b"
    buckets:       default: the manifest's (128 256 512 1024 2048)

Requires: torch, transformers (4.57, for the checkpoint's own code through
tools/classifier_reference.py's shims), coremltools, numpy, safetensors
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`.

The checkpoint's own modeling_fev.py lays out the gate set's inputs and
runs the fp32 reference (tools/classifier_reference.py's load_fev, which
pins its code by sha256). The converted graph is the library's plain-torch
Nandi, so the fp32 gate checks the port and every rewrite at once.

Core ML interface (all int32):
    input_ids      [1, S]
    attention_mask [1, S]
    marker_pos     [1, KMAX]   each option's </option> position; -1 pads
    decide_pos     [1]         the <decide> position (rank 1)
    logits         [1, KMAX]   one per option; padded slots are -1e4

Conversion constraints:
A. ATTENTION written out, with a finite causal-and-key-padding mask in which
   every query attends to itself (no fully masked row), and RoPE's cos and
   sin as constant buffers per bucket. No fused attention op.
B. fp16 RANGE: the residual stream reaches ~870 and every RMSNorm squares
   its input, past fp16's 65504; each RMSNorm gets a calibrated power-of-two
   input scale with eps compensated (backbones.nandi.fp16_norms). No linear
   output comes near the ANE's 2^15 (the largest is ~200), so there is no
   residual rewrite.
C. SILU as StableSilu (2 * silu from exp, without cancellation for either
   sign), up_proj taking the 1/2 (D20 amendment, D39). The earlier TanhSilu
   cancels in fp16 for negative inputs: StableSilu halves the worst ANE
   errors. Power-of-two input rescales add nothing measurable here (the
   linears' inputs are not small), so there are none.
D. BUCKET-INVARIANT SOFTMAX: every attention computes its softmax as
   exp(w - rowmax) and one matmul against [V | 1] (laya's constraint F,
   docs/DECISIONS.md D28 amendment).
E. INPUTS BUILT IN-GRAPH: marker_pos and decide_pos become one-hot rows
   against a position constant, and a matmul selects the hidden states. No
   data-dependent gather.
F. TEMPERATURE: the checkpoint's is 1.0, so the logits are fev's own. Any
   other value fails the conversion until it is folded into the graph.
G. THE FACTORIZED EMBEDDING is folded into one table (table @ proj^T,
   backbones.nandi.fold_embedding), exact up to fp32 rounding. Core ML keeps
   the first linear after the gather on the CPU however it is written (as a
   linear or a matmul, scaled or not), and every linear belongs on the ANE.
   The table grows from 131k x 196 to 131k x 832, about 166 MB more per
   bucket in fp16; the gather reads it on the CPU.

Gates, per bucket: the fp32 wrapper reproduces the checkpoint's own forward
(max |dlogit| <= FP32_TOL); no fused attention op and no native silu in the
program; on CPU_AND_NE, no argmax flip where fp32's top-2 margin is >=
MARGIN, and max raw |dp| <= DP_GATE; CPU_ONLY reported only; finite logits,
padded slots at -1e4, pad invariance; the compute plan. These gates catch a
broken conversion; accuracy is graded by the parity suite against the
model's ideal-fp16 ceiling.

Measured (M1 Max, macOS 27.0) by the parity suite on the 2,630-case corpus
(fixtures/classify/lumma-fev-0.1b.corpus.toml), against the checkpoint's
fp32 forward and its ideal-fp16 ceiling (|dp| max 0.0059, p99 0.0028):
- ANE: grade B, p99 1.43x the ceiling's; worst |dp| 0.0057; no flips;
  bucket invariance exact; 3,858 of 3,875 operations on the ANE at 2,048;
  17 / 36 / 115 / 305 / 1,308 ms at buckets 128 to 2,048. With TanhSilu
  (constraint C's earlier form): p99 1.57x, worst 0.010, about 12% faster.
- GPU: grade A, p99 0.92x the ceiling's; worst |dp| 0.0043; no flips.
- CPU_ONLY: D on accuracy (p99 5.7x), with one flip. Inputs over 512
  tokens move by up to 0.021 between the 1,024 and 2,048 buckets: Core
  ML's fp16 CPU matmul sums a contraction over 1,024 in a different order
  (tools/repro_cpu_matmul_accumulation.py), and slicing it costs more
  accuracy than it saves. That is the documented CPU limit over 1,024
  tokens (docs/DECISIONS.md D33).
"""

import math
import tomllib
from pathlib import Path

import numpy as np
import torch

import classifier_reference as cr
from convert_laya import GATE_STATES
from sidekick_convert import cli, core, tokenizer
from sidekick_convert.backbones import nandi
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.techniques.onehot import positions, positions_onehot

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "examples" / "classifiers" / "lumma-fev-0.1b" / "classifier.toml"
REVISION = "085f4705aa860a6404d6cc3ff17de8a2969ac0f4"
KMAX = 32
PAD_LOGIT = -1e4
FP32_TOL = 2e-3
MARGIN = 0.05       # argmax flips count only above this fp32 top-2 logit margin
DP_GATE = 0.08      # max raw |dp| vs fp32 on CPU_AND_NE (CPU_ONLY is reported)

# the gate questions, as request labels (the contract's rendering applies)
GATE_QUESTIONS = [
    ("noul", "The text discusses money, markets or finance.", ["false", "true"]),
    ("noul", "The text contains source code.", ["false: no, there is no code", "true: yes, there is code in it"]),
    ("choice", "What is the main topic of the text?",
     ["finance: money, companies, markets", "shipping: deliveries and parcels", "software: code and computers",
      "other"]),
    ("score", "How formal is the writing?", ["very informal", "informal", "neutral", "formal", "very formal"]),
    ("choice", "Which label fits best?", [f"label_{i}: description number {i}" for i in range(KMAX)]),
    ("choice", None, ["positive", "negative", "neutral"]),
]
_LONG = " ".join(f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail." for i in range(36))
EXTRA_STATES = [" ".join([_LONG] * 4)]   # past the 1,408-token state limit: lands in the largest bucket


class FevWrapper(torch.nn.Module):
    """The whole model at one bucket, on the int32 interface."""

    def __init__(self, model, head_q, head_k, head_scale, seq):
        super().__init__()
        self.model, self.q, self.k, self.seq = model, head_q, head_k, int(seq)
        self.head_scale = float(head_scale)
        cos, sin = model.rope(seq)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("positions", positions(seq))

    def forward(self, input_ids, attention_mask, marker_pos, decide_pos):
        h = self.model(input_ids.long(), attention_mask, self.cos, self.sin, self.seq)       # [1, S, d]
        # one selection matmul for the options and <decide> together: a lone one-row selection lands on the CPU
        where = torch.cat([marker_pos, decide_pos.reshape(1, 1)], dim=1)                   # [1, K+1]
        picked = positions_onehot(where, self.positions, h.dtype) @ h                       # [1, K+1, d]
        options, decide = picked[:, :-1], picked[:, -1:]
        logits = (self.k(options) @ self.q(decide).transpose(1, 2)).squeeze(-1) * self.head_scale   # [1, K]
        valid = (marker_pos >= 0).to(logits.dtype)
        return logits * valid + (1.0 - valid) * PAD_LOGIT


def gate_items(fev, model, tok, man):
    """Every gate state with every gate question, laid out by the
    checkpoint's own pack() and encode()."""
    cases = []
    for si, state in enumerate(GATE_STATES + EXTRA_STATES):
        for qi, (qtype, ins, labels) in enumerate(GATE_QUESTIONS):
            cases.append({"id": f"{si}/{qi}", "tags": [], "input": state, "candidate_labels": labels,
                          "question_type": qtype, "instructions": ins, "gold": None, "head": None})
    return cr.fev_build(cases, fev, model, tok, man)


def reference_logits(fev_model, items):
    """The checkpoint's own forward in fp32, one unpadded row at a time."""
    wrapped = cr.FevLogits(fev_model).eval()
    out = []
    with torch.no_grad():
        for it in items:
            ids = torch.tensor([it["ids"]])
            out.append(wrapped(ids, torch.ones_like(ids), torch.tensor(it["markers"]),
                               torch.tensor(it["decide"])).numpy().astype(np.float64))
    return out


def cases_for(items, refs):
    out = []
    for it, ref in zip(items, refs):
        marker_pos = np.full((1, KMAX), -1, dtype=np.int32)
        marker_pos[0, : it["k"]] = it["markers"]
        out.append(core.Case(ids=it["ids"], ref=ref, label=it["id"],
                             extra={"marker_pos": marker_pos, "decide_pos": np.array([it["decide"]], dtype=np.int32)}))
    return core.Evaluation(out)


def check_manifest():
    man = tomllib.loads(MANIFEST.read_text())
    if man["classify"]["max_labels"] != KMAX:
        raise SystemExit(f"{MANIFEST}: max_labels {man['classify']['max_labels']} != KMAX {KMAX}")
    if man["source"]["revision"] != REVISION:
        raise SystemExit(f"{MANIFEST}: source revision is not {REVISION}")
    if man["classify"].get("format") != "fev":
        raise SystemExit(f"{MANIFEST}: format must be \"fev\"")
    return man


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None)
    if args.chunks not in (None, "1"):
        raise SystemExit("--chunks: this converter's wrapper isn't a sidekick_convert backbone, so it can't be "
                         "chunked yet")
    src, install_dir = args.src, args.install_dir
    install_dir.mkdir(parents=True, exist_ok=True)
    man = check_manifest()
    buckets = args.buckets or man["buckets"]

    fev, fev_model, tok = cr.load_fev(src)          # the checkpoint's own code: layout and fp32 reference
    items = gate_items(fev, fev_model, tok, man)
    print(f"gate set: {len(items)} items, {min(len(i['ids']) for i in items)}-"
          f"{max(len(i['ids']) for i in items)} tokens; fp32 reference...", flush=True)
    refs = reference_logits(fev_model, items)

    model, cfg, rest = nandi.load(src)
    if set(rest) != {"head.q.weight", "head.q.bias", "head.k.weight", "head.k.bias"}:
        raise SystemExit(f"unexpected non-backbone weights: {sorted(rest)}")
    head_q = torch.nn.Linear(cfg["hidden_size"], cfg["head_dim"])
    head_k = torch.nn.Linear(cfg["hidden_size"], cfg["head_dim"])
    with torch.no_grad():
        for lin, name in ((head_q, "q"), (head_k, "k")):
            lin.weight.copy_(rest[f"head.{name}.weight"]); lin.bias.copy_(rest[f"head.{name}.bias"])

    def calibrate():   # the gate set's rows, unpadded: calibration on the converter's own inputs (D26)
        for it in items:
            n = len(it["ids"])
            cos, sin = model.rope(n)
            model(torch.tensor([it["ids"]]), torch.ones(1, n, dtype=torch.long), cos, sin, n)

    nandi.fp16_norms(model, calibrate)              # constraint B
    nandi.stable_silu(model)                        # constraint C
    nandi.fold_embedding(model)                     # constraint G
    model.softmax = "matmul"                        # constraint D

    tokenizer.prepare(src, install_dir / "tokenizer.json", mode="verbatim")
    ports = [core.sequence_port("input_ids"), core.sequence_port("attention_mask"),
             core.Port("marker_pos", lambda seq: (1, KMAX)), core.Port("decide_pos", lambda seq: (1,))]

    def make_wrapper(seq):
        return FevWrapper(model, head_q, head_k, 1.0 / math.sqrt(cfg["head_dim"]), seq).eval()

    def example(seq):
        ids = np.full((1, seq), 3, dtype=np.int32)
        ids[0, :3] = [131062, 131063, 131066]
        mask = np.zeros((1, seq), dtype=np.int32)
        mask[0, :3] = 1
        marker = np.full((1, KMAX), -1, dtype=np.int32)
        marker[0, 0] = 1
        return {"input_ids": ids, "attention_mask": mask, "marker_pos": marker,
                "decide_pos": np.array([2], dtype=np.int32)}

    job = core.Job(
        name="lumma-fev-0.1b", buckets=buckets, ports=ports, output="logits", make_wrapper=make_wrapper,
        example=example, evaluation=cases_for(items, refs),
        gates=ClassifierGates(fp32_tol=FP32_TOL, margin=MARGIN, dp_gate=DP_GATE, gated_paths=("CPU_AND_NE",),
                              report_paths=("CPU_ONLY",), pad_value=PAD_LOGIT, pad_id_range=(1000, 40000)),
        forbid_ops=frozenset({core.FUSED_ATTENTION, "silu"}),
        install_files=[(MANIFEST, "classifier.toml")], landing_required=True, gate_cases="landing",
        timing=args.time, int8_embedding=args.int8_embedding, ignore_ane_weight_cap=args.ignore_ane_weight_cap)
    core.run(job, install_dir)


if __name__ == "__main__":
    main()
