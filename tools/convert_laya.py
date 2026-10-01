"""Convert convaiinnovations/laya (English) into ANE-resident Core ML classifier
artifacts for sidekick's `POST /v1/classify` (docs/design/classify.md).

laya is a decision model: a ModernBERT-large encoder, then a two-layer
transformer head and a scorer that reads a [MASK] marker placed before each
candidate option. One artifact per sequence-length bucket runs the whole
model and returns one logit per marker.

Usage:
    python tools/convert_laya.py <laya-dir> <install-dir> [buckets...]

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

A. The encoder follows tools/convert_gte_modernbert.py: explicit (eager)
   attention (D25), fp16-safe masks, traceable rotate_half. Its helpers are
   imported, not copied.
B. RESIDUAL RANGE REWRITE PINNED AT K = 2 (D25's amendment). The ANE's
   linear op saturates above 2^15. laya's encoder writes up to ~27,500 in
   layer 19's MLP output projection, so K = 1 would leave only ~19%
   headroom under 32,768; the calibration rule would pick K = 1, and the
   design pins K = 2 instead. The converter measures the maxima on its
   gate set and fails if K = 2 leaves less than 1/0.85 headroom.
C. THE HEAD'S ATTENTION IS WRITTEN OUT. nn.TransformerEncoderLayer's fast
   path and the fused attention op are never traced: q/k/v projections,
   matmul -> softmax -> matmul, with a finite additive mask for pad keys.
D. INPUTS BUILT IN-GRAPH. marker_pos becomes a [KMAX, S] one-hot by
   comparison with a position constant (a -1 pad matches nothing), and
   qtype becomes a one-hot row that selects laya's question-type
   embedding. No data-dependent gather, so the graph stays static.
E. EXPLICIT GELU (docs/DECISIONS.md D28 amendment). Core ML's native gelu
   op is coarse on the ANE: up to 6e-3 off on [-1, 1] (the GPU: 3e-4), where
   most of the encoder's MLP inputs lie. laya's decisions amplify it. Its
   encoder's first layers are the sensitive ones, and there the native gelu
   made the MLP branch 9-15x less accurate than on the GPU. TwiceGelu
   computes x * (1 + erf(x / sqrt 2)) from erf, mul and add, 9x closer on
   [-1, 1]. The factor 2 goes into the weights that consume it: each
   encoder MLP's gate rows and the scorer's output linear. Exact in fp32.
   Keeping the 0.5 out of the graph matters: 0.5 * x * (1 + erf(x / sqrt 2))
   is fused back into the native op. convert_bucket() fails if a gelu op
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
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_gte_modernbert as gte  # noqa: E402  (constraint A)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "examples" / "classifiers" / "laya-en" / "classifier.toml"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
RL_COMMON_BLOB = "d90d564964bcdc77586a257b0acccb5b7b19d6cf"  # git blob of rl_common.py at LAYA_REVISION
KMAX = 32
K_RESIDUAL = 2
MASK_ADD = -30000.0
PAD_LOGIT = -1e4
FP32_TOL = 2e-3
MARGIN = 0.05       # argmax flips count only above this fp32 top-2 logit margin
DP_GATE = 0.05      # max raw |dp| vs fp32, per path

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


class TwiceGelu(torch.nn.Module):
    """TWICE the exact (erf) GELU, x * (1 + erf(x / sqrt 2)) (constraint E).
    Without the 0.5, conversion keeps the erf instead of fusing the pattern
    back into Core ML's native gelu; explicit_gelu() folds the 0.5 into the
    next linear's weights."""

    def forward(self, x):
        return x * (1.0 + torch.erf(x * 0.7071067811865476))


def explicit_gelu(dm):
    """Constraint E: every GELU in the encoder's MLPs and the scorer as
    TwiceGelu, the 0.5 folded into the weights that consume it (the gate
    rows of each Wi, the scorer's output linear). Exact in fp32."""
    if dm.encoder.config.hidden_activation != "gelu":
        raise SystemExit(f"encoder activation {dm.encoder.config.hidden_activation!r}, expected exact gelu")
    act = dm.scorer[2] if len(dm.scorer) == 4 else None
    if not (isinstance(act, torch.nn.GELU) and act.approximate == "none"
            and isinstance(dm.scorer[3], torch.nn.Linear)):
        raise SystemExit(f"unexpected scorer layout: {dm.scorer}")
    with torch.no_grad():
        for layer in dm.encoder.layers:
            ff = layer.mlp.Wo.in_features
            layer.mlp.act = TwiceGelu()                 # Wo(act(input) * gate)
            layer.mlp.Wi.weight[ff:].mul_(0.5)          # gate rows
            if layer.mlp.Wi.bias is not None:
                layer.mlp.Wi.bias[ff:].mul_(0.5)
        dm.scorer[2] = TwiceGelu()
        dm.scorer[3].weight.mul_(0.5)


def explicit_head_layer(layer, x, add, seq):
    """nn.TransformerEncoderLayer(norm_first=True) with its attention written
    out (constraint C). Shapes come from module constants, never from traced
    sizes, which coremltools can't cast to int under static shapes."""
    attn = layer.self_attn
    heads, d = int(attn.num_heads), int(attn.embed_dim)
    hd = d // heads
    y = layer.norm1(x)
    q, k, v = F.linear(y, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q.reshape(1, seq, heads, hd).transpose(1, 2)
    k = k.reshape(1, seq, heads, hd).transpose(1, 2)
    v = v.reshape(1, seq, heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * (hd ** -0.5) + add
    o = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(1, seq, d)
    x = x + attn.out_proj(o)
    return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))


class LayaWrapper(torch.nn.Module):
    """Static-shape laya with the int32 interface (constraint D)."""

    def __init__(self, dm, seq):
        super().__init__()
        self.enc, self.layers, self.scorer = dm.encoder, dm.head.layers, dm.scorer
        self.type_w = dm.type_emb.weight
        self.seq = seq
        self.register_buffer("position_ids", torch.arange(seq, dtype=torch.long).unsqueeze(0))
        self.register_buffer("positions", torch.arange(seq, dtype=torch.long).reshape(1, 1, seq))
        self.register_buffer("qtypes", torch.arange(3, dtype=torch.long).reshape(1, 3))

    def forward(self, input_ids, attention_mask, marker_pos, qtype):
        h = self.enc(input_ids=input_ids.long(), attention_mask=attention_mask.long(),
                     position_ids=self.position_ids).last_hidden_state
        qt = (qtype.long().reshape(1, 1) == self.qtypes).to(h.dtype)          # [1, 3]
        h = h + (qt @ self.type_w).unsqueeze(1)
        add = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * MASK_ADD
        for layer in self.layers:
            h = explicit_head_layer(layer, h, add, self.seq)
        pos = marker_pos.long().unsqueeze(-1)                                   # [1, KMAX, 1]
        onehot = (pos == self.positions).to(h.dtype)                            # [1, KMAX, S]
        valid = (marker_pos >= 0).to(h.dtype)                                   # [1, KMAX]
        logits = self.scorer(onehot @ h).squeeze(-1)
        return logits * valid + (valid - 1.0) * -PAD_LOGIT


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


def bucket_of(n, buckets):
    return next(b for b in buckets if n <= b)


def inputs_for(item, seq, pad_ids=None):
    n = len(item["ids"])
    ids = np.zeros((1, seq), dtype=np.int32)
    ids[0, :n] = item["ids"]
    if pad_ids is not None:
        ids[0, n:] = pad_ids[: seq - n]
    mask = np.zeros((1, seq), dtype=np.int32)
    mask[0, :n] = 1
    marker_pos = np.full((1, KMAX), -1, dtype=np.int32)
    marker_pos[0, : item["k"]] = item["markers"]
    return {"input_ids": ids, "attention_mask": mask, "marker_pos": marker_pos,
            "qtype": np.array([item["qtype"]], dtype=np.int32)}


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


def encoder_output_maxima(dm, items):
    """fp32 max |output| of every encoder linear over the gate set (constraint B)."""
    maxima = {}
    hooks = []
    for i, layer in enumerate(dm.encoder.layers):
        for name, lin in (("Wqkv", layer.attn.Wqkv), ("attn.Wo", layer.attn.Wo),
                          ("Wi", layer.mlp.Wi), ("mlp.Wo", layer.mlp.Wo)):
            def hook(mod, args, out, key=(i, name)):
                maxima[key] = max(maxima.get(key, 0.0), float(out.detach().abs().max()))
            hooks.append(lin.register_forward_hook(hook))
    reference_logits(dm, items)
    for h in hooks:
        h.remove()
    return maxima


def headroom_at(maxima, k):
    fixed = max(v for (i, n), v in maxima.items() if not n.endswith("Wo"))
    scaled = max(v for (i, n), v in maxima.items() if n.endswith("Wo"))
    return gte.ANE_LINEAR_MAX / max(fixed, scaled / k), max(maxima, key=maxima.get)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def fp32_gate(wrapper, items, refs, seq, buckets):
    worst = 0.0
    with torch.no_grad():
        for it, ref in zip(items, refs):
            if bucket_of(len(it["ids"]), buckets) != seq:
                continue
            x = {k: torch.from_numpy(v) for k, v in inputs_for(it, seq).items()}
            out = wrapper(**x)[0].numpy().astype(np.float64)
            pad = out[it["k"]:]
            d = np.abs(out[: it["k"]] - ref).max()
            if not (np.isfinite(out).all() and np.all(pad == PAD_LOGIT)):
                raise SystemExit(f"seq {seq}: fp32 wrapper output malformed")
            worst = max(worst, float(d))
    if not worst <= FP32_TOL:  # NaN fails too
        raise SystemExit(f"seq {seq}: fp32 wrapper vs laya's forward, max |dlogit| {worst:.2e} > {FP32_TOL}")
    return worst


def convert_bucket(wrapper, example, seq, workdir):
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(torch.from_numpy(example[k]) for k in
                                                ("input_ids", "attention_mask", "marker_pos", "qtype")))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="marker_pos", shape=(1, KMAX), dtype=np.int32),
                ct.TensorType(name="qtype", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    if "scaled_dot_product_attention" in ops:
        raise SystemExit(f"seq {seq}: converted graph contains the fused attention op (D25)")
    if "gelu" in ops:
        raise SystemExit(f"seq {seq}: converted graph contains Core ML's native gelu (constraint E)")
    pkg = Path(workdir) / f"model_{seq}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def parity_check(pkg, items, refs, seq, buckets):
    """Per path: finite, padded slots at -1e4 and pad invariance, gated on
    both; argmax agreement above MARGIN and max raw |dp| <= DP_GATE, gated on
    CPU_AND_NE and reported on CPU_ONLY. Returns metrics and latency."""
    idx = [i for i, it in enumerate(items) if bucket_of(len(it["ids"]), buckets) == seq]
    results = {}
    for label, cu in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        dp_max, dl_max, flips, ties, ms = 0.0, 0.0, 0, 0, []
        for i in idx:
            it, ref = items[i], refs[i]
            x = inputs_for(it, seq)
            t0 = time.perf_counter()
            out = m.predict(x)["logits"][0].astype(np.float64)
            ms.append((time.perf_counter() - t0) * 1e3)
            k = it["k"]
            if not (np.isfinite(out).all() and np.all(out[k:] == PAD_LOGIT)):
                raise SystemExit(f"seq {seq} [{label}]: non-finite or unpadded logits")
            got = out[:k]
            top2 = np.sort(ref)[-2:]
            if top2[1] - top2[0] >= MARGIN:
                flips += int(np.argmax(got) != np.argmax(ref))
            else:
                ties += 1
            dp_max = max(dp_max, float(np.abs(softmax(got) - softmax(ref)).max()))
            dl_max = max(dl_max, float(np.abs(got - ref).max()))
        if flips or not dp_max <= DP_GATE:
            message = (f"seq {seq} [{label}]: {flips} argmax flips above margin {MARGIN}, "
                       f"max |dp| {dp_max:.4f} (gate {DP_GATE})")
            if label == "CPU_AND_NE":
                raise SystemExit(message)
            print(f"WARNING, report only: {message}")
        # pad invariance: pad content must not reach the logits
        it = next(items[i] for i in idx if len(items[i]["ids"]) < seq)
        a = m.predict(inputs_for(it, seq))["logits"][0, : it["k"]]
        rng = np.random.default_rng(0)
        b = m.predict(inputs_for(it, seq, pad_ids=rng.integers(1000, 40000, seq)))["logits"][0, : it["k"]]
        pad_d = float(np.abs(a - b).max())
        if not pad_d <= 1e-3:
            raise SystemExit(f"seq {seq} [{label}]: logits depend on pad content (max |dlogit| {pad_d})")
        results[label] = {"n": len(idx), "dp_max": dp_max, "dlogit_max": dl_max, "near_ties": ties,
                          "flips": flips,
                          "pad_dlogit": pad_d, "ms_median": float(np.median(ms[1:] or ms))}
    return results


def plan_check(pkg):
    """Compute plan after conversion: every linear/matmul on the ANE and >= 80%
    of assigned operations (ane_check's verdict)."""
    from coremltools.models.compute_plan import MLComputePlan
    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    ane, total, off, heavy_off = 0, 0, {}, []
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        total += 1
        dev = type(usage.preferred_compute_device).__name__
        if "NeuralEngine" in dev:
            ane += 1
        else:
            name = op.operator_name.split(".")[-1]
            off[name] = off.get(name, 0) + 1
            if name in ("linear", "matmul", "conv"):
                heavy_off.append(name)
    if total == 0:
        raise SystemExit("compute plan assigns no operations; re-read from another path (MODELS.md)")
    if heavy_off or ane / total < 0.8:
        raise SystemExit(f"compute plan: {ane}/{total} ops on the ANE; off-ANE {off}")
    return ane, total, off


def compile_to_mlmodelc(pkg, install_dir, seq):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True,
                       stdout=subprocess.DEVNULL)
        compiled = next(Path(tmp).glob("*.mlmodelc"))
        dest = install_dir / f"model_{seq}.mlmodelc"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
    return dest


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


def main():
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    install_dir.mkdir(parents=True, exist_ok=True)

    rl = load_rl_common(src)
    cfg = json.loads((src / "rl_agent_config.json").read_text())
    check_manifest(cfg)
    tok = AutoTokenizer.from_pretrained(src / "tokenizer")
    items = gate_items(rl, tok, cfg)
    gte.install_patches()
    dm = load_decision_model(src, rl, cfg)
    print(f"gate set: {len(items)} items; fp32 reference...", flush=True)
    refs = reference_logits(dm, items)

    maxima = encoder_output_maxima(dm, items)
    headroom, at = headroom_at(maxima, K_RESIDUAL)
    print(f"residual scale K={K_RESIDUAL}: largest calibrated linear output (at layer {at[0]} {at[1]}) "
          f"is {headroom:.2f}x under the ANE linear's {gte.ANE_LINEAR_MAX:.0f} "
          f"({gte.ANE_LINEAR_MAX / max(maxima.values()):.2f}x at K=1)", flush=True)
    if not headroom >= 1.0 / gte.LINEAR_HEADROOM:
        raise SystemExit(f"K={K_RESIDUAL} leaves only {headroom:.2f}x headroom")
    # explicit attention for conversion (D25); laya's own build uses sdpa
    dm.encoder.config._attn_implementation = "eager"
    gte.range_rewrite(dm.encoder, K_RESIDUAL)
    explicit_gelu(dm)

    report = {}
    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            if not any(bucket_of(len(it["ids"]), buckets) == seq for it in items):
                raise SystemExit(f"no gate item lands in bucket {seq}")
            wrapper = LayaWrapper(dm, seq).eval()
            f32 = fp32_gate(wrapper, items, refs, seq, buckets)
            print(f"bucket {seq}: fp32 wrapper vs laya's forward, max |dlogit| {f32:.1e}; converting...",
                  flush=True)
            example = inputs_for(items[0], seq)
            pkg = convert_bucket(wrapper, example, seq, workdir)
            ane, total, off = plan_check(pkg)
            res = parity_check(pkg, items, refs, seq, buckets)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            print(f"bucket {seq}: compute plan {ane}/{total} ops on the ANE (off: {off})")
            for label, r in res.items():
                print(f"bucket {seq} [{label}]: n={r['n']} max |dp| {r['dp_max']:.4f} "
                      f"max |dlogit| {r['dlogit_max']:.3f} flips {r['flips']} near-ties {r['near_ties']} "
                      f"pad {r['pad_dlogit']:.1e} {r['ms_median']:.1f}ms")
            print(f"bucket {seq} -> {dest}", flush=True)
            report[seq] = res

    shutil.copy(src / "tokenizer" / "tokenizer.json", install_dir / "tokenizer.json")
    shutil.copy(MANIFEST, install_dir / "classifier.toml")
    print(f"installed classifier.toml + tokenizer -> {install_dir}")


if __name__ == "__main__":
    main()
