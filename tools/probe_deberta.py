"""Probe a DeBERTa-v2/v3 encoder for the Apple Neural Engine before writing its converter.

DeBERTa's disentangled attention adds two relative-position terms to the
attention scores: content-to-position (c2p) and position-to-content (p2c).
transformers builds each by projecting a table of relative-position
embeddings and gathering, for every query/key pair, the row for their
log-bucketed distance (`torch.gather` with an index that depends only on
q - k). Neither form survives Core ML as is:

  - transformers' attention doesn't trace under coremltools 9. Its
    TorchScript helpers and its arithmetic on tensor sizes hit coremltools'
    `int` conversion bug ("only 0-dimensional arrays can be converted to
    Python scalars").
  - Made traceable, each gather becomes `gather_along_axis`. On macOS 27 all
    of them run on the CPU and take the matmuls that feed them along, with an
    ANE/CPU hand-off per gather. ane_check rejects the plan, since a matmul is
    off the ANE.

The rewrite this tool verifies is gather-free. For a static bucket of length
L, the gather index depends only on the distance d = q - k, which takes the
2L - 1 values -(L-1)..(L-1). So each layer's position table is expanded over
those distances once, as a weight-only constant (Kfull, Qfull: 2L - 1 rows),
and the scores are read off a relative shift ("skew") of Q·Kfullᵀ and
K·Qfullᵀ that uses only reshape and slice:

    skew(x)[r, c] = x[r, c - r + L - 1]    for x of shape (BH, L, 2L - 1)

It equals transformers' output in fp32. Two more changes make the graph
fp16-safe: the mask fill is -30000 instead of finfo.min (which is -inf in
fp16), and pad queries attend to themselves (transformers' pairwise mask
masks their whole row).

Heads:
  --head cls      the encoder's [CLS] vector (any DeBERTa-v2/v3 snapshot)
  --head gliner2  a GLiNER2 checkpoint's classification logits at its [L]
                  label markers: the encoder plus the checkpoint's per-token
                  `classifier` MLP, output as logits [1, L]. Cases come from
                  fastino/fast-decisions (--corpus), laid out by the gliner2
                  package's own processor.

Measured on an M1 Max, macOS 27.0, coremltools 9.0 (September 2026):

  model                                bucket   ANE ops       ANE parity      CPU_ONLY parity
  deberta-v3-small, cls                128-512  262/278       cos 1.000000    cos 0.999994
  GLiNER2.5-Decide encoder (v3-large)  128      1036/1052     cos 0.999999    cos 0.999981

GLiNER2.5-Decide with --head gliner2, [L] logits on fast-decisions against
gliner2's own fp32 scoring (1039/1055 ops on the ANE at every bucket):

  bucket  rows/tasks  CPU_AND_NE: argmax, max |Δlogit|, max Δp  CPU_ONLY: argmax, max Δp
  128     92/110      110/110, 0.036, 0.0073                   110/110, 0.0101
  256     204/348     348/348, 0.081, 0.0082                   348/348, 0.0270
  512     204/348     348/348, 0.066, 0.0092                   348/348, 0.0270

Pad invariance is exact everywhere. The CPU ops are the embedding gather and
mask plumbing. In fp32 the rewritten graph, padded, matches gliner2's own
unpadded scoring (every argmax, max |Δlogit| < 1e-4). Neither --gelu twice
(ANE Δp 0.0089 at 128) nor --rescale-mlp 8 (0.0065, with |Δlogit| 0.048)
improved the ANE path. With --bias gather, deberta-v3-small's plan puts all
12 gathers and 6 matmuls on the CPU.

Converting takes 30-47 s, at about 10 GB peak memory. The ANE compile takes
18 s at 128, 32 s at 256 and 129 s at 512, once per artifact path:
Core ML caches the compiled bundle per executable, and a later process
loads in 0.2 s.

Usage:
    python tools/probe_deberta.py <model-dir> <seq> [options]

    <model-dir>        a DeBERTa-v2/v3 snapshot (config, weights, tokenizer), or a
                       GLiNER2 checkpoint (encoder_config/ + model.safetensors)
    <seq>              the static bucket length
    --head cls|gliner2
    --corpus DIR       fast-decisions snapshot, for --head gliner2
                       (hf download fastino/fast-decisions --repo-type dataset
                        --revision 1a33070cabf94ce2e29105482dd2ef6c157ad7f2 --local-dir DIR)
    --per-file N       fast-decisions rows per domain file (default 6)
    --bias skew|gather the rewrite, or transformers' gathers made traceable
    --gelu native|twice
                       Core ML's gelu op, or x·(1 + erf(x/√2)) with the 0.5 folded
                       into each layer's output.dense (no gelu op survives)
    --rescale-mlp S    scale each MLP down projection's input by S (a power of two) and
                       its weights by 1/S
    --convert DIR      also convert, compile into DIR, and report the compute
                       plan and parity on CPU_ONLY and CPU_AND_NE

Core ML caches compiled bundles per executable under
~/Library/Caches/<executable>/com.apple.e5rt.e5bundlecache; running with
CFFIXED_USER_HOME=<dir> puts that cache under <dir> instead.

Requires: torch, transformers, coremltools, safetensors, numpy, and for
--head gliner2 the gliner2 package (arm64-native Python).
"""

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.deberta_v2 import modeling_deberta_v2 as md

MASK_FILL = -30000.0

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Quarterly financial earnings exceeded analyst expectations by 12%.",
    "def add(a, b):\n    return a + b  # helper",
    "Visit https://example.com/path?q=1&r=2 for details, or call 555-0100.",
    "Is this review positive or negative? The food was cold but the staff were lovely.",
    "Die Katze sitzt auf der Matte. Le chat est sur le tapis.",
    " ".join(["token"] * 60),
    "Customer wants a refund for order #88213 after the package arrived damaged; escalate?",
]


# ---- loading ------------------------------------------------------------------------------

def load_encoder(src):
    """A DeBERTa-v2/v3 snapshot, or the encoder of a GLiNER2 checkpoint."""
    src = Path(src)
    if (src / "encoder_config").is_dir():
        from safetensors.torch import load_file
        model = AutoModel.from_config(AutoConfig.from_pretrained(src / "encoder_config"))
        weights = load_file(src / "model.safetensors")
        sd = {k[len("encoder."):]: v.clone() for k, v in weights.items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or any("position_ids" not in m for m in missing):
            raise SystemExit(f"encoder weights don't match its config: {missing} {unexpected}")
    else:
        model = AutoModel.from_pretrained(src, dtype=torch.float32)
    return model.float().eval()


# ---- the rewrite ----------------------------------------------------------------------------

def rel_index(d, span, max_pos):
    """transformers' c2p gather index for distance d = q - k: clamp(log_bucket(d) + span, 0, 2*span - 1)."""
    b = md.make_log_bucket_position(torch.as_tensor(d, dtype=torch.long), span, max_pos).long()
    return torch.clamp(b + span, 0, 2 * span - 1)


def skew(x, L: int):
    """(BH, L, 2L-1) -> (BH, L, L), out[r, c] = x[r, c - r + L - 1], by reshape and slice only.

    L must be a Python int: sizes read from tensors trace as arithmetic and hit coremltools'
    `int` conversion bug."""
    flat = x.reshape(-1, L * (2 * L - 1))[:, L - 1 : L - 1 + L * (2 * L - 2)]
    return flat.reshape(-1, L, 2 * L - 2)[:, :, :L]


def install(model, bias, seq):
    """Replace the attention with a traceable, fp16-safe form for bucket `seq`."""
    enc = model.encoder
    att0 = enc.layer[0].attention.self
    if not (att0.relative_attention and att0.share_att_key
            and set(att0.pos_att_type) == {"c2p", "p2c"}):
        raise SystemExit("only relative attention with share_att_key and both c2p and p2c is handled")
    span, max_pos = att0.pos_ebd_size, att0.max_relative_positions
    with torch.no_grad():
        rel = enc.get_rel_embedding()  # (2*span, hidden), LayerNorm applied
        if bias == "skew":
            e = torch.arange(2 * seq - 1)
            c2p_idx = rel_index((seq - 1) - e, span, max_pos)  # column e of Q·Kfullᵀ is d = (L-1) - e
            p2c_idx = rel_index(e - (seq - 1), span, max_pos)  # column e of K·Qfullᵀ is d = e - (L-1)
            for layer in enc.layer:
                a = layer.attention.self
                a.register_buffer("kfull", a.key_proj(rel[c2p_idx]))
                a.register_buffer("qfull", a.query_proj(rel[p2c_idx]))
        else:
            i = torch.arange(seq)
            d = i[:, None] - i[None, :]
            for layer in enc.layer:
                a = layer.attention.self
                a.register_buffer("kpos", a.key_proj(rel))
                a.register_buffer("qpos", a.query_proj(rel))
                a.register_buffer("c2p_pos", rel_index(d, span, max_pos))   # gathered per query row
                a.register_buffer("p2c_pos", rel_index(-d, span, max_pos))  # per key row, then transposed

    def forward(self, hidden_states, attention_mask, output_attentions=False, query_states=None,
                relative_pos=None, rel_embeddings=None):
        H, dh, L = self.num_attention_heads, self.attention_head_size, seq
        q = self.transpose_for_scores(self.query_proj(hidden_states), H)  # (BH, L, dh)
        k = self.transpose_for_scores(self.key_proj(hidden_states), H)
        v = self.transpose_for_scores(self.value_proj(hidden_states), H)
        scores = torch.bmm(q, k.transpose(-1, -2))
        if bias == "skew":
            kf = self.transpose_for_scores(self.kfull.unsqueeze(0), H)  # (H, 2L-1, dh)
            qf = self.transpose_for_scores(self.qfull.unsqueeze(0), H)
            c2p = skew(torch.bmm(q, kf.transpose(-1, -2)), L)
            p2c = skew(torch.bmm(k, qf.transpose(-1, -2)), L).transpose(-1, -2)
        else:
            kp = self.transpose_for_scores(self.kpos.unsqueeze(0), H)
            qp = self.transpose_for_scores(self.qpos.unsqueeze(0), H)
            c2p = torch.gather(torch.bmm(q, kp.transpose(-1, -2)), -1, self.c2p_pos.expand(H, L, L))
            p2c = torch.gather(torch.bmm(k, qp.transpose(-1, -2)), -1,
                               self.p2c_pos.expand(H, L, L)).transpose(-1, -2)
        scores = (scores + c2p + p2c) / math.sqrt(3 * dh)  # a literal, not q.size(-1)
        scores = scores.view(-1, H, L, L).masked_fill(~attention_mask.bool(), MASK_FILL)
        ctx = torch.bmm(torch.softmax(scores, dim=-1).view(-1, L, L), v)
        ctx = ctx.view(-1, H, L, dh).permute(0, 2, 1, 3).contiguous()
        return (ctx.view(ctx.size(0), L, H * dh), None)

    def get_attention_mask(self, attention_mask):
        m = attention_mask.unsqueeze(1).unsqueeze(2)
        m = m * m.squeeze(-2).unsqueeze(-1)
        return torch.maximum(m, torch.eye(seq, dtype=m.dtype).view(1, 1, seq, seq))

    md.DisentangledSelfAttention.forward = forward
    md.DebertaV2Encoder.get_attention_mask = get_attention_mask


class TwiceGelu(torch.nn.Module):
    """2·GELU(x) = x·(1 + erf(x/√2)); the 0.5 is folded into the next linear's weights.

    Core ML's native gelu op is coarse on the ANE (up to ~6e-3 off on [-1, 1]). Written with
    the 0.5 in the graph, coremltools fuses the expression straight back into that op."""

    def forward(self, x):
        return x * (1.0 + torch.erf(x * 0.7071067811865476))


class ScaledAct(torch.nn.Module):
    def __init__(self, act, scale):
        super().__init__()
        self.act, self.scale = act, scale

    def forward(self, x):
        return self.act(x) * self.scale


def rescale_mlp(encoder, scale):
    """Run each MLP down projection on a `scale`-times larger input (a power of two, exact in
    fp16) and divide its weights by the same factor. The ANE linear's precision floor is
    absolute on its input, and these inputs sit at rms 0.06-0.3."""
    with torch.no_grad():
        for layer in encoder.encoder.layer:
            layer.intermediate.intermediate_act_fn = ScaledAct(layer.intermediate.intermediate_act_fn, scale)
            layer.output.dense.weight.div_(scale)  # bias unchanged


def twice_gelu(encoder):
    with torch.no_grad():
        for layer in encoder.encoder.layer:
            layer.intermediate.intermediate_act_fn = TwiceGelu()
            layer.output.dense.weight.mul_(0.5)  # bias unchanged


# ---- heads and cases ----------------------------------------------------------------------

class ClsHead(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder, self.hidden = encoder, encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        return h[:, 0, :].reshape(1, self.hidden)  # a literal size: a traced one breaks coremltools


class GlinerHead(torch.nn.Module):
    """GLiNER2's classifier applied to every token; the host reads the [L] positions it placed."""

    def __init__(self, encoder, classifier, seq):
        super().__init__()
        self.encoder, self.classifier, self.seq = encoder, classifier, seq

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
        return self.classifier(h).reshape(1, self.seq)


def cls_cases(src, seq):
    tok = AutoTokenizer.from_pretrained(src)
    ref = load_encoder(src)  # before install(): stock transformers, unpadded
    out = []
    with torch.no_grad():
        for t in TEXTS:
            ids = np.array(tok(t, truncation=True, max_length=seq)["input_ids"], np.int32)
            out.append((ids, [ref(input_ids=torch.tensor(ids[None]).long()).last_hidden_state[0, 0].numpy()]))
    return out, None


def gliner_cases(src, seq, corpus, per_file):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from gliner2.classification import Classifier, ClassificationSchema, compile_schema

    clf = Classifier.from_pretrained(str(src), device="cpu", dtype=torch.float32).eval()
    gm, proc = clf.scorer.model, clf.scorer.processor
    out = []
    for f in sorted(Path(corpus).glob("*.jsonl")):
        kept = 0
        for line in open(f):
            if kept >= per_file:
                break
            row = json.loads(line)
            schema = ClassificationSchema()
            for t in row["output"]["classifications"]:
                (schema.multi if t.get("multi_label") else schema.single)(t["task"], t["labels"])
            b = proc.collate_fn_inference([(row["input"], compile_schema(schema).build())])
            ids = b.input_ids[0].numpy().astype(np.int32)
            if len(ids) > seq:
                continue
            tasks = [([int(p) for p in b.schema_special_indices[0][j]][1:],  # drop the [P] prompt marker
                      bool(row["output"]["classifications"][j].get("multi_label")))
                     for j in range(b.schema_counts[0])]
            with torch.no_grad():  # gliner2's own scoring: unpadded encoder, classifier at [L]
                h = gm.encoder(input_ids=torch.tensor(ids[None]).long(),
                               attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
                refs = [gm.classifier(h[pos]).squeeze(-1).numpy() for pos, _ in tasks]
            out.append((ids, refs, tasks))
            kept += 1
    return out, gm.classifier


# ---- metrics ------------------------------------------------------------------------------

def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def worst(values):
    """NaN-safe minimum: any NaN wins."""
    values = list(values)
    return float("nan") if any(v != v for v in values) else min(values)


def activation(z, multi):
    z = np.asarray(z, np.float64)
    if multi:
        return 1 / (1 + np.exp(-z))
    e = np.exp(z - z.max())
    return e / e.sum()


def feeds(ids, seq, pad_ids=None):
    x = np.zeros((1, seq), np.int32)
    if pad_ids is not None:
        x[0] = pad_ids
    x[0, : len(ids)] = ids
    m = np.zeros((1, seq), np.int32)
    m[0, : len(ids)] = 1
    return x, m


def grade(label, head_kind, cases, run):
    if head_kind == "cls":
        c = worst(cos(run(ids), refs[0]) for ids, refs in cases)
        print(f"{label}: worst CLS cosine vs fp32 {c:.7f}")
        return
    n = agree = flips = 0
    dz = dp = 0.0
    for ids, refs, tasks in cases:
        z = run(ids)
        if not np.isfinite(z).all():
            print(f"{label}: NON-FINITE output")
            return
        for (pos, multi), r in zip(tasks, refs):
            got = z[pos]
            n += 1
            dz = max(dz, float(np.abs(got - r).max()))
            dp = max(dp, float(np.abs(activation(got, multi) - activation(r, multi)).max()))
            if int(got.argmax()) == int(r.argmax()):
                agree += 1
            else:
                top = np.sort(r)
                flips += int(top[-1] - top[-2] >= 0.05)  # D28: near-ties aren't graded
    print(f"{label}: {n} tasks, argmax agreement {agree}/{n}, graded flips {flips}, "
          f"max |dlogit| {dz:.4f}, max dp {dp:.5f}")


def plan_summary(mlmodelc):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(path=str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    devices, off, order = Counter(), Counter(), []

    def walk(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            name = op.operator_name.split(".")[-1]
            if usage is not None:
                dev = type(usage.preferred_compute_device).__name__
                dev = "ANE" if "NeuralEngine" in dev else "GPU" if "GPU" in dev else "CPU"
                devices[dev] += 1
                order.append(dev)
                if dev != "ANE":
                    off[name] += 1
            for nested in op.blocks:
                walk(nested)

    walk(plan.model_structure.program.functions["main"].block)
    handoffs = sum(1 for a, b in zip(order, order[1:]) if a != b)
    total = sum(devices.values())
    print(f"compute plan: ANE {devices['ANE']}/{total} ({100 * devices['ANE'] / max(total, 1):.1f}%), "
          f"{handoffs} device changes along main; off the ANE: {dict(off.most_common())}")


# ---- main ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("seq", type=int)
    ap.add_argument("--head", choices=["cls", "gliner2"], default="cls")
    ap.add_argument("--corpus")
    ap.add_argument("--per-file", type=int, default=6)
    ap.add_argument("--bias", choices=["skew", "gather"], default="skew")
    ap.add_argument("--gelu", choices=["native", "twice"], default="native")
    ap.add_argument("--rescale-mlp", type=float, default=1.0)
    ap.add_argument("--convert", metavar="DIR")
    a = ap.parse_args()
    seq = a.seq

    if a.head == "gliner2":
        if not a.corpus:
            raise SystemExit("--head gliner2 needs --corpus (a fastino/fast-decisions snapshot)")
        cases, classifier = gliner_cases(a.src, seq, a.corpus, a.per_file)
    else:
        cases, classifier = cls_cases(a.src, seq)
    if not cases:
        raise SystemExit(f"no case fits a bucket of {seq}")
    print(f"{len(cases)} cases, {min(len(c[0]) for c in cases)}-{max(len(c[0]) for c in cases)} tokens")

    encoder = load_encoder(a.src)
    install(encoder, a.bias, seq)
    if a.gelu == "twice":
        twice_gelu(encoder)
    if a.rescale_mlp != 1.0:
        rescale_mlp(encoder, a.rescale_mlp)
    head = (GlinerHead(encoder, classifier, seq) if a.head == "gliner2" else ClsHead(encoder)).eval()

    with torch.no_grad():
        grade(f"fp32, {a.bias} bias, padded to {seq}", a.head, cases,
              lambda ids: head(*map(torch.tensor, feeds(ids, seq))).numpy()[0])
    if not a.convert:
        return

    import coremltools as ct

    out = Path(a.convert)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.head}_{a.bias}_{a.gelu}_x{a.rescale_mlp:g}_{seq}"
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(head, tuple(map(torch.tensor, feeds(cases[0][0], seq))))
    ml = ct.convert(traced,
                    inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                            ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32)],
                    outputs=[ct.TensorType(name="logits" if a.head == "gliner2" else "embedding")],
                    convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15)
    ops = Counter(op.op_type for f in ml._mil_program.functions.values() for op in f.operations)
    print(f"converted in {time.time() - t:.0f}s; gather_along_axis {ops['gather_along_axis']}, gelu {ops['gelu']}, mul {ops['mul']}")
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / f"{name}.mlpackage"
        ml.save(str(pkg))
        del ml
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True, capture_output=True)
        dst = out / f"{name}.mlmodelc"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(Path(tmp) / f"{name}.mlmodelc"), dst)
    t = time.time()
    plan_summary(dst)
    print(f"(plan read, which compiles for the ANE: {time.time() - t:.0f}s)")

    output = "logits" if a.head == "gliner2" else "embedding"
    noisy = np.random.default_rng(1).integers(1000, 100000, seq).astype(np.int32)
    kept = []
    for units in ("CPU_ONLY", "CPU_AND_NE"):
        t = time.time()
        model = ct.models.CompiledMLModel(str(dst), compute_units=getattr(ct.ComputeUnit, units))
        print(f"{units}: loaded in {time.time() - t:.0f}s")

        # Core ML releases a prediction's input about a second after the model
        # goes idle, from its own queue; coremltools hands it NumPy-backed
        # buffers that Python may already have freed by then (a segfault). Keep
        # every input alive for the life of the process.
        def run(ids, pads=None):
            x, m = feeds(ids, seq, pads)
            kept.append({"input_ids": x, "attention_mask": m})
            return model.predict(kept[-1])[output][0]

        grade(units, a.head, cases, run)
        if a.head == "gliner2":  # per-token logits: compare the real tokens only
            pad = max(float(np.abs(run(c[0])[: len(c[0])] - run(c[0], noisy)[: len(c[0])]).max()) for c in cases)
            print(f"{units}: pad invariance (max |dlogit| over real tokens, pad ids 0 vs random) {pad:.6f}")
        else:
            pad = worst(cos(run(c[0]), run(c[0], noisy)) for c in cases)
            print(f"{units}: pad invariance (cosine, pad ids 0 vs random) {pad:.7f}")


if __name__ == "__main__":
    main()
