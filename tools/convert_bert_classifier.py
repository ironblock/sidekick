"""Convert a BERT text-classification checkpoint into ANE-resident Core ML
classifier artifacts for sidekick's `POST /v1/classify`
(docs/design/classify.md). Validated on
nlptown/bert-base-multilingual-uncased-sentiment.

Produces one static-shape .mlmodelc per sequence-length bucket whose output
is the model's logits, statically (1, num_labels), in id2label order.

Usage:
    python tools/convert_bert_classifier.py <hf-model-dir> <install-dir> [buckets...]

    hf-model-dir: local snapshot of the checkpoint (config.json,
                  model.safetensors, vocab.txt or tokenizer.json)
    install-dir:  classifier directory the daemon scans. Its name is the
                  classifier id: the manifest is copied from
                  examples/classifiers/<name>/classifier.toml, e.g.
                  "~/Library/Application Support/sidekick/models/nlptown-sentiment"
    buckets:      default 128 256 512

Requires: torch, transformers, tokenizers, coremltools, numpy (arm64-native
Python), plus Xcode for `xcrun coremlcompiler`.

Conversion constraints (the bge-small recipe, D15, plus D25's lessons):

A. EXPLICIT ATTENTION WITH A FINITE MASK. transformers' sdpa path calls
   F.scaled_dot_product_attention without a scale, which coremltools turns
   into its fused attention op; D25 found that op dropping masks on the ANE.
   The eager path is explicit (matmul -> softmax -> matmul), but its
   extended attention mask is finfo(float32).min, -inf in fp16, which NaNs
   softmax (D15). get_extended_attention_mask is patched to the fp16-safe
   additive -30000; convert_bucket() fails if the fused op appears.
B. Static shapes per bucket, explicit position_ids and token_type_ids
   buffers (coremltools can't cast traced sizes to int under static shapes).
C. TOKENIZER. sidekick loads tokenizer.json. When the checkpoint ships only
   vocab.txt (nlptown does), the converter builds the fast tokenizer that
   transformers derives from it and saves it as tokenizer.json; the
   reference generator (tools/classifier_reference.py) tokenizes with that
   same file.
D. ANE RANGE. The ANE's linear op saturates above 2^15 (D25). The converter
   measures every linear's largest output on its gate set and fails if one
   exceeds 0.85 x 2^15. BERT is post-norm, so D25's residual rewrite
   doesn't apply as written.

Gates, per bucket: fp32 (the wrapper reproduces the checkpoint's forward,
max |dlogit| <= FP32_TOL); no fused attention op; compute plan (every
linear/matmul on the ANE and >= 80% of ops); on CPU_AND_NE and CPU_ONLY,
finite logits, argmax agreement with fp32 wherever fp32's top-2 margin is
>= MARGIN, max raw |dp| <= DP_GATE, and pad invariance. Every gate treats
NaN as a failure.

Measured on nlptown (M1 Max, macOS 27.0), with tools/classifier_reference.py
and tools/measure_classifier.py on the 51-input parity corpus against the
checkpoint in fp32: argmax agreement 100% on the ANE, CPU and GPU (46 cases
above the 0.05-logit margin; the 5 near-ties agree too); raw |dp| max 0.0026
on the ANE, 0.0034 on the CPU, 0.0007 on the GPU; 294 of 304 operations on
the ANE; ANE latency 4.3 / 10.4 / 27.3 ms at buckets 128 / 256 / 512 (CPU
18.9 / 32.3 / 62.3 ms). The generated tokenizer.json matches transformers'
slow BertTokenizer on accented, CJK, Cyrillic, emoji and special-token text.
"""

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
import coremltools as ct
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
MASK_ADD = -30000.0
ANE_LINEAR_MAX = 32768.0
LINEAR_HEADROOM = 0.85
FP32_TOL = 1e-3
MARGIN = 0.05
DP_GATE = 0.02

_REVIEW = ("The blender arrived quickly and works well, although the lid is a bit loose and "
           "the instructions were confusing at first.")
GATE_TEXTS = [
    "Absolutely terrible. It broke after two days and support never answered.",
    "Not bad, not great. It does the job.",
    "Great value for the money, I would buy it again!",
    "Das Produkt ist in Ordnung, aber die Lieferung hat zu lange gedauert.",
    "Producto excelente, llegó antes de lo previsto.",
    "Service client décevant, je ne recommande pas.",
    "Prodotto perfetto, lo consiglio a tutti.",
    "Het werkt prima, maar de batterij is snel leeg.",
    " ".join([_REVIEW] * 6),
    " ".join([_REVIEW] * 14),
    " ".join([_REVIEW] * 17),
]


def fp16_safe_extended_mask(self, attention_mask, input_shape=None, dtype=None, **_):
    # constraint A: the geometry of ModuleUtilsMixin.get_extended_attention_mask
    # for an encoder, (bsz, 1, 1, seq), with a finite additive constant
    return (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * MASK_ADD


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model, seq, num_labels):
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.register_buffer("position_ids", torch.arange(seq, dtype=torch.long).unsqueeze(0))
        self.register_buffer("token_type_ids", torch.zeros((1, seq), dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids.long(), attention_mask=attention_mask.long(),
                         token_type_ids=self.token_type_ids, position_ids=self.position_ids)
        return out.logits.reshape(1, self.num_labels)


def load_tokenizer(src, install_dir):
    """constraint C: tokenizer.json from the checkpoint, or built from vocab.txt."""
    dest = install_dir / "tokenizer.json"
    if (src / "tokenizer.json").exists():
        shutil.copy(src / "tokenizer.json", dest)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        if not tok.is_fast:
            raise SystemExit("no fast tokenizer can be built for this checkpoint")
        tok.backend_tokenizer.save(str(dest))
        print(f"generated tokenizer.json from {', '.join(p.name for p in src.glob('vocab*'))}")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(dest))


def encode(tok, text, max_len):
    ids = tok.encode(text, add_special_tokens=True).ids
    if len(ids) > max_len:
        raise SystemExit(f"gate text longer than {max_len} tokens")
    return ids


def bucket_of(n, buckets):
    return next(b for b in buckets if n <= b)


def inputs_for(ids, seq, pad_ids=None):
    n = len(ids)
    x = np.zeros((1, seq), dtype=np.int32)
    x[0, :n] = ids
    if pad_ids is not None:
        x[0, n:] = pad_ids[: seq - n]
    mask = np.zeros((1, seq), dtype=np.int32)
    mask[0, :n] = 1
    return {"input_ids": x, "attention_mask": mask}


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def reference_and_maxima(model, items):
    """fp32 logits of the checkpoint (unpadded), and every linear's largest
    output (constraint D)."""
    maxima = {}
    hooks = [m.register_forward_hook(
        lambda mod, a, out, name=name: maxima.__setitem__(name, max(maxima.get(name, 0.0),
                                                                    float(out.detach().abs().max()))))
        for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    refs = []
    with torch.no_grad():
        for ids in items:
            t = torch.tensor([ids])
            refs.append(model(input_ids=t, attention_mask=torch.ones_like(t)).logits[0].numpy().astype(np.float64))
    for h in hooks:
        h.remove()
    return refs, maxima


def fp32_gate(wrapper, items, refs, seq, buckets):
    worst = 0.0
    with torch.no_grad():
        for ids, ref in zip(items, refs):
            if bucket_of(len(ids), buckets) != seq:
                continue
            x = {k: torch.from_numpy(v) for k, v in inputs_for(ids, seq).items()}
            out = wrapper(**x)[0].numpy().astype(np.float64)
            worst = max(worst, float(np.abs(out - ref).max()))
    if not worst <= FP32_TOL:
        raise SystemExit(f"seq {seq}: fp32 wrapper vs the checkpoint, max |dlogit| {worst:.2e} > {FP32_TOL}")
    return worst


def convert_bucket(wrapper, seq, workdir):
    ids = torch.zeros((1, seq), dtype=torch.int32)
    ids[0, 0], ids[0, 1] = 101, 102
    mask = torch.zeros((1, seq), dtype=torch.int32)
    mask[0, :2] = 1
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ids, mask))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    ops = {op.type for fn in mlmodel.get_spec().mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations}
    if "scaled_dot_product_attention" in ops:
        raise SystemExit(f"seq {seq}: converted graph contains the fused attention op (constraint A)")
    pkg = Path(workdir) / f"model_{seq}.mlpackage"
    mlmodel.save(str(pkg))
    return pkg


def plan_check(pkg):
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
        if "NeuralEngine" in type(usage.preferred_compute_device).__name__:
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


def parity_check(pkg, items, refs, seq, buckets):
    idx = [i for i, ids in enumerate(items) if bucket_of(len(ids), buckets) == seq]
    results = {}
    for label, cu in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
        m = ct.models.MLModel(str(pkg), compute_units=cu)
        dp_max, dl_max, flips, ms = 0.0, 0.0, 0, []
        for i in idx:
            t0 = time.perf_counter()
            out = m.predict(inputs_for(items[i], seq))["logits"][0].astype(np.float64)
            ms.append((time.perf_counter() - t0) * 1e3)
            if not np.isfinite(out).all():
                raise SystemExit(f"seq {seq} [{label}]: non-finite logits")
            ref = refs[i]
            top2 = np.sort(ref)[-2:]
            if top2[1] - top2[0] >= MARGIN:
                flips += int(np.argmax(out) != np.argmax(ref))
            dp_max = max(dp_max, float(np.abs(softmax(out) - softmax(ref)).max()))
            dl_max = max(dl_max, float(np.abs(out - ref).max()))
        if flips or not dp_max <= DP_GATE:
            raise SystemExit(f"seq {seq} [{label}]: {flips} argmax flips above margin {MARGIN}, "
                             f"max |dp| {dp_max:.4f} (gate {DP_GATE})")
        ids = next(items[i] for i in idx if len(items[i]) < seq)
        a = m.predict(inputs_for(ids, seq))["logits"][0]
        b = m.predict(inputs_for(ids, seq, np.random.default_rng(0).integers(1000, 30000, seq)))["logits"][0]
        pad_d = float(np.abs(a - b).max())
        if not pad_d <= 1e-3:
            raise SystemExit(f"seq {seq} [{label}]: logits depend on pad content (max |dlogit| {pad_d})")
        results[label] = {"n": len(idx), "dp_max": dp_max, "dlogit_max": dl_max, "pad_dlogit": pad_d,
                          "ms_median": float(np.median(ms[1:] or ms))}
    return results


def compile_to_mlmodelc(pkg, install_dir, seq):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), tmp], check=True,
                       stdout=subprocess.DEVNULL)
        compiled = next(Path(tmp).glob("*.mlmodelc"))
        dest = install_dir / f"model_{seq}.mlmodelc"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(compiled), dest)
    return dest


def main():
    src = Path(sys.argv[1]).expanduser()
    install_dir = Path(sys.argv[2]).expanduser()
    buckets = [int(b) for b in sys.argv[3:]] or [128, 256, 512]
    manifest_path = REPO / "examples" / "classifiers" / install_dir.name / "classifier.toml"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}: name the install dir after the classifier id")
    manifest = tomllib.loads(manifest_path.read_text())
    install_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if manifest["classify"]["labels"] != labels:
        raise SystemExit(f"{manifest_path}: labels {manifest['classify']['labels']} != id2label {labels}")
    if manifest["max_seq_len"] > model.config.max_position_embeddings:
        raise SystemExit("max_seq_len exceeds the model's position embeddings")
    tok = load_tokenizer(src, install_dir)
    items = [encode(tok, t, manifest["max_seq_len"]) for t in GATE_TEXTS]
    refs, maxima = reference_and_maxima(model, items)
    top = max(maxima, key=maxima.get)
    print(f"gate set: {len(items)} texts ({min(map(len, items))}-{max(map(len, items))} tokens); "
          f"largest linear output {maxima[top]:.1f} at {top}, "
          f"{ANE_LINEAR_MAX / maxima[top]:.1f}x under the ANE linear's {ANE_LINEAR_MAX:.0f}")
    if maxima[top] > LINEAR_HEADROOM * ANE_LINEAR_MAX:
        raise SystemExit("a linear output is past the ANE linear's range (constraint D)")
    model.get_extended_attention_mask = fp16_safe_extended_mask.__get__(model)
    model.bert.get_extended_attention_mask = fp16_safe_extended_mask.__get__(model.bert)

    with tempfile.TemporaryDirectory() as workdir:
        for seq in buckets:
            if not any(bucket_of(len(ids), buckets) == seq for ids in items):
                raise SystemExit(f"no gate text lands in bucket {seq}")
            wrapper = LogitsWrapper(model, seq, len(labels)).eval()
            f32 = fp32_gate(wrapper, items, refs, seq, buckets)
            print(f"bucket {seq}: fp32 wrapper vs the checkpoint, max |dlogit| {f32:.1e}; converting...", flush=True)
            pkg = convert_bucket(wrapper, seq, workdir)
            ane, total, off = plan_check(pkg)
            res = parity_check(pkg, items, refs, seq, buckets)
            dest = compile_to_mlmodelc(pkg, install_dir, seq)
            print(f"bucket {seq}: compute plan {ane}/{total} ops on the ANE (off: {off})")
            for label, r in res.items():
                print(f"bucket {seq} [{label}]: n={r['n']} max |dp| {r['dp_max']:.4f} "
                      f"max |dlogit| {r['dlogit_max']:.3f} pad {r['pad_dlogit']:.1e} {r['ms_median']:.1f}ms")
            print(f"bucket {seq} -> {dest}", flush=True)

    shutil.copy(manifest_path, install_dir / "classifier.toml")
    print(f"installed classifier.toml + tokenizer.json -> {install_dir}")


if __name__ == "__main__":
    main()
