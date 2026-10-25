"""convert.py <hf_src> <out_dir> --cal <calibration.json> [--C 2048] [--cuts 0-30] [--chunk i]
                [--funcs decode,prefill:128,prefill:512] [--quant none|int8|int8:block=32|int4|pal4|pal8[:group=16]]
                [--head-split 1] [--silu stable|native] [--rescale <rescale.json>]

Converts a LlamaForCausalLM to one multifunction Core ML program per chunk
(llama_ane.py): every function of a chunk shares its weights (deduplicated by
save_multifunction) and its KV-cache states. Writes chunk_<i>.mlmodelc,
embed.npy (the token table / K, fp16, gathered on the host) and meta.json.

Memory: the checkpoint is held in bf16 and only the chunk being converted is
built in fp32, so a 2B model converts in well under its 10 GB fp32 size.
--chunk i converts one chunk (run one process per chunk to bound the peak);
the calibration (llama_ane.stream_calibrate, as reference.py writes it) is
read from --cal, or computed and written there if missing.
--rescale applies power-of-two input rescales to chosen projections
(llama_ane.apply_rescales; D17/D19's small-input rule). --qweights takes
gptq.py's weights, already on the int8 grid that --quant int8 then stores."""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA
import texts as T

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--cal", required=True)
ap.add_argument("--C", type=int, default=2048)
ap.add_argument("--cuts", default=None, help="e.g. 0-14,14-24,24-24 (last: head-only chunk)")
ap.add_argument("--chunk", default=None, help="comma-separated chunk indices to convert (default all)")
ap.add_argument("--funcs", default="decode,prefill:128,prefill:512")
ap.add_argument("--quant", default="none")
ap.add_argument("--head-split", type=int, default=1)
ap.add_argument("--silu", default="stable")
ap.add_argument("--rescale", default=None)
ap.add_argument("--qweights", default=None, help="gptq.py output: weights already on the int8 grid")
a = ap.parse_args()

import coremltools as ct
import coremltools.optimize as cto

torch.set_grad_enabled(False)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.src)
hf = LA.load_hf(a.src, dtype=torch.bfloat16)
cfg = hf.config
L = cfg.num_hidden_layers
cuts = [(0, L)] if not a.cuts else [tuple(map(int, c.split("-"))) for c in a.cuts.split(",")]
todo = range(len(cuts)) if a.chunk is None else [int(x) for x in a.chunk.split(",")]

cal_path = Path(a.cal)
if cal_path.exists():
    cal = json.load(open(cal_path))
else:
    cal = LA.stream_calibrate(hf, T.calib_texts(tok), silu=a.silu)
    json.dump(cal, open(cal_path, "w"))
K = cal["K"]
rescale = json.load(open(a.rescale)) if a.rescale else None
if not (out / "embed.npy").exists():
    np.save(out / "embed.npy", (hf.model.embed_tokens.weight.float() / K).half().numpy())

funcs = []
for f in a.funcs.split(","):
    funcs.append(("decode", 1) if f == "decode" else (f"prefill_{f.split(':')[1]}", int(f.split(":")[1])))
D, Dh = cfg.hidden_size, cfg.head_dim
head_only = any(lo == hi == L for lo, hi in cuts)


def quant_spec(q):
    name, _, opts = q.partition(":")
    kw = dict(kv.split("=") for kv in opts.split(",") if kv)
    return name, {k: int(v) for k, v in kw.items()}


def quantize(ml):
    name, kw = quant_spec(a.quant)
    if name == "none":
        return ml
    if name in ("int8", "int4"):
        if "block" in kw:
            op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=name, granularity="per_block",
                                                    block_size=kw["block"], weight_threshold=2048)
        elif name == "int8":
            op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=name, granularity="per_channel",
                                                    weight_threshold=2048)
        else:
            op = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=name, granularity="per_block",
                                                    block_size=32, weight_threshold=2048)
        return cto.coreml.linear_quantize_weights(ml, config=cto.coreml.OptimizationConfig(global_config=op))
    if name in ("pal4", "pal8"):
        op = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=int(name[3:]), granularity="per_grouped_channel",
                                            group_size=kw.get("group", 16), weight_threshold=2048,
                                            num_kmeans_workers=1)
        return cto.coreml.palettize_weights(ml, config=cto.coreml.OptimizationConfig(global_config=op))
    raise SystemExit(f"unknown quant {a.quant}")


meta_path = out / "meta.json"
meta = json.load(open(meta_path)) if meta_path.exists() else {}
meta.update({"src": a.src, "C": a.C, "cuts": cuts, "funcs": [f for f, _ in funcs], "quant": a.quant, "K": K,
             "silu": a.silu, "head_split": a.head_split, "hidden": D, "head_dim": Dh, "rope_theta": cfg.rope_theta,
             "vocab": cfg.vocab_size, "layers": L, "kv_heads": cfg.num_key_value_heads,
             "rescale": a.rescale and Path(a.rescale).name, "qweights": a.qweights and Path(a.qweights).name})
if len(meta.get("chunks", [])) != len(cuts):
    meta["chunks"] = [None] * len(cuts)
tmp = out / "tmp"
tmp.mkdir(exist_ok=True)
for ci in todo:
    lo, hi = cuts[ci]
    has_head = (hi == L and not head_only) or (lo == hi == L)
    layers = [LA.build_layer(hf, i, cal, a.silu) for i in range(lo, hi)]
    if a.qweights:
        import gptq
        qmeta = json.load(open(Path(a.qweights) / "meta.json"))
        for i, layer in zip(range(lo, hi), layers):
            for name, (q, s) in torch.load(Path(a.qweights) / f"layer_{i}.pt").items():
                conv = getattr(layer, name)
                conv.weight.data = gptq.dequant(q, s, qmeta["block"]).view_as(conv.weight)
        if has_head and qmeta["head"]:
            q, s = torch.load(Path(a.qweights) / "head.pt")["head"]
            hf.lm_head.weight.data = gptq.dequant(q, s, qmeta["block"])
    if rescale is not None:
        for i, layer in zip(range(lo, hi), layers):
            LA.apply_rescales(layer, rescale.get(str(i), {}))
    c = LA.Chunk(layers, cfg, lo, hi, a.C, LA.final_norm(hf, cal) if has_head else None,
                 hf.lm_head if has_head else None, a.head_split)
    c.K = float(K)
    c.eval()
    parts = []
    for fname, S in funcs:
        t0 = time.time()
        c.S = S
        c.reset()
        ex = [torch.zeros(1, D, 1, S), torch.tensor([0], dtype=torch.int32), LA.mask_for(0, S, a.C),
              torch.zeros(S, Dh), torch.zeros(S, Dh)]
        inputs = [ct.TensorType("hidden", (1, D, 1, S), np.float16), ct.TensorType("pos", (1,), np.int32),
                  ct.TensorType("mask", (1, 1, S, a.C), np.float16), ct.TensorType("cos", (S, Dh), np.float16),
                  ct.TensorType("sin", (S, Dh), np.float16)]
        if has_head and S > 1:
            ex.append(torch.zeros(S))
            inputs.append(ct.TensorType("sel", (S,), np.float16))
        tr = torch.jit.trace(c, tuple(ex), check_trace=False)
        ml = ct.convert(tr, inputs=inputs,
                        outputs=[ct.TensorType(name="logits" if has_head else "hidden_out", dtype=np.float16)],
                        states=[ct.StateType(ct.TensorType(shape=tuple(c.k_cache.shape), dtype=np.float16), name="k_cache"),
                                ct.StateType(ct.TensorType(shape=tuple(c.v_cache.shape), dtype=np.float16), name="v_cache")],
                        minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        del tr
        ml = quantize(ml)
        p = tmp / f"chunk_{ci}_{fname}.mlpackage"
        ml.save(str(p))
        parts.append((fname, p))
        del ml
        gc.collect()
        print(f"chunk {ci} ({lo}-{hi}{' +head' if has_head else ''}) {fname}: {time.time() - t0:.0f}s", flush=True)
    del c, layers
    gc.collect()
    desc = ct.utils.MultiFunctionDescriptor()
    for fname, p in parts:
        desc.add_function(str(p), src_function_name="main", target_function_name=fname)
    desc.default_function_name = parts[0][0]
    pkg = out / f"chunk_{ci}.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    ct.utils.save_multifunction(desc, str(pkg))
    mc = out / f"chunk_{ci}.mlmodelc"
    if mc.exists():
        shutil.rmtree(mc)
    shutil.move(ct.utils.compile_model(str(pkg)), str(mc))
    wb = sum(f.stat().st_size for f in (mc / "weights").rglob("*") if f.is_file())
    meta["chunks"][ci] = {"lo": lo, "hi": hi, "head": has_head, "weight_bytes": wb,
                          "cache_shape": [hi - lo, cfg.num_key_value_heads, a.C, Dh]}
    print(f"chunk {ci}: weights {wb / 2**30:.3f} GiB", flush=True)
    shutil.rmtree(pkg)
    for _, p in parts:
        shutil.rmtree(p)
    json.dump(meta, open(meta_path, "w"), indent=1)
shutil.rmtree(tmp)
print("done", flush=True)
