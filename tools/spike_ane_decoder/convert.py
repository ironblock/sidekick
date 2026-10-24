"""convert.py <hf_src> <out_dir> [--C 2048] [--cuts 0-30] [--funcs decode,prefill:128,prefill:512]
                [--quant none|int8|int4|pal4|pal8] [--head-split 1] [--silu stable|native]

Converts a LlamaForCausalLM to one multifunction Core ML program per chunk
(llama_ane.py): every function of a chunk shares its weights (deduplicated by
save_multifunction) and its KV-cache states. Writes chunk_<i>.mlmodelc,
embed.npy (the token table / K, fp16, gathered on the host) and meta.json."""
import argparse
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
ap.add_argument("--C", type=int, default=2048)
ap.add_argument("--cuts", default=None, help="e.g. 0-14,14-24,24-24 (last: head-only chunk)")
ap.add_argument("--funcs", default="decode,prefill:128,prefill:512")
ap.add_argument("--quant", default="none")
ap.add_argument("--head-split", type=int, default=1)
ap.add_argument("--silu", default="stable")
ap.add_argument("--K", type=int, default=None)
a = ap.parse_args()

import coremltools as ct
import coremltools.optimize as cto

torch.set_grad_enabled(False)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.src)
hf = LA.load_hf(a.src)
cfg = hf.config
L = cfg.num_hidden_layers
cuts = [(0, L)] if not a.cuts else [tuple(map(int, c.split("-"))) for c in a.cuts.split(",")]

# calibrate on a cache big enough for the calibration texts (fp32), then build
# the fp16-cache chunks over the same (rewritten) layers
cal_chunks, layers, norm = LA.build(hf, C=1024, cuts=[(0, L)], silu=a.silu, cache_dtype=torch.float32)
embed = hf.model.embed_tokens.weight.detach()
cal = LA.calibrate(cal_chunks, layers, norm, cfg, embed, T.calib_texts(tok), K=a.K)
K = cal["K"]
head = hf.lm_head
head_only = any(lo == hi == L for lo, hi in cuts)
chunks = []
for lo, hi in cuts:
    has_head = (hi == L and not head_only) or (lo == hi == L)
    c = LA.Chunk(layers[lo:hi], cfg, lo, hi, a.C, norm if has_head else None, head if has_head else None, a.head_split)
    c.K = float(K)
    chunks.append(c)
del cal_chunks

np.save(out / "embed.npy", (embed / K).half().numpy())
# free the checkpoint: the chunks hold their own conv copies of every weight
del embed, head, hf
import gc
gc.collect()
funcs = []
for f in a.funcs.split(","):
    funcs.append(("decode", 1) if f == "decode" else (f"prefill_{f.split(':')[1]}", int(f.split(":")[1])))

D, Dh = cfg.hidden_size, cfg.head_dim


def quantize(ml):
    if a.quant == "none":
        return ml
    if a.quant in ("int8", "int4"):
        op = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype=a.quant,
            granularity="per_channel" if a.quant == "int8" else "per_block",
            **({} if a.quant == "int8" else {"block_size": 32}), weight_threshold=2048)
        return cto.coreml.linear_quantize_weights(ml, config=cto.coreml.OptimizationConfig(global_config=op))
    if a.quant in ("pal4", "pal8"):
        op = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=int(a.quant[3:]), granularity="per_grouped_channel",
                                            group_size=16, weight_threshold=2048, num_kmeans_workers=1)
        return cto.coreml.palettize_weights(ml, config=cto.coreml.OptimizationConfig(global_config=op))
    raise SystemExit(f"unknown quant {a.quant}")


meta = {"src": a.src, "C": a.C, "cuts": cuts, "funcs": [f for f, _ in funcs], "quant": a.quant, "K": K,
        "silu": a.silu, "head_split": a.head_split, "hidden": D, "head_dim": Dh, "rope_theta": cfg.rope_theta,
        "vocab": cfg.vocab_size, "layers": L, "kv_heads": cfg.num_key_value_heads, "calibration": cal,
        "chunks": []}
tmp = out / "tmp"
tmp.mkdir(exist_ok=True)
for ci, c in enumerate(chunks):
    parts = []
    for fname, S in funcs:
        t0 = time.time()
        c.S = S
        c.reset()
        last = c.heads is not None
        ex = [torch.zeros(1, D, 1, S), torch.tensor([0], dtype=torch.int32), LA.mask_for(0, S, a.C),
              torch.zeros(S, Dh), torch.zeros(S, Dh)]
        inputs = [ct.TensorType("hidden", (1, D, 1, S), np.float16), ct.TensorType("pos", (1,), np.int32),
                  ct.TensorType("mask", (1, 1, S, a.C), np.float16), ct.TensorType("cos", (S, Dh), np.float16),
                  ct.TensorType("sin", (S, Dh), np.float16)]
        if last and S > 1:
            ex.append(torch.zeros(S))
            inputs.append(ct.TensorType("sel", (S,), np.float16))
        tr = torch.jit.trace(c, tuple(ex), check_trace=False)
        ml = ct.convert(tr, inputs=inputs,
                        outputs=[ct.TensorType(name="logits" if last else "hidden_out", dtype=np.float16)],
                        states=[ct.StateType(ct.TensorType(shape=tuple(c.k_cache.shape), dtype=np.float16), name="k_cache"),
                                ct.StateType(ct.TensorType(shape=tuple(c.v_cache.shape), dtype=np.float16), name="v_cache")],
                        minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        ml = quantize(ml)
        p = tmp / f"chunk_{ci}_{fname}.mlpackage"
        ml.save(str(p))
        parts.append((fname, p))
        del ml, tr
        print(f"chunk {ci} ({c.lo}-{c.hi}{' +head' if last else ''}) {fname}: {time.time() - t0:.0f}s", flush=True)
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
    meta["chunks"].append({"lo": c.lo, "hi": c.hi, "head": c.heads is not None, "weight_bytes": wb,
                           "cache_shape": list(c.k_cache.shape)})
    print(f"chunk {ci}: weights {wb / 2**30:.3f} GiB", flush=True)
    shutil.rmtree(pkg)
    for _, p in parts:
        shutil.rmtree(p)
shutil.rmtree(tmp)
json.dump(meta, open(out / "meta.json", "w"), indent=1)
print("done", flush=True)
