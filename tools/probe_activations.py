"""Triage a transformer encoder for the Apple Neural Engine before converting it.

The ANE computes in fp16. This probe runs the model in fp32 PyTorch on the
CPU — no Core ML, minutes not hours — hooks the input of every normalization
layer (where the residual stream is read), and reports per layer: the peak
activation, which feature dimension carries it, how many times the median
dimension that is, where it sits (first/last/middle token), and whether a
mean-subtracting LayerNorm or an RMS-style norm reads it.

Its one calibrated verdict is fp16 RANGE. Values above 65504 overflow fp16;
EmbeddingGemma peaks at ~152,000 and converts after a power-of-two range
rewrite (D17). Below ~30,000 a direct convert is fine.

Massive activations — one dimension hundreds of times the rest on a few
tokens — are reported but are NOT disqualifying. gte-modernbert-base peaks
at ~48,000 (502x the median) under LayerNorm and converts at 0.9998 on the
ANE with explicit attention; an earlier version of this tool called that
pattern ANE-hostile, a misdiagnosis of a Core ML attention-mask bug (D25).
Expect those tokens' own output vectors to be slightly less accurate on the
ANE; pooled outputs are unaffected.

The probe can't see graph-level conversion bugs. After converting, run
ane_check: its compute-plan and pad-invariance gates are the real test.

Usage:
    python tools/probe_activations.py <model-dir> [options]

    <model-dir>            local Hugging Face snapshot (config + weights)
    --config DIR           config directory, if not <model-dir> (e.g. a
                           checkpoint that keeps its encoder config in encoder/)
    --weights FILE         safetensors file to load into the model built from
                           the config, instead of from_pretrained
    --prefix STR           strip this prefix from weight names (e.g. "encoder.")
    --tokenizer DIR        tokenizer directory, if not <model-dir>
    --trust-remote-code    allow the checkpoint's own modeling code
    --json FILE            also write the full report as JSON

Requires: torch, transformers, safetensors (arm64-native Python).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

FP16_MAX = 65504.0
RANGE_WARN = 30000.0   # below this, convert without a range rewrite
MASSIVE_RATIO = 100.0  # peak dim this many times the median: report as massive

# A mixed probe corpus: short and long prose, code, numbers and URLs,
# punctuation-heavy text, and non-English (for multilingual models). Massive
# activations tend to appear on the first token and on delimiters, so varied
# token types matter more than volume.
CORPUS = [
    "A cat sat on the mat.",
    "Quarterly financial earnings exceeded expectations.",
    "def add(a, b):\n    return a + b  # simple helper\n",
    "Order #48213 shipped 2026-09-14 to 1600 Amphitheatre Pkwy; see https://example.com/track?id=48213.",
    "Wait... what?! (No, really — \"that\" isn't it.) [1] {2} <3>",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "東京は日本の首都であり、世界最大級の都市圏を形成している。",
    "El modelo convierte texto en vectores para la búsqueda semántica.",
    "SELECT name, COUNT(*) FROM users WHERE active = 1 GROUP BY name ORDER BY 2 DESC;",
    " ".join(
        f"Sentence number {i} discusses topic {i * 7 % 13} in considerable detail."
        for i in range(40)
    ),
]


def load(args):
    config_dir = Path(args.config or args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model_dir, trust_remote_code=args.trust_remote_code
    )
    if args.weights:
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=args.trust_remote_code)
        model = AutoModel.from_config(config, trust_remote_code=args.trust_remote_code)
        state = load_file(args.weights)
        if args.prefix:
            state = {k[len(args.prefix):]: v for k, v in state.items() if k.startswith(args.prefix)}
        missing, unexpected = model.load_state_dict(
            {k: v.float() for k, v in state.items()}, strict=False
        )
        loaded = len(state) - len(unexpected)
        print(f"loaded {loaded} tensors ({len(missing)} missing, {len(unexpected)} unexpected)")
        if missing:
            print(f"  missing (first 5): {missing[:5]}")
        if loaded == 0:
            sys.exit("no weights matched the model; check --prefix")
    else:
        model = AutoModel.from_pretrained(
            args.model_dir, torch_dtype=torch.float32, trust_remote_code=args.trust_remote_code
        )
    return tokenizer, model.float().eval()


def is_norm(module):
    name = type(module).__name__
    return name.endswith("Norm") or "LayerNorm" in name or "RMSNorm" in name


def subtracts_mean(module):
    """LayerNorm centers its input; RMSNorm (and friends) don't."""
    name = type(module).__name__
    return isinstance(module, torch.nn.LayerNorm) or ("LayerNorm" in name and "RMS" not in name)


class NormProbe:
    def __init__(self, name, module):
        self.name = name
        self.centers = subtracts_mean(module)
        self.per_dim_max = None  # max |x| per feature dim, over real tokens
        self.top_positions = {}  # where each text's largest value sits: count

    def observe(self, x, mask, lengths):
        # x: [B, T, D] (other layouts, e.g. per-head QK norms, are reshaped)
        if x.dim() != 3 or x.shape[:2] != mask.shape:
            x = x.reshape(mask.shape[0], mask.shape[1], -1) if x.numel() % mask.numel() == 0 else None
            if x is None:
                return
        x = x.detach().float()
        real = mask.bool()
        vals = x[real]  # [N, D]
        dim_max = vals.abs().amax(0)
        self.per_dim_max = dim_max if self.per_dim_max is None else torch.maximum(self.per_dim_max, dim_max)
        # Where the largest value sits: first token, last real token, or elsewhere.
        flat = x.abs().masked_fill(~real[..., None], 0)
        b, t, _ = torch.unravel_index(flat.argmax(), flat.shape)
        where = "first" if t == 0 else ("last" if t == lengths[b] - 1 else "middle")
        self.top_positions[where] = self.top_positions.get(where, 0) + 1

    def summary(self):
        d = self.per_dim_max
        top_vals, top_dims = d.topk(min(3, d.numel()))
        median = float(d.median()) or 1e-12
        return {
            "norm": self.name,
            "centers": self.centers,
            "max_abs": float(top_vals[0]),
            "top_dims": [[int(i), round(float(v), 1)] for i, v in zip(top_dims, top_vals)],
            "outlier_ratio": round(float(top_vals[0]) / median, 1),
            "top_positions": self.top_positions,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--config")
    ap.add_argument("--weights")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--tokenizer")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--json")
    args = ap.parse_args()

    tokenizer, model = load(args)
    probes, hooks = {}, []
    for name, module in model.named_modules():
        if is_norm(module):
            probes[name] = NormProbe(name, module)

            def hook(mod, inputs, _out, name=name):
                current["probes"][name].observe(inputs[0], current["mask"], current["lengths"])

            hooks.append(module.register_forward_hook(hook))
    if not probes:
        sys.exit("found no normalization layers to probe")

    current = {"probes": probes}
    with torch.no_grad():
        for text in CORPUS:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_len)
            current["mask"] = enc["attention_mask"]
            current["lengths"] = enc["attention_mask"].sum(1)
            model(**enc)
    for h in hooks:
        h.remove()

    rows = [p.summary() for p in probes.values() if p.per_dim_max is not None]
    worst_range = max(rows, key=lambda r: r["max_abs"])
    layernorms = [r for r in rows if r["centers"]]

    print(f"\nmodel: {type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"{len(rows)} norm layers ({len(layernorms)} LayerNorm, {len(rows) - len(layernorms)} RMS-style)")
    print(f"{'norm layer (input)':<44} {'kind':<5} {'max |x|':>10} {'outlier×':>9}  top dims (dim:value)")
    for r in rows:
        dims = " ".join(f"{d}:{v:g}" for d, v in r["top_dims"])
        kind = "LN" if r["centers"] else "RMS"
        print(f"{r['norm'][-44:]:<44} {kind:<5} {r['max_abs']:>10.1f} {r['outlier_ratio']:>9.1f}  {dims}")

    print("\nverdict:")
    verdict, notes = "no fp16 range issue", []
    if worst_range["max_abs"] >= FP16_MAX:
        verdict = "needs a range rewrite"
        notes.append(f"values reach {worst_range['max_abs']:.0f} (> fp16 max 65504) at {worst_range['norm']}: "
                     "apply a power-of-two range rewrite before converting (D17)")
    elif worst_range["max_abs"] >= RANGE_WARN:
        notes.append(f"values reach {worst_range['max_abs']:.0f} at {worst_range['norm']}: little fp16 "
                     "headroom; calibrate on your real inputs")
    massive = max(rows, key=lambda r: r["outlier_ratio"])
    if massive["outlier_ratio"] >= MASSIVE_RATIO and massive["max_abs"] >= 100:
        where = max(massive["top_positions"], key=massive["top_positions"].get)
        notes.append(f"massive activation: dim {massive['top_dims'][0][0]} reaches {massive['max_abs']:.0f} "
                     f"({massive['outlier_ratio']:.0f}x the median dim, mostly on the {where} token). Not "
                     "disqualifying (gte-modernbert: ~48,000, 0.9998 on the ANE); those tokens' own output "
                     "vectors will be slightly less accurate")
    print(f"  {verdict}")
    for n in notes:
        print(f"  - {n}")
    print("  next: convert with explicit attention and run ane_check (compute plan + pad invariance)")

    if args.json:
        Path(args.json).write_text(json.dumps({"verdict": verdict, "notes": notes, "norms": rows}, indent=2))


if __name__ == "__main__":
    main()
