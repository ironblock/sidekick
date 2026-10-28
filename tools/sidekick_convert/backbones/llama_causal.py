"""LlamaForCausalLM as `kv-state-1` chunk programs for the Neural Engine
(docs/DECISIONS.md D53; docs/design/open-llm.md, "The program contract").

The model is rewritten into the channels-first form the ANE runs, then
split at layer boundaries into chunk programs that each own their layers'
KV cache as Core ML state:
- **Channels first.** The residual stream is [1, D, 1, T] and every
  projection a 1x1 convolution: at one token per call Core ML places a
  linear-op program on the CPU, but the same work as convolutions on the
  ANE (#173).
- **The contract** (`Chunk`): inputs `hidden_in` [1, D, 1, T], `pos` [1]
  int32, `mask` [1, 1, T, C], `cos`/`sin` [T, head_dim] and, on the last
  chunk, `sel` [T]; outputs `hidden_out` [1, D, 1, T] or `logits` [1, V];
  states `k_cache`/`v_cache` [layers, kv_heads, C, head_dim], written with a
  slice update at `pos`. Everything fp16 but `pos`. The token embedding is
  the host's (`embedding_table`), as are the RoPE rows (`rope_tables`).
- **Explicit attention** (matmul, softmax, matmul; never the fused op,
  D25) under a finite additive mask (techniques.masks.MASK_ADD), with the
  probabilities scaled by PV_SCALE into the value matmul and the output
  scaled back after it.
- **Output projections' input scales** (techniques.precision): o_proj and
  down_proj read inputs whose rms is small (0.02-0.1 in MiniCPM5-2B's early
  layers), which costs them about 1% on the ANE. Each takes a calibrated
  power-of-two scale on its input and the inverse on its output, as an
  explicit multiply the conversion keeps (DESCALE_PASS).
- **StableSilu** (D39) for the MLP, with up_proj taking its factor 1/2.
- **RMSNorm pre-scales**: each norm's input is multiplied by a power of two
  that keeps its squares in fp16 range, with eps * s^2 (the norm is
  scale-invariant up to eps).
- **The residual stream at 1/K** (D25 amendment;
  techniques.saturation): o_proj and down_proj write 1/K of their output,
  the host's embedding table carries 1/K, and the norms take eps / K^2,
  where K is the smallest power of two that keeps every projection's output
  under the ANE linear's 2^15 limit. K is 1 when nothing crosses it.
- **R1 and R2 rotations** (QuaRot / SpinQuant): random Hadamard rotations
  folded into the weights offline, so the program is unchanged in shape and
  runs no extra op. R1 rotates the residual stream: every RMSNorm's gain is
  folded into the projections it feeds (rms(Qh) = rms(h)), readers take
  W diag(g) Q^T, writers (o, down) Q W, the embedding table E Q^T and the
  head W_head diag(g_final) Q^T. R2 rotates each head's values: v's rows per
  KV head H W_v, o's columns per head W_o H^T. Massive activations, which
  sit in a few channels, are spread over all of them, which cuts the
  weights' outliers to what Gaussian weights give and is what lets 8-bit
  palettes hold the model's accuracy (#173).

Weights are read tensor by tensor from the checkpoint's safetensors
(`Checkpoint`), so a 2B model's layers are built in fp32 one at a time and
never all at once. `stream()` runs the rewritten model layer by layer over
whole sequences, in fp32: the rewrite gate, the range calibration, GPTQ and
the PyTorch screen all use it.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..techniques.activations import StableSilu
from ..techniques.masks import MASK_ADD
from ..techniques import precision, saturation

NORM_SQ_MAX = 30000.0
"""The largest square an RMSNorm input may reach after its pre-scale: under
fp16's 65,504 with a margin."""

PV_SCALE = 2.0 ** 8
"""The attention probabilities' power-of-two scale into the value matmul,
undone right after it, against the ANE's small-input floor on the matmul's
input, which grows with the context as the probabilities shrink (D55).
Exact in fp32; the matmul's output is at most PV_SCALE x max |v|, which
calibrate_ranges checks against the ANE's 2^15."""

DESCALE_PASS = "common::fuse_conv_scale"
"""coremltools' pass that folds a multiply after a convolution into its
weights. It would fold an output projection's descale back into a plain
weight, which undoes its input scale on the ANE (D55), so these programs are
converted and palettized without it, and the converter checks that every
descale survived."""

PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")
HF_NAMES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
            "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
READERS = {"n1": ("q", "k", "v"), "n2": ("gate", "up")}
WRITERS = ("o", "down")


class Checkpoint:
    """A LlamaForCausalLM checkpoint read tensor by tensor (fp32 copies of
    bf16 or fp16 values are exact). Refuses an architecture `kv-state-1`
    doesn't cover."""

    def __init__(self, src):
        from safetensors import safe_open
        self.src = Path(src)
        self.config = json.loads((self.src / "config.json").read_text())
        index = self.src / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.src.glob("*.safetensors"))
        self._where = {}
        self._files = [safe_open(str(self.src / f), framework="pt") for f in files]
        for f in self._files:
            for k in f.keys():
                self._where[k] = f
        self.check()

    def check(self):
        c = self.config
        problems = []
        if c.get("architectures") != ["LlamaForCausalLM"] or c.get("model_type") != "llama":
            problems.append(f"architecture {c.get('architectures')} / {c.get('model_type')}, not LlamaForCausalLM")
        if c.get("rope_scaling"):
            problems.append(f"rope_scaling {c['rope_scaling']} (kv-state-1 ships plain RoPE tables)")
        rope = c.get("rope_parameters") or {}            # transformers 5's form of rope_theta and rope_scaling
        if rope.get("rope_type", "default") != "default":
            problems.append(f"rope_parameters {rope} (kv-state-1 ships plain RoPE tables)")
        if c.get("hidden_act", "silu") != "silu":
            problems.append(f"hidden_act {c['hidden_act']}")
        for flag in ("attention_bias", "mlp_bias"):
            if c.get(flag):
                problems.append(f"{flag} is set")
        if self.heads % self.kv_heads:
            problems.append(f"{self.heads} heads don't group over {self.kv_heads} KV heads")
        if problems:
            raise ValueError("this checkpoint doesn't fit the llama_causal backbone: " + "; ".join(problems))

    def tensor(self, name):
        if name == "lm_head.weight" and name not in self._where and self.config.get("tie_word_embeddings"):
            name = "model.embed_tokens.weight"
        return self._where[name].get_tensor(name).float()

    def layer_tensor(self, i, proj):
        return self.tensor(f"model.layers.{i}.{HF_NAMES[proj]}.weight")

    @property
    def layers(self):
        return int(self.config["num_hidden_layers"])

    @property
    def hidden(self):
        return int(self.config["hidden_size"])

    @property
    def heads(self):
        return int(self.config["num_attention_heads"])

    @property
    def kv_heads(self):
        return int(self.config.get("num_key_value_heads") or self.heads)

    @property
    def head_dim(self):
        return int(self.config.get("head_dim") or self.hidden // self.heads)

    @property
    def vocab(self):
        return int(self.config["vocab_size"])

    @property
    def eps(self):
        return float(self.config["rms_norm_eps"])

    @property
    def theta(self):
        theta = self.config.get("rope_theta") or (self.config.get("rope_parameters") or {}).get("rope_theta")
        return float(theta or 10000.0)


# ---------------------------------------------------------------- modules

class Norm(nn.Module):
    """RMSNorm over the channels of [1, D, 1, S], times a power-of-two
    pre-scale `mult` (eps * mult^2 keeps it the same norm)."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().float().clone().view(1, -1, 1, 1), requires_grad=False)
        self.eps = float(eps)
        self.mult = 1.0
        self.stat = None            # range calibration: max |x|

    def forward(self, x):
        if self.stat is not None:
            self.stat = max(self.stat, float(x.abs().max()))
        y = x * self.mult if self.mult != 1.0 else x
        return y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + self.eps) * self.weight


def conv(weight):
    """A 1x1 convolution holding `weight` [out, in] (fp32)."""
    c = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=False)
    with torch.no_grad():
        c.weight.copy_(weight.float().reshape(weight.shape[0], weight.shape[1], 1, 1))
    c.requires_grad_(False)
    return c


def weight_of(c):
    return c.weight.view(c.weight.shape[0], -1)


def rotate_half(x, h):
    # h = head_dim // 2, a Python int: a traced shape converts to an Int op coremltools can't take
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Layer(nn.Module):
    """One decoder layer, channels first, with its KV cache read and written
    at `pos` in the chunk's cache tensors."""

    def __init__(self, ck, i):
        super().__init__()
        self.H, self.KVH, self.Dh = ck.heads, ck.kv_heads, ck.head_dim
        self.rep = self.H // self.KVH
        self.scale = 1.0 / math.sqrt(self.Dh)
        p = f"model.layers.{i}."
        self.n1 = Norm(ck.tensor(p + "input_layernorm.weight"), ck.eps)
        self.n2 = Norm(ck.tensor(p + "post_attention_layernorm.weight"), ck.eps)
        for proj in PROJECTIONS:
            w = ck.layer_tensor(i, proj)
            if proj == "up":
                w = w / StableSilu.GAIN       # StableSilu returns 2 silu(x)
            setattr(self, proj, conv(w))
        self.act = StableSilu()
        self.in_scale = {"o": 1.0, "down": 1.0}   # output projections' input scales (apply_ranges)
        self.istat = None           # range calibration: o and down inputs' [sum x^2, count, max |x|]
        self.vstat = None           # range calibration: max |v out|
        self.ostat = None           # range calibration: [max |q,k,v,gate,up out|, max |o,down out|]

    def _track_in(self, which, x):
        if self.istat is not None:
            s = self.istat[which]
            s[0] += float(x.double().pow(2).sum())
            s[1] += x.numel()
            s[2] = max(s[2], float(x.abs().max()))

    def _track(self, which, *ts):
        if self.ostat is not None:
            self.ostat[which] = max(self.ostat[which], *(float(t.abs().max()) for t in ts))

    def forward(self, h, li, kc, vc, pos, mask, cos, sin, S, C):
        H, KVH, Dh, rep = self.H, self.KVH, self.Dh, self.rep
        x = self.n1(h)
        q, k, v = self.q(x), self.k(x), self.v(x)
        self._track(0, q, k, v)
        if self.vstat is not None:
            self.vstat = max(self.vstat, float(v.abs().max()))
        q = q.view(H, Dh, S).transpose(1, 2)
        k = k.view(KVH, Dh, S).transpose(1, 2)
        v = v.view(KVH, Dh, S).transpose(1, 2)
        q = q * cos + rotate_half(q, Dh // 2) * sin
        k = k * cos + rotate_half(k, Dh // 2) * sin
        kc[li:li + 1, :, pos:pos + S, :] = k.unsqueeze(0).to(kc.dtype)
        vc[li:li + 1, :, pos:pos + S, :] = v.unsqueeze(0).to(vc.dtype)
        K = kc[li].to(q.dtype)                                  # [KVH, C, Dh]
        V = vc[li].to(q.dtype)
        qg = (q * self.scale).reshape(KVH, rep * S, Dh)
        s = torch.matmul(qg, K.transpose(1, 2)).view(KVH, rep, S, C) + mask.view(1, 1, S, C)
        so, sd = self.in_scale["o"], self.in_scale["down"]
        p = torch.softmax(s, dim=-1).view(KVH, rep * S, C) * PV_SCALE
        o = torch.matmul(p, V) * (so / PV_SCALE)
        o = o.view(H, S, Dh).transpose(1, 2).reshape(1, H * Dh, 1, S)
        self._track_in("o", o)
        o = self.o(o)
        if so != 1.0:
            o = o * (1.0 / so)
        h = h + o
        x = self.n2(h)
        g, u = self.gate(x), self.up(x)
        a = self.act(g) * u
        self._track_in("down", a)
        d = self.down(a * sd if sd != 1.0 else a)
        if sd != 1.0:
            d = d * (1.0 / sd)
        self._track(0, g, u)
        self._track(1, o, d)
        return h + d


class Chunk(nn.Module):
    """Layers [lo, hi) as one `kv-state-1` program, owning their KV cache;
    the last chunk also holds the final norm and the head (split into
    `head_split` convolutions, concatenated). T (tokens per call) and C
    (cache capacity) are fixed per trace."""

    def __init__(self, layers, ck, T, C, norm=None, head=None, head_split=1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.T, self.C, self.D = T, C, ck.hidden
        n = len(layers)
        shape = (n, ck.kv_heads, C, ck.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=torch.float16))
        self.norm = norm
        self.heads = None
        if head is not None:
            self.heads = nn.ModuleList(conv(p) for p in torch.chunk(head, head_split, dim=0))

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, hidden_in, pos, mask, cos, sin, sel=None):
        T, C = self.T, self.C
        h = hidden_in.to(torch.float32)
        cos, sin, mask = cos.to(torch.float32), sin.to(torch.float32), mask.to(torch.float32)
        for i, layer in enumerate(self.layers):
            h = layer(h, i, self.k_cache, self.v_cache, pos, mask, cos, sin, T, C)
        if self.heads is None:
            return h
        h = torch.matmul(h.view(self.D, T), sel.to(torch.float32).view(T, 1)).view(1, self.D, 1, 1)
        h = self.norm(h)
        logits = torch.cat([c(h) for c in self.heads], dim=1) if len(self.heads) > 1 else self.heads[0](h)
        return logits.view(1, -1)


# ---------------------------------------------------------------- host tables, masks

def rope_tables(theta, head_dim, positions):
    """cos, sin [len(positions), head_dim] in float64, from fp64 angles."""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], -1)
    return np.cos(ang), np.sin(ang)


def mask_for(pos, S, C):
    """The additive causal mask [1, 1, S, C] for S queries from `pos` over a
    cache of C: 0 where key <= query position, MASK_ADD elsewhere."""
    q = pos + torch.arange(S)[:, None]
    k = torch.arange(C)[None, :]
    return torch.where(k <= q, 0.0, MASK_ADD).view(1, 1, S, C)


# ---------------------------------------------------------------- rewrites

def hadamard(n, seed):
    """A random orthogonal Hadamard matrix: Sylvester's construction with
    its columns' signs drawn from `seed`, over sqrt(n), in float64."""
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    if H.shape[0] != n:
        raise ValueError(f"no Sylvester Hadamard matrix of size {n}: R1 and R2 need powers of two")
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float64)
    return H * s.view(1, -1) / math.sqrt(n)


@dataclasses.dataclass
class Rotation:
    """R1 (the residual stream, `Q`) and R2 (per head on v and o, `H`, or
    None), from `seed` (R1) and seed + 1 (R2)."""
    seed: int
    r2: bool = True

    def __post_init__(self):
        self.Q = None
        self.H = None

    def bind(self, ck):
        self.Q = hadamard(ck.hidden, self.seed)
        self.H = hadamard(ck.head_dim, self.seed + 1) if self.r2 else None
        return self


def rotate_layer(layer, rot):
    """R1 (and R2) on a freshly built Layer, in float64: each norm's gain
    folded into its readers, then set to 1."""
    Q = rot.Q
    with torch.no_grad():
        for norm, readers in ((layer.n1, READERS["n1"]), (layer.n2, READERS["n2"])):
            g = norm.weight.view(-1).double()
            for p in readers:
                c = getattr(layer, p)
                W = weight_of(c).double()
                c.weight.copy_(((W * g) @ Q.T).float().reshape(c.weight.shape))
            norm.weight.fill_(1.0)
        for p in WRITERS:
            c = getattr(layer, p)
            c.weight.copy_((Q @ weight_of(c).double()).float().reshape(c.weight.shape))
        if rot.H is not None:
            Hh, Dh, KVH, H = rot.H, layer.Dh, layer.KVH, layer.H
            Wv = layer.v.weight.view(KVH, Dh, -1).double()
            layer.v.weight.copy_(torch.einsum("ij,kjd->kid", Hh, Wv).float().reshape(layer.v.weight.shape))
            Wo = layer.o.weight.view(-1, H, Dh).double()
            layer.o.weight.copy_(torch.einsum("ohj,ij->ohi", Wo, Hh).float().reshape(layer.o.weight.shape))
    return layer


@dataclasses.dataclass
class Ranges:
    """The range calibration's result: the residual stream's K and every
    norm's pre-scale (layer i's input norm at i, its post-attention norm at
    L + i, the final norm last), with the maxima they were chosen from."""
    K: int
    norm_scale: list
    norm_max: list
    linear_out_max: list           # per layer: [max |q,k,v,gate,up out|, max |o,down out|]
    in_scale: list = None          # per layer: [o's input scale, down's input scale]
    v_out_max: list = None         # per layer: max |v out|, what the value matmul reads
    in_stats: list = None          # per layer: [[rms, max] of o's input, [rms, max] of down's input]

    def to_json(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)


def apply_ranges(layer, i, L, ranges):
    """The residual 1/K and the layer's two norms' pre-scales, on a freshly
    built (and rotated) Layer i of L."""
    K = ranges.K
    with torch.no_grad():
        if K != 1:
            layer.o.weight.mul_(1.0 / K)
            layer.down.weight.mul_(1.0 / K)
    for n, s in ((layer.n1, ranges.norm_scale[i]), (layer.n2, ranges.norm_scale[L + i])):
        n.eps = n.eps / (K * K) * s * s
        n.mult = s
    if ranges.in_scale is not None:
        layer.in_scale = {"o": ranges.in_scale[i][0], "down": ranges.in_scale[i][1]}
    return layer


def build_layer(ck, i, rot=None, ranges=None):
    """Layer i in fp32 with the backbone's rewrites: the rotation, then the
    residual 1/K and the norms' pre-scales."""
    layer = Layer(ck, i)
    if rot is not None:
        rotate_layer(layer, rot)
    if ranges is not None:
        apply_ranges(layer, i, ck.layers, ranges)
    return layer


def final_norm(ck, rot=None, ranges=None):
    """The final RMSNorm: its gain folded into the head under R1 (weight 1),
    eps / K^2 and its pre-scale."""
    w = torch.ones(ck.hidden) if rot is not None else ck.tensor("model.norm.weight")
    n = Norm(w, ck.eps)
    if ranges is not None:
        s = ranges.norm_scale[-1]
        n.eps = n.eps / (ranges.K * ranges.K) * s * s
        n.mult = s
    return n


def head_weight(ck, rot=None):
    """The head [V, D] fp32: W_head diag(g_final) Q^T under R1, in float64,
    16k rows at a time."""
    W = ck.tensor("lm_head.weight")
    if rot is None:
        return W
    g = ck.tensor("model.norm.weight").double()
    return torch.cat([((W[j:j + 16384].double() * g) @ rot.Q.T).float() for j in range(0, W.shape[0], 16384)])


def embedding_table(ck, rot=None, K=1):
    """The host's embedding table [V, D] fp32: E Q^T under R1, at 1/K."""
    E = ck.tensor("model.embed_tokens.weight")
    if rot is not None:
        E = torch.cat([(E[j:j + 16384].double() @ rot.Q.T).float() for j in range(0, E.shape[0], 16384)])
    return E / K


# ---------------------------------------------------------------- layer-at-a-time execution

def run_layer(layer, h, ck):
    """One layer over a whole sequence h [1, D, 1, S] from position 0, with a
    cache of capacity S: the same arithmetic a chunk program does call by
    call."""
    S = h.shape[-1]
    kc = torch.zeros(1, ck.kv_heads, S, ck.head_dim)
    vc = torch.zeros(1, ck.kv_heads, S, ck.head_dim)
    cos, sin = (torch.from_numpy(t).float() for t in rope_tables(ck.theta, ck.head_dim, range(S)))
    return layer(h, 0, kc, vc, torch.tensor([0], dtype=torch.int32), mask_for(0, S, S), cos, sin, S, S)


def embed(table, ids):
    """Hidden states [1, D, 1, S] for token ids, from an fp32 table."""
    return table[torch.tensor(ids)].float().T.reshape(1, -1, 1, len(ids))


def stream(ck, sequences, make_layer, table, on_layer=None, on_hidden=None):
    """The rewritten model, fp32, one layer at a time over every sequence:
    layer i is `make_layer(i)`, `on_layer(i, layer)` may wrap or inspect it
    before it runs, `on_hidden(i, s, h)` sees its output for sequence s.
    Returns the final hidden states [1, D, 1, S] per sequence."""
    hs = [embed(table, ids) for ids in sequences]
    with torch.no_grad():
        for i in range(ck.layers):
            layer = make_layer(i)
            if on_layer is not None:
                on_layer(i, layer)
            for s, h in enumerate(hs):
                hs[s] = run_layer(layer, h, ck)
                if on_hidden is not None:
                    on_hidden(i, s, hs[s])
            del layer
    return hs


def logits(norm, head, h, rows=None):
    """fp32 final norm and head over hidden [1, D, 1, S]: [rows, V], for
    every position, a (lo, hi) range or a list of positions."""
    S = h.shape[-1]
    if rows is None:
        rows = list(range(S))
    elif isinstance(rows, tuple):
        rows = list(range(*rows))
    with torch.no_grad():
        x = norm(h[..., rows]).reshape(h.shape[1], len(rows)).T
        return torch.cat([x @ head[j:j + 16384].T for j in range(0, head.shape[0], 16384)], dim=1)


def calibrate_ranges(ck, sequences, rot=None, K=None, report=print):
    """Max |x| at every RMSNorm input and max |output| per projection over
    `sequences`, in fp32 on the rotated (unscaled) model; then the residual
    stream's K (saturation.choose_k: the smallest power of two keeping o and
    down's outputs under 0.85 x 2^15; or the recipe's K) and each norm's
    power-of-two pre-scale, the largest keeping (max |x| / K * s)^2 under
    NORM_SQ_MAX."""
    L = ck.layers
    nmax = [0.0] * (2 * L + 1)
    lin = [[0.0, 0.0] for _ in range(L)]
    cur = {}

    ins = [None] * L
    vmax = [0.0] * L

    def on_layer(i, layer):
        layer.n1.stat, layer.n2.stat, layer.ostat, layer.vstat = 0.0, 0.0, [0.0, 0.0], 0.0
        layer.istat = {"o": [0.0, 0, 0.0], "down": [0.0, 0, 0.0]}
        cur["layer"] = layer

    def on_hidden(i, s, h):
        lay = cur["layer"]
        nmax[i], nmax[L + i], lin[i] = lay.n1.stat, lay.n2.stat, list(lay.ostat)
        ins[i] = [[math.sqrt(st[0] / st[1]), st[2]] for st in (lay.istat["o"], lay.istat["down"])]
        vmax[i] = lay.vstat

    table = embedding_table(ck, rot)
    hs = stream(ck, sequences, lambda i: build_layer(ck, i, rot), table, on_layer, on_hidden)
    nmax[2 * L] = max(float(h.abs().max()) for h in hs)
    fixed, scaled = max(x[0] for x in lin), max(x[1] for x in lin)
    saturation.check({"the value matmul at PV_SCALE": PV_SCALE * max(vmax)})
    if K is None:
        K, _ = saturation.choose_k(fixed, scaled)
    scales = []
    for m in nmax:
        m = m / K
        s = 1.0
        while (m * s) ** 2 > NORM_SQ_MAX:
            s /= 2.0
        scales.append(s)
    # o and down write the residual stream at 1/K; their inputs don't scale with K
    in_scale = [[precision.input_scale(_Stat(*st), _Stat(0.0, lin[i][1] / K)) for st in ins[i]] for i in range(L)]
    report(f"ranges: K = {K} (o/down outputs up to {scaled:.0f}, other projections {fixed:.0f}); norm inputs up "
           f"to {max(nmax) / K:.1f}; {sum(s != 1.0 for s in scales)}/{len(scales)} norms pre-scaled; "
           f"{sum(s != 1.0 for x in in_scale for s in x)}/{2 * L} output projections' inputs scaled "
           f"(up to {max(s for x in in_scale for s in x):.0f})")
    return Ranges(K=int(K), norm_scale=scales, norm_max=[m / K for m in nmax], linear_out_max=lin,
                  in_scale=in_scale, in_stats=ins, v_out_max=vmax)


@dataclasses.dataclass
class _Stat:
    """What precision.input_scale reads of a calibrate.Stat."""
    rms: float
    max: float
