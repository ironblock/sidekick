"""A Llama-architecture decoder (LlamaForCausalLM: SmolLM2, MiniCPM5) rewritten
for the Neural Engine with its KV cache held as Core ML state.

Layout and programs
- Channels-first residual stream [1, D, 1, S] with 1x1 convolutions for every
  projection: at S = 1 (decode) Core ML places a linear-op program on the CPU
  but the same work as conv on the ANE (toy probe, M1 Max, macOS 27).
- A program ("chunk") covers a contiguous run of layers and owns their KV
  cache as two states, k_cache and v_cache, [layers, kv_heads, C, head_dim]
  fp16, C the cache capacity. Every function of a chunk (prefill at bucket S,
  decode at S = 1) has the same states, so one MLState serves them all.
- Inputs: hidden [1, D, 1, S] fp16 (the token embedding is gathered on the
  host: no table in the program), pos [1] int32 (where this call's first token
  sits), mask [1, 1, S, C] fp16 additive (0 or -30000; causal over the cache),
  cos/sin [S, head_dim] fp16 (RoPE for positions pos..pos+S-1, from fp32 on the
  host). The last chunk also takes sel [S] fp16 (one-hot of the position whose
  logits to return) when S > 1, and returns logits [1, V]; other chunks return
  hidden [1, D, 1, S].
- The cache is written with slice_update at the dynamic position (torch's
  tracer records a tensor slice bound as a dynamic value).

Rewrites carried from the encoder work (docs/DECISIONS.md)
- explicit attention (matmul, softmax, matmul), never the fused op (D25), with
  a finite additive mask;
- StableSilu (D39) for the MLP, up_proj taking the 1/2;
- RMSNorm input pre-scale by a calibrated power of two, eps * s^2, so its
  squares stay in fp16 range (Lumma-fev / EmbeddingGemma technique);
- optional power-of-two input rescales for small linear inputs (D17/D19):
  not applied unless measured to matter.
"""
import math

import torch
import torch.nn as nn

MASK = -30000.0
NORM_SQ_MAX = 30000.0


class StableSilu(nn.Module):
    """2 * silu(x), cancellation-free (tools/sidekick_convert/techniques/activations.py)."""
    GAIN = 2.0

    def forward(self, x):
        return 2.0 * x * torch.exp(torch.clamp(x, max=0.0)) / (1.0 + torch.exp(-torch.abs(x)))


class Norm(nn.Module):
    """RMSNorm over channels of [1, D, 1, S]; mult a power-of-two pre-scale."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().float().clone().view(1, -1, 1, 1), requires_grad=False)
        self.eps = float(eps)
        self.mult = 1.0
        self.stat = None            # calibration: max |x|

    def forward(self, x):
        if self.stat is not None:
            self.stat = max(self.stat, float(x.abs().max()))
        y = x * self.mult if self.mult != 1.0 else x
        return y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + self.eps) * self.weight


def conv(lin, scale=1.0):
    w = lin.weight.detach().float()
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=False)
    with torch.no_grad():
        c.weight.copy_((w * scale).view(w.shape[0], w.shape[1], 1, 1))
    c.requires_grad_(False)
    return c


def rot(x, h):
    # h = head_dim // 2, a Python int (a traced shape converts to the Int op that crashes coremltools)
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Layer(nn.Module):
    def __init__(self, hf_layer, cfg, silu="stable"):
        super().__init__()
        a, m = hf_layer.self_attn, hf_layer.mlp
        self.H, self.KVH, self.Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rep = self.H // self.KVH
        self.scale = 1.0 / math.sqrt(self.Dh)
        self.n1 = Norm(hf_layer.input_layernorm.weight, cfg.rms_norm_eps)
        self.n2 = Norm(hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        self.q, self.k, self.v, self.o = conv(a.q_proj), conv(a.k_proj), conv(a.v_proj), conv(a.o_proj)
        self.silu = silu
        if silu == "stable":
            self.act = StableSilu()
            self.up = conv(m.up_proj, 1.0 / StableSilu.GAIN)
        else:
            self.act = nn.SiLU()
            self.up = conv(m.up_proj)
        self.gate, self.down = conv(m.gate_proj), conv(m.down_proj)
        self.post_o, self.post_down = 1.0, 1.0      # apply_rescales: explicit 1/s after o and down
        self.ostat = None           # calibration: [max |q,k,v,gate,up out|, max |o,down out|]

    def _track(self, which, *ts):
        if self.ostat is not None:
            self.ostat[which] = max(self.ostat[which], *(float(t.abs().max()) for t in ts))

    def residual_scale(self, inv_k):
        """Run the residual stream at 1/K (D25 amendment): o_proj and down_proj
        write 1/K of their output; the norms are scale-invariant up to eps."""
        with torch.no_grad():
            self.o.weight.mul_(inv_k)
            self.down.weight.mul_(inv_k)
        self.n1.eps *= inv_k * inv_k
        self.n2.eps *= inv_k * inv_k

    def forward(self, h, li, kc, vc, pos, mask, cos, sin, S, C):
        H, KVH, Dh, rep = self.H, self.KVH, self.Dh, self.rep
        x = self.n1(h)
        q, k, v = self.q(x), self.k(x), self.v(x)
        self._track(0, q, k, v)
        q = q.view(H, Dh, S).transpose(1, 2)
        k = k.view(KVH, Dh, S).transpose(1, 2)
        v = v.view(KVH, Dh, S).transpose(1, 2)
        q = q * cos + rot(q, Dh // 2) * sin
        k = k * cos + rot(k, Dh // 2) * sin
        kc[li:li + 1, :, pos:pos + S, :] = k.unsqueeze(0).to(kc.dtype)
        vc[li:li + 1, :, pos:pos + S, :] = v.unsqueeze(0).to(vc.dtype)
        K = kc[li].to(q.dtype)                                  # [KVH, C, Dh]
        V = vc[li].to(q.dtype)
        qg = (q * self.scale).reshape(KVH, rep * S, Dh)
        s = torch.matmul(qg, K.transpose(1, 2)).view(KVH, rep, S, C) + mask.view(1, 1, S, C)
        p = torch.softmax(s, dim=-1).view(KVH, rep * S, C)
        o = torch.matmul(p, V).view(H, S, Dh).transpose(1, 2).reshape(1, H * Dh, 1, S)
        o = self.o(o)
        if self.post_o != 1.0:
            o = o * self.post_o
        h = h + o
        x = self.n2(h)
        g, u = self.gate(x), self.up(x)
        m = self.act(g) * u
        if getattr(self, "r4", None) is not None:      # R4: block-diagonal Hadamard at run time (grouped 1x1 conv)
            m = self.r4(m)
        d = self.down(m)
        if self.post_down != 1.0:
            d = d * self.post_down
        self._track(0, g, u)
        self._track(1, o, d)
        return h + d


class Chunk(nn.Module):
    """Layers [lo, hi) of the decoder, optionally with the final norm and head.
    S (tokens per call) and C (cache capacity) are Python ints, fixed per trace."""

    def __init__(self, layers, cfg, lo, hi, C, final_norm=None, head=None, head_split=1, cache_dtype=torch.float16):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.lo, self.hi, self.C, self.S = lo, hi, C, 1
        self.D = cfg.hidden_size
        n = len(layers)
        self.register_buffer("k_cache", torch.zeros(n, cfg.num_key_value_heads, C, cfg.head_dim, dtype=cache_dtype))
        self.register_buffer("v_cache", torch.zeros(n, cfg.num_key_value_heads, C, cfg.head_dim, dtype=cache_dtype))
        self.norm = final_norm
        if head is not None:
            w = head.weight.detach().float()
            parts = torch.chunk(w, head_split, dim=0)
            self.heads = nn.ModuleList()
            for p in parts:
                c = nn.Conv2d(p.shape[1], p.shape[0], 1, bias=False)
                with torch.no_grad():
                    c.weight.copy_(p.view(p.shape[0], p.shape[1], 1, 1))
                c.requires_grad_(False)
                self.heads.append(c)
        else:
            self.heads = None

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, hidden, pos, mask, cos, sin, sel=None):
        S, C = self.S, self.C
        h = hidden.to(torch.float32)
        cos, sin, mask = cos.to(torch.float32), sin.to(torch.float32), mask.to(torch.float32)
        for i, layer in enumerate(self.layers):
            h = layer(h, i, self.k_cache, self.v_cache, pos, mask, cos, sin, S, C)
        if self.heads is None:
            return h
        if sel is not None:
            h = torch.matmul(h.view(self.D, S), sel.to(torch.float32).view(S, 1)).view(1, self.D, 1, 1)
            S = 1
        h = self.norm(h)
        logits = torch.cat([c(h) for c in self.heads], dim=1) if len(self.heads) > 1 else self.heads[0](h)
        return logits.view(-1, S).transpose(0, 1)                # [S, V]


def load_hf(src, dtype=torch.float32):
    """The checkpoint, by default in fp32. bfloat16 keeps a 2B model at half
    the memory; Layer and Chunk copy what they use to fp32 exactly."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, attn_implementation="eager").eval()
    if getattr(m.config, "head_dim", None) is None:
        m.config.head_dim = m.config.hidden_size // m.config.num_attention_heads
    return m


def build(hf, C, cuts, silu="stable", head_split=1, cache_dtype=torch.float16):
    """Chunks over hf's layers. cuts: list of (lo, hi) layer ranges; a range
    (L, L) after the last layer is a head-only chunk. The head goes on the
    chunk that contains the last layer, or on a head-only chunk if given."""
    cfg = hf.config
    L = cfg.num_hidden_layers
    layers = [Layer(l, cfg, silu) for l in hf.model.layers]
    norm = Norm(hf.model.norm.weight, cfg.rms_norm_eps)
    head_only = any(lo == hi == L for lo, hi in cuts)
    chunks = []
    for lo, hi in cuts:
        has_head = (hi == L and not head_only) or (lo == hi == L)
        chunks.append(Chunk(layers[lo:hi], cfg, lo, hi, C, norm if has_head else None,
                            hf.lm_head if has_head else None, head_split, cache_dtype))
    return chunks, layers, norm


def rope(cfg, positions):
    """cos, sin [S, head_dim] fp32 for absolute positions."""
    Dh = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, Dh, 2, dtype=torch.float64) / Dh))
    ang = torch.tensor(positions, dtype=torch.float64)[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos().float(), emb.sin().float()


def mask_for(pos, S, C):
    q = pos + torch.arange(S)[:, None]
    k = torch.arange(C)[None, :]
    return torch.where(k <= q, 0.0, MASK).view(1, 1, S, C)


def run_eager(chunks, cfg, embed, ids, pos=0):
    """fp32 eager forward of `ids` at `pos` through all chunks (S = len(ids));
    returns logits [S, V]."""
    S = len(ids)
    C = chunks[0].C
    cos, sin = rope(cfg, list(range(pos, pos + S)))
    h = embed[torch.tensor(ids)].T.reshape(1, -1, 1, S) / getattr(chunks[0], "K", 1.0)
    m = mask_for(pos, S, C)
    p = torch.tensor([pos], dtype=torch.int32)
    for c in chunks:
        c.S = S
        h = c(h, p, m, cos, sin)
    return h


ANE_LINEAR_MAX = 32768.0
HEADROOM = 0.85


def calibrate(chunks, layers, norm, cfg, embed, texts_ids, report=print, K=None):
    """Max |x| at every RMSNorm input and max |linear output| per layer over
    texts (fp32), then: the residual stream's 1/K (smallest power of two that
    keeps o_proj/down_proj outputs under HEADROOM * 2^15, the ANE linear's
    limit), and each norm's power-of-two pre-scale (eps * s^2)."""
    norms = [l.n1 for l in layers] + [l.n2 for l in layers] + [norm]
    for n in norms:
        n.stat = 0.0
    for l in layers:
        l.ostat = [0.0, 0.0]
    with torch.no_grad():
        for ids in texts_ids:
            for c in chunks:
                c.reset()
            run_eager(chunks, cfg, embed, ids)
    lin = [list(l.ostat) for l in layers]
    fixed, scaled = max(x[0] for x in lin), max(x[1] for x in lin)
    if K is None:
        if fixed > HEADROOM * ANE_LINEAR_MAX:
            raise SystemExit(f"an unscaled linear output reaches {fixed:.0f}")
        K = 1
        while scaled / K > HEADROOM * ANE_LINEAR_MAX:
            K *= 2
    if K != 1:
        for l in layers:
            l.residual_scale(1.0 / K)
        norm.eps /= K * K
        for n in norms:
            n.stat /= K
        for c in chunks:
            c.K = float(K)
    scales = []
    for n in norms:
        s = 1.0
        while (n.stat * s) ** 2 > NORM_SQ_MAX:
            s /= 2.0
        n.mult, n.eps = s, n.eps * s * s
        scales.append((n.stat, s))
    for n in norms:
        n.stat = None
    for l in layers:
        l.ostat = None
    report(f"residual 1/K: K={K} (o/down max {scaled:.0f}, others {fixed:.0f}); norm input max |x| (scaled) "
           f"{max(v[0] for v in scales):.1f}; norm pre-scales != 1: {sum(v[1] != 1.0 for v in scales)}/{len(norms)}")
    return {"K": K, "norm_max": [v[0] for v in scales], "norm_scale": [v[1] for v in scales],
            "linear_out_max": lin}


# Layer-at-a-time execution, for checkpoints whose fp32 copy doesn't fit in
# memory beside the checkpoint itself (MiniCPM5-2B: 10 GB in fp32). Each layer
# is built in fp32 from the checkpoint (bf16 or fp32), applied to every text's
# hidden states, then freed. A per-text cache of capacity S makes it equal to
# run_eager at C = S.

def apply_calibration(layer, i, L, cal):
    """Give a freshly built Layer i of L the calibrated rewrites: the residual
    1/K and its norms' pre-scales (calibrate()'s order: 1/K first)."""
    K = cal["K"]
    if K != 1:
        layer.residual_scale(1.0 / K)
    for n, s in ((layer.n1, cal["norm_scale"][i]), (layer.n2, cal["norm_scale"][L + i])):
        n.mult, n.eps = s, n.eps * s * s
    return layer


def final_norm(hf, cal):
    n = Norm(hf.model.norm.weight, hf.config.rms_norm_eps)
    if cal is not None:
        K, s = cal["K"], cal["norm_scale"][-1]
        n.eps /= K * K
        n.mult, n.eps = s, n.eps * s * s
    return n


def apply_rescales(layer, spec):
    """D17/D19's small-input rule: bring a projection's input towards rms 1
    with a power-of-two scale s folded into the weights that produce it,
    undone by an explicit multiply by 1/s after the projection (not folded
    into its weights, which would push small ones toward fp16 subnormals).
    spec: {"o": s, "down": s}. o's input is attention's output, so v_proj
    takes s; down's input is act(gate) * up, so up_proj takes s."""
    with torch.no_grad():
        if spec.get("o", 1) != 1:
            layer.v.weight.mul_(spec["o"])
            layer.post_o = 1.0 / spec["o"]
        if spec.get("down", 1) != 1:
            layer.up.weight.mul_(spec["down"])
            layer.post_down = 1.0 / spec["down"]
    return layer


# Incoherence processing (QuaRot / SpinQuant's R1 and R2, the rotation half of
# QTIP / EXL3): orthogonal rotations folded into the weights offline, so the
# converted program is unchanged in shape and runs no extra op.
# - R1, a random Hadamard Q on the residual stream: every RMSNorm's weight is
#   folded into the linears it feeds (the norm itself is rotation-invariant:
#   rms(Qh) = rms(h)), readers take W' = W diag(g) Q^T, writers (o, down) Q W,
#   the embedding table E Q^T (the host gathers rows of it) and the head
#   W_head diag(g_final) Q^T. Massive activations, which sit in a few channels
#   of h, are spread over all of them.
# - R2, a random Hadamard per attention head on v and o: v's rows per KV head
#   H W_v, o's columns per head W_o H^T (attention mixes V's rows linearly).
# MiniCPM5-2B is a plain Llama (no muP scalars, untied embeddings).

def hadamard(n, seed):
    """A random orthogonal Hadamard (Sylvester, sign-flipped columns) / sqrt(n), fp64."""
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    if H.shape[0] != n:
        raise ValueError(f"no Sylvester Hadamard of size {n}")
    g = torch.Generator().manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float64)
    return H * s.view(1, -1) / math.sqrt(n)


def rotate_model(hf, seed=0, r2=True, r4=0):
    """Rotate the checkpoint's edges in place (embedding, final norm, head,
    in fp32) and record the rotation for build_layer, which rotates each
    layer as it builds it."""
    cfg = hf.config
    Q = hadamard(cfg.hidden_size, seed)
    Hh = hadamard(cfg.head_dim, seed + 1) if r2 else None
    with torch.no_grad():
        E = hf.model.embed_tokens.weight
        hf.model.embed_tokens.weight = nn.Parameter((E.double() @ Q.T).float(), requires_grad=False)
        gf = hf.model.norm.weight.double()
        W = hf.lm_head.weight
        hf.lm_head.weight = nn.Parameter(torch.cat([(W[j:j + 16384].double() * gf) @ Q.T
                                                    for j in range(0, W.shape[0], 16384)]).float(),
                                         requires_grad=False)
        hf.model.norm.weight = nn.Parameter(torch.ones_like(hf.model.norm.weight, dtype=torch.float32),
                                            requires_grad=False)
    Hb = hadamard(r4, seed + 2) if r4 else None
    hf._rotation = {"Q": Q, "H": Hh, "seed": seed, "r2": r2, "Hb": Hb}
    return hf


def rotate_layer(layer, rotation):
    """R1 (and R2) on a freshly built Layer (fp32 conv weights, norms with weights)."""
    Q, Hh = rotation["Q"], rotation["H"]
    with torch.no_grad():
        for norm, readers in ((layer.n1, ("q", "k", "v")), (layer.n2, ("gate", "up"))):
            g = norm.weight.view(-1).double()
            for p in readers:
                c = getattr(layer, p)
                W = c.weight.view(c.weight.shape[0], -1).double()
                c.weight.copy_(((W * g) @ Q.T).float().reshape(c.weight.shape))
            norm.weight.fill_(1.0)
        for p in ("o", "down"):
            c = getattr(layer, p)
            W = c.weight.view(c.weight.shape[0], -1).double()
            c.weight.copy_((Q @ W).float().reshape(c.weight.shape))
        Hb = rotation.get("Hb")
        if Hb is not None:
            # R4 on down_proj's input: blocks of b channels rotated by Hb at run time (a grouped 1x1 conv,
            # weights b x b per group), and W_down's columns by Hb^T offline, so down(Hb x) = down(x)
            b = Hb.shape[0]
            I = layer.down.weight.shape[1]
            r4 = nn.Conv2d(I, I, 1, groups=I // b, bias=False)
            r4.weight.copy_(Hb.float().view(1, b, b).expand(I // b, b, b).reshape(I, b, 1, 1))
            r4.requires_grad_(False)
            layer.r4 = r4
            Wd = layer.down.weight.view(-1, I // b, b).double()
            layer.down.weight.copy_(torch.einsum("ogj,ij->ogi", Wd, Hb).float().reshape(layer.down.weight.shape))
        if Hh is not None:
            Dh, KVH, H = layer.Dh, layer.KVH, layer.H
            Wv = layer.v.weight.view(KVH, Dh, -1).double()
            layer.v.weight.copy_(torch.einsum("ij,kjd->kid", Hh, Wv).float().reshape(layer.v.weight.shape))
            Wo = layer.o.weight.view(-1, H, Dh).double()
            layer.o.weight.copy_(torch.einsum("ohj,ij->ohi", Wo, Hh).float().reshape(layer.o.weight.shape))
    return layer


def build_layer(hf, i, cal=None, silu="stable"):
    layer = Layer(hf.model.layers[i], hf.config, silu)
    if getattr(hf, "_rotation", None) is not None:
        rotate_layer(layer, hf._rotation)
    if cal is not None:
        apply_calibration(layer, i, hf.config.num_hidden_layers, cal)
    return layer


def stream(hf, texts_ids, cal=None, silu="stable", on_layer=None, on_hidden=None, rows=None, head=True,
           layer_fn=None):
    """fp32 forward of every text, one layer at a time. on_layer(i, layer)
    may modify a layer before it runs (calibration stats, simulated
    quantisation); on_hidden(i, t, h) sees layer i's output for text t.
    layer_fn(i) builds layer i (default: build_layer with cal).
    rows: (lo, hi) logits rows to keep (default all). Returns the logits
    per text ([rows, V] fp32), or the final hidden states if head is False."""
    cfg = hf.config
    L, KVH, Dh = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    K = cal["K"] if cal is not None else 1
    emb = hf.model.embed_tokens.weight
    hs = [emb[torch.tensor(ids)].float().T.reshape(1, -1, 1, len(ids)) / K for ids in texts_ids]
    with torch.no_grad():
        for i in range(L):
            layer = layer_fn(i) if layer_fn is not None else build_layer(hf, i, cal, silu)
            if on_layer is not None:
                on_layer(i, layer)
            for t, h in enumerate(hs):
                S = h.shape[-1]
                kc = torch.zeros(1, KVH, S, Dh)
                vc = torch.zeros(1, KVH, S, Dh)
                cos, sin = rope(cfg, list(range(S)))
                hs[t] = layer(h, 0, kc, vc, torch.tensor([0], dtype=torch.int32), mask_for(0, S, S), cos, sin, S, S)
                if on_hidden is not None:
                    on_hidden(i, t, hs[t])
            del layer
        if not head:
            return hs
        return [logits_of(hf, cal, h, rows) for h in hs]


def logits_of(hf, cal, h, rows=None):
    """fp32 final norm and head over hidden [1, D, 1, S]; rows (lo, hi)."""
    S = h.shape[-1]
    lo, hi = rows if rows is not None else (0, S)
    W = hf.lm_head.weight
    with torch.no_grad():
        x = final_norm(hf, cal)(h[..., lo:hi]).view(hf.config.hidden_size, hi - lo).T     # [rows, D]
        return torch.cat([x @ W[j:j + 16384].float().T for j in range(0, W.shape[0], 16384)], dim=1)


def stream_calibrate(hf, texts_ids, silu="stable", report=print, K=None):
    """calibrate() by stream(): the same statistics and the same rule for K
    and the norms' pre-scales, without the model's fp32 copy in memory."""
    L = hf.config.num_hidden_layers
    nmax = [0.0] * (2 * L + 1)
    lin = [[0.0, 0.0] for _ in range(L)]
    cur = {}

    def on_layer(i, layer):
        layer.n1.stat, layer.n2.stat, layer.ostat = 0.0, 0.0, [0.0, 0.0]
        cur["layer"] = layer

    def on_hidden(i, t, h):          # running maxima over the texts seen so far
        lay = cur["layer"]
        nmax[i], nmax[L + i], lin[i] = lay.n1.stat, lay.n2.stat, list(lay.ostat)

    hs = stream(hf, texts_ids, silu=silu, on_layer=on_layer, on_hidden=on_hidden, head=False)
    nmax[2 * L] = max(float(h.abs().max()) for h in hs)
    fixed, scaled = max(x[0] for x in lin), max(x[1] for x in lin)
    if K is None:
        if fixed > HEADROOM * ANE_LINEAR_MAX:
            raise SystemExit(f"an unscaled linear output reaches {fixed:.0f}")
        K = 1
        while scaled / K > HEADROOM * ANE_LINEAR_MAX:
            K *= 2
    scales = []
    for m in nmax:
        m = m / K
        s = 1.0
        while (m * s) ** 2 > NORM_SQ_MAX:
            s /= 2.0
        scales.append((m, s))
    report(f"residual 1/K: K={K} (o/down max {scaled:.0f}, others {fixed:.0f}); norm input max |x| (scaled) "
           f"{max(v[0] for v in scales):.1f}; norm pre-scales != 1: {sum(v[1] != 1.0 for v in scales)}/{len(scales)}")
    return {"K": K, "norm_max": [v[0] for v in scales], "norm_scale": [v[1] for v in scales],
            "linear_out_max": lin}


