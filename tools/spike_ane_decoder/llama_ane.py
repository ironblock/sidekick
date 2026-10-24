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
        self.weight = nn.Parameter(weight.detach().clone().view(1, -1, 1, 1), requires_grad=False)
        self.eps = float(eps)
        self.mult = 1.0
        self.stat = None            # calibration: max |x|

    def forward(self, x):
        if self.stat is not None:
            self.stat = max(self.stat, float(x.abs().max()))
        y = x * self.mult if self.mult != 1.0 else x
        return y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + self.eps) * self.weight


def conv(lin, scale=1.0):
    w = lin.weight.detach()
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
        h = h + o
        x = self.n2(h)
        g, u = self.gate(x), self.up(x)
        d = self.down(self.act(g) * u)
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
            w = head.weight.detach()
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


def load_hf(src):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, attn_implementation="eager").eval()
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


