"""gptq.py <hf_src> <calibration.json> <out_dir> [--grid int8|pal8] [--group 16] [--act-order]
          [--tokens 8192] [--damp 0.01] [--skip-head]

GPTQ (Frantar et al., 2022) onto the two compressed weight formats that run
on the ANE (per-block int8 and int4 fall back to the CPU or fail to load):
- int8: coremltools' linear_symmetric int8, one scale per output channel
  (max |w| / 127, range [-127, 127]); the element at the max is pinned to
  +-127 so linear_quantize_weights recovers the same scale and integers;
- pal8: an 8-bit palette per group of --group output channels (palettize_
  weights' per_grouped_channel layout), the palette fixed beforehand by
  k-means as coremltools computes it; convert.py stores it with the
  palettizer's "unique" mode, which keeps the values exactly;
- pal4: a 4-bit palette per group (--group 1: one per output channel, which
  can hold any per-channel int4 grid), from k-means weighted by each input
  channel's mean squared activation.
Instead of rounding every weight to its nearest grid point, GPTQ rounds the
input channels in order (--act-order: by decreasing Hessian diagonal) and
moves each rounding error onto the channels not yet rounded, weighted by the
inverse Hessian of the layer's inputs (H = X^T X over calibration tokens),
so the layer's output error is smaller than round-to-nearest's on the same
grid.

Runs layer by layer over the rewritten decoder (llama_ane.stream's layers,
with reference.py's calibration), calibration inputs propagated through the
already-quantised layers. Calibration text: texts.gptq_texts (repository
text disjoint from the eval spans, and chat-formatted prompts).
Writes <out_dir>/layer_<i>.pt ({proj: (codes, scale or palette)}), head.pt
and meta.json; convert.py --qweights and probe_quant.py --qweights load it."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama_ane as LA
import texts as T

PROJ = ["q", "k", "v", "o", "gate", "up", "down"]
HESS = {"q": "attn", "k": "attn", "v": "attn", "o": "o", "gate": "mlp", "up": "mlp", "down": "down"}


def f16(x):
    return x.half().float()


class Int8PerChannel:
    """coremltools' linear_symmetric int8 grid, one scale per output channel
    (max |w| / 127), the element at the max pinned to +-127 so the scale
    survives requantisation."""

    def __init__(self, W, qmax=127):
        self.qmax = qmax              # 127; 7 gives the per-channel int4 grid (stored as a palette per channel)
        self.scale = f16(W.abs().amax(1).float() / qmax).double().clamp(min=1e-12)
        self.pin = W.abs().argmax(1)
        self.codes = torch.zeros(W.shape, dtype=torch.int8)

    def round(self, w, col):
        m = float(self.qmax)
        q = torch.clamp(torch.round(w / self.scale), -m, m)
        hit = self.pin == col
        q[hit] = m * torch.sign(w[hit])
        q[hit & (w == 0)] = m
        self.codes[:, col] = q.to(torch.int8)
        return q * self.scale

    def result(self):
        return self.codes, self.scale.float()


class Lut:
    """An 8-bit palette per group of g output channels (palettize_weights'
    per_grouped_channel layout): the LUT is fixed beforehand (k-means of the
    original weights), GPTQ only chooses each weight's entry."""

    def __init__(self, W, lut, g):
        self.g = g
        self.lut = lut.double()                                   # [G, 256], fp16 values
        rows = torch.arange(W.shape[0]) // g
        self.row_lut = self.lut[rows]                             # [out, 256]
        self.row_mid = (self.row_lut[:, 1:] + self.row_lut[:, :-1]) / 2
        self.codes = torch.zeros(W.shape, dtype=torch.uint8)

    def round(self, w, col):
        idx = torch.searchsorted(self.row_mid, w.unsqueeze(1).contiguous()).clamp(max=self.lut.shape[1] - 1)
        self.codes[:, col] = idx.squeeze(1).to(torch.uint8)
        return torch.gather(self.row_lut, 1, idx).squeeze(1)

    def result(self):
        return self.codes, self.lut.float()


def gptq(W, H, grid, damp=0.01, chunk=128, act_order=False):
    """GPTQ of W [out, in] under H [in, in] onto `grid` (Int8PerChannel or
    Lut, built on the original W). act_order rounds the columns in order of
    decreasing H diagonal (GPTQ's desc_act). Returns grid.result()."""
    W = W.clone().double()
    H = H.clone().double()
    n_out, n_in = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(n_in)
    W, H = W[:, perm], H[perm][:, perm]
    eye = torch.mean(torch.diag(H)) * torch.eye(n_in, dtype=H.dtype)
    while True:                         # raise the damping until H is numerically positive definite
        try:
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H + damp * eye)), upper=True)
            break
        except torch._C._LinAlgError:
            damp *= 2
            print(f"  damping raised to {damp}", flush=True)
    for i1 in range(0, n_in, chunk):
        i2 = min(i1 + chunk, n_in)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            wq = grid.round(w, int(perm[i1 + i]))
            err = (w - wq) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return grid.result()


def lloyd_lut(W, g, k, importance=None, iters=60, pin=False):
    """k-entry palette per group of g output channels, all groups at once:
    Lloyd's k-means from the group's quantiles, each weight weighted by the
    importance of its input channel (mean squared activation, the imatrix;
    sensitivity-weighted k-means in the spirit of SKMPalettizer). For small
    k (4-bit), where Lloyd converges well. pin keeps the group's min and max
    as the first and last entries (outlier-preserving: k-means alone lets
    the rare extreme weights, which this model depends on, fall into a
    centroid well inside them). Returns fp16 values [G, k]."""
    o, i = W.shape
    if (o // g) * g * i > 2 ** 23:            # torch.quantile's size limit: a slice of groups at a time
        step = max(1, 2 ** 23 // (g * i)) * g
        return torch.cat([lloyd_lut(W[r:r + step], g, k, importance, iters, pin) for r in range(0, o, step)])
    x = W.reshape(o // g, g * i).float()
    wt = torch.ones(i) if importance is None else importance.float().clamp(min=1e-12)
    wt = (wt / wt.mean()).repeat(g).view(1, -1).expand_as(x).contiguous()
    q = torch.linspace(0, 1, k) if pin else torch.linspace(0, 1, k + 2)[1:-1]
    c = torch.quantile(x, q, dim=1).T.contiguous()       # [G, k]
    lo, hi = x.min(1).values, x.max(1).values
    for _ in range(iters):
        idx = torch.searchsorted((c[:, 1:] + c[:, :-1]) / 2, x.contiguous())
        sums = torch.zeros_like(c).scatter_add_(1, idx, x * wt)
        cnt = torch.zeros_like(c).scatter_add_(1, idx, wt)
        c = torch.where(cnt > 0, sums / cnt.clamp(min=1e-30), c)
        c, _ = torch.sort(c, dim=1)
        if pin:
            c[:, 0], c[:, -1] = lo, hi
    return f16(c)


def kmeans_lut(W, g, k=256):
    """k-means LUT per group of g output channels, as palettize_weights
    computes it (scikit-learn k-means++ on the fp16 weights), on the unique
    values weighted by their counts (bf16 checkpoints have few)."""
    import numpy as np
    from sklearn.cluster import KMeans
    o, i = W.shape
    out = np.zeros((o // g, k), dtype=np.float16)
    x = W.reshape(o // g, g * i).half().numpy()
    for r in range(o // g):
        v, cnt = np.unique(x[r], return_counts=True)
        n = min(len(v), k)
        km = KMeans(n, init="k-means++", tol=1e-2, n_init=1, random_state=0).fit(
            v.reshape(-1, 1).astype(np.float32), sample_weight=cnt)
        c = np.sort(km.cluster_centers_.reshape(-1)).astype(np.float16)
        out[r, :n] = c
        out[r, n:] = c[-1]
    return torch.from_numpy(out.astype(np.float32))


def dequant(q, s, block=0):
    if q.dtype == torch.uint8:                                    # Lut codes, s = lut [G, 256]
        g = q.shape[0] // s.shape[0]
        rows = torch.arange(q.shape[0]) // g
        return torch.gather(s[rows], 1, q.long())
    if block == 0:
        return q.float() * s.view(-1, 1)
    return (q.float().view(q.shape[0], -1, block) * s.unsqueeze(-1)).view(q.shape)


class HessRec(torch.nn.Module):
    def __init__(self, conv, acc, key):
        super().__init__()
        self.conv, self.acc, self.key = conv, acc, key

    def forward(self, x):
        X = x.reshape(x.shape[1], -1).double()          # [in, S]
        h = X @ X.T
        if self.key in self.acc:
            self.acc[self.key][0].add_(h)
            self.acc[self.key][1] += X.shape[1]
        else:
            self.acc[self.key] = [h, X.shape[1]]
        return self.conv(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("cal")
    ap.add_argument("out")
    ap.add_argument("--grid", default="int8", choices=["int8", "int4", "pal8", "pal6", "pal4"])
    ap.add_argument("--group", type=int, default=16, help="pal8: output channels per palette")
    ap.add_argument("--pin", action="store_true", help="pal4/pal6: keep each group's min and max in its palette")
    ap.add_argument("--rot", type=int, default=None, help="rotate the model (llama_ane.rotate_model) with this seed")
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--r4", type=int, default=0)
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-head", action="store_true")
    ap.add_argument("--imatrix", default=None,
                    help="also save each linear's importance (mean squared input activation per input channel, "
                         "llama.cpp imatrix style) from the same calibration pass to this .pt file")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.src)
    hf = LA.load_hf(a.src, dtype=torch.bfloat16)
    if a.rot is not None:
        LA.rotate_model(hf, a.rot, r2=not a.no_r2, r4=a.r4)
    if Path(a.cal).exists():
        cal = json.load(open(a.cal))
    else:                       # e.g. a rotated model's own calibration
        cal = LA.stream_calibrate(hf, T.calib_texts(tok))
        json.dump(cal, open(a.cal, "w"))
    cfg = hf.config
    L, KVH, Dh = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    texts = T.gptq_texts(tok, a.tokens)

    def make_grid(W, H=None):
        if a.grid in ("int8", "int4"):
            return Int8PerChannel(W.double(), 127 if a.grid == "int8" else 7)
        if a.grid in ("pal4", "pal6"):
            k = 2 ** int(a.grid[3:])
            return Lut(W, lloyd_lut(W.float(), a.group, k, None if H is None else torch.diag(H), pin=a.pin), a.group)
        return Lut(W, kmeans_lut(W.float(), a.group), a.group)

    def rtn(W, grid):
        if a.grid in ("int8", "int4"):
            s, m = grid.scale.float().view(-1, 1), grid.qmax
            return f16(torch.clamp(torch.round(W / s), -m, m) * s)
        mid = (grid.row_lut[:, 1:] + grid.row_lut[:, :-1]) / 2
        idx = torch.searchsorted(mid, W.double().contiguous()).clamp(max=grid.lut.shape[1] - 1)
        return torch.gather(grid.row_lut, 1, idx).float()
    print(f"calibration: {len(texts)} texts, {sum(map(len, texts))} tokens", flush=True)
    emb = hf.model.embed_tokens.weight
    hs = [emb[torch.tensor(ids)].float().T.reshape(1, -1, 1, len(ids)) / cal["K"] for ids in texts]

    def run(layer):
        for t, h in enumerate(hs):
            S = h.shape[-1]
            cos, sin = LA.rope(cfg, list(range(S)))
            yield layer(h, 0, torch.zeros(1, KVH, S, Dh), torch.zeros(1, KVH, S, Dh),
                        torch.tensor([0], dtype=torch.int32), LA.mask_for(0, S, S), cos, sin, S, S)

    stats = {}
    imatrix = {}           # "<layer>.<input>": mean x^2 per input channel; inputs attn (q/k/v), o, mlp (gate/up), down
    for i in range(L):
        t0 = time.time()
        layer = LA.build_layer(hf, i, cal)
        acc = {}
        convs = {p: getattr(layer, p) for p in PROJ}
        for p in PROJ:
            if p in ("q", "gate", "o", "down"):         # one recorder per distinct input
                setattr(layer, p, HessRec(convs[p], acc, HESS[p]))
        for _ in run(layer):
            pass
        res = {}
        st = {}
        for p in PROJ:
            setattr(layer, p, convs[p])
            H = acc[HESS[p]][0] / acc[HESS[p]][1]
            W = convs[p].weight.view(convs[p].weight.shape[0], -1)
            grid = make_grid(W, H)
            q, s = gptq(W, H, grid, a.damp, act_order=a.act_order)
            Wq = dequant(q, s)
            # output error (relative, under H) of GPTQ vs round to nearest
            def herr(Wx):
                D = (Wx - W).double()
                return float(torch.sqrt(torch.einsum("oi,ij,oj->", D, H, D) / torch.einsum("oi,ij,oj->", W.double(), H, W.double())))
            Wr = rtn(W, grid)
            st[p] = {"gptq": herr(Wq), "rtn": herr(Wr) if Wr is not None else None}
            res[p] = (q, s)
            with torch.no_grad():
                convs[p].weight.copy_(Wq.view_as(convs[p].weight))
        torch.save(res, out / f"layer_{i}.pt")
        hs = list(run(layer))
        stats[i] = st
        for key, (h, n) in acc.items():
            imatrix[f"{i}.{key}"] = (torch.diag(h) / n).float()
        print(f"layer {i}: {time.time() - t0:.0f}s " + " ".join(
            f"{p} {v['gptq']:.4f}" + (f"/{v['rtn']:.4f}" if v["rtn"] else "") for p, v in st.items()), flush=True)
        del layer, convs, acc
    if not a.skip_head:
        t0 = time.time()
        norm = LA.final_norm(hf, cal)
        X = torch.cat([norm(h).reshape(cfg.hidden_size, -1) for h in hs], 1).double()
        H = X @ X.T / X.shape[1]
        imatrix["head"] = torch.diag(H).float()
        W = hf.lm_head.weight.float()
        q, s = gptq(W, H, make_grid(W, H), a.damp, act_order=a.act_order)
        torch.save({"head": (q, s)}, out / "head.pt")
        print(f"head: {time.time() - t0:.0f}s", flush=True)
    if a.imatrix:
        torch.save({"inputs": imatrix, "tokens": sum(map(len, texts)), "calibration": "texts.gptq_texts"}, a.imatrix)
    json.dump({"block": 0, "rot": a.rot, "r4": a.r4, "r2": not a.no_r2, "pin": a.pin, "grid": a.grid, "group": a.group, "act_order": a.act_order, "tokens": sum(map(len, texts)), "damp": a.damp, "head": not a.skip_head,
               "stats": stats}, open(out / "meta.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
