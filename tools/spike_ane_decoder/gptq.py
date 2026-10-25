"""gptq.py <hf_src> <calibration.json> <out_dir> [--block 0] [--tokens 8192] [--damp 0.01] [--skip-head]

GPTQ (Frantar et al., 2022) for the int8 weights Core ML stores: linear
symmetric int8, one scale per output channel (--block 0) or per block of
--block input channels, range [-127, 127] (coremltools' linear_symmetric).
Instead of rounding every weight to its nearest grid point, GPTQ rounds the
input channels in order and moves each rounding error onto the channels not
yet rounded, weighted by the inverse Hessian of the layer's inputs
(H = X^T X over calibration tokens), so the layer's output error is
smaller than round-to-nearest's for the same grid.

The result is a weight on coremltools' own grid: each row (or block) keeps
an element at +-127 * scale, so linear_quantize_weights, which takes the
scale from the row's max |w|, recovers the same scale and integers. convert.py
--qweights and probe_quant.py --qweights load it.

Runs layer by layer over the rewritten decoder (llama_ane.stream's layers,
with reference.py's calibration), calibration inputs propagated through the
already-quantised layers. Calibration text: texts.gptq_texts (repository
text disjoint from the eval spans, and chat-formatted prompts).
Writes <out_dir>/layer_<i>.pt ({proj: (int8 [out, in], fp32 scale)}),
head.pt and meta.json."""
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


def gptq(W, H, block=0, damp=0.01, chunk=128):
    """W [out, in] fp32, H [in, in]. Returns (int8 q, scale [out] or [out, in/block])."""
    W = W.clone().double()
    H = H.clone().double()
    n_out, n_in = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n_in, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(n_out, n_in, dtype=torch.int8)
    rows = torch.arange(n_out)
    if block == 0:
        scale = f16(W.abs().amax(1).float() / 127.0).double().clamp(min=1e-12)
        pin = W.abs().argmax(1)
        scales = scale
    else:
        scales = torch.zeros(n_out, n_in // block, dtype=torch.float64)
    for i1 in range(0, n_in, chunk):
        i2 = min(i1 + chunk, n_in)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if block and col % block == 0:
                blk = W1[:, i:i + block] if i + block <= i2 - i1 else W[:, col:col + block]
                scale = f16(blk.abs().amax(1).float() / 127.0).double().clamp(min=1e-12)
                pin = col + blk.abs().argmax(1)
                scales[:, col // block] = scale
            w = W1[:, i]
            q = torch.clamp(torch.round(w / scale), -127, 127)
            hit = pin == col
            q[hit] = 127.0 * torch.sign(w[hit])
            q[hit & (w == 0)] = 127.0
            Q[:, col] = q.to(torch.int8)
            err = (w - q * scale) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scales.float()


def dequant(q, s, block=0):
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
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--skip-head", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.src)
    cal = json.load(open(a.cal))
    hf = LA.load_hf(a.src, dtype=torch.bfloat16)
    cfg = hf.config
    L, KVH, Dh = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    texts = T.gptq_texts(tok, a.tokens)
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
            q, s = gptq(W, H, a.block, a.damp)
            Wq = dequant(q, s, a.block)
            # output error (relative, under H) of GPTQ vs round to nearest
            def herr(Wx):
                D = (Wx - W).double()
                return float(torch.sqrt(torch.einsum("oi,ij,oj->", D, H, D) / torch.einsum("oi,ij,oj->", W.double(), H, W.double())))
            sr = f16(W.abs().amax(1) / 127.0).clamp(min=1e-12).view(-1, 1)
            Wr = f16(torch.clamp(torch.round(W / sr), -127, 127) * sr) if a.block == 0 else None
            st[p] = {"gptq": herr(Wq), "rtn": herr(Wr) if Wr is not None else None}
            res[p] = (q, s)
            with torch.no_grad():
                convs[p].weight.copy_(Wq.view_as(convs[p].weight))
        torch.save(res, out / f"layer_{i}.pt")
        hs = list(run(layer))
        stats[i] = st
        print(f"layer {i}: {time.time() - t0:.0f}s " + " ".join(
            f"{p} {v['gptq']:.4f}" + (f"/{v['rtn']:.4f}" if v["rtn"] else "") for p, v in st.items()), flush=True)
        del layer, convs, acc
    if not a.skip_head:
        t0 = time.time()
        norm = LA.final_norm(hf, cal)
        X = torch.cat([norm(h).reshape(cfg.hidden_size, -1) for h in hs], 1).double()
        H = X @ X.T / X.shape[1]
        W = hf.lm_head.weight.float()
        q, s = gptq(W, H, a.block, a.damp)
        torch.save({"head": (q, s)}, out / "head.pt")
        print(f"head: {time.time() - t0:.0f}s", flush=True)
    json.dump({"block": a.block, "tokens": sum(map(len, texts)), "damp": a.damp, "head": not a.skip_head,
               "stats": stats}, open(out / "meta.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
