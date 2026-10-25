"""Weight quantisers simulated in PyTorch (fp32 arithmetic, fp16 scales),
shared by probe_quant.py and chunk_trace.py. Weights are [out, in], as
coremltools sees a 1x1 conv's weight."""
import torch

PROJ = ["q", "k", "v", "o", "gate", "up", "down"]


def f16(x):
    return x.half().float()


def q_int8pc(w, clip=None):
    amax = w.abs().amax(1, keepdim=True) if clip is None else clip
    s = f16(amax / 127.0).clamp(min=1e-12)
    return f16(torch.clamp(torch.round(w / s), -127, 127) * s)


def q_int8pc_mse(w):
    amax = w.abs().amax(1, keepdim=True)
    best, err = None, None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        qw = q_int8pc(w, amax * r)
        e = ((qw - w) ** 2).sum(1, keepdim=True)
        if best is None:
            best, err = qw, e
        else:
            m = e < err
            best, err = torch.where(m, qw, best), torch.where(m, e, err)
    return best


def q_int8b(w, n):
    o, i = w.shape
    wb = w.view(o, i // n, n)
    s = f16(wb.abs().amax(2, keepdim=True) / 127.0).clamp(min=1e-12)
    return f16(torch.clamp(torch.round(wb / s), -127, 127) * s).view(o, i)


def pal_lut(w, g, k=256, iters=30):
    """1-D k-means LUT (k entries, fp16) per group of g output channels, all
    groups at once: Lloyd's iterations from a uniform grid over the group's
    range, so the result is never worse than that grid. Returns lut [G, k]."""
    o, i = w.shape
    x = w.reshape(o // g, g * i)
    lo, hi = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values
    c = lo + (hi - lo) * torch.linspace(0, 1, k).view(1, k)
    for _ in range(iters):
        idx = torch.searchsorted((c[:, 1:] + c[:, :-1]) / 2, x)
        sums = torch.zeros_like(c).scatter_add_(1, idx, x)
        cnt = torch.zeros_like(c).scatter_add_(1, idx, torch.ones_like(x))
        c = torch.where(cnt > 0, sums / cnt.clamp(min=1), c)
    return f16(c)


def lut_round(w, lut, g):
    """Each weight to the nearest entry of its group's LUT."""
    o, i = w.shape
    x = w.reshape(o // g, g * i)
    idx = torch.searchsorted((lut[:, 1:] + lut[:, :-1]) / 2, x.contiguous())
    return torch.gather(lut, 1, idx).view(o, i)


def q_pal8(w, g):
    return lut_round(w, pal_lut(w, g), g)


def quantizer(name):
    if name == "int8pc":
        return q_int8pc
    if name == "int8pc-mse":
        return q_int8pc_mse
    if name.startswith("int8b"):
        n = int(name[5:])
        return lambda w: q_int8b(w, n)
    if name.startswith("pal8g"):
        g = int(name[5:])
        return lambda w: q_pal8(w, g)
    if name == "fp16":
        return f16
    if name == "none":
        return None
    raise SystemExit(f"unknown quantiser {name}")
