"""LFM2 hybrids (LFM2.5-Embedding): double-gated short convolutions
interleaved with full-attention blocks (grouped-query, QK-norm), made
bidirectional by the checkpoint's own remote code (docs/DECISIONS.md D19 and
its amendment). The remote code is loaded with trust_remote_code: read
modeling_lfm2_bidirectional.py before converting.

What conversion needs:
- MASK. The remote code's bidirectional mask uses a -1e9 pad bias, -inf in
  fp16; the patch builds the same pad-only (1, 1, 1, S) mask at -30000.
  transformers' sdpa path passes an explicit scale, so attention converts to
  explicit ops, never the fused one.
- TRACEABLE helpers: rotate_half and repeat_kv, and the short conv, whose
  upstream forward computes conv1d's padding and groups from tensor shapes.
- PAD ZEROING before every conv (D19's constraint D). Attention masks
  silence pad KEYS, but a symmetric conv mixes neighbours unconditionally,
  so pad states would leak into the last real tokens and, through
  attention, into CLS (parity 0.905 without it). Zeroing them reproduces the
  unpadded forward exactly and makes embeddings bucket-invariant.
  `--no-pad-zeroing` builds the negative control. It needs bias-free
  projections (conv_bias false), checked at load.
- PRECISION REWRITE (D19 amendment): this model's activations are tiny
  (output-projection inputs at rms 0.003-0.07), under the ANE linear's
  small-input floor, so power-of-two scales bring each projection's input to
  rms ~1 (folded into the operator norm, v_proj, and in_proj's B and C rows)
  and Descale undoes them before the residual add; the coarse native silu
  becomes TanhSilu, its factor 2 divided out with the MLP's rescale.

The patches are installed on transformers' LFM2 module and class, after the
remote code (which installs its own) has loaded.
"""

import torch

from ..techniques import activations, precision, traceable
from ..calibrate import Stat
from . import Backbone

MASK_ADD = -30000.0
QK_MAX = 150.0          # cap on |q|, |k| entering their RMSNorms, which square them


def _make_traceable_shortconv_forward(zero_pads):
    # transformers' upstream non-causal short-conv computes F.conv1d
    # padding/groups from tensor shapes, which jit.trace turns into traced
    # values that conv1d rejects. Shapes are static per bucket, so int() pins
    # them. Math is identical for odd kernels ('same' symmetric padding).
    def forward(
        self, hidden_states, past_key_values=None, cache_position=None,
        attention_mask=None,
    ):
        # pad zeroing (zero_pads=True): the symmetric conv mixes neighbours
        # unconditionally, so pad states are zeroed before EVERY conv, which
        # reproduces the unpadded forward exactly at all real positions
        # (F.conv1d edge-pads with zeros). The 4D additive mask is 0 for real
        # tokens and MASK_ADD for pads, so `== 0` recovers the keep-mask.
        if zero_pads and attention_mask is not None:
            keep = (attention_mask == 0).to(hidden_states.dtype).reshape(1, -1, 1)
            hidden_states = hidden_states * keep
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x
        k = int(self.conv.weight.shape[-1])
        assert k % 2 == 1, "even conv kernels need an output-length correction"
        conv_out = torch.nn.functional.conv1d(
            Bx, weight=self.conv.weight, bias=self.conv.bias,
            stride=1, padding=k // 2, dilation=1, groups=int(Bx.shape[1]),
        )
        y = C * conv_out
        return self.out_proj(y.transpose(-1, -2).contiguous())

    return forward


def _fp16_safe_bidirectional_mask(config, **kwargs):
    # the pad-only additive mask the remote code installs, at MASK_ADD
    # instead of -1e9. Cache-free trace: kv_len == q_len, and a (1, 1, 1, S)
    # mask broadcasts over query positions inside SDPA.
    embeds = kwargs.get("inputs_embeds")
    if embeds is None:
        embeds = kwargs.get("input_embeds")
    attention_mask = kwargs.get("attention_mask")
    pad = 1.0 - attention_mask.to(embeds.dtype)
    return pad[:, None, None, :] * MASK_ADD


def install_patches(conv_pad_zeroing=True):
    import transformers.integrations.sdpa_attention as _sdpa_attention
    import transformers.models.lfm2.modeling_lfm2 as _lfm2
    traceable.install(_lfm2)
    traceable.install(_sdpa_attention, names=("repeat_kv",))
    # after the model has loaded: the remote code installs its own
    # create_causal_mask/slow_forward, and the last patch installed wins
    _lfm2.create_causal_mask = _fp16_safe_bidirectional_mask
    _lfm2.Lfm2ShortConv.slow_forward = _make_traceable_shortconv_forward(conv_pad_zeroing)


class ScaledMLP(torch.nn.Module):
    """SwiGLU with the tanh silu, w3 pre-scaled by m and the output descaled."""

    def __init__(self, mlp, m):
        super().__init__()
        self.w1, self.w3, self.w2 = mlp.w1, mlp.w3, mlp.w2
        with torch.no_grad():
            self.w3.weight.mul_(m)
        self.act = activations.TanhSilu()
        self.inv = 1.0 / (activations.TanhSilu.GAIN * m)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x)) * self.inv


class Lfm2Backbone(Backbone):
    def call(self, w, x):
        return getattr(w, self.attr)(
            input_ids=x["input_ids"].long(),
            attention_mask=x["attention_mask"].long(),
            position_ids=w.position_ids,
        )

    def reference(self, ids, token_type_ids=None):
        t = torch.tensor([ids])
        with torch.no_grad():
            return self.model(input_ids=t, attention_mask=torch.ones_like(t))


def load(src, tok, *, conv_pad_zeroing=True):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
    model.config.use_cache = False
    if getattr(model.config, "conv_bias", False):
        raise SystemExit("conv_bias=true would break pad zeroing's exactness (zeroed pad states must map to zero)")
    install_patches(conv_pad_zeroing)
    from ..tokenizer import special_ids
    return Lfm2Backbone(family="lfm2", model=model, config=model.config, special_ids=tuple(special_ids(tok)),
                        forbid_ops={"silu", "gelu"})


def _calibrate(model, run):
    """fp32 stats (unpadded forwards) of every tensor the rewrite rescales."""
    stats = {}

    def stat(name):
        return stats.setdefault(name, Stat())

    def pre(name):
        return lambda mod, args: stat(name).add(args[0])

    def post(name):
        return lambda mod, args, out: stat(name).add(out)

    def conv_parts(i):
        def f(mod, args, out):
            B, C, x = out.detach().chunk(3, dim=-1)
            stat(f"{i}.bx").add(B * x)
        return f

    hooks = []
    for i, layer in enumerate(model.layers):
        if layer.is_attention_layer:
            a = layer.self_attn
            hooks += [a.q_proj.register_forward_pre_hook(pre(f"{i}.qkv_in")),
                      a.q_proj.register_forward_hook(post(f"{i}.q")),
                      a.k_proj.register_forward_hook(post(f"{i}.k")),
                      a.out_proj.register_forward_pre_hook(pre(f"{i}.o_in")),
                      a.out_proj.register_forward_hook(post(f"{i}.o_out"))]
        else:
            c = layer.conv
            hooks += [c.in_proj.register_forward_pre_hook(pre(f"{i}.in")),
                      c.in_proj.register_forward_hook(conv_parts(i)),
                      c.out_proj.register_forward_pre_hook(pre(f"{i}.y")),
                      c.out_proj.register_forward_hook(post(f"{i}.y_out"))]
        f = layer.feed_forward
        hooks += [f.w2.register_forward_pre_hook(pre(f"{i}.d_in")),
                  f.w2.register_forward_hook(post(f"{i}.d_out"))]
    with torch.no_grad():
        run()
    for h in hooks:
        h.remove()
    return stats


def precision_rewrite(calibration, tok, report=print):
    """A rewrite: calibrate on `calibration` (a core.Calibration), then apply
    the power-of-two rescales and the tanh silu. Exact in fp32."""
    from ..core import Calibration
    from ..tokenizer import encode
    if not isinstance(calibration, Calibration):
        raise TypeError("precision_rewrite calibrates on a core.Calibration, never on evaluation cases")

    def rewrite(backbone):
        model = backbone.model

        def run():
            for text in calibration.texts:
                backbone.reference(encode(tok, text))

        stats = _calibrate(model, run)
        scales = []
        for i, layer in enumerate(model.layers):
            if layer.is_attention_layer:
                a = layer.self_attn
                # q/k/v inputs: scale the norm feeding them. q and k are
                # re-normalized per head, so only their RMSNorm eps moves (x s^2).
                s_in = precision.input_scale(stats[f"{i}.qkv_in"])
                qk = max(stats[f"{i}.q"].max, stats[f"{i}.k"].max)
                while s_in > 1.0 and qk * s_in > QK_MAX:
                    s_in /= 2.0
                # out_proj input: v arrives scaled by s_in; v_proj adds the rest
                s_o = precision.input_scale(stats[f"{i}.o_in"], stats[f"{i}.o_out"])
                with torch.no_grad():
                    layer.operator_norm.weight.mul_(s_in)
                    a.v_proj.weight.mul_(s_o / s_in)
                a.q_layernorm.variance_epsilon *= s_in * s_in
                a.k_layernorm.variance_epsilon *= s_in * s_in
                a.out_proj = precision.Descale(a.out_proj, 1.0 / s_o)
                scales.append(f"L{i} attn {s_in:g}/{s_o:g}")
            else:
                c = layer.conv
                h = c.out_proj.weight.shape[0]
                # in_proj input via the norm; B*x (the conv's input) via B's
                # rows; y = C * conv(Bx) (out_proj's input) via C's rows. A row
                # factor below 1 is still an exact power-of-two weight scale.
                s_in = precision.input_scale(stats[f"{i}.in"])
                s_bx = precision.input_scale(stats[f"{i}.bx"])
                s_y = precision.input_scale(stats[f"{i}.y"], stats[f"{i}.y_out"])
                with torch.no_grad():
                    layer.operator_norm.weight.mul_(s_in)
                    c.in_proj.weight[:h].mul_(s_bx / (s_in * s_in))
                    c.in_proj.weight[h:2 * h].mul_(s_y / (s_in * s_bx))
                c.out_proj = precision.Descale(c.out_proj, 1.0 / s_y)
                scales.append(f"L{i} conv {s_in:g}/{s_bx:g}/{s_y:g}")
            m = precision.input_scale(stats[f"{i}.d_in"], stats[f"{i}.d_out"], activations.TanhSilu.GAIN)
            layer.feed_forward = ScaledMLP(layer.feed_forward, m)
            scales[-1] += f" mlp {activations.TanhSilu.GAIN * m:g}"
        report("scales (attn in/out, conv in/Bx/y, mlp): " + "; ".join(scales))
    return rewrite
