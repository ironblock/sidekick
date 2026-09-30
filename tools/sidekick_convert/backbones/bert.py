"""BERT-family encoders: BERT, RoBERTa and XLM-R (post-norm, absolute
positions, token types).

What conversion needs (docs/DECISIONS.md D15, D25, D28):
- ATTENTION. "explicit" loads transformers' eager path (matmul -> softmax ->
  matmul) and replaces its extended attention mask, which fills with
  finfo(float32).min (-inf in fp16, NaN through softmax), by a finite
  additive -30000 of the same geometry. "fused" loads the sdpa path, which
  coremltools turns into Core ML's fused attention op: the ANE ignores its
  mask here, and bge-small's artifact is correct only through a Core ML
  fallback (D25). It exists only to keep bge-small's existing artifact
  byte-identical; new models use "explicit", and moving bge-small over is a
  separate, measured change.
- STATIC SHAPES. position_ids is a buffer, never computed from a traced size.
  RoBERTa and XLM-R number positions from padding_idx + 1, so their buffer
  is offset by 2 (position_offset="auto" keys on model_type, never on the
  tokenizer or pad id: paraphrase-multilingual-MiniLM is model_type "bert"
  with an XLM-R tokenizer and takes no offset).
- TOKEN TYPES, one of:
  - "zeros": a zero buffer, as a single-text classifier sees them;
  - "input": an int32 token_type_ids port, for pair models (rerankers);
  - "none": pass nothing, so the embeddings use their own registered zeros.
    bge-small's artifact was built this way.
- RANGE. BERT is post-norm: there is no residual stream to rescale, so the
  ANE linear's 2^15 limit (D25 amendment) is only checked, over the gate
  texts, and the conversion fails when it doesn't hold.
"""

import torch

from ..techniques import masks, saturation
from ..calibrate import all_linears, linear_maxima
from . import Backbone

OFFSET_MODEL_TYPES = ("roberta", "xlm-roberta", "camembert")


def _finite_extended_mask(self, attention_mask, input_shape=None, dtype=None, **_):
    # the geometry of ModuleUtilsMixin.get_extended_attention_mask for an
    # encoder, (bsz, 1, 1, seq), with a finite additive constant
    return masks.key_padding(attention_mask)


class BertBackbone(Backbone):
    def __init__(self, *, token_types, position_offset, **kw):
        super().__init__(**kw)
        self.token_types = token_types
        self.position_offset = position_offset

    def buffers(self, seq):
        position_ids = torch.arange(seq, dtype=torch.long)
        if self.position_offset:
            position_ids = position_ids + self.position_offset
        b = {"position_ids": position_ids.unsqueeze(0)}
        if self.token_types == "zeros":
            b["token_type_ids"] = torch.zeros((1, seq), dtype=torch.long)
        return b

    def call(self, w, x):
        kw = {"input_ids": x["input_ids"].long(), "attention_mask": x["attention_mask"].long()}
        if self.token_types == "zeros":
            kw["token_type_ids"] = w.token_type_ids
        elif self.token_types == "input":
            kw["token_type_ids"] = x["token_type_ids"].long()
        kw["position_ids"] = w.position_ids
        return getattr(w, self.attr)(**kw)

    def reference(self, ids, token_type_ids=None):
        t = torch.tensor([ids])
        kw = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if token_type_ids is not None:
            kw["token_type_ids"] = torch.tensor([token_type_ids])
        with torch.no_grad():
            return self.model(**kw)

    def check_linear_range(self, texts, tok):
        """The ANE linear's 2^15 rule, checked over `texts` (a gate, so the
        evaluation texts will do: it decides nothing about the artifact).
        Returns (name, value, headroom factor) of the largest output."""
        from ..tokenizer import encode

        def run():
            for text in texts:
                self.reference(encode(tok, text))

        return saturation.check(linear_maxima(run, all_linears(self.model)))


def load(src, tok, *, task=None, attention="explicit", token_types="zeros", position_offset="auto"):
    """Load a BERT-family checkpoint for conversion.

    task: None for the bare encoder (pooling heads), "sequence-classification"
    for the checkpoint's own classification head (classifiers, rerankers).
    """
    from transformers import AutoModel, AutoModelForSequenceClassification
    if attention not in ("explicit", "fused"):
        raise ValueError(f"attention must be explicit or fused, not {attention!r}")
    if token_types not in ("zeros", "input", "none"):
        raise ValueError(f"token_types must be zeros, input or none, not {token_types!r}")
    cls = {None: AutoModel, "sequence-classification": AutoModelForSequenceClassification}[task]
    impl = "eager" if attention == "explicit" else "sdpa"
    model = cls.from_pretrained(src, dtype=torch.float32, attn_implementation=impl).eval()
    config = model.config
    if position_offset == "auto":
        position_offset = config.pad_token_id + 1 if config.model_type in OFFSET_MODEL_TYPES else 0
    if attention == "explicit":
        for m in {id(model): model, id(model.base_model): model.base_model}.values():
            m.get_extended_attention_mask = _finite_extended_mask.__get__(m)
    from ..tokenizer import special_ids
    return BertBackbone(family="bert", model=model, config=config, special_ids=tuple(special_ids(tok)),
                        token_types=token_types, position_offset=position_offset)
