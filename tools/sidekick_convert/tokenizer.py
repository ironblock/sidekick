"""The tokenizer sidekick loads (tokenizer.json), and encoding with it.

sidekick tokenizes with the Rust `tokenizers` crate from tokenizer.json, so
the converter's gate cases and the parity references encode with the same
file through Python's `tokenizers`, never with AutoTokenizer, whose Python
side can differ (pair handling, special tokens). sidekick truncates and pads
by itself, so the installed file must enable neither.

prepare(), with mode:
- "verbatim": copy the checkpoint's tokenizer.json byte for byte. Existing
  models use it, so their tokenizer_sha256 (pinned by parity references)
  never changes.
- "clean": copy it byte for byte unless it enables padding or truncation;
  then load it, call no_padding() and no_truncation(), and save(). A
  checkpoint with only vocab.txt gets the fast tokenizer transformers builds
  from it, saved the same way. The result is deterministic: all-MiniLM-L6's
  cleaned file is byte-identical to e5-small-v2's and bge-small's upstream
  ones (tokenizers 0.22).
"""

import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(src, dest, mode="clean", expected_sha256=None):
    """Write the tokenizer.json sidekick will load to `dest`; returns dest."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    original = src / "tokenizer.json"
    if mode == "verbatim":
        shutil.copy(original, dest)
    elif mode == "clean":
        if original.exists():
            raw = json.loads(original.read_text())
            if raw.get("padding") is None and raw.get("truncation") is None:
                shutil.copy(original, dest)
            else:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(original))
                tok.no_padding()
                tok.no_truncation()
                tok.save(str(dest))
        else:
            from transformers import AutoTokenizer
            fast = AutoTokenizer.from_pretrained(src)
            if not fast.is_fast:
                raise SystemExit(f"{src}: no tokenizer.json, and no fast tokenizer can be built")
            tok = fast.backend_tokenizer
            tok.no_padding()
            tok.no_truncation()
            tok.save(str(dest))
            print(f"generated tokenizer.json from {', '.join(p.name for p in src.glob('vocab*'))}")
    else:
        raise ValueError(f"unknown tokenizer mode {mode!r}")
    if expected_sha256 is not None and sha256(dest) != expected_sha256:
        raise SystemExit(f"{dest}: sha256 {sha256(dest)} != expected {expected_sha256}; "
                         "references pinned to the expected file would go stale")
    return dest


def load(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def encode(tok, text):
    """Token ids with special tokens, exactly as sidekick encodes one input."""
    return tok.encode(text, add_special_tokens=True).ids


def encode_pair(tok, a, b):
    """(ids, token_type_ids) of a text pair, as sidekick encodes a rerank pair."""
    enc = tok.encode(a, b, add_special_tokens=True)
    return enc.ids, enc.type_ids


def special_ids(tok):
    """Ids the tokenizer adds around an empty input, e.g. [CLS] [SEP]."""
    return tok.encode("", add_special_tokens=True).ids


def st_config(src):
    """sentence-transformers' sentence_bert_config.json, or {}."""
    p = Path(src) / "sentence_bert_config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def st_pooling(src):
    """sentence-transformers' pooling mode ("cls", "mean", "lasttoken", ...),
    or None without a Pooling module."""
    for p in sorted(Path(src).glob("*Pooling*/config.json")):
        cfg = json.loads(p.read_text())
        for key, mode in (("pooling_mode_cls_token", "cls"), ("pooling_mode_mean_tokens", "mean"),
                          ("pooling_mode_lasttoken", "lasttoken"), ("pooling_mode_max_tokens", "max")):
            if cfg.get(key):
                return mode
    return None


def snapshot_revision(src):
    """The revision of a Hugging Face cache snapshot (…/snapshots/<rev>), or None."""
    parts = Path(src).resolve().parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
