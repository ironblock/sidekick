"""Tiny ONNX test models for the onnx embedding backend (D40).

Writes crates/sidekick-embed/tests/fixtures/tiny-onnx/:
- `hidden.onnx`: a published export's shape. It takes `input_ids`,
  `attention_mask` and `token_type_ids` (int64) and `position_ids` (int32,
  so the backend's input types must follow the graph), and returns
  token-level `last_hidden_state` [batch, seq, 8]. It is one masked
  attention layer over word, type and position embeddings: a padded
  position that leaked into a real token's attention would change it.
- `pooled.onnx`: sidekick's own export's shape. It takes `input_ids` and
  `attention_mask` (int32), and returns the masked mean `embedding`
  [batch, 8], pooled in the graph.
- `tokenizer.json`: a word-level tokenizer over the 12 ids, splitting on
  spaces only, for tools/smoke_launch.sh's test model;
- `parity-model/`: `hidden.onnx` as an onnx model the parity suite can
  grade, with a tokenizer that wraps each input in [CLS] ... [SEP], and
  its reference over the parity corpus in `parity/` (the format
  tools/parity_reference.py writes, numpy as the oracle). CI runs the
  suite on it on Linux, where no Core ML model exists;
- `expected.json`: for fixed id sequences, numpy's unit-normalized
  vectors for each pooling the Rust side applies (mean, cls, last_token)
  to `hidden`, and for `pooled`. numpy computes the same function
  independently of ONNX Runtime.

Built with the onnx package alone (onnx 1.18), from seeded weights, so the
bytes are reproducible: `python tools/make_onnx_test_models.py [out_dir]`.
"""

import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB, D, POSITIONS = 12, 8, 64
OPSET = 17


def weights():
    rng = np.random.default_rng(40)
    f = lambda *shape: rng.normal(0.0, 0.5, size=shape).astype(np.float32)  # noqa: E731
    return {"word": f(VOCAB, D), "type": f(2, D), "pos": f(POSITIONS, D), "w1": f(D, D), "b1": f(D)}


def encoder_nodes(with_types):
    """x = word + [type + pos]; h = tanh(x W1 + b1); h + softmax(h hᵀ/√D + mask) h."""
    nodes = [helper.make_node("Gather", ["word", "input_ids"], ["x_word"])]
    if with_types:
        nodes += [
            helper.make_node("Gather", ["type", "token_type_ids"], ["x_type"]),
            helper.make_node("Gather", ["pos", "position_ids"], ["x_pos"]),
            helper.make_node("Add", ["x_word", "x_type"], ["x_wt"]),
            helper.make_node("Add", ["x_wt", "x_pos"], ["x"]),
        ]
    else:
        nodes += [helper.make_node("Identity", ["x_word"], ["x"])]
    nodes += [
        helper.make_node("MatMul", ["x", "w1"], ["xw"]),
        helper.make_node("Add", ["xw", "b1"], ["xwb"]),
        helper.make_node("Tanh", ["xwb"], ["h"]),
        helper.make_node("Transpose", ["h"], ["ht"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["h", "ht"], ["scores_raw"]),
        helper.make_node("Mul", ["scores_raw", "inv_sqrt_d"], ["scores"]),
        # Keys a mask of 0 hides: (1 - mask) * -1e4, broadcast over queries.
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "mask_f"], ["unmask"]),
        helper.make_node("Mul", ["unmask", "neg_big"], ["bias_flat"]),
        helper.make_node("Unsqueeze", ["bias_flat", "axis1"], ["bias"]),
        helper.make_node("Add", ["scores", "bias"], ["scores_masked"]),
        helper.make_node("Softmax", ["scores_masked"], ["att"], axis=-1),
        helper.make_node("MatMul", ["att", "h"], ["ctx"]),
        helper.make_node("Add", ["h", "ctx"], ["hidden"]),
    ]
    return nodes


def constants(w, with_types):
    c = {
        "word": w["word"],
        "w1": w["w1"],
        "b1": w["b1"],
        "inv_sqrt_d": np.array(1.0 / np.sqrt(D), np.float32),
        "one": np.array(1.0, np.float32),
        "neg_big": np.array(-1e4, np.float32),
        "axis1": np.array([1], np.int64),
    }
    if with_types:
        c["type"], c["pos"] = w["type"], w["pos"]
    return [numpy_helper.from_array(v, name=k) for k, v in c.items()]


def int_input(name, elem):
    return helper.make_tensor_value_info(name, elem, ["batch", "sequence"])


def hidden_model(w):
    nodes = encoder_nodes(True) + [helper.make_node("Identity", ["hidden"], ["last_hidden_state"])]
    graph = helper.make_graph(
        nodes,
        "tiny_hidden",
        [
            int_input("input_ids", TensorProto.INT64),
            int_input("attention_mask", TensorProto.INT64),
            int_input("token_type_ids", TensorProto.INT64),
            int_input("position_ids", TensorProto.INT32),
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", D])],
        constants(w, True),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="sidekick")


def pooled_model(w):
    nodes = encoder_nodes(False) + [
        # Masked mean over the sequence: sum(hidden * mask) / sum(mask).
        helper.make_node("Unsqueeze", ["mask_f", "axis2"], ["mask3"]),
        helper.make_node("Mul", ["hidden", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["mask3", "axis1"], ["count"], keepdims=0),
        helper.make_node("Div", ["summed", "count"], ["embedding"]),
    ]
    inits = constants(w, False) + [numpy_helper.from_array(np.array([2], np.int64), name="axis2")]
    graph = helper.make_graph(
        nodes,
        "tiny_pooled",
        [int_input("input_ids", TensorProto.INT32), int_input("attention_mask", TensorProto.INT32)],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", D])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="sidekick")


def numpy_hidden(w, ids, with_types):
    """The encoder in numpy, for one unpadded input."""
    ids = np.array(ids)
    x = w["word"][ids]
    if with_types:
        x = x + w["type"][np.zeros_like(ids)] + w["pos"][np.arange(len(ids))]
    h = np.tanh(x @ w["w1"] + w["b1"])
    scores = (h @ h.T) / np.sqrt(D)
    att = np.exp(scores - scores.max(axis=-1, keepdims=True))
    att = att / att.sum(axis=-1, keepdims=True)
    return h + att @ h


def unit(v):
    return [float(x) for x in v / np.linalg.norm(v)]


CASES = [[1, 4, 5, 2], [1, 6, 2], [1, 4, 5, 6, 7, 8, 9, 10, 11, 2], [1, 2]]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "crates", "sidekick-embed", "tests", "fixtures", "tiny-onnx"
    )
    os.makedirs(out, exist_ok=True)
    w = weights()
    for name, model in [("hidden.onnx", hidden_model(w)), ("pooled.onnx", pooled_model(w))]:
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, os.path.join(out, name))
    expected = []
    for ids in CASES:
        h = numpy_hidden(w, ids, True)
        expected.append({
            "ids": ids,
            "mean": unit(h.mean(axis=0)),
            "cls": unit(h[0]),
            "last_token": unit(h[-1]),
            "pooled": unit(numpy_hidden(w, ids, False).mean(axis=0)),
        })
    words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {w: i for i, w in enumerate(words)}, "unk_token": "[UNK]"},
    }
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=1)
        f.write("\n")
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"cases": expected}, f, indent=1)
        f.write("\n")
    parity_model(out, w)
    print(f"wrote {out}: hidden.onnx, pooled.onnx, expected.json ({len(expected)} cases), parity-model/")


PARITY_MAX_SEQ_LEN = 64


def parity_model(out, w):
    """`hidden.onnx` as a gradable onnx model, and its parity reference.

    Each case is materialized as tools/parity_reference.py does, and its ids
    are what the tokenizer below gives, truncated to the model's 64 tokens
    keeping [SEP], as sentence-transformers truncates. The vectors are
    numpy's encoder (CLS pooling, unit-normalized)."""
    import struct
    import tomllib

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity_reference import FORMAT, corpus_hash, materialize, target_length

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_dir = os.path.join(out, "parity-model")
    os.makedirs(os.path.join(model_dir, "parity"), exist_ok=True)
    words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    vocab = {w: i for i, w in enumerate(words)}
    special = lambda t: {"id": t, "ids": [vocab[t]], "tokens": [t]}  # noqa: E731
    template = [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}]
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {"type": "TemplateProcessing", "single": template,
                           "pair": template + [{"Sequence": {"id": "B", "type_id": 1}},
                                               {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
                           "special_tokens": {"[CLS]": special("[CLS]"), "[SEP]": special("[SEP]")}},
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    tokenizer_bytes = (json.dumps(tokenizer, indent=1) + "\n").encode()
    with open(os.path.join(model_dir, "tokenizer.json"), "wb") as f:
        f.write(tokenizer_bytes)
    with open(os.path.join(model_dir, "manifest.toml"), "w") as f:
        f.write(
            "# hidden.onnx (one directory up) as an onnx model for the parity suite: its\n"
            "# reference is in parity/. Built by tools/make_onnx_test_models.py.\n\n"
            'id = "tiny-onnx"\nbackend = "onnx"\nartifact = "../hidden.onnx"\ntokenizer = "tokenizer.json"\n'
            f'dims = {D}\npooling = "cls"\nmax_seq_len = {PARITY_MAX_SEQ_LEN}\n\n[io]\n'
            'input_ids = "input_ids"\nattention_mask = "attention_mask"\ntoken_type_ids = "token_type_ids"\n'
            'position_ids = "position_ids"\noutput = "last_hidden_state"\n'
        )

    def encode(text):
        ids = [vocab["[CLS]"]] + [vocab.get(t, vocab["[UNK]"]) for t in text.split()] + [vocab["[SEP]"]]
        if len(ids) > PARITY_MAX_SEQ_LEN:
            ids = ids[: PARITY_MAX_SEQ_LEN - 1] + [vocab["[SEP]"]]
        return ids

    corpus_bytes = open(os.path.join(root, "fixtures", "parity", "corpus.toml"), "rb").read()
    corpus = tomllib.loads(corpus_bytes.decode())
    filler = corpus["filler"]["text"].split()
    buckets = [16, 32, PARITY_MAX_SEQ_LEN]
    count = lambda text: len(text.split()) + 2  # noqa: E731
    cases, vectors = [], []
    for case in corpus["case"]:
        purpose = case.get("purpose", "document")
        text = materialize(filler, "", target_length(case["length"], buckets), count) if "length" in case else case["text"]
        ids = encode(text)
        h = numpy_hidden(w, ids, True)
        cases.append({"id": case["id"], "purpose": purpose, "tags": case.get("tags", []), "text": text, "ids": ids})
        vectors.append(np.array(unit(h[0]), np.float32))
    meta = {
        "format": FORMAT,
        "corpus_sha256": corpus_hash(corpus_bytes.decode()),
        "model": {"id": "tiny-onnx", "dims": D, "buckets": buckets, "max_seq_len": PARITY_MAX_SEQ_LEN,
                  "prefixes": {"query": "", "document": ""}},
        "source": {"id": "tools/make_onnx_test_models.py", "revision": None},
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "prompt_source": {"query": "manifest", "document": "manifest"},
        "pooling": "cls",
        "oracles": ["torch"],
        "versions": {"numpy": np.__version__, "onnx": onnx.__version__},
        "cases": cases,
    }
    with open(os.path.join(model_dir, "parity", "reference.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    # safetensors, one F32 tensor "torch" [cases, dims]: numpy is the oracle
    # here; the suite names its reference oracle "torch".
    data = np.stack(vectors).astype("<f4").tobytes()
    header = json.dumps({"torch": {"dtype": "F32", "shape": [len(cases), D], "data_offsets": [0, len(data)]}})
    header += " " * (-len(header) % 8)
    with open(os.path.join(model_dir, "parity", "reference.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header.encode() + data)


if __name__ == "__main__":
    main()
