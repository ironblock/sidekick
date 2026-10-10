"""ONNX export: the manifest it installs, the server's pooling, and a
dynamic-shape export through ONNX Runtime."""

import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import torch

from sidekick_convert import onnx_export

COREML_MANIFEST = """\
# A Core ML artifact's description, which the ONNX manifest drops.
# Second line of it.

id = "tiny"
backend = "coreml"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 4
pooling = "none"          # pooled in the graph
buckets = [8, 16]
max_seq_len = 16

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "embedding"

[prefixes]
query = "q: "
document = ""
"""


class TinyEncoder(torch.nn.Module):
    """A position-dependent encoder whose pads must not leak into real rows."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(50, 4)
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        context = (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)
        return torch.tanh(self.lin(h + context))


class ManifestTest(unittest.TestCase):
    def test_onnx_manifest_from_the_coreml_one(self):
        text = onnx_export.manifest_text(COREML_MANIFEST, "mean", onnx_export.TOKEN_OUTPUT)
        m = tomllib.loads(text)
        self.assertEqual(m["backend"], "onnx")
        self.assertEqual(m["artifact"], "model.onnx")
        self.assertEqual(m["pooling"], "mean")
        self.assertNotIn("buckets", m)
        self.assertEqual(m["max_seq_len"], 16)
        self.assertEqual(m["io"]["output"], "last_hidden_state")
        self.assertEqual(m["prefixes"]["query"], "q: ")
        self.assertNotIn("Core ML artifact's description", text)

    def test_pooled_in_graph(self):
        m = tomllib.loads(onnx_export.manifest_text(COREML_MANIFEST, None, onnx_export.POOLED_OUTPUT))
        self.assertEqual(m["pooling"], "none")
        self.assertEqual(m["io"]["output"], "embedding")


class PoolTest(unittest.TestCase):
    def test_modes_read_only_real_positions(self):
        h = np.arange(12, dtype=np.float64).reshape(4, 3)
        mask = np.array([1, 1, 1, 0])
        np.testing.assert_array_equal(onnx_export.pool(h, mask, "cls"), h[0])
        np.testing.assert_array_equal(onnx_export.pool(h, mask, "mean"), h[:3].mean(axis=0))
        np.testing.assert_array_equal(onnx_export.pool(h, mask, "last_token"), h[2])


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "needs onnx and onnxruntime")
class ExportTest(unittest.TestCase):
    def test_dynamic_batch_and_sequence(self):
        model = TinyEncoder().eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.onnx"
            onnx_export.export(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                               onnx_export.TOKEN_OUTPUT, path)
            runner = onnx_export.Runner(path, "mean")
            rows = [[1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]
            batched = runner.run(rows)
            padded = runner.run(rows, pad_ids=np.full(7, 33))
            for row, b, p in zip(rows, batched, padded):
                alone = runner.run([row])[0]
                with torch.no_grad():
                    t = torch.tensor([row])
                    want = model(t, torch.ones_like(t))[0].double().numpy().mean(axis=0)
                np.testing.assert_allclose(alone, want, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(b, alone, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(p, b, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
