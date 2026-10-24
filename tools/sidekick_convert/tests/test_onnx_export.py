"""ONNX export: the manifest it installs, the server's pooling, and a
dynamic-shape export through ONNX Runtime."""

import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import torch

from sidekick_convert import manifest, onnx_export

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

    def test_extra_inputs_are_named(self):
        m = tomllib.loads(onnx_export.manifest_text(COREML_MANIFEST, "cls", onnx_export.TOKEN_OUTPUT,
                                                    ["token_type_ids"]))
        self.assertEqual(m["io"]["token_type_ids"], "token_type_ids")

    def test_onnx_only_settings_are_appended(self):
        extra = "# only for ONNX\n[onnx]\ndefault_intra_op_threads = 16\n"
        m = tomllib.loads(onnx_export.manifest_text(COREML_MANIFEST, "last_token", onnx_export.TOKEN_OUTPUT,
                                                    onnx_extra=extra))
        self.assertEqual(m["onnx"]["default_intra_op_threads"], 16)
        self.assertEqual(m["prefixes"]["query"], "q: ")

    def test_onnx_build_has_its_own_max_seq_len(self):
        extra = ("# Settings for the ONNX build only.\n\n"
                 "# why the ONNX build takes longer inputs\nmax_seq_len = 64\n\n"
                 "[onnx]\ndefault_intra_op_threads = 16\n")
        text = onnx_export.manifest_text(COREML_MANIFEST, "last_token", onnx_export.TOKEN_OUTPUT, onnx_extra=extra)
        m = tomllib.loads(text)
        self.assertEqual(m["max_seq_len"], 64)
        self.assertNotIn("buckets", m)
        self.assertEqual(m["onnx"]["default_intra_op_threads"], 16)
        # the reason travels with the value; the sidecar's description doesn't
        self.assertIn("# why the ONNX build takes longer inputs\nmax_seq_len = 64\n", text)
        self.assertNotIn("Settings for the ONNX build only", text)
        self.assertEqual(text.count("max_seq_len"), 1)

    def test_core_ml_serving_keys_are_left_out(self):
        coreml = COREML_MANIFEST.replace(
            "max_seq_len = 16\n",
            "max_seq_len = 16\n# why the CPU serves less\n# of it\ncpu_max_seq_len = 8\ncompute_units = \"cpu_and_gpu\"\n")
        text = onnx_export.manifest_text(coreml, "mean", onnx_export.TOKEN_OUTPUT)
        m = tomllib.loads(text)
        self.assertNotIn("cpu_max_seq_len", m)
        self.assertNotIn("compute_units", m)
        self.assertNotIn("why the CPU serves less", text)
        self.assertEqual(text, onnx_export.manifest_text(COREML_MANIFEST, "mean", onnx_export.TOKEN_OUTPUT))

    def test_sidecar_sets_only_known_top_level_keys(self):
        with self.assertRaises(SystemExit):
            onnx_export.manifest_text(COREML_MANIFEST, "mean", onnx_export.TOKEN_OUTPUT, onnx_extra="dims = 8\n")

    def test_onnx_view_for_the_checks(self):
        m = tomllib.loads(COREML_MANIFEST)
        view = manifest.onnx_view(m, "# about\n\nmax_seq_len = 64\n[onnx]\ndefault_intra_op_threads = 2\n")
        self.assertEqual(view["max_seq_len"], 64)
        self.assertNotIn("buckets", view)
        self.assertNotIn("onnx", view)
        self.assertEqual(manifest.onnx_view(m, "")["max_seq_len"], 16)

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
