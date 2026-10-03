"""Chunked buckets (chunking.py, D37): the composed chunks are the unchunked
wrapper, cuts are planned under the budget, and the installed manifest and
placement say what was split.

A small random Qwen3 tree model and AgentJev head (test_agentjev's), no
download and no Core ML: the conversion itself is exercised by the tiny
chunked fixture (tools/make_classifier_test_models.py) and the Rust tests.
"""

import tempfile
import tomllib
import types
import unittest

import numpy as np
import torch

from sidekick_convert import chunking, plan
from sidekick_convert.backbones import qwen3
from sidekick_convert.tests.test_agentjev import load, tiny_checkpoint, tree_feed
from sidekick_convert.wrapper import Wrapper


def feed(prefix, suffixes, seq, kmax, pads=None):
    return {k: v.numpy() for k, v in tree_feed(prefix, suffixes, seq, kmax, pads).items()}


class ComposedChunks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tiny_checkpoint(cls.dir.name, seed=2, layers=4)
        cls.backbone, cls.head = load(cls.dir.name)
        qwen3.matmul_softmax(cls.backbone)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_composed_chunks_are_the_unchunked_wrapper(self):
        ports = self.head.ports()
        rng = np.random.default_rng(0)
        for seq, cuts in ((16, [2]), (32, [1, 3]), (32, [1, 2, 3])):
            f = feed([5, 9, 13, 2, 7], [[40, 41], [50], [60, 61, 62]], seq, self.head.kmax,
                     rng.integers(1, 120, seq).tolist())
            whole = Wrapper(self.backbone, self.head, ports, seq).eval()
            with torch.no_grad():
                want = whole(*(torch.from_numpy(f[p.name]) for p in ports))
            parts = chunking.chunks(self.backbone, self.head, ports, seq, cuts, self.head.output)
            got = chunking.run_torch(parts, f)
            self.assertLessEqual(float((got - want).abs().max()), 1e-6, (seq, cuts))

    def test_each_chunk_takes_only_what_it_reads(self):
        parts = chunking.chunks(self.backbone, self.head, self.head.ports(), 16, [1, 3], self.head.output)
        self.assertEqual([c.inputs for c in parts], [
            ("input_ids", "attention_mask", "seg", "position_ids"),
            ("hidden_in", "attention_mask", "seg", "position_ids"),
            ("hidden_in", "attention_mask", "seg", "position_ids", "cand_end")])
        self.assertEqual([c.output for c in parts], ["hidden_out", "hidden_out", self.head.output])
        self.assertEqual([(c.lo, c.hi) for c in parts], [(0, 1), (1, 3), (3, 4)])
        # weights go only where they're used
        self.assertTrue(hasattr(parts[0].module, "embed_tokens"))
        self.assertFalse(hasattr(parts[1].module, "embed_tokens") or hasattr(parts[1].module, "norm"))
        self.assertTrue(hasattr(parts[2].module, "norm") and hasattr(parts[2].module, "scorer"))

    def test_auto_finds_the_fewest_balanced_chunks_under_the_budget(self):
        first, per_layer, last, every = chunking.weight_plan(self.backbone, self.head, 32)
        whole = first + sum(per_layer) + last + every
        self.assertEqual(chunking.plan_cuts(self.backbone, self.head, 32, "auto", budget=whole), [])
        cuts = chunking.plan_cuts(self.backbone, self.head, 32, "auto", budget=whole // 2 + every)
        sizes = chunking.chunk_sizes(self.backbone, self.head, 32, cuts)
        self.assertTrue(all(s <= whole // 2 + every for s in sizes), sizes)
        # two chunks, balanced: no other cut has a smaller largest chunk
        best = min(max(chunking.chunk_sizes(self.backbone, self.head, 32, [c])) for c in (1, 2, 3))
        self.assertEqual(max(chunking.chunk_sizes(self.backbone, self.head, 32,
                                                  chunking.plan_cuts(self.backbone, self.head, 32, 2))), best)
        with self.assertRaises(ValueError):
            chunking.plan_cuts(self.backbone, self.head, 32, "auto", budget=every)

    def test_explicit_cuts_and_counts_are_checked(self):
        self.assertEqual(chunking.plan_cuts(self.backbone, self.head, 16, [3, 1]), [1, 3])
        for bad in ([0], [4], [2, 2], []):
            with self.assertRaises(ValueError, msg=bad):
                chunking.plan_cuts(self.backbone, self.head, 16, bad)
        with self.assertRaises(ValueError):
            chunking.plan_cuts(self.backbone, self.head, 16, 5)
        self.assertEqual([chunking.parse_spec(s) for s in (None, "1", "auto", "3", "10,20")],
                         [chunking.DEFAULT, None, "auto", 3, [10, 20]])


class Planning(unittest.TestCase):
    """core._plan_chunks: the default, explicit specs, and the template check."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        tiny_checkpoint(cls.dir.name, seed=4, layers=6)
        cls.backbone, cls.head = load(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def job(self, chunks, manifest='artifact = "model_{seq}.mlmodelc"\n', backbone=None):
        from pathlib import Path
        from sidekick_convert import core
        path = Path(self.dir.name) / "m.toml"
        path.write_text(manifest)
        return core.Job(name="t", buckets=[16, 32], ports=[], output="logits", make_wrapper=None, example=None,
                        evaluation=None, gates=None, install_files=[(path, "classifier.toml")], chunks=chunks,
                        backbone=backbone or self.backbone, head=self.head)

    def test_the_default_chunks_only_what_needs_it(self):
        from sidekick_convert import core
        # under the real budget: one program, silently
        self.assertIsNone(core._plan_chunks(self.job(chunking.DEFAULT)))
        self.assertIsNone(core._plan_chunks(self.job(None)))
        # a backbone that can't be chunked converts as before by default, and refuses an explicit request
        plain = types.SimpleNamespace(chunk_ports=None, family="bert")
        self.assertIsNone(core._plan_chunks(self.job(chunking.DEFAULT, backbone=plain)))
        with self.assertRaises(ValueError):
            core._plan_chunks(self.job("auto", backbone=plain))
        self.assertEqual(core._plan_chunks(self.job(3)), chunking.plan_cuts(self.backbone, self.head, 32, 3))
        self.assertEqual(core._plan_chunks(self.job([2, 4])), [2, 4])

    def test_a_manifest_that_cant_name_its_chunks_fails_before_converting(self):
        from sidekick_convert import core
        with self.assertRaises(ValueError):
            core._plan_chunks(self.job(2, manifest='artifact = "custom_{seq}.mlmodelc"\n'))

    def test_balancing_finds_the_best_split(self):
        import itertools
        first, per_layer, last, every = chunking.weight_plan(self.backbone, self.head, 32)
        for n in (2, 3, 4):
            best = min(max(chunking._chunk_bytes(list(c), first, per_layer, last, every))
                       for c in itertools.combinations(range(1, len(per_layer)), n - 1))
            got = chunking._balanced(n, first, per_layer, last, every)
            self.assertEqual(len(got), n - 1)
            self.assertEqual(max(chunking._chunk_bytes(got, first, per_layer, last, every)), best, n)


class Layouts(unittest.TestCase):
    def test_installing_one_layout_removes_the_other(self):
        from pathlib import Path
        from sidekick_convert import core
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("model_16.mlmodelc", "model_16.0.mlmodelc", "model_16.1.mlmodelc", "model_16.2.mlmodelc",
                         "model_32.mlmodelc"):
                (d / name).mkdir()
            core._remove_other_layout(d, 16, chunked=True)
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["model_32.mlmodelc"])
            for name in ("model_32.0.mlmodelc", "model_32.1.mlmodelc"):
                (d / name).mkdir()
            core._remove_other_layout(d, 32, chunked=False)
            self.assertEqual(sorted(p.name for p in d.iterdir()), ["model_32.mlmodelc"])
        self.assertEqual(chunking.artifact_name(512, 1), "model_512.1.mlmodelc")


class PrivateCoreMLCache(unittest.TestCase):
    def test_a_converter_runs_with_a_temporary_home_and_keeps_its_exit_code(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        if sys.platform != "darwin":
            self.skipTest("macOS only")
        tools = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "probe.py"
            probe.write_text("import os, sys\nsys.path.insert(0, %r)\nfrom sidekick_convert import cli\n"
                             "cli.parse('probe')\nhome = os.environ['CFFIXED_USER_HOME']\n"
                             "open(sys.argv[2] + '/home', 'w').write(home)\nsys.exit(3)\n" % str(tools))
            env = {k: v for k, v in os.environ.items() if k != "CFFIXED_USER_HOME"}
            r = subprocess.run([sys.executable, "-X", "utf8", str(probe), "src", d], env=env)
            self.assertEqual(r.returncode, 3)
            home = (Path(d) / "home").read_text()
            self.assertIn("sidekick-convert-cache-", home)
            self.assertFalse(Path(home).exists(), "the temporary home is removed")


class InstalledRecords(unittest.TestCase):
    def test_the_manifest_names_its_chunks(self):
        text = 'id = "x"\nartifact = "model_{seq}.mlmodelc"\ntokenizer = "t.json"\n\n[classify]\nformat = "agentjev"\n'
        out = tomllib.loads(chunking.manifest_text(text, 2, chunking.CHUNK_WEIGHT_BUDGET_BYTES))
        self.assertEqual(out["artifact"], "model_{seq}.{chunk}.mlmodelc")
        self.assertEqual(out["chunking"], {"chunks": 2, "weight_budget_bytes": int(0.9 * 2**30)})
        self.assertEqual(out["classify"], {"format": "agentjev"})
        self.assertNotIn("weight_budget_bytes", tomllib.loads(chunking.manifest_text(text, 3, None))["chunking"])
        with self.assertRaises(ValueError):
            chunking.manifest_text(text.replace("model_{seq}", "m"), 2, None)

    def test_placement_sums_the_chunks_and_lists_each(self):
        def chunk(ane, cpu, off):
            return {"ane": ane, "gpu": 0, "cpu": cpu, "unassigned": 5, "total": ane + cpu + 5,
                    "assigned": ane + cpu, "off": off, "heavy_off": [], "unassigned_heavy": [],
                    "masked_fused_attention": []}
        s = plan.combine([chunk(90, 10, {"gather": 1, "cast": 9}), chunk(200, 2, {"cast": 2})])
        s["units"] = "CPU_AND_NE"
        self.assertEqual((s["ane"], s["cpu"], s["total"], s["off"]), (290, 12, 312, {"gather": 1, "cast": 11}))
        doc = tomllib.loads(plan.placement_toml({512: s}, "Apple M1 Max", "26A1", "2026-10-03"))
        b = doc["placement"]["buckets"]["512"]
        self.assertEqual((b["ane"], b["total"], b["off_ane_ops"]), (290, 312, {"cast": 11, "gather": 1}))
        self.assertEqual([c["ane"] for c in b["chunks"]], [90, 200])
        self.assertEqual(b["chunks"][1]["off_ane_ops"], {"cast": 2})
        # The plan for the other compute units, beside it
        gpu = dict(s, units="CPU_AND_GPU")
        doc = tomllib.loads(plan.placement_toml({512: s}, "Apple M1 Max", "26A1", "2026-10-03",
                                                alternatives={"CPU_AND_GPU": {512: gpu}}))
        alt = doc["placement"]["alternatives"]["cpu_and_gpu"]["buckets"]["512"]
        self.assertEqual((alt["ane"], [c["ane"] for c in alt["chunks"]]), (290, [90, 200]))
        self.assertEqual(doc["placement"]["compute_units"], "cpu_and_ne")


if __name__ == "__main__":
    unittest.main()
