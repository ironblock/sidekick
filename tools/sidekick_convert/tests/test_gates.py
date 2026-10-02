"""The gates' Core ML models keep every prediction input alive, and a
classifier is gated on the path its manifest serves (no Core ML needed:
stand-ins replace the model and the compute plan)."""

import unittest
import weakref

import numpy as np

from sidekick_convert import gates, manifest
from sidekick_convert.core import GateFailure


class PredictionInputs(unittest.TestCase):
    def test_inputs_outlive_the_call(self):
        class StandIn:
            def predict(self, feed):
                return {"y": feed["x"] * 2}

        m = gates._Model(StandIn())
        x = np.ones((1, 8), dtype=np.float32)
        alive = weakref.ref(x)
        out = m.predict({"x": x})["y"]
        del x
        self.assertIsNotNone(alive(), "the input array was freed after predict() returned")
        np.testing.assert_array_equal(out, np.full((1, 8), 2.0, dtype=np.float32))



class ServedPath(unittest.TestCase):
    def test_manifest_compute_units(self):
        self.assertEqual(manifest.served_path({}), "CPU_AND_NE")
        for value, units in [("cpu_and_ne", "CPU_AND_NE"), ("cpu_and_gpu", "CPU_AND_GPU"),
                             ("cpu_only", "CPU_ONLY"), ("all", "ALL")]:
            self.assertEqual(manifest.served_path({"compute_units": value}), units)
        with self.assertRaises(SystemExit):
            manifest.served_path({"compute_units": "gpu"})

    def test_the_served_path_is_gated(self):
        gpu = gates.ClassifierGates.paths("CPU_AND_GPU")
        self.assertEqual(gpu, {"gated_paths": ("CPU_AND_GPU",), "report_paths": ("CPU_AND_NE", "CPU_ONLY"),
                               "plan_required": False, "served": "CPU_AND_GPU"})
        ane = gates.ClassifierGates.paths("CPU_AND_NE")
        self.assertEqual(ane, {"gated_paths": ("CPU_AND_NE",), "report_paths": ("CPU_ONLY",),
                               "plan_required": True, "served": "CPU_AND_NE"})

    def test_the_plan_gates_only_an_ane_served_model(self):
        summary = {"ane": 1, "assigned": 10, "share": 0.1, "off": {"linear": 9}}

        def failing_gate(compiled, min_ane):
            raise GateFailure("compute plan: 1/10 ops on the ANE")

        read_for = []

        def report(compiled, units="CPU_AND_NE"):
            read_for.append(units)
            return summary

        saved = gates._plan.gate, gates._plan.report
        gates._plan.gate, gates._plan.report = failing_gate, report
        try:
            off_ane = gates.ClassifierGates(**{**gates.ClassifierGates.paths("CPU_AND_GPU"),
                                               "gated_paths": (), "report_paths": ()})
            self.assertIs(off_ane.coreml("model.mlmodelc", 128, [], None, False)["plan"], summary)
            self.assertEqual(read_for, ["CPU_AND_GPU"])  # the plan of the units it is served with
            on_ane = gates.ClassifierGates(**{**gates.ClassifierGates.paths("CPU_AND_NE"),
                                              "gated_paths": (), "report_paths": ()})
            with self.assertRaises(GateFailure):
                on_ane.coreml("model.mlmodelc", 128, [], None, False)
        finally:
            gates._plan.gate, gates._plan.report = saved


class AneWeightCap(unittest.TestCase):
    """A model served on the ANE whose weights pass the Neural Engine's
    per-program limit is refused, unless the job ignores the limit; then the
    bypass is recorded (a sparse file stands in for the weights)."""

    def compiled(self, size):
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp()) / "model.mlmodelc"
        (d / "weights").mkdir(parents=True)
        with open(d / "weights" / "weight.bin", "wb") as f:
            f.truncate(size)
        return d

    def job(self, ignore=False, ane=True):
        from types import SimpleNamespace
        return SimpleNamespace(name="big-model", gates=SimpleNamespace(plan_required=ane),
                               ignore_ane_weight_cap=ignore)

    def test_over_the_limit_fails_and_names_the_options(self):
        from sidekick_convert import core, plan
        over = self.compiled(plan.MAX_ANE_PROGRAM_WEIGHT_BYTES + 2**20)
        with self.assertRaises(GateFailure) as caught:
            core.check_ane_weights(self.job(), 512, over)
        message = str(caught.exception)
        self.assertTrue(message.startswith("big-model's 512 bucket has 1.001 GiB of weights, over the Neural "
                                           "Engine's 1 GiB per-program limit (MAX_ANE_PROGRAM_WEIGHT_BYTES)"))
        for option in ("cpu_and_gpu", "--int8-embedding", "--ignore-ane-weight-cap"):
            self.assertIn(option, message)

    def test_the_bypass_warns_and_is_recorded(self):
        import tempfile
        import tomllib
        from pathlib import Path
        from sidekick_convert import core, plan
        over = self.compiled(plan.MAX_ANE_PROGRAM_WEIGHT_BYTES + 2**20)
        note = core.check_ane_weights(self.job(ignore=True), 512, over)
        self.assertIn("bucket 512 has 1.001 GiB", note)
        manifest = Path(tempfile.mkdtemp()) / "classifier.toml"
        manifest.write_text('id = "big-model"\n')
        core.note_bypass(manifest, [note])
        text = manifest.read_text()
        self.assertIn("# Converted with --ignore-ane-weight-cap:", text)
        self.assertEqual(tomllib.loads(text), {"id": "big-model"})

    def test_within_the_limit_or_off_the_ane_passes(self):
        from sidekick_convert import core, plan
        self.assertIsNone(core.check_ane_weights(self.job(), 128, self.compiled(int(0.964 * 2**30))))
        over = self.compiled(plan.MAX_ANE_PROGRAM_WEIGHT_BYTES + 1)
        self.assertIsNone(core.check_ane_weights(self.job(ane=False), 128, over))


class PlacementRecord(unittest.TestCase):
    """The [placement] table an installed manifest gets: sidekick_coreml's
    PlanSummary counts per bucket, for the units the model is served with."""

    def summary(self, units, ane, gpu, cpu, unassigned, off):
        return {"units": units, "ane": ane, "gpu": gpu, "cpu": cpu, "unassigned": unassigned,
                "total": ane + gpu + cpu + unassigned, "off": off}

    def test_counts_match_the_runtime_summary(self):
        from sidekick_convert import plan
        ops = [plan.Op("const", "none", ["w"]), plan.Op("linear", "NeuralEngine", ["a"]),
               plan.Op("gather", "CPU", ["b"]), plan.Op("matmul", "GPU", ["c"]), plan.Op("const", "none", ["d"])]
        s = plan.summarize(ops)
        self.assertEqual((s["ane"], s["gpu"], s["cpu"], s["unassigned"], s["total"]), (1, 1, 1, 2, 5))
        self.assertEqual(s["off"], {"gather": 1, "matmul": 1})

    def test_the_table_parses_as_the_runtime_reads_it(self):
        import tomllib
        from sidekick_convert import plan
        plans = {512: self.summary("CPU_AND_GPU", 0, 300, 14, 400, {"matmul": 290, "gather": 1}),
                 128: self.summary("CPU_AND_GPU", 0, 290, 12, 380, {"matmul": 280})}
        text = 'id = "m"\n' + plan.placement_toml(plans, "Apple M1 Max", "25A354", "2026-10-01")
        p = tomllib.loads(text)["placement"]
        self.assertEqual({k: p[k] for k in ("compute_units", "chip", "macos_build", "date")},
                         {"compute_units": "cpu_and_gpu", "chip": "Apple M1 Max", "macos_build": "25A354",
                          "date": "2026-10-01"})
        self.assertEqual(list(p["buckets"]), ["128", "512"])
        self.assertEqual(p["buckets"]["512"], {"ane": 0, "gpu": 300, "cpu": 14, "unassigned": 400, "total": 714,
                                               "off_ane_ops": {"gather": 1, "matmul": 290}})

    def test_one_set_of_units_per_table(self):
        from sidekick_convert import plan
        plans = {128: self.summary("CPU_AND_NE", 1, 0, 0, 0, {}), 256: self.summary("CPU_AND_GPU", 0, 1, 0, 0, {})}
        with self.assertRaises(ValueError):
            plan.placement_toml(plans, "chip", "build", "2026-10-01")


if __name__ == "__main__":
    unittest.main()
