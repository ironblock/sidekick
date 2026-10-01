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
                               "plan_required": False})
        ane = gates.ClassifierGates.paths("CPU_AND_NE")
        self.assertEqual(ane, {"gated_paths": ("CPU_AND_NE",), "report_paths": ("CPU_ONLY",),
                               "plan_required": True})

    def test_the_plan_gates_only_an_ane_served_model(self):
        summary = {"ane": 1, "assigned": 10, "share": 0.1, "off": {"linear": 9}}

        def failing_gate(compiled, min_ane):
            raise GateFailure("compute plan: 1/10 ops on the ANE")

        saved = gates._plan.gate, gates._plan.report
        gates._plan.gate, gates._plan.report = failing_gate, lambda compiled: summary
        try:
            off_ane = gates.ClassifierGates(**{**gates.ClassifierGates.paths("CPU_AND_GPU"),
                                               "gated_paths": (), "report_paths": ()})
            self.assertIs(off_ane.coreml("model.mlmodelc", 128, [], None, False)["plan"], summary)
            on_ane = gates.ClassifierGates(**{**gates.ClassifierGates.paths("CPU_AND_NE"),
                                              "gated_paths": (), "report_paths": ()})
            with self.assertRaises(GateFailure):
                on_ane.coreml("model.mlmodelc", 128, [], None, False)
        finally:
            gates._plan.gate, gates._plan.report = saved


if __name__ == "__main__":
    unittest.main()
