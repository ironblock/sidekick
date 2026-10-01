"""The gates' Core ML models keep every prediction input alive (no Core ML
needed: a stand-in model records what it was given)."""

import unittest
import weakref

import numpy as np

from sidekick_convert import gates


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


if __name__ == "__main__":
    unittest.main()
