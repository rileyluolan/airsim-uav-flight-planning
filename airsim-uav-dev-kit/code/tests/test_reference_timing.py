import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flight.reference_trajectory import Reference


class ReferenceTimingTests(unittest.TestCase):
    def test_scaled_duration_and_cubic_motion(self):
        x = [8 * math.comb(i, 3) / math.comb(7, 3) if i >= 3 else 0 for i in range(8)]
        controls = np.column_stack((x, np.zeros(8), np.repeat(-5, 8)))
        reference = Reference({"controls": [controls.tolist()], "durations": [2.0], "holds": {}}, scale=1.8)
        self.assertAlmostEqual(reference.duration, 3.6)
        state = reference.evaluate(1.8)
        np.testing.assert_allclose(state, [1, 0, -5, 3 / 1.8, 0, 0, 6 / 1.8 ** 2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(reference.polynomials[0].derivative(3)(1.8), [6 / 1.8 ** 3, 0, 0], atol=1e-12)

    def test_original_duration_and_linear_motion(self):
        controls = np.column_stack((np.arange(8), np.zeros(8), np.repeat(-5, 8)))
        reference = Reference({"controls": [controls.tolist()], "durations": [7.0], "holds": {}})
        self.assertEqual(reference.duration, 7.0)
        np.testing.assert_allclose(reference.evaluate(3.0), [3, 0, -5, 1, 0, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(reference.evaluate(7.0), [7, 0, -5, 0, 0, 0, 0, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
