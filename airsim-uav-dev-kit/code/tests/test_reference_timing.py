import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flight.reference_trajectory import Reference


class ReferenceTimingTests(unittest.TestCase):
    def test_original_duration_and_linear_motion(self):
        controls = np.column_stack((np.arange(8), np.zeros(8), np.repeat(-5, 8)))
        reference = Reference({"controls": [controls.tolist()], "durations": [7.0], "holds": {}})
        self.assertEqual(reference.duration, 7.0)
        np.testing.assert_allclose(reference.evaluate(3.0), [3, 0, -5, 1, 0, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(reference.evaluate(7.0), [7, 0, -5, 0, 0, 0, 0, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
