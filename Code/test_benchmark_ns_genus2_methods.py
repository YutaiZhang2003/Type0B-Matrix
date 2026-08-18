"""Unit tests for the NS genus-two convergence benchmark."""

from __future__ import annotations

import unittest

from benchmark_ns_genus2_methods import first_stable_cutoff, normalized_error


class NSGenus2EfficiencyBenchmarkTests(unittest.TestCase):
    def test_first_stable_cutoff_requires_stable_tail(self) -> None:
        errors = (1.0e-2, 5.0e-7, 2.0e-6, 1.0e-8, 0.0)
        self.assertEqual(first_stable_cutoff(errors, tolerance=1.0e-6), 3)

    def test_normalized_error_is_absolute_below_unit_scale(self) -> None:
        self.assertAlmostEqual(normalized_error(0.1, 0.100002), 2.0e-6)


if __name__ == "__main__":
    unittest.main()
