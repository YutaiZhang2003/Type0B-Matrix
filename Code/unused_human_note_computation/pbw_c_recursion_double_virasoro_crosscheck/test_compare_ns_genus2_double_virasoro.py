"""Regression tests for the mature all-NS double-Virasoro comparison."""

from __future__ import annotations

import unittest

from compare_ns_genus2_double_virasoro import (
    DEFAULT_SAMPLES,
    run_comparison,
    virasoro_b_square_rs_from_h,
)


class NSGenus2DoubleVirasoroComparisonTests(unittest.TestCase):
    def test_stable_s_one_pole_branch(self) -> None:
        for r in (2, 3, 4):
            for weight in (-1.7, -0.4, 0.8):
                observed = virasoro_b_square_rs_from_h(r, 1, weight)
                expected = 2.0 * (r - 1.0 + 2.0 * weight) / (1.0 - r * r)
                self.assertAlmostEqual(observed, expected)

    def test_total_level_four_all_lifts_and_sectors(self) -> None:
        result = run_comparison(
            maximum_total_physical_level=4,
            samples=DEFAULT_SAMPLES[:1],
        )
        sample = result.sample_results[0]
        self.assertEqual(sample.coefficient_count, 165)
        self.assertEqual(sample.star_quotient_mismatch_count, 0)
        self.assertEqual(sample.ordinary_quotient_mismatch_count, 91)
        self.assertLess(
            sample.maximum_c_recursion_vs_double_virasoro_relative_error,
            2.0e-8,
        )
        self.assertLess(sample.maximum_direct_vs_c_recursion_error, 2.0e-8)
        self.assertLess(
            sample.maximum_evaluated_lift_sector_relative_error,
            1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
