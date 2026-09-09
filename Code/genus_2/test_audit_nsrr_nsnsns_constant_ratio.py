#!/usr/bin/env python3

import math
import unittest

from audit_nsrr_nsnsns_constant_ratio import constant_fit


class ConstantRatioFitTests(unittest.TestCase):
    def test_exact_constant(self):
        result = constant_fit([
            {"coordinate": value, "ratio": 4.0} for value in (-1.0, 0.0, 2.0)
        ])
        self.assertAlmostEqual(result["normalization_geometric_mean"], 4.0)
        self.assertAlmostEqual(result["maximum_absolute_fractional_residual"], 0.0)
        self.assertAlmostEqual(result["linear_log_slope_per_coordinate"], 0.0)

    def test_log_linear_trend_is_detected(self):
        points = [
            {"coordinate": value, "ratio": 3.0 * math.exp(-0.2 * value)}
            for value in (-1.0, 0.0, 1.0)
        ]
        result = constant_fit(points)
        self.assertAlmostEqual(result["normalization_geometric_mean"], 3.0)
        self.assertAlmostEqual(result["linear_log_slope_per_coordinate"], -0.2)
        self.assertLess(
            result["linear_detrended_maximum_absolute_fractional_residual"], 1.0e-14
        )


if __name__ == "__main__":
    unittest.main()
