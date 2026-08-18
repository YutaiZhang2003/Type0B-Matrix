"""Checks for the pointwise human-frame NS c-recursion."""

from __future__ import annotations

import unittest

from compare_ns_genus2_double_virasoro import (
    DEFAULT_SAMPLES,
    evaluated_sector,
    ns_c_recursion_series,
)
from ns_genus2_pointwise_c_recursion import PointwiseHumanThetaCRecursion


class PointwiseHumanThetaCRecursionTests(unittest.TestCase):
    def test_matches_coefficient_recursion_at_small_q(self) -> None:
        sample = DEFAULT_SAMPLES[0]
        background = sample.b + 1.0 / sample.b
        central_charge = 1.5 + 3.0 * background * background
        weights = tuple(
            background * background / 8.0 - momentum * momentum / 2.0
            for momentum in sample.momenta
        )
        q_values = (0.00013, 0.00017, 0.00011)
        coefficient_series = ns_c_recursion_series(
            c=central_charge,
            weights=weights,
            cutoff=8,
        )
        pointwise = PointwiseHumanThetaCRecursion(q_values=q_values)
        for sector in (0, 1):
            expected = evaluated_sector(
                coefficient_series,
                q_values=q_values,
                lifts=(1, 1, 1),
                sector=sector,
            )
            observed = pointwise.block(
                central_charge=central_charge,
                weights=weights,
                sector=sector,
                recursion_order=8,
                lifts=(1, 1, 1),
            )
            self.assertLess(abs(observed - expected), 2.0e-12)


if __name__ == "__main__":
    unittest.main()
