"""Regression tests for arbitrary-primary-parity NS theta c-recursion."""

from __future__ import annotations

from itertools import product
import unittest

from check_ns_genus2_arbitrary_primary_parity import (
    absolute_parity,
    fusion_polynomial_label,
    odd_null_transport_sign,
    relative_block_label,
    run_checks,
)
from ns_genus12_finite_c_check import (
    DirectThetaOracle,
    level_tuples,
    recursion_theta_coefficient,
)


class NSGenus2ArbitraryPrimaryParityTests(unittest.TestCase):
    def test_note_uses_relative_a_and_keeps_absolute_parity_separate(self) -> None:
        levels = (3, 1, 0)
        primaries = (1, 0, 0)
        label = relative_block_label(levels)
        self.assertEqual(label, 0)
        self.assertEqual(fusion_polynomial_label(label), 0)
        self.assertEqual(absolute_parity(label, primaries), 1)

    def test_odd_null_transport_includes_primary_parity(self) -> None:
        self.assertEqual(
            odd_null_transport_sign(
                levels=(3, 0, 0),
                primary_parities=(0, 1, 0),
                edge=0,
            ),
            -1,
        )
        self.assertEqual(
            odd_null_transport_sign(
                levels=(3, 0, 0),
                primary_parities=(0, 1, 1),
                edge=0,
            ),
            1,
        )

    def test_direct_pbw_matches_c_recursion(self) -> None:
        summary = run_checks(maximum_total_twice_level=4)
        self.assertEqual(summary.parity_assignments, 8)
        self.assertEqual(summary.coefficient_count, 280)
        self.assertEqual(summary.direct_recursion_zero_count, 280)
        self.assertEqual(summary.regular_seed_covariance_count, 280)
        self.assertEqual(summary.direct_orientation_covariance_count, 280)
        self.assertEqual(summary.relative_label_projector_count, 280)
        self.assertEqual(summary.absolute_parity_count, 280)
        self.assertEqual(summary.fusion_label_identity_count, 280)
        self.assertGreater(summary.null_orientation_transport_count, 0)
        self.assertEqual(
            summary.null_orientation_transport_count,
            summary.fusion_crossing_square_count,
        )

    def test_numeric_pbw_matches_production_coefficient_recursion_for_all_p(self) -> None:
        c = 17.3
        weights = (0.71, 0.83, 0.94)
        lifts = (1, -1, 1)
        for primaries in product((0, 1), repeat=3):
            oracle = DirectThetaOracle(
                c=c,
                weights=weights,
                primary_parities=primaries,
            )
            for levels in level_tuples(4):
                label = sum(levels) % 2
                sectors = (label, label)
                direct = oracle.coefficient(
                    twice_levels=levels,
                    sectors=sectors,
                    lifts=lifts,
                )
                recursive = recursion_theta_coefficient(
                    c=c,
                    weights=weights,
                    twice_levels=levels,
                    sectors=sectors,
                    lifts=lifts,
                    primary_parities=primaries,
                )
                self.assertLess(
                    abs(direct - recursive),
                    2.0e-12,
                    (primaries, levels, direct, recursive),
                )


if __name__ == "__main__":
    unittest.main()
