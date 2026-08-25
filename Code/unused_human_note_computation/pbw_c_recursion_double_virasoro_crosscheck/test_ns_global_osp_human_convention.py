"""Regression tests for the direct Human Note global osp(1|2) checker."""

from __future__ import annotations

import unittest

from check_ns_global_osp_human_convention import (
    extracted_primary_lift_factor,
    relative_global_label,
    required_vacuum_lift_rephasing,
    run_checks,
    theta_cross_exponent,
    theta_orientation_exponent,
)


class NSGlobalOSPHumanConventionTests(unittest.TestCase):
    def test_arbitrary_primary_orientation(self) -> None:
        zero = (0, 0, 0)
        self.assertEqual(theta_orientation_exponent(zero, (0, 0, 0)), 0)
        self.assertEqual(theta_orientation_exponent(zero, (1, 0, 0)), 0)
        self.assertEqual(theta_orientation_exponent(zero, (1, 1, 0)), 1)
        self.assertEqual(theta_orientation_exponent(zero, (1, 1, 1)), 1)
        self.assertEqual(theta_orientation_exponent((1, 0, 0), (0, 1, 0)), 1)
        self.assertEqual(theta_cross_exponent((1, 0, 0), (0, 1, 0)), 1)
        self.assertEqual(
            required_vacuum_lift_rephasing((1, 0, 0), (0, 1, 0)),
            (-1, -1, 1),
        )

    def test_primary_lift_is_extracted(self) -> None:
        self.assertEqual(extracted_primary_lift_factor((1, 0, 1), (-1, 1, -1)), 1)
        self.assertEqual(extracted_primary_lift_factor((1, 1, 0), (-1, 1, -1)), -1)
        self.assertEqual(relative_global_label(1, (1, 0, 0)), 0)
        self.assertEqual(relative_global_label(1, (1, 1, 0)), 1)

    def test_direct_global_osp_block(self) -> None:
        summary = run_checks(maximum_total_occupation=1)
        self.assertEqual(summary.ground_table_identities, 8)
        self.assertEqual(summary.pbw_vertex_identities, 32)
        self.assertEqual(summary.superspace_vertex_identities, 32)
        self.assertEqual(summary.global_norm_identities, 12)
        self.assertEqual(summary.sewn_coefficient_identities, 32)
        self.assertEqual(summary.exact_production_seed_identities, 32)
        self.assertEqual(summary.numerical_production_term_identities, 32)
        self.assertEqual(summary.truncated_block_identities, 64)
        self.assertEqual(summary.arbitrary_primary_coefficient_identities, 256)
        self.assertEqual(summary.arbitrary_primary_block_identities, 512)
        self.assertEqual(summary.orientation_polarization_identities, 256)
        self.assertEqual(summary.large_c_vacuum_rephasing_identities, 256)
        self.assertEqual(summary.adapted_production_term_identities, 256)
        self.assertEqual(summary.extracted_primary_lift_identities, 512)
        self.assertLess(summary.maximum_numerical_production_error, 1.0e-13)


if __name__ == "__main__":
    unittest.main()
