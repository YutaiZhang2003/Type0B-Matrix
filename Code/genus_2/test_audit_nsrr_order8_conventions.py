"""Regression tests for the completed order-eight convention audit."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import audit_nsrr_order8_conventions as convention_audit


SUMMARY = (
    Path(__file__).resolve().parents[2]
    / "Data Set"
    / "nsrr_nsnsns_theta_order8_cannon_resume_summary_20260902.json"
)


class OrderEightConventionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cls.report = convention_audit.audit(cls.summary, free_max_mode=24)

    def test_all_three_quadrature_orders_are_kept_distinct(self):
        rows = self.report["counterfactual_comparisons"]
        self.assertEqual([row["quadrature_order"] for row in rows], [8, 10, 12])
        self.assertNotEqual(
            rows[0]["reported_source_over_target"],
            rows[-1]["reported_source_over_target"],
        )

    def test_legacy_free_conversion_fails_and_fixed_free_converges(self):
        free = self.report["free_theory"]
        self.assertFalse(free["legacy_theta_ratio_test"]["compatible"])
        self.assertGreater(
            free["legacy_theta_ratio_test"]["maximum_relative_incompatibility"],
            0.4,
        )
        fixed = free["fixed_free_for_declared_marked_spins"]
        self.assertAlmostEqual(fixed["source_nsrr"]["z_free"], 0.5754095923872395)
        self.assertAlmostEqual(fixed["target_nsnsns"]["z_free"], 0.5638924570404816)
        self.assertLess(fixed["source_nsrr"]["period_residual"], 1.0e-8)
        self.assertLess(fixed["target_nsnsns"]["period_residual"], 1.0e-8)

    def test_sign_and_coefficient_counterfactuals_are_algebraic(self):
        for row in self.report["counterfactual_comparisons"]:
            ratio = row["reported_source_over_target"]
            self.assertEqual(
                row["source_decomposition_minus_with_two_odd_vertex_i_phases"],
                ratio,
            )
            self.assertEqual(
                row["target_omit_both_odd_i_phase_and_decomposition_minus"],
                ratio,
            )
            self.assertEqual(
                row["source_replace_E_O_by_E_over_2_O_over_2_only"],
                ratio / 4,
            )
            self.assertLess(row["source_flip_only_f1"], 0)
            self.assertLess(abs(row["target_flip_only_odd"] - ratio), 0.005)

    def test_duplicate_channel_row_is_rejected(self):
        broken = deepcopy(self.summary)
        broken["rows"].append(deepcopy(broken["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            convention_audit.grouped_rows(broken)


if __name__ == "__main__":
    unittest.main()
