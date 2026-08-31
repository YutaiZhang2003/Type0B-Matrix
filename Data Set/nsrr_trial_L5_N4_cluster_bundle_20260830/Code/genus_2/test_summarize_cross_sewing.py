from __future__ import annotations

import unittest

from genus_2.summarize_cross_sewing import summarize_cross_sewing


class CrossSewingSummaryTests(unittest.TestCase):
    @staticmethod
    def _row(point_id: str, channel: str, q_l: float) -> dict:
        return {
            "point_id": point_id,
            "channel": channel,
            "recursion_order": 24,
            "quadrature_order": 10,
            "finite_part_radius": 0.035,
            "q_l": q_l,
        }

    def test_summary_separates_old_and_new_sign_effects(self) -> None:
        spin = {"alpha": [0, 0], "beta": [0, 0]}
        fresh = {
            "schema": "ns-genus2-cannon-v7-glasses-parity",
            "implementation_fingerprint": "fresh",
            "config": {
                "physical_lifts": {"theta": [1, -1, 1], "glasses": [1, 1, 1]},
                "points": [{"id": "p"}],
            },
            "spin_characteristics": {"p": {"theta": spin, "glasses": spin}},
            "analytic_checks": {
                "spin_source_characteristic": spin,
                "spin_target_characteristic": spin,
            },
            "rows": [self._row("p", "theta", 1.0), self._row("p", "glasses", 1.0)],
            "crossing": [
                {
                    "point_id": "p",
                    "recursion_order": 24,
                    "quadrature_order": 10,
                    "finite_part_radius": 0.035,
                    "theta_over_glasses": 1.0,
                }
            ],
        }
        old = {
            "rows": [self._row("p", "theta", 1.02), self._row("p", "glasses", 1.0)],
            "crossing": [
                {
                    "point_id": "p",
                    "recursion_order": 24,
                    "quadrature_order": 10,
                    "finite_part_radius": 0.035,
                    "theta_over_glasses": 1.02,
                }
            ],
        }
        audit = summarize_cross_sewing(
            fresh,
            old,
            {"corrected_rows": [{"point_id": "p", "q_l_corrected": 1.01}]},
        )
        self.assertEqual(audit["status"], "pass")
        self.assertAlmostEqual(
            audit["rows"][0]["parity_corrected_theta_over_old_glasses"],
            1.01,
        )
        self.assertAlmostEqual(
            audit["rows"][0]["fresh_theta_over_fresh_glasses"],
            1.0,
        )

    def test_summary_rejects_wrong_spin(self) -> None:
        fresh = {
            "schema": "ns-genus2-cannon-v7-glasses-parity",
            "config": {
                "physical_lifts": {"theta": [1, -1, 1], "glasses": [1, 1, 1]},
                "points": [{"id": "p"}],
            },
            "spin_characteristics": {
                "p": {
                    "theta": {"alpha": [0, 0], "beta": [1, 0]},
                    "glasses": {"alpha": [0, 0], "beta": [0, 0]},
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "spin mismatch"):
            summarize_cross_sewing(fresh, {}, {})


if __name__ == "__main__":
    unittest.main()
