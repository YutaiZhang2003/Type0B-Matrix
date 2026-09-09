"""Tests for the certified Type-0B four-point continuation optimizer."""

from __future__ import annotations

import unittest

from type0b_sphere_four_point_continuation import (
    centered_rectangles,
    certify_four_point_continuation_rectangle,
    search_four_point_continuation_rectangles,
)
from type0b_sphere_four_point_hybrid import (
    CONVERGENT_RAY_COEFFICIENTS,
    LARGE_RESIDUE_RAY_RECTANGLE,
    WALL_ONE_RAY_COEFFICIENTS,
    WALL_ONE_RAY_RECTANGLE,
    audit_four_point_convergence,
)


def _signature(record):
    return record.partition, record.side, record.kind, record.sector


class FourPointContinuationOptimizerTests(unittest.TestCase):
    def test_wall_one_rectangle_is_production_ready(self):
        certificate = certify_four_point_continuation_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
            ray_real_sign=1,
        )
        self.assertTrue(certificate.mathematically_certified)
        self.assertTrue(certificate.production_ready)
        self.assertEqual(certificate.crossed_walls, (1,))
        self.assertEqual(certificate.residue_record_count, 5)
        self.assertEqual(certificate.residue_cost, 10)
        self.assertEqual(certificate.maximum_combined_pole_order, 2)
        self.assertEqual(certificate.maximum_logarithm_power, 1)
        self.assertAlmostEqual(
            certificate.minimum_margin_lower_bound, 0.0512, places=13
        )
        self.assertAlmostEqual(
            certificate.maximum_phase_upper_bound, 1.527296, places=13
        )
        self.assertAlmostEqual(
            certificate.maximum_phase_to_margin_upper_bound, 29.83, places=12
        )
        self.assertEqual(
            certificate.limiting_signature,
            ((1, 2), (1, 2), "continuous", None),
        )

    def test_exponent_bounds_cover_rectangle_corners_and_center(self):
        certificate = certify_four_point_continuation_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
            ray_real_sign=1,
        )
        bound_by_signature = {
            bound.signature: bound for bound in certificate.exponent_bounds
        }
        x_values = (*WALL_ONE_RAY_RECTANGLE[0], sum(WALL_ONE_RAY_RECTANGLE[0]) / 2)
        t_values = (*WALL_ONE_RAY_RECTANGLE[1], sum(WALL_ONE_RAY_RECTANGLE[1]) / 2)
        for x_value in x_values:
            for t_value in t_values:
                base = complex(x_value, t_value)
                outgoing = tuple(
                    coefficient * base
                    for coefficient in WALL_ONE_RAY_COEFFICIENTS
                )
                audit = audit_four_point_convergence(
                    outgoing, include_residues=True
                )
                self.assertEqual(
                    set(bound_by_signature),
                    {_signature(record) for record in audit.records},
                )
                for record in audit.records:
                    bound = bound_by_signature[_signature(record)]
                    exponent = (
                        record.momentum * record.momentum
                        - record.channel_energy * record.channel_energy
                        - record.threshold
                    )
                    self.assertGreaterEqual(
                        exponent.real + 2.0e-12, bound.minimum_real_part
                    )
                    self.assertLessEqual(
                        abs(exponent.imag),
                        bound.maximum_absolute_imaginary_part + 2.0e-12,
                    )

    def test_wall_four_domain_is_mathematical_but_not_production_ready(self):
        certificate = certify_four_point_continuation_rectangle(
            LARGE_RESIDUE_RAY_RECTANGLE[0],
            LARGE_RESIDUE_RAY_RECTANGLE[1],
            ray_coefficients=CONVERGENT_RAY_COEFFICIENTS,
            ray_real_sign=-1,
        )
        self.assertTrue(certificate.mathematically_certified)
        self.assertFalse(certificate.production_ready)
        self.assertEqual(certificate.crossed_walls, (1, 2, 3, 4))
        self.assertEqual(certificate.maximum_combined_pole_order, 4)
        self.assertEqual(certificate.maximum_logarithm_power, 3)
        self.assertGreater(certificate.residue_cost, 70)
        self.assertGreater(
            certificate.maximum_phase_to_margin_upper_bound, 200.0
        )

    def test_hard_margin_requirement_can_reject_a_positive_domain(self):
        certificate = certify_four_point_continuation_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
            required_minimum_margin=0.06,
        )
        self.assertTrue(certificate.domain_certificate.certified)
        self.assertFalse(certificate.mathematically_certified)
        self.assertFalse(certificate.production_ready)

    def test_search_prefers_the_complete_wall_one_chamber(self):
        search = search_four_point_continuation_rectangles(
            ray_candidates=(
                WALL_ONE_RAY_COEFFICIENTS,
                CONVERGENT_RAY_COEFFICIENTS,
            ),
            rectangles=(WALL_ONE_RAY_RECTANGLE,),
            ray_real_sign=1,
            required_minimum_margin=0.02,
            required_wall_clearance=0.01,
            keep=2,
        )
        self.assertEqual(search.candidates_evaluated, 2)
        self.assertGreaterEqual(search.production_ready_count, 1)
        self.assertIsNotNone(search.best)
        assert search.best is not None
        self.assertEqual(search.best.ray_coefficients, WALL_ONE_RAY_COEFFICIENTS)
        self.assertTrue(search.best.production_ready)
        self.assertEqual(search.best.crossed_walls, (1,))

    def test_centered_rectangle_builder(self):
        rectangles = centered_rectangles(
            (0.25, 0.27),
            (0.60,),
            x_half_width=0.01,
            t_half_width=0.02,
        )
        self.assertEqual(
            rectangles,
            (
                ((0.24, 0.26), (0.58, 0.62)),
                ((0.26, 0.28), (0.58, 0.62)),
            ),
        )


if __name__ == "__main__":
    unittest.main()
