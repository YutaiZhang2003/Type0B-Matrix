#!/usr/bin/env python3
"""Exact tests for the first Ramond physical-L1 branch reduction."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import decompose_physical_l1 as reduction  # noqa: E402
import check_zero_mode_action as zero_modes  # noqa: E402


class PhysicalL1ReductionTests(unittest.TestCase):
    def test_first_ramond_reduction(self):
        for sigma in (-1, 1):
            upper = sigma * sp.Rational(3, 4)
            expected_lower = -sigma * sp.Rational(1, 4)
            for parity in (0, 1):
                decomposition = reduction.decompose_l1_image(upper, parity)
                for lower, coefficient in decomposition.items():
                    expected = (
                        reduction.closed_coefficient(upper, parity)
                        if lower == expected_lower
                        else 0
                    )
                    self.assertEqual(sp.cancel(coefficient - expected), 0)

    def test_first_step_matches_direct_free_field_action(self):
        label = sp.Rational(3, 4)
        for parity in (0, 1):
            pbw_image = {
                (modes, ground, (), (), physical_ground): coefficient
                for (
                    modes,
                    ground,
                    physical_ground,
                ), coefficient in reduction.physical_l1_image(
                    label, parity
                ).items()
            }
            free_field_image = reduction.physical_l1_positive_raw(label, parity)
            residual = reduction._linear_combination(
                ((1, pbw_image), (-1, free_field_image))
            )
            self.assertEqual(residual, {})

    def test_physical_l1_is_minus_auxiliary_l1_on_primaries(self):
        for label in (
            sp.Rational(1, 4),
            sp.Rational(3, 4),
            sp.Rational(5, 4),
            sp.Rational(7, 4),
        ):
            for parity in (0, 1):
                physical = reduction.physical_l1_positive_raw(label, parity)
                auxiliary = reduction.auxiliary_l1_positive_raw(label, parity)
                self.assertEqual(
                    reduction._linear_combination(
                        ((1, physical), (1, auxiliary))
                    ),
                    {},
                )

    def test_double_virasoro_l_minus_one_matches_existing_audit(self):
        b = sp.symbols("b", nonzero=True)
        q_value = b + 1 / b
        substitutions = {
            reduction.branch.Q: q_value,
            reduction.branch.P: reduction.branch.P,
        }
        for sheet in (-1, 1):
            label = sp.Rational(sheet, 4)
            for parity in (0, 1):
                for copy in (1, 2):
                    expected_abstract = zero_modes.double_vir_l_minus_one(
                        sheet, parity, copy, b
                    )
                    expected = {}
                    for (auxiliary, physical), outer in expected_abstract.items():
                        for final, inner in reduction.branch.descendant_to_fock(
                            physical, realization=-1
                        ).items():
                            reduction.add_term(
                                expected,
                                reduction._join_product_state(auxiliary, final),
                                (outer * inner).subs(
                                    substitutions, simultaneous=True
                                ),
                            )
                    calculated = reduction.double_virasoro_descendant(
                        label,
                        parity,
                        (1,) if copy == 1 else (),
                        (1,) if copy == 2 else (),
                        b,
                    )
                    residual = reduction._linear_combination(
                        ((1, calculated), (-1, expected))
                    )
                    self.assertEqual(residual, {})

    def test_general_auxiliary_virasoro_lowering_modes(self):
        ground = {((), 0, (), (), 0): sp.Integer(1)}
        for first, second, bracket_coefficient in ((1, 2, 1), (1, 3, 2)):
            left = reduction.auxiliary_lf_minus(
                reduction.auxiliary_lf_minus(ground, second), first
            )
            right = reduction.auxiliary_lf_minus(
                reduction.auxiliary_lf_minus(ground, first), second
            )
            bracket = reduction.auxiliary_lf_minus(
                ground, first + second
            )
            self.assertEqual(
                reduction._linear_combination(
                    ((1, left), (-1, right), (-bracket_coefficient, bracket))
                ),
                {},
            )

    def test_five_quarter_level_two_reduction(self):
        b = sp.Rational(3, 2)
        expected = reduction.closed_five_quarter_coefficients(b)
        even = reduction.decompose_positive_l1_at_five_quarters(0, b)
        odd = reduction.decompose_positive_l1_at_five_quarters(1, b)
        self.assertEqual(set(even), set(expected))
        self.assertEqual(set(odd), set(expected))
        for descriptor, coefficient in expected.items():
            self.assertEqual(sp.cancel(even[descriptor] - coefficient), 0)
            self.assertEqual(sp.cancel(odd[descriptor] - coefficient), 0)

    def test_seven_quarter_image_closes_in_three_quarter_module(self):
        decomposition = reduction.decompose_positive_l1_single_lower_module(
            sp.Rational(7, 4),
            0,
            sp.Rational(3, 2),
            momentum=sp.Rational(2, 5),
        )
        self.assertEqual(len(decomposition), 20)
        self.assertTrue(
            all(coefficient != 0 for coefficient in decomposition.values())
        )
        self.assertEqual(
            decomposition[(sp.Rational(3, 4), (), (4,))],
            -sp.Rational(2134586495, 39213529488),
        )
        self.assertEqual(
            decomposition[(sp.Rational(3, 4), (1, 1, 1, 1), ())],
            sp.Rational(18105170625, 1248489275744),
        )


if __name__ == "__main__":
    unittest.main()
