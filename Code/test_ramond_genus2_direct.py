"""Regression tests for the direct genus-two NRR theta sewing."""

from __future__ import annotations

import unittest

from ramond_descendant_blocks import RamondThreePointWardMatrix
from ramond_genus2_direct import (
    DirectNRRThetaOracle,
    RRNSDescendantThreeForm,
    analytic_first_r_kac_residue_checks,
    level_triples,
)


C = 37.25
H_NS = 0.73
BETA_1 = 0.67
BETA_2 = 0.83


class RRNSDescendantThreeFormTests(unittest.TestCase):
    def form(self) -> RRNSDescendantThreeForm:
        return RRNSDescendantThreeForm(
            c=C,
            left_weight=C / 24.0 - BETA_1**2,
            ns_weight=H_NS,
            right_weight=C / 24.0 - BETA_2**2,
            sign=1,
        )

    def test_reduces_to_independent_two_r_leg_ward_matrix(self) -> None:
        full = self.form()
        reference = RamondThreePointWardMatrix(
            left_module=full.left_module,
            right_module=full.right_module,
            external_ns_weight=H_NS,
            sign=1,
        )
        for total_level in range(5):
            for left_level in range(total_level + 1):
                right_level = total_level - left_level
                for left_parity in (0, 1):
                    for right_parity in (0, 1):
                        for left in full.left_module.basis(
                            left_level, left_parity
                        ):
                            for right in full.right_module.basis(
                                right_level, right_parity
                            ):
                                for component, middle in (
                                    (0, ()),
                                    (1, (("G", -1),)),
                                ):
                                    self.assertAlmostEqual(
                                        full.value(left, middle, right),
                                        reference.value(
                                            left, right, component
                                        ),
                                        places=11,
                                    )

    def test_middle_l_minus_one_scaling(self) -> None:
        full = self.form()
        expected = full.weights[0] - full.weights[1] - full.weights[2]
        self.assertAlmostEqual(
            full.value((), (("L", -2),), ()),
            expected,
            places=13,
        )


class DirectNRRThetaTests(unittest.TestCase):
    def oracle(self, *, c: complex = C) -> DirectNRRThetaOracle:
        return DirectNRRThetaOracle(
            c=c,
            h_ns=H_NS,
            beta_1=BETA_1,
            beta_2=BETA_2,
            signs=(1, 1),
        )

    def test_ground_normalization_and_level_two_anchors(self) -> None:
        oracle = self.oracle()
        self.assertAlmostEqual(oracle.coefficient(0, 0, 0), 1.0, places=13)
        anchors = {
            (0, 0, 2): 0.0893735696177088,
            (0, 1, 1): 1.55132652614021,
            (0, 2, 0): 0.265908249357633,
            (2, 0, 1): 0.206721162190762,
            (2, 1, 0): 0.0705842032831438,
            (4, 0, 0): 0.118109371891463,
        }
        for levels, expected in anchors.items():
            self.assertAlmostEqual(
                oracle.coefficient(*levels), expected, places=11
            )

    def test_equal_form_half_integer_total_levels_cancel(self) -> None:
        oracle = self.oracle()
        maximum = 0.0
        for ns_twice, r1, r2 in level_triples(5):
            if ns_twice % 2:
                maximum = max(
                    maximum,
                    abs(oracle.coefficient(ns_twice, r1, r2)),
                )
        self.assertLess(maximum, 4.0e-14)

    def test_first_r_kac_residue(self) -> None:
        checks = analytic_first_r_kac_residue_checks(
            h_ns=H_NS,
            beta_1=BETA_1,
            beta_2=BETA_2,
        )
        self.assertEqual(tuple(check.branch for check in checks), (1, -1))
        for check in checks:
            self.assertLess(check.relative_error, 5.0e-15)

    def test_level_five_anchors(self) -> None:
        oracle = self.oracle()
        anchors = {
            (0, 5, 0): 4.9524423357081,
            (2, 4, 0): 12.4638614359077,
            (4, 3, 0): 18.0867400551341,
            (6, 2, 0): 0.184292624555106,
            (8, 1, 0): 0.0506115762371577,
            (10, 0, 0): 0.07136801885572,
        }
        for levels, expected in anchors.items():
            self.assertAlmostEqual(
                oracle.coefficient(*levels), expected, places=9
            )


if __name__ == "__main__":
    unittest.main()
