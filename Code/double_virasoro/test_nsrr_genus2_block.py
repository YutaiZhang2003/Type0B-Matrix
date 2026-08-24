"""Regression tests for the low-order NS--R--R double-Virasoro block."""

from __future__ import annotations

from fractions import Fraction
import unittest

import sympy as sp

from nsrr_genus2_block import (
    BRANCHING_RECURSION,
    DirectHalfNSAnchor,
    ZERO_VECTOR,
    auxiliary_majorana_nsrr_series,
    branch_base_levels,
    compare_nsrr_low_order,
    direct_pbw_nsrr_series,
    enlarged_double_virasoro_series,
    ns_branch_labels,
    ramond_branch_labels,
    star_convolve_series,
)


B = sp.Rational(7, 5)
MOMENTA = (
    sp.Rational(11, 23),
    sp.Rational(13, 29),
    sp.Rational(17, 31),
)


class NSRRGenusTwoTests(unittest.TestCase):
    def test_branch_lattice_at_first_order(self) -> None:
        self.assertEqual(
            ns_branch_labels(2),
            (Fraction(-1, 2), Fraction(0), Fraction(1, 2)),
        )
        self.assertEqual(
            ramond_branch_labels(2),
            (
                Fraction(-3, 4),
                Fraction(-1, 4),
                Fraction(1, 4),
                Fraction(3, 4),
            ),
        )
        self.assertEqual(
            ramond_branch_labels(1),
            (Fraction(-1, 4), Fraction(1, 4)),
        )
        self.assertEqual(
            branch_base_levels(
                Fraction(0), Fraction(3, 4), Fraction(1, 4)
            ),
            (0, 2, 0),
        )

    def test_human_reflected_normalization_closes_the_ground_resolution(self) -> None:
        comparison = compare_nsrr_low_order(
            b=B,
            momenta=MOMENTA,
            form_parity=0,
            etas=(1, 1),
            maximum_total_twice_level=0,
        )
        self.assertLess(comparison.maximum_absolute_error, 5.0e-14)
        self.assertEqual(comparison.unsupported_level_triples, ())

    def test_direct_anchor_reproduces_the_human_ground_frame(self) -> None:
        anchor = DirectHalfNSAnchor(b=B, momenta=MOMENTA)
        second_module = BRANCHING_RECURSION.FreeFieldModule(
            "R",
            BRANCHING_RECURSION.real_number(float(B)),
            BRANCHING_RECURSION.real_number(float(MOMENTA[1])),
        )
        third_module = BRANCHING_RECURSION.FreeFieldModule(
            "R",
            BRANCHING_RECURSION.real_number(float(B)),
            BRANCHING_RECURSION.real_number(float(MOMENTA[2])),
        )
        for second_label in (Fraction(-1, 4), Fraction(1, 4)):
            for third_label in (Fraction(-1, 4), Fraction(1, 4)):
                for parity_r1 in (0, 1):
                    for parity_r2 in (0, 1):
                        for eta in (-1, 1):
                            expected = BRANCHING_RECURSION.direct_ground_value(
                                second_module,
                                third_module,
                                second_label,
                                third_label,
                                parity_r1,
                                parity_r2,
                                eta,
                            )
                            actual = anchor.numerator(
                                Fraction(0),
                                second_label,
                                third_label,
                                parity_r1,
                                parity_r2,
                                (parity_r1 + parity_r2) % 2,
                                eta,
                            )
                            self.assertAlmostEqual(
                                complex(sp.N(actual, 30)), expected, places=12
                            )

    def test_low_order_diagnostic_is_componentwise(self) -> None:
        cutoff = 2
        keywords = dict(
            b=B,
            momenta=MOMENTA,
            form_parity=0,
            etas=(1, 1),
            maximum_total_twice_level=cutoff,
        )
        enlarged = enlarged_double_virasoro_series(**keywords)
        auxiliary = auxiliary_majorana_nsrr_series(
            maximum_total_twice_level=cutoff
        )
        physical = direct_pbw_nsrr_series(**keywords)
        factorized = star_convolve_series(
            auxiliary,
            physical,
            maximum_total_twice_level=cutoff,
        )

        # The Human-basis ground resolution closes, but the first Ramond
        # descendant edges remain an explicit convention diagnostic.  Do not
        # silently bless the old zero that came from the removed frame phase.
        for levels in ((0, 2, 0), (0, 0, 2)):
            self.assertGreater(
                abs(
                    enlarged.get(levels, ZERO_VECTOR)[6]
                    - factorized.get(levels, ZERO_VECTOR)[6]
                ),
                1.0e-3,
            )

        # This is the newly supplied half-integral NS anchor.  Both allowed
        # Ramond-copy parity components agree with the independent PBW sewing.
        for component in (3, 5):
            self.assertAlmostEqual(
                enlarged.get((1, 0, 0), ZERO_VECTOR)[component],
                factorized.get((1, 0, 0), ZERO_VECTOR)[component],
                places=11,
            )

        # The other copy remains a real convention diagnostic rather than a
        # silently normalized equality.
        comparison = compare_nsrr_low_order(**keywords)
        self.assertEqual(comparison.unsupported_level_triples, ())
        self.assertEqual(comparison.coefficient_count, 40)
        self.assertGreater(comparison.maximum_relative_error, 0.1)
        self.assertEqual(comparison.worst_levels, (2, 0, 0))

    def test_odd_form_ground_frame(self) -> None:
        comparison = compare_nsrr_low_order(
            b=B,
            momenta=MOMENTA,
            form_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=0,
        )
        self.assertLess(comparison.maximum_absolute_error, 5.0e-14)


if __name__ == "__main__":
    unittest.main()
