"""Tests for the explicitly provisional intrinsic-NS-parity suggestion.

These tests certify only what the direct low-order PBW audit presently says.
They do not promote the machine suggestion to a human-verified convention.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import unittest

import sympy as sp

import nsrr_genus2_block as certified
from machine_suggestion_nsrr_primary_parity import (
    DISCLAIMER,
    HUMAN_VERIFIED,
    MACHINE_SUGGESTION,
    branch_component,
    compare_machine_suggestion_to_direct_pbw,
    machine_suggested_auxiliary_majorana_series,
    machine_suggested_direct_pbw_series,
    machine_suggested_double_virasoro_series,
    machine_suggested_level_triples,
    proposed_extra_branch_sign,
)
from theta_star_algebra import theta_quadratic_sign


B = sp.Rational(7, 5)
MOMENTA = (
    sp.Rational(11, 23),
    sp.Rational(13, 29),
    sp.Rational(17, 31),
)
SECOND_B = sp.Rational(9, 7)
SECOND_MOMENTA = (
    sp.Rational(5, 17),
    sp.Rational(7, 19),
    sp.Rational(11, 23),
)


class MachineSuggestedNSPrimaryParityTests(unittest.TestCase):
    def test_file_is_unambiguously_provisional(self) -> None:
        self.assertTrue(MACHINE_SUGGESTION)
        self.assertFalse(HUMAN_VERIFIED)
        self.assertIn("MACHINE SUGGESTION", DISCLAIMER)
        self.assertIn("not been fully human-verified", DISCLAIMER)

    def test_full_quadratic_sign_has_the_proposed_relative_factor(self) -> None:
        for p_ns, k_ns, alpha_2, alpha_3 in product((0, 1), repeat=4):
            old_component = k_ns | (alpha_2 << 1) | (alpha_3 << 2)
            new_component = (p_ns ^ k_ns) | (alpha_2 << 1) | (alpha_3 << 2)
            self.assertEqual(
                theta_quadratic_sign(new_component),
                theta_quadratic_sign(old_component)
                * proposed_extra_branch_sign(p_ns, alpha_2, alpha_3),
            )

    def test_branch_component_uses_full_ns_parity(self) -> None:
        self.assertEqual(
            branch_component(
                primary_parity=1,
                n_ns=Fraction(0),
                parity_r1=1,
                parity_r2=0,
            ),
            3,
        )
        self.assertEqual(
            branch_component(
                primary_parity=1,
                n_ns=Fraction(1, 2),
                parity_r1=1,
                parity_r2=0,
            ),
            2,
        )

    def test_provisional_lattice_and_auxiliary_factor_extend_to_level_four(
        self,
    ) -> None:
        triples = tuple(machine_suggested_level_triples(4))
        self.assertIn((4, 0, 0), triples)
        self.assertIn((2, 2, 0), triples)
        self.assertIn((0, 2, 2), triples)
        auxiliary = machine_suggested_auxiliary_majorana_series(
            maximum_total_twice_level=4
        )
        self.assertTrue(set(auxiliary).issubset(set(triples)))
        # The extension must be literal at the old certified cutoff.
        expected = certified.auxiliary_majorana_nsrr_series(
            maximum_total_twice_level=2
        )
        actual = machine_suggested_auxiliary_majorana_series(
            maximum_total_twice_level=2
        )
        self.assertEqual(expected, actual)

    def test_p_zero_direct_pbw_reduces_to_the_certified_oracle(self) -> None:
        keywords = dict(
            b=B,
            momenta=MOMENTA,
            form_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=2,
        )
        expected = certified.direct_pbw_nsrr_series(**keywords)
        actual = machine_suggested_direct_pbw_series(
            **keywords, primary_parity=0
        )
        self.assertEqual(expected.keys(), actual.keys())
        for levels in expected:
            for left, right in zip(expected[levels], actual[levels]):
                self.assertAlmostEqual(left, right, places=13)

    def test_every_ground_sector_matches_direct_pbw(self) -> None:
        for p_ns, relative_f, eta, eta_prime in product(
            (0, 1), (0, 1), (1, -1), (1, -1)
        ):
            comparison = compare_machine_suggestion_to_direct_pbw(
                b=B,
                momenta=MOMENTA,
                primary_parity=p_ns,
                form_parity=relative_f,
                etas=(eta, eta_prime),
                maximum_total_twice_level=0,
            )
            self.assertLess(
                comparison.maximum_absolute_error,
                5.0e-13,
                (p_ns, relative_f, eta, eta_prime),
            )

    def test_every_ground_sector_matches_at_a_second_exact_point(self) -> None:
        for p_ns, relative_f, eta, eta_prime in product(
            (0, 1), (0, 1), (1, -1), (1, -1)
        ):
            comparison = compare_machine_suggestion_to_direct_pbw(
                b=SECOND_B,
                momenta=SECOND_MOMENTA,
                primary_parity=p_ns,
                form_parity=relative_f,
                etas=(eta, eta_prime),
                maximum_total_twice_level=0,
            )
            self.assertLess(
                comparison.maximum_absolute_error,
                5.0e-13,
                (p_ns, relative_f, eta, eta_prime),
            )

    def test_direct_pbw_obeys_proposed_parity_lift_through_level_four(
        self,
    ) -> None:
        keywords = dict(
            b=B,
            momenta=MOMENTA,
            form_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=4,
        )
        even = machine_suggested_direct_pbw_series(
            **keywords, primary_parity=0
        )
        odd = machine_suggested_direct_pbw_series(
            **keywords, primary_parity=1
        )
        for levels in machine_suggested_level_triples(4):
            for component in range(8):
                parity_r1 = (component >> 1) & 1
                parity_r2 = (component >> 2) & 1
                expected = (
                    proposed_extra_branch_sign(1, parity_r1, parity_r2)
                    * even[levels][component]
                )
                self.assertAlmostEqual(
                    odd[levels][component ^ 1], expected, places=12
                )

    def test_double_virasoro_covariance_first_breaks_at_level_three(
        self,
    ) -> None:
        keywords = dict(
            b=B,
            momenta=MOMENTA,
            form_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=3,
        )
        even = machine_suggested_double_virasoro_series(
            **keywords, primary_parity=0
        )
        odd = machine_suggested_double_virasoro_series(
            **keywords, primary_parity=1
        )
        maximum_by_total = [0.0] * 4
        for levels in machine_suggested_level_triples(3):
            for component in range(8):
                parity_r1 = (component >> 1) & 1
                parity_r2 = (component >> 2) & 1
                expected = (
                    proposed_extra_branch_sign(1, parity_r1, parity_r2)
                    * even.get(levels, (0.0j,) * 8)[component]
                )
                error = abs(
                    odd.get(levels, (0.0j,) * 8)[component ^ 1]
                    - expected
                )
                maximum_by_total[sum(levels)] = max(
                    maximum_by_total[sum(levels)], error
                )
        self.assertLess(max(maximum_by_total[:3]), 5.0e-13)
        # This is an intentionally retained negative result.  It prevents
        # the provisional file from claiming that the unresolved branching
        # convention is already fixed by the final quadratic sign.
        self.assertGreater(maximum_by_total[3], 1.0)

    def test_proposed_extra_sign_is_needed_by_the_odd_ground_frame(self) -> None:
        keywords = dict(
            b=B,
            momenta=MOMENTA,
            form_parity=1,
            primary_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=0,
        )
        physical = machine_suggested_direct_pbw_series(**keywords)
        # Undo only the proposed p_NS*(alpha_2+alpha_3) orientation sign.
        without_sign = {
            levels: tuple(
                value
                * proposed_extra_branch_sign(
                    1, (component >> 1) & 1, (component >> 2) & 1
                )
                for component, value in enumerate(vector)
            )
            for levels, vector in physical.items()
        }
        auxiliary = certified.auxiliary_majorana_nsrr_series(
            maximum_total_twice_level=0
        )
        wrong = certified.star_convolve_series(
            auxiliary, without_sign, maximum_total_twice_level=0
        )
        correct = compare_machine_suggestion_to_direct_pbw(**keywords)
        self.assertLess(correct.maximum_absolute_error, 5.0e-13)
        # The omitted sign reverses both nonzero mixed-eta ground components
        # in the literal Human-Note odd-form phase.
        self.assertEqual(wrong[(0, 0, 0)][3], -2.0j)
        self.assertEqual(wrong[(0, 0, 0)][5], -2.0j)

    def test_first_level_report_closes_after_human_basis_conversion(self) -> None:
        passing = compare_machine_suggestion_to_direct_pbw(
            b=B,
            momenta=MOMENTA,
            primary_parity=1,
            form_parity=1,
            etas=(1, -1),
            maximum_total_twice_level=1,
        )
        unresolved = compare_machine_suggestion_to_direct_pbw(
            b=B,
            momenta=MOMENTA,
            primary_parity=1,
            form_parity=0,
            etas=(1, -1),
            maximum_total_twice_level=1,
        )
        self.assertLess(passing.maximum_absolute_error, 5.0e-13)
        self.assertLess(unresolved.maximum_absolute_error, 5.0e-13)


if __name__ == "__main__":
    unittest.main()
