"""Regression tests for the canonical human-note NS three-form convention."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import sympy as sp


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ns_genus12_finite_c_check import NSDescendantThreeForm  # noqa: E402
from ns_genus2_symbolic_low_order import (  # noqa: E402
    ExactNSDescendantThreeForm,
    exact_osp_three_point,
)
from ns_global_osp_block import osp_three_point  # noqa: E402
from ns_human_convention import (  # noqa: E402
    absolute_three_form_parity,
    glasses_primary_parity_rephasing,
    human_note_rho_sign,
    ns_double_null_factorization_sign,
    ns_null_factorization_sign,
    primary_parity_ward_sign,
    relative_label_from_absolute,
    relative_three_form_label,
    theta_primary_parity_rephasing,
)
from ns_osp_superspace import superspace_three_point  # noqa: E402


BITS = (
    (0, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
)


class NSHumanConventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c, self.h1, self.h2, self.h3 = sp.symbols("c h1 h2 h3")
        self.expected = {
            (0, 0, 0): sp.S.One,
            (1, 1, 0): self.h1 + self.h2 - self.h3,
            (1, 0, 1): self.h1 - self.h2 + self.h3,
            (0, 1, 1): self.h1 - self.h2 - self.h3,
            (1, 0, 0): sp.S.One,
            (0, 1, 0): sp.S.One,
            (0, 0, 1): -sp.S.One,
            (1, 1, 1): -(
                self.h1 + self.h2 + self.h3 - sp.Rational(1, 2)
            ),
        }

    def test_single_canonical_sign_function(self) -> None:
        expected_signs = {
            bits: (-1 if sum(bits) % 2 and bits[2] else 1)
            for bits in BITS
        }
        self.assertEqual(
            {bits: human_note_rho_sign(bits) for bits in BITS},
            expected_signs,
        )

    def test_relative_label_is_distinct_from_absolute_parity(self) -> None:
        descendants = (1, 1, 0)
        primaries = (1, 0, 0)
        label = relative_three_form_label(descendants)
        self.assertEqual(label, 0)
        absolute = absolute_three_form_parity(label, primaries)
        self.assertEqual(absolute, 1)
        self.assertEqual(relative_label_from_absolute(absolute, primaries), label)

    def test_current_one_and_double_null_sign_tables(self) -> None:
        for descendants in product((0, 1), repeat=3):
            a, _c, _e = descendants
            for primaries in product((0, 1), repeat=3):
                p1, p2, _p3 = primaries
                for delta in (0, 1):
                    expected_one = (
                        (-1) ** (delta * (p1 + a)),
                        1,
                        (-1) ** (delta * (1 + p2)),
                    )
                    observed_one = tuple(
                        ns_null_factorization_sign(
                            slot=slot,
                            null_parity=delta,
                            descendant_parities=descendants,
                            primary_parities=primaries,
                        )
                        for slot in range(3)
                    )
                    self.assertEqual(observed_one, expected_one)

                    expected_double = (
                        (-1) ** (delta * (p1 + a)),
                        (-1) ** (delta * (1 + p1 + p2 + a)),
                        (-1) ** (delta * p2),
                    )
                    observed_double = tuple(
                        ns_double_null_factorization_sign(
                            pair=pair,
                            null_parity=delta,
                            descendant_parities=descendants,
                            primary_parities=primaries,
                        )
                        for pair in ((0, 1), (0, 2), (1, 2))
                    )
                    self.assertEqual(observed_double, expected_double)

    def test_graph_primary_parity_rephasings_are_sign_characters(self) -> None:
        for lifts in product((-1, 1), repeat=3):
            for primaries in product((0, 1), repeat=3):
                for reducer in (
                    theta_primary_parity_rephasing,
                    glasses_primary_parity_rephasing,
                ):
                    prefactor, effective = reducer(lifts, primaries)
                    self.assertIn(prefactor, (-1, 1))
                    self.assertTrue(all(value in (-1, 1) for value in effective))
                    identity_prefactor, identity_lifts = reducer(
                        effective, (0, 0, 0)
                    )
                    self.assertEqual(identity_prefactor, 1)
                    self.assertEqual(identity_lifts, effective)

    def test_generic_primary_parities_recover_the_osp_table(self) -> None:
        """Recover all eight entries from rho_0(000)=rho_1(010)=1."""

        numeric_weights = (0.71, 0.83, 0.94)
        substitutions = dict(
            zip((self.h1, self.h2, self.h3), numeric_weights)
        )
        g = (("G", -1),)
        for primaries in product((0, 1), repeat=3):
            exact_form = ExactNSDescendantThreeForm(
                c=self.c,
                weights=(self.h1, self.h2, self.h3),
                primary_parities=primaries,
            )
            numeric_form = NSDescendantThreeForm(
                c=17.0,
                bra_weight=numeric_weights[0],
                middle_weight=numeric_weights[1],
                ket_weight=numeric_weights[2],
                primary_parities=primaries,
            )
            for bits, even_primary_value in self.expected.items():
                phase = (-1) ** (
                    primaries[0] * bits[0]
                    + primaries[1] * bits[2]
                )
                expected = phase * even_primary_value
                arguments = dict(
                    n1=0,
                    n2=0,
                    n3=0,
                    epsilon1=bits[0],
                    epsilon2=bits[1],
                    epsilon3=bits[2],
                    d1=self.h1,
                    d2=self.h2,
                    d3=self.h3,
                    primary_parities=primaries,
                )
                self.assertEqual(
                    sp.cancel(exact_osp_three_point(**arguments) - expected),
                    0,
                    (primaries, bits, "global"),
                )
                self.assertEqual(
                    sp.cancel(superspace_three_point(**arguments) - expected),
                    0,
                    (primaries, bits, "superspace"),
                )
                states = tuple(g if bit else () for bit in bits)
                self.assertEqual(
                    sp.cancel(exact_form.value(*states) - expected),
                    0,
                    (primaries, bits, "exact Ward"),
                )

                numeric_expected = complex(expected.subs(substitutions))
                numeric_arguments = {
                    **arguments,
                    "d1": numeric_weights[0],
                    "d2": numeric_weights[1],
                    "d3": numeric_weights[2],
                }
                global_value = osp_three_point(**numeric_arguments)
                ward_value = numeric_form.value(*states)
                self.assertAlmostEqual(
                    global_value.real, numeric_expected.real, places=13
                )
                self.assertAlmostEqual(
                    global_value.imag, numeric_expected.imag, places=13
                )
                self.assertAlmostEqual(
                    ward_value.real, numeric_expected.real, places=13
                )
                self.assertAlmostEqual(
                    ward_value.imag, numeric_expected.imag, places=13
                )

            self.assertEqual(exact_form.value((), (), ()), 1)
            self.assertEqual(exact_form.value((), g, ()), 1)

    def test_primary_parities_are_propagated_through_non_global_ward_steps(self) -> None:
        """Odd contour crossings obey the same graded phase beyond OSp."""

        even_form = ExactNSDescendantThreeForm(
            c=self.c, weights=(self.h1, self.h2, self.h3)
        )
        states_to_check = (
            ((('G', -3),), (), ()),
            ((), (('G', -3),), (('G', -1),)),
            ((('G', -1),), (('L', -4),), (('G', -3),)),
            ((('G', -3),), (('G', -1),), (('L', -4),)),
        )
        for primaries in product((0, 1), repeat=3):
            graded_form = ExactNSDescendantThreeForm(
                c=self.c,
                weights=(self.h1, self.h2, self.h3),
                primary_parities=primaries,
            )
            for states in states_to_check:
                descendant_parities = tuple(
                    sum(kind == "G" for kind, _ in state) % 2
                    for state in states
                )
                expected = (
                    primary_parity_ward_sign(
                        descendant_parities, primaries
                    )
                    * even_form.value(*states)
                )
                self.assertEqual(
                    sp.cancel(graded_form.value(*states) - expected),
                    0,
                    (primaries, states),
                )

    def test_exact_global_api_matches_human_note_table(self) -> None:
        for bits, expected in self.expected.items():
            observed = exact_osp_three_point(
                n1=0,
                n2=0,
                n3=0,
                epsilon1=bits[0],
                epsilon2=bits[1],
                epsilon3=bits[2],
                d1=self.h1,
                d2=self.h2,
                d3=self.h3,
            )
            self.assertEqual(sp.cancel(observed - expected), 0, bits)

    def test_exact_ward_api_matches_human_note_table(self) -> None:
        form = ExactNSDescendantThreeForm(
            c=self.c, weights=(self.h1, self.h2, self.h3)
        )
        g = (("G", -1),)
        for bits, expected in self.expected.items():
            states = tuple(g if bit else () for bit in bits)
            self.assertEqual(sp.cancel(form.value(*states) - expected), 0, bits)

    def test_numeric_global_and_ward_apis_match_human_note_table(self) -> None:
        values = {self.h1: 0.71, self.h2: 0.83, self.h3: 0.94}
        form = NSDescendantThreeForm(
            c=17.0,
            bra_weight=values[self.h1],
            middle_weight=values[self.h2],
            ket_weight=values[self.h3],
        )
        g = (("G", -1),)
        for bits, symbolic_expected in self.expected.items():
            expected = complex(symbolic_expected.subs(values))
            global_value = osp_three_point(
                n1=0,
                n2=0,
                n3=0,
                epsilon1=bits[0],
                epsilon2=bits[1],
                epsilon3=bits[2],
                d1=values[self.h1],
                d2=values[self.h2],
                d3=values[self.h3],
            )
            states = tuple(g if bit else () for bit in bits)
            self.assertAlmostEqual(global_value.real, expected.real, places=13)
            self.assertAlmostEqual(global_value.imag, expected.imag, places=13)
            ward_value = form.value(*states)
            self.assertAlmostEqual(ward_value.real, expected.real, places=13)
            self.assertAlmostEqual(ward_value.imag, expected.imag, places=13)

    def test_negative_middle_supercurrent_uses_the_human_note_ward_sign(self) -> None:
        """Check the first non-global G Ward identity in closed form."""

        form = ExactNSDescendantThreeForm(
            c=self.c, weights=(self.h1, self.h2, self.h3)
        )
        g_half = (("G", -1),)
        g_three_halves = (("G", -3),)
        cases = {
            ((), g_three_halves, ()): -sp.S.One,
            ((), g_three_halves, g_half): -self.h1 + self.h2 + 3 * self.h3,
            (g_half, g_three_halves, ()): self.h1 - self.h2 + self.h3,
            (g_half, g_three_halves, g_half): -(
                2 * self.h1 - 2 * self.h2 - 6 * self.h3 + 1
            ) / 2,
        }
        for states, expected in cases.items():
            self.assertEqual(sp.cancel(form.value(*states) - expected), 0, states)

    def test_first_non_global_descendants_in_each_slot(self) -> None:
        """Closed forms obtained by one T or G contour from the base table."""

        form = ExactNSDescendantThreeForm(
            c=self.c, weights=(self.h1, self.h2, self.h3)
        )
        g_half = (("G", -1),)
        g_three_halves = (("G", -3),)
        l_two = (("L", -4),)
        cases = {
            (g_three_halves, (), ()): sp.S.One,
            ((), g_three_halves, ()): -sp.S.One,
            ((), (), g_three_halves): -sp.S.One,
            (l_two, (), ()): self.h1 + 2 * self.h2 - self.h3,
            ((), l_two, ()): -self.h1 + self.h2 + 2 * self.h3,
            ((), (), l_two): -self.h1 + 2 * self.h2 + self.h3,
            (g_three_halves, g_half, ()): self.h1 + 3 * self.h2 - self.h3,
            (g_three_halves, (), g_half): self.h1 - self.h2 - self.h3,
            ((), g_half, g_three_halves): self.h1 - 3 * self.h2 - self.h3,
            (g_half, (), g_three_halves): -self.h1 - self.h2 + self.h3,
            (g_half, g_three_halves, ()): self.h1 - self.h2 + self.h3,
            ((), g_three_halves, g_half): -self.h1 + self.h2 + 3 * self.h3,
        }
        for states, expected in cases.items():
            self.assertEqual(sp.cancel(form.value(*states) - expected), 0, states)


if __name__ == "__main__":
    unittest.main()
