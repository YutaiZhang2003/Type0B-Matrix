"""Exact generic-parity checks for the NS double-Virasoro factorization."""

from __future__ import annotations

from itertools import product
import unittest

import sympy as sp

from check_second_virasoro_primary import (
    ExactNSDescendantThreeForm,
    product_rho,
    solve_v1,
    v0,
    vhalf,
)
from check_ungraded_branching_low_level import ungraded_tensor_three_point


class NSDoubleVirasoroPrimaryParityTests(unittest.TestCase):
    def test_direct_branching_vectors_with_generic_primary_parity(self) -> None:
        """Check 8 parity choices times all 27 ``k_i=0,1,2`` triples."""

        b = sp.Rational(3, 2)
        momenta = (
            sp.Rational(2, 5),
            sp.Rational(-1, 3),
            sp.Rational(3, 7),
        )
        q = b + 1 / b
        c = sp.Rational(3, 2) + 3 * q**2
        weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
        vectors_by_slot = []
        for momentum in momenta:
            _module, level_two = solve_v1(b, momentum)
            vectors_by_slot.append(
                {0: v0(), 1: vhalf(q, momentum), 2: level_two}
            )

        even_form = ExactNSDescendantThreeForm(c=c, weights=weights)
        even_values = {}
        for labels in product((0, 1, 2), repeat=3):
            vectors = tuple(
                vectors_by_slot[slot][label]
                for slot, label in enumerate(labels)
            )
            direct = ungraded_tensor_three_point(even_form, vectors)
            product_formula = product_rho(
                labels,
                momenta,
                b,
                q,
                third_slot_sign=True,
            )
            self.assertEqual(sp.cancel(direct - product_formula), 0, labels)
            even_values[labels] = direct

        checked = 0
        changed_squares = 0
        for primary_parities in product((0, 1), repeat=3):
            form = ExactNSDescendantThreeForm(
                c=c,
                weights=weights,
                primary_parities=primary_parities,
            )
            for labels in product((0, 1, 2), repeat=3):
                vectors = tuple(
                    vectors_by_slot[slot][label]
                    for slot, label in enumerate(labels)
                )
                direct = ungraded_tensor_three_point(form, vectors)
                even = even_values[labels]
                twisted_first = {
                    state: (-1) ** (len(state[0]) * primary_parities[0])
                    * coefficient
                    for state, coefficient in vectors[0].items()
                }
                transformed = ungraded_tensor_three_point(
                    even_form,
                    (twisted_first, vectors[1], vectors[2]),
                )
                sign = (-1) ** (
                    primary_parities[0] * (labels[0] % 2)
                    + primary_parities[1] * (labels[2] % 2)
                )
                self.assertEqual(
                    sp.cancel(direct - sign * transformed),
                    0,
                    (primary_parities, labels),
                )
                square_changed = sp.cancel(direct**2 - even**2) != 0
                if primary_parities[0] == 0:
                    self.assertFalse(square_changed, (primary_parities, labels))
                elif square_changed:
                    changed_squares += 1
                checked += 1
        self.assertEqual(checked, 216)
        self.assertEqual(changed_squares, 64)


if __name__ == "__main__":
    unittest.main()
