#!/usr/bin/env python3
"""Exact tests for the low double-Virasoro primary enumerator."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ns_branch_norms as ns  # noqa: E402
import enumerate_double_virasoro_primaries as primaries  # noqa: E402


class LowPrimaryEnumerationTests(unittest.TestCase):
    def assert_expression_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_component_counts(self):
        self.assertEqual(len(primaries.primary_components(0)), 1)
        self.assertEqual(len(primaries.primary_components(sp.Rational(1, 2))), 2)
        self.assertEqual(len(primaries.primary_components(1)), 7)

    def test_first_positive_and_negative_branches(self):
        positive = {
            (auxiliary, virasoro, supercurrent): coefficient
            for auxiliary, virasoro, supercurrent, coefficient
            in primaries.primary_components(sp.Rational(1, 2))
        }
        negative = {
            (auxiliary, virasoro, supercurrent): coefficient
            for auxiliary, virasoro, supercurrent, coefficient
            in primaries.primary_components(-sp.Rational(1, 2))
        }
        self.assert_expression_equal(
            positive[((1,), (), ())], ns.Q / 2 + ns.P
        )
        self.assert_expression_equal(
            negative[((1,), (), ())], ns.Q / 2 - ns.P
        )
        self.assertEqual(positive[((), (), (1,))], 1)
        self.assertEqual(negative[((), (), (1,))], 1)

    def test_reflection(self):
        for magnitude in (sp.Rational(1, 2), sp.Integer(1)):
            positive = primaries.primary_components(magnitude)
            negative = primaries.primary_components(-magnitude)
            self.assertEqual(
                tuple(term[:3] for term in negative),
                tuple(term[:3] for term in positive),
            )
            for negative_term, positive_term in zip(negative, positive):
                self.assert_expression_equal(
                    negative_term[3], positive_term[3].subs(ns.P, -ns.P)
                )

    def test_level_two_l_minus_one_squared_coefficient(self):
        coefficients = {
            (auxiliary, virasoro, supercurrent): coefficient
            for auxiliary, virasoro, supercurrent, coefficient
            in primaries.primary_components(1)
        }
        self.assertEqual(coefficients[((), (1, 1), ())], -2)

    def test_unified_level_two_formula(self):
        for sigma in (-1, 1):
            coefficients = {
                (auxiliary, virasoro, supercurrent): coefficient
                for auxiliary, virasoro, supercurrent, coefficient
                in primaries.primary_components(sigma)
            }
            d_sigma = (
                4 * ns.P**2
                + 8 * sigma * ns.P * ns.Q
                + 3 * ns.Q**2
                + 4
            )
            expected = {
                ((3, 1), (), ()): (
                    (ns.Q + sigma * ns.P)
                    * (ns.Q + 2 * sigma * ns.P)
                    * d_sigma
                    / 2
                ),
                ((3,), (), (1,)): (ns.Q + sigma * ns.P) * d_sigma,
                ((1,), (), (3,)): -(
                    (ns.Q + sigma * ns.P)
                    * (ns.Q + 2 * sigma * ns.P) ** 2
                ),
                ((1,), (1,), (1,)): -4 * (ns.Q + sigma * ns.P),
                ((), (2,), ()): -(ns.Q + 2 * sigma * ns.P) ** 2,
                ((), (1, 1), ()): -2,
                ((), (), (3, 1)): 2
                * (
                    2 * ns.P**2
                    + 3 * sigma * ns.P * ns.Q
                    + ns.Q**2
                    + 1
                ),
            }
            self.assertEqual(set(coefficients), set(expected))
            for state, expected_coefficient in expected.items():
                self.assert_expression_equal(
                    coefficients[state], expected_coefficient
                )


if __name__ == "__main__":
    unittest.main()
