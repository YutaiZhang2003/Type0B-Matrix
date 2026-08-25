#!/usr/bin/env python3
"""Exact low-level tests for Ramond double-Virasoro primary enumeration."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ramond_branching as branch  # noqa: E402
import enumerate_ramond_double_virasoro_primaries as primaries  # noqa: E402


def component_map(label, parity):
    _, components = primaries.primary_components(label, parity)
    return {
        component[:5]: component[5]
        for component in components
    }


class RamondPrimaryEnumerationTests(unittest.TestCase):
    def assert_expression_equal(self, left, right):
        self.assertEqual(sp.cancel(left - right), 0)

    def test_component_counts(self):
        for sign in (-1, 1):
            for parity in (0, 1):
                self.assertEqual(
                    len(component_map(sign * sp.Rational(1, 4), parity)), 2
                )
                self.assertEqual(
                    len(component_map(sign * sp.Rational(3, 4), parity)), 6
                )

    def test_onset_states(self):
        for sigma in (-1, 1):
            odd = component_map(sigma * sp.Rational(1, 4), 1)
            even = component_map(sigma * sp.Rational(1, 4), 0)
            self.assert_expression_equal(odd[((), 1, (), (), 0)], 1 / sp.sqrt(2))
            self.assert_expression_equal(
                odd[((), 0, (), (), 1)], -sp.I * sigma / sp.sqrt(2)
            )
            self.assert_expression_equal(even[((), 0, (), (), 0)], 1)
            self.assert_expression_equal(even[((), 1, (), (), 1)], sp.I * sigma)

    def test_three_quarters_unified_formulas(self):
        for sigma in (-1, 1):
            denominator = (
                4 * branch.P**2
                + 6 * sigma * branch.P * branch.Q
                + 2 * branch.Q**2
                + 1
            )
            shifted = branch.Q + 2 * sigma * branch.P
            even_expected = {
                ((1,), 1, (), (), 0): -1 / sp.sqrt(2),
                ((1,), 0, (), (), 1): sp.I * sigma / sp.sqrt(2),
                ((), 0, (1,), (), 0): 2 / denominator,
                ((), 1, (1,), (), 1): -2 * sp.I * sigma / denominator,
                ((), 1, (), (1,), 0): sp.sqrt(2) * shifted / denominator,
                ((), 0, (), (1,), 1): (
                    sp.I * sigma * sp.sqrt(2) * shifted / denominator
                ),
            }
            odd_expected = {
                ((1,), 0, (), (), 0): -1,
                ((1,), 1, (), (), 1): -sp.I * sigma,
                ((), 0, (1,), (), 1): (
                    2 * sp.sqrt(2) * sp.I * sigma / denominator
                ),
                ((), 1, (1,), (), 0): 2 * sp.sqrt(2) / denominator,
                ((), 0, (), (1,), 0): -2 * shifted / denominator,
                ((), 1, (), (1,), 1): (
                    2 * sp.I * sigma * shifted / denominator
                ),
            }
            for calculated, expected in (
                (component_map(sigma * sp.Rational(3, 4), 0), even_expected),
                (component_map(sigma * sp.Rational(3, 4), 1), odd_expected),
            ):
                self.assertEqual(set(calculated), set(expected))
                for state, coefficient in expected.items():
                    self.assert_expression_equal(calculated[state], coefficient)


if __name__ == "__main__":
    unittest.main()
