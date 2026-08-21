#!/usr/bin/env python3
"""Checks for the independent genus-two Majorana plumbing oracle."""

from __future__ import annotations

import math
import unittest

from free_majorana_pair_of_pants import (
    glasses_majorana_plumbing_partition,
    majorana_three_point,
    ns_fermion_states_at_twice_level,
    theta_majorana_plumbing_partition,
)


class FreeMajoranaPairOfPantsTests(unittest.TestCase):
    def test_bpz_dual_is_orthonormal(self) -> None:
        states = [
            state
            for level in range(13)
            for state in ns_fermion_states_at_twice_level(level)
        ]
        for left in states:
            for right in states:
                expected = int(left == right)
                self.assertEqual(majorana_three_point(left, (), right), expected)

    def test_elementary_sphere_coefficients(self) -> None:
        self.assertEqual(majorana_three_point((1,), (1,), ()), 1)
        self.assertEqual(majorana_three_point((), (1,), (1,)), 1)
        self.assertEqual(majorana_three_point((), (2,), (1,)), -1)
        self.assertEqual(majorana_three_point((2,), (1,), ()), 1)

    def test_glasses_separating_limit_is_two_ns_characters(self) -> None:
        q_left, q_right = 0.07, 0.11
        cutoff = 28
        result = glasses_majorana_plumbing_partition(
            q_left,
            q_right,
            0.0,
            max_total_twice_level=cutoff,
        )
        expected = 1.0
        for mode in range(20):
            expected *= (1.0 + q_left ** (mode + 0.5))
            expected *= (1.0 + q_right ** (mode + 0.5))
        self.assertLess(abs(result.chiral_value - expected) / expected, 2.0e-11)

    def test_theta_vacuum_and_parity_selection(self) -> None:
        result = theta_majorana_plumbing_partition(
            0.0,
            0.0,
            0.0,
            max_total_twice_level=8,
        )
        self.assertEqual(result.chiral_value, 1.0)
        self.assertTrue(math.isfinite(result.chiral_value.real))

    def test_direct_wick_selection_and_low_coefficients(self) -> None:
        # These are coefficients of the defining Fock-space sewing sum,
        # before any plumbing orientation sign.  No theta constant,
        # bosonization identity, or Schottky product enters this check.
        expected_nonzero = {
            (0, 0, 0): 1,
            (0, 1, 1): 1,
            (1, 0, 1): 1,
            (1, 1, 0): 1,
            (0, 1, 3): 1,
            (0, 3, 1): 1,
            (3, 1, 0): 1,
        }
        for levels in (
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 1, 3),
            (0, 3, 1),
            (3, 1, 0),
        ):
            states = [ns_fermion_states_at_twice_level(level) for level in levels]
            coefficient = sum(
                majorana_three_point(bra, middle, ket) ** 2
                for bra in states[0]
                for middle in states[1]
                for ket in states[2]
            )
            self.assertEqual(coefficient, expected_nonzero[levels], levels)

        # A Wick correlator with odd total fermion number vanishes.
        for levels in ((1, 0, 0), (1, 1, 1), (0, 2, 1), (3, 0, 0)):
            states = [ns_fermion_states_at_twice_level(level) for level in levels]
            coefficient = sum(
                majorana_three_point(bra, middle, ket) ** 2
                for bra in states[0]
                for middle in states[1]
                for ket in states[2]
            )
            self.assertEqual(coefficient, 0, levels)


if __name__ == "__main__":
    unittest.main()
