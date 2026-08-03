#!/usr/bin/env python3
"""Regression checks for the first genus-two NS partition experiment."""

from __future__ import annotations

import cmath
from itertools import product
import unittest

from ns_genus2_partition import (
    GLASSES_ORIENTATION,
    _free_superfield_chiral_log,
    _theta_schottky_data,
    direct_global_block,
    run_internal_checks,
)
from ns_vacuum_schottky import ccy_theta_generators, theta_lift_signs
from plumbing_algorithms import generators_for_theta


class GenusTwoNSPartitionTests(unittest.TestCase):
    @staticmethod
    def _projective_error(left, right) -> float:
        left_entries = (left.a, left.b, left.c, left.d)
        right_entries = (right.a, right.b, right.c, right.d)
        pivot = max(range(4), key=lambda index: abs(right_entries[index]))
        scale = left_entries[pivot] / right_entries[pivot]
        return max(
            abs(a - scale * b) for a, b in zip(left_entries, right_entries)
        ) / max(1.0, *(abs(value) for value in left_entries))

    def test_orientation_polynomial(self) -> None:
        self.assertEqual(GLASSES_ORIENTATION.edge_linear_bits, (0, 0, 0))
        for left in (0, 1):
            for right in (0, 1):
                for bridge in (0, 1):
                    self.assertEqual(
                        GLASSES_ORIENTATION.exponent((left, right, bridge)),
                        bridge * (left + right) % 2,
                    )

    def test_separating_and_spin_checks(self) -> None:
        checks = run_internal_checks()
        self.assertLess(checks["separating_global_relative_error"], 2.0e-10)
        self.assertLess(checks["handle_residue_torus_relative_error"], 2.0e-12)
        self.assertEqual(
            checks["spin_target_characteristic"],
            {"alpha": [0, 0], "beta": [1, 1]},
        )

    def test_odd_glasses_sector_starts_on_bridge(self) -> None:
        result = direct_global_block(
            channel="glasses",
            weights=(0.71, 0.83, 0.64),
            q_values=(0.12, 0.14, 0.0),
            sector=1,
            lifts=(1, 1, 1),
            tolerance=1.0e-12,
            max_total_occupation=10,
        )
        self.assertEqual(result.value, 0.0j)

    def test_theta_schottky_marking_matches_period_coordinates(self) -> None:
        q_values = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
        edge_lifts = (1, -1, -1)
        generators, signs = _theta_schottky_data(q_values, edge_lifts)
        expected = generators_for_theta(*q_values)
        self.assertEqual(signs, (-1, -1))
        for observed, target in zip(generators, expected):
            self.assertLess(
                self._projective_error(observed.gamma, target.gamma), 1.0e-14
            )

        # The same marked surface in the CCY frame swaps q_0 and q_infinity
        # and reverses the second Schottky generator.
        ccy = ccy_theta_generators(q_values[2], q_values[1], q_values[0])
        self.assertLess(
            self._projective_error(generators[0].gamma, ccy[0].gamma), 1.0e-14
        )
        self.assertLess(
            self._projective_error(generators[1].gamma, ccy[1].gamma.inv()),
            1.0e-14,
        )

    def test_theta_free_product_is_identical_in_both_markings(self) -> None:
        q_values = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
        ccy_generators = ccy_theta_generators(
            q_values[2], q_values[1], q_values[0]
        )
        for edge_lifts in product((-1, 1), repeat=3):
            generators, signs = _theta_schottky_data(q_values, edge_lifts)
            period_log, _ = _free_superfield_chiral_log(
                generators, signs, max_word_length=5, max_mode=30
            )

            swapped_lifts = (edge_lifts[2], edge_lifts[1], edge_lifts[0])
            ccy_log, _ = _free_superfield_chiral_log(
                ccy_generators,
                theta_lift_signs(swapped_lifts),
                max_word_length=5,
                max_mode=30,
            )
            self.assertLess(abs(period_log - ccy_log), 2.0e-13)

        # Lock against the original erroneous unswapped call.  This is a
        # genuine surface mismatch, not a numerically invisible refactoring.
        edge_lifts = (1, -1, -1)
        generators, signs = _theta_schottky_data(q_values, edge_lifts)
        period_log, _ = _free_superfield_chiral_log(
            generators, signs, max_word_length=5, max_mode=30
        )
        wrong_log, _ = _free_superfield_chiral_log(
            ccy_theta_generators(*q_values),
            theta_lift_signs(edge_lifts),
            max_word_length=5,
            max_mode=30,
        )
        self.assertGreater(abs(cmath.exp(period_log - wrong_log) - 1.0), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
