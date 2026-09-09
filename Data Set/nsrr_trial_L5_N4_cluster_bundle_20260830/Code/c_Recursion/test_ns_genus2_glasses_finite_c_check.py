"""Regressions for the independent finite-c glasses sewing oracle."""

from __future__ import annotations

import unittest

from ns_genus12_finite_c_check import level_tuples
from ns_genus2_glasses_finite_c_check import (
    DirectGlassesOracle,
    global_glasses_coefficient,
)


class DirectGlassesOracleTests(unittest.TestCase):
    def test_global_subspace_matches_analytic_network(self) -> None:
        weights = (0.73, 0.91, 1.17)
        oracle = DirectGlassesOracle(c=37.25, weights=weights)
        errors = []
        for levels in level_tuples(2):
            sector = levels[2] % 2
            direct = oracle.coefficient(levels, sector, (1, 1, 1))
            analytic = global_glasses_coefficient(
                weights=weights,
                twice_levels=levels,
                sector=sector,
                lifts=(1, 1, 1),
            )
            errors.append(abs(direct - analytic))
            self.assertEqual(
                oracle.coefficient(levels, sector ^ 1, (1, 1, 1)),
                0.0 + 0.0j,
            )
        self.assertLess(max(errors), 1.0e-14)

    def test_odd_handle_lifts_are_local(self) -> None:
        oracle = DirectGlassesOracle(c=37.25, weights=(0.73, 0.91, 1.17))
        levels = (1, 1, 0)
        reference = oracle.coefficient(levels, 0, (1, 1, 1))
        self.assertEqual(
            oracle.coefficient(levels, 0, (-1, 1, 1)), -reference
        )
        self.assertEqual(
            oracle.coefficient(levels, 0, (-1, -1, 1)), reference
        )


if __name__ == "__main__":
    unittest.main()
