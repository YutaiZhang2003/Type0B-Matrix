"""Regression tests for the exact three-way all-NS theta-block audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import sympy as sp


CODE_DIR = Path(__file__).resolve().parent
PYTHON_DIR = CODE_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from ccy_genus2_block import (  # noqa: E402
    c_rs_from_h,
    ccy_residue_prefactor_for_weights,
)
from ns_genus2_three_way_symbolic_check import (  # noqa: E402
    P0,
    divide_multivariate_series,
    ns_weight,
    run_checks,
    two_virasoro_parameters,
    virasoro_21_pole_and_residue,
)


class NSGenus2ThreeWaySymbolicTests(unittest.TestCase):
    def test_branch_weight_shift(self) -> None:
        for label in range(-2, 3):
            parameters = two_virasoro_parameters(P0, label)
            observed = parameters[0][1] + parameters[1][1]
            expected = ns_weight(P0) + sp.Rational(label * label, 2)
            self.assertEqual(sp.cancel(observed - expected), 0)

    def test_human_convention_uses_ordinary_series_division(self) -> None:
        x = sp.Symbol("x")
        numerator = {
            (0, 0, 0): sp.Integer(1),
            (0, 0, 1): x,
            (0, 1, 1): sp.Integer(1),
            (0, 1, 2): x,
        }
        denominator = {
            (0, 0, 0): sp.Integer(1),
            (0, 1, 1): sp.Integer(1),
        }
        quotient = divide_multivariate_series(
            numerator,
            denominator,
            max_total_twice_level=3,
        )
        self.assertEqual(quotient[(0, 0, 1)], x)
        self.assertEqual(quotient[(0, 1, 2)], 0)

    def test_exact_three_way_check_through_level_three_halves(self) -> None:
        summary = run_checks(max_total_twice_level=3)
        self.assertEqual(summary.coefficient_count, 20)
        self.assertEqual(summary.direct_vs_recursion_zero_count, 20)
        self.assertEqual(summary.direct_vs_two_virasoro_zero_count, 14)
        self.assertEqual(summary.double_virasoro_mismatch_count, 6)
        self.assertTrue(
            summary.first_double_virasoro_mismatch.startswith(
                "(0, 1, 2):"
            )
        )

    def test_exact_virasoro_21_kernel_matches_production_recursion(self) -> None:
        weights = (sp.Rational(7, 10), sp.Rational(9, 10), sp.Rational(6, 5))
        fusion_pairs = (
            (weights[2], weights[1]),
            (weights[2], weights[0]),
            (weights[0], weights[1]),
        )
        for edge in range(3):
            exact_pole, exact_residue = virasoro_21_pole_and_residue(
                edge=edge, weights=weights
            )
            numeric_pole = c_rs_from_h(2, 1, float(weights[edge]))
            numeric_residue = ccy_residue_prefactor_for_weights(
                2,
                1,
                float(weights[edge]),
                float(fusion_pairs[edge][0]),
                float(fusion_pairs[edge][1]),
            )
            self.assertAlmostEqual(complex(sp.N(exact_pole, 17)), numeric_pole)
            self.assertAlmostEqual(
                complex(sp.N(exact_residue, 17)), numeric_residue
            )


if __name__ == "__main__":
    unittest.main()
