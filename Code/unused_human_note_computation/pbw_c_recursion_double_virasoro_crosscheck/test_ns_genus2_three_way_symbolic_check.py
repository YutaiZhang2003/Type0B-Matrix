"""Regression tests for the exact three-way all-NS theta-block audit."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import sympy as sp


CODE_DIR = Path(__file__).resolve().parent
PYTHON_DIR = CODE_DIR.parent / "genus_2_cross_channel"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from ccy_genus2_block import (  # noqa: E402
    c_rs_from_h,
    ccy_residue_prefactor_for_weights,
)
from ns_genus2_three_way_symbolic_check import (  # noqa: E402
    DEFAULT_LEVEL_TWO_SAMPLES,
    P0,
    graded_gram_extra_exponent,
    divide_multivariate_series,
    ns_weight,
    run_checks,
    run_level_two_exact_samples,
    two_virasoro_parameters,
    theta_cross_exponent,
    theta_quadratic_exponent,
    virasoro_21_pole_and_residue,
)
from ns_genus2_symbolic_low_order import (  # noqa: E402
    C,
    H0,
    H1,
    HINF,
    ExactDirectThetaOracle,
)


class NSGenus2ThreeWaySymbolicTests(unittest.TestCase):
    def test_direct_sewing_uses_literal_human_note_sign(self) -> None:
        direct = ExactDirectThetaOracle(c=C, weights=(H0, H1, HINF))

        one_fermion = {
            (1, 0, 0): 1 / (2 * H0),
            (0, 1, 0): 1 / (2 * H1),
            (0, 0, 1): 1 / (2 * HINF),
        }
        for levels, expected in one_fermion.items():
            self.assertEqual(sp.cancel(direct.coefficient(levels) - expected), 0)

        two_fermion = {
            (1, 1, 0): -(H0 + H1 - HINF) ** 2 / (4 * H0 * H1),
            (1, 0, 1): -(H0 - H1 + HINF) ** 2 / (4 * H0 * HINF),
            (0, 1, 1): -(H0 - H1 - HINF) ** 2 / (4 * H1 * HINF),
        }
        for levels, expected in two_fermion.items():
            self.assertEqual(sp.cancel(direct.coefficient(levels) - expected), 0)

    def test_branch_weight_shift(self) -> None:
        for label in range(-2, 3):
            parameters = two_virasoro_parameters(P0, label)
            observed = parameters[0][1] + parameters[1][1]
            expected = ns_weight(P0) + sp.Rational(label * label, 2)
            self.assertEqual(sp.cancel(observed - expected), 0)

    def test_level_one_double_virasoro_gram_factorization(self) -> None:
        b = sp.Symbol("b", nonzero=True)
        momentum = sp.Symbol("P", nonzero=True)
        q = b + 1 / b
        weight = q**2 / 8 - momentum**2 / 2
        denominator = 1 / b - b

        # Columns are L_-1^(1)v_0 and L_-1^(2)v_0 in the ordered product
        # basis X=1 tensor L_-1 phi, Y=psi_-1/2 tensor G_-1/2 phi.
        change_of_basis = sp.Matrix(
            [
                [(1 / b) / denominator, -b / denominator],
                [1 / denominator, -1 / denominator],
            ]
        )
        shapovalov_product_gram = sp.diag(2 * weight, -2 * weight)
        graded_tensor_gram = sp.diag(2 * weight, 2 * weight)

        shapovalov_double_virasoro = sp.simplify(
            change_of_basis.T
            * shapovalov_product_gram
            * change_of_basis
        )
        graded_double_virasoro = sp.simplify(
            change_of_basis.T * graded_tensor_gram * change_of_basis
        )

        parameters = two_virasoro_parameters(momentum, 0, b)
        expected_factorized = sp.diag(
            2 * parameters[0][1], 2 * parameters[1][1]
        )
        for entry in shapovalov_double_virasoro - expected_factorized:
            self.assertEqual(sp.cancel(entry), 0)

        expected_graded = (
            2
            * weight
            / denominator**2
            * sp.Matrix([[b ** (-2) + 1, -2], [-2, b**2 + 1]])
        )
        for entry in graded_double_virasoro - expected_graded:
            self.assertEqual(sp.cancel(entry), 0)
        self.assertNotEqual(sp.cancel(graded_double_virasoro[0, 1]), 0)

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

    def test_graded_gram_cancels_theta_polarization(self) -> None:
        for primaries in product((0, 1), repeat=3):
            for sca in product((0, 1), repeat=3):
                for fermion in product((0, 1), repeat=3):
                    if sum(fermion) % 2:
                        continue
                    absolute_sca = tuple(
                        sca[edge] + primaries[edge] for edge in range(3)
                    )
                    total = tuple(
                        absolute_sca[edge] + fermion[edge]
                        for edge in range(3)
                    )
                    raw_graded = (
                        theta_quadratic_exponent(total)
                        + graded_gram_extra_exponent(
                            sca, fermion, primaries
                        )
                    ) % 2
                    separate = (
                        theta_quadratic_exponent(absolute_sca)
                        + theta_quadratic_exponent(fermion)
                    ) % 2
                    self.assertEqual(raw_graded, separate)
                    self.assertEqual(
                        graded_gram_extra_exponent(
                            sca, fermion, primaries
                        ),
                        theta_cross_exponent(absolute_sca, fermion),
                    )

    def test_exact_three_way_check_through_level_three_halves(self) -> None:
        summary = run_checks(max_total_twice_level=3)
        self.assertEqual(summary.coefficient_count, 20)
        self.assertEqual(summary.direct_vs_recursion_zero_count, 20)
        self.assertEqual(summary.direct_vs_two_virasoro_zero_count, 20)
        self.assertEqual(summary.double_virasoro_mismatch_count, 0)
        self.assertEqual(summary.first_double_virasoro_mismatch, "none")
        self.assertEqual(summary.ordinary_quotient_control_zero_count, 14)
        self.assertEqual(summary.ordinary_quotient_control_mismatch_count, 6)
        self.assertTrue(
            summary.first_ordinary_quotient_control_mismatch.startswith(
                "(0, 1, 2):"
            )
        )
        self.assertEqual(summary.old_hatted_vs_twisted_product_zero_count, 20)
        self.assertEqual(summary.direct_vs_twisted_two_virasoro_zero_count, 20)
        self.assertEqual(
            summary.corrected_hatted_vs_ordinary_product_zero_count, 20
        )
        self.assertEqual(
            summary.corrected_hatted_vs_old_double_virasoro_zero_count, 14
        )
        self.assertEqual(
            summary.direct_vs_corrected_gram_quotient_zero_count, 20
        )
        self.assertEqual(summary.graded_hatted_correction_count, 6)
        self.assertTrue(
            summary.first_graded_hatted_correction.startswith(
                "(0, 1, 2):"
            )
        )

    def test_level_two_mature_star_inverse_at_exact_sample(self) -> None:
        summary = run_level_two_exact_samples(
            samples=DEFAULT_LEVEL_TWO_SAMPLES[:1]
        )
        self.assertEqual(summary.coefficient_count_per_sample, 35)
        self.assertEqual(summary.top_shell_coefficient_count, 15)
        self.assertEqual(summary.exact_rational_sample_count, 1)
        self.assertEqual(summary.direct_vs_recursion_zero_count, 35)
        self.assertEqual(summary.direct_vs_two_virasoro_zero_count, 35)
        self.assertEqual(summary.double_virasoro_mismatch_count, 0)
        self.assertEqual(summary.top_shell_double_virasoro_mismatch_count, 0)

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
