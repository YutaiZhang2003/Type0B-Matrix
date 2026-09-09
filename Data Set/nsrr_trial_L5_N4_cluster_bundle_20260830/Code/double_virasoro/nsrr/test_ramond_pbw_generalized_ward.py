"""Tests for the physical-basis Ramond PBW and generalized Ward audit."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest

import sympy as sp


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ramond_pbw_generalized_ward import (  # noqa: E402
    GeneralizedNRRWard,
    RamondPBWModule,
    RamondState,
    clean,
    contract_level_one_null,
    fixed_beta_inverse_null_norm,
    fusion_polynomial_510,
    inverse_null_product_59,
    pole_equation_residual,
    ramond_degenerate_data,
    ramond_labels_at_level,
)


class RamondPBWGeneralizedWardTest(unittest.TestCase):
    def setUp(self):
        self.b = sp.symbols("b", nonzero=True)
        self.lambda_i, self.beta_j = sp.symbols("lambda_i beta_j")

    def assertExactZero(self, expression):
        self.assertEqual(clean(expression), 0)

    def test_53_normalization_and_primary_parity_shift(self):
        for second, third in product((0, 1), repeat=2):
            self.assertEqual(
                GeneralizedNRRWard.component_normalization(second, third), 1
            )

        pole = ramond_degenerate_data(2, 1, self.b)
        epsilon = []
        for p_phi in (0, 1):
            ward = GeneralizedNRRWard(
                p_phi=p_phi,
                form_parity=p_phi,
                eta=1,
                h_ns=sp.Symbol("h_i"),
                h_second=pole["h"],
                h_third=sp.Symbol("h_j"),
                beta_second=pole["beta"],
                beta_third=self.beta_j,
                central_charge=pole["c"],
            )
            self.assertEqual(ward.ground_value(0, 0), 1)
            epsilon.append(ward.epsilon((), (), 0))
        self.assertExactZero(epsilon[0] + epsilon[1])

    def test_both_generalized_ward_equations_with_both_primary_parities(self):
        h_i, h_2, h_3, beta_2, beta_3, c = sp.symbols(
            "h_i h_2 h_3 beta_2 beta_3 c"
        )
        for p_phi, eta, second, third in product(
            (0, 1), (1, -1), (0, 1), (0, 1)
        ):
            # Every term in either Ward equation has one additional G, so
            # this is the nonzero fixed-form sector for ground input states.
            form_parity = (p_phi + second + third + 1) % 2
            ward = GeneralizedNRRWard(
                p_phi=p_phi,
                form_parity=form_parity,
                eta=eta,
                h_ns=h_i,
                h_second=h_2,
                h_third=h_3,
                beta_second=beta_2,
                beta_third=beta_3,
                central_charge=c,
            )
            self.assertExactZero(
                ward.ward_first_residual(0, (), (), second, (), third)
            )
            self.assertExactZero(
                ward.ward_second_residual(1, (), (), second, (), third)
            )

    def test_56_57_degeneracy_and_shift(self):
        for r, s in ((2, 1), (1, 2), (4, 1), (1, 4)):
            pole = ramond_degenerate_data(r, s, self.b)
            self.assertExactZero(pole_equation_residual(r, s, self.b))
            self.assertExactZero(
                pole["h_shifted"]
                - pole["h"]
                - sp.Rational(r * s, 2)
            )

    def test_physical_ground_metric_and_g0(self):
        beta, h, c = sp.symbols("beta h c")
        module = RamondPBWModule(h, beta, c)
        plus_coefficient, plus_ground = module.g0_action(0)
        minus_coefficient, minus_ground = module.g0_action(1)
        self.assertEqual((plus_ground, minus_ground), (1, 0))
        self.assertExactZero(plus_coefficient * minus_coefficient + beta**2)
        self.assertEqual(module.ground_pairing(0, 0), 1)
        self.assertEqual(module.ground_pairing(1, 1), sp.I)
        self.assertEqual(module.ground_pairing(0, 1), 0)

    def test_level_one_gram_null_doublet(self):
        for r, s in ((2, 1), (1, 2)):
            pole = ramond_degenerate_data(r, s, self.b)
            module = RamondPBWModule(pole["h"], pole["beta"], pole["c"])
            for parity in (0, 1):
                basis, gram = module.gram_matrix(1, parity)
                null = module.normalized_null_vector(1, parity)
                leading = RamondState((("L", -1),), parity)
                self.assertEqual(null[leading], 1)
                vector = sp.Matrix([null.get(state, 0) for state in basis])
                for residual in gram * vector:
                    self.assertExactZero(residual)

    def test_symbolic_gram_null_doublets_through_level_three(self):
        for level in range(1, 4):
            for r, s in ramond_labels_at_level(level):
                pole = ramond_degenerate_data(r, s, self.b)
                module = RamondPBWModule(
                    pole["h"], pole["beta"], pole["c"]
                )
                for parity in (0, 1):
                    basis, gram = module.gram_matrix(level, parity)
                    _, kernel = module.gram_kernel(level, parity)
                    self.assertEqual(len(kernel), 1)
                    null = module.normalized_null_vector(level, parity)
                    leading = RamondState(
                        tuple(("L", -1) for _ in range(level)), parity
                    )
                    self.assertEqual(null[leading], 1)
                    column = sp.Matrix([null.get(state, 0) for state in basis])
                    for residual in gram * column:
                        self.assertExactZero(residual)

    def test_exact_sampled_gram_nullity_through_level_five(self):
        sample_b = sp.Rational(2, 3)
        for level in (4, 5):
            for r, s in ramond_labels_at_level(level):
                pole = ramond_degenerate_data(r, s, sample_b)
                module = RamondPBWModule(
                    pole["h"], pole["beta"], pole["c"]
                )
                for parity in (0, 1):
                    basis, gram = module.gram_matrix(level, parity)
                    _, kernel = module.gram_kernel(level, parity)
                    self.assertEqual(len(kernel), 1)
                    null = module.normalized_null_vector(level, parity)
                    column = sp.Matrix([null.get(state, 0) for state in basis])
                    for residual in gram * column:
                        self.assertExactZero(residual)

    def test_59_direct_norm_selects_even_sublattice(self):
        for r, s in ((2, 1), (1, 2)):
            pole = ramond_degenerate_data(r, s, self.b)
            module = RamondPBWModule(pole["h"], pole["beta"], pole["c"])
            null = module.normalized_null_vector(1, 0)
            direct = fixed_beta_inverse_null_norm(r, s, self.b, null)
            self.assertExactZero(
                direct - inverse_null_product_59(r, s, self.b, lattice="even")
            )
            self.assertNotEqual(
                clean(
                    direct
                    - inverse_null_product_59(r, s, self.b, lattice="literal")
                ),
                0,
            )
            self.assertNotEqual(
                clean(direct - inverse_null_product_59(r, s, self.b, lattice="odd")),
                0,
            )

    def test_510_generalized_ward_result_is_recorded_faithfully(self):
        for r, s in ((2, 1), (1, 2)):
            for p_phi, eta in product((0, 1), (1, -1)):
                result = contract_level_one_null(
                    r=r,
                    s=s,
                    p_phi=p_phi,
                    eta=eta,
                    b=self.b,
                    lambda_i=self.lambda_i,
                    beta_j=self.beta_j,
                )
                parity_corrected = fusion_polynomial_510(
                    r,
                    s,
                    self.lambda_i,
                    self.beta_j,
                    self.b,
                    (-1) ** p_phi * eta,
                )
                self.assertExactZero(result["direct"] + parity_corrected)
                # The printed formula has neither the plane-coordinate minus
                # sign nor the intrinsic-primary eta flip, so it must not be
                # silently accepted by this literal generalized-Ward audit.
                self.assertNotEqual(result["residual"], 0)


if __name__ == "__main__":
    unittest.main()
