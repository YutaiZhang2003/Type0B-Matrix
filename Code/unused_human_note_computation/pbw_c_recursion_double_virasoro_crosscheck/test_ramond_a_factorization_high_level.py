"""Regression checks for higher-level Ramond A_rs and factorization."""

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
    RamondPBWModule,
    clean,
    contract_ramond_null,
    fixed_beta_inverse_null_norm,
    inverse_null_product_59,
    ramond_degenerate_data,
    ramond_labels_at_level,
)


class RamondAAndFactorizationHighLevelTest(unittest.TestCase):
    def setUp(self):
        self.b = sp.symbols("b", nonzero=True)
        self.lambda_i, self.beta_j = sp.symbols("lambda_i beta_j", nonzero=True)

    def assertExactZero(self, expression):
        self.assertEqual(clean(expression), 0)

    def test_a_rs_symbolic_even_lattice_through_level_three(self):
        for level in range(1, 4):
            for r, s in ramond_labels_at_level(level):
                data = ramond_degenerate_data(r, s, self.b)
                module = RamondPBWModule(data["h"], data["beta"], data["c"])
                chi_plus = module.normalized_null_vector(level, 0)
                direct = fixed_beta_inverse_null_norm(r, s, self.b, chi_plus)
                self.assertExactZero(
                    direct
                    - inverse_null_product_59(r, s, self.b, lattice="even")
                )
                self.assertNotEqual(
                    clean(
                        direct
                        - inverse_null_product_59(
                            r, s, self.b, lattice="literal"
                        )
                    ),
                    0,
                )

    def test_a_rs_null_doublet_phase_through_level_three(self):
        for level in range(1, 4):
            # One representative per level is enough for the phase
            # regression; the plus kernels above cover every label.
            r, s = ramond_labels_at_level(level)[0]
            data = ramond_degenerate_data(r, s, self.b)
            module = RamondPBWModule(data["h"], data["beta"], data["c"])
            plus = fixed_beta_inverse_null_norm(
                r, s, self.b, module.normalized_null_vector(level, 0)
            )
            minus = fixed_beta_inverse_null_norm(
                r, s, self.b, module.normalized_null_vector(level, 1)
            )
            self.assertExactZero(minus + sp.I * plus)

    def test_full_component_factorization_symbolically_through_level_two(self):
        for level in (1, 2):
            for r, s in ramond_labels_at_level(level):
                for null_slot, null_ground, spectator_ground, p_phi, eta in product(
                    (2, 3), (0, 1), (0, 1), (0, 1), (1, -1)
                ):
                    result = contract_ramond_null(
                        r=r,
                        s=s,
                        p_phi=p_phi,
                        eta=eta,
                        b=self.b,
                        lambda_i=self.lambda_i,
                        beta_j=self.beta_j,
                        null_ground=null_ground,
                        spectator_ground=spectator_ground,
                        null_slot=null_slot,
                    )
                    self.assertExactZero(result["generalized_residual"])

    def test_level_three_factorization_all_labels_and_primary_parities(self):
        # The null/spectator ground phases are already exhausted above.
        # At level three, cover every label and both p_phi,eta choices on the
        # ++ component to keep the routine test time moderate.
        for r, s in ramond_labels_at_level(3):
            for null_slot, p_phi, eta in product((2, 3), (0, 1), (1, -1)):
                result = contract_ramond_null(
                    r=r,
                    s=s,
                    p_phi=p_phi,
                    eta=eta,
                    b=self.b,
                    lambda_i=self.lambda_i,
                    beta_j=self.beta_j,
                    null_slot=null_slot,
                )
                self.assertExactZero(result["generalized_residual"])


if __name__ == "__main__":
    unittest.main()
