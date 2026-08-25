"""Convention and regression tests for four-R sphere blocks."""

import math
import unittest

from ramond_sphere_blocks import (
    RamondExternalSphereFourPointBlock,
    b_from_c,
    ramond_beta,
    ramond_g0_matrix,
    ramond_liouville_weight,
)
from superconformal_blocks import central_charge


class RamondExternalSphereFourPointBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = 13.5 + 1.0e-5
        cls.b = b_from_c(cls.c)
        cls.block = RamondExternalSphereFourPointBlock.from_liouville_momenta(
            p1=1.0 / 2.0,
            p2=1.0 / 3.0,
            p3=1.0 / 4.0,
            p4=3.0 / 5.0,
            internal_momentum=math.sqrt(6.0 / 5.0),
            b=cls.b,
            sign3=1,
            sign2=1,
        )

    def test_bry_hjs_convention_dictionary(self):
        self.assertAlmostEqual(central_charge(self.b), self.c, places=12)
        beta = ramond_beta(3.0 / 5.0)
        expected_weight = self.c / 24.0 + 0.5 * (3.0 / 5.0) ** 2
        self.assertAlmostEqual(
            self.c / 24.0 - beta * beta,
            expected_weight,
            places=13,
        )
        self.assertAlmostEqual(
            ramond_liouville_weight(3.0 / 5.0, self.b),
            expected_weight,
            places=13,
        )

        g0 = ramond_g0_matrix(beta)
        g0_squared_00 = g0[0][0] * g0[0][0] + g0[0][1] * g0[1][0]
        g0_squared_11 = g0[1][0] * g0[0][1] + g0[1][1] * g0[1][1]
        self.assertAlmostEqual(g0_squared_00, -beta * beta, places=14)
        self.assertAlmostEqual(g0_squared_11, -beta * beta, places=14)

    def test_level_half_fusion_polynomial_and_residue_phase(self):
        block = self.block
        left = block.fusion_polynomial(
            1,
            1,
            lower_beta=block.beta4,
            upper_beta=block.beta3,
            upper_sign=block.sign3,
        )
        right = block.fusion_polynomial(
            1,
            1,
            lower_beta=block.beta1,
            upper_beta=block.beta2,
            upper_sign=block.sign2,
        )
        self.assertAlmostEqual(left, block.beta4 - block.beta3, places=14)
        self.assertAlmostEqual(right, block.beta1 - block.beta2, places=14)

        direct = block.direct_leading_coefficients()["odd_level_half"]
        q_half = block.elliptic_coefficients(1, "odd")[1]
        # q ~ z/16, hence H_{1/2} q^{1/2} contributes
        # (H_{1/2}/4) z^{1/2} to the local block.
        self.assertAlmostEqual(q_half / 4.0, direct, places=13)

    def test_elliptic_recursion_matches_direct_descendants(self):
        block = self.block
        direct = block.direct_leading_coefficients()
        z = 1.0e-5
        leading_power = block.internal_weight - block.h2 - block.h1

        even = block.elliptic_block(z, 8, "even")
        odd = block.elliptic_block(z, 8, "odd")
        extracted_even_one = (even / z**leading_power - 1.0) / z
        extracted_odd_half = odd / z ** (leading_power + 0.5)

        self.assertLess(
            abs(extracted_even_one - direct["even_level_1"]),
            4.0e-6,
        )
        self.assertLess(
            abs(extracted_odd_half - direct["odd_level_half"]),
            3.0e-7,
        )

    def test_bry_regulated_numerical_anchor(self):
        z = 1.0 / 3.0 + 3.0j / 5.0
        even = self.block.elliptic_block(z, 8, "even")
        odd = self.block.elliptic_block(z, 8, "odd")
        self.assertAlmostEqual(
            even,
            1.1208467964939917 + 0.12338733150859268j,
            places=11,
        )
        self.assertAlmostEqual(
            odd,
            0.0045330605547722865 + 0.014809671120852226j,
            places=12,
        )

        even_product = self.block.diagonal_block_product(z, 8, "even")
        odd_product = self.block.diagonal_block_product(z, 8, "odd")
        self.assertGreater(even_product.real, 0.0)
        self.assertGreater(odd_product.real, 0.0)
        self.assertLess(abs(even_product.imag), 2.0e-12)
        self.assertLess(abs(odd_product.imag), 2.0e-12)

    def test_q6_to_q8_is_stable_for_all_sign_branches(self):
        z = 1.0 / 3.0 + 3.0j / 5.0
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                block = RamondExternalSphereFourPointBlock.from_liouville_momenta(
                    p1=1.0 / 2.0,
                    p2=1.0 / 3.0,
                    p3=1.0 / 4.0,
                    p4=3.0 / 5.0,
                    internal_momentum=math.sqrt(6.0 / 5.0),
                    b=self.b,
                    sign3=sign3,
                    sign2=sign2,
                )
                for parity in ("even", "odd"):
                    value6 = block.elliptic_block(z, 6, parity)
                    value8 = block.elliptic_block(z, 8, parity)
                    self.assertLess(abs(value8 - value6), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
