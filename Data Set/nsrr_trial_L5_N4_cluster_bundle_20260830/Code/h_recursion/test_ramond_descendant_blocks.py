"""Direct descendant-sum tests for the generic long-R sphere block."""

import unittest

from compare_ramond_large_c_level5 import (
    FixedData,
    direct_local_coefficients,
    stable_normalized_elliptic_coefficients,
    strip_universal_power,
)
from mixed_ramond_sphere_blocks import MixedRExchangeSphereFourPointBlock
from ramond_descendant_blocks import BruteForceMixedRExchangeSphereBlock
from ramond_fixed_beta_c_recursion import FixedBetaRExchangeSphereFourPointBlock
from self_dual_superconformal_blocks import (
    SelfDualMixedRExchangeSphereFourPointBlock,
)


PARAMETERS = dict(
    p1_ns=0.31,
    p2_r=0.41,
    p3_r=0.23,
    p4_ns=0.37,
    internal_momentum=0.70,
)


class RamondDescendantBlockTests(unittest.TestCase):
    def test_level_one_gram_and_ward_vectors(self):
        brute = BruteForceMixedRExchangeSphereBlock(
            b=1.0, sign3=1, sign2=-1, **PARAMETERS
        )
        reference = MixedRExchangeSphereFourPointBlock(
            b=1.0,
            sign3=1,
            sign2=-1,
            **PARAMETERS,
        )

        gram = brute.module.gram_matrix(1, 0)
        h = brute.internal_weight
        kappa_squared = h - brute.c / 24.0
        expected_gram = (
            (2.0 * h, 1.5 * kappa_squared),
            (
                1.5 * kappa_squared,
                kappa_squared * (2.0 * h + brute.c / 4.0),
            ),
        )
        for row in range(2):
            for column in range(2):
                self.assertAlmostEqual(
                    gram[row][column],
                    expected_gram[row][column],
                    places=13,
                )

        expected_left = (
            h + brute.h3 - brute.h4,
            -0.5 * brute.internal_beta**2
            - brute.internal_beta * brute.beta3,
        )
        expected_right = (
            h + brute.h2 - brute.h1,
            -0.5 * brute.internal_beta**2
            + brute.internal_beta * brute.beta2,
        )
        for actual, expected in zip(brute.left.vector(1), expected_left):
            self.assertAlmostEqual(actual, expected, places=13)
        for actual, expected in zip(brute.right.vector(1), expected_right):
            self.assertAlmostEqual(actual, expected, places=13)

        self.assertAlmostEqual(
            brute.local_coefficients(1)[1],
            reference.direct_level_one_coefficient(),
            places=12,
        )

    def test_pbw_dimensions_and_gram_symmetry_through_level_three(self):
        brute = BruteForceMixedRExchangeSphereBlock(
            b=1.0, sign3=1, sign2=1, **PARAMETERS
        )
        self.assertEqual(
            [len(brute.module.basis(level, 0)) for level in range(4)],
            [1, 2, 4, 8],
        )
        for level in range(4):
            gram = brute.module.gram_matrix(level, 0)
            for row in range(len(gram)):
                for column in range(len(gram)):
                    self.assertAlmostEqual(
                        gram[row][column],
                        gram[column][row],
                        places=12,
                    )

    def test_all_sign_components_match_exact_recursion_through_q5(self):
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                with self.subTest(sign3=sign3, sign2=sign2):
                    direct = BruteForceMixedRExchangeSphereBlock(
                        b=1.0,
                        sign3=sign3,
                        sign2=sign2,
                        **PARAMETERS,
                    ).elliptic_coefficients(5)
                    recursive = SelfDualMixedRExchangeSphereFourPointBlock(
                        sign3=sign3,
                        sign2=sign2,
                        **PARAMETERS,
                    ).elliptic_coefficients(6)
                    for power in range(6):
                        self.assertLess(
                            abs(direct[power] - recursive[power]),
                            5.0e-9,
                        )

    def test_fixed_beta_c_recursion_matches_direct_sewing_through_q5(self):
        base = BruteForceMixedRExchangeSphereBlock(
            b=1.0, sign3=1, sign2=1, **PARAMETERS
        )
        data = FixedData(
            beta=base.internal_beta,
            beta2=base.beta2,
            beta3=base.beta3,
            h1=base.h1,
            h4=base.h4,
        )
        for c in (20.0, 60.0):
            for sign3 in (1, -1):
                for sign2 in (1, -1):
                    with self.subTest(c=c, sign3=sign3, sign2=sign2):
                        local = direct_local_coefficients(
                            c, data, sign3, sign2
                        )
                        direct = stable_normalized_elliptic_coefficients(
                            strip_universal_power(local, c), data
                        )
                        recursive = FixedBetaRExchangeSphereFourPointBlock(
                            c=c,
                            h1_ns=data.h1,
                            beta2_r=data.beta2,
                            beta3_r=data.beta3,
                            h4_ns=data.h4,
                            internal_beta=data.beta,
                            sign3=sign3,
                            sign2=sign2,
                        )
                        for power in range(6):
                            self.assertLess(
                                abs(direct[power] - recursive.coefficient(power)),
                                2.0e-6,
                            )


if __name__ == "__main__":
    unittest.main()
