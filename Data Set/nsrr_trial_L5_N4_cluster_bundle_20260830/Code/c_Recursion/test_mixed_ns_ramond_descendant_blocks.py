"""Direct Ward/Gram and regular-seed tests for the NSNSRR NS channel."""

from __future__ import annotations

import unittest

from mixed_ns_ramond_descendant_blocks import (
    BruteForceMixedNSExchangeSphereBlock,
)
from ramond_c_recursive_sphere_blocks import (
    CRecursiveMixedNSExchangeSphereFourPointBlock,
)
from ramond_c_regular_seed import DirectMixedRamondRegularSeed
from ramond_sphere_blocks import ramond_liouville_weight
from superconformal_blocks import central_charge, ns_liouville_weight


class MixedNSRamondDescendantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.b = 1.27
        cls.c = central_charge(cls.b)
        cls.momenta = (0.31, 0.41, 0.23, 0.37)
        cls.r_weights = tuple(
            ramond_liouville_weight(momentum, cls.b)
            for momentum in cls.momenta
        )
        cls.ns_weights = tuple(
            ns_liouville_weight(momentum, cls.b)
            for momentum in cls.momenta
        )
        cls.internal_weight = ns_liouville_weight(0.70, cls.b)
        cls.seed = DirectMixedRamondRegularSeed(
            h1_r=cls.r_weights[0],
            h2_r=cls.r_weights[1],
            h3_ns=cls.ns_weights[2],
            h4_ns=cls.ns_weights[3],
        )

    def direct_block(
        self, sign2: int, c: complex | None = None
    ) -> BruteForceMixedNSExchangeSphereBlock:
        return BruteForceMixedNSExchangeSphereBlock(
            c=self.c if c is None else c,
            h1_r=self.r_weights[0],
            h2_r=self.r_weights[1],
            h3_ns=self.ns_weights[2],
            h4_ns=self.ns_weights[3],
            internal_weight=self.internal_weight,
            sign2=sign2,
        )

    def test_direct_ward_gram_matches_h_recursion_through_level_13_over_2(
        self,
    ):
        for sign2 in (1, -1):
            direct = self.direct_block(sign2)
            reference = CRecursiveMixedNSExchangeSphereFourPointBlock(
                c=self.c,
                h1_r=self.r_weights[0],
                h2_r=self.r_weights[1],
                h3_ns=self.ns_weights[2],
                h4_ns=self.ns_weights[3],
                internal_weight=self.internal_weight,
                sign2=sign2,
            )
            for parity in ("even", "odd"):
                direct_values = direct.local_coefficients(7, parity)
                reference_values = reference._reference_local_coefficients(
                    c=self.c,
                    internal_weight=self.internal_weight,
                    sign3=1,
                    sign2=sign2,
                    order=7,
                    parity=parity,
                )
                for index, (actual, expected) in enumerate(
                    zip(direct_values, reference_values)
                ):
                    with self.subTest(
                        sign2=sign2, parity=parity, index=index
                    ):
                        self.assertLess(abs(actual - expected), 3.0e-12)

    def test_regular_seed_completes_c_recursion_through_level_9_over_2(
        self,
    ):
        for sign2 in (1, -1):
            recursive = CRecursiveMixedNSExchangeSphereFourPointBlock(
                c=self.c,
                h1_r=self.r_weights[0],
                h2_r=self.r_weights[1],
                h3_ns=self.ns_weights[2],
                h4_ns=self.ns_weights[3],
                internal_weight=self.internal_weight,
                sign2=sign2,
                regular_seed=self.seed,
            )
            reference = CRecursiveMixedNSExchangeSphereFourPointBlock(
                c=self.c,
                h1_r=self.r_weights[0],
                h2_r=self.r_weights[1],
                h3_ns=self.ns_weights[2],
                h4_ns=self.ns_weights[3],
                internal_weight=self.internal_weight,
                sign2=sign2,
            )
            for parity, twice_level in (
                ("even", 4),
                ("odd", 5),
                ("even", 8),
                ("odd", 9),
            ):
                expected = reference._reference_local_coefficients(
                    c=self.c,
                    internal_weight=self.internal_weight,
                    sign3=1,
                    sign2=sign2,
                    order=twice_level // 2 + 1,
                    parity=parity,
                )[twice_level // 2]
                with self.subTest(
                    sign2=sign2,
                    parity=parity,
                    twice_level=twice_level,
                ):
                    self.assertLess(
                        abs(
                            recursive.coefficient(twice_level, parity)
                            - expected
                        ),
                        3.0e-12,
                    )

    def test_plain_global_seed_fails_for_large_ramond_sign(self):
        c_large = 100000.0
        h = self.internal_weight
        regular_plus = self.seed.coefficient(
            4, h, c_large, "even", 1
        )
        regular_minus = self.seed.coefficient(
            4, h, c_large, "even", -1
        )
        trial = CRecursiveMixedNSExchangeSphereFourPointBlock(
            c=c_large,
            h1_r=self.r_weights[0],
            h2_r=self.r_weights[1],
            h3_ns=self.ns_weights[2],
            h4_ns=self.ns_weights[3],
            internal_weight=h,
            sign2=1,
        ).naive_global_seed(4, "even")
        self.assertLess(abs(regular_plus - trial), 3.0e-9)
        self.assertGreater(abs(regular_minus - trial), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
