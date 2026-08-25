"""Fixed-weight c-pole tests for sphere blocks with Ramond insertions."""

from __future__ import annotations

import unittest

from mixed_ramond_sphere_blocks import MixedNSExchangeSphereFourPointBlock
from ramond_c_recursive_sphere_blocks import (
    CRecursiveMixedNSExchangeSphereFourPointBlock,
    CRecursiveRamondExternalSphereFourPointBlock,
)
from ramond_sphere_blocks import (
    RamondExternalSphereFourPointBlock,
    ramond_liouville_weight,
)
from superconformal_blocks import central_charge, ns_liouville_weight


class RamondCRecursionPoleTests(unittest.TestCase):
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

    def rrrr_block(
        self, sign3: int, sign2: int
    ) -> CRecursiveRamondExternalSphereFourPointBlock:
        return CRecursiveRamondExternalSphereFourPointBlock(
            c=self.c,
            h1=self.r_weights[0],
            h2=self.r_weights[1],
            h3=self.r_weights[2],
            h4=self.r_weights[3],
            internal_weight=self.internal_weight,
            sign3=sign3,
            sign2=sign2,
        )

    def mixed_block(
        self, sign2: int
    ) -> CRecursiveMixedNSExchangeSphereFourPointBlock:
        return CRecursiveMixedNSExchangeSphereFourPointBlock(
            c=self.c,
            h1_r=self.r_weights[0],
            h2_r=self.r_weights[1],
            h3_ns=self.ns_weights[2],
            h4_ns=self.ns_weights[3],
            internal_weight=self.internal_weight,
            sign2=sign2,
        )

    def test_rrrr_direct_levels_zero_half_and_one(self) -> None:
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                with self.subTest(sign3=sign3, sign2=sign2):
                    c_block = self.rrrr_block(sign3, sign2)
                    h_block = (
                        RamondExternalSphereFourPointBlock.from_liouville_momenta(
                            p1=self.momenta[0],
                            p2=self.momenta[1],
                            p3=self.momenta[2],
                            p4=self.momenta[3],
                            internal_momentum=0.70,
                            b=self.b,
                            sign3=sign3,
                            sign2=sign2,
                        )
                    )
                    direct = h_block.direct_leading_coefficients()
                    self.assertAlmostEqual(
                        c_block.coefficient(0, "even"),
                        direct["even_level_0"],
                    )
                    self.assertAlmostEqual(
                        c_block.coefficient(1, "odd"),
                        direct["odd_level_half"],
                    )
                    self.assertAlmostEqual(
                        c_block.coefficient(2, "even"),
                        direct["even_level_1"],
                    )

    def test_scalar_global_seed_is_rejected_beyond_level_one(self) -> None:
        block = self.rrrr_block(1, 1)
        with self.assertRaisesRegex(
            NotImplementedError, "matrix-valued Ramond large-c"
        ):
            block.coefficient(3, "odd")
        with self.assertRaisesRegex(
            NotImplementedError, "matrix-valued Ramond large-c"
        ):
            block.coefficient(4, "even")

    def test_explicit_scalar_seed_reaches_elliptic_diagnostic(self) -> None:
        holder = {}

        def scalar_seed(
            twice_level, internal_weight, c, parity, sign3, sign2
        ):
            return holder["block"]._global_trial_seed(
                twice_level=twice_level,
                internal_weight=internal_weight,
                c=c,
                parity=parity,
                sign3=sign3,
                sign2=sign2,
            )

        block = CRecursiveMixedNSExchangeSphereFourPointBlock(
            c=self.c,
            h1_r=self.r_weights[0],
            h2_r=self.r_weights[1],
            h3_ns=self.ns_weights[2],
            h4_ns=self.ns_weights[3],
            internal_weight=self.internal_weight,
            sign2=1,
            regular_seed=scalar_seed,
        )
        holder["block"] = block
        reference = MixedNSExchangeSphereFourPointBlock(
            b=self.b,
            p1_r=self.momenta[0],
            p2_r=self.momenta[1],
            p3_ns=self.momenta[2],
            p4_ns=self.momenta[3],
            internal_momentum=0.70,
            sign2=1,
        )
        c_even = block.elliptic_coefficients(3, "even")
        h_even = reference.elliptic_coefficients(3, "even")
        c_odd = block.elliptic_coefficients(2, "odd")
        h_odd = reference.elliptic_coefficients(2, "odd")
        self.assertAlmostEqual(c_even[0], h_even[0])
        self.assertAlmostEqual(c_even[2], h_even[2])
        self.assertAlmostEqual(c_odd[1], h_odd[1])
        self.assertGreater(abs(c_even[4] - h_even[4]), 1.0e-4)
        self.assertGreater(abs(c_odd[3] - h_odd[3]), 1.0e-3)

    def test_rrrr_first_c_poles_all_sign_branches(self) -> None:
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                block = self.rrrr_block(sign3, sign2)
                for r, s, twice_level, parity in (
                    (3, 1, 3, "odd"),
                    (2, 2, 4, "even"),
                ):
                    with self.subTest(
                        sign3=sign3,
                        sign2=sign2,
                        r=r,
                        s=s,
                    ):
                        check = block.numerical_residue_check(
                            r=r,
                            s=s,
                            twice_level=twice_level,
                            parity=parity,
                            epsilon=1.0e-5,
                        )
                        self.assertLess(check.relative_error, 2.0e-8)

    def test_mixed_ns_exchange_first_c_poles(self) -> None:
        for sign2 in (1, -1):
            block = self.mixed_block(sign2)
            for r, s, twice_level, parity in (
                (3, 1, 3, "odd"),
                (2, 2, 4, "even"),
            ):
                with self.subTest(sign2=sign2, r=r, s=s):
                    check = block.numerical_residue_check(
                        r=r,
                        s=s,
                        twice_level=twice_level,
                        parity=parity,
                        epsilon=1.0e-5,
                    )
                    self.assertLess(check.relative_error, 2.0e-8)

    def test_fixed_weight_mixed_reference_constructor(self) -> None:
        block = MixedNSExchangeSphereFourPointBlock.from_weights(
            b=self.b,
            h1_r=self.r_weights[0],
            h2_r=self.r_weights[1],
            h3_ns=self.ns_weights[2],
            h4_ns=self.ns_weights[3],
            internal_weight=self.internal_weight,
            sign2=-1,
        )
        self.assertAlmostEqual(block.h1, self.r_weights[0])
        self.assertAlmostEqual(block.h2, self.r_weights[1])
        self.assertAlmostEqual(block.h3, self.ns_weights[2])
        self.assertAlmostEqual(block.h4, self.ns_weights[3])
        self.assertAlmostEqual(block.internal_weight, self.internal_weight)


if __name__ == "__main__":
    unittest.main()
