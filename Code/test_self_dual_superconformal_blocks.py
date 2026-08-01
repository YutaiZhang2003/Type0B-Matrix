"""Tests for collision-aware evaluation at the Type-0B self-dual point."""

import cmath
import unittest

from ramond_sphere_blocks import RamondExternalSphereFourPointBlock
from self_dual_superconformal_blocks import (
    SelfDualMixedNSExchangeSphereFourPointBlock,
    SelfDualMixedRExchangeSphereFourPointBlock,
    SelfDualRamondExternalSphereFourPointBlock,
    self_dual_finite_part,
)


class SelfDualFinitePartTests(unittest.TestCase):
    def test_cauchy_projection_removes_principal_part(self):
        def laurent_function(b):
            t = cmath.log(b)
            return 3.0 - 2.0j + 0.7 / t - 0.2j / (t * t) + t + t**3

        result = self_dual_finite_part(laurent_function, samples=16)
        self.assertAlmostEqual(result.value, 3.0 - 2.0j, places=12)
        self.assertLess(result.relative_error, 1.0e-12)

    def test_four_r_block_is_available_directly_at_b_one(self):
        generic = RamondExternalSphereFourPointBlock.from_liouville_momenta(
            p1=0.20,
            p2=0.35,
            p3=0.40,
            p4=0.55,
            internal_momentum=0.70,
            b=1.0,
        )
        with self.assertRaises(ZeroDivisionError):
            generic.elliptic_coefficients(2, "odd")

        block = SelfDualRamondExternalSphereFourPointBlock(
            p1=0.20,
            p2=0.35,
            p3=0.40,
            p4=0.55,
            internal_momentum=0.70,
        )
        self.assertAlmostEqual(
            block.elliptic_block(0.37, 12, "even"),
            1.874029257645511,
            places=12,
        )
        self.assertAlmostEqual(
            block.elliptic_block(0.37, 12, "odd"),
            -0.00504094652288641,
            places=14,
        )
        self.assertLess(
            max(
                diagnostic.relative_error
                for diagnostic in block.coefficient_diagnostics(
                    12, "odd"
                ).values()
            ),
            5.0e-8,
        )

    def test_long_r_g0_level_one_anchor_at_exact_c(self):
        block = SelfDualMixedRExchangeSphereFourPointBlock(
            p1_ns=0.31,
            p2_r=0.41,
            p3_r=0.23,
            p4_ns=0.37,
            internal_momentum=0.70,
            sign3=1,
            sign2=-1,
        )
        self.assertAlmostEqual(
            block.recursion_level_one_coefficient(),
            block.direct_level_one_coefficient(),
            places=13,
        )
        self.assertLess(
            max(
                diagnostic.relative_error
                for diagnostic in block.coefficient_diagnostics(12).values()
            ),
            1.0e-8,
        )

    def test_long_r_wrapper_rejects_the_short_ground_state(self):
        with self.assertRaisesRegex(ValueError, "short quotient"):
            SelfDualMixedRExchangeSphereFourPointBlock(
                p1_ns=0.31,
                p2_r=0.41,
                p3_r=0.23,
                p4_ns=0.37,
                internal_momentum=0.0,
            )

    def test_both_rrnn_channels_have_finite_order_twelve_blocks(self):
        ns_block = SelfDualMixedNSExchangeSphereFourPointBlock(
            p1_r=0.20,
            p2_r=0.40,
            p3_ns=0.30,
            p4_ns=0.30,
            internal_momentum=0.70,
        )
        r_block = SelfDualMixedRExchangeSphereFourPointBlock(
            p1_ns=0.30,
            p2_r=0.40,
            p3_r=0.20,
            p4_ns=0.30,
            internal_momentum=0.70,
        )
        self.assertAlmostEqual(
            ns_block.elliptic_block(0.37, 12, "even"),
            1.94636896290303,
            places=12,
        )
        self.assertAlmostEqual(
            r_block.elliptic_block(0.37, 12),
            1.82014451619952,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
