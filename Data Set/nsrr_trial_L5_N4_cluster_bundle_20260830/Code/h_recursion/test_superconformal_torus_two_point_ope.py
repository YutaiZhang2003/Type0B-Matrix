"""Tests for the sphere-three-point--torus-one-point sewing channel."""

from __future__ import annotations

import unittest

from mixed_ns_ramond_descendant_blocks import NSVermaModule
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)
from superconformal_torus_two_point_ope import (
    BruteForceNSOPEBridgeGroundBlock,
    NSTorusTwoPointOPEBlock,
    PlaneTorusExternalNSWardVector,
    RamondHandleTorusTwoPointOPEBlock,
    SelfDualNSTorusTwoPointOPEBlock,
)


class TorusTwoPointOPEBlockTests(unittest.TestCase):
    def test_external_torus_ward_vector_through_level_one(self):
        h = 0.72
        module = NSVermaModule(c=13.5, weight=h)
        ward = PlaneTorusExternalNSWardVector(h)
        self.assertEqual(ward.vector(module, 0), (1.0 + 0.0j,))
        self.assertEqual(ward.vector(module, 1), (0.0j,))
        self.assertEqual(ward.vector(module, 2), (-h + 0.0j,))

    def test_bridge_ward_gram_contraction(self):
        hs, d1, d2 = 0.72, 0.31, 0.44
        block = NSTorusTwoPointOPEBlock(
            b=1.27,
            bridge_weight=hs,
            handle_weight=0.83,
            external_weight_1=d1,
            external_weight_2=d2,
        )
        coefficients = block.bridge_coefficients(2)
        self.assertLess(abs(coefficients[0] - 1.0), 1.0e-14)
        self.assertLess(abs(coefficients[1]), 1.0e-14)
        # The sphere vector is hs+d1-d2.  Sewing with B_1=2hs and
        # <L_-1 V_hs>=-hs<V_hs>, followed by z-1=-(1-z), gives:
        expected = 0.5 * (hs + d1 - d2)
        self.assertLess(abs(coefficients[2] - expected), 1.0e-14)

    def test_ns_bridge_level_zero_is_torus_one_point_block(self):
        block = NSTorusTwoPointOPEBlock(
            b=1.27,
            bridge_weight=0.72,
            handle_weight=0.83,
            external_weight_1=0.31,
            external_weight_2=0.44,
        )
        x = 0.19
        plumbing = NSPlumbingParameter(0.03, 1)
        actual = block.chiral_block(x, plumbing, 0, 4)
        expected = (
            x
            ** (
                block.bridge_weight
                - block.external_weight_1
                - block.external_weight_2
            )
            * block.handle.chiral_block(plumbing, 4)
        )
        self.assertLess(abs(actual - expected), 2.0e-14)

    def test_ns_two_direction_coefficients_factorize_at_level_one(self):
        block = NSTorusTwoPointOPEBlock(
            b=1.27,
            bridge_weight=0.72,
            handle_weight=0.83,
            external_weight_1=0.31,
            external_weight_2=0.44,
        )
        coefficients = block.raw_coefficients(2, 4)
        handle = block.handle.raw_coefficients(4)
        bridge = block.bridge_coefficients(2)
        for bridge_level in range(3):
            for handle_level in range(5):
                with self.subTest(
                    bridge_level=bridge_level,
                    handle_level=handle_level,
                ):
                    self.assertLess(
                        abs(
                            coefficients[(bridge_level, handle_level)]
                            - bridge[bridge_level] * handle[handle_level]
                        ),
                        1.0e-14,
                    )

    def test_r_bridge_level_zero_is_normalized_torus_one_point_block(self):
        block = RamondHandleTorusTwoPointOPEBlock(
            b=1.27,
            bridge_weight=0.72,
            handle_beta=0.41j,
            external_weight_1=0.31,
            external_weight_2=0.44,
            sign=1,
        )
        x = 0.19
        plumbing = RamondPlumbingParameter(0.03, "identity")
        actual = block.normalized_chiral_block(x, plumbing, 0, 3)
        handle_series = sum(
            coefficient * plumbing.level_factor(level)
            for level, coefficient in enumerate(
                block.handle.raw_even_coefficients(3)
            )
        )
        expected = (
            x
            ** (
                block.bridge_weight
                - block.external_weight_1
                - block.external_weight_2
            )
            * plumbing.q
            ** (block.handle_weight - block.c / 24.0)
            * handle_series
        )
        self.assertLess(abs(actual - expected), 2.0e-14)

    def test_self_dual_wrapper_uses_finite_part_handle(self):
        block = SelfDualNSTorusTwoPointOPEBlock(
            bridge_momentum=0.61,
            handle_momentum=0.74,
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            samples=8,
        )
        coefficients = block.raw_coefficients(2, 2)
        self.assertLess(abs(coefficients[(0, 0)] - 1.0), 2.0e-12)
        expected_bridge = 0.5 * (
            block.bridge_weight
            + block.external_weight_1
            - block.external_weight_2
        )
        self.assertLess(
            abs(coefficients[(2, 0)] - expected_bridge), 2.0e-12
        )
    def test_rejects_unimplemented_bridge_level(self):
        block = NSTorusTwoPointOPEBlock(
            b=1.27,
            bridge_weight=0.72,
            handle_weight=0.83,
            external_weight_1=0.31,
            external_weight_2=0.44,
        )
        with self.assertRaisesRegex(ValueError, "through bridge level one"):
            block.raw_coefficients(3, 2)


if __name__ == "__main__":
    unittest.main()
