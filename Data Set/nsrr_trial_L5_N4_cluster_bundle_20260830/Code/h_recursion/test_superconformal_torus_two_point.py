"""Tests for the leading two-punctured torus and the modular frame change."""

import cmath
import math
import unittest

from mixed_ramond_sphere_blocks import (
    _r_a_beta,
    _r_beta_prime,
    _r_beta_rs,
)
from ramond_sphere_blocks import ramond_beta, ramond_liouville_weight
from super_liouville_torus_one_point import (
    run_type0b_ns_modular_s_check,
    run_type0b_ns_modular_s_convergence,
)
from super_liouville_torus_two_point import (
    Type0BNSTorusTwoPointHRecursionCorrelator,
    Type0BNSTorusTwoPointLeadingCorrelator,
    Type0BRamondTorusTwoPointBetaRecursionCorrelator,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    NSTorusOnePointBlock,
    RamondPlumbingParameter,
    RamondTorusOnePointBlock,
)
from superconformal_torus_two_point import (
    BruteForceRamondTorusTwoPointBlock,
    DirectRamondTorusTwoPointRegularSeed,
    MixedNSRamondTorusTwoPointGroundBlock,
    NSTorusTwoPointHRecursionBlock,
    NSTorusTwoPointLeadingBlock,
    RamondTorusTwoPointBetaRecursionBlock,
    RamondTorusTwoPointGroundBlock,
    SelfDualNSTorusTwoPointHRecursionBlock,
    SelfDualRamondTorusTwoPointBetaRecursionBlock,
)


class SuperconformalTorusTwoPointTests(unittest.TestCase):
    def test_mixed_ns_r_ground_block_has_two_external_r_punctures(self):
        block = MixedNSRamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_ns_weight=0.71,
            internal_r_weight=13.5 / 24.0 + 0.31,
        )
        plus = block.ground_coefficient(
            NSPlumbingParameter(0.07, 1),
            RamondPlumbingParameter(0.05, "identity"),
        )
        flipped_lift = block.ground_coefficient(
            NSPlumbingParameter(0.07, -1),
            RamondPlumbingParameter(0.05, "identity"),
        )
        self.assertAlmostEqual(plus, 1.0, places=14)
        self.assertAlmostEqual(flipped_lift, plus, places=14)

    def test_mixed_ns_r_ground_parity_acts_on_internal_r_component(self):
        block = MixedNSRamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_ns_weight=0.71,
            internal_r_weight=13.5 / 24.0 + 0.31,
            vertex_sign_1=1,
            vertex_sign_2=-1,
            external_ground_1="-",
            external_ground_2="-",
        )
        identity = block.ground_coefficient(
            NSPlumbingParameter(0.07),
            RamondPlumbingParameter(0.05, "identity"),
        )
        parity = block.ground_coefficient(
            NSPlumbingParameter(0.07),
            RamondPlumbingParameter(0.05, "parity"),
        )
        self.assertAlmostEqual(identity, -1.0, places=14)
        self.assertAlmostEqual(parity, 1.0, places=14)

    def test_h_recursion_matches_complete_leading_ward_gram_ledger(self):
        h1, h2 = 0.71, 0.83
        d1, d2 = 0.27, 0.36
        recursive = NSTorusTwoPointHRecursionBlock(
            b=1.27,
            internal_weight_1=h1,
            internal_weight_2=h2,
            external_weight_1=d1,
            external_weight_2=d2,
        )
        direct = NSTorusTwoPointLeadingBlock(
            central_charge=recursive.c,
            internal_weight_1=h1,
            internal_weight_2=h2,
            external_weight_1=d1,
            external_weight_2=d2,
        )
        actual = recursive.raw_coefficients(2, 2)
        expected = direct.raw_coefficients()
        for key in ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
            with self.subTest(key=key):
                self.assertLess(abs(actual[key] - expected[key]), 2.0e-14)

    def test_h_recursion_is_cyclic_through_four_half_levels(self):
        block = NSTorusTwoPointHRecursionBlock(
            b=1.27,
            internal_weight_1=0.71,
            internal_weight_2=0.83,
            external_weight_1=0.27,
            external_weight_2=0.36,
        )
        exchanged = NSTorusTwoPointHRecursionBlock(
            b=1.27,
            internal_weight_1=0.83,
            internal_weight_2=0.71,
            external_weight_1=0.36,
            external_weight_2=0.27,
        )
        coefficients = block.raw_coefficients(4, 4)
        exchanged_coefficients = exchanged.raw_coefficients(4, 4)
        for twice_level_1 in range(5):
            for twice_level_2 in range(5):
                with self.subTest(
                    twice_level_1=twice_level_1,
                    twice_level_2=twice_level_2,
                ):
                    self.assertLess(
                        abs(
                            coefficients[(twice_level_1, twice_level_2)]
                            - exchanged_coefficients[
                                (twice_level_2, twice_level_1)
                            ]
                        ),
                        1.2e-12,
                    )

    def test_h_recursion_identity_puncture_reduces_to_one_point_block(self):
        b = 1.27
        h = 0.81
        d = 0.36
        max_twice_level = 6
        samples = 32
        radius = 0.03
        totals = {
            (level_1, level_2): 0.0j
            for level_1 in range(max_twice_level + 1)
            for level_2 in range(max_twice_level + 1)
        }
        for index in range(samples):
            angle = 2.0 * math.pi * (index + 0.5) / samples
            displacement = radius * cmath.exp(1j * angle)
            values = NSTorusTwoPointHRecursionBlock(
                b=b,
                internal_weight_1=h,
                internal_weight_2=h + displacement,
                external_weight_1=0.0,
                external_weight_2=d,
            ).raw_coefficients(max_twice_level, max_twice_level)
            for key, value in values.items():
                totals[key] += value / samples

        one_point = NSTorusOnePointBlock(
            b=b,
            internal_weight=h,
            external_weight=d,
        ).raw_coefficients(max_twice_level)
        for level in range(max_twice_level + 1):
            self.assertLess(
                abs(totals[(level, level)] - one_point[level]),
                3.0e-12,
            )
        self.assertLess(
            max(
                abs(value)
                for (level_1, level_2), value in totals.items()
                if level_1 != level_2
            ),
            3.0e-12,
        )

    def test_self_dual_h_recursion_finite_part_matches_leading_layer(self):
        recursive = SelfDualNSTorusTwoPointHRecursionBlock(
            internal_momentum_1=0.61,
            internal_momentum_2=0.74,
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            samples=16,
        )
        direct = NSTorusTwoPointLeadingBlock(
            central_charge=recursive.c,
            internal_weight_1=recursive.internal_weight_1,
            internal_weight_2=recursive.internal_weight_2,
            external_weight_1=0.5 + 0.33**2 / 2.0,
            external_weight_2=0.5 + 0.41**2 / 2.0,
        )
        actual = recursive.raw_coefficients(2, 2)
        expected = direct.raw_coefficients()
        for key in ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
            with self.subTest(key=key):
                self.assertLess(abs(actual[key] - expected[key]), 3.0e-12)

    def test_self_dual_equal_weight_collision_is_radius_stable_and_cyclic(self):
        block = SelfDualNSTorusTwoPointHRecursionBlock(
            internal_momentum_1=0.61,
            internal_momentum_2=0.61,
            external_momentum_1=0.29,
            external_momentum_2=0.33,
            samples=12,
            difference_radius=0.025,
            difference_samples=16,
        )
        exchanged = SelfDualNSTorusTwoPointHRecursionBlock(
            internal_momentum_1=0.61,
            internal_momentum_2=0.61,
            external_momentum_1=0.33,
            external_momentum_2=0.29,
            samples=12,
            difference_radius=0.035,
            difference_samples=16,
        )
        actual = block.raw_coefficients(4, 4)
        expected = exchanged.raw_coefficients(4, 4)
        for level_1 in range(5):
            for level_2 in range(5):
                with self.subTest(level_1=level_1, level_2=level_2):
                    self.assertLess(
                        abs(
                            actual[(level_1, level_2)]
                            - expected[(level_2, level_1)]
                        ),
                        2.0e-9,
                    )

    def test_leading_coefficients_and_fermion_parity(self):
        h1, h2 = 0.71, 0.83
        d1, d2 = 0.27, 0.36
        block = NSTorusTwoPointLeadingBlock(
            central_charge=13.5,
            internal_weight_1=h1,
            internal_weight_2=h2,
            external_weight_1=d1,
            external_weight_2=d2,
        )
        coefficients = block.raw_coefficients()
        self.assertEqual(coefficients[(1, 0)], 0.0j)
        self.assertEqual(coefficients[(0, 1)], 0.0j)
        self.assertAlmostEqual(
            coefficients[(1, 1)],
            (h1 + h2 - d1)
            * (h1 + h2 - d2)
            / (4.0 * h1 * h2),
            places=14,
        )
        plus = block.evaluate(
            NSPlumbingParameter(0.07, 1),
            NSPlumbingParameter(0.05, 1),
        )
        flipped = block.evaluate(
            NSPlumbingParameter(0.07, -1),
            NSPlumbingParameter(0.05, 1),
        )
        expected_difference = (
            2.0
            * coefficients[(1, 1)]
            * (0.07 * 0.05) ** 0.5
        )
        self.assertAlmostEqual(
            plus - flipped, expected_difference, places=14
        )

    def test_identity_insertion_reduces_to_torus_one_point_half_level(self):
        h = 0.73
        d = 0.29
        block = NSTorusTwoPointLeadingBlock(
            central_charge=13.5,
            internal_weight_1=h,
            internal_weight_2=h,
            external_weight_1=0.0,
            external_weight_2=d,
        )
        coefficients = block.raw_coefficients()
        self.assertAlmostEqual(coefficients[(2, 0)], 0.0, places=14)
        self.assertAlmostEqual(coefficients[(0, 2)], 0.0, places=14)
        self.assertAlmostEqual(
            coefficients[(1, 1)],
            1.0 - d / (2.0 * h),
            places=14,
        )

    def test_cyclic_exchange_of_necklace_edges(self):
        block = NSTorusTwoPointLeadingBlock(
            central_charge=13.5,
            internal_weight_1=0.71,
            internal_weight_2=0.83,
            external_weight_1=0.27,
            external_weight_2=0.36,
        )
        exchanged = NSTorusTwoPointLeadingBlock(
            central_charge=13.5,
            internal_weight_1=0.83,
            internal_weight_2=0.71,
            external_weight_1=0.36,
            external_weight_2=0.27,
        )
        value = block.chiral_block(
            NSPlumbingParameter(0.07),
            NSPlumbingParameter(0.05),
        )
        exchanged_value = exchanged.chiral_block(
            NSPlumbingParameter(0.05),
            NSPlumbingParameter(0.07),
        )
        self.assertAlmostEqual(value, exchanged_value, places=14)

    def test_leading_nonchiral_integrand_obeys_cyclic_exchange(self):
        plumbing_1 = NSPlumbingParameter(0.07)
        plumbing_2 = NSPlumbingParameter(0.05)
        correlator = Type0BNSTorusTwoPointLeadingCorrelator(
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            quadrature_order=2,
        )
        exchanged = Type0BNSTorusTwoPointLeadingCorrelator(
            external_momentum_1=0.41,
            external_momentum_2=0.33,
            quadrature_order=2,
        )
        value = correlator.momentum_integrand(
            0.61, 0.74, plumbing_1, plumbing_2
        )
        exchanged_value = exchanged.momentum_integrand(
            0.74, 0.61, plumbing_2, plumbing_1
        )
        self.assertAlmostEqual(value, exchanged_value, places=13)

    def test_recursive_nonchiral_integrand_obeys_cyclic_exchange(self):
        plumbing_1 = NSPlumbingParameter(0.07)
        plumbing_2 = NSPlumbingParameter(0.05)
        correlator = Type0BNSTorusTwoPointHRecursionCorrelator(
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            max_twice_level_1=2,
            max_twice_level_2=3,
            quadrature_order=2,
            structure_precision=15,
            finite_part_samples=8,
            difference_samples=8,
        )
        exchanged = Type0BNSTorusTwoPointHRecursionCorrelator(
            external_momentum_1=0.41,
            external_momentum_2=0.33,
            max_twice_level_1=3,
            max_twice_level_2=2,
            quadrature_order=2,
            structure_precision=15,
            finite_part_samples=8,
            difference_samples=8,
        )
        value = correlator.momentum_integrand(
            0.61, 0.74, plumbing_1, plumbing_2
        )
        exchanged_value = exchanged.momentum_integrand(
            0.74, 0.61, plumbing_2, plumbing_1
        )
        self.assertLess(
            abs(value - exchanged_value) / max(abs(value), 1.0e-300),
            2.0e-12,
        )

    def test_ramond_beta_recursion_matches_two_leg_ward_sewing(self):
        b = 1.27
        c = central_charge(b)
        beta_1 = ramond_beta(0.61)
        beta_2 = ramond_beta(0.74)
        h_1 = ramond_liouville_weight(0.61, b)
        h_2 = ramond_liouville_weight(0.74, b)
        d_1 = ns_liouville_weight(0.33, b)
        d_2 = ns_liouville_weight(0.41, b)
        sectors = (
            (1, 1, "identity", "identity"),
            (-1, -1, "identity", "identity"),
            (1, -1, "parity", "identity"),
            (-1, 1, "parity", "identity"),
        )
        for sign_1, sign_2, cycle_1, cycle_2 in sectors:
            recursive = RamondTorusTwoPointBetaRecursionBlock(
                b=b,
                internal_beta_1=beta_1,
                internal_beta_2=beta_2,
                external_weight_1=d_1,
                external_weight_2=d_2,
                vertex_sign_1=sign_1,
                vertex_sign_2=sign_2,
                cycle_insertion_1=cycle_1,
                cycle_insertion_2=cycle_2,
            )
            direct = BruteForceRamondTorusTwoPointBlock(
                central_charge=c,
                internal_weight_1=h_1,
                internal_weight_2=h_2,
                external_weight_1=d_1,
                external_weight_2=d_2,
                vertex_sign_1=sign_1,
                vertex_sign_2=sign_2,
                cycle_insertion_1=cycle_1,
                cycle_insertion_2=cycle_2,
            )
            actual = recursive.raw_coefficients(1, 1)
            expected = direct.raw_coefficients(1, 1)
            for levels in actual:
                with self.subTest(
                    signs=(sign_1, sign_2),
                    cycles=(cycle_1, cycle_2),
                    levels=levels,
                ):
                    self.assertLess(
                        abs(actual[levels] - expected[levels]),
                        1.0e-12,
                    )

    def test_ramond_beta_poles_match_direct_two_leg_residues(self):
        b = 1.27
        beta_2 = ramond_beta(0.74)
        d_1 = ns_liouville_weight(0.33, b)
        d_2 = ns_liouville_weight(0.41, b)
        sign_1 = sign_2 = 1
        seed = DirectRamondTorusTwoPointRegularSeed(
            b=b,
            external_weight_1=d_1,
            external_weight_2=d_2,
        )
        kernel = RamondTorusTwoPointBetaRecursionBlock(
            b=b,
            internal_beta_1=ramond_beta(0.61),
            internal_beta_2=beta_2,
            external_weight_1=d_1,
            external_weight_2=d_2,
            use_direct_regular_seed=False,
        )
        epsilon = 1.0e-7
        for r, s in ((1, 2), (2, 1)):
            beta_rs = _r_beta_rs(b, r, s)
            beta_prime = _r_beta_prime(b, r, s)
            a_factor = _r_a_beta(b, r, s, 1.0e-12)
            for pole_sign in (1, -1):
                pole = pole_sign * beta_rs
                plus = seed.direct_coefficient(
                    1,
                    0,
                    pole + epsilon,
                    beta_2,
                    sign_1,
                    sign_2,
                )
                minus = seed.direct_coefficient(
                    1,
                    0,
                    pole - epsilon,
                    beta_2,
                    sign_1,
                    sign_2,
                )
                measured = epsilon * (plus - minus) / 2.0
                if pole_sign == 1:
                    predicted = (
                        a_factor
                        * kernel._fusion_product(
                            r=r,
                            s=s,
                            adjacent_beta=beta_2,
                            sign_1=sign_1,
                            sign_2=sign_2,
                        )
                        * seed.direct_coefficient(
                            0,
                            0,
                            beta_prime,
                            beta_2,
                            sign_1,
                            sign_2,
                        )
                    )
                else:
                    predicted = (
                        -a_factor
                        * kernel._fusion_product(
                            r=r,
                            s=s,
                            adjacent_beta=beta_2,
                            sign_1=-sign_1,
                            sign_2=-sign_2,
                        )
                        * seed.direct_coefficient(
                            0,
                            0,
                            beta_prime,
                            beta_2,
                            -sign_1,
                            -sign_2,
                        )
                    )
                with self.subTest(r=r, s=s, pole_sign=pole_sign):
                    self.assertLess(
                        abs(measured - predicted)
                        / max(abs(predicted), 1.0e-300),
                        3.0e-8,
                    )

    def test_ramond_identity_puncture_reduces_to_one_point_at_level_one(self):
        b = 1.27
        beta = ramond_beta(0.61)
        external_weight = 0.36
        for sign, cycle in ((1, "identity"), (-1, "parity")):
            two_point = RamondTorusTwoPointBetaRecursionBlock(
                b=b,
                internal_beta_1=beta,
                internal_beta_2=beta,
                external_weight_1=0.0,
                external_weight_2=external_weight,
                vertex_sign_1=1,
                vertex_sign_2=sign,
                cycle_insertion_1=cycle,
            ).raw_coefficients(1, 1)
            one_point = RamondTorusOnePointBlock(
                b=b,
                internal_beta=beta,
                external_weight=external_weight,
                sign=sign,
            ).cycle_projected_raw_coefficients(
                RamondPlumbingParameter(0.03, cycle), 1
            )
            for level in range(2):
                self.assertLess(
                    abs(two_point[(level, level)] - one_point[level]),
                    2.0e-12,
                )
            self.assertLess(
                max(
                    abs(value)
                    for (level_1, level_2), value in two_point.items()
                    if level_1 != level_2
                ),
                2.0e-12,
            )

    def test_self_dual_ramond_beta_recursion_matches_direct_level_one(self):
        recursive = SelfDualRamondTorusTwoPointBetaRecursionBlock(
            internal_momentum_1=0.61,
            internal_momentum_2=0.74,
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            samples=12,
        )
        direct = BruteForceRamondTorusTwoPointBlock(
            central_charge=central_charge(1.0),
            internal_weight_1=ramond_liouville_weight(0.61, 1.0),
            internal_weight_2=ramond_liouville_weight(0.74, 1.0),
            external_weight_1=ns_liouville_weight(0.33, 1.0),
            external_weight_2=ns_liouville_weight(0.41, 1.0),
        )
        actual = recursive.raw_coefficients(1, 1)
        expected = direct.raw_coefficients(1, 1)
        for levels in actual:
            with self.subTest(levels=levels):
                self.assertLess(
                    abs(actual[levels] - expected[levels]),
                    2.0e-11,
                )
        self.assertLess(
            max(
                diagnostic.relative_error
                for diagnostic in recursive.coefficient_diagnostics(
                    1, 1
                ).values()
            ),
            2.0e-10,
        )

    def test_ramond_channel_integrand_obeys_cyclic_exchange(self):
        plumbing_1 = RamondPlumbingParameter(0.07, "parity")
        plumbing_2 = RamondPlumbingParameter(0.05, "identity")
        correlator = Type0BRamondTorusTwoPointBetaRecursionCorrelator(
            external_momentum_1=0.33,
            external_momentum_2=0.41,
            max_level_1=1,
            max_level_2=0,
            cycle_insertion_1="parity",
            cycle_insertion_2="identity",
            quadrature_order=2,
            structure_precision=15,
            finite_part_samples=8,
        )
        exchanged = Type0BRamondTorusTwoPointBetaRecursionCorrelator(
            external_momentum_1=0.41,
            external_momentum_2=0.33,
            max_level_1=0,
            max_level_2=1,
            cycle_insertion_1="identity",
            cycle_insertion_2="parity",
            quadrature_order=2,
            structure_precision=15,
            finite_part_samples=8,
        )
        value = correlator.momentum_integrand(
            0.61, 0.74, plumbing_1, plumbing_2
        )
        exchanged_value = exchanged.momentum_integrand(
            0.74, 0.61, plumbing_2, plumbing_1
        )
        self.assertLess(
            abs(value - exchanged_value) / max(abs(value), 1.0e-300),
            3.0e-11,
        )

    def test_ramond_ground_identity_cycle_selects_equal_hjs_signs(self):
        block = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=13.5 / 24.0 + 0.31,
            internal_weight_2=13.5 / 24.0 + 0.47,
            external_weight_1=0.27,
            external_weight_2=0.36,
            vertex_sign_1=1,
            vertex_sign_2=1,
        )
        identity_1 = RamondPlumbingParameter(0.07, "identity")
        identity_2 = RamondPlumbingParameter(0.05, "identity")
        self.assertAlmostEqual(
            block.ground_coefficient(identity_1, identity_2),
            2.0,
            places=14,
        )
        opposite = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=block.internal_weight_1,
            internal_weight_2=block.internal_weight_2,
            external_weight_1=block.external_weight_1,
            external_weight_2=block.external_weight_2,
            vertex_sign_1=1,
            vertex_sign_2=-1,
        )
        self.assertAlmostEqual(
            opposite.ground_coefficient(identity_1, identity_2),
            0.0,
            places=14,
        )

    def test_ramond_ground_parity_cycle_selects_opposite_hjs_signs(self):
        parity_1 = RamondPlumbingParameter(0.07, "parity")
        identity_2 = RamondPlumbingParameter(0.05, "identity")
        same = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=13.5 / 24.0 + 0.31,
            internal_weight_2=13.5 / 24.0 + 0.47,
            external_weight_1=0.27,
            external_weight_2=0.36,
            vertex_sign_1=1,
            vertex_sign_2=1,
        )
        opposite = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=same.internal_weight_1,
            internal_weight_2=same.internal_weight_2,
            external_weight_1=same.external_weight_1,
            external_weight_2=same.external_weight_2,
            vertex_sign_1=1,
            vertex_sign_2=-1,
        )
        self.assertAlmostEqual(
            same.ground_coefficient(parity_1, identity_2),
            0.0,
            places=14,
        )
        self.assertAlmostEqual(
            opposite.ground_coefficient(parity_1, identity_2),
            2.0,
            places=14,
        )

    def test_ramond_identity_puncture_reduces_to_one_point_ground_trace(self):
        c = 13.5
        h = c / 24.0 + 0.31
        plumbing_1 = RamondPlumbingParameter(0.07, "identity")
        plumbing_2 = RamondPlumbingParameter(0.05, "identity")
        for sign in (-1, 1):
            block = RamondTorusTwoPointGroundBlock(
                central_charge=c,
                internal_weight_1=h,
                internal_weight_2=h,
                external_weight_1=0.0,
                external_weight_2=0.36,
                vertex_sign_1=1,
                vertex_sign_2=sign,
            )
            fiber = block.ground_fiber_1
            one_point = fiber.contract(
                fiber.even_vertex(sign),
                "identity",
            )
            self.assertAlmostEqual(
                block.ground_coefficient(plumbing_1, plumbing_2),
                one_point,
                places=14,
            )

    def test_ramond_ground_necklace_is_cyclic(self):
        plumbing_1 = RamondPlumbingParameter(0.07, "identity")
        plumbing_2 = RamondPlumbingParameter(0.05, "identity")
        block = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=13.5 / 24.0 + 0.31,
            internal_weight_2=13.5 / 24.0 + 0.47,
            external_weight_1=0.27,
            external_weight_2=0.36,
            vertex_sign_1=1,
            vertex_sign_2=-1,
        )
        exchanged = RamondTorusTwoPointGroundBlock(
            central_charge=13.5,
            internal_weight_1=block.internal_weight_2,
            internal_weight_2=block.internal_weight_1,
            external_weight_1=block.external_weight_2,
            external_weight_2=block.external_weight_1,
            vertex_sign_1=block.vertex_sign_2,
            vertex_sign_2=block.vertex_sign_1,
        )
        self.assertAlmostEqual(
            block.chiral_block(plumbing_1, plumbing_2),
            exchanged.chiral_block(plumbing_2, plumbing_1),
            places=14,
        )

    def test_type0b_ns_one_point_modular_s_frame_change(self):
        result = run_type0b_ns_modular_s_check(
            tau=0.2 + 0.9j,
            external_momentum=0.33,
            max_twice_level=12,
            p_max=4.5,
            quadrature_order=40,
            structure_precision=35,
            finite_part_samples=24,
        )
        self.assertNotEqual(result.q, result.q_tilde)
        self.assertEqual(result.max_twice_level, 12)
        self.assertEqual(
            (result.lift_sign, result.lift_sign_tilde), (1, 1)
        )
        self.assertLess(abs(result.relative_error), 2.0e-12)
        self.assertAlmostEqual(
            result.value_q.real, 0.0546233246078, places=12
        )

    def test_order_twelve_convergence_ledger(self):
        results = run_type0b_ns_modular_s_convergence(
            levels=(8, 10, 12),
            quadrature_order=32,
            structure_precision=35,
            finite_part_samples=24,
        )
        self.assertEqual(
            tuple(result.max_twice_level for result in results),
            (8, 10, 12),
        )
        self.assertLess(
            abs(results[-1].value_q - results[-2].value_q)
            / abs(results[-1].value_q),
            1.0e-12,
        )

    def test_modular_s_frame_change_preserves_nonprincipal_ns_lift(self):
        results = run_type0b_ns_modular_s_convergence(
            levels=(6, 8, 10, 12),
            tau=0.45 + 0.65j,
            quadrature_order=40,
            structure_precision=35,
            finite_part_samples=24,
        )
        self.assertTrue(
            all(
                (result.lift_sign, result.lift_sign_tilde) == (1, -1)
                for result in results
            )
        )
        residuals = tuple(abs(result.relative_error) for result in results)
        self.assertTrue(
            all(
                later < earlier
                for earlier, later in zip(residuals, residuals[1:])
            )
        )
        self.assertLess(residuals[-1], 2.0e-10)


if __name__ == "__main__":
    unittest.main()
