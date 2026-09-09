"""Decisive RRRR/RRNSNS crossing and long-R low-level checks."""

import math
import unittest

from mixed_ramond_sphere_blocks import MixedRExchangeSphereFourPointBlock
from ramond_sphere_blocks import b_from_c
from ramond_sphere_correlators import (
    RRRRSphereCorrelator,
    RRNNMixedChannelCorrelator,
    SelfDualRRRRSphereCorrelator,
    SymmetricRRNNMixedChannelCorrelator,
    _spectral_legendre_interval,
    relative_crossing_error,
)


class MixedRamondSphereTests(unittest.TestCase):
    def test_bry_spectral_quadrature_normalization(self):
        p_max = 3.7
        integrated_constant = sum(
            weight
            for _, weight in _spectral_legendre_interval(8, p_max)
        )
        self.assertAlmostEqual(
            integrated_constant, p_max / math.pi, places=14
        )

    def test_long_r_level_one_recursion_matches_direct_sewing(self):
        # c=6 keeps beta_12 and beta_21 well separated, so this tests the
        # residue formula without the b=1 collision regulator.
        b = b_from_c(6.0)
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                block = MixedRExchangeSphereFourPointBlock(
                    b=b,
                    p1_ns=0.31,
                    p2_r=0.41,
                    p3_r=0.23,
                    p4_ns=0.37,
                    internal_momentum=0.70,
                    sign3=sign3,
                    sign2=sign2,
                )
                self.assertAlmostEqual(
                    block.direct_level_one_coefficient(),
                    block.recursion_level_one_coefficient(),
                    places=13,
                )

    def test_rrrr_crossing(self):
        # Legacy displaced-c calibration.  The exact-c production path has
        # its own reduced-cost regression below and a high-accuracy stress
        # ledger in stress_rrrr_crossing.py.
        direct = RRRRSphereCorrelator(
            p1=0.20,
            p2=0.35,
            p3=0.40,
            p4=0.55,
            block_order=6,
            structure_precision=30,
        )
        left = direct.evaluate(0.37, p_max=5.0, quadrature_order=16)
        right = direct.crossed().evaluate(
            0.63, p_max=5.0, quadrature_order=16
        )
        # Absolute BRY normalization: the spectral measure is dP/pi.
        self.assertAlmostEqual(left.real, 0.2029150373725, places=11)
        self.assertLess(abs(left.imag), 1.0e-12)
        self.assertLess(relative_crossing_error(left, right), 1.0e-5)

    def test_exact_c_rrrr_correlator_path(self):
        correlator = SelfDualRRRRSphereCorrelator(
            p1=0.20,
            p2=0.35,
            p3=0.40,
            p4=0.55,
            block_order=4,
            structure_precision=30,
            finite_part_samples=8,
        )
        left = correlator.evaluate(
            0.37, p_max=4.0, quadrature_order=8
        )
        right = correlator.crossed().evaluate(
            0.63, p_max=4.0, quadrature_order=8
        )
        self.assertAlmostEqual(left.real, 0.20296889878, places=9)
        self.assertLess(abs(left.imag), 1.0e-12)
        # This deliberately cheap q^4/eight-node test is only an execution
        # and normalization regression.  The production q^10/32-node scan
        # reaches 1.32e-8.
        self.assertLess(relative_crossing_error(left, right), 1.5e-2)

    def test_rrnsns_crossing_between_ns_and_r_exchange(self):
        correlator = RRNNMixedChannelCorrelator(
            p1_r=0.20,
            p2_r=0.40,
            p3_ns=0.30,
            p4_ns=0.30,
            block_order=6,
            structure_precision=30,
            central_charge_shift=1.0e-4,
        )
        left = correlator.evaluate_ns_channel(
            0.37, p_max=5.0, quadrature_order=16
        )
        right = correlator.evaluate_crossed_r_channel(
            0.37, p_max=5.0, quadrature_order=16
        )
        self.assertLess(relative_crossing_error(left, right), 5.0e-5)

    def test_rrnsns_order_12_symmetric_regulator_stress(self):
        correlator = SymmetricRRNNMixedChannelCorrelator(
            p1_r=0.20,
            p2_r=0.40,
            p3_ns=0.30,
            p4_ns=0.30,
            block_order=12,
            structure_precision=30,
            central_charge_shift=1.0e-4,
        )
        for z in (0.05, 0.95):
            left = correlator.evaluate_ns_channel(
                z, p_max=5.0, quadrature_order=24
            )
            right = correlator.evaluate_crossed_r_channel(
                z, p_max=5.0, quadrature_order=24
            )
            self.assertLess(relative_crossing_error(left, right), 2.0e-7)


if __name__ == "__main__":
    unittest.main()
