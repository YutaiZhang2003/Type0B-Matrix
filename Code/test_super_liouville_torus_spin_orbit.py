"""Tests for the Type-0B NS-tilde/R torus modular orbit."""

from __future__ import annotations

import math
import unittest

from super_liouville_structure_constants import rr_ns_structure_constants
from super_liouville_torus_one_point import (
    build_type0b_r_torus_channel,
    run_type0b_ns_tilde_r_modular_s_convergence,
)


class SuperLiouvilleTorusSpinOrbitTests(unittest.TestCase):
    def test_bry_ramond_spectral_normalization(self):
        momentum = 0.61
        external_momentum = 0.33
        spectral_weight = 0.17
        c_even, _ = rr_ns_structure_constants(
            momentum,
            momentum,
            external_momentum,
            precision=35,
        )
        channel = build_type0b_r_torus_channel(
            momentum=momentum,
            spectral_weight=spectral_weight,
            external_momentum=external_momentum,
            structure_precision=35,
            finite_part_samples=16,
        )
        self.assertAlmostEqual(
            channel.weighted_structure_constant,
            2.0 * spectral_weight * c_even / math.pi,
            places=14,
        )
        self.assertEqual(channel.block.sign, 1)

    def test_spin_orbit_residual_decreases_with_recursion_order(self):
        results = run_type0b_ns_tilde_r_modular_s_convergence(
            levels=(2, 4, 6),
            tau=0.2 + 0.9j,
            external_momentum=0.33,
            p_max=4.5,
            quadrature_order=24,
            structure_precision=30,
            finite_part_samples=16,
        )
        residuals = [abs(result.relative_error) for result in results]
        self.assertGreater(residuals[0], residuals[1])
        self.assertGreater(residuals[1], residuals[2])
        self.assertLess(residuals[2], 2.0e-7)
        self.assertTrue(
            all(result.ns_tilde_lift_sign == -1 for result in results)
        )
        self.assertTrue(
            all(result.max_r_level * 2 == result.max_twice_level
                for result in results)
        )


if __name__ == "__main__":
    unittest.main()
