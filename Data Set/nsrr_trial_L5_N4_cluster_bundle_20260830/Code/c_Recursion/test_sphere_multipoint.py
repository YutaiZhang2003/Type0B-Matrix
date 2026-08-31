"""Tests for the Type-0B NS sphere multipoint correlator assembly."""

import unittest

from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from sphere_multipoint import (
    BRYNSSphereMultipointCorrelator,
    sphere_comb_frame,
)
from superconformal_blocks import HighPrecisionNSSphereFourPointBlock


class SphereMultipointCorrelatorTests(unittest.TestCase):
    def test_reduced_series_restores_the_known_four_point_z_block(self):
        central_charge = 14.19870372000744
        external_weights = (0.37, 0.61, 0.48, 0.29)
        internal_weight = 0.73
        z = 0.23
        reference = HighPrecisionNSSphereFourPointBlock(
            c=central_charge,
            h1=external_weights[0],
            h2=external_weights[1],
            h3=external_weights[2],
            h4=external_weights[3],
            internal_weight=internal_weight,
            working_precision=70,
        )
        for sectors, parity in (((0, 0), "even"), ((1, 1), "odd")):
            block = NSSphereLinearCRecursion(
                central_charge=central_charge,
                external_weights=external_weights,
                internal_weights=(internal_weight,),
                vertex_sectors=sectors,
                working_precision=70,
            )
            reduced = block.series_value((z,), (7,))
            value = z ** (internal_weight - sum(external_weights[:2])) * reduced
            with self.subTest(sectors=sectors):
                self.assertLess(
                    abs(value - reference.z_block(z, 4, parity)),
                    2.0e-14,
                )

    def test_comb_coordinates_for_clustered_five_points(self):
        points = (0.0, 0.05, 0.1, 1.0, 2.0)
        weights = (0.5,) * 5
        direct = sphere_comb_frame(
            points=points,
            weights=weights,
            order=(0, 1, 2, 3, 4),
        )
        crossed = sphere_comb_frame(
            points=points,
            weights=weights,
            order=(2, 1, 0, 3, 4),
        )
        self.assertLess(abs(direct.q_values[0] - 19.0 / 39.0), 1.0e-14)
        self.assertLess(abs(direct.q_values[1] - 1.0 / 19.0), 1.0e-14)
        self.assertLess(abs(crossed.q_values[0] - 20.0 / 39.0), 1.0e-14)
        self.assertLess(abs(crossed.q_values[1] + 1.0 / 18.0), 1.0e-14)

    def test_covariance_factor_has_the_primary_scaling_law(self):
        points = (0.0, 0.07 + 0.01j, 0.13 - 0.02j, 1.0, 2.0 + 0.1j)
        weights = (0.51, 0.56, 0.61, 0.67, 0.74)
        order = (2, 1, 0, 3, 4)
        original = sphere_comb_frame(points=points, weights=weights, order=order)
        scale = 2.3 - 0.7j
        translation = -0.4 + 0.2j
        transformed = sphere_comb_frame(
            points=tuple(scale * point + translation for point in points),
            weights=weights,
            order=order,
        )
        for left, right in zip(original.q_values, transformed.q_values):
            self.assertLess(abs(left - right), 2.0e-14)
        expected_ratio = abs(scale) ** (-2.0 * sum(weights))
        self.assertLess(
            abs(
                transformed.covariance_factor / original.covariance_factor
                - expected_ratio
            ),
            2.0e-13,
        )

    def test_five_point_integrand_contains_all_four_even_sectors(self):
        correlator = BRYNSSphereMultipointCorrelator(
            momenta=(0.5, 1.0 / 3.0, 0.25, 0.6, 0.4),
            points=(0.0, 0.05, 0.1, 1.0, 2.0),
            max_twice_levels=(1, 1),
            max_total_twice_level=2,
            structure_precision=20,
            block_working_precision=40,
        )
        frame = correlator.frame((0, 1, 2, 3, 4))
        value = correlator.momentum_integrand(frame, (0.7, 0.8))
        self.assertTrue(abs(value) > 0)
        # Three all-but-one sector truncations cannot reproduce the full sum.
        ordered_momenta = tuple(correlator.momenta[index] for index in frame.order)
        partial = 0.0j
        for sectors in ((0, 0, 0), (0, 1, 1), (1, 0, 1)):
            chiral = correlator.chiral_block(frame, (0.7, 0.8), sectors)
            partial += correlator._structure_product(
                ordered_momenta, (0.7, 0.8), sectors
            ) * abs(chiral) ** 2
        partial /= 3.141592653589793**2
        self.assertGreater(abs(value - partial), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
