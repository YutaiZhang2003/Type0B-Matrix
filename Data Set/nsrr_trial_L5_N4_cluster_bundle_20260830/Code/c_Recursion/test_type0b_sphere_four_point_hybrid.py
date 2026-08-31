"""Regression tests for the Type-0B four-point hybrid amplitude."""

from __future__ import annotations

import unittest
from unittest import mock

import type0b_sphere_four_point_hybrid as four_point_module
from sphere_four_point import (
    BRYFourTachyonSphere,
    HRecursiveNSSphereFourPointBlock,
)
from superconformal_blocks import HighPrecisionNSSphereFourPointBlock
from type0b_sphere_four_point_hybrid import (
    FourPointDensityComponents,
    WALL_ONE_RAY_COEFFICIENTS,
    WALL_ONE_RAY_RECTANGLE,
    Type0BSphereFourPointHybrid,
    audit_four_point_convergence,
    audit_four_point_crossing,
    certify_convergent_ray_rectangle,
    certify_residue_convergent_ray_rectangle,
    folded_unit_disk_density,
    four_point_channel_from_ordering,
    integrate_subtraction_free_four_point_component_cells,
    wall_one_momentum_rule,
)


class Type0BSphereFourPointHybridTests(unittest.TestCase):
    def test_pure_imaginary_fixed_contour_ope_domain(self):
        audit = audit_four_point_convergence(
            (0.6j, 0.6j, 0.6j), include_residues=False
        )
        self.assertTrue(audit.convergent)
        self.assertAlmostEqual(audit.minimum_margin, 0.44, places=13)
        self.assertEqual(len(audit.records), 6)

    def test_pure_imaginary_continued_contour_requires_finite_part(self):
        audit = audit_four_point_convergence(
            (0.6j, 0.6j, 0.6j), include_residues=True
        )
        self.assertFalse(audit.convergent)
        self.assertAlmostEqual(audit.minimum_margin, -1.52, places=13)
        self.assertEqual(
            {
                round(record.momentum.imag, 12)
                for record in audit.records
                if record.kind.startswith("residue")
            },
            {0.2, 0.4, 1.4},
        )
        with self.assertRaisesRegex(ValueError, "not subtraction-free"):
            Type0BSphereFourPointHybrid(
                outgoing_energies=(0.6j,) * 3,
                contour_prescription="continued",
                recursion_max_twice_level=2,
                momentum_order=2,
                momentum_maximum=3.0,
            )

    def test_tilted_continued_rectangle_is_certified(self):
        certificate = certify_convergent_ray_rectangle()
        self.assertTrue(certificate.certified)
        self.assertAlmostEqual(
            certificate.minimum_margin_lower_bound, 0.083, places=13
        )

    def test_large_wall_four_domain_has_uniform_residue_convergence(self):
        certificate = certify_residue_convergent_ray_rectangle()
        self.assertTrue(certificate.certified)
        self.assertTrue(certificate.chamber_stable)
        self.assertEqual(certificate.chamber_record_count, 30)
        self.assertEqual(certificate.continuum_record_count, 6)
        self.assertAlmostEqual(
            certificate.continuum_minimum_margin_lower_bound,
            0.051232,
            places=13,
        )
        self.assertAlmostEqual(
            certificate.minimum_wall_clearance, 0.0665, places=13
        )
        walls = {wall.wall: wall for wall in certificate.residue_walls}
        self.assertEqual(set(walls), {1, 2, 3, 4})
        expected = {
            1: (9, 0.1066, 2, 1),
            2: (9, 4.006, 4, 3),
            3: (3, 11.8406, 3, 2),
            4: (3, 14.4852, 4, 3),
        }
        for wall, (count, margin, order, logarithm) in expected.items():
            self.assertEqual(walls[wall].record_count, count)
            self.assertAlmostEqual(
                walls[wall].minimum_margin_lower_bound, margin, places=12
            )
            self.assertEqual(walls[wall].maximum_combined_pole_order, order)
            self.assertEqual(walls[wall].maximum_logarithm_power, logarithm)

    def test_unequal_ray_has_wall_one_only_convergent_domain(self):
        certificate = certify_residue_convergent_ray_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(certificate.chamber_record_count, 11)
        self.assertAlmostEqual(
            certificate.continuum_minimum_margin_lower_bound,
            0.0512,
            places=13,
        )
        self.assertAlmostEqual(
            certificate.minimum_wall_clearance, 0.0532, places=13
        )
        self.assertEqual(len(certificate.residue_walls), 1)
        wall = certificate.residue_walls[0]
        self.assertEqual(wall.wall, 1)
        self.assertEqual(wall.record_count, 5)
        self.assertAlmostEqual(
            wall.minimum_margin_lower_bound, 0.056584, places=13
        )
        self.assertEqual(wall.maximum_combined_pole_order, 2)
        self.assertEqual(wall.maximum_logarithm_power, 1)

    def test_wall_one_composite_momentum_rule_has_thirty_nodes(self):
        rule = wall_one_momentum_rule()
        self.assertEqual(len(rule), 30)
        self.assertAlmostEqual(sum(weight for _, weight in rule), 3.0, places=14)
        self.assertTrue(all(0.0 < momentum < 3.0 for momentum, _ in rule))

    def test_wall_one_rectangle_is_certified_on_positive_bry_sheet(self):
        certificate = certify_residue_convergent_ray_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
            ray_real_sign=1,
        )
        self.assertTrue(certificate.certified)
        self.assertEqual(tuple(wall.wall for wall in certificate.residue_walls), (1,))

    def test_h_coefficients_equal_c_for_bry_middle_superpartners(self):
        common = dict(
            c=13.50001,
            h1=0.8,
            h2=0.9,
            h3=1.1,
            h4=1.2,
            internal_weight=1.3,
            star2=True,
            star3=True,
            working_precision=50,
        )
        h_block = HRecursiveNSSphereFourPointBlock(**common)
        c_block = HighPrecisionNSSphereFourPointBlock(**common)
        for twice_level in range(7):
            self.assertLess(
                abs(
                    complex(h_block.coefficient(twice_level))
                    - complex(c_block.coefficient(twice_level))
                ),
                1.0e-40,
            )

    def test_standard_channel_matches_existing_bry_kernel(self):
        z = 0.5 + 0.3j
        momentum = 0.7
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="c",
            recursion_max_twice_level=6,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        channel = four_point_channel_from_ordering(
            kernel.fixed_positions(z), (3, 2, 1, 0)
        )
        measured = kernel.fixed_momentum_density(
            z, momentum, channel=channel
        )
        legacy = BRYFourTachyonSphere(
            omega=1.8j,
            omega1=0.6j,
            omega2=0.6j,
            omega3=0.6j,
            block_order=4,
            structure_precision=20,
            block_working_precision=40,
        )
        expected = legacy.combine_correlators(
            z, legacy.liouville.momentum_integrands(momentum, z)
        )
        self.assertLess(abs(measured - expected), 1.0e-12)

    def test_folded_bulk_h_and_c_routes_agree(self):
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="hybrid",
            recursion_max_twice_level=6,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        hybrid = folded_unit_disk_density(kernel, 0.5 + 0.3j)
        kernel.block_backend = "c"
        pure_c = folded_unit_disk_density(kernel, 0.5 + 0.3j)
        self.assertLess(abs(hybrid - pure_c), 1.0e-13)

    def test_fixed_contour_diagnostic_fails_crossing_gate(self):
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="c",
            recursion_max_twice_level=2,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        audit = audit_four_point_crossing(
            kernel,
            0.37 + 0.28j,
            relative_tolerance=5.0e-3,
        )
        self.assertFalse(audit.passed)
        self.assertGreater(audit.relative_spread, 0.5)
        self.assertEqual([value.backend for value in audit.values], ["c", "c"])

    def test_origin_and_infinity_collars_force_c_region(self):
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="hybrid",
            recursion_max_twice_level=2,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        calls = []

        def fake_density(z, *, channel=None, block_region="auto"):
            calls.append(block_region)
            return 0.0 + 0.0j

        with mock.patch.object(kernel, "density", side_effect=fake_density):
            folded_unit_disk_density(kernel, 0.05 + 0.02j)
        self.assertEqual(calls, ["corner", "corner"])

    def test_component_cells_use_only_verified_canonical_frames(self):
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="hybrid",
            recursion_max_twice_level=2,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        orderings = []

        def fake_components(z, *, channel=None, block_region="auto"):
            self.assertIsNotNone(channel)
            orderings.append(channel.ordering)
            return FourPointDensityComponents(1.0 + 0.0j, 2.0 + 0.0j)

        with mock.patch.object(
            kernel, "density_components", side_effect=fake_components
        ):
            result = integrate_subtraction_free_four_point_component_cells(
                kernel,
                radial_order=2,
                angular_order=4,
                replicates=2,
            )
        self.assertTrue(orderings)
        self.assertLessEqual(
            set(orderings),
            {
                (3, 2, 1, 0),
                (2, 1, 0, 3),
                (0, 2, 1, 3),
            },
        )
        self.assertLess(
            abs(result.mean - result.continuous_mean - result.residue_mean),
            1.0e-8,
        )

    def test_residue_coefficient_already_contains_dP_over_pi_measure(self):
        kernel = Type0BSphereFourPointHybrid(
            outgoing_energies=(0.6j,) * 3,
            contour_prescription="fixed",
            block_backend="c",
            recursion_max_twice_level=0,
            momentum_order=2,
            momentum_maximum=3.0,
            structure_precision=20,
            block_working_precision=40,
        )
        positions = kernel.fixed_positions(0.37 + 0.28j)
        channel = four_point_channel_from_ordering(
            positions, (3, 2, 1, 0)
        )
        pole = four_point_module.CrossedNSStructurePole(
            family="sum",
            momentum=0.2j,
            wall=1.0,
            contour_coefficient=-2.0j,
        )
        pole_calls = iter(((pole,), (), (), ()))
        with (
            mock.patch.object(
                four_point_module,
                "_positive_contour_structure_poles",
                side_effect=lambda *args, **kwargs: next(pole_calls),
            ),
            mock.patch.object(
                kernel,
                "_structure_product_laurent_coefficients",
                return_value=(1.0 + 0.0j,),
            ),
            mock.patch.object(
                kernel,
                "_sector_component_kernel",
                return_value=1.0 + 0.0j,
            ),
        ):
            measured = kernel._residue_density(
                positions, channel, block_region="corner"
            )
        self.assertEqual(measured, -2.0j)


if __name__ == "__main__":
    unittest.main()
