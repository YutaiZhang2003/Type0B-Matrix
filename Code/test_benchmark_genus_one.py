"""Checks for the first noncritical type 0B benchmark."""

import math
import unittest

from benchmark_genus_one import (
    bry_0a_genus_one_density,
    bry_0a_genus_one_free_energy,
    bry_0a_momentum_density,
    bry_0a_odd_spin_density,
    bry_0a_radius_from_0b,
    bry_0a_winding_density,
    bry_circle_modular_integral,
    bry_even_spin_density,
    bry_genus_one_density,
    bry_genus_one_free_energy,
    bry_liouville_volume_log,
    bry_momentum_density,
    bry_odd_spin_density,
    bry_winding_density,
    dimensionless_radius,
    dual_radius,
    load_baseline,
    torus_log_coefficient,
    torus_log_term,
    translated_liouville_volume_log,
)


class GenusOneBenchmarkTests(unittest.TestCase):
    def test_dimensionless_radius(self) -> None:
        self.assertAlmostEqual(dimensionless_radius(2.0, 2.0), 1.0)

    def test_baseline_uses_bry_conventions(self) -> None:
        config = load_baseline()
        self.assertEqual(config["convention_source"], "arXiv:2201.05621v2")
        self.assertEqual(config["alpha_prime"], 2.0)
        self.assertEqual(
            config["radii_bry_definition"],
            "rho=R_phys/ell_B",
        )
        self.assertEqual(config["omega_mm_over_omega_worldsheet"], 2.0)
        self.assertEqual(config["liouville_fermi_map_status"], "comparison_only")
        self.assertIn(2.0, config["radii_bry"])

    def test_bry_spectrum_decomposition(self) -> None:
        for radius in (0.5, 1.0, 2.0, 7.0):
            self.assertAlmostEqual(
                bry_genus_one_density(radius),
                bry_momentum_density(radius) + bry_winding_density(radius),
            )
            self.assertAlmostEqual(
                bry_genus_one_density(radius),
                (radius + 4.0 / radius) / 24.0,
            )

    def test_convergent_modular_orbit_integral(self) -> None:
        for radius in (0.5, 1.0, 2.0, 7.0):
            expected = math.pi / 3.0 * (1.0 + 2.0 / radius**2)
            self.assertAlmostEqual(
                bry_circle_modular_integral(radius),
                expected,
            )
            self.assertAlmostEqual(
                bry_even_spin_density(radius),
                3.0 * radius / (16.0 * math.pi) * expected,
            )

    def test_bry_spin_structure_decomposition(self) -> None:
        for radius in (0.5, 1.0, 2.0, 7.0):
            self.assertAlmostEqual(
                bry_even_spin_density(radius) + bry_odd_spin_density(radius),
                bry_genus_one_density(radius),
            )

    def test_0b_odd_spin_sign(self) -> None:
        self.assertAlmostEqual(bry_odd_spin_density(math.sqrt(2.0)), 0.0)
        self.assertGreater(bry_odd_spin_density(1.0), 0.0)
        self.assertLess(bry_odd_spin_density(4.0), 0.0)

    def test_0a_spectrum_and_spin_structure_decompositions(self) -> None:
        for radius in (0.5, 1.0, 2.0, 7.0):
            self.assertAlmostEqual(
                bry_0a_genus_one_density(radius),
                bry_0a_momentum_density(radius)
                + bry_0a_winding_density(radius),
            )
            self.assertAlmostEqual(
                bry_0a_genus_one_density(radius),
                (radius + 1.0 / radius) / 12.0,
            )
            self.assertAlmostEqual(
                bry_even_spin_density(radius)
                + bry_0a_odd_spin_density(radius),
                bry_0a_genus_one_density(radius),
            )
            self.assertAlmostEqual(
                bry_0a_odd_spin_density(radius),
                -bry_odd_spin_density(radius),
            )

    def test_0a_0b_t_duality(self) -> None:
        for radius_0b in (0.5, 1.0, 2.0, 7.0):
            radius_0a = bry_0a_radius_from_0b(radius_0b)
            self.assertAlmostEqual(
                bry_0a_genus_one_density(radius_0a),
                bry_genus_one_density(radius_0b),
            )
            self.assertAlmostEqual(
                bry_0a_radius_from_0b(radius_0a),
                radius_0b,
            )

    def test_0a_liouville_wall_volume(self) -> None:
        self.assertAlmostEqual(
            bry_0a_genus_one_free_energy(1.0, math.e),
            -1.0 / 6.0,
        )

    def test_bry_liouville_wall_volume(self) -> None:
        self.assertAlmostEqual(bry_liouville_volume_log(math.e), -1.0)
        self.assertAlmostEqual(
            bry_genus_one_free_energy(2.0, math.e),
            -1.0 / 6.0,
        )

    def test_volume_translation_tracks_field_and_power_maps(self) -> None:
        self.assertAlmostEqual(
            translated_liouville_volume_log(
                math.e,
                field_scale=2.0,
                liouville_power=3.0,
                b=1.5,
            ),
            -4.0,
        )

    def test_stationary_radius_anchor(self) -> None:
        for alpha_prime in (0.5, 1.0, 2.0, 7.0):
            radius = math.sqrt(2.0 * alpha_prime)
            self.assertAlmostEqual(
                torus_log_coefficient(radius, alpha_prime),
                -1.0 / 6.0,
            )

    def test_shape_is_invariant_under_radius_involution(self) -> None:
        alpha_prime = 1.7
        for radius in (0.2, 0.8, 1.0, 3.0, 11.0):
            reflected = dual_radius(radius, alpha_prime)
            self.assertAlmostEqual(
                torus_log_coefficient(radius, alpha_prime),
                torus_log_coefficient(reflected, alpha_prime),
            )
            self.assertAlmostEqual(
                dual_radius(reflected, alpha_prime),
                radius,
            )

    def test_log_term_uses_dimensionless_mu_ratio(self) -> None:
        coefficient = torus_log_coefficient(2.0, 2.0)
        self.assertAlmostEqual(
            torus_log_term(2.0, 2.0, math.e),
            coefficient,
        )

    def test_invalid_scales_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            torus_log_coefficient(0.0, 1.0)
        with self.assertRaises(ValueError):
            torus_log_coefficient(1.0, -1.0)
        with self.assertRaises(ValueError):
            torus_log_term(1.0, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
