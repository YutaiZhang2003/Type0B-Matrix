#!/usr/bin/env python3
"""Focused checks for the Spin(23) NS sphere-block implementation.

These tests are deliberately independent of a full momentum/moduli integral.
They verify the finite-level conformal-block object that such an integral
would consume.
"""

from __future__ import annotations

import math
import unittest

from spin23_general_kinematics import Kinematics, assemble_channel_densities
from spin23_ns_sphere_blocks import (
    build_spin23_block_bundle,
    channel_coordinates,
    direct_sphere_block_series,
    primary_external_words,
    recursive_sphere_block_series,
    relative_series_difference,
    two_star_external_words,
)


class Spin23NSSphereBlockTests(unittest.TestCase):
    """Check normalization, recursion, descendants, and channel routing."""

    def setUp(self) -> None:
        self.parameters = {
            "c": 13.5,
            "h_internal": 1.37,
            "external_weights": (0.41, 0.63, 0.82, 1.11),
            "maximum_twice_level": 8,
        }

    def test_primary_leading_coefficients(self) -> None:
        """Check the level-zero and first global NS coefficients."""

        even = direct_sphere_block_series(**self.parameters, component="even")
        odd = direct_sphere_block_series(**self.parameters, component="odd")
        h1, h2, h3, h4 = self.parameters["external_weights"]
        hp = self.parameters["h_internal"]

        expected_l_minus_one = (hp + h2 - h1) * (hp + h3 - h4) / (2 * hp)
        self.assertAlmostEqual(even.coefficients[0], 1.0)
        self.assertAlmostEqual(even.coefficients[2], expected_l_minus_one)
        self.assertAlmostEqual(odd.coefficients[1], 1 / (2 * hp))

    def test_primary_recursion_matches_inverse_gram_definition(self) -> None:
        """Compare both NS components coefficient by coefficient."""

        for component in ("even", "odd"):
            with self.subTest(component=component):
                direct = direct_sphere_block_series(
                    **self.parameters,
                    component=component,
                )
                recursive = recursive_sphere_block_series(
                    **self.parameters,
                    component=component,
                )
                self.assertLess(relative_series_difference(direct, recursive), 1.0e-12)

    def test_recursion_matches_for_complex_weights(self) -> None:
        """Exercise the analytically continued kinematic regime."""

        parameters = {
            "c": 13.5,
            "h_internal": 0.8 + 0.07j,
            "external_weights": (
                0.48 + 0.03j,
                0.62 - 0.02j,
                0.91 + 0.04j,
                1.15 - 0.05j,
            ),
            "maximum_twice_level": 6,
        }
        for component in ("even", "odd"):
            with self.subTest(component=component):
                direct = direct_sphere_block_series(**parameters, component=component)
                recursive = recursive_sphere_block_series(**parameters, component=component)
                self.assertLess(relative_series_difference(direct, recursive), 1.0e-12)

    def test_two_star_recursion_matches_direct_descendant_tensor(self) -> None:
        """The two-star seed and residues must be descendant-specific."""

        words = two_star_external_words()
        starred = direct_sphere_block_series(
            **self.parameters,
            component="even",
            external_words=words,
        )
        self.assertAlmostEqual(starred.coefficients[0], 1.0)
        recursive = recursive_sphere_block_series(
            **self.parameters, component="even", external_words=words,
        )
        self.assertLess(relative_series_difference(starred, recursive), 1.0e-12)

    def test_t_channel_coordinate_and_external_order(self) -> None:
        """Check the standardized ``x=1-z`` crossed-channel routing."""

        z = 0.31 + 0.17j
        weights = self.parameters["external_weights"]
        words = two_star_external_words()
        x, reordered_weights, reordered_words = channel_coordinates(
            z,
            weights,
            words,
            channel="t",
        )
        self.assertAlmostEqual(x, 1 - z)
        self.assertEqual(reordered_weights, (weights[2], weights[1], weights[0], weights[3]))
        self.assertEqual(reordered_words, (words[2], words[1], words[0], words[3]))

    def test_bundle_supplies_channel_density_inputs(self) -> None:
        """Smoke-test the interface to the unequal-energy assembly module."""

        bundle = build_spin23_block_bundle(**self.parameters, method="recursion")
        self.assertEqual(bundle.even_star.method, "recursion")
        self.assertEqual(bundle.odd_star.method, "recursion")
        z = 0.23 + 0.14j
        blocks = bundle.values_at_z(z)
        kin = Kinematics(0.10 + 0.15j, 0.17 + 0.18j, 0.23 + 0.20j)
        densities = assemble_channel_densities(kin, z, 0.7 - 0.2j, 0.3 + 0.1j, blocks)
        for value in (
            densities.G,
            densities.K,
            densities.common,
            densities.M1,
            densities.M2,
            densities.M3,
        ):
            self.assertTrue(math.isfinite(value.real))
            self.assertTrue(math.isfinite(value.imag))

    def test_primary_word_helper_is_four_empty_words(self) -> None:
        self.assertEqual(primary_external_words(), ((), (), (), ()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
