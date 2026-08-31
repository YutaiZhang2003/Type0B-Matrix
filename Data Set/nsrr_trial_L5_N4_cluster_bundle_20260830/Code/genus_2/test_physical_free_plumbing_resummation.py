#!/usr/bin/env python3
"""Focused tests for the physical free-superfield plumbing resummation."""

from __future__ import annotations

import cmath
import unittest

from physical_free_plumbing_resummation import (
    glasses_boson_loop_gaussian,
    glasses_charged_boson_resummation,
    glasses_physical_fermion_fredholm,
    physical_superfield_plumbing_partition,
    sphere_boson_kernel,
    sphere_fermion_kernel,
    theta_boson_loop_gaussian,
    theta_charged_boson_resummation,
    theta_physical_fermion_fredholm,
)

from free_boson_plumbing import glasses_free_boson_product, theta_free_boson_product
from free_majorana_pair_of_pants import (
    glasses_majorana_plumbing_partition,
    majorana_three_point,
    theta_majorana_plumbing_partition,
)


class PhysicalFreePlumbingResummationTests(unittest.TestCase):
    def test_sphere_kernel_is_antisymmetric(self) -> None:
        kernel, indices = sphere_fermion_kernel(7)
        self.assertEqual(kernel.shape, (21, 21))
        self.assertEqual(len(indices), 21)
        self.assertTrue((kernel.T == -kernel).all())

    def test_physical_bra_reversal_sign_is_not_hidden_in_kernel(self) -> None:
        kernel, indices = sphere_fermion_kernel(2)
        selected = [
            index
            for index, slot_mode in enumerate(indices)
            if slot_mode in ((0, 1), (0, 2), (2, 1), (2, 2))
        ]
        minor = kernel[selected][:, selected]
        generating_pfaffian = (
            minor[0, 1] * minor[2, 3]
            - minor[0, 2] * minor[1, 3]
            + minor[0, 3] * minor[1, 2]
        )
        physical_rho = majorana_three_point((1, 2), (), (1, 2))
        self.assertEqual(generating_pfaffian, -1.0 + 0.0j)
        self.assertEqual(physical_rho, 1)
        self.assertEqual(physical_rho, -generating_pfaffian)

    def test_boson_sphere_kernel_is_symmetric(self) -> None:
        kernel, indices = sphere_boson_kernel(7)
        self.assertEqual(kernel.shape, (21, 21))
        self.assertEqual(len(indices), 21)
        self.assertTrue((kernel.T == kernel).all())

    def test_vacuum_term_is_one(self) -> None:
        result = theta_physical_fermion_fredholm(
            (0.0, 0.0, 0.0),
            (1, -1, 1),
            max_mode=4,
        )
        self.assertEqual(result.chiral_value, 1.0 + 0.0j)
        self.assertEqual(result.nonchiral_value, 1.0)

    def test_fredholm_matches_direct_human_sewing(self) -> None:
        q_values = (0.004 + 0.001j, 0.006 - 0.0005j, 0.008 + 0.0007j)
        for lifts in ((1, 1, 1), (1, -1, 1), (-1, 1, -1)):
            with self.subTest(lifts=lifts):
                fredholm = theta_physical_fermion_fredholm(
                    q_values,
                    lifts,
                    max_mode=14,
                )
                direct = theta_majorana_plumbing_partition(
                    *q_values,
                    max_total_twice_level=24,
                    lifts=lifts,
                )
                self.assertLess(
                    abs(direct.chiral_value / fredholm.chiral_value - 1.0),
                    2.0e-12,
                )

    def test_boson_gaussian_vacuum_matches_primitive_product(self) -> None:
        q_values = (0.004 + 0.001j, 0.006 - 0.0005j, 0.008 + 0.0007j)
        gaussian = theta_charged_boson_resummation(
            q_values,
            alpha_zero=0.0,
            alpha_one=0.0,
            max_mode=10,
        )
        primitive = theta_free_boson_product(
            *q_values,
            max_word_length=8,
            max_mode=50,
            tolerance=1.0e-15,
        )
        expected = complex(cmath.exp(primitive.chiral_log_product))
        self.assertLess(abs(gaussian.vacuum_chiral / expected - 1.0), 2.0e-14)

    def test_loop_matrix_reproduces_charged_gaussian(self) -> None:
        import math
        import numpy as np

        q_values = (0.004 + 0.001j, 0.006 - 0.0005j, 0.008 + 0.0007j)
        loop = theta_boson_loop_gaussian(q_values, max_mode=10)
        matrix = np.asarray(loop.charge_quadratic_matrix)
        alpha = np.asarray((0.37, -0.21))
        charged = theta_charged_boson_resummation(
            q_values,
            alpha_zero=float(alpha[0]),
            alpha_one=float(alpha[1]),
            max_mode=10,
        )
        vacuum = theta_charged_boson_resummation(
            q_values,
            alpha_zero=0.0,
            alpha_one=0.0,
            max_mode=10,
        )
        direct_log = 2.0 * math.log(abs(charged.chiral_value / vacuum.chiral_value))
        quadratic_log = -math.pi * float(alpha @ matrix @ alpha)
        self.assertAlmostEqual(direct_log, quadratic_log, places=13)

    def test_glasses_fermion_fredholm_matches_direct_sewing(self) -> None:
        q_values = (0.001 + 0.00003j, 0.0012 - 0.00002j, 0.0008 + 0.00001j)
        for lifts in ((1, 1, 1), (1, -1, 1), (-1, 1, -1)):
            with self.subTest(lifts=lifts):
                fredholm = glasses_physical_fermion_fredholm(
                    q_values,
                    lifts,
                    max_mode=12,
                )
                direct = glasses_majorana_plumbing_partition(
                    *q_values,
                    max_total_twice_level=28,
                    lifts=lifts,
                )
                self.assertLess(
                    abs(direct.chiral_value / fredholm.chiral_value - 1.0),
                    2.0e-12,
                )

    def test_glasses_boson_gaussian_vacuum_matches_primitive_product(self) -> None:
        q_values = (0.004 + 0.0002j, 0.003 - 0.0001j, 0.002 + 0.0001j)
        gaussian = glasses_charged_boson_resummation(
            q_values,
            alpha_left=0.0,
            alpha_right=0.0,
            max_mode=10,
        )
        primitive = glasses_free_boson_product(
            *q_values,
            max_word_length=8,
            max_mode=50,
            tolerance=1.0e-15,
        )
        expected = complex(cmath.exp(primitive.chiral_log_product))
        self.assertLess(abs(gaussian.vacuum_chiral / expected - 1.0), 2.0e-14)

    def test_glasses_loop_matrix_reproduces_charged_gaussian(self) -> None:
        import math
        import numpy as np

        q_values = (0.004 + 0.0002j, 0.003 - 0.0001j, 0.002 + 0.0001j)
        loop = glasses_boson_loop_gaussian(q_values, max_mode=10)
        matrix = np.asarray(loop.charge_quadratic_matrix)
        alpha = np.asarray((0.31, -0.17))
        charged = glasses_charged_boson_resummation(
            q_values,
            alpha_left=float(alpha[0]),
            alpha_right=float(alpha[1]),
            max_mode=10,
        )
        vacuum = glasses_charged_boson_resummation(
            q_values,
            alpha_left=0.0,
            alpha_right=0.0,
            max_mode=10,
        )
        direct_log = 2.0 * math.log(abs(charged.chiral_value / vacuum.chiral_value))
        quadratic_log = -math.pi * float(alpha @ matrix @ alpha)
        self.assertAlmostEqual(direct_log, quadratic_log, places=13)

    def test_complete_partition_is_product_of_unpacked_factors(self) -> None:
        samples = {
            "theta": (
                (0.004 + 0.001j, 0.006 - 0.0005j, 0.008 + 0.0007j),
                (1, -1, 1),
            ),
            "glasses": (
                (0.004 + 0.0002j, 0.003 - 0.0001j, 0.002 + 0.0001j),
                (1, 1, 1),
            ),
        }
        for channel, (q_values, lifts) in samples.items():
            with self.subTest(channel=channel):
                result = physical_superfield_plumbing_partition(
                    channel,
                    q_values,
                    lifts,
                    max_mode=12,
                )
                unpacked = (
                    result.loop_gaussian
                    * result.boson_nonchiral_oscillator
                    * result.fermion_nonchiral_oscillator
                )
                self.assertAlmostEqual(result.one_superfield_value, unpacked)
                self.assertAlmostEqual(
                    result.nine_superfield_value,
                    result.one_superfield_value**9,
                )


if __name__ == "__main__":
    unittest.main()
