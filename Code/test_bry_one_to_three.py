"""Focused regression tests for the regulated BRY 1->3 benchmark."""

import math
import unittest

from bry_one_to_three import BRYOneToThreeBenchmark


class BRYOneToThreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The local formula checks do not require the production q^8 cutoff.
        cls.benchmark = BRYOneToThreeBenchmark(block_q_order=6)

    def test_complex_energy_family_and_counterterm_threshold(self):
        benchmark = self.benchmark
        self.assertEqual(benchmark.omega, 1.0 / 3.0 + 0.6j)
        self.assertEqual(benchmark.omega1, benchmark.omega / 3.0)
        expected = math.sqrt(
            1.0 + ((benchmark.omega2 + benchmark.omega3) ** 2).real
        )
        self.assertAlmostEqual(benchmark.t_threshold, expected, places=14)
        self.assertAlmostEqual(benchmark.t_threshold, 0.9430708966187976, places=14)

    def test_three_picture_raised_terms_form_the_ope_square(self):
        benchmark = self.benchmark
        momentum = 0.47
        h = 0.5 * (1.0 + momentum * momentum)
        h2 = 0.5 * (1.0 + benchmark.omega2**2)
        h3 = 0.5 * (1.0 + benchmark.omega3**2)
        c0 = h2 + h3 - h
        separated = (
            benchmark.omega2**2 * benchmark.omega3**2
            + c0**2
            + 2.0 * benchmark.omega2 * benchmark.omega3 * c0
        )
        self.assertAlmostEqual(
            benchmark.leading_t_coefficient(momentum), separated, places=14
        )

    def test_low_z_ope_matches_elliptic_recursion(self):
        benchmark = self.benchmark
        z = 6.0e-4 + 8.0e-4j
        direct = benchmark.direct_momentum_density(0.5, z)
        local = benchmark.s_local_momentum_density(0.5, z)
        self.assertLess(abs(local / direct - 1.0), 1.0e-5)

    def test_reduced_target_has_the_matrix_model_normalization(self):
        benchmark = self.benchmark
        converted = 8j * benchmark.reduced_moduli_target / math.pi
        self.assertAlmostEqual(
            converted, benchmark.matrix_amplitude_coefficient, places=14
        )

    def test_folded_counterterm_is_direct_plus_inversion_image(self):
        benchmark = self.benchmark
        momentum = 0.47
        z = 0.63 * complex(math.cos(0.71), math.sin(0.71))
        direct = benchmark.t_counterterm_momentum_density(momentum, z)
        inversion_image = (
            abs(z) ** -4
            * benchmark.t_counterterm_momentum_density(momentum, 1.0 / z)
        )
        folded = benchmark.folded_t_counterterm_momentum_density(momentum, z)
        self.assertLess(abs(folded - direct - inversion_image), 1.0e-11)

    def test_three_regions_partition_the_folded_unit_disk(self):
        benchmark = BRYOneToThreeBenchmark(
            angular_order=14,
            radial_order=14,
            cap_angular_order=14,
            cap_radial_order=10,
            block_q_order=2,
            structure_precision=20,
            block_working_precision=35,
        )
        benchmark.direct_momentum_density = lambda momentum, z: 1.0 + 0.0j
        benchmark.s_local_momentum_density = lambda momentum, z: 1.0 + 0.0j
        benchmark.t_local_momentum_density = lambda momentum, z: 1.0 + 0.0j
        benchmark.folded_t_counterterm_momentum_density = (
            lambda momentum, z: 0.0j
        )
        partition = benchmark.z_integral_at_momentum(0.5)
        self.assertLess(abs(partition - 2.0 * math.pi), 1.0e-6)

    def test_leading_t_channel_power_is_cancelled(self):
        benchmark = self.benchmark
        momentum = 0.5
        phase = complex(math.cos(0.31), math.sin(0.31))

        def relative_remainder(radius):
            z = 1.0 - radius * phase
            counterterm = benchmark.folded_t_counterterm_momentum_density(
                momentum, z
            )
            regulated = (
                2.0 * benchmark.t_local_momentum_density(momentum, z)
                - counterterm
            )
            return abs(regulated) / abs(counterterm)

        coarse = relative_remainder(1.0e-2)
        fine = relative_remainder(1.0e-3)
        self.assertLess(fine, 1.0e-4)
        self.assertLess(fine, coarse / 5.0)

    def test_reduced_order_end_to_end_anchor(self):
        benchmark = BRYOneToThreeBenchmark(
            epsilon=0.05,
            p_max=1.2,
            p_quadrature_order=2,
            angular_order=2,
            radial_order=2,
            cap_angular_order=2,
            cap_radial_order=2,
            block_q_order=2,
            structure_precision=20,
            block_working_precision=35,
        )
        result = benchmark.evaluate()
        expected = 0.11443650226495594 - 0.05003034229128169j
        self.assertLess(abs(result.reduced_moduli_integral - expected), 1.0e-11)
        self.assertEqual(result.block_q_order, 2)


if __name__ == "__main__":
    unittest.main()
