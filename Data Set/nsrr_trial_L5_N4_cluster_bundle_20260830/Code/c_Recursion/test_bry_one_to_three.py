"""Focused regression tests for the regulated BRY 1->3 benchmark."""

import math
import cmath
import unittest

from bry_one_to_three import BRYOneToThreeBenchmark
from superconformal_blocks import elliptic_nome


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

    def test_h_and_c_recursion_give_the_same_bry_density(self):
        common = dict(
            block_q_order=4,
            structure_precision=20,
            block_working_precision=40,
        )
        h_benchmark = BRYOneToThreeBenchmark(block_backend="h", **common)
        c_benchmark = BRYOneToThreeBenchmark(block_backend="c", **common)
        for momentum, z in (
            (0.37, 0.31 + 0.27j),
            (0.83, 0.71 + 0.16j),
        ):
            h_value = h_benchmark.direct_momentum_density(momentum, z)
            c_value = c_benchmark.direct_momentum_density(momentum, z)
            self.assertLess(
                abs(h_value - c_value),
                2.0e-13 * max(1.0, abs(c_value)),
            )

    def test_h_recursion_matches_the_low_z_boundary_expansion(self):
        benchmark = BRYOneToThreeBenchmark(
            block_q_order=6,
            block_backend="h",
            structure_precision=20,
            block_working_precision=40,
        )
        z = 6.0e-4 + 8.0e-4j
        direct = benchmark.direct_momentum_density(0.5, z)
        local = benchmark.s_local_momentum_density(0.5, z)
        self.assertLess(abs(local / direct - 1.0), 1.0e-5)

    def test_hybrid_switch_is_set_by_the_active_elliptic_nome(self):
        benchmark = BRYOneToThreeBenchmark(
            block_backend="hybrid",
            block_q_order=4,
            structure_precision=20,
            block_working_precision=40,
        )
        bulk = benchmark._hybrid_channel(0.5 + 0.3j, 0)
        self.assertLess(abs(elliptic_nome(bulk.q)), 0.3)
        self.assertEqual(
            benchmark.hybrid_atlas._selected_backend(bulk, "auto"), "h"
        )

        corner = benchmark._hybrid_channel(0.9999 + 2.0e-5j, 0)
        self.assertLess(abs(corner.q), 1.0e-3)
        self.assertEqual(
            benchmark.hybrid_atlas._selected_backend(corner, "auto"), "c"
        )

    def test_full_crossed_c_recursion_matches_bry_t_ope(self):
        benchmark = BRYOneToThreeBenchmark(
            block_backend="hybrid",
            block_q_order=4,
            structure_precision=20,
            block_working_precision=45,
        )
        z = 1.0 - 1.0e-3 * cmath.exp(0.31j)
        for momentum in (0.3, 0.7):
            full = benchmark.hybrid_momentum_density(
                momentum, z, preferred_chart=0
            )
            local = benchmark.t_local_momentum_density(momentum, z)
            self.assertLess(abs(full / local - 1.0), 2.0e-5)

    def test_hybrid_full_density_cancels_the_explicit_t_polynomial(self):
        benchmark = BRYOneToThreeBenchmark(
            block_backend="hybrid",
            block_q_order=4,
            structure_precision=20,
            block_working_precision=45,
        )
        z = 1.0 - 1.0e-3 * cmath.exp(0.31j)
        counterterm = benchmark.folded_t_counterterm_momentum_density(0.5, z)
        regulated = benchmark.regulated_folded_momentum_density(0.5, z)
        self.assertLess(abs(regulated) / abs(counterterm), 1.0e-4)

    def test_hybrid_t_cap_nome_split_preserves_the_exact_lens_area(self):
        benchmark = BRYOneToThreeBenchmark(
            block_backend="hybrid",
            epsilon=0.05,
            block_q_order=2,
            cap_angular_order=8,
            cap_radial_order=8,
            structure_precision=20,
            block_working_precision=35,
        )
        benchmark.regulated_folded_momentum_density = (
            lambda momentum, z: 1.0 + 0.0j
        )
        benchmark.integrated_t_asymptotic_remainder = (
            lambda momentum, radius: 0.5 * math.pi * radius**2
        )
        measured = benchmark._t_cap_integral(0.5)
        epsilon = benchmark.epsilon
        exact = (
            epsilon**2 * math.acos(epsilon / 2.0)
            + math.acos(1.0 - epsilon**2 / 2.0)
            - 0.5 * epsilon * math.sqrt(4.0 - epsilon**2)
        )
        self.assertLess(abs(measured - exact), 1.0e-12)

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
        # Anchor for the cusp-resolving sin(theta)=epsilon*sin(u) map in the
        # central slice of the excised z=1 disk.
        expected = 0.11536634312413474 - 0.05021924992788822j
        self.assertLess(abs(result.reduced_moduli_integral - expected), 1.0e-11)
        self.assertLess(
            abs(
                result.low_z_region_integral
                + result.bulk_region_integral
                + result.t_cap_region_integral
                - result.reduced_moduli_integral
            ),
            1.0e-14,
        )
        self.assertEqual(result.block_q_order, 2)

    def test_reduced_order_h_and_c_regulated_integrals_agree(self):
        common = dict(
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
        h_result = BRYOneToThreeBenchmark(block_backend="h", **common).evaluate()
        c_result = BRYOneToThreeBenchmark(block_backend="c", **common).evaluate()
        self.assertLess(
            abs(h_result.reduced_moduli_integral - c_result.reduced_moduli_integral),
            1.0e-13,
        )
        self.assertEqual(h_result.block_backend, "h")
        self.assertEqual(c_result.block_backend, "c")


if __name__ == "__main__":
    unittest.main()
