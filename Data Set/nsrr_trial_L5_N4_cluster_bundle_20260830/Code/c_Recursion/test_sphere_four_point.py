"""Tests for the BRY nonchiral genus-zero four-point layer."""

import unittest

import mpmath

from plot_bry_figure4 import BRYFigure4Benchmark
from sphere_four_point import (
    BRYFourTachyonSphere,
    BRYNSFourPointCorrelator,
)
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
    upsilon_1,
)
from superconformal_blocks import (
    HighPrecisionNSSphereFourPointBlock,
)


class StructureConstantTests(unittest.TestCase):
    def test_upsilon_reflection_and_shift(self):
        x = 0.3 + 0.2j
        self.assertAlmostEqual(upsilon_1(2 - x), upsilon_1(x), places=13)
        shifted = upsilon_1(x + 1)
        expected = (
            mpmath.gamma(x)
            / mpmath.gamma(1 - x)
            * upsilon_1(x)
        )
        self.assertAlmostEqual(shifted, complex(expected), places=13)

    def test_bry_real_structure_constant_anchor(self):
        c_value = ns_structure_constant(0.5, 1.0 / 3.0, 0.7)
        ct_value = ns_tilde_structure_constant(0.5, 1.0 / 3.0, 0.7)
        self.assertAlmostEqual(c_value.real, 0.6114166451496762, places=13)
        self.assertAlmostEqual(ct_value.real, 0.12291106222161676, places=13)
        self.assertAlmostEqual(c_value.imag, 0.0, places=13)
        self.assertAlmostEqual(ct_value.imag, 0.0, places=13)
        self.assertAlmostEqual(
            ns_structure_constant(0.7, 0.5, 1.0 / 3.0), c_value, places=13
        )


class HighPrecisionBlockTests(unittest.TestCase):
    def test_threshold_cancellation_at_eighth_order(self):
        # This is the first 32-point Gauss--Legendre node on [0,5].  At this
        # momentum, the level-15/2 recursion contains O(10^12) terms which
        # cancel.  The high-precision backend resolves the finite answer.
        momentum = 0.006840345376296353
        c = 13.5 + 1.0e-5
        q_squared = c / 3.0 - 0.5

        def weight(p):
            return 0.5 * (q_squared / 4.0 + p * p)

        block = HighPrecisionNSSphereFourPointBlock(
            c=c,
            h1=weight(0.5),
            h2=weight(1.0 / 3.0),
            h3=weight(0.25),
            h4=weight(0.6),
            internal_weight=weight(momentum),
            star2=True,
            star3=True,
            working_precision=60,
        )
        self.assertAlmostEqual(
            float(block.coefficient(15).real),
            -0.484272615796271,
            places=12,
        )


class CrossingPlotConventionTests(unittest.TestCase):
    def test_plot_uses_bry_cutoffs_and_shifted_c_exponent(self):
        benchmark = BRYFigure4Benchmark()
        self.assertEqual(benchmark.direct.bry_q_order, 8)
        self.assertEqual(benchmark.crossed.bry_q_order, 12)
        self.assertEqual(benchmark.quadrature_order, 24)
        self.assertAlmostEqual(
            benchmark.crossing_exponent,
            -4.063612777777777,
            places=14,
        )


class FourPointCorrelatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.correlator = BRYNSFourPointCorrelator(
            p1=0.5,
            p2=1.0 / 3.0,
            p3=0.25,
            p4=0.6,
            block_order=8,
        )
        cls.z = 0.1
        cls.values = cls.correlator.evaluate(
            cls.z, p_max=5.0, quadrature_order=20
        )

    def test_bry_sample_correlators_as_functions_of_z(self):
        self.assertAlmostEqual(self.values.G.real, 0.6251752342906804, places=10)
        self.assertAlmostEqual(self.values.H.real, -0.30827877654322783, places=10)
        self.assertAlmostEqual(self.values.J.real, -0.412543792684698, places=10)
        self.assertLess(abs(self.values.G.imag), 1.0e-12)
        self.assertLess(abs(self.values.H.imag), 1.0e-12)
        self.assertLess(abs(self.values.J.imag), 1.0e-12)

    def test_h_only_evaluator_matches_full_contraction(self):
        h_only = self.correlator.evaluate_h(
            self.z, p_max=5.0, quadrature_order=20
        )
        self.assertAlmostEqual(h_only.real, self.values.H.real, places=13)
        self.assertAlmostEqual(h_only.imag, self.values.H.imag, places=13)

    def test_g_only_evaluator_matches_full_contraction(self):
        g_only = self.correlator.evaluate_g(
            self.z, p_max=5.0, quadrature_order=20
        )
        self.assertAlmostEqual(g_only.real, self.values.G.real, places=13)
        self.assertAlmostEqual(g_only.imag, self.values.G.imag, places=13)

    def test_separated_g_families_sum_to_full_contraction(self):
        components = self.correlator.evaluate_g_components_grid(
            (self.z,), p_max=5.0, quadrature_order=20
        )[0]
        self.assertAlmostEqual(components.total.real, self.values.G.real, places=13)
        self.assertAlmostEqual(components.total.imag, self.values.G.imag, places=13)
        self.assertNotEqual(components.wrong_relative_sign, components.total)

    def test_four_tachyon_combination(self):
        sphere = BRYFourTachyonSphere(
            omega=0.6,
            omega1=0.5,
            omega2=1.0 / 3.0,
            omega3=0.25,
            block_order=8,
        )
        actual = sphere.combine_correlators(self.z, self.values)
        kinematic = abs(self.z) ** (-1.0 / 3.0) * abs(1 - self.z) ** (-1.0 / 6.0)
        expected = kinematic * (
            (1.0 / 3.0) ** 2 * 0.25**2 / abs(1 - self.z) ** 2 * self.values.G
            - self.values.H
            - (1.0 / 3.0) * 0.25 * self.values.J
        )
        self.assertAlmostEqual(actual, expected, places=13)
        self.assertGreater(actual.real, 0.0)

    def test_vwwv_crossing_at_complex_z(self):
        # BRY Figure 4 checks the same 2<->3 crossing relation.  A small
        # imaginary part fixes the conjugate boundary values when 1/z lies
        # beyond the real branch cut.
        z = 0.1 + 0.03j
        direct = BRYNSFourPointCorrelator(
            p1=0.5, p2=1.0 / 3.0, p3=0.25, p4=0.6, block_order=6
        )
        crossed = BRYNSFourPointCorrelator(
            p1=0.5, p2=0.25, p3=1.0 / 3.0, p4=0.6, block_order=6
        )
        lhs = direct.correlator("H", z, p_max=5.0, quadrature_order=16)
        rhs_block = crossed.correlator("H", 1 / z, p_max=5.0, quadrature_order=16)
        h1, h2, h3, h4 = [
            direct.block_weight(momentum)
            for momentum in (0.5, 1.0 / 3.0, 0.25, 0.6)
        ]
        factor = abs(z) ** (2 * (h4 - (h3 + 0.5) - (h2 + 0.5) - h1))
        rhs = factor * rhs_block
        self.assertLess(abs(lhs - rhs) / abs(lhs), 3.0e-3)

    def test_crossed_channel_quadrature_converges_near_threshold(self):
        z = 0.01
        crossed = BRYNSFourPointCorrelator(
            p1=0.5,
            p2=0.25,
            p3=1.0 / 3.0,
            p4=0.6,
            block_order=8,
        )
        h1, h2, h3, h4 = [
            crossed.block_weight(momentum)
            for momentum in (0.5, 1.0 / 3.0, 0.25, 0.6)
        ]
        exponent = 2 * (h4 - (h3 + 0.5) - (h2 + 0.5) - h1)

        def transformed(order):
            value = crossed.evaluate_h(
                1 / z + 1.0e-8j,
                p_max=5.0,
                quadrature_order=order,
            )
            return -(z**exponent * value).real

        order_20 = transformed(20)
        order_32 = transformed(32)
        self.assertLess(abs(order_32 - order_20), 1.0e-5)
        self.assertAlmostEqual(order_32, 20.44246182840761, places=8)


if __name__ == "__main__":
    unittest.main()
