#!/usr/bin/env python3
"""Focused checks for genus-one spectral, GSO, and free-field assembly."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from spin23_genus1_amplitude import (
    EMPTY_WORD,
    G_MINUS_HALF,
    GenusOneNSState,
    evaluate_genus_one_integrand,
    evaluate_genus_one_integrand_batch,
    even_pco_components,
    ordered_necklace_coordinates,
)
from spin23_genus1_free_fields import (
    dedekind_eta,
    even_superghost_chiral_partition,
    majorana_chiral_partition,
    noncompact_boson_partition_per_unit_volume,
)
from spin23_genus1_gso import diagonal_gso_table, ho_vacuum_fermion_character
from spin23_genus1_blocks import b1_ramond_liouville_necklace_series
from spin23_genus1_spectral import (
    _ramond_endpoint_powers,
    SpectralIntegralDiagnostics,
    adaptive_gauss_laguerre_spectral_integral,
    adaptive_gauss_laguerre_spectral_integral_batch,
    gauss_kronrod_3_7_spectral_integral,
    gauss_kronrod_7_15_spectral_integral,
    gauss_laguerre_spectral_integral,
    gauss_laguerre_spectral_integral_batch,
    integrate_liouville_necklace,
    ns_liouville_necklace_momentum_integrand,
    ramond_liouville_necklace_momentum_integrand,
    tensor_spectral_integral,
)
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_chiral_structure_constant,
    rr_ns_structure_constants,
)


class SpectralQuadratureTests(unittest.TestCase):
    """Check the continuum measure and its two independent diagnostics."""

    def test_vector_laguerre_rule_shares_nodes_without_changing_values(
        self,
    ) -> None:
        scales = (1.3, 0.7)
        batch = gauss_laguerre_spectral_integral_batch(
            lambda momenta: (1.0 + 0.0j, 2.0 + 0.0j),
            output_size=2,
            gaussian_scales=scales,
            quadrature_order=5,
            coarse_order=3,
        )
        expected = 1.0 / (
            4.0 * math.pi * math.sqrt(scales[0] * scales[1])
        )
        self.assertAlmostEqual(batch[0].value, expected, places=15)
        self.assertAlmostEqual(batch[1].value, 2.0 * expected, places=15)
        self.assertEqual(batch[0].function_evaluations, 5**2 + 3**2)
        self.assertEqual(batch[1].function_evaluations, 5**2 + 3**2)

    def test_embedded_gauss_kronrod_rule_is_exact_on_low_degree_polynomial(
        self,
    ) -> None:
        p_max = 1.3
        result = gauss_kronrod_3_7_spectral_integral(
            lambda momenta: (1.0 + momenta[0] ** 2)
            * (1.0 + momenta[1] ** 4),
            dimension=2,
            p_max=p_max,
        )
        expected = (
            (p_max + p_max**3 / 3.0)
            * (p_max + p_max**5 / 5.0)
            / math.pi**2
        )
        self.assertAlmostEqual(result.value, expected, places=14)
        self.assertAlmostEqual(result.coarse_value, expected, places=14)
        self.assertLess(result.quadrature_absolute_error, 1.0e-14)
        self.assertEqual(result.function_evaluations, 7**2)
        self.assertEqual(result.quadrature_method, "gauss_kronrod_3_7")
        self.assertEqual(result.coarse_order, 3)

    def test_nested_tail_probe_adds_only_disjoint_momentum_boxes(self) -> None:
        p_max = 1.0
        extended_p_max = 2.0
        result = gauss_kronrod_3_7_spectral_integral(
            lambda momenta: 1.0 + momenta[0] ** 2 + momenta[1] ** 2,
            dimension=2,
            p_max=p_max,
            extended_p_max=extended_p_max,
        )

        def exact(square_size: float) -> float:
            return (
                square_size**2 + 2.0 * square_size**4 / 3.0
            ) / math.pi**2

        self.assertAlmostEqual(result.value, exact(p_max), places=14)
        self.assertAlmostEqual(
            result.extended_value,
            exact(extended_p_max),
            places=14,
        )
        self.assertAlmostEqual(
            result.tail_absolute_error,
            exact(extended_p_max) - exact(p_max),
            places=14,
        )
        self.assertEqual(result.function_evaluations, 7**2 + 3 * 3**2)

    def test_embedded_gauss_kronrod_7_15_reuses_all_gauss_nodes(self) -> None:
        p_max = 1.1
        result = gauss_kronrod_7_15_spectral_integral(
            lambda momenta: (1.0 + momenta[0] ** 12)
            * (1.0 + momenta[1] ** 10),
            dimension=2,
            p_max=p_max,
            extended_p_max=1.4,
        )
        expected_core = (
            (p_max + p_max**13 / 13.0)
            * (p_max + p_max**11 / 11.0)
            / math.pi**2
        )
        self.assertAlmostEqual(result.value, expected_core, places=13)
        self.assertAlmostEqual(result.coarse_value, expected_core, places=13)
        self.assertLess(result.quadrature_absolute_error, 1.0e-13)
        self.assertEqual(result.function_evaluations, 15**2 + 3 * 7**2)
        self.assertEqual(result.quadrature_method, "gauss_kronrod_7_15")
        self.assertEqual(result.coarse_order, 7)

    def test_gauss_laguerre_integrates_known_gaussian_on_full_half_lines(
        self,
    ) -> None:
        scales = (1.7, 0.8)
        result = gauss_laguerre_spectral_integral(
            lambda momenta: (1.0 + momenta[0] ** 2)
            * (1.0 + 2.0 * momenta[1] ** 4),
            gaussian_scales=scales,
            quadrature_order=8,
            coarse_order=4,
        )

        def gaussian_moment(scale: float, power: int) -> float:
            return math.gamma(power + 0.5) / (
                2.0 * math.pi * scale ** (power + 0.5)
            )

        expected = (
            gaussian_moment(scales[0], 0)
            + gaussian_moment(scales[0], 1)
        ) * (
            gaussian_moment(scales[1], 0)
            + 2.0 * gaussian_moment(scales[1], 2)
        )
        self.assertAlmostEqual(result.value, expected, places=14)
        self.assertAlmostEqual(result.coarse_value, expected, places=14)
        self.assertLess(result.quadrature_absolute_error, 1.0e-14)
        self.assertIsNone(result.p_max)
        self.assertIsNone(result.tail_absolute_error)
        self.assertEqual(result.function_evaluations, 8**2 + 4**2)
        self.assertEqual(result.quadrature_method, "gauss_laguerre")
        self.assertEqual(result.endpoint_powers, (0, 0))

    def test_gauss_laguerre_factors_linear_endpoint_zeros_exactly(self) -> None:
        scales = (1.7, 0.8)
        result = gauss_laguerre_spectral_integral(
            lambda momenta: (
                momenta[0]
                * (1.0 + momenta[0] ** 2)
                * momenta[1]
                * (1.0 + 2.0 * momenta[1] ** 4)
            ),
            gaussian_scales=scales,
            quadrature_order=4,
            coarse_order=2,
            endpoint_powers=(1, 1),
        )

        def gaussian_moment(scale: float, power: int) -> float:
            return math.gamma((power + 1.0) / 2.0) / (
                2.0 * math.pi * scale ** ((power + 1.0) / 2.0)
            )

        expected = (
            gaussian_moment(scales[0], 1)
            + gaussian_moment(scales[0], 3)
        ) * (
            gaussian_moment(scales[1], 1)
            + 2.0 * gaussian_moment(scales[1], 5)
        )
        self.assertAlmostEqual(result.value, expected, places=14)
        self.assertAlmostEqual(result.coarse_value, expected, places=14)
        self.assertLess(result.quadrature_absolute_error, 1.0e-14)
        self.assertEqual(result.endpoint_powers, (1, 1))

    def test_parallel_gauss_laguerre_matches_serial_rule(self) -> None:
        options = dict(
            gaussian_scales=(1.3, 0.9),
            quadrature_order=5,
            coarse_order=3,
            endpoint_powers=(1, 1),
        )
        integrand = lambda momenta: (
            momenta[0]
            * momenta[1]
            * (1.0 + momenta[0] ** 2 + 0.4 * momenta[1] ** 4)
        )
        serial = gauss_laguerre_spectral_integral(
            integrand,
            **options,
            workers=1,
        )
        parallel = gauss_laguerre_spectral_integral(
            integrand,
            **options,
            workers=3,
        )
        self.assertEqual(parallel.value, serial.value)
        self.assertEqual(parallel.coarse_value, serial.coarse_value)
        self.assertEqual(
            parallel.quadrature_absolute_error,
            serial.quadrature_absolute_error,
        )
        self.assertEqual(parallel.function_evaluations, 5**2 + 3**2)
        self.assertEqual(parallel.worker_processes, 3)

    def test_parallel_adaptive_laguerre_matches_serial_rule(self) -> None:
        options = dict(
            gaussian_scales=(1.3, 0.9),
            quadrature_orders=(3, 5, 7),
            relative_tolerance=1.0e-13,
            endpoint_powers=(1, 1),
        )
        integrand = lambda momenta: (
            momenta[0]
            * momenta[1]
            * math.exp(0.7 * momenta[0] + 0.4 * momenta[1])
        )
        serial = adaptive_gauss_laguerre_spectral_integral(
            integrand,
            **options,
            workers=1,
        )
        parallel = adaptive_gauss_laguerre_spectral_integral(
            integrand,
            **options,
            workers=3,
        )
        self.assertEqual(parallel.value, serial.value)
        self.assertEqual(parallel.coarse_value, serial.coarse_value)
        self.assertEqual(
            parallel.quadrature_absolute_error,
            serial.quadrature_absolute_error,
        )
        self.assertEqual(parallel.orders_evaluated, serial.orders_evaluated)
        self.assertEqual(
            parallel.function_evaluations,
            serial.function_evaluations,
        )
        self.assertEqual(parallel.worker_processes, 3)

    def test_parallel_rule_rejects_slurm_oversubscription(self) -> None:
        with patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "2"}):
            with self.assertRaisesRegex(ValueError, "exceed the Slurm"):
                gauss_laguerre_spectral_integral(
                    lambda momenta: 1.0 + momenta[0],
                    gaussian_scales=(1.0,),
                    quadrature_order=3,
                    workers=3,
                )

    def test_shared_block_correction_matches_two_independent_integrals(
        self,
    ) -> None:
        common = dict(
            sector="NS",
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            temporal_lift_sign=1,
            p_max=None,
            quadrature_order=2,
            structure_precision=40,
            spectral_method="gauss_laguerre",
            block_backend="direct",
        )
        current = integrate_liouville_necklace(
            maximum_twice_levels=2,
            **common,
        )
        lower = integrate_liouville_necklace(
            maximum_twice_levels=0,
            **common,
        )
        correction = integrate_liouville_necklace(
            maximum_twice_levels=2,
            subtract_maximum_twice_levels=0,
            **common,
        )
        self.assertAlmostEqual(
            correction.value,
            current.value - lower.value,
            places=14,
        )

    def test_liouville_dispatch_strips_exact_plumbing_gaussian(self) -> None:
        plumbing = 0.08 + 0.01j
        scale = -math.log(abs(plumbing))

        def fake_liouville(
            momenta: tuple[float, ...],
            **options: object,
        ) -> complex:
            momentum = momenta[0]
            if options.get("strip_internal_gaussian"):
                return 1.0 + momentum**2
            return math.exp(-scale * momentum**2) * (1.0 + momentum**2)

        with patch(
            "spin23_genus1_spectral.ns_liouville_necklace_momentum_integrand",
            side_effect=fake_liouville,
        ):
            result = integrate_liouville_necklace(
                sector="NS",
                external_momenta=(0.41,),
                plumbing_parameters=(plumbing,),
                holomorphic_words=(EMPTY_WORD,),
                antiholomorphic_words=(EMPTY_WORD,),
                maximum_twice_levels=0,
                temporal_lift_sign=1,
                p_max=None,
                quadrature_order=4,
                spectral_method="gauss_laguerre",
            )
        expected = (
            math.gamma(0.5) / (2.0 * math.pi * scale**0.5)
            + math.gamma(1.5) / (2.0 * math.pi * scale**1.5)
        )
        self.assertAlmostEqual(result.value, expected, places=14)
        self.assertEqual(result.function_evaluations, 4)
        self.assertEqual(result.endpoint_powers, (0,))

    def test_ramond_dispatch_uses_the_linear_endpoint_factor(self) -> None:
        plumbing = (0.08 + 0.01j, 0.06 - 0.02j)
        scales = tuple(-math.log(abs(value)) for value in plumbing)

        def fake_liouville(
            momenta: tuple[float, ...],
            **options: object,
        ) -> complex:
            reduced = math.prod(
                momentum * (1.0 + momentum**2)
                for momentum in momenta
            )
            if options.get("strip_internal_gaussian"):
                return reduced
            return math.exp(
                -sum(
                    scale * momentum**2
                    for scale, momentum in zip(scales, momenta)
                )
            ) * reduced

        with patch(
            "spin23_genus1_spectral.ramond_liouville_necklace_momentum_integrand",
            side_effect=fake_liouville,
        ):
            result = integrate_liouville_necklace(
                sector="R",
                external_momenta=(0.41, 0.37),
                plumbing_parameters=plumbing,
                holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
                antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
                maximum_twice_levels=0,
                temporal_lift_sign=1,
                p_max=None,
                quadrature_order=3,
                spectral_method="gauss_laguerre",
            )
        expected = math.prod(
            1.0 / (2.0 * math.pi * scale)
            + 1.0 / (2.0 * math.pi * scale**2)
            for scale in scales
        )
        self.assertAlmostEqual(result.value, expected, places=14)
        self.assertEqual(result.endpoint_powers, (1, 1))

    def test_two_point_ramond_endpoint_class_follows_combined_word_parity(
        self,
    ) -> None:
        g_word = G_MINUS_HALF
        self.assertEqual(
            _ramond_endpoint_powers(
                (EMPTY_WORD, EMPTY_WORD),
                (EMPTY_WORD, EMPTY_WORD),
            ),
            (1, 1),
        )
        self.assertEqual(
            _ramond_endpoint_powers(
                (g_word, g_word),
                (EMPTY_WORD, EMPTY_WORD),
            ),
            (0, 0),
        )
        self.assertEqual(
            _ramond_endpoint_powers(
                (g_word, g_word),
                (g_word, g_word),
            ),
            (1, 1),
        )
        self.assertEqual(
            _ramond_endpoint_powers(
                (EMPTY_WORD,) * 3,
                (EMPTY_WORD,) * 3,
            ),
            (0, 0, 0),
        )

    def test_adaptive_laguerre_stops_or_reports_exhausted_order(self) -> None:
        polynomial = adaptive_gauss_laguerre_spectral_integral(
            lambda momenta: (1.0 + momenta[0] ** 2)
            * (1.0 + momenta[1] ** 4),
            gaussian_scales=(1.0, 1.2),
            quadrature_orders=(7, 15, 24),
            relative_tolerance=1.0e-10,
        )
        self.assertTrue(polynomial.converged)
        self.assertEqual(polynomial.orders_evaluated, (7, 15))
        self.assertEqual(polynomial.function_evaluations, 7**2 + 15**2)

        unresolved = adaptive_gauss_laguerre_spectral_integral(
            lambda momenta: math.exp(
                1.8 * momenta[0] + 1.3 * momenta[1]
            ),
            gaussian_scales=(1.0, 1.2),
            quadrature_orders=(7, 15, 24),
            relative_tolerance=1.0e-6,
        )
        self.assertFalse(unresolved.converged)
        self.assertEqual(unresolved.orders_evaluated, (7, 15, 24))
        self.assertEqual(
            unresolved.function_evaluations,
            7**2 + 15**2 + 24**2,
        )

    def test_tensor_gaussian_uses_one_dP_over_pi_per_edge(self) -> None:
        p_max = 2.3
        result = tensor_spectral_integral(
            lambda momenta: math.exp(-sum(momentum**2 for momentum in momenta)),
            dimension=2,
            p_max=p_max,
            quadrature_order=18,
            refined_order=24,
            extended_p_max=2.8,
        )
        one_edge = math.erf(p_max) / (2.0 * math.sqrt(math.pi))
        self.assertAlmostEqual(result.value, one_edge**2, places=14)
        self.assertIsNotNone(result.quadrature_absolute_error)
        self.assertIsNotNone(result.tail_absolute_error)
        self.assertEqual(
            result.function_evaluations,
            18**2 + 24**2 + 18**2,
        )

    def test_ns_level_zero_matches_one_point_spectral_formula(self) -> None:
        momentum = 0.37
        external = 0.41
        plumbing = 0.08 + 0.01j
        actual = ns_liouville_necklace_momentum_integrand(
            (momentum,),
            external_momenta=(external,),
            plumbing_parameters=(plumbing,),
            holomorphic_words=(EMPTY_WORD,),
            antiholomorphic_words=(EMPTY_WORD,),
            maximum_twice_levels=0,
            structure_precision=55,
        )
        structure = ns_structure_constants(
            momentum,
            external,
            momentum,
            precision=55,
        )[0]
        weight = ns_weight(momentum)
        expected = (
            structure
            * plumbing ** (weight - 13.5 / 24.0)
            * plumbing.conjugate() ** (weight - 13.5 / 24.0)
        )
        self.assertAlmostEqual(actual, expected, places=13)

    def test_ramond_level_zero_matches_trusted_long_fiber_formula(self) -> None:
        momentum = 0.37
        external = 0.41
        plumbing = 0.08 + 0.01j
        with patch(
            "spin23_genus1_spectral.b1_ramond_liouville_necklace_series",
            wraps=b1_ramond_liouville_necklace_series,
        ) as block_builder:
            actual = ramond_liouville_necklace_momentum_integrand(
                (momentum,),
                external_momenta=(external,),
                plumbing_parameters=(plumbing,),
                holomorphic_words=(EMPTY_WORD,),
                antiholomorphic_words=(EMPTY_WORD,),
                maximum_twice_levels=0,
                structure_precision=55,
            )
        self.assertEqual(block_builder.call_count, 1)
        block = b1_ramond_liouville_necklace_series(
            internal_momenta=(momentum,),
            external_ns_momenta=(external,),
            structure_signs=(1,),
            maximum_twice_levels=0,
            include_structure_constants=False,
            precision=55,
        )
        structure = rr_ns_chiral_structure_constant(
            momentum,
            momentum,
            external,
            structure_sign=1,
            precision=55,
        )
        expected = (
            2.0
            * structure
            * block.value((plumbing,))
            * block.value((plumbing.conjugate(),))
        )
        self.assertAlmostEqual(actual, expected, places=13)

    def test_equal_external_two_point_reuses_symmetric_structure_data(
        self,
    ) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=0,
            structure_precision=45,
        )
        with patch(
            "spin23_genus1_spectral.ns_structure_constants",
            wraps=ns_structure_constants,
        ) as ns_builder:
            ns_liouville_necklace_momentum_integrand(**common)
        with patch(
            "spin23_genus1_spectral.rr_ns_structure_constants",
            wraps=rr_ns_structure_constants,
        ) as ramond_builder:
            ramond_liouville_necklace_momentum_integrand(**common)
        self.assertEqual(ns_builder.call_count, 1)
        self.assertEqual(ramond_builder.call_count, 1)

    def test_ns_recursive_backend_matches_direct_two_point_integrand(self) -> None:
        options = dict(
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=2,
            recursion_radius=0.035,
            recursion_check_radius=0.05,
            recursion_samples=8,
        )
        direct = ns_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="direct",
            **options,
        )
        recursive = ns_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="recursion",
            **options,
        )
        self.assertAlmostEqual(recursive, direct)

    def test_ns_direct_fast_backend_matches_direct_two_point_integrand(
        self,
    ) -> None:
        for words in (
            (EMPTY_WORD, EMPTY_WORD),
            (G_MINUS_HALF, G_MINUS_HALF),
        ):
            with self.subTest(words=words):
                options = dict(
                    external_momenta=(0.41, 0.41),
                    plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
                    holomorphic_words=words,
                    antiholomorphic_words=words,
                    maximum_twice_levels=2,
                    temporal_lift_sign=-1,
                    structure_precision=35,
                )
                direct = ns_liouville_necklace_momentum_integrand(
                    (0.37, 0.52),
                    block_backend="direct",
                    **options,
                )
                fast = ns_liouville_necklace_momentum_integrand(
                    (0.37, 0.52),
                    block_backend="direct_fast",
                    **options,
                )
                self.assertTrue(
                    np.isclose(
                        fast,
                        direct,
                        rtol=3.0e-12,
                        atol=3.0e-12,
                    )
                )

    def test_ns_recursive_gg_backend_matches_direct_two_point_integrand(
        self,
    ) -> None:
        options = dict(
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(G_MINUS_HALF, G_MINUS_HALF),
            antiholomorphic_words=(G_MINUS_HALF, G_MINUS_HALF),
            maximum_twice_levels=2,
            recursion_radius=0.035,
            recursion_check_radius=0.05,
            recursion_samples=8,
        )
        direct = ns_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="direct",
            **options,
        )
        recursive = ns_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="recursion",
            **options,
        )
        self.assertAlmostEqual(recursive, direct)

    def test_auto_backend_records_direct_fallbacks(self) -> None:
        options = dict(
            sector="NS",
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=0,
            temporal_lift_sign=1,
            p_max=0.25,
            quadrature_order=2,
            structure_precision=35,
        )
        with (
            patch(
                "spin23_genus1_spectral.generated_ns_rectangle_available",
                return_value=False,
            ),
            patch(
                "spin23_genus1_spectral."
                "recursive_ns_torus_two_point_series",
                side_effect=ZeroDivisionError("synthetic confluent pole"),
            ),
        ):
            automatic = integrate_liouville_necklace(
                block_backend="auto",
                **options,
            )
            with self.assertRaisesRegex(ZeroDivisionError, "confluent"):
                integrate_liouville_necklace(
                    block_backend="recursion",
                    **options,
                )
        self.assertEqual(automatic.block_backend_requested, "auto")
        self.assertGreater(automatic.recursion_fallbacks, 0)
        self.assertGreaterEqual(
            automatic.direct_block_evaluations,
            automatic.recursion_fallbacks,
        )
        self.assertEqual(automatic.recursive_block_evaluations, 0)

    def test_auto_backend_avoids_known_confluent_recursion_locus(self) -> None:
        options = dict(
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=2,
            block_backend="auto",
            recursion_samples=8,
        )
        with patch(
            "spin23_genus1_spectral.recursive_ns_torus_two_point_series",
            side_effect=AssertionError("recursion should not be attempted"),
        ):
            automatic = ns_liouville_necklace_momentum_integrand(
                (0.37, 0.37),
                **options,
            )
        direct = ns_liouville_necklace_momentum_integrand(
            (0.37, 0.37),
            **{**options, "block_backend": "direct"},
        )
        self.assertAlmostEqual(automatic, direct)

    def test_auto_ramond_backend_uses_conditioned_direct_at_low_level(self) -> None:
        options = dict(
            sector="R",
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=2,
            temporal_lift_sign=1,
            p_max=0.25,
            quadrature_order=2,
            structure_precision=24,
            block_backend="auto",
        )
        with patch(
            "spin23_genus1_spectral.recursive_ramond_torus_two_point_series",
            side_effect=AssertionError("low-level auto must not call recursion"),
        ):
            automatic = integrate_liouville_necklace(**options)
        oracle = integrate_liouville_necklace(
            **{**options, "block_backend": "direct"}
        )
        self.assertGreater(automatic.fast_direct_block_evaluations, 0)
        self.assertEqual(automatic.recursive_block_evaluations, 0)
        self.assertEqual(automatic.recursion_fallbacks, 0)
        self.assertTrue(
            math.isclose(
                abs(automatic.value - oracle.value),
                0.0,
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
        )

    def test_hybrid_ramond_backend_fails_closed_on_conditioning(self) -> None:
        with self.assertRaises(np.linalg.LinAlgError):
            ramond_liouville_necklace_momentum_integrand(
                (0.37, 0.52),
                external_momenta=(0.41, 0.41),
                plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
                holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
                antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
                maximum_twice_levels=2,
                block_backend="hybrid",
                condition_limit=1.0,
            )

    def test_ramond_recursive_backend_matches_direct_two_point_integrand(
        self,
    ) -> None:
        options = dict(
            external_momenta=(0.41, 0.41),
            plumbing_parameters=(0.08 + 0.01j, 0.06 - 0.02j),
            holomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            antiholomorphic_words=(EMPTY_WORD, EMPTY_WORD),
            maximum_twice_levels=0,
            recursion_radius=0.035,
            recursion_check_radius=0.05,
            recursion_samples=8,
        )
        direct = ramond_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="direct",
            **options,
        )
        recursive = ramond_liouville_necklace_momentum_integrand(
            (0.37, 0.52),
            block_backend="recursion",
            **options,
        )
        self.assertAlmostEqual(recursive, direct)

    def test_exact_internal_gaussian_strip_in_both_sectors(self) -> None:
        momentum = 0.37
        plumbing = 0.08 + 0.01j
        common = dict(
            internal_momenta=(momentum,),
            external_momenta=(0.41,),
            plumbing_parameters=(plumbing,),
            holomorphic_words=(EMPTY_WORD,),
            antiholomorphic_words=(EMPTY_WORD,),
            maximum_twice_levels=0,
            structure_precision=55,
        )
        gaussian = math.exp(math.log(abs(plumbing)) * momentum**2)
        for evaluator in (
            ns_liouville_necklace_momentum_integrand,
            ramond_liouville_necklace_momentum_integrand,
        ):
            with self.subTest(evaluator=evaluator.__name__):
                full = evaluator(**common)
                stripped = evaluator(
                    **common,
                    strip_internal_gaussian=True,
                )
                self.assertAlmostEqual(full, gaussian * stripped, places=13)


class DiagonalGSOTests(unittest.TestCase):
    """Check the diagonal projector separately from determinant phases."""

    tau = 0.19 + 1.13j

    def test_effective_even_coefficients_are_plus_minus_minus(self) -> None:
        table = diagonal_gso_table()
        self.assertEqual(
            [entry.effective_even_coefficient for entry in table[:3]],
            [0.5, -0.5, -0.5],
        )
        self.assertIsNone(table[3].effective_even_coefficient)

    def test_ho_vacuum_character_is_24(self) -> None:
        for tau in (self.tau, 0.5 + 0.93j, 2.1j):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(
                    ho_vacuum_fermion_character(tau, precision=60),
                    24.0,
                    places=11,
                )

    def test_two_majoranas_times_superghost_leave_only_trace_phase(self) -> None:
        for entry in diagonal_gso_table()[:3]:
            with self.subTest(spin=entry.spin_structure.label):
                value = majorana_chiral_partition(
                    self.tau,
                    entry.spin_structure,
                    n_fermions=2,
                    precision=60,
                ) * even_superghost_chiral_partition(
                    self.tau,
                    entry.spin_structure,
                    precision=60,
                )
                self.assertAlmostEqual(
                    value,
                    entry.even_superghost_trace_phase,
                    places=12,
                )


class FreeBosonNormalizationTests(unittest.TestCase):
    """Check the explicit noncompact zero-mode Gaussian convention."""

    def test_alpha_prime_two_matches_heterotic_torus_normalization(self) -> None:
        tau = 0.23 + 1.31j
        eta = dedekind_eta(tau, precision=60)
        expected = 1.0 / (
            math.sqrt(8.0 * math.pi**2 * tau.imag) * abs(eta) ** 2
        )
        self.assertAlmostEqual(
            noncompact_boson_partition_per_unit_volume(
                tau,
                alpha_prime=2.0,
                precision=60,
            ),
            expected,
            places=14,
        )


def _unit_liouville_evaluator(**kwargs: object) -> SpectralIntegralDiagnostics:
    """Return an exact unit correlator for inexpensive assembly tests."""

    dimension = len(tuple(kwargs["external_momenta"]))  # type: ignore[arg-type]
    return SpectralIntegralDiagnostics(
        value=1.0 + 0.0j,
        refined_value=None,
        extended_value=None,
        quadrature_absolute_error=0.0,
        tail_absolute_error=0.0,
        estimated_absolute_error=0.0,
        dimension=dimension,
        p_max=float(kwargs["p_max"]),
        quadrature_order=int(kwargs["quadrature_order"]),
        refined_order=None,
        extended_p_max=None,
        function_evaluations=0,
    )


def _unit_liouville_batch_evaluator(
    **kwargs: object,
) -> tuple[SpectralIntegralDiagnostics, ...]:
    """Return one exact unit correlator per external momentum tuple."""

    external_batch = tuple(kwargs["external_momenta_batch"])  # type: ignore[arg-type]
    return tuple(
        _unit_liouville_evaluator(
            **{
                **kwargs,
                "external_momenta": external,
            }
        )
        for external in external_batch
    )


class PhysicalAssemblyBookkeepingTests(unittest.TestCase):
    """Check channel coordinates, PCO terms, flavor, and projector routing."""

    def test_fixed_geometry_batch_matches_independent_assembly(self) -> None:
        state_sets = tuple(
            (
                GenusOneNSState.vector(energy, -energy, 4),
                GenusOneNSState.vector(energy, energy, 4),
            )
            for energy in (0.31, 0.47)
        )
        options = dict(
            maximum_twice_levels=0,
            p_max=1.0,
            quadrature_order=2,
            spin_labels=("NS",),
        )
        independent = tuple(
            evaluate_genus_one_integrand(
                states,
                0.12 + 1.18j,
                (0.0, 0.21 + 0.46j),
                liouville_evaluator=_unit_liouville_evaluator,
                **options,
            )
            for states in state_sets
        )
        batched = evaluate_genus_one_integrand_batch(
            state_sets,
            0.12 + 1.18j,
            (0.0, 0.21 + 0.46j),
            liouville_batch_evaluator=_unit_liouville_batch_evaluator,
            **options,
        )
        for expected, actual in zip(independent, batched):
            self.assertAlmostEqual(actual.value, expected.value)
            self.assertAlmostEqual(
                actual.fixed_spin["NS"].value,
                expected.fixed_spin["NS"].value,
            )

    def test_real_fixed_geometry_batch_matches_scalar_all_spin_sum(self) -> None:
        states = (
            GenusOneNSState.vector(0.31, -0.31, 4),
            GenusOneNSState.vector(0.31, 0.31, 4),
        )
        options = dict(
            maximum_twice_levels=0,
            p_max=None,
            quadrature_order=2,
            refined_order=None,
            structure_precision=24,
            block_digits=30,
            condition_limit=1.0e11,
            free_field_precision=24,
            spectral_method="gauss_laguerre",
            block_backend="direct_fast",
            spectral_workers=1,
            spin_workers=1,
        )
        scalar = evaluate_genus_one_integrand(
            states,
            0.12 + 1.18j,
            (0.0, 0.21 + 0.46j),
            **options,
        )
        batched = evaluate_genus_one_integrand_batch(
            (states,),
            0.12 + 1.18j,
            (0.0, 0.21 + 0.46j),
            **options,
        )[0]
        self.assertTrue(
            np.isclose(
                batched.value,
                scalar.value,
                rtol=2.0e-14,
                atol=2.0e-14,
            )
        )
        for label in scalar.fixed_spin:
            self.assertTrue(
                np.isclose(
                    batched.fixed_spin[label].value,
                    scalar.fixed_spin[label].value,
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )
            )

    def test_two_point_necklace_coordinates(self) -> None:
        tau = 0.17 + 1.2j
        separation = 0.21 + 0.43j
        coordinates = ordered_necklace_coordinates(
            tau,
            (0.31 + 0.07j, 0.31 + 0.07j + separation),
        )
        self.assertAlmostEqual(
            coordinates.plumbing_parameters[0],
            complex(math.e) ** (2j * math.pi * separation),
        )
        self.assertAlmostEqual(
            coordinates.plumbing_parameters[1],
            complex(math.e) ** (2j * math.pi * (tau - separation)),
        )
        self.assertAlmostEqual(
            math.prod(coordinates.plumbing_parameters),
            complex(math.e) ** (2j * math.pi * tau),
        )
        translated = ordered_necklace_coordinates(
            tau,
            (0.61 + 0.18j, 0.61 + 0.18j + separation),
        )
        for left, right in zip(
            translated.additive_points,
            coordinates.additive_points,
        ):
            self.assertAlmostEqual(left, right)
        for left, right in zip(
            translated.plumbing_parameters,
            coordinates.plumbing_parameters,
        ):
            self.assertAlmostEqual(left, right)

    def test_pco_component_count_and_signed_momentum_coefficients(self) -> None:
        two_states = (
            GenusOneNSState.singlet(0.4, -0.4),
            GenusOneNSState.vector(0.4, 0.4, 3),
        )
        two_components = even_pco_components(two_states)
        self.assertEqual(len(two_components), 2)
        self.assertEqual(two_components[0].time_fermion_indices, ())
        self.assertEqual(
            two_components[0].holomorphic_liouville_words,
            (G_MINUS_HALF, G_MINUS_HALF),
        )
        self.assertEqual(two_components[1].time_fermion_indices, (0, 1))
        self.assertEqual(
            two_components[1].holomorphic_liouville_words,
            (EMPTY_WORD, EMPTY_WORD),
        )
        self.assertAlmostEqual(two_components[1].momentum_coefficient, -0.16)

        three_states = (
            GenusOneNSState.singlet(0.3, -0.3),
            GenusOneNSState.vector(0.1, 0.1, 2),
            GenusOneNSState.vector(0.2, 0.2, 2),
        )
        self.assertEqual(len(even_pco_components(three_states)), 4)

    def test_mismatched_vector_flavors_make_two_point_integrand_zero(self) -> None:
        states = (
            GenusOneNSState.vector(0.37, -0.37, 2),
            GenusOneNSState.vector(0.37, 0.37, 7),
        )
        evaluation = evaluate_genus_one_integrand(
            states,
            0.13 + 1.07j,
            (0.0, 0.19 + 0.42j),
            maximum_twice_levels=0,
            p_max=1.0,
            quadrature_order=2,
            liouville_evaluator=_unit_liouville_evaluator,
        )
        self.assertEqual(evaluation.value, 0.0j)

    def test_diagonal_projector_is_applied_once(self) -> None:
        states = (
            GenusOneNSState.singlet(0.31, -0.31),
            GenusOneNSState.singlet(0.31, 0.31),
        )
        evaluation = evaluate_genus_one_integrand(
            states,
            0.08 + 1.16j,
            (0.0, 0.22 + 0.47j),
            maximum_twice_levels=0,
            p_max=1.0,
            quadrature_order=2,
            liouville_evaluator=_unit_liouville_evaluator,
        )
        expected = 0.5 * sum(
            fixed.value for fixed in evaluation.fixed_spin.values()
        )
        self.assertAlmostEqual(evaluation.value, expected)
        self.assertEqual(evaluation.picture_raising_factor, 0.25)

        phased = evaluate_genus_one_integrand(
            states,
            0.08 + 1.16j,
            (0.0, 0.22 + 0.47j),
            maximum_twice_levels=0,
            p_max=1.0,
            quadrature_order=2,
            include_string_phase=True,
            liouville_evaluator=_unit_liouville_evaluator,
        )
        self.assertAlmostEqual(phased.value, -evaluation.value)

    def test_disjoint_pco_component_shards_sum_to_full_fixed_spin(self) -> None:
        states = (
            GenusOneNSState.vector(0.35, -0.35, 4),
            GenusOneNSState.vector(0.35, 0.35, 4),
        )
        options = dict(
            tau=0.12 + 1.18j,
            points=(0.0, 0.21 + 0.46j),
            maximum_twice_levels=0,
            p_max=0.25,
            quadrature_order=2,
            spin_labels=("R",),
            liouville_evaluator=_unit_liouville_evaluator,
        )
        full = evaluate_genus_one_integrand(states, **options)
        shards = tuple(
            evaluate_genus_one_integrand(
                states,
                component_indices=(component_index,),
                **options,
            )
            for component_index in range(2)
        )
        self.assertEqual(full.value, sum((row.value for row in shards), 0.0j))
        self.assertEqual(
            full.fixed_spin["R"].value,
            sum((row.fixed_spin["R"].value for row in shards), 0.0j),
        )
        self.assertEqual(
            tuple(
                row.fixed_spin["R"].components[0].component.time_fermion_indices
                for row in shards
            ),
            ((), (0, 1)),
        )

    def test_full_two_point_recursive_assembly_matches_direct(self) -> None:
        states = (
            GenusOneNSState.vector(0.35, -0.35, 4),
            GenusOneNSState.vector(0.35, 0.35, 4),
        )
        options = dict(
            tau=0.12 + 1.18j,
            points=(0.0, 0.21 + 0.46j),
            maximum_twice_levels=0,
            p_max=0.25,
            quadrature_order=2,
            structure_precision=35,
            block_digits=40,
            free_field_precision=35,
            recursion_radius=0.035,
            recursion_check_radius=0.05,
            recursion_samples=8,
        )
        direct = evaluate_genus_one_integrand(
            states,
            block_backend="direct",
            **options,
        )
        recursive = evaluate_genus_one_integrand(
            states,
            block_backend="recursion",
            **options,
        )
        relative_difference = abs(recursive.value - direct.value) / abs(
            direct.value
        )
        self.assertLess(relative_difference, 1.0e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
