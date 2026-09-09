"""Regression tests for the NS sphere linear-channel h-recursion."""

import unittest
import tempfile

import mpmath

from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from ns_multipoint_h_recursion import (
    NSSphereLinearHRecursion,
    ns_b_from_c,
    ns_central_charge_from_b,
)
from sphere_multipoint import BRYNSSphereMultipointCorrelator


class MultipointNSHRecursionTests(unittest.TestCase):
    def test_b_and_c_parameterizations_are_inverse_on_physical_sheet(self):
        with mpmath.workdps(70):
            for b in (mpmath.mpf("1"), mpmath.mpf("1.17"), mpmath.mpf("1.43")):
                recovered = ns_b_from_c(ns_central_charge_from_b(b))
                self.assertLess(abs(recovered - b), mpmath.mpf("1e-65"))

    def test_four_point_coefficients_match_c_recursion_through_level_eight(self):
        with mpmath.workdps(80):
            b = mpmath.mpf("1.27")
            c_value = ns_central_charge_from_b(b)
            external = tuple(
                map(mpmath.mpf, ("0.41", "0.27", "0.36", "0.53"))
            )
            internal = (mpmath.mpf("0.71"),)
            for sectors in ((0, 0), (1, 1)):
                h_block = NSSphereLinearHRecursion(
                    b=b,
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                c_block = NSSphereLinearCRecursion(
                    central_charge=c_value,
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                parity = sectors[0]
                for twice_level in range(parity, 17, 2):
                    with self.subTest(
                        sectors=sectors, twice_level=twice_level
                    ):
                        self.assertLess(
                            abs(
                                h_block.coefficient((twice_level,))
                                - c_block.coefficient((twice_level,))
                            ),
                            mpmath.mpf("1e-70"),
                        )

    def test_five_point_coefficients_match_c_recursion_in_all_routings(self):
        with mpmath.workdps(80):
            b = mpmath.mpf("1.27")
            c_value = ns_central_charge_from_b(b)
            external = tuple(
                map(mpmath.mpf, ("0.31", "0.42", "0.53", "0.47", "0.28"))
            )
            internal = tuple(map(mpmath.mpf, ("0.73", "0.81")))
            samples = {
                (0, 0, 0): ((0, 0), (2, 2), (4, 2), (2, 4), (4, 4)),
                (0, 1, 1): ((0, 1), (2, 1), (2, 3), (4, 3)),
                (1, 0, 1): ((1, 1), (3, 1), (1, 3), (3, 3)),
                (1, 1, 0): ((1, 0), (1, 2), (3, 2), (3, 4)),
            }
            for sectors, levels_to_check in samples.items():
                h_block = NSSphereLinearHRecursion(
                    b=b,
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                c_block = NSSphereLinearCRecursion(
                    central_charge=c_value,
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                for levels in levels_to_check:
                    with self.subTest(sectors=sectors, levels=levels):
                        self.assertLess(
                            abs(
                                h_block.coefficient(levels)
                                - c_block.coefficient(levels)
                            ),
                            mpmath.mpf("2e-69"),
                        )

    def test_functional_recursion_equals_rectangular_coefficient_sum(self):
        with mpmath.workdps(70):
            block = NSSphereLinearHRecursion(
                b=mpmath.mpf("1.31"),
                external_weights=tuple(
                    map(
                        mpmath.mpf,
                        ("0.31", "0.42", "0.53", "0.47", "0.28"),
                    )
                ),
                internal_weights=tuple(map(mpmath.mpf, ("0.73", "0.81"))),
                vertex_sectors=(1, 0, 1),
                working_precision=70,
            )
            q_values = (
                mpmath.mpc("0.07", "0.02"),
                mpmath.mpc("0.19", "-0.03"),
            )
            coefficient_sum = block.series_value(
                q_values, (6, 6), max_total_twice_level=8
            )
            functional = block.recursive_series_value(
                q_values,
                8,
                maximum_accumulated_twice_levels=(6, 6),
            )
            self.assertLess(
                abs(coefficient_sum - functional), mpmath.mpf("1e-65")
            )

    def test_self_dual_limit_fits_each_coefficient_before_series_evaluation(self):
        with mpmath.workdps(70):
            etas = tuple(
                map(mpmath.mpf, ("0.16", "0.13", "0.10", "0.075", "0.055"))
            )
            common = dict(
                external_weights=tuple(
                    map(mpmath.mpf, ("0.31", "0.42", "0.53", "0.47", "0.28"))
                ),
                internal_weights=tuple(map(mpmath.mpf, ("0.73", "0.81"))),
                vertex_sectors=(1, 0, 1),
                working_precision=70,
            )
            block = NSSphereLinearHRecursion(
                central_charge=mpmath.mpf("13.5"),
                self_dual_log_b_nodes=etas,
                self_dual_polynomial_degree=3,
                self_dual_comparison_degree=2,
                **common,
            )
            levels = (3, 3)
            samples = tuple(
                NSSphereLinearHRecursion(b=mpmath.exp(eta), **common).coefficient(
                    levels
                )
                for eta in etas
            )

            def fitted_intercept(degree):
                matrix = mpmath.matrix(
                    [
                        [(eta * eta) ** power for power in range(degree + 1)]
                        for eta in etas
                    ]
                )
                return mpmath.qr_solve(matrix, mpmath.matrix(samples))[0][0]

            self.assertLess(
                abs(
                    block.coefficient(levels, fit_variant="production")
                    - fitted_intercept(3)
                ),
                mpmath.mpf("1e-60"),
            )
            self.assertLess(
                abs(
                    block.coefficient(levels, fit_variant="comparison")
                    - fitted_intercept(2)
                ),
                mpmath.mpf("1e-60"),
            )
            value = block.series_value(
                (mpmath.mpc("0.07", "0.02"), mpmath.mpc("0.19", "-0.03")),
                (4, 4),
                max_total_twice_level=6,
            )
            self.assertTrue(mpmath.isfinite(value))
            self.assertIsNone(block._limit_backends)
            diagnostics = block.self_dual_fit_diagnostics()
            self.assertGreater(diagnostics["coefficient_count"], 0)
            self.assertEqual(diagnostics["polynomial_degree"], 3)
            self.assertEqual(diagnostics["comparison_degree"], 2)

    def test_compact_self_dual_table_is_reused_from_disk(self):
        with tempfile.TemporaryDirectory() as directory, mpmath.workdps(60):
            common = dict(
                central_charge=mpmath.mpf("13.5"),
                external_weights=tuple(
                    map(mpmath.mpf, ("0.31", "0.42", "0.53", "0.47", "0.28"))
                ),
                internal_weights=tuple(map(mpmath.mpf, ("0.73", "0.81"))),
                vertex_sectors=(1, 0, 1),
                self_dual_log_b_nodes=("0.16", "0.13", "0.10", "0.075", "0.055"),
                self_dual_polynomial_degree=3,
                self_dual_comparison_degree=2,
                coefficient_cache_directory=directory,
                working_precision=60,
            )
            q_values = (mpmath.mpc("0.07", "0.02"), mpmath.mpc("0.19", "-0.03"))
            first = NSSphereLinearHRecursion(**common)
            first_value = first.series_value(
                q_values, (2, 2), max_total_twice_level=4
            )
            self.assertEqual(
                first.self_dual_fit_diagnostics()[
                    "coefficient_cache_artifact_count"
                ],
                1,
            )

            second = NSSphereLinearHRecursion(**common)
            second_value = second.series_value(
                q_values, (2, 2), max_total_twice_level=4
            )
            self.assertLess(abs(first_value - second_value), mpmath.mpf("1e-55"))
            self.assertIsNone(second._limit_backends)
            self.assertEqual(
                second.self_dual_fit_diagnostics()[
                    "coefficient_cache_artifact_count"
                ],
                1,
            )

    def test_external_descendant_components_match_c_recursion(self):
        with mpmath.workdps(75):
            b = mpmath.mpf("1.27")
            c_value = ns_central_charge_from_b(b)
            external = tuple(
                map(mpmath.mpf, ("0.31", "0.42", "0.53", "0.47", "0.28"))
            )
            internal = tuple(map(mpmath.mpf, ("0.73", "0.81")))
            descendants = (0, 1, 0, 1, 0)
            for sectors in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
                h_block = NSSphereLinearHRecursion(
                    b=b,
                    external_weights=external,
                    external_descendants=descendants,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=75,
                )
                c_block = NSSphereLinearCRecursion(
                    central_charge=c_value,
                    external_weights=external,
                    external_descendants=descendants,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=75,
                )
                parities = h_block.compatible_level_parities()
                for levels in (
                    parities,
                    tuple(value + 2 for value in parities),
                    (parities[0] + 4, parities[1] + 2),
                ):
                    with self.subTest(sectors=sectors, levels=levels):
                        self.assertLess(
                            abs(
                                h_block.coefficient(levels)
                                - c_block.coefficient(levels)
                            ),
                            mpmath.mpf("2e-68"),
                        )

    def test_self_dual_confluent_limit_matches_c_recursion(self):
        with mpmath.workdps(80):
            external = tuple(
                map(mpmath.mpf, ("0.37", "0.61", "0.48", "0.29"))
            )
            internal = (mpmath.mpf("0.73"),)
            for sectors in ((0, 0), (1, 1)):
                h_block = NSSphereLinearHRecursion(
                    b=mpmath.mpf(1),
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                c_block = NSSphereLinearCRecursion(
                    central_charge=mpmath.mpf("13.5"),
                    external_weights=external,
                    internal_weights=internal,
                    vertex_sectors=sectors,
                    working_precision=80,
                )
                parity = sectors[0]
                for twice_level in range(parity, 7, 2):
                    with self.subTest(
                        sectors=sectors, twice_level=twice_level
                    ):
                        self.assertLess(
                            abs(
                                h_block.coefficient((twice_level,))
                                - c_block.coefficient((twice_level,))
                            ),
                            mpmath.mpf("2e-14"),
                        )

    def test_explicit_research_hybrid_uses_h_bulk_c_corner(self):
        correlator = BRYNSSphereMultipointCorrelator(
            block_backend="hybrid",
            momenta=(0.5, 1.0 / 3.0, 0.25, 0.6, 0.4),
            points=(0.0, 0.05, 0.1, 1.0, 2.0),
            max_twice_levels=(4, 4),
            max_total_twice_level=6,
            structure_precision=20,
            block_working_precision=50,
        )
        self.assertEqual(correlator.block_backend, "hybrid")
        frame = correlator.frame((0, 1, 2, 3, 4))
        bulk = correlator._block(frame, (0.7, 0.8), (0, 0, 0))
        corner = correlator._block(
            frame,
            (0.7, 0.8),
            (0, 0, 0),
            block_region="corner",
        )
        self.assertIsInstance(bulk, NSSphereLinearHRecursion)
        self.assertIsInstance(corner, NSSphereLinearCRecursion)
        self.assertIsNot(bulk, corner)

    def test_correlator_h_and_c_backends_agree(self):
        common = dict(
            momenta=(0.5, 1.0 / 3.0, 0.25, 0.6, 0.4),
            points=(0.0, 0.05, 0.1, 1.0, 2.0),
            max_twice_levels=(6, 6),
            max_total_twice_level=8,
            structure_precision=20,
            block_working_precision=60,
        )
        h_correlator = BRYNSSphereMultipointCorrelator(
            **common, block_backend="h"
        )
        c_correlator = BRYNSSphereMultipointCorrelator(
            **common, block_backend="c"
        )
        h_frame = h_correlator.frame((0, 1, 2, 3, 4))
        c_frame = c_correlator.frame((0, 1, 2, 3, 4))
        for sectors in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
            h_value = h_correlator.chiral_block(
                h_frame, (0.7, 0.8), sectors
            )
            c_value = c_correlator.chiral_block(
                c_frame, (0.7, 0.8), sectors
            )
            with self.subTest(sectors=sectors):
                self.assertLess(abs(h_value - c_value), 2.0e-12)


if __name__ == "__main__":
    unittest.main()
