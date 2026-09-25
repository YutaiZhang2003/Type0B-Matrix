#!/usr/bin/env python3
"""Focused checks for the genuine Spin(23) singlet amplitudes.

These tests stop below the expensive momentum/moduli quadrature used by the
production scan.  They certify the external-component signs against the
independently implemented VVVV block engine and exercise the complete folded
sphere integral at a deliberately tiny smoke-test truncation.
"""

from __future__ import annotations

import unittest

import numpy as np

import spin23_singlet_amplitudes as singlet
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3 as reference
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast


class GenuineSingletAmplitudeTests(unittest.TestCase):
    """Check component routing, Ward identities, and permutation symmetry."""

    def setUp(self) -> None:
        self.energies = (
            0.11 + 0.15j,
            0.17 + 0.16j,
            0.23 + 0.18j,
            0.51 + 0.49j,
        )
        self.momentum = 0.73

    def _table(self, energies, names, order=3):
        momenta = np.asarray([self.momentum])
        solvers, _ = singlet._gram_solvers(
            momenta,
            2 * order + 1,
            condition_limit=1.0e13,
        )
        weights = tuple(fast.h_of_p(value) for value in energies)
        table = singlet._coefficient_table(
            weights,
            momenta,
            solvers,
            order,
            names,
        )
        return weights, table

    def test_component_phase_and_internal_parity_table(self) -> None:
        """Lock the fixed-parity-to-superfield component convention."""

        expected = {
            "P": ((1, 1), (0, 1)),
            "M": ((1, -1), (1, 0)),
            "O": ((1, 1), (1, 0)),
            "A": ((-1, 1), (0, 1)),
            "L": ((-1, 1), (0, 1)),
            "R": ((1, 1), (0, 1)),
        }
        for name, (phases, parities) in expected.items():
            words = singlet.WORD_PATTERNS[name]
            with self.subTest(pattern=name):
                self.assertEqual(
                    tuple(singlet.component_phase(words, value) for value in (0, 1)),
                    phases,
                )
                self.assertEqual(
                    tuple(singlet.internal_parity(words, value) for value in (0, 1)),
                    parities,
                )

    def test_s_channel_primary_and_middle_descendants_match_reference(self) -> None:
        """Compare every retained direct coefficient with the audited engine."""

        order = 3
        weights, table = self._table(self.energies, ("P", "M"), order)
        h1, h2, h3, h4 = weights
        h_internal = fast.h_of_p(self.momentum)
        reference_blocks = {
            "P": reference.NSBlockComputer(
                h4, h3, h2, h1, False, False, max_level2=2 * order + 1
            ),
            "M": reference.NSBlockComputer(
                h4, h3, h2, h1, True, True, max_level2=2 * order + 1
            ),
        }
        for name, block in reference_blocks.items():
            even, odd = (values[0] for values in table.coefficients[name])
            expected_even = np.asarray(
                [1.0]
                + [
                    complex(block.coefficient(2 * level, h_internal))
                    for level in range(1, order + 1)
                ]
            )
            expected_odd = np.asarray(
                [
                    complex(block.coefficient(2 * level + 1, h_internal))
                    for level in range(order + 1)
                ]
            )
            phase_even = singlet.component_phase(singlet.WORD_PATTERNS[name], 0)
            phase_odd = singlet.component_phase(singlet.WORD_PATTERNS[name], 1)
            reference_odd_sign = -1 if name == "M" else 1
            with self.subTest(pattern=name, parity="even"):
                np.testing.assert_allclose(
                    phase_even * even,
                    expected_even,
                    rtol=2.0e-13,
                    atol=2.0e-13,
                )
            with self.subTest(pattern=name, parity="odd"):
                np.testing.assert_allclose(
                    phase_odd * odd,
                    reference_odd_sign * expected_odd,
                    rtol=2.0e-13,
                    atol=2.0e-13,
                )

    def test_crossed_descendant_matches_global_ward_identity(self) -> None:
        """Check both crossed components at equal effective series depth.

        For ``L=(G_-1/2 V_1,G_-1/2 V_2,V_3,V_4)``, the global Ward
        identity expresses the even component as a starred odd block minus a
        derivative of the primary even block.  The starred series must be
        shifted by one power of the crossed coordinate before coefficients
        are compared.
        """

        order = 4
        crossed_energies = (
            self.energies[2],
            self.energies[1],
            self.energies[0],
            self.energies[3],
        )
        weights, table = self._table(crossed_energies, ("L",), order)
        h1, h2, h3, h4 = weights
        h_internal = fast.h_of_p(self.momentum)
        primary = reference.NSBlockComputer(
            h4, h3, h2, h1, False, False, max_level2=2 * order + 1
        )
        starred = reference.NSBlockComputer(
            h4, h3, h2, h1, True, True, max_level2=2 * order + 1
        )
        primary_even = np.asarray(
            [1.0]
            + [
                complex(primary.coefficient(2 * level, h_internal))
                for level in range(1, order + 1)
            ]
        )
        primary_odd = np.asarray(
            [
                complex(primary.coefficient(2 * level + 1, h_internal))
                for level in range(order + 1)
            ]
        )
        starred_even = np.asarray(
            [1.0]
            + [
                complex(starred.coefficient(2 * level, h_internal))
                for level in range(1, order + 1)
            ]
        )
        # The audited reference block includes a minus sign in its starred
        # odd component.
        starred_odd = -np.asarray(
            [
                complex(starred.coefficient(2 * level + 1, h_internal))
                for level in range(order + 1)
            ]
        )
        base_exponent = complex(h_internal - h1 - h2)
        expected_even = -(base_exponent + np.arange(order + 1)) * primary_even
        expected_even[1:] += starred_odd[:-1]
        expected_odd = starred_even - (
            base_exponent + 0.5 + np.arange(order + 1)
        ) * primary_odd

        raw_even, raw_odd = (values[0] for values in table.coefficients["L"])
        actual_even = singlet.component_phase(singlet.L_WORDS, 0) * raw_even
        actual_odd = singlet.component_phase(singlet.L_WORDS, 1) * raw_odd
        np.testing.assert_allclose(
            actual_even,
            expected_even,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            actual_odd,
            expected_odd,
            rtol=3.0e-13,
            atol=3.0e-13,
        )

    def test_both_adjacent_pair_blocks_are_reconstructed_by_ward_identities(
        self,
    ) -> None:
        """Compare the production Ward reduction with independent Gram sewing."""

        order = 4
        momenta = np.asarray([0.006, self.momentum, 2.0])
        weights = tuple(fast.h_of_p(value) for value in self.energies)
        solvers, _ = singlet._gram_solvers(
            momenta,
            2 * order + 1,
            condition_limit=1.0e13,
        )
        direct = singlet._coefficient_table(
            weights,
            momenta,
            solvers,
            order,
            ("P", "M", "L", "R"),
        )
        reduced_l, reduced_r = (
            singlet._crossed_pair_coefficients_from_ward_identity(direct)
        )
        for name, observed in (("L", reduced_l), ("R", reduced_r)):
            for parity in (0, 1):
                with self.subTest(pattern=name, parity=parity):
                    np.testing.assert_allclose(
                        observed[parity],
                        direct.coefficients[name][parity],
                        rtol=6.0e-13,
                        atol=6.0e-13,
                    )

    def test_equilibrated_gram_solve_matches_high_precision_oracle(self) -> None:
        """Stress the worst low-P level used by the q=7 production block."""

        order = 7
        momentum = np.asarray([0.006])
        weights = tuple(fast.h_of_p(value) for value in self.energies)
        solvers, _ = singlet._gram_solvers(
            momentum,
            2 * order + 1,
            condition_limit=1.0e13,
        )
        solver = solvers[2 * order + 1][0]
        self.assertLess(solver.equilibrated_condition, solver.condition / 100.0)

        double = singlet._coefficient_table(
            weights,
            momentum,
            solvers,
            order,
            ("P", "M", "O", "A"),
        )
        high_precision = singlet._coefficient_table(
            weights,
            momentum,
            solvers,
            order,
            ("P", "M", "O", "A"),
            high_precision_condition=1.0,
            high_precision_digits=70,
            high_precision_max_momentum=0.18,
        )
        expected_fallbacks = sum(
            solver.equilibrated_condition > 1.0
            for level_solvers in solvers.values()
            for solver in level_solvers
        )
        self.assertEqual(
            high_precision.high_precision_solve_count,
            expected_fallbacks,
        )
        for name in ("P", "M", "O", "A"):
            for parity in (0, 1):
                with self.subTest(pattern=name, parity=parity):
                    np.testing.assert_allclose(
                        double.coefficients[name][parity],
                        high_precision.coefficients[name][parity],
                        rtol=5.0e-13,
                        atol=5.0e-13,
                    )

    def test_stable_accumulation_retains_small_complex_remainder(self) -> None:
        values = (1.0e16 + 1.0e16j, 1.0 - 2.0j, -1.0e16 - 1.0e16j)
        self.assertEqual(singlet._stable_complex_fsum(values), 1.0 - 2.0j)

    def test_endpoint_momentum_rule_integrates_even_taylor_basis(self) -> None:
        """Certify the P^2-weighted low-momentum quadrature through P^8."""

        p_cut = 0.03
        momenta, weights = singlet._momentum_quadrature(
            "segmented",
            2.5,
            p_cut,
        )
        endpoint_momenta = momenta[:4]
        endpoint_weights = weights[:4]
        self.assertTrue(np.all(endpoint_weights > 0))
        for power in (2, 4, 6, 8):
            with self.subTest(power=power):
                observed = np.sum(endpoint_weights * endpoint_momenta**power)
                expected = p_cut ** (power + 1) / (power + 1)
                np.testing.assert_allclose(observed, expected, rtol=2.0e-14)

    def test_tiny_full_integral_is_symmetric_in_outgoing_legs_2_and_3(self) -> None:
        """Exercise folding, both OPE patches, and physical leg routing."""

        options = {
            "q_order": 1,
            "lower_q_order": None,
            "p_nodes": 3,
            "p_max": 2.0,
            "theta_orders": (3, 3, 5),
            "radial_order": 3,
            "disk_total_order": 5,
            "crossed_disk_total_order": 5,
            "include_v_to_vss": True,
        }
        original = singlet.evaluate_singlet_amplitudes(self.energies, **options)
        exchanged_energies = (
            self.energies[0],
            self.energies[2],
            self.energies[1],
            self.energies[3],
        )
        exchanged = singlet.evaluate_singlet_amplitudes(
            exchanged_energies,
            **options,
        )
        for left, right in (
            (original.values.ssvv_raw, exchanged.values.ssvv_raw),
            (original.values.ssss_raw, exchanged.values.ssss_raw),
            (
                original.values.v_to_vss_raw,
                exchanged.values.v_to_vss_raw,
            ),
        ):
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertTrue(np.isfinite(left.real) and np.isfinite(left.imag))
            self.assertAlmostEqual(left.real, right.real, places=10)
            self.assertAlmostEqual(left.imag, right.imag, places=10)

    def test_lens_integral_is_picture_assignment_independent(self) -> None:
        """Regress the three inequivalent choices of picture-zero singlets.

        This test uses sewing level 4 and independently resolved endpoint,
        bulk and tail nodes. The former integer count 16 is insufficient
        when shared among all three regions of the new threshold rule.
        The historical elliptic q=7 scan is not a sewing accuracy certificate.
        """

        outgoing = (
            0.189391 + 0.167372j,
            0.286250 + 0.216344j,
            0.170545 + 0.147015j,
        )
        options = {
            "q_order": 4,
            "lower_q_order": None,
            "p_nodes": (8, 24, 12),
            "momentum_scheme": "threshold_weighted",
            "p_max": 2.5,
            "theta_orders": (12, 12, 48),
            "radial_order": 20,
            "disk_total_order": 15,
            "crossed_disk_total_order": 15,
        }
        amplitudes = []
        # Exchanging the two picture-zero legs is exact, so one representative
        # from each pair of the six outgoing permutations is sufficient.
        for permutation in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
            permuted = tuple(outgoing[index] for index in permutation)
            energies = (*permuted, sum(permuted))
            amplitudes.append(
                singlet.evaluate_singlet_amplitudes(
                    energies,
                    **options,
                ).values.ssss_raw
            )
        amplitudes = np.asarray(amplitudes)
        mean = np.mean(amplitudes)
        maximum_relative_residual = np.max(np.abs(amplitudes - mean)) / abs(mean)
        self.assertLess(maximum_relative_residual, 5.0e-3)

    def test_meromorphic_disks_match_convergent_radial_integrals(self) -> None:
        """Benchmark both local continuation formulas where no continuation is needed."""

        energies = (
            0.05 + 0.01j,
            0.06 + 0.01j,
            0.07 + 0.01j,
            0.18 + 0.03j,
        )
        atlas = singlet._build_atlas(
            energies,
            q_order=1,
            p_nodes=3,
            p_max=3.0,
            p_cut=0.03,
            gram_condition_limit=1.0e13,
        )
        for channel, data in (
            ("s", atlas.original_s),
            ("t", atlas.original_t),
        ):
            if channel == "s":
                coordinate, area_weights = fast.disk_grid(0.04, 40, 64, 3.0)
            else:
                coordinate, area_weights = reference._lens_grid(
                    0.04,
                    40,
                    64,
                    3.0,
                )
            # The largest sampled internal momentum gives safely integrable
            # powers in both local channels.
            one_kernel = singlet._ChannelData(
                energies=data.energies,
                kernels=(data.kernels[-1],),
            )
            original_z = coordinate if channel == "s" else 1 - coordinate
            for process in ("ssvv", "ssss"):
                numerical = singlet._grid_integral(
                    one_kernel,
                    coordinate,
                    area_weights,
                    original_z=original_z,
                    q=None,
                    theta3=None,
                    order=1,
                    process=process,
                )
                if channel == "s":
                    analytic = singlet._disk_integral(
                        one_kernel,
                        0.04,
                        block_order=1,
                        total_order=10,
                        process=process,
                    )
                else:
                    analytic = singlet._crossed_disk_integral(
                        one_kernel,
                        0.04,
                        block_order=1,
                        total_order=10,
                        process=process,
                    )
                with self.subTest(channel=channel, process=process):
                    np.testing.assert_allclose(
                        analytic,
                        numerical,
                        rtol=1.0e-11,
                        atol=1.0e-100,
                    )

    def test_lens_series_integral_matches_direct_quadrature(self) -> None:
        """Check the lens geometry independently of the conformal blocks."""

        epsilon = 0.1
        holomorphic = np.asarray([1.0, 0.2 - 0.1j, -0.03j])
        antiholomorphic = np.asarray([0.7 + 0.1j, -0.3 + 0.05j])
        holomorphic_power = 1.2 + 0.07j
        antiholomorphic_power = 0.2 + 0.07j
        coordinate, area_weights = reference._lens_grid(
            epsilon,
            90,
            180,
            2.0,
        )
        direct = np.sum(
            area_weights
            * np.exp(
                holomorphic_power * np.log(coordinate)
                + antiholomorphic_power * np.log(np.conj(coordinate))
            )
            * np.polynomial.polynomial.polyval(coordinate, holomorphic)
            * np.polynomial.polynomial.polyval(
                np.conj(coordinate),
                antiholomorphic,
            )
        )
        continued = singlet._integrate_series_lens(
            holomorphic,
            antiholomorphic,
            holomorphic_power,
            antiholomorphic_power,
            epsilon,
            angular_order=28,
        )
        np.testing.assert_allclose(
            continued,
            direct,
            rtol=2.0e-12,
            atol=2.0e-14,
        )

    def test_direct_disk_exact_logarithm_uses_finite_part(self) -> None:
        r"""At ``z^-1 zbar^-1`` the continued disk is ``2 pi log eps``."""

        epsilon = 0.073
        actual = fast._integrate_series_disk(
            np.asarray([1.0 + 0.0j]),
            np.asarray([1.0 + 0.0j]),
            -1.0,
            -1.0,
            epsilon,
        )
        expected = 2.0 * np.pi * np.log(epsilon)
        self.assertTrue(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)

    def test_energy_conservation_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "omega0"):
            singlet.evaluate_singlet_amplitudes(
                (*self.energies[:3], self.energies[3] + 0.1),
                q_order=1,
                lower_q_order=None,
                p_nodes=2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
