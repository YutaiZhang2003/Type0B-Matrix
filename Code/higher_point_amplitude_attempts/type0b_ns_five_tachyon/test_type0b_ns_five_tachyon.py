"""Regressions for the exploratory Type-0B NS five-point kernel."""

import cmath
import itertools
import math
import unittest
from unittest import mock

from sphere_four_point import BRYNSFourPointCorrelator
from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    ContinuedMomentumDensity,
    FIXED_INFINITY_LABEL,
    FIXED_ONE_LABEL,
    FIXED_ZERO_LABEL,
    MINIMAL_SUBTRACTION_T_MAX,
    MOVING_LABELS,
    ODD_SECTOR_ASSIGNMENTS,
    _fermion_pair,
    _oriented_bidisc_mixture_density,
    _oriented_bidisc_mixture_density_in_channel,
    _to_fixed_gauge,
    balanced_equal_energy,
    crossed_ns_structure_poles,
    crossed_ns_structure_poles_complex,
    equal_complex_energy_convergence_audit,
    imaginary_energy_chamber_audit,
    incoming_endpoint_linear_channels,
    pco_safe_linear_channels,
    pco_chiral_terms,
)
from sphere_five_point_liouville import (
    linear_channel_complex_jacobian_to_chart,
    linear_channel_from_ordering,
    linear_channel_positions_by_label,
    oriented_tree_orderings,
)


class Type0BNSFiveTachyonTests(unittest.TestCase):
    def test_three_pcos_leave_four_even_timelike_subsets(self):
        positions = (None, 1.0 + 0.0j, 0.0j, 0.2 + 0.1j, 0.4 - 0.05j)
        signed = (0.4, -0.1, -0.1, -0.1, -0.1)
        terms = pco_chiral_terms(
            positions=positions,
            signed_energies=signed,
        )
        self.assertEqual(len(terms), 4)
        self.assertEqual(
            tuple(term.timelike_labels for term in terms),
            ((), (0, 1), (0, 2), (1, 2)),
        )
        self.assertEqual(terms[0].liouville_descendants, (1, 1, 1, 0, 0))
        self.assertEqual(
            tuple(sum(term.liouville_descendants) for term in terms),
            (3, 1, 1, 1),
        )
        self.assertEqual(
            ODD_SECTOR_ASSIGNMENTS,
            ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)),
        )

    def test_timelike_fermion_sign_reproduces_bry_j_coefficient(self):
        z = 0.23 + 0.17j
        positions = (0.0j, z, 1.0 + 0.0j, None, 2.0 + 0.0j)
        omega2 = 0.19
        omega3 = 0.27
        coefficient = (
            omega2
            * omega3
            * _fermion_pair(positions, 1, 2)
        )
        self.assertLess(
            abs(coefficient - omega2 * omega3 / (1.0 - z)),
            1.0e-15,
        )

    def test_two_pco_specialization_is_exactly_bry_g_h_j_combination(self):
        omega2 = 0.17
        omega3 = 0.23
        z = 0.31 + 0.16j
        momentum = 0.47
        sphere = BRYNSFourPointCorrelator(
            p1=0.11,
            p2=omega2,
            p3=omega3,
            p4=0.51,
            c_recursion_order=2,
            structure_precision=15,
            block_working_precision=40,
        )
        values = sphere.momentum_integrands(momentum, z)
        c_product, ct_product = sphere._structure_products(momentum)
        primary, starred = sphere._blocks(momentum)

        def blocks(argument):
            return (
                sphere._block_value(primary, argument, "even"),
                -sphere._block_value(primary, argument, "odd"),
                sphere._block_value(starred, argument, "even"),
                -sphere._block_value(starred, argument, "odd"),
            )

        pe, po, se, so = blocks(z)
        pe_bar, po_bar, se_bar, so_bar = blocks(z.conjugate())
        pair = omega2 * omega3 / (1.0 - z)
        pair_bar = omega2 * omega3 / (1.0 - z.conjugate())
        component_sum = (
            c_product * (so + pair * pe) * (so_bar + pair_bar * pe_bar)
            + ct_product * (se + pair * po) * (se_bar + pair_bar * po_bar)
        ) / math.pi
        bry_sum = (
            omega2**2 * omega3**2 / abs(1.0 - z) ** 2 * values.G
            - values.H
            - omega2 * omega3 * values.J
        )
        self.assertLess(abs(component_sum - bry_sum), 2.0e-13)

    def test_atlas_chart_round_trip_and_mixture_density(self):
        ordering = oriented_tree_orderings()[37]
        q1 = 0.21 + 0.13j
        q2 = -0.17 + 0.29j
        positions = _to_fixed_gauge(q1, q2, ordering)
        recovered = linear_channel_from_ordering(positions, ordering)
        self.assertLess(abs(recovered.q1 - q1), 2.0e-14)
        self.assertLess(abs(recovered.q2 - q2), 2.0e-14)
        density = _oriented_bidisc_mixture_density(
            positions, radial_power=0.3
        )
        self.assertTrue(math.isfinite(density))
        self.assertGreater(density, 0.0)
        channel_density = _oriented_bidisc_mixture_density_in_channel(
            q1,
            q2,
            ordering,
            radial_power=0.3,
        )
        jacobian = linear_channel_complex_jacobian_to_chart(
            q1,
            q2,
            ordering,
            fixed_zero=FIXED_ZERO_LABEL,
            fixed_one=FIXED_ONE_LABEL,
            fixed_infinity=FIXED_INFINITY_LABEL,
            moving_labels=MOVING_LABELS,
        )
        self.assertLess(
            abs(channel_density - density * abs(jacobian) ** 2)
            / channel_density,
            2.0e-13,
        )

    def test_channel_space_mixture_density_survives_deep_collars(self):
        ordering = oriented_tree_orderings()[53]
        density = _oriented_bidisc_mixture_density_in_channel(
            1.0e-80 * cmath.exp(0.3j),
            1.0e-70 * cmath.exp(-0.7j),
            ordering,
            radial_power=0.04,
        )
        self.assertTrue(math.isfinite(density))
        self.assertGreater(density, 0.0)

    def test_all_c_atlas_log_density_survives_deep_collars(self):
        from type0b_ns_five_tachyon_domain import (
            all_c_atlas_orderings,
            certified_ray_frequencies,
        )

        t_value = 0.651
        outgoing = certified_ray_frequencies(t_value)
        orderings = all_c_atlas_orderings(outgoing)
        powers = {
            tuple(sorted(pair)): 0.04
            for ordering in orderings
            for pair in (ordering[:2], ordering[3:])
        }
        log_density = _oriented_bidisc_mixture_density_in_channel(
            1.0e-80 * cmath.exp(0.3j),
            1.0e-70 * cmath.exp(-0.7j),
            orderings[17],
            radial_power=min(powers.values()),
            orderings=orderings,
            pair_radial_powers=powers,
            return_log_density=True,
        )
        self.assertTrue(math.isfinite(log_density))

    def test_low_order_five_point_smoke_is_finite(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.05j,) * 4,
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        value = kernel.momentum_integrand(
            kernel.fixed_gauge_positions(0.22 + 0.11j, 0.43 - 0.08j),
            (0.37, 0.61),
        )
        self.assertTrue(math.isfinite(value.real))
        self.assertTrue(math.isfinite(value.imag))
        self.assertGreater(abs(value), 1.0e-12)

    def test_equal_imaginary_energy_has_no_raw_problem_free_chamber(self):
        low = imaginary_energy_chamber_audit((0.1j,) * 4)
        self.assertTrue(low["undeformed_positive_real_liouville_contour_valid"])
        self.assertFalse(low["raw_moduli_convergent_without_pco_subtraction"])
        high = imaginary_energy_chamber_audit((0.6j,) * 4)
        self.assertTrue(high["raw_moduli_convergent_without_pco_subtraction"])
        self.assertFalse(high["undeformed_positive_real_liouville_contour_valid"])
        self.assertFalse(low["simultaneously_subtraction_and_residue_free"])
        self.assertFalse(high["simultaneously_subtraction_and_residue_free"])

    def test_full_boundary_audit_distinguishes_two_channel_energies(self):
        audit = imaginary_energy_chamber_audit((0.1j,) * 4)
        thresholds = audit["boundary_pair_threshold_squared"]
        self.assertEqual(len(thresholds), 10)
        self.assertAlmostEqual(thresholds["1,2"], 0.96)
        self.assertAlmostEqual(thresholds["0,4"], 0.91)

    def test_first_complete_large_t_residue_ledger(self):
        c_in_out = crossed_ns_structure_poles(2.2, 0.55, 0)
        self.assertEqual(
            tuple((pole.family, round(pole.momentum.imag, 12)) for pole in c_in_out),
            (("difference", 0.65), ("sum", 1.75)),
        )
        tilde_in_out = crossed_ns_structure_poles(2.2, 0.55, 1)
        self.assertEqual(len(tilde_in_out), 1)
        self.assertAlmostEqual(tilde_in_out[0].momentum.imag, 0.75)
        c_out_out = crossed_ns_structure_poles(0.55, 0.55, 0)
        self.assertEqual(len(c_out_out), 1)
        self.assertAlmostEqual(c_out_out[0].momentum.imag, 0.1)
        self.assertEqual(crossed_ns_structure_poles(0.55, 0.55, 1), ())

    def test_complex_residue_ledger_retains_real_pole_parts(self):
        omega = 0.37 + 0.63j
        c_in_out = crossed_ns_structure_poles_complex(4.0 * omega, omega, 0)
        self.assertEqual(
            tuple((pole.family, pole.wall) for pole in c_in_out),
            (("sum", 3.0), ("difference", 1.0), ("sum", 1.0)),
        )
        expected = (5.0 * omega - 3.0j, 3.0 * omega - 1.0j, 5.0 * omega - 1.0j)
        self.assertLess(max(abs(pole.momentum - value) for pole, value in zip(c_in_out, expected)), 1.0e-14)
        c_out_out = crossed_ns_structure_poles_complex(omega, omega, 0)
        self.assertEqual(len(c_out_out), 1)
        self.assertLess(abs(c_out_out[0].momentum - (2.0 * omega - 1.0j)), 1.0e-14)

    def test_endpoint_balanced_path_exposes_middle_channel_obstruction(self):
        for t_value in (0.601, 0.602, 0.603, 0.604):
            omega = balanced_equal_energy(t_value)
            audit = equal_complex_energy_convergence_audit(omega)
            self.assertFalse(audit["all_moduli_boundaries_absolutely_convergent"])
            self.assertTrue(audit["minimal_subtraction_chamber"])
            self.assertEqual(
                audit["required_polynomial_subtraction_orbits"][0]["multiplicity"],
                2,
            )
            expected_endpoint_margin = 2.0 * t_value - 1.2
            self.assertAlmostEqual(
                audit["endpoint_channel_minimum_integrability_margin"],
                expected_endpoint_margin,
                places=12,
            )
            u_value = 0.5 * t_value - 0.05
            expected_global_margin = 4.0 * u_value - (4.0 * t_value - 1.0) ** 2
            self.assertAlmostEqual(
                audit["minimum_integrability_margin"],
                expected_global_margin,
                places=12,
            )
        self.assertAlmostEqual(
            MINIMAL_SUBTRACTION_T_MAX,
            (25.0 + math.sqrt(545.0)) / 80.0,
            places=15,
        )
        outside = equal_complex_energy_convergence_audit(
            balanced_equal_energy(0.605)
        )
        self.assertFalse(outside["minimal_subtraction_chamber"])

    def test_continued_complex_path_point_density_is_finite(self):
        omega = balanced_equal_energy(0.62)
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(omega,) * 4,
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        positions = kernel.fixed_gauge_positions(0.22 + 0.11j, 0.43 - 0.08j)
        channel = incoming_endpoint_linear_channels(positions)[0]
        result = kernel.continued_integrand_components_positions(
            positions, channel=channel
        )
        for value in (
            result.continuous,
            result.left_residues,
            result.right_residues,
            result.nested_residues,
            result.total,
        ):
            self.assertTrue(math.isfinite(value.real))
            self.assertTrue(math.isfinite(value.imag))

    def test_continued_separated_energy_middle_channel_is_finite(self):
        from type0b_ns_five_tachyon_domain import (
            CERTIFIED_OUTGOING_FREQUENCIES,
            all_c_atlas_orderings,
        )

        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=CERTIFIED_OUTGOING_FREQUENCIES,
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        orderings = all_c_atlas_orderings(CERTIFIED_OUTGOING_FREQUENCIES)
        ordering = next(
            item
            for item in orderings
            if item.index(0) == 2
        )
        result = kernel.continued_linear_q_components(
            0.22 + 0.11j,
            0.43 - 0.08j,
            ordering,
            evaluation_orderings=orderings,
        )
        for value in (
            result.continuous,
            result.left_residues,
            result.right_residues,
            result.nested_residues,
            result.total,
        ):
            self.assertTrue(math.isfinite(value.real))
            self.assertTrue(math.isfinite(value.imag))

    def test_all_c_split_selects_smallest_certified_bidisc(self):
        from type0b_ns_five_tachyon_domain import (
            CERTIFIED_OUTGOING_FREQUENCIES,
            all_c_atlas_orderings,
        )

        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=CERTIFIED_OUTGOING_FREQUENCIES,
            recursion_max_twice_level=0,
            global_max_twice_levels=(0, 0),
            global_max_total_twice_level=0,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        orderings = all_c_atlas_orderings(CERTIFIED_OUTGOING_FREQUENCIES)
        sampled_ordering = orderings[11]
        q1 = 0.22 + 0.11j
        q2 = 0.43 - 0.08j
        positions = linear_channel_positions_by_label(
            q1, q2, sampled_ordering
        )
        candidates = []
        for ordering in orderings:
            channel = linear_channel_from_ordering(positions, ordering)
            if channel.score < 1.0:
                candidates.append(channel)
        geometric = min(candidates, key=lambda channel: channel.score)
        expected_ordering = geometric.ordering
        if expected_ordering.index(0) in (3, 4):
            expected_ordering = tuple(reversed(expected_ordering))
        expected = linear_channel_from_ordering(positions, expected_ordering)
        zero = ContinuedMomentumDensity(0.0j, 0.0j, 0.0j, 0.0j)
        with mock.patch.object(
            kernel,
            "continued_integrand_components_positions",
            return_value=zero,
        ) as evaluate:
            kernel.continued_linear_q_components(
                q1,
                q2,
                sampled_ordering,
                evaluation_orderings=orderings,
            )
        selected = evaluate.call_args.kwargs["channel"]
        self.assertEqual(selected.ordering, expected.ordering)
        self.assertAlmostEqual(selected.score, expected.score, places=15)

    def test_antiholomorphic_block_uses_conjugate_tube_lift(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.1j,) * 4,
            recursion_max_twice_level=2,
            global_max_twice_levels=(5, 3),
            global_max_total_twice_level=8,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=35,
        )
        positions = (0.0j, 0.05 + 0.0j, 0.1 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j)
        channel = linear_channel_from_ordering(
            positions, (2, 1, 0, 3, 4)
        )
        self.assertLess(channel.q2.real, 0.0)
        descendants = (0, 0, 0, 0, 0)
        holomorphic = kernel._chiral_block(
            channel,
            positions,
            (0.37, 0.61),
            (0, 1, 1),
            descendants,
            antiholomorphic=False,
        )
        antiholomorphic = kernel._chiral_block(
            channel,
            positions,
            (0.37, 0.61),
            (0, 1, 1),
            descendants,
            antiholomorphic=True,
        )
        self.assertLess(
            abs(antiholomorphic - holomorphic.conjugate()),
            2.0e-12 * max(1.0, abs(holomorphic)),
        )

    def test_moving_middle_terms_reconstruct_diagnostic_component(self):
        from type0b_ns_five_tachyon_domain import (
            minimal_subtraction_ray_frequencies,
        )

        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=minimal_subtraction_ray_frequencies(0.99),
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        ordering = (1, 2, 0, 3, 4)
        q1 = 0.22 + 0.11j
        q2 = 0.31 - 0.08j
        terms = kernel.continued_linear_q_middle_line_terms(
            q1, q2, ordering
        )
        components = kernel.continued_linear_q_components(
            q1, q2, ordering
        )
        reconstructed = sum((term.value for term in terms), 0.0j)
        scale = max(1.0, abs(components.middle_line_residues))
        self.assertLess(
            abs(reconstructed - components.middle_line_residues) / scale,
            2.0e-11,
        )
        excluded = kernel.continued_linear_q_components(
            q1,
            q2,
            ordering,
            excluded_middle_walls=(1.0,),
        )
        wall_one = sum(
            (
                term.value
                for term in terms
                if abs(term.second_pole.wall - 1.0) < 1.0e-12
            ),
            0.0j,
        )
        self.assertLess(
            abs(
                components.middle_line_residues
                - excluded.middle_line_residues
                - wall_one
            )
            / scale,
            2.0e-11,
        )

    def test_all_c_middle_corner_projection_matches_off_diagonal_limit(self):
        from type0b_ns_five_tachyon_domain import (
            minimal_subtraction_ray_frequencies,
        )

        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=minimal_subtraction_ray_frequencies(0.99),
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        ordering = (1, 2, 0, 3, 4)
        q1, q2 = 1.0e-4, 2.0e-5
        raw = sum(
            (
                term.value
                for term in kernel.continued_linear_q_middle_line_terms(
                    q1, q2, ordering
                )
                if abs(term.second_pole.wall - 1.0) < 1.0e-12
            ),
            0.0j,
        )
        counterterm = kernel.continued_middle_line_corner_counterterm(
            q1,
            q2,
            ordering,
            projection_radius=1.0e-5,
        )
        self.assertLess(abs(raw / counterterm - 1.0), 2.0e-3)
        finite_part = kernel.continued_middle_line_corner_finite_part(
            ordering,
            collar_radius=0.05,
            projection_radius=1.0e-5,
        )
        self.assertTrue(math.isfinite(finite_part.real))
        self.assertTrue(math.isfinite(finite_part.imag))

    def test_continued_large_t_point_density_is_finite(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.55j,) * 4,
            recursion_max_twice_level=0,
            global_max_twice_levels=(1, 1),
            global_max_total_twice_level=2,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        positions = kernel.fixed_gauge_positions(0.22 + 0.11j, 0.43 - 0.08j)
        channel = incoming_endpoint_linear_channels(positions)[0]
        result = kernel.continued_integrand_components_positions(
            positions, channel=channel
        )
        for value in (
            result.continuous,
            result.left_residues,
            result.right_residues,
            result.nested_residues,
            result.total,
        ):
            self.assertTrue(math.isfinite(value.real))
            self.assertTrue(math.isfinite(value.imag))

    def test_pco_safe_channel_reversal_is_exact_at_fixed_momenta(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.05j,) * 4,
            recursion_max_twice_level=2,
            global_max_twice_levels=(5, 5),
            global_max_total_twice_level=8,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=35,
        )
        positions = kernel.fixed_gauge_positions(
            0.22 + 0.11j, 0.43 - 0.08j
        )
        channel = pco_safe_linear_channels(positions)[0]
        reverse = linear_channel_from_ordering(
            positions, tuple(reversed(channel.ordering))
        )
        first = kernel.momentum_integrand(
            positions, (0.37, 0.61), channel=channel
        )
        second = kernel.momentum_integrand(
            positions, (0.61, 0.37), channel=reverse
        )
        self.assertLess(
            abs(first - second) / max(abs(first), abs(second)),
            2.0e-12,
        )

    def test_endpoint_descendant_bpz_phase_makes_general_reversal_exact(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.05j,) * 4,
            recursion_max_twice_level=2,
            global_max_twice_levels=(5, 5),
            global_max_total_twice_level=8,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=35,
        )
        positions = kernel.fixed_gauge_positions(
            0.22 + 0.11j, 0.43 - 0.08j
        )
        channel = min(
            (
                linear_channel_from_ordering(positions, ordering)
                for ordering in oriented_tree_orderings()
                if ordering[0] in (0, 1, 2)
            ),
            key=lambda item: item.score,
        )
        reverse = linear_channel_from_ordering(
            positions, tuple(reversed(channel.ordering))
        )
        first = kernel.momentum_integrand(
            positions, (0.37, 0.61), channel=channel
        )
        second = kernel.momentum_integrand(
            positions, (0.61, 0.37), channel=reverse
        )
        self.assertLess(
            abs(first - second) / max(abs(first), abs(second)),
            2.0e-12,
        )

    def test_two_pco_face_factorizes_in_all_six_crossing_cells(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.1j,) * 4,
            recursion_max_twice_level=None,
            global_max_twice_levels=(4, 4),
            global_max_total_twice_level=6,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=35,
        )
        q1 = 1.0e-4
        q2 = 0.31 + 0.17j
        for tail in itertools.permutations((0, 3, 4)):
            ordering = (1, 2, *tail)
            positions = linear_channel_positions_by_label(q1, q2, ordering)
            channel = linear_channel_from_ordering(positions, ordering)
            direct = kernel.momentum_integrand(
                positions, (0.37, 0.61), channel=channel
            ) * abs(q2) ** 2
            coefficient = kernel.two_pco_face_asymptotic_momentum_density(
                ordering=ordering,
                normal_momentum=0.37,
                remaining_momentum=0.61,
                remaining_modulus=q2,
            )
            beta = kernel._two_pco_face_beta(ordering, 0.37)
            predicted = coefficient * abs(q1) ** beta.real
            self.assertLess(
                abs(direct / predicted - 1.0),
                1.5e-4,
                msg=f"failed face ordering {ordering}",
            )

        ordering = (1, 2, 3, 4, 0)
        ratios = []
        for angle in (0.0, 0.7, 1.9):
            normal = q1 * cmath.exp(1.0j * angle)
            positions = linear_channel_positions_by_label(normal, q2, ordering)
            channel = linear_channel_from_ordering(positions, ordering)
            direct = kernel.momentum_integrand(
                positions, (0.37, 0.61), channel=channel
            ) * abs(q2) ** 2
            coefficient = kernel.two_pco_face_asymptotic_momentum_density(
                ordering=ordering,
                normal_momentum=0.37,
                remaining_momentum=0.61,
                remaining_modulus=q2,
            )
            beta = kernel._two_pco_face_beta(ordering, 0.37)
            ratios.append(direct / (coefficient * abs(normal) ** beta.real))
        self.assertLess(max(abs(value - 1.0) for value in ratios), 1.5e-4)

    def test_boundary_primary_and_remainder_split_after_channel_reversal(self):
        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=(0.1j,) * 4,
            recursion_max_twice_level=0,
            global_max_twice_levels=(2, 2),
            global_max_total_twice_level=4,
            momentum_orders=(2, 3),
            momentum_maximum=1.0,
            structure_precision=15,
            block_working_precision=30,
        )
        arguments = (0.02 + 0.01j, 0.21 + 0.03j, (1, 2, 3, 4, 0))
        full = kernel.continued_linear_q_components(*arguments).total
        remainder = kernel.continued_linear_q_components(
            *arguments, subtracted_continuum_boundary_pair=(1, 2)
        ).total
        primary = kernel.continued_linear_q_components(
            *arguments, primary_continuum_boundary_pair=(1, 2)
        ).total
        self.assertLess(abs(full - remainder - primary) / abs(full), 2.0e-12)


if __name__ == "__main__":
    unittest.main()
    crossed_ns_structure_poles,
    incoming_endpoint_linear_channels,
