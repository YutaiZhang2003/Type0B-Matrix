#!/usr/bin/env python3
"""Focused exact and numerical checks for the Ramond block implementation."""

from __future__ import annotations

import cmath
import math
import unittest
from fractions import Fraction

import sympy as sp

from ns_algebra.ns_sca import G
from ramond_algebra.nrr_three_point_tensor import (
    hjs_to_polynomial_ground_tensor,
    rr_three_point_from_ground_tensor,
)
from ramond_algebra.ns_rr_three_point_tensor import (
    ns_rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_plumbing import zero_mode_matrix
from ramond_algebra.ramond_sca import (
    L,
    PBWState,
    gram_matrix,
    ground_gram_matrix,
    ground_state,
)
from spin23_ramond_blocks import (
    direct_ramond_sphere_block_series,
    direct_ramond_torus_one_point_series,
    hjs_ramond_sphere_block_series,
    ramond_weight,
)
from spin23_genus1_blocks import direct_ramond_torus_necklace_series
from spin23_genus1_recursion import (
    ns_liouville_weight,
    super_liouville_central_charge,
)
from spin23_ramond_torus_recursion import (
    generic_ramond_torus_two_point_coefficients,
    generic_ramond_torus_two_point_necklace_coefficients,
    self_dual_ramond_torus_two_point_necklace_coefficients,
)
from spin23_two_virasoro_ramond import (
    AuxiliaryFermionState,
    _embedded_l_minus_one_action,
    _elementary_product_form,
    _theta_elementary_product_form,
    auxiliary_fermion_mode_action,
    auxiliary_fermion_basis,
    auxiliary_fermion_inner_product,
    auxiliary_ns_ns_ns_three_point,
    auxiliary_ns_r_r_three_point,
    embedded_branch_state,
)
from spin23_virasoro_torus_recursion import _normalized_rho_two_edge


class Spin23RamondBlockTests(unittest.TestCase):
    """Check the full ground doublet, Ward tensors, and block contractions."""

    def test_zero_mode_squares_to_shifted_l_zero(self) -> None:
        """Verify the defining Ramond zero-mode relation on descendants."""

        h, c = sp.symbols("h c")
        for twice_level in (0, 2, 4):
            with self.subTest(twice_level=twice_level):
                basis, zero_mode = zero_mode_matrix(twice_level, h=h, c=c)
                expected = h + sp.Rational(twice_level, 2) - c / 24
                difference = zero_mode**2 - expected * sp.eye(len(basis))
                self.assertTrue(all(sp.factor(entry) == 0 for entry in difference))

    def test_identity_three_point_tensor_is_ramond_gram_matrix(self) -> None:
        """Use the NS identity as an independent Ward-tensor normalization."""

        h, c = sp.symbols("h c")
        terminal = ground_gram_matrix(h, c)
        for twice_level in (0, 2, 4):
            with self.subTest(twice_level=twice_level):
                basis, expected = gram_matrix(twice_level, h=h, c=c)
                actual = sp.Matrix(
                    [
                        [
                            rr_three_point_from_ground_tensor(
                                left,
                                (),
                                right,
                                h_infinity=h,
                                h_ns=0,
                                h_zero=h,
                                c=c,
                                ground_tensor=terminal,
                            )
                            for right in basis
                        ]
                        for left in basis
                    ]
                )
                self.assertTrue(
                    all(sp.factor(entry) == 0 for entry in actual - expected)
                )

    def test_unsimplified_ramond_ward_tensor_is_algebraically_identical(
        self,
    ) -> None:
        """Certify the faster numerical-template path against the default."""

        h_infinity, h_ns, h_zero, c = sp.symbols(
            "h_infinity h_ns h_zero c"
        )
        ground = sp.Matrix(2, 2, sp.symbols("g0:4"))
        infinity = PBWState((L(-1),), 0)
        middle = (G(Fraction(-1, 2)),)
        zero = PBWState((L(-1),), 0)
        options = dict(
            h_infinity=h_infinity,
            h_ns=h_ns,
            h_zero=h_zero,
            c=c,
            ground_tensor=ground,
        )
        simplified = rr_three_point_from_ground_tensor(
            infinity,
            middle,
            zero,
            **options,
        )
        unsimplified = rr_three_point_from_ground_tensor(
            infinity,
            middle,
            zero,
            simplify=False,
            **options,
        )
        self.assertEqual(sp.expand(unsimplified - simplified), 0)

    def test_ordered_ns_rr_identity_obeys_local_two_point_derivatives(self) -> None:
        """Check the PDF local coordinates against exact two-point derivatives."""

        h, c = sp.symbols("h c")
        ground = ground_state(0)
        # Construct the state explicitly; the deliberately local test is
        # rho(1_NS at infinity, R at z=1, R at z=0), not a BPZ Gram test.
        derivative = PBWState((L(-1),), 0)
        terminal = ground_gram_matrix(h, c)

        def evaluate(middle, zero):
            return ns_rr_three_point_from_ground_tensor(
                (),
                middle,
                zero,
                h_infinity=0,
                h_middle=h,
                h_zero=h,
                c=c,
                ground_tensor=terminal,
            )

        self.assertEqual(sp.factor(evaluate(derivative, ground)), -2 * h)
        self.assertEqual(sp.factor(evaluate(ground, derivative)), 2 * h)
        self.assertEqual(
            sp.factor(evaluate(derivative, derivative)),
            -2 * h * (2 * h + 1),
        )

    def test_pdf_ordered_global_virasoro_coefficients(self) -> None:
        """Check the cut-theta seed against elementary global Ward identities."""

        h_1, h_2, h_3 = 0.73, 1.11, 1.37
        a = h_3 + h_2 - h_1
        cases = {
            (0, 0): 1.0,
            (1, 0): -a / math.sqrt(2.0 * h_2),
            (0, 1): a / math.sqrt(2.0 * h_3),
            (1, 1): -a * (a + 1.0) / math.sqrt(4.0 * h_2 * h_3),
        }
        for levels, expected in cases.items():
            with self.subTest(levels=levels):
                actual = _normalized_rho_two_edge(
                    levels[0],
                    levels[1],
                    h_2,
                    h_1,
                    h_3,
                )
                self.assertAlmostEqual(actual, expected)

    def test_auxiliary_ns_rr_form_obeys_local_virasoro_derivatives(self) -> None:
        """Reconstruct free-fermion L_-1 descendants in PDF coordinates."""

        vacuum = AuxiliaryFermionState("NS")
        coefficient = 1.0 / (2.0 * math.sqrt(2.0))
        h_ramond = 1.0 / 16.0
        for form_parity in (0, 1):
            for middle_parity in (0, 1):
                for zero_parity in (0, 1):
                    ground_middle = AuxiliaryFermionState(
                        "R", (), middle_parity
                    )
                    ground_zero = AuxiliaryFermionState("R", (), zero_parity)
                    base = auxiliary_ns_r_r_three_point(
                        vacuum,
                        ground_middle,
                        ground_zero,
                        form_parity=form_parity,
                    )
                    if abs(base) < 1.0e-14:
                        continue
                    derivative_middle = AuxiliaryFermionState(
                        "R", (2,), 1 - middle_parity
                    )
                    derivative_zero = AuxiliaryFermionState(
                        "R", (2,), 1 - zero_parity
                    )
                    middle_value = coefficient * auxiliary_ns_r_r_three_point(
                        vacuum,
                        derivative_middle,
                        ground_zero,
                        form_parity=form_parity,
                    )
                    zero_value = coefficient * auxiliary_ns_r_r_three_point(
                        vacuum,
                        ground_middle,
                        derivative_zero,
                        form_parity=form_parity,
                    )
                    both_value = coefficient**2 * auxiliary_ns_r_r_three_point(
                        vacuum,
                        derivative_middle,
                        derivative_zero,
                        form_parity=form_parity,
                    )
                    self.assertAlmostEqual(middle_value, -2.0 * h_ramond * base)
                    self.assertAlmostEqual(zero_value, 2.0 * h_ramond * base)
                    self.assertAlmostEqual(
                        both_value,
                        -2.0 * h_ramond * (2.0 * h_ramond + 1.0) * base,
                    )

    def test_auxiliary_ns_three_point_reduces_to_the_bpz_pairing(self) -> None:
        """Check the Pfaffian form against the independent mode algebra."""

        vacuum = AuxiliaryFermionState("NS")
        for twice_level in range(8):
            basis = auxiliary_fermion_basis("NS", twice_level)
            for left in basis:
                for right in basis:
                    with self.subTest(
                        twice_level=twice_level,
                        left=left,
                        right=right,
                    ):
                        self.assertAlmostEqual(
                            auxiliary_ns_ns_ns_three_point(
                                left,
                                vacuum,
                                right,
                            ),
                            auxiliary_fermion_inner_product(left, right),
                        )

        fermion = AuxiliaryFermionState("NS", (1,))
        self.assertAlmostEqual(
            auxiliary_ns_ns_ns_three_point(vacuum, fermion, fermion),
            1.0,
        )
        self.assertAlmostEqual(
            auxiliary_ns_ns_ns_three_point(fermion, fermion, vacuum),
            -1.0,
        )

    def test_auxiliary_odd_ground_form_is_zero_mode_lift(self) -> None:
        """Derive the odd spin-field tensor from the Ramond zero mode."""

        vacuum = AuxiliaryFermionState("NS")
        for middle_parity in (0, 1):
            for zero_parity in (0, 1):
                middle = AuxiliaryFermionState("R", (), middle_parity)
                zero = AuxiliaryFermionState("R", (), zero_parity)
                lifted = 0.0j
                for output, coefficient in auxiliary_fermion_mode_action(
                    0, zero
                ).items():
                    lifted += coefficient * auxiliary_ns_r_r_three_point(
                        vacuum,
                        middle,
                        output,
                        form_parity=0,
                    )
                actual = auxiliary_ns_r_r_three_point(
                    vacuum,
                    middle,
                    zero,
                    form_parity=1,
                )
                self.assertAlmostEqual(actual, lifted)

    def test_two_virasoro_ground_sum_reproduces_pdf_equation_7_4(self) -> None:
        """Check every chiral-structure and spin-lift sign at q_i=0."""

        for eta in (-1, 1):
            for eta_prime in (-1, 1):
                for eta_2 in (-1, 1):
                    for eta_3 in (-1, 1):
                        with self.subTest(
                            eta=eta,
                            eta_prime=eta_prime,
                            eta_2=eta_2,
                            eta_3=eta_3,
                        ):
                            result = generic_ramond_torus_two_point_coefficients(
                                b=1.17,
                                momentum_2=0.37,
                                momentum_3=0.52,
                                external_ns_momentum=0.41,
                                left_structure_sign=eta,
                                right_structure_sign=eta_prime,
                                lift_sign_2=eta_2,
                                lift_sign_3=eta_3,
                                maximum_level_2=0,
                                maximum_level_3=0,
                            )
                            even = result.primary_components[0][(0, 0)]
                            odd = result.primary_components[1][(0, 0)]
                            self.assertAlmostEqual(
                                even,
                                1.0 + eta * eta_prime * eta_2 * eta_3,
                            )
                            self.assertAlmostEqual(
                                odd,
                                1j * (eta * eta_prime * eta_2 - eta_3),
                            )

    def test_odd_branch_restriction_obeys_both_virasoro_ward_identities(self) -> None:
        """Audit equation (4.10) before performing any sewn branch sum."""

        b = 1.17
        cut = embedded_branch_state(
            b=b,
            sector="NS",
            physical_momentum=0.41,
            branch_number=Fraction(0),
            parity=None,
        )
        edge_2 = embedded_branch_state(
            b=b,
            sector="R",
            physical_momentum=0.37,
            branch_number=Fraction(-1, 4),
            parity=0,
        )
        edge_3 = embedded_branch_state(
            b=b,
            sector="R",
            physical_momentum=0.52,
            branch_number=Fraction(-1, 4),
            parity=1,
        )

        def state_map(branch):
            return dict(zip(branch.basis, branch.coefficients))

        def descended_map(branch, copy):
            result = {}
            for state, state_coefficient in state_map(branch).items():
                for output, coefficient in _embedded_l_minus_one_action(
                    state,
                    sector=branch.sector,
                    copy=copy,
                    b=b,
                    h=branch.super_weight,
                    c=branch.super_central_charge,
                ).items():
                    result[output] = result.get(output, 0.0j) + (
                        state_coefficient * coefficient
                    )
            return result

        def form(edge_2_states, edge_3_states, copy):
            total = 0.0j
            for state_1, coefficient_1 in state_map(cut).items():
                for state_2, coefficient_2 in edge_2_states.items():
                    for state_3, coefficient_3 in edge_3_states.items():
                        total += (
                            coefficient_1
                            * coefficient_2
                            * coefficient_3
                            * _theta_elementary_product_form(
                                state_1,
                                state_2,
                                state_3,
                                orientation="left",
                                h_ns=cut.super_weight,
                                h_ramond_2=edge_2.super_weight,
                                h_ramond_3=edge_3.super_weight,
                                c=cut.super_central_charge,
                                beta_2=1j * edge_2.physical_momentum / math.sqrt(2),
                                beta_3=1j * edge_3.physical_momentum / math.sqrt(2),
                                structure_sign=1,
                                super_form_parity=1,
                                auxiliary_form_parity=0,
                            )
                        )
            return total

        base = form(state_map(edge_2), state_map(edge_3), 1)
        self.assertGreater(abs(base), 1.0e-8)
        for copy in (1, 2):
            h_cut = cut.parameters.h_1 if copy == 1 else cut.parameters.h_2
            h_2 = edge_2.parameters.h_1 if copy == 1 else edge_2.parameters.h_2
            h_3 = edge_3.parameters.h_1 if copy == 1 else edge_3.parameters.h_2
            middle = form(descended_map(edge_2, copy), state_map(edge_3), copy)
            zero = form(state_map(edge_2), descended_map(edge_3, copy), copy)
            self.assertAlmostEqual(middle, (h_cut - h_2 - h_3) * base)
            self.assertAlmostEqual(zero, (h_2 + h_3 - h_cut) * base)

    def test_necklace_branch_restrictions_obey_local_virasoro_wards(self) -> None:
        """Check each cyclicly ordered branch form before sewing it."""

        b = 1.17
        middle = embedded_branch_state(
            b=b,
            sector="NS",
            physical_momentum=0.41,
            branch_number=Fraction(0),
            parity=None,
        )
        previous = embedded_branch_state(
            b=b,
            sector="R",
            physical_momentum=0.37,
            branch_number=Fraction(-1, 4),
            parity=0,
        )
        current = embedded_branch_state(
            b=b,
            sector="R",
            physical_momentum=0.52,
            branch_number=Fraction(1, 4),
            parity=0,
        )

        def state_map(branch):
            return dict(zip(branch.basis, branch.coefficients))

        def descended_map(branch, copy):
            result = {}
            for state, state_coefficient in state_map(branch).items():
                for output, coefficient in _embedded_l_minus_one_action(
                    state,
                    sector=branch.sector,
                    copy=copy,
                    b=b,
                    h=branch.super_weight,
                    c=branch.super_central_charge,
                ).items():
                    result[output] = result.get(output, 0.0j) + (
                        state_coefficient * coefficient
                    )
            return result

        def form(infinity, zero, infinity_states, zero_states, orientation):
            total = 0.0j
            for state_infinity, coefficient_infinity in infinity_states.items():
                for state_middle, coefficient_middle in state_map(middle).items():
                    for state_zero, coefficient_zero in zero_states.items():
                        total += (
                            coefficient_infinity
                            * coefficient_middle
                            * coefficient_zero
                            * _elementary_product_form(
                                state_infinity,
                                state_middle,
                                state_zero,
                                orientation=orientation,
                                h_infinity=infinity.super_weight,
                                h_middle=middle.super_weight,
                                h_zero=zero.super_weight,
                                c=middle.super_central_charge,
                                beta_infinity=(
                                    1j * infinity.physical_momentum / math.sqrt(2)
                                ),
                                beta_zero=(
                                    1j * zero.physical_momentum / math.sqrt(2)
                                ),
                                structure_sign=-1,
                                super_form_parity=None,
                                auxiliary_form_parity=0,
                                auxiliary_parity_twist=1,
                            )
                        )
            return total

        for infinity, zero, orientation in (
            (previous, current, "left"),
            (current, previous, "right"),
        ):
            base = form(
                infinity,
                zero,
                state_map(infinity),
                state_map(zero),
                orientation,
            )
            self.assertGreater(abs(base), 1.0e-8)
            for copy in (1, 2):
                h_infinity = (
                    infinity.parameters.h_1
                    if copy == 1
                    else infinity.parameters.h_2
                )
                h_middle = (
                    middle.parameters.h_1
                    if copy == 1
                    else middle.parameters.h_2
                )
                h_zero = (
                    zero.parameters.h_1
                    if copy == 1
                    else zero.parameters.h_2
                )
                infinity_value = form(
                    infinity,
                    zero,
                    descended_map(infinity, copy),
                    state_map(zero),
                    orientation,
                )
                zero_value = form(
                    infinity,
                    zero,
                    state_map(infinity),
                    descended_map(zero, copy),
                    orientation,
                )
                self.assertAlmostEqual(
                    infinity_value,
                    (h_infinity + h_middle - h_zero) * base,
                )
                self.assertAlmostEqual(
                    zero_value,
                    (-h_infinity + h_middle + h_zero) * base,
                )

    def test_recursive_ramond_necklace_matches_direct_pp_and_gg(self) -> None:
        """Exercise recursion residues against the level-two direct oracle."""

        b = 1.17
        previous_momentum = 0.37
        current_momentum = 0.52
        external_momentum = 0.41
        central_charge = super_liouville_central_charge(b)

        def ramond_weight_at_b(momentum):
            return central_charge / 24.0 + momentum * momentum / 2.0

        def beta(momentum):
            return 1j * momentum / math.sqrt(2.0)

        signs = (-1, -1)
        tensors = (
            hjs_to_polynomial_ground_tensor(
                beta(previous_momentum),
                beta(current_momentum),
                structure_sign=signs[0],
            ),
            hjs_to_polynomial_ground_tensor(
                beta(current_momentum),
                beta(previous_momentum),
                structure_sign=signs[1],
            ),
        )
        recursive = generic_ramond_torus_two_point_necklace_coefficients(
            b=b,
            previous_internal_momentum=previous_momentum,
            current_internal_momentum=current_momentum,
            external_ns_momentum=external_momentum,
            left_structure_sign=signs[0],
            right_structure_sign=signs[1],
            maximum_previous_level=2,
            maximum_current_level=2,
        )
        half_mode = (G(Fraction(-1, 2)),)
        for words, recursive_table in (
            (((), ()), recursive.primary_primary),
            (
                (half_mode, half_mode),
                recursive.superdescendant_superdescendant,
            ),
        ):
            direct = direct_ramond_torus_necklace_series(
                c=central_charge,
                internal_weights=(
                    ramond_weight_at_b(current_momentum),
                    ramond_weight_at_b(previous_momentum),
                ),
                external_weights=(
                    ns_liouville_weight(external_momentum, b),
                )
                * 2,
                ground_tensors=tensors,
                maximum_twice_levels=4,
                edge_lift_signs=(1, 1),
                external_words=words,
            )
            direct_in_recursive_order = {
                (previous_twice_level // 2, current_twice_level // 2): value
                for (
                    current_twice_level,
                    previous_twice_level,
                ), value in direct.coefficients.items()
            }
            for key, expected in direct_in_recursive_order.items():
                with self.subTest(words=words, levels=key):
                    self.assertAlmostEqual(recursive_table[key], expected)

    def test_recursive_ramond_necklace_covers_hjs_signs_and_even_lifts(self) -> None:
        """Check all HJS structures in both invertible equal-lift traces."""

        b = 1.17
        previous_momentum = 0.37
        current_momentum = 0.52
        external_momentum = 0.41
        central_charge = super_liouville_central_charge(b)
        external_weight = ns_liouville_weight(external_momentum, b)
        ramond_weight = lambda momentum: (
            central_charge / 24.0 + momentum * momentum / 2.0
        )
        beta = lambda momentum: 1j * momentum / math.sqrt(2.0)
        half_mode = (G(Fraction(-1, 2)),)

        for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            tensors = (
                hjs_to_polynomial_ground_tensor(
                    beta(previous_momentum),
                    beta(current_momentum),
                    structure_sign=signs[0],
                ),
                hjs_to_polynomial_ground_tensor(
                    beta(current_momentum),
                    beta(previous_momentum),
                    structure_sign=signs[1],
                ),
            )
            for lift in (-1, 1):
                recursive = generic_ramond_torus_two_point_necklace_coefficients(
                    b=b,
                    previous_internal_momentum=previous_momentum,
                    current_internal_momentum=current_momentum,
                    external_ns_momentum=external_momentum,
                    left_structure_sign=signs[0],
                    right_structure_sign=signs[1],
                    previous_lift_sign=lift,
                    current_lift_sign=lift,
                    maximum_previous_level=1,
                    maximum_current_level=1,
                )
                for words, recursive_table in (
                    (((), ()), recursive.primary_primary),
                    (
                        (half_mode, half_mode),
                        recursive.superdescendant_superdescendant,
                    ),
                ):
                    direct = direct_ramond_torus_necklace_series(
                        c=central_charge,
                        internal_weights=(
                            ramond_weight(current_momentum),
                            ramond_weight(previous_momentum),
                        ),
                        external_weights=(external_weight, external_weight),
                        ground_tensors=tensors,
                        maximum_twice_levels=2,
                        edge_lift_signs=(lift, lift),
                        external_words=words,
                    )
                    direct_in_recursive_order = {
                        (
                            previous_twice_level // 2,
                            current_twice_level // 2,
                        ): value
                        for (
                            current_twice_level,
                            previous_twice_level,
                        ), value in direct.coefficients.items()
                    }
                    for key, expected in direct_in_recursive_order.items():
                        with self.subTest(
                            signs=signs,
                            lift=lift,
                            words=words,
                            levels=key,
                        ):
                            self.assertAlmostEqual(
                                recursive_table[key],
                                expected,
                            )

        with self.assertRaisesRegex(ValueError, "not invertible"):
            generic_ramond_torus_two_point_necklace_coefficients(
                b=b,
                previous_internal_momentum=previous_momentum,
                current_internal_momentum=current_momentum,
                external_ns_momentum=external_momentum,
                left_structure_sign=-1,
                right_structure_sign=1,
                previous_lift_sign=-1,
                current_lift_sign=1,
                maximum_previous_level=0,
                maximum_current_level=0,
            )

    def test_self_dual_ramond_projection_is_coefficientwise(self) -> None:
        """Compare the true b=1 finite part with the direct Ward oracle."""

        previous_momentum = 0.37
        current_momentum = 0.52
        external_momentum = 0.41
        signs = (-1, -1)
        beta = lambda momentum: 1j * momentum / math.sqrt(2.0)
        tensors = (
            hjs_to_polynomial_ground_tensor(
                beta(previous_momentum),
                beta(current_momentum),
                structure_sign=signs[0],
            ),
            hjs_to_polynomial_ground_tensor(
                beta(current_momentum),
                beta(previous_momentum),
                structure_sign=signs[1],
            ),
        )
        recursive = self_dual_ramond_torus_two_point_necklace_coefficients(
            previous_internal_momentum=previous_momentum,
            current_internal_momentum=current_momentum,
            external_ns_momentum=external_momentum,
            left_structure_sign=signs[0],
            right_structure_sign=signs[1],
            maximum_previous_level=1,
            maximum_current_level=1,
            radius=0.035,
            check_radius=0.05,
            samples=8,
        )
        half_mode = (G(Fraction(-1, 2)),)
        for words, recursive_table in (
            (((), ()), recursive.primary_primary),
            (
                (half_mode, half_mode),
                recursive.superdescendant_superdescendant,
            ),
        ):
            direct = direct_ramond_torus_necklace_series(
                c=13.5,
                internal_weights=(
                    13.5 / 24.0 + current_momentum**2 / 2.0,
                    13.5 / 24.0 + previous_momentum**2 / 2.0,
                ),
                external_weights=(
                    ns_liouville_weight(external_momentum, 1.0),
                )
                * 2,
                ground_tensors=tensors,
                maximum_twice_levels=2,
                edge_lift_signs=(1, 1),
                external_words=words,
            )
            for (current_level, previous_level), expected in direct.coefficients.items():
                key = (previous_level // 2, current_level // 2)
                with self.subTest(words=words, levels=key):
                    self.assertAlmostEqual(recursive_table[key], expected)

        # The complete generic-b branch sum is invariant under b -> 1/b, so
        # the two eight-point contours require four evaluations each.
        self.assertEqual(recursive.generic_evaluations, 8)
        self.assertLess(
            max(
                diagnostic.absolute_error
                for diagnostic in recursive.primary_diagnostics.values()
            ),
            1.0e-9,
        )
        self.assertLess(
            max(
                diagnostic.absolute_error
                for diagnostic in recursive.superdescendant_diagnostics.values()
            ),
            1.0e-9,
        )

    def test_complete_generic_necklace_is_b_inversion_symmetric(self) -> None:
        """Check the exact self-duality used to halve contour evaluations."""

        b = cmath.exp(0.035 * cmath.exp(0.37j))
        for signs in ((1, 1), (1, -1)):
            options = dict(
                previous_internal_momentum=0.37,
                current_internal_momentum=0.52,
                external_ns_momentum=0.41,
                left_structure_sign=signs[0],
                right_structure_sign=signs[1],
                maximum_previous_level=1,
                maximum_current_level=1,
            )
            forward = generic_ramond_torus_two_point_necklace_coefficients(
                b=b,
                **options,
            )
            inverted = generic_ramond_torus_two_point_necklace_coefficients(
                b=1.0 / b,
                **options,
            )
            for name in (
                "primary_primary",
                "superdescendant_superdescendant",
            ):
                forward_table = getattr(forward, name)
                inverted_table = getattr(inverted, name)
                with self.subTest(signs=signs, component=name):
                    self.assertLess(
                        max(
                            abs(forward_table[key] - inverted_table[key])
                            for key in forward_table
                        ),
                        1.0e-10,
                    )
    def test_sphere_ground_coefficient_is_explicit_doublet_contraction(self) -> None:
        """Compare level zero with a hand-written two-by-two contraction."""

        c = 13.5
        h_internal = 1.37
        left = sp.Matrix([[1, sp.Rational(2, 3)], [sp.Rational(3, 5), sp.Rational(4, 7)]])
        right = sp.Matrix([[sp.Rational(5, 4), sp.Rational(1, 2)], [sp.Rational(2, 5), sp.Rational(7, 6)]])
        external_states = (ground_state(1), ground_state(0))
        series = direct_ramond_sphere_block_series(
            c=c,
            h_internal=h_internal,
            external_weights=(0.8, 0.4, 0.5, 0.9),
            left_ground_tensor=left,
            right_ground_tensor=right,
            maximum_twice_level=0,
            form_parities=(0, 1),
            ramond_states=external_states,
        )
        # The selected left/right form parities retain only the even
        # internal ground state for these external parities.
        expected = complex(left[0, 0]) * complex(right[0, 1])
        self.assertAlmostEqual(series.coefficients[0], expected)

    def test_hjs_wrapper_keeps_internal_basis_change_inside_sewing(self) -> None:
        """Check the HJS wrapper against the explicit polynomial conversion."""

        c = 13.5
        beta_internal = 0.7
        beta_one = 0.3
        beta_four = 0.4
        wrapped = hjs_ramond_sphere_block_series(
            c=c,
            beta_internal=beta_internal,
            beta_one=beta_one,
            beta_four=beta_four,
            h_two=0.41,
            h_three=0.53,
            maximum_twice_level=4,
            left_structure_sign=-1,
            right_structure_sign=1,
        )
        direct = direct_ramond_sphere_block_series(
            c=c,
            h_internal=ramond_weight(c, beta_internal),
            external_weights=(
                ramond_weight(c, beta_one),
                0.41,
                0.53,
                ramond_weight(c, beta_four),
            ),
            left_ground_tensor=hjs_to_polynomial_ground_tensor(
                beta_four,
                beta_internal,
                structure_sign=-1,
            ),
            right_ground_tensor=hjs_to_polynomial_ground_tensor(
                beta_internal,
                beta_one,
                structure_sign=1,
            ),
            maximum_twice_level=4,
        )
        self.assertEqual(wrapped.external_ground_basis, "hjs")
        for level in wrapped.coefficients:
            self.assertAlmostEqual(wrapped.coefficients[level], direct.coefficients[level])

    def test_hjs_external_odd_state_conversion(self) -> None:
        """Check the nontrivial zero-slot HJS phase on an odd external state."""

        c = 13.5
        beta_internal = 0.7
        beta_one = 0.3
        beta_four = 0.4
        states = (ground_state(1), ground_state(0))
        wrapped = hjs_ramond_sphere_block_series(
            c=c,
            beta_internal=beta_internal,
            beta_one=beta_one,
            beta_four=beta_four,
            h_two=0.41,
            h_three=0.53,
            maximum_twice_level=2,
            form_parities=(0, 1),
            ramond_states=states,
        )
        direct = direct_ramond_sphere_block_series(
            c=c,
            h_internal=ramond_weight(c, beta_internal),
            external_weights=(
                ramond_weight(c, beta_one),
                0.41,
                0.53,
                ramond_weight(c, beta_four),
            ),
            left_ground_tensor=hjs_to_polynomial_ground_tensor(
                beta_four, beta_internal, structure_sign=1
            ),
            right_ground_tensor=hjs_to_polynomial_ground_tensor(
                beta_internal, beta_one, structure_sign=1
            ),
            maximum_twice_level=2,
            form_parities=(0, 1),
            ramond_states=states,
        )
        phase = (1 + 1j) / math.sqrt(2.0)
        for level in wrapped.coefficients:
            self.assertAlmostEqual(
                wrapped.coefficients[level],
                direct.coefficients[level] / (phase * beta_one),
            )

    def test_torus_identity_is_doubled_ramond_character(self) -> None:
        """Recover ``2 prod_n (1+q^n)/(1-q^n)`` through level three."""

        c = 13.5
        h = 1.37
        terminal = ground_gram_matrix(sp.Float(h), sp.Float(c))
        series = direct_ramond_torus_one_point_series(
            c=c,
            h_internal=h,
            h_external=0,
            ground_tensor=terminal,
            maximum_twice_level=6,
            lift_sign=1,
        )
        expected = {0: 2, 2: 4, 4: 8, 6: 16}
        for power, coefficient in expected.items():
            self.assertAlmostEqual(series.coefficients[power], coefficient)

    def test_identity_supertrace_and_odd_trace_vanish(self) -> None:
        """Check parity cancellation and absence of an unsaturated G0 trace."""

        c = 13.5
        h = 1.37
        terminal = ground_gram_matrix(sp.Float(h), sp.Float(c))
        supertrace = direct_ramond_torus_one_point_series(
            c=c,
            h_internal=h,
            h_external=0,
            ground_tensor=terminal,
            maximum_twice_level=6,
            lift_sign=-1,
        )
        odd = direct_ramond_torus_one_point_series(
            c=c,
            h_internal=h,
            h_external=0,
            ground_tensor=terminal,
            maximum_twice_level=6,
            kernel_kind="odd",
        )
        self.assertTrue(all(abs(value) < 1.0e-12 for value in supertrace.coefficients.values()))
        self.assertTrue(all(abs(value) < 1.0e-12 for value in odd.coefficients.values()))

    def test_odd_hjs_ground_state_rejects_shortening_point(self) -> None:
        """Do not divide by the singular HJS odd vector at beta zero."""

        with self.assertRaises(ValueError):
            hjs_ramond_sphere_block_series(
                c=13.5,
                beta_internal=0.7,
                beta_one=0.0,
                beta_four=0.4,
                h_two=0.41,
                h_three=0.53,
                maximum_twice_level=0,
                ramond_states=(ground_state(1), ground_state(0)),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
