#!/usr/bin/env python3
"""Focused checks for the genus-one Spin(23) CFT ingredients."""

from __future__ import annotations

import math
import unittest

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import G, pbw_basis as ns_pbw_basis
from ramond_algebra.ramond_plumbing import ramond_plumbing_kernels
from ramond_algebra.ramond_sca import ground_gram_matrix
from spin23_genus1_blocks import (
    _ramond_edge_data,
    b1_ramond_liouville_necklace_series,
    direct_ns_torus_necklace_series,
    direct_ramond_torus_necklace_series,
)
from spin23_genus1_free_fields import (
    dedekind_eta,
    flavored_majorana_wick_factor,
    majorana_wick_factor,
    odd_szego_nonzero_mode,
    pfaffian,
    prime_form,
    szego_kernel,
    theta_characteristic,
    torus_scalar_green,
)
from spin23_genus1_spin import (
    TorusSpinStructure,
    genus_one_ns_pco_prescription,
    torus_spin_structures,
)
from spin23_generated.ns_b1_kernels import (
    evaluate_gram as evaluate_generated_ns_gram,
    evaluate_vertex as evaluate_generated_ns_vertex,
)
from spin23_generated.ns_b1_level5_x3_kernels import (
    evaluate_gram as evaluate_sparse_ns_gram,
    evaluate_vertex as evaluate_sparse_ns_vertex,
)
from spin23_ns_fast import (
    _B1_P as _NS_B1_P,
    _VERTEX_H_NS as _NS_VERTEX_H,
    _VERTEX_P_LEFT as _NS_VERTEX_P_LEFT,
    _VERTEX_P_RIGHT as _NS_VERTEX_P_RIGHT,
    _b1_ns_gram_expressions,
    _b1_ns_vertex_expressions,
    _fast_edge_factor as _fast_ns_edge_factor,
    _prewhitened_vertex_coefficients as _prewhitened_ns_vertex_coefficients,
    direct_fast_b1_ns_necklace_series,
    direct_fast_b1_ns_two_point_polynomial_series,
)
from spin23_ramond_blocks import direct_ramond_torus_one_point_series
from spin23_ramond_fast import (
    _B1_P,
    _VERTEX_H_NS,
    _VERTEX_P_LEFT,
    _VERTEX_P_RIGHT,
    _fast_edge_factor,
    _normalized_gram_expressions,
    _normalized_vertex_sign_expressions,
    _parity_ordered_basis,
    _prewhitened_vertex_sign_coefficients,
    direct_fast_b1_ramond_necklace_series,
    direct_fast_b1_ramond_two_sign_polynomial_series,
    direct_fast_b1_ramond_two_sign_series,
    q_weighted_series_discrepancy,
)
from spin23_generated.ramond_b1_kernels import (
    evaluate_gram as evaluate_generated_ramond_gram,
    evaluate_vertex_parts as evaluate_generated_ramond_vertex_parts,
)
from spin23_generated.ramond_b1_level5_x3_kernels import (
    evaluate_gram as evaluate_sparse_ramond_gram,
    evaluate_vertex_parts as evaluate_sparse_ramond_vertex_parts,
)
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_chiral_structure_constant,
    rr_ns_structure_constants,
    upsilon_1,
)


class SuperLiouvilleDataTests(unittest.TestCase):
    """Check normalization, permutation symmetry, and fixed reference values."""

    def test_upsilon_normalization(self) -> None:
        self.assertAlmostEqual(upsilon_1(1.0), 1.0 + 0.0j, places=14)

    def test_structure_constant_reference_values(self) -> None:
        ns_even, ns_odd = ns_structure_constants(0.37, 0.52, 0.41, precision=60)
        rr_even, rr_odd = rr_ns_structure_constants(0.37, 0.52, 0.41, precision=60)
        self.assertAlmostEqual(ns_even, 0.5430163965269362, places=13)
        self.assertAlmostEqual(ns_odd, 0.0835427156034286, places=13)
        self.assertAlmostEqual(rr_even, 0.35981591223437687, places=13)
        self.assertAlmostEqual(rr_odd, 0.22019728810106723, places=13)

    def test_structure_constant_cache_reuses_only_identical_inputs(self) -> None:
        ns_structure_constants.cache_clear()
        first = ns_structure_constants(0.29, 0.43, 0.51, precision=45)
        second = ns_structure_constants(0.29, 0.43, 0.51, precision=45)
        changed_precision = ns_structure_constants(
            0.29,
            0.43,
            0.51,
            precision=46,
        )
        cache = ns_structure_constants.cache_info()
        self.assertIs(first, second)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 2)
        for left, right in zip(first, changed_precision):
            self.assertAlmostEqual(left, right, places=13)

    def test_rr_exchange_and_hjs_dictionary(self) -> None:
        direct = rr_ns_structure_constants(0.31, 0.57, 0.44, precision=55)
        exchanged = rr_ns_structure_constants(0.57, 0.31, 0.44, precision=55)
        for left, right in zip(direct, exchanged):
            self.assertAlmostEqual(left, right, places=13)
        self.assertAlmostEqual(
            rr_ns_chiral_structure_constant(
                0.31, 0.57, 0.44, structure_sign=1, precision=55
            ),
            direct[0],
            places=14,
        )
        self.assertAlmostEqual(
            rr_ns_chiral_structure_constant(
                0.31, 0.57, 0.44, structure_sign=-1, precision=55
            ),
            direct[1],
            places=14,
        )

    def test_complex128_structure_constants_are_stable_at_24_digits(self) -> None:
        samples = (
            (1.0e-6, 0.1, 0.35),
            (0.37, 0.52, 0.35),
            (1.0, 2.0, 0.35),
            (4.0, 4.0, 0.35),
        )
        for builder in (ns_structure_constants, rr_ns_structure_constants):
            for momenta in samples:
                with self.subTest(builder=builder.__name__, momenta=momenta):
                    production = builder(*momenta, precision=24)
                    audit = builder(*momenta, precision=80)
                    for actual, expected in zip(production, audit):
                        self.assertTrue(
                            np.isclose(
                                actual,
                                expected,
                                rtol=2.0e-14,
                                atol=2.0e-14,
                            )
                        )


class SpinAndPCOTests(unittest.TestCase):
    """Check the four traces and the even/odd PCO prescriptions."""

    def test_spin_structure_ledger(self) -> None:
        structures = torus_spin_structures()
        self.assertEqual(
            [structure.label for structure in structures],
            ["NS", "NS_tilde", "R", "R_tilde"],
        )
        self.assertEqual(sum(s.arf_invariant for s in structures), 1)
        self.assertEqual(
            [s.lift_sign for s in structures],
            [1, -1, 1, -1],
        )

    def test_even_and_odd_pco_words(self) -> None:
        even = genus_one_ns_pco_prescription(
            3,
            TorusSpinStructure(False, False),
        )
        odd = genus_one_ns_pco_prescription(
            3,
            TorusSpinStructure(True, True),
            distinguished_puncture=1,
        )
        self.assertEqual(even.distinguished_puncture, None)
        self.assertTrue(all(word == (G(sp.Rational(-1, 2)),) for word in even.external_words))
        self.assertEqual(odd.external_words[0], (G(sp.Rational(-1, 2)),))
        self.assertEqual(odd.external_words[1], (G(sp.Rational(-3, 2)),))
        self.assertEqual(odd.external_words[2], (G(sp.Rational(-1, 2)),))


class NecklaceDegenerationTests(unittest.TestCase):
    """Test identities, cyclic sewing, characters, and plumbing limits."""

    c = 13.5
    h = 1.37
    external_h = 0.41

    def test_ns_one_point_low_level_ward_coefficient(self) -> None:
        series = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[self.external_h],
            maximum_twice_levels=1,
        )
        expected_half_level = (2 * self.h - self.external_h) / (2 * self.h)
        self.assertAlmostEqual(series.coefficients[(0,)], 1.0)
        self.assertAlmostEqual(series.coefficients[(1,)], expected_half_level)

    def test_ns_identity_puncture_collapses_necklace(self) -> None:
        one_point = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[self.external_h],
            maximum_twice_levels=3,
        )
        two_point = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h, self.h],
            external_weights=[self.external_h, 0],
            maximum_twice_levels=3,
            # The identity has only its even three-point form.
            form_weights=[(1, 1), (1, 0)],
        )
        for left_level in range(4):
            for right_level in range(4):
                coefficient = two_point.coefficients[(left_level, right_level)]
                if left_level == right_level:
                    self.assertAlmostEqual(
                        coefficient,
                        one_point.coefficients[(left_level,)],
                    )
                else:
                    self.assertAlmostEqual(coefficient, 0.0)
        q1, q2 = 0.07 + 0.01j, 0.11 - 0.02j
        self.assertAlmostEqual(
            two_point.descendant_value((q1, q2)),
            one_point.descendant_value((q1 * q2,)),
        )

    def test_ns_two_edge_coefficients_match_closed_ward_formulas(self) -> None:
        h1, h2 = 1.21, 1.43
        d1, d2 = 0.31, 0.46
        series = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[h1, h2],
            external_weights=[d1, d2],
            maximum_twice_levels=2,
            form_weights=[(1, 0), (1, 0)],
        )
        expected_10 = (h1 + d1 - h2) * (h1 + d2 - h2) / (2 * h1)
        expected_01 = (h2 + d1 - h1) * (h2 + d2 - h1) / (2 * h2)
        expected_half_half = (
            (h1 + h2 - d1) * (h1 + h2 - d2) / (4 * h1 * h2)
        )
        self.assertAlmostEqual(series.coefficients[(2, 0)], expected_10)
        self.assertAlmostEqual(series.coefficients[(0, 2)], expected_01)
        self.assertAlmostEqual(series.coefficients[(1, 1)], expected_half_half)
        self.assertAlmostEqual(series.coefficients[(1, 0)], 0.0)
        self.assertAlmostEqual(series.coefficients[(0, 1)], 0.0)

    def test_ns_identity_is_spin_lifted_character(self) -> None:
        ordinary = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[0],
            maximum_twice_levels=4,
            form_weights=[(1, 0)],
        )
        parity = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[0],
            maximum_twice_levels=4,
            edge_lift_signs=[-1],
            form_weights=[(1, 0)],
        )
        expected = {0: 1, 1: 1, 2: 1, 3: 2, 4: 3}
        for level, multiplicity in expected.items():
            self.assertAlmostEqual(ordinary.coefficients[(level,)], multiplicity)
            self.assertAlmostEqual(
                parity.coefficients[(level,)],
                (-1) ** level * multiplicity,
            )

    def test_ramond_necklace_matches_independent_one_point(self) -> None:
        tensor = sp.Matrix([[1, sp.Rational(1, 3)], [sp.Rational(2, 5), 1]])
        necklace = direct_ramond_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[self.external_h],
            ground_tensors=[tensor],
            maximum_twice_levels=2,
        )
        independent = direct_ramond_torus_one_point_series(
            c=self.c,
            h_internal=self.h,
            h_external=self.external_h,
            ground_tensor=tensor,
            maximum_twice_level=2,
        )
        for level in (0, 2):
            self.assertAlmostEqual(
                necklace.coefficients[(level,)],
                independent.coefficients[level],
            )

    def test_numeric_ramond_edge_kernel_matches_exact_plumbing_oracle(self) -> None:
        for lift_sign in (-1, 1):
            with self.subTest(lift_sign=lift_sign):
                _, numeric, _ = _ramond_edge_data(
                    4,
                    complex(self.h),
                    complex(self.c),
                    lift_sign,
                    55,
                )
                exact = ramond_plumbing_kernels(
                    4,
                    h=sp.Float(self.h, 60),
                    c=sp.Float(self.c, 60),
                    lift_sign=lift_sign,
                ).even_kernel
                expected = np.asarray(exact.tolist(), dtype=np.complex128)
                self.assertTrue(
                    np.allclose(
                        numeric,
                        expected,
                        rtol=2.0e-12,
                        atol=2.0e-12,
                    )
                )

    def test_ramond_identity_puncture_collapses_necklace(self) -> None:
        tensor = sp.Matrix([[1, sp.Rational(1, 3)], [sp.Rational(2, 5), 1]])
        identity_tensor = ground_gram_matrix(sp.Float(self.h), sp.Float(self.c))
        one_point = direct_ramond_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h],
            external_weights=[self.external_h],
            ground_tensors=[tensor],
            maximum_twice_levels=2,
        )
        two_point = direct_ramond_torus_necklace_series(
            c=self.c,
            internal_weights=[self.h, self.h],
            external_weights=[self.external_h, 0],
            ground_tensors=[tensor, identity_tensor],
            maximum_twice_levels=2,
        )
        for left_level in (0, 2):
            for right_level in (0, 2):
                coefficient = two_point.coefficients[(left_level, right_level)]
                if left_level == right_level:
                    self.assertAlmostEqual(
                        coefficient,
                        one_point.coefficients[(left_level,)],
                    )
                else:
                    self.assertAlmostEqual(coefficient, 0.0)

    def test_compiled_ramond_vertices_match_direct_oracle_through_level_two(self) -> None:
        tensors = (
            sp.Matrix([[1, sp.Rational(1, 3)], [sp.Rational(2, 5), 1]]),
            sp.Matrix([[sp.Rational(7, 6), -sp.Rational(1, 4)], [0, 1]]),
        )
        options = dict(
            c=self.c,
            internal_weights=[self.h, self.h + 0.17],
            external_weights=[self.external_h, self.external_h + 0.11],
            ground_tensors=tensors,
            maximum_twice_levels=4,
            edge_lift_signs=[-1, 1],
            external_words=[(G(sp.Rational(-1, 2)),), ()],
        )
        direct = direct_ramond_torus_necklace_series(
            **options,
            vertex_backend="direct",
        )
        compiled = direct_ramond_torus_necklace_series(
            **options,
            vertex_backend="template",
        )
        self.assertEqual(set(direct.coefficients), set(compiled.coefficients))
        for levels, expected in direct.coefficients.items():
            self.assertTrue(
                np.isclose(
                    compiled.coefficients[levels],
                    expected,
                    rtol=2.0e-12,
                    atol=2.0e-12,
                ),
                msg=f"compiled Ramond coefficient mismatch at {levels}",
            )

    def test_conditioned_b1_ramond_backend_matches_direct_oracle(self) -> None:
        options = dict(
            internal_momenta=(0.37, 0.52),
            external_ns_momenta=(0.41, 0.41),
            structure_signs=(-1, -1),
            maximum_twice_levels=(4, 4),
            edge_lift_signs=(1, -1),
            external_words=((), ()),
        )
        oracle = b1_ramond_liouville_necklace_series(**options)
        fast = direct_fast_b1_ramond_necklace_series(**options)
        self.assertEqual(set(fast.coefficients), set(oracle.coefficients))
        for levels, expected in oracle.coefficients.items():
            self.assertTrue(
                np.isclose(
                    fast.coefficients[levels],
                    expected,
                    rtol=3.0e-12,
                    atol=3.0e-12,
                ),
                msg=f"conditioned Ramond coefficient mismatch at {levels}",
            )

    def test_conditioned_b1_ns_backend_matches_direct_oracle(self) -> None:
        internal_momenta = (0.37, 0.52)
        external_momenta = (0.41, 0.41)
        words = ((G(sp.Rational(-1, 2)),),) * 2
        common = dict(
            maximum_twice_levels=(4, 3),
            edge_lift_signs=(1, -1),
            external_words=words,
            form_weights=((0.0, 1.0), (0.0, 1.0)),
        )
        oracle = direct_ns_torus_necklace_series(
            c=13.5,
            internal_weights=tuple(
                ns_weight(momentum) for momentum in internal_momenta
            ),
            external_weights=tuple(
                ns_weight(momentum) for momentum in external_momenta
            ),
            vertex_backend="template",
            **common,
        )
        fast = direct_fast_b1_ns_necklace_series(
            internal_momenta=internal_momenta,
            external_ns_momenta=external_momenta,
            **common,
        )
        self.assertEqual(set(fast.coefficients), set(oracle.coefficients))
        self.assertLess(
            q_weighted_series_discrepancy(
                oracle,
                fast,
                (0.05556, 0.01085),
            ),
            2.0e-14,
        )

    def test_ns_whitened_polynomials_are_reused_across_external_weights(
        self,
    ) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            maximum_twice_levels=(4, 4),
            edge_lift_signs=(1, -1),
            external_words=((), ()),
            form_weights=((1.0, 0.0), (1.0, 0.0)),
        )
        _prewhitened_ns_vertex_coefficients.cache_clear()
        direct_fast_b1_ns_necklace_series(
            external_ns_momenta=(0.41, 0.41),
            **common,
        )
        after_first = _prewhitened_ns_vertex_coefficients.cache_info()
        direct_fast_b1_ns_necklace_series(
            external_ns_momenta=(0.47, 0.47),
            **common,
        )
        after_second = _prewhitened_ns_vertex_coefficients.cache_info()
        self.assertGreater(after_first.misses, 0)
        self.assertEqual(after_second.misses, after_first.misses)
        self.assertGreater(after_second.hits, after_first.hits)

    def test_ns_two_point_polynomial_reconstructs_direct_series(self) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            maximum_twice_levels=(4, 3),
            edge_lift_signs=(1, -1),
            external_words=((G(sp.Rational(-1, 2)),),) * 2,
            form_weights=((0.0, 1.0), (0.0, 1.0)),
        )
        external_momentum = 0.43 + 0.17j
        polynomial = direct_fast_b1_ns_two_point_polynomial_series(**common)
        reconstructed = polynomial.at_external_weight(
            ns_weight(external_momentum)
        )
        direct = direct_fast_b1_ns_necklace_series(
            external_ns_momenta=(external_momentum, external_momentum),
            **common,
        )
        self.assertLess(
            q_weighted_series_discrepancy(
                reconstructed,
                direct,
                (0.05556, 0.01085),
            ),
            3.0e-14,
        )

    def test_conditioned_b1_ns_endpoint_factorization_is_stable(self) -> None:
        factor = _fast_ns_edge_factor(8, 0.0, -1, 1.0e11)
        self.assertLess(factor.condition_number, 200.0)
        self.assertLess(factor.factorization_residual, 2.0e-15)
        self.assertTrue(
            np.allclose(
                factor.cholesky_lower @ factor.cholesky_lower.T,
                factor.equilibrated_gram,
                rtol=2.0e-13,
                atol=2.0e-13,
            )
        )

    def test_generated_ns_kernels_match_exact_symbolic_expressions(self) -> None:
        left_level, right_level = 6, 8
        p_left = sp.Rational(1, 3)
        p_right = sp.Rational(2, 5)
        h_ns = sp.Rational(7, 5)

        _, exact_gram = _b1_ns_gram_expressions(right_level)
        expected_gram = np.asarray(
            exact_gram.subs(_NS_B1_P, p_right),
            dtype=np.float64,
        )
        generated_gram = evaluate_generated_ns_gram(
            right_level,
            float(p_right),
        )
        self.assertTrue(
            np.allclose(
                generated_gram,
                expected_gram,
                rtol=3.0e-15,
                atol=3.0e-15,
            )
        )

        left_basis = tuple(spin_word for spin_word in ns_pbw_basis(left_level))
        right_basis = tuple(spin_word for spin_word in ns_pbw_basis(right_level))
        for component, word in (
            ("PP", ()),
            ("GG", (G(sp.Rational(-1, 2)),)),
        ):
            with self.subTest(component=component):
                exact_vertex = _b1_ns_vertex_expressions(
                    left_basis,
                    word,
                    right_basis,
                )
                expected_vertex = np.asarray(
                    exact_vertex.subs(
                        {
                            _NS_VERTEX_P_LEFT: p_left,
                            _NS_VERTEX_H: h_ns,
                            _NS_VERTEX_P_RIGHT: p_right,
                        }
                    ),
                    dtype=np.complex128,
                )
                generated_vertex = evaluate_generated_ns_vertex(
                    component,
                    left_level,
                    right_level,
                    float(p_left),
                    complex(h_ns),
                    float(p_right),
                )
                self.assertTrue(
                    np.allclose(
                        generated_vertex,
                        expected_vertex,
                        rtol=2.0e-14,
                        atol=2.0e-14,
                    )
                )

    def test_sparse_level5_ns_kernels_match_exact_expressions(self) -> None:
        p_left = sp.Rational(1, 3)
        p_right = sp.Rational(2, 5)
        h_ns = sp.Rational(7, 5)
        left_basis = tuple(ns_pbw_basis(10))
        right_basis = tuple(ns_pbw_basis(6))

        _, exact_gram = _b1_ns_gram_expressions(10)
        expected_gram = np.asarray(
            exact_gram.subs(_NS_B1_P, p_right),
            dtype=np.float64,
        )
        generated_gram = evaluate_sparse_ns_gram(10, float(p_right))
        self.assertTrue(
            np.allclose(
                generated_gram,
                expected_gram,
                rtol=2.0e-15,
                atol=2.0e-14,
            )
        )

        substitutions = {
            _NS_VERTEX_P_LEFT: p_left,
            _NS_VERTEX_H: h_ns,
            _NS_VERTEX_P_RIGHT: p_right,
        }
        for component, word in (
            ("PP", ()),
            ("GG", (G(sp.Rational(-1, 2)),)),
        ):
            with self.subTest(component=component):
                exact_vertex = _b1_ns_vertex_expressions(
                    left_basis,
                    word,
                    right_basis,
                )
                expected_vertex = np.asarray(
                    exact_vertex.subs(substitutions),
                    dtype=np.complex128,
                )
                generated_vertex = evaluate_sparse_ns_vertex(
                    component,
                    10,
                    6,
                    float(p_left),
                    complex(h_ns),
                    float(p_right),
                )
                self.assertTrue(
                    np.allclose(
                        generated_vertex,
                        expected_vertex,
                        rtol=3.0e-14,
                        atol=5.0e-14,
                    )
                )

    def test_sparse_level5_ns_backend_matches_direct_oracle(self) -> None:
        internal_momenta = (0.37, 0.52)
        external_momenta = (0.41, 0.41)
        common = dict(
            maximum_twice_levels=(10, 6),
            edge_lift_signs=(1, -1),
            external_words=((), ()),
            form_weights=((1.0, 0.0), (1.0, 0.0)),
        )
        oracle = direct_ns_torus_necklace_series(
            c=13.5,
            internal_weights=tuple(
                ns_weight(momentum) for momentum in internal_momenta
            ),
            external_weights=tuple(
                ns_weight(momentum) for momentum in external_momenta
            ),
            vertex_backend="template",
            **common,
        )
        fast = direct_fast_b1_ns_necklace_series(
            internal_momenta=internal_momenta,
            external_ns_momenta=external_momenta,
            **common,
        )
        self.assertLess(
            q_weighted_series_discrepancy(
                oracle,
                fast,
                (0.05556, 0.01085),
            ),
            2.0e-14,
        )

    def test_conditioned_b1_ramond_endpoint_is_q_weight_stable(self) -> None:
        options = dict(
            internal_momenta=(1.0e-4, 0.52),
            external_ns_momenta=(0.41, 0.41),
            structure_signs=(1, 1),
            maximum_twice_levels=(6, 6),
            external_words=((), ()),
        )
        oracle = b1_ramond_liouville_necklace_series(**options)
        fast = direct_fast_b1_ramond_necklace_series(**options)
        self.assertLess(
            q_weighted_series_discrepancy(
                oracle,
                fast,
                (0.05556, 0.01085),
            ),
            2.0e-8,
        )
        factor = _fast_edge_factor(6, 1.0e-4, 1, 1.0e11)
        self.assertLess(factor.condition_number, 100.0)
        self.assertTrue(
            np.allclose(
                factor.cholesky_lower @ factor.cholesky_lower.T,
                factor.equilibrated_gram,
                rtol=2.0e-13,
                atol=2.0e-13,
            )
        )
        endpoint = direct_fast_b1_ramond_necklace_series(
            **{**options, "internal_momenta": (0.0, 0.52)}
        )
        near_endpoint = direct_fast_b1_ramond_necklace_series(
            **{**options, "internal_momenta": (1.0e-8, 0.52)}
        )
        self.assertLess(
            q_weighted_series_discrepancy(
                endpoint,
                near_endpoint,
                (0.05556, 0.01085),
            ),
            2.0e-7,
        )

    def test_fused_hjs_sign_tables_match_independent_direct_fast_tables(self) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            external_ns_momenta=(0.41, 0.41),
            maximum_twice_levels=(4, 4),
            edge_lift_signs=(-1, 1),
            external_words=((), ()),
        )
        fused = direct_fast_b1_ramond_two_sign_series(
            **common,
            temporal_lift_sign=-1,
        )
        for first_sign in (-1, 1):
            independent = direct_fast_b1_ramond_necklace_series(
                **common,
                structure_signs=(first_sign, -first_sign),
            )
            self.assertLess(
                q_weighted_series_discrepancy(
                    fused[first_sign],
                    independent,
                    (0.05556, 0.01085),
                ),
                2.0e-14,
            )

    def test_ramond_whitened_polynomials_are_reused_across_external_weights(
        self,
    ) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            temporal_lift_sign=-1,
            maximum_twice_levels=(4, 4),
            edge_lift_signs=(-1, 1),
            external_words=((), ()),
        )
        _prewhitened_vertex_sign_coefficients.cache_clear()
        direct_fast_b1_ramond_two_sign_series(
            external_ns_momenta=(0.41, 0.41),
            **common,
        )
        after_first = _prewhitened_vertex_sign_coefficients.cache_info()
        direct_fast_b1_ramond_two_sign_series(
            external_ns_momenta=(0.47, 0.47),
            **common,
        )
        after_second = _prewhitened_vertex_sign_coefficients.cache_info()
        self.assertGreater(after_first.misses, 0)
        self.assertEqual(after_second.misses, after_first.misses)
        self.assertGreater(after_second.hits, after_first.hits)

    def test_ramond_two_sign_polynomial_reconstructs_fused_series(self) -> None:
        common = dict(
            internal_momenta=(0.37, 0.52),
            temporal_lift_sign=-1,
            maximum_twice_levels=(4, 4),
            edge_lift_signs=(-1, 1),
            external_words=((), ()),
        )
        external_momentum = 0.43 + 0.17j
        weight = ns_weight(external_momentum)
        polynomial = direct_fast_b1_ramond_two_sign_polynomial_series(**common)
        direct = direct_fast_b1_ramond_two_sign_series(
            external_ns_momenta=(external_momentum, external_momentum),
            **common,
        )
        for first_sign in (-1, 1):
            self.assertLess(
                q_weighted_series_discrepancy(
                    polynomial[first_sign].at_external_weight(weight),
                    direct[first_sign],
                    (0.05556, 0.01085),
                ),
                3.0e-14,
            )

    def test_generated_ramond_kernels_match_exact_symbolic_expressions(self) -> None:
        left_level, right_level = 2, 4
        p_left = sp.Rational(1, 3)
        p_right = sp.Rational(2, 5)
        h_ns = sp.Rational(7, 5)
        _, exact_gram = _normalized_gram_expressions(right_level)
        generated_gram = evaluate_generated_ramond_gram(
            right_level,
            float(p_right),
        )
        expected_gram = np.asarray(
            exact_gram.subs(_B1_P, p_right),
            dtype=np.float64,
        )
        self.assertTrue(
            np.allclose(
                generated_gram,
                expected_gram,
                rtol=2.0e-15,
                atol=2.0e-15,
            )
        )

        left_basis = _parity_ordered_basis(left_level)
        right_basis = _parity_ordered_basis(right_level)
        for component, word in (
            ("PP", ()),
            ("GG", (G(sp.Rational(-1, 2)),)),
        ):
            with self.subTest(component=component):
                exact_parts = _normalized_vertex_sign_expressions(
                    left_basis,
                    word,
                    right_basis,
                )
                substitutions = {
                    _VERTEX_P_LEFT: p_left,
                    _VERTEX_H_NS: h_ns,
                    _VERTEX_P_RIGHT: p_right,
                }
                expected = np.asarray(
                    [
                        np.asarray(
                            part.subs(substitutions),
                            dtype=np.complex128,
                        )
                        for part in exact_parts
                    ],
                    dtype=np.complex128,
                )
                generated = evaluate_generated_ramond_vertex_parts(
                    component,
                    left_level,
                    right_level,
                    float(p_left),
                    complex(h_ns),
                    float(p_right),
                )
                self.assertTrue(
                    np.allclose(
                        generated,
                        expected,
                        rtol=3.0e-15,
                        atol=3.0e-15,
                    )
                )

    def test_sparse_level5_ramond_kernels_match_exact_expressions(self) -> None:
        p_left = sp.Rational(1, 3)
        p_right = sp.Rational(2, 5)
        h_ns = sp.Rational(7, 5)
        left_basis = _parity_ordered_basis(10)
        right_basis = _parity_ordered_basis(0)

        _, exact_gram = _normalized_gram_expressions(10)
        expected_gram = np.asarray(
            exact_gram.subs(_B1_P, p_right),
            dtype=np.float64,
        )
        generated_gram = evaluate_sparse_ramond_gram(10, float(p_right))
        self.assertTrue(
            np.allclose(
                generated_gram,
                expected_gram,
                rtol=2.0e-15,
                atol=2.0e-14,
            )
        )

        substitutions = {
            _VERTEX_P_LEFT: p_left,
            _VERTEX_H_NS: h_ns,
            _VERTEX_P_RIGHT: p_right,
        }
        for component, word in (
            ("PP", ()),
            ("GG", (G(sp.Rational(-1, 2)),)),
        ):
            with self.subTest(component=component):
                exact_parts = _normalized_vertex_sign_expressions(
                    left_basis,
                    word,
                    right_basis,
                )
                expected = np.asarray(
                    [
                        np.asarray(part.subs(substitutions), dtype=np.complex128)
                        for part in exact_parts
                    ],
                    dtype=np.complex128,
                )
                generated = evaluate_sparse_ramond_vertex_parts(
                    component,
                    10,
                    0,
                    float(p_left),
                    complex(h_ns),
                    float(p_right),
                )
                self.assertTrue(
                    np.allclose(
                        generated,
                        expected,
                        rtol=2.0e-15,
                        atol=2.0e-14,
                    )
                )

    def test_compiled_ns_vertices_match_direct_oracle_through_level_two(self) -> None:
        options = dict(
            c=self.c,
            internal_weights=[self.h, self.h + 0.17],
            external_weights=[self.external_h, self.external_h + 0.11],
            maximum_twice_levels=4,
            edge_lift_signs=[-1, 1],
            external_words=[(G(sp.Rational(-1, 2)),), ()],
            form_weights=[(0.73, -0.21), (1.17, 0.34)],
        )
        direct = direct_ns_torus_necklace_series(
            **options,
            vertex_backend="direct",
        )
        compiled = direct_ns_torus_necklace_series(
            **options,
            vertex_backend="template",
        )
        self.assertEqual(set(direct.coefficients), set(compiled.coefficients))
        for levels, expected in direct.coefficients.items():
            self.assertTrue(
                np.isclose(
                    compiled.coefficients[levels],
                    expected,
                    rtol=2.0e-12,
                    atol=2.0e-12,
                ),
                msg=f"compiled NS coefficient mismatch at {levels}",
            )

    def test_physical_rrns_constants_scale_the_chiral_branch(self) -> None:
        internal = (0.37, 0.52)
        external = (0.41, 0.46)
        signs = (1, -1)
        normalized = b1_ramond_liouville_necklace_series(
            internal_momenta=internal,
            external_ns_momenta=external,
            structure_signs=signs,
            maximum_twice_levels=0,
            include_structure_constants=False,
        )
        physical = b1_ramond_liouville_necklace_series(
            internal_momenta=internal,
            external_ns_momenta=external,
            structure_signs=signs,
            maximum_twice_levels=0,
            include_structure_constants=True,
        )
        constants = (
            rr_ns_chiral_structure_constant(
                internal[1], internal[0], external[0], structure_sign=signs[0]
            ),
            rr_ns_chiral_structure_constant(
                internal[0], internal[1], external[1], structure_sign=signs[1]
            ),
        )
        self.assertAlmostEqual(
            physical.leading_coefficient,
            normalized.leading_coefficient * constants[0] * constants[1],
        )

    def test_cyclic_relabeling_and_one_edge_degeneration(self) -> None:
        original = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[1.21, 1.43, 1.68],
            external_weights=[0.31, 0.46, 0.57],
            maximum_twice_levels=1,
            form_weights=[(1.0, 0.7), (1.0, -0.4), (1.0, 1.2)],
        )
        rotated = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[1.43, 1.68, 1.21],
            external_weights=[0.46, 0.57, 0.31],
            maximum_twice_levels=1,
            form_weights=[(1.0, -0.4), (1.0, 1.2), (1.0, 0.7)],
        )
        for levels, coefficient in original.coefficients.items():
            self.assertAlmostEqual(
                coefficient,
                rotated.coefficients[(levels[1], levels[2], levels[0])],
            )

        q_values = (0.0, 0.08, 0.12)
        expected = sum(
            coefficient * q_values[1] ** (levels[1] / 2) * q_values[2] ** (levels[2] / 2)
            for levels, coefficient in original.coefficients.items()
            if levels[0] == 0
        )
        self.assertAlmostEqual(original.descendant_value(q_values), expected)

    def test_odd_spin_pco_descendants_enter_the_necklace_tensor(self) -> None:
        prescription = genus_one_ns_pco_prescription(
            2,
            TorusSpinStructure(True, True),
        )
        series = direct_ns_torus_necklace_series(
            c=self.c,
            internal_weights=[1.2, 1.4],
            external_weights=[0.4, 0.6],
            maximum_twice_levels=0,
            external_words=prescription.external_words,
        )
        self.assertTrue(math.isfinite(abs(series.leading_coefficient)))


class FreeFieldDegenerationTests(unittest.TestCase):
    """Check local OPE, cusp, theta, and fermion identities."""

    tau = 0.17 + 1.21j

    def test_prime_form_and_szego_local_limits(self) -> None:
        separation = 1.0e-7 * (1 + 0.3j)
        self.assertAlmostEqual(
            prime_form(separation, 0, self.tau) / separation,
            1.0,
            places=11,
        )
        for spin in torus_spin_structures():
            with self.subTest(spin=spin.label):
                self.assertAlmostEqual(
                    separation * szego_kernel(separation, self.tau, spin),
                    1.0,
                    places=10,
                )

    def test_scalar_green_has_logarithmic_ope_singularity(self) -> None:
        first = 2.0e-5
        second = 1.0e-5
        difference = (
            torus_scalar_green(second, 0, self.tau)
            - torus_scalar_green(first, 0, self.tau)
        )
        self.assertAlmostEqual(difference, 2 * math.log(2), places=7)

    def test_odd_nonzero_mode_kernel_is_doubly_periodic(self) -> None:
        z = 0.17 + 0.13j
        value = odd_szego_nonzero_mode(z, self.tau, precision=55)
        self.assertAlmostEqual(
            odd_szego_nonzero_mode(z + 1, self.tau, precision=55),
            value,
            places=12,
        )
        self.assertAlmostEqual(
            odd_szego_nonzero_mode(z + self.tau, self.tau, precision=55),
            value,
            places=12,
        )

    def test_jacobi_identity_and_eta_cusp(self) -> None:
        theta2 = theta_characteristic(0.5, 0.0, 0, self.tau)
        theta3 = theta_characteristic(0.0, 0.0, 0, self.tau)
        theta4 = theta_characteristic(0.0, 0.5, 0, self.tau)
        self.assertAlmostEqual(theta3**4, theta2**4 + theta4**4, places=12)

        cusp_tau = 5j
        leading = np.exp(1j * math.pi * cusp_tau / 12)
        self.assertLess(abs(dedekind_eta(cusp_tau) / leading - 1), 1.0e-12)

    def test_pfaffian_and_majorana_zero_mode_rules(self) -> None:
        matrix = np.asarray(
            [
                [0, 1 + 0.2j, 0.3, -0.4j],
                [-1 - 0.2j, 0, 0.7, 0.2],
                [-0.3, -0.7, 0, 1.1],
                [0.4j, -0.2, -1.1, 0],
            ],
            dtype=np.complex128,
        )
        self.assertAlmostEqual(pfaffian(matrix) ** 2, np.linalg.det(matrix))

        even = TorusSpinStructure(False, False)
        odd = TorusSpinStructure(True, True)
        points = (0.11 + 0.03j, 0.37 + 0.09j)
        self.assertAlmostEqual(
            majorana_wick_factor(points, self.tau, even),
            szego_kernel(points[0] - points[1], self.tau, even),
        )
        self.assertEqual(majorana_wick_factor((), self.tau, odd), 0.0j)
        self.assertAlmostEqual(majorana_wick_factor((0.2,), self.tau, odd), 1.0)
        self.assertEqual(majorana_wick_factor(points, self.tau, odd), 0.0j)

        same_flavor = flavored_majorana_wick_factor(
            points,
            (4, 4),
            self.tau,
            even,
            n_fermions=23,
            include_even_partition=False,
        )
        different_flavors = flavored_majorana_wick_factor(
            points,
            (4, 7),
            self.tau,
            even,
            n_fermions=23,
            include_even_partition=False,
        )
        self.assertAlmostEqual(
            same_flavor,
            szego_kernel(points[0] - points[1], self.tau, even),
        )
        self.assertEqual(different_flavors, 0.0j)
        self.assertEqual(
            flavored_majorana_wick_factor(
                (0.2,),
                (3,),
                self.tau,
                odd,
                n_fermions=23,
            ),
            0.0j,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
