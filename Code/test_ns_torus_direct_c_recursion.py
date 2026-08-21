"""Tests for the functional NS torus c-recursion and modular assembly."""

from __future__ import annotations

import unittest

import mpmath
import numpy as np

from compare_ns_torus_c_h_recursion import (
    TorusCRecursion,
    _central_charge,
    _regular_torus_block,
    _regular_torus_coefficient,
)
from ns_genus12_finite_c_check import (
    NSDescendantThreeForm,
    NumericNSVermaModule,
)
from ns_recursion_recipe import ns_self_loop_scalar_kernel_mp
from stress_ns_torus_modularity_c_recursion import direct_c_modularity_scan


class DirectNSTorusCRecursionTests(unittest.TestCase):
    def test_both_self_loop_sectors_match_independent_pbw_pole_residues(self):
        """Extract odd/even toric residues without using the c-recursion.

        Sector one is represented by a top-component insertion at the middle
        puncture.  A symmetric Laurent extraction, Richardson-improved in the
        detuning, is compared with the scalar self-loop kernel for both the
        first odd and first even NS nulls.  This checks the local ``a=1``
        rule; it does not construct a complete top-component torus block.
        """

        internal_weight = 0.73
        external_weight = 0.62
        top_component = (("G", -1),)

        def direct_coefficient(central_charge_value, twice_level, sector):
            module = NumericNSVermaModule(
                c=central_charge_value,
                weight=internal_weight,
            )
            three_form = NSDescendantThreeForm(
                c=central_charge_value,
                bra_weight=internal_weight,
                middle_weight=external_weight,
                ket_weight=internal_weight,
            )
            basis = module.basis(twice_level)
            inverse_gram = module.numeric_inverse_gram(twice_level)
            middle = top_component if sector else ()
            vertex = np.asarray(
                [
                    [three_form.value(bra, middle, ket) for ket in basis]
                    for bra in basis
                ],
                dtype=np.complex128,
            )
            return np.einsum("ab,ba->", inverse_gram, vertex)

        def symmetric_residue(pole, twice_level, sector, detuning):
            upper = direct_coefficient(
                pole + detuning, twice_level, sector
            )
            lower = direct_coefficient(
                pole - detuning, twice_level, sector
            )
            return detuning * (upper - lower) / 2

        detuning = 3.0e-3
        for r, s in ((3, 1), (2, 2)):
            for sector in (0, 1):
                pole, expected, child_sector = (
                    ns_self_loop_scalar_kernel_mp(
                        r=r,
                        s=s,
                        handle_weight=internal_weight,
                        external_weight=external_weight,
                        sector=sector,
                    )
                )
                pole_value = float(mpmath.re(pole.c))
                coarse = symmetric_residue(
                    pole_value, r * s, sector, detuning
                )
                fine = symmetric_residue(
                    pole_value, r * s, sector, detuning / 2
                )
                extrapolated = (4 * fine - coarse) / 3
                with self.subTest(r=r, s=s, sector=sector):
                    self.assertEqual(child_sector, sector)
                    self.assertLess(abs(extrapolated - expected), 1.0e-10)

    def test_c_recursion_matches_independent_pbw_trace_through_first_poles(self):
        """Compare with a direct finite-c torus descendant trace.

        This oracle uses only the NS commutators, Gram matrices, and Ward
        identities.  In particular it does not import the self-loop residue,
        its incidence ordering, or its odd-null sign from the c-recursion.
        """

        central_charge_value = 14.19870372000744
        internal_weight = 0.73
        external_weight = 0.62
        module = NumericNSVermaModule(
            c=central_charge_value,
            weight=internal_weight,
        )
        three_form = NSDescendantThreeForm(
            c=central_charge_value,
            bra_weight=internal_weight,
            middle_weight=external_weight,
            ket_weight=internal_weight,
        )
        block = TorusCRecursion(
            c=central_charge_value,
            internal_weight=internal_weight,
            external_weight=external_weight,
        )
        coefficients = block.raw_coefficients(8)

        for twice_level in range(9):
            basis = module.basis(twice_level)
            inverse_gram = module.numeric_inverse_gram(twice_level)
            vertex = np.asarray(
                [
                    [three_form.value(bra, (), ket) for ket in basis]
                    for bra in basis
                ],
                dtype=np.complex128,
            )
            direct = np.einsum("ab,ba->", inverse_gram, vertex)
            with self.subTest(twice_level=twice_level):
                self.assertLess(
                    abs(direct - coefficients[twice_level]),
                    5.0e-11,
                )

    def test_hard_modular_point_has_one_non_small_nome(self):
        tau = 0.25j
        s_tau = -1.0 / tau
        q_abs = abs(mpmath.exp(2.0j * mpmath.pi * tau))
        s_q_abs = abs(mpmath.exp(2.0j * mpmath.pi * s_tau))
        self.assertGreater(q_abs, mpmath.mpf("0.2"))
        self.assertLess(s_q_abs, mpmath.mpf("2e-11"))

    def test_exact_regular_leaf_resums_coefficient_formula(self):
        with mpmath.workdps(70):
            q = mpmath.mpf("1e-5")
            h = mpmath.mpf("0.73")
            d = mpmath.mpf("0.29")
            for lift_sign in (1, -1):
                with self.subTest(lift_sign=lift_sign):
                    exact = _regular_torus_block(q, lift_sign, h, d)
                    series = sum(
                        _regular_torus_coefficient(level, h, d)
                        * lift_sign**level
                        * q ** (mpmath.mpf(level) / 2)
                        for level in range(13)
                    )
                    self.assertLess(abs(exact - series), mpmath.mpf("1e-28"))

    def test_functional_recursion_matches_low_q_coefficients(self):
        with mpmath.workdps(70):
            b = mpmath.mpf("1.27")
            block = TorusCRecursion(
                c=_central_charge(b),
                internal_weight=mpmath.mpf("0.73"),
                external_weight=mpmath.mpf("0.29"),
            )
            q = mpmath.mpf("1e-8")
            coefficients = block.raw_coefficients(12)
            for lift_sign in (1, -1):
                with self.subTest(lift_sign=lift_sign):
                    direct = block.recursive_block(q, 4, lift_sign)
                    series = sum(
                        coefficient
                        * lift_sign**level
                        * q ** (mpmath.mpf(level) / 2)
                        for level, coefficient in coefficients.items()
                    )
                    self.assertLess(abs(direct - series), mpmath.mpf("1e-20"))

    def test_vectorized_recursion_matches_scalar_calls(self):
        with mpmath.workdps(70):
            b = mpmath.mpf(1)
            block = TorusCRecursion(
                c=_central_charge(b),
                internal_weight=mpmath.mpf("0.745"),
                external_weight=mpmath.mpf("0.55445"),
            )
            q_values = (
                mpmath.mpc("0.002", "0.001"),
                mpmath.mpc("0.004", "-0.001"),
            )
            signs = (1, -1)
            vector = block.recursive_blocks(q_values, 6, signs)
            scalar = tuple(
                block.recursive_block(q, 6, sign)
                for q, sign in zip(q_values, signs)
            )
            for actual, expected in zip(vector, scalar):
                self.assertLess(abs(actual - expected), mpmath.mpf("1e-65"))

    def test_nonchiral_modular_frames_agree_at_reduced_cost(self):
        result = direct_c_modularity_scan(
            taus=(0.2 + 0.9j,),
            recursion_orders=(6,),
            external_momentum=0.33,
            p_max=4.5,
            quadrature_order=24,
            structure_precision=30,
            working_precision=60,
        )[0]
        self.assertEqual((result.lift_sign, result.s_lift_sign), (1, 1))
        self.assertLess(result.relative_residual, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
