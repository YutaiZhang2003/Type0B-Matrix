#!/usr/bin/env python3
"""Tests for the fixed-spin NSRR auxiliary-Majorana quotient."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import sympy as sp

from nsrr_double_virasoro_block import NSRRDoubleVirasoroTheta
from compute_full_block import BranchingGrid
from nsrr_genus2_block import HumanNSRRThetaOracle, star_convolve_series, ZERO_VECTOR
from theta_star_algebra import fwht, star_spectrum


class NSRRDoubleVirasoroBlockTests(unittest.TestCase):
    def test_all_parities_and_lifts_against_independent_pbw(self):
        b = sp.Rational(7, 5)
        q = b + 1/b
        p = (sp.Rational(21, 100), sp.Rational(37, 100), sp.Rational(52, 100))
        for primary_parity in (0, 1):
            block = NSRRDoubleVirasoroTheta(b=float(b), physical_momenta=p,
                                           cutoff=2, primary_parity=primary_parity,
                                           completion="pbw_diagnostic")
            for f in (0, 1):
                for left, right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    oracle = HumanNSRRThetaOracle(
                        central_charge=sp.Rational(3, 2)+3*q*q,
                        h_ns=q*q/8+p[0]*p[0]/2,
                        beta_r1=sp.I*p[1]/sp.sqrt(2), beta_r2=sp.I*p[2]/sp.sqrt(2),
                        form_parity=f, primary_parity=primary_parity, etas=(left, right))
                    components = block.physical_components(f, left, right)
                    for exponent, actual in components.items():
                        expected = oracle.coefficient_components(
                            exponent[0], exponent[1]//2, exponent[2]//2)
                        self.assertLess(max(abs(a-b) for a, b in zip(actual, expected)), 2e-9,
                                        (primary_parity, f, left, right, exponent))
                        for character in range(8):
                            self.assertAlmostEqual(block.physical_series(
                                f, left, right, character)[exponent], fwht(expected)[character],
                                delta=3e-9)
                    sewn = star_convolve_series(block.auxiliary, components,
                                               maximum_total_twice_level=4)
                    enlarged = {}
                    for key, value in block.enlarged_series(f, left, right).items():
                        vector = enlarged.setdefault(key[:3], [0j]*8)
                        vector[key[3]+2*key[4]+4*key[5]] += value
                    for exponent in sewn.keys() | enlarged.keys():
                        self.assertLess(max(abs(a-b) for a, b in zip(
                            sewn.get(exponent, ZERO_VECTOR), enlarged.get(exponent, ZERO_VECTOR))), 4e-9)

    def test_missing_channels_are_not_silently_set_to_zero(self):
        block = NSRRDoubleVirasoroTheta(b=1.4, physical_momenta=(.21, .37, .52), cutoff=1,
                                       completion="pbw_diagnostic")
        self.assertAlmostEqual(block.physical_series(0, 1, -1, 4)[(0, 0, 0)], 2)
        self.assertAlmostEqual(block.star_character_series(0, 1, -1, 4).get((0, 0, 0), 0), 0)
        self.assertAlmostEqual(block.physical_series(0, 1, 1, 4)[(0, 0, 0)], 0)
        self.assertAlmostEqual(block.star_character_series(0, 1, 1, 4)[(0, 0, 0)], 2)
        block.pbw_completion_max_level = 0
        with self.assertRaisesRegex(NotImplementedError, "annihilated"):
            block.physical_components(1, 1, -1)
        with self.assertRaises(ZeroDivisionError):
            block.star_character_series(0, 1, 1, 0)

    def test_physical_liouville_momenta_and_requested_spin_are_finite(self) -> None:
        block = NSRRDoubleVirasoroTheta(
            b=1.4,
            physical_momenta=(0.21, 0.37, 0.52),
            cutoff=1,
            completion="pbw_diagnostic",
        )
        self.assertLess(block.ward_residual_maximum, 1.0e-10)
        for form_parity in (0, 1):
            for eta_left in (1, -1):
                for eta_right in (1, -1):
                    result = block.block(
                        q_values=(0.03 + 0.01j, 0.04 - 0.01j, -0.05j),
                        lifts=(1, 1, -1),
                        form_parity=form_parity,
                        eta_left=eta_left,
                        eta_right=eta_right,
                    )
                    self.assertEqual(result.spin_character, 4)
                    self.assertGreater(abs(result.auxiliary_ground), 0.5)
                    self.assertTrue(abs(result.value) < 100.0)

    def test_only_the_certified_branching_interface_is_called(self):
        original = BranchingGrid.solve
        calls = []

        def canonical_only(grid, a2, a3, **kwargs):
            self.assertEqual(kwargs, {})
            calls.append(grid.momenta)
            return original(grid, a2, a3)

        with patch.object(BranchingGrid, "solve", canonical_only):
            block = NSRRDoubleVirasoroTheta(
                b=1.4, physical_momenta=(.21, .37, .52), cutoff=1)
        self.assertEqual(len(calls), 8)
        self.assertEqual({p[1] for p in calls}, {.37j, -.37j})
        self.assertLess(block.ward_residual_maximum, 1e-10)

    def test_production_never_calls_the_pbw_oracle(self):
        with patch("nsrr_double_virasoro_block.HumanNSRRThetaOracle",
                   side_effect=AssertionError("PBW must not be production input")):
            block = NSRRDoubleVirasoroTheta(
                b=1.4, physical_momenta=(.21, .37, .52), cutoff=1)
            for f in (0, 1):
                for eta in (1, -1):
                    block.physical_components(f, eta, eta)
                with self.assertRaisesRegex(NotImplementedError, "silently use PBW"):
                    block.physical_components(f, 1, -1)


if __name__ == "__main__":
    unittest.main()
