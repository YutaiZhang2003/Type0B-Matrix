#!/usr/bin/env python3
"""Focused checks of the Spin(23) heterotic three-point amplitudes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mpmath as mp
import sympy as sp


BUNDLE_DIR = Path(__file__).resolve().parent
FOUR_POINT_DIR = BUNDLE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
for path in (BUNDLE_DIR, FOUR_POINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heterotic_so23_1to3 import c_odd  # noqa: E402
from ns_algebra.ns_sca import G  # noqa: E402
from ns_algebra.ns_three_point_tensor import ns_three_point  # noqa: E402
from spin23_three_point_amplitudes import (  # noqa: E402
    ns_weight,
    odd_structure_constant_on_shell,
    s_to_ss_raw,
    s_to_vv_raw,
)


class Spin23ThreePointAmplitudeTests(unittest.TestCase):
    def test_odd_structure_constant_resonance_identity(self) -> None:
        mp.mp.dps = 50
        samples = (
            (mp.mpc("0.37", "0.13"), mp.mpc("0.41", "0.09")),
            (mp.mpc("0.20", "0.17"), mp.mpc("0.33", "0.11")),
        )
        for omega1, omega2 in samples:
            omega0 = omega1 + omega2
            direct = c_odd(omega0, omega1, omega2)
            expected = omega0 * omega1 * omega2
            self.assertLess(abs(direct / expected - 1), mp.mpf("1e-45"))

    def test_global_ns_ward_factors(self) -> None:
        h0, h1, h2, central_charge = sp.symbols("h0 h1 h2 c")
        g_word = (G(sp.Rational(-1, 2)),)

        for words in ((g_word, (), ()), ((), g_word, ()), ((), (), g_word)):
            self.assertEqual(
                ns_three_point(
                    *words,
                    h_infinity=h0,
                    h_middle=h1,
                    h_zero=h2,
                    c=central_charge,
                ),
                1,
            )

        three_descendant_factor = ns_three_point(
            g_word,
            g_word,
            g_word,
            h_infinity=h0,
            h_middle=h1,
            h_zero=h2,
            c=central_charge,
        )
        self.assertEqual(
            sp.simplify(
                three_descendant_factor
                - (h0 + h1 + h2 - sp.Rational(1, 2))
            ),
            0,
        )

    def test_outgoing_exchange_symmetry(self) -> None:
        omega1 = 0.37 + 0.13j
        omega2 = 0.41 + 0.09j
        self.assertAlmostEqual(s_to_vv_raw(omega1, omega2), s_to_vv_raw(omega2, omega1))
        self.assertAlmostEqual(s_to_ss_raw(omega1, omega2), s_to_ss_raw(omega2, omega1))

    def test_sss_ward_factor_is_implemented_explicitly(self) -> None:
        omega1 = 0.2 + 0.17j
        omega2 = 0.33 + 0.11j
        omega0 = omega1 + omega2
        expected = odd_structure_constant_on_shell(omega1, omega2) * (
            ns_weight(omega0) + ns_weight(omega1) + ns_weight(omega2) - 0.5
        )
        self.assertAlmostEqual(s_to_ss_raw(omega1, omega2), expected)


if __name__ == "__main__":
    unittest.main()
