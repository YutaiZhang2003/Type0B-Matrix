#!/usr/bin/env python3
"""Independent checks for the log-DOZZ and cutoff-free momentum backends."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mpmath as mp
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
FIT_DIR = CODE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
for directory in (CODE_DIR, FIT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import heterotic_so23_1to3 as ref  # noqa: E402
import heterotic_so23_1to3_fast as fast  # noqa: E402
from spin23_singlet_amplitudes import _momentum_quadrature  # noqa: E402


def _direct_upsilon_ns(x: complex) -> mp.mpc:
    x = mp.mpc(x)
    u = ref._upsilon_1_barnes_direct(x / 2)
    return ref.gamma_ratio(x / 2) * u * u


def _direct_upsilon_r(x: complex) -> mp.mpc:
    x = mp.mpc(x)
    u = ref._upsilon_1_barnes_direct((x + 1) / 2)
    return u * u


def _direct_structure(ps: tuple[complex, complex, complex], odd: bool) -> mp.mpc:
    momenta = [mp.mpc(value) for value in ps]
    total = sum(momenta)
    denominator = _direct_upsilon_r if odd else _direct_upsilon_ns
    result = (1j if odd else 0.5j) / denominator(1 + 1j * total)
    for momentum in momenta:
        result *= mp.gamma(1 + 1j * momentum) / mp.gamma(1 - 1j * momentum)
        result *= _direct_upsilon_ns(2j * momentum)
        result /= denominator(1 + 1j * (total - 2 * momentum))
    return result


class LogStructureConstantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mp.mp.dps = 70

    def test_upsilon_shift_identity(self) -> None:
        for x in (-2.3 + 0.2j, -0.4 + 0.6j, 0.7 + 0.1j, 3.2 + 0.4j):
            lhs = ref.upsilon_1(x + 1)
            rhs = ref.gamma_ratio(x) * ref.upsilon_1(x)
            self.assertLess(float(abs(lhs - rhs) / abs(rhs)), 1.0e-60)

    def test_log_structure_matches_direct_barnes_product(self) -> None:
        samples = (
            (0.11 + 0.18j, 0.11 + 0.18j, 0.3),
            (0.20 + 0.05j, 0.17 + 0.10j, 2.5),
            (0.11 + 0.23j, 0.11 + 0.23j, 8.0),
        )
        for momenta in samples:
            for function, odd in ((ref.c_even, False), (ref.c_odd, True)):
                shifted_log_value = function(*momenta)
                direct_value = _direct_structure(momenta, odd)
                relative_error = abs(shifted_log_value - direct_value) / abs(direct_value)
                self.assertLess(float(relative_error), 1.0e-60)

    def test_binary64_log_backend_matches_reference(self) -> None:
        for momenta in (
            (0.11 + 0.18j, 0.11 + 0.18j, 0.3),
            (0.20 + 0.05j, 0.17 + 0.10j, 2.5),
            (0.11 + 0.23j, 0.11 + 0.23j, 8.0),
        ):
            for quick, precise in ((fast.c_even, ref.c_even), (fast.c_odd, ref.c_odd)):
                actual = quick(*momenta)
                expected = complex(precise(*momenta))
                self.assertLess(abs(actual - expected) / abs(expected), 1.0e-11)


class InfiniteMomentumQuadratureTests(unittest.TestCase):
    def test_full_half_line_map_and_jacobian(self) -> None:
        momenta, weights = _momentum_quadrature(
            64,
            0.0,
            0.0,
            scheme="infinite_gauss",
            infinite_gauss_scale=1.0,
        )
        self.assertTrue(np.all(np.diff(momenta) > 0))
        self.assertTrue(np.all(weights > 0))
        # With this map exp(-P) cancels the Jacobian to a constant, so the
        # Gaussian rule checks both the infinite map and its weights exactly.
        integral = float(np.dot(weights, np.exp(-momenta)))
        self.assertAlmostEqual(integral, 1.0, places=14)

    def test_scale_parameter(self) -> None:
        scale = 0.5
        momenta, weights = _momentum_quadrature(
            48,
            0.0,
            0.0,
            scheme="infinite_gauss",
            infinite_gauss_scale=scale,
        )
        integral = float(np.dot(weights, np.exp(-momenta / scale)))
        self.assertAlmostEqual(integral, scale, places=14)

    def test_composite_rule_includes_infinite_tail(self) -> None:
        momenta, weights = _momentum_quadrature(
            [20, 20, 20, 20, 20, 20, 24],
            0.0,
            0.0,
            scheme="infinite_composite",
            infinite_gauss_scale=1.0,
        )
        self.assertEqual(len(momenta), 144)
        self.assertTrue(np.all(np.diff(momenta) > 0))
        self.assertGreater(float(momenta[-1]), 8.0)
        integral = float(np.dot(weights, np.exp(-momenta)))
        self.assertAlmostEqual(integral, 1.0, places=14)

    def test_extended_composite_rule_resolves_high_momentum_before_tail(self) -> None:
        counts = [12] * 8
        momenta, weights = _momentum_quadrature(
            counts,
            0.0,
            0.0,
            scheme="infinite_composite_extended",
            infinite_gauss_scale=1.0,
        )
        self.assertEqual(len(momenta), sum(counts))
        self.assertTrue(np.any((momenta > 2.1) & (momenta < 3.2)))
        self.assertGreater(float(momenta[-1]), 7.5)
        self.assertAlmostEqual(float(np.dot(weights, np.exp(-momenta))), 1.0, places=13)


if __name__ == "__main__":
    unittest.main()
