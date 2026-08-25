#!/usr/bin/env python3
"""Checks for the post-freeze torus three-point matrix comparison."""

from __future__ import annotations

import math
import sys

try:
    from compare_genus1_three_point_matrix_after_freeze import (
        matrix_f1_bry_normalization,
        matrix_stripped_genus1_three_point,
    )
except ImportError:  # pragma: no cover
    from plumbing.compare_genus1_three_point_matrix_after_freeze import (
        matrix_f1_bry_normalization,
        matrix_stripped_genus1_three_point,
    )


def require_close(value: complex, expected: complex, tolerance: float, label: str) -> None:
    error = abs(complex(value) - complex(expected))
    print(f"{label}: error={error:.3e}")
    if error > tolerance:
        raise AssertionError(f"{label} failed: {value!r} != {expected!r}")


def run() -> None:
    t = 0.6
    omega = 1.0j * t
    omega_1 = omega_2 = 0.5j * t
    stripped_expected = (t - 1.0) * (t - 2.0) * (1.0 + t - 0.5 * t * t) / 48.0
    f1_expected = -1.0j * t**3 * (t - 1.0) * (t - 2.0) * (
        1.0 + t - 0.5 * t * t
    ) / 96.0
    require_close(
        matrix_stripped_genus1_three_point(omega_1, omega_2),
        stripped_expected,
        1.0e-15,
        "equal-ray stripped polynomial",
    )
    require_close(
        matrix_f1_bry_normalization(omega, omega_1, omega_2),
        f1_expected,
        1.0e-15,
        "equal-ray F1 matrix normalization",
    )
    require_close(
        4.0 * math.pi * f1_expected,
        -0.02248375030321143j,
        1.0e-15,
        "I target",
    )
    print("all genus-one three-point matrix comparison checks passed")


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
