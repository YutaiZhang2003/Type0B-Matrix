#!/usr/bin/env python3
"""Checks for the absolute free-boson long-tube normalization audit."""

from __future__ import annotations

import cmath
import math

try:
    from audit_free_boson_long_tube_normalization import evaluate_long_tube_point
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_free_boson_long_tube_normalization import evaluate_long_tube_point
    from plumbing.free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> None:
    rows = [
        evaluate_long_tube_point(
            0.08,
            0.11,
            q_bridge,
            period_algorithm="collocation",
            collocation_basis_order=60,
            collocation_samples=256,
            max_word_length=10,
            max_mode=100,
        )
        for q_bridge in (1.0e-2, 1.0e-3, 1.0e-4)
    ]

    normalization_errors = [abs(float(row["normalization_constant"]) - 1.0) for row in rows]
    period_errors = [abs(float(row["q_bridge_period_ratio"]) - 1.0) for row in rows]
    print("absolute free-boson long-tube normalization")
    for row in rows:
        print(
            f"  |q_B|={float(row['q_bridge_abs']):.1e}: "
            f"period ratio={float(row['q_bridge_period_ratio']):.12f}, "
            f"constant={float(row['normalization_constant']):.12f}"
        )

    _require(
        all(row["period_method"] == "holomorphic_form_collocation" for row in rows),
        "the normalization audit did not use normalized holomorphic one-forms",
    )
    _require(
        normalization_errors[2] < normalization_errors[1] < normalization_errors[0],
        "the extracted scalar normalization does not converge toward one",
    )
    _require(normalization_errors[-1] < 2.0e-9, "the long-tube scalar normalization is not accurate")
    _require(
        period_errors[2] < period_errors[1] < period_errors[0],
        "the plumbing/Fay coordinate conversion does not converge",
    )

    for row in rows:
        normalization = float(row["normalization_constant"])
        loop_factor = float(row["loop_momentum_gaussian"])
        oscillator_only = float(row["oscillator_only_normalization_constant"])
        mixed_low = float(row["mixed_plumbing_ordinary_volume_vs_canonical_V_over_2pi"])
        mixed_high = float(row["mixed_plumbing_V_over_2pi_vs_canonical_ordinary_volume"])
        _require(
            abs(oscillator_only - normalization / loop_factor) < 2.0e-14,
            "the handle-momentum Gaussian is not separated correctly",
        )
        _require(
            abs(mixed_low * (2.0 * math.pi) - normalization) < 2.0e-14,
            "the ordinary-volume conversion is inconsistent",
        )
        _require(
            abs(mixed_high / (2.0 * math.pi) - normalization) < 2.0e-14,
            "the normalized-volume conversion is inconsistent",
        )

    theta_vacuum = theta_heisenberg_plumbing_partition(0.2, 0.19, 0.18, max_total_level=0)
    glasses_vacuum = glasses_heisenberg_plumbing_partition(0.2, 0.19, 0.18, max_total_level=0)
    _require(theta_vacuum.chiral_value == 1.0, "theta pants sewing has a non-unit vacuum vertex")
    _require(glasses_vacuum.chiral_value == 1.0, "glasses pants sewing has a non-unit vacuum vertex")
    print("  theta and glasses level-zero pants normalizations = 1")

    complex_row = evaluate_long_tube_point(
        0.08 * cmath.exp(0.21j),
        0.11 * cmath.exp(-0.17j),
        1.0e-3 * cmath.exp(0.31j),
        period_algorithm="collocation",
        collocation_basis_order=60,
        collocation_samples=256,
        max_word_length=10,
        max_mode=100,
    )
    complex_error = abs(float(complex_row["normalization_constant"]) - 1.0)
    _require(complex_error < 2.0e-8, "the phase-deformed long-tube normalization failed")
    print(f"  phase-deformed |kappa_X-1| = {complex_error:.3e}")
    print("free-boson long-tube normalization checks passed")


if __name__ == "__main__":
    run()
