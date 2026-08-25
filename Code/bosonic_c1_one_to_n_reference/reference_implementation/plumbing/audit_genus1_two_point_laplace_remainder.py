#!/usr/bin/env python3
"""Intrinsic remainder audit for the torus two-point Laplace expansion.

Threshold quadrature is not used to select a truncation or to decide whether
the Laplace result passes.  It is read only after classification to measure
the behavior of the independent error estimate.
"""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path

try:
    from genus1_two_point_cusp_laplace import necklace_laplace_value
    from torus_two_point_blocks import (
        elliptic_nome,
        evaluate_bivariate,
        necklace_coefficients_in_elliptic_nomes,
        necklace_descendant_coefficients,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus1_two_point_cusp_laplace import necklace_laplace_value
    from plumbing.torus_two_point_blocks import (
        elliptic_nome,
        evaluate_bivariate,
        necklace_coefficients_in_elliptic_nomes,
        necklace_descendant_coefficients,
    )


ROOT = Path(__file__).resolve().parent
COMPARISON = (
    ROOT
    / "results/genus1_two_point_worldsheet/threshold_vs_laplace_multi_point_v1/summary.json"
)
OUTPUT = (
    ROOT
    / "results/genus1_two_point_worldsheet/threshold_vs_laplace_multi_point_v1/"
    "intrinsic_remainder_audit.json"
)
TERMINANT_SAFETY_FACTOR = 2.0


def _relative(first: complex, second: complex) -> float:
    return float(abs(second - first) / max(abs(first), abs(second), 1.0e-300))


def _shanks(first: complex, second: complex, third: complex) -> complex:
    denominator = third - 2.0 * second + first
    if abs(denominator) <= 1.0e-300:
        return third
    return third - (third - second) ** 2 / denominator


def _block_omission_envelope(
    *,
    x: float,
    z: complex,
    tau: complex,
    first_decay: float,
    second_decay: float,
) -> float:
    """Probe the fixed-order descendant correction on a Gamma-scaled grid."""

    external_weight = 1.0 - 0.25 * x * x
    q_first = cmath.exp(1.0j * z)
    q_second = cmath.exp(1.0j * (2.0 * math.pi * tau - z))
    hat_q_first = elliptic_nome(q_first)
    hat_q_second = elliptic_nome(q_second)
    maximum = 0.0
    # The threshold Gaussian gives a Gamma(3/2) law in a_i P_i^2.  This grid
    # covers its center and a conservative part of both tails without using
    # any threshold-integral value.
    for first_scaled in (0.0, 1.5, 3.0, 6.0, 10.0):
        for second_scaled in (0.0, 1.5, 3.0, 6.0, 10.0):
            coefficients = necklace_descendant_coefficients(
                25.0,
                1.0 + first_scaled / first_decay,
                1.0 + second_scaled / second_decay,
                external_weight,
                external_weight,
                6,
                3,
            )
            coefficients = necklace_coefficients_in_elliptic_nomes(
                coefficients, 6, 3
            )
            descendant = evaluate_bivariate(
                coefficients, hat_q_first, hat_q_second
            )
            maximum = max(maximum, abs(abs(descendant) ** 2 - 1.0))
    return float(maximum)


def audit_row(row: dict[str, object], *, maximum_degree: int = 14) -> dict[str, object]:
    x = float(row["x"])
    z = complex(float(row["z_real"]), float(row["z_imag"]))
    tau = complex(float(row["tau_real"]), float(row["tau_imag"]))
    raw = [
        necklace_laplace_value(
            x,
            z,
            tau,
            max_x_degree=degree,
            dps=60,
            cauchy_nodes=128,
        )
        for degree in range(maximum_degree + 1)
    ]
    accelerated = [
        _shanks(raw[degree - 2], raw[degree - 1], raw[degree])
        for degree in range(2, maximum_degree + 1)
    ]
    accelerated_steps = [
        _relative(accelerated[index - 1], accelerated[index])
        for index in range(1, len(accelerated))
    ]
    # Ignore the first few pre-asymptotic transforms.  The chosen index is
    # determined entirely by the smallest accelerated terminant.
    candidate_degrees = range(6, maximum_degree + 1)
    selected_degree = min(
        candidate_degrees,
        key=lambda degree: accelerated_steps[degree - 3],
    )
    selected_value = accelerated[selected_degree - 2]
    terminant = accelerated_steps[selected_degree - 3]

    first_decay = float(row["laplace_attempts"][0]["decay_first"])  # type: ignore[index]
    second_decay = float(row["laplace_attempts"][0]["decay_second"])  # type: ignore[index]
    block_envelope = _block_omission_envelope(
        x=x,
        z=z,
        tau=tau,
        first_decay=first_decay,
        second_decay=second_decay,
    )
    stability_value = necklace_laplace_value(
        x,
        z,
        tau,
        max_x_degree=selected_degree,
        dps=60,
        cauchy_nodes=192,
    )
    cauchy_stability = _relative(raw[selected_degree], stability_value)
    # A Shanks-to-Shanks step is a scale estimate, not an alternating-series
    # bound.  Reserve two such steps so a point close to the requested
    # tolerance is not certified by a single fortuitously small transform.
    # This factor is fixed without using the threshold value below.
    intrinsic_error = (
        TERMINANT_SAFETY_FACTOR * terminant
        + block_envelope
        + cauchy_stability
    )
    threshold_value = complex(float(row["threshold_selected_value_real"]), 0.0)
    observed_difference = _relative(selected_value, threshold_value)
    return {
        "name": str(row["name"]),
        "x": x,
        "tau_imag": float(tau.imag),
        "necklace_fraction": float(row["necklace_fraction"]),
        "minimum_decay_coefficient": min(first_decay, second_decay),
        "selected_degree": selected_degree,
        "selected_value_real": float(selected_value.real),
        "selected_value_imag": float(selected_value.imag),
        "accelerated_terminant_estimate": terminant,
        "accelerated_terminant_safety_factor": TERMINANT_SAFETY_FACTOR,
        "block_omission_envelope": block_envelope,
        "cauchy_coefficient_stability": cauchy_stability,
        "intrinsic_relative_error_estimate": intrinsic_error,
        "intrinsic_pass_5e_5": intrinsic_error < 5.0e-5,
        # Audit-only fields below do not enter the decision above.
        "threshold_observed_relative_difference": observed_difference,
        "observed_within_intrinsic_estimate": observed_difference <= intrinsic_error,
    }


def run() -> dict[str, object]:
    comparison = json.loads(COMPARISON.read_text())
    rows = [audit_row(row) for row in comparison["rows"]]
    payload = {
        "calculation": "intrinsic genus-one two-point Laplace remainder audit",
        "selection_uses_threshold": False,
        "maximum_x_degree": 14,
        "accelerated_terminant_safety_factor": TERMINANT_SAFETY_FACTOR,
        "relative_tolerance": 5.0e-5,
        "point_count": len(rows),
        "intrinsic_pass_count": sum(row["intrinsic_pass_5e_5"] for row in rows),
        "observed_bound_count": sum(
            row["observed_within_intrinsic_estimate"] for row in rows
        ),
        "rows": rows,
        "caveat": (
            "The accelerated terminant, even with the stated factor-of-two "
            "reserve, is an asymptotic error estimate rather than a "
            "mathematical inequality. The block envelope is a Gamma-scaled "
            "finite grid rather than a global complex-momentum supremum."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['name']:22s} D={row['selected_degree']:2d} "
            f"estimate={row['intrinsic_relative_error_estimate']:.3e} "
            f"observed={row['threshold_observed_relative_difference']:.3e} "
            f"pass={row['intrinsic_pass_5e_5']}",
            flush=True,
        )
    print(f"wrote {OUTPUT}")
    return payload


if __name__ == "__main__":
    run()
