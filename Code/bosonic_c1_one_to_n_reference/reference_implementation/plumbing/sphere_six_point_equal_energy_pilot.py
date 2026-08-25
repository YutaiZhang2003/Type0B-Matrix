#!/usr/bin/env python3
"""Analytic-chamber and block-cost pilot for sphere 1->5 scattering.

This is deliberately not yet a full Mbar_0,6 integral.  It freezes the
normalization and first Liouville contour wall, tabulates the analytic target
on omega=i t, and benchmarks the validated six-point block at a representative
Liouville momentum and plumbing point.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

try:
    from ccy_sphere_six_point import (
        evaluate_sphere_six_point_series,
        sphere_six_point_c_coefficients,
        sphere_six_point_h_c25_limit,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_sphere_six_point import (
        evaluate_sphere_six_point_series,
        sphere_six_point_c_coefficients,
        sphere_six_point_h_c25_limit,
    )


FIRST_RESIDUE_WALL = 1.0 / 3.0


def expected_q5_imaginary_ray(t: float) -> float:
    """Return Q5(i t)=(1-5t)(2-5t)(3-5t)."""

    t = float(t)
    return float((1.0 - 5.0 * t) * (2.0 - 5.0 * t) * (3.0 - 5.0 * t))


def expected_mu4_amplitude_imaginary_ray(t: float) -> complex:
    r"""Return mu^4 A_tree(i t)=-5 i t^6 Q5(i t)."""

    t = float(t)
    return complex(0.0, -5.0 * t**6 * expected_q5_imaginary_ray(t))


def expected_i6_imaginary_ray(t: float) -> float:
    r"""Return I6(i t)=-40 pi^3 t^6 Q5(i t)."""

    t = float(t)
    return float(-40.0 * math.pi**3 * t**6 * expected_q5_imaginary_ray(t))


def incoming_outgoing_poles(omega: complex) -> tuple[complex, complex]:
    r"""Nearest DOZZ pair P_+/- = +/- (3 omega-i)."""

    positive = 3.0 * complex(omega) - 1.0j
    return positive, -positive


def liouville_weights_on_imaginary_ray(
    t: float,
    internal_momenta: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the six external and three internal c=25 weights."""

    t = float(t)
    if not 0.0 < t < FIRST_RESIDUE_WALL:
        raise ValueError("the residue-free pilot requires 0<t<1/3")
    if len(internal_momenta) != 3:
        raise ValueError("internal_momenta must contain three entries")
    incoming = 1.0 - (2.5 * t) ** 2
    outgoing = 1.0 - (0.5 * t) ** 2
    external = (incoming,) + (outgoing,) * 5
    internal = tuple(1.0 + float(momentum) ** 2 for momentum in internal_momenta)
    return external, internal


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def block_benchmark(
    *,
    t: float,
    internal_momenta: Sequence[float],
    q_values: Sequence[float],
    orders: Sequence[int],
) -> list[dict[str, object]]:
    """Compare exact c=25 c-recursion with regulated h-recursion."""

    if len(q_values) != 3 or any(not 0.0 < abs(float(q)) < 1.0 for q in q_values):
        raise ValueError("q_values must contain three entries with 0<|q|<1")
    external, internal = liouville_weights_on_imaginary_ray(t, internal_momenta)
    records: list[dict[str, object]] = []
    for order_value in orders:
        order = int(order_value)
        if order < 0:
            raise ValueError("orders must be non-negative")

        start = time.perf_counter()
        c_table = sphere_six_point_c_coefficients(
            central_charge=25.0,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        c_seconds = time.perf_counter() - start
        c_value = evaluate_sphere_six_point_series(*q_values, c_table)

        start = time.perf_counter()
        h_table, h_fit_shifts = sphere_six_point_h_c25_limit(
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        h_seconds = time.perf_counter() - start
        h_value = evaluate_sphere_six_point_series(*q_values, h_table)
        h_fit_shift_value = evaluate_sphere_six_point_series(*q_values, h_fit_shifts)

        records.append(
            {
                "total_block_order": order,
                "coefficient_count": len(c_table),
                "c_recursion_seconds": c_seconds,
                "h_recursion_seconds": h_seconds,
                "c_recursion_block": _complex_record(c_value),
                "regulated_h_recursion_block": _complex_record(h_value),
                "h_minus_c": _complex_record(h_value - c_value),
                "regulated_h_fit_shift_at_q": _complex_record(h_fit_shift_value),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=0.18)
    parser.add_argument("--internal-momenta", nargs=3, type=float, default=(0.35, 0.70, 1.05))
    parser.add_argument("--q", nargs=3, type=float, default=(0.10, 0.12, 0.14))
    parser.add_argument("--orders", nargs="+", type=int, default=(4, 6, 8))
    parser.add_argument(
        "--target-t",
        nargs="+",
        type=float,
        default=(0.14, 0.17, 0.19, 0.199, 0.201, 0.22, 0.26, 0.30),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_six_point_1to5"
        / "block_and_chamber_pilot.json",
    )
    arguments = parser.parse_args()

    if not 0.0 < arguments.t < FIRST_RESIDUE_WALL:
        parser.error("--t must lie in the residue-free chamber 0<t<1/3")
    if any(not 0.0 < t < FIRST_RESIDUE_WALL for t in arguments.target_t):
        parser.error("every --target-t value must lie in 0<t<1/3")

    external, internal = liouville_weights_on_imaginary_ray(
        arguments.t, arguments.internal_momenta
    )
    target_points = []
    for t in arguments.target_t:
        amplitude = expected_mu4_amplitude_imaginary_ray(t)
        target_points.append(
            {
                "t": t,
                "distance_to_first_wall": FIRST_RESIDUE_WALL - t,
                "Q5_it": expected_q5_imaginary_ray(t),
                "mu4_A_tree_it": _complex_record(amplitude),
                "I6_it": expected_i6_imaginary_ray(t),
            }
        )

    payload = {
        "status": "block_validated_full_Mbar_0_6_integral_not_yet_run",
        "kinematics": "one incoming energy 5 omega and five labelled outgoing energies omega, with omega=i t",
        "normalization": {
            "amplitude": "mu^4 A_tree = i I6/(8 pi^3) = i 5 omega^6 Q5",
            "stripped_function": "Q5 = I6/(40 pi^3 omega^6)",
        },
        "analytic_target": {
            "Q5_omega": "(1+5 i omega)(2+5 i omega)(3+5 i omega)",
            "Q5_it": "(1-5 t)(2-5 t)(3-5 t)",
            "first_zero_in_chamber": 0.2,
        },
        "contour_chamber": {
            "nearest_poles": "P_+/- = +/- (3 omega-i)",
            "first_residue_wall": FIRST_RESIDUE_WALL,
            "production_domain": "0<t<1/3; use t<=0.30 for the first scan",
        },
        "representative_block_point": {
            "t": arguments.t,
            "internal_momenta": list(arguments.internal_momenta),
            "q_values": list(arguments.q),
            "external_weights": list(external),
            "internal_weights": list(internal),
        },
        "block_benchmark": block_benchmark(
            t=arguments.t,
            internal_momenta=arguments.internal_momenta,
            q_values=arguments.q,
            orders=arguments.orders,
        ),
        "target_points": target_points,
        "next_gates": [
            "construct and validate a complete Mbar_0,6 channel atlas",
            "precompute the three-momentum Liouville kernel on threshold-adapted panels",
            "run independent momentum-order, block-order, and six-real-dimensional moduli-QMC ladders",
            "hold out the sign change across t=1/5 from any cubic reconstruction",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

