#!/usr/bin/env python3
"""Recompute one asymmetric four-vector heterotic 1->3 amplitude point.

The default is the report's representative continuation
    omega0 = 1/3 + 0.6 i,
    (omega1,omega2,omega3) = (0.2,0.3,0.5) omega0.
Use --quick for a lower-resolution diagnostic calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heterotic_so23_1to3_fast import (
    pair_channel_ansatz,
    vector_amplitude_coefficients_regularized,
)


def pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=float, default=1 / 3, help="real part of omega0")
    parser.add_argument("--a", type=float, default=0.6, help="imaginary part of omega0")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.2, 0.3, 0.5),
        metavar=("R1", "R2", "R3"),
        help="real outgoing fractions; they must sum to one",
    )
    parser.add_argument("--q-order", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="use a cheaper diagnostic quadrature")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    ratios = np.asarray(args.ratios, dtype=float)
    if not np.isclose(ratios.sum(), 1.0, atol=1e-12):
        parser.error("--ratios must sum to one")

    omega0 = complex(args.real, args.a)
    energies = [complex(r * omega0) for r in ratios] + [omega0]

    if args.quick:
        settings = dict(
            q_order=min(args.q_order, 4),
            p_nodes=44,
            p_max=4.0,
            epsilon0=0.08,
            epsilon1=0.06,
            theta_orders=(16, 16, 36),
            radial_order=16,
            disk_total_order=12,
            lens_radial_order=14,
            lens_angular_order=36,
            lens_power=3.0,
        )
    else:
        settings = dict(
            q_order=args.q_order,
            p_nodes="segmented",
            p_max=4.0,
            epsilon0=0.08,
            epsilon1=0.06,
            theta_orders=(24, 24, 60),
            radial_order=24,
            disk_total_order=17 if args.q_order >= 6 else 14,
            lens_radial_order=20,
            lens_angular_order=64 if args.q_order >= 6 else 56,
            lens_power=3.0,
        )

    numerical = vector_amplitude_coefficients_regularized(energies, **settings)
    prediction = pair_channel_ansatz(energies)
    relative = np.abs((numerical - prediction) / prediction)

    result = {
        "energy_ordering": "[omega1, omega2, omega3, omega0]",
        "energies": [pair(z) for z in energies],
        "tensor_basis": ["delta03 delta12", "delta02 delta13", "delta01 delta23"],
        "settings": settings,
        "numerical": [pair(z) for z in numerical],
        "conjecture": [pair(z) for z in prediction],
        "relative_errors": [float(x) for x in relative],
        "max_relative_error": float(relative.max()),
    }
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
