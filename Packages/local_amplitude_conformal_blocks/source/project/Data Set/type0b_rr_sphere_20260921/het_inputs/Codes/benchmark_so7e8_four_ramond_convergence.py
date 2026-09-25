#!/usr/bin/env python3
"""Convergence driver for the full SO(7) four-Ramond Liouville integral.

The benchmark stays in the strict generic-complex-energy chamber.  It does
not use resonant or linear-dilaton kinematics.  Each row varies one of the
three independent numerical cutoffs: the super-Liouville block order, the
real-P quadrature/tail, or the sphere-modulus quadrature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "Codes"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from so7e8_four_ramond_liouville_integral import (  # noqa: E402
    evaluate_four_ramond_liouville_convergent,
)


MOMENTA = (0.02 + 0.22j, 0.03 + 0.23j, 0.04 + 0.24j, 0.09 + 0.69j)
FAMILIES = ("Psi_tilde", "Psi_tilde", "Psi_tilde", "Psi")

MODULI_GRIDS = {
    "coarse": dict(
        theta_orders=(4, 4, 8),
        radial_order=4,
        disk_radial_order=4,
        disk_angular_order=12,
        lens_radial_order=4,
        lens_angular_order=12,
    ),
    "medium": dict(
        theta_orders=(6, 6, 12),
        radial_order=6,
        disk_radial_order=6,
        disk_angular_order=18,
        lens_radial_order=6,
        lens_angular_order=18,
    ),
    "fine": dict(
        theta_orders=(8, 8, 16),
        radial_order=8,
        disk_radial_order=8,
        disk_angular_order=24,
        lens_radial_order=8,
        lens_angular_order=24,
    ),
    "ultra": dict(
        theta_orders=(12, 12, 24),
        radial_order=12,
        disk_radial_order=12,
        disk_angular_order=36,
        lens_radial_order=12,
        lens_angular_order=36,
    ),
    "ultra_p5": dict(
        theta_orders=(12, 12, 24),
        radial_order=12,
        disk_radial_order=12,
        disk_angular_order=36,
        disk_radial_power=5.0,
        lens_radial_order=12,
        lens_angular_order=36,
        lens_radial_power=5.0,
    ),
}


def _complex_pairs(values: tuple[complex, ...]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def run_case(
    *,
    order: int,
    p_nodes: int | str | tuple[int, ...],
    p_max: float,
    grid: str,
    momentum_scheme: str = "cutoff",
    epsilon0: float = 0.12,
    epsilon1: float = 0.10,
    lens_channel: str = "same",
) -> dict:
    started = time.perf_counter()
    result = evaluate_four_ramond_liouville_convergent(
        MOMENTA,
        families=FAMILIES,
        maximum_twice_level=order,
        p_nodes=p_nodes,
        p_max=p_max,
        momentum_scheme=momentum_scheme,
        epsilon0=epsilon0,
        epsilon1=epsilon1,
        digits=max(50, 4 * order + 40),
        adjacent_level=False,
        spin7_order_diagnostic=False,
        sld_block_backend="elliptic_recursion",
        spin7_backend="kz_ode",
        lens_channel=lens_channel,
        **MODULI_GRIDS[grid],
    )
    return {
        "momenta": _complex_pairs(MOMENTA),
        "families": list(FAMILIES),
        "production_certified": result.production_certified,
        "sld_block_backend": result.sld_block_backend,
        "spin7_backend": result.spin7_backend,
        "lens_channel": result.lens_channel,
        "sewing_basis": result.sewing_basis,
        "sewing_cocycle_status": result.sewing_cocycle_status,
        "order": order,
        "requested_bulk_p_nodes": p_nodes,
        "actual_p_nodes": result.momentum_nodes,
        "p_max": p_max,
        "momentum_scheme": momentum_scheme,
        "epsilon0": epsilon0,
        "epsilon1": epsilon1,
        "grid": grid,
        "seconds": time.perf_counter() - started,
        "coefficients": _complex_pairs(result.coefficients),
        "pieces": {
            name: _complex_pairs(values) for name, values in result.pieces.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--p-nodes", type=int, default=6)
    parser.add_argument("--p-max", type=float, default=3.0)
    parser.add_argument(
        "--momentum-scheme",
        choices=("cutoff", "infinite_gauss", "infinite_composite"),
        default="cutoff",
    )
    parser.add_argument(
        "--segmented-lite",
        action="store_true",
        help="use a 50-node seven-interval cutoff rule instead of --p-nodes",
    )
    parser.add_argument("--grid", choices=tuple(MODULI_GRIDS), default="coarse")
    parser.add_argument("--epsilon0", type=float, default=0.12)
    parser.add_argument("--epsilon1", type=float, default=0.10)
    parser.add_argument("--lens-channel", choices=("same","reordered"), default="same")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requested_nodes: int | str | tuple[int, ...] = (
        (6, 8, 12, 8, 6, 5, 7)
        if args.segmented_lite
        else (
            "segmented"
            if args.momentum_scheme == "infinite_composite"
            else args.p_nodes
        )
    )
    serialized = json.dumps(
        run_case(
            order=args.order,
            p_nodes=requested_nodes,
            p_max=args.p_max,
            grid=args.grid,
            momentum_scheme=args.momentum_scheme,
            epsilon0=args.epsilon0,
            epsilon1=args.epsilon1,
            lens_channel=args.lens_channel,
        ),
        indent=2,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
