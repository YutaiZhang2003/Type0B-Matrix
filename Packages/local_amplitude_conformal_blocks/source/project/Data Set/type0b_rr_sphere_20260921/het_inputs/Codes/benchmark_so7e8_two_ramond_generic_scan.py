#!/usr/bin/env python3
"""Generic-energy full-Liouville scan for two-Ramond contact reconstruction.

These points lie in the strict real-P-contour and collision-convergence
chamber.  No resonance or linear-dilaton reduction enters the calculation.
The first four momenta extend the existing order-15 data by supplying FSS;
the last two complete a minimally identifiable degree-one crossing design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


CODE_DIR = Path(__file__).resolve().parents[1] / "Codes"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from so7e8_two_ramond_liouville_integral import (  # noqa: E402
    evaluate_two_ramond_liouville_convergent,
)


MOMENTA = {
    "old0": (0.01 + 0.18j, 0.04 + 0.20j, 0.06 + 0.27j, 0.11 + 0.65j),
    "old1": (0.05 + 0.25j, -0.02 + 0.15j, 0.03 + 0.27j, 0.06 + 0.67j),
    "old2": (-0.03 + 0.19j, 0.05 + 0.28j, 0.02 + 0.17j, 0.04 + 0.64j),
    "old3": (0.04 + 0.23j, 0.02 + 0.25j, -0.01 + 0.18j, 0.05 + 0.66j),
    "new3": (
        -0.039221196556038264 + 0.14368643583796603j,
        -0.04737404286906298 + 0.2585133671744862j,
        -0.05813903940434442 + 0.20805362575627073j,
        -0.14473427882944567 + 0.6102534287687229j,
    ),
    "new4": (
        -0.04876846474487012 + 0.14863280473004373j,
        -0.009001229037608006 + 0.1383271011189602j,
        -0.035198614908733106 + 0.14126607983782732j,
        -0.09296830869121123 + 0.42822598568683123j,
    ),
}

SPECIES = {
    "FSS": ("S", "S"),
    "FSV": ("S", "V"),
    "FVS": ("V", "S"),
    "FVV": ("V", "V"),
}

GRIDS = {
    "coarse": dict(
        theta_orders=(8, 8, 16),
        radial_order=8,
        disk_radial_order=10,
        disk_angular_order=32,
        lens_radial_order=10,
        lens_angular_order=32,
    ),
    "fine": dict(
        theta_orders=(12, 12, 24),
        radial_order=12,
        disk_radial_order=14,
        disk_angular_order=44,
        lens_radial_order=14,
        lens_angular_order=44,
    ),
}


def _pairs(values: dict[str, complex]) -> dict[str, list[float]]:
    return {
        name: [float(value.real), float(value.imag)]
        for name, value in values.items()
    }


def run(
    target: str,
    species: str,
    *,
    order: int,
    p_nodes: int,
    grid: str,
    momentum_scheme: str,
) -> dict:
    ns_at_z, ns_at_one = SPECIES[species]
    started = time.perf_counter()
    result = evaluate_two_ramond_liouville_convergent(
        MOMENTA[target],
        ns_at_z=ns_at_z,
        ns_at_one=ns_at_one,
        maximum_twice_level=order,
        p_nodes=p_nodes,
        p_max=4.0,
        p_cut=0.03,
        momentum_scheme=momentum_scheme,
        epsilon0=0.08,
        epsilon1=0.06,
        digits=max(50, 4 * order + 40),
        crossed_block_backend="double_virasoro",
        **GRIDS[grid],
    )
    return {
        "target": target,
        "species": species,
        "momenta": [[value.real, value.imag] for value in MOMENTA[target]],
        "order": order,
        "p_nodes": result.momentum_nodes,
        "p_max": result.p_max,
        "momentum_scheme": momentum_scheme,
        "grid": grid,
        "seconds": time.perf_counter() - started,
        "coefficients": _pairs(dict(result.values.coefficients)),
        "pieces": {
            piece: _pairs(dict(values))
            for piece, values in result.values.pieces.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=tuple(MOMENTA))
    parser.add_argument("species", choices=tuple(SPECIES))
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--p-nodes", type=int, default=16)
    parser.add_argument("--grid", choices=tuple(GRIDS), default="coarse")
    parser.add_argument(
        "--momentum-scheme",
        choices=("cutoff", "infinite_gauss"),
        default="cutoff",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(
        run(
            args.target,
            args.species,
            order=args.order,
            p_nodes=args.p_nodes,
            grid=args.grid,
            momentum_scheme=args.momentum_scheme,
        ),
        indent=2,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
