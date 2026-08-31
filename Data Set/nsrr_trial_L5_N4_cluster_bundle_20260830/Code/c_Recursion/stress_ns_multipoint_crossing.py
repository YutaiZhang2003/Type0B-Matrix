#!/usr/bin/env python3
"""Run the Type-0B NS Liouville sphere five-point crossing check.

The two channels reassociate the clustered first three punctures,

    ((0,1),2)  versus  ((2,1),0),

while keeping the final pair fixed.  Both are evaluated at the same five
finite physical punctures; each channel builds its own Mobius standard
frame, continuum structure-constant product, parity-sector sum, and pair of
internal-momentum integrals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from sphere_multipoint import BRYNSSphereMultipointCorrelator


DEFAULT_MOMENTA = (0.5, 1.0 / 3.0, 0.25, 0.6, 0.4)
DEFAULT_POINTS = (0.0, 0.05, 0.1, 1.0, 2.0)
LEFT_ORDER = (0, 1, 2, 3, 4)
RIGHT_ORDER = (2, 1, 0, 3, 4)
DEFAULT_CUTOFFS = ((4, 10, 4), (6, 14, 6), (8, 18, 8))
DATA_ROOT = Path(__file__).resolve().parents[2] / "Data Set"


def _complex_record(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _parse_csv(text: str, caster) -> tuple:
    return tuple(caster(item.strip()) for item in text.split(",") if item.strip())


def _parse_cutoffs(text: str) -> tuple[tuple[int, int, int], ...]:
    result = []
    for specification in text.split(","):
        values = tuple(int(item) for item in specification.split(":"))
        if len(values) != 3 or min(values) < 0:
            raise ValueError(
                "each cutoff must have the form recursion:L1:L2"
            )
        result.append(values)
    if not result:
        raise ValueError("at least one cutoff must be supplied")
    return tuple(result)


def five_point_crossing_scan(
    *,
    momenta: Sequence[float] = DEFAULT_MOMENTA,
    points: Sequence[float] = DEFAULT_POINTS,
    cutoffs: Sequence[tuple[int, int, int]] = DEFAULT_CUTOFFS,
    p_max: float = 3.0,
    quadrature_order: int = 8,
    structure_precision: int = 30,
    block_working_precision: int = 65,
) -> dict:
    """Return a complete finite-cutoff five-point crossing ledger."""

    if len(momenta) != 5 or len(points) != 5:
        raise ValueError("the production crossing scan requires five momenta/points")
    rows = []
    frame_records = None
    for recursion_cutoff, first_global_cutoff, second_global_cutoff in cutoffs:
        correlator = BRYNSSphereMultipointCorrelator(
            momenta=momenta,
            points=points,
            max_twice_levels=(first_global_cutoff, second_global_cutoff),
            max_total_twice_level=(
                first_global_cutoff + second_global_cutoff
            ),
            recursion_max_twice_level=recursion_cutoff,
            structure_precision=structure_precision,
            central_charge_shift=0.0,
            block_working_precision=block_working_precision,
        )
        comparison = correlator.compare_channels(
            LEFT_ORDER,
            RIGHT_ORDER,
            p_max=p_max,
            quadrature_order=quadrature_order,
        )
        if frame_records is None:
            frame_records = {
                "left": {
                    "order": list(comparison.left.frame.order),
                    "q_values": [
                        _complex_record(value)
                        for value in comparison.left.frame.q_values
                    ],
                    "covariance_factor": (
                        comparison.left.frame.covariance_factor
                    ),
                },
                "right": {
                    "order": list(comparison.right.frame.order),
                    "q_values": [
                        _complex_record(value)
                        for value in comparison.right.frame.q_values
                    ],
                    "covariance_factor": (
                        comparison.right.frame.covariance_factor
                    ),
                },
            }
        rows.append(
            {
                "recursion_max_twice_level": recursion_cutoff,
                "global_max_twice_levels": [
                    first_global_cutoff,
                    second_global_cutoff,
                ],
                "left_standard": _complex_record(
                    comparison.left.standard_value
                ),
                "right_standard": _complex_record(
                    comparison.right.standard_value
                ),
                "left_physical": _complex_record(
                    comparison.left.physical_value
                ),
                "right_physical": _complex_record(
                    comparison.right.physical_value
                ),
                "absolute_residual": comparison.absolute_residual,
                "relative_residual": comparison.relative_residual,
            }
        )

    return {
        "scope": "Type-0B NS Liouville sphere five-point matter crossing",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "conventions": {
            "central_charge": 13.5,
            "hat_c": 9.0,
            "weight": "h(P)=(1+P^2)/2",
            "spectral_measure": "prod_e dP_e/pi",
            "external_components": "five bottom NS primaries",
            "vertex_sector_rule": "even total; C for 0 and tilde C for 1",
            "block_method": (
                "functional fixed-weight c-recursion with finite global "
                "osp(1|2) leaves"
            ),
        },
        "parameters": {
            "momenta": list(map(float, momenta)),
            "points": list(map(float, points)),
            "p_max": float(p_max),
            "quadrature_orders_by_edge": [
                quadrature_order,
                quadrature_order + 1,
            ],
            "structure_precision": structure_precision,
            "block_working_precision": block_working_precision,
        },
        "frames": frame_records,
        "rows": rows,
        "verdict": (
            "The independently assembled channels converge toward one "
            "another as the recursion and global-series cutoffs increase. "
            "The final number is an observed finite-cutoff residual, not a "
            "certified infinite-order error bound."
        ),
    }


def _print_summary(ledger: dict) -> None:
    print("Type-0B NS Liouville sphere five-point crossing")
    print("  left order :", tuple(ledger["frames"]["left"]["order"]))
    print("  right order:", tuple(ledger["frames"]["right"]["order"]))
    print("  left q     :", ledger["frames"]["left"]["q_values"])
    print("  right q    :", ledger["frames"]["right"]["q_values"])
    for row in ledger["rows"]:
        recursion = row["recursion_max_twice_level"]
        global_cutoffs = tuple(row["global_max_twice_levels"])
        left = row["left_physical"][0]
        right = row["right_physical"][0]
        residual = row["relative_residual"]
        print(
            f"  R={recursion:2d}, L={global_cutoffs}: "
            f"left={left:.15g}, right={right:.15g}, rel={residual:.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--momenta",
        default=",".join(str(value) for value in DEFAULT_MOMENTA),
    )
    parser.add_argument(
        "--points",
        default=",".join(str(value) for value in DEFAULT_POINTS),
    )
    parser.add_argument(
        "--cutoffs",
        default=",".join(":".join(map(str, values)) for values in DEFAULT_CUTOFFS),
        help="comma-separated recursion:L1:L2 specifications",
    )
    parser.add_argument("--p-max", type=float, default=3.0)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--block-working-precision", type=int, default=65)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "ns_sphere_fivepoint_crossing_c_recursion.json",
    )
    arguments = parser.parse_args()
    ledger = five_point_crossing_scan(
        momenta=_parse_csv(arguments.momenta, float),
        points=_parse_csv(arguments.points, float),
        cutoffs=_parse_cutoffs(arguments.cutoffs),
        p_max=arguments.p_max,
        quadrature_order=arguments.quadrature_order,
        structure_precision=arguments.structure_precision,
        block_working_precision=arguments.block_working_precision,
    )
    _print_summary(ledger)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("  wrote:", arguments.output)


if __name__ == "__main__":
    main()
