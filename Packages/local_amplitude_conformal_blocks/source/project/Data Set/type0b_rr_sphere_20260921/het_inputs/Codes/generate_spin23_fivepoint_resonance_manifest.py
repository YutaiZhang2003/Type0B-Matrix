#!/usr/bin/env python3
"""Generate unequal first-resonance partitions for the heterotic 1->4 scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from spin23_sphere_fivepoint_resonance import (
    singlet_resonance_collision_margin,
    sobol_convergent_resonance_partitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=230824)
    parser.add_argument("--first-leg-min", type=float, default=0.12)
    parser.add_argument("--first-leg-max", type=float, default=0.42)
    parser.add_argument("--minimum-margin", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "spin23_scan_manifests/spin23_fivepoint_resonance_s1_convergent_256.csv"
        ),
    )
    args = parser.parse_args()
    points = list(
        sobol_convergent_resonance_partitions(
            args.count,
            seed=args.seed,
            first_leg_interval=(args.first_leg_min, args.first_leg_max),
            minimum_margin=args.minimum_margin,
        )
    )
    # Add deliberately structured controls before the low-discrepancy core.
    controls = [
        (0.5j, 0.5j, 0.5j, 0.5j),
        (0.31j, 0.53j, 0.56j, 0.60j),
        (0.20j, 0.55j, 0.60j, 0.65j),
        (0.40j, 0.52j, 0.53j, 0.55j),
    ]
    control_kinds = (
        "boundary_negative_control",
        "strict_wedge_control",
        "strict_wedge_control",
        "strict_wedge_control",
    )
    rows = controls + points
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("index", "kind", "t1", "t2", "t3", "t4", "collision_margin")
        )
        for index, point in enumerate(rows):
            writer.writerow(
                (
                    index,
                    control_kinds[index] if index < len(controls) else "sobol",
                    *(f"{value.imag:.17g}" for value in point),
                    f"{singlet_resonance_collision_margin([value.imag for value in point]):.17g}",
                )
            )
    print(f"wrote {len(rows)} points to {args.output}")


if __name__ == "__main__":
    main()
