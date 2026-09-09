#!/usr/bin/env python3
"""Certify disjoint plumbing disks and nonempty sewing annuli at genus two.

At a standard trinion use local coordinates ``z``, ``z-1``, and ``1/z`` at
``0``, ``1``, and infinity.  For an edge with plumbing relation
``u_left*u_right=q``, embedded coordinate disks of radii ``r_left,r_right``
give a nonempty sewing collar when ``r_left*r_right>|q|``.  This audit starts
from the balanced radii ``sqrt(|q|)`` and enlarges every incidence by a common
factor ``lambda>1``.  Pairwise disjointness after the enlargement proves both
embedded, mutually disjoint coordinate disks and a positive-width annulus.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


def _sphere_margins(
    radius_zero: float,
    radius_one: float,
    radius_infinity: float,
) -> tuple[float, float, float]:
    """Clearances between the disks at ``0``, ``1``, and infinity."""

    return (
        1.0 - radius_zero - radius_one,
        1.0 / radius_infinity - radius_zero,
        1.0 / radius_infinity - (1.0 + radius_one),
    )


def plumbing_disk_margins(
    channel: str,
    q_values: Sequence[complex],
    inflation: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return all pairwise disk clearances in the marked plumbing graph."""

    if len(q_values) != 3:
        raise ValueError("genus-two plumbing requires three q values")
    if not inflation > 0:
        raise ValueError("disk inflation must be positive")
    q_tuple = tuple(complex(value) for value in q_values)
    if any(not 0.0 < abs(value) < 1.0 for value in q_tuple):
        raise ValueError("plumbing parameters must satisfy 0 < |q_e| < 1")
    radii = tuple(inflation * math.sqrt(abs(value)) for value in q_tuple)
    if channel == "theta":
        sphere = _sphere_margins(radii[0], radii[1], radii[2])
        return sphere, sphere
    if channel == "glasses":
        # On the left/right trinions the handle occupies 0 and infinity,
        # while the separating bridge occupies 1.
        return (
            _sphere_margins(radii[0], radii[2], radii[0]),
            _sphere_margins(radii[1], radii[2], radii[1]),
        )
    raise ValueError(f"unknown genus-two channel {channel!r}")


def maximum_uniform_inflation(
    channel: str,
    q_values: Sequence[complex],
) -> float:
    """Largest common radius multiplier retaining strict disjointness."""

    def minimum_margin(inflation: float) -> float:
        return min(
            value
            for sphere in plumbing_disk_margins(channel, q_values, inflation)
            for value in sphere
        )

    lower = 1.0
    if minimum_margin(lower) <= 0.0:
        return lower
    upper = 2.0
    while minimum_margin(upper) > 0.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if minimum_margin(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def audit(summary_path: Path, *, inflation: float = 1.1) -> dict:
    source = json.loads(summary_path.read_text())
    config = source.get("config", source)
    rows = []
    for point in config["points"]:
        for channel in ("theta", "glasses"):
            q_values = tuple(
                complex(value) for value in point["q_values"][channel]
            )
            balanced = plumbing_disk_margins(channel, q_values, 1.0)
            inflated = plumbing_disk_margins(channel, q_values, inflation)
            maximum = maximum_uniform_inflation(channel, q_values)
            minimum_balanced = min(value for sphere in balanced for value in sphere)
            minimum_inflated = min(value for sphere in inflated for value in sphere)
            rows.append(
                {
                    "point_id": str(point["id"]),
                    "channel": channel,
                    "q_absolute_values": [abs(value) for value in q_values],
                    "balanced_radii": [math.sqrt(abs(value)) for value in q_values],
                    "balanced_minimum_disk_margin": minimum_balanced,
                    "certified_inflation": float(inflation),
                    "inflated_minimum_disk_margin": minimum_inflated,
                    "maximum_uniform_inflation": maximum,
                    "annulus_product_ratio": float(inflation**2),
                    "annulus_modulus_lower_bound": float(
                        math.log(inflation) / math.pi
                    ),
                    "valid": bool(
                        inflation > 1.0
                        and minimum_balanced > 0.0
                        and minimum_inflated > 0.0
                    ),
                }
            )
    if not all(row["valid"] for row in rows):
        raise RuntimeError("one or more plumbing charts has overlapping annuli")
    return {
        "status": "pass",
        "criterion": (
            "pairwise disjoint standard trinion coordinate disks at a common "
            "inflation lambda>1, implying r_left*r_right>|q| and nonempty "
            "sewing annuli"
        ),
        "source": str(summary_path),
        "certified_inflation": float(inflation),
        "annulus_product_ratio": float(inflation**2),
        "annulus_modulus_lower_bound": float(math.log(inflation) / math.pi),
        "minimum_margin_all_points": min(
            row["inflated_minimum_disk_margin"] for row in rows
        ),
        "minimum_maximum_uniform_inflation": min(
            row["maximum_uniform_inflation"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--inflation", type=float, default=1.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.summary, inflation=args.inflation)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
