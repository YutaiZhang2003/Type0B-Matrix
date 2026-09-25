#!/usr/bin/env python3
r"""Scan the explicit top-component contact strata at the first resonance.

This is a diagnostic of the fixed-component prescription, not an instruction
to add these values to the meromorphically continued six-point soft result.
The latter can already contain the same local extension.  Moreover, auxiliary
field contractions generate additional collision strata that are absent from
this isolated external-contact sum.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from spin23_fivepoint_resonance_channels import all_nonzero_projections
from spin23_fivepoint_resonance_klt import fit_laurent_series
from spin23_fivepoint_s1_contacts import (
    DEFAULT_CONTACT_TWIST_DIRECTION,
    evaluate_contact_projection,
)


DEFAULT_OUTGOING = (0.31j, 0.53j, 0.56j, 0.60j)
DEFAULT_RADII = (0.024, 0.015, 0.009, 0.0055, 0.0033, 0.002)
ALTERNATE_CONTACT_TWIST_DIRECTION = {
    pair: complex(0.113 * (index + 2), -0.079 * (7 - index))
    for index, pair in enumerate(
        ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    )
}
TWIST_DIRECTIONS = {
    "default": DEFAULT_CONTACT_TWIST_DIRECTION,
    "alternate": ALTERNATE_CONTACT_TWIST_DIRECTION,
}


def _complex_pair(value: complex) -> list[float]:
    normalized = complex(value)
    return [float(normalized.real), float(normalized.imag)]


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) < 3 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("at least three positive radii are required")
    return values


def _parse_complex_tuple(raw: str) -> tuple[complex, ...]:
    values = tuple(complex(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError("four comma-separated momenta are required")
    return values


def _fit_record(radii: Sequence[float], values: Sequence[complex]) -> dict[str, object]:
    degree = min(4, len(radii) - 1)
    comparison_degree = max(1, degree - 1)
    primary = fit_laurent_series(
        radii, values, powers=tuple(range(degree + 1))
    )
    comparison = fit_laurent_series(
        radii, values, powers=tuple(range(comparison_degree + 1))
    )
    return {
        "powers": list(primary.powers),
        "coefficients": [_complex_pair(value) for value in primary.coefficients],
        "limit": _complex_pair(primary.finite_part),
        "residual_norm": primary.residual_norm,
        "condition_number": primary.condition_number,
        "lower_degree_limit": _complex_pair(comparison.finite_part),
        "degree_spread": abs(primary.finite_part - comparison.finite_part),
    }


def run_contact_scan(
    *,
    outgoing_momenta: Sequence[complex] = DEFAULT_OUTGOING,
    radii: Sequence[float] = DEFAULT_RADII,
    directions: Mapping[str, Mapping[tuple[int, int], complex]] = TWIST_DIRECTIONS,
    rule: str = "gauss-jacobi-10",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
) -> dict[str, object]:
    outgoing = tuple(complex(value) for value in outgoing_momenta)
    scan_radii = tuple(float(value) for value in radii)
    if len(outgoing) != 4 or abs(sum(outgoing) - 2j) > 1.0e-10:
        raise ValueError("the outgoing first-resonance momenta must sum to 2i")
    if len(scan_radii) < 3 or any(value <= 0 for value in scan_radii):
        raise ValueError("at least three positive regulator radii are required")

    projections = all_nonzero_projections()
    direction_payload: dict[str, object] = {}
    overall_start = perf_counter()
    for direction_name, direction in directions.items():
        by_label: dict[str, list[complex]] = {
            projection.label: [] for projection in projections
        }
        rows = []
        for radius in scan_radii:
            started = perf_counter()
            values = {}
            maximum_error = 0.0
            all_converged = True
            for projection in projections:
                result = evaluate_contact_projection(
                    projection,
                    outgoing,
                    twist_regulator=radius,
                    twist_direction=direction,
                    rule=rule,
                    rtol=rtol,
                    atol=atol,
                )
                by_label[projection.label].append(result.wall_stripped_value)
                values[projection.label] = {
                    "total": _complex_pair(result.wall_stripped_value),
                    "branches": [
                        {
                            "contact_world_leg": branch.contact_world_leg,
                            "original_charge": _complex_pair(branch.original_charge),
                            "value": _complex_pair(branch.wall_stripped_value),
                            "absolute_error_bound": float(branch.absolute_error_bound),
                        }
                        for branch in result.branches
                    ],
                }
                maximum_error = max(maximum_error, float(result.absolute_error_bound))
                all_converged = all_converged and result.periods_converged
            elapsed = perf_counter() - started
            rows.append(
                {
                    "radius": radius,
                    "elapsed_seconds": elapsed,
                    "all_periods_converged": all_converged,
                    "maximum_absolute_error_bound": maximum_error,
                    "values": values,
                }
            )
            print(
                f"direction={direction_name} radius={radius:.6g} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
        direction_payload[direction_name] = {
            "pair_direction": {
                f"{first}{second}": _complex_pair(value)
                for (first, second), value in direction.items()
            },
            "rows": rows,
            "limit_fits": {
                label: _fit_record(scan_radii, values)
                for label, values in by_label.items()
            },
        }

    direction_names = tuple(direction_payload)
    direction_spreads = {}
    if len(direction_names) >= 2:
        reference = direction_payload[direction_names[0]]["limit_fits"]
        for label in reference:
            limits = [
                complex(*direction_payload[name]["limit_fits"][label]["limit"])
                for name in direction_names
            ]
            direction_spreads[label] = max(
                abs(first - second)
                for index, first in enumerate(limits)
                for second in limits[index + 1 :]
            )

    return {
        "schema": "spin23_fivepoint_s1_external_contact_scan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "external_contact_diagnostic_not_full_resonance",
        "warning": (
            "Do not add these values to the soft-KLT continuation without a "
            "common local matching. Pairwise auxiliary-field collision strata "
            "and vertical/contact completion are not included."
        ),
        "outgoing_momenta": [_complex_pair(value) for value in outgoing],
        "radii": list(scan_radii),
        "quadrature_rule": rule,
        "rtol": rtol,
        "atol": atol,
        "directions": direction_payload,
        "direction_limit_spreads": direction_spreads,
        "total_elapsed_seconds": perf_counter() - overall_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outgoing", type=_parse_complex_tuple, default=DEFAULT_OUTGOING)
    parser.add_argument("--radii", type=_parse_float_tuple, default=DEFAULT_RADII)
    parser.add_argument("--rule", default="gauss-jacobi-10")
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--atol", type=float, default=1.0e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_exports/spin23_fivepoint_s1_external_contacts.json"),
    )
    args = parser.parse_args()
    payload = run_contact_scan(
        outgoing_momenta=args.outgoing,
        radii=args.radii,
        rule=args.rule,
        rtol=args.rtol,
        atol=args.atol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
