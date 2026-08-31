#!/usr/bin/env python3
"""Evaluate ten precision- and crossing-gated wall-one amplitudes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from type0b_sphere_four_point_hybrid import (
    Type0BSphereFourPointHybrid,
    WALL_ONE_RAY_COEFFICIENTS,
    WALL_ONE_RAY_RECTANGLE,
    audit_four_point_crossing,
    certify_residue_convergent_ray_rectangle,
    integrate_subtraction_free_four_point,
)


SCAN_POINTS = (
    (0.246, 0.604),
    (0.258, 0.604),
    (0.270, 0.604),
    (0.282, 0.604),
    (0.294, 0.604),
    (0.246, 0.620),
    (0.258, 0.620),
    (0.270, 0.620),
    (0.282, 0.620),
    (0.294, 0.620),
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "type0b_sphere_four_point_wall_one_ten_point_scan_positive_sheet_m30.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twice-level", type=int, default=8)
    parser.add_argument("--momentum-order", type=int, default=30)
    parser.add_argument("--momentum-maximum", type=float, default=3.0)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-precision", type=int, default=45)
    parser.add_argument("--sobol-power", type=int, default=9)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--corner-radius", type=float, default=0.15)
    parser.add_argument("--crossing-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--relative-standard-error-tolerance", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _compute_point(
    index: int,
    x_value: float,
    t_value: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    # BRY continue the amplitude from the positive real energy axis.  The
    # former ``-x+i*t`` scan starts from negative real Liouville momenta and
    # therefore lies on a different reflected continuation sheet.
    ray = complex(x_value, t_value)
    outgoing = tuple(
        coefficient * ray for coefficient in WALL_ONE_RAY_COEFFICIENTS
    )
    kernel = Type0BSphereFourPointHybrid(
        outgoing_energies=outgoing,
        contour_prescription="continued",
        block_backend="hybrid",
        hybrid_corner_radius=settings["corner_radius"],
        recursion_max_twice_level=settings["twice_level"],
        momentum_order=settings["momentum_order"],
        momentum_maximum=settings["momentum_maximum"],
        momentum_rule="wall-one-30",
        structure_precision=settings["structure_precision"],
        block_working_precision=settings["block_precision"],
    )
    crossing = audit_four_point_crossing(
        kernel,
        0.37 + 0.28j,
        frames=(0, 1),
        # Gate the same hybrid object used by the moduli integral.  Forcing
        # both frames to c-recursion would not certify the production route.
        block_region="auto",
        relative_tolerance=settings["crossing_tolerance"],
    )
    payload: dict[str, Any] = {
        "index": index,
        "x": x_value,
        "t": t_value,
        "outgoing_energies": [
            {"real": value.real, "imag": value.imag} for value in outgoing
        ],
        "convergence_audit": kernel.audit.to_json(),
        "crossing_audit": crossing.to_json(),
        "status": "crossing-passed" if crossing.passed else "blocked-by-crossing",
    }
    if not crossing.passed:
        return payload
    result = integrate_subtraction_free_four_point(
        kernel,
        sobol_power=settings["sobol_power"],
        replicates=settings["replicates"],
        seed=settings["seed"] + 1000 * index,
    )
    error_norm = (
        result.standard_error_real**2 + result.standard_error_imag**2
    ) ** 0.5
    relative_standard_error = error_norm / max(abs(result.mean), 1.0e-300)
    payload["relative_standard_error"] = relative_standard_error
    payload["precision_tolerance"] = settings["relative_standard_error_tolerance"]
    payload["status"] = (
        "integrated"
        if relative_standard_error <= settings["relative_standard_error_tolerance"]
        else "integrated-low-precision"
    )
    payload["amplitude"] = result.to_json()
    return payload


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    settings = {
        "twice_level": args.twice_level,
        "momentum_order": args.momentum_order,
        "momentum_maximum": args.momentum_maximum,
        "momentum_rule": "wall-one-30",
        "structure_precision": args.structure_precision,
        "block_precision": args.block_precision,
        "sobol_power": args.sobol_power,
        "replicates": args.replicates,
        "corner_radius": args.corner_radius,
        "crossing_tolerance": args.crossing_tolerance,
        "relative_standard_error_tolerance": args.relative_standard_error_tolerance,
        "seed": args.seed,
    }
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_compute_point, index, x_value, t_value, settings): index
            for index, (x_value, t_value) in enumerate(SCAN_POINTS)
        }
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"point {result['index'] + 1}/10: "
                f"x={result['x']:.3f}, t={result['t']:.3f}, "
                f"status={result['status']}",
                flush=True,
            )
    completed.sort(key=lambda value: value["index"])
    payload = {
        "description": __doc__,
        "ray_coefficients": list(WALL_ONE_RAY_COEFFICIENTS),
        "ray_base": "+x+i*t",
        "continuation_origin": "positive real energy axis",
        "certified_rectangle": {
            "x": list(WALL_ONE_RAY_RECTANGLE[0]),
            "t": list(WALL_ONE_RAY_RECTANGLE[1]),
        },
        "domain_certificate": certify_residue_convergent_ray_rectangle(
            WALL_ONE_RAY_RECTANGLE[0],
            WALL_ONE_RAY_RECTANGLE[1],
            ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
            ray_real_sign=1,
        ).to_json(),
        "settings": settings,
        "points": completed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    if any(point["status"] != "integrated" for point in completed):
        raise SystemExit(
            "at least one point failed the crossing or target-blind precision gate"
        )


if __name__ == "__main__":
    main()
