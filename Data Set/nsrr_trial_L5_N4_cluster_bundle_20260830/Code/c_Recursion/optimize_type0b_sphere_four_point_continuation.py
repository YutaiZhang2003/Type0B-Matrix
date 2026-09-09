#!/usr/bin/env python3
"""Search, certify, and optionally integrate an optimized four-point chamber."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_type0b_sphere_four_point_wall_one_to_matrix_model import (
    bry_worldsheet_coefficient,
    matrix_model_coefficient,
)
from type0b_sphere_four_point_continuation import (
    centered_rectangles,
    search_four_point_continuation_rectangles,
)
from type0b_sphere_four_point_hybrid import (
    Type0BSphereFourPointHybrid,
    WALL_ONE_RAY_RECTANGLE,
    audit_four_point_crossing,
    integrate_subtraction_free_four_point,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "type0b_sphere_four_point_continuation_optimization.json"
)


def _float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-first", type=_float_list, default=(0.1, 0.12, 0.15, 0.2, 0.25, 0.3))
    parser.add_argument("--ray-second", type=_float_list, default=(0.9, 0.95, 1.0))
    parser.add_argument("--x-centers", type=_float_list, default=(0.25, 0.27, 0.29))
    parser.add_argument("--t-centers", type=_float_list, default=(0.604, 0.620))
    parser.add_argument("--x-half-width", type=float, default=0.012)
    parser.add_argument("--t-half-width", type=float, default=0.008)
    parser.add_argument("--minimum-margin", type=float, default=0.02)
    parser.add_argument("--minimum-wall-clearance", type=float, default=0.01)
    parser.add_argument("--supported-product-pole-order", type=int, default=2)
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--integrate-best", action="store_true")
    parser.add_argument("--twice-level", type=int, default=8)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-precision", type=int, default=45)
    parser.add_argument("--sobol-power", type=int, default=9)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--corner-radius", type=float, default=0.15)
    parser.add_argument("--crossing-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--relative-standard-error-tolerance", type=float, default=0.15)
    return parser


def _default_rectangles(args: argparse.Namespace):
    local = list(
        centered_rectangles(
            args.x_centers,
            args.t_centers,
            x_half_width=args.x_half_width,
            t_half_width=args.t_half_width,
        )
    )
    # Include a larger high-volume candidate and the already audited wall-one
    # rectangle.  The quality metric decides whether their extra area offsets
    # their smaller margin and larger log-radial phase.
    local.extend(
        [
            ((0.241, 0.301), (0.598, 0.626)),
            WALL_ONE_RAY_RECTANGLE,
        ]
    )
    return tuple(local)


def _integrate_best(best, args: argparse.Namespace) -> dict[str, object]:
    if not best.production_ready:
        raise RuntimeError("the best continuation chamber is not production ready")
    if best.crossed_walls != (1,):
        raise RuntimeError(
            "the optimized driver currently uses the 30-node wall-one momentum rule"
        )
    kernel = Type0BSphereFourPointHybrid(
        outgoing_energies=best.center_outgoing_energies,
        contour_prescription="continued",
        block_backend="hybrid",
        hybrid_corner_radius=args.corner_radius,
        recursion_max_twice_level=args.twice_level,
        momentum_order=30,
        momentum_maximum=3.0,
        momentum_rule="wall-one-30",
        structure_precision=args.structure_precision,
        block_working_precision=args.block_precision,
    )
    crossing = audit_four_point_crossing(
        kernel,
        0.37 + 0.28j,
        frames=(0, 1),
        block_region="auto",
        relative_tolerance=args.crossing_tolerance,
    )
    payload: dict[str, object] = {
        "settings": {
            "recursion_max_twice_level": args.twice_level,
            "momentum_order": 30,
            "momentum_maximum": 3.0,
            "momentum_rule": "wall-one-30",
            "structure_precision": args.structure_precision,
            "block_working_precision": args.block_precision,
            "sobol_power": args.sobol_power,
            "samples_per_replicate": 2**args.sobol_power,
            "replicates": args.replicates,
            "seed": args.seed,
            "hybrid_corner_radius": args.corner_radius,
            "crossing_tolerance": args.crossing_tolerance,
            "relative_standard_error_tolerance": (
                args.relative_standard_error_tolerance
            ),
        },
        "center_convergence_audit": kernel.audit.to_json(),
        "crossing_audit": crossing.to_json(),
        "status": "crossing-passed" if crossing.passed else "blocked-by-crossing",
    }
    if not crossing.passed:
        return payload
    amplitude = integrate_subtraction_free_four_point(
        kernel,
        sobol_power=args.sobol_power,
        replicates=args.replicates,
        seed=args.seed,
    )
    error_norm = (
        amplitude.standard_error_real**2 + amplitude.standard_error_imag**2
    ) ** 0.5
    relative_standard_error = error_norm / max(abs(amplitude.mean), 1.0e-300)
    worldsheet = bry_worldsheet_coefficient(amplitude.mean)
    matrix = matrix_model_coefficient(
        amplitude.incoming_energy,
        amplitude.outgoing_energies,
    )
    residual = worldsheet - matrix
    worldsheet_error_real = 8.0 * amplitude.standard_error_imag / math.pi
    worldsheet_error_imag = 8.0 * amplitude.standard_error_real / math.pi
    precision_passed = (
        relative_standard_error <= args.relative_standard_error_tolerance
    )
    payload.update(
        {
            "status": "integrated" if precision_passed else "integrated-low-precision",
            "amplitude": amplitude.to_json(),
            "relative_standard_error": relative_standard_error,
            "relative_standard_error_tolerance": args.relative_standard_error_tolerance,
            "precision_passed": precision_passed,
            "bry_normalized_worldsheet_coefficient": {
                "real": worldsheet.real,
                "imag": worldsheet.imag,
            },
            "worldsheet_standard_error_real": worldsheet_error_real,
            "worldsheet_standard_error_imag": worldsheet_error_imag,
            "matrix_model_coefficient": {
                "real": matrix.real,
                "imag": matrix.imag,
            },
            "worldsheet_minus_matrix_model": {
                "real": residual.real,
                "imag": residual.imag,
            },
            "relative_complex_discrepancy": abs(residual) / max(abs(matrix), 1.0e-300),
        }
    )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    rays = tuple(
        (first, second, 1.0)
        for first in args.ray_first
        for second in args.ray_second
    )
    search = search_four_point_continuation_rectangles(
        ray_candidates=rays,
        rectangles=_default_rectangles(args),
        ray_real_sign=1,
        required_minimum_margin=args.minimum_margin,
        required_wall_clearance=args.minimum_wall_clearance,
        supported_product_pole_order=args.supported_product_pole_order,
        keep=args.keep,
    )
    best = search.best
    payload: dict[str, object] = {
        "description": __doc__,
        "continuation_sheet": "positive real BRY sheet",
        "search": search.to_json(include_exponent_bounds=False),
        "settings": {
            "ray_first": list(args.ray_first),
            "ray_second": list(args.ray_second),
            "x_centers": list(args.x_centers),
            "t_centers": list(args.t_centers),
            "x_half_width": args.x_half_width,
            "t_half_width": args.t_half_width,
            "minimum_margin": args.minimum_margin,
            "minimum_wall_clearance": args.minimum_wall_clearance,
            "supported_product_pole_order": args.supported_product_pole_order,
        },
    }
    if best is not None:
        payload["best_full_certificate"] = best.to_json(
            include_exponent_bounds=True
        )
        print(
            "best: "
            f"ray={best.ray_coefficients}, "
            f"x={best.x_interval}, t={best.t_interval}, "
            f"walls={best.crossed_walls}, "
            f"margin>={best.minimum_margin_lower_bound:.6g}, "
            f"phase/margin<={best.maximum_phase_to_margin_upper_bound:.6g}, "
            f"production_ready={best.production_ready}",
            flush=True,
        )
        if args.integrate_best:
            payload["best_center_integration"] = _integrate_best(best, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    if best is None or not best.production_ready:
        raise SystemExit("the search found no production-ready continuation chamber")


if __name__ == "__main__":
    main()
