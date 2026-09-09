#!/usr/bin/env python3
r"""Evaluate a crossing-gated Type-0B sphere four-point candidate.

The default mode uses the complete analytically continued Liouville contour
in the certified tilted complex-energy chamber. Before integrating over
moduli space it compares two convergent crossing frames at a fixed point and
refuses to produce an amplitude when they disagree beyond the requested
tolerance.

The old equal-pure-imaginary ``t=0.6`` fixed-contour calculation remains
available only as ``--mode fixed-diagnostic``. It omits crossed Liouville
poles and is not a Type-0B ``1->3`` amplitude. Its channel-patched folded
integral additionally requires the explicit ``--allow-crossing-failure``
override.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from type0b_sphere_four_point_hybrid import (
    CONVERGENT_RAY_REFERENCE,
    Type0BSphereFourPointHybrid,
    audit_four_point_crossing,
    certify_convergent_ray_rectangle,
    convergent_ray_energies,
    integrate_folded_unit_disk_qmc,
    integrate_subtraction_free_four_point,
)


RESULTS = Path(__file__).resolve().parent / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("continued-chamber", "fixed-diagnostic"),
        default="continued-chamber",
    )
    parser.add_argument(
        "--base-real", type=float, default=CONVERGENT_RAY_REFERENCE.real
    )
    parser.add_argument(
        "--base-imag", type=float, default=CONVERGENT_RAY_REFERENCE.imag
    )
    parser.add_argument("--t", type=float, default=0.6)
    parser.add_argument("--twice-level", type=int, default=10)
    parser.add_argument("--momentum-order", type=int, default=20)
    parser.add_argument("--momentum-maximum", type=float, default=6.0)
    parser.add_argument("--structure-precision", type=int, default=24)
    parser.add_argument("--block-precision", type=int, default=50)
    parser.add_argument("--sobol-power", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--corner-radius", type=float, default=0.15)
    parser.add_argument("--crossing-real", type=float, default=0.37)
    parser.add_argument("--crossing-imag", type=float, default=0.28)
    parser.add_argument("--crossing-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--crossing-only", action="store_true")
    parser.add_argument(
        "--allow-crossing-failure",
        action="store_true",
        help="diagnostic override; any resulting integral remains uncertified",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path)
    return parser


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    name = (
        "type0b_sphere_four_point_continued_candidate.json"
        if args.mode == "continued-chamber"
        else "type0b_sphere_four_point_fixed_contour_diagnostic.json"
    )
    return RESULTS / name


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "continued-chamber":
        base = complex(args.base_real, args.base_imag)
        outgoing = convergent_ray_energies(base)
        contour = "continued"
        description = (
            "Crossing-gated, complete-contour Type-0B four-point candidate "
            "in the tilted subtraction-free chamber"
        )
    else:
        if args.t <= 0.5:
            raise ValueError("the fixed diagnostic OPE test requires t>1/2")
        outgoing = (1.0j * args.t,) * 3
        contour = "fixed"
        description = (
            "UNCERTIFIED fixed-contour channel-patched diagnostic; crossed "
            "Liouville poles are omitted, so this is not a Type-0B amplitude"
        )

    kernel = Type0BSphereFourPointHybrid(
        outgoing_energies=outgoing,
        contour_prescription=contour,
        block_backend="hybrid",
        hybrid_corner_radius=args.corner_radius,
        recursion_max_twice_level=args.twice_level,
        momentum_order=args.momentum_order,
        momentum_maximum=args.momentum_maximum,
        structure_precision=args.structure_precision,
        block_working_precision=args.block_precision,
    )
    crossing = audit_four_point_crossing(
        kernel,
        complex(args.crossing_real, args.crossing_imag),
        frames=(0, 1),
        block_region="corner",
        relative_tolerance=args.crossing_tolerance,
    )
    output = _output_path(args)
    payload: dict[str, object] = {
        "description": description,
        "certification_status": (
            "crossing-passed" if crossing.passed else "blocked-by-crossing"
        ),
        "mode": args.mode,
        "audit": kernel.audit.to_json(),
        "crossing_audit": crossing.to_json(),
        "settings": {
            "contour_prescription": contour,
            "twice_level": args.twice_level,
            "momentum_order": args.momentum_order,
            "momentum_maximum": args.momentum_maximum,
            "structure_precision": args.structure_precision,
            "block_precision": args.block_precision,
            "sobol_power": args.sobol_power,
            "replicates": args.replicates,
            "corner_radius": args.corner_radius,
            "crossing_backend": "c",
            "crossing_tolerance": args.crossing_tolerance,
            "seed": args.seed,
        },
    }
    if args.mode == "continued-chamber":
        payload["ray_rectangle_certificate"] = (
            certify_convergent_ray_rectangle().to_json()
        )

    if not crossing.passed and not args.allow_crossing_failure:
        _write(output, payload)
        raise SystemExit(
            "crossing preflight failed: relative spread "
            f"{crossing.relative_spread:.6g} exceeds "
            f"{crossing.relative_tolerance:.6g}; wrote audit to {output}"
        )
    if args.crossing_only:
        _write(output, payload)
        print(f"crossing audit wrote {output}")
        return

    if args.mode == "continued-chamber":
        result = integrate_subtraction_free_four_point(
            kernel,
            sobol_power=args.sobol_power,
            replicates=args.replicates,
            seed=args.seed,
        )
    else:
        result = integrate_folded_unit_disk_qmc(
            kernel,
            sobol_power=args.sobol_power,
            replicates=args.replicates,
            seed=args.seed,
        )
    payload["certification_status"] = (
        "crossing-passed-integrated"
        if crossing.passed
        else "uncertified-crossing-override"
    )
    payload["result"] = result.to_json()
    _write(output, payload)
    print(
        f"I={result.mean.real:.12g}{result.mean.imag:+.12g}i "
        f"+/-({result.standard_error_real:.3g},"
        f"{result.standard_error_imag:.3g}); wrote {output}"
    )


if __name__ == "__main__":
    main()
