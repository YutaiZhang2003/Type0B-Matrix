#!/usr/bin/env python3
"""Target-blind analytic-continuation pilot with q-gated h/c recursion.

The default kinematics lie at the center of the certified wall-one chamber

    omega_j = a_j (x + i t),   a = (0.1, 1, 1).

The positive-real starting sheet is continued vertically to the endpoint.
Every crossed Liouville pole is included.  The moduli integrand uses linear
h-recursion only for the verified picture routing and only while
``|q_ell| < 0.3``; all other evaluations use a channel-adapted c-recursion.
No matrix-model value is imported or computed by this driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from superconformal_blocks import elliptic_nome
from type0b_sphere_four_point_continuation import (
    certify_four_point_continuation_rectangle,
)
from type0b_sphere_four_point_hybrid import (
    Type0BSphereFourPointHybrid,
    WALL_ONE_RAY_COEFFICIENTS,
    WALL_ONE_RAY_RECTANGLE,
    _crossing_cell_channel,
    audit_four_point_crossing,
    integrate_subtraction_free_four_point,
    integrate_subtraction_free_four_point_component_cells,
    integrate_subtraction_free_four_point_component_stratified_qmc,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE
    / "results"
    / "type0b_sphere_four_point_analytic_hybrid_wall_one_stratified_pilot.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _routing_audit(
    kernel: Type0BSphereFourPointHybrid,
    z: complex,
    frames: Sequence[int],
) -> dict[str, object]:
    positions = kernel.fixed_positions(z)
    records: list[dict[str, object]] = []
    h_nomes: list[float] = []
    for frame in frames:
        channel = _crossing_cell_channel(positions, frame)
        backend = kernel._selected_backend(channel, "auto")
        nome = float(abs(elliptic_nome(channel.q)))
        if backend == "h":
            h_nomes.append(nome)
        records.append(
            {
                "frame": frame,
                "ordering": list(channel.ordering),
                "cross_ratio": _pair(channel.q),
                "elliptic_nome_magnitude": nome,
                "backend": backend,
            }
        )
    maximum_h_nome = max(h_nomes, default=0.0)
    passed = maximum_h_nome < kernel.hybrid_elliptic_nome_threshold
    if not passed:
        raise ArithmeticError("an h-recursive frame escaped the nome gate")
    return {
        "z": _pair(z),
        "frames_checked": list(frames),
        "threshold": kernel.hybrid_elliptic_nome_threshold,
        "maximum_h_nome": maximum_h_nome,
        "passed": passed,
        "frames": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.271)
    parser.add_argument("--t", type=float, default=0.612)
    parser.add_argument("--twice-level", type=int, default=8)
    parser.add_argument("--q-threshold", type=float, default=0.3)
    parser.add_argument("--momentum-order", type=int, default=30)
    parser.add_argument("--momentum-maximum", type=float, default=3.0)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-precision", type=int, default=45)
    parser.add_argument("--sobol-power", type=int, default=5)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument(
        "--integration-method",
        choices=("verified-cells", "stratified-qmc", "qmc"),
        default="stratified-qmc",
    )
    parser.add_argument("--radial-order", type=int, default=2)
    parser.add_argument("--angular-order", type=int, default=4)
    parser.add_argument("--crossing-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crossing-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    x_interval, t_interval = WALL_ONE_RAY_RECTANGLE
    if not x_interval[0] < args.x < x_interval[1]:
        raise ValueError("x must lie strictly inside the certified rectangle")
    if not t_interval[0] < args.t < t_interval[1]:
        raise ValueError("t must lie strictly inside the certified rectangle")
    if args.momentum_order != 30 or abs(args.momentum_maximum - 3.0) > 1.0e-14:
        raise ValueError("the certified pilot uses the 30-node wall-one rule")

    certificate = certify_four_point_continuation_rectangle(
        x_interval,
        t_interval,
        ray_coefficients=WALL_ONE_RAY_COEFFICIENTS,
        ray_real_sign=1,
        required_minimum_margin=0.02,
        required_wall_clearance=0.01,
        supported_product_pole_order=2,
    )
    if not certificate.production_ready:
        raise ArithmeticError("the requested continuation chamber is not certified")
    if certificate.crossed_walls != (1,):
        raise ArithmeticError("the default pilot must remain in the wall-one chamber")

    ray = complex(args.x, args.t)
    outgoing = tuple(
        coefficient * ray for coefficient in WALL_ONE_RAY_COEFFICIENTS
    )
    kernel = Type0BSphereFourPointHybrid(
        outgoing_energies=outgoing,
        contour_prescription="continued",
        block_backend="hybrid",
        hybrid_elliptic_nome_threshold=args.q_threshold,
        recursion_max_twice_level=args.twice_level,
        momentum_order=args.momentum_order,
        momentum_maximum=args.momentum_maximum,
        momentum_rule="wall-one-30",
        structure_precision=args.structure_precision,
        block_working_precision=args.block_precision,
    )
    first_crossing_point = 0.37 + 0.28j
    second_crossing_point = 1.0 / (1.0 - first_crossing_point)
    routing = (
        _routing_audit(kernel, first_crossing_point, (0, 1)),
        _routing_audit(kernel, second_crossing_point, (1, 2)),
    )
    crossings = tuple(
        audit_four_point_crossing(
            kernel,
            point,
            frames=frames,
            block_region="auto",
            relative_tolerance=args.crossing_tolerance,
        )
        for point, frames in (
            (first_crossing_point, (0, 1)),
            (second_crossing_point, (1, 2)),
        )
    )
    crossing_passed = all(audit.passed for audit in crossings)
    maximum_crossing_spread = max(
        audit.relative_spread for audit in crossings
    )

    payload: dict[str, object] = {
        "status": (
            "target_blind_crossing_passed"
            if crossing_passed
            else "blocked_by_crossing"
        ),
        "comparison_performed": False,
        "matrix_model_data_included": False,
        "calculation": (
            "analytic continuation from the positive-real energy sheet, "
            "including the complete wall-one Liouville residue ledger"
        ),
        "kinematics": {
            "ray_coefficients": list(WALL_ONE_RAY_COEFFICIENTS),
            "ray_base": _pair(ray),
            "outgoing_energies": [_pair(value) for value in outgoing],
            "incoming_energy": _pair(sum(outgoing)),
        },
        "continuation": {
            "path": "lambda(s) = x + i s t, 0 <= s <= 1",
            "origin": "positive real energy axis",
            "crossed_walls": list(certificate.crossed_walls),
            "domain_certificate": certificate.to_json(
                include_exponent_bounds=True
            ),
            "pointwise_convergence_audit": kernel.audit.to_json(),
        },
        "recursion_atlas": {
            "h_rule": (
                "linear h recursion only for the verified picture routing "
                f"and |q_ell| < {args.q_threshold}"
            ),
            "c_rule": "channel-adapted c recursion everywhere else",
            "maximum_twice_level": args.twice_level,
            "routing_audits": routing,
        },
        "crossing_audits": [audit.to_json() for audit in crossings],
        "settings": {
            "q_threshold": args.q_threshold,
            "momentum_order": args.momentum_order,
            "momentum_maximum": args.momentum_maximum,
            "momentum_rule": "wall-one-30",
            "structure_precision": args.structure_precision,
            "block_precision": args.block_precision,
            "sobol_power": args.sobol_power,
            "replicates": args.replicates,
            "integration_method": args.integration_method,
            "radial_order": args.radial_order,
            "angular_order": args.angular_order,
            "crossing_tolerance": args.crossing_tolerance,
            "seed": args.seed,
        },
        "source_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__).resolve(),
                HERE / "type0b_sphere_four_point_hybrid.py",
                HERE / "type0b_sphere_four_point_continuation.py",
                HERE / "ns_multipoint_h_recursion.py",
                HERE / "ns_multipoint_c_recursion.py",
            )
        },
    }
    if crossing_passed and not args.crossing_only:
        if args.integration_method == "verified-cells":
            result = integrate_subtraction_free_four_point_component_cells(
                kernel,
                radial_order=args.radial_order,
                angular_order=args.angular_order,
                replicates=args.replicates,
            )
        elif args.integration_method == "stratified-qmc":
            result = integrate_subtraction_free_four_point_component_stratified_qmc(
                kernel,
                sobol_power=args.sobol_power,
                replicates=args.replicates,
                seed=args.seed,
            )
        else:
            result = integrate_subtraction_free_four_point(
                kernel,
                sobol_power=args.sobol_power,
                replicates=args.replicates,
                seed=args.seed,
            )
        payload["status"] = "target_blind_worldsheet_pilot_unfrozen"
        payload["worldsheet_result"] = result.to_json()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"maximum crossing spread={maximum_crossing_spread:.6g}; "
        f"status={payload['status']}; wrote {args.output}",
        flush=True,
    )
    return 0 if crossing_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
