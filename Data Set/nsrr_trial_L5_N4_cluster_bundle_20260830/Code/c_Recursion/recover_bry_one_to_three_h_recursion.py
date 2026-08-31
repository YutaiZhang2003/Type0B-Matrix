#!/usr/bin/env python3
"""Recover BRY's regulated Type-0B sphere four point with h-recursion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bry_one_to_three import BRYOneToThreeBenchmark


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "bry_one_to_three_h_recursion_recovery.json"
POINTWISE_AUDIT_POINTS = (
    (0.37, 0.31 + 0.27j),
    (0.83, 0.71 + 0.16j),
)
RECORDED_BRY_Q8 = 0.01750879244 - 0.00320541444j


def _integer_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("q orders must be positive integers")
    return result


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _uses_recorded_bry_grid(args: argparse.Namespace) -> bool:
    return (
        args.incoming_imaginary == 0.6
        and args.epsilon == 1.0e-2
        and args.p_max == 4.0
        and args.p_order == 24
        and args.angular_order == 14
        and args.radial_order == 14
        and args.cap_angular_order == 14
        and args.cap_radial_order == 10
        and args.structure_precision == 30
        and args.block_precision == 60
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-imaginary", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=1.0e-2)
    parser.add_argument("--p-max", type=float, default=4.0)
    parser.add_argument("--p-order", type=int, default=24)
    parser.add_argument("--angular-order", type=int, default=14)
    parser.add_argument("--radial-order", type=int, default=14)
    parser.add_argument("--cap-angular-order", type=int, default=14)
    parser.add_argument("--cap-radial-order", type=int, default=10)
    parser.add_argument("--q-orders", type=_integer_list, default=(8,))
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--block-precision", type=int, default=60)
    parser.add_argument("--matrix-tolerance", type=float, default=0.03)
    parser.add_argument("--backend-audit-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--bry-reference-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _benchmark(args: argparse.Namespace, q_order: int, backend: str):
    return BRYOneToThreeBenchmark(
        incoming_imaginary=args.incoming_imaginary,
        epsilon=args.epsilon,
        p_max=args.p_max,
        p_quadrature_order=args.p_order,
        angular_order=args.angular_order,
        radial_order=args.radial_order,
        cap_angular_order=args.cap_angular_order,
        cap_radial_order=args.cap_radial_order,
        block_q_order=q_order,
        block_backend=backend,
        structure_precision=args.structure_precision,
        block_working_precision=args.block_precision,
    )


def _pointwise_backend_audit(
    args: argparse.Namespace, q_order: int
) -> dict[str, object]:
    h_benchmark = _benchmark(args, q_order, "h")
    c_benchmark = _benchmark(args, q_order, "c")
    records = []
    maximum_relative_difference = 0.0
    for momentum, z in POINTWISE_AUDIT_POINTS:
        h_value = h_benchmark.direct_momentum_density(momentum, z)
        c_value = c_benchmark.direct_momentum_density(momentum, z)
        relative = abs(h_value - c_value) / max(abs(c_value), 1.0e-300)
        maximum_relative_difference = max(maximum_relative_difference, relative)
        records.append(
            {
                "momentum": momentum,
                "z": _complex_json(z),
                "h_density": _complex_json(h_value),
                "c_density": _complex_json(c_value),
                "relative_difference": relative,
            }
        )
    return {
        "q_order": q_order,
        "records": records,
        "maximum_relative_difference": maximum_relative_difference,
        "tolerance": args.backend_audit_tolerance,
        "passed": maximum_relative_difference <= args.backend_audit_tolerance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.matrix_tolerance <= 0.0
        or args.backend_audit_tolerance <= 0.0
        or args.bry_reference_tolerance <= 0.0
    ):
        raise ValueError("comparison tolerances must be positive")
    q_orders = tuple(sorted(set(args.q_orders)))
    backend_audit = _pointwise_backend_audit(args, max(q_orders))
    if not backend_audit["passed"]:
        raise RuntimeError("h- and c-recursion failed the pointwise density audit")

    results = []
    for q_order in q_orders:
        print(f"evaluating h-recursive BRY integral at q^{q_order}", flush=True)
        result = _benchmark(args, q_order, "h").evaluate()
        results.append(result.json_dict())
        print(
            f"q^{q_order}: M={result.reduced_moduli_integral.real:+.10f}"
            f"{result.reduced_moduli_integral.imag:+.10f}i, "
            f"relative target error={result.relative_error:.6%}",
            flush=True,
        )

    finest = results[-1]
    finest_error = float(finest["relative_error"])
    q8_result = next(
        (item for item in results if item["block_q_order"] == 8), None
    )
    reference_check = None
    if q8_result is not None and _uses_recorded_bry_grid(args):
        computed = complex(
            q8_result["reduced_moduli_integral"]["real"],
            q8_result["reduced_moduli_integral"]["imag"],
        )
        relative = abs(computed - RECORDED_BRY_Q8) / abs(RECORDED_BRY_Q8)
        reference_check = {
            "reference": _complex_json(RECORDED_BRY_Q8),
            "computed": _complex_json(computed),
            "relative_difference": relative,
            "tolerance": args.bry_reference_tolerance,
            "passed": relative <= args.bry_reference_tolerance,
        }
    payload = {
        "description": __doc__,
        "scheme": {
            "integral_backend": "h recursion only",
            "c_backend_role": "pointwise validation only; not used in the integral",
            "bulk": "elliptic sphere block with fixed-difference h-recursion",
            "low_z_boundary": "BRY direct OPE through the recorded local orders",
            "t_boundary": "BRY crossed OPE with the leading CC power subtracted",
            "analytic_subtraction": (
                "global folded t-channel counterterm on 0<=P<P_star"
            ),
            "momentum_contour": "positive real P, split at P_star",
            "liouville_pole_residues": "none crossed on the BRY (4.15) family",
        },
        "settings": {
            "incoming_imaginary": args.incoming_imaginary,
            "epsilon": args.epsilon,
            "p_max": args.p_max,
            "p_order_per_threshold_interval": args.p_order,
            "angular_order": args.angular_order,
            "radial_order": args.radial_order,
            "cap_angular_order": args.cap_angular_order,
            "cap_radial_order": args.cap_radial_order,
            "q_orders": list(q_orders),
            "structure_precision": args.structure_precision,
            "block_precision": args.block_precision,
        },
        "pointwise_h_vs_c_audit": backend_audit,
        "recorded_bry_q8_check": reference_check,
        "results": results,
        "finest_relative_error": finest_error,
        "matrix_tolerance": args.matrix_tolerance,
        "recovered_bry_curve": finest_error <= args.matrix_tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    if finest_error > args.matrix_tolerance:
        raise SystemExit("the h-recursive integral did not recover the BRY tolerance")
    if reference_check is not None and not reference_check["passed"]:
        raise SystemExit("the h-recursive integral did not recover the recorded q^8 value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
