#!/usr/bin/env python3
"""Target-blind BRY 1->3 integral with q-gated h/c recursion.

The linear s/u charts use h recursion only while the active elliptic nome
satisfies |q_ell| <= 0.3.  The complementary corner patch is evaluated in a
channel-adapted c-recursive chart.  BRY's leading t-channel OPE polynomial is
subtracted explicitly at the integrand level; the s- and u-channel ledgers
are empty for the selected complex-energy family.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from bry_one_to_three import BRYOneToThreeBenchmark, _legendre_rule


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "bry_one_to_three_hybrid_worldsheet.json"


def _pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _momentum_nodes(
    lower: float, upper: float, order: int, threshold: float
) -> tuple[tuple[int, float, float], ...]:
    breaks = [lower]
    if lower < threshold < upper:
        breaks.append(threshold)
    breaks.append(upper)
    records: list[tuple[int, float, float]] = []
    index = 0
    for left, right in zip(breaks, breaks[1:]):
        midpoint = 0.5 * (left + right)
        scale = 0.5 * (right - left)
        for node, weight in _legendre_rule(order):
            records.append(
                (index, midpoint + scale * node, scale * weight)
            )
            index += 1
    return tuple(records)


def _evaluate_node(task):
    index, momentum, weight, settings = task
    benchmark = BRYOneToThreeBenchmark(
        incoming_imaginary=settings["incoming_imaginary"],
        epsilon=settings["epsilon"],
        p_max=settings["p_max"],
        p_quadrature_order=settings["p_order"],
        angular_order=settings["angular_order"],
        radial_order=settings["radial_order"],
        cap_angular_order=settings["cap_angular_order"],
        cap_radial_order=settings["cap_radial_order"],
        block_q_order=settings["q_order"],
        block_backend="hybrid",
        hybrid_elliptic_nome_threshold=settings["q_threshold"],
        hybrid_asymptotic_radius=settings["asymptotic_radius"],
        structure_precision=settings["structure_precision"],
        block_working_precision=settings["block_precision"],
    )
    if settings["region"] == "t-cap":
        values = (0.0j, 0.0j, benchmark._t_cap_integral(momentum))
    else:
        values = (
            benchmark._s_disk_integral(momentum),
            benchmark._rest_of_disk_integral(momentum),
            benchmark._t_cap_integral(momentum),
        )
    return index, momentum, weight, values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-imaginary", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=5.0e-3)
    parser.add_argument("--p-max", type=float, default=5.0)
    parser.add_argument("--p-order", type=int, default=30)
    parser.add_argument("--angular-order", type=int, default=20)
    parser.add_argument("--radial-order", type=int, default=20)
    parser.add_argument("--cap-angular-order", type=int, default=20)
    parser.add_argument("--cap-radial-order", type=int, default=14)
    parser.add_argument("--q-order", type=int, default=12)
    parser.add_argument("--q-threshold", type=float, default=0.3)
    parser.add_argument("--asymptotic-radius", type=float, default=1.0e-4)
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--block-precision", type=int, default=70)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--region", choices=("all", "t-cap"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.p_order < 2 or args.workers < 1:
        raise ValueError("p-order must be at least two and workers positive")
    probe = BRYOneToThreeBenchmark(
        incoming_imaginary=args.incoming_imaginary,
        block_backend="hybrid",
        block_q_order=args.q_order,
        hybrid_elliptic_nome_threshold=args.q_threshold,
        hybrid_asymptotic_radius=args.asymptotic_radius,
        structure_precision=args.structure_precision,
        block_working_precision=args.block_precision,
    )
    settings = {
        "incoming_imaginary": args.incoming_imaginary,
        "epsilon": args.epsilon,
        "p_max": args.p_max,
        "p_order": args.p_order,
        "angular_order": args.angular_order,
        "radial_order": args.radial_order,
        "cap_angular_order": args.cap_angular_order,
        "cap_radial_order": args.cap_radial_order,
        "q_order": args.q_order,
        "q_threshold": args.q_threshold,
        "asymptotic_radius": args.asymptotic_radius,
        "structure_precision": args.structure_precision,
        "block_precision": args.block_precision,
        "region": args.region,
    }
    nodes = _momentum_nodes(0.0, args.p_max, args.p_order, probe.t_threshold)
    tasks = tuple(
        (index, momentum, weight, settings)
        for index, momentum, weight in nodes
    )
    print(
        f"hybrid worldsheet: nodes={len(tasks)}, workers={args.workers}, "
        f"q^{args.q_order}, |q_ell|<={args.q_threshold}",
        flush=True,
    )
    if args.workers == 1:
        evaluated = tuple(_evaluate_node(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = tuple(executor.map(_evaluate_node, tasks, chunksize=1))
    totals = [0.0j, 0.0j, 0.0j]
    for _, _, weight, values in sorted(evaluated):
        for region, value in enumerate(values):
            totals[region] += weight * value
    low, bulk, cap = (complex(value) for value in totals)
    reduced = low + bulk + cap
    payload = {
        "status": "worldsheet_only_unfrozen",
        "comparison_performed": False,
        "matrix_model_data_included": False,
        "calculation": "BRY regulated Type-0B sphere four-point integral",
        "recursion_atlas": {
            "bulk": "linear-channel h recursion",
            "bulk_gate": f"|q_ell| < {args.q_threshold}",
            "corners": "channel-adapted c recursion",
            "maximum_twice_level": 2 * args.q_order,
        },
        "polynomial_subtraction": {
            "s_channel": [],
            "t_channel": [
                {
                    "sector": "NS-C",
                    "momentum_range": [0.0, probe.t_threshold],
                    "coefficient": (
                        "C(omega1,omega,P) C(omega2,omega3,P) "
                        "[1+(omega2+omega3)^2-P^2]^2/(4*pi)"
                    ),
                    "radial_power": (
                        "-3+P^2-(omega2+omega3)^2"
                    ),
                }
            ],
            "u_channel": [],
            "implementation": "subtracted pointwise before moduli quadrature",
        },
        "settings": settings,
        "kinematics": {
            "incoming_energy": _pair(probe.omega),
            "each_outgoing_energy": _pair(probe.omega1),
        },
        "regions": {
            "low_z": _pair(low),
            "bulk": _pair(bulk),
            "t_cap": _pair(cap),
        },
        "reduced_moduli_integral": _pair(reduced),
        "worldsheet_amplitude_coefficient": _pair(8j * reduced / math.pi),
        "source_sha256": {
            path.name: _sha256(path)
            for path in (
                Path(__file__).resolve(),
                HERE / "bry_one_to_three.py",
                HERE / "type0b_sphere_four_point_hybrid.py",
                HERE / "ns_multipoint_c_recursion.py",
                HERE / "ns_multipoint_h_recursion.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"M={reduced.real:+.14f}{reduced.imag:+.14f}i; wrote {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
