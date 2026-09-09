#!/usr/bin/env python3
"""Target-free high-accuracy BRY sphere-four-point worldsheet campaign.

The output contains only native worldsheet quantities.  Matrix-model data and
comparison formulae deliberately live in a separate post-freeze program.
Several controls sharing the same Liouville-momentum nodes are evaluated in
one worker so their h-recursive coefficient caches can be reused.
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
DEFAULT_OUTPUT = HERE / "results" / "bry_one_to_three_high_accuracy_primary.json"

PRIMARY_VARIANTS = (
    {
        "name": "candidate_q12_eps005_z20",
        "q_order": 12,
        "epsilon": 0.005,
        "angular_order": 20,
        "radial_order": 20,
        "cap_angular_order": 20,
        "cap_radial_order": 14,
    },
    {
        "name": "block_control_q10_eps005_z20",
        "q_order": 10,
        "epsilon": 0.005,
        "angular_order": 20,
        "radial_order": 20,
        "cap_angular_order": 20,
        "cap_radial_order": 14,
    },
    {
        "name": "cap_control_q12_eps0075_z20",
        "q_order": 12,
        "epsilon": 0.0075,
        "angular_order": 20,
        "radial_order": 20,
        "cap_angular_order": 20,
        "cap_radial_order": 14,
    },
    {
        "name": "moduli_control_q12_eps005_z18",
        "q_order": 12,
        "epsilon": 0.005,
        "angular_order": 18,
        "radial_order": 18,
        "cap_angular_order": 18,
        "cap_radial_order": 12,
    },
)

CANDIDATE_ONLY = (PRIMARY_VARIANTS[0],)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _momentum_nodes(
    lower: float,
    upper: float,
    order: int,
    threshold: float,
) -> tuple[tuple[int, float, float], ...]:
    breaks = [lower]
    if lower < threshold < upper:
        breaks.append(threshold)
    breaks.append(upper)
    records = []
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
    index, momentum, weight, common, variants = task
    benchmark = BRYOneToThreeBenchmark(
        incoming_imaginary=common["incoming_imaginary"],
        epsilon=max(item["epsilon"] for item in variants),
        p_max=common["p_max"],
        p_quadrature_order=common["p_order"],
        angular_order=max(item["angular_order"] for item in variants),
        radial_order=max(item["radial_order"] for item in variants),
        cap_angular_order=max(item["cap_angular_order"] for item in variants),
        cap_radial_order=max(item["cap_radial_order"] for item in variants),
        block_q_order=max(item["q_order"] for item in variants),
        block_backend="h",
        structure_precision=common["structure_precision"],
        block_working_precision=common["block_precision"],
    )

    low_cache = {}
    bulk_cache = {}
    cap_cache = {}
    values = {}
    for variant in variants:
        benchmark.epsilon = variant["epsilon"]
        benchmark.angular_order = variant["angular_order"]
        benchmark.radial_order = variant["radial_order"]
        benchmark.cap_angular_order = variant["cap_angular_order"]
        benchmark.cap_radial_order = variant["cap_radial_order"]
        benchmark.block_q_order = variant["q_order"]
        benchmark.sphere.liouville.bry_q_order = variant["q_order"]

        cap_key = (
            benchmark.epsilon,
            benchmark.cap_angular_order,
            benchmark.cap_radial_order,
        )
        bulk_key = (
            benchmark.epsilon,
            benchmark.angular_order,
            benchmark.radial_order,
            benchmark.block_q_order,
        )
        if cap_key not in low_cache:
            low_cache[cap_key] = benchmark._s_disk_integral(momentum)
            cap_cache[cap_key] = benchmark._t_cap_integral(momentum)
        if bulk_key not in bulk_cache:
            bulk_cache[bulk_key] = benchmark._rest_of_disk_integral(momentum)
        values[variant["name"]] = (
            low_cache[cap_key],
            bulk_cache[bulk_key],
            cap_cache[cap_key],
        )
    return index, momentum, weight, values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        choices=("primary", "p-order-control", "tail-control"),
        default="primary",
    )
    parser.add_argument("--incoming-imaginary", type=float, default=0.6)
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--p-max", type=float)
    parser.add_argument("--p-order", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--block-precision", type=int, default=70)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _campaign_defaults(args: argparse.Namespace):
    if args.campaign == "primary":
        return (
            0.0 if args.p_min is None else args.p_min,
            5.0 if args.p_max is None else args.p_max,
            30 if args.p_order is None else args.p_order,
            PRIMARY_VARIANTS,
        )
    if args.campaign == "p-order-control":
        return (
            0.0 if args.p_min is None else args.p_min,
            5.0 if args.p_max is None else args.p_max,
            24 if args.p_order is None else args.p_order,
            CANDIDATE_ONLY,
        )
    return (
        5.0 if args.p_min is None else args.p_min,
        6.0 if args.p_max is None else args.p_max,
        10 if args.p_order is None else args.p_order,
        CANDIDATE_ONLY,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p_min, p_max, p_order, variants = _campaign_defaults(args)
    if not 0.0 <= p_min < p_max or p_order < 2 or args.workers < 1:
        raise ValueError("invalid momentum interval, order, or worker count")

    probe = BRYOneToThreeBenchmark(
        incoming_imaginary=args.incoming_imaginary,
        block_backend="h",
    )
    common = {
        "incoming_imaginary": args.incoming_imaginary,
        "p_max": p_max,
        "p_order": p_order,
        "structure_precision": args.structure_precision,
        "block_precision": args.block_precision,
    }
    nodes = _momentum_nodes(p_min, p_max, p_order, probe.t_threshold)
    tasks = tuple(
        (index, momentum, weight, common, variants)
        for index, momentum, weight in nodes
    )
    print(
        f"campaign={args.campaign}, nodes={len(tasks)}, workers={args.workers}, "
        f"variants={len(variants)}",
        flush=True,
    )
    if args.workers == 1:
        evaluated = tuple(_evaluate_node(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            evaluated = tuple(executor.map(_evaluate_node, tasks, chunksize=1))
    evaluated = tuple(sorted(evaluated, key=lambda item: item[0]))

    totals = {
        variant["name"]: [0.0j, 0.0j, 0.0j] for variant in variants
    }
    for _, _, weight, node_values in evaluated:
        for name, region_values in node_values.items():
            for region, value in enumerate(region_values):
                totals[name][region] += weight * value

    records = []
    for variant in variants:
        low, bulk, cap = (complex(value) for value in totals[variant["name"]])
        reduced = low + bulk + cap
        records.append(
            {
                **variant,
                "block_backend": "h",
                "low_z_region_integral": _complex_pair(low),
                "bulk_region_integral": _complex_pair(bulk),
                "t_cap_region_integral": _complex_pair(cap),
                "reduced_moduli_integral": _complex_pair(reduced),
                "worldsheet_amplitude_coefficient": _complex_pair(
                    8j * reduced / math.pi
                ),
            }
        )
        print(f"{variant['name']}: {reduced.real:+.14f}{reduced.imag:+.14f}i")

    payload = {
        "status": "worldsheet_only_unfrozen",
        "calculation": "BRY regulated Type-0B sphere four-point integral",
        "comparison_performed": False,
        "matrix_model_data_included": False,
        "recursion_backend": "h",
        "campaign": args.campaign,
        "kinematics": {
            "incoming_energy": _complex_pair(probe.omega),
            "each_outgoing_energy": _complex_pair(probe.omega1),
        },
        "momentum_quadrature": {
            "minimum": p_min,
            "maximum": p_max,
            "order_per_threshold_interval": p_order,
            "threshold": probe.t_threshold,
            "node_count": len(nodes),
        },
        "precision": {
            "structure_digits": args.structure_precision,
            "block_digits": args.block_precision,
        },
        "source_sha256": {
            Path(__file__).name: _sha256(Path(__file__)),
            "bry_one_to_three.py": _sha256(HERE / "bry_one_to_three.py"),
            "sphere_four_point.py": _sha256(HERE / "sphere_four_point.py"),
            "ns_multipoint_h_recursion.py": _sha256(
                HERE / "ns_multipoint_h_recursion.py"
            ),
        },
        "variants": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
