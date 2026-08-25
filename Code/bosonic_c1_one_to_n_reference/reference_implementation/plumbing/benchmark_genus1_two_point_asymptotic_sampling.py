#!/usr/bin/env python3
"""Compare product and correlated asymptotic momentum rules at fixed points."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

try:
    from genus1_two_point_adaptive_momentum import (
        _complex_record,
        _relative_change,
        default_points,
        evaluate_old_anchor,
        evaluate_point,
        evaluate_point_polar,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_adaptive_momentum import (
        _complex_record,
        _relative_change,
        default_points,
        evaluate_old_anchor,
        evaluate_point,
        evaluate_point_polar,
    )


def _parse_pair(text: str) -> tuple[int, int]:
    values = tuple(int(value) for value in text.lower().split("x"))
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"invalid order pair {text!r}")
    return values  # type: ignore[return-value]


def run(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--points", default="moderate-bulk,moderate-collision,moderate-disc")
    parser.add_argument("--polar-orders", default="4x6,6x8,8x10,10x12,12x14")
    parser.add_argument("--product-orders", default="8,12,16,20,24,28")
    parser.add_argument("--necklace-orders", default="6,3")
    parser.add_argument("--ope-orders", default="3,8")
    parser.add_argument("--dps", type=int, default=28)
    parser.add_argument("--old-order", type=int, default=16)
    parser.add_argument("--p-max", type=float, default=6.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        help="Optional output of analyze_genus1_two_point_momentum_asymptotics.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "plumbing/results/genus1_two_point_worldsheet/"
            "asymptotic_sampling_benchmark_x04.json"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    polar_orders = tuple(_parse_pair(value) for value in args.polar_orders.split(","))
    product_orders = tuple(int(value) for value in args.product_orders.split(","))
    necklace_orders = tuple(int(value) for value in args.necklace_orders.split(","))
    ope_orders = tuple(int(value) for value in args.ope_orders.split(","))
    if len(necklace_orders) != 2 or len(ope_orders) != 2:
        raise ValueError("block-order pairs must contain two values")
    point_map = {point.name: point for point in default_points()}
    names = tuple(value for value in args.points.split(",") if value)
    if any(name not in point_map for name in names):
        raise ValueError("unknown fixed audit point")
    tuned_proposals: dict[str, dict[str, float]] = {}
    if args.proposal_json is not None:
        proposal_payload = json.loads(args.proposal_json.read_text())
        tuned_proposals = {
            str(row["name"]): {
                key: float(value)
                for key, value in row["recommended_exact_proposal"].items()
                if key
                in {
                    "common_decay_scale",
                    "angular_jacobi_alpha",
                    "angular_jacobi_beta",
                }
            }
            for row in proposal_payload["results"]
        }

    records: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for name in names:
        point = point_map[name]
        point_record: dict[str, object] = {
            "name": name,
            "channel": point.channel,
            "product": [],
            "polar": [],
            "polar_tuned": [],
        }
        previous: complex | None = None
        for order in product_orders:
            started = time.perf_counter()
            value, _ = evaluate_point(
                point,
                x=args.x,
                order=order,
                necklace_orders=necklace_orders,  # type: ignore[arg-type]
                ope_orders=ope_orders,  # type: ignore[arg-type]
                dps=args.dps,
            )
            drift = None if previous is None else _relative_change(previous, value)
            row = {
                "order": order,
                "node_count": order * order,
                "value": _complex_record(value),
                "relative_change": drift,
                "runtime_seconds": time.perf_counter() - started,
            }
            point_record["product"].append(row)  # type: ignore[union-attr]
            print(
                f"{name:22s} product Q={order:2d} nodes={order*order:4d} "
                f"value={value.real:+.12e} "
                f"step={float('nan') if drift is None else drift:.3e}",
                flush=True,
            )
            previous = value

        if name in tuned_proposals:
            proposal = tuned_proposals[name]
            previous = None
            for radial_order, angular_order in polar_orders:
                started = time.perf_counter()
                value, rule = evaluate_point_polar(
                    point,
                    x=args.x,
                    radial_order=radial_order,
                    angular_order=angular_order,
                    necklace_orders=necklace_orders,  # type: ignore[arg-type]
                    ope_orders=ope_orders,  # type: ignore[arg-type]
                    dps=args.dps,
                    angular_jacobi_alpha=proposal["angular_jacobi_alpha"],
                    angular_jacobi_beta=proposal["angular_jacobi_beta"],
                    decay_scale=proposal["common_decay_scale"],
                )
                drift = (
                    None if previous is None else _relative_change(previous, value)
                )
                row = {
                    "radial_order": radial_order,
                    "angular_order": angular_order,
                    "node_count": radial_order * angular_order,
                    "value": _complex_record(value),
                    "relative_change": drift,
                    "runtime_seconds": time.perf_counter() - started,
                    "common_decay_scale": proposal["common_decay_scale"],
                    "angular_jacobi_alpha": rule.angular_jacobi_alpha,
                    "angular_jacobi_beta": rule.angular_jacobi_beta,
                }
                point_record["polar_tuned"].append(row)  # type: ignore[union-attr]
                print(
                    f"{name:22s} tuned {radial_order:2d}x{angular_order:2d} "
                    f"nodes={radial_order*angular_order:4d} "
                    f"value={value.real:+.12e} "
                    f"step={float('nan') if drift is None else drift:.3e}",
                    flush=True,
                )
                previous = value

        previous = None
        for radial_order, angular_order in polar_orders:
            started = time.perf_counter()
            value, rule = evaluate_point_polar(
                point,
                x=args.x,
                radial_order=radial_order,
                angular_order=angular_order,
                necklace_orders=necklace_orders,  # type: ignore[arg-type]
                ope_orders=ope_orders,  # type: ignore[arg-type]
                dps=args.dps,
            )
            drift = None if previous is None else _relative_change(previous, value)
            row = {
                "radial_order": radial_order,
                "angular_order": angular_order,
                "node_count": radial_order * angular_order,
                "value": _complex_record(value),
                "relative_change": drift,
                "runtime_seconds": time.perf_counter() - started,
                "radial_laguerre_alpha": rule.radial_laguerre_alpha,
                "angular_jacobi_alpha": rule.angular_jacobi_alpha,
                "angular_jacobi_beta": rule.angular_jacobi_beta,
            }
            point_record["polar"].append(row)  # type: ignore[union-attr]
            print(
                f"{name:22s} polar {radial_order:2d}x{angular_order:2d} "
                f"nodes={radial_order*angular_order:4d} "
                f"value={value.real:+.12e} "
                f"step={float('nan') if drift is None else drift:.3e}",
                flush=True,
            )
            previous = value

        old_started = time.perf_counter()
        old = evaluate_old_anchor(
            point,
            x=args.x,
            order=args.old_order,
            p_max=args.p_max,
            power=args.power,
            necklace_orders=necklace_orders,  # type: ignore[arg-type]
            ope_orders=ope_orders,  # type: ignore[arg-type]
            dps=args.dps,
        )
        point_record["old_anchor"] = {
            "order": args.old_order,
            "value": _complex_record(old),
            "runtime_seconds": time.perf_counter() - old_started,
        }
        records.append(point_record)

    payload: dict[str, object] = {
        "calculation": "genus-one two-point asymptotic momentum sampler benchmark",
        "x": float(args.x),
        "block_orders": {
            "necklace": list(necklace_orders),
            "ope": list(ope_orders),
        },
        "total_runtime_seconds": time.perf_counter() - total_started,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return payload


if __name__ == "__main__":
    run()
