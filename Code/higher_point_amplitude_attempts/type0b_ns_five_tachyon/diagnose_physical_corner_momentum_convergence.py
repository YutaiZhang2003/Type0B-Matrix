#!/usr/bin/env python3
"""Deterministic momentum-order audit for the physical five-point corners."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Sequence

from type0b_ns_five_tachyon import (
    BOUNDARY_CORNER_RAISED_ORBITS,
    BRYNSFiveTachyonIntegrand,
)


def _order_pair(value: str) -> tuple[int, int]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("an order pair must have the form N1,N2")
    result = tuple(int(field) for field in fields)
    if min(result) < 2:
        raise argparse.ArgumentTypeError("both momentum orders must be at least two")
    return result  # type: ignore[return-value]


def _corner_task(spec: tuple[object, ...]) -> dict[str, object]:
    (
        orders,
        ordering,
        multiplicity,
        epsilon,
        momentum_maximum,
        collar_radius,
        projection_radius,
        central_charge_shift,
        structure_precision,
        block_working_precision,
    ) = spec
    kernel = BRYNSFiveTachyonIntegrand(
        outgoing_energies=(0.25 + 1.0j * float(epsilon),) * 4,
        block_backend="c",
        recursion_max_twice_level=None,
        global_max_twice_levels=(4, 4),
        global_max_total_twice_level=6,
        momentum_orders=orders,
        momentum_maximum=float(momentum_maximum),
        structure_precision=int(structure_precision),
        central_charge_shift=float(central_charge_shift),
        block_working_precision=int(block_working_precision),
    )
    started = time.time()
    value = kernel.boundary_corner_finite_part(
        ordering=ordering,
        collar_radius=float(collar_radius),
        projection_radius=float(projection_radius),
        momentum_refinement_shells=-1,
        momentum_singularity_subtraction=True,
    )
    return {
        "orders": list(orders),
        "ordering": list(ordering),
        "multiplicity": int(multiplicity),
        "momentum_maximum": float(momentum_maximum),
        "value": {"real": value.real, "imag": value.imag},
        "seconds": time.time() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orders", type=_order_pair, nargs="+", default=((4, 5), (5, 6))
    )
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument(
        "--momentum-maximum", type=float, nargs="+", default=(1.5,)
    )
    parser.add_argument("--collar-radius", type=float, default=0.08)
    parser.add_argument("--projection-radius", type=float, default=1.0e-5)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-5)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-working-precision", type=int, default=45)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    specs = [
        (
            orders,
            ordering,
            multiplicity,
            args.epsilon,
            momentum_maximum,
            args.collar_radius,
            args.projection_radius,
            args.central_charge_shift,
            args.structure_precision,
            args.block_working_precision,
        )
        for orders in args.orders
        for momentum_maximum in args.momentum_maximum
        for ordering, multiplicity in BOUNDARY_CORNER_RAISED_ORBITS
    ]
    records: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as pool:
        futures = tuple(pool.submit(_corner_task, spec) for spec in specs)
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    totals: list[dict[str, object]] = []
    for orders in args.orders:
        for momentum_maximum in args.momentum_maximum:
            selected = [
                record
                for record in records
                if record["orders"] == list(orders)
                and record["momentum_maximum"] == momentum_maximum
            ]
            value = sum(
                int(record["multiplicity"])
                * complex(record["value"]["real"], record["value"]["imag"])
                for record in selected
            )
            totals.append(
                {
                    "orders": list(orders),
                    "momentum_maximum": momentum_maximum,
                    "value": {"real": value.real, "imag": value.imag},
                }
            )
    payload = {
        "status": "corner_momentum_convergence_diagnostic_not_worldsheet_freeze",
        "matrix_model_comparison_performed": False,
        "epsilon": args.epsilon,
        "momentum_maximum": list(args.momentum_maximum),
        "collar_radius": args.collar_radius,
        "projection_radius": args.projection_radius,
        "momentum_panels": "automatic fixed factor-four threshold hierarchy",
        "records": sorted(
            records,
            key=lambda item: (
                item["orders"],
                item["momentum_maximum"],
                item["ordering"],
            ),
        ),
        "totals": totals,
    }
    print(json.dumps({"totals": totals}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
