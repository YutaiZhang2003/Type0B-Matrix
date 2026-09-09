#!/usr/bin/env python3
"""Fixed-modulus momentum convergence audit for one physical boundary face."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    _four_point_fundamental_cell_sample,
)


def _order_pair(value: str) -> tuple[int, int]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("an order pair must have the form N1,N2")
    result = tuple(int(field) for field in fields)
    if min(result) < 2:
        raise argparse.ArgumentTypeError("both momentum orders must be at least two")
    return result  # type: ignore[return-value]


def _task(spec: tuple[object, ...]) -> dict[str, object]:
    orders, ordering, modulus, settings = spec
    kernel = BRYNSFiveTachyonIntegrand(
        outgoing_energies=(0.25 + 1.0j * settings["epsilon"],) * 4,
        block_backend="c",
        recursion_max_twice_level=None,
        global_max_twice_levels=(4, 4),
        global_max_total_twice_level=6,
        momentum_orders=orders,
        momentum_maximum=settings["momentum_maximum"],
        structure_precision=settings["structure_precision"],
        central_charge_shift=settings["central_charge_shift"],
        block_working_precision=settings["block_working_precision"],
    )
    started = time.time()
    value = kernel.boundary_face_finite_part_density(
        ordering=ordering,
        remaining_modulus=modulus,
        collar_radius=settings["collar_radius"],
        projection_radius=settings["projection_radius"],
        momentum_refinement_shells=-1,
        momentum_singularity_subtraction=True,
    )
    return {
        "orders": list(orders),
        "ordering": list(ordering),
        "remaining_modulus": {"real": modulus.real, "imag": modulus.imag},
        "value": {"real": value.real, "imag": value.imag},
        "seconds": time.time() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=_order_pair, nargs="+", required=True)
    parser.add_argument("--ordering", type=int, nargs=5, default=(0, 1, 2, 3, 4))
    parser.add_argument("--sample-uv", type=float, nargs=2, default=(0.37, 0.61))
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument("--collar-radius", type=float, default=0.08)
    parser.add_argument("--projection-radius", type=float, default=1.0e-5)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-5)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-working-precision", type=int, default=45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    ordering = tuple(args.ordering)
    if set(ordering) != set(range(5)):
        raise ValueError("ordering must permute labels 0,...,4")
    modulus, area_jacobian = _four_point_fundamental_cell_sample(*args.sample_uv)
    settings = {
        "epsilon": args.epsilon,
        "momentum_maximum": args.momentum_maximum,
        "collar_radius": args.collar_radius,
        "projection_radius": args.projection_radius,
        "central_charge_shift": args.central_charge_shift,
        "structure_precision": args.structure_precision,
        "block_working_precision": args.block_working_precision,
    }
    specs = tuple((orders, ordering, modulus, settings) for orders in args.orders)
    records: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as pool:
        futures = tuple(pool.submit(_task, spec) for spec in specs)
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    payload = {
        "status": "face_momentum_convergence_diagnostic_not_worldsheet_freeze",
        "matrix_model_comparison_performed": False,
        "sample_uv": list(args.sample_uv),
        "remaining_modulus": {"real": modulus.real, "imag": modulus.imag},
        "area_jacobian": area_jacobian,
        **settings,
        "records": sorted(records, key=lambda item: item["orders"]),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
