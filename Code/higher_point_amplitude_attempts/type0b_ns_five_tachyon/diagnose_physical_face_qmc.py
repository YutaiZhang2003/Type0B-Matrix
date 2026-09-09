#!/usr/bin/env python3
"""Face-only nested Sobol audit for the physical five-point finite part."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from type0b_ns_five_tachyon import (
    BOUNDARY_FACE_RAISED_ORBITS,
    BRYNSFiveTachyonIntegrand,
    _four_point_fundamental_cell_sample,
)


def _replicate_task(spec: tuple[object, ...]) -> dict[str, object]:
    replicate, powers, settings = spec
    kernel = BRYNSFiveTachyonIntegrand(
        outgoing_energies=(0.25 + 1.0j * settings["epsilon"],) * 4,
        block_backend="c",
        recursion_max_twice_level=None,
        global_max_twice_levels=tuple(settings["global_max_twice_levels"]),
        global_max_total_twice_level=settings["global_max_total_twice_level"],
        momentum_orders=tuple(settings["momentum_orders"]),
        momentum_maximum=settings["momentum_maximum"],
        structure_precision=settings["structure_precision"],
        central_charge_shift=settings["central_charge_shift"],
        block_working_precision=settings["block_working_precision"],
    )
    maximum_power = max(powers)
    sampler = qmc.Sobol(
        d=2, scramble=True, seed=settings["seed"] + 10000 + replicate
    )
    samples = sampler.random_base2(maximum_power)
    values: list[complex] = []
    started = time.time()
    for sample_index, sample in enumerate(samples):
        modulus, area_jacobian = _four_point_fundamental_cell_sample(
            sample[0], sample[1]
        )
        if abs(modulus) < settings["collar_radius"]:
            value = 0.0j
        else:
            density = sum(
                multiplicity
                * kernel.boundary_face_finite_part_density(
                    ordering=ordering,
                    remaining_modulus=modulus,
                    collar_radius=settings["collar_radius"],
                    projection_radius=settings["projection_radius"],
                    momentum_refinement_shells=-1,
                    momentum_singularity_subtraction=True,
                )
                for ordering, multiplicity in BOUNDARY_FACE_RAISED_ORBITS
            )
            value = complex(area_jacobian * density)
        values.append(value)
        print(
            json.dumps(
                {
                    "event": "face_sample",
                    "replicate": replicate,
                    "sample": sample_index,
                    "value": {"real": value.real, "imag": value.imag},
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    estimates = []
    for power in powers:
        estimate = complex(np.mean(np.asarray(values[: 2**power], dtype=complex)))
        estimates.append(
            {
                "power": power,
                "value": {"real": estimate.real, "imag": estimate.imag},
            }
        )
    return {
        "replicate": replicate,
        "seconds": time.time() - started,
        "estimates": estimates,
        "backend_counts": dict(kernel._block_backend_evaluation_counts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--powers", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--momentum-orders", type=int, nargs=2, default=(5, 7))
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument("--global-max-twice-levels", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--global-max-total-twice-level", type=int, default=6)
    parser.add_argument("--collar-radius", type=float, default=0.08)
    parser.add_argument("--projection-radius", type=float, default=1.0e-5)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-5)
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--block-working-precision", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    powers = tuple(sorted(set(args.powers)))
    if not powers or min(powers) < 1:
        raise ValueError("powers must contain positive integers")
    if args.replicates < 2:
        raise ValueError("at least two replicates are required")
    settings = {
        "epsilon": args.epsilon,
        "momentum_orders": list(args.momentum_orders),
        "momentum_maximum": args.momentum_maximum,
        "global_max_twice_levels": list(args.global_max_twice_levels),
        "global_max_total_twice_level": args.global_max_total_twice_level,
        "collar_radius": args.collar_radius,
        "projection_radius": args.projection_radius,
        "central_charge_shift": args.central_charge_shift,
        "structure_precision": args.structure_precision,
        "block_working_precision": args.block_working_precision,
        "seed": args.seed,
    }
    specs = tuple((replicate, powers, settings) for replicate in range(args.replicates))
    records: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as pool:
        futures = tuple(pool.submit(_replicate_task, spec) for spec in specs)
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"event": "replicate", **record}, sort_keys=True), flush=True)
    summaries = []
    for power in powers:
        estimates = np.asarray(
            [
                complex(item["value"]["real"], item["value"]["imag"])
                for record in records
                for item in record["estimates"]
                if item["power"] == power
            ],
            dtype=complex,
        )
        mean = complex(np.mean(estimates))
        summaries.append(
            {
                "power": power,
                "samples_per_replicate": 2**power,
                "mean": {"real": mean.real, "imag": mean.imag},
                "standard_error_real": float(
                    np.std(estimates.real, ddof=1) / math.sqrt(len(estimates))
                ),
                "standard_error_imag": float(
                    np.std(estimates.imag, ddof=1) / math.sqrt(len(estimates))
                ),
            }
        )
    payload = {
        "status": "face_qmc_convergence_diagnostic_not_worldsheet_freeze",
        "matrix_model_comparison_performed": False,
        "powers": list(powers),
        "replicates": args.replicates,
        **settings,
        "records": sorted(records, key=lambda item: item["replicate"]),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summaries": summaries}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
