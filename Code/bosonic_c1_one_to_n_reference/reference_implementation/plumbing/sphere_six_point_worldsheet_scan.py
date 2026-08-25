#!/usr/bin/env python3
"""Worldsheet-only convergent-ray scan for sphere 1->5 scattering.

This driver intentionally contains no matrix-model formula.  It freezes I6,
the stripped Q5 inferred from the worldsheet normalization, and independent
QMC errors before any analytic comparison is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

try:
    from sphere_six_point_equal_energy import (
        EqualEnergySixPointKernel,
        FIRST_RESIDUE_WALL,
        integrate_convergent_equal_energy_atlas_qmc,
    )
except ImportError:  # pragma: no cover
    from plumbing.sphere_six_point_equal_energy import (
        EqualEnergySixPointKernel,
        FIRST_RESIDUE_WALL,
        integrate_convergent_equal_energy_atlas_qmc,
    )


def evaluate_point(
    t: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_power: float,
    sobol_power: int,
    replicates: int,
    radial_power: float,
    seed: int,
) -> dict[str, object]:
    start = time.perf_counter()
    kernel = EqualEnergySixPointKernel(
        t,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_maximum=momentum_maximum,
        momentum_power=momentum_power,
    )
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = integrate_convergent_equal_energy_atlas_qmc(
        kernel,
        sobol_power=sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=seed,
    )
    integration_seconds = time.perf_counter() - start

    q5 = -result.mean / (40.0 * math.pi**3 * t**6)
    q5_error_real = result.standard_error_real / (40.0 * math.pi**3 * t**6)
    q5_error_imag = result.standard_error_imag / (40.0 * math.pi**3 * t**6)
    amplitude = 1.0j * result.mean / (8.0 * math.pi**3)
    amplitude_error_real = result.standard_error_imag / (8.0 * math.pi**3)
    amplitude_error_imag = result.standard_error_real / (8.0 * math.pi**3)
    return {
        "t": t,
        "distance_to_first_residue_wall": FIRST_RESIDUE_WALL - t,
        "I6": {"real": result.mean.real, "imag": result.mean.imag},
        "I6_standard_error": {
            "real": result.standard_error_real,
            "imag": result.standard_error_imag,
        },
        "Q5_worldsheet": {"real": q5.real, "imag": q5.imag},
        "Q5_worldsheet_standard_error": {
            "real": q5_error_real,
            "imag": q5_error_imag,
        },
        "mu4_A_tree_worldsheet": {
            "real": amplitude.real,
            "imag": amplitude.imag,
        },
        "mu4_A_tree_worldsheet_standard_error": {
            "real": amplitude_error_real,
            "imag": amplitude_error_imag,
        },
        "replicate_I6": [
            {"real": value.real, "imag": value.imag} for value in result.estimates
        ],
        "channel_selection": {
            "comb_fraction": result.comb_selection_fraction,
            "star_fraction": result.star_selection_fraction,
            "maximum_selected_radius": result.maximum_selected_radius,
        },
        "block_fallback_counts": kernel.fallback_counts,
        "timing_seconds": {
            "kernel_build": build_seconds,
            "moduli_integration": integration_seconds,
        },
    }


def scan_payload(
    arguments: argparse.Namespace,
    points: list[dict[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    return {
        "status": status,
        "normalization": {
            "amplitude": "mu^4 A_tree = i I6/(8 pi^3)",
            "stripped": "Q5 = I6/(40 pi^3 omega^6) = -I6/(40 pi^3 t^6) on omega=i t",
        },
        "kinematic_domain": {
            "omega": "i t",
            "first_residue_wall": FIRST_RESIDUE_WALL,
            "all_points_below_wall": True,
        },
        "settings": {
            "block_order": arguments.block_order,
            "momentum_base_order": arguments.momentum_order,
            "momentum_edge_orders": [
                arguments.momentum_order,
                arguments.momentum_order + 1,
                arguments.momentum_order + 2,
            ],
            "momentum_maximum": arguments.momentum_maximum,
            "momentum_power": arguments.momentum_power,
            "sobol_power": arguments.sobol_power,
            "samples_per_replicate": 2**arguments.sobol_power,
            "replicates": arguments.replicates,
            "radial_power": arguments.radial_power,
            "seed": arguments.seed,
            "atlas": "equal mixture of 720 comb and 720 star plumbing charts",
        },
        "points": points,
    }


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t",
        nargs="+",
        type=float,
        default=(0.14, 0.17, 0.19, 0.199, 0.201, 0.22, 0.26, 0.30),
    )
    parser.add_argument("--block-order", type=int, default=4)
    parser.add_argument("--momentum-order", type=int, default=8)
    parser.add_argument("--momentum-maximum", type=float, default=5.0)
    parser.add_argument("--momentum-power", type=float, default=1.25)
    parser.add_argument("--sobol-power", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--radial-power", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_six_point_1to5"
        / "worldsheet_convergent_scan.json",
    )
    arguments = parser.parse_args()
    if any(not 0.0 < t < FIRST_RESIDUE_WALL for t in arguments.t):
        parser.error("every t must lie in the residue-free chamber 0<t<1/3")

    points = []
    for index, t in enumerate(arguments.t):
        print(
            f"[{index + 1}/{len(arguments.t)}] worldsheet t={t:.8g}",
            flush=True,
        )
        point = evaluate_point(
            t,
            block_order=arguments.block_order,
            momentum_order=arguments.momentum_order,
            momentum_maximum=arguments.momentum_maximum,
            momentum_power=arguments.momentum_power,
            sobol_power=arguments.sobol_power,
            replicates=arguments.replicates,
            radial_power=arguments.radial_power,
            seed=arguments.seed,
        )
        points.append(point)
        print(json.dumps(point, sort_keys=True), flush=True)
        write_payload(
            arguments.output,
            scan_payload(
                arguments,
                points,
                status="worldsheet_only_partial_no_matrix_model_imported",
            ),
        )

    payload = scan_payload(
        arguments,
        points,
        status="worldsheet_only_no_matrix_model_imported",
    )
    write_payload(arguments.output, payload)
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
