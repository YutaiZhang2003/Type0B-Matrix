#!/usr/bin/env python3
"""Worldsheet-only numerical-systematics audit for sphere 1->5.

The audit deliberately contains no matrix-model expression.  It compares
adjacent Liouville-momentum rules, block truncations, momentum cutoffs, and
radial importance samplers at a representative point in the residue-free
chamber.
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


CONFIGURATIONS = (
    {
        "name": "production_proxy",
        "block_order": 4,
        "momentum_order": 8,
        "momentum_maximum": 5.0,
        "radial_power": 0.20,
    },
    {
        "name": "momentum_order_minus_one",
        "block_order": 4,
        "momentum_order": 7,
        "momentum_maximum": 5.0,
        "radial_power": 0.20,
    },
    {
        "name": "block_order_plus_two",
        "block_order": 6,
        "momentum_order": 7,
        "momentum_maximum": 5.0,
        "radial_power": 0.20,
    },
    {
        "name": "momentum_cutoff_lower",
        "block_order": 4,
        "momentum_order": 7,
        "momentum_maximum": 4.0,
        "radial_power": 0.20,
    },
    {
        "name": "radial_sampler_alternate",
        "block_order": 4,
        "momentum_order": 7,
        "momentum_maximum": 5.0,
        "radial_power": 0.30,
    },
)


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=0.18)
    parser.add_argument("--sobol-power", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--momentum-power", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_six_point_1to5"
        / "worldsheet_numerical_audit.json",
    )
    arguments = parser.parse_args()
    if not 0.0 < arguments.t < FIRST_RESIDUE_WALL:
        parser.error("t must lie in the residue-free chamber 0<t<1/3")

    kernels: dict[tuple[int, int, float], EqualEnergySixPointKernel] = {}
    evaluations: list[dict[str, object]] = []
    for index, configuration in enumerate(CONFIGURATIONS):
        print(
            f"[{index + 1}/{len(CONFIGURATIONS)}] {configuration['name']}",
            flush=True,
        )
        kernel_key = (
            int(configuration["block_order"]),
            int(configuration["momentum_order"]),
            float(configuration["momentum_maximum"]),
        )
        build_start = time.perf_counter()
        if kernel_key not in kernels:
            kernels[kernel_key] = EqualEnergySixPointKernel(
                arguments.t,
                block_order=kernel_key[0],
                momentum_order=kernel_key[1],
                momentum_maximum=kernel_key[2],
                momentum_power=arguments.momentum_power,
            )
        build_seconds = time.perf_counter() - build_start
        kernel = kernels[kernel_key]

        integration_start = time.perf_counter()
        result = integrate_convergent_equal_energy_atlas_qmc(
            kernel,
            sobol_power=arguments.sobol_power,
            replicates=arguments.replicates,
            radial_power=float(configuration["radial_power"]),
            seed=arguments.seed,
        )
        integration_seconds = time.perf_counter() - integration_start
        denominator = 40.0 * math.pi**3 * arguments.t**6
        q5 = -result.mean / denominator
        q5_standard_error = result.standard_error_real / denominator
        evaluation = {
            **configuration,
            "momentum_edge_orders": [
                int(configuration["momentum_order"]),
                int(configuration["momentum_order"]) + 1,
                int(configuration["momentum_order"]) + 2,
            ],
            "I6": complex_pair(result.mean),
            "I6_standard_error_real": result.standard_error_real,
            "Q5_worldsheet": complex_pair(q5),
            "Q5_worldsheet_standard_error_real": q5_standard_error,
            "replicate_I6": [complex_pair(value) for value in result.estimates],
            "maximum_selected_radius": result.maximum_selected_radius,
            "comb_selection_fraction": result.comb_selection_fraction,
            "block_fallback_counts": kernel.fallback_counts,
            "timing_seconds": {
                "kernel_build": build_seconds,
                "moduli_integration": integration_seconds,
            },
        }
        evaluations.append(evaluation)
        print(json.dumps(evaluation, sort_keys=True), flush=True)

    by_name = {str(item["name"]): item for item in evaluations}
    production = by_name["production_proxy"]
    lower_order = by_name["momentum_order_minus_one"]
    higher_block = by_name["block_order_plus_two"]
    lower_cutoff = by_name["momentum_cutoff_lower"]
    radial_alternate = by_name["radial_sampler_alternate"]

    def q5_real(item: dict[str, object]) -> float:
        pair = item["Q5_worldsheet"]
        assert isinstance(pair, dict)
        return float(pair["real"])

    diagnostics = {
        "adjacent_momentum_order_shift_Q5": abs(
            q5_real(production) - q5_real(lower_order)
        ),
        "block_order_4_to_6_shift_Q5_at_momentum_order_7": abs(
            q5_real(higher_block) - q5_real(lower_order)
        ),
        "momentum_cutoff_4_to_5_shift_Q5_at_momentum_order_7": abs(
            q5_real(lower_cutoff) - q5_real(lower_order)
        ),
        "radial_sampler_shift_Q5_at_momentum_order_7": abs(
            q5_real(radial_alternate) - q5_real(lower_order)
        ),
    }
    deterministic_terms = (
        diagnostics["adjacent_momentum_order_shift_Q5"],
        diagnostics["block_order_4_to_6_shift_Q5_at_momentum_order_7"],
        diagnostics["momentum_cutoff_4_to_5_shift_Q5_at_momentum_order_7"],
    )
    diagnostics["combined_discretization_Q5"] = math.sqrt(
        sum(float(value) ** 2 for value in deterministic_terms)
    )

    payload = {
        "status": "worldsheet_only_no_matrix_model_imported",
        "t": arguments.t,
        "first_residue_wall": FIRST_RESIDUE_WALL,
        "samples_per_replicate": 2**arguments.sobol_power,
        "replicates": arguments.replicates,
        "momentum_power": arguments.momentum_power,
        "seed": arguments.seed,
        "evaluations": evaluations,
        "diagnostics": diagnostics,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
