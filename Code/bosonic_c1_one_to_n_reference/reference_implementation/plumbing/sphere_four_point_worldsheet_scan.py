#!/usr/bin/env python3
"""Produce and freeze a blind ``omega=i*t`` sphere ``1->3`` scan.

This executable imports only worldsheet ingredients.  Comparison functions
belong in a separate program that may run only after the JSON table and its
SHA-256 manifest have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

try:
    from sphere_four_point_imaginary_energy import (
        FIRST_RESIDUE_WALL,
        ImaginaryOneToThreeKernel,
        integrate_convergent_one_to_three_atlas_qmc,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_imaginary_energy import (
        FIRST_RESIDUE_WALL,
        ImaginaryOneToThreeKernel,
        integrate_convergent_one_to_three_atlas_qmc,
    )


DEFAULT_T_VALUES = tuple(0.16 + 0.02 * index for index in range(16))


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_point(
    t: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_panels: int,
    sobol_power: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    kernel = ImaginaryOneToThreeKernel(
        t,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_maximum=momentum_maximum,
        momentum_panels=momentum_panels,
    )
    result = integrate_convergent_one_to_three_atlas_qmc(
        kernel,
        sobol_power=sobol_power,
        replicates=replicates,
        seed=seed,
    )
    denominator = 6.0 * math.pi * kernel.omega**4
    reduced = result.mean / denominator
    reduced_error_real = result.standard_error_real / abs(denominator)
    reduced_error_imag = result.standard_error_imag / abs(denominator)
    return {
        "t": float(t),
        "omega": _complex_record(kernel.omega),
        "raw_integral_I4": _complex_record(result.mean),
        "raw_standard_error": {
            "real": result.standard_error_real,
            "imag": result.standard_error_imag,
        },
        "Q3": _complex_record(reduced),
        "Q3_standard_error": {
            "real": reduced_error_real,
            "imag": reduced_error_imag,
        },
        "boundary_radial_power": kernel.leading_boundary_radial_power,
        "proposal_radial_power": result.radial_power,
        "replicate_estimates_I4": [
            _complex_record(value) for value in result.estimates
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent / "results" / "sphere_four_point_1to3"
    parser.add_argument("--t", nargs="+", type=float, default=DEFAULT_T_VALUES)
    parser.add_argument("--block-order", type=int, default=10)
    parser.add_argument("--momentum-order", type=int, default=24)
    parser.add_argument("--momentum-maximum", type=float, default=8.0)
    parser.add_argument("--momentum-panels", type=int, default=2)
    parser.add_argument("--sobol-power", type=int, default=13)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=base / "worldsheet_scan.json")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=base / "worldsheet_scan_frozen.json",
    )
    arguments = parser.parse_args()

    t_values = tuple(float(value) for value in arguments.t)
    if len(t_values) != len(set(t_values)) or t_values != tuple(sorted(t_values)):
        parser.error("--t values must be distinct and strictly increasing")
    if any(not 0.0 < value < FIRST_RESIDUE_WALL for value in t_values):
        parser.error("all --t values must lie in the residue-free chamber 0<t<1/2")

    points = []
    for index, t in enumerate(t_values):
        point = scan_point(
            t,
            block_order=arguments.block_order,
            momentum_order=arguments.momentum_order,
            momentum_maximum=arguments.momentum_maximum,
            momentum_panels=arguments.momentum_panels,
            sobol_power=arguments.sobol_power,
            replicates=arguments.replicates,
            seed=arguments.seed + 1009 * index,
        )
        points.append(point)
        print(json.dumps(point, sort_keys=True), flush=True)

    payload = {
        "status": "worldsheet_only_frozen_before_external_comparison",
        "calculation": "direct labelled c=1 sphere 1->3 worldsheet integral",
        "matrix_model_information_used": False,
        "kinematics": "three equal outgoing energies omega and incoming energy 3 omega",
        "domain": "omega=i*t with 0<t<1/2",
        "liouville_contour": {
            "path": "each internal momentum is integrated on P>=0",
            "nearest_poles": "P_+/-=+/-(2 omega-i)",
            "first_residue_wall": FIRST_RESIDUE_WALL,
            "residue_terms_included": 0,
        },
        "moduli_prescription": (
            "raw convergent correlator integrated with an exact equal-weight "
            "three-chart mixture at z=0,1,infinity; no finite-part subtraction"
        ),
        "normalization": {
            "amplitude": "mu^2 A_tree=i I4/(2 pi)",
            "stripped_definition": "Q3=I4/(6 pi omega^4)",
        },
        "settings": {
            "block_order": arguments.block_order,
            "momentum_order_per_panel": arguments.momentum_order,
            "momentum_maximum": arguments.momentum_maximum,
            "momentum_panels": arguments.momentum_panels,
            "total_momentum_nodes": (
                arguments.momentum_order * arguments.momentum_panels
            ),
            "sobol_power": arguments.sobol_power,
            "samples_per_replicate": 2**arguments.sobol_power,
            "replicates": arguments.replicates,
            "base_seed": arguments.seed,
        },
        "points": points,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "status": "worldsheet_only_frozen_before_external_comparison",
        "artifact": str(arguments.output.resolve()),
        "sha256": _sha256(arguments.output),
        "frozen_on": date.today().isoformat(),
        "point_count": len(points),
        "matrix_model_information_used": False,
    }
    arguments.freeze_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

