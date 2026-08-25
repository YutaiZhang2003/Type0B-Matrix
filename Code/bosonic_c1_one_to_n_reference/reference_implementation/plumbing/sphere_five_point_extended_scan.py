#!/usr/bin/env python3
"""Extend the convergent-ray sphere-five scan to larger imaginary energies.

This driver deliberately uses the same kernel and 120-chart plumbing mixture
as the original equal-energy calculation.  It only accepts omega=i*t with
positive real t.  The raw integral is converted to

Q(omega) = I_5 / (16*pi^2*omega^5).

The output is worldsheet-only.  No matrix-model function is imported here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        integrate_convergent_equal_energy_atlas_qmc,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        integrate_convergent_equal_energy_atlas_qmc,
    )


def scan_point(
    t_value: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_panels: int,
    momentum_power: float,
    block_scheme: str,
    liouville_contour: str,
    sobol_power: int,
    replicates: int,
    radial_power: float,
    seed: int,
) -> dict[str, object]:
    """Evaluate one convergent point and return a JSON-ready record."""

    t_value = float(t_value)
    if not math.isfinite(t_value) or t_value <= 0.0:
        raise ValueError("each t value must be positive and finite")
    omega = 1.0j * t_value
    kernel = EqualEnergyFivePointKernel(
        omega,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_maximum=momentum_maximum,
        momentum_panels=momentum_panels,
        momentum_power=momentum_power,
        block_scheme=block_scheme,
        liouville_contour=liouville_contour,
    )
    result = integrate_convergent_equal_energy_atlas_qmc(
        kernel,
        sobol_power=sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=seed,
    )
    denominator = 16.0 * math.pi**2 * omega**5
    reduced = result.mean / denominator
    denominator_magnitude = abs(denominator)
    # Division by i*positive swaps the raw real/imaginary uncertainties.
    reduced_standard_error_real = (
        result.standard_error_imag / denominator_magnitude
    )
    reduced_standard_error_imag = (
        result.standard_error_real / denominator_magnitude
    )
    return {
        "t": t_value,
        "omega": {"real": omega.real, "imag": omega.imag},
        "raw_integral": {"real": result.mean.real, "imag": result.mean.imag},
        "raw_standard_error": {
            "real": result.standard_error_real,
            "imag": result.standard_error_imag,
        },
        "Q": {"real": reduced.real, "imag": reduced.imag},
        "Q_standard_error": {
            "real": reduced_standard_error_real,
            "imag": reduced_standard_error_imag,
        },
        "replicate_estimates": [
            {"real": value.real, "imag": value.imag}
            for value in result.estimates
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=(0.38, 0.42, 0.46, 0.48),
        help="positive imaginary-energy coordinates, omega=i*t",
    )
    parser.add_argument("--block-order", type=int, default=6)
    parser.add_argument("--momentum-order", type=int, default=16)
    parser.add_argument("--momentum-maximum", type=float, default=6.0)
    parser.add_argument("--momentum-panels", type=int, default=1)
    parser.add_argument(
        "--momentum-power",
        type=float,
        default=1.25,
        help=(
            "power in P=Pmax*u^power; 5/4 improves threshold resolution "
            "without placing c-recursion nodes too close to P=0"
        ),
    )
    parser.add_argument("--block-scheme", choices=("h", "c"), default="c")
    parser.add_argument(
        "--liouville-contour",
        choices=("real", "continued"),
        default="continued",
        help=(
            "real uses P>=0; continued adds the crossed-pole residue needed "
            "across the t=2/5 DOZZ endpoint pinch"
        ),
    )
    parser.add_argument("--sobol-power", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--radial-power", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_five_point_1to4"
        / "worldsheet_extended_pilot.json",
    )
    arguments = parser.parse_args()

    records: list[dict[str, object]] = []
    for t_value in arguments.t:
        print(f"evaluating omega=i*{t_value:.6g}", flush=True)
        try:
            record = scan_point(
                t_value,
                block_order=arguments.block_order,
                momentum_order=arguments.momentum_order,
                momentum_maximum=arguments.momentum_maximum,
                momentum_panels=arguments.momentum_panels,
                momentum_power=arguments.momentum_power,
                block_scheme=arguments.block_scheme,
                liouville_contour=arguments.liouville_contour,
                sobol_power=arguments.sobol_power,
                replicates=arguments.replicates,
                radial_power=arguments.radial_power,
                seed=arguments.seed,
            )
        except Exception as error:  # preserve successful points in a long scan
            record = {
                "t": float(t_value),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "description": (
            "Worldsheet-only extension on omega=i*t; no matrix-model values "
            "were imported or evaluated."
        ),
        "settings": {
            "block_order": arguments.block_order,
            "momentum_order": arguments.momentum_order,
            "momentum_maximum": arguments.momentum_maximum,
            "momentum_panels": arguments.momentum_panels,
            "momentum_power": arguments.momentum_power,
            "block_scheme": arguments.block_scheme,
            "liouville_contour": arguments.liouville_contour,
            "continuation": (
                "real P>=0 double integral plus -2i times the residue at "
                "P=(5/2)omega-i on an incoming-outgoing cherry edge"
            ),
            "validity_range": "0<t<1/2; a second DOZZ pole family pinches at t=1/2",
            "sobol_power": arguments.sobol_power,
            "replicates": arguments.replicates,
            "radial_power": arguments.radial_power,
            "seed": arguments.seed,
        },
        "points": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
