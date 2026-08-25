#!/usr/bin/env python3
"""Pointwise physical-domain sphere-five calculation with BRY subtraction.

Every requested real outgoing energy is evaluated independently at one or
more positive i-epsilon values. No fit or analytic continuation in omega is
performed, and no matrix-model expression is imported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        integrate_physical_equal_energy_finite_part_qmc,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        integrate_physical_equal_energy_finite_part_qmc,
    )


def evaluate_point(
    outgoing_energy: float,
    epsilon: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_panels: int,
    endpoint_refinement: int,
    block_scheme: str,
    collar_radius: float,
    bulk_sobol_power: int,
    face_sobol_power: int,
    replicates: int,
    radial_power: float,
    seed: int,
    kernel: EqualEnergyFivePointKernel | None = None,
) -> dict[str, object]:
    """Evaluate one real omega and one positive epsilon."""

    outgoing_energy = float(outgoing_energy)
    epsilon = float(epsilon)
    if outgoing_energy <= 0.0 or epsilon <= 0.0:
        raise ValueError("outgoing_energy and epsilon must be positive")
    # BRY convention: omega_in=4*omega+i*epsilon and each outgoing
    # frequency is omega+i*epsilon/4.
    outgoing_complex = complex(outgoing_energy, epsilon / 4.0)
    if kernel is None:
        kernel = EqualEnergyFivePointKernel(
            outgoing_complex,
            block_order=block_order,
            momentum_order=momentum_order,
            momentum_maximum=momentum_maximum,
            momentum_panels=momentum_panels,
            endpoint_refinement=endpoint_refinement,
            block_scheme=block_scheme,
        )
    elif abs(kernel.omega - outgoing_complex) > 1.0e-15:
        raise ValueError("the precomputed kernel has the wrong complex frequency")
    result = integrate_physical_equal_energy_finite_part_qmc(
        kernel,
        collar_radius=collar_radius,
        bulk_sobol_power=bulk_sobol_power,
        face_sobol_power=face_sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=seed,
    )
    xi_amplitude = 1.0j * result.mean / (4.0 * math.pi**2)
    xi_error_real = result.standard_error_imag / (4.0 * math.pi**2)
    xi_error_imag = result.standard_error_real / (4.0 * math.pi**2)
    return {
        "omega": outgoing_energy,
        "epsilon": epsilon,
        "collar_radius": collar_radius,
        "complex_outgoing_frequency": {
            "real": outgoing_complex.real,
            "imag": outgoing_complex.imag,
        },
        "complex_incoming_frequency": {
            "real": (4.0 * outgoing_complex).real,
            "imag": (4.0 * outgoing_complex).imag,
        },
        "I5": {"real": result.mean.real, "imag": result.mean.imag},
        "I5_standard_error": {
            "real": result.standard_error_real,
            "imag": result.standard_error_imag,
        },
        "mu3_A_tree": {
            "real": xi_amplitude.real,
            "imag": xi_amplitude.imag,
        },
        "mu3_A_tree_standard_error": {
            "real": xi_error_real,
            "imag": xi_error_imag,
        },
        "replicate_I5": [
            {"real": value.real, "imag": value.imag}
            for value in result.estimates
        ],
        "strata": {
            "bulk_mean": {
                "real": complex(sum(result.bulk_estimates) / len(result.bulk_estimates)).real,
                "imag": complex(sum(result.bulk_estimates) / len(result.bulk_estimates)).imag,
            },
            "faces_mean": {
                "real": complex(sum(result.face_estimates) / len(result.face_estimates)).real,
                "imag": complex(sum(result.face_estimates) / len(result.face_estimates)).imag,
            },
            "corners": {
                "real": result.corner_contribution.real,
                "imag": result.corner_contribution.imag,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", nargs="+", type=float, default=(0.35,))
    parser.add_argument("--epsilon", nargs="+", type=float, default=(0.04,))
    parser.add_argument("--block-order", type=int, default=4)
    parser.add_argument("--momentum-order", type=int, default=4)
    parser.add_argument("--momentum-maximum", type=float, default=6.0)
    parser.add_argument("--momentum-panels", type=int, default=1)
    parser.add_argument("--endpoint-refinement", type=int, default=0)
    parser.add_argument("--block-scheme", choices=("h", "c"), default="h")
    parser.add_argument("--collar-radius", nargs="+", type=float, default=(0.14, 0.10, 0.07))
    parser.add_argument("--bulk-sobol-power", type=int, default=5)
    parser.add_argument("--face-sobol-power", type=int, default=6)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--radial-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_five_point_1to4"
        / "physical_iepsilon_pilot.json",
    )
    arguments = parser.parse_args()

    points: list[dict[str, object]] = []
    for omega in arguments.omega:
        for epsilon in arguments.epsilon:
            outgoing_complex = complex(float(omega), float(epsilon) / 4.0)
            kernel = EqualEnergyFivePointKernel(
                outgoing_complex,
                block_order=arguments.block_order,
                momentum_order=arguments.momentum_order,
                momentum_maximum=arguments.momentum_maximum,
                momentum_panels=arguments.momentum_panels,
                endpoint_refinement=arguments.endpoint_refinement,
                block_scheme=arguments.block_scheme,
            )
            for collar_radius in arguments.collar_radius:
                print(
                    f"evaluating physical omega={omega:.8g}, epsilon={epsilon:.3g}, "
                    f"rho={collar_radius:.3g}",
                    flush=True,
                )
                record = evaluate_point(
                    omega,
                    epsilon,
                    block_order=arguments.block_order,
                    momentum_order=arguments.momentum_order,
                    momentum_maximum=arguments.momentum_maximum,
                    momentum_panels=arguments.momentum_panels,
                    endpoint_refinement=arguments.endpoint_refinement,
                    block_scheme=arguments.block_scheme,
                    collar_radius=collar_radius,
                    bulk_sobol_power=arguments.bulk_sobol_power,
                    face_sobol_power=arguments.face_sobol_power,
                    replicates=arguments.replicates,
                    radial_power=arguments.radial_power,
                    seed=arguments.seed,
                    kernel=kernel,
                )
                points.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    collar_audits: list[dict[str, object]] = []
    for omega in arguments.omega:
        for epsilon in arguments.epsilon:
            selected = [
                point
                for point in points
                if point["omega"] == float(omega) and point["epsilon"] == float(epsilon)
            ]
            real_values = [float(point["I5"]["real"]) for point in selected]
            imag_values = [float(point["I5"]["imag"]) for point in selected]
            real_errors = [
                float(point["I5_standard_error"]["real"]) for point in selected
            ]
            imag_errors = [
                float(point["I5_standard_error"]["imag"]) for point in selected
            ]
            real_spread = max(real_values) - min(real_values)
            imag_spread = max(imag_values) - min(imag_values)
            collar_audits.append(
                {
                    "omega": float(omega),
                    "epsilon": float(epsilon),
                    "real_spread": real_spread,
                    "imag_spread": imag_spread,
                    "two_sigma_stable": (
                        real_spread <= 2.0 * max(real_errors)
                        and imag_spread <= 2.0 * max(imag_errors)
                    ),
                }
            )

    payload = {
        "status": "physical_i_epsilon_worldsheet_unfrozen_requires_convergence",
        "normalization": "mu^3 A_tree=i I5/(4 pi^2)",
        "subtraction": "local finite part: excised bulk + 10 analytic faces + 15 double-primitive corners",
        "settings": {
            "block_order": arguments.block_order,
            "momentum_order_per_endpoint_panel": arguments.momentum_order,
            "momentum_maximum": arguments.momentum_maximum,
            "momentum_panels": arguments.momentum_panels,
            "endpoint_refinement": arguments.endpoint_refinement,
            "block_scheme": arguments.block_scheme,
            "collar_radii": arguments.collar_radius,
            "bulk_sobol_power": arguments.bulk_sobol_power,
            "face_sobol_power": arguments.face_sobol_power,
            "replicates": arguments.replicates,
            "radial_power": arguments.radial_power,
            "seed": arguments.seed,
        },
        "collar_audits": collar_audits,
        "promotion_ready": False,
        "remaining_promotion_gates": [
            "block-order convergence",
            "Liouville momentum-grid convergence at every power endpoint",
            "moduli-grid convergence",
            "three-radius collar stability",
            "epsilon-to-zero sequence at fixed real omega",
        ],
        "points": points,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
