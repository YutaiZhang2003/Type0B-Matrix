#!/usr/bin/env python3
"""Export the compact, post-moduli-integration genus-two free-energy release."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from genus2_integrand_normalization import (
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        integration_kernel_scale_to_current,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus2_integrand_normalization import (
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        integration_kernel_scale_to_current,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256/"
    "final_stable_cutoff_v4/direct_low_radius_reweight_v1/"
    "merged_radius_sweep_R05_R2_39/radius_sweep_R05_R2_39_direct.csv"
)
DEFAULT_PROVENANCE = DEFAULT_INPUT.with_name("radius_provenance.csv")
DEFAULT_DESIGN_SUMMARY = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256/summary.json"
)
DEFAULT_OUTPUT = ROOT / "output/data/genus2_c1_free_energy_direct_39"

OUTPUT_FIELDS = (
    "radius",
    "integration_kernel_convention",
    "coarse_domain_integral_K2_c1",
    "coarse_domain_integral_standard_error",
    "connected_logZ_genus2_over_gs_squared",
    "connected_logZ_standard_error",
    "thermal_free_energy_over_gs_squared",
    "thermal_free_energy_standard_error",
    "normalized_shape",
    "normalized_shape_jackknife_standard_error",
    "contribution_effective_sample_size",
    "largest_node_fraction",
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _release_rows(
    rows: list[dict[str, str]],
    *,
    source_kernel_convention: str,
) -> list[dict[str, object]]:
    release: list[dict[str, object]] = []
    radii = [float(row["radius"]) for row in rows]
    if radii != sorted(radii) or len(set(radii)) != len(radii):
        raise ValueError("radius rows must be strictly increasing and unique")
    if len(rows) != 39 or not math.isclose(radii[0], 0.5) or not math.isclose(radii[-1], 2.0):
        raise ValueError("expected the finalized 39-point radius grid on [0.5, 2]")

    normalization_scale = integration_kernel_scale_to_current(
        source_kernel_convention
    )
    for row in rows:
        connected_logz = (
            normalization_scale * float(row["free_energy_over_gs_squared"])
        )
        connected_logz_standard_error = (
            normalization_scale * float(row["rqmc_scramble_standard_error"])
        )
        radius = float(row["radius"])
        coarse_integral = 2.0 * connected_logz
        coarse_integral_standard_error = 2.0 * connected_logz_standard_error
        circumference = 2.0 * math.pi * radius
        release.append(
            {
                "radius": row["radius"],
                "integration_kernel_convention": (
                    STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
                ),
                "coarse_domain_integral_K2_c1": coarse_integral,
                "coarse_domain_integral_standard_error": coarse_integral_standard_error,
                "connected_logZ_genus2_over_gs_squared": connected_logz,
                "connected_logZ_standard_error": connected_logz_standard_error,
                "thermal_free_energy_over_gs_squared": -connected_logz / circumference,
                "thermal_free_energy_standard_error": (
                    connected_logz_standard_error / circumference
                ),
                "normalized_shape": row["normalized_worldsheet_shape"],
                "normalized_shape_jackknife_standard_error": row[
                    "normalized_worldsheet_shape_jackknife_se"
                ],
                "contribution_effective_sample_size": row[
                    "contribution_effective_sample_size"
                ],
                "largest_node_fraction": row["largest_node_fraction"],
            }
        )
    return release


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input-kernel-convention",
        default=None,
        help=(
            "explicit convention of free_energy_over_gs_squared in --input; "
            "required for custom CSVs that do not carry a convention column"
        ),
    )
    args = parser.parse_args()

    source_rows = _load_csv(args.input)
    provenance_rows = _load_csv(args.provenance)
    embedded_conventions = {
        str(row.get("integration_kernel_convention", "")).strip()
        for row in source_rows
        if str(row.get("integration_kernel_convention", "")).strip()
    }
    if len(embedded_conventions) > 1:
        raise ValueError("input rows mix integration-kernel conventions")
    if embedded_conventions:
        embedded_convention = embedded_conventions.pop()
        if (
            args.input_kernel_convention is not None
            and args.input_kernel_convention != embedded_convention
        ):
            raise ValueError(
                "--input-kernel-convention disagrees with the CSV convention"
            )
        source_kernel_convention = embedded_convention
    elif args.input_kernel_convention is not None:
        source_kernel_convention = args.input_kernel_convention
    elif args.input.resolve() == DEFAULT_INPUT.resolve():
        source_kernel_convention = (
            LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION
        )
    else:
        raise ValueError(
            "custom input CSV has no integration_kernel_convention column; "
            "pass --input-kernel-convention explicitly"
        )
    release_rows = _release_rows(
        source_rows,
        source_kernel_convention=source_kernel_convention,
    )
    if [row["radius"] for row in source_rows] != [row["radius"] for row in provenance_rows]:
        raise ValueError("radius data and provenance grids do not agree")

    design_summary = json.loads(args.design_summary.read_text(encoding="utf-8"))
    radius_one = next(
        row for row in release_rows if math.isclose(float(row["radius"]), 1.0)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "radius_sweep.csv"
    provenance_path = args.output_dir / "radius_provenance.csv"
    summary_path = args.output_dir / "summary.json"
    _write_csv(data_path, release_rows, OUTPUT_FIELDS)
    _write_csv(provenance_path, provenance_rows, ("radius", "source"))

    summary = {
        "scope": "Compact final data after genus-two moduli integration; no sampling-level or CFT-node data are included.",
        "radius_count": len(release_rows),
        "radius_interval": [float(release_rows[0]["radius"]), float(release_rows[-1]["radius"])],
        "all_radius_values_directly_evaluated": True,
        "rqmc": {
            "sampling_scheme": design_summary["sampling_scheme"],
            "replicate_count": design_summary["replicate_count"],
            "integrated_cft_node_count": design_summary["domain_cft_node_count"],
        },
        "self_dual_radius": {
            "radius": 1.0,
            "coarse_domain_integral_K2_c1": radius_one[
                "coarse_domain_integral_K2_c1"
            ],
            "connected_logZ_genus2_over_gs_squared": radius_one[
                "connected_logZ_genus2_over_gs_squared"
            ],
            "thermal_free_energy_over_gs_squared": radius_one[
                "thermal_free_energy_over_gs_squared"
            ],
            "coarse_domain_integral_standard_error": radius_one[
                "coarse_domain_integral_standard_error"
            ],
            "connected_logZ_standard_error": radius_one[
                "connected_logZ_standard_error"
            ],
            "thermal_free_energy_standard_error": radius_one[
                "thermal_free_energy_standard_error"
            ],
        },
        "normalization": {
            "monte_carlo_measure_normalization_included": True,
            "generic_genus_two_stack_weight": 0.5,
            "source_integration_kernel_convention": source_kernel_convention,
            "source_to_current_kernel_scale": integration_kernel_scale_to_current(
                source_kernel_convention
            ),
            "integration_kernel_convention": (
                STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
            ),
            "connected_functional_relation": (
                "mathcal_F_2^str/g_s^2=(1/2)*integral_coarse K_2^c1"
            ),
            "thermal_free_energy_relation_alpha_prime_1": (
                "F_2^therm/g_s^2=-(mathcal_F_2^str/g_s^2)/(2*pi*R)"
            ),
            "external_worldsheet_to_matrix_model_normalization_applied": False,
            "normalized_matrix_model_shape": "(7 R + 10/R + 7/R^3)/24",
        },
        "invariant_moduli_volume_control": {
            "exact": design_summary["exact_invariant_volume_control"],
            "estimate": design_summary["invariant_volume_control_estimate"],
            "standard_error": design_summary[
                "invariant_volume_control_standard_error"
            ],
            "z_score": design_summary["invariant_volume_control_z_score"],
        },
        "files": {
            "data": data_path.name,
            "provenance": provenance_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
