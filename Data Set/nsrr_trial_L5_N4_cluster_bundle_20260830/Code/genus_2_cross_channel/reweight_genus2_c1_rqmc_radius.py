#!/usr/bin/env python3
"""Paired compact-radius sweep on complete genus-two RQMC CFT data.

The expensive Liouville, ghost, and noncompact-scalar factors are evaluated
once at ``R=1``.  At every saved period matrix the only radius-dependent
factor is then replaced exactly,

    g_i(R) = g_i(1) R Theta_R(Omega_i) / Theta_1(Omega_i).

Independent complete Sobol scrambles, rather than individual nodes, determine
the error on both the integral and the normalization-free radius shape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from genus2_hybrid_period_map import (
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        is_schottky_algorithm,
    )
    from genus2_c1_string_integrand import (
        COMPACT_THETA_IMPLEMENTATION,
        compact_boson_winding_evaluation_genus2,
    )
    from genus2_integrand_normalization import (
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    )
    from genus2_moduli_rqmc import estimate_rqmc_integral
    from genus2_plumbing_atlas import symplectic_matrix_from_csv_row
    from genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        estimate_physical_mixture_integral,
    )
    from monte_carlo_integrate_genus2_c1 import (
        RQMC_SAMPLING_SCHEMES,
        canonicalize_string_note_kernel_row,
        omega_from_csv_row,
        sampling_scheme_for_rows,
    )
    from reweight_genus2_c1_radius import logarithmic_reciprocal_radii, parse_radii
except ImportError:  # pragma: no cover
    from plumbing.genus2_hybrid_period_map import (
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        is_schottky_algorithm,
    )
    from plumbing.genus2_c1_string_integrand import (
        COMPACT_THETA_IMPLEMENTATION,
        compact_boson_winding_evaluation_genus2,
    )
    from plumbing.genus2_integrand_normalization import (
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    )
    from plumbing.genus2_moduli_rqmc import estimate_rqmc_integral
    from plumbing.genus2_plumbing_atlas import symplectic_matrix_from_csv_row
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        estimate_physical_mixture_integral,
    )
    from plumbing.monte_carlo_integrate_genus2_c1 import (
        RQMC_SAMPLING_SCHEMES,
        canonicalize_string_note_kernel_row,
        omega_from_csv_row,
        sampling_scheme_for_rows,
    )
    from plumbing.reweight_genus2_c1_radius import logarithmic_reciprocal_radii, parse_radii


DEFAULT_INPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_holomorphic_R8_M16_b8q8/assembled/combined_samples.csv"
)
DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_holomorphic_R8_M16_b8q8/radius_sweep"
)

# The difficult period-map recovery was explicitly promoted with this common
# production ceiling.  Original rows retain their earlier requested 1e-6 in
# the CSV, so final assembly must audit the actual certificates against the
# promoted common bar rather than pretend the heterogeneous requests are the
# final acceptance policy.
PRODUCTION_PERIOD_CERTIFICATE_CEILING = 1.0e-5


@dataclass(frozen=True)
class RadiusShapeResult:
    radius: float
    integration_kernel_convention: str
    free_energy_over_gs_squared: float
    rqmc_scramble_standard_error: float
    volume_calibrated_free_energy_over_gs_squared: float
    volume_calibrated_scramble_standard_error: float
    normalized_worldsheet_shape: float
    normalized_worldsheet_shape_jackknife_se: float
    contribution_effective_sample_size: float
    largest_node_fraction: float


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _period_certificate_follows_routing_policy(row: dict[str, str]) -> bool:
    """Validate the final period certificate independently of its atlas seed.

    ``atlas_period_algorithm`` records how the table supplied the initial
    plumbing chart.  ``period_algorithm`` records the subsequent fixed-q
    certificate used by the CFT evaluation.  In the two-method overlap these
    may intentionally differ, so requiring the same solver family for both
    discards a useful cross-method check.  The final certificate is the
    authoritative object: all solvers must meet the residual/stability bar,
    and a Schottky certificate additionally needs an all-small/overlap region
    and a converged word-tail bound.
    """

    holomorphic_algorithms = {
        "holomorphic-form-collocation",
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
    }
    supported_algorithms = holomorphic_algorithms | {
        "adaptive-schottky",
        "calibrated-schottky",
    }
    atlas_algorithm = row.get("atlas_period_algorithm", "")
    certificate_algorithm = row.get("period_algorithm", "")
    if (
        atlas_algorithm not in supported_algorithms
        or certificate_algorithm not in supported_algorithms
    ):
        return False

    try:
        requested_tolerance = float(row["period_validation_tolerance"])
        residual = float(row["period_final_residual"])
        map_step = float(row["period_final_map_step"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not all(
            math.isfinite(value)
            for value in (requested_tolerance, residual, map_step)
        )
        or requested_tolerance <= 0.0
        or requested_tolerance > PRODUCTION_PERIOD_CERTIFICATE_CEILING
        or residual > PRODUCTION_PERIOD_CERTIFICATE_CEILING
        or map_step > PRODUCTION_PERIOD_CERTIFICATE_CEILING
    ):
        return False

    if certificate_algorithm in holomorphic_algorithms:
        return True
    if not is_schottky_algorithm(certificate_algorithm):
        return False
    if row.get("period_map_region", "") not in {
        "schottky-all-small",
        "two-method-overlap",
    }:
        return False
    try:
        certified_bound = float(row["period_certified_error_bound"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(certified_bound)
        and certified_bound >= 0.0
        and certified_bound <= PRODUCTION_PERIOD_CERTIFICATE_CEILING
    )


def validate_production_rows(
    rows: Sequence[dict[str, str]],
    *,
    require_holomorphic_period_map: bool = True,
) -> dict[str, object]:
    """Require complete scrambles, one fixed CFT order, and certified periods."""

    if not rows:
        raise ValueError("the RQMC input is empty")
    if any(row.get("status") != "ok" for row in rows):
        failed = [row.get("rqmc_node_id", "?") for row in rows if row.get("status") != "ok"]
        raise ValueError(f"RQMC input contains failed or missing nodes: {failed[:8]}")
    sampling_scheme = sampling_scheme_for_rows(rows)
    if sampling_scheme not in RQMC_SAMPLING_SCHEMES:
        raise ValueError("the input is not a homogeneous RQMC design")
    conventions = {row.get("integration_kernel_convention", "") for row in rows}
    if conventions != {STRING_NOTE_INTEGRATION_KERNEL_CONVENTION}:
        raise ValueError(
            "RQMC rows must use the string-note integration kernel; reassemble "
            f"legacy rows first. Got {sorted(conventions)!r}"
        )
    if {row.get("igusa_measure_frame", "") for row in rows} != {
        "siegel-fundamental-domain"
    }:
        raise ValueError("RQMC rows do not explicitly lock the Igusa factor to the fundamental domain")
    if {row.get("compact_theta_frame", "") for row in rows} != {
        "siegel-fundamental-domain"
    }:
        raise ValueError("RQMC rows do not lock the compact theta sum to the fundamental domain")
    if any(
        not row.get("liouville_scalar_quotient_frame", "").startswith("plumbing:")
        for row in rows
    ):
        raise ValueError("RQMC rows do not declare a plumbing frame for Z_L/Z_X^25")
    if any(symplectic_matrix_from_csv_row(row) is None for row in rows):
        raise ValueError("RQMC rows lack the exact Sp(4,Z) marking needed for radius reweighting")

    groups: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(int(row["rqmc_replicate"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("at least two complete independent scrambles are required")
    for replicate, group in groups.items():
        expected = int(group[0]["rqmc_domain_count"])
        if len(group) != expected:
            raise ValueError(
                f"replicate {replicate} is incomplete: {len(group)} of {expected} nodes"
            )

    required_order_fields = (
        "block_order_low",
        "block_order_high",
        "quadrature_order_low",
        "quadrature_order_high",
        "scalar_word_low",
        "scalar_word_high",
    )
    if any(any(row.get(key, "") == "" for key in required_order_fields) for row in rows):
        raise ValueError("input predates the auditable fixed-order row schema")
    orders = {
        tuple(int(row[key]) for key in required_order_fields)
        for row in rows
    }
    if len(orders) != 1:
        raise ValueError(f"the sample mixes CFT truncations: {sorted(orders)}")
    order = next(iter(orders))
    if order[0] != order[1] or order[2] != order[3] or order[4] != order[5]:
        raise ValueError(f"production input is not evaluated at one fixed order: {order}")
    if any(not _as_bool(row.get("fixed_cft_order", "")) for row in rows):
        raise ValueError("one or more rows are not marked as fixed-order evaluations")

    period_algorithms = {row.get("period_algorithm", "") for row in rows}
    atlas_algorithms = {row.get("atlas_period_algorithm", "") for row in rows}
    certified_schottky_count = 0
    mixed_period_provenance_count = 0
    if require_holomorphic_period_map:
        for row in rows:
            atlas_algorithm = row.get("atlas_period_algorithm", "")
            certificate_algorithm = row.get("period_algorithm", "")
            if not _period_certificate_follows_routing_policy(row):
                raise ValueError(
                    "every final period certificate must follow the shared routing "
                    "policy and meet its numerical bar: a holomorphic certificate, "
                    "or a Schottky certificate with a converged word-tail bound in "
                    "the all-small/overlap region. Got "
                    f"atlas={atlas_algorithm!r}, certificate={certificate_algorithm!r}, "
                    f"node={row.get('rqmc_node_id', '?')}"
                )
            if is_schottky_algorithm(certificate_algorithm):
                certified_schottky_count += 1
            if is_schottky_algorithm(atlas_algorithm) != is_schottky_algorithm(
                certificate_algorithm
            ):
                mixed_period_provenance_count += 1

    return {
        "sampling_scheme": sampling_scheme,
        "replicate_count": len(groups),
        "node_count": len(rows),
        "block_order": order[0],
        "quadrature_order": order[2],
        "scalar_word_length": order[4],
        "atlas_period_algorithms": sorted(atlas_algorithms),
        "certificate_period_algorithms": sorted(period_algorithms),
        "unsupported_period_map_node_count": 0,
        "certified_schottky_node_count": certified_schottky_count,
        "mixed_atlas_certificate_algorithm_count": mixed_period_provenance_count,
        "period_certificate_ceiling": PRODUCTION_PERIOD_CERTIFICATE_CEILING,
        # Compatibility key for older summary readers.
        "calibrated_schottky_node_count": certified_schottky_count,
        "maximum_period_residual": max(float(row["period_final_residual"]) for row in rows),
        "maximum_period_map_step": max(float(row["period_final_map_step"]) for row in rows),
        "maximum_q": max(float(row["q_max"]) for row in rows),
    }


def paired_shape_from_replicates(
    radius_replicates: Sequence[float],
    radius_one_replicates: Sequence[float],
) -> tuple[float, float]:
    """Return the ratio of RQMC means and delete-one-scramble uncertainties."""

    radius_values = np.asarray(radius_replicates, dtype=float)
    one_values = np.asarray(radius_one_replicates, dtype=float)
    if radius_values.shape != one_values.shape or radius_values.ndim != 1:
        raise ValueError("paired replicate estimates must be one-dimensional and aligned")
    if radius_values.size < 2 or np.any(radius_values <= 0.0) or np.any(one_values <= 0.0):
        raise ValueError("need at least two positive paired replicate estimates")
    sum_radius = float(np.sum(radius_values))
    sum_one = float(np.sum(one_values))
    worldsheet_shape = sum_radius / sum_one
    leave_one_shape = (sum_radius - radius_values) / (sum_one - one_values)

    def jackknife_se(values: np.ndarray) -> float:
        center = float(np.mean(values))
        count = values.size
        return float(math.sqrt((count - 1.0) / count * np.sum((values - center) ** 2)))

    worldsheet_se = jackknife_se(leave_one_shape)
    return worldsheet_shape, worldsheet_se


def _effective_sample_size(values: np.ndarray) -> float:
    total = float(np.sum(values))
    return total * total / float(np.sum(values * values))


def _estimate_view(
    rows: Sequence[dict[str, str]],
    values: np.ndarray,
    sampling_scheme: str,
) -> dict[str, object]:
    """Normalize the two RQMC estimator result schemas for a radius sweep."""

    if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        result = estimate_physical_mixture_integral(rows, values)
        return {
            "estimate": result.estimate,
            "standard_error": result.scramble_standard_error,
            "replicate_estimates": result.replicate_estimates,
            "volume_calibrated_estimate": math.nan,
            "volume_calibrated_standard_error": math.nan,
        }
    result = estimate_rqmc_integral(rows, values)
    return {
        "estimate": result.raw_estimate,
        "standard_error": result.raw_scramble_standard_error,
        "replicate_estimates": result.raw_replicate_estimates,
        "volume_calibrated_estimate": result.volume_calibrated_estimate,
        "volume_calibrated_standard_error": (
            result.volume_calibrated_scramble_standard_error
        ),
    }


def evaluate_radius_sweep(
    rows: Sequence[dict[str, str]],
    radii: Sequence[float],
    *,
    lattice_tolerance: float,
) -> tuple[list[RadiusShapeResult], dict[str, object]]:
    rows = tuple(
        canonicalize_string_note_kernel_row(dict(row))
        for row in rows
    )
    certificate = validate_production_rows(rows)
    sampling_scheme = str(certificate["sampling_scheme"])
    radii = sorted(float(radius) for radius in radii)
    if not any(math.isclose(radius, 1.0, abs_tol=1.0e-13) for radius in radii):
        raise ValueError("the radius grid must contain R=1")

    omegas = [omega_from_csv_row(row) for row in rows]
    base_values = np.asarray(
        [float(row["transformed_integrand_high"]) for row in rows], dtype=float
    )
    saved_one = np.asarray([float(row["compact_winding_sum"]) for row in rows], dtype=float)
    if np.any(base_values <= 0.0) or np.any(saved_one <= 0.0):
        raise ValueError("saved integrands and winding sums must be positive")

    values_by_radius: dict[float, np.ndarray] = {}
    estimates: dict[float, object] = {}
    compact_theta_diagnostics: dict[str, object] = {}
    direct_radii = [radius for radius in radii if radius >= 1.0 - 1.0e-13]
    for radius in direct_radii:
        if math.isclose(radius, 1.0, abs_tol=1.0e-13):
            values = base_values.copy()
            saved_algorithms = [
                row.get("compact_theta_algorithm", "saved-legacy") for row in rows
            ]
            compact_theta_diagnostics[f"{radius:.16g}"] = {
                "source": "saved-radius-one-values",
                "algorithm_counts": {
                    name: saved_algorithms.count(name) for name in sorted(set(saved_algorithms))
                },
            }
        else:
            evaluations = [
                compact_boson_winding_evaluation_genus2(
                    omega,
                    radius,
                    tolerance=lattice_tolerance,
                )
                for omega in omegas
            ]
            winding = np.asarray([evaluation.value for evaluation in evaluations], dtype=float)
            values = base_values * (radius * winding / saved_one)
            algorithms = [evaluation.algorithm for evaluation in evaluations]
            compact_theta_diagnostics[f"{radius:.16g}"] = {
                "source": "evaluated-at-fundamental-period-matrix",
                "algorithm_counts": {
                    name: algorithms.count(name) for name in sorted(set(algorithms))
                },
                "maximum_momentum_nmax": max(
                    evaluation.momentum_nmax for evaluation in evaluations
                ),
                "maximum_winding_nmax": max(
                    evaluation.winding_nmax for evaluation in evaluations
                ),
                "maximum_estimated_term_count": max(
                    evaluation.estimated_term_count for evaluation in evaluations
                ),
            }
        values_by_radius[radius] = values
        estimates[radius] = _estimate_view(rows, values, sampling_scheme)

    for radius in (value for value in radii if value < 1.0 - 1.0e-13):
        inverse = min(direct_radii, key=lambda value: abs(value - 1.0 / radius))
        if math.isclose(inverse, 1.0 / radius, rel_tol=2.0e-13, abs_tol=2.0e-13):
            values = values_by_radius[inverse] / radius**2
            compact_theta_diagnostics[f"{radius:.16g}"] = {
                "source": "exact-t-duality",
                "inverse_radius": inverse,
            }
        else:
            evaluations = [
                compact_boson_winding_evaluation_genus2(
                    omega,
                    radius,
                    tolerance=lattice_tolerance,
                )
                for omega in omegas
            ]
            winding = np.asarray([evaluation.value for evaluation in evaluations], dtype=float)
            values = base_values * (radius * winding / saved_one)
            algorithms = [evaluation.algorithm for evaluation in evaluations]
            compact_theta_diagnostics[f"{radius:.16g}"] = {
                "source": "evaluated-at-fundamental-period-matrix",
                "algorithm_counts": {
                    name: algorithms.count(name) for name in sorted(set(algorithms))
                },
                "maximum_momentum_nmax": max(
                    evaluation.momentum_nmax for evaluation in evaluations
                ),
                "maximum_winding_nmax": max(
                    evaluation.winding_nmax for evaluation in evaluations
                ),
                "maximum_estimated_term_count": max(
                    evaluation.estimated_term_count for evaluation in evaluations
                ),
            }
        values_by_radius[radius] = values
        estimates[radius] = _estimate_view(rows, values, sampling_scheme)

    one_radius = min(radii, key=lambda value: abs(value - 1.0))
    one_estimate = estimates[one_radius]
    results: list[RadiusShapeResult] = []
    for radius in radii:
        estimate = estimates[radius]
        shape = paired_shape_from_replicates(
            estimate["replicate_estimates"],
            one_estimate["replicate_estimates"],
        )
        values = values_by_radius[radius]
        weighted_contributions = values * np.asarray(
            [float(row["rqmc_stack_integration_weight"]) for row in rows],
            dtype=float,
        )
        results.append(
            RadiusShapeResult(
                radius=radius,
                integration_kernel_convention=(
                    STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
                ),
                free_energy_over_gs_squared=float(estimate["estimate"]),
                rqmc_scramble_standard_error=float(estimate["standard_error"]),
                volume_calibrated_free_energy_over_gs_squared=(
                    float(estimate["volume_calibrated_estimate"])
                ),
                volume_calibrated_scramble_standard_error=(
                    float(estimate["volume_calibrated_standard_error"])
                ),
                normalized_worldsheet_shape=shape[0],
                normalized_worldsheet_shape_jackknife_se=shape[1],
                contribution_effective_sample_size=_effective_sample_size(
                    weighted_contributions
                ),
                largest_node_fraction=float(
                    np.max(weighted_contributions) / np.sum(weighted_contributions)
                ),
            )
        )

    nodewise_duality: list[float] = []
    integrated_duality: list[float] = []
    for radius in radii:
        inverse = min(radii, key=lambda value: abs(value - 1.0 / radius))
        if not math.isclose(inverse, 1.0 / radius, rel_tol=2.0e-13, abs_tol=2.0e-13):
            continue
        left = values_by_radius[radius]
        right = values_by_radius[inverse] / radius**2
        nodewise_duality.append(float(np.max(np.abs(left / right - 1.0))))
        integrated_duality.append(
            abs(
                float(estimates[radius]["estimate"])
                / (float(estimates[inverse]["estimate"]) / radius**2)
                - 1.0
            )
        )

    diagnostics = {
        **certificate,
        "radius_count": len(radii),
        "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
        "compact_theta_tolerance": lattice_tolerance,
        "compact_theta_by_radius": compact_theta_diagnostics,
        "radius_one_winding_source": "saved by the fixed-order R=1 integrand evaluation",
        "maximum_nodewise_t_duality_relative_residual": max(nodewise_duality, default=0.0),
        "maximum_integrated_t_duality_relative_residual": max(
            integrated_duality, default=0.0
        ),
        "external_comparison_target": None,
        "minimum_effective_sample_size": min(
            row.contribution_effective_sample_size for row in results
        ),
        "maximum_largest_node_fraction": max(row.largest_node_fraction for row in results),
    }
    return results, diagnostics


def _write_csv(path: Path, rows: Sequence[RadiusShapeResult]) -> None:
    payload = [asdict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def _write_plot(path: Path, rows: Sequence[RadiusShapeResult]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "stringmc-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row.radius)
    radii = np.asarray([row.radius for row in ordered])
    x = np.log2(radii)
    worldsheet = np.asarray([row.normalized_worldsheet_shape for row in ordered])
    worldsheet_se = np.asarray(
        [row.normalized_worldsheet_shape_jackknife_se for row in ordered]
    )
    free_energy = np.asarray([row.free_energy_over_gs_squared for row in ordered])
    free_energy_se = np.asarray([row.rqmc_scramble_standard_error for row in ordered])

    figure, axes = plt.subplots(2, 1, figsize=(10.8, 8.4), sharex=True)
    axes[0].errorbar(
        x,
        free_energy,
        yerr=free_energy_se,
        color="#16717c",
        marker="o",
        markersize=4.2,
        capsize=2.5,
        linewidth=1.6,
        label="genus-two RQMC",
    )
    axes[0].set_ylabel(r"$F_2(R)/g_s^2$")
    axes[0].set_title("Genus-two free energy")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        x,
        worldsheet,
        yerr=worldsheet_se,
        color="#16717c",
        marker="o",
        markersize=4.2,
        capsize=2.5,
        linewidth=1.6,
    )
    axes[1].set_ylabel(r"$F_2(R)/F_2(1)$")
    axes[1].set_xlabel("compactification radius R (log2 scale)")
    axes[1].set_title("Normalization-independent radius shape")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    ticks = np.asarray([0.5, 2.0 / 3.0, 1.0, 1.5, 2.0])
    ticks = ticks[(ticks >= radii.min()) & (ticks <= radii.max())]
    axes[1].set_xticks(np.log2(ticks), [f"{value:.3g}" for value in ticks])
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radii", help="comma-separated radii; R=1 is inserted")
    parser.add_argument("--radius-max", type=float, default=2.0)
    parser.add_argument("--radius-count", type=int, default=17)
    parser.add_argument("--lattice-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    radii = (
        parse_radii(args.radii)
        if args.radii
        else logarithmic_reciprocal_radii(args.radius_max, args.radius_count)
    )
    rows = list(csv.DictReader(args.input_csv.open()))
    results, diagnostics = evaluate_radius_sweep(
        rows,
        radii,
        lattice_tolerance=args.lattice_tolerance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "radius_sweep.csv"
    json_path = args.out_dir / "summary.json"
    png_path = args.out_dir / "radius_dependence.png"
    svg_path = args.out_dir / "radius_dependence.svg"
    _write_csv(csv_path, results)
    json_path.write_text(
        json.dumps(
            {
                "scope": (
                    "String-note normalized genus-two free energy divided by g_s^2, "
                    "plus its paired radius shape on complete RQMC scrambles with "
                    "certified hybrid period-map plumbing coordinates."
                ),
                "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
                "input_csv": str(args.input_csv),
                "external_comparison_target": None,
                "reweighting_identity": (
                    "g_i(R)=g_i(1) R Theta_R(Omega_i)/Theta_1(Omega_i)"
                ),
                "diagnostics": diagnostics,
                "rows": [asdict(row) for row in results],
            },
            indent=2,
        )
        + "\n"
    )
    if not args.skip_plot:
        _write_plot(png_path, results)
        _write_plot(svg_path, results)

    print("Genus-two c=1 RQMC radius sweep")
    print(
        f"  nodes={diagnostics['node_count']}, scrambles={diagnostics['replicate_count']}, "
        f"fixed order={diagnostics['block_order']}/{diagnostics['quadrature_order']}"
    )
    for selected_radius in (0.5, 1.0, 2.0):
        row = min(results, key=lambda item: abs(item.radius - selected_radius))
        print(
            f"  R={row.radius:.6g}: F2/g_s^2={row.free_energy_over_gs_squared:.7g} "
            f"+/- {row.rqmc_scramble_standard_error:.2g}, "
            f"shape={row.normalized_worldsheet_shape:.7g} +/- "
            f"{row.normalized_worldsheet_shape_jackknife_se:.2g}"
        )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    run()
