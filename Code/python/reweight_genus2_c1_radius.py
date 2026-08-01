#!/usr/bin/env python3
"""Reweight the refined genus-two ``c=1`` pilot across compactification radii.

At fixed period matrix the only radius-dependent factor in the implemented
matter--ghost density is the compact-boson zero mode and winding lattice,

    Z_R / Z_1 = R Theta_R(Omega) / Theta_1(Omega).

The expensive Liouville and plumbing calculations can therefore be reused
node by node.  This script preserves all correlations between radii and uses
a paired delete-one jackknife for the normalization-free radius-shape test.

The historical output field labelled ``conditional_strict_bry`` now uses the
string-note normalized, ``g_s^2``-stripped target

    (F_2/g_s^2)^target(R) = 16 pi^2 f_2(R).

The legacy field name is retained for compatibility with saved pilot analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from genus2_c1_string_integrand import compact_boson_winding_sum_genus2  # noqa: E402
from genus2_integrand_normalization import (  # noqa: E402
    GENUS2_GENERIC_STACK_WEIGHT,
    LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
    PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    c1_genus2_topology_correction,
    c1_sphere_normalized_genus2_kernel_multiplier,
    string_note_integration_kernel_target,
    xi_full_replacement_over_dimensionless,
)
from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2  # noqa: E402


DEFAULT_INPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "pilot_R1_N64_refined/refined_samples.csv"
)
DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "radius_reweight_N64_refined"
)
TWO_TO_TWELVE = float(2**12)


@dataclass(frozen=True)
class RefinedNode:
    sample_index: int
    omega: np.ndarray
    saved_radius_one_winding_sum: float
    radius_one_transformed_integrand: float


@dataclass(frozen=True)
class RadiusResult:
    radius: float
    local_moduli_integral: float
    monte_carlo_standard_error: float
    normalized_worldsheet_shape: float
    normalized_worldsheet_shape_jackknife_se: float
    matrix_model_coefficient_f2: float
    conditional_strict_bry_target: float
    normalized_matrix_model_shape: float
    conditional_target_over_local_integral: float
    conditional_target_over_local_integral_standard_error: float
    conditional_factor_over_2pow12: float
    normalized_mismatch: float
    normalized_mismatch_jackknife_se: float
    normalized_mismatch_z_from_one: float
    contribution_effective_sample_size: float
    largest_contribution_fraction: float


def c1_matrix_model_genus2_coefficient(radius: float) -> float:
    r"""Return ``f_2(R)`` in ``F_2=f_2(R) mu^-2``."""

    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    return (7.0 * radius**2 + 10.0 + 7.0 / radius**2) / (5760.0 * radius)


def conditional_strict_bry_target(radius: float) -> float:
    r"""Compatibility alias for the string-note kernel target."""

    return string_note_integration_kernel_target(radius)


def period_matrix_from_row(row: dict[str, str]) -> np.ndarray:
    """Reconstruct the symmetric period matrix saved in a refined CSV row."""

    omega = np.array(
        [
            [
                float(row["x11"]) + 1j * float(row["y11"]),
                float(row["x12"]) + 1j * float(row["y12"]),
            ],
            [
                float(row["x12"]) + 1j * float(row["y12"]),
                float(row["x22"]) + 1j * float(row["y22"]),
            ],
        ],
        dtype=np.complex128,
    )
    if np.min(np.linalg.eigvalsh(omega.imag)) <= 0.0:
        raise ValueError(f"sample {row.get('sample_index')} is outside Siegel space")
    return omega


def load_refined_nodes(path: Path) -> list[RefinedNode]:
    """Load successful refined nodes without dropping or refitting any sample."""

    nodes: list[RefinedNode] = []
    for row in csv.DictReader(path.open()):
        if row.get("status") != "ok":
            raise ValueError(f"sample {row.get('sample_index')} is not successful")
        winding = float(row["compact_winding_sum"])
        value = float(row["transformed_integrand_final"])
        convention = row.get("integration_kernel_convention", "")
        if convention == "":
            value *= (
                c1_sphere_normalized_genus2_kernel_multiplier()
                * xi_full_replacement_over_dimensionless()
            )
        elif convention == LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION:
            value *= (
                xi_full_replacement_over_dimensionless()
                * c1_genus2_topology_correction()
            )
        elif convention == PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
            value *= c1_genus2_topology_correction()
        elif convention != STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
            raise ValueError(
                f"sample {row['sample_index']} has unsupported kernel convention "
                f"{convention!r}"
            )
        if not (math.isfinite(winding) and winding > 0.0):
            raise ValueError(f"sample {row['sample_index']} has an invalid winding sum")
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"sample {row['sample_index']} has an invalid integrand")
        nodes.append(
            RefinedNode(
                sample_index=int(row["sample_index"]),
                omega=period_matrix_from_row(row),
                saved_radius_one_winding_sum=winding,
                radius_one_transformed_integrand=value,
            )
        )
    if len(nodes) < 2:
        raise ValueError("need at least two successful refined nodes")
    indices = [node.sample_index for node in nodes]
    if len(indices) != len(set(indices)):
        raise ValueError("sample indices are not unique")
    return nodes


def parse_radii(text: str) -> list[float]:
    radii = sorted({float(piece.strip()) for piece in text.split(",") if piece.strip()})
    if not radii:
        raise ValueError("need at least one radius")
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("radii must be positive and finite")
    if not any(math.isclose(radius, 1.0, rel_tol=0.0, abs_tol=1.0e-13) for radius in radii):
        radii.append(1.0)
        radii.sort()
    return radii


def logarithmic_reciprocal_radii(radius_max: float, count: int) -> list[float]:
    """Return a log-uniform grid closed under ``R -> 1/R``."""

    radius_max = float(radius_max)
    if not math.isfinite(radius_max) or radius_max <= 1.0:
        raise ValueError("radius_max must be finite and greater than one")
    if count < 3 or count % 2 == 0:
        raise ValueError("radius_count must be an odd integer at least three")
    return [float(value) for value in np.geomspace(1.0 / radius_max, radius_max, count)]


def _sample_mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a one-dimensional sample of size at least two")
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def _jackknife_standard_error(values: np.ndarray) -> float:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("jackknife values must be one-dimensional")
    center = float(np.mean(values))
    return float(math.sqrt((values.size - 1.0) / values.size * np.sum((values - center) ** 2)))


def paired_shape_jackknife(
    radius_values: np.ndarray,
    radius_one_values: np.ndarray,
    *,
    target_shape: float,
) -> tuple[float, float, float, float]:
    """Return worldsheet shape, its SE, target/worldsheet shape, and its SE."""

    if radius_values.shape != radius_one_values.shape or radius_values.ndim != 1:
        raise ValueError("paired samples must be one-dimensional and have equal shape")
    count = radius_values.size
    if count < 2:
        raise ValueError("need at least two paired samples")
    total_radius = float(np.sum(radius_values))
    total_one = float(np.sum(radius_one_values))
    worldsheet_shape = total_radius / total_one
    leave_one_shape = (total_radius - radius_values) / (total_one - radius_one_values)
    leave_one_mismatch = float(target_shape) / leave_one_shape
    mismatch = float(target_shape) / worldsheet_shape
    return (
        float(worldsheet_shape),
        _jackknife_standard_error(leave_one_shape),
        float(mismatch),
        _jackknife_standard_error(leave_one_mismatch),
    )


def _effective_sample_size(values: np.ndarray) -> float:
    total = float(np.sum(values))
    return total * total / float(np.sum(values * values))


def reweighted_node_values(
    nodes: Sequence[RefinedNode],
    radius: float,
    radius_one_winding_sums: np.ndarray,
    *,
    lattice_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed node values and newly evaluated winding sums."""

    winding_sums = np.array(
        [
            compact_boson_winding_sum_genus2(
                node.omega,
                radius,
                tolerance=lattice_tolerance,
            )
            for node in nodes
        ],
        dtype=np.float64,
    )
    base_values = np.array(
        [node.radius_one_transformed_integrand for node in nodes], dtype=np.float64
    )
    factors = float(radius) * winding_sums / radius_one_winding_sums
    return base_values * factors, winding_sums


def evaluate_radius_sweep(
    nodes: Sequence[RefinedNode],
    radii: Sequence[float],
    *,
    lattice_tolerance: float,
) -> tuple[list[RadiusResult], dict[str, object], dict[float, np.ndarray]]:
    """Evaluate all radii and return rows, diagnostics, and nodewise values."""

    radii = sorted(float(radius) for radius in radii)
    if not any(math.isclose(radius, 1.0, abs_tol=1.0e-13) for radius in radii):
        raise ValueError("the sweep must contain R=1")
    prefactor = float(GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2)
    recomputed_one = np.array(
        [
            compact_boson_winding_sum_genus2(
                node.omega,
                1.0,
                tolerance=lattice_tolerance,
            )
            for node in nodes
        ],
        dtype=np.float64,
    )
    saved_one = np.array(
        [node.saved_radius_one_winding_sum for node in nodes], dtype=np.float64
    )
    saved_relative_error = np.abs(recomputed_one / saved_one - 1.0)
    base_values = np.array(
        [node.radius_one_transformed_integrand for node in nodes], dtype=np.float64
    )

    values_by_radius: dict[float, np.ndarray] = {}
    winding_by_radius: dict[float, np.ndarray] = {1.0: recomputed_one}
    for radius in radii:
        if math.isclose(radius, 1.0, rel_tol=0.0, abs_tol=1.0e-13):
            values_by_radius[radius] = base_values.copy()
            winding_by_radius[radius] = recomputed_one
        else:
            values, winding = reweighted_node_values(
                nodes,
                radius,
                recomputed_one,
                lattice_tolerance=lattice_tolerance,
            )
            values_by_radius[radius] = values
            winding_by_radius[radius] = winding

    radius_one_key = min(radii, key=lambda radius: abs(radius - 1.0))
    one_values = values_by_radius[radius_one_key]
    one_mean, _one_node_se = _sample_mean_and_standard_error(one_values)
    one_integral = prefactor * one_mean
    one_target = conditional_strict_bry_target(1.0)

    results: list[RadiusResult] = []
    for radius in radii:
        node_values = values_by_radius[radius]
        node_mean, node_se = _sample_mean_and_standard_error(node_values)
        integral = prefactor * node_mean
        integral_se = prefactor * node_se
        target = conditional_strict_bry_target(radius)
        target_shape = target / one_target
        worldsheet_shape, shape_se, mismatch, mismatch_se = paired_shape_jackknife(
            node_values,
            one_values,
            target_shape=target_shape,
        )
        factor = target / integral
        factor_se = factor * integral_se / integral
        mismatch_z = 0.0 if mismatch_se == 0.0 else (mismatch - 1.0) / mismatch_se
        results.append(
            RadiusResult(
                radius=radius,
                local_moduli_integral=integral,
                monte_carlo_standard_error=integral_se,
                normalized_worldsheet_shape=worldsheet_shape,
                normalized_worldsheet_shape_jackknife_se=shape_se,
                matrix_model_coefficient_f2=c1_matrix_model_genus2_coefficient(radius),
                conditional_strict_bry_target=target,
                normalized_matrix_model_shape=target_shape,
                conditional_target_over_local_integral=factor,
                conditional_target_over_local_integral_standard_error=factor_se,
                conditional_factor_over_2pow12=factor / TWO_TO_TWELVE,
                normalized_mismatch=mismatch,
                normalized_mismatch_jackknife_se=mismatch_se,
                normalized_mismatch_z_from_one=mismatch_z,
                contribution_effective_sample_size=_effective_sample_size(node_values),
                largest_contribution_fraction=float(np.max(node_values) / np.sum(node_values)),
            )
        )

    duality_residuals: list[float] = []
    nodewise_duality_residuals: list[float] = []
    for radius in radii:
        inverse = min(radii, key=lambda candidate: abs(candidate - 1.0 / radius))
        if not math.isclose(inverse, 1.0 / radius, rel_tol=2.0e-13, abs_tol=2.0e-13):
            continue
        left = values_by_radius[radius]
        right = values_by_radius[inverse] / radius**2
        nodewise_duality_residuals.append(float(np.max(np.abs(left / right - 1.0))))
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        duality_residuals.append(abs(left_mean / right_mean - 1.0))

    diagnostics: dict[str, object] = {
        "sample_count": len(nodes),
        "radius_count": len(radii),
        "integration_prefactor": prefactor,
        "radius_one_local_moduli_integral": one_integral,
        "radius_one_conditional_strict_bry_target": one_target,
        "radius_one_conditional_target_over_local_integral": one_target / one_integral,
        "radius_one_factor_over_2pow12": one_target / one_integral / TWO_TO_TWELVE,
        "maximum_recomputed_vs_saved_radius_one_winding_relative_error": float(
            np.max(saved_relative_error)
        ),
        "maximum_integrated_t_duality_relative_residual": max(duality_residuals, default=0.0),
        "maximum_nodewise_t_duality_relative_residual": max(
            nodewise_duality_residuals, default=0.0
        ),
        "minimum_normalized_mismatch": min(row.normalized_mismatch for row in results),
        "maximum_normalized_mismatch": max(row.normalized_mismatch for row in results),
        "minimum_contribution_effective_sample_size": min(
            row.contribution_effective_sample_size for row in results
        ),
        "maximum_largest_contribution_fraction": max(
            row.largest_contribution_fraction for row in results
        ),
    }
    if len(nodes) % 2 == 0:
        midpoint = len(nodes) // 2
        split_rows: list[dict[str, float]] = []
        for radius, result in zip(radii, results):
            target_shape = result.normalized_matrix_model_shape
            first = paired_shape_jackknife(
                values_by_radius[radius][:midpoint],
                one_values[:midpoint],
                target_shape=target_shape,
            )
            second = paired_shape_jackknife(
                values_by_radius[radius][midpoint:],
                one_values[midpoint:],
                target_shape=target_shape,
            )
            split_rows.append(
                {
                    "radius": radius,
                    "first_half_normalized_mismatch": first[2],
                    "first_half_jackknife_se": first[3],
                    "second_half_normalized_mismatch": second[2],
                    "second_half_jackknife_se": second[3],
                }
            )
        diagnostics["split_half_shape_check"] = split_rows
    return results, diagnostics, values_by_radius


def write_csv(path: Path, results: Sequence[RadiusResult]) -> None:
    rows = [asdict(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, results: Sequence[RadiusResult]) -> None:
    """Write the radius-shape and normalization diagnostics as one figure."""

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "stringmc-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(results, key=lambda row: row.radius)
    radii = np.array([row.radius for row in ordered])
    x_values = np.log2(radii)
    worldsheet_shape = np.array([row.normalized_worldsheet_shape for row in ordered])
    worldsheet_shape_se = np.array(
        [row.normalized_worldsheet_shape_jackknife_se for row in ordered]
    )
    target_shape = np.array([row.normalized_matrix_model_shape for row in ordered])
    factor_scaled = np.array([row.conditional_factor_over_2pow12 for row in ordered])
    factor_scaled_se = np.array(
        [
            row.conditional_target_over_local_integral_standard_error / TWO_TO_TWELVE
            for row in ordered
        ]
    )
    mismatch = np.array([row.normalized_mismatch for row in ordered])
    mismatch_se = np.array([row.normalized_mismatch_jackknife_se for row in ordered])

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfa",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#d9d9d6",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 11.5), sharex=True)
    teal = "#16717c"
    rust = "#a34a34"
    ink = "#34383d"

    axes[0].plot(x_values, target_shape, color=rust, linewidth=2.3, label="matrix-model shape")
    axes[0].errorbar(
        x_values,
        worldsheet_shape,
        yerr=worldsheet_shape_se,
        color=teal,
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        capsize=2.5,
        label="reweighted genus-two integral",
    )
    axes[0].set_ylabel(r"value normalized at $R=1$")
    axes[0].set_title("Radius dependence: worldsheet pilot versus matrix-model shape")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].axhline(1.0, color=rust, linewidth=2.0, linestyle="--", label=r"exactly $2^{12}$")
    axes[1].errorbar(
        x_values,
        factor_scaled,
        yerr=factor_scaled_se,
        color=ink,
        marker="o",
        markersize=4.2,
        linewidth=1.7,
        capsize=2.5,
        label=r"conditional target / worldsheet / $2^{12}$",
    )
    axes[1].set_ylabel(r"absolute factor / $2^{12}$")
    axes[1].set_title("Absolute comparison (ordinary Monte Carlo errors)")
    axes[1].legend(frameon=False)

    axes[2].axhline(1.0, color=rust, linewidth=2.0, linestyle="--", label="one constant at all radii")
    axes[2].errorbar(
        x_values,
        mismatch,
        yerr=mismatch_se,
        color=teal,
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        capsize=2.5,
        label="target shape / worldsheet shape",
    )
    axes[2].set_ylabel("normalization-free mismatch")
    axes[2].set_title("Paired radius-shape test")
    axes[2].legend(frameon=False)

    tick_radii = np.array([0.5, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5, 2.0])
    tick_radii = tick_radii[(tick_radii >= radii.min()) & (tick_radii <= radii.max())]
    axes[2].set_xticks(np.log2(tick_radii), [f"{radius:.3g}" for radius in tick_radii])
    axes[2].set_xlabel(r"compactification radius $R$ (logarithmic axis)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(float(x_values.min()), float(x_values.max()))
    figure.suptitle(
        "Genus-two c=1 radius reweighting of the same 64 period matrices",
        fontsize=16,
        fontweight="semibold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _selected_rows(results: Sequence[RadiusResult]) -> list[RadiusResult]:
    targets = (0.5, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5, 2.0)
    selected: list[RadiusResult] = []
    for target in targets:
        row = min(results, key=lambda item: abs(math.log(item.radius / target)))
        if row not in selected:
            selected.append(row)
    return selected


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reweight the refined genus-two pilot in R.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--radii",
        help="comma-separated radii; R=1 is inserted if absent",
    )
    parser.add_argument("--radius-max", type=float, default=2.0)
    parser.add_argument("--radius-count", type=int, default=25)
    parser.add_argument("--lattice-tolerance", type=float, default=1.0e-13)
    args = parser.parse_args(list(argv) if argv is not None else None)

    radii = (
        parse_radii(args.radii)
        if args.radii
        else logarithmic_reciprocal_radii(args.radius_max, args.radius_count)
    )
    nodes = load_refined_nodes(args.input_csv)
    results, diagnostics, _values = evaluate_radius_sweep(
        nodes,
        radii,
        lattice_tolerance=args.lattice_tolerance,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "radius_sweep.csv"
    json_path = args.out_dir / "summary.json"
    png_path = args.out_dir / "radius_dependence.png"
    pdf_path = args.out_dir / "radius_dependence.pdf"
    write_csv(csv_path, results)
    payload = {
        "scope": (
            "Nodewise compact-boson reweighting of the same refined genus-two "
            "period-matrix sample; no Liouville block or normalization is refitted."
        ),
        "input_csv": str(args.input_csv),
        "lattice_tolerance": args.lattice_tolerance,
        "absolute_target_status": (
            "string-note convention: F2/g_s^2=16*pi^2*f2(R)"
        ),
        "shape_test_status": (
            "independent of every radius-independent overall normalization, including "
            "the radius-independent string-note multiplier"
        ),
        "matrix_model_coefficient": (
            "f2(R)=(7*R^2+10+7/R^2)/(5760*R)"
        ),
        "reweighting_identity": "g_i(R)=g_i(1)*R*Theta_R(Omega_i)/Theta_1(Omega_i)",
        "diagnostics": diagnostics,
        "rows": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_plot(png_path, results)
    write_plot(pdf_path, results)

    print("Genus-two c=1 compact-radius reweighting")
    print(f"  samples={len(nodes)}, radii={len(results)}")
    print(
        "  R=1 conditional target/code="
        f"{diagnostics['radius_one_conditional_target_over_local_integral']:.9g} "
        f"= {diagnostics['radius_one_factor_over_2pow12']:.9g} * 2^12"
    )
    print("  selected radius diagnostics:")
    print("    R          J2_local       target/code    normalized mismatch")
    for row in _selected_rows(results):
        print(
            f"    {row.radius:<10.6g} {row.local_moduli_integral:<14.7g} "
            f"{row.conditional_target_over_local_integral:<14.7g} "
            f"{row.normalized_mismatch:.7g} +/- "
            f"{row.normalized_mismatch_jackknife_se:.2g}"
        )
    print(
        "  max nodewise T-duality residual="
        f"{diagnostics['maximum_nodewise_t_duality_relative_residual']:.3e}"
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    print(f"  wrote {png_path}")
    print(f"  wrote {pdf_path}")


if __name__ == "__main__":
    run()
