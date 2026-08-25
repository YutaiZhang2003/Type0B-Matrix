#!/usr/bin/env python3
"""Compare a frozen sphere 1->4 imaginary-ray fit with the matrix model.

The upstream fit is produced by ``sphere_five_point_imaginary_ray_fit.py``,
which contains no matrix-model coefficients.  This downstream program checks
the frozen source hash, evaluates both curves on omega=i*t, and produces the
comparison artifact and figure.  The separate direct physical-i-epsilon
finite-part program is intentionally outside this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from sphere_five_point_imaginary_ray_fit import FIT_STATUS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MATRIX_COEFFICIENTS_IN_T = np.asarray([2.0, -12.0, 16.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduced_amplitude(t: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + coefficients[1] * t + coefficients[2] * t**2


def normalized_amplitude(t: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    return -4.0 * t**5 * reduced


def _plot_error(point: dict[str, Any]) -> float:
    qmc = float(point["Q_standard_error"])
    if "Q_block_order_4_6_8" not in point:
        return qmc
    values = np.asarray(point["Q_block_order_4_6_8"], dtype=float)
    block_spread = float(np.max(np.abs(values - float(point["Q"]))))
    return float(np.hypot(qmc, block_spread))


def _draw_plot(
    frozen: dict[str, Any],
    fit_coefficients: np.ndarray,
    figure_path: Path,
) -> None:
    primary = frozen["primary_points"]
    diagnostics = frozen["diagnostic_points_excluded_from_primary_fit"]
    grid = np.linspace(min(point["t"] for point in primary) - 0.01, 0.50, 500)
    q_fit = reduced_amplitude(grid, fit_coefficients)
    q_matrix = reduced_amplitude(grid, MATRIX_COEFFICIENTS_IN_T)

    figure, axes = plt.subplots(2, 1, figsize=(10.4, 8.3), sharex=True)
    axes[0].plot(
        grid,
        normalized_amplitude(grid, q_matrix),
        color="#2b83ba",
        linewidth=2.0,
        label="matrix model",
    )
    axes[0].plot(
        grid,
        normalized_amplitude(grid, q_fit),
        color="#222222",
        linewidth=1.7,
        linestyle="--",
        label="quadratic fit to imaginary-energy data",
    )
    axes[1].plot(grid, q_matrix, color="#2b83ba", linewidth=2.0)
    axes[1].plot(grid, q_fit, color="#222222", linewidth=1.7, linestyle="--")

    categories = [
        (
            "real contour",
            lambda point: point["contour"] == "real",
            "#f28e2b",
            "o",
        ),
        (
            "continued contour; residue inactive",
            lambda point: point["contour"] == "continued"
            and point.get("residue_status", "").startswith("inactive"),
            "#9467bd",
            "s",
        ),
        (
            "residue corrected",
            lambda point: point.get("residue_status") == "included",
            "#e76f9a",
            "D",
        ),
    ]
    for label, selector, color, marker in categories:
        selected = [point for point in primary if selector(point)]
        if not selected:
            continue
        t = np.asarray([point["t"] for point in selected], dtype=float)
        q = np.asarray([point["Q"] for point in selected], dtype=float)
        sigma_q = np.asarray([_plot_error(point) for point in selected], dtype=float)
        axes[0].errorbar(
            t,
            normalized_amplitude(t, q),
            yerr=4.0 * t**5 * sigma_q,
            fmt=marker,
            color=color,
            markersize=5.0,
            capsize=2.2,
            linewidth=1.0,
            label=label,
            zorder=5,
        )
        axes[1].errorbar(
            t,
            q,
            yerr=sigma_q,
            fmt=marker,
            color=color,
            markersize=5.0,
            capsize=2.2,
            linewidth=1.0,
            zorder=5,
        )

    for point in diagnostics:
        t = float(point["t"])
        q = float(point["Q"])
        sigma_q = _plot_error(point)
        axes[0].errorbar(
            [t],
            [normalized_amplitude(np.asarray([t]), np.asarray([q]))[0]],
            yerr=[4.0 * t**5 * sigma_q],
            fmt="D",
            markerfacecolor="none",
            markeredgecolor="#e76f9a",
            ecolor="#e76f9a",
            capsize=2.2,
            label="near-second-wall diagnostic (excluded from fit)",
            zorder=5,
        )
        axes[1].errorbar(
            [t],
            [q],
            yerr=[sigma_q],
            fmt="D",
            markerfacecolor="none",
            markeredgecolor="#e76f9a",
            ecolor="#e76f9a",
            capsize=2.2,
            zorder=5,
        )

    for axis in axes:
        axis.axvline(0.4, color="#e76f9a", linestyle=":", linewidth=1.1)
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.grid(alpha=0.20)
    axes[0].set_ylabel(r"$\mu^3\mathcal{A}_{1\to4}^{\mathrm{tree}}(it)$")
    axes[1].set_ylabel(r"$Q_4(it)$")
    axes[1].set_xlabel(r"$t$ in $\omega=it$")
    axes[0].set_title(
        r"Sphere $1\to4$: frozen imaginary-energy fit and later comparison"
    )
    axes[0].legend(loc="best", fontsize=8.3, ncol=2)
    axes[0].text(
        0.402,
        0.96,
        "first residue wall",
        transform=axes[0].get_xaxis_transform(),
        color="#aa3d68",
        fontsize=8,
        va="top",
    )
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def compare(frozen_path: Path, output_path: Path, figure_path: Path) -> dict[str, Any]:
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("status") != FIT_STATUS:
        raise ValueError(
            "comparison requires the frozen target-free imaginary-ray fit; "
            "old convergent-only fits and physical-i-epsilon smoke tests are rejected"
        )
    source_path = Path(frozen["source_worldsheet_points"])
    if not source_path.is_absolute():
        source_path = frozen_path.parent / source_path
    actual_source_hash = sha256(source_path)
    if actual_source_hash != frozen["source_worldsheet_points_sha256"]:
        raise ValueError("the source imaginary-ray worldsheet points changed after fitting")

    fit_coefficients = np.asarray(
        frozen["primary_fit"]["coefficients_in_t"], dtype=float
    )
    coefficient_relative_errors = np.abs(
        fit_coefficients / MATRIX_COEFFICIENTS_IN_T - 1.0
    )
    grid = np.linspace(
        min(point["t"] for point in frozen["primary_points"]), 0.50, 321
    )
    q_fit = reduced_amplitude(grid, fit_coefficients)
    q_matrix = reduced_amplitude(grid, MATRIX_COEFFICIENTS_IN_T)
    curve = [
        {
            "t": float(t),
            "Q_worldsheet_fit": float(q_ws),
            "Q_matrix_model": float(q_mm),
            "amplitude_worldsheet_fit": float(a_ws),
            "amplitude_matrix_model": float(a_mm),
        }
        for t, q_ws, q_mm, a_ws, a_mm in zip(
            grid,
            q_fit,
            q_matrix,
            normalized_amplitude(grid, q_fit),
            normalized_amplitude(grid, q_matrix),
            strict=True,
        )
    ]
    result: dict[str, Any] = {
        "status": "imaginary_ray_fit_compared_after_hash_verification",
        "worldsheet_frozen_fit": frozen_path.name,
        "worldsheet_frozen_fit_sha256": sha256(frozen_path),
        "verified_worldsheet_source_sha256": actual_source_hash,
        "comparison_domain": "omega=i*t",
        "direct_physical_iepsilon_claimed": False,
        "matrix_model_reduced_amplitude": "Q_4(i*t)=2-12*t+16*t^2",
        "matrix_model_normalized_amplitude": "mu^3*A_tree(i*t)=-4*t^5*Q_4(i*t)",
        "worldsheet_fit_coefficients_in_t": [
            float(value) for value in fit_coefficients
        ],
        "matrix_model_coefficients_in_t": [
            float(value) for value in MATRIX_COEFFICIENTS_IN_T
        ],
        "coefficient_relative_errors": [
            float(value) for value in coefficient_relative_errors
        ],
        "maximum_coefficient_relative_error": float(
            np.max(coefficient_relative_errors)
        ),
        "figure": figure_path.name,
        "curve": curve,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    _draw_plot(frozen, fit_coefficients, figure_path)
    return result


def main() -> None:
    base = Path(__file__).parent
    results = base / "results" / "sphere_five_point_1to4"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-worldsheet-fit",
        type=Path,
        default=results / "worldsheet_imaginary_ray_fit_frozen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results / "matrix_model_comparison_imaginary_ray.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=base / "figures" / "sphere_one_to_four_amplitude_imaginary_ray.png",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            compare(
                arguments.frozen_worldsheet_fit,
                arguments.output,
                arguments.figure,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
