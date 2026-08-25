#!/usr/bin/env python3
"""Compare the frozen target-free sphere 1->5 cubic fit with the matrix model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from sphere_six_point_imaginary_ray_fit import FIT_STATUS, POINTS_STATUS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MATRIX_COEFFICIENTS_IN_T = np.asarray([6.0, -55.0, 150.0, -125.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _design(t: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(t), t, t**2, t**3))


def reduced_amplitude(t: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return _design(t) @ coefficients


def amplitude_imaginary(t: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    return -5.0 * t**6 * reduced


def _draw_plot(
    points: dict[str, Any],
    fit_coefficients: np.ndarray,
    fit_covariance: np.ndarray,
    figure_path: Path,
) -> None:
    observations = points["points"]
    t = np.asarray([point["t"] for point in observations], dtype=float)
    q = np.asarray([point["Q5"] for point in observations], dtype=float)
    qmc = np.asarray(
        [point["Q5_qmc_standard_error"] for point in observations], dtype=float
    )
    discretization = float(points["Q5_discretization_error"])
    displayed_error = np.sqrt(qmc**2 + discretization**2)

    grid = np.linspace(0.13, 1.0 / 3.0, 700, endpoint=False)
    grid_design = _design(grid)
    q_fit = grid_design @ fit_coefficients
    q_matrix = grid_design @ MATRIX_COEFFICIENTS_IN_T
    q_fit_error = np.sqrt(
        np.einsum("ij,jk,ik->i", grid_design, fit_covariance, grid_design)
    )

    figure, axes = plt.subplots(2, 1, figsize=(10.4, 8.3), sharex=True)
    axes[0].fill_between(
        grid,
        amplitude_imaginary(grid, q_fit - q_fit_error),
        amplitude_imaginary(grid, q_fit + q_fit_error),
        color="#777777",
        alpha=0.17,
        label="worldsheet fit $1\sigma$ band",
    )
    axes[0].plot(
        grid,
        amplitude_imaginary(grid, q_matrix),
        color="#d55e00",
        linewidth=2.1,
        label="matrix model",
    )
    axes[0].plot(
        grid,
        amplitude_imaginary(grid, q_fit),
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        label="target-free cubic fit",
    )
    axes[0].errorbar(
        t,
        amplitude_imaginary(t, q),
        yerr=5.0 * t**6 * displayed_error,
        fmt="o",
        color="#0072b2",
        markersize=5.0,
        capsize=2.2,
        linewidth=1.0,
        label="worldsheet points",
        zorder=5,
    )

    axes[1].fill_between(
        grid,
        q_fit - q_fit_error,
        q_fit + q_fit_error,
        color="#777777",
        alpha=0.17,
    )
    axes[1].plot(grid, q_matrix, color="#d55e00", linewidth=2.1)
    axes[1].plot(
        grid, q_fit, color="#222222", linestyle="--", linewidth=1.8
    )
    axes[1].errorbar(
        t,
        q,
        yerr=displayed_error,
        fmt="o",
        color="#0072b2",
        markersize=5.0,
        capsize=2.2,
        linewidth=1.0,
        zorder=5,
    )

    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.axvline(1.0 / 3.0, color="#555555", linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.20)
    axes[0].set_ylabel(r"$\mathrm{Im}[\mu^4\mathcal{A}_{1\to5}^{\mathrm{tree}}(it)]$")
    axes[1].set_ylabel(r"$Q_5(it)$")
    axes[1].set_xlabel(r"$t$ in $\omega=it$")
    axes[0].set_title(
        r"Sphere $1\to5$: frozen imaginary-energy fit and later comparison"
    )
    axes[0].legend(loc="best", fontsize=8.5, ncol=2)
    axes[0].text(
        1.0 / 3.0 - 0.001,
        0.96,
        "first residue wall",
        transform=axes[0].get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def compare(fit_path: Path, output_path: Path, figure_path: Path) -> dict[str, Any]:
    frozen = json.loads(fit_path.read_text())
    if frozen.get("status") != FIT_STATUS:
        raise ValueError("comparison requires the frozen target-free 1->5 cubic fit")
    points_path = Path(frozen["source_worldsheet_points"])
    if not points_path.is_absolute():
        points_path = fit_path.parent / points_path
    if sha256(points_path) != frozen["source_worldsheet_points_sha256"]:
        raise ValueError("the frozen 1->5 points table changed after fitting")
    points = json.loads(points_path.read_text())
    if points.get("status") != POINTS_STATUS:
        raise ValueError("fit source is not the points-only 1->5 table")

    primary = frozen["primary_fit"]
    coefficients = np.asarray(primary["coefficients_in_t"], dtype=float)
    covariance = np.asarray(primary["coefficient_covariance"], dtype=float)
    difference = coefficients - MATRIX_COEFFICIENTS_IN_T
    coefficient_relative_errors = np.abs(
        coefficients / MATRIX_COEFFICIENTS_IN_T - 1.0
    )
    coefficient_standard_errors = np.sqrt(np.diag(covariance))
    coefficient_marginal_pulls = difference / coefficient_standard_errors
    coefficient_difference_quadratic_form = float(
        difference @ np.linalg.solve(covariance, difference)
    )

    observations = points["points"]
    t = np.asarray([point["t"] for point in observations], dtype=float)
    q = np.asarray([point["Q5"] for point in observations], dtype=float)
    qmc = np.asarray(
        [point["Q5_qmc_standard_error"] for point in observations], dtype=float
    )
    discretization = float(points["Q5_discretization_error"])
    data_covariance = np.diag(qmc**2) + discretization**2 * np.ones(
        (len(t), len(t))
    )
    matrix_at_points = reduced_amplitude(t, MATRIX_COEFFICIENTS_IN_T)
    residuals = q - matrix_at_points
    matrix_point_chi_squared = float(
        residuals @ np.linalg.solve(data_covariance, residuals)
    )
    diagonal_errors = np.sqrt(np.diag(data_covariance))

    grid = np.linspace(0.13, 1.0 / 3.0, 401, endpoint=False)
    grid_design = _design(grid)
    q_fit = grid_design @ coefficients
    q_matrix = grid_design @ MATRIX_COEFFICIENTS_IN_T
    q_fit_error = np.sqrt(
        np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design)
    )
    curve = [
        {
            "t": float(value),
            "Q5_worldsheet_fit": float(fit_value),
            "Q5_worldsheet_fit_standard_error": float(fit_error),
            "Q5_matrix_model": float(matrix_value),
            "amplitude_imaginary_worldsheet_fit": float(amplitude_fit),
            "amplitude_imaginary_matrix_model": float(amplitude_matrix),
        }
        for value, fit_value, fit_error, matrix_value, amplitude_fit, amplitude_matrix in zip(
            grid,
            q_fit,
            q_fit_error,
            q_matrix,
            amplitude_imaginary(grid, q_fit),
            amplitude_imaginary(grid, q_matrix),
            strict=True,
        )
    ]
    result: dict[str, Any] = {
        "status": "sphere_1to5_imaginary_ray_fit_compared_after_hash_verification",
        "worldsheet_frozen_fit": fit_path.name,
        "worldsheet_frozen_fit_sha256": sha256(fit_path),
        "verified_worldsheet_points_sha256": sha256(points_path),
        "comparison_domain": "omega=i*t with 0<t<1/3",
        "matrix_model_coefficients_in_t": [
            float(value) for value in MATRIX_COEFFICIENTS_IN_T
        ],
        "matrix_model_reduced_amplitude": (
            "Q_5(i*t)=6-55*t+150*t^2-125*t^3"
        ),
        "worldsheet_fit_coefficients_in_t": [float(value) for value in coefficients],
        "worldsheet_fit_coefficient_standard_errors": primary[
            "coefficient_standard_errors"
        ],
        "coefficient_marginal_pulls": [
            float(value) for value in coefficient_marginal_pulls
        ],
        "maximum_absolute_coefficient_marginal_pull": float(
            np.max(np.abs(coefficient_marginal_pulls))
        ),
        "coefficient_relative_errors": [
            float(value) for value in coefficient_relative_errors
        ],
        "maximum_coefficient_relative_error": float(
            np.max(coefficient_relative_errors)
        ),
        "coefficient_difference_quadratic_form": coefficient_difference_quadratic_form,
        "coefficient_comparison_degrees_of_freedom": 4,
        "pointwise_matrix_chi_squared_with_correlated_discretization": (
            matrix_point_chi_squared
        ),
        "pointwise_matrix_degrees_of_freedom": len(t),
        "maximum_absolute_diagonal_pull": float(
            np.max(np.abs(residuals / diagonal_errors))
        ),
        "direct_physical_iepsilon_claimed": False,
        "figure": figure_path.name,
        "curve": curve,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    _draw_plot(points, coefficients, covariance, figure_path)
    return result


def main() -> None:
    base = Path(__file__).parent
    results = base / "results" / "sphere_six_point_1to5"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit",
        type=Path,
        default=results / "worldsheet_imaginary_ray_cubic_fit_frozen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results / "matrix_model_fit_comparison_16point_local.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=results / "sphere_one_to_five_amplitude_16point_fit.png",
    )
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.fit, arguments.output, arguments.figure), indent=2))


if __name__ == "__main__":
    main()
