#!/usr/bin/env python3
"""Freeze and fit the sphere 1->5 worldsheet data on omega=i*t.

This module contains no matrix-model coefficient.  It verifies the existing
16-point worldsheet freeze, writes a points-only input, and fits the explicit
cubic ansatz

    Q_5(i t) = a + b t + c t^2 + d t^3.

The primary fit is generalized least squares.  QMC errors are independent,
while the single stored discretization estimate is treated as a fully
correlated additive uncertainty across t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POINTS_STATUS = "sphere_1to5_imaginary_ray_points_frozen_for_target_free_fit"
FIT_STATUS = "sphere_1to5_imaginary_ray_cubic_fit_frozen_for_separate_comparison"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_points(
    scan_path: Path,
    manifest_path: Path,
    audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    scan = json.loads(scan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    actual_scan_hash = sha256(scan_path)
    if actual_scan_hash != manifest.get("sha256"):
        raise ValueError("the merged 16-point worldsheet scan does not match its freeze")
    if manifest.get("status") != "worldsheet_only_frozen_before_matrix_model_comparison":
        raise ValueError("the merged worldsheet scan is not frozen for comparison")
    if scan.get("status") != "worldsheet_only_no_matrix_model_imported":
        raise ValueError("the merged scan is not labelled worldsheet-only")
    if scan.get("provenance", {}).get("matrix_model_information_used") is not False:
        raise ValueError("merged scan provenance does not exclude matrix-model information")
    if audit.get("status") != "worldsheet_only_no_matrix_model_imported":
        raise ValueError("the discretization audit is not labelled worldsheet-only")

    points: list[dict[str, float]] = []
    for source in scan["points"]:
        points.append(
            {
                "t": float(source["t"]),
                "Q5": float(source["Q5_worldsheet"]["real"]),
                "Q5_qmc_standard_error": float(
                    source["Q5_worldsheet_standard_error"]["real"]
                ),
                "amplitude_imaginary": float(
                    source["mu4_A_tree_worldsheet"]["imag"]
                ),
                "amplitude_imaginary_qmc_standard_error": float(
                    source["mu4_A_tree_worldsheet_standard_error"]["imag"]
                ),
            }
        )
    t_values = [point["t"] for point in points]
    if t_values != sorted(t_values) or len(set(t_values)) != len(t_values):
        raise ValueError("worldsheet t values must be unique and increasing")
    if len(points) != 16 or any(not 0.0 < t < 1.0 / 3.0 for t in t_values):
        raise ValueError("expected sixteen points strictly below the first residue wall")

    discretization = float(audit["diagnostics"]["combined_discretization_Q5"])
    result: dict[str, Any] = {
        "status": POINTS_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_scan": scan_path.name,
        "source_worldsheet_scan_sha256": actual_scan_hash,
        "source_freeze_manifest": manifest_path.name,
        "source_freeze_manifest_sha256": sha256(manifest_path),
        "source_worldsheet_audit": audit_path.name,
        "source_worldsheet_audit_sha256": sha256(audit_path),
        "target_information_present": False,
        "kinematic_domain": "omega=i*t with 0<t<1/3",
        "Q5_discretization_error": discretization,
        "discretization_correlation_model": (
            "one fully correlated additive Q5 nuisance across all t"
        ),
        "points": points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _design(t: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(t), t, t**2, t**3))


def _fit_with_covariance(
    t: np.ndarray,
    q: np.ndarray,
    data_covariance: np.ndarray,
    *,
    weighting: str,
) -> dict[str, Any]:
    design = _design(t)
    covariance_inverse_design = np.linalg.solve(data_covariance, design)
    normal = design.T @ covariance_inverse_design
    coefficient_covariance = np.linalg.inv(normal)
    coefficients = coefficient_covariance @ design.T @ np.linalg.solve(
        data_covariance, q
    )
    residuals = q - design @ coefficients
    chi_squared = float(residuals @ np.linalg.solve(data_covariance, residuals))
    return {
        "number_of_points": len(t),
        "degree": 3,
        "weighting": weighting,
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors": [
            float(value) for value in np.sqrt(np.diag(coefficient_covariance))
        ],
        "coefficient_covariance": [
            [float(value) for value in row] for row in coefficient_covariance
        ],
        "residuals": [float(value) for value in residuals],
        "chi_squared": chi_squared,
        "degrees_of_freedom": len(t) - 4,
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def _unweighted_sensitivity(t: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    design = _design(t)
    coefficients, *_ = np.linalg.lstsq(design, q, rcond=None)
    residuals = q - design @ coefficients
    residual_variance = float(residuals @ residuals / (len(t) - 4))
    covariance = np.linalg.inv(design.T @ design) * residual_variance
    return {
        "weighting": "unweighted ordinary least squares sensitivity",
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors_from_residual_scatter": [
            float(value) for value in np.sqrt(np.diag(covariance))
        ],
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def freeze_fit(points_path: Path, output_path: Path) -> dict[str, Any]:
    source = json.loads(points_path.read_text())
    if source.get("status") != POINTS_STATUS:
        raise ValueError("fit requires the frozen points-only 1->5 table")
    if source.get("target_information_present") is not False:
        raise ValueError("fit input must explicitly exclude target information")
    points = source["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q = np.asarray([point["Q5"] for point in points], dtype=float)
    qmc = np.asarray(
        [point["Q5_qmc_standard_error"] for point in points], dtype=float
    )
    discretization = float(source["Q5_discretization_error"])
    correlated_covariance = np.diag(qmc**2) + discretization**2 * np.ones(
        (len(t), len(t))
    )
    diagonal_covariance = np.diag(qmc**2 + discretization**2)

    result: dict[str, Any] = {
        "status": FIT_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_points": points_path.name,
        "source_worldsheet_points_sha256": sha256(points_path),
        "target_information_used": False,
        "fit_ansatz": "Q_5(i*t)=a+b*t+c*t^2+d*t^3",
        "fit_degree_assumption": (
            "degree three is an explicit analytic ansatz and no target "
            "coefficient is present in this program"
        ),
        "primary_fit": _fit_with_covariance(
            t,
            q,
            correlated_covariance,
            weighting=(
                "generalized least squares: diagonal QMC covariance plus one "
                "fully correlated additive discretization nuisance"
            ),
        ),
        "diagonal_combined_error_sensitivity_fit": _fit_with_covariance(
            t,
            q,
            diagonal_covariance,
            weighting=(
                "sensitivity: QMC and discretization combined independently "
                "at every t"
            ),
        ),
        "unweighted_sensitivity_fit": _unweighted_sensitivity(t, q),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    base = Path(__file__).parent / "results" / "sphere_six_point_1to5"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan",
        type=Path,
        default=base / "worldsheet_convergent_scan_16point_local.json",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=base / "worldsheet_freeze_manifest_16point_local.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=base / "worldsheet_numerical_audit.json",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=base / "worldsheet_imaginary_ray_points_frozen.json",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=base / "worldsheet_imaginary_ray_cubic_fit_frozen.json",
    )
    arguments = parser.parse_args()
    freeze_points(
        arguments.scan,
        arguments.freeze_manifest,
        arguments.audit,
        arguments.points_output,
    )
    print(json.dumps(freeze_fit(arguments.points_output, arguments.fit_output), indent=2))


if __name__ == "__main__":
    main()
