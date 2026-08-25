#!/usr/bin/env python3
"""Compare the independently frozen 30-point sphere ``1->3`` fit to the matrix model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCAN_STATUS = "worldsheet_only_merged_and_frozen_before_external_comparison"
FIT_STATUS = "sphere_1to3_worldsheet_affine_fit_frozen_for_separate_comparison"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, manifest_path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError(f"artifact does not match its freeze manifest: {path}")
    if payload.get("status") != status or manifest.get("status") != status:
        raise RuntimeError(f"artifact has the wrong frozen status: {path}")
    return payload


def matrix_model_q3(omega: np.ndarray) -> np.ndarray:
    return 1.0 + 3.0j * omega


def normalized_amplitude(omega: np.ndarray, q3: np.ndarray) -> np.ndarray:
    return 3.0j * omega**4 * q3


def _cohort_metrics(
    difference: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected_difference = difference[mask]
    selected_pull = selected_difference / sigma[mask]
    return {
        "point_count": int(np.sum(mask)),
        "qmc_only_chi_squared": float(selected_pull @ selected_pull),
        "qmc_only_degrees_of_freedom": int(np.sum(mask)),
        "maximum_absolute_qmc_pull": float(np.max(np.abs(selected_pull))),
        "maximum_absolute_Q3_difference": float(
            np.max(np.abs(selected_difference))
        ),
        "rms_Q3_difference": float(
            math.sqrt(np.mean(selected_difference**2))
        ),
    }


def _coefficient_comparison(fit_record: dict[str, Any]) -> dict[str, Any]:
    coefficients = np.asarray(fit_record["coefficients_in_t"], dtype=float)
    errors = np.asarray(fit_record["coefficient_standard_errors"], dtype=float)
    covariance = np.asarray(fit_record["coefficient_covariance"], dtype=float)
    matrix_coefficients = np.asarray([1.0, -3.0])
    difference = coefficients - matrix_coefficients
    return {
        "worldsheet_coefficients_in_t": [float(value) for value in coefficients],
        "matrix_model_coefficients_in_t": [float(value) for value in matrix_coefficients],
        "fit_minus_matrix_coefficients": [float(value) for value in difference],
        "marginal_difference_in_fit_standard_errors": [
            float(value) for value in difference / errors
        ],
        "joint_coefficient_chi_squared": float(
            difference @ np.linalg.solve(covariance, difference)
        ),
        "joint_coefficient_degrees_of_freedom": 2,
    }


def compare(
    scan_path: Path,
    scan_manifest_path: Path,
    fit_path: Path,
    fit_manifest_path: Path,
    output_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    scan = _verify(scan_path, scan_manifest_path, SCAN_STATUS)
    fit = _verify(fit_path, fit_manifest_path, FIT_STATUS)
    scan_hash = sha256_file(scan_path)
    if fit.get("source_worldsheet_scan_sha256") != scan_hash:
        raise RuntimeError("the frozen fit was not derived from this worldsheet scan")
    if fit.get("target_information_used") is not False:
        raise RuntimeError("the worldsheet fit does not certify target-free production")

    t = np.asarray([point["t"] for point in scan["points"]], dtype=float)
    q_worldsheet = np.asarray(
        [point["Q3"]["real"] for point in scan["points"]], dtype=float
    )
    sigma = np.asarray(
        [point["Q3_standard_error"]["real"] for point in scan["points"]], dtype=float
    )
    cohort = np.asarray([point["scan_cohort"] for point in scan["points"]])
    known_mask = cohort == "known_base"
    extension_mask = cohort == "new_extension"
    if int(np.sum(known_mask)) != 16 or int(np.sum(extension_mask)) != 14:
        raise RuntimeError("expected 16 known points and 14 new extension points")
    omega = 1.0j * t
    q_matrix = matrix_model_q3(omega).real
    difference = q_worldsheet - q_matrix
    amplitude_worldsheet = normalized_amplitude(omega, q_worldsheet.astype(complex))
    amplitude_matrix = normalized_amplitude(omega, q_matrix.astype(complex))
    amplitude_sigma = 3.0 * t**4 * sigma

    best_q = q_worldsheet.copy()
    conservative_sigma = sigma.copy()
    index_by_t = {round(float(value), 12): index for index, value in enumerate(t)}
    audit_indices = []
    for record in fit["deep_audit_replacements"]:
        index = index_by_t[round(float(record["t"]), 12)]
        audit_indices.append(index)
        best_q[index] = float(record["deep_Q3"])
        conservative_sigma[index] = float(record["conservative_standard_error"])
    audit_indices_array = np.asarray(audit_indices, dtype=int)
    conservative_difference = best_q - q_matrix

    matrix_coefficients = np.asarray([1.0, -3.0])
    merged_coefficient_comparison = _coefficient_comparison(fit["primary_fit"])

    point_records = []
    for index, value in enumerate(t):
        point_records.append(
            {
                "t": float(value),
                "scan_cohort": str(cohort[index]),
                "worldsheet_Q3": float(q_worldsheet[index]),
                "worldsheet_Q3_qmc_standard_error": float(sigma[index]),
                "matrix_model_Q3": float(q_matrix[index]),
                "Q3_difference": float(difference[index]),
                "Q3_qmc_pull": float(difference[index] / sigma[index]),
                "worldsheet_amplitude": {
                    "real": float(amplitude_worldsheet[index].real),
                    "imag": float(amplitude_worldsheet[index].imag),
                },
                "amplitude_qmc_standard_error": float(amplitude_sigma[index]),
                "matrix_model_amplitude": {
                    "real": float(amplitude_matrix[index].real),
                    "imag": float(amplitude_matrix[index].imag),
                },
            }
        )

    dense_t = np.linspace(float(t.min()) - 0.01, float(t.max()) + 0.01, 600)
    dense_omega = 1.0j * dense_t
    dense_q = matrix_model_q3(dense_omega).real
    dense_amplitude = normalized_amplitude(dense_omega, dense_q.astype(complex)).imag
    figure, axes = plt.subplots(3, 1, figsize=(8.4, 9.6), sharex=True)
    axes[0].plot(dense_t, dense_q, color="#222222", linewidth=1.7, label="matrix model")
    cohort_styles = (
        (known_mask, "o", "#d36c45", "16 known worldsheet points"),
        (extension_mask, "D", "#26876b", "14 new worldsheet points"),
    )
    for mask, marker, color, label in cohort_styles:
        axes[0].errorbar(
            t[mask], q_worldsheet[mask], yerr=sigma[mask], fmt=marker,
            markersize=4.3, capsize=2.2, color=color, ecolor=color, label=label,
        )
    axes[0].errorbar(
        t[audit_indices_array], best_q[audit_indices_array],
        yerr=conservative_sigma[audit_indices_array], fmt="s", markersize=4.1,
        capsize=2.2, color="#315b9d", ecolor="#315b9d", label="5 deep audits",
    )
    axes[0].set_ylabel(r"$Q_3(i t)$")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.18)

    axes[1].plot(dense_t, dense_amplitude, color="#222222", linewidth=1.7)
    for mask, marker, color, _ in cohort_styles:
        axes[1].errorbar(
            t[mask], amplitude_worldsheet.imag[mask], yerr=amplitude_sigma[mask],
            fmt=marker, markersize=4.3, capsize=2.2, color=color, ecolor=color,
        )
    axes[1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1].set_ylabel(r"$\mathrm{Im}[\mu^2\mathcal{A}^{\rm tree}_{1\to3}]$")
    axes[1].grid(alpha=0.18)

    for mask, marker, color, _ in cohort_styles:
        axes[2].errorbar(
            t[mask], difference[mask], yerr=sigma[mask], fmt=marker,
            markersize=4.3, capsize=2.2, color=color, ecolor=color,
        )
    axes[2].errorbar(
        t[audit_indices_array], conservative_difference[audit_indices_array],
        yerr=conservative_sigma[audit_indices_array], fmt="s", markersize=4.1,
        capsize=2.2, color="#315b9d", ecolor="#315b9d",
    )
    axes[2].axhline(0.0, color="#222222", linewidth=1.0)
    axes[2].set_xlabel(r"$t$ in $\omega=i t$")
    axes[2].set_ylabel(r"$Q_3^{\rm WS}-Q_3^{\rm MM}$")
    axes[2].grid(alpha=0.18)
    figure.suptitle(r"Sphere $1\to3$: independently frozen worldsheet fit vs. matrix model")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)

    result: dict[str, Any] = {
        "comparison_order": (
            "the 30-point worldsheet scan and affine fit were independently frozen "
            "before this program evaluated the matrix-model result"
        ),
        "verified_worldsheet_scan": {
            "path": str(scan_path.resolve()),
            "sha256": scan_hash,
        },
        "verified_worldsheet_fit": {
            "path": str(fit_path.resolve()),
            "sha256": sha256_file(fit_path),
        },
        "point_count": len(t),
        "matrix_model_Q3": "Q3(omega)=1+3 i omega",
        "matrix_model_amplitude": "mu^2 A_tree=3 i omega^4 Q3(omega)",
        "worldsheet_primary_affine_fit": fit["primary_fit"],
        "matrix_model_coefficients_in_t": [float(value) for value in matrix_coefficients],
        "affine_fit_coefficient_comparisons": {
            "merged_30point": merged_coefficient_comparison,
            "known_base_16point": _coefficient_comparison(
                fit["known_base_cohort_fit"]
            ),
            "new_extension_14point": _coefficient_comparison(
                fit["new_extension_cohort_fit"]
            ),
            "deep_audit_replacement_30point": _coefficient_comparison(
                fit["deep_audit_replacement_fit"]
            ),
        },
        "cohort_Q3_comparisons": {
            "merged_30point": _cohort_metrics(
                difference, sigma, np.ones(len(t), dtype=bool)
            ),
            "known_base_16point": _cohort_metrics(
                difference, sigma, known_mask
            ),
            "new_extension_14point": _cohort_metrics(
                difference, sigma, extension_mask
            ),
        },
        "deep_replacement_conservative_Q3_comparisons": {
            "merged_30point": _cohort_metrics(
                conservative_difference,
                conservative_sigma,
                np.ones(len(t), dtype=bool),
            ),
            "known_base_16point": _cohort_metrics(
                conservative_difference, conservative_sigma, known_mask
            ),
            "new_extension_14point": _cohort_metrics(
                conservative_difference, conservative_sigma, extension_mask
            ),
        },
        "audited_worldsheet_points": fit["deep_audit_replacements"],
        "maximum_absolute_Q3_difference": float(np.max(np.abs(difference))),
        "rms_Q3_difference": float(math.sqrt(np.mean(difference**2))),
        "qmc_only_chi_squared": float(np.sum((difference / sigma) ** 2)),
        "qmc_only_degrees_of_freedom": len(t),
        "maximum_absolute_qmc_pull": float(np.max(np.abs(difference / sigma))),
        "points": point_records,
        "figure": str(figure_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    run_dir = (
        Path(__file__).parent
        / "results"
        / "sphere_four_point_1to3"
        / "blind30_20260824"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=run_dir / "worldsheet_scan_30point.json")
    parser.add_argument(
        "--scan-manifest", type=Path, default=run_dir / "worldsheet_scan_30point_frozen.json"
    )
    parser.add_argument(
        "--fit", type=Path, default=run_dir / "worldsheet_affine_fit_30point_frozen.json"
    )
    parser.add_argument(
        "--fit-manifest", type=Path, default=run_dir / "worldsheet_affine_fit_30point_manifest.json"
    )
    parser.add_argument("--output", type=Path, default=run_dir / "matrix_comparison_30point.json")
    parser.add_argument("--figure", type=Path, default=run_dir / "amplitude_comparison_30point.png")
    arguments = parser.parse_args()
    result = compare(
        arguments.scan,
        arguments.scan_manifest,
        arguments.fit,
        arguments.fit_manifest,
        arguments.output,
        arguments.figure,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
