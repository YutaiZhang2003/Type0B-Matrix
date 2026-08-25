#!/usr/bin/env python3
"""Compare frozen sphere ``1->3`` worldsheet artifacts with the matrix model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, manifest_path: Path) -> tuple[dict[str, object], str]:
    manifest = json.loads(manifest_path.read_text())
    actual_hash = _sha256(path)
    if actual_hash != manifest["sha256"]:
        raise RuntimeError(f"frozen artifact hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if not str(payload["status"]).startswith("worldsheet_only"):
        raise RuntimeError(f"input is not a frozen worldsheet-only artifact: {path}")
    if payload.get("matrix_model_information_used") is not False:
        raise RuntimeError(f"input does not certify blind production: {path}")
    return payload, actual_hash


def matrix_model_q3(omega: np.ndarray) -> np.ndarray:
    return 1.0 + 3.0j * omega


def normalized_amplitude(omega: np.ndarray, q3: np.ndarray) -> np.ndarray:
    return 3.0j * omega**4 * q3


def compare(
    scan_path: Path,
    scan_manifest_path: Path,
    audit_paths: list[Path],
    audit_manifest_paths: list[Path],
    output_path: Path,
    figure_path: Path,
) -> dict[str, object]:
    scan, scan_hash = _verify(scan_path, scan_manifest_path)
    audits = []
    audit_hashes = []
    for path, manifest in zip(audit_paths, audit_manifest_paths):
        payload, artifact_hash = _verify(path, manifest)
        if payload["verified_scan_sha256"] != scan_hash:
            raise RuntimeError("an audit refers to a different frozen scan")
        audits.append(payload)
        audit_hashes.append(artifact_hash)

    t = np.asarray([point["t"] for point in scan["points"]], dtype=float)
    q_worldsheet = np.asarray(
        [point["Q3"]["real"] for point in scan["points"]], dtype=float
    )
    sigma_worldsheet = np.asarray(
        [point["Q3_standard_error"]["real"] for point in scan["points"]],
        dtype=float,
    )
    omega = 1.0j * t
    q_matrix = matrix_model_q3(omega).real
    difference = q_worldsheet - q_matrix

    deep_by_t: dict[float, tuple[float, float]] = {}
    for audit in audits:
        for record in audit["points"]:
            point = record["evaluations"]["deep_rqmc"]
            deep_by_t[round(float(record["t"]), 12)] = (
                float(point["Q3"]["real"]),
                float(point["Q3_standard_error"]["real"]),
            )

    audited_records = []
    best_q = q_worldsheet.copy()
    conservative_sigma = sigma_worldsheet.copy()
    for index, value in enumerate(t):
        key = round(float(value), 12)
        if key not in deep_by_t:
            continue
        deep_value, deep_sigma = deep_by_t[key]
        independent_spread = abs(deep_value - q_worldsheet[index])
        best_q[index] = deep_value
        conservative_sigma[index] = max(
            sigma_worldsheet[index], deep_sigma, independent_spread
        )
        audited_records.append(
            {
                "t": float(value),
                "production_Q3": float(q_worldsheet[index]),
                "production_standard_error": float(sigma_worldsheet[index]),
                "deep_Q3": deep_value,
                "deep_standard_error": deep_sigma,
                "independent_spread": independent_spread,
                "conservative_standard_error": float(conservative_sigma[index]),
                "matrix_model_Q3": float(q_matrix[index]),
                "conservative_z_score": float(
                    (deep_value - q_matrix[index]) / conservative_sigma[index]
                ),
            }
        )

    design = np.column_stack((np.ones_like(t), t))
    coefficients, _, _, _ = np.linalg.lstsq(design, q_worldsheet, rcond=None)
    fitted = design @ coefficients
    residuals = q_worldsheet - fitted
    residual_scale = math.sqrt(float(np.sum(residuals**2)) / (len(t) - 2))
    covariance = residual_scale**2 * np.linalg.inv(design.T @ design)
    coefficient_errors = np.sqrt(np.diag(covariance))
    best_coefficients, _, _, _ = np.linalg.lstsq(design, best_q, rcond=None)
    best_residuals = best_q - design @ best_coefficients
    best_residual_scale = math.sqrt(
        float(np.sum(best_residuals**2)) / (len(t) - 2)
    )
    best_covariance = best_residual_scale**2 * np.linalg.inv(design.T @ design)
    best_coefficient_errors = np.sqrt(np.diag(best_covariance))

    amplitude_worldsheet = normalized_amplitude(omega, q_worldsheet.astype(complex))
    amplitude_matrix = normalized_amplitude(omega, q_matrix.astype(complex))
    amplitude_sigma = 3.0 * t**4 * sigma_worldsheet
    conservative_z_scores = (best_q - q_matrix) / conservative_sigma

    point_records = []
    for index, value in enumerate(t):
        point_records.append(
            {
                "t": float(value),
                "worldsheet_Q3": float(q_worldsheet[index]),
                "worldsheet_standard_error": float(sigma_worldsheet[index]),
                "matrix_model_Q3": float(q_matrix[index]),
                "difference": float(difference[index]),
                "naive_qmc_z_score": float(difference[index] / sigma_worldsheet[index]),
                "worldsheet_amplitude": {
                    "real": float(amplitude_worldsheet[index].real),
                    "imag": float(amplitude_worldsheet[index].imag),
                },
                "amplitude_standard_error": float(amplitude_sigma[index]),
                "matrix_model_amplitude": {
                    "real": float(amplitude_matrix[index].real),
                    "imag": float(amplitude_matrix[index].imag),
                },
            }
        )

    dense_t = np.linspace(0.145, 0.475, 500)
    dense_omega = 1.0j * dense_t
    dense_q = matrix_model_q3(dense_omega).real
    dense_amplitude = normalized_amplitude(dense_omega, dense_q.astype(complex)).imag
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)
    axes[0].plot(dense_t, dense_q, color="#222222", linewidth=1.7, label="matrix model")
    axes[0].errorbar(
        t,
        q_worldsheet,
        yerr=sigma_worldsheet,
        fmt="o",
        markersize=4.8,
        capsize=2.5,
        color="#d36c45",
        ecolor="#d36c45",
        label="frozen worldsheet scan",
    )
    if audited_records:
        audited_t = np.asarray([record["t"] for record in audited_records])
        audited_q = np.asarray([record["deep_Q3"] for record in audited_records])
        audited_sigma = np.asarray(
            [record["conservative_standard_error"] for record in audited_records]
        )
        axes[0].errorbar(
            audited_t,
            audited_q,
            yerr=audited_sigma,
            fmt="s",
            markersize=4.5,
            capsize=2.5,
            color="#315b9d",
            ecolor="#315b9d",
            label="independent deep audit",
        )
    axes[0].axhline(0.0, color="#888888", linewidth=0.7)
    axes[0].set_ylabel(r"$Q_3(i t)$")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.18)

    axes[1].plot(
        dense_t,
        dense_amplitude,
        color="#222222",
        linewidth=1.7,
        label="matrix model",
    )
    axes[1].errorbar(
        t,
        amplitude_worldsheet.imag,
        yerr=amplitude_sigma,
        fmt="o",
        markersize=4.8,
        capsize=2.5,
        color="#d36c45",
        ecolor="#d36c45",
        label="worldsheet",
    )
    axes[1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1].set_xlabel(r"$t$ in $\omega=i t$")
    axes[1].set_ylabel(r"$\mathrm{Im}[\mu^2\mathcal{A}_{1\to3}^{\rm tree}(i t)]$")
    axes[1].grid(alpha=0.18)
    figure.suptitle(r"Sphere $1\to3$ amplitude: blind worldsheet computation and later comparison")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)

    result = {
        "verified_blind_worldsheet_scan": {
            "path": str(scan_path.resolve()),
            "sha256": scan_hash,
        },
        "verified_blind_worldsheet_audits": [
            {"path": str(path.resolve()), "sha256": artifact_hash}
            for path, artifact_hash in zip(audit_paths, audit_hashes)
        ],
        "comparison_order": (
            "matrix-model function evaluated only after the scan and convergence "
            "audits were frozen and their hashes verified"
        ),
        "matrix_model_Q3": "Q3(omega)=1+3 i omega",
        "matrix_model_amplitude": "mu^2 A_tree=3 i omega^4 Q3(omega)",
        "worldsheet_unweighted_affine_fit_in_t": {
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
            "intercept_standard_error_from_residuals": float(coefficient_errors[0]),
            "slope_standard_error_from_residuals": float(coefficient_errors[1]),
            "residual_rms": float(math.sqrt(np.mean(residuals**2))),
        },
        "worldsheet_affine_fit_with_deep_audit_replacements": {
            "intercept": float(best_coefficients[0]),
            "slope": float(best_coefficients[1]),
            "intercept_standard_error_from_residuals": float(
                best_coefficient_errors[0]
            ),
            "slope_standard_error_from_residuals": float(best_coefficient_errors[1]),
            "residual_rms": float(math.sqrt(np.mean(best_residuals**2))),
        },
        "point_count": len(t),
        "maximum_absolute_Q3_difference": float(np.max(np.abs(difference))),
        "production_qmc_chi_squared": float(np.sum((difference / sigma_worldsheet) ** 2)),
        "production_qmc_degrees_of_freedom": len(t),
        "deep_replacement_conservative_chi_squared": float(
            np.sum(conservative_z_scores**2)
        ),
        "deep_replacement_conservative_degrees_of_freedom": len(t),
        "maximum_deep_replacement_conservative_absolute_z_score": float(
            np.max(np.abs(conservative_z_scores))
        ),
        "audited_points": audited_records,
        "maximum_audited_conservative_absolute_z_score": max(
            abs(float(record["conservative_z_score"])) for record in audited_records
        ),
        "points": point_records,
        "figure": str(figure_path.resolve()),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent / "results" / "sphere_four_point_1to3"
    parser.add_argument("--scan", type=Path, default=base / "worldsheet_scan.json")
    parser.add_argument(
        "--scan-manifest", type=Path, default=base / "worldsheet_scan_frozen.json"
    )
    parser.add_argument(
        "--audits",
        nargs="+",
        type=Path,
        default=(base / "worldsheet_audit.json", base / "worldsheet_audit_t032.json"),
    )
    parser.add_argument(
        "--audit-manifests",
        nargs="+",
        type=Path,
        default=(
            base / "worldsheet_audit_frozen.json",
            base / "worldsheet_audit_t032_frozen.json",
        ),
    )
    parser.add_argument("--output", type=Path, default=base / "matrix_comparison.json")
    parser.add_argument("--figure", type=Path, default=base / "amplitude_comparison.png")
    arguments = parser.parse_args()
    if len(arguments.audits) != len(arguments.audit_manifests):
        parser.error("--audits and --audit-manifests must have equal lengths")
    result = compare(
        arguments.scan,
        arguments.scan_manifest,
        list(arguments.audits),
        list(arguments.audit_manifests),
        arguments.output,
        arguments.figure,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
