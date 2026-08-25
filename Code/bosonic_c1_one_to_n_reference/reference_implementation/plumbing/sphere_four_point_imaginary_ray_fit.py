#!/usr/bin/env python3
"""Freeze a target-free affine fit to sphere ``1->3`` worldsheet data.

No matrix-model function or coefficient is present in this module.  The fit
ansatz is the same affine-in-``t`` design used for the original 16-point
analysis.  The unweighted fit is primary because the recorded QMC errors omit
correlated block and momentum-discretization effects; a QMC-weighted fit is
stored only as a sensitivity check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


SCAN_STATUS = "worldsheet_only_merged_and_frozen_before_external_comparison"
AUDIT_STATUS = "worldsheet_only_audit_frozen_before_external_comparison"
POINTS_STATUS = "sphere_1to3_worldsheet_points_frozen_for_target_free_fit"
FIT_STATUS = "sphere_1to3_worldsheet_affine_fit_frozen_for_separate_comparison"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _fit(t: np.ndarray, q: np.ndarray, sigma: np.ndarray | None) -> dict[str, Any]:
    design = np.column_stack((np.ones_like(t), t))
    if sigma is None:
        coefficients, *_ = np.linalg.lstsq(design, q, rcond=None)
        residuals = q - design @ coefficients
        residual_variance = float(residuals @ residuals / (len(t) - 2))
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        chi_squared = None
        weighting = "unweighted ordinary least squares"
    else:
        inverse_variance = 1.0 / sigma**2
        normal = design.T @ (inverse_variance[:, None] * design)
        covariance = np.linalg.inv(normal)
        coefficients = covariance @ design.T @ (inverse_variance * q)
        residuals = q - design @ coefficients
        chi_squared = float(np.sum((residuals / sigma) ** 2))
        weighting = "diagonal QMC inverse-variance sensitivity fit"
    return {
        "number_of_points": len(t),
        "degree": 1,
        "weighting": weighting,
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors": [
            float(value) for value in np.sqrt(np.diag(covariance))
        ],
        "coefficient_covariance": [
            [float(value) for value in row] for row in covariance
        ],
        "residuals": [float(value) for value in residuals],
        "chi_squared": chi_squared,
        "degrees_of_freedom": len(t) - 2,
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(math.sqrt(np.mean(residuals**2))),
    }


def freeze_points(
    scan_path: Path,
    manifest_path: Path,
    audit_paths: list[Path],
    audit_manifest_paths: list[Path],
    output_path: Path,
    *,
    expected_point_count: int = 30,
) -> dict[str, Any]:
    scan = json.loads(scan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scan_hash = sha256_file(scan_path)
    if scan_hash != manifest.get("sha256"):
        raise RuntimeError("merged worldsheet scan does not match its freeze manifest")
    if scan.get("status") != SCAN_STATUS or manifest.get("status") != SCAN_STATUS:
        raise RuntimeError("fit input is not the frozen merged worldsheet scan")
    if scan.get("matrix_model_information_used") is not False:
        raise RuntimeError("fit input does not certify target-free production")
    if len(scan.get("points", [])) != expected_point_count:
        raise RuntimeError(f"fit requires exactly {expected_point_count} worldsheet points")
    if len(audit_paths) != len(audit_manifest_paths):
        raise RuntimeError("audit and audit-manifest counts differ")

    points = [
        {
            "t": float(point["t"]),
            "Q3": float(point["Q3"]["real"]),
            "Q3_qmc_standard_error": float(point["Q3_standard_error"]["real"]),
            "raw_integral_I4": point["raw_integral_I4"],
            "scan_cohort": str(point["scan_cohort"]),
        }
        for point in scan["points"]
    ]
    t_values = [point["t"] for point in points]
    if t_values != sorted(t_values) or len(set(t_values)) != len(t_values):
        raise RuntimeError("worldsheet t values must be distinct and increasing")

    base_scan_hash = scan["provenance"]["base_sha256"]
    source_audits = []
    deep_replacements = []
    seen_audit_t: set[float] = set()
    point_by_t = {round(point["t"], 12): point for point in points}
    for audit_path, audit_manifest_path in zip(
        audit_paths, audit_manifest_paths, strict=True
    ):
        audit = json.loads(audit_path.read_text())
        audit_manifest = json.loads(audit_manifest_path.read_text())
        audit_hash = sha256_file(audit_path)
        if audit_hash != audit_manifest.get("sha256"):
            raise RuntimeError(f"worldsheet audit does not match its freeze: {audit_path}")
        if (
            audit.get("status") != AUDIT_STATUS
            or audit_manifest.get("status") != AUDIT_STATUS
        ):
            raise RuntimeError(f"worldsheet audit has the wrong status: {audit_path}")
        if audit.get("matrix_model_information_used") is not False:
            raise RuntimeError(f"worldsheet audit does not certify blinding: {audit_path}")
        if audit.get("verified_scan_sha256") != base_scan_hash:
            raise RuntimeError("worldsheet audit does not refer to the known base scan")
        source_audits.append(
            {
                "artifact": str(audit_path.resolve()),
                "sha256": audit_hash,
                "manifest": str(audit_manifest_path.resolve()),
                "manifest_sha256": sha256_file(audit_manifest_path),
            }
        )
        for record in audit["points"]:
            t_value = round(float(record["t"]), 12)
            if t_value in seen_audit_t or t_value not in point_by_t:
                raise RuntimeError("audit t values are duplicate or absent from the merged scan")
            seen_audit_t.add(t_value)
            production = point_by_t[t_value]
            deep = record["evaluations"]["deep_rqmc"]
            deep_q = float(deep["Q3"]["real"])
            deep_sigma = float(deep["Q3_standard_error"]["real"])
            spread = abs(deep_q - production["Q3"])
            deep_replacements.append(
                {
                    "t": float(record["t"]),
                    "production_Q3": production["Q3"],
                    "production_Q3_qmc_standard_error": production[
                        "Q3_qmc_standard_error"
                    ],
                    "deep_Q3": deep_q,
                    "deep_Q3_qmc_standard_error": deep_sigma,
                    "independent_spread": spread,
                    "conservative_standard_error": max(
                        production["Q3_qmc_standard_error"], deep_sigma, spread
                    ),
                }
            )
    deep_replacements.sort(key=lambda record: record["t"])
    payload: dict[str, Any] = {
        "status": POINTS_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_scan": str(scan_path.resolve()),
        "source_worldsheet_scan_sha256": scan_hash,
        "source_freeze_manifest": str(manifest_path.resolve()),
        "source_freeze_manifest_sha256": sha256_file(manifest_path),
        "target_information_present": False,
        "point_count": len(points),
        "source_worldsheet_audits": source_audits,
        "deep_audit_replacements": deep_replacements,
        "points": points,
    }
    write_json(output_path, payload)
    return payload


def freeze_fit(
    points_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(points_path.read_text())
    if source.get("status") != POINTS_STATUS:
        raise RuntimeError("fit requires a frozen target-free points table")
    if source.get("target_information_present") is not False:
        raise RuntimeError("fit input contains target information")
    t = np.asarray([point["t"] for point in source["points"]], dtype=float)
    q = np.asarray([point["Q3"] for point in source["points"]], dtype=float)
    sigma = np.asarray(
        [point["Q3_qmc_standard_error"] for point in source["points"]], dtype=float
    )
    cohort = np.asarray([point["scan_cohort"] for point in source["points"]])
    known_mask = cohort == "known_base"
    extension_mask = cohort == "new_extension"
    if int(np.sum(known_mask)) != 16 or int(np.sum(extension_mask)) != 14:
        raise RuntimeError("expected 16 known points and 14 new extension points")
    deep_q = q.copy()
    index_by_t = {round(float(value), 12): index for index, value in enumerate(t)}
    for record in source["deep_audit_replacements"]:
        deep_q[index_by_t[round(float(record["t"]), 12)]] = float(record["deep_Q3"])
    payload: dict[str, Any] = {
        "status": FIT_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_points": str(points_path.resolve()),
        "source_worldsheet_points_sha256": sha256_file(points_path),
        "source_worldsheet_scan": source["source_worldsheet_scan"],
        "source_worldsheet_scan_sha256": source["source_worldsheet_scan_sha256"],
        "target_information_used": False,
        "fit_ansatz": "Q_3(i*t)=a+b*t",
        "fit_degree_assumption": "the pre-existing sphere 1->3 affine design",
        "primary_fit": _fit(t, q, None),
        "known_base_cohort_fit": _fit(t[known_mask], q[known_mask], None),
        "new_extension_cohort_fit": _fit(
            t[extension_mask], q[extension_mask], None
        ),
        "deep_audit_replacement_fit": _fit(t, deep_q, None),
        "deep_audit_replacements": source["deep_audit_replacements"],
        "qmc_weighted_sensitivity_fit": _fit(t, q, sigma),
    }
    write_json(output_path, payload)
    manifest: dict[str, Any] = {
        "status": FIT_STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": len(t),
        "target_information_used": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest


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
    base_dir = run_dir.parent
    parser.add_argument(
        "--audits",
        nargs="+",
        type=Path,
        default=(base_dir / "worldsheet_audit.json", base_dir / "worldsheet_audit_t032.json"),
    )
    parser.add_argument(
        "--audit-manifests",
        nargs="+",
        type=Path,
        default=(
            base_dir / "worldsheet_audit_frozen.json",
            base_dir / "worldsheet_audit_t032_frozen.json",
        ),
    )
    parser.add_argument(
        "--points-output", type=Path, default=run_dir / "worldsheet_points_30point_frozen.json"
    )
    parser.add_argument(
        "--fit-output", type=Path, default=run_dir / "worldsheet_affine_fit_30point_frozen.json"
    )
    parser.add_argument(
        "--fit-manifest", type=Path, default=run_dir / "worldsheet_affine_fit_30point_manifest.json"
    )
    arguments = parser.parse_args()
    freeze_points(
        arguments.scan,
        arguments.scan_manifest,
        list(arguments.audits),
        list(arguments.audit_manifests),
        arguments.points_output,
    )
    fit, _ = freeze_fit(arguments.points_output, arguments.fit_output, arguments.fit_manifest)
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
