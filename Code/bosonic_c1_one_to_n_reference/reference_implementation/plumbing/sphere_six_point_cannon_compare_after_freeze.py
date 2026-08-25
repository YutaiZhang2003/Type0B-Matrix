#!/usr/bin/env python3
"""Compare the Cannon worldsheet scan only after its blind freeze gate passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_frozen_scan(run_dir: Path) -> tuple[dict[str, object], str]:
    assembled = run_dir / "assembled"
    manifest_path = assembled / "worldsheet_freeze_manifest.json"
    report_path = assembled / "accuracy_report.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "worldsheet_freeze_valid":
        raise RuntimeError("the blind worldsheet freeze manifest is not valid")
    passed_accuracy_gate = manifest.get(
        "all_points_pass_accuracy_gate",
        manifest.get("all_points_pass_1e_minus_3_gate", False),
    )
    if not bool(passed_accuracy_gate):
        raise RuntimeError("the declared blind accuracy gate did not pass at every point")
    if not bool(manifest.get("comparison_allowed")):
        raise RuntimeError("the freeze manifest does not allow comparison")
    scan_path = assembled / str(manifest["worldsheet_frozen_file"])
    scan_hash = sha256_file(scan_path)
    if scan_hash != str(manifest["worldsheet_frozen_sha256"]):
        raise RuntimeError("the frozen worldsheet scan checksum changed")
    if sha256_file(report_path) != str(manifest["accuracy_report_sha256"]):
        raise RuntimeError("the frozen accuracy report checksum changed")
    report = json.loads(report_path.read_text())
    if int(report.get("failed_point_count", -1)) != 0:
        raise RuntimeError("one or more worldsheet points failed before comparison")
    scan = json.loads(scan_path.read_text())
    if scan.get("status") != "worldsheet_only_frozen_before_comparison":
        raise RuntimeError("the input scan is not a frozen worldsheet-only result")
    if bool(scan.get("target_formula_available", True)):
        raise RuntimeError("the worldsheet result records a blinding violation")
    point_count = int(manifest.get("point_count", -1))
    if point_count < 1 or int(scan.get("point_count", -1)) != point_count:
        raise RuntimeError("the frozen scan point count does not match its manifest")
    declared_target = float(scan["accuracy"]["target"])
    manifest_target = float(manifest.get("accuracy_target_Q5", declared_target))
    if manifest_target != declared_target:
        raise RuntimeError("the frozen accuracy target does not match its manifest")
    if not all(bool(point["passes_accuracy_gate"]) for point in scan["points"]):
        raise RuntimeError("a frozen point does not pass the declared accuracy gate")
    return scan, scan_hash


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=base / "results" / "sphere_six_point_1to5" / "cannon_blind50_v1",
    )
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or arguments.run_dir / "comparison"

    scan, scan_hash = verified_frozen_scan(arguments.run_dir)

    # This import is deliberately below the checksum and declared accuracy gate.
    try:
        from sphere_six_point_matrix_comparison import (
            amplitude_imaginary_matrix_model,
            q5_matrix_model,
            render_comparison_png,
        )
    except ImportError:  # pragma: no cover
        from plumbing.sphere_six_point_matrix_comparison import (
            amplitude_imaginary_matrix_model,
            q5_matrix_model,
            render_comparison_png,
        )

    points: list[dict[str, float]] = []
    for source in scan["points"]:
        t = float(source["t"])
        q5_worldsheet = float(source["Q5_worldsheet"]["real"])
        q5_error = float(source["stability_envelope_Q5"])
        amplitude_worldsheet = float(source["mu4_A_tree_worldsheet"]["imag"])
        amplitude_error = 5.0 * t**6 * q5_error
        q5_target = float(q5_matrix_model(t))
        amplitude_target = float(amplitude_imaginary_matrix_model(t))
        q5_residual = q5_worldsheet - q5_target
        amplitude_residual = amplitude_worldsheet - amplitude_target
        points.append(
            {
                "t": t,
                "q5_worldsheet": q5_worldsheet,
                "q5_worldsheet_stability_envelope": q5_error,
                "q5_matrix_model": q5_target,
                "q5_residual": q5_residual,
                "q5_pull": q5_residual / q5_error,
                "amplitude_imaginary_worldsheet": amplitude_worldsheet,
                "amplitude_imaginary_stability_envelope": amplitude_error,
                "amplitude_imaginary_matrix_model": amplitude_target,
                "amplitude_imaginary_residual": amplitude_residual,
                "amplitude_imaginary_pull": amplitude_residual / amplitude_error,
            }
        )

    pulls = np.asarray([point["q5_pull"] for point in points], dtype=float)
    residuals = np.asarray([point["q5_residual"] for point in points], dtype=float)
    result = {
        "status": "matrix_comparison_completed_after_blind_worldsheet_freeze",
        "comparison_performed_after_freeze": True,
        "input_worldsheet_sha256": scan_hash,
        "point_count": len(points),
        "error_model": "per-point maximum of production QMC and paired discretization stability bounds",
        "goodness": {
            "maximum_absolute_q5_residual": float(np.max(np.abs(residuals))),
            "rms_q5_residual": float(np.sqrt(np.mean(residuals**2))),
            "maximum_absolute_pull": float(np.max(np.abs(pulls))),
            "chi_squared": float(np.sum(pulls**2)),
            "degrees_of_freedom": len(points),
            "rms_pull": float(np.sqrt(np.mean(pulls**2))),
        },
        "points": points,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "matrix_model_comparison.json"
    csv_path = output_dir / "matrix_model_comparison.csv"
    figure_path = output_dir / (
        f"sphere_one_to_five_cannon_blind{len(points)}_comparison.png"
    )
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    render_comparison_png(
        figure_path,
        t_points=np.asarray([point["t"] for point in points]),
        amplitude_points=np.asarray(
            [point["amplitude_imaginary_worldsheet"] for point in points]
        ),
        amplitude_errors=np.asarray(
            [point["amplitude_imaginary_stability_envelope"] for point in points]
        ),
        q5_points=np.asarray([point["q5_worldsheet"] for point in points]),
        q5_errors=np.asarray(
            [point["q5_worldsheet_stability_envelope"] for point in points]
        ),
        first_wall=float(scan["kinematic_domain"]["first_residue_wall"]),
    )
    print(json.dumps(result["goodness"], indent=2, sort_keys=True))
    print(json_path)
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    main()
