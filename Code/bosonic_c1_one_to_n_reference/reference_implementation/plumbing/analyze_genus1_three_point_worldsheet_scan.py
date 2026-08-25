#!/usr/bin/env python3
"""Target-free shape analysis of the blind torus three-point t scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_SCAN_DIR = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "equal_split_imaginary_t_scan10_p12_n256_v1"
)


def _replicate_curve(point: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [-float(value["imag"]) for value in point["replicate_finals"]],
        dtype=float,
    )


def _quadratic_vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] >= 0.0:
        raise ValueError("the local three-point quadratic is not concave")
    location = float(-coefficients[1] / (2.0 * coefficients[0]))
    return location, float(np.polyval(coefficients, location))


def analyze_manifest(manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("comparison_stage_present") is not False:
        raise ValueError("shape analysis requires a worldsheet-only manifest")
    points = manifest["points"]
    t_values = np.asarray(manifest["t_values"], dtype=float)
    mean_curve = np.asarray([-float(point["I_1,3"]["imag"]) for point in points])
    standard_errors = np.asarray(
        [float(point["rqmc_standard_error"]["imag"]) for point in points]
    )
    replicate_curves = np.stack([_replicate_curve(point) for point in points], axis=1)
    if replicate_curves.shape[0] < 2:
        raise ValueError("at least two aligned scrambles are required")

    maximum_index = int(np.argmax(mean_curve))
    if maximum_index in (0, len(t_values) - 1):
        raise ValueError("the discrete maximum is not bracketed by scan points")
    local_slice = slice(maximum_index - 1, maximum_index + 2)
    mean_peak_t, mean_peak_height = _quadratic_vertex(
        t_values[local_slice],
        mean_curve[local_slice],
    )
    replicate_peak_locations = np.asarray(
        [
            _quadratic_vertex(t_values[local_slice], curve[local_slice])[0]
            for curve in replicate_curves
        ]
    )
    peak_location_se = float(
        np.std(replicate_peak_locations, ddof=1)
        / math.sqrt(len(replicate_peak_locations))
    )

    adjacent_differences = np.diff(replicate_curves, axis=1)
    adjacent_means = np.mean(adjacent_differences, axis=0)
    adjacent_standard_errors = np.std(adjacent_differences, axis=0, ddof=1) / math.sqrt(
        replicate_curves.shape[0]
    )
    reduced_t3 = mean_curve / t_values**3
    tail_fractions = np.asarray([float(point["tail_fraction"]) for point in points])
    tail_residuals = np.asarray(
        [float(point["tail_fit_relative_residual"]) for point in points]
    )
    block_shifts = np.asarray(
        [abs(float(point["bulk_block_order_shift"]["imag"])) for point in points]
    )

    return {
        "calculation": "target-free shape analysis of blind genus-one three-point scan",
        "comparison_stage_present": False,
        "ordinate": "-Im I_1,3(it/2,it/2)",
        "t_values": t_values.tolist(),
        "mean_curve": mean_curve.tolist(),
        "rqmc_standard_errors": standard_errors.tolist(),
        "relative_rqmc_standard_errors": (standard_errors / mean_curve).tolist(),
        "common_scramble_covariance": np.cov(replicate_curves, rowvar=False, ddof=1).tolist(),
        "discrete_maximum": {
            "t": float(t_values[maximum_index]),
            "value": float(mean_curve[maximum_index]),
        },
        "local_quadratic_peak": {
            "fit_points": t_values[local_slice].tolist(),
            "mean_curve_t": mean_peak_t,
            "mean_curve_height": mean_peak_height,
            "replicate_peak_locations": replicate_peak_locations.tolist(),
            "replicate_mean_t": float(np.mean(replicate_peak_locations)),
            "replicate_standard_error_t": peak_location_se,
            "status": "descriptive local-smoke statistic; no systematic error assigned",
        },
        "adjacent_changes": [
            {
                "t_left": float(t_values[index]),
                "t_right": float(t_values[index + 1]),
                "mean_change": float(adjacent_means[index]),
                "common_scramble_standard_error": float(adjacent_standard_errors[index]),
                "pull": float(adjacent_means[index] / adjacent_standard_errors[index]),
            }
            for index in range(len(adjacent_means))
        ],
        "t_cubed_reduced_curve": reduced_t3.tolist(),
        "t_cubed_reduced_curve_is_strictly_decreasing": bool(
            np.all(np.diff(reduced_t3) < 0.0)
        ),
        "numerical_control": {
            "tail_fraction_range": [float(np.min(tail_fractions)), float(np.max(tail_fractions))],
            "tail_fit_relative_residual_range": [
                float(np.min(tail_residuals)),
                float(np.max(tail_residuals)),
            ],
            "relative_rqmc_standard_error_range": [
                float(np.min(standard_errors / mean_curve)),
                float(np.max(standard_errors / mean_curve)),
            ],
            "bulk_block_shift_over_mean_range": [
                float(np.min(block_shifts / mean_curve)),
                float(np.max(block_shifts / mean_curve)),
            ],
            "largest_hat_q_seen": float(
                max(float(point["largest_hat_q_seen"]) for point in points)
            ),
        },
    }


def write_csv(path: Path, manifest: dict[str, object], analysis: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "t",
                "minus_imag_I_1_3",
                "rqmc_standard_error",
                "tail_fraction",
                "tail_fit_relative_residual",
                "fitted_tail_exponent",
                "bulk_block_order_shift_abs",
            ]
        )
        for point, t, value, error in zip(
            manifest["points"],
            analysis["t_values"],
            analysis["mean_curve"],
            analysis["rqmc_standard_errors"],
        ):
            writer.writerow(
                [
                    t,
                    value,
                    error,
                    point["tail_fraction"],
                    point["tail_fit_relative_residual"],
                    point["fitted_tail_exponent"],
                    abs(float(point["bulk_block_order_shift"]["imag"])),
                ]
            )


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    return out


def main() -> None:
    args = parser().parse_args()
    manifest_path = args.scan_dir / "worldsheet_scan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = analyze_manifest(manifest)
    output = args.scan_dir / "worldsheet_shape_analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.scan_dir / "worldsheet_t_dependence.csv", manifest, result)
    peak = result["local_quadratic_peak"]
    print(f"wrote {output}")
    print(
        "descriptive peak t={:.6f}, replicate SE={:.6f}".format(
            peak["replicate_mean_t"],
            peak["replicate_standard_error_t"],
        )
    )


if __name__ == "__main__":
    main()
