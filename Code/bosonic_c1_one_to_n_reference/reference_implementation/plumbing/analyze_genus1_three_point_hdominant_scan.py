#!/usr/bin/env python3
"""Validate, analyze, freeze, and post-compare the h-dominant t scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CHANNELS = ("necklace", "pair_ope", "comb_ope")
EXPECTED_T = tuple(round(0.05 + 0.10 * index, 2) for index in range(10))
EXPECTED_DESIGN = {
    "momentum_order": 8,
    "necklace_order": 8,
    "necklace_low_order": 3,
    "necklace_backend": "regulated-h-recursion",
    "necklace_c_regulator": 0.025,
    "necklace_qhat_threshold": 0.30,
    "necklace_second_qhat_threshold": 0.07,
    "ope_order": 6,
    "ope_loop_order": 4,
    "ope_low_loop_order": 2,
    "evaluation_order_cap": None,
    "sobol_power": 8,
    "replicates": 4,
    "position_alpha": 0.3,
    "patch_epsilon": 0.15,
    "triple_patch_epsilon": 0.10,
    "tail_integrated_directly_to_infinity": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def artifact_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def complex_value(record: dict[str, float]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def validate_point(data: dict[str, object], expected_t: float) -> None:
    if data.get("matrix_model_present") is not False:
        raise ValueError(f"t={expected_t}: point is not target-blind")
    if abs(float(data["t"]) - expected_t) > 1.0e-12:
        raise ValueError(f"t={expected_t}: kinematic mismatch")
    design = data["design"]
    for key, expected in EXPECTED_DESIGN.items():
        if design.get(key) != expected:
            raise ValueError(f"t={expected_t}: design mismatch for {key}")
    for key in ("mean", "rqmc_standard_error"):
        value = complex_value(data[key])
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError(f"t={expected_t}: non-finite {key}")


def validate_freeze(
    scan_dir: Path,
    reused_t075: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assembled = scan_dir / "assembled"
    shards = scan_dir / "shards"
    complete = json.loads((assembled / "RUN_COMPLETE.json").read_text())
    freeze = json.loads((assembled / "worldsheet_freeze_manifest.json").read_text())
    manifest_path = assembled / "worldsheet_scan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if complete.get("status") != "complete" or complete.get("comparison_performed") is not False:
        raise ValueError("remote assembly is not a completed blind run")
    if freeze.get("status") != "blind_worldsheet_scan_frozen":
        raise ValueError("remote freeze status is invalid")
    if freeze.get("matrix_model_present") is not False or freeze.get("point_count") != 10:
        raise ValueError("remote freeze is not a ten-point target-blind freeze")
    if manifest.get("matrix_model_present") is not False or manifest.get("complete") is not True:
        raise ValueError("scan manifest is not complete and target-blind")
    if tuple(float(value) for value in manifest["t_values"]) != EXPECTED_T:
        raise ValueError("scan t design changed")
    if manifest["design"] != EXPECTED_DESIGN:
        raise ValueError("scan numerical design changed")

    local_by_name = {
        "worldsheet_scan_manifest.json": manifest_path,
        "worldsheet_t_dependence.csv": assembled / "worldsheet_t_dependence.csv",
        "t075.json": reused_t075,
    }
    local_by_name.update({f"t{int(round(100*t)):03d}.json": shards / f"t{int(round(100*t)):03d}.json" for t in EXPECTED_T if t != 0.75})
    validated_artifacts = []
    for record in freeze["artifacts"]:
        name = Path(record["path"]).name
        local = local_by_name[name]
        if not local.is_file() or sha256(local) != record["sha256"]:
            raise ValueError(f"freeze digest mismatch for {name}")
        validated_artifacts.append(artifact_record(local))

    for t, point in zip(EXPECTED_T, manifest["points"]):
        source = reused_t075 if t == 0.75 else shards / f"t{int(round(100*t)):03d}.json"
        source_data = json.loads(source.read_text())
        validate_point(source_data, t)
        if sha256(source) != point["source_sha256"]:
            raise ValueError(f"manifest source digest mismatch at t={t}")
        if source_data["mean"] != point["I_1,3"]:
            raise ValueError(f"assembled central value mismatch at t={t}")
    return manifest, validated_artifacts


def quadratic_peak(t: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    coefficients = np.polyfit(t, values, 2)
    if coefficients[0] >= 0.0:
        raise ValueError("local peak fit is not concave")
    location = float(-coefficients[1] / (2.0 * coefficients[0]))
    return location, float(np.polyval(coefficients, location))


def blind_analysis(manifest: dict[str, object]) -> dict[str, object]:
    points = manifest["points"]
    t_values = np.asarray(manifest["t_values"], dtype=float)
    means = np.asarray([-float(point["I_1,3"]["imag"]) for point in points])
    errors = np.asarray([float(point["rqmc_standard_error"]["imag"]) for point in points])
    replicates = np.asarray(
        [[-float(value["imag"]) for value in point["replicate_values"]] for point in points]
    ).T
    recomputed_errors = np.std(replicates, axis=0, ddof=1) / math.sqrt(replicates.shape[0])
    if not np.allclose(errors, recomputed_errors, rtol=5.0e-14, atol=1.0e-18):
        raise ValueError("stored RQMC errors do not match the four replicates")

    maximum_index = int(np.argmax(means))
    local = slice(maximum_index - 1, maximum_index + 2)
    peak_t, peak_height = quadratic_peak(t_values[local], means[local])
    replicate_peak_t = np.asarray(
        [quadratic_peak(t_values[local], row[local])[0] for row in replicates]
    )
    peak_t_se = float(np.std(replicate_peak_t, ddof=1) / math.sqrt(len(replicate_peak_t)))

    adjacent = np.diff(replicates, axis=1)
    channel_totals = {
        channel: np.asarray(
            [
                sum(-float(point["channel_components"][region][channel]["mean"]["imag"]) for region in ("bulk", "tail"))
                for point in points
            ]
        )
        for channel in CHANNELS
    }
    tail_totals = np.asarray(
        [
            sum(-float(point["channel_components"]["tail"][channel]["mean"]["imag"]) for channel in CHANNELS)
            for point in points
        ]
    )
    return {
        "calculation": "target-blind shape analysis of the h-dominant genus-one three-point scan",
        "status": "blind_worldsheet_shape_analysis_complete",
        "matrix_model_present": False,
        "ordinate": "-Im I_1,3(it/2,it/2)",
        "t_values": t_values.tolist(),
        "mean_curve": means.tolist(),
        "rqmc_standard_errors": errors.tolist(),
        "relative_rqmc_standard_errors": (errors / means).tolist(),
        "replicate_curves": replicates.tolist(),
        "common_scramble_covariance": np.cov(replicates, rowvar=False, ddof=1).tolist(),
        "shape": {
            "strictly_increasing_through_t": float(t_values[maximum_index]),
            "strictly_decreasing_after_t": float(t_values[maximum_index]),
            "discrete_maximum": {"t": float(t_values[maximum_index]), "value": float(means[maximum_index])},
            "local_quadratic_peak": {
                "fit_t_values": t_values[local].tolist(),
                "central_t": peak_t,
                "central_height": peak_height,
                "replicate_t_values": replicate_peak_t.tolist(),
                "replicate_mean_t": float(np.mean(replicate_peak_t)),
                "replicate_standard_error_t": peak_t_se,
                "status": "descriptive three-point local fit; no truncation systematic assigned",
            },
            "adjacent_changes": [
                {
                    "t_left": float(t_values[index]),
                    "t_right": float(t_values[index + 1]),
                    "mean": float(np.mean(adjacent[:, index])),
                    "common_scramble_standard_error": float(np.std(adjacent[:, index], ddof=1) / math.sqrt(replicates.shape[0])),
                }
                for index in range(adjacent.shape[1])
            ],
        },
        "channel_decomposition": {
            channel: {
                "total_curve": channel_totals[channel].tolist(),
                "fraction_of_total": (channel_totals[channel] / means).tolist(),
            }
            for channel in CHANNELS
        },
        "tail": {
            "total_curve": tail_totals.tolist(),
            "fraction_of_total": (tail_totals / means).tolist(),
            "fraction_range": [float(np.min(tail_totals / means)), float(np.max(tail_totals / means))],
        },
        "numerical_diagnostics": {
            "relative_rqmc_error_range": [float(np.min(errors / means)), float(np.max(errors / means))],
            "largest_hat_q_seen": float(max(point["necklace_diagnostics"]["largest_hat_q_seen"] for point in points)),
            "largest_second_hat_q_seen": float(max(point["necklace_diagnostics"]["largest_second_hat_q_seen"] for point in points)),
            "maximum_ope_abs_hat_q": float(max(point["atlas_diagnostics"]["maximum_ope_abs_hat_q"] for point in points)),
            "c_recursion_collision_count": int(sum(point["atlas_diagnostics"]["c_recursion_collision_count"] for point in points)),
            "channel_counts_per_point": points[0]["atlas_diagnostics"]["channel_counts"],
            "adaptive_order_histogram_per_point": points[0]["necklace_diagnostics"]["adaptive_order_histogram"],
        },
    }


def matrix_ordinate(t: float) -> float:
    stripped = (t - 1.0) * (t - 2.0) * (1.0 + t - 0.5 * t * t) / 48.0
    return 2.0 * math.pi * t**3 * stripped


def post_freeze_comparison(
    analysis: dict[str, object],
    analysis_freeze: Path,
) -> dict[str, object]:
    t_values = np.asarray(analysis["t_values"], dtype=float)
    worldsheet = np.asarray(analysis["mean_curve"], dtype=float)
    errors = np.asarray(analysis["rqmc_standard_errors"], dtype=float)
    matrix = np.asarray([matrix_ordinate(t) for t in t_values])
    ratios = worldsheet / matrix
    pulls = (worldsheet - matrix) / errors
    dense_t = np.linspace(0.0, 1.0, 1001)[1:-1]
    dense_matrix = np.asarray([matrix_ordinate(t) for t in dense_t])
    matrix_peak_index = int(np.argmax(dense_matrix))
    return {
        "calculation": "post-freeze BRY-normalized matrix comparison for the genus-one three-point t scan",
        "status": "post_freeze_comparison_complete",
        "worldsheet_was_frozen_before_comparison": True,
        "blind_analysis_freeze": artifact_record(analysis_freeze),
        "normalization": {
            "coupling_dictionary": "mu^-1=2*pi*g_s^BRY",
            "matrix_genus_one": "F1_MM=2*F0*S_hat_1,3",
            "equal_split_kinematics": "omega_in=i*t, omega_out_1=omega_out_2=i*t/2",
            "F0": "omega_in*omega_out_1*omega_out_2",
            "S_hat_1,3_equal_split": "(t-1)*(t-2)*(1+t-t^2/2)/48",
            "worldsheet_ordinate": "-Im I_1,3",
            "matrix_ordinate": "-Im I_1,3^MM=2*pi*t^3*S_hat_1,3",
        },
        "t_values": t_values.tolist(),
        "worldsheet": worldsheet.tolist(),
        "rqmc_standard_errors": errors.tolist(),
        "matrix_model": matrix.tolist(),
        "worldsheet_over_matrix": ratios.tolist(),
        "rqmc_pulls": pulls.tolist(),
        "summary": {
            "ratio_range": [float(np.min(ratios)), float(np.max(ratios))],
            "maximum_absolute_pointwise_rqmc_pull": float(np.max(np.abs(pulls))),
            "all_points_within_one_rqmc_standard_error": bool(np.all(np.abs(pulls) < 1.0)),
            "worldsheet_peak_t_local_quadratic": analysis["shape"]["local_quadratic_peak"]["central_t"],
            "worldsheet_peak_t_replicate_standard_error": analysis["shape"]["local_quadratic_peak"]["replicate_standard_error_t"],
            "matrix_peak_t_dense_grid": float(dense_t[matrix_peak_index]),
            "interpretation": "pointwise RQMC comparison only; no calibrated h/OPE/momentum truncation systematic has been assigned",
        },
    }


def write_curve_csv(path: Path, analysis: dict[str, object], comparison: dict[str, object] | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["t", "minus_imag_I_1_3", "rqmc_standard_error"]
        if comparison is not None:
            header.extend(["matrix_model", "worldsheet_over_matrix", "rqmc_pull"])
        writer.writerow(header)
        for index, t in enumerate(analysis["t_values"]):
            row = [t, analysis["mean_curve"][index], analysis["rqmc_standard_errors"][index]]
            if comparison is not None:
                row.extend([comparison["matrix_model"][index], comparison["worldsheet_over_matrix"][index], comparison["rqmc_pulls"][index]])
            writer.writerow(row)


def write_comparison_plot(
    svg_path: Path,
    png_path: Path,
    analysis: dict[str, object],
    comparison: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    t_values = np.asarray(analysis["t_values"], dtype=float)
    worldsheet = np.asarray(analysis["mean_curve"], dtype=float)
    errors = np.asarray(analysis["rqmc_standard_errors"], dtype=float)
    dense_t = np.linspace(0.001, 0.999, 800)
    dense_matrix = np.asarray([matrix_ordinate(t) for t in dense_t])
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.errorbar(
        t_values,
        worldsheet,
        yerr=errors,
        fmt="o",
        linestyle="none",
        markersize=5.5,
        capsize=3,
        label="Worldsheet (RQMC)",
        zorder=3,
    )
    axis.plot(dense_t, dense_matrix, linewidth=2.0, label="Matrix model (BRY normalization)")
    axis.set_xlabel(r"$t$ in $\omega_{\rm in}=it$")
    axis.set_ylabel(r"$-\mathrm{Im}\, I_{1,3}$")
    axis.set_title("Genus-one three-point amplitude")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(svg_path)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--reused-t075", type=Path, required=True)
    args = parser.parse_args()
    output = args.scan_dir / "assembled"
    manifest, validated = validate_freeze(args.scan_dir, args.reused_t075)
    analysis = blind_analysis(manifest)
    analysis_path = output / "worldsheet_shape_analysis_blind.json"
    analysis_csv = output / "worldsheet_shape_analysis_blind.csv"
    write_json(analysis_path, analysis)
    write_curve_csv(analysis_csv, analysis)
    analysis_freeze_path = output / "worldsheet_analysis_freeze_manifest.json"
    write_json(
        analysis_freeze_path,
        {
            "status": "blind_worldsheet_analysis_frozen",
            "matrix_model_present": False,
            "validated_input_artifacts": validated,
            "blind_analysis_artifacts": [artifact_record(analysis_path), artifact_record(analysis_csv)],
        },
    )
    comparison = post_freeze_comparison(analysis, analysis_freeze_path)
    comparison_path = output / "matrix_comparison_after_blind_freeze.json"
    comparison_csv = output / "worldsheet_vs_matrix_bry.csv"
    comparison_svg = output / "worldsheet_vs_matrix_bry.svg"
    comparison_png = output / "worldsheet_vs_matrix_bry.png"
    write_json(comparison_path, comparison)
    write_curve_csv(comparison_csv, analysis, comparison)
    write_comparison_plot(comparison_svg, comparison_png, analysis, comparison)
    print(json.dumps({
        "analysis": str(analysis_path),
        "analysis_freeze": str(analysis_freeze_path),
        "comparison": str(comparison_path),
        "comparison_csv": str(comparison_csv),
        "comparison_svg": str(comparison_svg),
        "comparison_png": str(comparison_png),
        "peak": analysis["shape"]["local_quadratic_peak"],
        "comparison_summary": comparison["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
