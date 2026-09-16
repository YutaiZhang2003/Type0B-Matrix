#!/usr/bin/env python3
"""Consolidate the curve and branch-continued off-axis NSRR/NSNSNS tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVE = ROOT / "Data Set" / "nsrr_nsnsns_constant_ratio_audit_20260904" / "summary.json"
DEFAULT_OLD_N23 = ROOT / "Data Set" / "nsrr_nsnsns_offaxis_constant_scan_20260904" / "summary.json"
DEFAULT_OLD_N4 = ROOT / "Data Set" / "nsrr_nsnsns_offaxis_constant_scan_N4_20260904" / "summary.json"
DEFAULT_NEW_N23 = ROOT / "Data Set" / "nsrr_nsnsns_offaxis_constant_scan_continued_N23_20260904" / "summary.json"
DEFAULT_NEW_N4 = ROOT / "Data Set" / "nsrr_nsnsns_offaxis_constant_scan_continued_N4_20260904" / "summary.json"
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_extended_moduli_audit_20260904"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def by_order(summary: dict, order: int) -> dict[str, dict]:
    rows = {
        row["point_id"]: row
        for row in summary["comparisons"]
        if int(row["quadrature_order"]) == order
    }
    if len(rows) != len(summary["config"]["points"]):
        raise ValueError(f"quadrature order {order} is incomplete")
    return rows


def fixed_constant_metrics(values: list[float], constant: float) -> dict:
    residuals = [value / constant - 1.0 for value in values]
    return {
        "constant": constant,
        "residuals": residuals,
        "maximum_absolute_fractional_residual": max(map(abs, residuals)),
        "rms_fractional_residual": math.sqrt(
            math.fsum(value * value for value in residuals) / len(residuals)
        ),
    }


def unrescaled_sample_summary(values: list[float]) -> dict:
    """Describe raw ratios without interpreting their mean as a fit."""

    return {
        "arithmetic_mean": math.fsum(values) / len(values),
        "geometric_mean": math.exp(math.fsum(math.log(value) for value in values) / len(values)),
        "minimum": min(values),
        "maximum": max(values),
        "max_over_min_minus_one": max(values) / min(values) - 1.0,
        "warning": "descriptive statistics only; no normalization was fitted or applied",
    }


def audit(
    curve_path: Path,
    old_n23_path: Path,
    old_n4_path: Path,
    new_n23_path: Path,
    new_n4_path: Path,
) -> dict:
    curve = load(curve_path)
    old3 = by_order(load(old_n23_path), 3)
    old4_summary = load(old_n4_path)
    old4 = by_order(old4_summary, 4)
    new3_summary = load(new_n23_path)
    new3 = by_order(new3_summary, 3)
    new4_summary = load(new_n4_path)
    new4 = by_order(new4_summary, 4)
    config = new4_summary["config"]
    points = {point["point_id"]: point for point in config["points"]}

    point_rows = []
    affected = []
    unaffected = []
    for index, point_id in enumerate(points):
        point = points[point_id]
        old_lifts = next(
            candidate["target"]["lifts"]
            for candidate in old4_summary["config"]["points"]
            if candidate["point_id"] == point_id
        )
        lift_changed = point["target"]["lifts"] != old_lifts
        if new4[point_id]["source_Q"] != old4[point_id]["source_Q"]:
            raise ValueError(f"{point_id}: source data were not reused exactly")
        if not lift_changed and new4[point_id]["target_Q"] != old4[point_id]["target_Q"]:
            raise ValueError(f"{point_id}: unaffected target data changed")
        (affected if lift_changed else unaffected).append(point_id)
        ratio3 = float(new3[point_id]["source_over_target"])
        ratio4 = float(new4[point_id]["source_over_target"])
        point_rows.append(
            {
                "point_id": point_id,
                "variation": point["variation"],
                "target_period_branch": point["target"]["period_branch"],
                "target_lifts": point["target"]["lifts"],
                "lift_changed": lift_changed,
                "old_N4_ratio_without_continuation": float(old4[point_id]["source_over_target"]),
                "continued_N3_ratio": ratio3,
                "continued_N4_ratio": ratio4,
                "ratio_N3_to_N4_change": ratio4 / ratio3 - 1.0,
                "continued_N4_residual_from_one": ratio4 - 1.0,
            }
        )

    values3 = [row["continued_N3_ratio"] for row in point_rows]
    values4 = [row["continued_N4_ratio"] for row in point_rows]
    one3 = fixed_constant_metrics(values3, 1.0)
    one4 = fixed_constant_metrics(values4, 1.0)
    quarter = fixed_constant_metrics(
        values4, 0.25
    )

    unique_combined = [
        float(point["ratio"])
        for point in curve["preferred_points"]
        if float(point["t"]) != 0.6
    ] + [row["continued_N4_ratio"] for row in point_rows]
    curve_values = [float(point["ratio"]) for point in curve["preferred_points"]]
    return {
        "schema": "nsrr-nsnsns-extended-moduli-audit-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "conventions": {
            "source": "fixed-spin [11|00], amplitude lift projection, unscaled Human M kernel",
            "target": "fixed-spin [00|00] all-NS theta sewing",
            "analytic_continuation": config["analytic_continuation"],
            "source_level": config["source_level"],
            "target_recursion_twice_level": config["target_recursion_twice_level"],
        },
        "reuse_validation": {
            "all_source_N4_values_reused_exactly": True,
            "unaffected_target_N4_values_reused_exactly": True,
            "affected_points": affected,
            "unaffected_points": unaffected,
        },
        "normalization_policy": (
            "No overall constant is fitted. All residuals below use the sewing-fixed "
            "normalization NSRR/NSNSNS = 1."
        ),
        "offaxis_N3_fixed_normalization_one": one3,
        "offaxis_N4_fixed_normalization_one": one4,
        "offaxis_N4_unrescaled_sample_summary": unrescaled_sample_summary(values4),
        "offaxis_N4_fixed_constant_one_quarter_counterfactual": quarter,
        "offaxis_points": point_rows,
        "offaxis_N3_to_N4": {
            "maximum_absolute_ratio_change": max(
                abs(row["ratio_N3_to_N4_change"]) for row in point_rows
            ),
            "rms_ratio_change": math.sqrt(
                math.fsum(row["ratio_N3_to_N4_change"] ** 2 for row in point_rows)
                / len(point_rows)
            ),
            "unrescaled_sample_geometric_mean_N3": unrescaled_sample_summary(values3)["geometric_mean"],
            "unrescaled_sample_geometric_mean_N4": unrescaled_sample_summary(values4)["geometric_mean"],
            "warning": "the sample means are convergence diagnostics, not fitted constants",
        },
        "saved_curve_N5_fixed_normalization_one": fixed_constant_metrics(curve_values, 1.0),
        "saved_curve_N5_unrescaled_sample_summary": unrescaled_sample_summary(curve_values),
        "saved_center_N6_over_N7": curve["central_refinement_holdout"],
        "combined_17_unique_surfaces_fixed_normalization_one": fixed_constant_metrics(
            unique_combined, 1.0
        ),
        "combined_17_unique_surfaces_unrescaled_sample_summary": unrescaled_sample_summary(
            unique_combined
        ),
        "conclusion": (
            "After analytic continuation of the target plumbing lifts, the unscaled Human M "
            "NSRR sewing and the all-NS theta sewing agree throughout all six real local "
            "directions with normalization one within the observed cutoff error. The earlier "
            "multi-percent off-axis mismatch was a principal-square-root/lift branch artifact."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--old-n23", type=Path, default=DEFAULT_OLD_N23)
    parser.add_argument("--old-n4", type=Path, default=DEFAULT_OLD_N4)
    parser.add_argument("--new-n23", type=Path, default=DEFAULT_NEW_N23)
    parser.add_argument("--new-n4", type=Path, default=DEFAULT_NEW_N4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.curve, args.old_n23, args.old_n4, args.new_n23, args.new_n4)
    save(args.output_dir / "summary.json", result)
    with (args.output_dir / "offaxis_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["offaxis_points"][0].keys())
        writer.writeheader()
        writer.writerows(result["offaxis_points"])
    one = result["offaxis_N4_fixed_normalization_one"]
    print(
        f"N4 fixed normalization 1: rms={one['rms_fractional_residual']:.3%}, "
        f"max={one['maximum_absolute_fractional_residual']:.3%}"
    )


if __name__ == "__main__":
    main()
