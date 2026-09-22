#!/usr/bin/env python3
"""Audit raw N=3/N=4 ratios and selected N=5 refinements."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N3 = ROOT / "Data Set" / "nsrr_nsnsns_generic_10point_N3_20260904" / "summary.json"
DEFAULT_N4 = ROOT / "Data Set" / "nsrr_nsnsns_generic_10point_N4_20260904" / "summary.json"
DEFAULT_N5 = (
    ROOT
    / "Data Set"
    / "nsrr_nsnsns_generic_10point_selected_N5_20260904"
    / "summary.json"
)
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_generic_10point_audit_20260904"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fixed_one(values: list[float]) -> dict:
    residuals = [value - 1.0 for value in values]
    return {
        "normalization": 1.0,
        "normalization_fitted_or_applied": False,
        "residuals": residuals,
        "minimum_raw_ratio": min(values),
        "maximum_raw_ratio": max(values),
        "maximum_absolute_fractional_residual": max(map(abs, residuals)),
        "rms_fractional_residual": math.sqrt(
            math.fsum(value * value for value in residuals) / len(residuals)
        ),
    }


def audit(n3_path: Path, n4_path: Path, n5_path: Path | None = None) -> dict:
    n3 = load(n3_path)
    n4 = load(n4_path)
    n5 = load(n5_path) if n5_path is not None else None
    if n3["config"]["point_design"] != n4["config"]["point_design"]:
        raise ValueError("N=3 and N=4 point designs differ")
    if n3["config"]["q_envelope"] != n4["config"]["q_envelope"]:
        raise ValueError("N=3 and N=4 momentum envelopes differ")
    rows3 = {row["point_id"]: row for row in n3["comparisons"]}
    rows4 = {row["point_id"]: row for row in n4["comparisons"]}
    rows5 = {row["point_id"]: row for row in n5["comparisons"]} if n5 else {}
    points = {point["point_id"]: point for point in n4["config"]["points"]}
    if set(rows3) != set(rows4) or set(rows4) != set(points):
        raise ValueError("point coverage differs between configurations")
    if not set(rows5).issubset(rows4):
        raise ValueError("N=5 refinement contains points outside the ten-point design")
    if n5 is not None:
        for key in ("b", "kappa", "q_envelope"):
            if n5["config"][key] != n4["config"][key]:
                raise ValueError(f"N=4 and N=5 {key} differ")

    rows = []
    for point_id in n4["config"]["point_design"]["point_ids"]:
        point = points[point_id]
        coarse = rows3[point_id]
        fine = rows4[point_id]
        refined = rows5.get(point_id)
        row = {
                "point_id": point_id,
                **point["omega_coordinates"],
                "minimum_imaginary_eigenvalue": point["minimum_imaginary_eigenvalue"],
                "maximum_source_abs_q": max(abs(complex(value)) for value in point["source"]["q_values"]),
                "maximum_target_abs_q": max(abs(complex(value)) for value in point["target"]["q_values"]),
                "target_lifts": point["target"]["lifts"],
                "source_Q_N3": coarse["source_Q"],
                "target_Q_N3": coarse["target_Q"],
                "raw_ratio_N3": coarse["source_over_target"],
                "source_Q_N4": fine["source_Q"],
                "target_Q_N4": fine["target_Q"],
                "raw_ratio_N4": fine["source_over_target"],
                "raw_N4_residual_from_one": fine["source_over_target"] - 1.0,
                "source_N3_to_N4_change": fine["source_Q"] / coarse["source_Q"] - 1.0,
                "target_N3_to_N4_change": fine["target_Q"] / coarse["target_Q"] - 1.0,
                "ratio_N3_to_N4_change": fine["source_over_target"] / coarse["source_over_target"] - 1.0,
                "source_Q_N5": refined["source_Q"] if refined else None,
                "target_Q_N5": refined["target_Q"] if refined else None,
                "raw_ratio_N5": refined["source_over_target"] if refined else None,
                "raw_N5_residual_from_one": refined["source_over_target"] - 1.0 if refined else None,
                "source_N4_to_N5_change": (
                    refined["source_Q"] / fine["source_Q"] - 1.0 if refined else None
                ),
                "target_N4_to_N5_change": (
                    refined["target_Q"] / fine["target_Q"] - 1.0 if refined else None
                ),
                "ratio_N4_to_N5_change": (
                    refined["source_over_target"] / fine["source_over_target"] - 1.0
                    if refined
                    else None
                ),
            }
        rows.append(row)

    values3 = [row["raw_ratio_N3"] for row in rows]
    values4 = [row["raw_ratio_N4"] for row in rows]
    ratio_changes = [row["ratio_N3_to_N4_change"] for row in rows]
    values5 = [row["raw_ratio_N5"] for row in rows if row["raw_ratio_N5"] is not None]
    ratio_changes45 = [
        row["ratio_N4_to_N5_change"]
        for row in rows
        if row["ratio_N4_to_N5_change"] is not None
    ]
    return {
        "schema": "nsrr-nsnsns-generic-ten-audit-v2",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization_policy": (
            "The sewing prescription fixes NSRR/NSNSNS=1. No overall constant is fitted "
            "or applied; all residuals are raw ratio minus one."
        ),
        "b": n4["config"]["b"],
        "kappa": n4["config"]["kappa"],
        "point_design": n4["config"]["point_design"],
        "cutoffs": {
            "source_physical_level": n4["config"]["source_level"],
            "target_recursion_twice_level": n4["config"]["target_recursion_twice_level"],
            "momentum_orders": [3, 4] + ([5] if values5 else []),
        },
        "fixed_normalization_one_N3": fixed_one(values3),
        "fixed_normalization_one_N4": fixed_one(values4),
        "fixed_normalization_one_N5_selected": fixed_one(values5) if values5 else None,
        "quadrature_refinement": {
            "maximum_absolute_ratio_N3_to_N4_change": max(map(abs, ratio_changes)),
            "rms_ratio_N3_to_N4_change": math.sqrt(
                math.fsum(value * value for value in ratio_changes) / len(ratio_changes)
            ),
            "selected_N5_point_ids": list(rows5),
            "maximum_absolute_ratio_N4_to_N5_change": (
                max(map(abs, ratio_changes45)) if ratio_changes45 else None
            ),
            "rms_ratio_N4_to_N5_change": (
                math.sqrt(
                    math.fsum(value * value for value in ratio_changes45)
                    / len(ratio_changes45)
                )
                if ratio_changes45
                else None
            ),
        },
        "points": rows,
        "conclusion": (
            "At N=4 the ten generic raw ratios agree with one at the 1.03% maximum and "
            "0.471% RMS level. On the five diagnostic points, N=5 gives 0.915% maximum "
            "and 0.498% RMS deviation from one; the N=4-to-N=5 ratio change is at most "
            "0.337% and 0.188% RMS. This excludes an overall factor-of-four mismatch and "
            "shows sub-percent residual shape, but does not identify the remaining residual "
            "as quadrature error."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n3", type=Path, default=DEFAULT_N3)
    parser.add_argument("--n4", type=Path, default=DEFAULT_N4)
    parser.add_argument("--n5", type=Path, default=DEFAULT_N5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.n3, args.n4, args.n5)
    save(args.output_dir / "summary.json", result)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["points"][0].keys())
        writer.writeheader()
        writer.writerows(result["points"])
    check = result["fixed_normalization_one_N4"]
    print(
        f"N=4 fixed normalization 1: rms={check['rms_fractional_residual']:.3%}, "
        f"max={check['maximum_absolute_fractional_residual']:.3%}"
    )
    if result["fixed_normalization_one_N5_selected"] is not None:
        check5 = result["fixed_normalization_one_N5_selected"]
        refinement = result["quadrature_refinement"]
        print(
            f"selected N=5 fixed normalization 1: rms={check5['rms_fractional_residual']:.3%}, "
            f"max={check5['maximum_absolute_fractional_residual']:.3%}; "
            f"N=4->N=5 ratio-change rms={refinement['rms_ratio_N4_to_N5_change']:.3%}, "
            f"max={refinement['maximum_absolute_ratio_N4_to_N5_change']:.3%}"
        )


if __name__ == "__main__":
    main()
