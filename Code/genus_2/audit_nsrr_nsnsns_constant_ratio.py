#!/usr/bin/env python3
"""Test whether saved NSRR/NSNSNS data differ by one constant.

This audit treats the Human-block ``M`` contraction (the convention whose
identity-degeneration ground expression is eight) as the source candidate.
It does not choose or fit the absolute CFT normalization.  Instead it asks
whether one multiplicative constant removes the moduli dependence on the
five saved surfaces, and repeats the fit over all reusable level and
quadrature cutoffs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_SOURCE = ROOT / "Data Set" / "nsrr_nsnsns_human_convention_20260903" / "summary.json"
DEFAULT_TARGET = ROOT / "Data Set" / "nsrr_nsnsns_target_R8_R12_R16_N5_20260830" / "summary.json"
DEFAULT_OLD_SCAN = ROOT / "Data Set" / "nsrr_nsnsns_fivepoint_L4_N5_20260830" / "summary.json"
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_constant_ratio_audit_20260904"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def constant_fit(points: list[dict]) -> dict:
    """Equal-fractional-weight fit of ``source = A * target``."""

    if len(points) < 2:
        raise ValueError("at least two points are required")
    ratios = [float(point["ratio"]) for point in points]
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("ratios must be finite and positive")
    log_values = [math.log(value) for value in ratios]
    log_a = statistics.fmean(log_values)
    normalization = math.exp(log_a)
    residuals = [value / normalization - 1.0 for value in ratios]

    coordinates = [float(point["coordinate"]) for point in points]
    x_mean = statistics.fmean(coordinates)
    variance = math.fsum((value - x_mean) ** 2 for value in coordinates)
    slope = (
        math.fsum((x - x_mean) * (y - log_a) for x, y in zip(coordinates, log_values))
        / variance
        if variance
        else 0.0
    )
    linear_predictions = [math.exp(log_a + slope * (x - x_mean)) for x in coordinates]
    linear_residuals = [value / prediction - 1.0 for value, prediction in zip(ratios, linear_predictions)]
    return {
        "normalization_geometric_mean": normalization,
        "normalization_arithmetic_mean": statistics.fmean(ratios),
        "residuals": residuals,
        "maximum_absolute_fractional_residual": max(abs(value) for value in residuals),
        "rms_fractional_residual": math.sqrt(statistics.fmean(value * value for value in residuals)),
        "minimum_ratio": min(ratios),
        "maximum_ratio": max(ratios),
        "max_over_min_minus_one": max(ratios) / min(ratios) - 1.0,
        "linear_log_slope_per_coordinate": slope,
        "linear_change_over_scan": math.exp(slope * (max(coordinates) - min(coordinates))) - 1.0,
        "linear_detrended_maximum_absolute_fractional_residual": max(
            abs(value) for value in linear_residuals
        ),
        "linear_detrended_rms_fractional_residual": math.sqrt(
            statistics.fmean(value * value for value in linear_residuals)
        ),
    }


def _target_lookup(target: dict) -> dict[float, dict]:
    rows = [
        row
        for row in target["rows"]
        if int(row["quadrature_order"]) == 5 and int(row["recursion_order"]) == 16
    ]
    result = {float(row["t"]): row for row in rows}
    if len(result) != 5:
        raise ValueError("expected five R16/N5 target rows")
    return result


def audit(source_path: Path, target_path: Path, old_scan_path: Path) -> dict:
    source = load(source_path)
    target = _target_lookup(load(target_path))
    old_scan = load(old_scan_path)
    comparisons = {float(row["t"]): row for row in source["comparisons"]}
    if set(comparisons) != set(target):
        raise ValueError("source and target surface designs differ")

    # Q/Z is fixed entirely by the already checked same-frame free factor.
    source_q_per_z = {
        t: float(row["source_Q_L3_N5"]) / float(row["source_Z_L3_N5"])
        for t, row in comparisons.items()
    }
    # The historical target summary stores Q with its earlier free adapter.
    # The repaired comparison recomputes Q from the same target numerator and
    # the independently fixed-spin target free factor; use that matched value.
    target_q = {
        t: float(row["target_Q_R16_N5_fixed_free"])
        for t, row in comparisons.items()
    }
    for t, row in target.items():
        if not math.isclose(
            float(row["target_Z"]),
            float(comparisons[t]["target_Z_R16_N5_reused"]),
            rel_tol=2.0e-15,
        ):
            raise ValueError("the reused all-NS target numerator changed")
    target_quadrature_change = {
        float(row["t"]): abs(float(row["target_quadrature_relative_change"]))
        for row in old_scan["convergence_diagnostics"]
    }

    fits = []
    point_rows = []

    def add_fit(label: str, rows: list[dict], value_key: str) -> None:
        points = []
        for row in sorted(rows, key=lambda item: float(item["t"])):
            t = float(row["t"])
            source_q = float(row[value_key]) * source_q_per_z[t]
            ratio = source_q / target_q[t]
            points.append({"coordinate": t, "ratio": ratio})
            point_rows.append(
                {
                    "fit": label,
                    "t": t,
                    "ratio": ratio,
                    "source_Q": source_q,
                    "target_Q": target_q[t],
                }
            )
        fits.append({"fit": label, **constant_fit(points)})

    l3_rows = source["source_rows_L3"]
    for order in (3, 4, 5):
        for level in (0.0, 1.0, 2.0, 3.0):
            selected = [
                row
                for row in l3_rows
                if int(row["quadrature_order"]) == order and float(row["level"]) == level
            ]
            add_fit(f"M_source_L{level:g}_N{order}_vs_target_R16_N5", selected, "source_Z")

    l5_rows = source["source_rows_L5"]
    for level in (3.0, 4.0, 5.0):
        selected = [
            row
            for row in l5_rows
            if int(row["quadrature_order"]) == 3 and float(row["level"]) == level
        ]
        add_fit(f"M_source_L{level:g}_N3_deep_vs_target_R16_N5", selected, "source_Z")

    best = [
        row
        for row in l3_rows
        if int(row["quadrature_order"]) == 5 and float(row["level"]) == 3.0
    ]
    for label, key in (
        ("M_correct_interference_L3_N5", "source_Z"),
        ("diagonal_only_L3_N5", "diagonal_Z"),
        ("opposite_interference_L3_N5", "opposite_interference_sign_Z"),
    ):
        add_fit(label, best, key)

    preferred_label = "M_correct_interference_L3_N5"
    preferred = next(item for item in fits if item["fit"] == preferred_label)
    preferred_points = [row for row in point_rows if row["fit"] == preferred_label]
    for row, residual in zip(preferred_points, preferred["residuals"]):
        t = float(row["t"])
        comparison = comparisons[t]
        row["constant_fit_residual"] = residual
        row["source_level_L2_to_L3_change"] = abs(
            float(comparison["source_level_L2_to_L3_relative_change_N5"])
        )
        row["source_quadrature_N4_to_N5_change"] = abs(
            float(comparison["source_quadrature_N4_to_N5_relative_change_L3"])
        )
        row["target_quadrature_N4_to_N5_change"] = target_quadrature_change[t]
        row["sum_last_change_proxy"] = (
            row["source_level_L2_to_L3_change"]
            + row["source_quadrature_N4_to_N5_change"]
            + row["target_quadrature_N4_to_N5_change"]
        )

    central = source.get("central_refinement")
    central_check = None
    if central:
        central_ratio = float(central["source_over_target"])
        central_check = {
            "t": float(central["t"]),
            "ratio_source_N6_target_N7": central_ratio,
            "residual_from_five_point_constant": (
                central_ratio / preferred["normalization_geometric_mean"] - 1.0
            ),
        }

    return {
        "schema": "nsrr-nsnsns-constant-ratio-audit-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_convention": (
            "unscaled Human-block M contraction; the ground block expression is eight; "
            "no absolute normalization is imposed"
        ),
        "surface_scope": (
            "five points on the one-real-dimensional family "
            "Omega=[[i,t+i/2],[t+i/2,i]], t=0.52,...,0.68"
        ),
        "normalization_fit_policy": (
            "equal fractional weight: minimize squared log(source/target/A)"
        ),
        "preferred_fit": preferred,
        "preferred_points": preferred_points,
        "central_refinement_holdout": central_check,
        "all_cutoff_and_kernel_fits": fits,
        "conclusion": (
            "The unscaled M candidate is close to one constant on the saved curve, but "
            "the residual is monotone and the curve samples only one of six real genus-two "
            "moduli. This is evidence for an overall normalization, not yet a proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--old-scan", type=Path, default=DEFAULT_OLD_SCAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.source, args.target, args.old_scan)
    write_json(args.output_dir / "summary.json", result)
    rows = result["preferred_points"]
    with (args.output_dir / "preferred_fit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit = result["preferred_fit"]
    print(f"A={fit['normalization_geometric_mean']:.12f}")
    print(f"max |residual|={fit['maximum_absolute_fractional_residual']:.6%}")
    print(f"RMS residual={fit['rms_fractional_residual']:.6%}")
    print(f"linear change over scan={fit['linear_change_over_scan']:.6%}")
    if result["central_refinement_holdout"]:
        print(
            "central N6/N7 residual="
            f"{result['central_refinement_holdout']['residual_from_five_point_constant']:.6%}"
        )


if __name__ == "__main__":
    main()
