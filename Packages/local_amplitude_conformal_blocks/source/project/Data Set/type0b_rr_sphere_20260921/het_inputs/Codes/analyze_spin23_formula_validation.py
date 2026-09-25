#!/usr/bin/env python3
"""Analyze the targeted high-accuracy validation of the singlet formulas."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analyze_spin23_genuine_formulas import (
    _best_complex_scale,
    _candidate_values,
    _complex,
    _configure_matplotlib,
    _kinematics,
    _load_rows,
    _percentiles,
    _ssss_compact_coefficient_comparison,
    _ssvv_compact_coefficient_comparison,
)


PROFILE_ORDER = (
    "q7_p2p5_l20",
    "q6_p2p5_l20",
    "q6_p2p5_l24",
    "q6_p4_l20",
    "q6_p6_l20",
    "q6_p6_l24",
    "q6_p6_l28",
    "q6_p6_l24_rsmall",
    "q6_p6_l24_rlarge",
    "q6_p6_l24_dense",
)
PROFILE_LABELS = {
    "q7_p2p5_l20": r"$q_7,P_{2.5},L_{20}$",
    "q6_p2p5_l20": r"$q_6,P_{2.5},L_{20}$",
    "q6_p2p5_l24": r"$q_6,P_{2.5},L_{24}$",
    "q6_p4_l20": r"$q_6,P_4,L_{20}$",
    "q6_p6_l20": r"$q_6,P_6,L_{20}$",
    "q6_p6_l24": r"$q_6,P_6,L_{24}$",
    "q6_p6_l28": r"$q_6,P_6,L_{28}$",
    "q6_p6_l24_rsmall": r"small $r$",
    "q6_p6_l24_rlarge": r"large $r$",
    "q6_p6_l24_dense": r"dense quad.",
}
_TINY = np.finfo(float).tiny


def _residual(numerical: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(numerical - candidate) / np.maximum(np.abs(numerical), _TINY)


def _symmetric_change(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first - second) / np.maximum.reduce(
        (np.abs(first), np.abs(second), np.full(len(first), _TINY))
    )


def _profile_data(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)

    profiles: dict[str, dict[str, object]] = {}
    for profile, profile_rows in grouped.items():
        profile_rows = sorted(profile_rows, key=lambda row: row["parent_id"])
        omega = _kinematics(profile_rows)
        processes = {}
        for process in ("ssvv", "ssss"):
            numerical = np.asarray(
                [_complex(row, f"{process}_raw") for row in profile_rows],
                dtype=complex,
            )
            candidate = _candidate_values(omega, process)
            processes[process] = {
                "numerical": numerical,
                "candidate": candidate,
                "residual": _residual(numerical, candidate),
            }
        profiles[profile] = {
            "rows": profile_rows,
            "omega": omega,
            "processes": processes,
        }
    return profiles


def _coefficient_fit_for_profile(
    profile: Mapping[str, object],
) -> dict[str, object]:
    omega = profile["omega"]
    processes = profile["processes"]
    # The comparison helpers need two datasets.  Passing the same values as
    # parent and control gives one consistently normalized coefficient fit.
    ssvv = processes["ssvv"]["numerical"]
    ssss = processes["ssss"]["numerical"]
    return {
        "ssvv": _ssvv_compact_coefficient_comparison(omega, ssvv, ssvv),
        "ssss": _ssss_compact_coefficient_comparison(omega, ssss, ssss),
    }


def _paired_profile_summary(
    profiles: Mapping[str, Mapping[str, object]],
    first_name: str,
    second_name: str,
) -> dict[str, object] | None:
    if first_name not in profiles or second_name not in profiles:
        return None
    first = profiles[first_name]
    second = profiles[second_name]
    first_by_parent = {
        row["parent_id"]: index for index, row in enumerate(first["rows"])
    }
    second_by_parent = {
        row["parent_id"]: index for index, row in enumerate(second["rows"])
    }
    common = sorted(set(first_by_parent) & set(second_by_parent))
    summary: dict[str, object] = {"points": len(common)}
    for process in ("ssvv", "ssss"):
        first_values = np.asarray(
            [
                first["processes"][process]["numerical"][first_by_parent[parent]]
                for parent in common
            ]
        )
        second_values = np.asarray(
            [
                second["processes"][process]["numerical"][second_by_parent[parent]]
                for parent in common
            ]
        )
        first_candidate = np.asarray(
            [
                first["processes"][process]["candidate"][first_by_parent[parent]]
                for parent in common
            ]
        )
        second_candidate = np.asarray(
            [
                second["processes"][process]["candidate"][second_by_parent[parent]]
                for parent in common
            ]
        )
        if not np.allclose(first_candidate, second_candidate, rtol=0, atol=1.0e-14):
            raise ValueError("paired profiles disagree on formula values")
        first_gap = np.abs(first_values - first_candidate)
        second_gap = np.abs(second_values - second_candidate)
        summary[process] = {
            "numerical_relative_change": _percentiles(
                _symmetric_change(first_values, second_values)
            ),
            "first_formula_relative_residual": _percentiles(
                _residual(first_values, first_candidate)
            ),
            "second_formula_relative_residual": _percentiles(
                _residual(second_values, second_candidate)
            ),
            "fraction_second_closer_to_formula": float(np.mean(second_gap < first_gap)),
        }
    return summary


def _composite_high_accuracy_summary(
    profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    r"""Construct the stable high-q plus high-momentum composite estimator.

    For each process the estimator is

    ``M(q7,P2.5,L20) + M(q6,P6,L28) - M(q6,P2.5,L20)``.

    The q=5 values stored alongside both q=6 evaluations test whether the
    high-momentum/local-order correction is itself stable in q order.
    """

    names = ("q7_p2p5_l20", "q6_p2p5_l20", "q6_p6_l20", "q6_p6_l28")
    missing = [name for name in names if name not in profiles]
    if missing:
        return {"status": "incomplete", "missing_profiles": missing}

    indices = {
        name: {
            row["parent_id"]: index
            for index, row in enumerate(profiles[name]["rows"])
        }
        for name in names
    }
    common = sorted(set.intersection(*(set(table) for table in indices.values())))
    if not common:
        return {"status": "incomplete", "missing_profiles": [], "points": 0}

    reference_rows = [
        profiles["q7_p2p5_l20"]["rows"][indices["q7_p2p5_l20"][parent]]
        for parent in common
    ]
    omega = _kinematics(reference_rows)
    composite_values: dict[str, np.ndarray] = {}
    process_summary: dict[str, object] = {}

    def values(profile: str, process: str, *, lower: bool = False) -> np.ndarray:
        stem = f"{process}_raw_lower_q" if lower else f"{process}_raw"
        return np.asarray(
            [
                _complex(
                    profiles[profile]["rows"][indices[profile][parent]],
                    stem,
                )
                for parent in common
            ],
            dtype=complex,
        )

    for process in ("ssvv", "ssss"):
        q7_low_p = values("q7_p2p5_l20", process)
        q6_low_p = values("q6_p2p5_l20", process)
        q6_high_p_l20 = values("q6_p6_l20", process)
        q6_high_p_l28 = values("q6_p6_l28", process)
        composite = q7_low_p + q6_high_p_l28 - q6_low_p
        candidate = _candidate_values(omega, process)
        composite_values[process] = composite

        q6_tail = q6_high_p_l20 - q6_low_p
        q5_tail = (
            values("q6_p6_l20", process, lower=True)
            - values("q6_p2p5_l20", process, lower=True)
        )
        q6_local = q6_high_p_l28 - q6_high_p_l20
        q5_local = (
            values("q6_p6_l28", process, lower=True)
            - values("q6_p6_l20", process, lower=True)
        )
        baseline_gap = np.abs(q7_low_p - candidate)
        composite_gap = np.abs(composite - candidate)
        scale = _best_complex_scale(candidate, composite)
        process_summary[process] = {
            "formula_relative_residual": _percentiles(
                _residual(composite, candidate)
            ),
            "baseline_formula_relative_residual": _percentiles(
                _residual(q7_low_p, candidate)
            ),
            "fraction_composite_closer_than_baseline": float(
                np.mean(composite_gap < baseline_gap)
            ),
            "q5_to_q6_tail_correction_change_relative_to_amplitude": _percentiles(
                np.abs(q6_tail - q5_tail)
                / np.maximum(np.abs(composite), _TINY)
            ),
            "q5_to_q6_local_correction_change_relative_to_amplitude": _percentiles(
                np.abs(q6_local - q5_local)
                / np.maximum(np.abs(composite), _TINY)
            ),
            "best_complex_scale_not_applied": {
                "real": scale.real,
                "imag": scale.imag,
            },
        }

    process_summary["coefficient_fits"] = {
        "ssvv": _ssvv_compact_coefficient_comparison(
            omega, composite_values["ssvv"], composite_values["ssvv"]
        ),
        "ssss": _ssss_compact_coefficient_comparison(
            omega, composite_values["ssss"], composite_values["ssss"]
        ),
    }
    return {
        "status": "complete",
        "points": len(common),
        "definition": "q7_p2p5_l20 + q6_p6_l28 - q6_p2p5_l20",
        **process_summary,
    }


def _plot_profile_residuals(
    profiles: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> None:
    available = [profile for profile in PROFILE_ORDER if profile in profiles]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    for axis, process in zip(axes, ("ssvv", "ssss")):
        for x_value, profile in enumerate(available):
            values = profiles[profile]["processes"][process]["residual"]
            jitter = np.linspace(-0.13, 0.13, len(values))
            axis.scatter(
                x_value + jitter,
                values,
                s=12,
                alpha=0.45,
                linewidths=0,
                color="#0072B2",
            )
            axis.plot(
                x_value,
                np.median(values),
                marker="_",
                markersize=16,
                markeredgewidth=2.0,
                color="black",
            )
        axis.set_yscale("log")
        axis.set_xticks(range(len(available)), [PROFILE_LABELS[p] for p in available])
        axis.tick_params(axis="x", rotation=38)
        axis.set_ylabel(
            r"$|\mathcal{M}_{\rm num}-\mathcal{M}_{\rm form}|/"
            r"|\mathcal{M}_{\rm num}|$"
        )
        axis.set_title(r"$S\to SVV$" if process == "ssvv" else r"$S\to SSS$")
        axis.grid(alpha=0.18, linewidth=0.6)
    for extension, options in (("pdf", {}), ("png", {"dpi": 240})):
        figure.savefig(output_dir / f"spin23_formula_validation_profiles.{extension}", **options)
    plt.close(figure)


def analyze(results_dir: Path, output_dir: Path) -> dict[str, object]:
    rows = _load_rows(results_dir)
    profiles = _profile_data(rows)
    profile_summary = {}
    for profile in PROFILE_ORDER:
        if profile not in profiles:
            continue
        data = profiles[profile]
        profile_summary[profile] = {
            "points": len(data["rows"]),
            "ssvv_formula_relative_residual": _percentiles(
                data["processes"]["ssvv"]["residual"]
            ),
            "ssss_formula_relative_residual": _percentiles(
                data["processes"]["ssss"]["residual"]
            ),
            "coefficient_fits": _coefficient_fit_for_profile(data),
        }

    comparisons = {}
    for label, first, second in (
        ("q6_to_q7_at_p2p5", "q6_p2p5_l20", "q7_p2p5_l20"),
        ("local_order_20_to_24_at_p2p5", "q6_p2p5_l20", "q6_p2p5_l24"),
        ("momentum_cutoff_p2p5_to_p4", "q6_p2p5_l20", "q6_p4_l20"),
        ("momentum_cutoff_p4_to_p6", "q6_p4_l20", "q6_p6_l20"),
        ("local_order_20_to_24", "q6_p6_l20", "q6_p6_l24"),
        ("local_order_24_to_28", "q6_p6_l24", "q6_p6_l28"),
        ("radius_baseline_to_small", "q6_p6_l24", "q6_p6_l24_rsmall"),
        ("radius_baseline_to_large", "q6_p6_l24", "q6_p6_l24_rlarge"),
        ("quadrature_baseline_to_dense", "q6_p6_l24", "q6_p6_l24_dense"),
    ):
        comparison = _paired_profile_summary(profiles, first, second)
        if comparison is not None:
            comparisons[label] = comparison

    parent_counts = Counter(row["parent_id"] for row in rows)
    summary = {
        "successful_rows": len(rows),
        "profile_counts": dict(Counter(row["setting"] for row in rows)),
        "complete_kinematic_points": sum(
            count == len(PROFILE_ORDER) for count in parent_counts.values()
        ),
        "profiles": profile_summary,
        "paired_comparisons": comparisons,
        "composite_high_accuracy": _composite_high_accuracy_summary(profiles),
        "formulas_are_unscaled": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spin23_formula_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _configure_matplotlib()
    _plot_profile_residuals(profiles, output_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    summary = analyze(args.results_dir, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
