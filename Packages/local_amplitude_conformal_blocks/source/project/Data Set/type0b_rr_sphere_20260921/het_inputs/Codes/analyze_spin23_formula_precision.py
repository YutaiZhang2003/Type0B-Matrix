#!/usr/bin/env python3
"""Audit the large-OPE-disk precision validation of the singlet formulas.

The base central estimator applies a high-q correction only where the
level-seven Gram matrices are controlled,

``q7(P=2.5,r=.30) + q6(P=4,r=.30) - q6(P=2.5,r=.30)``.

When the optional ``Pmax=6`` supplement is supplied, its momentum-tail
extension replaces the ``Pmax=4`` term and the independently measured dense
quadrature correction is added.  Larger-radius evaluations are retained as
numerical diagnostics.  None of these controls is fitted to the candidate
formulas.
"""

from __future__ import annotations

import argparse
import json
import math
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


PROFILES = (
    "q7_p2p5_r30",
    "q6_p2p5_r30",
    "q6_p4_r30",
    "q6_p4_r30_dense",
    "q6_p4_r36",
)
PROFILE_LABELS = {
    "q6_p4_r30": r"$q_6,P_4,r=.30$",
    "composite": r"matched $q_7+P_4$",
    "composite_dense": r"matched + dense",
    "composite_p6_dense": r"matched $q_7+P_6$",
    "q6_p4_r36": r"$q_6,P_4,r=.36$",
}
PROCESS_LABELS = {"ssvv": r"$S\to SVV$", "ssss": r"$S\to SSS$"}
FAMILY_COLORS = {
    "core": "#0072B2",
    "structured": "#D55E00",
    "soft": "#009E73",
    "near-real": "#CC79A7",
}
_TINY = np.finfo(float).tiny


def _parent_family(parent_id: str) -> str:
    """Return the source family while preserving hyphenated family names."""

    for family in FAMILY_COLORS:
        if parent_id.startswith(f"{family}-"):
            return family
    return parent_id.split("-", maxsplit=1)[0]


def _relative_change(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first - second) / np.maximum.reduce(
        (np.abs(first), np.abs(second), np.full(len(first), _TINY))
    )


def _relative_residual(numerical: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(numerical - candidate) / np.maximum(np.abs(numerical), _TINY)


def _aligned_profiles(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], dict[str, list[Mapping[str, str]]]]:
    grouped: dict[str, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["setting"]][row["parent_id"]] = row
    missing = [profile for profile in PROFILES if profile not in grouped]
    if missing:
        raise ValueError(f"missing precision profiles: {missing}")
    parents = sorted(set.intersection(*(set(grouped[name]) for name in PROFILES)))
    if not parents:
        raise ValueError("no kinematic point is complete across all precision profiles")
    aligned = {
        profile: [grouped[profile][parent] for parent in parents]
        for profile in PROFILES
    }
    return parents, aligned


def _values(
    rows: Sequence[Mapping[str, str]],
    process: str,
    *,
    lower: bool = False,
) -> np.ndarray:
    stem = f"{process}_raw_lower_q" if lower else f"{process}_raw"
    return np.asarray([_complex(row, stem) for row in rows], dtype=complex)


def _precision_data(
    rows: Sequence[Mapping[str, str]],
    p6_rows: Sequence[Mapping[str, str]] | None = None,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, object]]:
    parents, aligned = _aligned_profiles(rows)
    p6_by_parent: dict[str, Mapping[str, str]] = {}
    if p6_rows is not None:
        p6_by_parent = {
            row["parent_id"]: row
            for row in p6_rows
            if row.get("setting") == "q6_p6_r30"
        }
        missing_p6 = sorted(set(parents) - set(p6_by_parent))
        if missing_p6:
            raise ValueError(
                "Pmax=6 supplemental results are missing parents: "
                + ", ".join(missing_p6)
            )
    omega = _kinematics(aligned["q7_p2p5_r30"])
    process_data: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, object] = {
        "complete_points": len(parents),
        "parent_ids": parents,
        "source_family_counts": dict(
            Counter(_parent_family(parent) for parent in parents)
        ),
    }

    for process in ("ssvv", "ssss"):
        q7_low = _values(aligned["q7_p2p5_r30"], process)
        q6_low = _values(aligned["q6_p2p5_r30"], process)
        q6_high = _values(aligned["q6_p4_r30"], process)
        q6_dense = _values(aligned["q6_p4_r30_dense"], process)
        q6_radius = _values(aligned["q6_p4_r36"], process)
        q6_p6 = (
            _values([p6_by_parent[parent] for parent in parents], process)
            if p6_by_parent
            else None
        )
        q5_low = _values(aligned["q6_p2p5_r30"], process, lower=True)
        q5_high = _values(aligned["q6_p4_r30"], process, lower=True)

        composite = q7_low + q6_high - q6_low
        composite_dense = composite + q6_dense - q6_high
        composite_p6_dense = (
            q7_low + q6_p6 - q6_low + q6_dense - q6_high
            if q6_p6 is not None
            else None
        )
        central = (
            composite_p6_dense
            if composite_p6_dense is not None
            else composite_dense
        )
        candidate = _candidate_values(omega, process)
        q_correction = q7_low - q6_low
        tail_q_change = (q6_high - q6_low) - (q5_high - q5_low)
        dense_change = q6_dense - q6_high
        radius_change = q6_radius - q6_high
        scale = np.maximum(np.abs(central), _TINY)

        # This is an observed-variation diagnostic, not a rigorous error bar.
        variation_envelope = (
            np.abs(q_correction)
            + np.abs(tail_q_change)
            + np.abs(dense_change)
            + np.abs(radius_change)
        ) / scale
        residual = _relative_residual(central, candidate)
        process_data[process] = {
            "q6_p4_r30": q6_high,
            "q6_p4_r36": q6_radius,
            "composite": composite,
            "composite_dense": composite_dense,
            "central": central,
            "candidate": candidate,
            "residual": residual,
            "variation_envelope": variation_envelope,
        }
        if composite_p6_dense is not None and q6_p6 is not None:
            process_data[process]["composite_p6_dense"] = composite_p6_dense
        summary[process] = {
            "unscaled_formula_relative_residual": _percentiles(residual),
            "best_complex_scale_not_applied": {
                "real": _best_complex_scale(candidate, central).real,
                "imag": _best_complex_scale(candidate, central).imag,
            },
            "q7_minus_q6_low_momentum_relative_change": _percentiles(
                np.abs(q_correction) / scale
            ),
            "q6_vs_q5_momentum_tail_relative_change": _percentiles(
                np.abs(tail_q_change) / scale
            ),
            "dense_quadrature_relative_change": _percentiles(
                np.abs(dense_change) / scale
            ),
            "radius_0p30_to_0p36_relative_change": _percentiles(
                np.abs(radius_change) / scale
            ),
            "observed_variation_envelope": _percentiles(variation_envelope),
            "formula_residual_over_variation_envelope": _percentiles(
                residual / np.maximum(variation_envelope, _TINY)
            ),
            "fraction_formula_residual_below_variation_envelope": float(
                np.mean(residual <= variation_envelope)
            ),
        }
        if q6_p6 is not None:
            summary[process]["pmax_4_to_6_relative_change"] = _percentiles(
                np.abs(q6_p6 - q6_high) / scale
            )
            summary[process]["matched_p6_formula_relative_residual"] = _percentiles(
                _relative_residual(composite_p6_dense, candidate)
            )

    summary["coefficient_fits"] = {
        "ssvv": _ssvv_compact_coefficient_comparison(
            omega,
            process_data["ssvv"]["central"],
            process_data["ssvv"]["central"],
        ),
        "ssss": _ssss_compact_coefficient_comparison(
            omega,
            process_data["ssss"]["central"],
            process_data["ssss"]["central"],
        ),
    }
    return parents, omega, process_data, summary


def _plot_residuals(
    process_data: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    profiles = ["q6_p4_r30", "composite", "composite_dense"]
    if "composite_p6_dense" in process_data["ssvv"]:
        profiles.append("composite_p6_dense")
    profiles.append("q6_p4_r36")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for axis, process in zip(axes, ("ssvv", "ssss")):
        candidate = process_data[process]["candidate"]
        for index, profile in enumerate(profiles):
            numerical = process_data[process][profile]
            residual = _relative_residual(numerical, candidate)
            jitter = np.linspace(-0.12, 0.12, len(residual))
            axis.scatter(
                index + jitter,
                residual,
                color="#0072B2",
                alpha=0.50,
                linewidths=0,
                s=15,
            )
            axis.plot(
                index,
                np.median(residual),
                marker="_",
                markersize=17,
                markeredgewidth=2.2,
                color="black",
            )
        axis.set_yscale("log")
        axis.set_xticks(range(len(profiles)), [PROFILE_LABELS[p] for p in profiles])
        axis.tick_params(axis="x", rotation=24)
        axis.set_ylabel(
            r"$|\mathcal{M}_{\mathrm{num}}-\mathcal{M}_{\mathrm{form}}|/"
            r"|\mathcal{M}_{\mathrm{num}}|$"
        )
        axis.set_title(PROCESS_LABELS[process])
        axis.grid(alpha=0.18, linewidth=0.6)
    for extension, options in (("pdf", {}), ("png", {"dpi": 250})):
        figure.savefig(output_dir / f"spin23_formula_precision_residuals.{extension}", **options)
    plt.close(figure)


def _plot_normalized_differences(
    parents: Sequence[str],
    process_data: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    families = np.asarray([_parent_family(parent) for parent in parents])
    for axis, process in zip(axes, ("ssvv", "ssss")):
        numerical = process_data[process]["central"]
        candidate = process_data[process]["candidate"]
        difference = (numerical - candidate) / numerical
        for family, color in FAMILY_COLORS.items():
            mask = families == family
            if np.any(mask):
                axis.scatter(
                    difference.real[mask],
                    difference.imag[mask],
                    s=24,
                    alpha=0.75,
                    linewidths=0,
                    color=color,
                    label=family,
                )
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.45)
        axis.axvline(0, color="black", linewidth=0.7, alpha=0.45)
        axis.set_xlabel(
            r"$\Re[(\mathcal{M}_{\mathrm{num}}-\mathcal{M}_{\mathrm{form}})"
            r"/\mathcal{M}_{\mathrm{num}}]$"
        )
        axis.set_ylabel(
            r"$\Im[(\mathcal{M}_{\mathrm{num}}-\mathcal{M}_{\mathrm{form}})"
            r"/\mathcal{M}_{\mathrm{num}}]$"
        )
        axis.set_title(PROCESS_LABELS[process])
        axis.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
        axis.grid(alpha=0.16, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8)
    for extension, options in (("pdf", {}), ("png", {"dpi": 250})):
        figure.savefig(
            output_dir / f"spin23_formula_precision_normalized_difference.{extension}",
            **options,
        )
    plt.close(figure)


def analyze(
    results_dir: Path,
    output_dir: Path,
    *,
    p6_results_dir: Path | None = None,
) -> dict[str, object]:
    rows = _load_rows(results_dir)
    p6_rows = _load_rows(p6_results_dir) if p6_results_dir is not None else None
    parents, _, process_data, summary = _precision_data(rows, p6_rows)
    summary["successful_rows"] = len(rows)
    summary["profile_counts"] = dict(Counter(row["setting"] for row in rows))
    summary["supplemental_p6_rows"] = len(p6_rows) if p6_rows is not None else 0
    summary["central_estimator"] = (
        "q7_p2p5_r30 + q6_p6_r30 - q6_p2p5_r30 "
        "+ q6_p4_r30_dense - q6_p4_r30"
        if p6_rows is not None
        else "q7_p2p5_r30 + q6_p4_r30_dense - q6_p2p5_r30"
    )
    summary["formulas_are_unscaled"] = True
    summary["variation_envelope_is_rigorous_bound"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spin23_formula_precision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _configure_matplotlib()
    _plot_residuals(process_data, output_dir)
    _plot_normalized_differences(parents, process_data, output_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--p6-results-dir",
        type=Path,
        help="optional q6, Pmax=6 supplemental result directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    summary = analyze(
        args.results_dir,
        output_dir,
        p6_results_dir=args.p6_results_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
