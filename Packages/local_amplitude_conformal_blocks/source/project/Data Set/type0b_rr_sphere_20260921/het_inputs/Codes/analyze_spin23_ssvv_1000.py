#!/usr/bin/env python3
"""Build and analyze a deterministic 1,000-point raw ``S -> S V V`` sample.

The completed VVVV scan already evaluated the required worldsheet integral.
In its column convention, ``M1=C`` is the exact Spin(23) tensor projection for
``S(omega0) -> S(omega1) V(omega2) V(omega3)``.  This script selects 1,000
distinct kinematic rows, performs an independent affine-inverse fit, retains a
deterministic held-out set, and writes machine-readable diagnostics and plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spin23_ssvv_projection import (
    SSVVKinematics,
    normalized_inverse,
    raw_ssvv_candidate,
)


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "Data Set"
DEFAULT_INPUT = (
    DATA_DIR
    / "results"
    / "spin23_vvvv_stable_all_6888_optimized_v1"
    / "merged.csv"
)
DEFAULT_OUTPUT = DATA_DIR / "results" / "spin23_ssvv_1000"
DEFAULT_QUOTAS = {
    "core": 700,
    "structured": 150,
    "soft": 75,
    "near-real": 75,
}


def _evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if count < 0 or count > length:
        raise ValueError(f"cannot select {count} rows from a family of size {length}")
    if count == 0:
        return np.asarray([], dtype=int)
    indices = np.floor((np.arange(count) + 0.5) * length / count).astype(int)
    if len(np.unique(indices)) != count:
        raise AssertionError("evenly spaced selector produced duplicate indices")
    return indices


def select_stratified_rows(
    frame: pd.DataFrame,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> pd.DataFrame:
    """Select deterministic, distinct rows from the independent scan families."""

    required = {"point_id", "family", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"input is missing required columns: {sorted(missing)}")

    selected: list[pd.DataFrame] = []
    for family, count in quotas.items():
        family_rows = frame.loc[
            (frame["family"] == family) & (frame["status"] == "ok")
        ].sort_values("point_id", kind="stable")
        indices = _evenly_spaced_indices(len(family_rows), int(count))
        selected.append(family_rows.iloc[indices])

    result = pd.concat(selected, ignore_index=True)
    expected = sum(int(value) for value in quotas.values())
    if len(result) != expected:
        raise AssertionError(f"selected {len(result)} rows, expected {expected}")
    if result["point_id"].duplicated().any():
        raise AssertionError("the SSVV sample contains duplicate point IDs")
    return result


def fit_inverse_affine(
    vector_pair_sum: np.ndarray,
    normalized_inverse_values: np.ndarray,
) -> tuple[complex, complex]:
    """Fit ``Y=alpha+beta*(omega2+omega3)`` by complex least squares."""

    design = np.column_stack(
        [np.ones(len(vector_pair_sum), dtype=complex), vector_pair_sum]
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        normalized_inverse_values,
        rcond=None,
    )
    return complex(coefficients[0]), complex(coefficients[1])


def complex_column(frame: pd.DataFrame, stem: str) -> np.ndarray:
    """Read adjacent ``*_re`` and ``*_im`` columns as a complex array."""

    return (
        pd.to_numeric(frame[f"{stem}_re"], errors="raise").to_numpy(dtype=float)
        + 1j
        * pd.to_numeric(frame[f"{stem}_im"], errors="raise").to_numpy(dtype=float)
    )


def build_ssvv_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Add raw SSVV amplitudes, predictions, and inverse-fit variables."""

    result = source.copy()
    omega1 = complex_column(result, "omega1")
    omega2 = complex_column(result, "omega2")
    omega3 = complex_column(result, "omega3")
    numerical = complex_column(result, "M1")

    kinematics = [
        SSVVKinematics(w1, w2, w3)
        for w1, w2, w3 in zip(omega1, omega2, omega3, strict=True)
    ]
    candidate = np.asarray([raw_ssvv_candidate(kin) for kin in kinematics])
    inverse = np.asarray(
        [normalized_inverse(value, kin) for value, kin in zip(numerical, kinematics, strict=True)]
    )
    pair_sum = omega2 + omega3
    absolute_residual = np.abs(numerical - candidate)
    relative_residual = absolute_residual / np.maximum(
        np.abs(candidate), np.finfo(float).tiny
    )

    result["ssvv_raw_re"] = numerical.real
    result["ssvv_raw_im"] = numerical.imag
    result["ssvv_candidate_re"] = candidate.real
    result["ssvv_candidate_im"] = candidate.imag
    result["ssvv_abs_residual"] = absolute_residual
    result["ssvv_rel_residual"] = relative_residual
    result["vector_pair_sum_re"] = pair_sum.real
    result["vector_pair_sum_im"] = pair_sum.imag
    result["normalized_inverse_re"] = inverse.real
    result["normalized_inverse_im"] = inverse.imag
    result["held_out"] = np.arange(len(result)) % 5 == 0
    return result


def _plot_results(frame: pd.DataFrame, output_dir: Path) -> None:
    numerical = complex_column(frame, "ssvv_raw")
    candidate = complex_column(frame, "ssvv_candidate")
    inverse = complex_column(frame, "normalized_inverse")
    pair_sum = complex_column(frame, "vector_pair_sum")
    expected_inverse = 1.0 + 1j * pair_sum

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    components = ((numerical.real, candidate.real, "Real part"), (numerical.imag, candidate.imag, "Imaginary part"))
    for axis, (observed, predicted, title) in zip(axes[0], components, strict=True):
        limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
        axis.scatter(predicted, observed, s=9, alpha=0.55, linewidths=0)
        axis.plot(limits, limits, color="black", linewidth=1.1)
        axis.set_xlabel("candidate")
        axis.set_ylabel("numerical block integral")
        axis.set_title(title)

    axes[1, 0].scatter(
        np.abs(candidate),
        frame["ssvv_rel_residual"],
        s=9,
        alpha=0.55,
        linewidths=0,
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$|\mathcal{M}^{\mathrm{candidate}}_{S\to SVV}|$")
    axes[1, 0].set_ylabel("relative discrepancy")
    axes[1, 0].set_title("Pointwise formula residual")

    inverse_error = np.abs(inverse - expected_inverse)
    axes[1, 1].scatter(
        np.abs(expected_inverse),
        inverse_error,
        s=9,
        alpha=0.55,
        linewidths=0,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"$|1+i(\omega_2+\omega_3)|$")
    axes[1, 1].set_ylabel(r"$|Y-[1+i(\omega_2+\omega_3)]|$")
    axes[1, 1].set_title("Affine-inverse residual")

    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"spin23_ssvv_1000_diagnostics.{suffix}", dpi=240)
    plt.close(figure)


def analyze(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Run the complete deterministic projection, fit, and diagnostic analysis."""

    source = pd.read_csv(input_path)
    selected = select_stratified_rows(source)
    frame = build_ssvv_frame(selected)

    pair_sum = complex_column(frame, "vector_pair_sum")
    inverse = complex_column(frame, "normalized_inverse")
    train = ~frame["held_out"].to_numpy(dtype=bool)
    held_out = ~train
    alpha, beta = fit_inverse_affine(pair_sum[train], inverse[train])
    fit_inverse = alpha + beta * pair_sum
    fit_residual = np.abs(inverse - fit_inverse)

    frame["fitted_inverse_re"] = fit_inverse.real
    frame["fitted_inverse_im"] = fit_inverse.imag
    frame["fitted_inverse_abs_residual"] = fit_residual

    relative = frame["ssvv_rel_residual"].to_numpy(dtype=float)
    q_change = pd.to_numeric(frame["estimated_abs_error_M1"], errors="coerce").to_numpy()
    formula_absolute = frame["ssvv_abs_residual"].to_numpy(dtype=float)
    summary: dict[str, object] = {
        "input": str(input_path),
        "sample_count": int(len(frame)),
        "family_counts": {
            str(key): int(value) for key, value in frame["family"].value_counts().items()
        },
        "train_count": int(train.sum()),
        "held_out_count": int(held_out.sum()),
        "fit_alpha": {"real": alpha.real, "imag": alpha.imag},
        "fit_beta": {"real": beta.real, "imag": beta.imag},
        "distance_alpha_from_1": abs(alpha - 1.0),
        "distance_beta_from_i": abs(beta - 1j),
        "formula_relative_residual": {
            "median": float(np.median(relative)),
            "rms": float(np.sqrt(np.mean(relative**2))),
            "maximum": float(np.max(relative)),
        },
        "inverse_fit_abs_residual": {
            "train_rms": float(np.sqrt(np.mean(fit_residual[train] ** 2))),
            "held_out_rms": float(np.sqrt(np.mean(fit_residual[held_out] ** 2))),
            "held_out_maximum": float(np.max(fit_residual[held_out])),
        },
        "formula_residual_over_adjacent_q_change": {
            "median": float(np.nanmedian(formula_absolute / q_change)),
            "fraction_below_one": float(np.nanmean(formula_absolute <= q_change)),
        },
        "interpretation": (
            "Raw reduced worldsheet projection only; no asymptotic singlet leg "
            "normalization or common heterotic sphere normalization is included."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "spin23_ssvv_1000.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_results(frame, output_dir)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = analyze(args.input, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
