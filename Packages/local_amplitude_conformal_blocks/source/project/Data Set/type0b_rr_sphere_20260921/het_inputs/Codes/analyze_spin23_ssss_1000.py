#!/usr/bin/env python3
"""Analyze the raw ``S -> S S S`` projection on 1,000 numerical points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_spin23_ssvv_1000 import (
    DEFAULT_INPUT,
    complex_column,
    select_stratified_rows,
)
from spin23_ssss_projection import (
    SSSSKinematics,
    channel_features,
    normalized_amplitude,
    raw_ssss_candidate,
)


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CODE_DIR.parent / "Data Set" / "results" / "spin23_ssss_1000"


def fit_channel_weights(
    features: np.ndarray,
    normalized_values: np.ndarray,
) -> np.ndarray:
    """Fit independent complex weights of the ``12``, ``13``, and ``23`` terms."""

    coefficients, _, _, _ = np.linalg.lstsq(features, normalized_values, rcond=None)
    return np.asarray(coefficients, dtype=complex)


def build_ssss_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Add the summed amplitude, candidate, channel basis, and residuals."""

    result = source.copy()
    omega1 = complex_column(result, "omega1")
    omega2 = complex_column(result, "omega2")
    omega3 = complex_column(result, "omega3")
    channels = np.column_stack(
        [complex_column(result, name) for name in ("M1", "M2", "M3")]
    )
    numerical = np.sum(channels, axis=1)
    kinematics = [
        SSSSKinematics(w1, w2, w3)
        for w1, w2, w3 in zip(omega1, omega2, omega3, strict=True)
    ]
    features = np.asarray([channel_features(kin) for kin in kinematics])
    candidate = np.asarray([raw_ssss_candidate(kin) for kin in kinematics])
    normalized = np.asarray(
        [normalized_amplitude(value, kin) for value, kin in zip(numerical, kinematics, strict=True)]
    )
    residual = np.abs(numerical - candidate)
    relative = residual / np.maximum(np.abs(candidate), np.finfo(float).tiny)

    result["ssss_raw_re"] = numerical.real
    result["ssss_raw_im"] = numerical.imag
    result["ssss_candidate_re"] = candidate.real
    result["ssss_candidate_im"] = candidate.imag
    result["ssss_abs_residual"] = residual
    result["ssss_rel_residual"] = relative
    result["ssss_normalized_re"] = normalized.real
    result["ssss_normalized_im"] = normalized.imag
    for index, label in enumerate(("12", "13", "23")):
        result[f"feature_{label}_re"] = features[:, index].real
        result[f"feature_{label}_im"] = features[:, index].imag
    result["held_out"] = np.arange(len(result)) % 5 == 0
    return result


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [complex_column(frame, f"feature_{label}") for label in ("12", "13", "23")]
    )


def _plot_results(frame: pd.DataFrame, coefficients: np.ndarray, output_dir: Path) -> None:
    numerical = complex_column(frame, "ssss_raw")
    candidate = complex_column(frame, "ssss_candidate")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    components = (
        (numerical.real, candidate.real, "Real part"),
        (numerical.imag, candidate.imag, "Imaginary part"),
    )
    for axis, (observed, predicted, title) in zip(axes[0], components, strict=True):
        limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
        axis.scatter(predicted, observed, s=9, alpha=0.55, linewidths=0)
        axis.plot(limits, limits, color="black", linewidth=1.1)
        axis.set_xlabel("candidate")
        axis.set_ylabel("numerical block integral")
        axis.set_title(title)

    axes[1, 0].scatter(
        np.abs(candidate),
        frame["ssss_rel_residual"],
        s=9,
        alpha=0.55,
        linewidths=0,
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"$|\mathcal{M}^{\mathrm{candidate}}_{S\to SSS}|$")
    axes[1, 0].set_ylabel("relative discrepancy")
    axes[1, 0].set_title("Pointwise formula residual")

    positions = np.arange(3)
    width = 0.36
    axes[1, 1].bar(
        positions - width / 2,
        coefficients.real - 1.0,
        width,
        label=r"$\operatorname{Re}c_{ij}-1$",
    )
    axes[1, 1].bar(
        positions + width / 2,
        coefficients.imag,
        width,
        label=r"$\operatorname{Im}c_{ij}$",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(positions, (r"$c_{12}$", r"$c_{13}$", r"$c_{23}$"))
    axes[1, 1].set_ylabel("deviation from unit weight")
    axes[1, 1].set_title("Unconstrained channel-weight residuals")
    axes[1, 1].legend(frameon=False)

    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"spin23_ssss_1000_diagnostics.{suffix}", dpi=240)
    plt.close(figure)


def analyze(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Run the deterministic SSSS projection and held-out channel fit."""

    source = pd.read_csv(input_path)
    selected = select_stratified_rows(source)
    frame = build_ssss_frame(selected)
    features = _feature_matrix(frame)
    normalized = complex_column(frame, "ssss_normalized")
    train = ~frame["held_out"].to_numpy(dtype=bool)
    held_out = ~train
    coefficients = fit_channel_weights(features[train], normalized[train])

    fitted_normalized = features @ coefficients
    fit_amplitude = np.empty(len(frame), dtype=complex)
    for index, row in enumerate(frame.itertuples(index=False)):
        kin = SSSSKinematics(
            row.omega1_re + 1j * row.omega1_im,
            row.omega2_re + 1j * row.omega2_im,
            row.omega3_re + 1j * row.omega3_im,
        )
        fit_amplitude[index] = -math.pi * kin.energy_product * fitted_normalized[index]
    numerical = complex_column(frame, "ssss_raw")
    fit_relative = np.abs(numerical - fit_amplitude) / np.maximum(
        np.abs(numerical), np.finfo(float).tiny
    )
    frame["ssss_fit_re"] = fit_amplitude.real
    frame["ssss_fit_im"] = fit_amplitude.imag
    frame["ssss_fit_rel_residual"] = fit_relative

    formula_absolute = frame["ssss_abs_residual"].to_numpy(dtype=float)
    formula_relative = frame["ssss_rel_residual"].to_numpy(dtype=float)
    conservative_q_error = sum(
        pd.to_numeric(frame[f"estimated_abs_error_M{index}"], errors="coerce").to_numpy()
        for index in (1, 2, 3)
    )
    summary: dict[str, object] = {
        "input": str(input_path),
        "sample_count": int(len(frame)),
        "family_counts": {
            str(key): int(value) for key, value in frame["family"].value_counts().items()
        },
        "train_count": int(train.sum()),
        "held_out_count": int(held_out.sum()),
        "fitted_channel_weights": {
            label: {"real": value.real, "imag": value.imag, "distance_from_1": abs(value - 1.0)}
            for label, value in zip(("c12", "c13", "c23"), coefficients, strict=True)
        },
        "formula_relative_residual": {
            "median": float(np.median(formula_relative)),
            "rms": float(np.sqrt(np.mean(formula_relative**2))),
            "maximum": float(np.max(formula_relative)),
        },
        "unconstrained_fit_relative_residual": {
            "train_rms": float(np.sqrt(np.mean(fit_relative[train] ** 2))),
            "held_out_rms": float(np.sqrt(np.mean(fit_relative[held_out] ** 2))),
            "held_out_maximum": float(np.max(fit_relative[held_out])),
        },
        "formula_residual_over_conservative_adjacent_q_change": {
            "median": float(np.nanmedian(formula_absolute / conservative_q_error)),
            "fraction_below_one": float(np.nanmean(formula_absolute <= conservative_q_error)),
        },
        "interpretation": (
            "Raw reduced worldsheet projection A+B+C only; no asymptotic singlet "
            "leg factors or common heterotic sphere normalization are included."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "spin23_ssss_1000.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_results(frame, coefficients, output_dir)
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
