#!/usr/bin/env python3
"""Create full-campaign plots for the Spin(23) four-vector ansatz test."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


CHANNEL_COLORS = ("#0072B2", "#D55E00", "#009E73")
FAMILY_COLORS = {
    "core": "#0072B2",
    "structured": "#D55E00",
    "soft": "#009E73",
    "near-real": "#CC79A7",
    "symmetry": "#E69F00",
    "convergence": "#56B4E9",
}
FAMILY_LABELS = {
    "core": "generic Sobol",
    "structured": "structured kinematics",
    "soft": "soft limits",
    "near-real": "near-real continuation",
    "symmetry": "permutation checks",
    "convergence": "convergence controls",
}
FAMILY_MARKERS = {
    "core": "o",
    "structured": "s",
    "soft": "^",
    "near-real": "D",
    "symmetry": "P",
    "convergence": "X",
}
def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def _complex_columns(frame: pd.DataFrame, prefix: str, channel: int) -> np.ndarray:
    return (
        frame[f"{prefix}M{channel}_re"].to_numpy(dtype=float)
        + 1j * frame[f"{prefix}M{channel}_im"].to_numpy(dtype=float)
    )


def _relative_residuals(frame: pd.DataFrame) -> np.ndarray:
    return frame[
        [f"formula_rel_residual_M{channel}" for channel in range(1, 4)]
    ].to_numpy(dtype=float)


def _limits(*values: np.ndarray) -> tuple[float, float]:
    data = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    lower = float(np.min(data))
    upper = float(np.max(data))
    padding = 0.035 * max(upper - lower, np.finfo(float).eps)
    return lower - padding, upper + padding


def _extract_note_number(notes: pd.Series, key: str) -> np.ndarray:
    pattern = re.compile(rf"(?:^|;\s*){re.escape(key)}=([^;]+)")
    values = []
    for note in notes.fillna(""):
        match = pattern.search(str(note))
        if match is None:
            raise ValueError(f"missing {key!r} in note {note!r}")
        values.append(float(match.group(1)))
    return np.asarray(values)


def _symmetry_residuals(frame: pd.DataFrame) -> np.ndarray:
    """Compare all six permutations after mapping channels to original legs."""
    symmetry = frame.loc[frame["family"] == "symmetry"]
    residuals: list[float] = []
    for _, group in symmetry.groupby("parent_id"):
        canonical_rows = []
        for _, row in group.iterrows():
            permutation = str(row["permutation"]).zfill(3)
            if sorted(permutation) != ["0", "1", "2"]:
                raise ValueError(f"invalid permutation {permutation!r}")
            canonical = np.empty(3, dtype=complex)
            for local_index, original_index in enumerate(map(int, permutation), start=1):
                canonical[original_index] = complex(
                    row[f"M{local_index}_re"], row[f"M{local_index}_im"]
                )
            canonical_rows.append(canonical)
        values = np.asarray(canonical_rows)
        mean = np.mean(values, axis=0)
        residuals.extend(
            (np.abs(values - mean) / np.maximum(np.abs(mean), np.finfo(float).tiny)).ravel()
        )
    return np.asarray(residuals)


def _control_shifts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    indexed = frame.set_index("point_id")
    controls = frame.loc[frame["family"] == "convergence"]
    shifts: dict[str, list[float]] = {}
    for setting, group in controls.groupby("setting"):
        setting_values = shifts.setdefault(str(setting), [])
        for _, row in group.iterrows():
            parent = indexed.loc[row["parent_id"]]
            for channel in range(1, 4):
                value = complex(row[f"M{channel}_re"], row[f"M{channel}_im"])
                reference = complex(
                    parent[f"M{channel}_re"], parent[f"M{channel}_im"]
                )
                setting_values.append(
                    abs(value - reference) / max(abs(value), np.finfo(float).tiny)
                )
    return {key: np.asarray(value) for key, value in shifts.items()}


def _plot_direct_agreement(frame: pd.DataFrame, output_dir: Path, tag: str) -> None:
    numerical = {
        channel: _complex_columns(frame, "", channel) for channel in range(1, 4)
    }
    ansatz = {
        channel: _complex_columns(frame, "candidate_", channel)
        for channel in range(1, 4)
    }

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)
    for axis, component, label in (
        (axes[0, 0], np.real, r"\operatorname{Re}"),
        (axes[0, 1], np.imag, r"\operatorname{Im}"),
    ):
        all_numerical = [component(numerical[channel]) for channel in range(1, 4)]
        all_ansatz = [component(ansatz[channel]) for channel in range(1, 4)]
        limits = _limits(*all_numerical, *all_ansatz)
        axis.plot(limits, limits, color="black", linewidth=1.25, zorder=1)
        for channel in range(1, 4):
            axis.scatter(
                component(ansatz[channel]),
                component(numerical[channel]),
                s=7,
                color=CHANNEL_COLORS[channel - 1],
                alpha=0.22,
                edgecolors="none",
                rasterized=True,
                label=rf"$M_{channel}$",
                zorder=2,
            )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(rf"${label}\,M_i^{{\rm ansatz}}$")
        axis.set_ylabel(rf"${label}\,M_i^{{\rm numerical}}$")
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[0, 0].legend(frameon=False, markerscale=2.5)

    for family, group in frame.groupby("family", sort=False):
        residual = _relative_residuals(group).ravel()
        x_values, y_values = _ecdf(residual)
        axes[1, 0].step(
            x_values,
            y_values,
            where="post",
            color=FAMILY_COLORS[family],
            linewidth=1.55,
            label=FAMILY_LABELS[family],
        )
    axes[1, 0].axvline(1.0e-6, color="black", linestyle="--", linewidth=1.1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim(3.0e-9, 2.2e-5)
    axes[1, 0].set_ylim(0.0, 1.01)
    axes[1, 0].set_xlabel(r"Relative residual $|M_i-M_i^{\rm ansatz}|/|M_i^{\rm ansatz}|$")
    axes[1, 0].set_ylabel("Empirical cumulative fraction")
    axes[1, 0].grid(alpha=0.18, which="both", linewidth=0.6)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")

    for family, group in frame.groupby("family", sort=False):
        for channel in range(1, 4):
            axes[1, 1].scatter(
                group[f"estimated_abs_error_M{channel}"],
                group[f"formula_abs_residual_M{channel}"],
                s=7,
                color=FAMILY_COLORS[family],
                alpha=0.18,
                edgecolors="none",
                rasterized=True,
            )
    all_change = np.concatenate(
        [frame[f"estimated_abs_error_M{channel}"] for channel in range(1, 4)]
    )
    all_residual = np.concatenate(
        [frame[f"formula_abs_residual_M{channel}"] for channel in range(1, 4)]
    )
    limits = _limits(np.log10(all_change), np.log10(all_residual))
    log_grid = np.linspace(*limits, 100)
    grid = 10.0**log_grid
    axes[1, 1].plot(grid, grid, color="black", linewidth=1.2, label="equal")
    axes[1, 1].plot(grid, 2.0 * grid, color="black", linewidth=1.0, linestyle="--", label="factor 2")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(10.0**limits[0], 10.0**limits[1])
    axes[1, 1].set_ylim(10.0**limits[0], 2.0 * 10.0**limits[1])
    axes[1, 1].set_xlabel(r"Adjacent-$q$ absolute change")
    axes[1, 1].set_ylabel("Ansatz absolute residual")
    axes[1, 1].grid(alpha=0.18, which="both", linewidth=0.6)
    axes[1, 1].legend(frameon=False, loc="upper left")

    figure.suptitle(
        rf"Spin(23) four-vector amplitude: all {len(frame):,} kinematic points",
        fontsize=14,
    )
    base = output_dir / f"spin23_{tag}_ansatz_agreement"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_stress_tests(frame: pd.DataFrame, output_dir: Path, tag: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    # Divide out the complete conjecture; all channels and kinematics should collapse to one.
    for channel in range(1, 4):
        numerical = _complex_columns(frame, "", channel)
        ansatz = _complex_columns(frame, "candidate_", channel)
        deviation = 1.0e6 * (numerical / ansatz - 1.0)
        axes[0, 0].scatter(
            deviation.real,
            deviation.imag,
            s=7,
            alpha=0.20,
            edgecolors="none",
            color=CHANNEL_COLORS[channel - 1],
            rasterized=True,
            label=rf"$M_{channel}$",
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel(r"$10^6\,\operatorname{Re}(M_i/M_i^{\rm ansatz}-1)$")
    axes[0, 0].set_ylabel(r"$10^6\,\operatorname{Im}(M_i/M_i^{\rm ansatz}-1)$")
    axes[0, 0].set_title("Channel collapse after dividing by the ansatz")
    axes[0, 0].grid(alpha=0.18, linewidth=0.6)
    axes[0, 0].legend(frameon=False, markerscale=2.5)

    for family, group in frame.groupby("family", sort=False):
        residual = np.max(_relative_residuals(group), axis=1)
        axes[0, 1].scatter(
            group["max_pair_re_square"],
            residual,
            s=8,
            alpha=0.28,
            edgecolors="none",
            color=FAMILY_COLORS[family],
            rasterized=True,
            label=FAMILY_LABELS[family],
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"Largest $\operatorname{Re}[(\omega_j+\omega_k)^2]$")
    axes[0, 1].set_ylabel("Largest channel relative residual")
    axes[0, 1].set_title("Residual across the kinematic domain")
    axes[0, 1].grid(alpha=0.18, which="both", linewidth=0.6)
    axes[0, 1].legend(frameon=False, fontsize=8)

    soft = frame.loc[frame["family"] == "soft"]
    for channel in range(1, 4):
        numerical = np.abs(_complex_columns(soft, "", channel))
        ansatz = np.abs(_complex_columns(soft, "candidate_", channel))
        axes[1, 0].scatter(
            ansatz,
            numerical,
            s=14,
            alpha=0.42,
            edgecolors="none",
            color=CHANNEL_COLORS[channel - 1],
            rasterized=True,
            label=rf"$M_{channel}$",
        )
    soft_limits = _limits(
        np.log10(np.abs(_complex_columns(soft, "", 1))),
        np.log10(np.abs(_complex_columns(soft, "candidate_", 1))),
        np.log10(np.abs(_complex_columns(soft, "", 2))),
        np.log10(np.abs(_complex_columns(soft, "candidate_", 2))),
        np.log10(np.abs(_complex_columns(soft, "", 3))),
        np.log10(np.abs(_complex_columns(soft, "candidate_", 3))),
    )
    soft_grid = 10.0 ** np.linspace(*soft_limits, 100)
    axes[1, 0].plot(soft_grid, soft_grid, color="black", linewidth=1.2)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim(10.0**soft_limits[0], 10.0**soft_limits[1])
    axes[1, 0].set_ylim(10.0**soft_limits[0], 10.0**soft_limits[1])
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel(r"$|M_i^{\rm ansatz}|$")
    axes[1, 0].set_ylabel(r"$|M_i^{\rm numerical}|$")
    axes[1, 0].set_title("Soft-limit stress test")
    axes[1, 0].grid(alpha=0.18, which="both", linewidth=0.6)

    near_real = frame.loc[frame["family"] == "near-real"].copy()
    near_real["epsilon"] = _extract_note_number(near_real["note"], "common_epsilon")
    near_real["max_residual"] = np.max(_relative_residuals(near_real), axis=1)
    axes[1, 1].scatter(
        near_real["epsilon"],
        near_real["max_residual"],
        s=12,
        color=FAMILY_COLORS["near-real"],
        alpha=0.24,
        edgecolors="none",
        rasterized=True,
    )
    epsilon_values = np.sort(near_real["epsilon"].unique())
    medians = []
    p90 = []
    for epsilon in epsilon_values:
        values = near_real.loc[
            np.isclose(near_real["epsilon"], epsilon), "max_residual"
        ].to_numpy()
        medians.append(np.median(values))
        p90.append(np.percentile(values, 90))
    axes[1, 1].plot(
        epsilon_values, medians, color="black", marker="o", markersize=4,
        linewidth=1.4, label="median",
    )
    axes[1, 1].plot(
        epsilon_values, p90, color="black", linestyle="--", marker="s",
        markersize=3.5, linewidth=1.1, label="90th percentile",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_xlabel(r"Common imaginary part $\epsilon$")
    axes[1, 1].set_ylabel("Largest channel relative residual")
    axes[1, 1].set_title("Continuation toward real energies")
    axes[1, 1].grid(alpha=0.18, which="both", linewidth=0.6)
    axes[1, 1].legend(frameon=False)

    base = output_dir / f"spin23_{tag}_ansatz_stress_tests"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_internal_checks(frame: pd.DataFrame, output_dir: Path, tag: str) -> dict:
    symmetry = _symmetry_residuals(frame)
    controls = _control_shifts(frame)
    setting_order = ["p_extended", "ope_radius_low", "q_plus2", "quadrature_double"]
    setting_labels = [
        r"extended $P$ range",
        "smaller OPE disk",
        r"block order $q+2$",
        "double quadrature",
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    x_values, y_values = _ecdf(symmetry)
    axes[0].step(x_values, y_values, where="post", color="#0072B2", linewidth=1.7)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_xlabel("Relative spread after mapping to the original leg labels")
    axes[0].set_ylabel("Empirical cumulative fraction")
    axes[0].set_title("Outgoing-leg permutation covariance")
    axes[0].grid(alpha=0.18, which="both", linewidth=0.6)

    box = axes[1].boxplot(
        [controls[setting] for setting in setting_order],
        tick_labels=setting_labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], ("#56B4E9", "#009E73", "#E69F00", "#CC79A7")):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("Relative change from the production evaluation")
    axes[1].set_title("Single-control numerical variations")
    axes[1].grid(alpha=0.18, axis="y", which="both", linewidth=0.6)

    base = output_dir / f"spin23_{tag}_internal_checks"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    return {
        "permutation_covariance_relative_spread": {
            "samples": int(symmetry.size),
            "median": float(np.median(symmetry)),
            "p90": float(np.percentile(symmetry, 90)),
            "p99": float(np.percentile(symmetry, 99)),
            "max": float(np.max(symmetry)),
        },
        "single_control_relative_change": {
            setting: {
                "samples": int(values.size),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
            for setting, values in controls.items()
        },
    }


def _plot_a_vs_total_energy(frame: pd.DataFrame, output_dir: Path, tag: str) -> None:
    """Plot fixed-shape ansatz trajectories for the coefficient A=M3."""
    structured = frame.loc[frame["family"] == "structured"].copy()
    shapes = (
        ("10/30,10/30,10/30", r"(10,10,10)/30"),
        ("8/30,10/30,12/30", r"(8,10,12)/30"),
        ("6/30,9/30,15/30", r"(6,9,15)/30"),
        ("4/30,4/30,22/30", r"(4,4,22)/30"),
    )
    total_real_parts = np.sort(structured["omega_re"].unique())
    color_norm = Normalize(vmin=float(total_real_parts[0]), vmax=float(total_real_parts[-1]))
    color_map = plt.get_cmap("viridis")
    figure, axes = plt.subplots(2, 4, figsize=(14.8, 6.6), constrained_layout=True)
    residual_figure, residual_axes = plt.subplots(
        2, 4, figsize=(14.8, 6.6), constrained_layout=True
    )

    for column, (fraction_text, fraction_label) in enumerate(shapes):
        shape = structured.loc[
            structured["note"].eq(f"common phase; fractions={fraction_text}")
        ]
        if len(shape) != 40:
            raise ValueError(f"expected 40 structured points for {fraction_text}, found {len(shape)}")

        for total_re, trajectory in shape.groupby("omega_re", sort=True):
            trajectory = trajectory.sort_values("omega_im")
            numerical = _complex_columns(trajectory, "", 3)
            ansatz = _complex_columns(trajectory, "candidate_", 3)
            fractional = numerical / ansatz - 1.0
            color = color_map(color_norm(float(total_re)))

            for row, component in enumerate((np.real, np.imag)):
                axes[row, column].plot(
                    trajectory["omega_im"],
                    component(ansatz),
                    color=color,
                    linewidth=1.35,
                    alpha=0.92,
                    zorder=1,
                )
                axes[row, column].scatter(
                    trajectory["omega_im"],
                    component(numerical),
                    s=19,
                    color=color,
                    edgecolors="black",
                    linewidths=0.28,
                    zorder=2,
                )
                residual_axes[row, column].plot(
                    trajectory["omega_im"],
                    1.0e6 * component(fractional),
                    color=color,
                    linewidth=1.1,
                    marker="o",
                    markersize=2.8,
                    alpha=0.88,
                )

        axes[0, column].set_title(
            rf"$(\omega_1,\omega_2,\omega_3)/\omega={fraction_label}$"
        )
        residual_axes[0, column].set_title(
            rf"$(\omega_1,\omega_2,\omega_3)/\omega={fraction_label}$"
        )
        for row in range(2):
            axes[row, column].set_xlabel(r"Total $\operatorname{Im}\omega$")
            axes[row, column].grid(alpha=0.18, linewidth=0.6)
            residual_axes[row, column].axhline(0.0, color="black", linewidth=0.8)
            residual_axes[row, column].set_xlabel(r"Total $\operatorname{Im}\omega$")
            residual_axes[row, column].grid(alpha=0.18, linewidth=0.6)

    axes[0, 0].set_ylabel(r"$\operatorname{Re}A$")
    axes[1, 0].set_ylabel(r"$\operatorname{Im}A$")
    residual_axes[0, 0].set_ylabel(
        r"$10^6\operatorname{Re}(A^{\rm numerical}/A^{\rm ansatz}-1)$"
    )
    residual_axes[1, 0].set_ylabel(
        r"$10^6\operatorname{Im}(A^{\rm numerical}/A^{\rm ansatz}-1)$"
    )

    axes[0, 0].plot([], [], color="black", linewidth=1.35, label="analytic ansatz")
    axes[0, 0].scatter(
        [], [], s=19, color="white", edgecolors="black", linewidths=0.5,
        label="numerical evaluation"
    )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.015,
    )
    colorbar.set_label(r"Total $\operatorname{Re}\omega$")
    figure.suptitle(
        r"Spin(23) coefficient $A=M_3$: fixed-shape kinematic trajectories",
        fontsize=15,
    )
    residual_colorbar = residual_figure.colorbar(
        ScalarMappable(norm=color_norm, cmap=color_map),
        ax=residual_axes,
        location="right",
        shrink=0.92,
        pad=0.015,
    )
    residual_colorbar.set_label(r"Total $\operatorname{Re}\omega$")
    residual_figure.suptitle(
        r"Signed numerical residual along the fixed-shape trajectories",
        fontsize=15,
    )

    base = output_dir / f"spin23_{tag}_A_vs_total_energy"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    residual_base = output_dir / f"spin23_{tag}_A_vs_total_energy_residuals"
    residual_figure.savefig(residual_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    residual_figure.savefig(residual_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(residual_figure)


def _plot_all_a_vs_total_energy(frame: pd.DataFrame, output_dir: Path, tag: str) -> None:
    """Plot A=M3 for every completed scan evaluation."""
    numerical = _complex_columns(frame, "", 3)
    ansatz = _complex_columns(frame, "candidate_", 3)
    fractional = numerical / ansatz - 1.0
    color_norm = Normalize(
        vmin=float(frame["omega_re"].min()),
        vmax=float(frame["omega_re"].max()),
    )
    color_map = plt.get_cmap("viridis")
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)

    for axis, component, label in (
        (axes[0, 0], np.real, r"\operatorname{Re}"),
        (axes[0, 1], np.imag, r"\operatorname{Im}"),
    ):
        for family, marker in FAMILY_MARKERS.items():
            mask = frame["family"].eq(family).to_numpy()
            axis.scatter(
                frame.loc[mask, "omega_im"],
                component(numerical[mask]),
                c=frame.loc[mask, "omega_re"],
                cmap=color_map,
                norm=color_norm,
                s=9,
                marker=marker,
                alpha=0.42,
                edgecolors="none",
                rasterized=True,
                label=FAMILY_LABELS[family],
                zorder=1,
            )
        axis.scatter(
            frame["omega_im"],
            component(ansatz),
            s=7,
            marker="x",
            color="black",
            linewidths=0.38,
            alpha=0.22,
            rasterized=True,
            label="analytic ansatz",
            zorder=2,
        )
        axis.set_xlabel(r"Total $\operatorname{Im}\omega$")
        axis.set_ylabel(rf"${label}\,A$")
        axis.grid(alpha=0.18, linewidth=0.6)

    for axis, component, label in (
        (axes[1, 0], np.real, r"\operatorname{Re}"),
        (axes[1, 1], np.imag, r"\operatorname{Im}"),
    ):
        for family, marker in FAMILY_MARKERS.items():
            mask = frame["family"].eq(family).to_numpy()
            axis.scatter(
                frame.loc[mask, "omega_im"],
                1.0e6 * component(fractional[mask]),
                c=frame.loc[mask, "omega_re"],
                cmap=color_map,
                norm=color_norm,
                s=9,
                marker=marker,
                alpha=0.42,
                edgecolors="none",
                rasterized=True,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(r"Total $\operatorname{Im}\omega$")
        axis.set_ylabel(
            rf"$10^6\,{label}(A^{{\rm numerical}}/A^{{\rm ansatz}}-1)$"
        )
        axis.grid(alpha=0.18, linewidth=0.6)

    axes[0, 0].set_title(r"Real part of $A=M_3$")
    axes[0, 1].set_title(r"Imaginary part of $A=M_3$")
    axes[1, 0].set_title("Signed real fractional discrepancy")
    axes[1, 1].set_title("Signed imaginary fractional discrepancy")
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=1.8, ncol=2)
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.015,
    )
    colorbar.set_label(r"Total $\operatorname{Re}\omega$")
    figure.suptitle(
        rf"Spin(23) coefficient $A$: all {len(frame):,} completed evaluations",
        fontsize=15,
    )

    base = output_dir / f"spin23_{tag}_all_A_vs_total_energy"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_re_a_analytic_and_numerical(
    frame: pd.DataFrame, output_dir: Path, tag: str
) -> None:
    """Plot all analytic and numerical Re(A) values on matched axes."""
    numerical = frame["M3_re"].to_numpy(dtype=float)
    ansatz = frame["candidate_M3_re"].to_numpy(dtype=float)
    x_values = frame["omega_im"].to_numpy(dtype=float)
    color_values = frame["omega_re"].to_numpy(dtype=float)
    color_norm = Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    color_map = plt.get_cmap("viridis")
    x_limits = _limits(x_values)
    y_limits = _limits(numerical, ansatz)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), constrained_layout=True)
    for axis, values, title in (
        (axes[0], ansatz, r"Analytic ansatz"),
        (axes[1], numerical, r"Numerical evaluation"),
    ):
        axis.scatter(
            x_values,
            values,
            c=color_values,
            cmap=color_map,
            norm=color_norm,
            s=9,
            alpha=0.42,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_xlabel(r"Total $\operatorname{Im}\omega$")
        axis.set_ylabel(r"$\operatorname{Re}A$")
        axis.set_title(title)
        axis.grid(alpha=0.18, linewidth=0.6)

    colorbar = figure.colorbar(
        ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.015,
    )
    colorbar.set_label(r"Total $\operatorname{Re}\omega$")
    figure.suptitle(
        rf"Spin(23) coefficient $A=M_3$: all {len(frame):,} evaluations",
        fontsize=15,
    )

    base = output_dir / f"spin23_{tag}_ReA_vs_Imomega_analytic_numerical"
    figure.savefig(base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csv", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", default="full6888")
    args = parser.parse_args()

    frame = pd.read_csv(args.merged_csv, dtype={"permutation": str})
    required_rows = len(frame)
    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        required_rows = len(manifest)
        if set(frame["point_id"]) != set(manifest["point_id"]):
            raise SystemExit("merged result IDs do not match the supplied manifest")
    if len(frame) != required_rows or frame["point_id"].nunique() != required_rows:
        raise SystemExit("merged results are incomplete or contain duplicate point IDs")
    if set(frame["status"]) != {"ok"}:
        raise SystemExit(f"non-success statuses found: {frame['status'].value_counts().to_dict()}")

    output_dir = args.output_dir or args.merged_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )

    _plot_direct_agreement(frame, output_dir, args.tag)
    _plot_stress_tests(frame, output_dir, args.tag)
    checks = _plot_internal_checks(frame, output_dir, args.tag)
    _plot_a_vs_total_energy(frame, output_dir, args.tag)
    _plot_all_a_vs_total_energy(frame, output_dir, args.tag)
    _plot_re_a_analytic_and_numerical(frame, output_dir, args.tag)

    all_relative = _relative_residuals(frame).ravel()
    all_abs = np.concatenate(
        [frame[f"formula_abs_residual_M{channel}"] for channel in range(1, 4)]
    )
    all_q_change = np.concatenate(
        [frame[f"estimated_abs_error_M{channel}"] for channel in range(1, 4)]
    )
    residual_over_change = all_abs / np.maximum(all_q_change, np.finfo(float).tiny)
    summary = {
        "complete": True,
        "completed_points": int(len(frame)),
        "channel_evaluations": int(all_relative.size),
        "families": {str(key): int(value) for key, value in frame.groupby("family").size().items()},
        "relative_ansatz_residual": {
            "min": float(np.min(all_relative)),
            "median": float(np.median(all_relative)),
            "p90": float(np.percentile(all_relative, 90)),
            "p99": float(np.percentile(all_relative, 99)),
            "max": float(np.max(all_relative)),
            "fraction_below_1e-6": float(np.mean(all_relative < 1.0e-6)),
            "fraction_below_2e-6": float(np.mean(all_relative < 2.0e-6)),
            "fraction_below_5e-6": float(np.mean(all_relative < 5.0e-6)),
        },
        "ansatz_residual_vs_adjacent_q_change": {
            "median_ratio": float(np.median(residual_over_change)),
            "fraction_below_one": float(np.mean(residual_over_change <= 1.0)),
            "fraction_below_two": float(np.mean(residual_over_change <= 2.0)),
        },
        **checks,
    }
    summary_path = output_dir / f"spin23_{args.tag}_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
