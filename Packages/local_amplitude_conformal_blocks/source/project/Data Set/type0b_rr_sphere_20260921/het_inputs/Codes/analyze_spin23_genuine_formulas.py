#!/usr/bin/env python3
"""Validate and plot the genuine Spin(23) singlet closed-form candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from spin23_genuine_formulas import (
    s_to_sss_raw_candidate,
    s_to_svv_raw_candidate,
)


_TINY = np.finfo(float).tiny
_FAMILY_ORDER = ("core", "structured", "soft", "near-real", "symmetry")
_COLORS = {
    "core": "#0072B2",
    "structured": "#D55E00",
    "soft": "#009E73",
    "near-real": "#CC79A7",
    "symmetry": "#E69F00",
}


def _complex(row: Mapping[str, str], stem: str) -> complex:
    return complex(float(row[f"{stem}_re"]), float(row[f"{stem}_im"]))


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return {
            key: math.nan
            for key in ("minimum", "median", "rms", "p90", "p99", "maximum")
        }
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "maximum": float(np.max(data)),
    }


def _load_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("result_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            loaded = list(csv.DictReader(handle))
        if len(loaded) != 1:
            raise ValueError(f"{path} contains {len(loaded)} rows rather than one")
        if loaded[0].get("status") == "ok":
            rows.append(loaded[0])
    if not rows:
        raise ValueError(f"no successful result rows found in {results_dir}")
    rows.sort(key=lambda row: row["point_id"])
    return rows


def _kinematics(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(float(row[f"omega{leg}_re"]), float(row[f"omega{leg}_im"]))
                for leg in (1, 2, 3)
            ]
            for row in rows
        ],
        dtype=complex,
    )


def _candidate_values(omega: np.ndarray, process: str) -> np.ndarray:
    function = {
        "ssvv": s_to_svv_raw_candidate,
        "ssss": s_to_sss_raw_candidate,
    }[process]
    return np.asarray([function(*point) for point in omega], dtype=complex)


def _best_complex_scale(candidate: np.ndarray, numerical: np.ndarray) -> complex:
    return complex(np.vdot(candidate, numerical) / np.vdot(candidate, candidate))


def _residual_summary(
    rows: Sequence[Mapping[str, str]],
    process: str,
    numerical: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, object]:
    residual = np.abs(numerical - candidate) / np.maximum(np.abs(numerical), _TINY)
    q_change = np.asarray(
        [float(row[f"{process}_raw_adjacent_q_rel_change"]) for row in rows]
    )
    by_family = {}
    for family in _FAMILY_ORDER:
        mask = np.asarray([row["family"] == family for row in rows])
        if np.any(mask):
            by_family[family] = _percentiles(residual[mask])
    finite_q = np.isfinite(q_change) & (q_change > 0)
    scale = _best_complex_scale(candidate, numerical)
    return {
        "unscaled_relative_residual": _percentiles(residual),
        "by_family": by_family,
        "best_complex_scale_not_applied": {"real": scale.real, "imag": scale.imag},
        "adjacent_q_relative_change": _percentiles(q_change),
        "formula_residual_over_adjacent_q_change": _percentiles(
            residual[finite_q] / q_change[finite_q]
        ),
    }


def _relative_formula_residual(numerical: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return ``|numerical-candidate|/|numerical|`` point by point."""

    return np.abs(numerical - candidate) / np.maximum(np.abs(numerical), _TINY)


def _convergence_formula_summary(
    primary_rows: Sequence[Mapping[str, str]],
    control_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """Measure whether independent numerical controls approach the formulas.

    ``p_extended`` is a q=6, larger-momentum-cutoff calculation.  It is
    therefore compared with the q=6 value already stored in the production
    parent's ``*_raw_lower_q`` columns.  Every other profile is compared with
    the parent's production ``*_raw`` value.
    """

    parents = {row["point_id"]: row for row in primary_rows}
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    missing_parents: list[str] = []
    for row in control_rows:
        parent_id = row.get("parent_id", "")
        if parent_id not in parents:
            missing_parents.append(parent_id)
            continue
        grouped[row["setting"]].append(row)

    profiles: dict[str, object] = {}
    for profile, profile_rows in sorted(grouped.items()):
        profile_summary: dict[str, object] = {"points": len(profile_rows)}
        omega = _kinematics(profile_rows)
        for process in ("ssvv", "ssss"):
            candidate = _candidate_values(omega, process)
            parent_stem = (
                f"{process}_raw_lower_q"
                if profile == "p_extended"
                else f"{process}_raw"
            )
            parent_values = np.asarray(
                [_complex(parents[row["parent_id"]], parent_stem) for row in profile_rows],
                dtype=complex,
            )
            control_values = np.asarray(
                [_complex(row, f"{process}_raw") for row in profile_rows],
                dtype=complex,
            )
            parent_residual = _relative_formula_residual(parent_values, candidate)
            control_residual = _relative_formula_residual(control_values, candidate)
            parent_gap = np.abs(parent_values - candidate)
            control_gap = np.abs(control_values - candidate)
            nonzero = parent_gap > _TINY
            gap_removed = np.full(len(profile_rows), np.nan, dtype=float)
            gap_removed[nonzero] = (
                parent_gap[nonzero] - control_gap[nonzero]
            ) / parent_gap[nonzero]
            shift = np.abs(control_values - parent_values) / np.maximum(
                np.abs(parent_values), _TINY
            )
            profile_summary[process] = {
                "parent_formula_relative_residual": _percentiles(parent_residual),
                "control_formula_relative_residual": _percentiles(control_residual),
                "control_over_parent_residual": _percentiles(
                    control_residual / np.maximum(parent_residual, _TINY)
                ),
                "numerical_control_shift": _percentiles(shift),
                "fraction_of_formula_gap_removed": _percentiles(gap_removed),
                "fraction_moving_closer_to_formula": float(
                    np.mean(control_gap < parent_gap)
                ),
            }
        profile_summary["coefficient_fits"] = {
            "ssvv": _ssvv_compact_coefficient_comparison(
                omega,
                np.asarray(
                    [
                        _complex(
                            parents[row["parent_id"]],
                            (
                                "ssvv_raw_lower_q"
                                if profile == "p_extended"
                                else "ssvv_raw"
                            ),
                        )
                        for row in profile_rows
                    ],
                    dtype=complex,
                ),
                np.asarray(
                    [_complex(row, "ssvv_raw") for row in profile_rows],
                    dtype=complex,
                ),
            ),
            "ssss": _ssss_compact_coefficient_comparison(
                omega,
                np.asarray(
                    [
                        _complex(
                            parents[row["parent_id"]],
                            (
                                "ssss_raw_lower_q"
                                if profile == "p_extended"
                                else "ssss_raw"
                            ),
                        )
                        for row in profile_rows
                    ],
                    dtype=complex,
                ),
                np.asarray(
                    [_complex(row, "ssss_raw") for row in profile_rows],
                    dtype=complex,
                ),
            ),
        }
        profiles[profile] = profile_summary

    return {
        "control_rows": len(control_rows),
        "missing_parent_ids": sorted(set(missing_parents)),
        "profiles": profiles,
    }


def _fit_real_coefficients(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit real coefficients to a complex linear system with column scaling."""

    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales == 0):
        raise ValueError("the compact-fit design matrix has a zero column")
    scaled = matrix / scales
    real_system = np.vstack((scaled.real, scaled.imag))
    real_target = np.concatenate((target.real, target.imag))
    return np.linalg.lstsq(real_system, real_target, rcond=1.0e-12)[0] / scales


def _coefficient_comparison(
    matrix: np.ndarray,
    parent_target: np.ndarray,
    control_target: np.ndarray,
    expected: np.ndarray,
    labels: Sequence[str],
) -> dict[str, object]:
    """Fit parent/control coefficient vectors and compare with integers."""

    parent = _fit_real_coefficients(matrix, parent_target)
    control = _fit_real_coefficients(matrix, control_target)
    return {
        "basis": list(labels),
        "expected": expected.tolist(),
        "parent": parent.tolist(),
        "control": control.tolist(),
        "parent_minus_expected_l2": float(np.linalg.norm(parent - expected)),
        "control_minus_expected_l2": float(np.linalg.norm(control - expected)),
    }


def _ssvv_compact_coefficient_comparison(
    omega: np.ndarray,
    parent: np.ndarray,
    control: np.ndarray,
) -> dict[str, object]:
    """Compare the seven quadratic ``S -> SVV`` numerator coefficients."""

    x = 1j * omega
    u = x[:, 0]
    v = x[:, 1] + x[:, 2]
    p = x[:, 1] * x[:, 2]
    x0 = u + v
    product = x0 * np.prod(x, axis=1)
    denominator = 1 + v
    matrix = np.column_stack(
        (np.ones(len(omega)), p, v, v**2, u, u * v, u**2)
    )
    normalization = denominator / (np.pi * product)
    expected = np.asarray((1, 0, 3, 2, 2, 3, 1), dtype=float)
    return _coefficient_comparison(
        matrix,
        parent * normalization,
        control * normalization,
        expected,
        ("1", "x2*x3", "v", "v^2", "u", "u*v", "u^2"),
    )


def _ssss_compact_coefficient_comparison(
    omega: np.ndarray,
    parent: np.ndarray,
    control: np.ndarray,
) -> dict[str, object]:
    """Compare the seven factorized ``S -> SSS`` coefficients."""

    x = 1j * omega
    s1 = np.sum(x, axis=1)
    s2 = x[:, 0] * x[:, 1] + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
    product = s1 * np.prod(x, axis=1)
    denominators = np.column_stack(
        (
            1 + x[:, 0] + x[:, 1],
            1 + x[:, 0] + x[:, 2],
            1 + x[:, 1] + x[:, 2],
        )
    )
    matrix = np.column_stack(
        (
            product * np.sum(1 / denominators, axis=1),
            np.ones(len(omega)),
            s2,
            s1,
            s1 * s2,
            s1**2,
            s1**3,
        )
    )
    expected = np.asarray((1, 1, 1, 2, 2, -1, -2), dtype=float)
    normalization = 1 / (np.pi * product)
    return _coefficient_comparison(
        matrix,
        parent * normalization,
        control * normalization,
        expected,
        ("pair residue", "1", "s2", "s1", "s1*s2", "s1^2", "s1^3"),
    )


def _ssvv_exchange_summary(
    rows: Sequence[Mapping[str, str]],
    numerical: np.ndarray,
) -> dict[str, float]:
    groups: dict[tuple[str, str], list[complex]] = defaultdict(list)
    for row, value in zip(rows, numerical):
        if row["family"] != "symmetry":
            continue
        groups[(row["parent_id"], row["permutation"][0])].append(value)
    errors = []
    for values in groups.values():
        if len(values) == 2:
            errors.append(abs(values[0] - values[1]) / max(map(abs, values), default=_TINY))
    return _percentiles(errors)


def _ssss_permutation_summary(
    rows: Sequence[Mapping[str, str]],
    numerical: np.ndarray,
) -> dict[str, float]:
    groups: dict[str, list[complex]] = defaultdict(list)
    for row, value in zip(rows, numerical):
        if row["family"] == "symmetry":
            groups[row["parent_id"]].append(value)
    errors = []
    for values in groups.values():
        if len(values) != 6:
            continue
        mean = sum(values) / len(values)
        errors.extend(abs(value - mean) / max(abs(mean), _TINY) for value in values)
    return _percentiles(errors)


def _ssvv_formula_components(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized numerical variables used by the compact formula."""

    x = 1j * omega
    x0 = np.sum(x, axis=1)
    product = x0 * np.prod(x, axis=1)
    shape = (1 + x0) * (1 + x0 + x[:, 1] + x[:, 2]) / (
        1 + x[:, 1] + x[:, 2]
    )
    return product, shape


def _core_training_mask(
    rows: Sequence[Mapping[str, str]],
) -> np.ndarray:
    """Return the deterministic four-fifths core training split."""

    return np.asarray(
        [
            row["family"] == "core"
            and int(
                hashlib.blake2b(row["point_id"].encode(), digest_size=2).hexdigest(),
                16,
            )
            % 5
            != 0
            for row in rows
        ]
    )


def _fit_ssvv_quadratic_coefficients(
    rows: Sequence[Mapping[str, str]],
    omega: np.ndarray,
    numerical: np.ndarray,
) -> dict[str, object]:
    r"""Fit the minimal vector-exchange-symmetric ``S -> SVV`` numerator.

    With ``u=x1``, ``v=x2+x3``, and ``p=x2*x3``, factor out the universal
    soft factor and multiply by the factorization denominator ``1+v``.  The
    most general quadratic numerator invariant under ``x2 <-> x3`` then has
    basis ``(1, p, v, v**2, u, u*v, u**2)``.
    """

    x = 1j * omega
    u = x[:, 0]
    v = x[:, 1] + x[:, 2]
    p = x[:, 1] * x[:, 2]
    x0 = u + v
    product = x0 * np.prod(x, axis=1)
    denominator = 1 + v
    matrix = np.column_stack(
        (np.ones(len(rows)), p, v, v**2, u, u * v, u**2)
    )
    target = numerical * denominator / (np.pi * product)
    training = _core_training_mask(rows)
    if np.sum(training) < matrix.shape[1]:
        raise ValueError("not enough core training rows for the SSVV coefficient fit")

    coefficients = _fit_real_coefficients(matrix[training], target[training])
    expected = np.asarray((1, 0, 3, 2, 2, 3, 1), dtype=float)
    fitted_amplitude = np.pi * product / denominator * (matrix @ coefficients)
    expected_amplitude = np.pi * product / denominator * (matrix @ expected)
    fitted_residual = _relative_formula_residual(numerical, fitted_amplitude)
    expected_residual = _relative_formula_residual(numerical, expected_amplitude)
    core = np.asarray([row["family"] == "core" for row in rows])
    core_holdout = core & ~training

    nested_models = {}
    for name, columns in (
        ("constant", (0,)),
        ("linear", (0, 2, 4)),
        ("quadratic_without_x2_x3", (0, 2, 3, 4, 5, 6)),
        ("general_symmetric_quadratic", tuple(range(matrix.shape[1]))),
    ):
        nested_matrix = matrix[:, columns]
        nested_coefficients = _fit_real_coefficients(
            nested_matrix[training], target[training]
        )
        nested_amplitude = (
            np.pi
            * product
            / denominator
            * (nested_matrix @ nested_coefficients)
        )
        nested_residual = _relative_formula_residual(numerical, nested_amplitude)
        nested_models[name] = {
            "coefficient_count": len(columns),
            "core_holdout_relative_residual": _percentiles(
                nested_residual[core_holdout]
            ),
        }

    labels = ("1", "x2*x3", "v", "v^2", "u", "u*v", "u^2")
    return {
        "basis": list(labels),
        "expected_coefficients": expected.tolist(),
        "fitted_coefficients": coefficients.tolist(),
        "fitted_minus_expected": (coefficients - expected).tolist(),
        "fitted_minus_expected_l2": float(np.linalg.norm(coefficients - expected)),
        "training_points": int(np.sum(training)),
        "core_holdout_points": int(np.sum(core_holdout)),
        "nested_model_selection": nested_models,
        "fitted_training_relative_residual": _percentiles(fitted_residual[training]),
        "fitted_core_holdout_relative_residual": _percentiles(
            fitted_residual[core_holdout]
        ),
        "expected_training_relative_residual": _percentiles(
            expected_residual[training]
        ),
        "expected_core_holdout_relative_residual": _percentiles(
            expected_residual[core_holdout]
        ),
        "expected_nontraining_relative_residual": _percentiles(
            expected_residual[~training]
        ),
        "expected_by_family": {
            family: _percentiles(
                expected_residual[
                    np.asarray([row["family"] == family for row in rows])
                ]
            )
            for family in _FAMILY_ORDER
        },
    }


def _fit_ssss_factorized_coefficients(
    rows: Sequence[Mapping[str, str]],
    omega: np.ndarray,
    numerical: np.ndarray,
) -> dict[str, object]:
    r"""Fit the residue and six symmetric contact coefficients as real numbers.

    The seven columns multiply

    ``Pi*sum(1/dij), 1, s2, s1, s1*s2, s1**2, s1**3``

    in ``M/(pi*Pi)``.  The conjectured exact coefficients are
    ``(1, 1, 1, 2, 2, -1, -2)``.
    """

    x = 1j * omega
    s1 = np.sum(x, axis=1)
    s2 = x[:, 0] * x[:, 1] + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
    product = s1 * np.prod(x, axis=1)
    denominators = np.column_stack(
        (
            1 + x[:, 0] + x[:, 1],
            1 + x[:, 0] + x[:, 2],
            1 + x[:, 1] + x[:, 2],
        )
    )
    matrix = np.column_stack(
        (
            product * np.sum(1 / denominators, axis=1),
            np.ones(len(rows)),
            s2,
            s1,
            s1 * s2,
            s1**2,
            s1**3,
        )
    )
    target = numerical / (np.pi * product)
    training = _core_training_mask(rows)
    if np.sum(training) < matrix.shape[1]:
        raise ValueError("not enough core training rows for the SSSS coefficient fit")
    scales = np.linalg.norm(matrix[training], axis=0)
    scaled = matrix / scales
    real_system = np.vstack((scaled[training].real, scaled[training].imag))
    real_target = np.concatenate((target[training].real, target[training].imag))
    coefficients = np.linalg.lstsq(real_system, real_target, rcond=1.0e-12)[0] / scales
    expected = np.asarray((1, 1, 1, 2, 2, -1, -2), dtype=float)
    fitted_amplitude = np.pi * product * (matrix @ coefficients)
    expected_amplitude = np.pi * product * (matrix @ expected)
    fitted_residual = np.abs(fitted_amplitude - numerical) / np.maximum(
        np.abs(numerical), _TINY
    )
    expected_residual = np.abs(expected_amplitude - numerical) / np.maximum(
        np.abs(numerical), _TINY
    )
    labels = ("pair residue", "1", "s2", "s1", "s1*s2", "s1^2", "s1^3")
    return {
        "basis": list(labels),
        "expected_coefficients": expected.tolist(),
        "fitted_coefficients": coefficients.tolist(),
        "fitted_minus_expected": (coefficients - expected).tolist(),
        "training_points": int(np.sum(training)),
        "core_holdout_points": int(np.sum(~training & np.asarray([r["family"] == "core" for r in rows]))),
        "fitted_training_relative_residual": _percentiles(fitted_residual[training]),
        "expected_training_relative_residual": _percentiles(expected_residual[training]),
        "expected_nontraining_relative_residual": _percentiles(expected_residual[~training]),
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _identity_plot(
    rows: Sequence[Mapping[str, str]],
    numerical: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    for column, process in enumerate(("ssvv", "ssss")):
        for row_index, (part, accessor) in enumerate((("Re", np.real), ("Im", np.imag))):
            axis = axes[row_index, column]
            x_values = accessor(candidate[process])
            y_values = accessor(numerical[process])
            low = min(float(np.min(x_values)), float(np.min(y_values)))
            high = max(float(np.max(x_values)), float(np.max(y_values)))
            padding = 0.04 * max(high - low, 1.0e-12)
            axis.plot(
                [low - padding, high + padding],
                [low - padding, high + padding],
                color="black",
                linewidth=1.1,
                zorder=0,
            )
            for family in _FAMILY_ORDER:
                mask = np.asarray([row["family"] == family for row in rows])
                if np.any(mask):
                    axis.scatter(
                        x_values[mask],
                        y_values[mask],
                        s=8,
                        alpha=0.55,
                        linewidths=0,
                        color=_COLORS[family],
                        label=family if row_index == 0 and column == 0 else None,
                    )
            axis.set_xlabel(rf"${part}\,\mathcal{{M}}_{{\rm formula}}$")
            axis.set_ylabel(rf"${part}\,\mathcal{{M}}_{{\rm numerical}}$")
            axis.grid(alpha=0.18, linewidth=0.6)
        axes[0, column].set_title(
            r"$S\to SVV$" if process == "ssvv" else r"$S\to SSS$"
        )
    axes[0, 0].legend(frameon=False, ncol=2)
    for extension, options in (("pdf", {}), ("png", {"dpi": 240})):
        figure.savefig(output_dir / f"spin23_genuine_formula_identity.{extension}", **options)
    plt.close(figure)


def _residual_plot(
    rows: Sequence[Mapping[str, str]],
    omega: np.ndarray,
    numerical: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    total_energy = np.abs(np.sum(omega, axis=1))
    for axis, process in zip(axes, ("ssvv", "ssss")):
        residual = np.abs(numerical[process] - candidate[process]) / np.maximum(
            np.abs(numerical[process]), _TINY
        )
        q_change = np.asarray(
            [float(row[f"{process}_raw_adjacent_q_rel_change"]) for row in rows]
        )
        axis.scatter(
            total_energy,
            q_change,
            s=7,
            alpha=0.18,
            linewidths=0,
            color="black",
            label=r"adjacent-$q$ change",
        )
        for family in _FAMILY_ORDER:
            mask = np.asarray([row["family"] == family for row in rows])
            if np.any(mask):
                axis.scatter(
                    total_energy[mask],
                    residual[mask],
                    s=9,
                    alpha=0.60,
                    linewidths=0,
                    color=_COLORS[family],
                    label=family,
                )
        axis.set_yscale("log")
        axis.set_xlabel(r"$|\omega_0|$")
        axis.set_ylabel("relative difference")
        axis.set_title(r"$S\to SVV$" if process == "ssvv" else r"$S\to SSS$")
        axis.grid(alpha=0.18, linewidth=0.6)
    axes[1].legend(frameon=False, ncol=2)
    for extension, options in (("pdf", {}), ("png", {"dpi": 240})):
        figure.savefig(output_dir / f"spin23_genuine_formula_residuals.{extension}", **options)
    plt.close(figure)


def _ssvv_global_error_histogram(
    rows: Sequence[Mapping[str, str]],
    numerical: np.ndarray,
    candidate: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    r"""Plot and summarize the global production S-to-SVV error.

    The plotted pointwise metric is
    abs(M_numerical-M_formula)/abs(M_numerical).
    The right panel normalizes each scan family separately because the family
    sizes are design choices rather than samples from a common distribution.
    """

    residual = _relative_formula_residual(numerical, candidate)
    finite = np.isfinite(residual) & (residual > 0)
    if not np.any(finite):
        raise ValueError("no finite positive SSVV residuals for the histogram")
    residual = residual[finite]
    finite_rows = [row for row, keep in zip(rows, finite) if keep]

    lower_decade = math.floor(math.log10(float(np.min(residual))))
    upper_decade = math.ceil(math.log10(float(np.max(residual))))
    if lower_decade == upper_decade:
        upper_decade += 1
    bins = np.geomspace(10.0**lower_decade, 10.0**upper_decade, 33)
    counts, _ = np.histogram(residual, bins=bins)
    percentiles = {
        "median": float(np.median(residual)),
        "p90": float(np.percentile(residual, 90)),
        "p99": float(np.percentile(residual, 99)),
    }

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    axes[0].hist(
        residual,
        bins=bins,
        color="#0072B2",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
    )
    line_styles = {
        "median": ("black", "-", "median"),
        "p90": ("#D55E00", "--", "90th percentile"),
        "p99": ("#CC79A7", ":", "99th percentile"),
    }
    for name, value in percentiles.items():
        color, style, label = line_styles[name]
        axes[0].axvline(
            value,
            color=color,
            linestyle=style,
            linewidth=1.5,
            label=rf"{label}: {value:.2e}",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(
        r"$|\mathcal{M}_{\rm num}-\mathcal{M}_{\rm formula}|/"
        r"|\mathcal{M}_{\rm num}|$"
    )
    axes[0].set_ylabel("number of points")
    axes[0].set_title(rf"All production points ($N={len(residual):,}$)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.6)

    family_counts: dict[str, list[int]] = {}
    for family in _FAMILY_ORDER:
        mask = np.asarray([row["family"] == family for row in finite_rows])
        if not np.any(mask):
            continue
        family_residual = residual[mask]
        weights = np.full(len(family_residual), 1.0 / len(family_residual))
        family_histogram, _ = np.histogram(family_residual, bins=bins)
        family_counts[family] = family_histogram.tolist()
        axes[1].hist(
            family_residual,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=1.5,
            color=_COLORS[family],
            label=rf"{family} ($N={len(family_residual):,}$)",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(
        r"$|\mathcal{M}_{\rm num}-\mathcal{M}_{\rm formula}|/"
        r"|\mathcal{M}_{\rm num}|$"
    )
    axes[1].set_ylabel("fraction of family per bin")
    axes[1].set_title("Breakdown by scan family")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.6)

    for extension, options in (("pdf", {}), ("png", {"dpi": 250})):
        figure.savefig(
            output_dir / f"spin23_ssvv_global_error_histogram.{extension}",
            **options,
        )
    plt.close(figure)

    thresholds = (1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)
    return {
        "metric": "abs(numerical-formula)/abs(numerical)",
        "point_count": len(residual),
        "mean": float(np.mean(residual)),
        "median": percentiles["median"],
        "rms": float(np.sqrt(np.mean(residual**2))),
        "p90": percentiles["p90"],
        "p99": percentiles["p99"],
        "maximum": float(np.max(residual)),
        "fractions_below": {
            f"{threshold:.0e}": float(np.mean(residual < threshold))
            for threshold in thresholds
        },
        "bin_edges": bins.tolist(),
        "all_counts": counts.tolist(),
        "family_counts": family_counts,
        "sampling_note": (
            "This is the empirical distribution over the designed scan, not "
            "a probability distribution over physical kinematics."
        ),
    }


def analyze(results_dir: Path, output_dir: Path) -> dict[str, object]:
    rows = _load_rows(results_dir)
    primary_mask = np.asarray([row["family"] != "convergence" for row in rows])
    primary_rows = [row for row, keep in zip(rows, primary_mask) if keep]
    control_rows = [row for row, keep in zip(rows, primary_mask) if not keep]
    omega = _kinematics(primary_rows)
    numerical = {
        process: np.asarray([_complex(row, f"{process}_raw") for row in primary_rows])
        for process in ("ssvv", "ssss")
    }
    candidate = {
        process: _candidate_values(omega, process) for process in ("ssvv", "ssss")
    }
    ssvv_summary = _residual_summary(
        primary_rows, "ssvv", numerical["ssvv"], candidate["ssvv"]
    )
    summary: dict[str, object] = {
        "result_rows": len(rows),
        "primary_rows": len(primary_rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "formulas_are_unscaled": True,
        "ssvv": ssvv_summary,
        "ssss": _residual_summary(
            primary_rows, "ssss", numerical["ssss"], candidate["ssss"]
        ),
        "ssvv_quadratic_coefficient_fit": _fit_ssvv_quadratic_coefficients(
            primary_rows, omega, numerical["ssvv"]
        ),
        "ssss_factorized_coefficient_fit": _fit_ssss_factorized_coefficients(
            primary_rows, omega, numerical["ssss"]
        ),
        "independent_convergence_controls": _convergence_formula_summary(
            primary_rows, control_rows
        ),
        "symmetry": {
            "ssvv_vector_exchange": _ssvv_exchange_summary(
                primary_rows, numerical["ssvv"]
            ),
            "ssss_outgoing_permutations": _ssss_permutation_summary(
                primary_rows, numerical["ssss"]
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    ssvv_summary["global_error_histogram"] = _ssvv_global_error_histogram(
        primary_rows,
        numerical["ssvv"],
        candidate["ssvv"],
        output_dir,
    )
    (output_dir / "spin23_genuine_formula_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _identity_plot(primary_rows, numerical, candidate, output_dir)
    _residual_plot(primary_rows, omega, numerical, candidate, output_dir)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir or args.results_dir / "formula_analysis"
    summary = analyze(args.results_dir, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
