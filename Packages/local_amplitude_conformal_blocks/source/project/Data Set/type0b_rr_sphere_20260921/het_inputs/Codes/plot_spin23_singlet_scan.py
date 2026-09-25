#!/usr/bin/env python3
"""Plot genuine singlet amplitudes against the legacy projection hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


_TINY = np.finfo(float).tiny


def _complex(row: Mapping[str, str], stem: str) -> complex:
    return complex(float(row[f"{stem}_re"]), float(row[f"{stem}_im"]))


def _fit_scale(candidate: np.ndarray, numerical: np.ndarray) -> complex:
    denominator = np.vdot(candidate, candidate)
    if abs(denominator) <= _TINY:
        raise ZeroDivisionError("candidate vanishes on the training sample")
    return complex(np.vdot(candidate, numerical) / denominator)


def _limits(*values: np.ndarray) -> tuple[float, float]:
    data = np.concatenate([np.asarray(value, dtype=float) for value in values])
    finite = data[np.isfinite(data)]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    padding = 0.05 * max(hi - lo, 1.0e-12)
    return lo - padding, hi + padding


def _read_primary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("family") != "convergence"
    ]


def _process_data(
    rows: Sequence[Mapping[str, str]],
    process: str,
) -> dict[str, object]:
    numerical = np.asarray([_complex(row, f"{process}_raw") for row in rows])
    candidate = np.asarray(
        [_complex(row, f"legacy_projection_{process}_candidate") for row in rows]
    )
    training = np.asarray(
        [row.get("family") == "core" and index % 5 != 0 for index, row in enumerate(rows)]
    )
    validation = ~training
    scale = _fit_scale(candidate[training], numerical[training])
    prediction = scale * candidate
    relative = np.abs(numerical - prediction) / np.maximum(np.abs(numerical), _TINY)
    q_change = np.asarray(
        [float(row[f"{process}_raw_adjacent_q_rel_change"]) for row in rows]
    )
    return {
        "numerical": numerical,
        "prediction": prediction,
        "relative": relative,
        "q_change": q_change,
        "validation": validation,
        "scale": scale,
    }


def make_plots(merged_csv: Path, output_dir: Path) -> dict[str, object]:
    """Write direct-agreement and residual figures plus a compact summary."""

    rows = _read_primary_rows(merged_csv)
    if not rows:
        raise ValueError("merged CSV contains no successful primary rows")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {process: _process_data(rows, process) for process in ("ssvv", "ssss")}
    labels = {"ssvv": r"$S\to SVV$", "ssss": r"$S\to SSS$"}

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), constrained_layout=True)
    for column, process in enumerate(("ssvv", "ssss")):
        numerical = data[process]["numerical"]
        prediction = data[process]["prediction"]
        validation = data[process]["validation"]
        assert isinstance(numerical, np.ndarray)
        assert isinstance(prediction, np.ndarray)
        assert isinstance(validation, np.ndarray)
        for row_index, component in enumerate((np.real, np.imag)):
            x = component(prediction[validation])
            y = component(numerical[validation])
            limits = _limits(x, y)
            axis = axes[row_index, column]
            axis.plot(limits, limits, color="black", linewidth=1.1, label="equal")
            axis.scatter(x, y, s=7, alpha=0.45, linewidths=0, rasterized=True)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            part = r"\operatorname{Re}" if row_index == 0 else r"\operatorname{Im}"
            axis.set_xlabel(rf"legacy hypothesis, ${part}\,\mathcal{{A}}$")
            axis.set_ylabel(rf"genuine descendant, ${part}\,\mathcal{{A}}$")
            axis.set_title(labels[process])
    figure.savefig(output_dir / "spin23_singlet_legacy_hypothesis_agreement.pdf")
    figure.savefig(output_dir / "spin23_singlet_legacy_hypothesis_agreement.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    summary: dict[str, object] = {"primary_rows": len(rows), "processes": {}}
    for axis, process in zip(axes, ("ssvv", "ssss")):
        relative = data[process]["relative"]
        q_change = data[process]["q_change"]
        validation = data[process]["validation"]
        scale = complex(data[process]["scale"])
        assert isinstance(relative, np.ndarray)
        assert isinstance(q_change, np.ndarray)
        assert isinstance(validation, np.ndarray)
        axis.scatter(
            np.maximum(q_change[validation], 1.0e-16),
            np.maximum(relative[validation], 1.0e-16),
            s=7,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("adjacent-$q$ relative change")
        axis.set_ylabel("legacy-hypothesis relative residual")
        axis.set_title(labels[process])
        summary["processes"][process] = {
            "fitted_complex_scale": {"real": scale.real, "imag": scale.imag},
            "validation_points": int(np.sum(validation)),
            "validation_residual_median": float(np.median(relative[validation])),
            "validation_residual_p90": float(np.percentile(relative[validation], 90)),
            "adjacent_q_change_median": float(np.median(q_change[validation])),
            "adjacent_q_change_p90": float(np.percentile(q_change[validation], 90)),
        }
    figure.savefig(output_dir / "spin23_singlet_residual_vs_q_error.pdf")
    figure.savefig(output_dir / "spin23_singlet_residual_vs_q_error.png", dpi=220)
    plt.close(figure)

    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir or args.merged_csv.parent / "plots"
    summary = make_plots(args.merged_csv, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
