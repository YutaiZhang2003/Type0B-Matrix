#!/usr/bin/env python3
"""Merge and diagnose the genuine ``S -> SVV`` and ``S -> SSS`` scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_TINY = np.finfo(float).tiny


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {name: math.nan for name in ("min", "median", "p90", "p99", "max")}
    return {
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _complex(row: Mapping[str, str], stem: str) -> complex:
    return complex(float(row[f"{stem}_re"]), float(row[f"{stem}_im"]))


def _block_backend_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    """Handle both legacy Gram rows and new recursion rows with blank Gram data."""
    summary: dict[str, object] = {
        "block_backend_counts": dict(Counter(row.get("block_backend") or "inverse_gram_legacy" for row in rows)),
        "series_parameter_counts": dict(Counter(row.get("series_parameter") or "elliptic_nome_legacy" for row in rows)),
        "momentum_scheme_counts": dict(Counter(row.get("momentum_scheme") or "unrecorded_legacy" for row in rows)),
    }
    for field in ("maximum_gram_condition", "maximum_recursion_cancellation"):
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        summary[field] = _percentiles(values) if values else None
    return summary


def _fit_complex_scale(candidate: np.ndarray, numerical: np.ndarray) -> complex:
    denominator = np.vdot(candidate, candidate)
    if abs(denominator) <= _TINY:
        raise ZeroDivisionError("candidate values vanish on the fit sample")
    return complex(np.vdot(candidate, numerical) / denominator)


def _scaled_fit_summary(
    rows: Sequence[Mapping[str, str]],
    process: str,
    normalization: str,
) -> dict[str, object]:
    numerical_stem = f"{process}_{normalization}"
    candidate_stem = f"legacy_projection_{process}_candidate"
    numerical = np.asarray([_complex(row, numerical_stem) for row in rows])
    candidate = np.asarray([_complex(row, candidate_stem) for row in rows])
    train = np.asarray(
        [row["family"] == "core" and index % 5 != 0 for index, row in enumerate(rows)]
    )
    validation = ~train
    scale = _fit_complex_scale(candidate[train], numerical[train])
    residual = np.abs(numerical - scale * candidate) / np.maximum(
        np.abs(numerical),
        _TINY,
    )
    return {
        "complex_scale": {"real": scale.real, "imag": scale.imag},
        "training_points": int(np.sum(train)),
        "validation_points": int(np.sum(validation)),
        "training_relative_residual": _percentiles(residual[train]),
        "validation_relative_residual": _percentiles(residual[validation]),
    }


def _symmetry_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("family") == "symmetry":
            groups[row.get("parent_id", "")].append(row)

    ssss_errors: list[float] = []
    ssvv_errors: list[float] = []
    for group in groups.values():
        if len(group) != 6:
            continue
        ssss = np.asarray([_complex(row, "ssss_raw") for row in group])
        mean = np.mean(ssss)
        ssss_errors.extend(np.abs(ssss - mean) / max(abs(mean), _TINY))

        # SSVV designates the first *current* outgoing leg as the singlet.
        # For each possible physical singlet, the two permutations with that
        # first entry differ only by exchanging the two vector legs.
        by_singlet: dict[str, list[complex]] = defaultdict(list)
        for row in group:
            by_singlet[row["permutation"][0]].append(_complex(row, "ssvv_raw"))
        for pair in by_singlet.values():
            if len(pair) == 2:
                scale = max(abs(pair[0]), abs(pair[1]), _TINY)
                ssvv_errors.append(abs(pair[0] - pair[1]) / scale)
    return {
        "complete_six_permutation_groups": sum(len(group) == 6 for group in groups.values()),
        "ssss_all_outgoing_permutations": _percentiles(ssss_errors),
        "ssvv_vector_pair_exchange": _percentiles(ssvv_errors),
    }


def _relative_difference(first: complex, second: complex) -> float:
    """Return a symmetric relative difference between two complex values."""

    return float(abs(first - second) / max(abs(first), abs(second), _TINY))


def _convergence_control_summary(
    primary_rows: Sequence[Mapping[str, str]],
    control_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """Compare repeated numerical-control rows with their parent evaluations."""

    parents = {row.get("point_id", ""): row for row in primary_rows}
    controls: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing_parents: list[str] = []
    for row in control_rows:
        parent_id = row.get("parent_id", "")
        parent = parents.get(parent_id)
        if parent is None:
            missing_parents.append(parent_id)
            continue
        profile = row.get("setting", "")
        for process in ("ssvv", "ssss"):
            parent_stem = f"{process}_raw"
            if profile == "p_extended":
                parent_stem = f"{process}_raw_lower_q"
            controls[profile][process].append(
                _relative_difference(
                    _complex(row, f"{process}_raw"),
                    _complex(parent, parent_stem),
                )
            )

    return {
        "control_points": len(control_rows),
        "missing_parent_ids": sorted(set(missing_parents)),
        "profiles": {
            profile: {
                process: _percentiles(values)
                for process, values in process_values.items()
            }
            for profile, process_values in controls.items()
        },
    }


def reduce_results(
    results_dir: Path,
    *,
    expected: int | None,
    relative_q_tolerance: float,
) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    malformed: list[str] = []
    for path in sorted(results_dir.glob("result_*.csv")):
        try:
            with path.open(newline="") as handle:
                file_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            malformed.append(path.name)
            continue
        if len(file_rows) != 1:
            malformed.append(path.name)
            continue
        rows.append(file_rows[0])
    if not rows:
        raise SystemExit(f"no one-row result CSV files found in {results_dir}")

    point_ids = [row.get("point_id", "") for row in rows]
    duplicates = sorted(point_id for point_id, count in Counter(point_ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"duplicate point IDs found: {duplicates[:5]}")
    if expected is not None and len(rows) != expected:
        raise SystemExit(f"found {len(rows)} result rows; expected {expected}")

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with (results_dir / "merged.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    summary: dict[str, object] = {
        "result_files": len(rows),
        "successful_points": len(ok),
        "failed_points": len(failed),
        "malformed_files": malformed,
        "failure_types": dict(Counter(row.get("error_type", "unknown") for row in failed)),
        "family_counts": dict(Counter(row.get("family", "") for row in ok)),
        "setting_counts": dict(Counter(row.get("setting", "") for row in ok)),
    }
    if not ok:
        (results_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    primary = [row for row in ok if row.get("family") != "convergence"]
    controls = [row for row in ok if row.get("family") == "convergence"]
    summary["primary_points"] = len(primary)
    summary["convergence_control_points"] = len(controls)
    summary["runtime_seconds"] = _percentiles(float(row["runtime_seconds"]) for row in ok)
    summary["total_cpu_hours"] = float(
        sum(float(row["runtime_seconds"]) for row in ok) / 3600
    )
    summary.update(_block_backend_summary(ok))
    summary["amplitudes"] = {}
    jointly_converged: list[dict[str, str]] = []
    for process in ("ssvv", "ssss"):
        q_field = f"{process}_raw_adjacent_q_rel_change"
        q_changes = np.asarray([float(row[q_field]) for row in primary])
        converged = np.isfinite(q_changes) & (q_changes <= relative_q_tolerance)
        candidate_residual = [
            float(row[f"legacy_projection_{process}_rel_residual"])
            for row in primary
        ]
        summary["amplitudes"][process] = {
            "adjacent_q_relative_change": _percentiles(q_changes),
            "points_below_relative_q_tolerance": int(np.sum(converged)),
            "relative_q_tolerance": relative_q_tolerance,
            "unscaled_legacy_projection_relative_residual": _percentiles(candidate_residual),
            "raw_scaled_legacy_projection_fit": _scaled_fit_summary(
                primary,
                process,
                "raw",
            ),
            "unit_descendant_scaled_legacy_projection_fit": _scaled_fit_summary(
                primary,
                process,
                "unit_descendants",
            ),
        }
    for row in primary:
        if all(
            math.isfinite(float(row[f"{process}_raw_adjacent_q_rel_change"]))
            and float(row[f"{process}_raw_adjacent_q_rel_change"])
            <= relative_q_tolerance
            for process in ("ssvv", "ssss")
        ):
            jointly_converged.append(dict(row))
    summary["jointly_converged_points"] = len(jointly_converged)
    summary["symmetry"] = _symmetry_summary(primary)
    summary["independent_numerical_controls"] = _convergence_control_summary(
        primary,
        controls,
    )

    if jointly_converged:
        with (results_dir / "converged.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(jointly_converged)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--expected", type=int, default=6888)
    parser.add_argument("--relative-q-tolerance", type=float, default=1.0e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = reduce_results(
        args.results_dir,
        expected=args.expected,
        relative_q_tolerance=args.relative_q_tolerance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
