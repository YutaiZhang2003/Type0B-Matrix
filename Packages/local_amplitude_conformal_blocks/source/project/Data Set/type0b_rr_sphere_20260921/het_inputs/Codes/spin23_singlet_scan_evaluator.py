#!/usr/bin/env python3
r"""Manifest evaluator for the genuine Spin(23) singlet amplitudes.

Unlike the earlier ``spin23_ssvv_projection`` and ``spin23_ssss_projection``
utilities, this module inserts the physical anti-holomorphic
super-Liouville descendants explicitly.  It evaluates both

.. math::

   S(\omega_0)\longrightarrow S(\omega_1)V^a(\omega_2)V^b(\omega_3),
   \qquad
   S(\omega_0)\longrightarrow S(\omega_1)S(\omega_2)S(\omega_3),

with ``omega0=omega1+omega2+omega3``.  One output CSV is written atomically
per manifest row, which makes Slurm requeues safe.

The ``legacy_projection_*`` columns contain the simple formulas inferred by
treating the singlet as a projected 24th vector.  They are retained as
hypotheses to test; they are not inputs to the genuine calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from spin23_highdim_scan import read_row, row_energies
from spin23_singlet_amplitudes import (
    SingletAmplitudeEvaluation,
    SingletAmplitudeValues,
    evaluate_singlet_amplitudes,
)
from spin23_ssss_projection import SSSSKinematics, raw_ssss_candidate
import ns_elliptic_conversion as elliptic_conversion
from spin23_ssvv_projection import SSVVKinematics, raw_ssvv_candidate


_TINY = np.finfo(float).tiny


def _settings_signature(settings: Mapping[str, Any]) -> str:
    options = _numerical_options(settings)
    canonical = json.dumps({"controls":options,
        "representation":elliptic_conversion.representation_metadata(options["series_parameter"],options["q_order"])},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _numerical_options(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a scan profile into the amplitude-engine controls."""

    q_order = int(settings.get("q_order", 5))
    estimate_q_error = bool(settings.get("estimate_q_error", True))
    angular = int(settings.get("z_angular_nodes", 60))
    radial = int(settings.get("z_radial_nodes", 24))
    return {
        "q_order": q_order,
        "lower_q_order": q_order - 1 if estimate_q_error and q_order > 1 else None,
        "p_nodes": settings.get("p_nodes", "segmented"),
        "p_max": float(settings.get("p_max", 4.0)),
        "p_cut": float(settings.get("p_cut", 0.03)),
        "momentum_scheme": str(settings.get("momentum_scheme", "threshold_weighted")),
        "momentum_threshold_options": settings.get("momentum_threshold_options"),
        "infinite_gauss_scale": float(settings.get("infinite_gauss_scale", 1.0)),
        "block_backend": str(settings.get("block_backend", "c_recursion")),
        "series_parameter": str(settings.get("series_parameter", "elliptic_nome")),
        "recursion_digits": int(settings.get("recursion_digits", 70)),
        "recursion_reference_p_max": float(settings.get("recursion_reference_p_max", 0.18)),
        "recursion_cancellation_limit": float(settings.get("recursion_cancellation_limit", 1.0e6)),
        "epsilon0": float(settings.get("ope_radius", 0.08)),
        "epsilon1": float(settings.get("crossed_ope_radius", 0.06)),
        "theta_orders": tuple(
            int(value)
            for value in settings.get(
                "theta_orders",
                (max(12, angular // 4), max(12, angular // 4), angular),
            )
        ),
        "radial_order": radial,
        "disk_total_order": int(settings.get("disk_total_order", q_order + 11)),
        "crossed_disk_total_order": int(
            settings.get("crossed_disk_total_order", q_order + 11)
        ),
        "gram_condition_limit": float(
            settings.get("gram_condition_limit", 1.0e13)
        ),
        "gram_high_precision_condition": (
            None
            if settings.get("gram_high_precision_condition") is None
            else float(settings["gram_high_precision_condition"])
        ),
        "gram_high_precision_digits": int(
            settings.get("gram_high_precision_digits", 60)
        ),
        "gram_high_precision_max_momentum": (
            None
            if settings.get("gram_high_precision_max_momentum") is None
            else float(settings["gram_high_precision_max_momentum"])
        ),
    }


def _put_complex(result: dict[str, Any], name: str, value: complex) -> None:
    value = complex(value)
    result[f"{name}_re"] = float(value.real)
    result[f"{name}_im"] = float(value.imag)


def _relative_change(current: complex, previous: complex) -> float:
    return float(abs(current - previous) / max(abs(current), _TINY))


def _piece_cancellation_ratio(pieces: Mapping[str, complex]) -> float:
    """Return ``sum(abs(piece))/abs(sum(piece))`` for one amplitude."""

    values = tuple(complex(value) for value in pieces.values())
    total = math.fsum(value.real for value in values) + 1j * math.fsum(
        value.imag for value in values
    )
    return float(sum(abs(value) for value in values) / max(abs(total), _TINY))


def _add_values(
    result: dict[str, Any],
    values: SingletAmplitudeValues,
    lower: SingletAmplitudeValues | None,
) -> None:
    amplitudes = {
        "ssvv_raw": values.ssvv_raw,
        "ssss_raw": values.ssss_raw,
        "ssvv_unit_descendants": values.ssvv_unit_descendants,
        "ssss_unit_descendants": values.ssss_unit_descendants,
    }
    lower_amplitudes = None
    if lower is not None:
        lower_amplitudes = {
            "ssvv_raw": lower.ssvv_raw,
            "ssss_raw": lower.ssss_raw,
            "ssvv_unit_descendants": lower.ssvv_unit_descendants,
            "ssss_unit_descendants": lower.ssss_unit_descendants,
        }

    for name, value in amplitudes.items():
        _put_complex(result, name, value)
        if lower_amplitudes is None:
            result[f"{name}_adjacent_q_abs_change"] = math.nan
            result[f"{name}_adjacent_q_rel_change"] = math.nan
            continue
        lower_value = lower_amplitudes[name]
        _put_complex(result, f"{name}_lower_q", lower_value)
        result[f"{name}_adjacent_q_abs_change"] = float(abs(value - lower_value))
        result[f"{name}_adjacent_q_rel_change"] = _relative_change(
            value,
            lower_value,
        )

    for process, pieces in values.pieces.items():
        result[f"{process}_piece_cancellation_ratio"] = (
            _piece_cancellation_ratio(pieces)
        )
        for region, value in pieces.items():
            _put_complex(result, f"{process}_piece_{region}", value)


def _add_legacy_hypotheses(
    result: dict[str, Any],
    omega1: complex,
    omega2: complex,
    omega3: complex,
    values: SingletAmplitudeValues,
) -> None:
    ssvv_candidate = raw_ssvv_candidate(SSVVKinematics(omega1, omega2, omega3))
    ssss_candidate = raw_ssss_candidate(SSSSKinematics(omega1, omega2, omega3))
    for process, numerical, candidate in (
        ("ssvv", values.ssvv_raw, ssvv_candidate),
        ("ssss", values.ssss_raw, ssss_candidate),
    ):
        _put_complex(result, f"legacy_projection_{process}_candidate", candidate)
        result[f"legacy_projection_{process}_abs_residual"] = float(
            abs(numerical - candidate)
        )
        result[f"legacy_projection_{process}_rel_residual"] = float(
            abs(numerical - candidate) / max(abs(candidate), _TINY)
        )


def evaluate_point(
    omega1: complex,
    omega2: complex,
    omega3: complex,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate both genuine singlet amplitudes at one manifest point."""

    started = time.perf_counter()
    omega1, omega2, omega3 = map(complex, (omega1, omega2, omega3))
    energies = (omega1, omega2, omega3, omega1 + omega2 + omega3)
    options = _numerical_options(settings)
    evaluation = evaluate_singlet_amplitudes(energies, **options)

    result: dict[str, Any] = {}
    _add_values(result, evaluation.values, evaluation.lower_order_values)
    _add_legacy_hypotheses(
        result,
        omega1,
        omega2,
        omega3,
        evaluation.values,
    )
    result.update(
        {
            "q_order": evaluation.q_order,
            "lower_q_order": (
                evaluation.lower_q_order
                if evaluation.lower_q_order is not None
                else ""
            ),
            "momentum_nodes": evaluation.momentum_nodes,
            "momentum_scheme": evaluation.momentum_quadrature["scheme"],
            "momentum_quadrature": json.dumps(evaluation.momentum_quadrature, sort_keys=True),
            "numerical_settings_sha256": _settings_signature(settings),
            "block_backend": evaluation.block_backend,
            **elliptic_conversion.representation_metadata(evaluation.series_parameter, evaluation.q_order),
            "maximum_recursion_cancellation": evaluation.maximum_recursion_cancellation,
            "high_precision_recursion_node_count": evaluation.high_precision_recursion_node_count,
            "maximum_gram_condition": evaluation.maximum_gram_condition,
            "maximum_equilibrated_gram_condition": (
                evaluation.maximum_equilibrated_gram_condition
            ),
            "high_precision_gram_solve_count": (
                evaluation.high_precision_gram_solve_count
            ),
            "template_seconds": evaluation.template_seconds,
            "build_seconds": evaluation.build_seconds,
            "integration_seconds": evaluation.integration_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "status": "ok",
            "singlet_definition": "explicit_super_liouville_descendant",
            "crossed_ope_geometry": "unit_disk_lens",
            "legacy_projection_is_input": False,
        }
    )
    return result


def _atomic_write_csv(
    path: Path,
    manifest_row: Mapping[str, str],
    result: Mapping[str, Any],
) -> None:
    """Write one complete row, replacing an earlier requeued task safely."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(manifest_row)
    row.update({name: str(value) for name, value in result.items()})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_manifest_row(
    manifest: Path,
    index: int,
    settings_path: Path,
    output: Path,
    *,
    settings_name: str | None = None,
) -> None:
    row = read_row(manifest, index)
    omega1, omega2, omega3 = row_energies(row)
    settings_table = json.loads(settings_path.read_text(encoding="utf-8"))
    setting_name = row["setting"] if settings_name is None else settings_name
    if setting_name not in settings_table:
        raise KeyError(f"unknown numerical setting {setting_name!r}")
    result = evaluate_point(
        omega1,
        omega2,
        omega3,
        settings_table[setting_name],
    )
    result["settings_profile"] = setting_name
    _atomic_write_csv(output, row, result)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_successfully(path: Path, expected_signature: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
    except (OSError, StopIteration, csv.Error):
        return False
    if row.get("status") != "ok":
        return False
    if expected_signature is not None and row.get("numerical_settings_sha256") != expected_signature:
        raise ValueError(f"{path} has different or unrecorded numerical settings; use a fresh result directory")
    return True


def run_manifest_shard(
    manifest: Path,
    task_index: int,
    task_count: int,
    settings_path: Path,
    output_dir: Path,
    *,
    settings_name: str | None = None,
    progress_every: int = 1,
) -> int:
    """Evaluate one strided shard while retaining point-level output files.

    Successful files with identical numerical settings are skipped on restart.
    Legacy/different-rule files are never silently reused or overwritten.
    A numerical exception is
    recorded for that point and the remaining shard continues; the process
    exits nonzero after the shard so Slurm still reports that intervention is
    required.
    """

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if not 0 <= task_index < task_count:
        raise ValueError("task_index must lie in [0, task_count)")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    rows = _read_manifest(manifest)
    settings_table = json.loads(settings_path.read_text(encoding="utf-8"))
    indices = list(range(task_index, len(rows), task_count))
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    completed = 0
    skipped = 0
    shard_started = time.perf_counter()
    for local_index, global_index in enumerate(indices, start=1):
        row = rows[global_index]
        output = output_dir / f"result_{global_index}.csv"
        effective_setting_name = row["setting"] if settings_name is None else settings_name
        if effective_setting_name not in settings_table:
            raise KeyError(f"unknown numerical setting {effective_setting_name!r}")
        point_settings = settings_table[effective_setting_name]
        if _completed_successfully(output, _settings_signature(point_settings)):
            skipped += 1
            continue
        point_started = time.perf_counter()
        try:
            omega1, omega2, omega3 = row_energies(row)
            result = evaluate_point(
                omega1,
                omega2,
                omega3,
                point_settings,
            )
            result["settings_profile"] = effective_setting_name
            _atomic_write_csv(output, row, result)
            completed += 1
        except Exception as error:  # keep unrelated points in this shard running
            failures += 1
            failure = {
                "status": "error",
                "runtime_seconds": time.perf_counter() - point_started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc().replace("\n", "\\n"),
            }
            _atomic_write_csv(output, row, failure)
        if local_index % progress_every == 0 or local_index == len(indices):
            elapsed = time.perf_counter() - shard_started
            print(
                f"task {task_index}/{task_count}: {local_index}/{len(indices)} "
                f"rows visited, {completed} computed, {skipped} skipped, "
                f"{failures} failed, {elapsed:.1f} s",
                flush=True,
            )
    return 1 if failures else 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int)
    mode.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--settings-json", type=Path, required=True)
    parser.add_argument("--settings-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.index is not None:
        if args.output is None or args.output_dir is not None:
            raise SystemExit("--index requires --output and forbids --output-dir")
        run_manifest_row(
            args.manifest,
            args.index,
            args.settings_json,
            args.output,
            settings_name=args.settings_name,
        )
        return 0
    if args.task_count is None or args.output_dir is None or args.output is not None:
        raise SystemExit(
            "--task-index requires --task-count and --output-dir, and forbids --output"
        )
    return run_manifest_shard(
        args.manifest,
        args.task_index,
        args.task_count,
        args.settings_json,
        args.output_dir,
        settings_name=args.settings_name,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
