#!/usr/bin/env python3
"""Cluster worker and assembler for the mixed genus-two period table.

Nothing is evaluated unless the ``worker`` command is given an explicit
``--execute`` flag.  Each array task owns one JSONL file; there are no shared
appends.  The manifest fixes routing before submission:

* all three q values small: adaptive Schottky with a word-tail certificate;
* every other ordinary point: normalized-holomorphic-form collocation;
* mixed deep cusps or failed primary certificates: high-precision
  holomorphic-form collocation.

The last backend is deliberately a plugin boundary.  A production launch is
not complete until ``genus2_multiprecision_collocation`` supplies a function
named ``evaluate_multiprecision_holomorphic_period_map`` and declares
``BACKEND_READY = True``.  Missing support is recorded, never silently
replaced by Schottky in a mixed cusp.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

try:
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        MethodEvaluation,
        evaluate_holomorphic_period_map,
        evaluate_schottky_period_map,
        period_max_residual,
    )
    from genus2_period_table import Genus2PeriodMapTable, format_complex
    from genus2_period_table_grid import (
        DEFAULT_CONFIG,
        config_sha256,
        load_config,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        MethodEvaluation,
        evaluate_holomorphic_period_map,
        evaluate_schottky_period_map,
        period_max_residual,
    )
    from plumbing.genus2_period_table import Genus2PeriodMapTable, format_complex
    from plumbing.genus2_period_table_grid import DEFAULT_CONFIG, config_sha256, load_config


RESULT_SCHEMA_VERSION = 2
MP_MODULE = "genus2_multiprecision_collocation"


def _load_mp_backend() -> Callable[..., MethodEvaluation] | None:
    candidates = (MP_MODULE, f"plumbing.{MP_MODULE}")
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        evaluator = getattr(module, "evaluate_multiprecision_holomorphic_period_map", None)
        if bool(getattr(module, "BACKEND_READY", False)) and callable(evaluator):
            return evaluator
    return None


def _hybrid_config(
    payload: dict[str, object],
    *,
    maximum_basis_override: int | None = None,
) -> HybridPeriodMapConfig:
    numerics = payload["numerics"]  # type: ignore[index]
    domain = payload["q_domain"]  # type: ignore[index]
    return HybridPeriodMapConfig(
        tolerance=float(numerics["period_tolerance"]),
        agreement_tolerance=float(numerics["agreement_tolerance"]),
        collocation_min_q=float(domain["standard_collocation_min_q"]),
        collocation_comfortable_min_q=float(domain["standard_collocation_min_q"]),
        schottky_all_small_q_max_theta=float(
            payload["schottky_all_small_q_max"]["theta"]  # type: ignore[index]
        ),
        schottky_all_small_q_max_glasses=float(
            payload["schottky_all_small_q_max"]["glasses"]  # type: ignore[index]
        ),
        method_boundary_log_half_width=float(
            payload["method_boundary_log_half_width"]
        ),
        overlap_q_max=max(
            float(value)
            for value in payload["schottky_all_small_q_max"].values()  # type: ignore[index]
        ),
        minimum_geometry_margin=float(domain["geometry_margin_min"]),
        comfortable_geometry_margin=float(domain["geometry_refinement_margin"]),
        maximum_collocation_basis=(
            int(maximum_basis_override)
            if maximum_basis_override is not None
            else int(numerics["maximum_collocation_basis"])
        ),
        minimum_schottky_word=int(numerics["minimum_schottky_word"]),
        maximum_schottky_word=int(numerics["maximum_schottky_word"]),
        schottky_tail_safety_factor=float(numerics["schottky_tail_safety_factor"]),
        crosscheck_overlap=False,
        require_convergence=True,
    )


def _manifest_rows(path: Path, shard_id: int | None = None) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if shard_id is None or int(row["shard_id"]) == int(shard_id):
                yield row


def _q_and_logs(row: dict[str, str]) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    q = tuple(complex(row[f"q{edge}"]) for edge in (1, 2, 3))
    logs = tuple(
        complex(float(row[f"log_abs_q{edge}"]), float(row[f"phase_q{edge}"]))
        for edge in (1, 2, 3)
    )
    return q, logs


def _stable_fraction(row_id: str) -> float:
    value = int.from_bytes(
        hashlib.blake2b(row_id.encode("utf-8"), digest_size=8).digest(), "big"
    )
    return value / float(1 << 64)


def _requires_crosscheck(row: dict[str, str], payload: dict[str, object]) -> bool:
    if row["planned_backend"] != "adaptive-schottky":
        return False
    numerics = payload["numerics"]  # type: ignore[index]
    if float(row["q_min"]) < float(numerics["crosscheck_min_q"]):
        return False
    if float(row["geometry_margin"]) < float(numerics["crosscheck_min_geometry_margin"]):
        return False
    if row["stratum"] in {str(value) for value in numerics["crosscheck_strata"]}:
        return True
    return _stable_fraction(row["row_id"]) < float(numerics["crosscheck_bulk_fraction"])


def _run_mp(
    evaluator: Callable[..., MethodEvaluation] | None,
    topology: str,
    q: Sequence[complex],
    logs: Sequence[complex],
    payload: dict[str, object],
    *,
    maximum_basis_override: int | None = None,
) -> MethodEvaluation | None:
    if evaluator is None:
        return None
    numerics = payload["numerics"]  # type: ignore[index]
    return evaluator(
        topology,
        q,
        log_q_values=logs,
        tolerance=float(numerics["period_tolerance"]),
        maximum_basis=(
            int(maximum_basis_override)
            if maximum_basis_override is not None
            else int(numerics["maximum_collocation_basis"])
        ),
    )


def _method_fields(evaluation: MethodEvaluation) -> dict[str, object]:
    omega = np.asarray(evaluation.omega, dtype=np.complex128)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(omega.imag)))
    def finite_or_none(value: float) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    return {
        "actual_backend": evaluation.algorithm,
        "omega11": format_complex(omega[0, 0]),
        "omega12": format_complex(omega[0, 1]),
        "omega22": format_complex(omega[1, 1]),
        "error_estimate": finite_or_none(evaluation.error_estimate),
        "low_order": int(evaluation.low_order),
        "high_order": int(evaluation.high_order),
        "seam_residual": finite_or_none(evaluation.seam_residual),
        "symmetry_error": finite_or_none(evaluation.symmetry_error),
        "used_multiprecision": bool(evaluation.used_multiprecision),
        "actual_precision_tier": (
            "multiprecision" if evaluation.used_multiprecision else "binary64"
        ),
        "calibrated": bool(evaluation.calibrated),
        "im_omega_min_eigenvalue": finite_or_none(minimum_eigenvalue),
        "method_message": evaluation.message,
    }


def evaluate_manifest_row(
    row: dict[str, str],
    payload: dict[str, object],
    *,
    mp_evaluator: Callable[..., MethodEvaluation] | None,
) -> dict[str, object]:
    """Evaluate one pre-routed row; used only by the explicit worker command."""

    started = time.monotonic()
    q, logs = _q_and_logs(row)
    topology = row["topology"]
    planned = row["planned_backend"]
    retry_basis_text = str(row.get("retry_maximum_collocation_basis", "")).strip()
    retry_basis = int(retry_basis_text) if retry_basis_text else None
    configured_basis = int(payload["numerics"]["maximum_collocation_basis"])  # type: ignore[index]
    if retry_basis is not None and retry_basis < configured_basis:
        raise ValueError("retry Laurent-basis ceiling cannot be below the production ceiling")
    config = _hybrid_config(payload, maximum_basis_override=retry_basis)
    primary: MethodEvaluation | None = None
    crosscheck: MethodEvaluation | None = None
    status = "failed"
    failure = ""
    try:
        if planned == "adaptive-schottky":
            primary = evaluate_schottky_period_map(
                topology, q, config=config, log_q_values=logs  # type: ignore[arg-type]
            )
            if not primary.converged:
                promoted = _run_mp(
                    mp_evaluator,
                    topology,
                    q,
                    logs,
                    payload,
                    maximum_basis_override=retry_basis,
                )
                if promoted is None:
                    status = "needs-multiprecision-collocation"
                    failure = "Schottky word-tail certificate failed and the high-precision holomorphic backend is unavailable"
                else:
                    primary = promoted
            elif _requires_crosscheck(row, payload):
                crosscheck = evaluate_holomorphic_period_map(
                    topology, q, config=config, log_q_values=logs  # type: ignore[arg-type]
                )
        elif planned == "holomorphic-form-collocation" and row["precision_tier"] == "binary64-adaptive":
            try:
                primary = evaluate_holomorphic_period_map(
                    topology, q, config=config, log_q_values=logs  # type: ignore[arg-type]
                )
            except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
                primary = _run_mp(
                    mp_evaluator,
                    topology,
                    q,
                    logs,
                    payload,
                    maximum_basis_override=retry_basis,
                )
            if primary is not None and not primary.converged:
                primary = _run_mp(
                    mp_evaluator,
                    topology,
                    q,
                    logs,
                    payload,
                    maximum_basis_override=retry_basis,
                )
                if primary is None:
                    status = "needs-multiprecision-collocation"
                    failure = "binary64 collocation failed and the high-precision holomorphic backend is unavailable"
            elif primary is None:
                status = "needs-multiprecision-collocation"
                failure = "binary64 collocation raised numerically and the high-precision holomorphic backend is unavailable"
        elif planned == "holomorphic-form-collocation":
            primary = _run_mp(
                mp_evaluator,
                topology,
                q,
                logs,
                payload,
                maximum_basis_override=retry_basis,
            )
            if primary is None:
                status = "needs-multiprecision-collocation"
                failure = "mixed-cusp policy requires the unavailable high-precision holomorphic backend"
        else:
            failure = f"unknown planned backend {planned!r}"

        agreement: float | None = None
        if primary is not None:
            if not primary.converged:
                status = "failed-certificate"
                failure = primary.message
            else:
                omega = np.asarray(primary.omega, dtype=np.complex128)
                min_eigenvalue = float(np.min(np.linalg.eigvalsh(omega.imag)))
                if not np.all(np.isfinite(omega)) or min_eigenvalue <= 0.0:
                    status = "failed-riemann-matrix"
                    failure = f"Im(Omega) is not positive definite: lambda_min={min_eigenvalue:.3e}"
                elif crosscheck is not None:
                    if not crosscheck.converged:
                        status = "failed-crosscheck-certificate"
                        failure = crosscheck.message
                    else:
                        agreement = period_max_residual(primary.omega, crosscheck.omega)
                        if not math.isfinite(agreement):
                            status = "failed-crosscheck-agreement"
                            failure = "Schottky/holomorphic agreement residual is nonfinite"
                            agreement = None
                        elif agreement > float(payload["numerics"]["agreement_tolerance"]):  # type: ignore[index]
                            status = "failed-crosscheck-agreement"
                            failure = f"Schottky/holomorphic disagreement {agreement:.3e} exceeds the configured bar"
                        else:
                            status = "ok"
                else:
                    status = "ok"
        output: dict[str, object] = {
            **row,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "status": status,
            "certified": status == "ok",
            "failure": failure,
            "crosscheck_performed": crosscheck is not None,
            "crosscheck_agreement": agreement,
            "elapsed_seconds": time.monotonic() - started,
        }
        if primary is not None:
            output.update(_method_fields(primary))
        if crosscheck is not None:
            output.update(
                {
                    f"crosscheck_{key}": value
                    for key, value in _method_fields(crosscheck).items()
                }
            )
        return output
    except Exception as exc:  # a failed point must not kill unrelated array work
        return {
            **row,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "status": "exception",
            "certified": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "crosscheck_performed": False,
            "crosscheck_agreement": None,
            "elapsed_seconds": time.monotonic() - started,
        }


def _read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    last_good_offset = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                break
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at {path}:{line_number}") from exc
            completed.add(str(row["row_id"]))
            last_good_offset = handle.tell()
    if path.stat().st_size != last_good_offset:
        # A scheduler kill may interrupt the final write.  Only that incomplete
        # suffix is discarded; every fsync-completed JSON line is retained.
        with path.open("r+b") as handle:
            handle.truncate(last_good_offset)
    return completed


def run_worker(
    *,
    manifest: Path,
    output_dir: Path,
    shard_id: int,
    payload: dict[str, object],
    limit: int | None,
) -> dict[str, object]:
    expected_digest = config_sha256(payload)
    task_count = int(payload["array_task_count"])
    if not 0 <= int(shard_id) < task_count:
        raise ValueError(f"shard id must lie in [0,{task_count})")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / f"shard-{int(shard_id):04d}.partial.jsonl"
    final = output_dir / f"shard-{int(shard_id):04d}.jsonl"
    if final.exists():
        raise FileExistsError(f"completed shard already exists: {final}")
    completed = _read_completed_ids(partial)
    mp_evaluator = _load_mp_backend()
    counts: Counter[str] = Counter()
    written = 0
    with partial.open("a") as handle:
        for row in _manifest_rows(manifest, shard_id):
            if row["config_sha256"] != expected_digest:
                raise ValueError(f"manifest/config digest mismatch at row {row['row_id']}")
            if row["row_id"] in completed:
                continue
            result = evaluate_manifest_row(row, payload, mp_evaluator=mp_evaluator)
            handle.write(
                json.dumps(
                    result,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            counts[str(result["status"])] += 1
            written += 1
            if limit is not None and written >= int(limit):
                break
    if limit is None:
        os.replace(partial, final)
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": expected_digest,
        "shard_id": int(shard_id),
        "new_rows": written,
        "status_counts_for_new_rows": dict(counts),
        "multiprecision_collocation_backend_ready": mp_evaluator is not None,
        "complete": limit is None,
        "output": str(final if limit is None else partial),
    }
    summary_path = output_dir / f"shard-{int(shard_id):04d}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def preflight(manifest: Path, payload: dict[str, object]) -> dict[str, object]:
    expected_digest = config_sha256(payload)
    routes: Counter[str] = Counter()
    shards: Counter[int] = Counter()
    crosschecks = 0
    rows = 0
    selected_targets: set[int] = set()
    selector_provenance_complete = True
    for row in _manifest_rows(manifest):
        if row["config_sha256"] != expected_digest:
            raise ValueError(f"manifest/config digest mismatch at row {row['row_id']}")
        rows += 1
        if not row.get("atlas_target_index", "").strip():
            selector_provenance_complete = False
        else:
            selected_targets.add(int(row["atlas_target_index"]))
        routes[f"{row['planned_backend']}:{row['precision_tier']}"] += 1
        shards[int(row["shard_id"])] += 1
        crosschecks += int(_requires_crosscheck(row, payload))
    expected_shards = int(payload["array_task_count"])
    expected_targets = 1 << int(payload["atlas_design"]["target_sample_power"])  # type: ignore[index]
    target_fraction = len(selected_targets) / float(expected_targets)
    required_target_fraction = float(
        payload["atlas_design"]["required_selected_target_fraction"]  # type: ignore[index]
    )
    selector_ready = bool(
        payload["useful_region"]["status"] == "selector-implemented-and-certified"  # type: ignore[index]
        and selector_provenance_complete
        and target_fraction >= required_target_fraction
    )
    mp_ready = _load_mp_backend() is not None
    return {
        "manifest": str(manifest),
        "config_sha256": expected_digest,
        "row_count": rows,
        "route_counts": dict(routes),
        "planned_crosschecks": crosschecks,
        "atlas_target_count": expected_targets,
        "selected_atlas_target_count": len(selected_targets),
        "selected_atlas_target_fraction": target_fraction,
        "required_selected_atlas_target_fraction": required_target_fraction,
        "selector_provenance_complete": selector_provenance_complete,
        "useful_region_selector_ready": selector_ready,
        "nonempty_shards": len(shards),
        "expected_shards": expected_shards,
        "minimum_rows_per_nonempty_shard": min(shards.values()) if shards else 0,
        "maximum_rows_per_shard": max(shards.values()) if shards else 0,
        "multiprecision_collocation_backend_ready": mp_ready,
        "production_ready": len(shards) == expected_shards and mp_ready and selector_ready,
        "period_matrices_evaluated": False,
    }


def _iter_result_rows(inputs: Iterable[Path]) -> Iterator[dict[str, object]]:
    for path in inputs:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                yield row


def build_retry_manifest(
    *,
    manifest: Path,
    failed_results: Path,
    output: Path,
    payload: dict[str, object],
    maximum_basis: int,
) -> dict[str, object]:
    """Select rejected rows and route them to the stabilized high-order solver."""

    configured_basis = int(payload["numerics"]["maximum_collocation_basis"])  # type: ignore[index]
    if int(maximum_basis) <= configured_basis:
        raise ValueError("retry basis ceiling must exceed the production ceiling")
    failures: dict[str, dict[str, object]] = {}
    for result in _iter_result_rows((failed_results,)):
        row_id = str(result["row_id"])
        if row_id in failures:
            raise ValueError(f"duplicate failed result row {row_id}")
        if result.get("status") == "ok" or bool(result.get("certified", False)):
            raise ValueError(f"retry input contains a successful row {row_id}")
        failures[row_id] = result
    if not failures:
        raise ValueError("retry input contains no failed rows")

    temporary = output.with_name(f".{output.name}.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    shards: Counter[int] = Counter()
    source_statuses: Counter[str] = Counter()
    with manifest.open(newline="") as source, temporary.open("w", newline="") as target:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("source manifest has no header")
        extra_fields = (
            "retry_maximum_collocation_basis",
            "retry_source_status",
            "retry_source_failure",
        )
        fieldnames = [*reader.fieldnames, *(name for name in extra_fields if name not in reader.fieldnames)]
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            row_id = row["row_id"]
            failure = failures.get(row_id)
            if failure is None:
                continue
            row["planned_backend"] = "holomorphic-form-collocation"
            row["precision_tier"] = "multiprecision-rescaled"
            row["retry_maximum_collocation_basis"] = str(int(maximum_basis))
            row["retry_source_status"] = str(failure.get("status", ""))
            row["retry_source_failure"] = str(failure.get("failure", ""))
            writer.writerow(row)
            found.add(row_id)
            shards[int(row["shard_id"])] += 1
            source_statuses[str(failure.get("status", ""))] += 1
    missing = sorted(set(failures) - found)
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{len(missing)} failed rows are absent from the source manifest")
    os.replace(temporary, output)
    summary = {
        "schema_version": 1,
        "config_sha256": config_sha256(payload),
        "source_manifest": str(manifest),
        "source_failed_results": str(failed_results),
        "retry_manifest": str(output),
        "retry_row_count": len(found),
        "retry_shard_count": len(shards),
        "retry_maximum_collocation_basis": int(maximum_basis),
        "source_status_counts": dict(source_statuses),
        "routing": "stabilized high-order holomorphic-form collocation",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def retry_preflight(manifest: Path, payload: dict[str, object]) -> dict[str, object]:
    expected_digest = config_sha256(payload)
    configured_basis = int(payload["numerics"]["maximum_collocation_basis"])  # type: ignore[index]
    rows = 0
    row_ids: set[str] = set()
    shards: Counter[int] = Counter()
    ceilings: Counter[int] = Counter()
    source_statuses: Counter[str] = Counter()
    for row in _manifest_rows(manifest):
        row_id = row["row_id"]
        if row_id in row_ids:
            raise ValueError(f"duplicate retry row {row_id}")
        row_ids.add(row_id)
        if row["config_sha256"] != expected_digest:
            raise ValueError(f"retry manifest/config digest mismatch at row {row_id}")
        if row["planned_backend"] != "holomorphic-form-collocation":
            raise ValueError(f"retry row {row_id} is not routed to holomorphic collocation")
        if row["precision_tier"] != "multiprecision-rescaled":
            raise ValueError(f"retry row {row_id} is not routed to the stabilized precision tier")
        ceiling = int(row["retry_maximum_collocation_basis"])
        if ceiling <= configured_basis:
            raise ValueError(f"retry row {row_id} does not raise the Laurent-basis ceiling")
        rows += 1
        shards[int(row["shard_id"])] += 1
        ceilings[ceiling] += 1
        source_statuses[row["retry_source_status"]] += 1
    return {
        "manifest": str(manifest),
        "config_sha256": expected_digest,
        "retry_row_count": rows,
        "nonempty_retry_shards": len(shards),
        "minimum_rows_per_nonempty_shard": min(shards.values()) if shards else 0,
        "maximum_rows_per_shard": max(shards.values()) if shards else 0,
        "retry_basis_ceiling_counts": dict(ceilings),
        "source_status_counts": dict(source_statuses),
        "multiprecision_collocation_backend_ready": _load_mp_backend() is not None,
        "retry_ready": bool(rows and _load_mp_backend() is not None),
    }


def overlay_retry_results(
    *,
    base_shard_dir: Path,
    retry_shard_dir: Path,
    output_dir: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    """Atomically replace every rejected base row with one certified retry row."""

    expected_digest = config_sha256(payload)
    base_paths = sorted(base_shard_dir.glob("shard-*.jsonl"))
    expected_shards = int(payload["array_task_count"])
    if len(base_paths) != expected_shards:
        raise ValueError(f"expected {expected_shards} base shards, found {len(base_paths)}")

    replacements: dict[str, dict[str, object]] = {}
    for retry_path in sorted(retry_shard_dir.glob("shard-*.jsonl")):
        for row in _iter_result_rows((retry_path,)):
            row_id = str(row["row_id"])
            if row_id in replacements:
                raise ValueError(f"duplicate retry result row {row_id}")
            if str(row.get("config_sha256")) != expected_digest:
                raise ValueError(f"retry result/config digest mismatch at row {row_id}")
            if row.get("status") != "ok" or not bool(row.get("certified", False)):
                raise RuntimeError(f"retry row {row_id} is not certified: {row.get('status')}")
            replacements[row_id] = row
    if not replacements:
        raise ValueError("retry shard directory contains no result rows")

    failed_base_ids: set[str] = set()
    successful_base_ids: set[str] = set()
    for row in _iter_result_rows(base_paths):
        row_id = str(row["row_id"])
        if str(row.get("config_sha256")) != expected_digest:
            raise ValueError(f"base result/config digest mismatch at row {row_id}")
        if row.get("status") == "ok" and bool(row.get("certified", False)):
            successful_base_ids.add(row_id)
        else:
            failed_base_ids.add(row_id)
    retry_ids = set(replacements)
    if retry_ids & successful_base_ids:
        raise RuntimeError("retry overlay attempted to replace already certified base rows")
    missing = failed_base_ids - retry_ids
    unexpected = retry_ids - failed_base_ids
    if missing or unexpected:
        raise RuntimeError(
            f"retry overlay mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("shard-*.jsonl")):
        raise FileExistsError(f"overlay output directory is not empty: {output_dir}")
    status_counts: Counter[str] = Counter()
    total_rows = 0
    for base_path in base_paths:
        destination = output_dir / base_path.name
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w") as handle:
            for base_row in _iter_result_rows((base_path,)):
                row_id = str(base_row["row_id"])
                row = replacements.get(row_id, base_row)
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                status_counts[str(row["status"])] += 1
                total_rows += 1
        os.replace(temporary, destination)

    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": expected_digest,
        "base_shard_count": len(base_paths),
        "retry_rows_applied": len(replacements),
        "preserved_certified_rows": len(successful_base_ids),
        "merged_row_count": total_rows,
        "status_counts": dict(status_counts),
        "complete": status_counts == Counter({"ok": total_rows}),
        "output_directory": str(output_dir),
    }
    if not summary["complete"]:
        raise RuntimeError("retry overlay did not produce an all-successful shard set")
    (output_dir / "retry_overlay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deterministic_gzip(source: Path, destination: Path) -> None:
    """Write a reproducible gzip copy suitable for long-term archiving."""

    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1 << 20)
    os.replace(temporary, destination)


def _artifact_record(path: Path, *, role: str) -> dict[str, object]:
    return {
        "path": path.name,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def assemble(
    *,
    shard_dir: Path,
    output_dir: Path,
    payload: dict[str, object],
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Assemble completed shards, rejecting duplicates and uncertified rows."""

    paths = sorted(shard_dir.glob("shard-*.jsonl"))
    if len(paths) != int(payload["array_task_count"]):
        raise ValueError(
            f"expected {payload['array_task_count']} completed shards, found {len(paths)}"
        )
    expected_digest = config_sha256(payload)
    seen: set[str] = set()
    accepted: list[dict[str, object]] = []
    statuses: Counter[str] = Counter()
    for row in _iter_result_rows(paths):
        row_id = str(row["row_id"])
        if row_id in seen:
            raise ValueError(f"duplicate result row {row_id}")
        seen.add(row_id)
        if str(row["config_sha256"]) != expected_digest:
            raise ValueError(f"result/config digest mismatch at row {row_id}")
        statuses[str(row["status"])] += 1
        if row["status"] == "ok" and bool(row["certified"]):
            accepted.append(row)
    if statuses.get("ok", 0) != sum(statuses.values()):
        raise RuntimeError(
            "assembly refused an incomplete table; status counts are " + json.dumps(dict(statuses), sort_keys=True)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "table.csv"
    temporary = output_dir / ".table.csv.tmp"
    fieldnames = sorted({key for row in accepted for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(accepted)
    os.replace(temporary, table_path)
    digest = _file_sha256(table_path)
    domain = payload["q_domain"]  # type: ignore[index]
    table = Genus2PeriodMapTable.from_csv(table_path, config_path=config_path)
    index_path = output_dir / "index_features.npz"
    table.write_portable_index(index_path, table_sha256=digest)
    compressed_table_path = output_dir / "table.csv.gz"
    _deterministic_gzip(table_path, compressed_table_path)

    config_snapshot_path = output_dir / "config.snapshot.json"
    config_snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    shard_inventory_path = output_dir / "raw_shard_inventory.json"
    shard_inventory = {
        "schema_version": 1,
        "config_sha256": expected_digest,
        "shard_count": len(paths),
        "source_directory": str(shard_dir),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in paths
        ],
    }
    shard_inventory_path.write_text(json.dumps(shard_inventory, indent=2) + "\n")

    artifacts = [
        _artifact_record(table_path, role="canonical-uncompressed-table"),
        _artifact_record(compressed_table_path, role="canonical-archival-table"),
        _artifact_record(index_path, role="portable-query-feature-index"),
        _artifact_record(config_snapshot_path, role="immutable-run-configuration"),
        _artifact_record(shard_inventory_path, role="raw-shard-provenance"),
    ]
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = {
        "dataset_schema_version": 1,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": expected_digest,
        "row_count": len(accepted),
        "topology_counts": dict(Counter(str(row["topology"]) for row in accepted)),
        "storage_policy": payload["storage"],
        "canonical_table": compressed_table_path.name,
        "working_table": table_path.name,
        "portable_query_index": index_path.name,
        "artifacts": artifacts,
        "retention": {
            "raw_shards": "retain until coverage and round-trip validation pass; archive afterward",
            "assembled_artifacts": "retain permanently on lab storage with checksums",
        },
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2) + "\n")
    manifest_record = _artifact_record(dataset_manifest_path, role="dataset-manifest")

    checksums_path = output_dir / "SHA256SUMS"
    checksums_path.write_text(
        "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in [*artifacts, manifest_record]
        )
    )
    summary = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "config_sha256": expected_digest,
        "row_count": len(accepted),
        "status_counts": dict(statuses),
        "q_abs_bounds": [float(domain["q_abs_tail_min"]), float(domain["q_abs_max"])],
        "table": str(table_path),
        "table_sha256": digest,
        "compressed_table": str(compressed_table_path),
        "portable_feature_index": str(index_path),
        "dataset_manifest": str(dataset_manifest_path),
        "checksums": str(checksums_path),
        "raw_shard_inventory": str(shard_inventory_path),
    }
    (output_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight", help="inspect routing without evaluating Omega")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    retry_manifest_parser = subparsers.add_parser(
        "build-retry-manifest", help="select failed rows for stabilized high-order recovery"
    )
    retry_manifest_parser.add_argument("--manifest", type=Path, required=True)
    retry_manifest_parser.add_argument("--failed-results", type=Path, required=True)
    retry_manifest_parser.add_argument("--output", type=Path, required=True)
    retry_manifest_parser.add_argument("--maximum-basis", type=int, default=224)
    retry_preflight_parser = subparsers.add_parser(
        "retry-preflight", help="validate a failed-row-only retry manifest"
    )
    retry_preflight_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser = subparsers.add_parser("worker", help="evaluate one explicitly authorized array shard")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--shard-id", type=int, required=True)
    worker_parser.add_argument("--limit", type=int)
    worker_parser.add_argument(
        "--execute",
        action="store_true",
        help="required guard acknowledging that period-map evaluations will run",
    )
    assemble_parser = subparsers.add_parser("assemble", help="assemble only fully successful shards")
    assemble_parser.add_argument("--shard-dir", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    overlay_parser = subparsers.add_parser(
        "overlay-retry", help="replace rejected base rows with certified retry rows"
    )
    overlay_parser.add_argument("--base-shard-dir", type=Path, required=True)
    overlay_parser.add_argument("--retry-shard-dir", type=Path, required=True)
    overlay_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = load_config(args.config)
    if args.command == "preflight":
        print(json.dumps(preflight(args.manifest, payload), indent=2))
    elif args.command == "build-retry-manifest":
        print(
            json.dumps(
                build_retry_manifest(
                    manifest=args.manifest,
                    failed_results=args.failed_results,
                    output=args.output,
                    payload=payload,
                    maximum_basis=args.maximum_basis,
                ),
                indent=2,
            )
        )
    elif args.command == "retry-preflight":
        print(json.dumps(retry_preflight(args.manifest, payload), indent=2))
    elif args.command == "worker":
        if not args.execute:
            parser.error("worker requires --execute; no period computation was launched")
        print(
            json.dumps(
                run_worker(
                    manifest=args.manifest,
                    output_dir=args.output_dir,
                    shard_id=args.shard_id,
                    payload=payload,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
    elif args.command == "assemble":
        print(
            json.dumps(
                assemble(
                    shard_dir=args.shard_dir,
                    output_dir=args.output_dir,
                    payload=payload,
                    config_path=args.config,
                ),
                indent=2,
            )
        )
    elif args.command == "overlay-retry":
        print(
            json.dumps(
                overlay_retry_results(
                    base_shard_dir=args.base_shard_dir,
                    retry_shard_dir=args.retry_shard_dir,
                    output_dir=args.output_dir,
                    payload=payload,
                ),
                indent=2,
            )
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    run()
