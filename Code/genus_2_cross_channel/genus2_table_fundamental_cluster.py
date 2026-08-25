#!/usr/bin/env python3
"""Cluster worker, assembly, and validation for fundamental table periods."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from bolza_torus_plumbing_reach import transform_omega
    from genus2_siegel_fundamental_domain import gottschling_min_margin
    from genus2_table_fundamental_reduction import (
        J4,
        omega_from_row,
        reduce_table_row_adaptive,
    )
    from liouville_genus2 import parse_complex
except ImportError:  # pragma: no cover
    from plumbing.bolza_torus_plumbing_reach import transform_omega
    from plumbing.genus2_siegel_fundamental_domain import gottschling_min_margin
    from plumbing.genus2_table_fundamental_reduction import (
        J4,
        omega_from_row,
        reduce_table_row_adaptive,
    )
    from plumbing.liouville_genus2 import parse_complex


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def _load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    accepted: list[str] = []
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        row_id = str(row["row_id"])
        if row_id in rows:
            raise RuntimeError(f"duplicate checkpoint row {row_id}")
        rows[row_id] = row
        accepted.append(json.dumps(row, separators=(",", ":")))
    # Drop an interrupted final line, if present, before resuming.
    path.write_text("\n".join(accepted) + ("\n" if accepted else ""))
    return rows


def worker(
    table_path: Path,
    output_dir: Path,
    *,
    task_id: int,
    task_count: int,
    correction_depths: Sequence[int],
    failed_row_file: Path | None = None,
    accept_first_certified: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"fundamental-shard-{task_id:04d}.jsonl"
    summary_path = output_dir / f"fundamental-shard-{task_id:04d}.summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        if payload.get("status") == "complete":
            print(f"task {task_id} already complete")
            return

    existing = _load_checkpoint(result_path)
    wanted: set[str] | None = None
    if failed_row_file is not None:
        if not failed_row_file.exists():
            raise RuntimeError(f"failed-row inventory is missing: {failed_row_file}")
        wanted = set()
        with failed_row_file.open() as handle:
            for line in handle:
                if line.strip():
                    wanted.add(str(json.loads(line)["row_id"]))
        if not wanted:
            raise RuntimeError("failed-row inventory is empty; no retry is required")
    assigned = 0
    processed = 0
    failed = 0
    started = time.time()
    with _open_csv(table_path) as handle, result_path.open("a") as output:
        for row in csv.DictReader(handle):
            if wanted is not None and str(row["row_id"]) not in wanted:
                continue
            shard_id = int(row["shard_id"])
            if shard_id % int(task_count) != int(task_id):
                continue
            assigned += 1
            if row["row_id"] in existing:
                if existing[row["row_id"]].get("fd_status") != "ok":
                    failed += 1
                continue
            reduced = reduce_table_row_adaptive(
                row,
                correction_depths=tuple(int(value) for value in correction_depths),
                accept_first_certified=bool(accept_first_certified),
            )
            reduced["fd_schema_version"] = SCHEMA_VERSION
            output.write(json.dumps(reduced, separators=(",", ":")) + "\n")
            output.flush()
            processed += 1
            if reduced["fd_status"] != "ok":
                failed += 1
            if processed % 50 == 0:
                print(
                    f"task={task_id} processed={processed} assigned_so_far={assigned} "
                    f"failed={failed}",
                    flush=True,
                )

    # Include failures already present in a resumed checkpoint.
    failed = sum(
        1 for row in _load_checkpoint(result_path).values() if row["fd_status"] != "ok"
    )
    final_count = len(_load_checkpoint(result_path))
    if final_count != assigned:
        raise RuntimeError(
            f"task {task_id} checkpoint has {final_count} rows but assignment has {assigned}"
        )
    summary = {
        "status": "complete",
        "task_id": int(task_id),
        "task_count": int(task_count),
        "assigned_count": assigned,
        "failed_count": failed,
        "correction_depths": [int(value) for value in correction_depths],
        "failed_row_file": str(failed_row_file) if failed_row_file is not None else None,
        "accept_first_certified": bool(accept_first_certified),
        "elapsed_seconds": time.time() - started,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _read_shards(shard_dir: Path, task_count: int) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for task_id in range(int(task_count)):
        summary_path = shard_dir / f"fundamental-shard-{task_id:04d}.summary.json"
        result_path = shard_dir / f"fundamental-shard-{task_id:04d}.jsonl"
        if not summary_path.exists() or not result_path.exists():
            raise RuntimeError(f"missing fundamental shard {task_id}")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete":
            raise RuntimeError(f"fundamental shard {task_id} is incomplete")
        if _sha256(result_path) != summary["result_sha256"]:
            raise RuntimeError(f"fundamental shard {task_id} checksum failed")
        with result_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                row_id = str(row["row_id"])
                if row_id in rows:
                    raise RuntimeError(f"duplicate fundamental row {row_id}")
                rows[row_id] = row
    return rows


def assemble(
    table_path: Path,
    shard_dir: Path,
    output_dir: Path,
    *,
    task_count: int,
    retry_shard_dir: Path | None = None,
    retry_task_count: int | None = None,
) -> None:
    augment = _read_shards(shard_dir, task_count)
    base_failures = {
        row_id for row_id, row in augment.items() if row["fd_status"] != "ok"
    }
    retry_count = 0
    if retry_shard_dir is not None or retry_task_count is not None:
        if retry_shard_dir is None or retry_task_count is None:
            raise RuntimeError("retry shard directory and task count must be supplied together")
        retry_rows = _read_shards(retry_shard_dir, retry_task_count)
        if set(retry_rows) != base_failures:
            missing = sorted(base_failures - set(retry_rows))
            extra = sorted(set(retry_rows) - base_failures)
            raise RuntimeError(
                f"retry rows do not exactly match base failures: missing={missing[:5]} "
                f"extra={extra[:5]}"
            )
        augment.update(retry_rows)
        retry_count = len(retry_rows)
    failures = [row for row in augment.values() if row["fd_status"] != "ok"]
    if failures:
        failure_path = output_dir / "failed_fundamental_rows.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in failures)
        )
        raise RuntimeError(
            f"fundamental assembly is fail-closed: {len(failures)} rows failed; "
            f"see {failure_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "fundamental_reduction.csv.gz"
    enriched_path = output_dir / "table_fundamental.csv.gz"
    row_ids: list[str] = []
    omega_fund: list[tuple[complex, complex, complex]] = []
    sp4: list[np.ndarray] = []
    branches: list[tuple[int, int, int]] = []
    margins: list[float] = []
    residuals: list[float] = []
    correction_indices: list[int] = []
    correction_depth_limits: list[int] = []
    attempted_depths: list[str] = []
    used: set[str] = set()

    with _open_csv(table_path) as source, gzip.open(
        sidecar_path, "wt", newline=""
    ) as sidecar, gzip.open(enriched_path, "wt", newline="") as enriched:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise RuntimeError("base table has no header")
        first_augmentation = next(iter(augment.values()))
        augmentation_fields = [key for key in first_augmentation if key != "row_id"]
        side_writer = csv.DictWriter(
            sidecar, fieldnames=["row_id", *augmentation_fields], lineterminator="\n"
        )
        full_writer = csv.DictWriter(
            enriched,
            fieldnames=[*reader.fieldnames, *augmentation_fields],
            lineterminator="\n",
        )
        side_writer.writeheader()
        full_writer.writeheader()
        for base in reader:
            row_id = str(base["row_id"])
            if row_id not in augment:
                raise RuntimeError(f"base row {row_id} has no fundamental augmentation")
            reduction = augment[row_id]
            used.add(row_id)
            side_writer.writerow(reduction)
            full_writer.writerow({**base, **{k: v for k, v in reduction.items() if k != "row_id"}})
            row_ids.append(row_id)
            omega_fund.append(
                (
                    parse_complex(str(reduction["fd_omega11"])),
                    parse_complex(str(reduction["fd_omega12"])),
                    parse_complex(str(reduction["fd_omega22"])),
                )
            )
            sp4.append(
                np.asarray(
                    [
                        [int(reduction[f"fd_sp4_{i}{j}"]) for j in range(4)]
                        for i in range(4)
                    ],
                    dtype=np.int64,
                )
            )
            branches.append(
                (
                    int(reduction["fd_branch11"]),
                    int(reduction["fd_branch12"]),
                    int(reduction["fd_branch22"]),
                )
            )
            margins.append(float(reduction["fd_domain_margin"]))
            residuals.append(float(reduction["fd_raw_to_fund_residual"]))
            correction_indices.append(int(reduction["fd_correction_index"]))
            correction_depth_limits.append(int(reduction["fd_correction_depth_limit"]))
            attempted_depths.append(str(reduction["fd_attempted_depths"]))

    if used != set(augment):
        raise RuntimeError(f"{len(set(augment) - used)} augmentation rows are absent from base table")
    index_path = output_dir / "fundamental_index.npz"
    np.savez_compressed(
        index_path,
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int64),
        base_table_sha256=np.asarray([_sha256(table_path)]),
        row_ids=np.asarray(row_ids),
        omega_fund=np.asarray(omega_fund, dtype=np.complex128),
        sp4_raw_to_fund=np.asarray(sp4, dtype=np.int64),
        b_period_branches=np.asarray(branches, dtype=np.int64),
        domain_margins=np.asarray(margins, dtype=np.float64),
        transform_residuals=np.asarray(residuals, dtype=np.float64),
        correction_indices=np.asarray(correction_indices, dtype=np.int64),
        correction_depth_limits=np.asarray(correction_depth_limits, dtype=np.int64),
        attempted_depths=np.asarray(attempted_depths),
    )
    depth_counts: dict[str, int] = {}
    for value in attempted_depths:
        depth_counts[value] = depth_counts.get(value, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(row_ids),
        "base_table": str(table_path),
        "base_table_sha256": _sha256(table_path),
        "minimum_domain_margin": min(margins),
        "maximum_transform_residual": max(residuals),
        "adaptive_depth_counts": depth_counts,
        "identity_correction_count": sum(index == 0 for index in correction_indices),
        "retried_row_count": retry_count,
        "files": {
            "sidecar": sidecar_path.name,
            "enriched_table": enriched_path.name,
            "portable_index": index_path.name,
        },
    }
    summary_path = output_dir / "fundamental_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    checksum_paths = (sidecar_path, enriched_path, index_path, summary_path)
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    print(json.dumps(summary, indent=2))


def validate(table_path: Path, index_path: Path, output_dir: Path, run_root: Path) -> None:
    with np.load(index_path, allow_pickle=False) as archive:
        row_ids = np.asarray(archive["row_ids"]).astype(str)
        omega_fund = np.asarray(archive["omega_fund"], dtype=np.complex128)
        sp4 = np.asarray(archive["sp4_raw_to_fund"], dtype=np.int64)
        stored_margins = np.asarray(archive["domain_margins"], dtype=np.float64)
    raw: list[np.ndarray] = []
    base_ids: list[str] = []
    with _open_csv(table_path) as handle:
        for row in csv.DictReader(handle):
            base_ids.append(str(row["row_id"]))
            raw.append(omega_from_row(row))
    if not np.array_equal(row_ids, np.asarray(base_ids)):
        raise RuntimeError("fundamental index row order differs from base table")
    if sp4.shape != (len(raw), 4, 4) or omega_fund.shape != (len(raw), 3):
        raise RuntimeError("fundamental index arrays have inconsistent shapes")

    omega_batch = np.empty((len(raw), 2, 2), dtype=np.complex128)
    omega_batch[:, 0, 0] = omega_fund[:, 0]
    omega_batch[:, 0, 1] = omega_fund[:, 1]
    omega_batch[:, 1, 0] = omega_fund[:, 1]
    omega_batch[:, 1, 1] = omega_fund[:, 2]
    margins = np.asarray(gottschling_min_margin(omega_batch), dtype=np.float64)
    maximum_residual = 0.0
    maximum_symplectic_error = 0
    for source, target, matrix in zip(raw, omega_batch, sp4):
        direct = transform_omega(matrix, source)
        direct = 0.5 * (direct + direct.T)
        maximum_residual = max(maximum_residual, float(np.max(np.abs(direct - target))))
        maximum_symplectic_error = max(
            maximum_symplectic_error,
            int(np.max(np.abs(matrix.T @ J4 @ matrix - J4))),
        )
    margin_reproduction = float(np.max(np.abs(margins - stored_margins)))
    passed = bool(
        len(raw) > 0
        and float(np.min(margins)) >= -2.0e-9
        and maximum_residual <= 2.0e-9
        and maximum_symplectic_error == 0
        and margin_reproduction <= 2.0e-12
    )
    payload = {
        "status": "passed" if passed else "failed",
        "row_count": len(raw),
        "minimum_domain_margin": float(np.min(margins)),
        "maximum_raw_to_fund_residual": maximum_residual,
        "maximum_symplectic_error": maximum_symplectic_error,
        "maximum_margin_reproduction_error": margin_reproduction,
        "base_table_sha256": _sha256(table_path),
        "fundamental_index_sha256": _sha256(index_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fundamental_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    if not passed:
        raise RuntimeError("fundamental-table validation failed")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_unix_time": time.time(),
                "assembled": str(index_path.parent),
                "validation": str(output_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload, indent=2))


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--table", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--task-id", type=int, required=True)
    worker_parser.add_argument("--task-count", type=int, required=True)
    worker_parser.add_argument("--correction-depths", type=int, nargs="+", default=(3, 5, 6, 7))
    worker_parser.add_argument("--failed-row-file", type=Path)
    worker_parser.add_argument("--accept-first-certified", action="store_true")
    assembly_parser = subparsers.add_parser("assemble")
    assembly_parser.add_argument("--table", type=Path, required=True)
    assembly_parser.add_argument("--shard-dir", type=Path, required=True)
    assembly_parser.add_argument("--output-dir", type=Path, required=True)
    assembly_parser.add_argument("--task-count", type=int, required=True)
    assembly_parser.add_argument("--retry-shard-dir", type=Path)
    assembly_parser.add_argument("--retry-task-count", type=int)
    validation_parser = subparsers.add_parser("validate")
    validation_parser.add_argument("--table", type=Path, required=True)
    validation_parser.add_argument("--index", type=Path, required=True)
    validation_parser.add_argument("--output-dir", type=Path, required=True)
    validation_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "worker":
        worker(
            args.table,
            args.output_dir,
            task_id=args.task_id,
            task_count=args.task_count,
            correction_depths=args.correction_depths,
            failed_row_file=args.failed_row_file,
            accept_first_certified=args.accept_first_certified,
        )
    elif args.command == "assemble":
        assemble(
            args.table,
            args.shard_dir,
            args.output_dir,
            task_count=args.task_count,
            retry_shard_dir=args.retry_shard_dir,
            retry_task_count=args.retry_task_count,
        )
    else:
        validate(args.table, args.index, args.output_dir, args.run_root)


if __name__ == "__main__":
    run()
