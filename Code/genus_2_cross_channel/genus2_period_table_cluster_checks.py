#!/usr/bin/env python3
"""Recovery-workflow checks for the genus-two period-table cluster driver."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
from pathlib import Path

try:
    from genus2_period_table_cluster import (
        build_retry_manifest,
        overlay_retry_results,
        retry_preflight,
    )
    from genus2_period_table_grid import config_sha256, load_config
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_period_table_cluster import (
        build_retry_manifest,
        overlay_retry_results,
        retry_preflight,
    )
    from plumbing.genus2_period_table_grid import config_sha256, load_config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def check_retry_manifest_and_overlay() -> None:
    payload = copy.deepcopy(load_config())
    payload["array_task_count"] = 2
    digest = config_sha256(payload)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_manifest = root / "manifest.csv"
        fieldnames = (
            "row_id",
            "config_sha256",
            "planned_backend",
            "precision_tier",
            "shard_id",
        )
        source_rows = [
            {
                "row_id": f"row-{index}",
                "config_sha256": digest,
                "planned_backend": "adaptive-schottky",
                "precision_tier": "multiprecision",
                "shard_id": str(index % 2),
            }
            for index in range(3)
        ]
        with source_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(source_rows)

        failed = root / "failed.jsonl"
        failed_rows = [
            {
                **source_rows[index],
                "status": "exception",
                "certified": False,
                "failure": "OverflowError: complex exponentiation",
            }
            for index in (1, 2)
        ]
        write_jsonl(failed, failed_rows)
        retry_manifest = root / "retry.csv"
        summary = build_retry_manifest(
            manifest=source_manifest,
            failed_results=failed,
            output=retry_manifest,
            payload=payload,
            maximum_basis=224,
        )
        require(summary["retry_row_count"] == 2, "retry manifest selected the wrong rows")
        preflight = retry_preflight(retry_manifest, payload)
        require(bool(preflight["retry_ready"]), "retry preflight did not pass")

        base_dir = root / "base"
        retry_dir = root / "retry"
        merged_dir = root / "merged"
        base_dir.mkdir()
        retry_dir.mkdir()
        base_results = [
            {
                **source_rows[0],
                "status": "ok",
                "certified": True,
            },
            *failed_rows,
        ]
        write_jsonl(base_dir / "shard-0000.jsonl", [base_results[0], base_results[2]])
        write_jsonl(base_dir / "shard-0001.jsonl", [base_results[1]])
        retry_results = [
            {
                **row,
                "status": "ok",
                "certified": True,
                "failure": "",
                "retry_maximum_collocation_basis": "224",
            }
            for row in failed_rows
        ]
        write_jsonl(retry_dir / "shard-0000.jsonl", [retry_results[1]])
        write_jsonl(retry_dir / "shard-0001.jsonl", [retry_results[0]])
        overlay = overlay_retry_results(
            base_shard_dir=base_dir,
            retry_shard_dir=retry_dir,
            output_dir=merged_dir,
            payload=payload,
        )
        require(bool(overlay["complete"]), "retry overlay is not complete")
        require(overlay["retry_rows_applied"] == 2, "retry overlay applied the wrong count")
        require(overlay["preserved_certified_rows"] == 1, "base successes were not preserved")


def run() -> None:
    check_retry_manifest_and_overlay()
    print("genus2 period-table cluster recovery checks passed")


if __name__ == "__main__":
    run()
