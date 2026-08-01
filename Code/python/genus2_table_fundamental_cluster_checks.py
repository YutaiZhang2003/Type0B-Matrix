#!/usr/bin/env python3
"""End-to-end check for failed-row retry, override assembly, and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

try:
    from genus2_siegel_fundamental_domain import sample_invariant_domain
    from genus2_table_fundamental_cluster import assemble, validate, worker
except ImportError:  # pragma: no cover
    from plumbing.genus2_siegel_fundamental_domain import sample_invariant_domain
    from plumbing.genus2_table_fundamental_cluster import assemble, validate, worker


def _format(value: complex) -> str:
    return f"{value.real:+.17e}{value.imag:+.17e}j"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    target = sample_invariant_domain(1, seed=20260720).omega[0]
    row = {
        "row_id": "synthetic-retry",
        "shard_id": "0",
        "omega11": _format(complex(target[0, 0])),
        "omega12": _format(complex(target[0, 1])),
        "omega22": _format(complex(target[1, 1])),
        "atlas_target_omega11": _format(complex(target[0, 0])),
        "atlas_target_omega12": _format(complex(target[0, 1])),
        "atlas_target_omega22": _format(complex(target[1, 1])),
        "atlas_marking_word": "I",
        "atlas_marking_matrix_index": "0",
        "atlas_search_depth": "0",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        table = root / "table.csv"
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

        base = root / "base"
        base.mkdir()
        failed = {
            "row_id": row["row_id"],
            "fd_status": "failed",
            "fd_failure": "synthetic depth exhaustion",
        }
        base_result = base / "fundamental-shard-0000.jsonl"
        base_result.write_text(json.dumps(failed, separators=(",", ":")) + "\n")
        (base / "fundamental-shard-0000.summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "task_id": 0,
                    "task_count": 1,
                    "assigned_count": 1,
                    "failed_count": 1,
                    "result_sha256": _sha256(base_result),
                }
            )
            + "\n"
        )
        failure_file = root / "failed.jsonl"
        failure_file.write_text(json.dumps(failed) + "\n")

        retry = root / "retry"
        worker(
            table,
            retry,
            task_id=0,
            task_count=1,
            correction_depths=(2,),
            failed_row_file=failure_file,
            accept_first_certified=True,
        )
        retry_summary = json.loads(
            (retry / "fundamental-shard-0000.summary.json").read_text()
        )
        assert retry_summary["assigned_count"] == 1
        assert retry_summary["failed_count"] == 0

        assembled = root / "assembled"
        assemble(
            table,
            base,
            assembled,
            task_count=1,
            retry_shard_dir=retry,
            retry_task_count=1,
        )
        summary = json.loads((assembled / "fundamental_summary.json").read_text())
        assert summary["row_count"] == 1
        assert summary["retried_row_count"] == 1
        validation = root / "validation"
        run_root = root / "run"
        validate(table, assembled / "fundamental_index.npz", validation, run_root)
        assert (run_root / "RUN_COMPLETE.json").exists()
    print("table fundamental retry-cluster checks passed")


if __name__ == "__main__":
    run()
