#!/usr/bin/env python3
"""Prepare and assemble the blind Cannon h-dominant torus-three-point scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from run_genus1_three_point_hdominant_scan import (
        T_VALUES,
        _expected_design,
        _sha256,
        _tag,
        validate_point,
        write_manifest,
    )
except ImportError:  # pragma: no cover
    from plumbing.run_genus1_three_point_hdominant_scan import (
        T_VALUES,
        _expected_design,
        _sha256,
        _tag,
        validate_point,
        write_manifest,
    )


REUSED_T = 0.75


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(design_dir: Path, reused_t075: Path) -> dict[str, object]:
    reused = validate_point(reused_t075, REUSED_T)
    design_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": index,
            "t": f"{t_value:.2f}",
            "tag": _tag(t_value),
        }
        for index, t_value in enumerate(
            value for value in T_VALUES if value != REUSED_T
        )
    ]
    manifest_path = design_dir / "cannon_tasks.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "t", "tag"))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "calculation": "blind Cannon design for h-dominant torus-three-point scan",
        "matrix_model_present": False,
        "task_count": len(rows),
        "task_t_values": [float(row["t"]) for row in rows],
        "reused_t": REUSED_T,
        "reused_t075_path": str(reused_t075),
        "reused_t075_sha256": _sha256(reused_t075),
        "reused_t075_I_1,3": reused["mean"],
        "design": _expected_design(),
    }
    _write_json(design_dir / "design_summary.json", payload)
    return payload


def assemble(
    task_manifest: Path,
    shards_dir: Path,
    reused_t075: Path,
    legacy_scan: Path,
    output_dir: Path,
) -> dict[str, object]:
    validate_point(reused_t075, REUSED_T)
    with task_manifest.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    if len(tasks) != 9:
        raise ValueError("the Cannon assembly requires exactly nine new t tasks")

    sources: dict[float, Path] = {round(REUSED_T, 12): reused_t075}
    for expected_task_id, row in enumerate(tasks):
        if int(row["task_id"]) != expected_task_id:
            raise ValueError("task ids are not contiguous")
        t_value = float(row["t"])
        shard = shards_dir / f"{row['tag']}.json"
        validate_point(shard, t_value)
        sources[round(t_value, 12)] = shard
    if len(sources) != len(T_VALUES):
        raise ValueError("the assembled scan does not contain ten unique t values")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(
        output_dir,
        sources,
        reused_t075=reused_t075,
        legacy_scan=legacy_scan,
        complete=True,
    )
    csv_path = output_dir / "worldsheet_t_dependence.csv"
    artifacts = [manifest, csv_path, reused_t075]
    artifacts.extend(
        shards_dir / f"{row['tag']}.json"
        for row in tasks
    )
    freeze = {
        "status": "blind_worldsheet_scan_frozen",
        "matrix_model_present": False,
        "comparison_allowed_only_after_this_freeze": True,
        "point_count": len(T_VALUES),
        "design": _expected_design(),
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifacts
        ],
    }
    freeze_path = output_dir / "worldsheet_freeze_manifest.json"
    _write_json(freeze_path, freeze)
    complete = {
        "status": "complete",
        "worldsheet_scan_manifest": str(manifest),
        "worldsheet_freeze_manifest": str(freeze_path),
        "worldsheet_workers_received_target_formula": False,
        "comparison_performed": False,
    }
    _write_json(output_dir / "RUN_COMPLETE.json", complete)
    return complete


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--design-dir", type=Path, required=True)
    prepare_parser.add_argument("--reused-t075", type=Path, required=True)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--task-manifest", type=Path, required=True)
    assemble_parser.add_argument("--shards-dir", type=Path, required=True)
    assemble_parser.add_argument("--reused-t075", type=Path, required=True)
    assemble_parser.add_argument("--legacy-scan", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        payload = prepare(args.design_dir, args.reused_t075)
        print(f"prepared {payload['task_count']} Cannon tasks")
        return
    result = assemble(
        args.task_manifest,
        args.shards_dir,
        args.reused_t075,
        args.legacy_scan,
        args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
