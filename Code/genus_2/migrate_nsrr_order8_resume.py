#!/usr/bin/env python3
"""Build an audited continuation of the historical order-eight theta run.

The original run used a Ward-residual acceptance threshold of 1e-7.  One
binary64 node is finite and reproducible but lies slightly above that gate.
This tool creates a NEW dataset whose only configuration change is a relaxed
acceptance threshold.  It validates every inherited shard against the frozen
original config and implementation before copying it; the original dataset is
never edited.

The exceptional binary64 node is compared with an independently evaluated
multiprecision probe.  The multiprecision value is always selected for that
node; disagreement above the requested tolerance is recorded as evidence that
the binary64 result was unstable rather than silently weakening the check.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import nsrr_nsnsns_theta_cannon as historical


REPORT_SCHEMA = "nsrr-nsnsns-order8-resume-migration-v1"
SCIENTIFIC_SCOPE = (
    "Historical diagnostic only. The nonchiral NSRR assembly and source "
    "spin/free-factor adapter used by the 2026-08-29 snapshot were later "
    "retired; this continuation cannot certify a physical NSRR/NSNSNS "
    "partition-function match."
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_gate_only_change(original: dict, continuation: dict) -> tuple[float, float]:
    old_gate = float(original["numerics"]["maximum_ward_residual"])
    new_gate = float(continuation["numerics"]["maximum_ward_residual"])
    left = deepcopy(original)
    right = deepcopy(continuation)
    left["numerics"].pop("maximum_ward_residual")
    right["numerics"].pop("maximum_ward_residual")
    if left != right:
        raise RuntimeError(
            "continuation config differs from the original in fields other "
            "than numerics.maximum_ward_residual"
        )
    if not 0.0 < old_gate < new_gate:
        raise RuntimeError(
            f"continuation Ward gate must be a relaxation: {old_gate=} {new_gate=}"
        )
    return old_gate, new_gate


def relative_vector_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("outlier probes have different sector counts")
    return max(
        abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0e-300)
        for a, b in zip(left, right)
    )


def migrate(
    *,
    original_config_path: Path,
    continuation_config_path: Path,
    original_shard_dir: Path,
    continuation_shard_dir: Path,
    outlier_task: int,
    outlier_shard_path: Path,
    check_config_path: Path,
    check_shard_path: Path,
    report_path: Path,
    agreement_tolerance: float,
    summary_path: Path | None,
) -> dict:
    original = load_json(original_config_path)
    continuation = load_json(continuation_config_path)
    check_config = load_json(check_config_path)
    historical._validate_config(original)
    historical._validate_config(continuation)
    historical._validate_config(check_config)
    old_gate, new_gate = require_gate_only_change(original, continuation)

    if int(check_config["numerics"]["branching_mp_dps"]) <= 0:
        raise RuntimeError("the independent outlier check is not multiprecision")
    check_without_precision = deepcopy(check_config)
    original_without_precision = deepcopy(original)
    check_without_precision["numerics"].pop("branching_mp_dps")
    original_without_precision["numerics"].pop("branching_mp_dps")
    if check_without_precision != original_without_precision:
        raise RuntimeError(
            "multiprecision check config differs from the original in fields "
            "other than numerics.branching_mp_dps"
        )

    implementation = historical._implementation_fingerprint()
    old_digest = historical._digest(original)
    new_digest = historical._digest(continuation)
    check_digest = historical._digest(check_config)
    count = historical.task_count(original)
    if not 0 <= outlier_task < count:
        raise RuntimeError(f"outlier task {outlier_task} lies outside 0..{count - 1}")

    outlier = load_json(outlier_shard_path)
    check = load_json(check_shard_path)
    historical._validate_shard(continuation, outlier_task, outlier, implementation)
    historical._validate_shard(check_config, outlier_task, check, implementation)
    outlier_ward = float(outlier["maximum_ward_residual"])
    check_ward = float(check["maximum_ward_residual"])
    if not old_gate < outlier_ward <= new_gate:
        raise RuntimeError(
            f"outlier residual must lie between the two gates: {outlier_ward=}"
        )
    if check_ward > old_gate:
        raise RuntimeError(
            f"multiprecision outlier residual {check_ward:.3e} exceeds {old_gate:.3e}"
        )
    sector_difference = relative_vector_difference(
        outlier["sector_contributions"], check["sector_contributions"]
    )
    precision_stable = sector_difference <= agreement_tolerance

    continuation_shard_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(continuation_shard_dir.glob("task-*.json"))
    if unexpected:
        raise RuntimeError(
            f"continuation shard directory is not empty: {unexpected[0]}"
        )

    source_count = 0
    target_count = 0
    maximum_inherited_ward = 0.0
    for task_index in range(count):
        if task_index == outlier_task:
            payload = deepcopy(check)
            origin = "multiprecision_replacement_for_unstable_binary64_node"
            source_digest = check_digest
        else:
            source_path = original_shard_dir / f"task-{task_index:06d}.json"
            if not source_path.is_file():
                raise RuntimeError(f"missing original shard {source_path}")
            payload = load_json(source_path)
            historical._validate_shard(original, task_index, payload, implementation)
            ward = payload.get("maximum_ward_residual")
            if ward is not None:
                maximum_inherited_ward = max(maximum_inherited_ward, float(ward))
                if float(ward) > old_gate:
                    raise RuntimeError(
                        f"inherited shard {task_index} exceeds original Ward gate"
                    )
            origin = "validated_original_shard"
            source_digest = old_digest

        if payload["channel"] == "source_nsrr":
            source_count += 1
        elif payload["channel"] == "target_nsnsns":
            target_count += 1
        else:
            raise RuntimeError(f"unexpected channel in task {task_index}")

        payload["config_digest"] = new_digest
        payload["continuation_provenance"] = {
            "schema": REPORT_SCHEMA,
            "origin": origin,
            "source_config_digest": source_digest,
            "continuation_config_digest": new_digest,
        }
        write_json(
            continuation_shard_dir / f"task-{task_index:06d}.json", payload
        )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "validated_and_migrated_with_multiprecision_replacement",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": SCIENTIFIC_SCOPE,
        "original_config": str(original_config_path),
        "continuation_config": str(continuation_config_path),
        "original_config_digest": old_digest,
        "continuation_config_digest": new_digest,
        "multiprecision_config_digest": check_digest,
        "implementation_fingerprint": implementation,
        "task_count": count,
        "source_shards": source_count,
        "target_shards": target_count,
        "original_ward_gate": old_gate,
        "continuation_ward_gate": new_gate,
        "maximum_inherited_ward_residual": maximum_inherited_ward,
        "outlier_task": outlier_task,
        "outlier_binary64_ward_residual": outlier_ward,
        "outlier_multiprecision_ward_residual": check_ward,
        "outlier_sector_max_relative_difference": sector_difference,
        "outlier_agreement_tolerance": agreement_tolerance,
        "outlier_binary64_precision_stable": precision_stable,
        "outlier_selected_shard": str(check_shard_path),
        "outlier_selection_reason": (
            "The multiprecision shard satisfies the original Ward gate and is "
            "used as the canonical replacement. The binary64 comparison is "
            "retained only as an instability diagnostic."
        ),
    }
    write_json(report_path, report)

    if summary_path is not None:
        summary = historical.reduce(
            continuation_config_path, continuation_shard_dir, summary_path
        )
        summary["status"] = "completed_historical_diagnostic_only"
        summary["scientific_scope"] = SCIENTIFIC_SCOPE
        summary["continuation_provenance"] = report
        write_json(summary_path, summary)
        report["summary"] = str(summary_path)
        report["comparisons"] = summary["comparisons"]
        write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--continuation-config", type=Path, required=True)
    parser.add_argument("--original-shard-dir", type=Path, required=True)
    parser.add_argument("--continuation-shard-dir", type=Path, required=True)
    parser.add_argument("--outlier-task", type=int, default=4608)
    parser.add_argument("--outlier-shard", type=Path, required=True)
    parser.add_argument("--check-config", type=Path, required=True)
    parser.add_argument("--check-shard", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--agreement-tolerance", type=float, default=1.0e-8)
    args = parser.parse_args(argv)
    if not math.isfinite(args.agreement_tolerance) or args.agreement_tolerance <= 0:
        parser.error("--agreement-tolerance must be finite and positive")
    report = migrate(
        original_config_path=args.original_config,
        continuation_config_path=args.continuation_config,
        original_shard_dir=args.original_shard_dir,
        continuation_shard_dir=args.continuation_shard_dir,
        outlier_task=args.outlier_task,
        outlier_shard_path=args.outlier_shard,
        check_config_path=args.check_config,
        check_shard_path=args.check_shard,
        report_path=args.report,
        agreement_tolerance=args.agreement_tolerance,
        summary_path=args.summary,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
