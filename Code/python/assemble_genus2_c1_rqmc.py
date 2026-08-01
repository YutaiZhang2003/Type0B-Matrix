#!/usr/bin/env python3
"""Assemble complete scrambled-Sobol genus-two CFT replicates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from monte_carlo_integrate_genus2_c1 import (  # noqa: E402
    RQMC_SAMPLING_SCHEMES,
    canonicalize_string_note_kernel_row,
    sampling_scheme_for_rows,
    summarize_rqmc_rows,
)
from genus2_integrand_normalization import (  # noqa: E402
    STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    c1_sphere_normalized_genus2_kernel_multiplier,
)


DEFAULT_DESIGN = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_design_R8_M64/production_nodes.csv"
)

PERIOD_COORDINATE_KEYS = ("x11", "x12", "x22", "y11", "y12", "y22")
PERIOD_COORDINATE_ABSOLUTE_TOLERANCE = 5.0e-14


def _node_id(row: dict[str, object]) -> str:
    value = str(row.get("rqmc_node_id", ""))
    if not value:
        raise ValueError("RQMC row lacks rqmc_node_id")
    return value


def index_evaluations(
    rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, object]]:
    """Index unique CFT evaluations by stable nested-design node id."""

    indexed: dict[str, dict[str, object]] = {}
    for source_row in rows:
        row = canonicalize_string_note_kernel_row(source_row)
        node_id = _node_id(row)
        if node_id in indexed:
            old = indexed[node_id]
            if old.get("status") != row.get("status"):
                if row.get("status") == "ok":
                    indexed[node_id] = row
                elif old.get("status") != "ok":
                    raise ValueError(f"duplicate node {node_id} has inconsistent status")
                continue
            if row.get("status") == "ok":
                for key in ("transformed_integrand_low", "transformed_integrand_high"):
                    if not math.isclose(
                        float(old[key]),
                        float(row[key]),
                        rel_tol=2.0e-12,
                        abs_tol=0.0,
                    ):
                        raise ValueError(f"duplicate node {node_id} disagrees in {key}")
            continue
        indexed[node_id] = row
    return indexed


def verify_evaluation_coordinates(
    design: dict[str, object],
    evaluation: dict[str, object],
    *,
    absolute_tolerance: float = PERIOD_COORDINATE_ABSOLUTE_TOLERANCE,
) -> None:
    """Reject a node-id match whose saved period-matrix coordinates differ."""

    node_id = _node_id(design)
    for key in PERIOD_COORDINATE_KEYS:
        if key not in design or key not in evaluation:
            raise ValueError(f"node {node_id} lacks coordinate {key}")
        design_value = float(design[key])
        evaluation_value = float(evaluation[key])
        if not math.isclose(
            design_value,
            evaluation_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"node {node_id} evaluation disagrees in {key}: "
                f"design={design_value:.17g}, evaluation={evaluation_value:.17g}"
            )


def assemble_rows(
    design_rows: Sequence[dict[str, str]],
    evaluation_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Merge current design weights with reusable CFT values."""

    design_ids = [_node_id(row) for row in design_rows]
    if len(design_ids) != len(set(design_ids)):
        raise ValueError("design node ids are not unique")
    scheme = sampling_scheme_for_rows(design_rows)
    if scheme not in RQMC_SAMPLING_SCHEMES:
        raise ValueError("design mixes or omits the RQMC sampling scheme")
    evaluated = index_evaluations(evaluation_rows)
    unknown = sorted(set(evaluated) - set(design_ids))

    combined: list[dict[str, object]] = []
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    reused_ids: list[str] = []
    protected = {
        "sample_index",
        "sampling_scheme",
        "x11",
        "x12",
        "x22",
        "y11",
        "y12",
        "y22",
    }
    for design in design_rows:
        node_id = _node_id(design)
        merged: dict[str, object] = dict(design)
        evaluation = evaluated.get(node_id)
        if evaluation is None:
            merged.update({"status": "missing", "error": "CFT evaluation is absent"})
            missing_ids.append(node_id)
        else:
            verify_evaluation_coordinates(design, evaluation)
            for key, value in evaluation.items():
                if key.startswith("rqmc_") or key in protected:
                    continue
                merged[key] = value
            if evaluation.get("status") != "ok":
                failed_ids.append(node_id)
            elif int(evaluation.get("rqmc_power", design["rqmc_power"])) < int(
                design["rqmc_power"]
            ):
                reused_ids.append(node_id)
            if evaluation.get("status") == "ok":
                weight = float(merged["rqmc_stack_integration_weight"])
                for order in ("low", "high"):
                    transformed = float(merged[f"transformed_integrand_{order}"])
                    merged[f"node_contribution_{order}"] = weight * transformed
        combined.append(merged)

    complete = not missing_ids and not failed_ids
    high = (
        summarize_rqmc_rows(combined, value_key="transformed_integrand_high")
        if complete
        else None
    )
    low = (
        summarize_rqmc_rows(combined, value_key="transformed_integrand_low")
        if complete
        else None
    )
    diagnostics = {
        "design_node_count": len(design_rows),
        "provided_unique_evaluation_count": len(evaluated),
        "unknown_evaluation_count": len(unknown),
        "unknown_evaluation_ids": unknown,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "nested_reused_node_count": len(reused_ids),
        "nested_reused_node_ids": reused_ids,
        "complete": complete,
        "headline_available": high is not None and low is not None,
        "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        "string_note_kernel_multiplier": (
            c1_sphere_normalized_genus2_kernel_multiplier()
        ),
        "high_order_summary": high,
        "low_order_summary": low,
        "aggregate_low_high_relative_change": (
            None
            if high is None or low is None
            else float(high["estimate"]) / float(low["estimate"]) - 1.0
        ),
    }
    return combined, diagnostics


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble complete genus-two RQMC CFT data.")
    parser.add_argument("--design-csv", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--evaluation-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    design_rows = list(csv.DictReader(args.design_csv.open()))
    evaluation_rows = [
        row
        for path in args.evaluation_csv
        for row in csv.DictReader(path.open())
    ]
    combined, diagnostics = assemble_rows(design_rows, evaluation_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "combined_samples.csv"
    json_path = args.out_dir / "summary.json"
    _write_csv(csv_path, combined)
    json_path.write_text(
        json.dumps(
            {
                "scope": (
                    "Strict assembly of complete independent scrambled-Sobol replicates; "
                    "current design weights override stale weights from nested lower powers."
                ),
                "design_csv": str(args.design_csv),
                "evaluation_csv": [str(path) for path in args.evaluation_csv],
                "radius": args.radius,
                "external_comparison_target": None,
                **diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    print("Genus-two c=1 RQMC assembly")
    print(
        f"  complete={diagnostics['complete']}, headline={diagnostics['headline_available']}, "
        f"missing={diagnostics['missing_count']}, failed={diagnostics['failed_count']}"
    )
    high = diagnostics["high_order_summary"]
    if high is not None:
        print(
            f"  F2/g_s^2={float(high['estimate']):.12g} +/- "
            f"{float(high['standard_error']):.3g} across "
            f"{int(high['replicate_count'])} scrambles"
        )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    run()
