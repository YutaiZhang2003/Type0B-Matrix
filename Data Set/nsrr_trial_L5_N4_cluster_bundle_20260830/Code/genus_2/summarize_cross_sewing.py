#!/usr/bin/env python3
"""Audit a fresh parity-correct theta/glasses genus-two comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


EXPECTED_SCHEMA = "ns-genus2-cannon-v7-glasses-parity"
SPIN_ZERO = {"alpha": [0, 0], "beta": [0, 0]}


def _rows_by_channel(summary: dict) -> dict[tuple[str, str], dict]:
    result = {}
    for row in summary["rows"]:
        if (
            int(row["recursion_order"]) == 24
            and int(row["quadrature_order"]) == 10
            and math.isclose(float(row["finite_part_radius"]), 0.035)
        ):
            result[(str(row["point_id"]), str(row["channel"]))] = row
    return result


def _crossing_by_point(summary: dict) -> dict[str, dict]:
    return {
        str(row["point_id"]): row
        for row in summary["crossing"]
        if int(row["recursion_order"]) == 24
        and int(row["quadrature_order"]) == 10
        and math.isclose(float(row["finite_part_radius"]), 0.035)
    }


def _certify_spin(summary: dict) -> None:
    config = summary["config"]
    if config.get("physical_lifts") != {
        "theta": [1, -1, 1],
        "glasses": [1, 1, 1],
    }:
        raise ValueError(
            "fresh summary does not use the matched physical Human Note lifts"
        )
    for point in config["points"]:
        point_id = str(point["id"])
        ledger = summary["spin_characteristics"][point_id]
        for channel in ("theta", "glasses"):
            if ledger[channel] != SPIN_ZERO:
                raise ValueError(
                    f"spin mismatch at {point_id}/{channel}: {ledger[channel]!r}"
                )


def summarize_cross_sewing(
    fresh: dict,
    old: dict,
    theta_parity: dict,
) -> dict:
    """Compare the fresh result with both historical sign conventions."""

    if fresh.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"fresh summary has schema {fresh.get('schema')!r}")
    _certify_spin(fresh)
    correction = fresh.get("analytic_checks", {})
    if correction.get("spin_source_characteristic") != SPIN_ZERO:
        raise ValueError("analytic spin-source check is missing or incorrect")
    if correction.get("spin_target_characteristic") != SPIN_ZERO:
        raise ValueError("analytic spin-target check is missing or incorrect")

    fresh_rows = _rows_by_channel(fresh)
    old_rows = _rows_by_channel(old)
    fresh_crossing = _crossing_by_point(fresh)
    old_crossing = _crossing_by_point(old)
    theta_corrections = {
        str(row["point_id"]): row for row in theta_parity["corrected_rows"]
    }
    point_ids = [str(point["id"]) for point in fresh["config"]["points"]]
    if not all(
        point_id in fresh_crossing
        and point_id in old_crossing
        and point_id in theta_corrections
        for point_id in point_ids
    ):
        raise ValueError("the three inputs do not cover the same five points")

    rows = []
    for point_id in point_ids:
        fresh_theta = float(fresh_rows[(point_id, "theta")]["q_l"])
        fresh_glasses = float(fresh_rows[(point_id, "glasses")]["q_l"])
        old_theta = float(old_rows[(point_id, "theta")]["q_l"])
        old_glasses = float(old_rows[(point_id, "glasses")]["q_l"])
        corrected_theta = float(theta_corrections[point_id]["q_l_corrected"])
        archived_corrected_ratio = corrected_theta / old_glasses
        fresh_ratio = fresh_theta / fresh_glasses
        rows.append(
            {
                "point_id": point_id,
                "old_unsigned_theta_over_old_glasses": old_theta / old_glasses,
                "parity_corrected_theta_over_old_glasses": archived_corrected_ratio,
                "fresh_theta_over_fresh_glasses": fresh_ratio,
                "fresh_relative_difference": fresh_ratio - 1.0,
                "fresh_theta_vs_archived_parity_corrected_theta": (
                    fresh_theta / corrected_theta - 1.0
                ),
                "fresh_glasses_vs_old_glasses": fresh_glasses / old_glasses - 1.0,
                "fresh_theta_q_l": fresh_theta,
                "fresh_glasses_q_l": fresh_glasses,
            }
        )

    residuals = [float(row["fresh_relative_difference"]) for row in rows]
    old_residuals = [
        float(row["old_unsigned_theta_over_old_glasses"]) - 1.0 for row in rows
    ]
    parity_only_residuals = [
        float(row["parity_corrected_theta_over_old_glasses"]) - 1.0
        for row in rows
    ]
    return {
        "schema": "ns-genus2-cross-sewing-audit-v1",
        "status": "pass" if max(map(abs, residuals)) < 0.01 else "mismatch",
        "quantity": "Q_L = Z_L / Z_(X+psi)^9",
        "recursion_order": 24,
        "quadrature_order": 10,
        "finite_part_radius": 0.035,
        "spin_structure": {
            "characteristic": SPIN_ZERO,
            "theta_human_note_lifts": [1, 1, 1],
            "glasses_human_note_lifts": [1, 1, 1],
            "transport": "branch-composed affine Sp(4,Z) action",
        },
        "sewing_signs": {
            "theta": "(-1)^(a+p1+p2+p3)",
            "glasses": "(-1)^(a+p_bridge)",
        },
        "glasses_regular_seed": "ordinary vacuum/global product",
        "rows": rows,
        "aggregate": {
            "old_mean_relative_difference": math.fsum(old_residuals) / len(rows),
            "parity_only_mean_relative_difference": (
                math.fsum(parity_only_residuals) / len(rows)
            ),
            "fresh_mean_relative_difference": math.fsum(residuals) / len(rows),
            "fresh_maximum_absolute_relative_difference": max(map(abs, residuals)),
            "fresh_rms_relative_difference": math.sqrt(
                math.fsum(value * value for value in residuals) / len(rows)
            ),
        },
        "fresh_summary_implementation_fingerprint": fresh.get(
            "implementation_fingerprint"
        ),
    }


def _markdown(audit: dict) -> str:
    lines = [
        "# Genus-two theta/glasses cross-sewing audit",
        "",
        "Spin characteristic: `[00|00]` in both transported markings.",
        "",
        "| point | old unsigned | theta sign only | fully corrected | residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit["rows"]:
        lines.append(
            "| {point_id} | {old:.8f} | {parity:.8f} | {fresh:.8f} | {res:+.3e} |".format(
                point_id=row["point_id"],
                old=row["old_unsigned_theta_over_old_glasses"],
                parity=row["parity_corrected_theta_over_old_glasses"],
                fresh=row["fresh_theta_over_fresh_glasses"],
                res=row["fresh_relative_difference"],
            )
        )
    aggregate = audit["aggregate"]
    lines.extend(
        [
            "",
            "Mean fresh residual: `{:+.6e}`.".format(
                aggregate["fresh_mean_relative_difference"]
            ),
            "Maximum absolute fresh residual: `{:.6e}`.".format(
                aggregate["fresh_maximum_absolute_relative_difference"]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--theta-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    audit = summarize_cross_sewing(
        json.loads(args.fresh.read_text(encoding="utf-8")),
        json.loads(args.old_summary.read_text(encoding="utf-8")),
        json.loads(args.theta_parity.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps(audit["aggregate"], indent=2))


if __name__ == "__main__":
    main()
