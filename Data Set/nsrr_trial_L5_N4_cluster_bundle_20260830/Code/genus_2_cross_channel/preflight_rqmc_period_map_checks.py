#!/usr/bin/env python3
"""Checks for tier-balanced RQMC period-map preflight selection."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from preflight_rqmc_period_map import (
        _checkpoint_settings,
        select_stress_rows,
        select_tail_stress_rows,
        select_tier_stress_rows,
    )
except ImportError:  # pragma: no cover
    from plumbing.preflight_rqmc_period_map import (
        _checkpoint_settings,
        select_stress_rows,
        select_tail_stress_rows,
        select_tier_stress_rows,
    )


def run_checks() -> None:
    rows = []
    for replicate in range(3):
        for tier in ("reference", "moderate", "hard"):
            for index, q_value in enumerate((0.1, 0.2)):
                rows.append(
                    {
                        "rqmc_replicate": str(replicate),
                        "plumbing_difficulty_tier": tier,
                        "best_leading_q_max": str(q_value + 0.01 * replicate),
                        "rqmc_t3_tail_level": str(index + replicate),
                        "rqmc_t3": str(index + 0.1 * replicate),
                        "rqmc_node_id": f"{replicate}-{tier}-{index}",
                    }
                )
    selected = select_tier_stress_rows(rows)
    if len(selected) != 9:
        raise AssertionError("preflight did not select one node per tier and scramble")
    if any(float(row["best_leading_q_max"]) < 0.2 for row in selected):
        raise AssertionError("preflight did not select the hardest node in a tier")
    tail = select_tail_stress_rows(rows)
    if len(tail) != 3 or any(not row["rqmc_node_id"].endswith("-1") for row in tail):
        raise AssertionError("preflight did not select one deepest tail node per scramble")
    combined = select_stress_rows(rows, mode="tiers-and-tail")
    if len({row["rqmc_node_id"] for row in combined}) != len(combined):
        raise AssertionError("combined stress selection contains duplicate nodes")
    settings = _checkpoint_settings(
        argparse.Namespace(
            input_csv=Path("input.csv"),
            out_dir=Path("output"),
            resume=True,
            validation_tolerance=1.0e-6,
            node_timeout_seconds=90.0,
        )
    )
    if settings != {"validation_tolerance": 1.0e-6, "node_timeout_seconds": 90.0}:
        raise AssertionError("checkpoint signature omits or includes the wrong settings")
    print("preflight_rqmc_period_map checks passed")


if __name__ == "__main__":
    run_checks()
