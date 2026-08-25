#!/usr/bin/env python3
"""Checks for the blind 30-point, three-hour sphere 1->5 campaign."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

try:
    import sphere_six_point_cannon_blind as campaign
except ImportError:  # pragma: no cover
    from plumbing import sphere_six_point_cannon_blind as campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "plumbing"
    / "config"
    / "sphere_six_point_1to5_cannon_blind30_3h_v1.json"
)
SOURCE = ROOT / "plumbing" / "sphere_six_point_cannon_blind.py"


def _seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def check_design() -> None:
    config = campaign.load_config(CONFIG)
    points = [float(value) for value in config["kinematics"]["t_points"]]
    rows = campaign.design_rows(config)
    production = [row for row in rows if row["task_kind"] == "production"]
    systematics = [row for row in rows if row["task_kind"] == "systematics"]
    assert len(points) == 30
    assert points[0] == 0.1805 and points[-1] == 0.3255
    assert all(0.0 < t < 1.0 / 3.0 for t in points)
    assert all(abs(t - 0.2) > 1.0e-12 for t in points)
    assert len(rows) == 450
    assert len(production) == 420
    assert len(systematics) == 30
    assert int(config["production"]["replicates"]) == 14
    assert 2 ** int(config["production"]["sobol_power"]) == 32768
    assert 14 * 32768 == 458752
    assert int(config["paired_systematics"]["replicates"]) == 6
    assert 6 * 2 ** int(config["paired_systematics"]["sobol_power"]) == 12288
    assert float(config["accuracy"]["target"]) == 5.0e-4
    cluster = config["cluster"]
    allocated = sum(
        _seconds(str(cluster[name]))
        for name in (
            "worker_wall_time",
            "assembly_wall_time",
            "validation_wall_time",
        )
    )
    assert allocated == _seconds(str(cluster["allocated_critical_path_wall_time"]))
    assert allocated == 10 * 3600 + 50 * 60
    assert allocated < _seconds(str(cluster["campaign_target_wall_time"]))
    assert cluster["queue_time_guaranteed"] is False
    print("30-point, 450-task design and 10h50 allocated critical path passed")


def check_blinding() -> None:
    source = SOURCE.read_text()
    config_text = CONFIG.read_text()
    assert "target_formula_available\": False" in source
    config = json.loads(config_text)
    assert config["blinding"]["worldsheet_workers_receive_target_formula"] is False
    assert config["blinding"]["comparison_code_staged_with_workers"] is False
    assert config["blinding"]["comparison_allowed_only_after_freeze_manifest"] is True
    print("target-formula exclusion and freeze barrier passed")


def _synthetic_production(row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    replicate = int(row["replicate"])
    q5 = 0.25 + 0.1 * t + (replicate - 6.5) * 1.0e-5
    i6 = -40.0 * math.pi**3 * t**6 * q5
    amplitude = 1.0j * i6 / (8.0 * math.pi**3)
    return {
        "status": "blind_worldsheet_production_shard",
        "task": row,
        "I6": campaign.complex_pair(i6),
        "Q5_worldsheet": campaign.complex_pair(q5),
        "mu4_A_tree_worldsheet": campaign.complex_pair(amplitude),
        "channel_selection": {
            "sample_count": 32768,
            "source_chart_recovery_count": 0,
        },
        "block_fallback_counts": {"comb_regulated_h": 0, "star_direct": 0},
        "target_formula_available": False,
    }


def _synthetic_systematics(row: dict[str, str]) -> dict[str, object]:
    diagnostic = {
        "paired_shift_Q5": {"real": 1.0e-4, "imag": 0.0},
        "paired_standard_error_Q5": {"real": 5.0e-5, "imag": 0.0},
        "two_sigma_absolute_bound_Q5": 2.0e-4,
        "replicate_differences_Q5": [
            {"real": 1.0e-4, "imag": 0.0}
        ]
        * 6,
    }
    return {
        "status": "blind_worldsheet_paired_systematics_shard",
        "task": row,
        "diagnostics": {
            "block_order": diagnostic,
            "momentum_order": diagnostic,
            "momentum_cutoff": diagnostic,
        },
        "block_fallback_counts": {
            "reference": {"comb_regulated_h": 0, "star_direct": 0},
            "block_order_plus_two": {"comb_regulated_h": 0, "star_direct": 0},
            "momentum_order_plus_two": {"comb_regulated_h": 0, "star_direct": 0},
            "momentum_cutoff_plus_one": {"comb_regulated_h": 0, "star_direct": 0},
        },
        "channel_selection": [
            {"sample_count": 2048, "source_chart_recovery_count": 0}
        ],
        "target_formula_available": False,
    }


def check_synthetic_freeze() -> None:
    with tempfile.TemporaryDirectory(prefix="sphere6_blind30_check_") as temporary:
        root = Path(temporary)
        design = root / "design"
        shards = root / "shards"
        assembled = root / "assembled"
        summary = campaign.prepare_design(CONFIG, design)
        assert summary["task_count"] == 450
        assert summary["t_point_count"] == 30
        rows = campaign.read_manifest(design / "manifest.csv")
        shards.mkdir()
        for row in rows:
            payload = (
                _synthetic_production(row)
                if row["task_kind"] == "production"
                else _synthetic_systematics(row)
            )
            campaign.atomic_write_json(
                shards / f"task_{int(row['task_id']):04d}.json", payload
            )
        report = campaign.assemble_worldsheet(
            CONFIG, design / "manifest.csv", shards, assembled
        )
        assert report["failed_point_count"] == 0
        manifest = campaign.validate_and_freeze(
            CONFIG, design / "manifest.csv", assembled
        )
        assert manifest["point_count"] == 30
        assert manifest["accuracy_target_Q5"] == 5.0e-4
        assert manifest["all_points_pass_accuracy_gate"] is True
        assert manifest["all_points_pass_1e_minus_3_gate"] is False
        frozen = json.loads((assembled / "worldsheet_scan_frozen.json").read_text())
        assert all(
            point["source_chart_recovery"]["count"] == 0
            for point in frozen["points"]
        )
    print("synthetic 30-point assembly, 5e-4 gate, and freeze passed")


def main() -> None:
    check_design()
    check_blinding()
    check_synthetic_freeze()
    print("all blind 30-point sphere 1->5 campaign checks passed")


if __name__ == "__main__":
    main()
