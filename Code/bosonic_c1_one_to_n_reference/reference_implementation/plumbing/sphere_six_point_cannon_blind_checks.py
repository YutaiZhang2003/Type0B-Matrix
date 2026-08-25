#!/usr/bin/env python3
"""Checks for the blind sphere 1->5 Cannon campaign."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

try:
    import sphere_six_point_cannon_blind as campaign
except ImportError:  # pragma: no cover
    from plumbing import sphere_six_point_cannon_blind as campaign


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "plumbing" / "config" / "sphere_six_point_1to5_cannon_blind50_v1.json"
SOURCE = ROOT / "plumbing" / "sphere_six_point_cannon_blind.py"


def check_design() -> None:
    config = campaign.load_config(CONFIG)
    points = [float(value) for value in config["kinematics"]["t_points"]]
    rows = campaign.design_rows(config)
    production = [row for row in rows if row["task_kind"] == "production"]
    systematics = [row for row in rows if row["task_kind"] == "systematics"]
    assert len(points) == 50
    assert points[0] == 0.18 and points[-1] == 0.325
    assert all(0.0 < t < 1.0 / 3.0 for t in points)
    assert all(abs(t - 0.2) > 1.0e-12 for t in points)
    assert len(rows) == 450
    assert len(production) == 400
    assert len(systematics) == 50
    assert int(config["production"]["replicates"]) == 8
    assert 2 ** int(config["production"]["sobol_power"]) == 32768
    assert 8 * 32768 == 262144
    print("50-point residue-free design and 450-task manifest passed")


def check_blinding() -> None:
    source = SOURCE.read_text()
    config_text = CONFIG.read_text()
    assert "target_formula_available\": False" in source
    config = json.loads(config_text)
    assert config["blinding"]["worldsheet_workers_receive_target_formula"] is False
    assert config["blinding"]["comparison_code_staged_with_workers"] is False
    assert config["blinding"]["comparison_allowed_only_after_freeze_manifest"] is True
    print("static target-formula exclusion and freeze barrier passed")


def check_geometry_sampling() -> None:
    rng = np.random.default_rng(20260823)
    for point in rng.random((24, 7)):
        positions, log_density, topology, channel = campaign._sample_geometry(
            point, radial_power=0.2
        )
        assert len(positions) == 6
        assert sum(value is None for value in positions) == 1
        assert math.isfinite(log_density)
        assert topology in ("comb", "star")
        assert all(
            abs(value) > 0.0 for value in (channel.q1, channel.q2, channel.q3)
        )
        assert channel.score < 1.0
    print("mixed-atlas sampling smoke passed")


def check_collapsed_channel_recovery() -> None:
    class CollapsedSelectionKernel:
        def select_channel(self, positions):
            del positions
            collapsed = type(
                "CollapsedChannel",
                (),
                {"q1": 0.0j, "q2": 0.1 + 0.0j, "q3": 0.2 + 0.0j},
            )()
            return "comb", collapsed

        def integrand_in_channel(
            self, positions, topology, channel, *, logarithmic_weight
        ):
            del positions, logarithmic_weight
            assert topology == "comb"
            assert isinstance(channel, campaign.SixPointLinearChannel)
            assert all(
                abs(value) > 0.0
                for value in (channel.q1, channel.q2, channel.q3)
            )
            return 2.0 + 3.0j, topology, channel.score

    point = np.asarray([[0.5, 0.1, 0.6, 0.2, 0.7, 0.3, 0.0]])
    values, summary = campaign.evaluate_common_points(
        [CollapsedSelectionKernel()], point, radial_power=0.2
    )
    assert values == [2.0 + 3.0j]
    assert summary["sample_count"] == 1
    assert summary["source_chart_recovery_count"] == 1
    assert summary["source_chart_recovery_fraction"] == 1.0
    print("collapsed reselected channel recovers in the exact source chart")


def _synthetic_production(row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    replicate = int(row["replicate"])
    q5 = 0.25 + 0.1 * t + (replicate - 3.5) * 1.0e-5
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
        * 4,
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
            {"sample_count": 1024, "source_chart_recovery_count": 0}
        ],
        "target_formula_available": False,
    }


def check_synthetic_assembly_and_freeze() -> None:
    with tempfile.TemporaryDirectory(prefix="sphere6_blind_check_") as temporary:
        root = Path(temporary)
        design = root / "design"
        shards = root / "shards"
        assembled = root / "assembled"
        summary = campaign.prepare_design(CONFIG, design)
        assert summary["task_count"] == 450
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
        assert manifest["point_count"] == 50
        assert manifest["all_points_pass_1e_minus_3_gate"] is True
        frozen = json.loads((assembled / "worldsheet_scan_frozen.json").read_text())
        assert frozen["comparison_performed"] is False
        assert frozen["target_formula_available"] is False
        assert all(
            point["source_chart_recovery"]["count"] == 0
            for point in frozen["points"]
        )
    print("synthetic 50-point assembly, accuracy gate, and freeze passed")


def main() -> None:
    check_design()
    check_blinding()
    check_geometry_sampling()
    check_collapsed_channel_recovery()
    check_synthetic_assembly_and_freeze()
    print("all blind sphere 1->5 Cannon checks passed")


if __name__ == "__main__":
    main()
