#!/usr/bin/env python3
"""Checks for the paired block-order-10 sphere 1->5 upgrade."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np

try:
    import sphere_six_point_cannon_block10_upgrade as upgrade
except ImportError:  # pragma: no cover
    from plumbing import sphere_six_point_cannon_block10_upgrade as upgrade


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "plumbing"
    / "config"
    / "sphere_six_point_1to5_cannon_block10_upgrade_v2.json"
)
PRIOR_ROOT = (
    ROOT
    / "plumbing"
    / "results"
    / "sphere_six_point_1to5"
    / "cannon_blind30_3h_v2"
)
SOURCE = ROOT / "plumbing" / "sphere_six_point_cannon_block10_upgrade.py"


def check_design_and_exact_reuse() -> None:
    config = upgrade.load_config(CONFIG)
    rows = upgrade.design_rows(config)
    with (PRIOR_ROOT / "manifest.csv").open(newline="") as source:
        prior_rows = list(csv.DictReader(source))
    assert len(rows) == len(prior_rows) == 450
    assert sum(row["task_kind"] == "production_order10" for row in rows) == 420
    assert sum(row["task_kind"] == "systematics_order10" for row in rows) == 30
    for new, old in zip(rows, prior_rows):
        for field in ("task_id", "t_index", "t", "replicate", "sobol_power", "seed"):
            assert new[field] == old[field], (field, new, old)
    assert int(config["production"]["block_order"]) == 10
    assert int(config["production"]["replicates"]) == 14
    assert int(config["production"]["sobol_power"]) == 15
    assert int(config["paired_systematics"]["replicates"]) == 6
    assert int(config["paired_systematics"]["sobol_power"]) == 11
    assert (
        config["paired_systematics"]["execution_mode"]
        == "isolated_process_per_configuration"
    )
    print("450 order-10 tasks exactly match the stored t, seed, and Sobol design")


def check_stored_shard_provenance() -> None:
    shards = PRIOR_ROOT / "shards"
    paths = sorted(shards.glob("task_*.json"))
    assert len(paths) == 450
    for task_id in (0, 13, 419):
        shard = json.loads((shards / f"task_{task_id:04d}.json").read_text())
        assert shard["code_version"] == "sphere_six_point_cannon_blind_v2_source_chart_recovery"
        assert int(shard["settings"]["block_order"]) == 6
        assert shard["target_formula_available"] is False
    for task_id in (420, 449):
        shard = json.loads((shards / f"task_{task_id:04d}.json").read_text())
        assert len(shard["replicate_Q5"]["block_order_plus_two"]) == 6
        assert shard["target_formula_available"] is False
    print("stored order-6 and order-8 shard provenance passed")


def check_blinding() -> None:
    source = SOURCE.read_text()
    config = json.loads(CONFIG.read_text())
    assert "sphere_six_point_matrix_comparison" not in source
    assert "q5_matrix_model" not in source
    assert config["blinding"]["worldsheet_workers_receive_target_formula"] is False
    assert config["blinding"]["comparison_code_staged_with_workers"] is False
    assert config["blinding"]["comparison_allowed_only_after_freeze_manifest"] is True
    print("target-formula exclusion and freeze barrier passed")


def check_isolated_common_point_equivalence() -> None:
    t = 0.2055
    production = {"momentum_power": 1.25, "radial_power": 0.2}
    systematics = {"replicates": 2, "sobol_power": 2, "base_seed": 2026088400}
    configurations = [
        {
            "name": "reference",
            "block_order": 0,
            "momentum_base_order": 1,
            "momentum_maximum": 1.0,
        },
        {
            "name": "momentum_order_plus_two",
            "block_order": 0,
            "momentum_base_order": 2,
            "momentum_maximum": 1.0,
        },
        {
            "name": "momentum_cutoff_plus_one",
            "block_order": 0,
            "momentum_base_order": 2,
            "momentum_maximum": 1.5,
        },
    ]
    kernels = [
        upgrade._build_from_item(t, item, float(production["momentum_power"]))
        for item in configurations
    ]
    batched = {str(item["name"]): [] for item in configurations}
    batched_channels = []
    for replicate in range(int(systematics["replicates"])):
        seed = int(systematics["base_seed"]) + replicate
        points = upgrade.qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(
            int(systematics["sobol_power"])
        )
        values, channel_summary = upgrade.evaluate_common_points(
            kernels, points, radial_power=float(production["radial_power"])
        )
        batched_channels.append(channel_summary)
        for item, value in zip(configurations, values):
            batched[str(item["name"])].append(upgrade._q5_from_i6(value, t))
    sequential = [
        upgrade._evaluate_systematics_configuration(
            t, production, systematics, item
        )
        for item in configurations
    ]
    for item, result in zip(configurations, sequential):
        np.testing.assert_allclose(
            result["replicate_Q5"], batched[str(item["name"])], rtol=0.0, atol=0.0
        )
        assert result["channel_selection"] == batched_channels
    isolated = upgrade._evaluate_configuration_isolated(
        t, production, systematics, configurations[0]
    )
    np.testing.assert_allclose(
        isolated["replicate_Q5"], sequential[0]["replicate_Q5"], rtol=0.0, atol=0.0
    )
    assert isolated["channel_selection"] == sequential[0]["channel_selection"]
    print("isolated kernels preserve exact common-point numerical pairing")


def _channel(sample_count: int) -> dict[str, object]:
    return {
        "sample_count": sample_count,
        "source_chart_recovery_count": 0,
    }


def _old_production(row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    replicate = int(row["replicate"])
    q6 = 0.25 + 0.1 * t + (replicate - 6.5) * 1.0e-5
    old_task = {**row, "task_kind": "production"}
    return {
        "task": old_task,
        "Q5_worldsheet": upgrade.complex_pair(q6),
        "settings": {"block_order": 6},
        "target_formula_available": False,
    }


def _new_production(row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    replicate = int(row["replicate"])
    q10 = 0.2501 + 0.1 * t + (replicate - 6.5) * 1.0e-5
    i6 = -40.0 * math.pi**3 * t**6 * q10
    amplitude = 1.0j * i6 / (8.0 * math.pi**3)
    return {
        "task": row,
        "I6_order10": upgrade.complex_pair(i6),
        "Q5_order10": upgrade.complex_pair(q10),
        "mu4_A_tree_order10": upgrade.complex_pair(amplitude),
        "settings": {"block_order": 10},
        "channel_selection": _channel(32768),
        "block_fallback_counts": {"comb_regulated_h": 0, "star_direct": 0},
        "target_formula_available": False,
    }


def _old_systematics(row: dict[str, str]) -> dict[str, object]:
    old_task = {**row, "task_kind": "systematics"}
    values = [upgrade.complex_pair(0.3 + replicate * 1.0e-3) for replicate in range(6)]
    return {
        "task": old_task,
        "replicate_Q5": {"block_order_plus_two": values},
        "target_formula_available": False,
    }


def _new_systematics(row: dict[str, str]) -> dict[str, object]:
    reference = [0.3001 + replicate * 1.0e-3 for replicate in range(6)]
    momentum = [value + 5.0e-5 for value in reference]
    cutoff = [value + 5.0e-5 for value in momentum]
    return {
        "task": row,
        "replicate_Q5_order10": {
            "reference": [upgrade.complex_pair(value) for value in reference],
            "momentum_order_plus_two": [upgrade.complex_pair(value) for value in momentum],
            "momentum_cutoff_plus_one": [upgrade.complex_pair(value) for value in cutoff],
        },
        "channel_selection": [_channel(2048) for _ in range(6)],
        "block_fallback_counts": {
            name: {"comb_regulated_h": 0, "star_direct": 0}
            for name in (
                "reference",
                "momentum_order_plus_two",
                "momentum_cutoff_plus_one",
            )
        },
        "target_formula_available": False,
    }


def check_synthetic_assembly_and_freeze() -> None:
    with tempfile.TemporaryDirectory(prefix="sphere6_block10_upgrade_check_") as temporary:
        root = Path(temporary)
        design = root / "design"
        new_shards = root / "new_shards"
        prior_shards = root / "prior_shards"
        assembled = root / "assembled"
        summary = upgrade.prepare_design(CONFIG, design)
        assert summary["task_count"] == 450
        rows = upgrade.read_manifest(design / "manifest.csv")
        new_shards.mkdir()
        prior_shards.mkdir()
        for row in rows:
            if row["task_kind"] == "production_order10":
                new_payload = _new_production(row)
                old_payload = _old_production(row)
            else:
                new_payload = _new_systematics(row)
                old_payload = _old_systematics(row)
            path = f"task_{int(row['task_id']):04d}.json"
            upgrade.atomic_write_json(new_shards / path, new_payload)
            upgrade.atomic_write_json(prior_shards / path, old_payload)
        report = upgrade.assemble_upgrade(
            CONFIG,
            design / "manifest.csv",
            new_shards,
            prior_shards,
            assembled,
        )
        assert report["failed_point_count"] == 0
        assert report["maximum_stability_envelope_Q5"] < 5.0e-4
        manifest = upgrade.validate_and_freeze(CONFIG, assembled)
        assert manifest["point_count"] == 30
        assert manifest["all_points_pass_accuracy_gate"] is True
        frozen = json.loads((assembled / "worldsheet_block10_scan_frozen.json").read_text())
        assert frozen["target_formula_available"] is False
        assert frozen["comparison_performed"] is False
    print("synthetic paired reuse, assembly, accuracy gate, and freeze passed")


def main() -> None:
    check_design_and_exact_reuse()
    check_stored_shard_provenance()
    check_blinding()
    check_isolated_common_point_equivalence()
    check_synthetic_assembly_and_freeze()
    print("all block-order-10 sphere 1->5 upgrade checks passed")


if __name__ == "__main__":
    main()
