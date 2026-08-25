#!/usr/bin/env python3
"""Preparation-only checks for the genus-two period-table pipeline."""

from __future__ import annotations

import cmath
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile

import numpy as np

try:
    from genus2_period_table import (
        Genus2PeriodMapTable,
        PeriodTableEntry,
        leading_omega,
        omega_feature,
        q_feature,
        symmetric_omega,
    )
    from genus2_period_table_cluster import _requires_crosscheck, assemble
    from genus2_period_table_grid import (
        DEFAULT_CONFIG,
        _stable_shard,
        config_sha256,
        describe,
        iter_manifest_rows,
        load_config,
        nominal_point_count,
        plan_backend,
        write_manifest,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_period_table import (
        Genus2PeriodMapTable,
        PeriodTableEntry,
        leading_omega,
        omega_feature,
        q_feature,
        symmetric_omega,
    )
    from plumbing.genus2_period_table_cluster import _requires_crosscheck, assemble
    from plumbing.genus2_period_table_grid import (
        DEFAULT_CONFIG,
        _stable_shard,
        config_sha256,
        describe,
        iter_manifest_rows,
        load_config,
        nominal_point_count,
        plan_backend,
        write_manifest,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_design_and_routing() -> None:
    config = load_config(DEFAULT_CONFIG)
    require(nominal_point_count(config) == 33_568_768, "selector row ceiling changed")
    description = describe(config)
    require(description["note"].startswith("Description only"), "description lost its no-compute guard")
    schottky = plan_backend("theta", (0.1, 0.1, 0.1), 0.1, config)
    require(schottky.backend == "adaptive-schottky", "all-small point was not routed to Schottky")
    ordinary = plan_backend("theta", (0.1, 0.1, 0.16), 0.1, config)
    require(
        ordinary.backend == "holomorphic-form-collocation"
        and ordinary.precision_tier == "binary64-adaptive",
        "ordinary mixed-size point was not routed to holomorphic forms",
    )
    mixed_cusp = plan_backend("theta", (1.0e-12, 0.1, 0.16), 0.1, config)
    require(
        mixed_cusp.backend == "holomorphic-form-collocation"
        and mixed_cusp.precision_tier == "multiprecision-rescaled",
        "one-small-q point was not promoted to high-precision holomorphic forms",
    )
    require(config["array_task_count"] == 960, "Cannon array size changed")
    require(_stable_shard("theta-test", 960) == _stable_shard("theta-test", 960), "sharding is unstable")
    tiny = copy.deepcopy(config)
    tiny["array_task_count"] = 4
    tiny["q_domain"]["q_abs_max"] = 0.8
    tiny["atlas_design"].update(
        {
            "target_sample_power": 3,
            "selector_batch_size": 8,
            "search_depth": 1,
            "tail_search_depth": 1,
            "rescue_search_depth": 1,
            "tail_refine_q_abs_min": 1.0,
            "markings_per_topology": 2,
            "tail_markings_per_topology": 2,
            "rescue_markings_per_topology": 2,
            "minimum_selector_geometry_margin": 1.0e-8,
            "required_selected_target_fraction": 0.0,
            "cusp_augmentation_fraction": 0.0,
        }
    )
    first = next(iter_manifest_rows(tiny))
    require(first["topology"] in {"theta", "glasses"} and 0 <= int(first["shard_id"]) < 4, "tiny manifest row is malformed")
    require("atlas_target_index" in first, "manifest row lost its atlas-selection provenance")
    require("omega11" not in first, "grid preparation unexpectedly evaluated a period matrix")


def check_useful_region_manifest_guard() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["useful_region"]["status"] = "selector-required-before-manifest"
    with tempfile.TemporaryDirectory(prefix="g2-period-design-guard-") as temporary:
        output = Path(temporary) / "design"
        try:
            write_manifest(config, output)
        except RuntimeError as error:
            require("production manifest refused" in str(error), "manifest guard gave an unclear failure")
        else:
            raise AssertionError("broad candidate pool was accepted as a production manifest")
        require(not output.exists(), "manifest guard created output before refusing the design")


def check_crosscheck_policy() -> None:
    config = load_config(DEFAULT_CONFIG)
    row = {
        "planned_backend": "adaptive-schottky",
        "q_min": "0.14",
        "geometry_margin": "0.1",
        "stratum": "method-boundary",
        "row_id": "theta-method-boundary-test",
    }
    require(_requires_crosscheck(row, config), "method boundary lost its mandatory cross-check")
    row["planned_backend"] = "holomorphic-form-collocation"
    require(not _requires_crosscheck(row, config), "Schottky cross-check escaped the all-small policy")


def check_period_adapted_features() -> None:
    log_min = math.log(1.0e-14)
    log_max = math.log(0.3)
    epsilon = 1.0e-9
    left = (0.1 * cmath.exp(1j * (math.pi - epsilon)), 0.08 + 0.01j, 0.05 - 0.01j)
    right = (0.1 * cmath.exp(1j * (-math.pi + epsilon)), left[1], left[2])
    distance = float(
        np.linalg.norm(
            q_feature(left, log_abs_min=log_min, log_abs_max=log_max)
            - q_feature(right, log_abs_min=log_min, log_abs_max=log_max)
        )
    )
    require(distance < 2.0e-9, "phase boundary is discontinuous in q index")
    for topology, q in (
        ("theta", (0.08 + 0.01j, 0.09 - 0.02j, 0.07 + 0.005j)),
        ("glasses", (0.08 + 0.01j, 0.09 - 0.02j, 0.03 + 0.005j)),
    ):
        q_coordinates = q_feature(q, log_abs_min=log_min, log_abs_max=log_max)
        inverse_coordinates = omega_feature(
            topology,
            leading_omega(topology, q),
            log_abs_min=log_min,
            log_abs_max=log_max,
        )
        require(
            float(np.max(np.abs(q_coordinates - inverse_coordinates))) < 1.0e-12,
            f"{topology} leading inverse index is inconsistent",
        )


def check_regularized_local_fit() -> None:
    correction = symmetric_omega(0.003 + 0.002j, -0.001 + 0.0005j, 0.002 - 0.001j)
    entries: list[PeriodTableEntry] = []
    for index in range(16):
        q = tuple(
            (0.055 + 0.002 * ((index + edge) % 5))
            * cmath.exp(1j * (0.02 * index + 0.03 * edge))
            for edge in range(3)
        )
        entries.append(
            PeriodTableEntry(
                row_id=f"synthetic-{index:02d}",
                topology="theta",
                q=q,  # type: ignore[arg-type]
                omega=leading_omega("theta", q) + correction,
                actual_backend="synthetic",
                precision_tier="test",
                error_estimate=0.0,
                geometry_margin=0.1,
                certified=True,
            )
        )
    table = Genus2PeriodMapTable(entries, q_abs_min=1.0e-14, q_abs_max=0.3)
    query = (0.059 * cmath.exp(0.07j), 0.061 * cmath.exp(0.11j), 0.057 * cmath.exp(0.04j))
    result = table.interpolate_omega("theta", query, count=12)
    expected = leading_omega("theta", query) + correction
    require(
        float(np.max(np.abs(result.omega - expected))) < 1.0e-10,
        "regularized local affine fit did not reproduce a constant correction",
    )


def check_cluster_guard_is_present() -> None:
    root = Path(__file__).resolve().parent
    worker = (root / "genus2_period_table_cluster.py").read_text()
    slurm = (root / "cluster" / "genus2_period_table_array.slurm").read_text()
    submit = (root / "cluster" / "stage_submit_genus2_period_table.sh").read_text()
    require("worker requires --execute" in worker, "worker lost its explicit execution guard")
    require("--execute" in slurm, "Slurm template does not acknowledge the execution guard")
    require("--partition=yin" in slurm, "Slurm template lost the dedicated Cannon partition")
    require("--array=0-959%192" in slurm, "Slurm template lost its 192-core cap")
    require("--mem=3G" in slurm, "Slurm template memory no longer permits full-node packing")
    require("PERIOD_TABLE_ARRAY_CAP:-192" in submit, "submission default lost its 192-core cap")
    require("PERIOD_TABLE_TASK_MEMORY:-3G" in submit, "submission default lost its memory policy")
    require("STRINGMC_CLUSTER_PARTITION:-yin" in submit, "submission default lost the yin partition")
    require("STRINGMC_PYTHON" in slurm, "Slurm template does not pin the Python environment")


def check_archive_storage_contract() -> None:
    """Assemble one synthetic certified row without evaluating a period map."""

    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    config["array_task_count"] = 1
    config["cluster"]["maximum_concurrent_tasks"] = 1
    digest = config_sha256(config)
    q = (0.05 + 0.002j, 0.06 - 0.001j, 0.04 + 0.003j)
    omega = leading_omega("theta", q)
    row = {
        "row_id": "synthetic-archive-row",
        "config_sha256": digest,
        "status": "ok",
        "certified": True,
        "topology": "theta",
        "q1": str(q[0]),
        "q2": str(q[1]),
        "q3": str(q[2]),
        "omega11": str(omega[0, 0]),
        "omega12": str(omega[0, 1]),
        "omega22": str(omega[1, 1]),
        "actual_backend": "synthetic",
        "actual_precision_tier": "binary64",
        "precision_tier": "binary64",
        "error_estimate": 0.0,
        "geometry_margin": 0.1,
    }
    with tempfile.TemporaryDirectory(prefix="g2-period-storage-") as temporary:
        root = Path(temporary)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config) + "\n")
        shard_dir = root / "shards"
        shard_dir.mkdir()
        (shard_dir / "shard-0000.jsonl").write_text(json.dumps(row) + "\n")
        output_dir = root / "assembled"
        summary = assemble(
            shard_dir=shard_dir,
            output_dir=output_dir,
            payload=config,
            config_path=config_path,
        )
        required = (
            "table.csv",
            "table.csv.gz",
            "index_features.npz",
            "config.snapshot.json",
            "raw_shard_inventory.json",
            "dataset_manifest.json",
            "SHA256SUMS",
        )
        require(all((output_dir / name).is_file() for name in required), "archive is incomplete")
        with gzip.open(output_dir / "table.csv.gz", "rb") as handle:
            require(handle.read() == (output_dir / "table.csv").read_bytes(), "gzip table differs")
        manifest = json.loads((output_dir / "dataset_manifest.json").read_text())
        require(manifest["row_count"] == 1, "dataset manifest has the wrong row count")
        require(manifest["canonical_table"] == "table.csv.gz", "canonical table changed")
        require(Path(summary["dataset_manifest"]).name == "dataset_manifest.json", "summary lost manifest")
        loaded = Genus2PeriodMapTable.from_portable_index(
            output_dir / "index_features.npz",
            verify_table_path=output_dir / "table.csv",
        )
        require(loaded.entries[0].row_id == row["row_id"], "portable index lost its table row")
        require(
            np.max(np.abs(loaded.entries[0].omega - omega)) < 1.0e-14,
            "portable index changed the period matrix",
        )


def run() -> None:
    check_design_and_routing()
    check_useful_region_manifest_guard()
    check_crosscheck_policy()
    check_period_adapted_features()
    check_regularized_local_fit()
    check_cluster_guard_is_present()
    check_archive_storage_contract()
    print("genus2 period-table preparation checks passed (no period matrices evaluated)")


if __name__ == "__main__":
    run()
