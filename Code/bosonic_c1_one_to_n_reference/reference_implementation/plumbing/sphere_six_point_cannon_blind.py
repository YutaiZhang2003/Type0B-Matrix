#!/usr/bin/env python3
"""Blind Cannon campaign for the sphere 1->5 worldsheet amplitude.

This module contains no target-space or matrix-model amplitude formula.  It
prepares a declared residue-free design, evaluates production and paired
numerical-systematics shards, assembles the worldsheet result, and creates a
freeze manifest only when every point satisfies the declared accuracy gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc

try:
    from sphere_six_point_atlas import (
        SixPointLinearChannel,
        SixPointStarChannel,
        linear_channel_from_plumbing_coordinates,
        linear_channel_positions_by_label,
        mixed_atlas_log_density_in_frame,
        oriented_comb_orderings,
        star_channel_from_plumbing_coordinates,
        star_channel_positions_by_label,
    )
    from sphere_six_point_equal_energy import (
        EqualEnergySixPointKernel,
        FIRST_RESIDUE_WALL,
        _power_disk_sample,
    )
except ImportError:  # pragma: no cover
    from plumbing.sphere_six_point_atlas import (
        SixPointLinearChannel,
        SixPointStarChannel,
        linear_channel_from_plumbing_coordinates,
        linear_channel_positions_by_label,
        mixed_atlas_log_density_in_frame,
        oriented_comb_orderings,
        star_channel_from_plumbing_coordinates,
        star_channel_positions_by_label,
    )
    from plumbing.sphere_six_point_equal_energy import (
        EqualEnergySixPointKernel,
        FIRST_RESIDUE_WALL,
        _power_disk_sample,
    )


CODE_VERSION = "sphere_six_point_cannon_blind_v2_source_chart_recovery"
MANIFEST_FIELDS = (
    "task_id",
    "task_kind",
    "t_index",
    "t",
    "replicate",
    "sobol_power",
    "seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_wall_time(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"wall time must use HH:MM:SS, received {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid wall time {value!r}")
    return float(3600 * hours + 60 * minutes + seconds)


def complex_pair(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def pair_complex(value: dict[str, float]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    points = [float(value) for value in payload["kinematics"]["t_points"]]
    declared_point_count = int(payload["kinematics"].get("point_count", len(points)))
    if len(points) != declared_point_count or len(set(points)) != declared_point_count:
        raise ValueError(
            "the blind design must contain its declared number of distinct t points"
        )
    if declared_point_count < 4:
        raise ValueError("the blind design must contain at least four t points")
    if points != sorted(points):
        raise ValueError("t points must be strictly ordered")
    if any(not 0.0 < value < FIRST_RESIDUE_WALL for value in points):
        raise ValueError("all t points must lie in the residue-free chamber 0<t<1/3")
    accuracy_target = float(payload["accuracy"]["target"])
    if not math.isfinite(accuracy_target) or not 0.0 < accuracy_target <= 1.0e-2:
        raise ValueError("the absolute Q5 accuracy target must lie in (0, 1e-2]")
    if payload["accuracy"].get("require_every_point") is not True:
        raise ValueError("blind freezing requires every point to pass its accuracy gate")
    production = payload["production"]
    base_order = int(production["momentum_base_order"])
    if list(production["momentum_edge_orders"]) != [
        base_order,
        base_order + 1,
        base_order + 2,
    ]:
        raise ValueError("production momentum edge orders must be adjacent")
    configurations = payload["paired_systematics"]["configurations"]
    expected_names = {
        "reference",
        "block_order_plus_two",
        "momentum_order_plus_two",
        "momentum_cutoff_plus_one",
    }
    if {str(item["name"]) for item in configurations} != expected_names:
        raise ValueError("paired-systematics configuration names changed")
    blinding = payload["blinding"]
    if bool(blinding["worldsheet_workers_receive_target_formula"]):
        raise ValueError("worldsheet workers must not receive the target formula")
    if bool(blinding["comparison_code_staged_with_workers"]):
        raise ValueError("comparison code must not be staged with blind workers")
    cluster = payload["cluster"]
    allocated_seconds = sum(
        parse_wall_time(str(cluster[name]))
        for name in (
            "worker_wall_time",
            "assembly_wall_time",
            "validation_wall_time",
        )
    )
    campaign_seconds = parse_wall_time(str(cluster["campaign_target_wall_time"]))
    if allocated_seconds > campaign_seconds:
        raise ValueError("allocated dependency-chain wall time exceeds campaign target")
    declared_critical_path = cluster.get("allocated_critical_path_wall_time")
    if declared_critical_path is not None and parse_wall_time(
        str(declared_critical_path)
    ) != allocated_seconds:
        raise ValueError("declared allocated critical path does not match job stages")
    task_count = declared_point_count * (int(production["replicates"]) + 1)
    if bool(cluster.get("launch_all_tasks_concurrently")) and int(
        cluster["array_cap"]
    ) < task_count:
        raise ValueError("array cap is too small to expose every blind task concurrently")
    return payload


def design_rows(config: dict[str, object]) -> list[dict[str, str]]:
    points = [float(value) for value in config["kinematics"]["t_points"]]
    production = config["production"]
    systematics = config["paired_systematics"]
    rows: list[dict[str, str]] = []
    task_id = 0
    for t_index, t in enumerate(points):
        for replicate in range(int(production["replicates"])):
            rows.append(
                {
                    "task_id": str(task_id),
                    "task_kind": "production",
                    "t_index": str(t_index),
                    "t": f"{t:.12g}",
                    "replicate": str(replicate),
                    "sobol_power": str(int(production["sobol_power"])),
                    "seed": str(int(production["base_seed"]) + replicate),
                }
            )
            task_id += 1
    for t_index, t in enumerate(points):
        rows.append(
            {
                "task_id": str(task_id),
                "task_kind": "systematics",
                "t_index": str(t_index),
                "t": f"{t:.12g}",
                "replicate": "-1",
                "sobol_power": str(int(systematics["sobol_power"])),
                "seed": str(int(systematics["base_seed"])),
            }
        )
        task_id += 1
    return rows


def prepare_design(config_path: Path, design_dir: Path) -> dict[str, object]:
    config = load_config(config_path)
    rows = design_rows(config)
    design_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = design_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    snapshot_path = design_dir / "config.snapshot.json"
    snapshot_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    production_tasks = sum(row["task_kind"] == "production" for row in rows)
    systematics_tasks = sum(row["task_kind"] == "systematics" for row in rows)
    summary = {
        "campaign": config["campaign"],
        "code_version": CODE_VERSION,
        "status": "blind_worldsheet_design_prepared",
        "t_point_count": len(config["kinematics"]["t_points"]),
        "task_count": len(rows),
        "production_task_count": production_tasks,
        "systematics_task_count": systematics_tasks,
        "samples_per_production_replicate": 2
        ** int(config["production"]["sobol_power"]),
        "production_replicates_per_t": int(config["production"]["replicates"]),
        "total_production_samples_per_t": int(config["production"]["replicates"])
        * 2 ** int(config["production"]["sobol_power"]),
        "paired_systematics_samples_per_t": int(
            config["paired_systematics"]["replicates"]
        )
        * 2 ** int(config["paired_systematics"]["sobol_power"]),
        "manifest_sha256": sha256_file(manifest_path),
        "config_snapshot_sha256": sha256_file(snapshot_path),
        "comparison_formula_present": False,
    }
    atomic_write_json(design_dir / "design_summary.json", summary)
    return summary


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or tuple(rows[0]) != MANIFEST_FIELDS:
        raise ValueError("unexpected manifest schema")
    for expected, row in enumerate(rows):
        if int(row["task_id"]) != expected:
            raise ValueError("manifest task ids must be contiguous from zero")
    return rows


def _sample_geometry(
    point: Sequence[float],
    *,
    radial_power: float,
) -> tuple[
    tuple[complex | None, ...],
    float,
    str,
    SixPointLinearChannel | SixPointStarChannel,
]:
    orderings = oriented_comb_orderings()
    proposal_count = 2 * len(orderings)
    q_values = (
        _power_disk_sample(point[0], point[1], radial_power),
        _power_disk_sample(point[2], point[3], radial_power),
        _power_disk_sample(point[4], point[5], radial_power),
    )
    proposal_index = min(int(point[6] * proposal_count), proposal_count - 1)
    if proposal_index < len(orderings):
        sampled_topology = "comb"
        ordering = orderings[proposal_index]
        positions = linear_channel_positions_by_label(*q_values, ordering)
        sampled_channel = linear_channel_from_plumbing_coordinates(
            *q_values, ordering
        )
        fixed = (ordering[0], ordering[4], ordering[5])
        moving = (ordering[1], ordering[2], ordering[3])
    else:
        sampled_topology = "star"
        ordering = orderings[proposal_index - len(orderings)]
        positions = star_channel_positions_by_label(*q_values, ordering)
        sampled_channel = star_channel_from_plumbing_coordinates(*q_values, ordering)
        fixed = (ordering[0], ordering[2], ordering[5])
        moving = (ordering[1], ordering[3], ordering[4])
    log_density = mixed_atlas_log_density_in_frame(
        positions,
        fixed_zero=fixed[0],
        fixed_one=fixed[1],
        fixed_infinity=fixed[2],
        moving_labels=moving,
        radial_power=radial_power,
    )
    return positions, log_density, sampled_topology, sampled_channel


def evaluate_common_points(
    kernels: Sequence[EqualEnergySixPointKernel],
    points: np.ndarray,
    *,
    radial_power: float,
) -> tuple[list[complex], dict[str, object]]:
    if not kernels:
        raise ValueError("at least one kernel is required")
    totals = np.zeros(len(kernels), dtype=complex)
    comb_count = 0
    star_count = 0
    source_chart_recovery_count = 0
    maximum_score = 0.0
    for point in points:
        positions, log_density, sampled_topology, sampled_channel = _sample_geometry(
            point, radial_power=radial_power
        )
        topology, channel = kernels[0].select_channel(positions)
        selected_q_values = (channel.q1, channel.q2, channel.q3)
        if any(abs(complex(value)) == 0.0 for value in selected_q_values):
            if not sampled_channel.score < 1.0:
                raise RuntimeError(
                    "collapsed channel cannot be recovered in a convergent source chart"
                )
            topology = sampled_topology
            channel = sampled_channel
            source_chart_recovery_count += 1
        for index, kernel in enumerate(kernels):
            value, selected_topology, score = kernel.integrand_in_channel(
                positions,
                topology,
                channel,
                logarithmic_weight=-log_density,
            )
            if selected_topology != topology:
                raise RuntimeError("common-channel evaluation changed topology")
            totals[index] += value
            if index == 0:
                if topology == "comb":
                    comb_count += 1
                else:
                    star_count += 1
                maximum_score = max(maximum_score, score)
    sample_count = len(points)
    return (
        [complex(value / sample_count) for value in totals],
        {
            "sample_count": sample_count,
            "comb_fraction": comb_count / sample_count,
            "star_fraction": star_count / sample_count,
            "source_chart_recovery_count": source_chart_recovery_count,
            "source_chart_recovery_fraction": source_chart_recovery_count
            / sample_count,
            "maximum_selected_radius": maximum_score,
        },
    )


def build_kernel(
    t: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_power: float,
) -> EqualEnergySixPointKernel:
    return EqualEnergySixPointKernel(
        t,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_maximum=momentum_maximum,
        momentum_power=momentum_power,
    )


def _q5_from_i6(value: complex, t: float) -> complex:
    return -complex(value) / (40.0 * math.pi**3 * float(t) ** 6)


def production_shard(
    config: dict[str, object], row: dict[str, str]
) -> dict[str, object]:
    t = float(row["t"])
    production = config["production"]
    build_start = time.perf_counter()
    kernel = build_kernel(
        t,
        block_order=int(production["block_order"]),
        momentum_order=int(production["momentum_base_order"]),
        momentum_maximum=float(production["momentum_maximum"]),
        momentum_power=float(production["momentum_power"]),
    )
    build_seconds = time.perf_counter() - build_start
    sampler = qmc.Sobol(d=7, scramble=True, seed=int(row["seed"]))
    points = sampler.random_base2(int(row["sobol_power"]))
    integration_start = time.perf_counter()
    values, channel_summary = evaluate_common_points(
        [kernel], points, radial_power=float(production["radial_power"])
    )
    integration_seconds = time.perf_counter() - integration_start
    i6 = values[0]
    q5 = _q5_from_i6(i6, t)
    amplitude = 1.0j * i6 / (8.0 * math.pi**3)
    return {
        "status": "blind_worldsheet_production_shard",
        "code_version": CODE_VERSION,
        "task": row,
        "t": t,
        "distance_to_first_residue_wall": FIRST_RESIDUE_WALL - t,
        "I6": complex_pair(i6),
        "Q5_worldsheet": complex_pair(q5),
        "mu4_A_tree_worldsheet": complex_pair(amplitude),
        "channel_selection": channel_summary,
        "block_fallback_counts": dict(kernel.fallback_counts),
        "settings": {
            "block_order": kernel.block_order,
            "momentum_edge_orders": [
                kernel.momentum_order,
                kernel.momentum_order + 1,
                kernel.momentum_order + 2,
            ],
            "momentum_maximum": kernel.momentum_maximum,
            "momentum_power": kernel.momentum_power,
            "radial_power": float(production["radial_power"]),
            "sobol_power": int(row["sobol_power"]),
            "sample_count": len(points),
            "seed": int(row["seed"]),
        },
        "timing_seconds": {
            "kernel_build": build_seconds,
            "moduli_integration": integration_seconds,
            "total": build_seconds + integration_seconds,
        },
        "target_formula_available": False,
    }


def _mean_and_standard_error(values: Sequence[complex]) -> tuple[complex, float, float]:
    array = np.asarray(values, dtype=complex)
    mean = complex(np.mean(array))
    if len(array) < 2:
        return mean, math.inf, math.inf
    return (
        mean,
        float(np.std(array.real, ddof=1) / math.sqrt(len(array))),
        float(np.std(array.imag, ddof=1) / math.sqrt(len(array))),
    )


def systematics_shard(
    config: dict[str, object], row: dict[str, str]
) -> dict[str, object]:
    t = float(row["t"])
    production = config["production"]
    systematics = config["paired_systematics"]
    configurations = list(systematics["configurations"])
    build_start = time.perf_counter()
    kernels = [
        build_kernel(
            t,
            block_order=int(item["block_order"]),
            momentum_order=int(item["momentum_base_order"]),
            momentum_maximum=float(item["momentum_maximum"]),
            momentum_power=float(production["momentum_power"]),
        )
        for item in configurations
    ]
    build_seconds = time.perf_counter() - build_start
    replicate_q5: dict[str, list[complex]] = {
        str(item["name"]): [] for item in configurations
    }
    channel_summaries: list[dict[str, object]] = []
    integration_start = time.perf_counter()
    for replicate in range(int(systematics["replicates"])):
        seed = int(systematics["base_seed"]) + replicate
        sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
        points = sampler.random_base2(int(systematics["sobol_power"]))
        values, channel_summary = evaluate_common_points(
            kernels,
            points,
            radial_power=float(production["radial_power"]),
        )
        channel_summaries.append(channel_summary)
        for item, value in zip(configurations, values):
            replicate_q5[str(item["name"])].append(_q5_from_i6(value, t))
    integration_seconds = time.perf_counter() - integration_start

    differences = {
        "block_order": [
            high - low
            for high, low in zip(
                replicate_q5["block_order_plus_two"],
                replicate_q5["reference"],
            )
        ],
        "momentum_order": [
            high - low
            for high, low in zip(
                replicate_q5["momentum_order_plus_two"],
                replicate_q5["reference"],
            )
        ],
        "momentum_cutoff": [
            high - low
            for high, low in zip(
                replicate_q5["momentum_cutoff_plus_one"],
                replicate_q5["momentum_order_plus_two"],
            )
        ],
    }
    diagnostics: dict[str, object] = {}
    for name, values in differences.items():
        mean, standard_error_real, standard_error_imag = _mean_and_standard_error(values)
        diagnostics[name] = {
            "paired_shift_Q5": complex_pair(mean),
            "paired_standard_error_Q5": {
                "real": standard_error_real,
                "imag": standard_error_imag,
            },
            "two_sigma_absolute_bound_Q5": abs(mean.real)
            + 2.0 * standard_error_real,
            "replicate_differences_Q5": [complex_pair(value) for value in values],
        }
    return {
        "status": "blind_worldsheet_paired_systematics_shard",
        "code_version": CODE_VERSION,
        "task": row,
        "t": t,
        "distance_to_first_residue_wall": FIRST_RESIDUE_WALL - t,
        "settings": {
            "sobol_power": int(systematics["sobol_power"]),
            "samples_per_replicate": 2 ** int(systematics["sobol_power"]),
            "replicates": int(systematics["replicates"]),
            "radial_power": float(production["radial_power"]),
            "configurations": configurations,
        },
        "replicate_Q5": {
            name: [complex_pair(value) for value in values]
            for name, values in replicate_q5.items()
        },
        "diagnostics": diagnostics,
        "channel_selection": channel_summaries,
        "block_fallback_counts": {
            str(item["name"]): dict(kernel.fallback_counts)
            for item, kernel in zip(configurations, kernels)
        },
        "timing_seconds": {
            "kernel_build": build_seconds,
            "moduli_integration": integration_seconds,
            "total": build_seconds + integration_seconds,
        },
        "target_formula_available": False,
    }


def run_worker(
    config_path: Path,
    manifest_path: Path,
    task_id: int,
    shards_dir: Path,
) -> Path:
    config = load_config(config_path)
    rows = read_manifest(manifest_path)
    task_id = int(task_id)
    if not 0 <= task_id < len(rows):
        raise IndexError(f"task id {task_id} outside [0,{len(rows) - 1}]")
    row = rows[task_id]
    if row["task_kind"] == "production":
        payload = production_shard(config, row)
    elif row["task_kind"] == "systematics":
        payload = systematics_shard(config, row)
    else:
        raise ValueError(f"unknown task kind {row['task_kind']!r}")
    output = shards_dir / f"task_{task_id:04d}.json"
    atomic_write_json(output, payload)
    return output


def _load_all_shards(
    manifest_rows: Sequence[dict[str, str]], shards_dir: Path
) -> dict[int, dict[str, object]]:
    loaded: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for row in manifest_rows:
        task_id = int(row["task_id"])
        path = shards_dir / f"task_{task_id:04d}.json"
        if not path.exists():
            missing.append(task_id)
            continue
        payload = json.loads(path.read_text())
        if int(payload["task"]["task_id"]) != task_id:
            raise ValueError(f"shard task mismatch in {path}")
        if bool(payload.get("target_formula_available", True)):
            raise ValueError(f"blindness violation recorded in {path}")
        loaded[task_id] = payload
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} shards: {missing[:20]}")
    return loaded


def assemble_worldsheet(
    config_path: Path,
    manifest_path: Path,
    shards_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    rows = read_manifest(manifest_path)
    shards = _load_all_shards(rows, shards_dir)
    production_by_t: dict[int, list[dict[str, object]]] = defaultdict(list)
    systematics_by_t: dict[int, dict[str, object]] = {}
    for row in rows:
        task_id = int(row["task_id"])
        t_index = int(row["t_index"])
        if row["task_kind"] == "production":
            production_by_t[t_index].append(shards[task_id])
        else:
            if t_index in systematics_by_t:
                raise ValueError(f"duplicate systematics shard for t index {t_index}")
            systematics_by_t[t_index] = shards[task_id]

    target = float(config["accuracy"]["target"])
    leakage_limit = float(config["accuracy"]["maximum_imaginary_leakage"])
    expected_replicates = int(config["production"]["replicates"])
    points: list[dict[str, object]] = []
    failed_indices: list[int] = []
    for t_index, t_source in enumerate(config["kinematics"]["t_points"]):
        t = float(t_source)
        production = sorted(
            production_by_t[t_index], key=lambda item: int(item["task"]["replicate"])
        )
        if len(production) != expected_replicates:
            raise ValueError(f"t index {t_index} has {len(production)} production replicates")
        if t_index not in systematics_by_t:
            raise ValueError(f"t index {t_index} has no systematics shard")
        i6_values = [pair_complex(item["I6"]) for item in production]
        q5_values = [pair_complex(item["Q5_worldsheet"]) for item in production]
        amplitude_values = [
            pair_complex(item["mu4_A_tree_worldsheet"]) for item in production
        ]
        i6_mean, i6_se_real, i6_se_imag = _mean_and_standard_error(i6_values)
        q5_mean, q5_se_real, q5_se_imag = _mean_and_standard_error(q5_values)
        amp_mean, amp_se_real, amp_se_imag = _mean_and_standard_error(amplitude_values)
        systematics = systematics_by_t[t_index]
        bounds = {
            name: float(value["two_sigma_absolute_bound_Q5"])
            for name, value in systematics["diagnostics"].items()
        }
        stability_envelope = max([q5_se_real, *bounds.values()])
        fallback_total = sum(
            int(count)
            for item in production
            for count in item["block_fallback_counts"].values()
        ) + sum(
            int(count)
            for counts in systematics["block_fallback_counts"].values()
            for count in counts.values()
        )
        production_recoveries = sum(
            int(item["channel_selection"]["source_chart_recovery_count"])
            for item in production
        )
        systematics_recoveries = sum(
            int(item["source_chart_recovery_count"])
            for item in systematics["channel_selection"]
        )
        source_chart_recovery_count = (
            production_recoveries + systematics_recoveries
        )
        geometric_sample_count = (
            len(production) * int(production[0]["channel_selection"]["sample_count"])
            + sum(
                int(item["sample_count"])
                for item in systematics["channel_selection"]
            )
        )
        stable = (
            stability_envelope <= target
            and abs(q5_mean.imag) <= leakage_limit
            and fallback_total == 0
            and t < FIRST_RESIDUE_WALL
        )
        if not stable:
            failed_indices.append(t_index)
        points.append(
            {
                "t_index": t_index,
                "t": t,
                "distance_to_first_residue_wall": FIRST_RESIDUE_WALL - t,
                "I6_worldsheet": complex_pair(i6_mean),
                "I6_qmc_standard_error": {
                    "real": i6_se_real,
                    "imag": i6_se_imag,
                },
                "Q5_worldsheet": complex_pair(q5_mean),
                "Q5_qmc_standard_error": {
                    "real": q5_se_real,
                    "imag": q5_se_imag,
                },
                "mu4_A_tree_worldsheet": complex_pair(amp_mean),
                "mu4_A_tree_qmc_standard_error": {
                    "real": amp_se_real,
                    "imag": amp_se_imag,
                },
                "production_replicate_Q5": [
                    complex_pair(value) for value in q5_values
                ],
                "paired_systematics": systematics["diagnostics"],
                "stability_components_Q5": {
                    "production_qmc_standard_error": q5_se_real,
                    **{f"{name}_two_sigma_bound": value for name, value in bounds.items()},
                },
                "stability_envelope_Q5": stability_envelope,
                "accuracy_target_Q5": target,
                "passes_accuracy_gate": stable,
                "fallback_total": fallback_total,
                "source_chart_recovery": {
                    "count": source_chart_recovery_count,
                    "geometric_sample_count": geometric_sample_count,
                    "fraction": source_chart_recovery_count
                    / geometric_sample_count,
                    "meaning": (
                        "exact originating plumbing chart used when floating-point "
                        "channel reconstruction produced q=0"
                    ),
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scan = {
        "status": "worldsheet_only_complete_unvalidated",
        "campaign": config["campaign"],
        "code_version": CODE_VERSION,
        "normalization": {
            "amplitude": "mu^4 A_tree = i I6/(8 pi^3)",
            "stripped": "Q5 = -I6/(40 pi^3 t^6) on omega=i t",
        },
        "kinematic_domain": {
            "omega": "i t",
            "first_residue_wall": FIRST_RESIDUE_WALL,
            "all_points_below_wall": True,
        },
        "accuracy": config["accuracy"],
        "production_settings": config["production"],
        "paired_systematics_settings": config["paired_systematics"],
        "point_count": len(points),
        "points": points,
        "target_formula_available": False,
    }
    scan_path = output_dir / "worldsheet_scan_unfrozen.json"
    atomic_write_json(scan_path, scan)
    report = {
        "status": "blind_accuracy_passed" if not failed_indices else "blind_accuracy_failed",
        "campaign": config["campaign"],
        "point_count": len(points),
        "passed_point_count": len(points) - len(failed_indices),
        "failed_point_count": len(failed_indices),
        "failed_t_indices": failed_indices,
        "failed_t_values": [points[index]["t"] for index in failed_indices],
        "maximum_stability_envelope_Q5": max(
            float(item["stability_envelope_Q5"]) for item in points
        ),
        "accuracy_target_Q5": target,
        "unfrozen_scan_sha256": sha256_file(scan_path),
        "comparison_performed": False,
    }
    atomic_write_json(output_dir / "accuracy_report.json", report)
    return report


def validate_and_freeze(
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    scan_path = output_dir / "worldsheet_scan_unfrozen.json"
    report_path = output_dir / "accuracy_report.json"
    scan = json.loads(scan_path.read_text())
    report = json.loads(report_path.read_text())
    failure_reasons: list[str] = []
    expected_point_count = len(config["kinematics"]["t_points"])
    accuracy_target = float(config["accuracy"]["target"])
    if int(scan.get("point_count", -1)) != expected_point_count:
        failure_reasons.append(
            f"assembled scan does not contain {expected_point_count} points"
        )
    if int(report.get("failed_point_count", -1)) != 0:
        failure_reasons.append(
            f"{report.get('failed_point_count')} points failed the "
            f"{accuracy_target:.6g} gate"
        )
    if bool(scan.get("target_formula_available", True)):
        failure_reasons.append("target formula leaked into the worldsheet scan")
    if not all(bool(item["passes_accuracy_gate"]) for item in scan["points"]):
        failure_reasons.append("one or more point records fail the accuracy gate")
    if failure_reasons:
        blocked = {
            "status": "freeze_blocked",
            "campaign": config["campaign"],
            "reasons": failure_reasons,
            "comparison_allowed": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(output_dir / "freeze_blocked.json", blocked)
        raise RuntimeError("; ".join(failure_reasons))

    frozen = dict(scan)
    frozen["status"] = "worldsheet_only_frozen_before_comparison"
    frozen["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["comparison_performed"] = False
    frozen_path = output_dir / "worldsheet_scan_frozen.json"
    atomic_write_json(frozen_path, frozen)
    manifest = {
        "status": "worldsheet_freeze_valid",
        "campaign": config["campaign"],
        "worldsheet_frozen_file": frozen_path.name,
        "worldsheet_frozen_sha256": sha256_file(frozen_path),
        "config_sha256": sha256_file(config_path),
        "design_manifest_sha256": sha256_file(manifest_path),
        "accuracy_report_sha256": sha256_file(report_path),
        "point_count": expected_point_count,
        "accuracy_target_Q5": accuracy_target,
        "all_points_pass_accuracy_gate": True,
        "all_points_pass_1e_minus_3_gate": accuracy_target == 1.0e-3,
        "comparison_performed": False,
        "comparison_allowed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "worldsheet_freeze_manifest.json", manifest)
    return manifest


def benchmark(
    config_path: Path,
    *,
    t: float,
    sobol_power: int,
    seed: int,
    kind: str,
    safety_factor: float,
) -> dict[str, object]:
    config = load_config(config_path)
    production = config["production"]
    systematics = config["paired_systematics"]

    if kind not in {"production", "systematics", "both"}:
        raise ValueError("benchmark kind must be production, systematics, or both")
    if sobol_power < 2:
        raise ValueError("benchmark Sobol power must be at least two")
    if safety_factor < 1.0:
        raise ValueError("benchmark safety factor must be at least one")

    projections: dict[str, object] = {}
    projected_worker_seconds: list[float] = []
    if kind in {"production", "both"}:
        row = {
            "task_id": "-1",
            "task_kind": "benchmark_production",
            "t_index": "-1",
            "t": f"{float(t):.12g}",
            "replicate": "0",
            "sobol_power": str(int(sobol_power)),
            "seed": str(int(seed)),
        }
        reduced = json.loads(json.dumps(config))
        reduced["production"]["sobol_power"] = int(sobol_power)
        payload = production_shard(reduced, row)
        sample_count = int(payload["settings"]["sample_count"])
        integration_seconds = float(
            payload["timing_seconds"]["moduli_integration"]
        )
        seconds_per_sample = integration_seconds / sample_count
        projected_seconds = (
            float(payload["timing_seconds"]["kernel_build"])
            + seconds_per_sample * 2 ** int(production["sobol_power"])
        )
        projected_worker_seconds.append(projected_seconds)
        projections["production"] = {
            "benchmark_sample_count": sample_count,
            "seconds_per_sample": seconds_per_sample,
            "kernel_build_seconds": payload["timing_seconds"]["kernel_build"],
            "projected_worker_seconds": projected_seconds,
            "projected_worker_hours": projected_seconds / 3600.0,
        }

    if kind in {"systematics", "both"}:
        row = {
            "task_id": "-1",
            "task_kind": "benchmark_systematics",
            "t_index": "-1",
            "t": f"{float(t):.12g}",
            "replicate": "-1",
            "sobol_power": str(int(sobol_power)),
            "seed": str(int(seed)),
        }
        reduced = json.loads(json.dumps(config))
        reduced["paired_systematics"]["sobol_power"] = int(sobol_power)
        reduced["paired_systematics"]["replicates"] = 1
        payload = systematics_shard(reduced, row)
        benchmark_evaluations = 2 ** int(sobol_power)
        integration_seconds = float(
            payload["timing_seconds"]["moduli_integration"]
        )
        seconds_per_common_point = integration_seconds / benchmark_evaluations
        target_evaluations = (
            int(systematics["replicates"])
            * 2 ** int(systematics["sobol_power"])
        )
        projected_seconds = (
            float(payload["timing_seconds"]["kernel_build"])
            + seconds_per_common_point * target_evaluations
        )
        projected_worker_seconds.append(projected_seconds)
        projections["systematics"] = {
            "benchmark_common_point_count": benchmark_evaluations,
            "seconds_per_common_point_four_kernels": seconds_per_common_point,
            "kernel_build_seconds": payload["timing_seconds"]["kernel_build"],
            "projected_common_point_count": target_evaluations,
            "projected_worker_seconds": projected_seconds,
            "projected_worker_hours": projected_seconds / 3600.0,
        }

    slowest_worker_seconds = max(projected_worker_seconds)
    conservative_worker_seconds = safety_factor * slowest_worker_seconds
    cluster = config["cluster"]
    assembly_seconds = parse_wall_time(str(cluster["assembly_wall_time"]))
    validation_seconds = parse_wall_time(str(cluster["validation_wall_time"]))
    campaign_seconds = conservative_worker_seconds + assembly_seconds + validation_seconds
    worker_bound_seconds = parse_wall_time(str(cluster["worker_wall_time"]))
    campaign_bound_seconds = parse_wall_time(str(cluster["campaign_target_wall_time"]))
    return {
        "status": "blind_runtime_benchmark",
        "t": float(t),
        "benchmark_kind": kind,
        "benchmark_sobol_power": int(sobol_power),
        "projections": projections,
        "slowest_projected_worker_seconds": slowest_worker_seconds,
        "slowest_projected_worker_hours": slowest_worker_seconds / 3600.0,
        "safety_factor": float(safety_factor),
        "conservative_worker_seconds": conservative_worker_seconds,
        "conservative_worker_hours": conservative_worker_seconds / 3600.0,
        "post_worker_budget_seconds": assembly_seconds + validation_seconds,
        "conservative_campaign_critical_path_seconds": campaign_seconds,
        "conservative_campaign_critical_path_hours": campaign_seconds / 3600.0,
        "queue_allowance_seconds": campaign_bound_seconds - campaign_seconds,
        "queue_allowance_hours": (campaign_bound_seconds - campaign_seconds) / 3600.0,
        "within_worker_wall_time": conservative_worker_seconds < worker_bound_seconds,
        "within_campaign_wall_time_if_concurrent": campaign_seconds
        < campaign_bound_seconds,
        "target_formula_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--design-dir", type=Path, required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--config", type=Path, required=True)
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--task-id", type=int, required=True)
    worker_parser.add_argument("--shards-dir", type=Path, required=True)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--config", type=Path, required=True)
    assemble_parser.add_argument("--manifest", type=Path, required=True)
    assemble_parser.add_argument("--shards-dir", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--manifest", type=Path, required=True)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--t", type=float, default=0.30)
    benchmark_parser.add_argument("--sobol-power", type=int, default=5)
    benchmark_parser.add_argument("--seed", type=int, default=2026082399)
    benchmark_parser.add_argument(
        "--kind",
        choices=("production", "systematics", "both"),
        default="both",
    )
    benchmark_parser.add_argument("--safety-factor", type=float, default=1.5)
    benchmark_parser.add_argument("--output", type=Path)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        payload = prepare_design(arguments.config, arguments.design_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.command == "worker":
        output = run_worker(
            arguments.config,
            arguments.manifest,
            arguments.task_id,
            arguments.shards_dir,
        )
        print(output)
    elif arguments.command == "assemble":
        payload = assemble_worldsheet(
            arguments.config,
            arguments.manifest,
            arguments.shards_dir,
            arguments.output_dir,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.command == "freeze":
        payload = validate_and_freeze(
            arguments.config,
            arguments.manifest,
            arguments.output_dir,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.command == "benchmark":
        payload = benchmark(
            arguments.config,
            t=arguments.t,
            sobol_power=arguments.sobol_power,
            seed=arguments.seed,
            kind=arguments.kind,
            safety_factor=arguments.safety_factor,
        )
        if arguments.output is not None:
            atomic_write_json(arguments.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
