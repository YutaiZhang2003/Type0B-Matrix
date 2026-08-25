#!/usr/bin/env python3
"""Blind block-order-10 upgrade for the sphere 1->5 Cannon campaign.

The upgrade preserves the completed order-6 campaign and reuses its exact
Sobol seeds and stored shard values.  New production workers evaluate only the
order-10 kernel.  New paired-systematics workers evaluate order 10 at the
reference, raised-momentum, and raised-cutoff settings.  Assembly then forms

* a full-sample paired order-10 minus order-6 diagnostic,
* an order-10 minus stored order-8 truncation diagnostic on common points,
* order-10 momentum-order and cutoff diagnostics.

No target-space or matrix-model amplitude formula is present in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import qmc

try:
    from sphere_six_point_cannon_blind import (
        _q5_from_i6,
        atomic_write_json,
        build_kernel,
        complex_pair,
        evaluate_common_points,
    )
    from sphere_six_point_equal_energy import FIRST_RESIDUE_WALL
except ImportError:  # pragma: no cover
    from plumbing.sphere_six_point_cannon_blind import (
        _q5_from_i6,
        atomic_write_json,
        build_kernel,
        complex_pair,
        evaluate_common_points,
    )
    from plumbing.sphere_six_point_equal_energy import FIRST_RESIDUE_WALL


CODE_VERSION = "sphere_six_point_block10_upgrade_v2_isolated_systematics"
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


def pair_complex(value: dict[str, object]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    points = [float(value) for value in config["kinematics"]["t_points"]]
    if len(points) != 30 or any(not 0.0 < t < FIRST_RESIDUE_WALL for t in points):
        raise ValueError("the block-10 upgrade requires 30 points in 0<t<1/3")
    production = config["production"]
    if int(production["block_order"]) != 10:
        raise ValueError("production block order must equal 10")
    if int(production["replicates"]) != 14 or int(production["sobol_power"]) != 15:
        raise ValueError("production must preserve the 14 by 2^15 design")
    names = [str(item["name"]) for item in config["paired_systematics"]["configurations"]]
    if names != ["reference", "momentum_order_plus_two", "momentum_cutoff_plus_one"]:
        raise ValueError("unexpected block-10 systematics configurations")
    if any(int(item["block_order"]) != 10 for item in config["paired_systematics"]["configurations"]):
        raise ValueError("all new systematics kernels must use block order 10")
    if config["paired_systematics"].get("execution_mode") != "isolated_process_per_configuration":
        raise ValueError("block-10 systematics must isolate each kernel in a fresh process")
    return config


def design_rows(config: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    production = config["production"]
    systematics = config["paired_systematics"]
    for t_index, t_source in enumerate(config["kinematics"]["t_points"]):
        t = float(t_source)
        for replicate in range(int(production["replicates"])):
            rows.append(
                {
                    "task_id": str(len(rows)),
                    "task_kind": "production_order10",
                    "t_index": str(t_index),
                    "t": repr(t),
                    "replicate": str(replicate),
                    "sobol_power": str(int(production["sobol_power"])),
                    "seed": str(int(production["base_seed"]) + replicate),
                }
            )
    for t_index, t_source in enumerate(config["kinematics"]["t_points"]):
        rows.append(
            {
                "task_id": str(len(rows)),
                "task_kind": "systematics_order10",
                "t_index": str(t_index),
                "t": repr(float(t_source)),
                "replicate": "-1",
                "sobol_power": str(int(systematics["sobol_power"])),
                "seed": str(int(systematics["base_seed"])),
            }
        )
    return rows


def prepare_design(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = load_config(config_path)
    rows = design_rows(config)
    if len(rows) != 450:
        raise ValueError("the block-10 upgrade must contain 450 tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    snapshot_path = output_dir / "config.snapshot.json"
    atomic_write_json(snapshot_path, config)
    summary = {
        "status": "blind_block10_upgrade_design",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(rows),
        "production_task_count": 420,
        "systematics_task_count": 30,
        "reuses_prior_order6_production_shards": True,
        "reuses_prior_order8_systematics_shards": True,
        "new_target_formula_available": False,
        "manifest_sha256": sha256_file(manifest_path),
        "config_snapshot_sha256": sha256_file(snapshot_path),
    }
    atomic_write_json(output_dir / "design_summary.json", summary)
    return summary


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or tuple(rows[0]) != MANIFEST_FIELDS:
        raise ValueError("unexpected manifest schema")
    for expected, row in enumerate(rows):
        if int(row["task_id"]) != expected:
            raise ValueError("manifest task ids must be contiguous")
    return rows


def _build_from_item(t: float, item: dict[str, object], momentum_power: float):
    return build_kernel(
        t,
        block_order=int(item["block_order"]),
        momentum_order=int(item["momentum_base_order"]),
        momentum_maximum=float(item["momentum_maximum"]),
        momentum_power=momentum_power,
    )


def production_shard(config: dict[str, object], row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    production = config["production"]
    build_start = time.perf_counter()
    kernel = _build_from_item(t, production, float(production["momentum_power"]))
    build_seconds = time.perf_counter() - build_start
    points = qmc.Sobol(d=7, scramble=True, seed=int(row["seed"])).random_base2(
        int(row["sobol_power"])
    )
    integration_start = time.perf_counter()
    values, channel_summary = evaluate_common_points(
        [kernel], points, radial_power=float(production["radial_power"])
    )
    integration_seconds = time.perf_counter() - integration_start
    i6 = values[0]
    return {
        "status": "blind_worldsheet_block10_production_shard",
        "code_version": CODE_VERSION,
        "task": row,
        "t": t,
        "distance_to_first_residue_wall": FIRST_RESIDUE_WALL - t,
        "I6_order10": complex_pair(i6),
        "Q5_order10": complex_pair(_q5_from_i6(i6, t)),
        "mu4_A_tree_order10": complex_pair(1.0j * i6 / (8.0 * math.pi**3)),
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


def _evaluate_systematics_configuration(
    t: float,
    production: dict[str, object],
    systematics: dict[str, object],
    item: dict[str, object],
) -> dict[str, object]:
    build_start = time.perf_counter()
    kernel = _build_from_item(t, item, float(production["momentum_power"]))
    build_seconds = time.perf_counter() - build_start
    q5_values: list[complex] = []
    channel_summaries: list[dict[str, object]] = []
    integration_start = time.perf_counter()
    for replicate in range(int(systematics["replicates"])):
        seed = int(systematics["base_seed"]) + replicate
        points = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(
            int(systematics["sobol_power"])
        )
        values, channel_summary = evaluate_common_points(
            [kernel], points, radial_power=float(production["radial_power"])
        )
        channel_summaries.append(channel_summary)
        q5_values.append(_q5_from_i6(values[0], t))
    integration_seconds = time.perf_counter() - integration_start
    return {
        "name": str(item["name"]),
        "replicate_Q5": q5_values,
        "channel_selection": channel_summaries,
        "block_fallback_counts": dict(kernel.fallback_counts),
        "timing_seconds": {
            "kernel_build": build_seconds,
            "moduli_integration": integration_seconds,
            "total": build_seconds + integration_seconds,
        },
    }


def _systematics_configuration_child(
    connection,
    t: float,
    production: dict[str, object],
    systematics: dict[str, object],
    item: dict[str, object],
) -> None:
    try:
        payload = _evaluate_systematics_configuration(
            t, production, systematics, item
        )
        connection.send({"ok": True, "payload": payload})
    except BaseException:  # pragma: no cover - exercised by worker failure paths
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _evaluate_configuration_isolated(
    t: float,
    production: dict[str, object],
    systematics: dict[str, object],
    item: dict[str, object],
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_systematics_configuration_child,
        args=(child_connection, t, production, systematics, item),
    )
    process.start()
    child_connection.close()
    process.join()
    if process.exitcode != 0:
        parent_connection.close()
        raise RuntimeError(
            f"isolated systematics configuration {item['name']!r} exited "
            f"with code {process.exitcode}"
        )
    if not parent_connection.poll():
        parent_connection.close()
        raise RuntimeError(
            f"isolated systematics configuration {item['name']!r} returned no payload"
        )
    message = parent_connection.recv()
    parent_connection.close()
    if not bool(message.get("ok")):
        raise RuntimeError(
            f"isolated systematics configuration {item['name']!r} failed:\n"
            f"{message.get('traceback', 'no traceback available')}"
        )
    return message["payload"]


def systematics_shard(config: dict[str, object], row: dict[str, str]) -> dict[str, object]:
    t = float(row["t"])
    production = config["production"]
    systematics = config["paired_systematics"]
    configurations = list(systematics["configurations"])
    started = time.perf_counter()
    results = [
        _evaluate_configuration_isolated(t, production, systematics, item)
        for item in configurations
    ]
    names = [str(item["name"]) for item in configurations]
    if [str(result["name"]) for result in results] != names:
        raise RuntimeError("isolated systematics results changed configuration order")
    reference_channels = results[0]["channel_selection"]
    if any(result["channel_selection"] != reference_channels for result in results[1:]):
        raise RuntimeError("identical Sobol points selected different channel summaries")
    build_seconds = sum(
        float(result["timing_seconds"]["kernel_build"]) for result in results
    )
    integration_seconds = sum(
        float(result["timing_seconds"]["moduli_integration"]) for result in results
    )
    return {
        "status": "blind_worldsheet_block10_systematics_shard",
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
            "execution_mode": "isolated_process_per_configuration",
            "pairing": "identical Sobol seeds and powers regenerated in every isolated process",
        },
        "replicate_Q5_order10": {
            str(result["name"]): [
                complex_pair(value) for value in result["replicate_Q5"]
            ]
            for result in results
        },
        "channel_selection": reference_channels,
        "channel_selection_by_configuration": {
            str(result["name"]): result["channel_selection"] for result in results
        },
        "block_fallback_counts": {
            str(result["name"]): result["block_fallback_counts"]
            for result in results
        },
        "configuration_timing_seconds": {
            str(result["name"]): result["timing_seconds"] for result in results
        },
        "timing_seconds": {
            "kernel_build": build_seconds,
            "moduli_integration": integration_seconds,
            "total": time.perf_counter() - started,
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
    row = rows[int(task_id)]
    if row["task_kind"] == "production_order10":
        payload = production_shard(config, row)
    elif row["task_kind"] == "systematics_order10":
        payload = systematics_shard(config, row)
    else:
        raise ValueError(f"unknown task kind {row['task_kind']!r}")
    output = shards_dir / f"task_{int(task_id):04d}.json"
    atomic_write_json(output, payload)
    return output


def _mean_and_se(values: Sequence[complex]) -> tuple[complex, float, float]:
    array = np.asarray(values, dtype=complex)
    if len(array) < 2:
        return complex(np.mean(array)), math.inf, math.inf
    return (
        complex(np.mean(array)),
        float(np.std(array.real, ddof=1) / math.sqrt(len(array))),
        float(np.std(array.imag, ddof=1) / math.sqrt(len(array))),
    )


def _diagnostic(values: Sequence[complex]) -> dict[str, object]:
    mean, se_real, se_imag = _mean_and_se(values)
    return {
        "paired_shift_Q5": complex_pair(mean),
        "paired_standard_error_Q5": {"real": se_real, "imag": se_imag},
        "two_sigma_absolute_bound_Q5": abs(mean.real) + 2.0 * se_real,
        "replicate_differences_Q5": [complex_pair(value) for value in values],
    }


def _load_shards(
    rows: Sequence[dict[str, str]], shards_dir: Path
) -> dict[int, dict[str, object]]:
    loaded: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for row in rows:
        task_id = int(row["task_id"])
        path = shards_dir / f"task_{task_id:04d}.json"
        if not path.exists():
            missing.append(task_id)
            continue
        payload = json.loads(path.read_text())
        if bool(payload.get("target_formula_available", True)):
            raise ValueError(f"blindness violation in {path}")
        task = payload.get("task")
        if not isinstance(task, dict):
            raise ValueError(f"missing task provenance in {path}")
        for field in ("task_id", "t_index", "t", "replicate", "sobol_power", "seed"):
            if str(task.get(field)) != str(row[field]):
                raise ValueError(
                    f"task provenance mismatch for {field} in {path}: "
                    f"{task.get(field)!r} != {row[field]!r}"
                )
        loaded[task_id] = payload
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} shards: {missing[:20]}")
    return loaded


def assemble_upgrade(
    config_path: Path,
    manifest_path: Path,
    new_shards_dir: Path,
    prior_shards_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    rows = read_manifest(manifest_path)
    new = _load_shards(rows, new_shards_dir)
    prior_rows = [
        {
            **row,
            "task_kind": "production" if int(row["task_id"]) < 420 else "systematics",
        }
        for row in rows
    ]
    prior = _load_shards(prior_rows, prior_shards_dir)
    for task_id in range(420):
        if int(new[task_id]["settings"]["block_order"]) != 10:
            raise ValueError("new production shard is not block order 10")
        if int(prior[task_id]["settings"]["block_order"]) != 6:
            raise ValueError("stored production shard is not block order 6")
    production_by_t: dict[int, list[int]] = defaultdict(list)
    for row in rows[:420]:
        production_by_t[int(row["t_index"])].append(int(row["task_id"]))

    target = float(config["accuracy"]["target"])
    leakage_limit = float(config["accuracy"]["maximum_imaginary_leakage"])
    points: list[dict[str, object]] = []
    failed_indices: list[int] = []
    for t_index, t_source in enumerate(config["kinematics"]["t_points"]):
        t = float(t_source)
        task_ids = production_by_t[t_index]
        q10 = [pair_complex(new[task_id]["Q5_order10"]) for task_id in task_ids]
        i10 = [pair_complex(new[task_id]["I6_order10"]) for task_id in task_ids]
        a10 = [pair_complex(new[task_id]["mu4_A_tree_order10"]) for task_id in task_ids]
        q6 = [pair_complex(prior[task_id]["Q5_worldsheet"]) for task_id in task_ids]
        for task_id, high, low in zip(task_ids, q10, q6):
            if new[task_id]["task"]["seed"] != prior[task_id]["task"]["seed"]:
                raise ValueError("order-10 and order-6 production seeds do not match")
            if new[task_id]["task"]["sobol_power"] != prior[task_id]["task"]["sobol_power"]:
                raise ValueError("order-10 and order-6 Sobol powers do not match")
        q10_mean, q10_se_real, q10_se_imag = _mean_and_se(q10)
        i10_mean, i10_se_real, i10_se_imag = _mean_and_se(i10)
        a10_mean, a10_se_real, a10_se_imag = _mean_and_se(a10)
        production_shift = _diagnostic([high - low for high, low in zip(q10, q6)])

        new_systematics = new[420 + t_index]
        old_systematics = prior[420 + t_index]
        if "block_order_plus_two" not in old_systematics["replicate_Q5"]:
            raise ValueError("stored systematics shard has no order-8 values")
        q10_sys = {
            name: [pair_complex(value) for value in values]
            for name, values in new_systematics["replicate_Q5_order10"].items()
        }
        q8_sys = [
            pair_complex(value)
            for value in old_systematics["replicate_Q5"]["block_order_plus_two"]
        ]
        if len(q8_sys) != len(q10_sys["reference"]):
            raise ValueError("stored order-8 and new order-10 replicate counts differ")
        diagnostics = {
            "block_order_8_to_10": _diagnostic(
                [high - low for high, low in zip(q10_sys["reference"], q8_sys)]
            ),
            "momentum_order_at_block10": _diagnostic(
                [
                    high - low
                    for high, low in zip(
                        q10_sys["momentum_order_plus_two"], q10_sys["reference"]
                    )
                ]
            ),
            "momentum_cutoff_at_block10": _diagnostic(
                [
                    high - low
                    for high, low in zip(
                        q10_sys["momentum_cutoff_plus_one"],
                        q10_sys["momentum_order_plus_two"],
                    )
                ]
            ),
        }
        bounds = {
            name: float(item["two_sigma_absolute_bound_Q5"])
            for name, item in diagnostics.items()
        }
        envelope = max([q10_se_real, *bounds.values()])
        fallback_total = sum(
            int(value)
            for task_id in task_ids
            for value in new[task_id]["block_fallback_counts"].values()
        ) + sum(
            int(value)
            for counts in new_systematics["block_fallback_counts"].values()
            for value in counts.values()
        )
        recovery_count = sum(
            int(new[task_id]["channel_selection"]["source_chart_recovery_count"])
            for task_id in task_ids
        ) + sum(
            int(item["source_chart_recovery_count"])
            for item in new_systematics["channel_selection"]
        )
        sample_count = sum(
            int(new[task_id]["channel_selection"]["sample_count"])
            for task_id in task_ids
        ) + sum(
            int(item["sample_count"])
            for item in new_systematics["channel_selection"]
        )
        stable = (
            envelope <= target
            and abs(q10_mean.imag) <= leakage_limit
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
                "I6_worldsheet_order10": complex_pair(i10_mean),
                "I6_qmc_standard_error": {"real": i10_se_real, "imag": i10_se_imag},
                "Q5_worldsheet_order10": complex_pair(q10_mean),
                "Q5_qmc_standard_error": {"real": q10_se_real, "imag": q10_se_imag},
                "mu4_A_tree_worldsheet_order10": complex_pair(a10_mean),
                "mu4_A_tree_qmc_standard_error": {"real": a10_se_real, "imag": a10_se_imag},
                "production_replicate_Q5_order10": [complex_pair(value) for value in q10],
                "paired_order10_minus_order6_production": production_shift,
                "paired_systematics": diagnostics,
                "stability_components_Q5": {
                    "production_qmc_standard_error": q10_se_real,
                    **{f"{name}_two_sigma_bound": value for name, value in bounds.items()},
                },
                "stability_envelope_Q5": envelope,
                "accuracy_target_Q5": target,
                "passes_accuracy_gate": stable,
                "fallback_total": fallback_total,
                "source_chart_recovery": {
                    "count": recovery_count,
                    "geometric_sample_count": sample_count,
                    "fraction": recovery_count / sample_count,
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scan = {
        "status": "worldsheet_block10_upgrade_complete_unvalidated",
        "campaign": config["campaign"],
        "code_version": CODE_VERSION,
        "normalization": {
            "amplitude": "mu^4 A_tree = i I6/(8 pi^3)",
            "stripped": "Q5 = -I6/(40 pi^3 t^6) on omega=i t",
        },
        "reuse": config["reuse"],
        "accuracy": config["accuracy"],
        "production_settings": config["production"],
        "paired_systematics_settings": config["paired_systematics"],
        "point_count": len(points),
        "points": points,
        "target_formula_available": False,
        "comparison_performed": False,
    }
    scan_path = output_dir / "worldsheet_block10_scan_unfrozen.json"
    atomic_write_json(scan_path, scan)
    report = {
        "status": "blind_accuracy_passed" if not failed_indices else "blind_accuracy_failed",
        "campaign": config["campaign"],
        "point_count": len(points),
        "passed_point_count": len(points) - len(failed_indices),
        "failed_point_count": len(failed_indices),
        "failed_t_indices": failed_indices,
        "failed_t_values": [points[index]["t"] for index in failed_indices],
        "maximum_stability_envelope_Q5": max(float(item["stability_envelope_Q5"]) for item in points),
        "accuracy_target_Q5": target,
        "comparison_performed": False,
        "unfrozen_scan_sha256": sha256_file(scan_path),
    }
    atomic_write_json(output_dir / "accuracy_report.json", report)
    return report


def validate_and_freeze(config_path: Path, assembled_dir: Path) -> dict[str, object]:
    config = load_config(config_path)
    report_path = assembled_dir / "accuracy_report.json"
    scan_path = assembled_dir / "worldsheet_block10_scan_unfrozen.json"
    report = json.loads(report_path.read_text())
    scan = json.loads(scan_path.read_text())
    if int(report["failed_point_count"]) != 0:
        blocked = {
            "status": "freeze_blocked",
            "reason": "one or more order-10 point records fail the blind accuracy gate",
            "accuracy_report_sha256": sha256_file(report_path),
            "comparison_performed": False,
        }
        atomic_write_json(assembled_dir / "freeze_blocked.json", blocked)
        raise RuntimeError(
            f"{report['failed_point_count']} points failed the {config['accuracy']['target']} gate"
        )
    frozen = {
        **scan,
        "status": "worldsheet_block10_scan_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_performed": False,
    }
    frozen_path = assembled_dir / "worldsheet_block10_scan_frozen.json"
    atomic_write_json(frozen_path, frozen)
    manifest = {
        "status": "blind_worldsheet_block10_freeze_complete",
        "campaign": config["campaign"],
        "point_count": len(frozen["points"]),
        "all_points_pass_accuracy_gate": True,
        "accuracy_target_Q5": float(config["accuracy"]["target"]),
        "frozen_scan_sha256": sha256_file(frozen_path),
        "comparison_allowed": True,
        "comparison_performed": False,
    }
    atomic_write_json(assembled_dir / "freeze_manifest.json", manifest)
    return manifest


def benchmark(
    config_path: Path,
    t: float,
    kind: str,
    sobol_power: int,
    safety_factor: float,
) -> dict[str, object]:
    config = load_config(config_path)
    if kind == "production":
        row = {
            "task_id": "0",
            "task_kind": "production_order10",
            "t_index": "0",
            "t": repr(float(t)),
            "replicate": "0",
            "sobol_power": str(int(sobol_power)),
            "seed": str(int(config["production"]["base_seed"])),
        }
        reduced = json.loads(json.dumps(config))
        reduced["production"]["sobol_power"] = int(sobol_power)
        started = time.perf_counter()
        result = production_shard(reduced, row)
        observed = time.perf_counter() - started
        timing = result["timing_seconds"]
        build_seconds = float(timing["kernel_build"])
        sample_seconds = float(timing["moduli_integration"])
        benchmark_samples = 2 ** int(sobol_power)
        full_samples = 2 ** int(config["production"]["sobol_power"])
    elif kind == "systematics":
        row = {
            "task_id": "420",
            "task_kind": "systematics_order10",
            "t_index": "0",
            "t": repr(float(t)),
            "replicate": "-1",
            "sobol_power": str(int(sobol_power)),
            "seed": str(int(config["paired_systematics"]["base_seed"])),
        }
        reduced = json.loads(json.dumps(config))
        reduced["paired_systematics"]["sobol_power"] = int(sobol_power)
        reduced["paired_systematics"]["replicates"] = 1
        started = time.perf_counter()
        result = systematics_shard(reduced, row)
        observed = time.perf_counter() - started
        timing = result["timing_seconds"]
        build_seconds = float(timing["kernel_build"])
        sample_seconds = float(timing["moduli_integration"])
        benchmark_samples = 2 ** int(sobol_power)
        full_samples = int(config["paired_systematics"]["replicates"]) * 2 ** int(
            config["paired_systematics"]["sobol_power"]
        )
    else:
        raise ValueError("benchmark kind must be production or systematics")
    seconds_per_sample = sample_seconds / benchmark_samples
    projected = build_seconds + seconds_per_sample * full_samples
    return {
        "status": "blind_block10_runtime_benchmark",
        "code_version": CODE_VERSION,
        "benchmark_kind": kind,
        "t": float(t),
        "benchmark_sobol_power": int(sobol_power),
        "benchmark_sample_count": benchmark_samples,
        "observed_seconds": observed,
        "kernel_build_seconds": build_seconds,
        "seconds_per_geometric_sample": seconds_per_sample,
        "projected_worker_seconds": projected,
        "projected_worker_hours": projected / 3600.0,
        "conservative_worker_seconds": float(safety_factor) * projected,
        "conservative_worker_hours": float(safety_factor) * projected / 3600.0,
        "safety_factor": float(safety_factor),
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
    assemble_parser.add_argument("--new-shards-dir", type=Path, required=True)
    assemble_parser.add_argument("--prior-shards-dir", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    freeze_parser.add_argument("--assembled-dir", type=Path, required=True)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--t", type=float, required=True)
    benchmark_parser.add_argument("--kind", choices=("production", "systematics"), required=True)
    benchmark_parser.add_argument("--sobol-power", type=int, default=4)
    benchmark_parser.add_argument("--safety-factor", type=float, default=1.5)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        payload = prepare_design(arguments.config, arguments.design_dir)
    elif arguments.command == "worker":
        path = run_worker(
            arguments.config, arguments.manifest, arguments.task_id, arguments.shards_dir
        )
        payload = {"status": "worker_complete", "output": str(path)}
    elif arguments.command == "assemble":
        payload = assemble_upgrade(
            arguments.config,
            arguments.manifest,
            arguments.new_shards_dir,
            arguments.prior_shards_dir,
            arguments.output_dir,
        )
    elif arguments.command == "freeze":
        payload = validate_and_freeze(arguments.config, arguments.assembled_dir)
    else:
        payload = benchmark(
            arguments.config,
            arguments.t,
            arguments.kind,
            arguments.sobol_power,
            arguments.safety_factor,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
