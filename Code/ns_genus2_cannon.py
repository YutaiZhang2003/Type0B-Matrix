#!/usr/bin/env python3
"""Deterministic Cannon array for genus-two NS partition convergence runs.

Each array element evaluates one Cartesian momentum node for one point and
channel.  The primary finite-part radius is used at every point; configured
audit points also evaluate a second radius in the same shard.  Shards are immutable JSON files;
the reducer sums them in node order with ``math.fsum`` and only then forms
``Q_L=Z_L/Z_(X+psi)^9`` and the theta/glasses ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import mpmath
import numpy as np

from ns_genus2_partition import (
    C_HAT9,
    GLASSES_CCY_DESCENDANT_EDGE_ORDER,
    GLASSES_GEOMETRY_EDGE_ORDER,
    NSGenus2CRecursion,
    THETA_CCY_DESCENDANT_EDGE_ORDER,
    THETA_GEOMETRY_EDGE_ORDER,
    _primary_gaussian_rule,
    _structure_weight,
    free_superfield_partition,
    ns_weight,
    run_internal_checks,
)


SCHEMA = "ns-genus2-cannon-v4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _implementation_fingerprint(root: Path) -> str:
    files = (
        "ns_genus2_cannon.py",
        "ns_genus2_partition.py",
        "compare_ns_torus_c_h_recursion.py",
        "ns_genus_c_recursion_checks.py",
        "ns_recursion_recipe.py",
        "ns_global_osp_block.py",
        "ns_regular_block.py",
        "ns_vacuum_schottky.py",
        "super_liouville_structure_constants.py",
        "superconformal_blocks.py",
        "python/ccy_genus2_block.py",
        "python/free_boson_plumbing.py",
        "python/genus2_vacuum_blocks.py",
        "python/plumbing_algorithms.py",
        "python/virasoro_blocks.py",
    )
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    digest.update(sys.version.encode())
    digest.update(mpmath.__version__.encode())
    digest.update(np.__version__.encode())
    return digest.hexdigest()


def _recursion_orders(config: dict) -> tuple[int, ...]:
    """Return one or more accumulated twice-level cutoffs.

    ``recursion_order`` remains supported for archived production configs;
    new convergence runs may use ``recursion_orders`` to evaluate several
    cutoffs with the same geometry and momentum rule.
    """

    if "recursion_orders" in config:
        orders = tuple(int(value) for value in config["recursion_orders"])
    else:
        orders = (int(config["recursion_order"]),)
    if not orders or len(set(orders)) != len(orders):
        raise ValueError("recursion orders must be a nonempty unique list")
    return orders


def _designs(config: dict) -> list[dict]:
    designs = []
    for point in config["points"]:
        for recursion_order in _recursion_orders(config):
            for quadrature_order in config["quadrature_orders"]:
                node_count = int(quadrature_order) ** 3
                for channel in ("theta", "glasses"):
                    designs.append(
                        {
                            "point_id": str(point["id"]),
                            "channel": channel,
                            "recursion_order": int(recursion_order),
                            "quadrature_order": int(quadrature_order),
                            "node_count": node_count,
                        }
                    )
    return designs


def task_count(config: dict) -> int:
    return sum(design["node_count"] for design in _designs(config))


def decode_task(config: dict, task_index: int) -> tuple[dict, int]:
    remaining = int(task_index)
    for design in _designs(config):
        if remaining < design["node_count"]:
            return design, remaining
        remaining -= design["node_count"]
    raise IndexError(f"task index {task_index} is outside 0..{task_count(config)-1}")


def _point(config: dict, point_id: str) -> dict:
    return next(point for point in config["points"] if point["id"] == point_id)


def _q_values(config: dict, point_id: str, channel: str) -> tuple[complex, complex, complex]:
    return tuple(complex(value) for value in _point(config, point_id)["q_values"][channel])  # type: ignore[return-value]


def _omega_from_config(config: dict, point_id: str, channel: str) -> np.ndarray:
    entries = _point(config, point_id)["omega"][channel]
    return np.asarray(
        [[complex(entries[i][j]) for j in range(2)] for i in range(2)],
        dtype=np.complex128,
    )


def _node_data(config: dict, design: dict, node_index: int):
    order = int(design["quadrature_order"])
    channel = str(design["channel"])
    q_values = _q_values(config, str(design["point_id"]), channel)
    rules = [_primary_gaussian_rule(value, order) for value in q_values]
    i0, remainder = divmod(int(node_index), order * order)
    i1, i2 = divmod(remainder, order)
    indices = (i0, i1, i2)
    momenta = tuple(float(rules[edge][0][indices[edge]]) for edge in range(3))
    measure = math.prod(float(rules[edge][1][indices[edge]]) for edge in range(3))
    return q_values, indices, momenta, measure


def evaluate_task(config: dict, task_index: int) -> dict:
    design, node_index = decode_task(config, task_index)
    q_values, indices, momenta, measure = _node_data(config, design, node_index)
    channel = str(design["channel"])
    if channel == "theta":
        geometry_edge_order = THETA_GEOMETRY_EDGE_ORDER
        descendant_edge_order = THETA_CCY_DESCENDANT_EDGE_ORDER
    elif channel == "glasses":
        geometry_edge_order = GLASSES_GEOMETRY_EDGE_ORDER
        descendant_edge_order = GLASSES_CCY_DESCENDANT_EDGE_ORDER
    else:  # pragma: no cover - designs only contain the two known channels
        raise ValueError(f"unknown channel {channel!r}")
    lifts = tuple(int(value) for value in config["physical_lifts"][channel])
    numerics = config["numerics"]
    recursion = NSGenus2CRecursion(
        channel=channel,
        q_values=q_values,
        global_method=str(numerics.get("global_method", "auto")),
        global_tolerance=float(numerics["global_tolerance"]),
        global_max_total_occupation=int(numerics["global_max_total_occupation"]),
        vacuum_word_length=int(numerics["vacuum_word_length"]),
        vacuum_max_mode=int(numerics["vacuum_max_mode"]),
    )
    weights = tuple(ns_weight(momentum) for momentum in momenta)
    primary = np.exp(
        sum(weight * np.log(q) for weight, q in zip(weights, q_values))
    )
    point = _point(config, str(design["point_id"]))
    radii = [float(config["finite_part_radii"][0])]
    if bool(point.get("secondary_finite_part_radius", False)):
        radii.append(float(config["finite_part_radii"][1]))
    structures = {
        sector: _structure_weight(
            channel,
            sector,
            momenta,
            int(numerics["structure_precision"]),
        )
        for sector in (0, 1)
    }
    radius_results = []
    started = time.time()
    block_method = str(numerics.get("block_method", "contour_finite_part"))
    if block_method not in {"contour_finite_part", "collision_aware_mp"}:
        raise ValueError(
            "numerics.block_method must be contour_finite_part or "
            "collision_aware_mp"
        )
    block_working_precision = int(
        numerics.get("block_working_precision", 60)
    )
    for radius in radii:
        sectors = []
        contribution = 0.0
        for sector in (0, 1):
            if block_method == "collision_aware_mp":
                block = recursion.collision_aware_block_mp(
                    weights=weights,
                    sector=sector,
                    recursion_order=int(design["recursion_order"]),
                    lifts=lifts,
                    central_charge=C_HAT9,
                    working_precision=block_working_precision,
                )
            else:
                block = recursion.finite_part_block(
                    momenta=momenta,
                    sector=sector,
                    recursion_order=int(design["recursion_order"]),
                    lifts=lifts,
                    radius=radius,
                    samples=int(numerics["finite_part_samples"]),
                )
            sector_value = measure * structures[sector] * abs(primary * block) ** 2
            contribution += float(sector_value)
            sectors.append(
                {
                    "sector": sector,
                    "structure_weight": structures[sector],
                    "block": [float(block.real), float(block.imag)],
                    "contribution": float(sector_value),
                }
            )
        radius_results.append(
            {
                "finite_part_radius": radius,
                "contribution": float(contribution),
                "sectors": sectors,
            }
        )
    return {
        "schema": SCHEMA,
        "task_index": int(task_index),
        "node_index": int(node_index),
        **design,
        "q_edge_order": list(geometry_edge_order),
        "descendant_tensor_edge_order": list(descendant_edge_order),
        "indices": list(indices),
        "momenta": list(momenta),
        "measure": float(measure),
        "radius_results": radius_results,
        "runtime_seconds": float(time.time() - started),
        "block_method": block_method,
        "block_working_precision": block_working_precision,
        "global_method": recursion.effective_global_method,
        "global_method_requested": recursion.global_method,
        "global_resummed_calls": recursion.global_resummed_calls,
        "global_max_occupation_used": recursion.global_max_used,
        "global_nonconverged_calls": recursion.global_nonconverged_calls,
        "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative,
        "block_calls": recursion.block_calls,
        "confluent_moment_groups": recursion.confluent_moment_groups,
        "confluent_direct_groups": recursion.confluent_direct_groups,
        "confluent_max_moment_terms": recursion.confluent_max_moment_terms,
        "confluent_max_moment_ratio": recursion.confluent_max_moment_ratio,
    }


def worker(config_path: Path, output_dir: Path, task_index: int, force: bool) -> Path:
    config = _load(config_path)
    config_digest = _digest(config)
    implementation = _implementation_fingerprint(Path(__file__).resolve().parent)
    path = output_dir / f"task-{int(task_index):06d}.json"
    if path.exists() and not force:
        existing = _load(path)
        if (
            existing.get("config_digest") == config_digest
            and existing.get("implementation_fingerprint") == implementation
            and existing.get("schema") == SCHEMA
        ):
            return path
        raise RuntimeError(f"stale shard exists: {path}; pass --force intentionally")
    result = evaluate_task(config, int(task_index))
    result["config_digest"] = config_digest
    result["implementation_fingerprint"] = implementation
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    return path


def reduce(config_path: Path, shard_dir: Path, output: Path) -> dict:
    config = _load(config_path)
    count = task_count(config)
    shards = []
    for task_index in range(count):
        path = shard_dir / f"task-{task_index:06d}.json"
        if not path.exists():
            raise RuntimeError(f"missing shard {path}")
        shard = _load(path)
        if shard.get("config_digest") != _digest(config):
            raise RuntimeError(f"configuration mismatch in {path}")
        shards.append(shard)

    free = {}
    for point in config["points"]:
        point_id = str(point["id"])
        free[point_id] = {}
        for channel in ("theta", "glasses"):
            free[point_id][channel] = asdict(
                free_superfield_partition(
                    channel=channel,
                    q_values=_q_values(config, point_id, channel),
                    omega=_omega_from_config(config, point_id, channel),
                    physical_lifts=tuple(config["physical_lifts"][channel]),
                    max_word_length=int(config["numerics"]["free_word_length"]),
                    max_mode=int(config["numerics"]["free_max_mode"]),
                )
            )
            free[point_id][channel]["chiral_log"] = [
                float(free[point_id][channel]["chiral_log"].real),
                float(free[point_id][channel]["chiral_log"].imag),
            ]

    rows = []
    for design in _designs(config):
        selected = [
            shard
            for shard in shards
            if shard["point_id"] == design["point_id"]
            and shard["channel"] == design["channel"]
            and int(shard.get("recursion_order", config.get("recursion_order", -1)))
            == design["recursion_order"]
            and shard["quadrature_order"] == design["quadrature_order"]
        ]
        selected.sort(key=lambda row: row["node_index"])
        if len(selected) != design["node_count"]:
            raise RuntimeError(f"incomplete design {design}")
        radii = [float(config["finite_part_radii"][0])]
        if bool(_point(config, design["point_id"]).get("secondary_finite_part_radius", False)):
            radii.append(float(config["finite_part_radii"][1]))
        for radius in radii:
            contributions = []
            for shard in selected:
                result = next(
                    result
                    for result in shard["radius_results"]
                    if result["finite_part_radius"] == radius
                )
                contributions.append(float(result["contribution"]))
            z_l = math.fsum(contributions)
            denominator = float(free[design["point_id"]][design["channel"]]["value"])
            rows.append(
                {
                    **design,
                    "finite_part_radius": radius,
                    "z_liouville": z_l,
                    "z_free_superfield": denominator,
                    "q_l": z_l / denominator**9,
                    "runtime_seconds_sum": math.fsum(
                        float(row["runtime_seconds"]) for row in selected
                    ),
                    "global_method": selected[0].get("global_method", "direct"),
                    "global_resummed_calls": sum(
                        int(row.get("global_resummed_calls", 0)) for row in selected
                    ),
                    "global_nonconverged_calls": sum(
                        int(row["global_nonconverged_calls"]) for row in selected
                    ),
                    "global_worst_last_shell_relative": max(
                        float(row["global_worst_last_shell_relative"])
                        for row in selected
                    ),
                    "confluent_moment_groups": sum(
                        int(row.get("confluent_moment_groups", 0))
                        for row in selected
                    ),
                    "confluent_direct_groups": sum(
                        int(row.get("confluent_direct_groups", 0))
                        for row in selected
                    ),
                    "confluent_max_moment_terms": max(
                        int(row.get("confluent_max_moment_terms", 0))
                        for row in selected
                    ),
                    "confluent_max_moment_ratio": max(
                        float(row.get("confluent_max_moment_ratio", 0.0))
                        for row in selected
                    ),
                }
            )

    crossing = []
    for point in config["points"]:
        point_id = str(point["id"])
        radii = [float(config["finite_part_radii"][0])]
        if bool(point.get("secondary_finite_part_radius", False)):
            radii.append(float(config["finite_part_radii"][1]))
        for recursion_order in _recursion_orders(config):
            for order in config["quadrature_orders"]:
                for radius in radii:
                    pair = {
                        row["channel"]: row
                        for row in rows
                        if row["point_id"] == point_id
                        and row["recursion_order"] == recursion_order
                        and row["quadrature_order"] == int(order)
                        and row["finite_part_radius"] == radius
                    }
                    ratio = pair["theta"]["q_l"] / pair["glasses"]["q_l"]
                    crossing.append(
                        {
                            "point_id": point_id,
                            "recursion_order": recursion_order,
                            "quadrature_order": int(order),
                            "finite_part_radius": radius,
                            "theta_over_glasses": ratio,
                            "relative_difference": ratio - 1.0,
                        }
                    )

    radius_stability = []
    radius_a, radius_b = (float(value) for value in config["finite_part_radii"])
    for point in config["points"]:
        if not bool(point.get("secondary_finite_part_radius", False)):
            continue
        for channel in ("theta", "glasses"):
            for recursion_order in _recursion_orders(config):
                for order in config["quadrature_orders"]:
                    values = {
                        row["finite_part_radius"]: row["q_l"]
                        for row in rows
                        if row["point_id"] == point["id"]
                        and row["channel"] == channel
                        and row["recursion_order"] == recursion_order
                        and row["quadrature_order"] == int(order)
                    }
                    relative = abs(values[radius_a] - values[radius_b]) / max(
                        abs(values[radius_a]), abs(values[radius_b]), 1.0e-300
                    )
                    radius_stability.append(
                        {
                            "point_id": point["id"],
                            "channel": channel,
                            "recursion_order": recursion_order,
                            "quadrature_order": int(order),
                            "relative": relative,
                        }
                    )

    summary = {
        "schema": SCHEMA,
        "config_digest": _digest(config),
        "implementation_fingerprint": shards[0]["implementation_fingerprint"],
        "config": config,
        "task_count": count,
        "analytic_checks": run_internal_checks(),
        "free_superfield": free,
        "rows": rows,
        "crossing": crossing,
        "radius_stability": radius_stability,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def recompute_free(summary_path: Path, output: Path) -> dict:
    """Recompute only the free denominator and its derived quotient fields."""

    source = _load(summary_path)
    config = source["config"]
    numerics = config["numerics"]
    free = {}
    for point in config["points"]:
        point_id = str(point["id"])
        free[point_id] = {}
        for channel in ("theta", "glasses"):
            diagnostics = asdict(
                free_superfield_partition(
                    channel=channel,
                    q_values=_q_values(config, point_id, channel),
                    omega=_omega_from_config(config, point_id, channel),
                    physical_lifts=tuple(config["physical_lifts"][channel]),
                    max_word_length=int(numerics["free_word_length"]),
                    max_mode=int(numerics["free_max_mode"]),
                )
            )
            diagnostics["chiral_log"] = [
                float(diagnostics["chiral_log"].real),
                float(diagnostics["chiral_log"].imag),
            ]
            free[point_id][channel] = diagnostics

    rows = []
    for old_row in source["rows"]:
        row = dict(old_row)
        denominator = float(free[row["point_id"]][row["channel"]]["value"])
        row["z_free_superfield"] = denominator
        row["q_l"] = float(row["z_liouville"]) / denominator**9
        rows.append(row)

    crossing = []
    for old_crossing in source["crossing"]:
        pair = {
            row["channel"]: row
            for row in rows
            if row["point_id"] == old_crossing["point_id"]
            and row["quadrature_order"] == old_crossing["quadrature_order"]
            and row["finite_part_radius"]
            == old_crossing["finite_part_radius"]
        }
        ratio = pair["theta"]["q_l"] / pair["glasses"]["q_l"]
        crossing.append(
            {
                **old_crossing,
                "theta_over_glasses": ratio,
                "relative_difference": ratio - 1.0,
            }
        )

    radius_stability = []
    for old_stability in source["radius_stability"]:
        selected = [
            row
            for row in rows
            if row["point_id"] == old_stability["point_id"]
            and row["channel"] == old_stability["channel"]
            and row["quadrature_order"] == old_stability["quadrature_order"]
        ]
        values = [float(row["q_l"]) for row in selected]
        relative = abs(values[0] - values[1]) / max(
            *(abs(value) for value in values), 1.0e-300
        )
        radius_stability.append({**old_stability, "relative": relative})

    summary = dict(source)
    summary["numerator_implementation_fingerprint"] = source[
        "implementation_fingerprint"
    ]
    summary["free_implementation_fingerprint"] = _implementation_fingerprint(
        Path(__file__).resolve().parent
    )
    summary["free_rerun"] = {
        "scope": "free-superfield denominator only",
        "source_summary": str(summary_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "liouville_numerators_preserved": True,
        "theta_schottky_marking": "period-matched two-pants coordinates",
    }
    summary["analytic_checks"] = run_internal_checks()
    summary["free_superfield"] = free
    summary["rows"] = rows
    summary["crossing"] = crossing
    summary["radius_stability"] = radius_stability
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def recombine_channel(
    config_path: Path,
    rerun_shard_dir: Path,
    source_summary_path: Path,
    output: Path,
    rerun_channel: str,
) -> dict:
    """Combine one freshly evaluated channel with the preserved other channel."""

    if rerun_channel not in ("theta", "glasses"):
        raise ValueError("rerun_channel must be 'theta' or 'glasses'")
    preserved_channel = "glasses" if rerun_channel == "theta" else "theta"

    config = _load(config_path)
    source = _load(source_summary_path)
    if source["config_digest"] != _digest(config):
        # A channel-only spin correction intentionally changes that channel's
        # plumbing lifts while preserving every momentum node, geometry,
        # cutoff, and the opposite-channel numerator.  Permit exactly that
        # one-field difference so audited source shards can be reused.
        source_config = json.loads(json.dumps(source["config"]))
        current_config = json.loads(json.dumps(config))
        source_config["physical_lifts"][rerun_channel] = current_config[
            "physical_lifts"
        ][rerun_channel]
        if source_config != current_config:
            raise RuntimeError(
                "source summary and channel rerun differ beyond the rerun "
                "channel's physical lifts"
            )
    current_implementation = _implementation_fingerprint(
        Path(__file__).resolve().parent
    )
    rerun_shards = []
    for task_index in range(task_count(config)):
        design, _ = decode_task(config, task_index)
        if design["channel"] != rerun_channel:
            continue
        path = rerun_shard_dir / f"task-{task_index:06d}.json"
        if not path.exists():
            raise RuntimeError(f"missing {rerun_channel} shard {path}")
        shard = _load(path)
        if shard.get("config_digest") != _digest(config):
            raise RuntimeError(f"configuration mismatch in {path}")
        if shard.get("implementation_fingerprint") != current_implementation:
            raise RuntimeError(f"implementation mismatch in {path}")
        if shard.get("channel") != rerun_channel:
            raise RuntimeError(f"channel mismatch in {path}")
        rerun_shards.append(shard)

    numerics = config["numerics"]
    free = {}
    for point in config["points"]:
        point_id = str(point["id"])
        free[point_id] = {}
        for channel in ("theta", "glasses"):
            diagnostics = asdict(
                free_superfield_partition(
                    channel=channel,
                    q_values=_q_values(config, point_id, channel),
                    omega=_omega_from_config(config, point_id, channel),
                    physical_lifts=tuple(config["physical_lifts"][channel]),
                    max_word_length=int(numerics["free_word_length"]),
                    max_mode=int(numerics["free_max_mode"]),
                )
            )
            diagnostics["chiral_log"] = [
                float(diagnostics["chiral_log"].real),
                float(diagnostics["chiral_log"].imag),
            ]
            free[point_id][channel] = diagnostics

    rows = []
    for old_row in source["rows"]:
        row = dict(old_row)
        if row["channel"] == rerun_channel:
            selected = [
                shard
                for shard in rerun_shards
                if shard["point_id"] == row["point_id"]
                and shard["quadrature_order"] == row["quadrature_order"]
            ]
            selected.sort(key=lambda shard: shard["node_index"])
            if len(selected) != int(row["node_count"]):
                raise RuntimeError(
                    f"incomplete {rerun_channel} design for {row['point_id']}"
                )
            contributions = []
            for shard in selected:
                radius_result = next(
                    result
                    for result in shard["radius_results"]
                    if result["finite_part_radius"]
                    == row["finite_part_radius"]
                )
                contributions.append(float(radius_result["contribution"]))
            row["z_liouville"] = math.fsum(contributions)
            row["runtime_seconds_sum"] = math.fsum(
                float(shard["runtime_seconds"]) for shard in selected
            )
            row["global_nonconverged_calls"] = sum(
                int(shard["global_nonconverged_calls"])
                for shard in selected
            )
            row["global_worst_last_shell_relative"] = max(
                float(shard["global_worst_last_shell_relative"])
                for shard in selected
            )
            row["confluent_moment_groups"] = sum(
                int(shard.get("confluent_moment_groups", 0))
                for shard in selected
            )
            row["confluent_direct_groups"] = sum(
                int(shard.get("confluent_direct_groups", 0))
                for shard in selected
            )
            row["confluent_max_moment_terms"] = max(
                int(shard.get("confluent_max_moment_terms", 0))
                for shard in selected
            )
            row["confluent_max_moment_ratio"] = max(
                float(shard.get("confluent_max_moment_ratio", 0.0))
                for shard in selected
            )
        denominator = float(free[row["point_id"]][row["channel"]]["value"])
        row["z_free_superfield"] = denominator
        row["q_l"] = float(row["z_liouville"]) / denominator**9
        rows.append(row)

    crossing = []
    for old_crossing in source["crossing"]:
        pair = {
            row["channel"]: row
            for row in rows
            if row["point_id"] == old_crossing["point_id"]
            and row["quadrature_order"] == old_crossing["quadrature_order"]
            and row["finite_part_radius"]
            == old_crossing["finite_part_radius"]
        }
        ratio = pair["theta"]["q_l"] / pair["glasses"]["q_l"]
        crossing.append(
            {
                **old_crossing,
                "theta_over_glasses": ratio,
                "relative_difference": ratio - 1.0,
            }
        )

    radius_stability = []
    for old_stability in source["radius_stability"]:
        selected = [
            row
            for row in rows
            if row["point_id"] == old_stability["point_id"]
            and row["channel"] == old_stability["channel"]
            and row["quadrature_order"] == old_stability["quadrature_order"]
        ]
        values = [float(row["q_l"]) for row in selected]
        relative = abs(values[0] - values[1]) / max(
            *(abs(value) for value in values), 1.0e-300
        )
        radius_stability.append({**old_stability, "relative": relative})

    summary = dict(source)
    summary["config"] = config
    summary["config_digest"] = _digest(config)
    summary["task_count"] = task_count(config)
    source_fingerprints = source.get(
        "channel_numerator_implementation_fingerprints", {}
    )
    summary["channel_numerator_implementation_fingerprints"] = {
        rerun_channel: current_implementation,
        preserved_channel: source_fingerprints.get(
            preserved_channel, source["implementation_fingerprint"]
        ),
    }
    summary["free_implementation_fingerprint"] = current_implementation
    summary["consistent_recombination"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rerun_channel": rerun_channel,
        f"{rerun_channel}_liouville_numerator": (
            "fresh Cannon shards with current implementation"
        ),
        f"{preserved_channel}_liouville_numerator": (
            "preserved from source summary; code path unaffected"
        ),
        "free_superfield": (
            "fresh bosonized scalar-Majorana evaluation in both channels"
        ),
        "source_summary": str(source_summary_path),
        f"{rerun_channel}_shard_count": len(rerun_shards),
    }
    if rerun_channel == "glasses":
        summary["consistent_recombination"]["glasses_correction"] = (
            "self-loop toric sign S_rs^alpha=(-1)^(alpha*rs)"
        )
    summary.pop("free_rerun", None)
    summary["analytic_checks"] = run_internal_checks()
    summary["free_superfield"] = free
    summary["rows"] = rows
    summary["crossing"] = crossing
    summary["radius_stability"] = radius_stability
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def recombine_theta(
    config_path: Path,
    theta_shard_dir: Path,
    source_summary_path: Path,
    output: Path,
) -> dict:
    """Combine freshly evaluated theta numerators with unchanged glasses data."""

    return recombine_channel(
        config_path,
        theta_shard_dir,
        source_summary_path,
        output,
        "theta",
    )


def recombine_glasses(
    config_path: Path,
    glasses_shard_dir: Path,
    source_summary_path: Path,
    output: Path,
) -> dict:
    """Combine corrected glasses numerators with unchanged theta data."""

    return recombine_channel(
        config_path,
        glasses_shard_dir,
        source_summary_path,
        output,
        "glasses",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--task-count-only", action="store_true")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    worker_parser.add_argument("--force", action="store_true")
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--shard-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    free_parser = subparsers.add_parser("recompute-free")
    free_parser.add_argument("--summary", type=Path, required=True)
    free_parser.add_argument("--output", type=Path, required=True)
    recombine_parser = subparsers.add_parser("recombine-theta")
    recombine_parser.add_argument("--theta-shard-dir", type=Path, required=True)
    recombine_parser.add_argument("--source-summary", type=Path, required=True)
    recombine_parser.add_argument("--output", type=Path, required=True)
    recombine_glasses_parser = subparsers.add_parser("recombine-glasses")
    recombine_glasses_parser.add_argument(
        "--glasses-shard-dir", type=Path, required=True
    )
    recombine_glasses_parser.add_argument(
        "--source-summary", type=Path, required=True
    )
    recombine_glasses_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _load(args.config)
    if args.command == "plan":
        if args.task_count_only:
            print(task_count(config))
        else:
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "task_count": task_count(config),
                        "config_digest": _digest(config),
                        "designs": _designs(config),
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "worker":
        print(worker(args.config, args.output_dir, args.task_index, args.force))
        return 0
    if args.command == "reduce":
        summary = reduce(args.config, args.shard_dir, args.output)
        print(json.dumps({"output": str(args.output), "crossing": summary["crossing"]}, indent=2))
        return 0
    if args.command == "recompute-free":
        summary = recompute_free(args.summary, args.output)
        print(
            json.dumps(
                {"output": str(args.output), "crossing": summary["crossing"]},
                indent=2,
            )
        )
        return 0
    if args.command == "recombine-theta":
        summary = recombine_theta(
            args.config,
            args.theta_shard_dir,
            args.source_summary,
            args.output,
        )
        print(
            json.dumps(
                {"output": str(args.output), "crossing": summary["crossing"]},
                indent=2,
            )
        )
        return 0
    if args.command == "recombine-glasses":
        summary = recombine_glasses(
            args.config,
            args.glasses_shard_dir,
            args.source_summary,
            args.output,
        )
        print(
            json.dumps(
                {"output": str(args.output), "crossing": summary["crossing"]},
                indent=2,
            )
        )
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
