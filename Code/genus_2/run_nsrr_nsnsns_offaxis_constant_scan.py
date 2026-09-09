#!/usr/bin/env python3
"""Off-axis genus-two NSRR/NSNSNS constant-ratio test.

The scan stays near the already validated t=0.60 surface but independently
varies all six real entries of a symmetric genus-two period matrix.  Source
and target are evaluated in the same re-marked NSRR and transformed all-NS
theta charts used by the five-point comparison.

The source uses the amplitude-level [11|00] lift projection and the unscaled
Human-block ``M`` contraction.  Equivalently, it multiplies the local ``M/4``
helper result by four.  This script tests moduli dependence only; it does not
declare that overall factor to be the final CFT normalization.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import audit_nsrr_nsnsns_constant_ratio as ratio_audit
import nsrr_factorized_sign_trial as trial
import nsrr_human_note_geometry as human_geometry
import nsrr_nsnsns_theta_omega_scan as scan
import recompute_all_ns_reference as all_ns
import refine_nsrr_factorized_sign_trial as refined_source
from fixed_spin_free_plumbing import charged_frame, fixed_spin_partition
from physical_nsrr_sewing import (
    CHANNELS,
    SOURCE_FIXED_SPIN_LIFTS,
    contract_physical_blocks,
    project_source_fixed_spin,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_GEOMETRY = ROOT / "Data Set" / "nsrr_nsnsns_human_note_repair_20260830" / "geometry.json"
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_offaxis_constant_scan_20260904"
SCHEMA = "nsrr-nsnsns-offaxis-constant-scan-v1"
SOURCE_SPIN = ((1, 1), (0, 0))
TARGET_SPIN = ((0, 0), (0, 0))
TARGET_LIFTS = (1, -1, 1)
UNSCALED_M_OVER_LOCAL_KERNEL = 4.0


VARIATIONS = (
    ("center", {}),
    ("re_omega11_plus", {"re11": +0.035}),
    ("re_omega11_minus", {"re11": -0.035}),
    ("re_omega22_plus", {"re22": +0.010}),
    ("re_omega22_minus", {"re22": -0.010}),
    ("re_omega12_plus", {"re12": +0.035}),
    ("re_omega12_minus", {"re12": -0.035}),
    ("im_omega11_plus", {"im11": +0.060}),
    ("im_omega11_minus", {"im11": -0.060}),
    ("im_omega22_plus", {"im22": +0.060}),
    ("im_omega22_minus", {"im22": -0.060}),
    ("im_omega12_plus", {"im12": +0.035}),
    ("im_omega12_minus", {"im12": -0.035}),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def encoded_matrix(matrix: np.ndarray) -> list[list[str]]:
    return [[str(complex(value)) for value in row] for row in matrix]


def matrix(values) -> np.ndarray:
    return np.asarray([[complex(value) for value in row] for row in values], dtype=complex)


def varied_omega(changes: dict[str, float]) -> np.ndarray:
    omega = np.asarray([[1j, 0.6 + 0.5j], [0.6 + 0.5j, 1j]], dtype=complex)
    omega[0, 0] += changes.get("re11", 0.0) + 1j * changes.get("im11", 0.0)
    omega[1, 1] += changes.get("re22", 0.0) + 1j * changes.get("im22", 0.0)
    omega[0, 1] += changes.get("re12", 0.0) + 1j * changes.get("im12", 0.0)
    omega[1, 0] = omega[0, 1]
    if np.linalg.eigvalsh(omega.imag)[0] <= 0:
        raise ValueError("the proposed period matrix is not in Siegel space")
    return omega


def charge_period_branch(q_values, omega: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    frame = charged_frame(tuple(complex(value) for value in q_values), max_mode=32)
    branch = np.rint((frame.omega_charge - omega).real).astype(int)
    if not np.array_equal(branch, branch.T):
        raise ArithmeticError("charged-frame period branch is not symmetric")
    if np.max(abs(frame.omega_charge - omega - branch)) > 2.0e-8:
        raise ArithmeticError("charged and marked periods do not differ by an integer branch")
    return tuple(tuple(int(value) for value in row) for row in branch)


def continued_target_lifts(
    q_values,
    reference_branch,
    current_branch,
    reference_lifts=TARGET_LIFTS,
) -> tuple[int, int, int]:
    r"""Continue the target spin lift across principal-log branch cuts.

    The inverse plumbing chart returns ordinary complex ``q`` values, so its
    principal square roots jump when an edge crosses the negative real axis.
    The lift on that edge must jump at the same time to represent the same
    continuously transported block.  Calibrating at the center chart, the
    affine integer-period action shifts the principal-frame beta label by
    ``diag(B-B_ref)`` modulo two.  Enumerating the four representatives with
    the first edge lift fixed removes the irrelevant simultaneous sign flip.
    """

    reference = np.asarray(reference_branch, dtype=int)
    current = np.asarray(current_branch, dtype=int)
    if reference.shape != (2, 2) or current.shape != (2, 2):
        raise ValueError("period branches must be 2x2 matrices")
    if not np.array_equal(reference, reference.T) or not np.array_equal(current, current.T):
        raise ValueError("period branches must be symmetric")
    base = scan._spin_characteristic_from_lifts("theta", q_values, reference_lifts)
    desired_beta = tuple(
        int((base[1][index] + current[index, index] - reference[index, index]) % 2)
        for index in range(2)
    )
    desired = base[0], desired_beta
    candidates = [
        (int(reference_lifts[0]), eta_one, eta_infinity)
        for eta_one in (1, -1)
        for eta_infinity in (1, -1)
    ]
    matches = [
        lifts
        for lifts in candidates
        if scan._spin_characteristic_from_lifts("theta", q_values, lifts) == desired
    ]
    if len(matches) != 1:
        raise ArithmeticError(
            f"period-branch continuation has {len(matches)} lift representatives, expected one"
        )
    return matches[0]


def prepare_config(
    geometry_path: Path,
    orders: tuple[int, ...],
    *,
    reuse_parent_output: Path | None = None,
) -> dict:
    baseline = load(geometry_path)
    center = next(point for point in baseline["points"] if float(point["t"]) == 0.6)
    source_seed = tuple(complex(value) for value in center["source_chart"]["q_values"])
    target_seed = tuple(complex(value) for value in center["target_chart"]["q_values"])
    points = []
    for point_id, changes in VARIATIONS:
        omega = varied_omega(changes)
        source_omega = human_geometry.action(human_geometry.SOURCE_REMARKING, omega)
        target_omega = scan.omega_action(omega)
        source_chart = scan.inverse_chart(source_omega, source_seed)
        target_chart = scan.inverse_chart(target_omega, target_seed)
        source_q = tuple(complex(value) for value in source_chart["q_values"])
        target_q = tuple(complex(value) for value in target_chart["q_values"])
        source_branch = charge_period_branch(source_q, source_omega)
        target_branch = charge_period_branch(target_q, target_omega)
        source_free = fixed_spin_partition(
            source_q,
            source_omega,
            SOURCE_SPIN,
            period_branch=source_branch,
            max_mode=32,
        )
        target_free = fixed_spin_partition(
            target_q,
            target_omega,
            TARGET_SPIN,
            period_branch=target_branch,
            max_mode=32,
        )
        points.append(
            {
                "point_id": point_id,
                "variation": changes,
                "omega_reference": encoded_matrix(omega),
                "source": {
                    **source_chart,
                    "characteristic": SOURCE_SPIN,
                    "period_branch": source_branch,
                    "Z_free": source_free["Z_free"],
                },
                "target": {
                    **target_chart,
                    "characteristic": TARGET_SPIN,
                    "period_branch": target_branch,
                    "principal_lifts_at_reference": TARGET_LIFTS,
                    "Z_free": target_free["Z_free"],
                },
            }
        )
        print(
            f"prepared {point_id}: max|q| source={max(map(abs, source_q)):.5f}, "
            f"target={max(map(abs, target_q)):.5f}",
            flush=True,
        )
    reference_target_branch = points[0]["target"]["period_branch"]
    for point in points:
        q = tuple(complex(value) for value in point["target"]["q_values"])
        point["target"]["lifts"] = continued_target_lifts(
            q,
            reference_target_branch,
            point["target"]["period_branch"],
        )
    b = 1.4
    config = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "b": b,
        "kappa": 1.0 + 2.0 * (b + 1.0 / b) ** 2,
        "orders": list(orders),
        "source_level": 3.0,
        "target_recursion_twice_level": 16,
        "source_contraction": (
            "amplitude-level [11|00] projection followed by the unscaled Human M kernel; "
            "four times the M/4 local helper"
        ),
        "normalization_status": "not fixed; this run tests constancy over moduli",
        "analytic_continuation": {
            "reference_point": points[0]["point_id"],
            "reference_target_period_branch": reference_target_branch,
            "reference_target_lifts": TARGET_LIFTS,
            "rule": "principal-frame beta shifts by diag(B-B_reference) mod 2",
        },
        "source_to_target": (scan.MATRIX @ np.rint(np.linalg.inv(human_geometry.SOURCE_REMARKING)).astype(int)).tolist(),
        "points": points,
        "q_envelope": {
            channel: [
                max(abs(complex(point[channel]["q_values"][edge])) for point in points)
                for edge in range(3)
            ]
            for channel in ("source", "target")
        },
        "protected_kernel_hashes": all_ns.protected_hashes(),
    }
    if reuse_parent_output is not None:
        config["reuse_parent_output"] = str(reuse_parent_output.resolve())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config["schema"] != SCHEMA or config["orders"] != sorted(set(config["orders"])):
        raise ValueError("invalid off-axis scan configuration")
    if not config["orders"] or any(order not in (2, 3, 4, 5) for order in config["orders"]):
        raise ValueError("this bounded audit supports momentum orders 2 through 5")
    if config["source_level"] != 3.0:
        raise ValueError("this bounded audit requires source L=3")
    point_design = config.get("point_design")
    expected_point_ids = (
        [str(value) for value in point_design["point_ids"]]
        if point_design is not None
        else [item[0] for item in VARIATIONS]
    )
    if config["target_recursion_twice_level"] != 16 or len(config["points"]) != len(expected_point_ids):
        raise ValueError("the bounded target cutoff or point design changed")
    if config["protected_kernel_hashes"] != all_ns.protected_hashes():
        raise ValueError("a protected conformal-block kernel changed")
    if [point["point_id"] for point in config["points"]] != expected_point_ids:
        raise ValueError("off-axis point ordering changed")
    for point in config["points"]:
        omega = matrix(point["omega_reference"])
        source = matrix(point["source"]["omega"])
        target = matrix(point["target"]["omega"])
        if np.max(abs(human_geometry.action(human_geometry.SOURCE_REMARKING, omega) - source)) > 1.0e-11:
            raise ValueError("source marking mismatch")
        if np.max(abs(scan.omega_action(omega) - target)) > 1.0e-11:
            raise ValueError("target marking mismatch")
        for channel in ("source", "target"):
            if point[channel]["Z_free"] <= 0:
                raise ValueError("nonpositive free factor")
            for q, envelope in zip(point[channel]["q_values"], config["q_envelope"][channel]):
                if not 0 < abs(complex(q)) <= envelope * (1.0 + 1.0e-13) < 1:
                    raise ValueError("plumbing parameter outside the shared envelope")
    continuation = config.get("analytic_continuation")
    if continuation is not None:
        reference_branch = continuation["reference_target_period_branch"]
        for point in config["points"]:
            q = tuple(complex(value) for value in point["target"]["q_values"])
            expected = continued_target_lifts(
                q, reference_branch, point["target"]["period_branch"]
            )
            if tuple(point["target"]["lifts"]) != expected:
                raise ValueError(f"{point['point_id']}: target lift continuation mismatch")


def tasks(config: dict) -> list[tuple[int, int]]:
    return [(order, node) for order in config["orders"] for node in range(order**3)]


def node_data(config: dict, channel: str, task_index: int):
    order, node = tasks(config)[task_index]
    indices = np.unravel_index(node, (order,) * 3)
    rules = trial._rules(config["q_envelope"][channel], order)
    momenta = tuple(float(rules[edge][0][indices[edge]]) for edge in range(3))
    measure = float(trial._measure(rules, indices))
    return order, node, momenta, measure


def reusable_parent(config: dict, channel: str, task_index: int):
    """Return a compatible parent config/shard pair, if one was requested."""

    parent_value = config.get("reuse_parent_output")
    if parent_value is None:
        return None
    parent_dir = Path(parent_value)
    parent_config = load(parent_dir / "config.json")
    order, node, momenta, measure = node_data(config, channel, task_index)
    parent_tasks = tasks(parent_config)
    matches = [
        index
        for index, pair in enumerate(parent_tasks)
        if pair == (order, node)
    ]
    if len(matches) != 1:
        raise ValueError(f"parent does not contain unique task {(order, node)}")
    parent_index = matches[0]
    shard = load(parent_dir / channel / "shards" / f"node-{parent_index:03d}.json")
    if shard["quadrature_order"] != order or shard["node"] != node:
        raise ValueError("parent shard order/node mismatch")
    if not np.allclose(shard["momenta"], momenta, rtol=0, atol=0) or shard["measure"] != measure:
        raise ValueError("parent quadrature design differs")
    if [row["point_id"] for row in shard["values"]] != [point["point_id"] for point in config["points"]]:
        raise ValueError("parent point design differs")
    return parent_dir, parent_config, shard


def source_worker(config_path: Path, output_dir: Path, task_index: int) -> None:
    config = load(config_path)
    validate_config(config)
    order, node, momenta, measure = node_data(config, "source", task_index)
    parent = reusable_parent(config, "source", task_index)
    if parent is not None:
        parent_dir, parent_config, parent_shard = parent
        for new, old in zip(config["points"], parent_config["points"]):
            for key in ("q_values", "omega", "Z_free"):
                if new["source"][key] != old["source"][key]:
                    raise ValueError(f"source {key} differs from reuse parent")
        shard = dict(parent_shard)
        shard["config_digest"] = digest(config)
        shard["task_index"] = task_index
        shard["reused_from"] = str(parent_dir.resolve())
        save(output_dir / f"node-{task_index:03d}.json", shard)
        return
    constants = trial.GenericSuperLiouvilleConstants(config["b"], dps=30)
    bry = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    components, checks = refined_source.block_components(config["b"], momenta[::-1], 3)
    values = []
    for point in config["points"]:
        q = tuple(complex(value) for value in point["source"]["q_values"])
        amplitudes = {}
        for lift in SOURCE_FIXED_SPIN_LIFTS:
            plumbing = trial.NSRRPlumbingInputs(q, lift, trial.GEOMETRY_SECTORS)
            primary = plumbing.primary(config["b"], momenta)
            blocks = trial.evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, 3.0)
            amplitudes[lift] = {channel: primary * blocks[channel] for channel in CHANNELS}
        projected = project_source_fixed_spin(amplitudes)
        local = contract_physical_blocks(projected, bry)
        values.append(
            {
                "point_id": point["point_id"],
                "source_Z_unscaled_M_node": UNSCALED_M_OVER_LOCAL_KERNEL * local["total"],
                "source_Z_local_M_over_4_node": local["total"],
                "diagonal_Z_unscaled_node": UNSCALED_M_OVER_LOCAL_KERNEL * local["diagonal"],
                "interference_Z_unscaled_node": UNSCALED_M_OVER_LOCAL_KERNEL * local["interference"],
            }
        )
    shard = {
        "schema": SCHEMA,
        "config_digest": digest(config),
        "channel": "source",
        "task_index": task_index,
        "quadrature_order": order,
        "node": node,
        "momenta": momenta,
        "measure": measure,
        "checks": checks,
        "values": values,
    }
    save(output_dir / f"node-{task_index:03d}.json", shard)


def target_worker(config_path: Path, output_dir: Path, task_index: int) -> None:
    config = load(config_path)
    validate_config(config)
    order, node, momenta, measure = node_data(config, "target", task_index)
    parent = reusable_parent(config, "target", task_index)
    parent_rows = {}
    parent_points = {}
    parent_dir = None
    if parent is not None:
        parent_dir, parent_config, parent_shard = parent
        parent_rows = {row["point_id"]: row for row in parent_shard["values"]}
        parent_points = {point["point_id"]: point for point in parent_config["points"]}
    constants = None
    values = []
    for point in config["points"]:
        old_point = parent_points.get(point["point_id"])
        if (
            old_point is not None
            and point["target"]["q_values"] == old_point["target"]["q_values"]
            and point["target"]["lifts"] == old_point["target"]["lifts"]
        ):
            reused = dict(parent_rows[point["point_id"]])
            reused["reused_from"] = str(parent_dir.resolve())
            values.append(reused)
            continue
        if constants is None:
            constants = scan.GenericSuperLiouvilleConstants(config["b"], dps=30)
        q = tuple(complex(value) for value in point["target"]["q_values"])
        recursion = scan.NSGenus2CRecursion(
            channel="theta",
            q_values=q,
            global_method="resummed",
            global_tolerance=2.0e-8,
            global_max_total_occupation=36,
            vacuum_word_length=7,
            vacuum_max_mode=50,
        )
        sectors = scan.all_ns_node(
            b=config["b"],
            q_values=q,
            lifts=point["target"]["lifts"],
            recursion_order=config["target_recursion_twice_level"],
            momenta=momenta,
            measure=measure,
            constants=constants,
            recursion=recursion,
            block_method="collision_aware_mp",
            block_working_precision=50,
        )
        if recursion.global_nonconverged_calls:
            raise ArithmeticError("target global block did not converge")
        values.append(
            {
                "point_id": point["point_id"],
                "sector_contributions": sectors,
                "global_max_occupation_used": recursion.global_max_used,
                "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative,
            }
        )
    shard = {
        "schema": SCHEMA,
        "config_digest": digest(config),
        "channel": "target",
        "task_index": task_index,
        "quadrature_order": order,
        "node": node,
        "momenta": momenta,
        "measure": measure,
        "values": values,
    }
    save(output_dir / f"node-{task_index:03d}.json", shard)


def validate_shard(config: dict, channel: str, task_index: int, shard: dict) -> None:
    order, node, momenta, measure = node_data(config, channel, task_index)
    expected = {
        "schema": SCHEMA,
        "config_digest": digest(config),
        "channel": channel,
        "task_index": task_index,
        "quadrature_order": order,
        "node": node,
    }
    for key, value in expected.items():
        if shard.get(key) != value:
            raise ValueError(f"{channel} shard {task_index}: {key} mismatch")
    if not np.allclose(shard["momenta"], momenta, rtol=0, atol=0) or shard["measure"] != measure:
        raise ValueError("quadrature node changed")
    if [row["point_id"] for row in shard["values"]] != [point["point_id"] for point in config["points"]]:
        raise ValueError("off-axis values are missing or reordered")


def reduce(config_path: Path, output_dir: Path) -> dict:
    config = load(config_path)
    validate_config(config)
    shards = {}
    for channel in ("source", "target"):
        rows = []
        for task_index in range(len(tasks(config))):
            shard = load(output_dir / channel / "shards" / f"node-{task_index:03d}.json")
            validate_shard(config, channel, task_index, shard)
            rows.append(shard)
        shards[channel] = rows

    comparisons = []
    fits = []
    fixed_normalization_checks = []
    fixed_normalization = "normalization_policy" in config
    for order in config["orders"]:
        source_shards = [row for row in shards["source"] if row["quadrature_order"] == order]
        target_shards = [row for row in shards["target"] if row["quadrature_order"] == order]
        fit_points = []
        order_rows = []
        for point_index, point in enumerate(config["points"]):
            source_z = math.fsum(
                shard["measure"] * shard["values"][point_index]["source_Z_unscaled_M_node"]
                for shard in source_shards
            )
            target_sectors = [
                math.fsum(shard["values"][point_index]["sector_contributions"][sector] for shard in target_shards)
                for sector in (0, 1)
            ]
            target_z = math.fsum(target_sectors)
            source_q = source_z / float(point["source"]["Z_free"]) ** config["kappa"]
            target_q = target_z / float(point["target"]["Z_free"]) ** config["kappa"]
            ratio = source_q / target_q
            fit_points.append({"coordinate": float(point_index), "ratio": ratio})
            order_rows.append(
                {
                    "point_id": point["point_id"],
                    "quadrature_order": order,
                    "source_Z_unscaled_M": source_z,
                    "target_Z": target_z,
                    "source_Q": source_q,
                    "target_Q": target_q,
                    "source_over_target": ratio,
                    "target_odd_fraction": target_sectors[1] / target_z,
                }
            )
        if fixed_normalization:
            residuals = [float(point["ratio"]) - 1.0 for point in fit_points]
            check = {
                "quadrature_order": order,
                "normalization": 1.0,
                "normalization_applied": False,
                "residuals": residuals,
                "maximum_absolute_fractional_residual": max(map(abs, residuals)),
                "rms_fractional_residual": math.sqrt(
                    math.fsum(value * value for value in residuals) / len(residuals)
                ),
                "minimum_raw_ratio": min(point["ratio"] for point in fit_points),
                "maximum_raw_ratio": max(point["ratio"] for point in fit_points),
            }
            for row, residual in zip(order_rows, residuals):
                row["fixed_normalization_one_residual"] = residual
            fixed_normalization_checks.append(check)
        else:
            fit = ratio_audit.constant_fit(fit_points)
            fit["quadrature_order"] = order
            for row, residual in zip(order_rows, fit["residuals"]):
                row["constant_fit_residual"] = residual
            fits.append(fit)
        comparisons.extend(order_rows)

    fine_order = max(config["orders"])
    fine = {row["point_id"]: row for row in comparisons if row["quadrature_order"] == fine_order}
    if len(config["orders"]) >= 2:
        coarse_order = config["orders"][-2]
        coarse = {
            row["point_id"]: row
            for row in comparisons
            if row["quadrature_order"] == coarse_order
        }
        for point_id, row in fine.items():
            row[f"source_quadrature_N{coarse_order}_to_N{fine_order}_change"] = (
                row["source_Q"] / coarse[point_id]["source_Q"] - 1.0
            )
            row[f"target_quadrature_N{coarse_order}_to_N{fine_order}_change"] = (
                row["target_Q"] / coarse[point_id]["target_Q"] - 1.0
            )
            row[f"ratio_N{coarse_order}_to_N{fine_order}_change"] = (
                row["source_over_target"] / coarse[point_id]["source_over_target"] - 1.0
            )

    center_check = None
    if "center" in fine:
        center = fine["center"]
        saved = load(ROOT / "Data Set" / "nsrr_nsnsns_human_convention_20260903" / "summary.json")
        saved_center_source = next(
            row["source_Z"]
            for row in saved["source_rows_L3"]
            if float(row["t"]) == 0.6 and int(row["quadrature_order"]) == 3 and float(row["level"]) == 3.0
        )
        center_check = {
            "new_source_order": fine_order,
            "new_source_Z_L3": center["source_Z_unscaled_M"],
            "saved_source_Z_N3_L3": saved_center_source,
            "relative_difference": center["source_Z_unscaled_M"] / saved_center_source - 1.0,
            "interpretation": (
                "The off-axis scan uses a wider common-|q| importance-sampling envelope, "
                "so equality is expected only after momentum-quadrature convergence."
            ),
        }
    result = {
        "schema": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "fits": fits,
        "fixed_normalization_checks": fixed_normalization_checks,
        "comparisons": comparisons,
        "center_cross_quadrature_design_check": center_check,
        "interpretation": (
            "For fixed-normalization configurations, compare every raw source/target ratio "
            "directly with one and assess the change under momentum-quadrature refinement. "
            "Legacy configurations retain their descriptive constant-ratio diagnostic."
        ),
    }
    save(output_dir / "summary.json", result)
    return result


def run_workers(config_path: Path, output_dir: Path, channel: str, workers: int) -> None:
    config = load(config_path)
    validate_config(config)
    shard_dir = output_dir / channel / "shards"
    log_dir = output_dir / channel / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def execute(task_index: int) -> int:
        path = shard_dir / f"node-{task_index:03d}.json"
        if path.exists():
            validate_shard(config, channel, task_index, load(path))
            return task_index
        with (log_dir / f"node-{task_index:03d}.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    f"{channel}-worker",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(shard_dir),
                    "--task-index",
                    str(task_index),
                ],
                check=True,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return task_index

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, index) for index in range(len(tasks(config)))]
        for count, future in enumerate(as_completed(futures), 1):
            print(
                f"{channel}: {count}/{len(futures)} complete; last={future.result()}; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    prepare_parser.add_argument("--reuse-parent-output", type=Path)
    prepare_order_parser = subparsers.add_parser("prepare-order")
    prepare_order_parser.add_argument("--base-config", type=Path, required=True)
    prepare_order_parser.add_argument("--orders", type=int, nargs="+", required=True)
    prepare_order_parser.add_argument("--output-dir", type=Path, required=True)
    for command in ("source-worker", "target-worker"):
        worker_parser = subparsers.add_parser(command)
        worker_parser.add_argument("--config", type=Path, required=True)
        worker_parser.add_argument("--output-dir", type=Path, required=True)
        worker_parser.add_argument("--task-index", type=int, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--source-workers", type=int, default=2)
    run_parser.add_argument("--target-workers", type=int, default=4)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.command == "prepare":
        config = prepare_config(
            args.geometry,
            tuple(sorted(set(args.orders))),
            reuse_parent_output=args.reuse_parent_output,
        )
        save(args.output_dir / "config.json", config)
    elif args.command == "prepare-order":
        base = load(args.base_config)
        validate_config(base)
        config = dict(base)
        config["orders"] = sorted(set(args.orders))
        config["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        config["parent_config_path"] = str(args.base_config.resolve())
        config["parent_config_digest"] = digest(base)
        validate_config(config)
        save(args.output_dir / "config.json", config)
    elif args.command == "source-worker":
        source_worker(args.config, args.output_dir, args.task_index)
    elif args.command == "target-worker":
        target_worker(args.config, args.output_dir, args.task_index)
    elif args.command == "reduce":
        result = reduce(args.output_dir / "config.json", args.output_dir)
        for fit in result["fits"]:
            print(
                f"N={fit['quadrature_order']} A={fit['normalization_geometric_mean']:.12f} "
                f"max={fit['maximum_absolute_fractional_residual']:.3%} "
                f"rms={fit['rms_fractional_residual']:.3%}"
            )
        for check in result["fixed_normalization_checks"]:
            print(
                f"N={check['quadrature_order']} fixed A=1 "
                f"max={check['maximum_absolute_fractional_residual']:.3%} "
                f"rms={check['rms_fractional_residual']:.3%}"
            )
    else:
        config_path = args.output_dir / "config.json"
        if not config_path.exists():
            save(config_path, prepare_config(DEFAULT_GEOMETRY, (2, 3)))
        run_workers(config_path, args.output_dir, "source", args.source_workers)
        run_workers(config_path, args.output_dir, "target", args.target_workers)
        result = reduce(config_path, args.output_dir)
        for fit in result["fits"]:
            print(
                f"N={fit['quadrature_order']} A={fit['normalization_geometric_mean']:.12f} "
                f"max={fit['maximum_absolute_fractional_residual']:.3%} "
                f"rms={fit['rms_fractional_residual']:.3%}"
            )
        for check in result["fixed_normalization_checks"]:
            print(
                f"N={check['quadrature_order']} fixed A=1 "
                f"max={check['maximum_absolute_fractional_residual']:.3%} "
                f"rms={check['rms_fractional_residual']:.3%}"
            )


if __name__ == "__main__":
    main()
