#!/usr/bin/env python3
"""Deterministic Cannon array for genus-two NS partition convergence runs.

Each array element evaluates one Cartesian momentum node for one point and
channel.  The primary finite-part radius is used at every point; configured
audit points also evaluate a second radius in the same shard.  Theta shards
    record the absolute three-form parity and the human-note nonchiral sign for
    both sectors in both channels.  Shards are immutable JSON files;
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
import scipy

from ns_genus2_partition import (
    C_ORDINARY_AT_HAT_C_9,
    GLASSES_CCY_DESCENDANT_EDGE_ORDER,
    GLASSES_GEOMETRY_EDGE_ORDER,
    HAT_C_TARGET,
    NSGenus2CRecursion,
    THETA_CCY_DESCENDANT_EDGE_ORDER,
    THETA_GEOMETRY_EDGE_ORDER,
    _primary_gaussian_rule,
    _spin_characteristic_from_lifts,
    _transport_spin_characteristic,
    _structure_weight,
    free_superfield_partition,
    ns_weight,
    run_internal_checks,
)
from genus_2.glasses_partition import (
    glasses_diagonal_sector_contribution,
    glasses_sector_pair,
)
from genus_2.theta_partition import (
    TYPE0B_NS_PRIMARY_PARITIES,
    theta_diagonal_sector_contribution,
    theta_sector_pair,
)


SCHEMA = "ns-genus2-cannon-v7-glasses-parity"
THETA_PARITY_SCHEMA = "ns-genus2-cannon-v6-theta-parity"
PHYSICAL_FREE_DENOMINATOR_SCOPE = {
    "role": "physical free-theory denominator of Q_L",
    "one_superfield": (
        "one noncompact real scalar plus one physical NS Majorana"
    ),
    "formula": "det(Im Omega)^(-1/2) |theta[delta](0|Omega)| |P_X|^3",
    "nonchiral_sign": (
        "Human Note sewing signs are resummed into the fixed-spin Majorana "
        "determinant; no extra sign multiplies its absolute value"
    ),
    "auxiliary_double_virasoro_fermion_used": False,
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_config_spin_characteristics(config: dict) -> dict:
    """Fail closed unless the two channel spins are related by the period map."""

    expected = config.get("expected_spin_characteristics")
    if expected is None:
        raise ValueError(
            "production configs must specify expected_spin_characteristics"
        )
    if set(expected) != {"theta", "glasses"}:
        raise ValueError(
            "expected_spin_characteristics must specify theta and glasses"
        )
    if "physical_lifts" not in config:
        raise ValueError("spin-checked configs must specify physical_lifts")

    provenance = config.get("provenance", {})
    matrix_entries = provenance.get(
        "symplectic_matrix_glasses_to_theta_after_branch"
    )
    if matrix_entries is None:
        raise ValueError(
            "production configs must specify the glasses-to-theta "
            "symplectic matrix"
        )
    matrix = np.asarray(matrix_entries, dtype=int)
    if provenance.get("spin_transport_source_channel") != "glasses":
        raise ValueError("spin transport source channel must be glasses")
    if provenance.get("spin_transport_target_channel") != "theta":
        raise ValueError("spin transport target channel must be theta")
    period_tolerance = float(
        provenance.get("spin_transport_period_tolerance", 5.0e-10)
    )
    if period_tolerance <= 0:
        raise ValueError("spin transport period tolerance must be positive")

    ledger: dict[str, dict[str, dict[str, list[int]]]] = {}
    for point in config.get("points", ()):
        point_id = str(point["id"])
        ledger[point_id] = {}
        actual_characteristics = {}
        for channel in ("theta", "glasses"):
            actual_alpha, actual_beta = _spin_characteristic_from_lifts(
                channel,
                tuple(complex(value) for value in point["q_values"][channel]),
                tuple(int(value) for value in config["physical_lifts"][channel]),
            )
            declared = expected[channel]
            declared_alpha = tuple(int(value) for value in declared["alpha"])
            declared_beta = tuple(int(value) for value in declared["beta"])
            actual = (actual_alpha, actual_beta)
            actual_characteristics[channel] = actual
            if actual != (declared_alpha, declared_beta):
                raise ValueError(
                    f"spin characteristic mismatch at {point_id}/{channel}: "
                    f"lifts give {actual}, config declares "
                    f"{(declared_alpha, declared_beta)}"
                )
            ledger[point_id][channel] = {
                "alpha": list(actual_alpha),
                "beta": list(actual_beta),
            }

        transported = _transport_spin_characteristic(
            matrix, actual_characteristics["glasses"]
        )
        if transported != actual_characteristics["theta"]:
            raise ValueError(
                f"modular spin mismatch at {point_id}: glasses "
                f"{actual_characteristics['glasses']} transports to "
                f"{transported}, not theta {actual_characteristics['theta']}"
            )

        omega_glasses = _omega_from_config(config, point_id, "glasses")
        omega_theta = _omega_from_config(config, point_id, "theta")
        A, B = matrix[:2, :2], matrix[:2, 2:]
        C, D = matrix[2:, :2], matrix[2:, 2:]
        transported_omega = (A @ omega_glasses + B) @ np.linalg.inv(
            C @ omega_glasses + D
        )
        period_residual = float(np.max(np.abs(transported_omega - omega_theta)))
        if period_residual > period_tolerance:
            raise ValueError(
                f"period-map direction mismatch at {point_id}: residual "
                f"{period_residual:.3e} exceeds {period_tolerance:.3e}"
            )
    return ledger


def _digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _implementation_fingerprint(root: Path) -> str:
    files = (
        "c_Recursion/ns_genus2_cannon.py",
        "c_Recursion/ns_genus2_partition.py",
        "c_Recursion/compare_ns_torus_c_h_recursion.py",
        "c_Recursion/ns_genus_c_recursion_checks.py",
        "c_Recursion/ns_human_convention.py",
        "c_Recursion/ns_recursion_recipe.py",
        "c_Recursion/ns_global_osp_block.py",
        "c_Recursion/ns_regular_block.py",
        "c_Recursion/ns_vacuum_schottky.py",
        "c_Recursion/super_liouville_structure_constants.py",
        "c_Recursion/superconformal_blocks.py",
        "genus_2/glasses_partition.py",
        "genus_2/theta_partition.py",
        "genus_2_cross_channel/ccy_genus2_block.py",
        "genus_2_cross_channel/free_boson_plumbing.py",
        "genus_2_cross_channel/genus2_vacuum_blocks.py",
        "genus_2_cross_channel/free_majorana_pair_of_pants.py",
        "genus_2_cross_channel/plumbing_algorithms.py",
        "genus_2_cross_channel/virasoro_blocks.py",
    )
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    digest.update(sys.version.encode())
    digest.update(mpmath.__version__.encode())
    digest.update(np.__version__.encode())
    digest.update(scipy.__version__.encode())
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Return the numerical runtime versions that affect production shards."""

    return {
        "python": sys.version,
        "mpmath": mpmath.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


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


def _cutoff_pairs(config: dict) -> tuple[tuple[int, int], ...]:
    """Return the requested ``(recursion order, quadrature order)`` pairs.

    The legacy fields define a Cartesian product.  ``convergence_designs``
    permits an axis sweep without evaluating irrelevant off-axis pairs, e.g.
    ``(R,N)=(20,10),(22,10),(24,8),(24,10),(24,12)``.
    """

    explicit = config.get("convergence_designs")
    if explicit is None:
        pairs = tuple(
            (recursion_order, int(quadrature_order))
            for recursion_order in _recursion_orders(config)
            for quadrature_order in config["quadrature_orders"]
        )
    else:
        pairs = tuple(
            (int(item["recursion_order"]), int(item["quadrature_order"]))
            for item in explicit
        )
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("cutoff pairs must be a nonempty unique list")
    if any(
        recursion_order < 0 or quadrature_order <= 0
        for recursion_order, quadrature_order in pairs
    ):
        raise ValueError(
            "recursion orders must be nonnegative and quadrature orders positive"
        )
    return pairs


def _designs(config: dict) -> list[dict]:
    designs = []
    for point in config["points"]:
        for recursion_order, quadrature_order in _cutoff_pairs(config):
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


def channel_task_chunks(
    config: dict, channel: str, tasks_per_chunk: int
) -> list[tuple[int, int]]:
    """Return inclusive task ranges for one channel without crossing designs."""

    if channel not in ("theta", "glasses"):
        raise ValueError("channel must be theta or glasses")
    chunk_size = int(tasks_per_chunk)
    if chunk_size <= 0:
        raise ValueError("tasks_per_chunk must be positive")
    chunks: list[tuple[int, int]] = []
    design_start = 0
    for design in _designs(config):
        design_stop = design_start + int(design["node_count"])
        if design["channel"] == channel:
            for start in range(design_start, design_stop, chunk_size):
                chunks.append((start, min(start + chunk_size, design_stop) - 1))
        design_start = design_stop
    return chunks


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


def _validate_shard(
    config: dict,
    task_index: int,
    shard: dict,
    expected_implementation: str,
) -> None:
    """Validate an immutable shard against its filename-derived task design."""

    expected_design, expected_node_index = decode_task(config, int(task_index))
    prefix = f"task-{int(task_index):06d}"
    if shard.get("schema") != SCHEMA:
        raise RuntimeError(f"schema mismatch in {prefix}")
    if int(shard.get("task_index", -1)) != int(task_index):
        raise RuntimeError(f"task_index mismatch in {prefix}")
    if int(shard.get("node_index", -1)) != expected_node_index:
        raise RuntimeError(f"node_index mismatch in {prefix}")
    if shard.get("config_digest") != _digest(config):
        raise RuntimeError(f"configuration mismatch in {prefix}")
    if shard.get("implementation_fingerprint") != expected_implementation:
        raise RuntimeError(f"implementation mismatch in {prefix}")

    for key, expected in expected_design.items():
        observed = shard.get(key)
        if isinstance(expected, int):
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                pass
        if observed != expected:
            raise RuntimeError(
                f"decoded design mismatch for {key} in {prefix}: "
                f"expected {expected!r}, got {observed!r}"
            )

    _, expected_indices, expected_momenta, expected_measure = _node_data(
        config, expected_design, expected_node_index
    )
    if tuple(int(value) for value in shard.get("indices", ())) != expected_indices:
        raise RuntimeError(f"quadrature indices mismatch in {prefix}")
    observed_momenta = tuple(float(value) for value in shard.get("momenta", ()))
    if len(observed_momenta) != 3 or any(
        not math.isclose(observed, expected, rel_tol=2.0e-14, abs_tol=1.0e-15)
        for observed, expected in zip(observed_momenta, expected_momenta)
    ):
        raise RuntimeError(f"quadrature momenta mismatch in {prefix}")
    if not math.isclose(
        float(shard.get("measure", math.nan)),
        expected_measure,
        rel_tol=2.0e-14,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(f"quadrature measure mismatch in {prefix}")

    channel = str(expected_design["channel"])
    expected_geometry_order = (
        THETA_GEOMETRY_EDGE_ORDER
        if channel == "theta"
        else GLASSES_GEOMETRY_EDGE_ORDER
    )
    expected_descendant_order = (
        THETA_CCY_DESCENDANT_EDGE_ORDER
        if channel == "theta"
        else GLASSES_CCY_DESCENDANT_EDGE_ORDER
    )
    if tuple(shard.get("q_edge_order", ())) != expected_geometry_order:
        raise RuntimeError(f"geometry edge order mismatch in {prefix}")
    if tuple(shard.get("descendant_tensor_edge_order", ())) != expected_descendant_order:
        raise RuntimeError(f"descendant edge order mismatch in {prefix}")
    radius_results = shard.get("radius_results")
    if not isinstance(radius_results, list) or not radius_results:
        raise RuntimeError(f"missing radius parity ledger in {prefix}")
    for radius_result in radius_results:
        sector_rows = radius_result.get("sectors", ())
        if len(sector_rows) != 2:
            raise RuntimeError(f"incomplete sector parity ledger in {prefix}")
        if {int(row.get("sector", -1)) for row in sector_rows} != {0, 1}:
            raise RuntimeError(f"invalid sector labels in {prefix}")
        for row in sector_rows:
            sector = int(row["sector"])
            pair_function = (
                theta_sector_pair if channel == "theta" else glasses_sector_pair
            )
            pair = pair_function(
                sector,
                holomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                antiholomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
            )
            if int(row.get("partition_sign", 0)) != pair.sign:
                raise RuntimeError(f"{channel} partition sign mismatch in {prefix}")
            if (
                int(row.get("absolute_three_form_parity", -1))
                != pair.absolute_parity
            ):
                raise RuntimeError(
                    f"{channel} absolute-parity mismatch in {prefix}"
                )
            if (
                int(row.get("antiholomorphic_sector", -1))
                != pair.antiholomorphic_sector
            ):
                raise RuntimeError(
                    f"{channel} antiholomorphic-sector mismatch in {prefix}"
                )
        sector_sum = math.fsum(float(row["contribution"]) for row in sector_rows)
        if not math.isclose(
            sector_sum,
            float(radius_result.get("contribution", math.nan)),
            rel_tol=2.0e-15,
            abs_tol=1.0e-300,
        ):
            raise RuntimeError(f"signed sector reduction mismatch in {prefix}")


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
                    central_charge=C_ORDINARY_AT_HAT_C_9,
                    working_precision=block_working_precision,
                    primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
            else:
                block = recursion.finite_part_block(
                    momenta=momenta,
                    sector=sector,
                    recursion_order=int(design["recursion_order"]),
                    lifts=lifts,
                    radius=radius,
                    samples=int(numerics["finite_part_samples"]),
                    primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
            if channel == "theta":
                pair = theta_sector_pair(
                    sector,
                    holomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                    antiholomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
                sector_value = theta_diagonal_sector_contribution(
                    sector=sector,
                    measure=measure,
                    structure_weight=structures[sector],
                    primary_times_block=primary * block,
                    primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
                partition_sign = pair.sign
                absolute_parity = pair.absolute_parity
                antiholomorphic_sector = pair.antiholomorphic_sector
            else:
                pair = glasses_sector_pair(
                    sector,
                    holomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                    antiholomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
                sector_value = glasses_diagonal_sector_contribution(
                    sector=sector,
                    measure=measure,
                    structure_weight=structures[sector],
                    primary_times_block=primary * block,
                    primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                )
                partition_sign = pair.sign
                absolute_parity = pair.absolute_parity
                antiholomorphic_sector = pair.antiholomorphic_sector
            contribution += float(sector_value)
            sectors.append(
                {
                    "sector": sector,
                    "antiholomorphic_sector": antiholomorphic_sector,
                    "absolute_three_form_parity": absolute_parity,
                    "partition_sign": partition_sign,
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
        "central_charge_convention": "ordinary c; hat_c=2c/3",
        "central_charge": C_ORDINARY_AT_HAT_C_9,
        "hat_c": HAT_C_TARGET,
        "theta_partition_formula": (
            "(-1)^(a+p1+p2+p3) C_a^2 |prod(q_i^h_i) F_a|^2"
        ),
        "glasses_partition_formula": (
            "(-1)^(a+p_bridge) C_LBL^a C_RBR^a "
            "|prod(q_i^h_i) F_a|^2"
        ),
        "nonchiral_assembly_scope": {
            "theta": "parity-correct human-note formula",
            "glasses": "parity-correct glasses sewing formula",
        },
        "structure_constant_convention": {
            "source": "real BRY b=1 coefficients",
            "human_note_boundary": (
                "C_HN^(0)=C_BRY; C_HN^(1)=i*tilde_C_BRY"
            ),
            "odd_two_pants_phase": -1,
            "human_note_sewing_sign_retained": True,
        },
        "type0b_ns_primary_parities": list(TYPE0B_NS_PRIMARY_PARITIES),
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
        "confluent_max_direct_cancellation_ratio": (
            recursion.confluent_max_direct_cancellation_ratio
        ),
        "confluent_max_moment_cancellation_ratio": (
            recursion.confluent_max_moment_cancellation_ratio
        ),
        "confluent_max_moment_series_cancellation_ratio": (
            recursion.confluent_max_moment_series_cancellation_ratio
        ),
        "confluent_max_total_cancellation_ratio": (
            recursion.confluent_max_total_cancellation_ratio
        ),
    }


def worker(config_path: Path, output_dir: Path, task_index: int, force: bool) -> Path:
    config = _load(config_path)
    _validate_config_spin_characteristics(config)
    config_digest = _digest(config)
    implementation = _implementation_fingerprint(Path(__file__).resolve().parents[1])
    path = output_dir / f"task-{int(task_index):06d}.json"
    if path.exists() and not force:
        existing = _load(path)
        if (
            existing.get("config_digest") == config_digest
            and existing.get("implementation_fingerprint") == implementation
            and existing.get("schema") == SCHEMA
        ):
            _validate_shard(config, int(task_index), existing, implementation)
            return path
        raise RuntimeError(f"stale shard exists: {path}; pass --force intentionally")
    result = evaluate_task(config, int(task_index))
    result["config_digest"] = config_digest
    result["implementation_fingerprint"] = implementation
    result["runtime_versions"] = _runtime_versions()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    return path


def reduce(config_path: Path, shard_dir: Path, output: Path) -> dict:
    config = _load(config_path)
    spin_characteristics = _validate_config_spin_characteristics(config)
    count = task_count(config)
    implementation = _implementation_fingerprint(Path(__file__).resolve().parents[1])
    expected_names = {f"task-{task_index:06d}.json" for task_index in range(count)}
    observed_names = {path.name for path in shard_dir.glob("task-*.json")}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise RuntimeError(
            "shard filename set mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    shards = []
    for task_index in range(count):
        path = shard_dir / f"task-{task_index:06d}.json"
        if not path.exists():
            raise RuntimeError(f"missing shard {path}")
        shard = _load(path)
        _validate_shard(config, task_index, shard, implementation)
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
                    "confluent_max_direct_cancellation_ratio": max(
                        float(
                            row.get(
                                "confluent_max_direct_cancellation_ratio", 0.0
                            )
                        )
                        for row in selected
                    ),
                    "confluent_max_moment_cancellation_ratio": max(
                        float(
                            row.get(
                                "confluent_max_moment_cancellation_ratio", 0.0
                            )
                        )
                        for row in selected
                    ),
                    "confluent_max_moment_series_cancellation_ratio": max(
                        float(
                            row.get(
                                "confluent_max_moment_series_cancellation_ratio",
                                0.0,
                            )
                        )
                        for row in selected
                    ),
                    "confluent_max_total_cancellation_ratio": max(
                        float(
                            row.get(
                                "confluent_max_total_cancellation_ratio", 0.0
                            )
                        )
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
        for recursion_order, order in _cutoff_pairs(config):
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
            for recursion_order, order in _cutoff_pairs(config):
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
        "quantity": "Q_L = Z_L / Z_(X+psi)^9",
        "free_denominator_scope": PHYSICAL_FREE_DENOMINATOR_SCOPE,
        "theta_partition_formula": (
            "(-1)^(a+p1+p2+p3) C_a^2 |prod(q_i^h_i) F_a|^2"
        ),
        "type0b_ns_primary_parities": list(TYPE0B_NS_PRIMARY_PARITIES),
        "config_digest": _digest(config),
        "implementation_fingerprint": implementation,
        "runtime_versions": _runtime_versions(),
        "config": config,
        "task_count": count,
        "spin_characteristics": spin_characteristics,
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
    spin_characteristics = _validate_config_spin_characteristics(config)
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
            and row["recursion_order"] == old_crossing["recursion_order"]
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
            and row["recursion_order"] == old_stability["recursion_order"]
            and row["quadrature_order"] == old_stability["quadrature_order"]
        ]
        values = [float(row["q_l"]) for row in selected]
        relative = abs(values[0] - values[1]) / max(
            *(abs(value) for value in values), 1.0e-300
        )
        radius_stability.append({**old_stability, "relative": relative})

    summary = dict(source)
    summary["quantity"] = "Q_L = Z_L / Z_(X+psi)^9"
    summary["free_denominator_scope"] = PHYSICAL_FREE_DENOMINATOR_SCOPE
    summary["numerator_implementation_fingerprint"] = source[
        "implementation_fingerprint"
    ]
    summary["free_implementation_fingerprint"] = _implementation_fingerprint(
        Path(__file__).resolve().parents[1]
    )
    summary["runtime_versions"] = _runtime_versions()
    summary["free_rerun"] = {
        "scope": "free-superfield denominator only",
        "source_summary": str(summary_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "liouville_numerators_preserved": True,
        "theta_schottky_marking": "period-matched two-pants coordinates",
    }
    summary["analytic_checks"] = run_internal_checks()
    if spin_characteristics:
        summary["spin_characteristics"] = spin_characteristics
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
    spin_characteristics = _validate_config_spin_characteristics(config)
    source = _load(source_summary_path)
    source_schema = source.get("schema")
    allowed_source_schemas = (
        {SCHEMA, THETA_PARITY_SCHEMA}
        if rerun_channel == "glasses"
        else {SCHEMA}
    )
    if source_schema not in allowed_source_schemas:
        raise RuntimeError(
            f"cannot preserve the {preserved_channel} numerator from schema "
            f"{source_schema!r}; rerun both channels or use a parity-correct "
            "source"
        )
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
        Path(__file__).resolve().parents[1]
    )
    expected_task_indices = {
        task_index
        for task_index in range(task_count(config))
        if decode_task(config, task_index)[0]["channel"] == rerun_channel
    }
    expected_names = {
        f"task-{task_index:06d}.json" for task_index in expected_task_indices
    }
    observed_names = {path.name for path in rerun_shard_dir.glob("task-*.json")}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise RuntimeError(
            f"{rerun_channel} shard filename set mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
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
        _validate_shard(config, task_index, shard, current_implementation)
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
                and shard["recursion_order"] == row["recursion_order"]
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
            for diagnostic in (
                "confluent_max_direct_cancellation_ratio",
                "confluent_max_moment_cancellation_ratio",
                "confluent_max_moment_series_cancellation_ratio",
                "confluent_max_total_cancellation_ratio",
            ):
                row[diagnostic] = max(
                    float(shard.get(diagnostic, 0.0)) for shard in selected
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
            and row["recursion_order"] == old_crossing["recursion_order"]
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
            and row["recursion_order"] == old_stability["recursion_order"]
            and row["quadrature_order"] == old_stability["quadrature_order"]
        ]
        values = [float(row["q_l"]) for row in selected]
        relative = abs(values[0] - values[1]) / max(
            *(abs(value) for value in values), 1.0e-300
        )
        radius_stability.append({**old_stability, "relative": relative})

    summary = dict(source)
    summary["schema"] = SCHEMA
    summary["quantity"] = "Q_L = Z_L / Z_(X+psi)^9"
    summary["free_denominator_scope"] = PHYSICAL_FREE_DENOMINATOR_SCOPE
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
    summary["runtime_versions"] = _runtime_versions()
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
        "theta_partition_formula": (
            "(-1)^(a+p1+p2+p3) C_a^2 |prod(q_i^h_i) F_a|^2"
        ),
        "glasses_partition_formula": (
            "(-1)^(a+p_bridge) C_LBL^a C_RBR^a "
            "|prod(q_i^h_i) F_a|^2"
        ),
    }
    if rerun_channel == "glasses":
        summary["consistent_recombination"]["glasses_correction"] = (
            "ordinary vacuum/global product; incidence-ordered "
            "P_rs^a P_rs^(a+rs); parity-correct nonchiral sewing"
        )
    summary.pop("free_rerun", None)
    summary["analytic_checks"] = run_internal_checks()
    if spin_characteristics:
        summary["spin_characteristics"] = spin_characteristics
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
    chunks_parser = subparsers.add_parser("channel-chunks")
    chunks_parser.add_argument("--channel", choices=("theta", "glasses"), required=True)
    chunks_parser.add_argument("--tasks-per-chunk", type=int, required=True)
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
    _validate_config_spin_characteristics(config)
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
    if args.command == "channel-chunks":
        for start, stop in channel_task_chunks(
            config, args.channel, args.tasks_per_chunk
        ):
            print(f"{start}\t{stop}")
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
