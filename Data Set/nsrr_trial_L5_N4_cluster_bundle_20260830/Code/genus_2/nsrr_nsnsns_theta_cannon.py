#!/usr/bin/env python3
"""Immutable Cannon shards for the NSRR/NSNSNS theta modular check.

One worker evaluates one Cartesian Gauss--Laguerre momentum node in one
channel.  The source uses the fixed-spin NSRR double-Virasoro construction;
the target uses the direct N=1 genus-two c-recursion.  The reducer validates
every shard and performs deterministic node-ordered ``math.fsum`` reductions
before dividing by the physical free ``X+psi`` factor in the same local frame.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import mpmath
import numpy as np
import scipy
import sympy


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for directory in (
    HERE,
    CODE_ROOT,
    CODE_ROOT / "c_Recursion",
    CODE_ROOT / "full_ramond_block_runtime",
    CODE_ROOT / "genus_2_cross_channel",
    CODE_ROOT / "ramond_branching_recursion",
    CODE_ROOT / "double_virasoro" / "nsrr",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compare_nsrr_nsnsns_theta import (  # noqa: E402
    RAMOND_GROUND_COMPLETENESS,
    _measure,
    _rules,
    all_ns_node,
    nsrr_node,
    same_frame_free_factors,
)
from generic_super_liouville_structure_constants import (  # noqa: E402
    GenericSuperLiouvilleConstants,
)
from ns_genus2_partition import (  # noqa: E402
    NSGenus2CRecursion,
    _spin_characteristic_from_lifts,
    _transport_spin_characteristic,
)


SCHEMA = "nsrr-nsnsns-theta-cannon-v1"
CHANNELS = ("source_nsrr", "target_nsnsns")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _complex_triple(values: Sequence[str]) -> tuple[complex, complex, complex]:
    parsed = tuple(complex(value) for value in values)
    if len(parsed) != 3:
        raise ValueError("each theta chart must specify three q values")
    if any(not 0 < abs(value) < 1 for value in parsed):
        raise ValueError("all plumbing parameters must satisfy 0<|q|<1")
    return parsed  # type: ignore[return-value]


def _characteristic(value: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    alpha = tuple(int(bit) for bit in value["alpha"])
    beta = tuple(int(bit) for bit in value["beta"])
    if len(alpha) != 2 or len(beta) != 2 or any(
        bit not in (0, 1) for bit in alpha + beta
    ):
        raise ValueError("a characteristic must contain two binary alpha/beta bits")
    return alpha, beta  # type: ignore[return-value]


def _omega(value: Sequence[Sequence[str]]) -> np.ndarray:
    result = np.asarray(
        [[complex(value[i][j]) for j in range(2)] for i in range(2)],
        dtype=np.complex128,
    )
    if not np.allclose(result, result.T, atol=2.0e-14, rtol=0.0):
        raise ValueError("period matrices must be symmetric")
    if min(np.linalg.eigvalsh(result.imag)) <= 0:
        raise ValueError("period matrices must lie in Siegel upper half-space")
    return result


def _validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    b = float(config["parameters"]["b"])
    if not math.isfinite(b) or b <= 0:
        raise ValueError("b must be finite and positive")
    designs = _cutoff_pairs(config)
    if any(order != 8 for order, _ in designs):
        raise ValueError("this precision run requires block order eight on both sides")
    if float(config["numerics"]["maximum_ward_residual"]) <= 0:
        raise ValueError("maximum Ward residual must be positive")

    charts = config["marked_surface"]["charts"]
    source_q = _complex_triple(charts["source_nsrr"]["q_values"])
    target_q = _complex_triple(charts["target_nsnsns"]["q_values"])
    source_lifts = tuple(int(value) for value in charts["source_nsrr"]["lifts"])
    target_lifts = tuple(int(value) for value in charts["target_nsnsns"]["lifts"])
    source_reference = _characteristic(
        charts["source_nsrr"]["physical_free_reference_characteristic"]
    )
    target_characteristic = _characteristic(
        charts["target_nsnsns"]["characteristic"]
    )
    actual_source_reference = _spin_characteristic_from_lifts(
        "theta", source_q, source_lifts
    )
    actual_target = _spin_characteristic_from_lifts(
        "theta", target_q, target_lifts
    )
    if actual_source_reference != source_reference:
        raise ValueError(
            "source physical-free reference lifts select "
            f"{actual_source_reference}, not {source_reference}"
        )
    if actual_target != target_characteristic:
        raise ValueError(
            f"target lifts select {actual_target}, not {target_characteristic}"
        )

    source_characteristic = _characteristic(
        charts["source_nsrr"]["characteristic"]
    )
    matrix = np.asarray(
        config["marked_surface"]["symplectic_matrix_source_to_target"],
        dtype=int,
    )
    transported_characteristic = _transport_spin_characteristic(
        matrix, source_characteristic
    )
    if transported_characteristic != target_characteristic:
        raise ValueError(
            f"source spin transports to {transported_characteristic}, "
            f"not {target_characteristic}"
        )

    source_omega = _omega(charts["source_nsrr"]["omega"])
    target_omega = _omega(charts["target_nsnsns"]["omega"])
    A, B = matrix[:2, :2], matrix[:2, 2:]
    C, D = matrix[2:, :2], matrix[2:, 2:]
    transported_omega = (A @ source_omega + B) @ np.linalg.inv(
        C @ source_omega + D
    )
    period_residual = float(np.max(np.abs(transported_omega - target_omega)))
    tolerance = float(config["marked_surface"]["period_transport_tolerance"])
    if period_residual > tolerance:
        raise ValueError(
            f"period transport residual {period_residual:.3e} exceeds {tolerance:.3e}"
        )
    for channel in CHANNELS:
        inverse_residual = float(charts[channel]["inverse_period_residual"])
        if inverse_residual > tolerance:
            raise ValueError(
                f"{channel} inverse-period residual {inverse_residual:.3e} "
                f"exceeds {tolerance:.3e}"
            )
    return {
        "source_characteristic": {
            "alpha": list(source_characteristic[0]),
            "beta": list(source_characteristic[1]),
        },
        "source_physical_free_reference_characteristic": {
            "alpha": list(source_reference[0]),
            "beta": list(source_reference[1]),
        },
        "target_characteristic": {
            "alpha": list(target_characteristic[0]),
            "beta": list(target_characteristic[1]),
        },
        "transported_characteristic": {
            "alpha": list(transported_characteristic[0]),
            "beta": list(transported_characteristic[1]),
        },
        "period_transport_residual": period_residual,
    }


def _digest(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _implementation_fingerprint() -> str:
    local_files = (
        CODE_ROOT / "genus_2" / "nsrr_nsnsns_theta_cannon.py",
        CODE_ROOT / "genus_2" / "compare_nsrr_nsnsns_theta.py",
        CODE_ROOT / "genus_2" / "nsrr_plumbing_adapter.py",
        CODE_ROOT / "genus_2" / "audit_nsrr_free_spin_conversion.py",
        CODE_ROOT / "genus_2" / "nsrr_checked_kernel_manifest.json",
        CODE_ROOT / "genus_2" / "physical_free_plumbing_resummation.py",
        CODE_ROOT / "genus_2" / "theta_partition.py",
        CODE_ROOT / "c_Recursion" / "generic_super_liouville_structure_constants.py",
        CODE_ROOT / "c_Recursion" / "ns_genus2_partition.py",
        CODE_ROOT / "full_ramond_block_runtime" / "nsrr_double_virasoro_block.py",
        CODE_ROOT / "full_ramond_block_runtime" / "compute_full_block.py",
        CODE_ROOT / "full_ramond_block_runtime" / "compute_q_expansion.py",
        CODE_ROOT / "ramond_branching_recursion" / "compute_target.py",
        CODE_ROOT / "ramond_branching_recursion" / "direct_state_check.py",
        CODE_ROOT / "double_virasoro" / "nsrr" / "nsrr_genus2_block.py",
        CODE_ROOT / "double_virasoro" / "nsrr" / "ramond_pbw_generalized_ward.py",
        CODE_ROOT / "c_Recursion" / "theta_star_algebra.py",
        CODE_ROOT / "c_Recursion" / "mixed_ns_ramond_descendant_blocks.py",
        CODE_ROOT / "genus_2_cross_channel" / "liouville_torus.py",
        CODE_ROOT / "genus_2_cross_channel" / "free_boson_plumbing.py",
    )
    external_modules = (
        "plumbing.ccy_genus2_block",
        "plumbing.genus2_vacuum_blocks",
        "plumbing.virasoro_plumbing_graph",
    )
    digest = hashlib.sha256()
    for path in local_files:
        digest.update(str(path.relative_to(CODE_ROOT)).encode())
        digest.update(path.read_bytes())
    for module_name in external_modules:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        digest.update(module_name.encode())
        digest.update(path.read_bytes())
    for value in (
        sys.version,
        mpmath.__version__,
        np.__version__,
        scipy.__version__,
        sympy.__version__,
    ):
        digest.update(value.encode())
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "mpmath": mpmath.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
    }


def _cutoff_pairs(config: dict) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (int(item["block_order"]), int(item["quadrature_order"]))
        for item in config["convergence_designs"]
    )
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("convergence designs must be nonempty and unique")
    if any(block < 0 or quadrature <= 0 for block, quadrature in pairs):
        raise ValueError("block orders must be nonnegative and quadrature positive")
    return pairs


def _designs(config: dict) -> list[dict]:
    designs = []
    for block_order, quadrature_order in _cutoff_pairs(config):
        for channel in CHANNELS:
            designs.append(
                {
                    "channel": channel,
                    "block_order": block_order,
                    "quadrature_order": quadrature_order,
                    "node_count": quadrature_order**3,
                }
            )
    return designs


def task_count(config: dict) -> int:
    return sum(int(design["node_count"]) for design in _designs(config))


def decode_task(config: dict, task_index: int) -> tuple[dict, int]:
    remaining = int(task_index)
    for design in _designs(config):
        if remaining < int(design["node_count"]):
            return design, remaining
        remaining -= int(design["node_count"])
    raise IndexError(f"task {task_index} lies outside 0..{task_count(config)-1}")


def channel_task_ranges(config: dict, channel: str) -> tuple[tuple[int, int], ...]:
    """Return inclusive global task ranges for one channel."""

    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    ranges = []
    start = 0
    for design in _designs(config):
        stop = start + int(design["node_count"]) - 1
        if design["channel"] == channel:
            ranges.append((start, stop))
        start = stop + 1
    return tuple(ranges)


def channel_task_indices(config: dict, channel: str) -> tuple[int, ...]:
    """Return global task indices for one channel in deterministic order."""

    return tuple(
        task_index
        for start, stop in channel_task_ranges(config, channel)
        for task_index in range(start, stop + 1)
    )


def channel_chunk_count(config: dict, channel: str, tasks_per_chunk: int) -> int:
    chunk_size = int(tasks_per_chunk)
    if chunk_size <= 0:
        raise ValueError("tasks_per_chunk must be positive")
    return math.ceil(len(channel_task_indices(config, channel)) / chunk_size)


def _q_values(config: dict, channel: str) -> tuple[complex, complex, complex]:
    return _complex_triple(
        config["marked_surface"]["charts"][channel]["q_values"]
    )


def _node_data(config: dict, design: dict, node_index: int):
    order = int(design["quadrature_order"])
    q_values = _q_values(config, str(design["channel"]))
    rules = _rules(q_values, order)
    i0, remainder = divmod(int(node_index), order * order)
    i1, i2 = divmod(remainder, order)
    indices = (i0, i1, i2)
    momenta = tuple(float(rules[edge][0][indices[edge]]) for edge in range(3))
    measure = _measure(rules, indices)
    return q_values, indices, momenta, measure


def evaluate_task(config: dict, task_index: int) -> dict:
    design, node_index = decode_task(config, task_index)
    channel = str(design["channel"])
    q_values, indices, momenta, measure = _node_data(config, design, node_index)
    parameters = config["parameters"]
    numerics = config["numerics"]
    charts = config["marked_surface"]["charts"]
    b = float(parameters["b"])
    constants = GenericSuperLiouvilleConstants(
        b,
        dps=int(numerics["structure_precision"]),
        mu=complex(parameters["mu"]),
        include_cosmological_prefactor=bool(
            parameters["include_cosmological_prefactor"]
        ),
    )
    started = time.perf_counter()
    if channel == "source_nsrr":
        sectors, ward_residual = nsrr_node(
            b=b,
            q_values=q_values,
            lifts=tuple(int(value) for value in charts[channel]["lifts"]),
            block_order=int(design["block_order"]),
            momenta=momenta,
            measure=measure,
            constants=constants,
            branching_mp_dps=int(numerics["branching_mp_dps"]),
        )
        ward_limit = float(numerics["maximum_ward_residual"])
        if ward_residual > ward_limit:
            raise ArithmeticError(
                f"NSRR branching Ward residual {ward_residual:.3e} "
                f"exceeds configured limit {ward_limit:.3e}"
            )
        block_method = (
            "NSRR branching-coefficient Ward recursion followed by the "
            "product of two ordinary genus-two Virasoro c-recursions"
        )
    elif channel == "target_nsnsns":
        recursion = NSGenus2CRecursion(
            channel="theta",
            q_values=q_values,
            global_method=str(numerics["global_method"]),
            global_tolerance=float(numerics["global_tolerance"]),
            global_max_total_occupation=int(
                numerics["global_max_total_occupation"]
            ),
            vacuum_word_length=int(numerics["vacuum_word_length"]),
            vacuum_max_mode=int(numerics["vacuum_max_mode"]),
        )
        sectors = all_ns_node(
            b=b,
            q_values=q_values,
            lifts=tuple(int(value) for value in charts[channel]["lifts"]),
            recursion_order=int(design["block_order"]),
            momenta=momenta,
            measure=measure,
            constants=constants,
            recursion=recursion,
            block_method=str(numerics["all_ns_block_method"]),
            block_working_precision=int(numerics["block_working_precision"]),
        )
        ward_residual = None
        block_method = str(numerics["all_ns_block_method"])
    else:  # pragma: no cover
        raise AssertionError(f"unknown channel {channel}")
    return {
        "schema": SCHEMA,
        "task_index": int(task_index),
        "node_index": int(node_index),
        **design,
        "indices": list(indices),
        "momenta": list(momenta),
        "measure": float(measure),
        "sector_contributions": [float(value) for value in sectors],
        "contribution": math.fsum(float(value) for value in sectors),
        "maximum_ward_residual": ward_residual,
        "block_method": block_method,
        "source_algorithm_ledger": (
            {
                "branching_coefficients": (
                    "finite-cutoff Ward recursion with certified low-state anchors"
                ),
                "double_virasoro_factors": (
                    "ordinary genus-two Virasoro c-recursion for each factor"
                ),
                "direct_pbw_genus_two_block_used": False,
                "maximum_allowed_ward_residual": ward_limit,
            }
            if channel == "source_nsrr"
            else None
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def _validate_shard(
    config: dict,
    task_index: int,
    shard: dict,
    implementation_fingerprint: str,
) -> None:
    design, node_index = decode_task(config, task_index)
    prefix = f"task-{task_index:06d}"
    if shard.get("schema") != SCHEMA:
        raise RuntimeError(f"schema mismatch in {prefix}")
    if int(shard.get("task_index", -1)) != task_index:
        raise RuntimeError(f"task index mismatch in {prefix}")
    if int(shard.get("node_index", -1)) != node_index:
        raise RuntimeError(f"node index mismatch in {prefix}")
    if shard.get("config_digest") != _digest(config):
        raise RuntimeError(f"config digest mismatch in {prefix}")
    if shard.get("implementation_fingerprint") != implementation_fingerprint:
        raise RuntimeError(f"implementation fingerprint mismatch in {prefix}")
    for key, expected in design.items():
        if shard.get(key) != expected:
            raise RuntimeError(
                f"decoded design mismatch for {key} in {prefix}: "
                f"expected {expected!r}, got {shard.get(key)!r}"
            )
    _, indices, momenta, measure = _node_data(config, design, node_index)
    if tuple(int(value) for value in shard.get("indices", ())) != indices:
        raise RuntimeError(f"quadrature index mismatch in {prefix}")
    observed_momenta = tuple(float(value) for value in shard.get("momenta", ()))
    if len(observed_momenta) != 3 or any(
        not math.isclose(left, right, rel_tol=2.0e-14, abs_tol=1.0e-15)
        for left, right in zip(observed_momenta, momenta)
    ):
        raise RuntimeError(f"momentum mismatch in {prefix}")
    if not math.isclose(
        float(shard.get("measure", math.nan)),
        measure,
        rel_tol=2.0e-14,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(f"measure mismatch in {prefix}")
    sectors = shard.get("sector_contributions", ())
    if len(sectors) != 2 or not math.isclose(
        math.fsum(float(value) for value in sectors),
        float(shard.get("contribution", math.nan)),
        rel_tol=2.0e-15,
        abs_tol=1.0e-300,
    ):
        raise RuntimeError(f"sector reduction mismatch in {prefix}")


def worker(config_path: Path, output_dir: Path, task_index: int, force: bool) -> Path:
    config = _load(config_path)
    _validate_config(config)
    config_digest = _digest(config)
    implementation = _implementation_fingerprint()
    path = output_dir / f"task-{int(task_index):06d}.json"
    if path.exists() and not force:
        existing = _load(path)
        _validate_shard(config, int(task_index), existing, implementation)
        return path
    result = evaluate_task(config, int(task_index))
    result["config_digest"] = config_digest
    result["implementation_fingerprint"] = implementation
    result["runtime_versions"] = _runtime_versions()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def channel_worker(
    config_path: Path,
    output_dir: Path,
    channel: str,
    chunk_index: int,
    tasks_per_chunk: int,
) -> dict:
    """Evaluate one restartable contiguous chunk of a channel's task list."""

    config = _load(config_path)
    _validate_config(config)
    task_indices = channel_task_indices(config, channel)
    chunk_size = int(tasks_per_chunk)
    start = int(chunk_index) * chunk_size
    stop = min(start + chunk_size, len(task_indices))
    if chunk_size <= 0 or start < 0 or start >= len(task_indices):
        raise IndexError(
            f"chunk {chunk_index} is outside channel {channel} with "
            f"{channel_chunk_count(config, channel, chunk_size)} chunks"
        )
    completed = []
    for task_index in task_indices[start:stop]:
        completed.append(str(worker(config_path, output_dir, task_index, False)))
    return {
        "channel": channel,
        "chunk_index": int(chunk_index),
        "tasks_per_chunk": chunk_size,
        "first_global_task": task_indices[start],
        "last_global_task": task_indices[stop - 1],
        "completed_shards": len(completed),
    }


def reduce(config_path: Path, shard_dir: Path, output: Path) -> dict:
    config = _load(config_path)
    geometry_validation = _validate_config(config)
    implementation = _implementation_fingerprint()
    count = task_count(config)
    expected_names = {f"task-{index:06d}.json" for index in range(count)}
    observed_names = {path.name for path in shard_dir.glob("task-*.json")}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise RuntimeError(
            f"shard set mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    shards = []
    for task_index in range(count):
        shard = _load(shard_dir / f"task-{task_index:06d}.json")
        _validate_shard(config, task_index, shard, implementation)
        shards.append(shard)

    parameters = config["parameters"]
    b = float(parameters["b"])
    q_background = b + 1 / b
    central_charge = 1.5 + 3 * q_background * q_background
    kappa = central_charge / 1.5
    source_free, target_free, spin_change_ratio = same_frame_free_factors(
        int(config["numerics"]["free_max_mode"])
    )
    free = {
        "source_nsrr": source_free,
        "target_nsnsns": target_free,
        "source_majorana_spin_change_ratio": spin_change_ratio,
        "power_kappa": kappa,
    }

    rows = []
    for design in _designs(config):
        selected = [
            shard
            for shard in shards
            if shard["channel"] == design["channel"]
            and int(shard["block_order"]) == int(design["block_order"])
            and int(shard["quadrature_order"])
            == int(design["quadrature_order"])
        ]
        selected.sort(key=lambda shard: int(shard["node_index"]))
        if len(selected) != int(design["node_count"]):
            raise RuntimeError(f"incomplete design {design}")
        sector_values = tuple(
            math.fsum(float(shard["sector_contributions"][sector]) for shard in selected)
            for sector in (0, 1)
        )
        partition = math.fsum(sector_values)
        denominator = source_free if design["channel"] == "source_nsrr" else target_free
        ward_values = [
            float(shard["maximum_ward_residual"])
            for shard in selected
            if shard["maximum_ward_residual"] is not None
        ]
        rows.append(
            {
                **design,
                "sector_values": list(sector_values),
                "z_super_liouville": partition,
                "z_free_superfield": denominator,
                "q_observable": partition / denominator**kappa,
                "runtime_seconds_sum": math.fsum(
                    float(shard["runtime_seconds"]) for shard in selected
                ),
                "maximum_ward_residual": max(ward_values) if ward_values else None,
            }
        )

    comparisons = []
    for block_order, quadrature_order in _cutoff_pairs(config):
        pair = {
            row["channel"]: row
            for row in rows
            if row["block_order"] == block_order
            and row["quadrature_order"] == quadrature_order
        }
        source = float(pair["source_nsrr"]["q_observable"])
        target = float(pair["target_nsnsns"]["q_observable"])
        comparisons.append(
            {
                "block_order": block_order,
                "quadrature_order": quadrature_order,
                "source_nsrr_over_target_nsnsns": source / target,
                "relative_difference": source / target - 1.0,
                "symmetric_relative_difference": 2 * (source - target) / (source + target),
            }
        )

    cosmological_pants = GenericSuperLiouvilleConstants(
        b,
        dps=int(config["numerics"]["structure_precision"]),
        mu=complex(parameters["mu"]),
        include_cosmological_prefactor=bool(
            parameters["include_cosmological_prefactor"]
        ),
    ).cosmological_three_point_factor()
    summary = {
        "schema": SCHEMA,
        "calculation": "generic-b NSRR theta / NSNSNS theta modular check",
        "quantity": "Q = Z_superLiouville / Z_(X+psi)^kappa",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_digest": _digest(config),
        "implementation_fingerprint": implementation,
        "runtime_versions": _runtime_versions(),
        "config": config,
        "task_count": count,
        "geometry_validation": geometry_validation,
        "parameters": {
            "b": b,
            "Q_background": q_background,
            "c_super_liouville": central_charge,
            "weyl_cancelling_free_superfield_power": kappa,
            "one_pants_cosmological_factor": [
                float(cosmological_pants.real),
                float(cosmological_pants.imag),
            ],
        },
        "free_superfield_same_local_frame": free,
        "rows": rows,
        "comparisons": comparisons,
        "convention_ledger": {
            "all_NS_three_point": "C_HN(0)=C_BRY; C_HN(1)=i*tilde_C_BRY exactly once",
            "NSRR_HJS_map": "eta=+ -> C_even; eta=- -> C_odd; HJS completeness 1/2",
            "NSRR_internal_R_ground_completeness": RAMOND_GROUND_COMPLETENESS,
            "source_branching_coefficients": "Ward/branching recursion on the finite order-eight lattice, anchored only by certified low states",
            "source_double_virasoro_block": "product of two ordinary genus-two Virasoro blocks, each computed by c-recursion at total order 8",
            "source_direct_PBW_genus_two_block_used": False,
            "target_block": "collision-aware multiprecision N=1 genus-two c-recursion at total order 8",
            "generic_b_leg_normalization": "reflection-symmetric NS/R metrics with common b^(-3) per pants",
            "free_denominator": "physical free scalar+Majorana in each numerator's own theta plumbing frame",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--task-count-only", action="store_true")
    ranges_parser = subparsers.add_parser("channel-ranges")
    ranges_parser.add_argument("--channel", choices=CHANNELS, required=True)
    chunk_count_parser = subparsers.add_parser("channel-chunk-count")
    chunk_count_parser.add_argument("--channel", choices=CHANNELS, required=True)
    chunk_count_parser.add_argument("--tasks-per-chunk", type=int, required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    worker_parser.add_argument("--force", action="store_true")
    channel_worker_parser = subparsers.add_parser("channel-worker")
    channel_worker_parser.add_argument("--output-dir", type=Path, required=True)
    channel_worker_parser.add_argument("--channel", choices=CHANNELS, required=True)
    channel_worker_parser.add_argument("--chunk-index", type=int, required=True)
    channel_worker_parser.add_argument("--tasks-per-chunk", type=int, required=True)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--shard-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = _load(args.config)
    geometry = _validate_config(config)
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
                        "geometry_validation": geometry,
                    },
                    indent=2,
                )
            )
        return 0
    if args.command == "channel-ranges":
        for start, stop in channel_task_ranges(config, args.channel):
            print(f"{start}-{stop}")
        return 0
    if args.command == "channel-chunk-count":
        print(channel_chunk_count(config, args.channel, args.tasks_per_chunk))
        return 0
    if args.command == "worker":
        print(worker(args.config, args.output_dir, args.task_index, args.force))
        return 0
    if args.command == "channel-worker":
        print(
            json.dumps(
                channel_worker(
                    args.config,
                    args.output_dir,
                    args.channel,
                    args.chunk_index,
                    args.tasks_per_chunk,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "reduce":
        summary = reduce(args.config, args.shard_dir, args.output)
        print(json.dumps({"output": str(args.output), "comparisons": summary["comparisons"]}, indent=2))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
