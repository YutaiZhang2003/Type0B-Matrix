#!/usr/bin/env python3
"""Raise only the Virasoro descendant cutoff in saved NSRR source nodes.

The saved source integrand is retained in full.  At the same momentum node,
the equal-HJS-sign double-Virasoro contribution is evaluated with a fixed
branching cutoff at two descendant cutoffs.  Only their difference is added
to the saved integrand.  Consequently the momentum rule, mixed-sign block,
three-point coefficients, spin projection, free factor, and target channel
are unchanged.
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

import nsrr_branching_cutoff_probe as probe
import nsrr_factorized_sign_trial as trial
import run_nsrr_nsnsns_offaxis_constant_scan as scan
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants
from nsrr_genus2_block import auxiliary_majorana_nsrr_series
from physical_nsrr_sewing import SOURCE_FIXED_SPIN_LIFTS


SCHEMA = "nsrr-virasoro-c-recursion-refinement-v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(probe.__file__),
        Path(probe.reduced_virasoro_series.__code__.co_filename),
    )
    answer = {}
    for path in paths:
        resolved = path.resolve()
        answer[str(resolved.relative_to(ROOT))] = file_digest(resolved)
    return answer


def auxiliary_values(q_geometry: tuple[complex, complex, complex], cutoff: int) -> list[complex]:
    series = auxiliary_majorana_nsrr_series(maximum_total_twice_level=2 * cutoff)
    return probe.evaluate_components(series, q_geometry[::-1])


def prepare(parent_output: Path, output_dir: Path, *, low: int, high: int, branch: int) -> dict:
    parent_output = parent_output.resolve()
    parent_config = load(parent_output / "config.json")
    parent_summary = load(parent_output / "summary.json")
    scan.validate_config(parent_config)
    if len(parent_config["orders"]) != 1:
        raise ValueError("the refinement requires one momentum order per parent dataset")
    if not 0 <= low < high or branch < 0:
        raise ValueError("require 0 <= low < high and a nonnegative branching cutoff")
    if branch != int(parent_config["source_level"]):
        raise ValueError("branching cutoff must equal the saved source block level")
    if {row["point_id"] for row in parent_summary["comparisons"]} != {
        point["point_id"] for point in parent_config["points"]
    }:
        raise ValueError("parent summary point coverage is incomplete")

    auxiliary_cutoff = 16
    point_auxiliary = {}
    maximum_auxiliary_change = 0.0
    for point in parent_config["points"]:
        q = tuple(complex(value) for value in point["source"]["q_values"])
        fine = auxiliary_values(q, auxiliary_cutoff)
        coarse = auxiliary_values(q, auxiliary_cutoff - 2)
        error = max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(fine, coarse))
        maximum_auxiliary_change = max(maximum_auxiliary_change, error)
        point_auxiliary[point["point_id"]] = [trial.encode(value) for value in fine]
    if maximum_auxiliary_change > 1.0e-10:
        raise ArithmeticError("auxiliary quotient is not converged")

    config = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_output": str(parent_output),
        "parent_config_digest": digest(parent_config),
        "parent_summary_digest": digest(parent_summary),
        "b": parent_config["b"],
        "kappa": parent_config["kappa"],
        "quadrature_order": parent_config["orders"][0],
        "point_ids": [point["point_id"] for point in parent_config["points"]],
        "branching_cutoff": branch,
        "virasoro_descendant_cutoffs": [low, high],
        "auxiliary_cutoff": auxiliary_cutoff,
        "auxiliary_values": point_auxiliary,
        "maximum_auxiliary_L14_to_L16_scaled_change": maximum_auxiliary_change,
        "source_update": (
            "saved full source node plus the equal-HJS-sign D_high minus D_low "
            "double-Virasoro correction at fixed branching cutoff"
        ),
        "fixed_inputs": (
            "momentum nodes and measures; mixed-sign block; three-point coefficients; "
            "[11|00] lift projection; unscaled Human M contraction; free factors; target Q"
        ),
        "virasoro_backend": "CCY genus-two c-recursion only",
        "normalization_policy": parent_config["normalization_policy"],
        "implementation_fingerprint": implementation_fingerprint(),
        "protected_kernel_hashes": trial.protected_hashes(),
    }
    validate(config)
    save(output_dir / "config.json", config)
    return config


def validate(config: dict) -> tuple[dict, dict]:
    if config.get("schema") != SCHEMA:
        raise ValueError("wrong c-recursion refinement schema")
    parent = Path(config["parent_output"])
    parent_config = load(parent / "config.json")
    parent_summary = load(parent / "summary.json")
    scan.validate_config(parent_config)
    if digest(parent_config) != config["parent_config_digest"]:
        raise ValueError("parent config changed")
    if digest(parent_summary) != config["parent_summary_digest"]:
        raise ValueError("parent summary changed")
    if config["implementation_fingerprint"] != implementation_fingerprint():
        raise ValueError("refinement implementation changed")
    if config["protected_kernel_hashes"] != trial.protected_hashes():
        raise ValueError("protected kernel changed")
    if config["virasoro_backend"] != "CCY genus-two c-recursion only":
        raise ValueError("a non-c-recursive Virasoro backend was requested")
    if config["point_ids"] != [point["point_id"] for point in parent_config["points"]]:
        raise ValueError("point order differs from the parent")
    if config["quadrature_order"] != parent_config["orders"][0]:
        raise ValueError("momentum order differs from the parent")
    return parent_config, parent_summary


def projected_equal_contribution(
    *,
    b: float,
    momenta_geometry: tuple[float, float, float],
    q_geometry: tuple[complex, complex, complex],
    shells: dict,
    auxiliary: list[complex],
    cutoff: int,
    branching_cutoff: int,
    coefficients: dict[int, complex],
) -> tuple[float, float]:
    q_slots = q_geometry[::-1]
    amplitudes: dict[tuple[int, int], complex] = {}
    maximum_leakage = 0.0
    for eta in (1, -1):
        even_enlarged = probe.cumulative_vector(shells[eta, cutoff], branching_cutoff)
        for form_parity, enlarged in (
            (0, even_enlarged),
            (1, probe.odd_partner(even_enlarged)),
        ):
            components, leakage = probe.supported_quotient(enlarged, auxiliary)
            maximum_leakage = max(maximum_leakage, leakage)
            lift_blocks = []
            for lift_geometry in SOURCE_FIXED_SPIN_LIFTS:
                character = trial.dv.spin_character_index(lift_geometry[::-1])
                lift_blocks.append(trial.fwht(components)[character])
            primary = trial.NSRRPlumbingInputs(
                q_geometry,
                SOURCE_FIXED_SPIN_LIFTS[0],
                trial.GEOMETRY_SECTORS,
            ).primary(b, momenta_geometry)
            amplitudes[form_parity, eta] = primary * sum(lift_blocks) / math.sqrt(2.0)

    total = 0.0
    for eta in (1, -1):
        weight = coefficients[eta] * coefficients[eta]
        if abs(weight.imag) > 1.0e-10 * max(1.0, abs(weight)):
            raise ArithmeticError("real-momentum equal-sign structure weight is complex")
        physical = amplitudes[0, eta] + 1.0j * amplitudes[1, eta]
        total += float(weight.real) * abs(physical) ** 2
    return total, maximum_leakage


def evaluate_node(config_path: Path, output_dir: Path, task_index: int) -> dict:
    config = load(config_path)
    parent_config, _ = validate(config)
    parent_dir = Path(config["parent_output"])
    parent_shard = load(parent_dir / "source" / "shards" / f"node-{task_index:03d}.json")
    scan.validate_shard(parent_config, "source", task_index, parent_shard)
    _, node, momenta_geometry_raw, measure = scan.node_data(parent_config, "source", task_index)
    momenta_geometry = tuple(momenta_geometry_raw)
    if parent_shard["momenta"] != list(momenta_geometry) or parent_shard["measure"] != measure:
        raise ValueError("parent momentum node changed")

    low, high = config["virasoro_descendant_cutoffs"]
    branch = config["branching_cutoff"]
    momenta_slots = momenta_geometry[::-1]
    cache_dir = output_dir / "actions_cache"
    grid, raw, labels, ward_error = probe.branch_data(
        config["b"], momenta_slots, branch, cache_dir=cache_dir
    )
    products = probe.make_products(grid, labels, high)
    constants = GenericSuperLiouvilleConstants(config["b"], dps=30)
    bry = constants.rr_ns_constants(
        momenta_geometry[1], momenta_geometry[0], momenta_geometry[2]
    )
    coefficients = {1: complex(bry[0]) / 2.0, -1: complex(bry[1]) / 2.0}

    parent_rows = {row["point_id"]: row for row in parent_shard["values"]}
    values = []
    maximum_leakage = 0.0
    for point in parent_config["points"]:
        point_id = point["point_id"]
        q_geometry = tuple(complex(value) for value in point["source"]["q_values"])
        shells = probe.numerical_shells(
            config["b"],
            momenta_slots,
            raw,
            products,
            q_geometry[::-1],
            (low, high),
        )
        auxiliary = [trial.decode(value) for value in config["auxiliary_values"][point_id]]
        equal_low, leakage_low = projected_equal_contribution(
            b=config["b"],
            momenta_geometry=momenta_geometry,
            q_geometry=q_geometry,
            shells=shells,
            auxiliary=auxiliary,
            cutoff=low,
            branching_cutoff=branch,
            coefficients=coefficients,
        )
        equal_high, leakage_high = projected_equal_contribution(
            b=config["b"],
            momenta_geometry=momenta_geometry,
            q_geometry=q_geometry,
            shells=shells,
            auxiliary=auxiliary,
            cutoff=high,
            branching_cutoff=branch,
            coefficients=coefficients,
        )
        maximum_leakage = max(maximum_leakage, leakage_low, leakage_high)
        saved = float(parent_rows[point_id]["source_Z_unscaled_M_node"])
        values.append(
            {
                "point_id": point_id,
                "saved_source_node": saved,
                "equal_sign_low": equal_low,
                "equal_sign_high": equal_high,
                "c_recursion_correction": equal_high - equal_low,
                "refined_source_node": saved + equal_high - equal_low,
            }
        )
    return {
        "schema": SCHEMA,
        "config_digest": digest(config),
        "task_index": task_index,
        "node": node,
        "momenta": list(momenta_geometry),
        "measure": measure,
        "branch_triple_count": len(labels),
        "maximum_branching_ward_residual": ward_error,
        "maximum_unsupported_character_leakage": maximum_leakage,
        "values": values,
    }


def worker(config_path: Path, output_dir: Path, task_index: int) -> None:
    result = evaluate_node(config_path, output_dir, task_index)
    save(output_dir / "shards" / f"node-{task_index:03d}.json", result)


def reduce(config_path: Path, output_dir: Path) -> dict:
    config = load(config_path)
    parent_config, parent_summary = validate(config)
    count = config["quadrature_order"] ** 3
    shards = []
    for index in range(count):
        shard = load(output_dir / "shards" / f"node-{index:03d}.json")
        if shard["schema"] != SCHEMA or shard["config_digest"] != digest(config):
            raise ValueError(f"incompatible shard {index}")
        if shard["task_index"] != index:
            raise ValueError(f"mislabelled shard {index}")
        shards.append(shard)

    parent_rows = {row["point_id"]: row for row in parent_summary["comparisons"]}
    comparisons = []
    for point_index, point in enumerate(parent_config["points"]):
        point_id = point["point_id"]
        saved_z = math.fsum(
            shard["measure"] * shard["values"][point_index]["saved_source_node"]
            for shard in shards
        )
        correction = math.fsum(
            shard["measure"] * shard["values"][point_index]["c_recursion_correction"]
            for shard in shards
        )
        refined_z = saved_z + correction
        source_q = refined_z / float(point["source"]["Z_free"]) ** config["kappa"]
        target_q = float(parent_rows[point_id]["target_Q"])
        comparisons.append(
            {
                "point_id": point_id,
                "saved_source_Z": saved_z,
                "c_recursion_correction_Z": correction,
                "refined_source_Z": refined_z,
                "saved_source_Q": float(parent_rows[point_id]["source_Q"]),
                "refined_source_Q": source_q,
                "target_Q_reused": target_q,
                "source_relative_change": source_q / float(parent_rows[point_id]["source_Q"]) - 1.0,
                "refined_raw_ratio": source_q / target_q,
                "refined_residual_from_one": source_q / target_q - 1.0,
            }
        )
    residuals = [row["refined_residual_from_one"] for row in comparisons]
    result = {
        "schema": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "complete_nodes": count,
        "comparisons": comparisons,
        "fixed_normalization_one": {
            "normalization": 1.0,
            "normalization_fitted_or_applied": False,
            "rms_fractional_residual": math.sqrt(
                math.fsum(value * value for value in residuals) / len(residuals)
            ),
            "maximum_absolute_fractional_residual": max(map(abs, residuals)),
            "minimum_raw_ratio": min(row["refined_raw_ratio"] for row in comparisons),
            "maximum_raw_ratio": max(row["refined_raw_ratio"] for row in comparisons),
        },
        "maximum_source_relative_change": max(
            abs(row["source_relative_change"]) for row in comparisons
        ),
        "maximum_branching_ward_residual": max(
            shard["maximum_branching_ward_residual"] for shard in shards
        ),
        "maximum_unsupported_character_leakage": max(
            shard["maximum_unsupported_character_leakage"] for shard in shards
        ),
    }
    save(output_dir / "summary.json", result)
    return result


def run(config_path: Path, output_dir: Path, workers: int) -> dict:
    config = load(config_path)
    validate(config)
    count = config["quadrature_order"] ** 3
    shard_dir = output_dir / "shards"
    log_dir = output_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def execute(index: int) -> int:
        path = shard_dir / f"node-{index:03d}.json"
        if path.exists():
            shard = load(path)
            if shard.get("config_digest") != digest(config) or shard.get("task_index") != index:
                raise ValueError(f"invalid saved shard {index}")
            return index
        with (log_dir / f"node-{index:03d}.log").open("a", encoding="utf-8") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--output-dir",
                    str(output_dir),
                    "--task-index",
                    str(index),
                ],
                check=True,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        return index

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            print(
                f"{completed}/{count} complete; last={future.result()}; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    return reduce(config_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--parent-output", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--low", type=int, default=3)
    prepare_parser.add_argument("--high", type=int, default=5)
    prepare_parser.add_argument("--branch", type=int, default=3)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=4)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        config = prepare(
            args.parent_output,
            args.output_dir,
            low=args.low,
            high=args.high,
            branch=args.branch,
        )
        print(json.dumps(config, indent=2))
    elif args.command == "worker":
        worker(args.output_dir / "config.json", args.output_dir, args.task_index)
    elif args.command == "run":
        result = run(args.output_dir / "config.json", args.output_dir, args.workers)
        check = result["fixed_normalization_one"]
        print(
            f"fixed A=1 rms={check['rms_fractional_residual']:.3%} "
            f"max={check['maximum_absolute_fractional_residual']:.3%}; "
            f"max source change={result['maximum_source_relative_change']:.3%}"
        )
    else:
        result = reduce(args.output_dir / "config.json", args.output_dir)
        print(json.dumps(result["fixed_normalization_one"], indent=2))


if __name__ == "__main__":
    main()
