#!/usr/bin/env python3
"""Plan, execute, and reduce the order-eight Type-0B five-point RQMC array."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

from evaluate_type0b_ns_five_tachyon_physical_i_epsilon import (
    main as evaluate_main,
)


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    schema = config.get("schema")
    if schema not in (
        "type0b-ns-fivepoint-order8-coefficient-table-subtraction-v4",
        "type0b-ns-fivepoint-order8-c-recursion-subtraction-v5",
    ):
        raise ValueError("unexpected cluster config schema")
    recursion = config["recursion"]
    subtraction = config["subtraction"]
    if recursion["global_max_twice_levels"] != [8, 8]:
        raise ValueError("this production bundle is fixed at recursion order (8,8)")
    if int(recursion["global_max_total_twice_level"]) < 16:
        raise ValueError("full rectangular order (8,8) requires total twice-level 16")
    backend = recursion["block_backend"]
    if backend != "c":
        raise ValueError("amplitude production requires all-c recursion; h/hybrid bundles are retired")
    if schema != "type0b-ns-fivepoint-order8-c-recursion-subtraction-v5":
        raise ValueError("pure c-recursion requires the v5 production schema")
    if recursion.get("h_recursion_role") != "disabled":
        raise ValueError("h-recursion must be disabled in the c-only bundle")
    if recursion.get("c_recursion_role") != "production_in_every_selected_chart":
        raise ValueError("c-recursion must be the all-chart production backend")
    if "self_dual_regulator" in config:
        raise ValueError("the c-only bundle must not define an h regulator")
    atlas = config.get("atlas", {})
    if int(atlas.get("oriented_linear_charts", 0)) != 120:
        raise ValueError("the production bundle requires the 120-chart atlas")
    if int(atlas.get("unoriented_trivalent_trees", 0)) != 15:
        raise ValueError("the production bundle requires all 15 five-point trees")
    if subtraction["scheme"] != "pointwise_F-P1-P2+P12_plus_analytic_forest":
        raise ValueError("the cluster job requires the consistent local forest")
    radii = tuple(float(value) for value in subtraction["collar_radii"])
    if len(radii) < 3 or any(not 0.0 < value <= 0.01 for value in radii):
        raise ValueError("all production collar radii must lie in (0,0.01]")
    shard_count = int(config["array"]["shards"])
    if shard_count < 2:
        raise ValueError("at least two shards per regulator-radius point are required")
    corner_count = int(
        config["array"].get(
            "deterministic_corner_shards", 0
        )
    )
    if corner_count != 1:
        raise ValueError("exactly one shard per regulator-radius point needs the corner")
    if int(config["qmc"]["replicates_per_shard"]) < 2:
        raise ValueError("each shard must contain at least two RQMC replicates")
    certificate = config.get("collar_certificate", {})
    audit_count = int(certificate.get("audit_shards", 0))
    if not 1 <= audit_count <= shard_count:
        raise ValueError("each regulator-radius point requires a certificate shard")
    if certificate.get("reference_backend") != "c":
        raise ValueError("the collar overlap certificate must use c-recursion")
    if not isinstance(certificate.get("enforce"), bool):
        raise ValueError("the collar certificate enforce policy must be boolean")
    if certificate.get("reference_max_twice_levels") != [8, 8]:
        raise ValueError("the c-recursion collar check is fixed at edge order (8,8)")
    expected_reference_total = 16 if backend == "c" else 8
    if int(certificate.get("reference_max_total_twice_level", -1)) != expected_reference_total:
        raise ValueError("the c-recursion collar check has the wrong total cutoff")
    if certificate.get("previous_reference_max_twice_levels") != [6, 6]:
        raise ValueError("the preceding c-recursion check is fixed at edge order (6,6)")
    expected_previous_total = 12 if backend == "c" else 6
    if int(certificate.get("previous_reference_max_total_twice_level", -1)) != expected_previous_total:
        raise ValueError("the preceding c-recursion check has the wrong total cutoff")
    compatible_hashes = config.get("merge", {}).get(
        "compatible_shard_config_sha256", {}
    )
    if not isinstance(compatible_hashes, dict):
        raise ValueError("compatible shard hashes must be an index-to-hash mapping")
    for key, value in compatible_hashes.items():
        try:
            shard_index = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("compatible shard hash keys must be integer indices") from error
        if shard_index <= 0 or shard_index >= shard_count:
            raise ValueError("only non-audit shard hashes may be merge-compatible")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("compatible shard hashes must be lowercase SHA-256 values")
    expected_tasks = shard_count
    if int(config["array"].get("task_count", -1)) != expected_tasks:
        raise ValueError("configured task_count does not match the regulator array")
    return config


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tasks(config: dict[str, object]) -> tuple[dict[str, object], ...]:
    shard_count = int(config["array"]["shards"])
    base_seed = int(config["array"]["base_seed"])
    return tuple(
        {
            "task_index": shard_index,
            "shard_index": shard_index,
            "central_charge_shift": 0.0,
            # Every collar is evaluated inside this worker, so one seed couples
            # all collar-stability differences exactly.
            "seed": base_seed + shard_index,
        }
        for shard_index in range(shard_count)
    )


def _worker_arguments(
    config: dict[str, object], task: dict[str, object], output: Path
) -> list[str]:
    physics = config["physics"]
    recursion = config["recursion"]
    momentum = config["momentum"]
    precision = config["precision"]
    qmc = config["qmc"]
    certificate = config["collar_certificate"]
    radii = tuple(float(value) for value in config["subtraction"]["collar_radii"])
    arguments = [
        "--energies",
        *(str(value) for value in physics["real_outgoing_energies"]),
        "--epsilon",
        str(physics["epsilon"]),
        "--epsilon-weights",
        *(str(value) for value in physics["epsilon_weights"]),
        "--block-backend",
        str(recursion["block_backend"]),
        "--recursion-max-twice-level",
        "-1",
        "--global-max-twice-levels",
        *(str(value) for value in recursion["global_max_twice_levels"]),
        "--global-max-total-twice-level",
        str(recursion["global_max_total_twice_level"]),
        "--momentum-orders",
        *(str(value) for value in momentum["orders"]),
        "--momentum-maximum",
        str(momentum["maximum"]),
        "--momentum-refinement-shells",
        str(momentum["refinement_shells"]),
        "--structure-precision",
        str(precision["structure_digits"]),
        "--central-charge-shift",
        str(task["central_charge_shift"]),
        "--block-working-precision",
        str(precision["block_digits"]),
        "--collar-radii",
        *(str(value) for value in radii),
        "--projection-radius",
        str(config["subtraction"]["projection_radius"]),
        "--bulk-sobol-power",
        str(qmc["bulk_sobol_power"]),
        "--face-sobol-power",
        str(qmc["face_sobol_power"]),
        "--replicates",
        str(qmc["replicates_per_shard"]),
        "--radial-power",
        str(qmc["radial_power"]),
        "--seed",
        str(task["seed"]),
        "--output",
        str(output),
    ]
    if recursion["block_backend"] == "h":
        regulator = config["self_dual_regulator"]
        insertion = arguments.index("--block-working-precision")
        arguments[insertion:insertion] = [
            "--h-regulator-etas",
            *(str(value) for value in regulator["eta_values"]),
            "--h-regulator-polynomial-degree",
            str(regulator["polynomial_degree"]),
            "--h-regulator-comparison-degree",
            str(regulator["comparison_degree"]),
            "--include-comparison-fit",
        ]
    if int(task["shard_index"]) < int(
        certificate["audit_shards"]
    ):
        arguments.extend(
            [
                "--face-collar-relative-tolerance",
                str(certificate["relative_tolerance"]),
                "--face-collar-absolute-tolerance",
                str(certificate["absolute_tolerance"]),
                "--face-collar-samples-per-orbit",
                str(certificate["samples_per_orbit"]),
                "--face-collar-normal-angle-count",
                str(certificate["normal_angle_count"]),
                "--face-collar-reference-backend",
                str(certificate["reference_backend"]),
                "--face-collar-reference-max-twice-levels",
                *(str(value) for value in certificate["reference_max_twice_levels"]),
                "--face-collar-reference-max-total-twice-level",
                str(certificate["reference_max_total_twice_level"]),
                "--face-collar-previous-reference-max-twice-levels",
                *(
                    str(value)
                    for value in certificate["previous_reference_max_twice_levels"]
                ),
                "--face-collar-previous-reference-max-total-twice-level",
                str(certificate["previous_reference_max_total_twice_level"]),
                "--face-collar-reference-convergence-relative-tolerance",
                str(certificate["reference_convergence_relative_tolerance"]),
            ]
        )
        if bool(certificate["enforce"]):
            arguments.append("--enforce-face-collar-certificate")
    else:
        arguments.append("--skip-face-collar-diagnostic")
    if int(task["shard_index"]) >= int(
        config["array"]["deterministic_corner_shards"]
    ):
        arguments.append("--skip-corner-contribution")
    runtime = config.get("runtime", {})
    arguments.extend([
        "--block-cache-limit", str(runtime.get("block_cache_limit", 2048)),
        "--auxiliary-cache-limit", str(runtime.get("auxiliary_cache_limit", 4096)),
        "--c-coefficient-cache", str(output.with_suffix(".coefficients.sqlite")),
        "--checkpoint-directory", str(output.with_suffix(".checkpoints")),
    ])
    return arguments


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_worker(
    config_path: Path, output_dir: Path, task_index: int
) -> dict[str, object]:
    config = _load_config(config_path)
    tasks = _tasks(config)
    index = int(task_index)
    if not 0 <= index < len(tasks):
        raise ValueError("task_index is outside the planned array")
    task = tasks[index]
    output = output_dir / f"task_{index:05d}.json"
    config_hash = _config_sha256(config_path)
    if output.exists():
        existing = json.loads(output.read_text())
        metadata = existing.get("cluster_task", {})
        if metadata.get("config_sha256") != config_hash:
            raise ValueError(f"existing shard has the wrong config hash: {output}")
        return existing

    started = time.perf_counter()
    status = evaluate_main(_worker_arguments(config, task, output))
    worker_wall_seconds = time.perf_counter() - started
    if status != 0 or not output.exists():
        raise RuntimeError(f"five-point worker failed for task {index}")
    payload = json.loads(output.read_text())
    payload["cluster_task"] = {
        **task,
        "config": str(config_path.resolve()),
        "config_sha256": config_hash,
        "stdout_streamed": True,
        "worker_wall_seconds": worker_wall_seconds,
    }
    payload["status"] = (
        "order8_c_recursion_forest_cluster_shard_not_frozen"
        if config["recursion"]["block_backend"] == "c"
        else "order8_coefficient_extrapolated_h_forest_cluster_shard_not_frozen"
    )
    _atomic_json(output, payload)
    return payload


def _complex(record: dict[str, object]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def _encoded(value: complex) -> dict[str, float]:
    return {"real": complex(value).real, "imag": complex(value).imag}


def reduce_shards(
    config_path: Path, output_dir: Path, summary_path: Path
) -> dict[str, object]:
    config = _load_config(config_path)
    tasks = _tasks(config)
    config_hash = _config_sha256(config_path)
    compatible_hashes = {
        int(index): value
        for index, value in config.get("merge", {}).get(
            "compatible_shard_config_sha256", {}
        ).items()
    }
    groups: dict[tuple[str, float], list[tuple[int, dict[str, object]]]] = {}
    shard_diagnostics: list[dict[str, object]] = []
    missing: list[int] = []
    for task in tasks:
        path = output_dir / f"task_{int(task['task_index']):05d}.json"
        if not path.exists():
            missing.append(int(task["task_index"]))
            continue
        payload = json.loads(path.read_text())
        task_index = int(task["task_index"])
        actual_hash = payload["cluster_task"]["config_sha256"]
        allowed_hashes = {config_hash}
        if task_index in compatible_hashes:
            allowed_hashes.add(compatible_hashes[task_index])
        if actual_hash not in allowed_hashes:
            raise ValueError(f"config hash mismatch in {path}")
        if payload.get("schema") != "type0b-ns-fivepoint-coupled-collar-fit-bundle-v1":
            raise ValueError(f"unexpected worker bundle schema in {path}")
        shard_index = int(payload["cluster_task"]["shard_index"])
        diagnostic = payload.get("self_dual_coefficient_fit")
        if diagnostic is not None:
            shard_diagnostics.append(diagnostic)
        for result in payload["results"]:
            key = (str(result["h_fit_variant"]), float(result["collar_radius"]))
            groups.setdefault(key, []).append((shard_index, result))
    if missing:
        raise FileNotFoundError(f"missing cluster shards: {missing[:16]}")

    radii = tuple(float(value) for value in config["subtraction"]["collar_radii"])
    shard_count = int(config["array"]["shards"])
    components: dict[tuple[str, float], dict[str, object]] = {}
    variants = (
        ("production", "comparison")
        if config["recursion"]["block_backend"] == "h"
        else ("production",)
    )
    for variant in variants:
        for radius in radii:
            key = (variant, radius)
            if key not in groups:
                raise ArithmeticError(f"missing coefficient-fit/radius group {key}")
            shards = sorted(
                groups[key],
                key=lambda item: item[0],
            )
            if [index for index, _result in shards] != list(range(shard_count)):
                raise ArithmeticError(f"group {key} has incomplete shard indices")
            certificates = [
                result["face_collar_certificate"]
                for _index, result in shards
                if result.get("face_collar_certificate") is not None
            ]
            required_certificates = (
                int(config["collar_certificate"]["audit_shards"])
                if variant == "production"
                else 0
            )
            if len(certificates) != required_certificates:
                raise ArithmeticError(
                    f"coefficient-fit/radius {key} has {len(certificates)} "
                    f"certificates; expected {required_certificates}"
                )
            certificates_passed = not any(
                not bool(certificate.get("passed"))
                for certificate in certificates
            )
            if bool(config["collar_certificate"]["enforce"]) and not certificates_passed:
                raise ArithmeticError(f"collar certificate failed at {key}")
            bulks = np.asarray(
                [
                    _complex(value)
                    for _index, result in shards
                    for value in result["bulk_estimates"]
                ],
                dtype=complex,
            )
            faces = np.asarray(
                [
                    _complex(value)
                    for _index, result in shards
                    for value in result["face_estimates"]
                ],
                dtype=complex,
            )
            corner_shards = [
                result
                for _index, result in shards
                if result.get("corner_contribution_computed")
            ]
            if len(corner_shards) != int(config["array"]["deterministic_corner_shards"]):
                raise ArithmeticError(
                    f"coefficient-fit/radius {key} lacks one common corner value"
                )
            corner = _complex(corner_shards[0]["corner_contribution"])
            totals = np.asarray(
                [
                    _complex(bulk) + _complex(face) + corner
                    for _index, result in shards
                    for bulk, face in zip(
                        result["bulk_estimates"], result["face_estimates"]
                    )
                ],
                dtype=complex,
            )
            components[key] = {
                "totals": totals,
                "bulks": bulks,
                "faces": faces,
                "corner": corner,
                "certificates": certificates,
                "certificates_passed": certificates_passed,
                "first_result": shards[0][1],
            }

    radius_summaries: list[dict[str, object]] = []
    production_by_radius: dict[float, np.ndarray] = {}
    for radius in radii:
        production = components[("production", radius)]
        totals = production["totals"]
        production_by_radius[radius] = totals
        first_result = production["first_result"]
        radius_summary = {
                "collar_radius": radius,
                "block_backend": config["recursion"]["block_backend"],
                "regulator_extrapolated_per_coefficient": (
                    config["recursion"]["block_backend"] == "h"
                ),
                "replicate_count": int(totals.size),
                "integral_mean": _encoded(complex(np.mean(totals))),
                "standard_error_real": float(
                    np.std(totals.real, ddof=1) / math.sqrt(totals.size)
                ),
                "standard_error_imag": float(
                    np.std(totals.imag, ddof=1) / math.sqrt(totals.size)
                ),
                "bulk_mean": _encoded(complex(np.mean(production["bulks"]))),
                "face_mean": _encoded(complex(np.mean(production["faces"]))),
                "corner_contribution": _encoded(production["corner"]),
                "face_collar_certificates": production["certificates"],
                "face_collar_certificates_passed": production[
                    "certificates_passed"
                ],
                "bulk_samples": int(
                    shard_count
                    * int(first_result["replicates"])
                    * int(first_result["bulk_samples_per_replicate"])
                ),
                "face_samples": int(
                    shard_count
                    * int(first_result["replicates"])
                    * int(first_result["face_samples_per_replicate"])
                ),
            }
        if config["recursion"]["block_backend"] == "h":
            comparison_totals = components[("comparison", radius)]["totals"]
            if totals.shape != comparison_totals.shape:
                raise ArithmeticError("paired coefficient-fit shapes disagree")
            fit_shifts = totals - comparison_totals
            radius_summary.update(
                {
                    "fit_variable": "eta^2",
                    "polynomial_degree": int(
                        config["self_dual_regulator"]["polynomial_degree"]
                    ),
                    "comparison_degree": int(
                        config["self_dual_regulator"]["comparison_degree"]
                    ),
                    "coefficient_fit_shift_mean": _encoded(
                        complex(np.mean(fit_shifts))
                    ),
                    "maximum_paired_coefficient_fit_shift": float(
                        np.max(np.abs(fit_shifts))
                    ),
                }
            )
        radius_summaries.append(radius_summary)

    payload: dict[str, object] = {
        "schema": (
            "type0b-ns-fivepoint-order8-c-recursion-summary-v5"
            if config["recursion"]["block_backend"] == "c"
            else "type0b-ns-fivepoint-order8-coefficient-table-summary-v4"
        ),
        "status": "worldsheet_cluster_preflight_not_frozen",
        "config": str(config_path.resolve()),
        "config_sha256": config_hash,
        "merged_shard_config_sha256": {
            str(int(task["task_index"])): json.loads(
                (output_dir / f"task_{int(task['task_index']):05d}.json").read_text()
            )["cluster_task"]["config_sha256"]
            for task in tasks
        },
        "task_count": len(tasks),
        "subtraction_scheme": (
            "pointwise F-P1-P2+P12 numerical remainder plus analytic face "
            "and corner finite parts"
        ),
        "recursion": config["recursion"],
        "self_dual_regulator": config.get("self_dual_regulator"),
        "atlas": config["atlas"],
        "collar_certificate_policy": config["collar_certificate"],
        "coefficient_fit_diagnostics_by_shard": shard_diagnostics,
        "radius_summaries": radius_summaries,
        "matrix_model_used": False,
    }
    collar_differences: list[dict[str, object]] = []
    for larger, smaller in zip(radii, radii[1:]):
        paired = production_by_radius[smaller] - production_by_radius[larger]
        collar_differences.append(
            {
                "larger_radius": larger,
                "smaller_radius": smaller,
                "paired_difference_mean": _encoded(complex(np.mean(paired))),
                "paired_difference_standard_error_real": float(
                    np.std(paired.real, ddof=1) / math.sqrt(paired.size)
                ),
                "paired_difference_standard_error_imag": float(
                    np.std(paired.imag, ddof=1) / math.sqrt(paired.size)
                ),
                "maximum_paired_difference": float(np.max(np.abs(paired))),
            }
        )
    payload["collar_stability_differences"] = collar_differences
    _atomic_json(summary_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--task-count-only", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--task-index", type=int, required=True)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--output-dir", type=Path, required=True)
    reduce_parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args.config.resolve())
    if args.command == "plan":
        count = len(_tasks(config))
        print(count if args.task_count_only else json.dumps({"task_count": count}, indent=2))
    elif args.command == "worker":
        payload = run_worker(args.config.resolve(), args.output_dir.resolve(), args.task_index)
        print(json.dumps(payload["cluster_task"], indent=2, sort_keys=True))
    elif args.command == "reduce":
        payload = reduce_shards(
            args.config.resolve(), args.output_dir.resolve(), args.summary.resolve()
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
