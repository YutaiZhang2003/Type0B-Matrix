#!/usr/bin/env python3
"""Parallel cluster driver for Type-0B torus modular checks.

The expensive unit of work is one spectral quadrature node.  A worker
computes its BRY structure constant and exact-c finite-part coefficients
once, then reuses them for every requested tau and recursion cutoff.  Slurm
array tasks own disjoint JSONL shard files; the reducer performs a
deterministic, node-ordered sum and never relies on shared-file appends.
"""

from __future__ import annotations

import argparse
import cmath
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import mpmath
import numpy

from super_liouville_torus_one_point import (
    build_type0b_ns_torus_channel,
    build_type0b_r_torus_channel,
    ns_lift_sign_from_tau,
    type0b_ns_channel_contribution,
    type0b_ns_gauss_legendre_rule,
    type0b_r_channel_contribution,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)


CONFIG_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 2
IMPLEMENTATION_SOURCE_FILES = (
    "super_liouville_torus_modular_cluster.py",
    "super_liouville_torus_one_point.py",
    "super_liouville_structure_constants.py",
    "superconformal_torus_blocks.py",
    "superconformal_blocks.py",
    "mixed_ramond_sphere_blocks.py",
    "ramond_sphere_blocks.py",
    "self_dual_superconformal_blocks.py",
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "type0b_torus_modular_cluster.json"
)


def _complex_pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def _complex_from_pair(value: Sequence[float]) -> complex:
    if len(value) != 2:
        raise ValueError("a complex pair must have two entries")
    return complex(float(value[0]), float(value[1]))


def _tau_from_entry(entry: Mapping[str, Any]) -> complex:
    value = entry.get("tau")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("each tau entry must contain tau=[real, imag]")
    tau = _complex_from_pair(value)
    if tau.imag <= 0.0:
        raise ValueError("all tau values must lie in the upper half-plane")
    return tau


def config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def implementation_manifest(
    source_root: Path | str | None = None,
) -> dict[str, Any]:
    """Fingerprint the numerical source closure and runtime dependencies."""

    root = (
        Path(__file__).resolve().parent
        if source_root is None
        else Path(source_root).resolve()
    )
    source_files = {}
    for relative_path in IMPLEMENTATION_SOURCE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"implementation source is missing: {path}"
            )
        source_files[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    source_tree_sha256 = config_sha256(source_files)
    environment = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": str(numpy.__version__),
        "mpmath_version": str(mpmath.__version__),
    }
    fingerprint_payload = {
        "source_tree_sha256": source_tree_sha256,
        "environment": environment,
    }
    return {
        "sha256": config_sha256(fingerprint_payload),
        "source_tree_sha256": source_tree_sha256,
        "source_files": source_files,
        "environment": environment,
    }


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text())
    validate_config(payload)
    return payload


def validate_config(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    external_momentum = float(payload["external_momentum"])
    if not math.isfinite(external_momentum):
        raise ValueError("external_momentum must be finite")
    modular_orbit = str(payload.get("modular_orbit", "ns_to_ns"))
    if modular_orbit not in ("ns_to_ns", "ns_tilde_to_r"):
        raise ValueError(
            "modular_orbit must be 'ns_to_ns' or 'ns_tilde_to_r'"
        )

    levels = tuple(int(level) for level in payload["levels"])
    if not levels or levels != tuple(sorted(set(levels))) or levels[0] < 0:
        raise ValueError("levels must be sorted, unique, and nonnegative")
    if modular_orbit == "ns_tilde_to_r" and any(
        level % 2 for level in levels
    ):
        raise ValueError(
            "ns_tilde_to_r levels must be even so both sectors end at "
            "the same power of q"
        )

    taus = payload["taus"]
    if not isinstance(taus, list) or not taus:
        raise ValueError("taus must be a nonempty list")
    tau_names = [str(entry["name"]) for entry in taus]
    if len(tau_names) != len(set(tau_names)):
        raise ValueError("tau names must be unique")
    for entry in taus:
        _tau_from_entry(entry)

    studies = payload["studies"]
    if not isinstance(studies, list) or not studies:
        raise ValueError("studies must be a nonempty list")
    study_names = [str(entry["name"]) for entry in studies]
    if len(study_names) != len(set(study_names)):
        raise ValueError("study names must be unique")
    for study in studies:
        if int(study["quadrature_order"]) < 2:
            raise ValueError("quadrature_order must be at least two")
        if float(study["p_max"]) <= 0.0:
            raise ValueError("p_max must be positive")
        if int(study["structure_precision"]) < 15:
            raise ValueError("structure_precision must be at least 15")
        if int(study["finite_part_samples"]) < 8:
            raise ValueError("finite_part_samples must be at least eight")

    default_shards = int(payload["default_shard_count"])
    if default_shards < 1:
        raise ValueError("default_shard_count must be positive")

    comparison_names: set[str] = set()
    for comparison in payload.get("comparisons", []):
        name = str(comparison["name"])
        if name in comparison_names:
            raise ValueError("comparison names must be unique")
        comparison_names.add(name)
        if str(comparison["left"]) not in study_names:
            raise ValueError(f"unknown comparison study {comparison['left']}")
        if str(comparison["right"]) not in study_names:
            raise ValueError(f"unknown comparison study {comparison['right']}")

    for target in payload.get("accuracy_targets", []):
        kind = str(target["kind"])
        if float(target["maximum"]) <= 0.0:
            raise ValueError("accuracy target maxima must be positive")
        if kind == "modular_residual":
            if str(target["study"]) not in study_names:
                raise ValueError(f"unknown target study {target['study']}")
            if str(target["tau"]) not in tau_names:
                raise ValueError(f"unknown target tau {target['tau']}")
            if int(target["level"]) not in levels:
                raise ValueError(f"unknown target level {target['level']}")
        elif kind == "comparison_relative_change":
            if str(target["comparison"]) not in comparison_names:
                raise ValueError(
                    f"unknown target comparison {target['comparison']}"
                )
        elif kind == "finite_part_two_radius":
            if str(target["study"]) not in study_names:
                raise ValueError(f"unknown target study {target['study']}")
        else:
            raise ValueError(f"unsupported accuracy target kind {kind}")


def atomic_jobs(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    jobs = []
    job_id = 0
    for study_index, study in enumerate(payload["studies"]):
        for node_index in range(int(study["quadrature_order"])):
            jobs.append(
                {
                    "job_id": job_id,
                    "study_index": study_index,
                    "study_name": str(study["name"]),
                    "node_index": node_index,
                }
            )
            job_id += 1
    return tuple(jobs)


def jobs_for_shard(
    payload: Mapping[str, Any], shard_id: int, shard_count: int
) -> tuple[dict[str, Any], ...]:
    shard_id = int(shard_id)
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_id < shard_count:
        raise ValueError("shard_id must satisfy 0 <= id < shard_count")
    return tuple(
        job
        for job in atomic_jobs(payload)
        if int(job["job_id"]) % shard_count == shard_id
    )


def _evaluate_atomic_job(
    payload: Mapping[str, Any],
    digest: str,
    job: Mapping[str, Any],
    *,
    shard_id: int,
    shard_count: int,
) -> dict[str, Any]:
    started = time.monotonic()
    study = payload["studies"][int(job["study_index"])]
    rule = type0b_ns_gauss_legendre_rule(
        float(study["p_max"]), int(study["quadrature_order"])
    )
    momentum, spectral_weight = rule[int(job["node_index"])]
    ns_channel = build_type0b_ns_torus_channel(
        momentum=momentum,
        spectral_weight=spectral_weight,
        external_momentum=float(payload["external_momentum"]),
        structure_precision=int(study["structure_precision"]),
        finite_part_samples=int(study["finite_part_samples"]),
    )

    maximum_level = max(int(level) for level in payload["levels"])
    ns_raw_coefficients = ns_channel.block.raw_coefficients(maximum_level)
    diagnostics = list(
        ns_channel.block.coefficient_diagnostics(maximum_level).values()
    )
    modular_orbit = str(payload.get("modular_orbit", "ns_to_ns"))
    r_channel = None
    r_raw_coefficients = None
    if modular_orbit == "ns_tilde_to_r":
        r_channel = build_type0b_r_torus_channel(
            momentum=momentum,
            spectral_weight=spectral_weight,
            external_momentum=float(payload["external_momentum"]),
            structure_precision=int(study["structure_precision"]),
            finite_part_samples=int(study["finite_part_samples"]),
        )
        maximum_r_level = maximum_level // 2
        r_raw_coefficients = r_channel.block.raw_even_coefficients(
            maximum_r_level
        )
        diagnostics.extend(
            r_channel.block.coefficient_diagnostics(
                maximum_r_level
            ).values()
        )
    maximum_two_radius_difference = max(
        (item.absolute_error for item in diagnostics), default=0.0
    )
    c = central_charge(1.0)

    tau_results: dict[str, Any] = {}
    for tau_entry in payload["taus"]:
        tau_name = str(tau_entry["name"])
        tau = _tau_from_entry(tau_entry)
        s_tau = -1.0 / tau
        q = cmath.exp(2.0j * math.pi * tau)
        q_tilde = cmath.exp(2.0j * math.pi * s_tau)
        lift_sign = ns_lift_sign_from_tau(tau)
        if modular_orbit == "ns_to_ns":
            lift_sign_tilde = ns_lift_sign_from_tau(s_tau)
            direct_plumbing = NSPlumbingParameter(q, lift_sign)
            transformed_plumbing = NSPlumbingParameter(
                q_tilde, lift_sign_tilde
            )
            direct_spin_structure = "NS"
            transformed_spin_structure = "NS"
        else:
            lift_sign_tilde = -lift_sign
            direct_plumbing = NSPlumbingParameter(q, lift_sign_tilde)
            transformed_plumbing = RamondPlumbingParameter(
                q_tilde, "identity"
            )
            direct_spin_structure = "NS_tilde"
            transformed_spin_structure = "R"
        levels: dict[str, Any] = {}
        for level in payload["levels"]:
            level = int(level)
            direct_contribution = type0b_ns_channel_contribution(
                ns_channel,
                direct_plumbing,
                ns_raw_coefficients,
                level,
                c=c,
            )
            if modular_orbit == "ns_to_ns":
                transformed_contribution = type0b_ns_channel_contribution(
                    ns_channel,
                    transformed_plumbing,
                    ns_raw_coefficients,
                    level,
                    c=c,
                )
            else:
                assert r_channel is not None
                assert r_raw_coefficients is not None
                transformed_contribution = type0b_r_channel_contribution(
                    r_channel,
                    transformed_plumbing,
                    r_raw_coefficients,
                    level // 2,
                    c=c,
                )
            levels[str(level)] = {
                "q": _complex_pair(direct_contribution),
                "q_tilde": _complex_pair(transformed_contribution),
            }
        tau_results[tau_name] = {
            "tau": _complex_pair(tau),
            "s_tau": _complex_pair(s_tau),
            "lift_sign": lift_sign,
            "lift_sign_tilde": lift_sign_tilde,
            "direct_spin_structure": direct_spin_structure,
            "transformed_spin_structure": transformed_spin_structure,
            "levels": levels,
        }

    result = {
        "record_type": "node",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": digest,
        "shard_id": int(shard_id),
        "shard_count": int(shard_count),
        "job_id": int(job["job_id"]),
        "study_index": int(job["study_index"]),
        "study_name": str(job["study_name"]),
        "node_index": int(job["node_index"]),
        "momentum": momentum,
        "spectral_weight": spectral_weight,
        "weighted_structure_constant": _complex_pair(
            ns_channel.weighted_structure_constant
        ),
        "maximum_two_radius_difference": maximum_two_radius_difference,
        "taus": tau_results,
        "runtime_seconds": time.monotonic() - started,
    }
    if r_channel is not None:
        result["weighted_structure_constants"] = {
            "NS_tilde": _complex_pair(
                ns_channel.weighted_structure_constant
            ),
            "R": _complex_pair(r_channel.weighted_structure_constant),
        }
    return result


def _shard_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / f"shard-{int(shard_id):04d}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _existing_shard_is_complete(
    path: Path,
    *,
    digest: str,
    implementation_sha256: str,
    shard_id: int,
    shard_count: int,
    expected_job_ids: set[int],
) -> bool:
    if not path.exists():
        return False
    try:
        records = _read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not records or records[0].get("record_type") != "metadata":
        return False
    metadata = records[0]
    if (
        metadata.get("config_sha256") != digest
        or metadata.get("implementation_sha256")
        != implementation_sha256
        or int(metadata.get("schema_version", -1))
        != ARTIFACT_SCHEMA_VERSION
        or int(metadata.get("shard_id", -1)) != shard_id
        or int(metadata.get("shard_count", -1)) != shard_count
    ):
        return False
    observed = {
        int(record["job_id"])
        for record in records[1:]
        if record.get("record_type") == "node"
    }
    return observed == expected_job_ids


def run_shard(
    config_path: Path | str,
    output_dir: Path | str,
    shard_id: int,
    shard_count: int,
    force: bool = False,
) -> dict[str, Any]:
    payload = load_config(config_path)
    digest = config_sha256(payload)
    implementation = implementation_manifest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for_shard(payload, shard_id, shard_count)
    expected_ids = {int(job["job_id"]) for job in jobs}
    output_path = _shard_path(output_dir, shard_id)
    if not force and _existing_shard_is_complete(
        output_path,
        digest=digest,
        implementation_sha256=str(implementation["sha256"]),
        shard_id=int(shard_id),
        shard_count=int(shard_count),
        expected_job_ids=expected_ids,
    ):
        return {
            "shard_id": int(shard_id),
            "job_count": len(jobs),
            "output": str(output_path),
            "status": "already-complete",
        }

    started = time.monotonic()
    records = [
        _evaluate_atomic_job(
            payload,
            digest,
            job,
            shard_id=int(shard_id),
            shard_count=int(shard_count),
        )
        for job in jobs
    ]
    metadata = {
        "record_type": "metadata",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": digest,
        "implementation_sha256": implementation["sha256"],
        "implementation": implementation,
        "shard_id": int(shard_id),
        "shard_count": int(shard_count),
        "job_count": len(jobs),
    }
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    with temporary.open("w") as handle:
        for record in (metadata, *records):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return {
        "shard_id": int(shard_id),
        "job_count": len(jobs),
        "output": str(output_path),
        "runtime_seconds": time.monotonic() - started,
        "status": "computed",
    }


def _local_shard_entry(arguments: tuple[Any, ...]) -> dict[str, Any]:
    return run_shard(*arguments)


def run_local_parallel(
    *,
    config_path: Path | str,
    output_dir: Path | str,
    workers: int,
    shard_count: int,
    force: bool,
) -> tuple[dict[str, Any], ...]:
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    arguments = [
        (
            str(config_path),
            str(output_dir),
            shard_id,
            int(shard_count),
            bool(force),
        )
        for shard_id in range(int(shard_count))
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_local_shard_entry, item): item[2]
            for item in arguments
        }
        for future in as_completed(futures):
            results.append(future.result())
    return tuple(sorted(results, key=lambda item: int(item["shard_id"])))


def _fsum_complex(values: Iterable[complex]) -> complex:
    values = tuple(complex(value) for value in values)
    return complex(
        math.fsum(value.real for value in values),
        math.fsum(value.imag for value in values),
    )


def _relative_change(left: complex, right: complex) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def _load_complete_rows(
    *,
    payload: Mapping[str, Any],
    implementation: Mapping[str, Any],
    input_dir: Path,
    shard_count: int,
) -> tuple[dict[str, Any], ...]:
    digest = config_sha256(payload)
    rows: list[dict[str, Any]] = []
    for shard_id in range(shard_count):
        path = _shard_path(input_dir, shard_id)
        if not path.exists():
            raise FileNotFoundError(f"missing shard {path}")
        records = _read_jsonl(path)
        if not records or records[0].get("record_type") != "metadata":
            raise ValueError(f"shard {path} has no metadata record")
        metadata = records[0]
        if metadata.get("config_sha256") != digest:
            raise ValueError(f"configuration digest mismatch in {path}")
        if int(metadata.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"artifact schema mismatch in {path}")
        if (
            metadata.get("implementation_sha256")
            != implementation["sha256"]
        ):
            raise ValueError(f"implementation fingerprint mismatch in {path}")
        if metadata.get("implementation") != implementation:
            raise ValueError(f"implementation manifest mismatch in {path}")
        if int(metadata.get("shard_id", -1)) != shard_id:
            raise ValueError(f"shard id mismatch in {path}")
        if int(metadata.get("shard_count", -1)) != shard_count:
            raise ValueError(f"shard count mismatch in {path}")
        rows.extend(records[1:])

    expected = {int(job["job_id"]) for job in atomic_jobs(payload)}
    observed = [int(row["job_id"]) for row in rows]
    if set(observed) != expected or len(observed) != len(expected):
        missing = sorted(expected - set(observed))
        duplicates = len(observed) - len(set(observed))
        raise ValueError(
            f"incomplete node ledger: missing={missing[:8]}, "
            f"duplicate_count={duplicates}"
        )
    return tuple(sorted(rows, key=lambda row: int(row["job_id"])))


def reduce_shards(
    *,
    config_path: Path | str,
    input_dir: Path | str,
    output_path: Path | str,
    shard_count: int,
) -> dict[str, Any]:
    payload = load_config(config_path)
    digest = config_sha256(payload)
    implementation = implementation_manifest()
    input_dir = Path(input_dir)
    rows = _load_complete_rows(
        payload=payload,
        implementation=implementation,
        input_dir=input_dir,
        shard_count=int(shard_count),
    )
    external_weight = ns_liouville_weight(
        float(payload["external_momentum"]), 1.0
    )
    study_summaries: dict[str, Any] = {}

    for study in payload["studies"]:
        study_name = str(study["name"])
        study_rows = sorted(
            (row for row in rows if row["study_name"] == study_name),
            key=lambda row: int(row["node_index"]),
        )
        if len(study_rows) != int(study["quadrature_order"]):
            raise ValueError(f"study {study_name} has the wrong node count")
        tau_summaries: dict[str, Any] = {}
        for tau_entry in payload["taus"]:
            tau_name = str(tau_entry["name"])
            tau = _tau_from_entry(tau_entry)
            expected_ratio = abs(tau) ** (2.0 * external_weight.real)
            level_summaries: dict[str, Any] = {}
            for level in payload["levels"]:
                key = str(int(level))
                value_q = _fsum_complex(
                    _complex_from_pair(row["taus"][tau_name]["levels"][key]["q"])
                    for row in study_rows
                )
                value_q_tilde = _fsum_complex(
                    _complex_from_pair(
                        row["taus"][tau_name]["levels"][key]["q_tilde"]
                    )
                    for row in study_rows
                )
                numeric_ratio = value_q_tilde / value_q
                relative_error = numeric_ratio / expected_ratio - 1.0
                level_summaries[key] = {
                    "value_q": _complex_pair(value_q),
                    "value_q_tilde": _complex_pair(value_q_tilde),
                    "numeric_ratio": _complex_pair(numeric_ratio),
                    "expected_ratio": expected_ratio,
                    "relative_error": _complex_pair(relative_error),
                    "relative_error_abs": abs(relative_error),
                }
            first = study_rows[0]["taus"][tau_name]
            tau_summaries[tau_name] = {
                "tau": first["tau"],
                "s_tau": first["s_tau"],
                "lift_sign": int(first["lift_sign"]),
                "lift_sign_tilde": int(first["lift_sign_tilde"]),
                "direct_spin_structure": str(
                    first.get("direct_spin_structure", "NS")
                ),
                "transformed_spin_structure": str(
                    first.get("transformed_spin_structure", "NS")
                ),
                "levels": level_summaries,
            }
        study_summaries[study_name] = {
            "parameters": dict(study),
            "node_count": len(study_rows),
            "maximum_two_radius_difference": max(
                float(row["maximum_two_radius_difference"])
                for row in study_rows
            ),
            "total_node_runtime_seconds": math.fsum(
                float(row["runtime_seconds"]) for row in study_rows
            ),
            "taus": tau_summaries,
        }

    comparison_summaries: dict[str, Any] = {}
    for comparison in payload.get("comparisons", []):
        name = str(comparison["name"])
        left_name = str(comparison["left"])
        right_name = str(comparison["right"])
        points: dict[str, Any] = {}
        maximum_change = 0.0
        for tau_entry in payload["taus"]:
            tau_name = str(tau_entry["name"])
            tau_points: dict[str, Any] = {}
            for level in payload["levels"]:
                key = str(int(level))
                left = study_summaries[left_name]["taus"][tau_name]["levels"][
                    key
                ]
                right = study_summaries[right_name]["taus"][tau_name][
                    "levels"
                ][key]
                direct_change = _relative_change(
                    _complex_from_pair(left["value_q"]),
                    _complex_from_pair(right["value_q"]),
                )
                transformed_change = _relative_change(
                    _complex_from_pair(left["value_q_tilde"]),
                    _complex_from_pair(right["value_q_tilde"]),
                )
                maximum_change = max(
                    maximum_change, direct_change, transformed_change
                )
                tau_points[key] = {
                    "value_q_relative_change": direct_change,
                    "value_q_tilde_relative_change": transformed_change,
                }
            points[tau_name] = tau_points
        comparison_summaries[name] = {
            "left": left_name,
            "right": right_name,
            "maximum_relative_change": maximum_change,
            "points": points,
        }

    checks = []
    for target in payload.get("accuracy_targets", []):
        kind = str(target["kind"])
        maximum = float(target["maximum"])
        if kind == "modular_residual":
            observed = study_summaries[str(target["study"])]["taus"][
                str(target["tau"])
            ]["levels"][str(int(target["level"]))]["relative_error_abs"]
        elif kind == "comparison_relative_change":
            observed = comparison_summaries[str(target["comparison"])][
                "maximum_relative_change"
            ]
        elif kind == "finite_part_two_radius":
            observed = study_summaries[str(target["study"])][
                "maximum_two_radius_difference"
            ]
        else:  # validate_config rejects this before the reduction begins.
            raise ValueError(f"unsupported accuracy target kind {kind}")
        checks.append(
            {
                "name": str(target["name"]),
                "kind": kind,
                "observed": observed,
                "maximum": maximum,
                "passed": observed <= maximum,
            }
        )

    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config_sha256": digest,
        "implementation_sha256": implementation["sha256"],
        "implementation": implementation,
        "modular_orbit": str(
            payload.get("modular_orbit", "ns_to_ns")
        ),
        "external_momentum": float(payload["external_momentum"]),
        "external_weight": _complex_pair(external_weight),
        "levels": [int(level) for level in payload["levels"]],
        "shard_count": int(shard_count),
        "atomic_job_count": len(rows),
        "studies": study_summaries,
        "comparisons": comparison_summaries,
        "accuracy_checks": checks,
        "accuracy_targets_passed": all(check["passed"] for check in checks),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return summary


def _print_plan(payload: Mapping[str, Any], shard_count: int) -> None:
    jobs = atomic_jobs(payload)
    counts = [
        len(jobs_for_shard(payload, shard_id, shard_count))
        for shard_id in range(shard_count)
    ]
    print("Type-0B torus modular cluster plan")
    print(
        "  modular_orbit="
        f"{payload.get('modular_orbit', 'ns_to_ns')}"
    )
    print(f"  config_sha256={config_sha256(payload)}")
    print(f"  implementation_sha256={implementation_manifest()['sha256']}")
    print(f"  studies={len(payload['studies'])}")
    print(f"  tau values={len(payload['taus'])}")
    print(f"  recursion cutoffs={list(payload['levels'])}")
    print(f"  atomic momentum jobs={len(jobs)}")
    print(f"  shards={shard_count}")
    print(f"  jobs per shard={min(counts)}..{max(counts)}")
    for study in payload["studies"]:
        print(
            f"    {study['name']}: Np={study['quadrature_order']}, "
            f"Pmax={study['p_max']}, samples={study['finite_part_samples']}, "
            f"dps={study['structure_precision']}"
        )


def _print_summary(summary: Mapping[str, Any]) -> None:
    print("Type-0B torus modular cluster reduction")
    print(f"  config_sha256={summary['config_sha256']}")
    print(f"  implementation_sha256={summary['implementation_sha256']}")
    print(f"  atomic jobs={summary['atomic_job_count']}")
    print(f"  shards={summary['shard_count']}")
    for study_name, study in summary["studies"].items():
        print(f"  study={study_name}")
        for tau_name, tau in study["taus"].items():
            lifts = (tau["lift_sign"], tau["lift_sign_tilde"])
            highest = str(max(int(level) for level in summary["levels"]))
            row = tau["levels"][highest]
            print(
                f"    {tau_name}: "
                f"{tau['direct_spin_structure']}->"
                f"{tau['transformed_spin_structure']}, lifts={lifts}, "
                f"level-{highest} residual="
                f"{row['relative_error_abs']:.6e}"
            )
    for name, comparison in summary["comparisons"].items():
        print(
            f"  comparison={name}: maximum relative change="
            f"{comparison['maximum_relative_change']:.6e}"
        )
    for check in summary["accuracy_checks"]:
        state = "PASS" if check["passed"] else "FAIL"
        print(
            f"  [{state}] {check['name']}: {check['observed']:.6e} "
            f"<= {check['maximum']:.6e}"
        )
    print(
        "  accuracy targets="
        f"{'PASS' if summary['accuracy_targets_passed'] else 'FAIL'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Type-0B genus-one one-point modular check"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--shard-count", type=int)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--shard-id", type=int, required=True)
    worker_parser.add_argument("--shard-count", type=int)
    worker_parser.add_argument("--force", action="store_true")
    worker_parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement before numerical evaluation",
    )

    local_parser = subparsers.add_parser("local")
    local_parser.add_argument("--output-dir", type=Path, required=True)
    local_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    local_parser.add_argument("--shard-count", type=int)
    local_parser.add_argument("--force", action="store_true")
    local_parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement before numerical evaluation",
    )

    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--input-dir", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--shard-count", type=int)

    args = parser.parse_args()
    payload = load_config(args.config)
    shard_count = (
        int(args.shard_count)
        if args.shard_count is not None
        else int(payload["default_shard_count"])
    )

    if args.command == "plan":
        _print_plan(payload, shard_count)
        return
    if args.command == "worker":
        if not args.execute:
            raise SystemExit("worker requires --execute")
        result = run_shard(
            args.config,
            args.output_dir,
            args.shard_id,
            shard_count,
            args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "local":
        if not args.execute:
            raise SystemExit("local evaluation requires --execute")
        results = run_local_parallel(
            config_path=args.config,
            output_dir=args.output_dir,
            workers=args.workers,
            shard_count=shard_count,
            force=args.force,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    if args.command == "reduce":
        summary = reduce_shards(
            config_path=args.config,
            input_dir=args.input_dir,
            output_path=args.output,
            shard_count=shard_count,
        )
        _print_summary(summary)
        if not summary["accuracy_targets_passed"]:
            raise SystemExit(1)
        return
    raise SystemExit(f"unsupported command {args.command}")


if __name__ == "__main__":
    main()
