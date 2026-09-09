#!/usr/bin/env python3
"""Audit an archived genus-two Cannon run without reevaluating CFT blocks.

The archived production module is loaded from ``RUN_ROOT/code/Code``.  Each
immutable shard is checked against the task decoder and Gaussian quadrature
rule from that exact source snapshot.  The Liouville numerator sums are then
reduced independently and compared with ``summary.json``.

This does not establish convergence in recursion or momentum order.  It only
establishes that the reported fixed-cutoff summary is the deterministic
reduction of the supplied, internally consistent production shards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=2.0e-15, abs_tol=1.0e-300)


def _quadrature_close(observed: float, expected: float) -> bool:
    # NumPy's Hermite nodes differ by a few ulps across BLAS/platform builds.
    return math.isclose(observed, expected, rel_tol=2.0e-13, abs_tol=1.0e-300)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_archived_cannon(code_root: Path):
    search_paths = (
        code_root,
        code_root / "c_Recursion",
        code_root / "h_recursion",
        code_root / "double_virasoro",
        code_root / "PBW_c_recursion_double_virasoro crosscheck",
        code_root / "genus_2_cross_channel",
        code_root / "python",
    )
    for search_path in reversed(search_paths):
        if search_path.exists():
            sys.path.insert(0, str(search_path))
    return importlib.import_module("ns_genus2_cannon")


def _design_key(row: dict[str, Any]) -> tuple[str, str, int, int, float]:
    return (
        str(row["point_id"]),
        str(row["channel"]),
        int(row["recursion_order"]),
        int(row["quadrature_order"]),
        float(row["finite_part_radius"]),
    )


def audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    code_root = run_root / "code" / "Code"
    config_path = (
        code_root / "config" / "ns_genus2_cannon_fivepoint_r24_n10_affine.json"
    )
    shard_root = run_root / "shards"
    summary_path = run_root / "summary.json"
    for required in (code_root, config_path, shard_root, summary_path):
        _require(required.exists(), f"missing required production artifact: {required}")

    cannon = _load_archived_cannon(code_root)
    config = cannon._load(config_path)
    summary = json.loads(summary_path.read_text())
    count = int(cannon.task_count(config))
    config_digest = str(cannon._digest(config))
    production_fingerprint = str(summary["implementation_fingerprint"])

    expected_names = {f"task-{index:06d}.json" for index in range(count)}
    observed_names = {path.name for path in shard_root.glob("task-*.json")}
    _require(
        observed_names == expected_names,
        "shard filename set differs from the exact decoded task range: "
        f"missing={sorted(expected_names - observed_names)[:5]}, "
        f"extra={sorted(observed_names - expected_names)[:5]}",
    )
    _require(int(summary["task_count"]) == count, "summary task_count mismatch")
    _require(str(summary["config_digest"]) == config_digest, "summary config mismatch")

    radius_sums: dict[tuple[str, str, int, int, float], list[float]] = {}
    shard_set_digest = hashlib.sha256()
    fingerprints: set[str] = set()
    for task_index in range(count):
        path = shard_root / f"task-{task_index:06d}.json"
        payload = path.read_bytes()
        shard_set_digest.update(path.name.encode())
        shard_set_digest.update(b"\0")
        shard_set_digest.update(payload)
        shard = json.loads(payload)
        design, node_index = cannon.decode_task(config, task_index)
        q_values, indices, momenta, measure = cannon._node_data(
            config, design, node_index
        )

        _require(shard.get("schema") == cannon.SCHEMA, f"schema mismatch in {path}")
        _require(int(shard.get("task_index", -1)) == task_index, f"task mismatch in {path}")
        _require(int(shard.get("node_index", -1)) == node_index, f"node mismatch in {path}")
        _require(shard.get("config_digest") == config_digest, f"config mismatch in {path}")
        fingerprint = str(shard.get("implementation_fingerprint"))
        fingerprints.add(fingerprint)
        _require(
            fingerprint == production_fingerprint,
            f"implementation fingerprint mismatch in {path}",
        )
        for field in (
            "point_id",
            "channel",
            "recursion_order",
            "quadrature_order",
            "node_count",
        ):
            _require(shard.get(field) == design[field], f"{field} mismatch in {path}")
        _require(tuple(shard["indices"]) == tuple(indices), f"indices mismatch in {path}")
        _require(
            all(
                _quadrature_close(float(a), float(b))
                for a, b in zip(shard["momenta"], momenta)
            ),
            f"momenta mismatch in {path}",
        )
        _require(
            _quadrature_close(float(shard["measure"]), float(measure)),
            f"measure mismatch in {path}",
        )

        if design["channel"] == "theta":
            q_order = tuple(cannon.THETA_GEOMETRY_EDGE_ORDER)
            descendant_order = tuple(cannon.THETA_CCY_DESCENDANT_EDGE_ORDER)
        else:
            q_order = tuple(cannon.GLASSES_GEOMETRY_EDGE_ORDER)
            descendant_order = tuple(cannon.GLASSES_CCY_DESCENDANT_EDGE_ORDER)
        _require(tuple(shard["q_edge_order"]) == q_order, f"q order mismatch in {path}")
        _require(
            tuple(shard["descendant_tensor_edge_order"]) == descendant_order,
            f"descendant order mismatch in {path}",
        )
        _require(len(q_values) == 3, f"invalid plumbing tuple in {path}")

        point = cannon._point(config, str(design["point_id"]))
        expected_radii = [float(config["finite_part_radii"][0])]
        if bool(point.get("secondary_finite_part_radius", False)):
            expected_radii.append(float(config["finite_part_radii"][1]))
        observed_radii = [float(item["finite_part_radius"]) for item in shard["radius_results"]]
        _require(observed_radii == expected_radii, f"radius ledger mismatch in {path}")
        for item in shard["radius_results"]:
            contribution = float(item["contribution"])
            _require(math.isfinite(contribution), f"nonfinite contribution in {path}")
            key = (
                str(design["point_id"]),
                str(design["channel"]),
                int(design["recursion_order"]),
                int(design["quadrature_order"]),
                float(item["finite_part_radius"]),
            )
            radius_sums.setdefault(key, []).append(contribution)

    _require(fingerprints == {production_fingerprint}, "mixed shard fingerprints")
    summary_rows = {_design_key(row): row for row in summary["rows"]}
    _require(set(summary_rows) == set(radius_sums), "summary design/radius set mismatch")
    reductions = []
    for key in sorted(radius_sums):
        observed = math.fsum(radius_sums[key])
        expected = float(summary_rows[key]["z_liouville"])
        _require(_close(observed, expected), f"Liouville reduction mismatch for {key}")
        reductions.append(
            {
                "point_id": key[0],
                "channel": key[1],
                "recursion_order": key[2],
                "quadrature_order": key[3],
                "finite_part_radius": key[4],
                "z_liouville": observed,
            }
        )

    crossing_by_key = {
        (
            str(row["point_id"]),
            int(row["recursion_order"]),
            int(row["quadrature_order"]),
            float(row["finite_part_radius"]),
        ): row
        for row in summary["crossing"]
    }
    for crossing_key, crossing in crossing_by_key.items():
        point_id, recursion_order, quadrature_order, radius = crossing_key
        theta = summary_rows[(point_id, "theta", recursion_order, quadrature_order, radius)]
        glasses = summary_rows[(point_id, "glasses", recursion_order, quadrature_order, radius)]
        ratio = float(theta["q_l"]) / float(glasses["q_l"])
        _require(
            _close(ratio, float(crossing["theta_over_glasses"])),
            f"crossing ratio mismatch for {crossing_key}",
        )

    return {
        "status": "pass",
        "scope": "fixed-cutoff shard identity and deterministic reduction only",
        "run_root": str(run_root),
        "schema": str(summary["schema"]),
        "task_count": count,
        "design_radius_count": len(radius_sums),
        "config_digest": config_digest,
        "production_implementation_fingerprint": production_fingerprint,
        "shard_set_sha256": shard_set_digest.hexdigest(),
        "reductions": reductions,
        "convergence_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
