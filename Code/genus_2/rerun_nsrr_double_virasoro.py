#!/usr/bin/env python3
"""Rerun the saved modular comparison with every NSRR block from native CCY.

Recompute all source momentum nodes, including both HJS signs and form parities.
Keep the saved all-NS target, quadrature, geometry, spin projection, structure
constant conventions, and normalization fixed. Report cumulative NSRR orders
3, 4, and 5 and the change from the old order-3 PBW-assisted source.
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

import nsrr_factorized_sign_trial as trial
import run_nsrr_nsnsns_offaxis_constant_scan as scan
from nsrr_cpp_backend import METHOD, implementation_hashes
from physical_nsrr_sewing import SOURCE_FIXED_SPIN_LIFTS, project_source_fixed_spin, contract_physical_blocks


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(baseline, output, levels):
    baseline = baseline.resolve()
    old = trial.load(baseline / "config.json")
    if old.get("normalization_policy", "").startswith("fixed by sewing:") is False:
        raise ValueError("Use an existing fixed-normalization modular comparison")
    # The old source kernels changed upstream; the target's protected kernels
    # must still be identical before its saved coefficients can serve as control.
    for name in ("Code/c_Recursion/ns_genus2_partition.py", "Code/c_Recursion/theta_star_algebra.py"):
        if sha(ROOT / name) != old["protected_kernel_hashes"][name]:
            raise ValueError(f"Saved all-NS control kernel changed: {name}")
    if len(old["orders"]) != 1:
        raise ValueError("Use one quadrature order per rerun")
    summary = trial.load(baseline / "summary.json")
    if trial.digest(summary["config"]) != trial.digest(old):
        raise ValueError("Saved summary/config mismatch")
    input_hashes = {"config.json": sha(baseline / "config.json"),
                    "summary.json": sha(baseline / "summary.json")}
    for index in range(len(scan.tasks(old))):
        for channel in ("source", "target"):
            relative = f"{channel}/shards/node-{index:03d}.json"
            shard = trial.load(baseline / relative)
            scan.validate_shard(old, channel, index, shard)
            input_hashes[relative] = sha(baseline / relative)
    files = [Path(__file__), Path(trial.__file__), Path(trial.dv.__file__),
             ROOT / "Code/genus_2/physical_nsrr_sewing.py", ROOT / "Code/genus_2/nsrr_plumbing_adapter.py",
             ROOT / "Code/c_Recursion/generic_super_liouville_structure_constants.py"]
    config = dict(schema="native-nsrr-modular-rerun-v1", baseline=str(baseline),
                  baseline_config=old, input_sha256=input_hashes, source_levels=levels,
                  method=METHOD, native_dps=40, target_policy="reuse unchanged saved all-NS target",
                  normalization_policy=old["normalization_policy"],
                  implementation_sha256={**implementation_hashes(), **{str(p.relative_to(ROOT)): sha(p) for p in files}})
    path = output / "config.json"
    if path.exists() and trial.load(path) != config:
        raise ValueError("Output directory belongs to a different run")
    trial.save(path, config)
    return config


def worker(config_path, index):
    config = trial.load(config_path)
    old = config["baseline_config"]
    order, node, momenta, measure = scan.node_data(old, "source", index)
    started = time.perf_counter()
    constants = trial.GenericSuperLiouvilleConstants(old["b"], dps=30)
    bry = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    components, checks = trial.block_components(old["b"], momenta[::-1], max(config["source_levels"]),
                                                native_dps=config["native_dps"])
    if checks["physical_PBW_used"] or checks["native_pipeline_calls"] != 8:
        raise AssertionError("All eight NSRR channels must use native double-Virasoro")
    values = []
    for point in old["points"]:
        q = tuple(complex(value) for value in point["source"]["q_values"])
        for level in config["source_levels"]:
            amplitudes = {}
            for lift in SOURCE_FIXED_SPIN_LIFTS:
                plumbing = trial.NSRRPlumbingInputs(q, lift, trial.GEOMETRY_SECTORS)
                primary = plumbing.primary(old["b"], momenta)
                blocks = trial.evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, level)
                amplitudes[lift] = {channel: primary * value for channel, value in blocks.items()}
            local = contract_physical_blocks(project_source_fixed_spin(amplitudes), bry)
            values.append(dict(point_id=point["point_id"], source_level=level,
                               source_Z_unscaled_M_node=4 * local["total"],
                               diagonal_Z_unscaled_node=4 * local["diagonal"],
                               interference_Z_unscaled_node=4 * local["interference"]))
    result = dict(config_digest=trial.digest(config), index=index, order=order, node=node,
                  momenta=momenta, measure=measure, C_BRY=[trial.encode(z) for z in bry],
                  checks=checks, values=values, elapsed_seconds=time.perf_counter()-started)
    trial.save(config_path.parent / "source" / f"node-{index:03d}.json", result)


def reduce(config_path):
    config = trial.load(config_path)
    old = config["baseline_config"]
    baseline = Path(config["baseline"])
    for relative, expected in config["input_sha256"].items():
        if sha(baseline / relative) != expected:
            raise ValueError(f"Baseline changed during rerun: {relative}")
    for relative, expected in config["implementation_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Implementation changed during rerun: {relative}")
    source = [trial.load(config_path.parent / "source" / f"node-{i:03d}.json")
              for i in range(len(scan.tasks(old)))]
    target = [trial.load(baseline / "target" / "shards" / f"node-{i:03d}.json") for i in range(len(source))]
    prior = {row["point_id"]: row for row in trial.load(baseline / "summary.json")["comparisons"]}
    for i, shard in enumerate(source):
        if shard["config_digest"] != trial.digest(config) or shard["index"] != i:
            raise ValueError("Wrong source shard provenance")
        _, _, momenta, measure = scan.node_data(old, "source", i)
        if shard["momenta"] != list(momenta) or shard["measure"] != measure:
            raise ValueError("Source quadrature changed")
    rows = []
    for point_index, point in enumerate(old["points"]):
        point_id = point["point_id"]
        target_z = math.fsum(value for shard in target
                             for value in shard["values"][point_index]["sector_contributions"])
        target_q = target_z / point["target"]["Z_free"]**old["kappa"]
        if abs(target_q / prior[point_id]["target_Q"] - 1) > 1e-12:
            raise ValueError("Saved target reduction does not reproduce the baseline")
        for level in config["source_levels"]:
            z = math.fsum(shard["measure"] * next(
                value["source_Z_unscaled_M_node"] for value in shard["values"]
                if value["point_id"] == point_id and value["source_level"] == level) for shard in source)
            q = z / point["source"]["Z_free"]**old["kappa"]
            rows.append(dict(point_id=point_id, quadrature_order=old["orders"][0], source_level=level,
                             source_Z=z, source_Q=q, target_Q=target_q, source_over_target=q/target_q,
                             relative_disagreement=q/target_q-1,
                             previous_L3_source_over_target=prior[point_id]["source_over_target"],
                             source_change_from_previous_L3=q/prior[point_id]["source_Q"]-1))
    result = dict(schema=config["schema"], completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  config=config, complete_source_nodes=len(source),
                  native_pipeline_calls=sum(s["checks"]["native_pipeline_calls"] for s in source),
                  physical_PBW_used=False, comparisons=rows,
                  maximum_ward_residual=max(s["checks"]["branching_ward_residual"] for s in source),
                  maximum_analytic_error=max(s["checks"]["analytic_ground_half_level_max_error"] for s in source),
                  by_source_level=[dict(source_level=level,
                    maximum_relative_disagreement=max(abs(r["relative_disagreement"]) for r in rows if r["source_level"]==level),
                    maximum_source_change_from_previous_L3=max(abs(r["source_change_from_previous_L3"]) for r in rows if r["source_level"]==level))
                    for level in config["source_levels"]])
    trial.save(config_path.parent / "summary.json", result)
    print(json.dumps(result["by_source_level"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--levels", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.worker_config:
        worker(args.worker_config, args.index)
        return
    if not args.baseline or not args.output or args.workers < 1:
        parser.error("--baseline, --output, and a positive worker count are required")
    if args.levels != sorted(set(args.levels)) or min(args.levels) < 1 or max(args.levels) > 8:
        parser.error("Use distinct ascending source levels in 1..8")
    args.output = args.output.resolve()
    config = prepare(args.baseline, args.output, args.levels)
    path = args.output / "config.json"
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    def run(index):
        shard = args.output / "source" / f"node-{index:03d}.json"
        if shard.exists() and trial.load(shard)["config_digest"] == trial.digest(config):
            return index
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        with (log_dir / f"node-{index:03d}.log").open("w") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--worker-config", str(path), "--index", str(index)], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Source node {index} failed; see {log_dir / f'node-{index:03d}.log'}")
        return index
    count = len(scan.tasks(config["baseline_config"]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, index) for index in range(count)]
        for done, future in enumerate(as_completed(futures), 1):
            index = future.result()
            print(f"Completed source node {index}: {done}/{count}", flush=True)
    reduce(path)


if __name__ == "__main__":
    main()
