#!/usr/bin/env python3
"""Run a small local toy comparison before committing to the larger scan.

No cluster submission or automatic precision escalation. The default toy
uses physical levels 2/3 and quadrature orders 3/4. The explicitly selected
fivepoint-l4 design raises these to levels 3/4 and quadrature orders 4/5.
Each node gets a fresh process, with bounded concurrent children.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import nsrr_nsnsns_theta_omega_scan as scan


def toy_config(base):
    config = copy.deepcopy(base)
    config["experiment"] = "toy-first feasibility check; no automatic accuracy escalation"
    config["points"] = [p for p in config["points"] if p["t"] in (.56, .60, .64)]
    config["quadrature_orders"] = [3, 4]
    config["source_physical_levels"] = [2, 3]
    config["target_physical_level"] = 3
    config["target_recursion_order_twice_level"] = 6
    config["numerics"].update(structure_precision=25, block_working_precision=40,
                              global_tolerance=1e-6, global_max_total_occupation=28)
    config["quadrature_reference_abs_q"] = {
        channel: [max(abs(complex(p["charts"][channel]["q_values"][edge]))
                      for p in config["points"]) for edge in range(3)]
        for channel in scan.CHANNELS}
    scan.validate_config(config)
    return config


def refined_fivepoint_config(base):
    """One-order local refinement, only when explicitly selected."""
    config = copy.deepcopy(base)
    config["experiment"] = "user-requested local five-point one-order refinement"
    config["quadrature_orders"] = [4, 5]
    config["source_physical_levels"] = [3, 4]
    config["target_physical_level"] = 4
    config["target_recursion_order_twice_level"] = 8
    config["numerics"].update(structure_precision=30, block_working_precision=40,
                              global_tolerance=1e-7, global_max_total_occupation=36)
    config["quadrature_reference_abs_q"] = {
        channel: [max(abs(complex(p["charts"][channel]["q_values"][edge]))
                      for p in config["points"]) for edge in range(3)]
        for channel in scan.CHANNELS}
    scan.validate_config(config)
    if [p["t"] for p in config["points"]] != [.52, .56, .60, .64, .68]:
        raise ValueError("the refined five-point design requires its certified Omega grid")
    return config


def run(base_config, output_dir, workers, design="toy"):
    output_dir = output_dir.resolve()
    factory = {"toy": toy_config, "fivepoint-l4": refined_fivepoint_config}[design]
    config = factory(scan._load(base_config))
    config_path = output_dir / "config.json"
    if config_path.exists() and scan._digest(scan._load(config_path)) != scan._digest(config):
        raise RuntimeError("run directory contains a different config; use a new output directory")
    scan.write_json(config_path, config)
    shards = output_dir / "shards"
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    count = len(scan.tasks(config))
    started = time.perf_counter()

    def evaluate(index):
        with (logs / f"task-{index:06d}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(Path(scan.__file__).resolve()), "--config", str(config_path),
                 "worker", "--output-dir", str(shards), "--task-index", str(index)],
                env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"local task {index} failed; see {logs / f'task-{index:06d}.log'}")
        return index

    print(f"{design}: {len(config['points'])} period points, {count} nodes, {workers} fresh-process workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, index) for index in range(count)]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                index = future.result()
                if completed % 10 == 0 or completed == count:
                    print(f"{completed}/{count} nodes; elapsed={time.perf_counter()-started:.1f}s; last={index}", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    result = scan.reduce_scan(config_path, shards, output_dir / "summary.json")
    print(f"{design} completed in {time.perf_counter()-started:.1f}s", flush=True)
    for row in result["convergence_diagnostics"]:
        print(row, flush=True)
    print(output_dir / "summary.svg", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path(__file__).parents[1] / "config/nsrr_nsnsns_theta_omega_scan_20260830.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--design", choices=("toy", "fivepoint-l4"), default="toy")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("use between one and four concurrent node workers")
    run(args.base_config, args.output_dir, args.workers, args.design)


if __name__ == "__main__":
    main()
