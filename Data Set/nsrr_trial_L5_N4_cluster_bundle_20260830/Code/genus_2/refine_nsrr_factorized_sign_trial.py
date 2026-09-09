#!/usr/bin/env python3
"""Raise numerical accuracy of the UNCHANGED, explicitly hypothetical NSRR trial.

The original trial, its archived data, and all checked kernels stay untouched.
Equal-sign blocks use branching and Virasoro c-recursions. Mixed signs retain
the explicit PBW diagnostic completion, here bounded to chiral level three.
This runner makes no new physical spin or antiholomorphic identification.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

import nsrr_factorized_sign_trial as trial

SCHEMA = "nsrr-factorized-sign-trial-refinement-v1"
FORMULA_KEYS = ("b", "channels", "lifts_geometry", "q_envelope", "vertex_ansatz",
                "antiholomorphic_ansatz", "formula", "control", "method",
                "cosmological_factor", "reference_free_spin", "normalization")


def fingerprint():
    paths = [Path(__file__), Path(trial._rules.__code__.co_filename),
             Path(trial.protected_hashes.__code__.co_filename)]
    return {str(p.resolve().relative_to(trial.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def make_config(baseline_dir, orders=(3, 4, 5), max_level=3):
    baseline_path = baseline_dir.resolve()/"config.json"
    baseline = trial.load(baseline_path)
    trial.validate_config(baseline)
    fresh = trial.make_config(baseline["geometry_path"])
    for old, new in zip(baseline["points"], fresh["points"]):
        if old["t"] != new["t"] or old["q_geometry"] != new["q_geometry"]:
            raise ValueError("plumbing changed under the baseline trial")
        if abs(old["Z_free_reference"]/new["Z_free_reference"]-1) > 1e-12:
            raise ValueError("the reference free factor changed")
    config = dict(baseline)
    config.update(schema=SCHEMA, max_level=max_level,
                  levels=[n/2 for n in range(2*max_level+1)],
                  quadrature_orders=list(orders), baseline_config_path=str(baseline_path),
                  baseline_config_digest=trial.digest(baseline),
                  refinement_implementation_sha256=fingerprint(),
                  fresh_geometry_and_free_audit=fresh["points"],
                  refinement_scope="numerics only; the original trial contraction is reused verbatim")
    validate_config(config)
    return config


def validate_config(config):
    if config["schema"] != SCHEMA:
        raise ValueError("wrong refinement schema")
    if config["implementation_sha256"] != trial.fingerprint() or \
            config["refinement_implementation_sha256"] != fingerprint():
        raise ValueError("refinement or original trial implementation changed")
    if config["protected_kernel_sha256"] != trial.protected_hashes():
        raise ValueError("protected kernel changed")
    baseline = trial.load(config["baseline_config_path"])
    trial.validate_config(baseline)
    if trial.digest(baseline) != config["baseline_config_digest"]:
        raise ValueError("baseline provenance changed")
    if any(config[k] != baseline[k] for k in FORMULA_KEYS) or config["points"] != baseline["points"]:
        raise ValueError("this refinement must not change the trial or geometry")
    if config["max_level"] not in (2, 3) or config["levels"] != [n/2 for n in range(2*config["max_level"]+1)]:
        raise ValueError("unsupported or incomplete level sweep")
    orders = config["quadrature_orders"]
    if len(orders) < 2 or orders != sorted(set(orders)) or any(n not in range(3, 7) for n in orders):
        raise ValueError("use at least two distinct ascending quadrature orders in 3..6")
    for key in ("physical_Z", "physical_Q", "physical_Ramond_projector", "physical_lift_spin_dictionary"):
        if config[key] is not None:
            raise ValueError("the trial must not be labelled a physical fixed-spin partition")


def block_components(b, momenta_slots, cutoff):
    if cutoff not in (2, 3):
        raise ValueError("the refined diagnostic completion is bounded to level two or three")
    if cutoff == 2:
        return trial.block_components(b, momenta_slots, cutoff)
    with patch.object(trial.dv, "HumanNSRRThetaOracle", wraps=trial.dv.HumanNSRRThetaOracle) as oracle:
        runtime = trial.dv.NSRRDoubleVirasoroTheta(
            b=b, physical_momenta=momenta_slots, cutoff=cutoff,
            completion="pbw_diagnostic", pbw_completion_max_level=3)
        components = {channel: runtime.physical_components(*channel) for channel in trial.CHANNELS}
        calls = oracle.call_count
    if calls != 4:
        raise ArithmeticError("expected exactly four explicitly requested PBW completions")
    error = 0.
    for channel, vectors in components.items():
        for lifts in product((1, -1), repeat=3):
            k = trial.dv.spin_character_index(lifts)
            expected = trial.low_level_coefficients(b, momenta_slots, *channel, lifts)
            for exponent, target in zip(((0, 0, 0), (1, 0, 0)), expected):
                actual = trial.fwht(vectors[exponent])[k]
                error = max(error, abs(actual-target)/max(1., abs(target)))
    if error > 1e-10 or runtime.ward_residual_maximum > 1e-8:
        raise ArithmeticError("low-level or branching Ward check failed")
    return components, {"explicit_PBW_completion_calls": calls,
                        "analytic_ground_half_level_max_error": error,
                        "branching_ward_residual": runtime.ward_residual_maximum}


def baseline_node_check(config, shard):
    """At shared N=3 nodes compare every old block and weighted total through L=2."""
    baseline = trial.load(config["baseline_config_path"])
    key = (shard["quadrature_order"], shard["node"])
    if key not in trial.tasks(baseline):
        return None
    old_index = trial.tasks(baseline).index(key)
    old = trial.load(Path(config["baseline_config_path"]).parent/"shards"/f"node-{old_index:03d}.json")
    trial.validate_shard(baseline, old_index, old)
    if old["momenta_geometry"] != list(shard["momenta_geometry"]) or old["measure"] != shard["measure"]:
        raise ValueError("shared quadrature grid changed")
    lookup = {(r["t"], r["level"], tuple(r["lifts_geometry"])): r for r in shard["rows"]}
    block_error = total_error = 0.
    for old_row in old["rows"]:
        new_row = lookup[old_row["t"], old_row["level"], tuple(old_row["lifts_geometry"])]
        for a, b in zip(old_row["blocks"], new_row["blocks"]):
            a, b = trial.decode(a), trial.decode(b)
            block_error = max(block_error, abs(a-b)/max(1., abs(a)))
        a, b = trial.decode(old_row["total"]), trial.decode(new_row["total"])
        total_error = max(total_error, abs(a-b)/max(abs(a), 1e-280))
    if max(block_error, total_error) > 1e-9:
        raise ArithmeticError("refined runner does not reproduce the previous low-order trial")
    return {"block_scaled_error": block_error, "total_relative_error": total_error,
            "baseline_shard_index": old_index}


def evaluate_node(config, index):
    validate_config(config)
    started = time.monotonic()
    n, node = trial.tasks(config)[index]
    indices = np.unravel_index(node, (n,)*3)
    rules = trial._rules(config["q_envelope"], n)
    momenta = tuple(float(rules[e][0][indices[e]]) for e in range(3))
    constants = trial.GenericSuperLiouvilleConstants(config["b"], dps=30)
    c = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    components, checks = block_components(config["b"], momenta[::-1], config["max_level"])
    rows = []
    for point, lifts, level in product(config["points"], trial.LIFTS, config["levels"]):
        plumbing = trial.NSRRPlumbingInputs(tuple(complex(z) for z in point["q_geometry"]),
                                            lifts, trial.GEOMETRY_SECTORS)
        primary = plumbing.primary(config["b"], momenta)
        blocks = trial.evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, level)
        anti = {k: z.conjugate() for k, z in blocks.items()}
        result = trial.contract(blocks, anti, c)
        wrong_sign = trial.contract(blocks, anti, c, sewing_sign=False)["total"]
        formal_anti = trial.evaluate_blocks(components, tuple(z.conjugate() for z in plumbing.q_slots),
                                            plumbing.lifts_slots, level)
        formal = trial.contract(blocks, formal_anti, c)["total"]
        rows.append({"t": point["t"], "level": level, "lifts_geometry": list(lifts),
                     "primary": trial.encode(primary), "blocks": [trial.encode(blocks[k]) for k in trial.CHANNELS],
                     "weighted_terms": [trial.encode(abs(primary)**2*result["terms"][k]) for k in trial.CHANNELS],
                     **{key: trial.encode(abs(primary)**2*result[key]) for key in ("even", "odd", "equal", "mixed", "total")},
                     "without_sewing_sign": trial.encode(abs(primary)**2*wrong_sign),
                     "formal_same_convention_tilde": trial.encode(abs(primary)**2*formal)})
    shard = {"schema": SCHEMA, "config_digest": trial.digest(config), "index": index,
             "quadrature_order": n, "node": node, "momenta_geometry": momenta,
             "momenta_slots": momenta[::-1], "measure": trial._measure(rules, indices),
             "C_BRY": [trial.encode(z) for z in c], "checks": checks, "rows": rows}
    checks["baseline_L2_reproduction"] = baseline_node_check(config, shard)
    shard["elapsed_seconds"] = time.monotonic()-started
    validate_config(config)
    return shard


def reduced_rows(config, shards):
    kappa = 1+2*(config["b"]+1/config["b"])**2
    rows = []
    for n in config["quadrature_orders"]:
        selected = [s for s in shards if s["quadrature_order"] == n]
        if len(selected) != n**3:
            raise ValueError("incomplete momentum integration")
        for j, design in enumerate(selected[0]["rows"]):
            row = {key: design[key] for key in ("t", "level", "lifts_geometry")}
            row["quadrature_order"] = n
            for key in ("even", "odd", "equal", "mixed", "total", "without_sewing_sign", "formal_same_convention_tilde"):
                values = [s["measure"]*trial.decode(s["rows"][j][key]) for s in selected]
                z = complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))
                row[key] = trial.encode(z) if key == "formal_same_convention_tilde" else trial.real_value(z)
            free = next(p["Z_free_reference"] for p in config["points"] if p["t"] == row["t"])
            row["Q_trial_reference"] = row["total"]/free**kappa
            row["mixed_fraction"] = row["mixed"]/row["total"]
            rows.append(row)
    return rows


def reduce_run(output):
    config = trial.load(output/"config.json")
    validate_config(config)
    shards = [trial.load(output/"shards"/f"node-{i:03d}.json") for i in range(len(trial.tasks(config)))]
    for i, shard in enumerate(shards):
        trial.validate_shard(config, i, shard)
    rows = reduced_rows(config, shards)
    fine_n, coarse_n = config["quadrature_orders"][-1], config["quadrature_orders"][-2]
    fine_l = config["max_level"]
    diagnostics = []
    for point in config["points"]:
        def select(n, level):
            return next(r for r in rows if r["t"] == point["t"] and r["quadrature_order"] == n
                        and r["level"] == level and r["lifts_geometry"] == [1, 1, 1])
        fine, coarse, low = select(fine_n, fine_l), select(coarse_n, fine_l), select(fine_n, fine_l-1)
        diagnostics.append({"t": point["t"], "Z_trial": fine["total"],
                            "Q_trial_reference": fine["Q_trial_reference"],
                            "level_coarse": fine_l-1, "level_fine": fine_l,
                            "level_relative_change": fine["total"]/low["total"]-1,
                            "quadrature_coarse": coarse_n, "quadrature_fine": fine_n,
                            "quadrature_relative_change": fine["total"]/coarse["total"]-1,
                            "mixed_fraction": fine["mixed_fraction"]})
    baseline_checks = [s["checks"]["baseline_L2_reproduction"] for s in shards
                       if s["checks"]["baseline_L2_reproduction"] is not None]
    result = {"schema": SCHEMA, "config": config, "rows": rows, "diagnostics": diagnostics,
              "checks": {"protected_kernel_sha256": trial.protected_hashes(),
                         "analytic_max_error": max(s["checks"]["analytic_ground_half_level_max_error"] for s in shards),
                         "ward_residual_maximum": max(s["checks"]["branching_ward_residual"] for s in shards),
                         "explicit_PBW_completion_calls": sum(s["checks"]["explicit_PBW_completion_calls"] for s in shards),
                         "baseline_shared_node_count": len(baseline_checks),
                         "baseline_block_scaled_error": max((c["block_scaled_error"] for c in baseline_checks), default=None),
                         "baseline_total_relative_error": max((c["total_relative_error"] for c in baseline_checks), default=None)},
              "physical_Z": None, "physical_Q": None}
    trial.save(output/"summary.json", result)
    with (output/"fivepoint_trial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    return result


def run(args):
    output = args.output_dir.resolve()
    config_path = output/"config.json"
    if config_path.exists():
        config = trial.load(config_path)
        validate_config(config)
        if config["quadrature_orders"] != args.orders or config["max_level"] != args.max_level:
            raise ValueError("do not reuse a directory with different accuracy settings")
    else:
        config = make_config(args.baseline_dir, args.orders, args.max_level)
        trial.save(config_path, config)
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def worker(index):
        with (logs/f"node-{index:03d}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output-dir", str(output),
                            "--index", str(index)], stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        return index
    print(f"Refining unchanged NSRR trial: L={config['max_level']}, N={config['quadrature_orders']}, "
          f"{len(trial.tasks(config))} nodes, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, i) for i in range(len(trial.tasks(config)))]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                index = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            print(f"{completed}/{len(futures)} nodes complete; last={index}", flush=True)
    result = reduce_run(output)
    for row in result["diagnostics"]:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--max-level", type=int, choices=(2, 3), default=3)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    for name in ("worker", "reduce"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, required=True)
        if name == "worker":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "worker":
        config = trial.load(args.output_dir/"config.json")
        validate_config(config)
        path = args.output_dir/"shards"/f"node-{args.index:03d}.json"
        if path.exists():
            trial.validate_shard(config, args.index, trial.load(path))
        else:
            shard = evaluate_node(config, args.index)
            trial.validate_shard(config, args.index, shard)
            trial.save(path, shard)
    else:
        result = reduce_run(args.output_dir)
        print(result["diagnostics"])


if __name__ == "__main__":
    main()
