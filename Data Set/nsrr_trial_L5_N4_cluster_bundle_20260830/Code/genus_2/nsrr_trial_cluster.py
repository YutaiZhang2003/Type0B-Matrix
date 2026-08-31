#!/usr/bin/env python3
"""Portable, three-hour-bounded L=5 NSRR trial; not a physical spin partition.

The N=4 nodes, vertex coefficients, plumbing, and L=3 reference are frozen
from the audited local trial. No protected kernel or old runner is changed.
Equal signs use branching plus two Virasoro c-recursions; mixed signs use
explicit PBW diagnostic completion through the SAME total chiral level five.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
from itertools import product
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

import nsrr_factorized_sign_trial as trial
import refine_nsrr_factorized_sign_trial as previous

SCHEMA = "nsrr-L5-bounded-trial-v1"


def frozen_inputs_digest(config):
    keys = previous.FORMULA_KEYS+("points", "reference_nodes", "reference_allNS", "geometry")
    return trial.digest({k: config[k] for k in keys})


def implementation_hashes():
    return {**trial.fingerprint(), **previous.fingerprint(),
            str(Path(__file__).relative_to(trial.ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def prepare(reference_dir, output, order=4):
    old = trial.load(reference_dir/"summary.json")
    previous.validate_config(old["config"])
    verification = trial.load(reference_dir/"verification.json")
    comparison = trial.load(reference_dir/"comparison.json")
    if verification["summary_digest"] != trial.digest(old) or comparison["refined_summary_digest"] != trial.digest(old):
        raise ValueError("reference audit provenance mismatch")
    geometry = trial.load(old["config"]["geometry_path"])
    if hashlib.sha256(Path(old["config"]["geometry_path"]).read_bytes()).hexdigest() != old["config"]["geometry_sha256"]:
        raise ValueError("reference geometry changed")
    config = {key: old["config"][key] for key in previous.FORMULA_KEYS}
    config.update(schema=SCHEMA, max_level=5, levels=[n/2 for n in range(11)], quadrature_orders=[order],
                  points=old["config"]["points"], implementation_sha256=implementation_hashes(),
                  protected_kernel_sha256=trial.protected_hashes(), source_summary_digest=trial.digest(old),
                  source_config_digest=trial.digest(old["config"]),
                  reference_comparison_digest=trial.digest(comparison),
                  reference_allNS=comparison["rows"], geometry=geometry,
                  physical_Z=None, physical_Q=None, physical_Ramond_projector=None,
                  physical_lift_spin_dictionary=None,
                  resources={"cpus": 8, "memory_gib": 16, "wall_seconds": 10800,
                             "compute_seconds": 9900, "node_timeout_seconds": 900},
                  coefficients_policy=f"reuse the audited N={order} BRY coefficients unchanged; two vertex products, not absolute squares",
                  reference_nodes=[])
    for index, (n, node) in enumerate(trial.tasks(old["config"])):
        if n != order:
            continue
        shard = trial.load(reference_dir/"shards"/f"node-{index:03d}.json")
        trial.validate_shard(old["config"], index, shard)
        config["reference_nodes"].append({
            "node": node, "momenta_geometry": shard["momenta_geometry"],
            "measure": shard["measure"], "C_BRY": shard["C_BRY"],
            "source_shard_digest": trial.digest(shard),
            "L3_rows": [{k: r[k] for k in ("t", "level", "lifts_geometry", "blocks", "total")}
                        for r in shard["rows"] if r["level"] == 3]})
    config["frozen_inputs_digest"] = frozen_inputs_digest(config)
    validate_config(config)
    if output.exists():
        if trial.load(output) != config:
            raise FileExistsError("refusing to replace a different frozen configuration")
    else:
        trial.save(output, config)
    return config


def validate_config(config):
    if config["schema"] != SCHEMA or config["implementation_sha256"] != implementation_hashes():
        raise ValueError("cluster trial implementation/configuration mismatch")
    if config["frozen_inputs_digest"] != frozen_inputs_digest(config):
        raise ValueError("frozen sewing assumptions, geometry, coefficients, or reference changed")
    if config["protected_kernel_sha256"] != trial.protected_hashes():
        raise ValueError("protected kernel changed")
    if config["max_level"] != 5 or config["levels"] != [n/2 for n in range(11)] or config["quadrature_orders"] not in ([3], [4]):
        raise ValueError("this bounded toy supports L=5, N=3 or N=4")
    order = config["quadrature_orders"][0]
    if config["channels"] != [list(c) for c in trial.CHANNELS] or config["lifts_geometry"] != [list(l) for l in trial.LIFTS]:
        raise ValueError("do not omit mixed channels or spin controls")
    if len(config["points"]) != 5 or len(config["reference_nodes"]) != order**3:
        raise ValueError("five surfaces and the complete quadrature grid are required")
    if config["resources"] != {"cpus": 8, "memory_gib": 16, "wall_seconds": 10800,
                               "compute_seconds": 9900, "node_timeout_seconds": 900}:
        raise ValueError("the approved three-hour resource boundary changed")
    for key in ("physical_Z", "physical_Q", "physical_Ramond_projector", "physical_lift_spin_dictionary"):
        if config[key] is not None:
            raise ValueError("this remains the same hypothetical nonchiral trial")
    rules = trial._rules(config["q_envelope"], order)
    for i, node in enumerate(config["reference_nodes"]):
        indices = np.unravel_index(i, (order,)*3)
        expected = [float(rules[e][0][indices[e]]) for e in range(3)]
        if node["node"] != i or not np.allclose(node["momenta_geometry"], expected, rtol=1e-12, atol=1e-14):
            raise ValueError("cached quadrature node is inconsistent")
        if not math.isclose(node["measure"], trial._measure(rules, indices), rel_tol=1e-12):
            raise ValueError("cached quadrature measure is inconsistent")
        design = [(p["t"], list(l)) for p, l in product(config["points"], trial.LIFTS)]
        if [(r["t"], r["lifts_geometry"]) for r in node["L3_rows"]] != design:
            raise ValueError("incomplete L=3 reference")


def preflight(config):
    validate_config(config)
    errors = []
    for point, reference in zip(config["geometry"]["points"], config["reference_allNS"]):
        if point["t"] != reference["t"]:
            raise ValueError("target reference ordering changed")
        for label, spin, branch, expected in (
            ("source_chart", ((1, 1), (0, 0)), trial.SOURCE_BRANCH, reference["source_free"]),
            ("target_chart", ((0, 0), (0, 0)), ((-1, -1), (-1, 0)), reference["target_free"])):
            chart = point[label]
            value = trial.fixed_spin_partition(
                tuple(complex(z) for z in chart["q_values"]),
                np.array([[complex(z) for z in r] for r in chart["omega"]]),
                spin, period_branch=branch, max_mode=32)
            errors.append(abs(value["Z_free"]/expected-1))
    if max(errors) > 1e-11:
        raise ArithmeticError("same-plumbing free factor changed")
    return {"protected_kernels": len(trial.protected_hashes()),
            "free_factor_max_relative_error": max(errors), "nodes": len(config["reference_nodes"]), "max_level": 5}


def block_components(config, momenta):
    started = time.monotonic()
    timings = []
    with patch.object(trial.dv, "HumanNSRRThetaOracle", wraps=trial.dv.HumanNSRRThetaOracle) as oracle:
        runtime = trial.dv.NSRRDoubleVirasoroTheta(
            b=config["b"], physical_momenta=momenta[::-1], cutoff=5,
            completion="pbw_diagnostic", pbw_completion_max_level=5)
        print(f"branching and Virasoro setup: {time.monotonic()-started:.3f}s", flush=True)
        components = {}
        for channel in trial.CHANNELS:
            tick = time.monotonic()
            components[channel] = runtime.physical_components(*channel)
            timings.append({"channel": list(channel), "seconds": time.monotonic()-tick})
            print(f"channel {channel}: {timings[-1]['seconds']:.3f}s", flush=True)
        calls = oracle.call_count
    error = 0.
    for channel, vectors in components.items():
        for lifts in product((1, -1), repeat=3):
            k = trial.dv.spin_character_index(lifts)
            expected = trial.low_level_coefficients(config["b"], momenta[::-1], *channel, lifts)
            for exponent, target in zip(((0, 0, 0), (1, 0, 0)), expected):
                error = max(error, abs(trial.fwht(vectors[exponent])[k]-target)/max(1., abs(target)))
    if calls != 4 or error > 1e-10 or runtime.ward_residual_maximum > 1e-8:
        raise ArithmeticError("completion count, low-level identity, or branching Ward check failed")
    return components, {"explicit_PBW_completion_calls": calls, "analytic_max_error": error,
                        "branching_ward_residual": runtime.ward_residual_maximum,
                        "channel_timings": timings}


def evaluate_node(config, index):
    validate_config(config)
    started = time.monotonic()
    node = config["reference_nodes"][index]
    momenta = tuple(node["momenta_geometry"])
    c = tuple(trial.decode(z) for z in node["C_BRY"])
    components, checks = block_components(config, momenta)
    rows = []
    for point, lifts, level in product(config["points"], trial.LIFTS, config["levels"]):
        plumbing = trial.NSRRPlumbingInputs(tuple(complex(z) for z in point["q_geometry"]), lifts, trial.GEOMETRY_SECTORS)
        primary = plumbing.primary(config["b"], momenta)
        blocks = trial.evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, level)
        anti = {k: z.conjugate() for k, z in blocks.items()}
        result = trial.contract(blocks, anti, c)
        wrong_sign = trial.contract(blocks, anti, c, sewing_sign=False)["total"]
        formal = trial.contract(blocks, trial.evaluate_blocks(components,
            tuple(z.conjugate() for z in plumbing.q_slots), plumbing.lifts_slots, level), c)["total"]
        rows.append({"t": point["t"], "level": level, "lifts_geometry": list(lifts),
                     "primary": trial.encode(primary), "blocks": [trial.encode(blocks[k]) for k in trial.CHANNELS],
                     "weighted_terms": [trial.encode(abs(primary)**2*result["terms"][k]) for k in trial.CHANNELS],
                     **{key: trial.encode(abs(primary)**2*result[key]) for key in ("even", "odd", "equal", "mixed", "total")},
                     "without_sewing_sign": trial.encode(abs(primary)**2*wrong_sign),
                     "formal_same_convention_tilde": trial.encode(abs(primary)**2*formal)})
    error = total_error = 0.
    for new, old in zip((r for r in rows if r["level"] == 3), node["L3_rows"]):
        for a, b in zip(new["blocks"], old["blocks"]):
            error = max(error, abs(trial.decode(a)-trial.decode(b))/max(1., abs(trial.decode(b))))
        total_error = max(total_error, abs(trial.decode(new["total"])/trial.decode(old["total"])-1))
    if max(error, total_error) > 2e-8:
        raise ArithmeticError("L=5 calculation does not reproduce the frozen L=3 trial")
    checks.update(L3_block_scaled_error=error, L3_total_relative_error=total_error)
    validate_config(config)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"schema": SCHEMA, "config_digest": trial.digest(config), "index": index,
            "quadrature_order": config["quadrature_orders"][0], "node": index, "momenta_geometry": momenta,
            "momenta_slots": momenta[::-1], "measure": node["measure"], "C_BRY": node["C_BRY"],
            "checks": checks, "rows": rows, "elapsed_seconds": time.monotonic()-started,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss*1024}


def audit_shard(config, index, shard):
    trial.validate_shard(config, index, shard)
    node = config["reference_nodes"][index]
    if shard["momenta_geometry"] != node["momenta_geometry"] or shard["measure"] != node["measure"] or shard["C_BRY"] != node["C_BRY"]:
        raise ValueError("shard changed the frozen integration data")
    error = 0.
    c = [trial.decode(z)/2 for z in shard["C_BRY"]]
    for row in shard["rows"]:
        primary2 = abs(trial.decode(row["primary"]))**2
        terms = [primary2*c[0 if a == 1 else 1]*c[0 if b == 1 else 1]*abs(trial.decode(z))**2
                 for (_, a, b), z in zip(trial.CHANNELS, row["blocks"])]
        expected = sum(terms)
        error = max(error, abs(expected-trial.decode(row["total"]))/max(abs(expected), 1e-280))
        for a, b in zip(terms, row["weighted_terms"]):
            error = max(error, abs(a-trial.decode(b))/max(abs(expected), abs(a), 1e-280))
    if error > 1e-11:
        raise ArithmeticError("independent contraction reconstruction failed")
    return error


def reduce_run(config, output):
    validate_config(config)
    order = config["quadrature_orders"][0]
    shards = [trial.load(output/"shards"/f"node-{i:03d}.json") for i in range(order**3)]
    errors = [audit_shard(config, i, s) for i, s in enumerate(shards)]
    rows = previous.reduced_rows(config, shards)
    diagnostics = []
    for point, reference in zip(config["points"], config["reference_allNS"]):
        selected = {level: next(r for r in rows if r["t"] == point["t"] and r["level"] == level
                               and r["lifts_geometry"] == [1, 1, 1]) for level in (3, 4, 5)}
        qn = reference["allNS_Q_diagnostic"]
        diagnostics.append({"t": point["t"], "Q_allNS_diagnostic": qn,
                            **{f"Q_trial_L{l}_N{order}": selected[l]["Q_trial_reference"] for l in (3, 4, 5)},
                            "Z_trial_L5": selected[5]["total"],
                            "L3_to_L4_relative": selected[4]["total"]/selected[3]["total"]-1,
                            "L4_to_L5_relative": selected[5]["total"]/selected[4]["total"]-1,
                            "trial_over_allNS_minus_one": selected[5]["Q_trial_reference"]/qn-1})
    result = {"schema": SCHEMA, "config": config, "rows": rows, "diagnostics": diagnostics,
              "maximum_independent_term_error": max(errors),
              "maximum_L3_block_reproduction_error": max(s["checks"]["L3_block_scaled_error"] for s in shards),
              "maximum_branching_ward_residual": max(s["checks"]["branching_ward_residual"] for s in shards),
              "protected_kernel_sha256": trial.protected_hashes(), "physical_Z": None, "physical_Q": None}
    trial.save(output/"summary.json", result)
    with (output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    return result


def run(config_path, output, workers):
    started = time.monotonic()
    config = trial.load(config_path)
    node_count = len(config["reference_nodes"])
    if not 1 <= workers <= config["resources"]["cpus"]:
        raise ValueError("worker count exceeds the eight-core allocation")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(preflight(config), flush=True)
        deadline = started+config["resources"]["compute_seconds"]
        (output/"logs").mkdir(exist_ok=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        completed = []
        def worker(index):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("compute budget exhausted; existing shards preserved")
            with (output/"logs"/f"node-{index:03d}.log").open("a") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config_path),
                                "--output-dir", str(output), "--index", str(index)], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=min(config["resources"]["node_timeout_seconds"], remaining))
            return index
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(worker, i) for i in range(node_count)]
                try:
                    for future in as_completed(futures):
                        completed.append(future.result())
                        trial.save(output/"status.json", {"status": "running", "completed_nodes": len(completed),
                            "required_nodes": node_count, "elapsed_seconds": time.monotonic()-started})
                        print(f"{len(completed)}/{node_count} nodes complete", flush=True)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
            result = reduce_run(config, output)
        except Exception as error:
            trial.save(output/"status.json", {"status": "incomplete", "error": str(error),
                "completed_nodes": len(completed), "required_nodes": node_count, "elapsed_seconds": time.monotonic()-started})
            raise
        trial.save(output/"status.json", {"status": "complete", "completed_nodes": node_count,
            "elapsed_seconds": time.monotonic()-started, "summary_digest": trial.digest(result)})
        print(result["diagnostics"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order", type=int, choices=(3, 4), default=4)
    for name in ("preflight", "worker", "run", "reduce"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name != "preflight":
            p.add_argument("--output-dir", type=Path, required=True)
        if name == "worker":
            p.add_argument("--index", type=int, required=True, choices=range(64))
        if name == "run":
            p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        c = prepare(args.reference_dir, args.output, args.order)
        print({"config_digest": trial.digest(c), "nodes": len(c["reference_nodes"]), "resources": c["resources"]})
    elif args.command == "run":
        run(args.config.resolve(), args.output_dir.resolve(), args.workers)
    else:
        c = trial.load(args.config)
        validate_config(c)
        if args.command == "preflight":
            print(preflight(c))
        elif args.command == "reduce":
            print(reduce_run(c, args.output_dir)["diagnostics"])
        else:
            path = args.output_dir/"shards"/f"node-{args.index:03d}.json"
            if path.exists():
                audit_shard(c, args.index, trial.load(path))
            else:
                shard = evaluate_node(c, args.index)
                # Normalize tuple/list representation before provenance checks.
                shard["momenta_geometry"] = list(shard["momenta_geometry"])
                audit_shard(c, args.index, shard)
                trial.save(path, shard)
            print({"complete": True, "index": args.index}, flush=True)


if __name__ == "__main__":
    main()
