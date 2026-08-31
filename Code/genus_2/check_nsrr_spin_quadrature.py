#!/usr/bin/env python3
"""Fixed-formula momentum convergence; no physical spin assignment is inferred.

Use the existing R16 all-NS evaluator and the unchanged L3 NSRR trial. Each
node is a fresh subprocess. Only literal one-module Ward actions are cached;
all three-module Ward solutions and all conformal blocks are recalculated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

import recompute_all_ns_reference as fresh
import refine_nsrr_factorized_sign_trial as refine
from nsrr_branching_cutoff_probe import cached_actions

trial = refine.trial
ROOT = trial.ROOT
SCHEMA = "nsrr-spin-quadrature-diagnostic-v1"
DEFAULT_OUTPUT = ROOT / "Data Set/nsrr_spin_quadrature_t060_20260830"
REFERENCES = {
    "target": ROOT / "Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830",
    "source": ROOT / "Data Set/nsrr_factorized_sign_trial_L3_N5_20260830",
    "free": ROOT / "Data Set/fixed_spin_free_NSrr_20260830",
}


def fingerprint():
    paths = [Path(__file__), Path(refine.__file__), Path(trial.__file__),
             Path(fresh.__file__), Path(cached_actions.__code__.co_filename),
             Path(trial.dv.__file__), ROOT/"Human Notes/SCblock.tex"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(output, t=.60, target_orders=(6, 7), source_orders=(6,)):
    target = trial.load(REFERENCES["target"]/"config.json")
    source = trial.load(REFERENCES["source"]/"config.json")
    fresh.validate_config(target)
    refine.validate_config(source)
    free = trial.load(REFERENCES["free"]/"summary.json")
    point = next(p for p in free["points"] if p["t"] == t)
    config = {"schema": SCHEMA, "t": t, "target": target, "source": source,
              "target_orders": list(target_orders), "source_orders": list(source_orders),
              "target_point": next(p for p in target["points"] if p["t"] == t),
              "source_point": next(p for p in source["points"] if p["t"] == t),
              "Z_free_target": point["target_NSnsns"]["Z_free"],
              "Z_free_source": point["source_NSrr"]["Z_free"],
              "references": {k: str(v) for k, v in REFERENCES.items()},
              "reference_summary_digests": {k: trial.digest(trial.load(v/"summary.json")) for k, v in REFERENCES.items()},
              "implementation": fingerprint(), "protected": fresh.protected_hashes(),
              "scope": "fixed-cutoff numerical diagnostic, not an established fixed-spin Liouville comparison",
              "physical_Q_NSrr": None}
    validate(config)
    path = output/"config.json"
    if path.exists() and trial.load(path) != config:
        raise ValueError("refuse to overwrite a different frozen run")
    trial.save(path, config)
    return config


def validate(c):
    if c["schema"] != SCHEMA or c["implementation"] != fingerprint():
        raise ValueError("implementation provenance mismatch")
    if c["protected"] != fresh.protected_hashes():
        raise ValueError("checked kernel changed")
    fresh.validate_config(c["target"])
    refine.validate_config(c["source"])
    if c["target"]["recursion_order_twice_level"] != 16 or c["source"]["max_level"] != 3:
        raise ValueError("block cutoffs must stay fixed at R16 / L3")
    if c["physical_Q_NSrr"] is not None:
        raise ValueError("uncertified spin assignment")
    for channel in ("target", "source"):
        ns = c[channel+"_orders"]
        if not ns or ns != sorted(set(ns)) or any(n not in range(3, 11) for n in ns):
            raise ValueError("invalid quadrature sweep")
        if c[channel+"_point"] != next(p for p in c[channel]["points"] if p["t"] == c["t"]):
            raise ValueError("changed plumbing point")


def node_data(c, channel, n, index):
    if channel not in ("source", "target") or not 0 <= index < n**3:
        raise ValueError("invalid node")
    envelope = c[channel]["q_envelope" if channel == "source" else "quadrature_reference_abs_q"]
    rules = trial._rules(envelope, n)
    indices = np.unravel_index(index, (n,)*3)
    return tuple(float(rules[e][0][indices[e]]) for e in range(3)), trial._measure(rules, indices)


def evaluate(c, channel, n, index, cache_dir=None):
    validate(c)
    started = time.monotonic()
    momenta, measure = node_data(c, channel, n, index)
    point = c[channel+"_point"]
    if channel == "target":
        p, num = c["target"]["parameters"], c["target"]["numerics"]
        constants = fresh.scan.GenericSuperLiouvilleConstants(
            p["b"], dps=num["structure_precision"], mu=complex(p["mu"]),
            include_cosmological_prefactor=p["include_cosmological_prefactor"])
        q = tuple(map(complex, point["q_values"]))
        recursion = fresh.scan.NSGenus2CRecursion(
            channel="theta", q_values=q, global_method="resummed",
            global_tolerance=num["global_tolerance"],
            global_max_total_occupation=num["global_max_total_occupation"],
            vacuum_word_length=num["vacuum_word_length"], vacuum_max_mode=num["vacuum_max_mode"])
        sectors = fresh.scan.all_ns_node(
            b=p["b"], q_values=q, lifts=point["lifts"], recursion_order=16,
            momenta=momenta, measure=measure, constants=constants, recursion=recursion,
            block_method="collision_aware_mp", block_working_precision=num["block_working_precision"])
        checks = {"global_nonconverged_calls": recursion.global_nonconverged_calls,
                  "global_max_occupation_used": recursion.global_max_used,
                  "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative}
        if checks["global_nonconverged_calls"]:
            raise ArithmeticError("global resummation failed")
        rows = [{"lifts": point["lifts"], "sectors_weighted": list(sectors),
                 "Z_weighted": math.fsum(sectors)}]
    else:
        b = c["source"]["b"]
        constants = trial.GenericSuperLiouvilleConstants(b, dps=30)
        coefficients = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
        if cache_dir is None:
            components, checks = refine.block_components(b, momenta[::-1], 3)
        else:
            # A process-local exact memoization wrapper, never a source edit.
            with patch.object(trial.dv.BranchingGrid, "build_actions", lambda grid: cached_actions(grid, cache_dir)):
                components, checks = refine.block_components(b, momenta[::-1], 3)
        rows = []
        for lifts in trial.LIFTS:
            plumbing = trial.NSRRPlumbingInputs(tuple(map(complex, point["q_geometry"])), lifts, trial.GEOMETRY_SECTORS)
            primary = plumbing.primary(b, momenta)
            blocks = trial.evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, 3)
            contraction = trial.contract(blocks, {k: z.conjugate() for k, z in blocks.items()}, coefficients)
            factor = measure*abs(primary)**2
            values = {key+"_weighted": trial.real_value(factor*contraction[key])
                      for key in ("total", "even", "odd", "equal", "mixed")}
            rows.append({"lifts": list(lifts), "Z_weighted": values.pop("total_weighted"),
                         "blocks": [trial.encode(blocks[k]) for k in trial.CHANNELS], **values})
    return {"schema": SCHEMA, "config_digest": trial.digest(c), "channel": channel,
            "quadrature_order": n, "index": index, "momenta": list(momenta), "measure": measure,
            "rows": rows, "checks": checks, "seconds": time.monotonic()-started}


def validate_shard(c, channel, n, index, shard):
    momenta, measure = node_data(c, channel, n, index)
    for k, v in {"schema": SCHEMA, "config_digest": trial.digest(c), "channel": channel,
                 "quadrature_order": n, "index": index, "momenta": list(momenta), "measure": measure}.items():
        if shard.get(k) != v:
            raise ValueError("shard mismatch: "+k)
    lifts = [c["target_point"]["lifts"]] if channel == "target" else list(map(list, trial.LIFTS))
    if [r["lifts"] for r in shard["rows"]] != lifts:
        raise ValueError("incomplete or reordered lift rows")
    for r in shard["rows"]:
        if not math.isfinite(r["Z_weighted"]) or r["Z_weighted"] < 0:
            raise ValueError("invalid weighted node")
        if channel == "target" and not math.isclose(math.fsum(r["sectors_weighted"]), r["Z_weighted"], rel_tol=1e-14):
            raise ValueError("sector reduction mismatch")
        if channel == "source" and (not math.isclose(r["even_weighted"]+r["odd_weighted"], r["Z_weighted"], rel_tol=1e-13)
                                    or not math.isclose(r["equal_weighted"]+r["mixed_weighted"], r["Z_weighted"], rel_tol=1e-13)):
            raise ValueError("NSRR channel reduction mismatch")


def shard_path(output, channel, n, index):
    return output/"shards"/f"{channel}-N{n}-node-{index:03d}.json"


def worker(output, channel, n, index):
    c = trial.load(output/"config.json")
    validate(c)
    if n not in c[channel+"_orders"]:
        raise ValueError("node is not in the frozen design")
    path = shard_path(output, channel, n, index)
    if path.exists():
        validate_shard(c, channel, n, index, trial.load(path))
        return
    shard = evaluate(c, channel, n, index, output/"action_cache")
    validate(c)
    validate_shard(c, channel, n, index, shard)
    trial.save(path, shard)


def reduce_run(output):
    c = trial.load(output/"config.json")
    validate(c)
    b = c["source"]["b"]
    kappa = 1+2*(b+1/b)**2
    rows = []
    for channel in ("target", "source"):
        baseline = trial.load(Path(c["references"][channel])/"summary.json")
        if trial.digest(baseline) != c["reference_summary_digests"][channel]:
            raise ValueError("archived reference changed")
        if channel == "target":
            previous = [(5, next(r for r in baseline["rows"] if r["t"] == c["t"])["target_Z"])]
        else:
            previous = [(r["quadrature_order"], r["total"]) for r in baseline["rows"]
                        if r["t"] == c["t"] and r["level"] == 3 and r["lifts_geometry"] == [1, 1, 1]]
        for n, z in previous:
            rows.append({"channel": channel, "N": n, "Z": z, "origin": "validated reference",
                         "Q_diagnostic": z/c["Z_free_"+channel]**kappa})
        for n in c[channel+"_orders"]:
            shards = [trial.load(shard_path(output, channel, n, i)) for i in range(n**3)]
            for i, shard in enumerate(shards):
                validate_shard(c, channel, n, i, shard)
            z = math.fsum(s["rows"][0]["Z_weighted"] for s in shards)
            row = {"channel": channel, "N": n, "Z": z, "origin": "fresh fixed-cutoff nodes",
                   "Q_diagnostic": z/c["Z_free_"+channel]**kappa,
                   "nodes": n**3, "seconds_sum": math.fsum(s["seconds"] for s in shards)}
            if channel == "source":
                row["Z_by_literal_lift"] = [math.fsum(s["rows"][j]["Z_weighted"] for s in shards) for j in range(4)]
                row["mixed_fraction"] = math.fsum(s["rows"][0]["mixed_weighted"] for s in shards)/z
            rows.append(row)
    for channel in ("target", "source"):
        ordered = sorted((r for r in rows if r["channel"] == channel), key=lambda r: r["N"])
        for a, b in zip(ordered, ordered[1:]):
            b["relative_change_from_previous_N"] = b["Z"]/a["Z"]-1
    finest = {ch: max((r for r in rows if r["channel"] == ch), key=lambda r: r["N"]) for ch in ("source", "target")}
    report = {"schema": SCHEMA, "t": c["t"], "config_digest": trial.digest(c), "kappa": kappa,
              "rows": rows, "finest_diagnostic_gap": finest["source"]["Q_diagnostic"]/finest["target"]["Q_diagnostic"]-1,
              "physical_Q_NSrr": None, "protected": fresh.protected_hashes(),
              "warning": "Successive differences are convergence observations, not rigorous error bounds; numerator spin identification remains separate."}
    trial.save(output/"quadrature_summary.json", report)
    return report


def run(output, workers=2):
    c = trial.load(output/"config.json")
    validate(c)
    tasks = [(ch, n, i) for ch in ("target", "source") for n in c[ch+"_orders"] for i in range(n**3)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    def dispatch(task):
        ch, n, i = task
        with (logs/f"{ch}-N{n}-node-{i:03d}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                            "--channel", ch, "--N", str(n), "--index", str(i)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        return task
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(dispatch, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            task = future.result()
            print(f"{count}/{len(tasks)} complete; {task}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    report = reduce_run(output)
    for row in report["rows"]:
        print(row, flush=True)
    print("finest diagnostic gap", report["finest_diagnostic_gap"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "worker", "run", "reduce"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--channel", choices=("source", "target"))
    p.add_argument("--N", type=int)
    p.add_argument("--index", type=int)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "worker":
        worker(args.output, args.channel, args.N, args.index)
    elif args.mode == "run":
        run(args.output, args.workers)
    else:
        print(reduce_run(args.output))


if __name__ == "__main__":
    main()
