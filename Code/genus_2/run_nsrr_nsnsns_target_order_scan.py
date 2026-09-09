#!/usr/bin/env python3
"""Isolate all-NS c-recursion order on an immutable existing NSRR/NSNSNS grid.

R limits accumulated twice-level of Kac residues, NOT the full plumbing
polynomial: the regular/global blocks remain resummed. No NSRR recomputation,
quadrature remapping, convention change, or fitted normalization is performed.
Each momentum node runs in a fresh subprocess to bound retained caches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html import escape
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import nsrr_nsnsns_theta_omega_scan as scan

SCHEMA = "nsrr-nsnsns-target-order-scan-v1"


def fingerprint():
    return hashlib.sha256((scan.fingerprint() + hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def make_config(baseline_dir, orders, quadratures):
    baseline_dir = Path(baseline_dir).resolve()
    base = scan._load(baseline_dir / "config.json")
    summary = scan._load(baseline_dir / "summary.json")
    scan.validate_config(base)
    if summary["config"] != base or summary["implementation_fingerprint"] != scan.fingerprint():
        raise ValueError("baseline config/kernel no longer matches the saved run")
    config = {"schema": SCHEMA, "baseline_run_directory": str(baseline_dir),
              "baseline_config": base, "baseline_summary_digest": scan._digest(summary),
              "baseline_implementation_fingerprint": scan.fingerprint(),
              "recursion_orders": list(orders), "quadrature_orders": list(quadratures)}
    validate_config(config)
    return config


def validate_config(config):
    if config["schema"] != SCHEMA:
        raise ValueError("wrong order-scan schema")
    base = config["baseline_config"]
    scan.validate_config(base)
    orders, quadratures = config["recursion_orders"], config["quadrature_orders"]
    if (not orders or orders != sorted(set(orders))
            or orders[0] != base["target_recursion_order_twice_level"]
            or any(not isinstance(r, int) or not 0 <= r <= 24 for r in orders)):
        raise ValueError("orders must start at the baseline R and increase within 0..24")
    if (not quadratures or quadratures != sorted(set(quadratures))
            or not set(quadratures).issubset(base["quadrature_orders"])):
        raise ValueError("quadrature grids must be selected from the unchanged baseline")
    if config["baseline_implementation_fingerprint"] != scan.fingerprint():
        raise ValueError("numerical kernel changed since the baseline")


def task_indices(config):
    return [i for i, (channel, n, _) in enumerate(scan.tasks(config["baseline_config"]))
            if channel == "target_nsnsns" and n in config["quadrature_orders"]]


def baseline_shard(config, index):
    shard = scan._load(Path(config["baseline_run_directory"]) / "shards" / f"task-{index:06d}.json")
    scan.validate_shard(config["baseline_config"], index, shard,
                        config["baseline_implementation_fingerprint"])
    return shard


def validate_shard(config, index, shard):
    base = config["baseline_config"]
    channel, n, node, indices, momenta, measure = scan.node_data(base, index)
    if index not in task_indices(config):
        raise ValueError("unexpected target task index")
    expected = {"schema": SCHEMA, "config_digest": scan._digest(config),
                "implementation_fingerprint": fingerprint(), "task_index": index,
                "channel": channel, "quadrature_order": n, "node_index": node,
                "indices": list(indices), "momenta": list(momenta), "measure": measure,
                "baseline_shard_digest": scan._digest(baseline_shard(config, index))}
    for key, value in expected.items():
        if shard.get(key) != value:
            raise ValueError(f"order-scan shard {index}: {key} mismatch")
    design = [(p["t"], r) for p in base["points"] for r in config["recursion_orders"]]
    if [(row["t"], row["recursion_order"]) for row in shard["values"]] != design:
        raise ValueError("order-scan shard has incomplete or reordered evaluations")
    for row in shard["values"]:
        values = row["sector_contributions"]
        if len(values) != 2 or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("invalid target sector values")
        if row["global_nonconverged_calls"] != 0:
            raise ValueError("unconverged global block")
    if not 0 <= shard["baseline_sector_relative_error_max"] <= 1e-10:
        raise ValueError("baseline R was not reproduced")


def worker(config_path, output_dir, index):
    config = scan._load(config_path)
    validate_config(config)
    path = Path(output_dir) / f"task-{index:06d}.json"
    if path.exists():
        validate_shard(config, index, scan._load(path))
        return path
    if index not in task_indices(config):
        raise ValueError("only selected target tasks may run")
    base = config["baseline_config"]
    prior = baseline_shard(config, index)
    channel, n, node, indices, momenta, measure = scan.node_data(base, index)
    p, numerics = base["parameters"], base["numerics"]
    constants = scan.GenericSuperLiouvilleConstants(
        p["b"], dps=numerics["structure_precision"], mu=complex(p["mu"]),
        include_cosmological_prefactor=p["include_cosmological_prefactor"])
    started = time.perf_counter()
    values, errors = [], []
    for point in base["points"]:
        chart = point["charts"][channel]
        q = tuple(complex(x) for x in chart["q_values"])
        recursion = scan.NSGenus2CRecursion(
            channel="theta", q_values=q, global_method="resummed",
            global_tolerance=numerics["global_tolerance"],
            global_max_total_occupation=numerics["global_max_total_occupation"],
            vacuum_word_length=numerics["vacuum_word_length"],
            vacuum_max_mode=numerics["vacuum_max_mode"])
        for r in config["recursion_orders"]:
            tick = time.perf_counter()
            sectors = scan.all_ns_node(
                b=p["b"], q_values=q, lifts=chart["lifts"], recursion_order=r,
                momenta=momenta, measure=measure, constants=constants, recursion=recursion,
                block_method="collision_aware_mp",
                block_working_precision=numerics["block_working_precision"])
            if r == config["recursion_orders"][0]:
                expected = next(x["sector_contributions"] for x in prior["values"] if x["t"] == point["t"])
                errors.extend(abs(a-b) / max(abs(b), 1e-280) for a, b in zip(sectors, expected))
            values.append({"t": point["t"], "recursion_order": r,
                           "sector_contributions": list(sectors),
                           "runtime_seconds": time.perf_counter()-tick,
                           "global_max_occupation_used": recursion.global_max_used,
                           "global_nonconverged_calls": recursion.global_nonconverged_calls,
                           "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative,
                           "confluent_max_total_cancellation_ratio": float(recursion.confluent_max_total_cancellation_ratio)})
            print(f"task={index} t={point['t']:.2f} R={r} done", flush=True)
    result = {"schema": SCHEMA, "config_digest": scan._digest(config),
              "implementation_fingerprint": fingerprint(), "runtime_versions": scan._runtime_versions(),
              "task_index": index, "channel": channel, "quadrature_order": n,
              "node_index": node, "indices": list(indices), "momenta": list(momenta),
              "measure": measure, "baseline_shard_digest": scan._digest(prior), "values": values,
              "baseline_sector_relative_error_max": max(errors),
              "runtime_seconds": time.perf_counter()-started}
    validate_shard(config, index, result)
    scan.write_json(path, result)
    return path


def reduce_scan(config_path, shard_dir, output):
    config = scan._load(config_path)
    validate_config(config)
    indices = task_indices(config)
    expected = {f"task-{i:06d}.json" for i in indices}
    observed = {p.name for p in Path(shard_dir).glob("task-*.json")}
    if expected != observed:
        raise RuntimeError(f"incomplete order scan: {len(expected-observed)} missing, {len(observed-expected)} unexpected")
    baseline = scan._load(Path(config["baseline_run_directory"]) / "summary.json")
    if scan._digest(baseline) != config["baseline_summary_digest"]:
        raise ValueError("baseline summary changed")
    shards = [scan._load(Path(shard_dir) / f"task-{i:06d}.json") for i in indices]
    for index, shard in zip(indices, shards):
        validate_shard(config, index, shard)
    base = config["baseline_config"]
    source_level = max(base["source_physical_levels"])
    rows = []
    for point in base["points"]:
        for n in config["quadrature_orders"]:
            old = next(row for row in baseline["rows"] if row["t"] == point["t"] and row["quadrature_order"] == n)
            source = old["values"][f"source_nsrr_L{source_level}"]
            free = point["charts"]["target_nsnsns"]["physical_free_superfield"]
            first = previous = None
            for r in config["recursion_orders"]:
                nodes = [row for shard in shards if shard["quadrature_order"] == n
                         for row in shard["values"] if row["t"] == point["t"] and row["recursion_order"] == r]
                if len(nodes) != n**3:
                    raise RuntimeError("missing quadrature nodes")
                sectors = [math.fsum(row["sector_contributions"][s] for row in nodes) for s in (0, 1)]
                z = math.fsum(sectors)
                q_value = z / free**baseline["kappa"]
                if first is None:
                    first = q_value
                    old_target = old["values"][f"target_nsnsns_L{base['target_physical_level']}"]["Q"]
                    if not math.isclose(first, old_target, rel_tol=1e-10, abs_tol=0):
                        raise ValueError("baseline integrated Q not reproduced")
                rows.append({"t": point["t"], "quadrature_order": n, "recursion_order": r,
                             "target_Z": z, "target_Q": q_value, "target_sector_values": sectors,
                             "target_odd_fraction": sectors[1]/z, "target_Z_free": free,
                             "source_physical_level": source_level, "source_Q_fixed": source["Q"],
                             "source_over_target": source["Q"]/q_value,
                             "target_relative_change_from_baseline": q_value/first-1,
                             "target_relative_change_from_previous": None if previous is None else q_value/previous-1})
                previous = q_value
    result = {"schema": SCHEMA, "config": config, "implementation_fingerprint": fingerprint(),
              "kappa": baseline["kappa"], "shards_validated": len(shards), "rows": rows,
              "baseline_sector_relative_error_max": max(s["baseline_sector_relative_error_max"] for s in shards),
              "global_nonconverged_calls": sum(row["global_nonconverged_calls"] for s in shards for row in s["values"]),
              "global_max_occupation_used": max(row["global_max_occupation_used"] for s in shards for row in s["values"]),
              "global_worst_last_shell_relative": max(row["global_worst_last_shell_relative"] for s in shards for row in s["values"]),
              "runtime_seconds_sum": math.fsum(s["runtime_seconds"] for s in shards),
              "interpretation": "Only accumulated Kac-residue twice-level R changes. Global blocks remain resummed; all other settings and the NSRR answer are fixed. Differences are diagnostics, not certified error bounds."}
    scan.write_json(output, result)
    plot_svg(result, Path(output).with_suffix(".svg"))
    return result


def plot_svg(result, path):
    config = result["config"]
    n = max(config["quadrature_orders"])
    orders = config["recursion_orders"]
    colors = ("#777777", "#d77a13", "#176dad", "#84469b", "#26804b")
    selected = [row for row in result["rows"] if row["quadrature_order"] == n]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1000" viewBox="0 0 1100 1000">',
           '<rect width="1100" height="1000" fill="white"/><g font-family="Arial,sans-serif" fill="#20242b">']
    def label(x, y, text, size=14, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(text)}</text>')
    label(90, 35, "All-NS c-recursion convergence: fixed geometry and NSRR answer", 22)
    label(90, 61, f"b={config['baseline_config']['parameters']['b']}; N={n}; R = accumulated null twice-level; global blocks resummed")
    source = [(row["t"], row["source_Q_fixed"]) for row in selected if row["recursion_order"] == orders[0]]
    for panel, (key, title) in enumerate((("target_Q", "Q versus source period coordinate"),
                                          ("source_over_target", "Q_NS RR / Q_NS NS NS"),
                                          ("target_relative_change_from_baseline", f"All-NS fractional change relative to R={orders[0]}"))):
        top, left, width, height = 115+panel*285, 125, 645, 195
        curves = [(f"all-NS R={r}", colors[i % len(colors)],
                   [(row["t"], row[key]) for row in selected if row["recursion_order"] == r]) for i, r in enumerate(orders)]
        if panel == 0:
            curves.insert(0, ("NSRR fixed L=4", "#ba432f", source))
        points = [point for _, _, curve in curves for point in curve]
        xmin, xmax = min(x for x, y in points), max(x for x, y in points)
        ys = [y for x, y in points] + ([1.] if panel == 1 else [0.] if panel == 2 else [])
        low, high = min(ys), max(ys)
        margin = .10*(high-low or max(abs(high), 1e-10))
        low, high = low-margin, high+margin
        def xy(x, y):
            return left+width*(x-xmin)/(xmax-xmin or 1), top+height*(high-y)/(high-low)
        label(left, top-18, title, 17)
        for j in range(5):
            y = low+(high-low)*j/4
            yy = xy(xmin, y)[1]
            svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e2e5e9"/>')
            label(left-12, yy+5, f"{y:.4g}", 12, "end")
        for x in sorted(set(x for x, y in points)):
            label(xy(x, low)[0], top+height+21, f"{x:.2f}", 12, "middle")
        for i, (name, color, curve) in enumerate(curves):
            coords = " ".join(f"{xy(x,y)[0]:.3f},{xy(x,y)[1]:.3f}" for x, y in curve)
            dash = ' stroke-dasharray="7 4"' if i % 2 else ""
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
            for x, y in curve:
                xx, yy = xy(x,y)
                svg.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{color}"/>')
            label(820, top+20+26*i, name, 13)
        label(left+width/2, top+height+43, "t = Re Omega_12 (source marking)", 14, "middle")
    label(90, 980, "Same local-frame free normalization, unchanged odd i / sewing sign; no fitted rescaling. Cutoff differences are not error bars.", 12)
    svg.append('</g></svg>')
    Path(path).write_text("\n".join(svg)+"\n")


def run(baseline_dir, output_dir, orders, quadratures, workers):
    config = make_config(baseline_dir, orders, quadratures)
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "config.json"
    if config_path.exists() and scan._load(config_path) != config:
        raise ValueError("output directory has a different design")
    scan.write_json(config_path, config)
    logs, shards = output_dir/"logs", output_dir/"shards"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def evaluate(index):
        with (logs/f"task-{index:06d}.log").open("a") as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker",
                "--config", str(config_path), "--output-dir", str(shards), "--task-index", str(index)],
                env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"target task {index} failed; see {logs/f'task-{index:06d}.log'}")
        return index
    indices = task_indices(config)
    print(f"Target-only order scan: R={orders}, N={quadratures}, {len(indices)} nodes, {workers} workers", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, index) for index in indices]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                index = future.result()
                print(f"{count}/{len(indices)} complete; last={index}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    result = reduce_scan(config_path, shards, output_dir/"summary.json")
    print(f"Completed in {time.perf_counter()-started:.1f}s", flush=True)
    for row in result["rows"]:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--orders", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--quadratures", type=int, nargs="+", default=[5])
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-index", type=int, required=True)
    p = sub.add_parser("reduce")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not 1 <= args.workers <= 4:
            parser.error("use 1..4 concurrent workers")
        run(args.baseline_dir, args.output_dir, args.orders, args.quadratures, args.workers)
    elif args.command == "worker":
        print(worker(args.config, args.output_dir, args.task_index), flush=True)
    else:
        reduce_scan(args.config, args.shard_dir, args.output)


if __name__ == "__main__":
    main()
