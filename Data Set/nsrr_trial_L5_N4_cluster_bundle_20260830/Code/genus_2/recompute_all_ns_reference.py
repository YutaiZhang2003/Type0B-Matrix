#!/usr/bin/env python3
"""Fresh all-NS reference on the former five-point NSRR comparison geometry.

Only geometry, quadrature design, and numerical settings are imported. No
archived partition values or uncertified NSRR sewing prescriptions are used.
One subprocess per momentum node bounds the recursion caches. The output is
explicitly partial: it cannot establish an NSRR/all-NS modular comparison.
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

import numpy as np

import nsrr_nsnsns_theta_omega_scan as scan

SCHEMA = "fresh-all-ns-reference-v1"
ROOT = Path(__file__).resolve().parents[2]


def protected_hashes():
    expected = scan._load(Path(__file__).with_name("nsrr_checked_kernel_manifest.json"))["sha256"]
    actual = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in expected}
    if actual != expected:
        raise ValueError("a protected kernel changed; stop before evaluating")
    return actual


def fingerprint():
    return hashlib.sha256((scan.fingerprint() + hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def prepare(baseline_path, order, recursion_order):
    base = scan._load(baseline_path)
    if order not in base["quadrature_orders"]:
        raise ValueError("use an existing common momentum grid")
    if not 0 <= recursion_order <= 24:
        raise ValueError("recursion twice-level must be in 0..24")
    protected_hashes()
    result = {
        "schema": SCHEMA, "implementation_fingerprint": fingerprint(),
        "input_geometry_path": str(Path(baseline_path).resolve()),
        "input_geometry_digest": scan._digest(base),
        "input_policy": "geometry and settings only; no archived partition values",
        "parameters": base["parameters"], "numerics": base["numerics"],
        "quadrature_order": order, "recursion_order_twice_level": recursion_order,
        "quadrature_reference_abs_q": base["quadrature_reference_abs_q"]["target_nsnsns"],
        "conventions": {
            "three_point": "C_HN=(C_BRY,i*tilde_C_BRY); square, not absolute square",
            "sewing_sign": "separate (-1)^f for even NS primaries",
            "measure": "dP1*dP2*dP3/pi^3 on nonnegative momenta",
            "free_frame": "direct all-NS X+psi evaluator in the same q and lift frame",
            "normalization_status": "existing all-NS convention; NSRR spin conversion not used",
            "R": "accumulated null twice-level; global blocks remain resummed",
        },
        "nsrr_status": "not computed: nonchiral contraction and free-spin conversion unresolved",
        "points": [],
    }
    for point in base["points"]:
        chart = point["charts"]["target_nsnsns"]
        q = tuple(complex(x) for x in chart["q_values"])
        omega = scan.complex_matrix(chart["omega"])
        source = np.array([[1j, point["t"] + .5j], [point["t"] + .5j, 1j]])
        if np.max(abs(scan.omega_action(source) - omega)) > 1e-12:
            raise ValueError("target marking does not match the requested family")
        if scan._spin_characteristic_from_lifts("theta", q, chart["lifts"]) != scan.TARGET_SPIN:
            raise ValueError("unexpected all-NS lifts")
        forward = scan.solve_theta_collocation(*q, basis_order=32, samples_per_seam=160)
        error = float(np.max(abs(forward.omega - omega)))
        if error > 1e-8 or forward.max_seam_residual > 1e-8:
            raise ValueError("fresh forward period check failed")
        free = [float(scan.physical_superfield_plumbing_partition(
            "theta", q, chart["lifts"], max_mode=mode).one_superfield_value)
            for mode in (base["numerics"]["free_check_mode"], base["numerics"]["free_max_mode"])]
        if not all(math.isfinite(z) and z > 0 for z in free) or abs(free[0]/free[1]-1) > 1e-8:
            raise ValueError("all-NS free factor did not converge")
        result["points"].append({
            "t": point["t"], "q_values": chart["q_values"], "lifts": chart["lifts"],
            "omega": chart["omega"], "forward_period_error": error,
            "forward_basis_order": 32, "forward_samples_per_seam": 160,
            "seam_residual": float(forward.max_seam_residual),
            "Z_free": free[1], "free_mode_relative_change": abs(free[0]/free[1]-1),
        })
        print(f"fresh geometry/free audit t={point['t']:.2f}: period error {error:.3e}", flush=True)
    return result


def validate_config(config):
    if config["schema"] != SCHEMA or config["implementation_fingerprint"] != fingerprint():
        raise ValueError("config schema or implementation mismatch")
    protected_hashes()
    if config["quadrature_order"] < 1 or not 0 <= config["recursion_order_twice_level"] <= 24:
        raise ValueError("invalid numerical cutoff")
    ts = [p["t"] for p in config["points"]]
    if not ts or ts != sorted(set(ts)):
        raise ValueError("invalid point design")
    for point in config["points"]:
        if not 0 <= point["forward_period_error"] <= 1e-8:
            raise ValueError("failed geometry")
        if not math.isfinite(point["Z_free"]) or point["Z_free"] <= 0:
            raise ValueError("invalid free factor")
        for q, envelope in zip(point["q_values"], config["quadrature_reference_abs_q"]):
            if not 0 < abs(complex(q)) <= envelope*(1+1e-14) < 1:
                raise ValueError("q outside the continuum quadrature envelope")


def node_data(config, index):
    n = config["quadrature_order"]
    if not 0 <= index < n**3:
        raise ValueError("invalid momentum node")
    indices = np.unravel_index(index, (n,)*3)
    rules = scan._rules(config["quadrature_reference_abs_q"], n)
    return (tuple(float(rules[e][0][indices[e]]) for e in range(3)),
            float(scan._measure(rules, indices)))


def validate_shard(config, index, shard):
    momenta, measure = node_data(config, index)
    for key, expected in {
        "schema": SCHEMA, "config_digest": scan._digest(config),
        "implementation_fingerprint": fingerprint(), "index": index,
        "momenta": list(momenta), "measure": measure,
    }.items():
        if shard.get(key) != expected:
            raise ValueError(f"node {index}: {key} mismatch")
    if [r["t"] for r in shard["values"]] != [p["t"] for p in config["points"]]:
        raise ValueError("missing or reordered evaluations")
    for row in shard["values"]:
        sectors = row["sector_contributions"]
        if len(sectors) != 2 or not all(math.isfinite(x) and x >= 0 for x in sectors):
            raise ValueError("invalid all-NS sector values")
        if row["global_nonconverged_calls"]:
            raise ValueError("global resummation did not converge")


def worker(config_path, output_dir, index):
    config = scan._load(config_path)
    validate_config(config)
    path = Path(output_dir) / f"node-{index:03d}.json"
    if path.exists():
        validate_shard(config, index, scan._load(path))
        return
    started = time.perf_counter()
    momenta, measure = node_data(config, index)
    p, n = config["parameters"], config["numerics"]
    constants = scan.GenericSuperLiouvilleConstants(
        p["b"], dps=n["structure_precision"], mu=complex(p["mu"]),
        include_cosmological_prefactor=p["include_cosmological_prefactor"])
    values = []
    for point in config["points"]:
        q = tuple(complex(x) for x in point["q_values"])
        recursion = scan.NSGenus2CRecursion(
            channel="theta", q_values=q, global_method="resummed",
            global_tolerance=n["global_tolerance"],
            global_max_total_occupation=n["global_max_total_occupation"],
            vacuum_word_length=n["vacuum_word_length"], vacuum_max_mode=n["vacuum_max_mode"])
        sectors = scan.all_ns_node(
            b=p["b"], q_values=q, lifts=point["lifts"],
            recursion_order=config["recursion_order_twice_level"],
            momenta=momenta, measure=measure, constants=constants, recursion=recursion,
            block_method="collision_aware_mp", block_working_precision=n["block_working_precision"])
        values.append({"t": point["t"], "sector_contributions": list(sectors),
                       "global_nonconverged_calls": recursion.global_nonconverged_calls,
                       "global_max_occupation_used": recursion.global_max_used,
                       "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative})
        print(f"node={index} t={point['t']:.2f} done", flush=True)
    if config["implementation_fingerprint"] != fingerprint():
        raise ValueError("implementation changed during evaluation")
    result = {"schema": SCHEMA, "config_digest": scan._digest(config),
              "implementation_fingerprint": fingerprint(), "index": index,
              "momenta": list(momenta), "measure": measure, "values": values,
              "runtime_seconds": time.perf_counter()-started}
    validate_shard(config, index, result)
    scan.write_json(path, result)


def reduce_run(config_path, output_dir):
    config = scan._load(config_path)
    validate_config(config)
    output_dir = Path(output_dir)
    count = config["quadrature_order"]**3
    paths = [output_dir / "shards" / f"node-{i:03d}.json" for i in range(count)]
    if set((output_dir/"shards").glob("node-*.json")) != set(paths):
        raise ValueError("missing or unexpected momentum shards")
    shards = [scan._load(p) for p in paths]
    for index, shard in enumerate(shards):
        validate_shard(config, index, shard)
    b = config["parameters"]["b"]
    kappa = 1+2*(b+1/b)**2
    rows = []
    for j, point in enumerate(config["points"]):
        sectors = [math.fsum(s["values"][j]["sector_contributions"][f] for s in shards) for f in (0, 1)]
        z = math.fsum(sectors)
        rows.append({"t": point["t"], "target_Z": z, "target_Q": z/point["Z_free"]**kappa,
                     "target_Z_free": point["Z_free"], "sector_values": sectors,
                     "target_odd_fraction": sectors[1]/z, "source_Z": None, "source_Q": None,
                     "source_over_target": None})
    result = {"schema": SCHEMA, "status": "partial: all-NS only", "config": config,
              "implementation_fingerprint": fingerprint(), "protected_kernel_hashes": protected_hashes(),
              "runtime_versions": scan._runtime_versions(), "kappa": kappa,
              "shards_validated": count, "rows": rows,
              "global_nonconverged_calls": sum(r["global_nonconverged_calls"] for s in shards for r in s["values"]),
              "global_max_occupation_used": max(r["global_max_occupation_used"] for s in shards for r in s["values"]),
              "runtime_seconds_sum": math.fsum(s["runtime_seconds"] for s in shards),
              "interpretation": "Fresh all-NS sewing values, not a corrected NSRR comparison. No fitted normalization. Fixed quadrature is not a certified error bound."}
    scan.write_json(output_dir/"summary.json", result)
    plot_svg(result, output_dir/"all_ns_reference.svg")
    return result


def plot_svg(result, path):
    config, rows = result["config"], result["rows"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="680" viewBox="0 0 1040 680">',
           '<rect width="1040" height="680" fill="white"/><g font-family="Arial,sans-serif" fill="#253044">']
    def label(x, y, value, size=15, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')
    label(80, 38, "Fresh all-NS genus-two reference — NSRR comparison pending", 23)
    label(80, 66, f"b={config['parameters']['b']}; N={config['quadrature_order']}; c-recursion R={config['recursion_order_twice_level']}; resummed global blocks")
    label(80, 91, "Omega_source(t) = [[i, t+0.5i], [t+0.5i, i]]; values evaluated in the transformed all-NS chart")
    xmin, xmax = rows[0]["t"], rows[-1]["t"]
    cosmological = "included" if config["parameters"]["include_cosmological_prefactor"] else "omitted"
    for panel, (key, title) in enumerate((("target_Z", f"Z_SL (cosmological prefactor {cosmological})"),
                                         ("target_Q", "Q = Z_SL / Z_free^kappa; kappa = %.8f" % result["kappa"]))):
        left, top, width, height = 120+panel*475, 165, 350, 330
        low, high = min(r[key] for r in rows), max(r[key] for r in rows)
        margin = .1*(high-low or abs(high) or 1)
        low, high = low-margin, high+margin
        def xy(x, y):
            return left+width*(x-xmin)/(xmax-xmin or 1), top+height*(high-y)/(high-low)
        label(left, top-28, title, 15)
        for j in range(5):
            y = low+(high-low)*j/4
            yy = xy(xmin, y)[1]
            svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e0e5ec"/>')
            label(left-10, yy+4, f"{y:.3e}", 12, "end")
        for row in rows:
            label(xy(row["t"], low)[0], top+height+24, f"{row['t']:.2f}", 13, "middle")
        coords = " ".join(f"{xy(r['t'],r[key])[0]:.3f},{xy(r['t'],r[key])[1]:.3f}" for r in rows)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="#176dad" stroke-width="2.5"/>')
        for row in rows:
            x, y = xy(row["t"], row[key])
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#176dad"/>')
        label(left+width/2, top+height+54, "t = Re Omega_source,12", 15, "middle")
    label(80, 590, "Only the all-NS curve has been recomputed. No NSRR values or modular ratios are supplied.", 16)
    label(80, 617, "Odd three-point i factor and separate sewing sign retained; free field evaluated at the same q and lifts.", 14)
    label(80, 644, f"Fixed N={config['quadrature_order']} quadrature: displayed precision does not represent a certified integration error.", 14)
    svg.append('</g></svg>')
    Path(path).write_text("\n".join(svg)+"\n")


def run(args):
    output = args.output_dir.resolve()
    config_path = output/"config.json"
    if config_path.exists():
        config = scan._load(config_path)
        validate_config(config)
        if (config["input_geometry_digest"] != scan._digest(scan._load(args.baseline_config))
                or config["quadrature_order"] != args.quadrature_order
                or config["recursion_order_twice_level"] != args.recursion_order):
            raise ValueError("existing output belongs to a different run")
    else:
        config = prepare(args.baseline_config, args.quadrature_order, args.recursion_order)
        validate_config(config)
        scan.write_json(config_path, config)
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def evaluate(index):
        with (logs/f"node-{index:03d}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker",
                            "--config", str(config_path), "--output-dir", str(output/"shards"),
                            "--index", str(index)], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return index
    started = time.perf_counter()
    count = config["quadrature_order"]**3
    print(f"Fresh all-NS reference: {count} nodes x {len(config['points'])} points, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, i) for i in range(count)]
        try:
            for done, future in enumerate(as_completed(futures), 1):
                print(f"{done}/{count} nodes complete; last={future.result()}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    result = reduce_run(config_path, output)
    print(f"Completed in {time.perf_counter()-started:.1f}s", flush=True)
    for row in result["rows"]:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--baseline-config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--quadrature-order", type=int, default=5)
    p.add_argument("--recursion-order", type=int, default=16)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    p = sub.add_parser("worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p = sub.add_parser("reduce")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "worker":
        worker(args.config, args.output_dir, args.index)
    else:
        reduce_run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
