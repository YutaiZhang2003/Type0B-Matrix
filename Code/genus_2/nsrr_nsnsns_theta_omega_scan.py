#!/usr/bin/env python3
"""Moderate-cost, common-momentum-grid NSRR/NSNSNS period-matrix scan.

Each immutable shard computes one momentum node at every marked surface.
The expensive Ward branching and both Virasoro c-recursions are reused over
the scan. Batch workers launch a NEW PROCESS per node to bound cache memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from nsrr_nsnsns_theta_cannon import (
    CHANNELS, _digest, _implementation_fingerprint, _load, _runtime_versions,
    _spin_characteristic_from_lifts, _transport_spin_characteristic,
)
from compare_nsrr_nsnsns_theta import (
    SOURCE_Q, TARGET_Q, SOURCE_LIFTS, TARGET_LIFTS, RAMOND_GROUND_COMPLETENESS,
    _rules, _measure, _primary, all_ns_node, GenericSuperLiouvilleConstants,
    hjs_rr_ns_constant, NSGenus2CRecursion,
    physical_superfield_plumbing_partition, riemann_theta_constant_genus2,
    require_certified_nsrr_partition_sewing,
)
from nsrr_double_virasoro_block import (
    NSRRDoubleVirasoroTheta, spin_character_index, evaluate_twice_level_series,
)
from plumbing_algorithms import (
    solve_theta_collocation, schottky_theta_period_matrix_cross_ratio,
)

SCHEMA = "nsrr-nsnsns-theta-omega-scan-v1"
MATRIX = np.array([[1, 0, 0, -1], [0, -1, 1, -1],
                   [0, 0, 1, 0], [0, 1, -1, 0]])
SOURCE_SPIN = ((0, 1), (1, 0))
REFERENCE_SPIN = ((0, 0), (1, 0))
TARGET_SPIN = ((0, 0), (0, 0))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def omega_action(omega):
    return ((MATRIX[:2, :2] @ omega + MATRIX[:2, 2:])
            @ np.linalg.inv(MATRIX[2:, :2] @ omega + MATRIX[2:, 2:]))


def complex_matrix(values):
    return np.array([[complex(x) for x in row] for row in values])


def inverse_chart(omega, seed):
    def unpack(x):
        return np.exp(x[:3] + 1j * x[3:])

    def residual(x):
        raw = solve_theta_collocation(*unpack(x), basis_order=16,
                                      samples_per_seam=80).omega
        delta = raw - omega
        # Used only during optimization. Final direct, marked-basis residual
        # below MUST pass without rounding or a silent spin/basis change.
        delta -= np.rint(delta.real)
        upper = delta[np.triu_indices(2)]
        return np.r_[upper.real, upper.imag]

    logs = np.log(np.asarray(seed))
    solved = least_squares(residual, np.r_[logs.real, logs.imag],
                           diff_step=1e-5, xtol=1e-11, gtol=1e-11,
                           ftol=1e-11, max_nfev=40)
    q = unpack(solved.x)
    audit = solve_theta_collocation(*q, basis_order=24, samples_per_seam=112)
    direct_error = float(np.max(np.abs(audit.omega - omega)))
    cross = schottky_theta_period_matrix_cross_ratio(*q, max_word_len=8)
    branch = np.rint((omega - cross).real).astype(int)
    cross_error = float(np.max(np.abs(cross + branch - omega)))
    if (not solved.success or direct_error > 1e-8 or cross_error > 1e-8
            or audit.max_seam_residual > 1e-8):
        raise ArithmeticError(f"inverse period map failed: {direct_error=}, {cross_error=}")
    return {
        "q_values": [str(complex(x)) for x in q],
        "omega": [[str(complex(x)) for x in row] for row in omega],
        "inverse_period_residual": direct_error,
        "seam_residual": float(audit.max_seam_residual),
        "cross_ratio_period_residual": cross_error,
        "cross_ratio_integer_branch": branch.tolist(),
        "collocation_basis_order": 24, "cross_ratio_word_length": 8,
    }


def build_config(output):
    old = _load(Path(__file__).parents[1] / "config/nsrr_nsnsns_theta_order8_cannon_20260829.json")
    config = {
        "schema": SCHEMA,
        "parameters": old["parameters"],
        "convention_ledger": old["convention_ledger"],
        "scan": {"coordinate": "Re Omega_12", "family": "[[i,t+0.5i],[t+0.5i,i]]",
                 "symplectic_matrix_source_to_target": MATRIX.tolist()},
        "quadrature_orders": [4, 6],
        "source_physical_levels": [4, 6],
        "target_physical_level": 8,
        "target_recursion_order_twice_level": 16,
        "numerics": {"structure_precision": 30, "branching_mp_dps": 0,
                     "maximum_ward_residual": 1e-5, "block_working_precision": 50,
                     "global_tolerance": 2e-8, "global_max_total_occupation": 36,
                     "vacuum_word_length": 7, "vacuum_max_mode": 50,
                     "free_max_mode": 44, "free_check_mode": 36},
        "points": [],
    }
    for t in (.52, .56, .60, .64, .68):
        omega = np.array([[1j, t + .5j], [t + .5j, 1j]])
        point = {"t": t, "charts": {}}
        for channel, desired, seed, lifts in (
            (CHANNELS[0], omega, SOURCE_Q, SOURCE_LIFTS),
            (CHANNELS[1], omega_action(omega), TARGET_Q, TARGET_LIFTS),
        ):
            chart = inverse_chart(desired, seed)
            chart["lifts"] = list(lifts)
            q = tuple(complex(x) for x in chart["q_values"])
            free = [float(physical_superfield_plumbing_partition(
                "theta", q, lifts, max_mode=mode).one_superfield_value)
                for mode in (36, 44)]
            ratio = 1.0
            if channel == CHANNELS[0]:
                from audit_nsrr_free_spin_conversion import require_compatible_theta_ratio
                require_compatible_theta_ratio(q, omega)
                ratio = float(abs(riemann_theta_constant_genus2(omega, SOURCE_SPIN, tol=1e-15)
                                  / riemann_theta_constant_genus2(omega, REFERENCE_SPIN, tol=1e-15)))
            chart.update(physical_free_superfield=free[1] * ratio,
                         free_mode_relative_change=abs(free[0] / free[1] - 1),
                         source_majorana_spin_change_ratio=ratio)
            if chart["free_mode_relative_change"] > 1e-8:
                raise ArithmeticError("physical free denominator is not converged")
            point["charts"][channel] = chart
        config["points"].append(point)
        print(f"geometry t={t:.2f} validated", flush=True)
    config["quadrature_reference_abs_q"] = {
        channel: [max(abs(complex(p["charts"][channel]["q_values"][edge]))
                      for p in config["points"]) for edge in range(3)]
        for channel in CHANNELS}
    validate_config(config)
    write_json(output, config)
    return config


def validate_config(config):
    if config["schema"] != SCHEMA:
        raise ValueError("wrong scan schema")
    if config["target_recursion_order_twice_level"] != 2 * config["target_physical_level"]:
        raise ValueError("all-NS recursion uses twice-level units")
    if not config["source_physical_levels"] or min(config["source_physical_levels"]) < 0:
        raise ValueError("invalid NSRR levels")
    if sorted(set(config["quadrature_orders"])) != config["quadrature_orders"] or min(config["quadrature_orders"]) < 1:
        raise ValueError("invalid quadrature orders")
    if config["scan"]["symplectic_matrix_source_to_target"] != MATRIX.tolist():
        raise ValueError("unexpected period-basis transformation")
    if _transport_spin_characteristic(MATRIX, SOURCE_SPIN) != TARGET_SPIN:
        raise ValueError("spin transport mismatch")
    ts = [p["t"] for p in config["points"]]
    if not ts or ts != sorted(set(ts)):
        raise ValueError("scan coordinates must be unique and increasing")
    for point in config["points"]:
        source, target = (complex_matrix(point["charts"][ch]["omega"]) for ch in CHANNELS)
        expected = np.array([[1j, point["t"] + .5j], [point["t"] + .5j, 1j]])
        if np.max(abs(source - expected)) > 1e-12 or np.max(abs(omega_action(source) - target)) > 1e-12:
            raise ValueError("period family or transformation mismatch")
        for channel, spin in zip(CHANNELS, (REFERENCE_SPIN, TARGET_SPIN)):
            chart = point["charts"][channel]
            q = tuple(complex(x) for x in chart["q_values"])
            if any(not 0 < abs(x) < 1 for x in q):
                raise ValueError("invalid q")
            if _spin_characteristic_from_lifts("theta", q, chart["lifts"]) != spin:
                raise ValueError("spin lift/branch mismatch")
            for key in ("inverse_period_residual", "seam_residual", "cross_ratio_period_residual", "free_mode_relative_change"):
                if not 0 <= chart[key] <= 1e-8:
                    raise ValueError(f"geometry/free audit failed: {key}")
            if not math.isfinite(chart["physical_free_superfield"]) or chart["physical_free_superfield"] <= 0:
                raise ValueError("invalid free denominator")
            if any(abs(q[e]) > config["quadrature_reference_abs_q"][channel][e] * (1 + 1e-14) for e in range(3)):
                raise ValueError("common quadrature envelope too small")


def tasks(config):
    return [(channel, order, node) for channel in CHANNELS
            for order in config["quadrature_orders"] for node in range(order ** 3)]


def node_data(config, task_index):
    channel, order, node = tasks(config)[task_index]
    indices = np.unravel_index(node, (order,) * 3)
    rules = _rules(config["quadrature_reference_abs_q"][channel], order)
    return (channel, order, node, tuple(int(x) for x in indices),
            tuple(float(rules[e][0][indices[e]]) for e in range(3)),
            float(_measure(rules, indices)))


def source_values(config, momenta, measure, constants):
    """Block production Q until nonchiral Ramond sewing has a certificate."""
    require_certified_nsrr_partition_sewing()


def _legacy_source_values(config, momenta, measure, constants):
    """Do not feed corrected blocks into the obsolete contraction."""
    raise NotImplementedError(
        "The obsolete factor-four scan assembler is retired; archived "
        "results are preserved, but are not regenerated with corrected blocks."
    )


def target_values(config, momenta, measure, constants):
    n = config["numerics"]
    results = []
    for point in config["points"]:
        chart = point["charts"][CHANNELS[1]]
        q = tuple(complex(x) for x in chart["q_values"])
        recursion = NSGenus2CRecursion(
            channel="theta", q_values=q, global_method="resummed",
            global_tolerance=n["global_tolerance"],
            global_max_total_occupation=n["global_max_total_occupation"],
            vacuum_word_length=n["vacuum_word_length"], vacuum_max_mode=n["vacuum_max_mode"])
        sectors = all_ns_node(
            b=config["parameters"]["b"], q_values=q, lifts=chart["lifts"],
            recursion_order=config["target_recursion_order_twice_level"],
            momenta=momenta, measure=measure, constants=constants, recursion=recursion,
            block_method="collision_aware_mp", block_working_precision=n["block_working_precision"])
        results.append({"t": point["t"], "physical_level": config["target_physical_level"],
                        "sector_contributions": list(sectors)})
        print(f"target point t={point['t']:.2f} complete; global shell max={recursion.global_max_used}",
              file=sys.stderr, flush=True)
    return results, None


def fingerprint():
    return hashlib.sha256((_implementation_fingerprint()
                          + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def validate_shard(config, task_index, shard, implementation):
    channel, order, node, indices, momenta, measure = node_data(config, task_index)
    for key, value in {"schema": SCHEMA, "config_digest": _digest(config),
                       "implementation_fingerprint": implementation,
                       "task_index": task_index, "channel": channel,
                       "quadrature_order": order, "node_index": node,
                       "indices": list(indices)}.items():
        if shard.get(key) != value:
            raise ValueError(f"shard {task_index} mismatch in {key}")
    if not np.allclose(shard["momenta"], momenta, atol=1e-15, rtol=2e-14) or not math.isclose(shard["measure"], measure, rel_tol=2e-14):
        raise ValueError("shard momentum/measure mismatch")
    levels = config["source_physical_levels"] if channel == CHANNELS[0] else [config["target_physical_level"]]
    expected = [(p["t"], level) for p in config["points"] for level in levels]
    if [(r["t"], r["physical_level"]) for r in shard["values"]] != expected:
        raise ValueError("shard scan points/levels mismatch")
    for row in shard["values"]:
        values = row["sector_contributions"]
        if len(values) != 2 or not all(math.isfinite(x) for x in values):
            raise ValueError("invalid sector contributions")
    if channel == CHANNELS[0]:
        ward = shard["maximum_ward_residual"]
        if not math.isfinite(ward) or not 0 <= ward <= config["numerics"]["maximum_ward_residual"]:
            raise ValueError("shard Ward residual fails acceptance")


def worker(config_path, output_dir, task_index):
    config = _load(config_path)
    validate_config(config)
    implementation = fingerprint()
    path = output_dir / f"task-{task_index:06d}.json"
    if path.exists():
        validate_shard(config, task_index, _load(path), implementation)
        return path
    started = time.perf_counter()
    channel, order, node, indices, momenta, measure = node_data(config, task_index)
    p = config["parameters"]
    constants = GenericSuperLiouvilleConstants(
        p["b"], dps=config["numerics"]["structure_precision"], mu=complex(p["mu"]),
        include_cosmological_prefactor=p["include_cosmological_prefactor"])
    values, ward = (source_values if channel == CHANNELS[0] else target_values)(config, momenta, measure, constants)
    result = {"schema": SCHEMA, "config_digest": _digest(config),
              "implementation_fingerprint": implementation, "runtime_versions": _runtime_versions(),
              "task_index": task_index, "channel": channel, "quadrature_order": order,
              "node_index": node, "indices": list(indices), "momenta": list(momenta),
              "measure": measure, "values": values, "maximum_ward_residual": ward,
              "runtime_seconds": time.perf_counter() - started}
    validate_shard(config, task_index, result, implementation)
    write_json(path, result)
    return path


def channel_indices(config, channel):
    return [i for i, item in enumerate(tasks(config)) if item[0] == channel]


def channel_worker(config_path, output_dir, channel, chunk_index, chunk_size):
    indices = channel_indices(_load(config_path), channel)
    start = chunk_index * chunk_size
    if chunk_size <= 0 or start < 0 or start >= len(indices):
        raise ValueError("invalid chunk")
    for task_index in indices[start:start + chunk_size]:
        # Do not replace this with an in-process worker loop: the underlying
        # momentum-dependent lru_caches retain evaluator instances indefinitely.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                        "worker", "--output-dir", str(output_dir), "--task-index", str(task_index)], check=True)


def reduce_scan(config_path, shard_dir, output):
    config = _load(config_path)
    validate_config(config)
    count = len(tasks(config))
    expected = {f"task-{i:06d}.json" for i in range(count)}
    observed = {p.name for p in shard_dir.glob("task-*.json")}
    if observed != expected:
        raise RuntimeError(f"incomplete shard set: {len(expected-observed)} missing, {len(observed-expected)} unexpected")
    implementation = fingerprint()
    shards = []
    for i in range(count):
        shard = _load(shard_dir / f"task-{i:06d}.json")
        validate_shard(config, i, shard, implementation)
        shards.append(shard)
    b = config["parameters"]["b"]
    kappa = 1 + 2 * (b + 1 / b) ** 2
    rows = []
    for point in config["points"]:
        for order in config["quadrature_orders"]:
            values = {}
            for channel in CHANNELS:
                levels = config["source_physical_levels"] if channel == CHANNELS[0] else [config["target_physical_level"]]
                selected = [s for s in shards if s["channel"] == channel and s["quadrature_order"] == order]
                for level in levels:
                    matched = [r for s in selected for r in s["values"] if r["t"] == point["t"] and r["physical_level"] == level]
                    if len(matched) != order ** 3:
                        raise RuntimeError("incomplete reduction design")
                    sectors = [math.fsum(r["sector_contributions"][sector] for r in matched) for sector in (0, 1)]
                    free = point["charts"][channel]["physical_free_superfield"]
                    z = math.fsum(sectors)
                    values[f"{channel}_L{level}"] = {"Z": z, "Q": z / free ** kappa,
                                                    "Z_free": free, "sector_values": sectors}
            target = values[f"{CHANNELS[1]}_L{config['target_physical_level']}"]["Q"]
            ratios = {str(level): values[f"{CHANNELS[0]}_L{level}"]["Q"] / target
                      for level in config["source_physical_levels"]}
            rows.append({"t": point["t"], "quadrature_order": order,
                         "values": values, "source_over_target_by_source_level": ratios})
    diagnostics = []
    for point in config["points"]:
        coarse, fine = (next(r for r in rows if r["t"] == point["t"] and r["quadrature_order"] == order)
                        for order in (min(config["quadrature_orders"]), max(config["quadrature_orders"])))
        low, high = min(config["source_physical_levels"]), max(config["source_physical_levels"])
        source_key = f"source_nsrr_L{high}"
        target_key = f"target_nsnsns_L{config['target_physical_level']}"
        diagnostics.append({
            "t": point["t"],
            "source_quadrature_relative_change": fine["values"][source_key]["Q"] / coarse["values"][source_key]["Q"] - 1,
            "target_quadrature_relative_change": fine["values"][target_key]["Q"] / coarse["values"][target_key]["Q"] - 1,
            "source_level_relative_change": fine["values"][source_key]["Q"] / fine["values"][f"source_nsrr_L{low}"]["Q"] - 1,
            "raw_modular_ratio": fine["source_over_target_by_source_level"][str(high)],
        })
    result = {"schema": SCHEMA, "config": config, "implementation_fingerprint": implementation,
              "quantity": "Q=Z_superLiouville/(Z_free_X+psi)^kappa", "kappa": kappa,
              "rows": rows, "convergence_diagnostics": diagnostics, "shards_validated": count,
              "maximum_ward_residual": max(s["maximum_ward_residual"] for s in shards if s["channel"] == CHANNELS[0]),
              "runtime_seconds_sum": math.fsum(s["runtime_seconds"] for s in shards),
              "interpretation": "Raw normalization; no fitted prefactor. Quadrature/level variations are convergence diagnostics, not certified error bars."}
    write_json(output, result)
    plot_svg(result, output.with_suffix(".svg"))
    return result


def plot_svg(result, path):
    """Dependency-free scientific SVG, produced by the cluster reducer."""
    from html import escape
    cfg = result["config"]
    levels = cfg["source_physical_levels"]
    target_level = cfg["target_physical_level"]
    orders = cfg["quadrature_orders"]
    source_key = f"source_nsrr_L{max(levels)}"
    target_key = f"target_nsnsns_L{target_level}"
    curves = []
    colors = {CHANNELS[0]: "#ba432f", CHANNELS[1]: "#176dad"}
    for order in orders:
        selected = [r for r in result["rows"] if r["quadrature_order"] == order]
        for channel, key, level in ((CHANNELS[0], source_key, max(levels)), (CHANNELS[1], target_key, target_level)):
            curves.append((f"{channel} L={level}, N={order}", colors[channel], order != max(orders),
                           [(r["t"], r["values"][key]["Q"]) for r in selected]))
    ratio_curves = []
    for order in orders:
        selected = [r for r in result["rows"] if r["quadrature_order"] == order]
        for level in levels:
            ratio_curves.append((f"NSRR L={level} / NSNSNS L={target_level}, N={order}",
                                 "#7544a0" if level == max(levels) else "#4a8460", order != max(orders),
                                 [(r["t"], r["source_over_target_by_source_level"][str(level)]) for r in selected]))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="890" viewBox="0 0 1100 890">',
           '<rect width="1100" height="890" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#20242b">']
    def label(x, y, value, size=15, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(value)}</text>')
    label(80, 34, "Genus-two modular check: NSRR and transformed NSNSNS", 23)
    label(80, 61, f"b={cfg['parameters']['b']}; Omega(t)=[[i,t+0.5i],[t+0.5i,i]]; kappa={result['kappa']:.8f}")
    label(80, 84, "Same-frame free X+psi normalization; raw values, no fitted rescaling", 14)
    for top, data, title, unity in ((135, curves, "Q = Z_SL / (Z_free)^kappa", False),
                                    (480, ratio_curves, "Q_NS RR / Q_NS NS NS", True)):
        left, width, height = 100, 620, 245
        points = [p for _, _, _, curve in data for p in curve]
        xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
        ys = [p[1] for p in points] + ([1.] if unity else [])
        low, high = min(ys), max(ys)
        margin = .10 * (high - low or max(abs(high), 1e-10))
        low, high = low - margin, high + margin
        def xy(x, y):
            return left + width * (x - xmin) / (xmax - xmin or 1), top + height * (high - y) / (high - low)
        label(left, top - 17, title, 18)
        for j in range(6):
            y = low + (high - low) * j / 5
            yy = xy(xmin, y)[1]
            svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e2e5e9"/>')
            label(left - 10, yy + 5, f"{y:.4g}", 13, "end")
        for x in sorted(set(p[0] for p in points)):
            xx = xy(x, low)[0]
            label(xx, top + height + 23, f"{x:.2f}", 13, "middle")
        svg.append(f'<path d="M {left} {top} v {height} h {width}" fill="none" stroke="#444"/>')
        if unity:
            yy = xy(xmin, 1.)[1]
            svg.append(f'<path d="M {left} {yy} h {width}" stroke="#777" stroke-dasharray="3 4"/>')
        for i, (name, color, dashed, curve) in enumerate(data):
            coords = " ".join(f"{xy(x,y)[0]:.3f},{xy(x,y)[1]:.3f}" for x,y in curve)
            dash = ' stroke-dasharray="7 4"' if dashed else ""
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
            for x, y in curve:
                xx, yy = xy(x, y)
                svg.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{color}"/>')
            yy = top + 25 + 35 * i
            svg.append(f'<path d="M 750 {yy-5} h 30" stroke="{color}" stroke-width="2"{dash}/>')
            label(790, yy, name, 12)
        label(left + width / 2, top + height + 48, "t = Re Omega_12 (source marking)", 15, "middle")
    label(80, 826, f"Ward residual maximum: {result['maximum_ward_residual']:.3e}; acceptance cutoff: {cfg['numerics']['maximum_ward_residual']:.0e}", 14)
    label(80, 849, "N and level changes show numerical sensitivity; they are not certified error bars.", 14)
    label(80, 872, "The Liouville cosmological prefactor is " + ("included." if cfg['parameters']['include_cosmological_prefactor'] else "omitted consistently in both channels."), 14)
    svg.append('</g></svg>')
    Path(path).write_text("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-config"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan"); p.add_argument("--task-count-only", action="store_true")
    p = sub.add_parser("worker")
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--task-index", type=int, required=True)
    for name in ("channel-worker", "channel-chunk-count"):
        p = sub.add_parser(name)
        p.add_argument("--channel", choices=CHANNELS, required=True)
        p.add_argument("--tasks-per-chunk", type=int, required=True)
        if name == "channel-worker":
            p.add_argument("--chunk-index", type=int, required=True)
            p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("reduce")
    p.add_argument("--shard-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-config":
        build_config(args.output); return
    config = _load(args.config)
    validate_config(config)
    if args.command == "plan":
        print(len(tasks(config)) if args.task_count_only else json.dumps({
            "tasks": len(tasks(config)), "scan_points": len(config["points"]),
            "source_physical_levels": config["source_physical_levels"],
            "target_physical_level": config["target_physical_level"],
            "quadrature_orders": config["quadrature_orders"], "fresh_process_per_node": True}, indent=2))
    elif args.command == "worker":
        print(worker(args.config, args.output_dir, args.task_index), flush=True)
    elif args.command == "channel-chunk-count":
        print(math.ceil(len(channel_indices(config, args.channel)) / args.tasks_per_chunk))
    elif args.command == "channel-worker":
        channel_worker(args.config, args.output_dir, args.channel, args.chunk_index, args.tasks_per_chunk)
    elif args.command == "reduce":
        reduce_scan(args.config, args.shard_dir, args.output)
        print(args.output); print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
