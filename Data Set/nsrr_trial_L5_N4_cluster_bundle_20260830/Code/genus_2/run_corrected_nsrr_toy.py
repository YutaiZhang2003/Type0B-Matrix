#!/usr/bin/env python3
"""Corrected NSRR diagonal-norm toy, NOT a physical partition function.

For a=(f,eta), A_a=C_eta * prod(q_e**h_e) * F_f^(eta,eta), this
computes H_ab=integral A_a conjugate(A_b) prod(dP_e/pi) and D=trace(H).
The identity metric defining D is a diagnostic choice, NOT a derivation of
the nonchiral Ramond sewing projector. Opposite-HJS blocks are omitted, not
set to zero. No free-spin conversion, multiplicity, or fitted normalization
is used. All four inequivalent literal lift choices are retained.

Only the supported corrected double-Virasoro path runs in production.
The genus-two PBW oracle is replaced by a raising mock inside each worker
to ensure that no hidden fallback supplies physical block coefficients.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html import escape
from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

import nsrr_nsnsns_theta_omega_scan as scan
import nsrr_double_virasoro_block as dv
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from recompute_all_ns_reference import protected_hashes


SCHEMA = "corrected-nsrr-diagonal-norm-toy-v1"
CHANNELS = ((0, 1), (0, -1), (1, 1), (1, -1))
LIFTS = tuple((r0, r1, 1) for r0, r1 in product((1, -1), repeat=2))


def encode(z):
    return [float(complex(z).real), float(complex(z).imag)]


def decode(z):
    return complex(*z)


def fingerprint():
    return hashlib.sha256((scan.fingerprint()+hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def make_config(geometry_path):
    geometry = scan._load(geometry_path)
    if geometry["schema"] != "nsrr-human-note-marked-geometry-v2":
        raise ValueError("use the corrected, re-marked NSRR geometry")
    if tuple(geometry["geometry_edge_sectors"]) != GEOMETRY_SECTORS:
        raise ValueError("expected geometric sectors (R,R,NS)")
    points = []
    for point in geometry["points"]:
        chart = point["source_chart"]
        q = tuple(complex(x) for x in chart["q_values"])
        measured = scan.solve_theta_collocation(*q, basis_order=32, samples_per_seam=160)
        error = float(np.max(abs(measured.omega-scan.complex_matrix(chart["omega"]))))
        if error > 1e-8:
            raise ValueError("fresh NSRR forward-period check failed")
        points.append({"t": point["t"], "q_geometry": chart["q_values"],
                       "omega_remarked_source": chart["omega"], "forward_period_error": error})
    result = {
        "schema": SCHEMA, "implementation_fingerprint": fingerprint(),
        "geometry_path": str(Path(geometry_path).resolve()), "geometry_digest": scan._digest(geometry),
        "b": 1.4, "quadrature_orders": [2, 3], "levels": [0, 1, 2],
        "structure_dps": 30, "include_cosmological_prefactor": False,
        "geometry_sectors": list(GEOMETRY_SECTORS), "points": points,
        "lifts_geometry": [list(x) for x in LIFTS],
        "channels": [{"form_parity": f, "eta_left": eta, "eta_right": eta} for f, eta in CHANNELS],
        "quadrature_envelope_geometry": [max(abs(complex(p["q_geometry"][e])) for p in points) for e in range(3)],
        "quantity": "H_ab=integral A_a*conjugate(A_b); D=trace(H); A_(f,eta)=C_eta*primary*F_f^(eta,eta)",
        "scope": "coefficient-weighted diagonal norm of supported chiral channels, NOT physical Z or Q",
        "physical_Z": None, "physical_Q": None,
        "opposite_HJS_channels": "not computed and not asserted zero",
        "literal_lift_to_physical_spin_dictionary": "unassigned; all four representatives retained",
        "sewing_metric": "identity for the diagnostic trace only; not the physical Ramond projector",
        "coefficient_convention": "C_eta is BRY C_even/C_odd; rho_1 phases remain in the corrected block; a norm cannot certify the missing nonchiral odd sign",
        "method": "branching recursion times two ordinary Virasoro c-recursions; equal-sign Ward support; ordinary lift sum",
        "PBW_production_fallback": False,
    }
    validate_config(result)
    return result


def validate_config(config):
    protected_hashes()
    if config["schema"] != SCHEMA or config["implementation_fingerprint"] != fingerprint():
        raise ValueError("config implementation mismatch")
    if config["physical_Z"] is not None or config["physical_Q"] is not None:
        raise ValueError("the toy cannot be labelled a physical partition")
    if config["levels"] != [0, 1, 2] or config["quadrature_orders"] != [2, 3]:
        raise ValueError("unexpected toy cutoffs")
    if config["lifts_geometry"] != [list(x) for x in LIFTS] or config["PBW_production_fallback"]:
        raise ValueError("do not select a preferred physical lift or enable fallback")
    ts = [p["t"] for p in config["points"]]
    if ts != [.52, .56, .60, .64, .68]:
        raise ValueError("unexpected period family")


def tasks(config):
    return [(n, i) for n in config["quadrature_orders"] for i in range(n**3)]


def node_data(config, index):
    if not isinstance(index, int) or not 0 <= index < len(tasks(config)):
        raise ValueError("invalid toy node index")
    n, node = tasks(config)[index]
    indices = np.unravel_index(node, (n,)*3)
    rules = scan._rules(config["quadrature_envelope_geometry"], n)
    return (n, node, tuple(float(rules[e][0][indices[e]]) for e in range(3)),
            float(scan._measure(rules, indices)))


def evaluate_node(config, index):
    """Return actual NSRR block data; no all-NS computation is invoked."""
    validate_config(config)
    started = time.perf_counter()
    n, node, momenta, measure = node_data(config, index)
    probe = NSRRPlumbingInputs(tuple(complex(x) for x in config["points"][0]["q_geometry"]),
                              LIFTS[0], GEOMETRY_SECTORS)
    p_ns, p_one, p_zero = probe.momenta_slots(momenta)
    constants = scan.GenericSuperLiouvilleConstants(config["b"], dps=config["structure_dps"])
    c_eta = constants.rr_ns_constants(p_one, p_zero, p_ns)
    with patch.object(dv, "HumanNSRRThetaOracle", side_effect=AssertionError("PBW production fallback forbidden")) as forbidden:
        runtime = dv.NSRRDoubleVirasoroTheta(
            b=config["b"], physical_momenta=(p_ns, p_one, p_zero),
            cutoff=max(config["levels"]), completion="none")
        series = {(f, eta, k): runtime.physical_series(f, eta, eta, k)
                  for f, eta in CHANNELS for k in range(8)}
        rows = []
        for point in config["points"]:
            for level in config["levels"]:
                for lifts in LIFTS:
                    plumbing = NSRRPlumbingInputs(tuple(complex(x) for x in point["q_geometry"]),
                                                  lifts, GEOMETRY_SECTORS)
                    k = dv.spin_character_index(plumbing.lifts_slots)
                    primary = plumbing.primary(config["b"], momenta)
                    blocks = [dv.evaluate_twice_level_series(
                        {e: v for e, v in series[f, eta, k].items() if sum(e) <= 2*level},
                        plumbing.q_slots) for f, eta in CHANNELS]
                    amplitudes = [c_eta[0 if eta == 1 else 1]*primary*block
                                  for (f, eta), block in zip(CHANNELS, blocks)]
                    rows.append({"t": point["t"], "level": level, "lifts_geometry": list(lifts),
                                 "primary": encode(primary), "blocks": [encode(z) for z in blocks],
                                 "amplitudes": [encode(z) for z in amplitudes]})
        fallback_calls = forbidden.call_count
    result = {"schema": SCHEMA, "implementation_fingerprint": fingerprint(),
              "config_digest": scan._digest(config), "index": index, "quadrature_order": n,
              "node": node, "momenta_geometry": list(momenta), "momenta_slots": [p_ns, p_one, p_zero],
              "measure": measure, "C_eta": [encode(z) for z in c_eta], "values": rows,
              "ward_residual": runtime.ward_residual_maximum,
              "PBW_production_calls": fallback_calls, "runtime_seconds": time.perf_counter()-started}
    validate_shard(config, index, result)
    return result


def validate_shard(config, index, shard):
    n, node, momenta, measure = node_data(config, index)
    expected = {"schema": SCHEMA, "implementation_fingerprint": fingerprint(),
                "config_digest": scan._digest(config), "index": index, "quadrature_order": n,
                "node": node, "momenta_geometry": list(momenta), "momenta_slots": list(momenta[::-1]),
                "measure": measure, "PBW_production_calls": 0}
    for key, value in expected.items():
        if shard.get(key) != value:
            raise ValueError(f"NSRR shard {index}: {key} mismatch")
    design = [(p["t"], level, list(lifts)) for p in config["points"]
              for level in config["levels"] for lifts in LIFTS]
    if [(r["t"], r["level"], r["lifts_geometry"]) for r in shard["values"]] != design:
        raise ValueError("missing/reordered NSRR evaluations")
    if not 0 <= shard["ward_residual"] < 1e-7:
        raise ValueError("branching Ward solve failed")
    for row in shard["values"]:
        if len(row["blocks"]) != 4 or len(row["amplitudes"]) != 4:
            raise ValueError("missing supported HJS channels")
        if not all(math.isfinite(x) for z in row["blocks"]+row["amplitudes"] for x in z):
            raise ValueError("nonfinite NSRR value")
    if config["implementation_fingerprint"] != fingerprint():
        raise ValueError("implementation changed during NSRR evaluation")


def worker(config_path, output_dir, index):
    config = scan._load(config_path)
    validate_config(config)
    path = Path(output_dir)/f"node-{index:03d}.json"
    if path.exists():
        validate_shard(config, index, scan._load(path))
    else:
        scan.write_json(path, evaluate_node(config, index))


def reduce_run(config_path, output_dir):
    config = scan._load(config_path)
    validate_config(config)
    paths = [Path(output_dir)/"shards"/f"node-{i:03d}.json" for i in range(len(tasks(config)))]
    if set(paths) != set((Path(output_dir)/"shards").glob("node-*.json")):
        raise ValueError("incomplete toy grid")
    shards = [scan._load(p) for p in paths]
    for i, shard in enumerate(shards):
        validate_shard(config, i, shard)
    rows = []
    for n in config["quadrature_orders"]:
        selected = [s for s in shards if s["quadrature_order"] == n]
        for j, design in enumerate(selected[0]["values"]):
            matrix = np.zeros((4, 4), dtype=complex)
            for shard in selected:
                amplitudes = np.array([decode(z) for z in shard["values"][j]["amplitudes"]])
                matrix += shard["measure"]*np.outer(amplitudes, amplitudes.conjugate())
            norm = max(float(np.trace(matrix).real), 1e-280)
            if np.max(abs(matrix-matrix.conjugate().T)) > 1e-12*norm or np.linalg.eigvalsh(matrix).min() < -1e-12*norm:
                raise ArithmeticError("the diagnostic Gram matrix is not Hermitian positive")
            rows.append({"t": design["t"], "quadrature_order": n, "level": design["level"],
                         "lifts_geometry": design["lifts_geometry"],
                         "H": [[encode(z) for z in row] for row in matrix],
                         "diagonal_channels": [float(z.real) for z in np.diag(matrix)],
                         "D_diagnostic": float(np.trace(matrix).real), "physical_Z": None, "physical_Q": None})
    result = {"schema": SCHEMA, "config": config, "implementation_fingerprint": fingerprint(),
              "status": "completed corrected NSRR diagonal-norm toy; physical partition not computed",
              "rows": rows, "shards_validated": len(shards),
              "PBW_production_calls": sum(s["PBW_production_calls"] for s in shards),
              "maximum_ward_residual": max(s["ward_residual"] for s in shards),
              "runtime_seconds_sum": sum(s["runtime_seconds"] for s in shards),
              "protected_kernel_hashes": protected_hashes()}
    scan.write_json(Path(output_dir)/"summary.json", result)
    plot_svg(result, Path(output_dir)/"nsrr_diagonal_toy.svg")
    return result


def plot_svg(result, path):
    rows = [r for r in result["rows"] if r["quadrature_order"] == 3]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="810" viewBox="0 0 1060 810">',
           '<rect width="1060" height="810" fill="white"/><g font-family="Arial,sans-serif" fill="#253044">']
    def label(x, y, s, size=15, anchor="start"):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(s))}</text>')
    label(75, 35, "Corrected NSRR toy: diagonal norm D, not physical Z or Q", 23)
    label(75, 62, "b=1.4; N=3; branching recursion x two Virasoro c-recursions; no PBW production fallback")
    label(75, 86, "Four literal lift representatives; no assignment to a physical Ramond spin structure")
    for panel, lifts in enumerate(LIFTS):
        left, top, width, height = 120+(panel%2)*495, 153+(panel//2)*290, 345, 195
        selected = [r for r in rows if r["lifts_geometry"] == list(lifts)]
        low, high = min(r["D_diagnostic"] for r in selected), max(r["D_diagnostic"] for r in selected)
        margin = .1*(high-low or high or 1)
        low, high = low-margin, high+margin
        def xy(x, y):
            return left+width*(x-.52)/.16, top+height*(high-y)/(high-low)
        label(left, top-20, f"lifts (R0,R1,NSinf) = {lifts}", 16)
        for j in range(4):
            y = low+(high-low)*j/3
            yy = xy(.52, y)[1]
            svg.append(f'<path d="M {left} {yy} h {width}" stroke="#e1e6ee"/>')
            label(left-8, yy+4, f"{y:.2e}", 12, "end")
        for t in (.52, .56, .60, .64, .68):
            label(xy(t, low)[0], top+height+23, f"{t:.2f}", 12, "middle")
        for level, color in ((0, "#8a8a8a"), (1, "#d67a20"), (2, "#176dad")):
            curve = [r for r in selected if r["level"] == level]
            coords = " ".join(f"{xy(r['t'],r['D_diagnostic'])[0]:.3f},{xy(r['t'],r['D_diagnostic'])[1]:.3f}" for r in curve)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            label(left+level*110, top+height+47, f"L={level}", 13)
            svg.append(f'<path d="M {left+33+level*110} {top+height+43} h 30" stroke="{color}" stroke-width="3"/>')
    label(75, 750, "Horizontal axis: t = Re Omega_original,12. Opposite-HJS channels are uncomputed, not zero.")
    label(75, 777, "D = integral sum_(f,eta) |C_eta * primary * F_f^(eta,eta)|^2. Unit diagnostic metric; no free normalization.", 14)
    svg.append('</g></svg>')
    Path(path).write_text("\n".join(svg)+"\n")


def run(args):
    output = args.output_dir.resolve()
    config_path = output/"config.json"
    if config_path.exists():
        config = scan._load(config_path)
        validate_config(config)
        if config["geometry_digest"] != scan._digest(scan._load(args.geometry)):
            raise ValueError("geometry changed since this toy run")
    else:
        config = make_config(args.geometry)
        scan.write_json(config_path, config)
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def evaluate(index):
        with (logs/f"node-{index:03d}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config_path),
                            "--output-dir", str(output/"shards"), "--index", str(index)],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return index
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, i) for i in range(len(tasks(config)))]
        for completed, future in enumerate(as_completed(futures), 1):
            print(f"{completed}/35 NSRR nodes complete; last={future.result()}", flush=True)
    result = reduce_run(config_path, output)
    print(f"Completed corrected NSRR toy in {time.perf_counter()-started:.1f}s", flush=True)
    for row in result["rows"]:
        if row["level"] == 2 and row["quadrature_order"] == 3:
            print({k: row[k] for k in ("t", "lifts_geometry", "D_diagnostic")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
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
