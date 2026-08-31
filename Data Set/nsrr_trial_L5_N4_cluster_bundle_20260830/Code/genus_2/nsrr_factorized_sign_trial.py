#!/usr/bin/env python3
r"""Low-order test of a specified NSRR factorized sewing ANSATZ.

This is not the physical Ramond projector or a certified partition.
All eight literal Human-Note chiral components are retained. Equal-sign
components use branching recursion and two Virasoro c-recursions; missing
opposite-sign components use the explicit, capped PBW DIAGNOSTIC completion.

Trial vertex: t_(f;eta,zeta)=i**f*c_eta*delta_eta,zeta,
              (c_+,c_-)=(C_even,C_odd)/2.
Trial antiholomorphic identification: Ftilde=conjugate(F), at real momenta.
Both are hypotheses at this boundary, not consequences of the grading proof.
The (-1)**f factor and the two vertex phases are evaluated separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

import nsrr_double_virasoro_block as dv
from theta_star_algebra import fwht
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants
from fixed_spin_free_plumbing import fixed_spin_partition
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from recompute_all_ns_reference import protected_hashes
from compare_nsrr_nsnsns_theta import _rules, _measure
from plumbing_algorithms import solve_theta_collocation


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "nsrr-factorized-sign-trial-v1"
CHANNELS = tuple(product((0, 1), (1, -1), (1, -1)))
LIFTS = tuple((r0, r1, 1) for r0, r1 in product((1, -1), repeat=2))
SOURCE_BRANCH = ((0, 0), (0, 1))


def encode(z):
    return [float(complex(z).real), float(complex(z).imag)]


def decode(z):
    return complex(*z)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def fingerprint():
    files = [Path(__file__), Path(dv.__file__),
             ROOT/"Code/genus_2/fixed_spin_free_plumbing.py",
             ROOT/"Code/genus_2/physical_free_plumbing_resummation.py",
             ROOT/"Code/genus_2/nsrr_plumbing_adapter.py",
             ROOT/"Code/c_Recursion/generic_super_liouville_structure_constants.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def low_level_coefficients(b, momenta_slots, f, eta, etap, lifts_slots):
    """Independent analytic checks, equations (14)/(16) of the derivation."""
    pns, pr1, pr2 = momenta_slots
    h = (b+1/b)**2/8+pns*pns/2
    beta2, beta3 = 1j*pr1/math.sqrt(2), 1j*pr2/math.sqrt(2)
    l1, l2, l3 = lifts_slots
    k = eta*etap
    a = (beta3-eta*beta2)*(beta3-etap*beta2)
    if f == 0:
        return 1+k*l2*l3, -l1*a*(l3-k*l2)/(2*h)
    return -1j*(l3-k*l2), -1j*l1*a*(1+k*l2*l3)/(2*h)


def block_components(b, momenta_slots, cutoff):
    if cutoff not in (1, 2):
        raise ValueError("this diagnostic completion is bounded to level 1 or 2")
    with patch.object(dv, "HumanNSRRThetaOracle", wraps=dv.HumanNSRRThetaOracle) as oracle:
        runtime = dv.NSRRDoubleVirasoroTheta(
            b=b, physical_momenta=momenta_slots, cutoff=cutoff,
            completion="pbw_diagnostic", pbw_completion_max_level=2)
        components = {channel: runtime.physical_components(*channel) for channel in CHANNELS}
        calls = oracle.call_count
    if calls != 4:
        raise ArithmeticError("expected explicit PBW completion for exactly four opposite-sign channels")
    error = 0.
    for channel, vectors in components.items():
        for lifts in product((1, -1), repeat=3):
            k = dv.spin_character_index(lifts)
            expected = low_level_coefficients(b, momenta_slots, *channel, lifts)
            for exponent, target in zip(((0, 0, 0), (1, 0, 0)), expected):
                actual = fwht(vectors[exponent])[k]
                error = max(error, abs(actual-target)/max(1., abs(target)))
    if error > 1e-10 or runtime.ward_residual_maximum > 1e-8:
        raise ArithmeticError("a low-level normalization or branching check failed")
    return components, {"explicit_PBW_completion_calls": calls,
                        "analytic_ground_half_level_max_error": error,
                        "branching_ward_residual": runtime.ward_residual_maximum}


def evaluate_blocks(components, q_slots, lifts_slots, level):
    k = dv.spin_character_index(lifts_slots)
    return {channel: dv.evaluate_twice_level_series(
        {e: fwht(v)[k] for e, v in vectors.items() if sum(e) <= round(2*level)}, q_slots)
        for channel, vectors in components.items()}


def contract(blocks, anti_blocks, bry_constants, *, odd_vertex_phase=1j, sewing_sign=True):
    """Explicit diagonal-vertex trial; NO inferred physical projection.

    The two vertex coefficients are multiplied, not absolute-squared.
    C_even/C_odd select eta, while odd_vertex_phase selects f.
    """
    if set(blocks) != set(CHANNELS) or set(anti_blocks) != set(CHANNELS):
        raise ValueError("all eight channels, including mixed signs, are required")
    if len(bry_constants) != 2 or not np.isclose(abs(odd_vertex_phase), 1):
        raise ValueError("two BRY constants and a unit odd vertex phase are required")
    c = {1: complex(bry_constants[0])/2, -1: complex(bry_constants[1])/2}
    terms = {}
    for f, eta, etap in CHANNELS:
        phase = odd_vertex_phase**f
        left, right = phase*c[eta], phase*c[etap]
        sign = (-1)**f if sewing_sign else 1
        terms[f, eta, etap] = sign*left*right*blocks[f, eta, etap]*anti_blocks[f, eta, etap]
    return {"terms": terms,
            "even": sum(v for (f, _, _), v in terms.items() if f == 0),
            "odd": sum(v for (f, _, _), v in terms.items() if f == 1),
            "equal": sum(v for (_, a, b), v in terms.items() if a == b),
            "mixed": sum(v for (_, a, b), v in terms.items() if a != b),
            "total": sum(terms.values())}


def make_config(geometry_path):
    protected = protected_hashes()
    geometry = load(geometry_path)
    if geometry["schema"] != "nsrr-human-note-marked-geometry-v2":
        raise ValueError("use the re-marked NS-at-infinity geometry")
    if tuple(geometry["geometry_edge_sectors"]) != GEOMETRY_SECTORS:
        raise ValueError("geometry must have sectors (R,R,NS)")
    points = []
    for point in geometry["points"]:
        chart = point["source_chart"]
        q = tuple(complex(z) for z in chart["q_values"])
        omega = np.asarray([[complex(z) for z in row] for row in chart["omega"]])
        forward = solve_theta_collocation(*q, basis_order=32, samples_per_seam=160)
        error = float(np.max(abs(forward.omega-omega)))
        if error > 1e-8:
            raise ValueError("fresh forward period audit failed")
        free = fixed_spin_partition(q, omega, ((1, 1), (0, 0)),
                                    period_branch=SOURCE_BRANCH, max_mode=32)
        points.append({"t": point["t"], "q_geometry": chart["q_values"],
                       "omega_source": chart["omega"], "forward_period_error": error,
                       "Z_free_reference": free["Z_free"],
                       "charged_period_error": free["period_residual"]})
    config = {
        "schema": SCHEMA, "b": 1.4, "max_level": 2,
        "levels": [0, .5, 1, 1.5, 2], "quadrature_orders": [2, 3],
        "channels": [list(c) for c in CHANNELS], "lifts_geometry": [list(l) for l in LIFTS],
        "points": points, "geometry_path": str(Path(geometry_path).resolve()),
        "geometry_sha256": hashlib.sha256(Path(geometry_path).read_bytes()).hexdigest(),
        "implementation_sha256": fingerprint(), "protected_kernel_sha256": protected,
        "q_envelope": [max(abs(complex(p["q_geometry"][e])) for p in points) for e in range(3)],
        "vertex_ansatz": "t_(f;eta,zeta)=i^f c_eta delta_eta,zeta; (c_+,c_-)=(C_even,C_odd)/2",
        "antiholomorphic_ansatz": "coefficientwise conjugate chiral block at real momenta; an explicit hypothesis, not a proved physical BPZ dictionary",
        "formula": "Z_trial=integral primary*conj(primary) sum_(f,eta,eta') (-1)^f (i^f c_eta)(i^f c_eta') F_f^(eta,eta') conj(F_f^(eta,eta')) dP^3/pi^3",
        "control": "also evaluate without the explicit sewing sign and with a formal same-chiral-convention antichiral block",
        "method": "branching recursion and two Virasoro c-recursions, with explicit PBW diagnostic completion only for missing opposite-sign star channels",
        "cosmological_factor": "common factor omitted consistently, as in previous toys",
        "reference_free_spin": [[1, 1], [0, 0]], "physical_lift_spin_dictionary": None,
        "normalization": "no fitted multiplicity; Q_trial_reference=Z_trial/Z_free_reference^kappa is a diagnostic only",
        "physical_Ramond_projector": None, "physical_Z": None, "physical_Q": None,
    }
    return config


def validate_config(config):
    if config["schema"] != SCHEMA or config["implementation_sha256"] != fingerprint():
        raise ValueError("trial configuration/implementation mismatch")
    if config["protected_kernel_sha256"] != protected_hashes():
        raise ValueError("protected kernel changed")
    if config["physical_Z"] is not None or config["physical_Q"] is not None:
        raise ValueError("this ansatz is not a certified physical partition")
    if config["channels"] != [list(c) for c in CHANNELS] or config["max_level"] != 2:
        raise ValueError("missing channels or unsupported diagnostic cutoff")


def tasks(config):
    return [(n, j) for n in config["quadrature_orders"] for j in range(n**3)]


def evaluate_node(config, index):
    validate_config(config)
    started = time.monotonic()
    n, node = tasks(config)[index]
    idx = np.unravel_index(node, (n,)*3)
    rules = _rules(config["q_envelope"], n)
    momenta = tuple(float(rules[e][0][idx[e]]) for e in range(3))
    measure = _measure(rules, idx)
    constants = GenericSuperLiouvilleConstants(config["b"], dps=30)
    c = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
    components, checks = block_components(config["b"], momenta[::-1], config["max_level"])
    rows = []
    for point, lifts, level in product(config["points"], LIFTS, config["levels"]):
        plumbing = NSRRPlumbingInputs(tuple(complex(z) for z in point["q_geometry"]),
                                      lifts, GEOMETRY_SECTORS)
        primary = plumbing.primary(config["b"], momenta)
        blocks = evaluate_blocks(components, plumbing.q_slots, plumbing.lifts_slots, level)
        anti = {k: z.conjugate() for k, z in blocks.items()}
        result = contract(blocks, anti, c)
        wrong_sign = contract(blocks, anti, c, sewing_sign=False)["total"]
        formal_anti = evaluate_blocks(components, tuple(z.conjugate() for z in plumbing.q_slots),
                                      plumbing.lifts_slots, level)
        formal = contract(blocks, formal_anti, c)["total"]
        rows.append({"t": point["t"], "level": level, "lifts_geometry": list(lifts),
                     "primary": encode(primary), "blocks": [encode(blocks[k]) for k in CHANNELS],
                     "weighted_terms": [encode(abs(primary)**2*result["terms"][k]) for k in CHANNELS],
                     **{key: encode(abs(primary)**2*result[key]) for key in ("even", "odd", "equal", "mixed", "total")},
                     "without_sewing_sign": encode(abs(primary)**2*wrong_sign),
                     "formal_same_convention_tilde": encode(abs(primary)**2*formal)})
    validate_config(config)
    return {"schema": SCHEMA, "config_digest": digest(config), "index": index,
            "quadrature_order": n, "node": node, "momenta_geometry": momenta,
            "momenta_slots": momenta[::-1], "measure": measure, "C_BRY": [encode(z) for z in c],
            "checks": checks, "rows": rows, "elapsed_seconds": time.monotonic()-started}


def validate_shard(config, index, shard):
    if shard["config_digest"] != digest(config) or shard["index"] != index:
        raise ValueError("wrong shard provenance")
    n, node = tasks(config)[index]
    if (shard["quadrature_order"], shard["node"]) != (n, node):
        raise ValueError("wrong quadrature node")
    design = [(p["t"], l, list(lifts)) for p, lifts, l in product(config["points"], LIFTS, config["levels"])]
    if [(r["t"], r["level"], r["lifts_geometry"]) for r in shard["rows"]] != design:
        raise ValueError("missing or reordered evaluations")
    if shard["checks"]["explicit_PBW_completion_calls"] != 4:
        raise ValueError("missing explicit diagnostic completion")
    for row in shard["rows"]:
        if len(row["blocks"]) != 8 or len(row["weighted_terms"]) != 8:
            raise ValueError("a chiral component was omitted")


def real_value(z):
    if abs(z.imag) > 1e-10*max(abs(z), 1e-280):
        raise ArithmeticError(f"conjugate trial unexpectedly complex: {z}")
    return float(z.real)


def reduce_run(config_path, output):
    config = load(config_path)
    validate_config(config)
    shards = [load(output/"shards"/f"node-{i:03d}.json") for i in range(len(tasks(config)))]
    for i, shard in enumerate(shards):
        validate_shard(config, i, shard)
    kappa = 1+2*(config["b"]+1/config["b"])**2
    rows = []
    for n in config["quadrature_orders"]:
        selected = [s for s in shards if s["quadrature_order"] == n]
        for j, design in enumerate(selected[0]["rows"]):
            row = {key: design[key] for key in ("t", "level", "lifts_geometry")}
            row["quadrature_order"] = n
            for key in ("even", "odd", "equal", "mixed", "total", "without_sewing_sign", "formal_same_convention_tilde"):
                values = [s["measure"]*decode(s["rows"][j][key]) for s in selected]
                z = complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))
                row[key] = encode(z) if key == "formal_same_convention_tilde" else real_value(z)
            free = next(p["Z_free_reference"] for p in config["points"] if p["t"] == row["t"])
            row["Q_trial_reference"] = row["total"]/free**kappa
            row["mixed_fraction"] = row["mixed"]/row["total"]
            rows.append(row)
    diagnostics = []
    for point in config["points"]:
        select = lambda n, level: next(r for r in rows if r["t"] == point["t"]
            and r["quadrature_order"] == n and r["level"] == level and r["lifts_geometry"] == [1, 1, 1])
        fine, low, coarse = select(3, 2), select(3, 1), select(2, 2)
        diagnostics.append({"t": point["t"], "Z_trial": fine["total"], "Q_trial_reference": fine["Q_trial_reference"],
                            "level_1_to_2_relative_change": fine["total"]/low["total"]-1,
                            "quadrature_2_to_3_relative_change": fine["total"]/coarse["total"]-1,
                            "mixed_fraction": fine["mixed_fraction"],
                            "without_sewing_sign": fine["without_sewing_sign"],
                            "formal_same_convention_tilde": fine["formal_same_convention_tilde"]})
    result = {"schema": SCHEMA, "config": config, "rows": rows, "diagnostics": diagnostics,
              "checks": {"protected_kernel_sha256": protected_hashes(),
                         "analytic_max_error": max(s["checks"]["analytic_ground_half_level_max_error"] for s in shards),
                         "explicit_PBW_completion_calls": sum(s["checks"]["explicit_PBW_completion_calls"] for s in shards)},
              "physical_Z": None, "physical_Q": None}
    save(output/"summary.json", result)
    with (output/"fivepoint_trial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    return result


def run(args):
    output = args.output_dir.resolve()
    config_path = output/"config.json"
    if config_path.exists():
        config = load(config_path)
        validate_config(config)
    else:
        config = make_config(args.geometry)
        save(config_path, config)
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def worker(index):
        with (logs/f"node-{index:03d}.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--config", str(config_path),
                            "--output-dir", str(output), "--index", str(index)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        return index
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, i) for i in range(len(tasks(config)))]
        for completed, future in enumerate(as_completed(futures), 1):
            print(f"{completed}/{len(futures)} trial nodes complete; last={future.result()}", flush=True)
    result = reduce_run(config_path, output)
    for row in result["diagnostics"]:
        print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--geometry", type=Path, required=True)
    runp.add_argument("--output-dir", type=Path, required=True)
    runp.add_argument("--workers", type=int, choices=(1, 2), default=2)
    for name in ("worker", "reduce"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        if name == "worker":
            p.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "worker":
        config = load(args.config)
        validate_config(config)
        path = args.output_dir/"shards"/f"node-{args.index:03d}.json"
        if path.exists():
            validate_shard(config, args.index, load(path))
        else:
            shard = evaluate_node(config, args.index)
            validate_shard(config, args.index, shard)
            save(path, shard)
    else:
        reduce_run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
