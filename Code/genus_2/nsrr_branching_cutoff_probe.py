#!/usr/bin/env python3
"""Independent branching/descendant cutoffs for the unchanged NSRR trial.

Only equal-HJS-sign components are varied. The opposite-sign PBW components
are frozen at L5. A supported-character pointwise quotient is a diagnostic,
not a new physical spin prescription. No checked kernel is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import fcntl
import hashlib
from itertools import product
import math
from pathlib import Path
import subprocess
import sys
import time

import nsrr_factorized_sign_trial as trial
import nsrr_trial_cluster as previous
from compute_full_block import BranchingGrid, base_twice_level
from compute_q_expansion import (
    add_to, large_c_vacuum_series, reduced_virasoro_series, series_multiply,
)
from compute_target import ActionTerm, norm_product, solve_ns_l1, solve_ramond_lminus
from nsrr_genus2_block import auxiliary_majorana_nsrr_series
from theta_star_algebra import from_star_spectrum, fwht, star_spectrum

SCHEMA = "nsrr-independent-branching-cutoff-v1"
BRANCH_CUTOFFS = (3, 4, 5, 6, 8, 10)
DESCENDANT_CUTOFFS = (4, 5, 6)


def fingerprint():
    return {**previous.implementation_hashes(),
            str(Path(__file__).resolve().relative_to(trial.ROOT)):
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def evaluate_components(series, q):
    answer = [0j]*8
    for e, vector in series.items():
        power = math.prod(q[j]**(e[j]/2) for j in range(3))
        for j, value in enumerate(vector):
            answer[j] += value*power
    return answer


def supported_quotient(numerator, denominator):
    a, b = star_spectrum(numerator), star_spectrum(denominator)
    supported = (2, 3, 4, 5)
    scale = max(1., *(abs(z) for z in b))
    if any(abs(b[k]) < 1e-10*scale for k in supported):
        raise ArithmeticError("supported auxiliary character is singular")
    if any(abs(b[k]) > 1e-12*scale for k in (0, 1, 6, 7)):
        raise ArithmeticError("auxiliary support differs from the checked convention")
    leakage = max(abs(a[k]) for k in (0, 1, 6, 7))/max(1., *(abs(z) for z in a))
    return from_star_spectrum([a[k]/b[k] if k in supported else 0j for k in range(8)]), leakage


def odd_partner(vector):
    answer = [0j]*8
    for index, z in enumerate(vector):
        p0, p1 = index & 1, (index >> 1) & 1
        answer[index ^ 4] += -1j*(-1)**(p0+p1)*z
    return answer


def cached_actions(grid, cache_dir):
    """Cache literal protected one-module action outputs, never Ward solutions.

    Each action depends only on its module, label, parity, and implementation.
    No coefficient, normalization, or precision conversion is approximated.
    Per-entry locks prevent two workers from solving the same action twice.
    """
    if grid.mp_dps:
        grid.build_actions()
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kernel = trial.protected_hashes()["Code/ramond_branching_recursion/compute_target.py"]
    for slot, module in enumerate(grid.modules):
        labels = grid.ns if slot == 0 else grid.r
        for parity, label in product((0,) if slot == 0 else (0, 1), labels):
            if slot == 0 and abs(label) < 1:
                grid.ns_actions[label] = ()
                continue
            identity = {"kernel_sha256": kernel, "sector": module.sector,
                        "b": float(module.b), "momentum": trial.encode(module.momentum),
                        "realization": module.realization, "label": str(label), "parity": parity,
                        "precision": "binary64"}
            key = trial.digest(identity)
            path = cache_dir/f"{key}.json"
            with (cache_dir/f"{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if path.exists():
                    saved = trial.load(path)
                    if saved["identity"] != identity or saved["payload_digest"] != trial.digest(saved["payload"]):
                        raise ValueError("branch action cache provenance mismatch")
                else:
                    terms, fit = solve_ns_l1(module, label) if slot == 0 else solve_ramond_lminus(module, label, parity)
                    payload = {"terms": [[str(t.label), list(t.first), list(t.second), trial.encode(t.coefficient)] for t in terms],
                               "relative_residual": float(fit["relative_residual"])}
                    saved = {"identity": identity, "payload": payload, "payload_digest": trial.digest(payload)}
                    trial.save(path, saved)
            terms = tuple(ActionTerm(Fraction(label), tuple(first), tuple(second), trial.decode(coefficient))
                          for label, first, second, coefficient in saved["payload"]["terms"])
            if slot == 0:
                grid.ns_actions[label] = terms
            else:
                grid.r_actions[slot, parity, label] = terms
            grid.action_diagnostics.append({"sector": module.sector, "slot": slot+1, "label": str(label),
                                            "parity": parity, "relative_residual": saved["payload"]["relative_residual"]})


def branch_data(b, momenta, cutoff, mp_dps=0, cache_dir=None):
    raw, diagnostics = {}, []
    original = None
    for eta in (1, -1):
        note = (1j*momenta[0], eta*1j*momenta[1], 1j*momenta[2])
        grid = BranchingGrid(b, note, cutoff, mp_dps=mp_dps)
        tick = time.monotonic()
        if cache_dir is None:
            grid.build_actions()
        else:
            cached_actions(grid, cache_dir)
        if eta == 1:
            original = grid
        for a2, a3 in product((0, 1), repeat=2):
            values, check = grid.solve(a2, a3)
            if eta == -1:
                values = {(n1, -n2, n3): z for (n1, n2, n3), z in values.items()}
            raw[eta, a2, a3] = values
            diagnostics.append(float(check["relative_residual"]))
        print(f"branching eta={eta}: {time.monotonic()-tick:.2f}s", flush=True)
    assert original is not None
    labels = tuple(n for n in product(original.ns, original.r, original.r)
                   if base_twice_level(n) <= 2*cutoff)
    return original, raw, labels, max(diagnostics)


def branch_prefactors(b, momenta, raw, labels, eta):
    twice_n = int(2*labels[0])
    pairs = ((0, 0), (1, 1)) if twice_n % 2 == 0 else ((0, 1), (1, 0))
    answer = []
    for a2, a3 in pairs:
        norm = norm_product(labels, a2, a3, b, tuple(1j*p for p in momenta))
        coefficient = (-1)**(twice_n+twice_n*(a2+a3)+a2*a3)*raw[eta, a2, a3][labels]**2/norm**2
        answer.append(((twice_n % 2) | (a2 << 1) | (a3 << 2), complex(coefficient)))
    return answer


def make_products(grid, labels, descendant_cutoff):
    vacuum, _ = large_c_vacuum_series(descendant_cutoff)
    vacuum2 = series_multiply(vacuum, vacuum, descendant_cutoff)
    answer = {}
    for n in labels:
        copies = [reduced_virasoro_series(grid.weights.central_charges[c],
                  grid.weights.triple(n, c), descendant_cutoff) for c in (0, 1)]
        answer[n] = series_multiply(series_multiply(*copies, descendant_cutoff),
                                    vacuum2, descendant_cutoff)
    return answer


def formal_enlarged(b, momenta, raw, products, total_cutoff, eta):
    """Independent assembly for comparison with the protected total-L series."""
    answer = {}
    for n, descendants in products.items():
        base = (int(4*n[0]**2), int(4*n[1]**2-Fraction(1, 4)),
                int(4*n[2]**2-Fraction(1, 4)))
        for parity, prefactor in branch_prefactors(b, momenta, raw, n, eta):
            bits = tuple((parity >> j) & 1 for j in range(3))
            for d, z in descendants.items():
                e = tuple(base[j]+2*d[j] for j in range(3))
                if sum(e) <= 2*total_cutoff:
                    add_to(answer, e+bits, prefactor*z)
    return answer


def numerical_shells(b, momenta, raw, products, q, descendant_cutoffs):
    """No global total-level filter: keep the branch shift PLUS D descendants."""
    answer = {(eta, d): {} for eta, d in product((1, -1), descendant_cutoffs)}
    for n, descendants in products.items():
        shell = base_twice_level(n)
        base = (2*n[0]**2, 2*n[1]**2-Fraction(1, 8), 2*n[2]**2-Fraction(1, 8))
        primary = math.prod(q[j]**float(base[j]) for j in range(3))
        values = {D: sum(z*math.prod(q[j]**d[j] for j in range(3))
                         for d, z in descendants.items() if sum(d) <= D)
                  for D in descendant_cutoffs}
        for eta in (1, -1):
            prefactors = branch_prefactors(b, momenta, raw, n, eta)
            for D, value in values.items():
                vector = answer[eta, D].setdefault(shell, [0j]*8)
                for index, prefactor in prefactors:
                    vector[index] += prefactor*primary*value
    return answer


def cumulative_vector(shells, cutoff):
    return [sum(v[j] for s, v in shells.items() if s <= 2*cutoff) for j in range(8)]


def prepare(reference_dir, output, t=.6):
    reference_dir = reference_dir.resolve()
    summary = trial.load(reference_dir/"summary.json")
    previous.validate_config(summary["config"])
    audit = trial.load(reference_dir/"verification.json")
    if audit["summary_digest"] != trial.digest(summary):
        raise ValueError("reference is not the audited L5 run")
    if summary["config"]["quadrature_orders"] != [3]:
        raise ValueError("this local probe fixes the N3 quadrature")
    point = next(p for p in summary["config"]["points"] if p["t"] == t)
    q = tuple(complex(z) for z in point["q_geometry"])[::-1]
    auxiliary = auxiliary_majorana_nsrr_series(maximum_total_twice_level=32)
    aux_values = {L: evaluate_components({e: v for e, v in auxiliary.items() if sum(e) <= 2*L}, q)
                  for L in (10, 12, 14, 16)}
    aux_error = max(abs(a-b)/max(1., abs(b)) for a, b in zip(aux_values[14], aux_values[16]))
    if aux_error > 1e-11:
        raise ArithmeticError("auxiliary pointwise denominator is not converged")
    config = {"schema": SCHEMA, "reference_dir": str(reference_dir),
              "action_cache_dir": str((output/"actions_cache").resolve()),
              "reference_summary_digest": trial.digest(summary), "point": point,
              "b": summary["config"]["b"], "quadrature_order": 3,
              "branch_cutoffs": BRANCH_CUTOFFS, "descendant_cutoffs": DESCENDANT_CUTOFFS,
              "auxiliary_values": {str(L): [trial.encode(z) for z in v] for L, v in aux_values.items()},
              "auxiliary_L14_L16_scaled_error": aux_error,
              "implementation_sha256": fingerprint(), "protected_kernel_sha256": trial.protected_hashes(),
              "scope": "equal-sign supported-character branch-sum diagnostic; opposite-sign blocks frozen at PBW L5",
              "physical_Q": None, "physical_Z": None}
    if (output/"config.json").exists():
        if trial.digest(trial.load(output/"config.json")) != trial.digest(config):
            raise FileExistsError("refusing to replace different probe inputs")
    else:
        trial.save(output/"config.json", config)
    return config


def validate(config):
    if config["schema"] != SCHEMA or config["implementation_sha256"] != fingerprint():
        raise ValueError("probe implementation changed")
    if config["protected_kernel_sha256"] != trial.protected_hashes():
        raise ValueError("protected kernels changed")
    if config["physical_Q"] is not None or config["physical_Z"] is not None:
        raise ValueError("this is not a physical partition computation")


def evaluate_node(config, index, mp_dps=0):
    validate(config)
    start = time.monotonic()
    reference = trial.load(Path(config["reference_dir"])/f"shards/node-{index:03d}.json")
    source = trial.load(Path(config["reference_dir"])/"summary.json")
    if trial.digest(source) != config["reference_summary_digest"]:
        raise ValueError("reference summary changed")
    previous.audit_shard(source["config"], index, reference)
    momenta = tuple(reference["momenta_slots"])
    K, D = max(config["branch_cutoffs"]), max(config["descendant_cutoffs"])
    grid, raw, labels, ward_error = branch_data(config["b"], momenta, K, mp_dps, config["action_cache_dir"])
    tick = time.monotonic()
    products = make_products(grid, labels, D)
    print(f"{len(labels)} products through D={D}: {time.monotonic()-tick:.2f}s", flush=True)
    q = tuple(complex(z) for z in config["point"]["q_geometry"])[::-1]
    shells = numerical_shells(config["b"], momenta, raw, products, q, config["descendant_cutoffs"])
    auxiliary = [trial.decode(z) for z in config["auxiliary_values"]["16"]]
    # A separate low-L runtime tests high-grid conditioning as well as assembly.
    small = trial.dv.NSRRDoubleVirasoroTheta(b=config["b"], physical_momenta=momenta,
                                            cutoff=5, completion="none")
    formal_error = point_error = 0.
    for eta in (1, -1):
        actual = formal_enlarged(config["b"], momenta, raw, products, 5, eta)
        expected = small.enlarged_series(0, eta, eta)
        for key in actual.keys() | expected.keys():
            formal_error = max(formal_error, abs(actual.get(key, 0j)-expected.get(key, 0j))/max(1., abs(expected.get(key, 0j))))
        vectors = []
        for series in (actual, expected):
            value = [0j]*8
            for key, z in series.items():
                parity_index = key[3] | (key[4] << 1) | (key[5] << 2)
                value[parity_index] += z*math.prod(q[j]**(key[j]/2) for j in range(3))
            vectors.append(value)
        point_error = max(point_error, *(abs(a-e)/max(1., abs(e)) for a, e in zip(*vectors)))
    if ward_error > 1e-7 or formal_error > 2e-6 or point_error > 1e-8:
        raise ArithmeticError(f"high-grid L5 reproduction/conditioning failed: {ward_error=}, {formal_error=}, {point_error=}")
    rows = []
    max_old_error = 0.
    for lift in trial.LIFTS:
        old = next(r for r in reference["rows"] if r["t"] == config["point"]["t"] and r["level"] == 5 and r["lifts_geometry"] == list(lift))
        original_blocks = {k: trial.decode(z) for k, z in zip(trial.CHANNELS, old["blocks"])}
        k = trial.dv.spin_character_index(lift[::-1])
        for f, eta in product((0, 1), (1, -1)):
            value = small.block(q_values=q, lifts=lift[::-1], form_parity=f, eta_left=eta, eta_right=eta).value
            target = original_blocks[f, eta, eta]
            max_old_error = max(max_old_error, abs(value-target)/max(1., abs(target)))
        for D, K in product(config["descendant_cutoffs"], config["branch_cutoffs"]):
            blocks = dict(original_blocks)
            leakage = 0.
            for eta in (1, -1):
                even = cumulative_vector(shells[eta, D], K)
                for f, enlarged in ((0, even), (1, odd_partner(even))):
                    vector, error = supported_quotient(enlarged, auxiliary)
                    leakage = max(leakage, error)
                    blocks[f, eta, eta] = fwht(vector)[k]
            c = tuple(trial.decode(z) for z in reference["C_BRY"])
            result = trial.contract(blocks, {key: z.conjugate() for key, z in blocks.items()}, c)
            primary2 = abs(trial.decode(old["primary"]))**2
            rows.append({"branch_cutoff": K, "descendant_cutoff": D, "lifts_geometry": list(lift),
                         "blocks": [trial.encode(blocks[c]) for c in trial.CHANNELS],
                         "equal": trial.encode(primary2*result["equal"]),
                         "mixed_frozen_L5": trial.encode(primary2*result["mixed"]),
                         "total_hybrid": trial.encode(primary2*result["total"]),
                         "enlarged_missing_character_relative_leakage": leakage})
    if max_old_error > 1e-10:
        raise ArithmeticError("stored L5 blocks not reproduced")
    validate(config)
    return {"schema": SCHEMA, "config_digest": trial.digest(config), "index": index,
            "source_shard_digest": trial.digest(reference), "momenta_slots": momenta,
            "measure": reference["measure"], "branch_triple_counts": {
                str(K): sum(base_twice_level(n) <= 2*K for n in labels) for K in config["branch_cutoffs"]},
            "shells": [{"eta": eta, "descendant_cutoff": D, "twice_branch_shift": s,
                        "enlarged_components": [trial.encode(z) for z in v]}
                       for (eta, D), values in shells.items() for s, v in sorted(values.items())],
            "checks": {"ward_residual": ward_error, "L5_formal_scaled_error": formal_error,
                       "L5_point_scaled_error": point_error, "archived_L5_block_scaled_error": max_old_error,
                       "branching_mp_dps": mp_dps},
            "rows": rows, "elapsed_seconds": time.monotonic()-start}


def reduce_run(config, output):
    validate(config)
    shards = [trial.load(output/f"shards/node-{i:03d}.json") for i in range(27)]
    for i, shard in enumerate(shards):
        if shard["config_digest"] != trial.digest(config) or shard["index"] != i:
            raise ValueError("incompatible probe shard")
    kappa = 1+2*(config["b"]+1/config["b"])**2
    free_power = config["point"]["Z_free_reference"]**kappa
    rows = []
    for j, row in enumerate(shards[0]["rows"]):
        reduced = {k: row[k] for k in ("branch_cutoff", "descendant_cutoff", "lifts_geometry")}
        for key in ("equal", "mixed_frozen_L5", "total_hybrid"):
            numbers = [s["measure"]*trial.decode(s["rows"][j][key]) for s in shards]
            value = complex(math.fsum(z.real for z in numbers), math.fsum(z.imag for z in numbers))
            if abs(value.imag) > 1e-10*max(abs(value.real), 1e-280):
                raise ArithmeticError("non-real hybrid trial")
            reduced[key] = value.real
        reduced["Q_hybrid"] = reduced["total_hybrid"]/free_power
        reduced["Q_equal"] = reduced["equal"]/free_power
        rows.append(reduced)
    result = {"config": config, "rows": rows, "complete_nodes": len(shards),
              "branch_triple_counts": shards[0]["branch_triple_counts"],
              "maximum_checks": {key: max(s["checks"][key] for s in shards) for key in shards[0]["checks"]},
              "maximum_support_leakage": max(r["enlarged_missing_character_relative_leakage"] for s in shards for r in s["rows"]),
              "physical_Q": None, "physical_Z": None}
    trial.save(output/"summary.json", result)
    for row in rows:
        if row["lifts_geometry"] == [1, 1, 1]:
            print(row, flush=True)
    return result


def run(config, output, workers):
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        def work(index):
            path = output/f"shards/node-{index:03d}.json"
            if path.exists():
                s = trial.load(path)
                if s["config_digest"] != trial.digest(config) or s["index"] != index:
                    raise ValueError("invalid saved node")
                return index
            log = output/f"logs/node-{index:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as handle:
                subprocess.run([sys.executable, __file__, "node", "--output", str(output),
                                "--index", str(index)], stdout=handle, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
            return index
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(27)]
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    index = future.result()
                    status = {"completed_nodes": done, "last_node": index,
                              "elapsed_seconds": time.monotonic()-start, "status": "running"}
                    trial.save(output/"status.json", status)
                    print(status, flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                trial.save(output/"status.json", {"status": "failed", "elapsed_seconds": time.monotonic()-start})
                raise
        reduce_run(config, output)
        trial.save(output/"status.json", {"completed_nodes": 27, "elapsed_seconds": time.monotonic()-start,
                                         "status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "node", "run", "reduce"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path,
                        default=trial.ROOT/"Data Set/nsrr_trial_L5_N3_local_20260830")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mp-dps", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.reference_dir, args.output), flush=True)
        return
    config = trial.load(args.output/"config.json")
    validate(config)
    if args.mode == "node":
        if not 0 <= args.index < 27:
            raise ValueError("node index must be in 0..26")
        result = evaluate_node(config, args.index, args.mp_dps)
        trial.save(args.output/f"shards/node-{args.index:03d}.json", result)
        print({"index": args.index, "checks": result["checks"], "elapsed_seconds": result["elapsed_seconds"]}, flush=True)
    elif args.mode == "run":
        run(config, args.output, args.workers)
    else:
        reduce_run(config, args.output)


if __name__ == "__main__":
    main()
