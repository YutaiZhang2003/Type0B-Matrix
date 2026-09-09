#!/usr/bin/env python3
"""Reconstruct the branching-sweep contraction and measure shell increments."""
import argparse
import csv
import hashlib
from html import escape
from itertools import product
import math
from pathlib import Path

import nsrr_branching_cutoff_probe as probe

trial = probe.trial


def recovered_blocks(shard, config, D, K, lifts):
    auxiliary = [trial.decode(z) for z in config["auxiliary_values"]["16"]]
    result = {}
    character = trial.dv.spin_character_index(lifts[::-1])
    for eta in (1, -1):
        terms = [r for r in shard["shells"] if r["eta"] == eta and r["descendant_cutoff"] == D
                 and r["twice_branch_shift"] <= 2*K]
        vector = [sum(trial.decode(r["enlarged_components"][j]) for r in terms) for j in range(8)]
        for f, value in ((0, vector), (1, probe.odd_partner(vector))):
            physical, _ = probe.supported_quotient(value, auxiliary)
            result[f, eta, eta] = probe.fwht(physical)[character]
    return result


def audit(root):
    summary = trial.load(root/"summary.json")
    config = summary["config"]
    probe.validate(config)
    if config != trial.load(root/"config.json"):
        raise ValueError("summary/config mismatch")
    source = trial.load(Path(config["reference_dir"])/"summary.json")
    if trial.digest(source) != config["reference_summary_digest"]:
        raise ValueError("source provenance changed")
    t = config["point"]["t"]
    source_row = next(r for r in source["rows"] if r["t"] == t and r["level"] == 5
                      and r["lifts_geometry"] == [1, 1, 1])
    baseline = source_row["Q_trial_reference"]
    kappa = 1+2*(config["b"]+1/config["b"])**2
    free_power = config["point"]["Z_free_reference"]**kappa
    errors = {"block": 0., "contraction": 0., "frozen_mixed": 0., "reduction": 0.}
    totals = {}
    shell_totals = {s: [] for s in range(21)}
    shell_bounds = {s: [] for s in range(21)}
    for i in range(27):
        shard = trial.load(root/f"shards/node-{i:03d}.json")
        ref = trial.load(Path(config["reference_dir"])/f"shards/node-{i:03d}.json")
        if shard["config_digest"] != trial.digest(config) or shard["index"] != i or shard["source_shard_digest"] != trial.digest(ref):
            raise ValueError("bad node provenance")
        if shard["measure"] != ref["measure"] or shard["momenta_slots"] != ref["momenta_slots"]:
            raise ValueError("momentum grid changed")
        design = [(D, K, list(lifts)) for lifts, D, K in
                  product(trial.LIFTS, config["descendant_cutoffs"], config["branch_cutoffs"])]
        if [(r["descendant_cutoff"], r["branch_cutoff"], r["lifts_geometry"]) for r in shard["rows"]] != design:
            raise ValueError("missing, duplicated, or reordered cutoff/lift combination")
        c = {eta: trial.decode(z)/2 for eta, z in zip((1, -1), ref["C_BRY"])}
        for row in shard["rows"]:
            D, K, lifts = row["descendant_cutoff"], row["branch_cutoff"], row["lifts_geometry"]
            old = next(r for r in ref["rows"] if r["t"] == t and r["level"] == 5 and r["lifts_geometry"] == lifts)
            blocks = recovered_blocks(shard, config, D, K, lifts)
            for key, value in zip(trial.CHANNELS, old["blocks"]):
                if key[1] != key[2]:
                    blocks[key] = trial.decode(value)
            actual_blocks = {k: trial.decode(z) for k, z in zip(trial.CHANNELS, row["blocks"])}
            errors["block"] = max(errors["block"], *(abs(blocks[k]-actual_blocks[k])/max(1., abs(blocks[k])) for k in blocks))
            primary2 = abs(trial.decode(old["primary"]))**2
            # Independently simplify (-1)^f (i^f)^2 = 1 only in this audit.
            equal = primary2*sum(c[a]*c[b]*abs(z)**2 for (f, a, b), z in blocks.items() if a == b)
            mixed = primary2*sum(c[a]*c[b]*abs(z)**2 for (f, a, b), z in blocks.items() if a != b)
            z = equal+mixed
            for name, value in (("equal", equal), ("mixed_frozen_L5", mixed), ("total_hybrid", z)):
                errors["contraction"] = max(errors["contraction"], abs(value-trial.decode(row[name]))/max(abs(z), 1e-280))
            errors["frozen_mixed"] = max(errors["frozen_mixed"], abs(mixed-trial.decode(old["mixed"]))/max(abs(z), 1e-280))
            totals.setdefault((D, K, tuple(lifts)), []).append(shard["measure"]*z)
        old = next(r for r in ref["rows"] if r["t"] == t and r["level"] == 5 and r["lifts_geometry"] == [1, 1, 1])
        primary2 = abs(trial.decode(old["primary"]))**2
        last_blocks = {(f, eta, eta): 0j for f, eta in product((0, 1), (1, -1))}
        last_equal = 0j
        for twice_s in range(21):
            blocks = recovered_blocks(shard, config, 5, twice_s/2, [1, 1, 1])
            equal = primary2*sum(c[a]*c[b]*abs(z)**2 for (f, a, b), z in blocks.items())
            shell_totals[twice_s].append(shard["measure"]*(equal-last_equal)/free_power)
            bound = primary2*sum(abs(c[a]*c[b])*(2*abs(last_blocks[key])*abs(z-last_blocks[key])+abs(z-last_blocks[key])**2)
                                 for key, z in blocks.items() for f, a, b in (key,))
            shell_bounds[twice_s].append(shard["measure"]*bound/free_power)
            last_equal, last_blocks = equal, blocks
    for row in summary["rows"]:
        numbers = totals[row["descendant_cutoff"], row["branch_cutoff"], tuple(row["lifts_geometry"])]
        value = complex(math.fsum(z.real for z in numbers), math.fsum(z.imag for z in numbers))/free_power
        errors["reduction"] = max(errors["reduction"], abs(value-row["Q_hybrid"])/max(abs(value), 1e-280))
    if max(errors.values()) > 2e-11:
        raise ArithmeticError(f"independent audit failed: {errors}")
    rows = []
    for D in config["descendant_cutoffs"]:
        selected = [r for r in summary["rows"] if r["lifts_geometry"] == [1, 1, 1] and r["descendant_cutoff"] == D]
        base = next(r for r in selected if r["branch_cutoff"] == 5)
        previous = None
        for r in selected:
            row = {**r, "relative_to_K5": r["Q_hybrid"]/base["Q_hybrid"]-1,
                   "relative_to_old_total_L5": r["Q_hybrid"]/baseline-1,
                   "relative_to_preceding_K": r["Q_hybrid"]/previous["Q_hybrid"]-1 if previous else None}
            rows.append(row)
            previous = r
    shell_rows = [{"branch_shift": s/2, "delta_Q_hybrid_D5": math.fsum(z.real for z in shell_totals[s]),
                   "absolute_increment_bound_Q": math.fsum(shell_bounds[s])}
                  for s in range(21)]
    fixed_mixed_Q = next(r for r in rows if r["descendant_cutoff"] == 5 and r["branch_cutoff"] == 5)["mixed_frozen_L5"]/free_power
    controls = [{"branch_cutoff": s/2,
                 "Q_hybrid_D5": fixed_mixed_Q+math.fsum(r["delta_Q_hybrid_D5"] for r in shell_rows[:s+1])}
                for s in range(21)]
    q5 = controls[10]["Q_hybrid_D5"]
    smoking_gun = {
        "fixed_descendant_cutoff": 5,
        "Q_K0_ground_branches_only": controls[0]["Q_hybrid_D5"],
        "Q_K5": q5, "Q_K10": controls[20]["Q_hybrid_D5"],
        "K0_to_K5_relative": q5/controls[0]["Q_hybrid_D5"]-1,
        "K5_to_K10_relative": controls[20]["Q_hybrid_D5"]/q5-1,
        "added_shell_absolute_change_bound_relative": math.fsum(r["absolute_increment_bound_Q"] for r in shell_rows[11:])/q5,
        "bound_scope": "triangle-inequality bound on quadratic integrand change, summed over half-level shells, equal-sign channels, and momentum nodes; coherent sums within each shell are preserved",
    }
    uncached_check = None
    old_probe = root.with_name(root.name+"_initial_metadata_bug")/"shards/node-000.json"
    if old_probe.exists():
        old = trial.load(old_probe)
        new = trial.load(root/"shards/node-000.json")
        if old["source_shard_digest"] != new["source_shard_digest"] or old["momenta_slots"] != new["momenta_slots"]:
            raise ValueError("uncached timing probe is at a different momentum")
        block_error = total_error = 0.
        for a, b in zip(old["rows"], new["rows"]):
            for x, y in zip(a["blocks"], b["blocks"]):
                x, y = trial.decode(x), trial.decode(y)
                block_error = max(block_error, abs(x-y)/max(1., abs(x)))
            x, y = trial.decode(a["total_hybrid"]), trial.decode(b["total_hybrid"])
            total_error = max(total_error, abs(x-y)/max(abs(x), 1e-280))
        if max(block_error, total_error) > 1e-11:
            raise ArithmeticError("cached K10 point differs from uncached timing probe")
        uncached_check = {"maximum_block_scaled_error": block_error,
                          "maximum_integrand_relative_error": total_error,
                          "initial_shard_digest": trial.digest(old),
                          "initial_metadata_bug": "old node index overwritten by parity temporary; old data are not used in integration"}
    result = {"summary_digest": trial.digest(summary), "protected_kernel_sha256": trial.protected_hashes(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "errors": errors, "baseline_total_L5_Q": baseline,
              "rows": rows, "D5_shell_increments": shell_rows,
              "D5_positive_control": controls, "smoking_gun": smoking_gun,
              "cached_vs_uncached_highK_probe": uncached_check,
              "physical_Q": None, "physical_Z": None}
    cache_errors = []
    for path in Path(config["action_cache_dir"]).glob("*.json"):
        entry = trial.load(path)
        if path.stem != trial.digest(entry["identity"]) or entry["payload_digest"] != trial.digest(entry["payload"]):
            raise ValueError("action cache key/payload hash mismatch")
        if entry["identity"]["kernel_sha256"] != result["protected_kernel_sha256"]["Code/ramond_branching_recursion/compute_target.py"]:
            raise ValueError("action cache was produced by a different kernel")
        cache_errors.append(entry["payload"]["relative_residual"])
    result["action_cache_audit"] = {"entries": len(cache_errors), "maximum_fit_relative_residual": max(cache_errors)}
    trial.save(root/"verification.json", result)
    with (root/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root/"branch_shells_D5.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shell_rows[0]))
        writer.writeheader()
        writer.writerows(shell_rows)
    print({"errors": errors, "baseline_total_L5_Q": baseline}, flush=True)
    for row in rows:
        print(row, flush=True)
    plot(root, result)
    return result


def plot(root, audit):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">',
             '<rect width="1120" height="650" fill="#fafbfd"/><g font-family="Arial,sans-serif" fill="#253248">']
    def text(x, y, value, size=14, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(value))}</text>')
    text(50, 42, "NSRR branching-cutoff test at t = Re Omega12 = 0.60", 25)
    text(50, 74, "b = 1.4, N = 3; K = branching shift cutoff; D = combined descendant order of two Virasoro blocks", 14)
    text(50, 101, "Equal-sign double-Virasoro sectors varied; mixed-sign PBW blocks held at L5. Hybrid diagnostic only.", 14)
    for panel in (0, 1):
        left, top, width, height = 100+530*panel, 174, 415, 340
        rows = [r for r in audit["rows"] if r["branch_cutoff"] >= 5]
        value = lambda r: r["Q_hybrid"]*1e7 if panel == 0 else r["relative_to_K5"]*100
        values = [value(r) for r in rows]
        lo, hi = min(values), max(values)
        pad = max((hi-lo)*.12, 1e-9)
        lo, hi = lo-pad, hi+pad
        xy = lambda k, v: (left+(k-5)*width/5, top+(hi-v)*height/(hi-lo))
        text(left, top-30, "Hybrid Q (units of 10^-7)" if panel == 0 else "Change from K=5 at fixed D (%)", 18)
        for j in range(6):
            v = lo+(hi-lo)*j/5
            y = xy(5, v)[1]
            parts.append(f'<path d="M{left},{y}h{width}" stroke="#dce3ec"/>')
            text(left-10, y+4, f"{v:.7f}" if panel == 0 else f"{v:.2g}", 12, "end")
        for K in (5, 6, 8, 10):
            text(xy(K, lo)[0], 538, K, 13, "middle")
        for D, color in ((4, "#aa7a2c"), (5, "#087c9d"), (6, "#7847a1")):
            selected = [r for r in rows if r["descendant_cutoff"] == D]
            coords = " ".join(f"{xy(r['branch_cutoff'],value(r))[0]},{xy(r['branch_cutoff'],value(r))[1]}" for r in selected)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for r in selected:
                x, y = xy(r["branch_cutoff"], value(r))
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
        text(left+width/2, 567, "Branching cutoff K", 14, "middle")
    for j, (D, color) in enumerate(((4, "#aa7a2c"), (5, "#087c9d"), (6, "#7847a1"))):
        x = 335+180*j
        parts.append(f'<path d="M{x},599h25" stroke="{color}" stroke-width="3"/>')
        text(x+32, 604, f"D = {D}", 14)
    text(50, 635, "No final total-L5 truncation; no fitted normalization. This does not test the missing physical spin / nonchiral dictionary.", 13)
    parts.append('</g></svg>')
    (root/"comparison.svg").write_text("\n".join(parts)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    audit(parser.parse_args().run_dir)
