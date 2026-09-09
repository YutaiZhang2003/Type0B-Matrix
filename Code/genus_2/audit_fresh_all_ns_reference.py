#!/usr/bin/env python3
"""Independently reduce fresh all-NS nodes and compare target-only history.

Historical NSRR values are never consulted. Agreement with an earlier run
tests reproducibility, not the correctness of the unresolved spin adapter.
"""
import argparse
import math
from pathlib import Path

import recompute_all_ns_reference as fresh


def audit(run_dir, previous_dir):
    scan = fresh.scan
    result = scan._load(run_dir/"summary.json")
    config = result["config"]
    fresh.validate_config(config)
    previous = scan._load(previous_dir/"summary.json")
    baseline = previous["config"]["baseline_config"]
    n, order = config["quadrature_order"], config["recursion_order_twice_level"]
    indices = [i for i, (channel, qorder, _) in enumerate(scan.tasks(baseline))
               if channel == "target_nsnsns" and qorder == n]
    if len(indices) != n**3:
        raise ValueError("historical all-NS grid not found")
    shards = [scan._load(run_dir/"shards"/f"node-{i:03d}.json") for i in range(n**3)]
    node_errors = []
    for i, (old_index, shard) in enumerate(zip(indices, shards)):
        fresh.validate_shard(config, i, shard)
        old = scan._load(previous_dir/"shards"/f"task-{old_index:06d}.json")
        if old["implementation_fingerprint"] != previous["implementation_fingerprint"]:
            raise ValueError("historical target shard provenance mismatch")
        if old["momenta"] != shard["momenta"] or old["measure"] != shard["measure"]:
            raise ValueError("momentum grids differ")
        for row in shard["values"]:
            old_row = next(r for r in old["values"] if r["t"] == row["t"] and r["recursion_order"] == order)
            node_errors.extend(abs(x/y-1) if y else abs(x)
                               for x, y in zip(row["sector_contributions"], old_row["sector_contributions"]))
    b = config["parameters"]["b"]
    kappa = (1.5+3*(b+1/b)**2)/1.5
    rows = []
    for j, point in enumerate(config["points"]):
        output = next(r for r in result["rows"] if r["t"] == point["t"])
        old = next(r for r in previous["rows"] if r["t"] == point["t"]
                   and r["quadrature_order"] == n and r["recursion_order"] == order)
        if output["source_Z"] is not None or output["source_Q"] is not None or output["source_over_target"] is not None:
            raise ValueError("fresh output must not contain uncertified NSRR values")
        sectors = [math.fsum(s["values"][j]["sector_contributions"][f] for s in shards) for f in (0, 1)]
        z = math.fsum(sectors)
        q = z/point["Z_free"]**kappa
        if not math.isclose(z, output["target_Z"], rel_tol=1e-14) or not math.isclose(q, output["target_Q"], rel_tol=1e-13):
            raise ValueError("independent reduction failed")
        free48 = float(scan.physical_superfield_plumbing_partition(
            "theta", tuple(complex(x) for x in point["q_values"]), point["lifts"], max_mode=48).one_superfield_value)
        rows.append({"t": point["t"], "fresh_Z": z, "fresh_Q": q,
                     "fresh_vs_previous_R16_Z_relative_change": z/old["target_Z"]-1,
                     "fresh_vs_previous_R16_Q_relative_change": q/old["target_Q"]-1,
                     "free_mode_44_to_48_relative_change": free48/point["Z_free"]-1})
    source = scan._load(run_dir/"source_geometry_audit.json")
    report = {
        "schema": "fresh-all-ns-reference-audit-v1",
        "independent_reduction_passed": True,
        "protected_kernel_hashes": fresh.protected_hashes(),
        "summary_digest": scan._digest(result),
        "previous_target_summary_digest": scan._digest(previous),
        "source_geometry_audit_digest": scan._digest(source),
        "historical_source_values_used": False,
        "nodes_checked": len(shards), "sector_evaluations_compared": len(node_errors),
        "maximum_relative_node_reproduction_error": max(node_errors),
        "rows": rows,
        "nsrr_partition_status": config["nsrr_status"],
        "nsrr_replumbing_maximum_forward_error": max(r["high_order_forward_period_residual"] for r in source["points"]),
        "nsrr_free_spin_conversion_compatible": all(r["theta_ratio_free_conversion_audit"]["compatible"] for r in source["points"]),
        "interpretation": "Reproducibility and finite-cutoff checks only; no NSRR comparison or certified quadrature error.",
    }
    if max(node_errors) > 1e-9 or max(abs(r["free_mode_44_to_48_relative_change"]) for r in rows) > 1e-8:
        raise ArithmeticError("target reference reproducibility audit failed")
    scan.write_json(run_dir/"verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir, args.previous_dir)
    print("Independent reduction and target-only reproduction passed")
    print(f"Maximum node discrepancy: {report['maximum_relative_node_reproduction_error']:.3e}")
    for row in report["rows"]:
        print(row)
