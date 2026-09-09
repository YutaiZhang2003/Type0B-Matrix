#!/usr/bin/env python3
"""Same all-NS integrand at four literal lifts on dominant N5 nodes.

This is a sensitivity check, not a replacement fixed-spin partition. No
free-theory basis conversion is applied to these Liouville blocks.
"""
import math
from pathlib import Path
import time

import check_nsrr_spin_quadrature as check
from audit_nsrr_comparison_spin_basis import LIFTS, raw_spin, TARGET_BRANCH


def run():
    c = check.trial.load(check.DEFAULT_OUTPUT/"config.json")
    check.validate(c)
    p, num, point = c["target"]["parameters"], c["target"]["numerics"], c["target_point"]
    q = tuple(map(complex, point["q_values"]))
    results = []
    started = time.monotonic()
    for index in (31, 30, 26):
        momenta, measure = check.node_data(c, "target", 5, index)
        constants = check.fresh.scan.GenericSuperLiouvilleConstants(
            p["b"], dps=num["structure_precision"], mu=complex(p["mu"]),
            include_cosmological_prefactor=p["include_cosmological_prefactor"])
        recursion = check.fresh.scan.NSGenus2CRecursion(
            channel="theta", q_values=q, global_method="resummed",
            global_tolerance=num["global_tolerance"], global_max_total_occupation=num["global_max_total_occupation"],
            vacuum_word_length=num["vacuum_word_length"], vacuum_max_mode=num["vacuum_max_mode"])
        rows = []
        for lifts in LIFTS:
            sectors = check.fresh.scan.all_ns_node(
                b=p["b"], q_values=q, lifts=lifts, recursion_order=16, momenta=momenta, measure=measure,
                constants=constants, recursion=recursion, block_method="collision_aware_mp",
                block_working_precision=num["block_working_precision"])
            rows.append({"lifts": list(lifts), "Z_node": math.fsum(sectors), "sectors": list(sectors),
                         "corresponding_unfiltered_free_spin_only": raw_spin(lifts, TARGET_BRANCH)})
        reference = next(r for r in rows if r["lifts"] == point["lifts"])
        saved = check.trial.load(Path(c["references"]["target"])/"shards"/f"node-{index:03d}.json")
        expected = math.fsum(next(r for r in saved["values"] if r["t"] == c["t"])["sector_contributions"])
        error = abs(reference["Z_node"]/expected-1)
        if error > 1e-12 or recursion.global_nonconverged_calls:
            raise ArithmeticError("reference or global-convergence check failed")
        for row in rows:
            row["relative_change_from_selected_lifts"] = row["Z_node"]/reference["Z_node"]-1
        results.append({"index": index, "momenta": momenta, "reference_reproduction_error": error, "rows": rows})
        print(index, [(r["lifts"], r["relative_change_from_selected_lifts"]) for r in rows], flush=True)
    report = {"config_digest": check.trial.digest(c), "t": c["t"], "N": 5, "R": 16,
              "nodes": results, "seconds": time.monotonic()-started,
              "interpretation": "Pointwise sensitivity of the unchanged HN Liouville numerator. Labels on unfiltered FREE controls are not assignments of Liouville spin; no alternative integral or physical correction is claimed."}
    check.trial.save(check.DEFAULT_OUTPUT/"target_lift_sensitivity.json", report)
    return report


if __name__ == "__main__":
    run()
