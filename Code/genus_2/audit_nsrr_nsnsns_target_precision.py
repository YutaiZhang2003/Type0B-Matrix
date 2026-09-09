#!/usr/bin/env python3
"""Spot-check global-sum/MP precision separately from c-recursion cutoff.

This is a bounded node audit, not a replacement for full-grid convergence.
Three-point constants, local frames, momenta and measures remain unchanged.
"""
import argparse
from pathlib import Path
import time

import nsrr_nsnsns_theta_omega_scan as scan


def audit(baseline_dir, task_indices, output):
    base = scan._load(baseline_dir/"config.json")
    summary = scan._load(baseline_dir/"summary.json")
    scan.validate_config(base)
    if base != summary["config"] or summary["implementation_fingerprint"] != scan.fingerprint():
        raise ValueError("baseline config or numerical implementation changed")
    p, n = base["parameters"], base["numerics"]
    settings = {
        "baseline": {"global_tolerance": n["global_tolerance"],
                     "global_max_total_occupation": n["global_max_total_occupation"],
                     "block_working_precision": n["block_working_precision"]},
        "tight": {"global_tolerance": 1e-11, "global_max_total_occupation": 60,
                  "block_working_precision": 60},
    }
    constants = scan.GenericSuperLiouvilleConstants(p["b"], dps=n["structure_precision"],
        mu=complex(p["mu"]), include_cosmological_prefactor=p["include_cosmological_prefactor"])
    rows = []
    for index in task_indices:
        ch, order, node, indices, momenta, measure = scan.node_data(base, index)
        if ch != "target_nsnsns":
            raise ValueError("precision audit requires target nodes")
        for point in base["points"]:
            if point["t"] not in (.52, .60, .68):
                continue
            chart = point["charts"][ch]
            q = tuple(complex(x) for x in chart["q_values"])
            values = {}
            for name, control in settings.items():
                recursion = scan.NSGenus2CRecursion(channel="theta", q_values=q,
                    global_method="resummed", global_tolerance=control["global_tolerance"],
                    global_max_total_occupation=control["global_max_total_occupation"],
                    vacuum_word_length=n["vacuum_word_length"], vacuum_max_mode=n["vacuum_max_mode"])
                for r in (8, 16):
                    tick = time.perf_counter()
                    sectors = scan.all_ns_node(b=p["b"], q_values=q, lifts=chart["lifts"],
                        recursion_order=r, momenta=momenta, measure=measure, constants=constants,
                        recursion=recursion, block_method="collision_aware_mp",
                        block_working_precision=control["block_working_precision"])
                    values[f"{name}_R{r}"] = {"sector_contributions": list(sectors),
                        "sum": sum(sectors), "runtime_seconds": time.perf_counter()-tick,
                        "global_max_used": recursion.global_max_used,
                        "global_nonconverged_calls": recursion.global_nonconverged_calls,
                        "global_worst_last_shell_relative": recursion.global_worst_last_shell_relative}
                    print(f"precision audit node={index} t={point['t']:.2f} {name} R={r} complete", flush=True)
            rows.append({"task_index": index, "t": point["t"], "quadrature_order": order,
                         "node_index": node, "indices": list(indices), "momenta": list(momenta),
                         "values": values,
                         "tight_R16_over_baseline_R16_minus_one": values["tight_R16"]["sum"]/values["baseline_R16"]["sum"]-1,
                         "baseline_R16_over_R8_minus_one": values["baseline_R16"]["sum"]/values["baseline_R8"]["sum"]-1,
                         "tight_R16_over_R8_minus_one": values["tight_R16"]["sum"]/values["tight_R8"]["sum"]-1})
    result = {"schema": "nsrr-nsnsns-target-precision-audit-v1", "baseline_dir": str(baseline_dir.resolve()),
              "baseline_config_digest": scan._digest(base), "numerical_kernel_fingerprint": scan.fingerprint(),
              "settings": settings, "rows": rows,
              "interpretation": "Selected-node stability audit, not an integrated or certified error bound. Both global tolerance and working precision are tightened together."}
    scan.write_json(output, result)
    for row in rows:
        print({key: value for key, value in row.items() if key != "values"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--task-indices", nargs="+", type=int, default=[283, 284])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.baseline_dir, args.task_indices, args.output)


if __name__ == "__main__":
    main()
