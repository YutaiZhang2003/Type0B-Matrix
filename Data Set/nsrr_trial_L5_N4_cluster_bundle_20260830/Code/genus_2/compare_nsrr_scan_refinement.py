#!/usr/bin/env python3
"""Compare completed scans without fitting or changing their normalization."""
import argparse
import json
from pathlib import Path


def comparison(refined, reference):
    cfg, old_cfg = refined["config"], reference["config"]
    fine_n, coarse_n = max(cfg["quadrature_orders"]), min(cfg["quadrature_orders"])
    old_n = max(old_cfg["quadrature_orders"])
    source = f"source_nsrr_L{max(cfg['source_physical_levels'])}"
    target = f"target_nsnsns_L{cfg['target_physical_level']}"
    old_source = f"source_nsrr_L{max(old_cfg['source_physical_levels'])}"
    old_target = f"target_nsnsns_L{old_cfg['target_physical_level']}"
    old_rows = {r["t"]: r for r in reference["rows"] if r["quadrature_order"] == old_n}
    old_points = {p["t"]: p for p in old_cfg["points"]}
    diagnostics = {r["t"]: r for r in refined["convergence_diagnostics"]}
    result = []
    for point in cfg["points"]:
        t = point["t"]
        fine = next(r for r in refined["rows"] if r["t"] == t and r["quadrature_order"] == fine_n)
        coarse = next(r for r in refined["rows"] if r["t"] == t and r["quadrature_order"] == coarse_n)
        a, b = (fine["values"][key]["Q"] for key in (source, target))
        ca, cb = (coarse["values"][key]["Q"] for key in (source, target))
        row = {
            "t": t, "source_Q": a, "target_Q": b, "raw_ratio": a / b,
            "relative_mismatch": a / b - 1,
            "relative_ratio_change_within_new_quadrature_axis": (a / b) / (ca / cb) - 1,
            "source_level_relative_change": diagnostics[t]["source_level_relative_change"],
            "target_odd_fraction_after_both_signs": fine["values"][target]["sector_values"][1] / fine["values"][target]["Z"],
        }
        if t in old_rows:
            old = old_rows[t]
            old_ratio = old["values"][old_source]["Q"] / old["values"][old_target]["Q"]
            same_geometry = point["charts"] == old_points[t]["charts"]
            if not same_geometry:
                raise ValueError(f"changed geometry or free denominator at shared point {t}")
            row.update(toy_ratio=old_ratio, relative_ratio_change_from_toy=(a / b) / old_ratio - 1,
                       shared_geometry_and_free_denominators_identical=same_geometry)
        result.append(row)
    return {
        "same_numerical_kernel_fingerprint": refined["implementation_fingerprint"] == reference["implementation_fingerprint"],
        "same_convention_ledger": cfg["convention_ledger"] == old_cfg["convention_ledger"],
        "common_quadrature_reference_changed": cfg["quadrature_reference_abs_q"] != old_cfg["quadrature_reference_abs_q"],
        "comparison_scope": "Toy-to-refined changes combine cutoffs, precision controls and quadrature envelope; the new internal N and source-level axes isolate those two refinements.",
        "toy_quadrature_order": old_n, "refined_quadrature_order": fine_n,
        "toy_source_physical_level": max(old_cfg["source_physical_levels"]),
        "refined_source_physical_level": max(cfg["source_physical_levels"]),
        "toy_target_physical_level": old_cfg["target_physical_level"],
        "refined_target_physical_level": cfg["target_physical_level"],
        "fitted_normalization_used": False, "rows": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = comparison(json.loads(args.refined.read_text()), json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["rows"]:
        print(row)


if __name__ == "__main__":
    main()
