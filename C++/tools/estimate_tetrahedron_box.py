#!/usr/bin/env python3
"""Count the Section 6 six-edge workload without evaluating conformal blocks.

Counts follow the C++ contraction and recursion loops before the Section 6
consolidation. In particular, the SCA reference is the historical target-by-target
recursion, not the newly batched reference. Runtime scenarios use saved 40-digit
measurements; they are neither bounds nor confidence intervals.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def character(n, fermions, bosons):
    values = [1] + [0] * n
    for k in bosons:
        for j in range(k, n + 1):
            values[j] += values[j - k]
    for k in fermions:
        for j in range(n, k - 1, -1):
            values[j] += values[j - k]
    return values


def convolve(a, b):
    return [sum(a[i] * b[k - i] for i in range(k + 1))
            for k in range(len(a))]


def cumulative_product(*arrays):
    result = [1] + [0] * (len(arrays[0]) - 1)
    for a in arrays:
        result = convolve(result, a)
    return sum(result)


def ramond_pbw(level, box):
    ns = character(2 * level, range(1, 2 * level + 1, 2),
                   range(2, 2 * level + 1, 2))
    rr = character(level, range(1, level + 1), range(1, level + 1))
    degrees = products = 0
    for k in itertools.product(range(2 * level + 1), repeat=3):
        for r in itertools.product(range(level + 1), repeat=3):
            if not box and sum(k) + 2 * sum(r) > 2 * level:
                continue
            n = [ns[i] for i in k]
            a, b, c = [rr[i] for i in r]
            degrees += 1
            # 32 nonzero form/sign cases times two allowed rim parities.
            # Three spoke Gram transforms and the actual five ring products.
            products += 64 * math.prod(n) * (
                sum(n) + c*a*a + c*a*b + c*b*b + b*c*c + c*c*c)
    return {"multidegrees": degrees, "dense_contraction_products": products}


def virasoro_counts(level):
    limit = 2 * level
    reachable = [int(i == 0 or i >= 2) for i in range(level + 1)]
    poles = [r*s for r in range(2, level + 1)
             for s in range(1, level // r + 1)]
    arrivals = [sum(reachable[u-t] for t in poles if t <= u)
                for u in range(level + 1)]
    polynomials = {name: [0] * (limit + 1)
                   for name in ("C", "B", "R", "A", "OR", "OA", "G", "P")}
    box = dict.fromkeys(polynomials, 0)
    for label in range(-math.isqrt(limit), math.isqrt(limit) + 1):
        base = label * label
        remaining = (limit - base) // 2
        polynomials["C"][base] += 1
        box["C"] += 1
        for u in range(remaining + 1):
            degree = base + 2*u
            for key, value in (("R", reachable[u]), ("A", arrivals[u]),
                               ("G", sum(reachable[:u+1])), ("P", u+1)):
                polynomials[key][degree] += value
                box[key] += value
            for t in poles:
                if u+t <= remaining:
                    for key, value in (("OR", reachable[u]), ("OA", arrivals[u])):
                        polynomials[key][base + 2*(u+t)] += value
                        box[key] += value
        for t in poles:
            if t <= remaining:
                polynomials["B"][base + 2*t] += 1
                box["B"] += 1

    def count(keys, independent):
        if independent:
            return math.prod(box[k] for k in keys)
        return cumulative_product(*(polynomials[k] for k in keys))

    result = {}
    for independent in (False, True):
        result["per_edge" if independent else "total"] = {
            "branch_tuples": count(["C"]*6, independent),
            "ccy_transitions": 2 * (
                6 * count(["B"] + ["C"]*5, independent)
                + 6 * count(["OA"] + ["R"]*5, independent)
                + 30 * count(["A", "OR"] + ["R"]*4, independent)),
            "global_seed_terms": 2 * count(["G"]*6, independent),
            "two_copy_products": count(["P"]*6, independent),
        }
    return result


def ns_recursion_counts(level):
    n = 2 * level
    poles = [r*s for r in range(2, n+1) for s in range(1, n//r+1)
             if (r+s) % 2 == 0]
    reachable = [1] + [0]*n
    for k in range(1, n+1):
        reachable[k] = int(any(k >= t and reachable[k-t] for t in poles))
    arrivals = [sum(reachable[k-t] for t in poles if t <= k) for k in range(n+1)]
    outgoing = [sum(t <= k for t in poles) for k in range(n+1)]
    ones = [1]*(n+1)
    rp, ap = convolve(reachable, ones), convolve(arrivals, ones)
    at, rt = convolve(arrivals, outgoing), convolve(reachable, outgoing)
    result = {}
    for box in (False, True):
        def count(*arrays):
            return (math.prod(sum(a) for a in arrays) if box
                    else cumulative_product(*arrays))
        result["per_edge" if box else "total"] = {
            "multidegrees": count(*([ones]*6)),
            "recursion_states": count(*([ones]*6)) + 6*count(ap, *([rp]*5)),
            "recursion_transitions": (6*count(outgoing, *([ones]*5))
                                      + 6*count(at, *([rp]*5))
                                      + 30*count(ap, rt, *([rp]*4))),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    saved = ROOT / "C++/experiments/mercedes_all_ns_level10_2026-09-16"
    dv_timing = json.loads((saved / "timing_dv_L10.json").read_text())
    ns_timing = json.loads((saved / "timing_ns_L10.json").read_text())
    pbw_timing = json.loads((ROOT / "C++/experiments/total_level6_2026-09-16/mercedes_L6_results.json").read_text())
    vir, ns = virasoro_counts(10), ns_recursion_counts(10)
    assert vir["total"]["ccy_transitions"] == dv_timing["ccy_transitions"]
    assert vir["total"]["branch_tuples"] == dv_timing["branches"]
    assert ns["total"]["multidegrees"] == ns_timing["multidegrees"]
    pbw_small, pbw_box = ramond_pbw(6, False), ramond_pbw(5, True)
    seconds_per_year = 365.25 * 86400
    report = {
        "dps": 40, "new_block_evaluations": 0,
        "ramond_pbw_total6": pbw_small, "ramond_pbw_per_edge5": pbw_box,
        "all_ns_double_virasoro_level10": vir,
        "all_ns_recursion_level10": ns,
        "runtime_scenarios": {
            "pbw_days_scaling_same_graph_stage": sum(c["physical_pbw_seconds"] for c in pbw_timing["cases"])
                * pbw_box["dense_contraction_products"] / pbw_small["dense_contraction_products"] / 86400,
            "dv_years_by_work_component": {
                k: dv_timing["dv_seconds"] * vir["per_edge"][k] / vir["total"][k] / seconds_per_year
                for k in ("ccy_transitions", "global_seed_terms", "two_copy_products")},
            "ns_years_by_work_component": {
                k: ns_timing["recursion_and_output_seconds"] * ns["per_edge"][k] / ns["total"][k] / seconds_per_year
                for k in ("recursion_states", "recursion_transitions")},
        },
        "assumptions": [
            "Six-edge algorithms before the Section 6 consolidation, sequential 40-digit MPC arithmetic.",
            "The SCA-reference scenario uses historical target-by-target recursion, not the new batched reference.",
            "Generic nonzero residues; counts are structural work, not a new accuracy test.",
            "PBW count includes all 64 form/sign cases; zero matrix entries can reduce executed products.",
            "Runtime scenarios scale individual work components against a measured aggregate stage; they are not measured runtimes or bounds.",
            "NS seed growth, memory pressure, paging, and new algorithmic optimizations are not modeled.",
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(text)


if __name__ == "__main__":
    main()
