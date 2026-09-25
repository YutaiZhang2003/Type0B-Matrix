"""Compare like-for-like mixed-channel estimates, keeping error sources separate.

Optional multilevel correction: I_high ~= I_base(fine P rule) +
[I_high-I_base](coarse P rule). Both terms must have a momentum refinement.
Reported differences are empirical convergence indicators, not rigorous bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_so7e8_momentum_refinement import pairs, unpairs
from so7e8_sphere_branches import require_same_coordinate_branch


KINEMATICS = ("momenta", "points", "picture", "species", "normalization")


def require_compatible(a, b, *, same_channel=False):
    require_same_coordinate_branch(a["parameters"], b["parameters"])
    for key in KINEMATICS:
        if a["parameters"][key] != b["parameters"][key]:
            raise ValueError(f"incompatible {key}")
    if same_channel and a["channel"] != b["channel"]:
        raise ValueError("multilevel correction must use the same channel")


def value(row, order, name):
    return unpairs(row["blocks"][str(order)][name]["value"])


def estimate(report, correction=None):
    """Return central values and separate absolute momentum/block indicators."""
    if report.get("status") != "complete" or len(report["rows"]) < 2:
        raise ValueError("a completed momentum-refinement sequence is required")
    last, previous = report["rows"][-1], report["rows"][-2]
    orders = sorted(map(int, last["blocks"]))
    base = orders[-1]
    result = {}
    if correction is not None:
        require_compatible(report, correction, same_channel=True)
        if correction.get("status") != "complete" or len(correction["rows"]) < 2:
            raise ValueError("the block correction needs its own momentum refinement")
        fine_c, coarse_c = correction["rows"][-1], correction["rows"][-2]
        correction_orders = sorted(map(int, fine_c["blocks"]))
        high = correction_orders[-1]
        if base not in correction_orders or high <= base:
            raise ValueError("correction must contain the base and a higher block order")
    for name in ("F0", "F2"):
        total = value(last, base, name)
        p_error = abs(total - value(previous, base, name))
        if correction is None:
            if len(orders) < 2:
                raise ValueError("a block-order refinement is also required")
            h_error = abs(total - value(last, orders[-2], name))
            correction_value = np.zeros_like(total)
            correction_p_error = np.zeros_like(p_error)
        else:
            correction_value = value(fine_c, high, name) - value(fine_c, base, name)
            coarse_delta = value(coarse_c, high, name) - value(coarse_c, base, name)
            correction_p_error = abs(correction_value - coarse_delta)
            h_error = abs(value(fine_c, high, name) - value(fine_c, correction_orders[-2], name))
            total = total + correction_value
        scale = np.maximum(abs(total), 1e-300)
        result[name] = dict(value=pairs(total), multilevel_correction=pairs(correction_value),
            momentum_base_absolute=p_error.tolist(),
            momentum_correction_absolute=correction_p_error.tolist(),
            momentum_relative=((p_error+correction_p_error)/scale).tolist(),
            block_absolute=h_error.tolist(), block_relative=(h_error/scale).tolist(),
            summed_indicator_relative=((p_error+correction_p_error+h_error)/scale).tolist())
    return result


def compare(ns, r, *, ns_correction=None, r_correction=None):
    require_compatible(ns, r)
    if ns["channel"] != "crossed" or r["channel"] != "direct":
        raise ValueError("expected NS crossed and R direct reports")
    nsv = estimate(ns, ns_correction)
    rv = estimate(r, r_correction)
    overlap = {}
    for name in nsv:
        a, b = unpairs(nsv[name]["value"]), unpairs(rv[name]["value"])
        scale = np.maximum(np.maximum(abs(a), abs(b)), 1e-300)
        overlap[name] = dict(ratio_NS_over_R=pairs(a/b),
            absolute_difference=abs(a-b).tolist(), relative_difference=(abs(a-b)/scale).tolist())
    return dict(status="refined_diagnostic", physical_sewing_certified=False,
        definition="delta=abs(F_NS-F_R)/max(abs(F_NS),abs(F_R)); target zero; complex ratio target one",
        error_interpretation="successive-refinement indicators, not rigorous error bounds",
        parameters={**{k:ns["parameters"][k] for k in KINEMATICS},
                    "coordinate_branch":ns["parameters"].get("coordinate_branch")}, NS=nsv, R=rv, overlap=overlap)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns",type=Path,required=True)
    parser.add_argument("--r",type=Path,required=True)
    parser.add_argument("--ns-correction",type=Path)
    parser.add_argument("--r-correction",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    paths={k:getattr(args,k) for k in ("ns","r","ns_correction","r_correction")}
    reports={k:json.loads(p.read_text()) if p else None for k,p in paths.items()}
    result=compare(**reports)
    result["inputs"]={k:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for k,p in paths.items() if p}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("NS","R","overlap")},indent=2))


if __name__=="__main__":
    main()
