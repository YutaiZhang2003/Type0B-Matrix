#!/usr/bin/env python3
"""Compare accepted binary64 NS-recursion coefficients with the reference backend.

This is a coefficient-level audit, independent of the momentum and moduli
quadratures.  Nodes rejected by the production precision policy are recorded
as reference fallbacks; accepted nodes must agree with the high-precision
recursion through the requested order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np


FIT_DIR=Path(__file__).resolve().parent/"heterotic_so23_1to3_vvvv_fit_bundle"
DEFAULT_OUTPUT=FIT_DIR.parents[1]/"Data Set"/"block_stability_q10.json"
sys.path.insert(0,str(FIT_DIR))

import heterotic_so23_1to3 as reference  # noqa: E402
import heterotic_so23_1to3_fast as stable  # noqa: E402


KINEMATICS=(
    (0.10j,0.12j,0.14j,0.36j),
    (0.11+0.13j,0.17+0.16j,0.23+0.18j,0.51+0.47j),
    ((0.20*(1/3+0.60j)),(0.30*(1/3+0.60j)),(0.50*(1/3+0.60j)),1/3+0.60j),
)
MOMENTA=(0.04,0.10,0.16,0.19,0.22,0.30,0.40,0.55,0.75,0.95,1.0,1.05,1.35,1.75,2.0,2.10,3.0)
STAR_PATTERNS=((False,False),(False,True),(True,False),(True,True))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-order",type=int,default=10)
    parser.add_argument("--mp-dps",type=int,default=70)
    parser.add_argument("--reference-p-max",type=float,default=0.18)
    parser.add_argument("--cancellation-limit",type=float,default=1.0e7)
    parser.add_argument("--accepted-relative-tolerance",type=float,default=1.0e-6)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    mp.mp.dps=args.mp_dps
    max_level2=2*args.q_order+1

    records=[]
    for kinematic_index,energies in enumerate(KINEMATICS):
        hf=tuple(stable.h_of_p(value) for value in energies)
        hr=tuple(reference.h_of_p(value) for value in energies)
        for momentum in MOMENTA:
            for star3,star2 in STAR_PATTERNS:
                fast=stable.FastNSBlockComputer(
                    hf[3],hf[2],hf[1],hf[0],star3,star2,
                    max_level2=max_level2,
                )
                exact=reference.NSBlockComputer(
                    hr[3],hr[2],hr[1],hr[0],star3,star2,
                    max_level2=max_level2,
                )
                errors=[]
                failure=""
                try:
                    for level2 in range(1,max_level2+1):
                        observed=complex(fast.coefficient(level2,stable.h_of_p(momentum)))
                        expected=complex(exact.coefficient(level2,reference.h_of_p(momentum)))
                        errors.append(abs(observed-expected)/max(abs(expected),1.0e-300))
                except Exception as exc:  # the production policy falls back on any such failure
                    failure=f"{type(exc).__name__}: {exc}"

                condition=float(fast.max_condition)
                accepted=(
                    momentum > args.reference_p_max
                    and not failure
                    and np.isfinite(condition)
                    and condition <= args.cancellation_limit
                )
                max_error=float(max(errors)) if errors else float("nan")
                records.append({
                    "kinematic_index":kinematic_index,
                    "momentum":momentum,
                    "star3":star3,
                    "star2":star2,
                    "accepted_fast":accepted,
                    "max_relative_coefficient_error":max_error,
                    "max_cancellation_ratio":condition,
                    "failure":failure,
                })

    accepted=[row for row in records if row["accepted_fast"]]
    violations=[
        row for row in accepted
        if row["max_relative_coefficient_error"] > args.accepted_relative_tolerance
    ]
    summary={
        "q_order":args.q_order,
        "mp_dps":args.mp_dps,
        "reference_p_max":args.reference_p_max,
        "cancellation_limit":args.cancellation_limit,
        "accepted_relative_tolerance":args.accepted_relative_tolerance,
        "total_block_cases":len(records),
        "accepted_fast_cases":len(accepted),
        "reference_fallback_cases":len(records)-len(accepted),
        "max_accepted_relative_coefficient_error":max(
            row["max_relative_coefficient_error"] for row in accepted
        ),
        "violations":len(violations),
        "records":records,
    }
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({key:value for key,value in summary.items() if key != "records"},indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
