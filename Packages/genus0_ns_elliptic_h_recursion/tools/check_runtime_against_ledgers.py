#!/usr/bin/env python3
"""Recompute the packaged runtime against all saved independent audit ledgers."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp

from genus0_ns_elliptic_h_recursion import compute_h_recursion

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points",type=int,choices=(4,5,6),required=True)
    parser.add_argument("--case",choices=tuple("ABCD"),default="A")
    parser.add_argument("--degree",type=int,default=10)
    parser.add_argument("--dps",type=int,default=80)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    ledger = ROOT/"validation/Data Set/h-Recursion"/f"ns_pillow_c_comparison_n{args.points}_{args.case}_degree10_dps{args.dps}_20260830.json"
    data = json.loads(ledger.read_text())
    if not 0<=args.degree<=10:
        parser.error("saved ledgers cover total degrees zero through ten")
    start = time.monotonic()
    table = compute_h_recursion(b=data["b"],external_weights=data["external"],
                internal_weights=data["internal"],order=args.degree,dps=args.dps)
    with mp.workdps(args.dps):
        maxima = {name:mp.mpf(0) for name in ("c_recursion_H","h_recursion_H","pbw_H")}
        for row in data["coefficients"]:
            key = tuple(row["twice_levels"])
            if sum(key)>2*args.degree:
                continue
            actual = table.coefficients[key]
            for name in maxima:
                expected = mp.mpc(row[name]["real"],row[name]["imag"])
                error = abs(actual-expected)/max(1,abs(actual),abs(expected))
                if not mp.isfinite(error):
                    raise ArithmeticError("nonfinite comparison")
                maxima[name] = max(maxima[name],error)
        passed = all(value<mp.mpf("1e-50") for value in maxima.values())
        result = {"passed":passed,"points":args.points,"case":args.case,"degree":args.degree,
                  "dps":args.dps,"coefficient_count":len(table.coefficients),
                  "maximum_scaled_errors":{name:mp.nstr(value,12) for name,value in maxima.items()},
                  "reference_file":str(ledger.relative_to(ROOT)),
                  "reference_sha256":hashlib.sha256(ledger.read_bytes()).hexdigest(),
                  "runtime_source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT/"src/genus0_ns_elliptic_h_recursion").glob("*.py"))},
                  "seconds":time.monotonic()-start}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
