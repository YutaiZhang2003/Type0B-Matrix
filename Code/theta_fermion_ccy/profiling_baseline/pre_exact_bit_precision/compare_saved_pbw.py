"""Compare a saved physical CCY block with direct PBW through a chosen level.

This performs only the requested physical coefficient comparison, retaining
unequal split powers and comparing them to zero. It does not recompute CCY
coefficients or perform split-parameter evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp

from compare_saved_precision import read_block, encoded


def stats(records):
    if not records:
        return {"slots":0}
    absolute = max(records,key=lambda item:item[0])
    scaled = max(records,key=lambda item:item[1])
    return {"slots":len(records),
            "maximum_absolute_error":str(absolute[0]),
            "maximum_scaled_error":str(scaled[1]),
            "worst_absolute_slot":absolute[2], "worst_scaled_slot":scaled[2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production",type=Path,required=True)
    parser.add_argument("--pbw",type=Path,required=True)
    parser.add_argument("--level",type=int,required=True)
    parser.add_argument("--json",type=Path,required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    mp.mp.dps = 70
    production, full, production_hash = read_block(args.production)
    raw = args.pbw.read_bytes()
    pbw = json.loads(raw)
    if pbw["status"] != "reference computed; comparison pending":
        raise ValueError("The PBW reference is incomplete")
    if not 0 <= args.level <= min(production["total_q_level"],pbw["total_q_level"]):
        raise ValueError("Comparison level exceeds a saved block's cutoff")
    for key in ("b","momenta","p","f","etas"):
        if production[key] != pbw[key]:
            raise ValueError(f"Block parameters differ: {key}")
    cutoff = 2*args.level
    calculated = {key:value for key,value in full.items() if sum(key) <= cutoff}
    expected = {}
    for item in pbw["coefficients"]:
        first,second,third = item["twice_levels"]
        if second % 2 or third % 2 or min(first,second,third) < 0:
            raise ValueError("PBW level labels do not match the NS-R-R channel")
        if first+second+third > cutoff:
            continue
        key = (first,second//2,second//2,third)
        if key in expected or len(item["values"]) != 8:
            raise ValueError("Duplicate PBW coefficient or incomplete parity vector")
        expected[key] = tuple(mp.mpc(value["real"],value["imag"]) for value in item["values"])
        if any(not mp.isfinite(value) for value in expected[key]):
            raise ValueError("Nonfinite PBW coefficient")
    zero = (mp.mpc(0),)*8
    records, levels = [], {}
    for key in sorted(calculated.keys() | expected.keys(),key=lambda x:(sum(x),x)):
        for parity,(got,want) in enumerate(zip(calculated.get(key,zero),expected.get(key,zero))):
            absolute = abs(got-want)
            scaled = absolute/max(1,abs(want))
            record = (absolute,scaled,{
                "exponents":key,"parity_index":parity,
                "candidate":encoded(got),"pbw":encoded(want),
                "absolute_error":str(absolute),"scaled_error":str(scaled)})
            records.append(record)
            levels.setdefault(sum(key),[]).append(record)
    if not records:
        raise ValueError("No physical coefficients to compare")
    cumulative = []
    for level in range(args.level+1):
        subset = [record for record in records if sum(record[2]["exponents"]) <= 2*level]
        cumulative.append({"through_total_level":level,**stats(subset)})
    report = {
        "status":"comparison completed",
        "scope":"Physical PBW comparison only; all retained parity slots and unequal split powers included",
        "production_file":str(args.production),"pbw_file":str(args.pbw),
        "production_sha256":production_hash,"pbw_sha256":hashlib.sha256(raw).hexdigest(),
        "production_source_hashes":production.get("source_hashes",{}),
        "production_computed_through_level":production["total_q_level"],
        "comparison_through_level":args.level,
        "parameters":{key:production[key] for key in ("b","momenta","p","f","etas")},
        "pbw_precision_bits":pbw["precision_bits"],
        "pbw_arithmetic":"FLINT complex midpoints, inherited scalar clipping below 1e-80",
        "comparison_arithmetic_dps":mp.mp.dps,
        "scaled_error_definition":"abs(candidate-PBW)/max(1,abs(PBW))",
        "pbw_split_embedding":"(a,b,c) twice-levels -> (a,b/2,b/2,c); all other split powers have zero PBW coefficient",
        "candidate_coefficient_vectors":len(calculated),
        "pbw_diagonal_vectors":len(expected),
        "overall":stats(records),
        "equal_split_powers":stats([r for r in records if r[2]["exponents"][1] == r[2]["exponents"][2]]),
        "unequal_split_powers":stats([r for r in records if r[2]["exponents"][1] != r[2]["exponents"][2]]),
        "by_twice_total_level":[{"twice_total_level":level,**stats(values)} for level,values in sorted(levels.items())],
        "cumulative_by_integer_level":cumulative,
        "top_scaled_errors":[r[2] for r in sorted(records,key=lambda r:r[1],reverse=True)[:10]],
        "comparison_seconds":time.perf_counter()-started,
    }
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"overall":report["overall"],
                      "equal_split_powers":report["equal_split_powers"],
                      "unequal_split_powers":report["unequal_split_powers"],
                      "cumulative":[{"level":x["through_total_level"],"maximum_scaled_error":x["maximum_scaled_error"]} for x in cumulative],
                      "comparison_seconds":report["comparison_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
