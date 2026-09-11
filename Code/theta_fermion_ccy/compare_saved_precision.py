"""Compare two saved physical blocks coefficient by coefficient; no recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp


def read_block(path):
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("status") != "computed; validation pending":
        raise ValueError(f"Incomplete physical block: {path}")
    result = {}
    for item in document["coefficients"]:
        key = tuple(item["exponents"])
        if len(key) != 4 or any(type(x) is not int or x < 0 for x in key):
            raise ValueError(f"Invalid coefficient exponents: {key}")
        if key in result or len(item["values"]) != 8:
            raise ValueError(f"Duplicate coefficient or incomplete parity vector: {key}")
        if sum(key) > 2*document["total_q_level"]:
            raise ValueError(f"Coefficient exceeds physical cutoff: {key}")
        values = tuple(mp.mpc(value["real"],value["imag"]) for value in item["values"])
        if any(not mp.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite coefficient: {key}")
        result[key] = values
    return document, result, hashlib.sha256(raw).hexdigest()


def encoded(value):
    return {"real": str(value.real), "imag": str(value.imag)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tolerance", default="1e-8")
    args = parser.parse_args()
    started = time.perf_counter()
    mp.mp.dps = 70
    tolerance = mp.mpf(args.tolerance)
    candidate, calculated, candidate_hash = read_block(args.candidate)
    reference, expected, reference_hash = read_block(args.reference)
    parameter_keys = ("total_q_level","b","momenta","p","f","etas",
                      "auxiliary_backend","universal_seed_power_after_recovery")
    for key in parameter_keys:
        if candidate[key] != reference[key]:
            raise ValueError(f"Incompatible saved blocks: {key}")
    zero = (mp.mpc(0),)*8
    records = []
    by_level = {}
    for key in sorted(calculated.keys() | expected.keys(), key=lambda x: (sum(x),x)):
        for parity, (got,want) in enumerate(zip(calculated.get(key,zero),expected.get(key,zero))):
            absolute = abs(got-want)
            scaled = absolute/max(1,abs(want))
            record = {"exponents": key, "parity_index": parity,
                      "candidate": encoded(got), "reference": encoded(want),
                      "absolute_error": str(absolute), "scaled_error": str(scaled)}
            records.append((absolute,scaled,record))
            level = sum(key)
            stats = by_level.setdefault(level, {"slots":0,"failed_slots":0,
                                               "max_absolute":mp.mpf(0),"max_scaled":mp.mpf(0)})
            stats["slots"] += 1
            stats["failed_slots"] += int(scaled > tolerance)
            stats["max_absolute"] = max(stats["max_absolute"],absolute)
            stats["max_scaled"] = max(stats["max_scaled"],scaled)
    if not records:
        raise ValueError("No saved physical coefficients to compare")
    worst_absolute = max(records,key=lambda x:x[0])
    worst_scaled = max(records,key=lambda x:x[1])
    failures = sum(scaled > tolerance for _,scaled,_ in records)
    report = {
        "status":"passed" if not failures else "failed",
        "scope":"All saved physical-block coefficients; no component, PBW or split-evaluation checks",
        "candidate_file":str(args.candidate), "reference_file":str(args.reference),
        "candidate_sha256":candidate_hash, "reference_sha256":reference_hash,
        "candidate_source_hashes":candidate.get("source_hashes",{}),
        "reference_source_hashes":reference.get("source_hashes",{}),
        "parameters":{key:candidate[key] for key in parameter_keys},
        "candidate_precision":{"ward_dps":candidate["ward_dps"],"ccy_dps":candidate["ccy_dps"]},
        "reference_precision":{"ward_dps":reference["ward_dps"],"ccy_dps":reference["ccy_dps"]},
        "comparison_arithmetic_dps":mp.mp.dps,
        "scaled_error_definition":"abs(candidate-reference)/max(1,abs(reference))",
        "tolerance":str(tolerance), "component_slots":len(records),
        "coefficient_vectors":len(calculated.keys() | expected.keys()),
        "keys_only_in_candidate":sorted(calculated.keys()-expected.keys()),
        "keys_only_in_reference":sorted(expected.keys()-calculated.keys()),
        "failed_slots":failures,
        "maximum_absolute_error":str(worst_absolute[0]),
        "maximum_scaled_error":str(worst_scaled[1]),
        "worst_absolute_slot":worst_absolute[2], "worst_scaled_slot":worst_scaled[2],
        "top_scaled_errors":[record for _,_,record in sorted(records,key=lambda x:x[1],reverse=True)[:10]],
        "by_twice_total_level":[{"twice_total_level":level,**stats} for level,stats in sorted(by_level.items())],
        "comparison_seconds":time.perf_counter()-started,
        "limitation":"Agreement with the saved 40-digit result is not an independent correctness proof or a guarantee at other levels/parameters",
    }
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps({key:report[key] for key in (
        "status","component_slots","failed_slots","maximum_absolute_error",
        "maximum_scaled_error","worst_scaled_slot","comparison_seconds")}),flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
