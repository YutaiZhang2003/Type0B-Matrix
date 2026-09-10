"""Only the two numerical cross-checks explicitly requested by the user.

Consumes the production output and the independently generated PBW output.
It neither computes nor substitutes any numerator/auxiliary coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp

from series_algebra import evaluate


def number(value):
    return mp.mpc(value["real"], value["imag"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--pbw", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tolerance", default="1e-8")
    args = parser.parse_args()
    mp.mp.dps = 70
    tolerance = mp.mpf(args.tolerance)
    production = json.loads(args.production.read_text())
    reference = json.loads(args.pbw.read_text())
    if production["total_q_level"] != reference["total_q_level"]:
        raise ValueError("Production and PBW cutoffs differ")
    for key in ("b", "momenta", "p", "f", "etas"):
        if production[key] != reference[key]:
            raise ValueError(f"Production and PBW parameter mismatch: {key}")
    if reference["status"] != "reference computed; comparison pending":
        raise ValueError("The independent PBW output is incomplete")
    if production["status"] != "computed; validation pending":
        raise ValueError("The production output is incomplete")
    calculated = {tuple(item["exponents"]): tuple(map(number, item["values"]))
                  for item in production["coefficients"]}
    expected = {}
    for item in reference["coefficients"]:
        first, second, third = item["twice_levels"]
        expected[first, second//2, second//2, third] = tuple(map(number, item["values"]))
    zero = (mp.mpc(0),)*8
    maximum = mp.mpf(0)
    maximum_scaled = mp.mpf(0)
    worst = None
    compared = 0
    for key in calculated.keys() | expected.keys():
        for parity, (got, want) in enumerate(zip(calculated.get(key, zero),
                                                expected.get(key, zero))):
            error = abs(got-want)
            scaled = error/max(1, abs(want))
            maximum = max(maximum, error)
            if scaled > maximum_scaled:
                maximum_scaled, worst = scaled, (key, parity)
            compared += 1

    q1, q2, q3 = map(mp.mpf, ("0.013", "0.007", "0.009"))
    split_records = []
    split_maximum = mp.mpf(0)
    for index in range(8):
        signs = tuple(-1 if (index >> edge) & 1 else 1 for edge in range(3))
        values = []
        for u in map(mp.mpf, ("0.5", "1", "2")):
            left, right = u*mp.sqrt(q2), mp.sqrt(q2)/u
            value = evaluate(calculated, q1=q1, q2_left=left,
                             q2_right=right, q3=q3, spin_signs=signs)
            values.append(value)
            split_records.append({
                "spin_signs": signs, "u": str(u),
                "q2_left": str(left), "q2_right": str(right),
                "product": str(left*right),
                "value": {"real": str(value.real), "imag": str(value.imag)},
            })
        split_maximum = max(split_maximum,
                            *(abs(value-values[1])/max(1, abs(values[1]))
                              for value in values))
    passed = maximum_scaled <= tolerance and split_maximum <= tolerance
    report = {
        "status": "passed" if passed else "failed",
        "production_file": str(args.production), "pbw_file": str(args.pbw),
        "production_source_hashes": production.get("source_hashes", {}),
        "total_q_level": production["total_q_level"],
        "parameters": {key: production[key] for key in ("b", "momenta", "p", "f", "etas")},
        "tolerance": str(tolerance),
        "pbw_comparison": {
            "component_slots": compared, "maximum_absolute_error": str(maximum),
            "maximum_scaled_error": str(maximum_scaled), "worst_slot": worst,
        },
        "split_product_comparison": {
            "q1": str(q1), "q2_product": str(q2), "q3": str(q3),
            "maximum_scaled_difference": str(split_maximum), "evaluations": split_records,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"],
                      "pbw_max_scaled_error": str(maximum_scaled),
                      "split_max_scaled_difference": str(split_maximum)}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
