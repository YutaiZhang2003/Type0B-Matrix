"""Check the C++ physical pipelines against saved Python blocks; never run PBW.

Use --run to recompute the C++ blocks in fresh processes. The reference data are
the existing inserted 40-digit level-5 run and ordinary 384-bit level-10 run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mpmath as mp

CPP = Path(__file__).resolve().parents[1]
ROOT = CPP.parent
sys.path.insert(0, str(ROOT / "Code/ramond_zero_mode_recovery"))
from compare_saved_recovery_methods import load_series  # noqa: E402

REFERENCES = {
    "inserted": ROOT / "Code/theta_fermion_ccy/results/forward_diagonal_40dps_level5.json",
    "ordinary": ROOT / "Code/ramond_zero_mode_recovery/restricted_level10.json",
}


def label(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def numeric_rows(record, field="coefficients"):
    result = {}
    for row in record[field]:
        a, left, right, d = row["exponents"]
        if left != right or len(row["values"]) != 8:
            raise ValueError("Not a diagonal eight-component coefficient")
        key = (a, 2 * left, d)
        if key in result:
            raise ValueError("Duplicate coefficient")
        result[key] = tuple(mp.mpc(v["real"], v["imag"]) for v in row["values"])
    return result


def check(mode, level, dps, directory):
    precision = "machine" if dps == 0 else f"{dps}dps"
    path = directory / f"{mode}_level{level}_{precision}.json"
    candidate = json.loads(path.read_text())
    reference_path = REFERENCES[mode]
    reference = json.loads(reference_path.read_text())
    for key, expected in dict(mode=mode, total_q_level=level, dps=dps, b="7/5",
                              momenta=["11/23", "13/29", "17/31"], p=0, f=0,
                              etas=[1, -1 if mode == "inserted" else 1]).items():
        if candidate.get(key) != expected:
            raise ValueError(f"Wrong {key} in {path}")
    if mode == "inserted":
        for key in ["b", "momenta", "p", "f", "etas"]:
            if candidate[key] != reference[key]:
                raise ValueError(f"Reference {key} mismatch")
        saved = numeric_rows(reference)
    else:
        expected = dict(b="7/5", P=["11/23", "13/29", "17/31"],
                        fermion_parity=0, primary_parity=0, three_point_eta=[1, 1])
        if reference["parameters"] != expected:
            raise ValueError("Ordinary reference parameter mismatch")
        saved = load_series(reference)
    cutoff = 2 * level
    wanted = {(a, b, c) for a in range(cutoff + 1)
              for b in range(0, cutoff - a + 1, 2)
              for c in range(0, cutoff - a - b + 1, 2)}
    saved = {k: v for k, v in saved.items() if sum(k) <= cutoff}
    current = numeric_rows(candidate)
    if current.keys() != wanted or saved.keys() != wanted:
        raise ValueError("Incomplete physical total-level truncation")
    differences = []
    for key in sorted(wanted):
        for parity, (a, b) in enumerate(zip(current[key], saved[key])):
            absolute = abs(a - b)
            scaled = absolute / max(mp.mpf(1), abs(a), abs(b))
            differences.append((absolute, scaled, key, parity))
    worst = max(differences, key=lambda x: x[1])
    tolerance = mp.mpf("1e-8" if dps == 0 else "1e-20")
    return dict(status="passed" if worst[1] <= tolerance else "failed",
                candidate=label(path),
                reference=label(reference_path),
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                total_level=level, dps=dps,
                precision_bits=candidate.get("precision_bits"),
                monomials=len(wanted), parity_components=len(differences),
                maximum_absolute_difference=mp.nstr(max(d[0] for d in differences), 18),
                maximum_scaled_difference=mp.nstr(worst[1], 18),
                worst_twice_levels=worst[2], worst_parity=worst[3],
                scaled_tolerance=str(tolerance), timing_seconds=candidate["timing_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--level", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--results", type=Path, default=CPP / "results")
    args = parser.parse_args()
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    mp.mp.dps = 100
    checks = []
    for mode in ["ordinary", "inserted"]:
        for dps in [0, 40]:
            precision = "machine" if dps == 0 else f"{dps}dps"
            path = args.results / f"{mode}_level{args.level}_{precision}.json"
            if args.run:
                subprocess.run([str(CPP / "bin/ramond"), "--mode", mode, "--level",
                                str(args.level), "--dps", str(dps), "--json", str(path)],
                               check=True, cwd=ROOT)
            checks.append(check(mode, args.level, dps, args.results))
    report = dict(status="passed" if all(c["status"] == "passed" for c in checks) else "failed",
                  scaled_difference="abs(a-b)/max(1,abs(a),abs(b))",
                  scope="All physical coefficients through the stated cutoff; saved references only",
                  limitation="Agreement with saved numerical results is not a certified error bound",
                  checks=checks)
    path = args.results / f"pipeline_validation_level{args.level}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for c in checks:
        print(f"{c['candidate']}: {c['status']}; max scaled {c['maximum_scaled_difference']}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
