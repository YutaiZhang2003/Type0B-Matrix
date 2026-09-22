"""Compare the current C++ NSRR block with freshly computed physical PBW.

By default --below-order 5 checks all total levels 0, 1/2, ..., 9/2.
Use --through-order 5 to include level 5 in the independent comparison.
No saved numerical reference, auxiliary block, or double-Virasoro numerator
is used to generate the reference coefficients.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import mpmath as mp

CPP = Path(__file__).resolve().parents[1]
ROOT = CPP.parent
sys.path.insert(0, str(ROOT / "Code/ramond_zero_mode_recovery"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def number(value):
    result = mp.mpc(value["real"], value["imag"])
    if not (mp.isfinite(result.real) and mp.isfinite(result.imag)):
        raise ValueError("Nonfinite coefficient")
    return result


def encode_flint(value):
    return {"real": value.x.real.mid().str(105, radius=False),
            "imag": value.x.imag.mid().str(105, radius=False)}


def exact_complex(value, backend):
    parts = value.split(",")
    if len(parts) == 1:
        return backend.F(Fraction(parts[0]))
    if len(parts) == 2:
        return backend.F(Fraction(parts[0])) + backend.I * backend.F(Fraction(parts[1]))
    raise ValueError(f"Invalid complex rational: {value}")


def forbid_enlarged(*args, **kwargs):
    raise AssertionError("The PBW reference must not evaluate an enlarged block")


def candidate_rows(record, settings, cpp_level, dps):
    expected = dict(settings, status="computed", total_q_level=cpp_level,
                    truncation="total", dps=dps, sector_policy="error",
                    branching_method="stored_recursion")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Candidate {key}: expected {value!r}, got {record.get(key)!r}")
    rows = {}
    for row in record["coefficients"]:
        a, left, right, d = row["exponents"]
        if left != right or d % 2 or len(row["values"]) != 8:
            raise ValueError("Invalid diagonal physical coefficient")
        key = a, 2 * left, d
        if key in rows:
            raise ValueError(f"Duplicate coefficient: {key}")
        rows[key] = tuple(map(number, row["values"]))
    cutoff = 2 * cpp_level
    complete = {(a, b, c) for a in range(cutoff + 1)
                for b in range(0, cutoff - a + 1, 2)
                for c in range(0, cutoff - a - b + 1, 2)}
    if rows.keys() != complete:
        raise ValueError("Candidate has missing or unexpected monomials")
    return rows


def compare(calculated, expected, tolerance):
    differences = []
    for levels, reference in expected.items():
        for parity, (got, want) in enumerate(zip(calculated[levels], reference)):
            absolute = abs(got - want)
            scaled = absolute / max(mp.mpf(1), abs(want))
            differences.append((absolute, scaled, levels, parity, got, want))
    worst = max(differences, key=lambda row: row[1])
    per_order = []
    for order in sorted({sum(row[2]) for row in differences}):
        subset = [row for row in differences if sum(row[2]) == order]
        per_order.append(dict(total_order=str(Fraction(order, 2)),
                              monomials=len(subset) // 8,
                              maximum_absolute_error=mp.nstr(max(row[0] for row in subset), 18),
                              maximum_scaled_error=mp.nstr(max(row[1] for row in subset), 18)))
    return dict(
        status="passed" if worst[1] <= tolerance else "failed",
        monomials=len(expected), parity_components=len(differences),
        nonzero_pbw_components=sum(value != 0 for row in expected.values() for value in row),
        tolerance=str(tolerance),
        failing_components=sum(row[1] > tolerance for row in differences),
        maximum_absolute_error=mp.nstr(max(row[0] for row in differences), 18),
        maximum_scaled_error=mp.nstr(worst[1], 18),
        worst=dict(twice_levels=worst[2], parity_component=worst[3],
                   cpp=mp.nstr(worst[4], 25), pbw=mp.nstr(worst[5], 25)),
        by_total_order=per_order,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument("--below-order", type=int, choices=range(1, 6),
                              help="Strict upper total order; defaults to 5")
    cutoff_group.add_argument("--through-order", type=int, choices=range(1, 6),
                              help="Inclusive upper total order")
    parser.add_argument("--dps", type=int, nargs="+", default=[0, 40])
    parser.add_argument("--modes", nargs="+", choices=("ordinary", "inserted"),
                        default=["ordinary", "inserted"])
    parser.add_argument("--all-parities", action="store_true",
                        help="Check p,f=0,1 and eta=+/-1 in the selected production modes")
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--momenta", nargs=3, default=["11/23", "13/29", "17/31"])
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.below_order is None and args.through_order is None:
        args.below_order = 5
    cpp_level = args.through_order if args.through_order is not None else args.below_order
    maximum_twice_level = 2*cpp_level-(args.through_order is None)
    if any(dps != 0 and dps < 30 for dps in args.dps):
        parser.error("C++ precision must be 0 or at least 30 digits")
    binary = CPP / "bin/ramond"
    if not binary.is_file():
        parser.error("Build C++/bin/ramond with make -C C++ first")
    directory = args.results.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mp.mp.dps = max(120, max(args.dps) + 20)

    import complex_arithmetic as backend
    from check_level10_complex import load_runner

    runner = load_runner()
    started = time.perf_counter()
    oracle = runner.Check(exact_complex(args.b, backend),
                          tuple(exact_complex(value, backend) for value in args.momenta))
    # Make accidental use of the enlarged/double-Virasoro route fail immediately.
    oracle.enlarged_coefficient = forbid_enlarged
    oracle.edge = forbid_enlarged
    oracle.primary = forbid_enlarged
    levels = list(runner.level_triples(maximum_twice_level))
    source_paths = sorted(CPP.glob("include/ramond/*.hpp")) + [CPP / "src/main.cpp", Path(__file__)]
    source_paths += [ROOT / "Code/ramond_zero_mode_recovery" / name for name in (
        "check_level10_complex.py", "check_level10_modular.py", "modular_backend.py",
        "complex_arithmetic.py", "zero_mode_recovery.py")]
    source_paths += [ROOT / "Code/double_virasoro/nsrr" / name for name in (
        "nsrr_genus2_block.py", "ramond_pbw_generalized_ward.py")]
    source_paths.append(ROOT / "Code/c_Recursion/theta_star_algebra.py")
    source_paths += [ROOT / "Code/ramond_branching_recursion" / name for name in (
        "compute_target.py", "direct_state_check.py")]
    report = dict(status="running", git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        platform=platform.platform(), python=sys.version,
        below_total_order=args.below_order,
        through_total_order=args.through_order,
        maximum_compared_total_order=str(Fraction(maximum_twice_level, 2)),
        cpp_total_level=cpp_level, pbw_precision_bits=backend.ctx.prec,
        reference="Fresh physical SCA PBW Gram matrices and Human-Note Ward sewing",
        reference_numerics="FLINT complex midpoints; terms below 1e-80 discarded",
        error_definition="abs(C++ - PBW) / max(1, abs(PBW))",
        b=args.b, momenta=args.momenta, binary_sha256=digest(binary),
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        checks=[])
    cases = list(product((0, 1), (0, 1), (1, -1))) if args.all_parities else [(0, 0, 1)]
    for p, f, eta in cases:
        for mode in args.modes:
            etas = (eta, -eta if mode == "inserted" else eta)
            case = f"{mode}_p{p}_f{f}_eta{eta:+d}"
            pbw_started = time.perf_counter()
            reference = []
            for key in levels:
                values = oracle.physical_coefficient(key, p, f, etas)
                reference.append(dict(twice_levels=key, values=[encode_flint(x) for x in values]))
            pbw_seconds = time.perf_counter() - pbw_started
            settings = dict(mode=mode, p=p, f=f, etas=list(etas), b=args.b, momenta=args.momenta)
            pbw_path = directory / f"{case}_pbw.json"
            write_json(pbw_path, dict(settings, precision_bits=backend.ctx.prec,
                                     below_total_order=args.below_order,
                                     through_total_order=args.through_order,
                                     elapsed_seconds=pbw_seconds, coefficients=reference))
            expected = {tuple(row["twice_levels"]): tuple(map(number, row["values"]))
                        for row in reference}
            print(f"PBW {case}: {len(reference)} monomials, {pbw_seconds:.3f} s", flush=True)
            for dps in args.dps:
                output = directory / f"{case}_cpp_{dps}dps.json"
                command = [str(binary), "--mode", mode, "--level", str(cpp_level),
                           "--dps", str(dps), "--p", str(p), "--f", str(f), "--eta", str(eta),
                           "--b", args.b, "--P1", args.momenta[0], "--P2", args.momenta[1],
                           "--P3", args.momenta[2], "--json", str(output)]
                tick = time.perf_counter()
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
                wall_seconds = time.perf_counter() - tick
                log = directory / f"{case}_cpp_{dps}dps.log"
                log.write_text(result.stdout + result.stderr)
                check = dict(settings, dps=dps, command=command,
                             pbw_file=pbw_path.name, cpp_file=output.name,
                             pbw_sha256=digest(pbw_path), pbw_seconds=pbw_seconds,
                             cpp_wall_seconds=wall_seconds)
                if result.returncode:
                    check.update(status="failed", error=result.stderr.strip(), exit_code=result.returncode)
                else:
                    candidate = json.loads(output.read_text())
                    calculated = candidate_rows(candidate, settings, cpp_level, dps)
                    check.update(compare(calculated, expected, mp.mpf("1e-8" if dps == 0 else "1e-20")))
                    check.update(cpp_sha256=digest(output), diagnostics=candidate["diagnostics"],
                                 counts=candidate["counts"], timing_seconds=candidate["timing_seconds"])
                report["checks"].append(check)
                write_json(directory / "summary.json", report)
                print(f"C++ {case}, {dps} dps: {check['status']}; max scaled error "
                      f"{check.get('maximum_scaled_error', check.get('error'))}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["checks"]) else "failed"
    write_json(directory / "summary.json", report)
    print(f"{report['status']}: {len(report['checks'])} checks in {report['elapsed_seconds']:.3f} s", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
