#!/usr/bin/env python3
"""Compare the NS pillow h-recursion with Belavin--Geiko c-recursion.

The c-recursion is evaluated in plane plumbing ratios using the independent
Code/c_Recursion implementation of arXiv:1806.09563, sections 3.1--3.2 and
Appendix A. Its seed is the osp(1|2) global block. It never calls the pillow
h-recursion or a PBW solver. Only the coordinate pullback is shared with the
earlier PBW audit. All blocks use the human note's ordinary central charge
and fixed-parity three-forms; H always means the unit-seed elliptic block.

Degree N means sum(twice_levels) <= 2*N, including every parity sector.
An optional existing PBW ledger provides a third, independently computed
comparison; it is not used to construct either recursion's coefficients.
"""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import mpmath as mp

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "c_Recursion"))
from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from ns_pillow_elliptic_audit import PillowMap, NSEllipticRecursion, indices


# The same generic fixtures as the preceding independent PBW audit. They
# are duplicated here so this comparison does not import a PBW engine.
CASES = {
    "A": ("1.27", (".31", ".42", ".53", ".37", ".47", ".28"), (".73", "1.10", "1.37")),
    "B": ("1.43", (".22", ".61", ".39", ".58", ".74", ".45"), ("1.13", ".85", "1.69")),
    "C": (".83", (".17", ".83", "1.21", ".46", ".34", ".67"), (".19", "1.47", ".82")),
    "D": ("1.61", ("1.12", ".26", ".79", "1.33", ".58", ".91"), ("2.31", ".64", "1.09")),
}


def sector_labels(twice_levels):
    epsilon = tuple(n % 2 for n in twice_levels)
    return (epsilon[0],) + tuple(a ^ b for a, b in zip(epsilon, epsilon[1:])) + (epsilon[-1],)


class CachedLiteratureCRecursion(NSSphereLinearCRecursion):
    """Memoization and an all-parity entry point; no changed recursion rules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Per-instance caches retain the precision of this single run.
        self._edge_residue = lru_cache(None)(self._edge_residue)
        self._global_coefficient = lru_cache(None)(self._global_coefficient)

    def coefficient_for_parity(self, twice_levels):
        key = tuple(twice_levels)
        if len(key) != self.edge_count or any(type(n) is not int or n < 0 for n in key):
            raise ValueError("invalid nonnegative twice-level tuple")
        with mp.workdps(self.working_precision):
            return self._coefficient(
                key, self.central_charge, self.internal_weights, sector_labels(key)
            )


def scaled_error(left, right):
    return abs(left - right) / max(1, abs(left), abs(right))


def number(value, digits):
    return {"real": mp.nstr(mp.re(value), digits), "imag": mp.nstr(mp.im(value), digits)}


def source_hashes():
    paths = (
        Path(__file__),
        CODE / "c_Recursion/ns_multipoint_c_recursion.py",
        CODE / "c_Recursion/ns_recursion_recipe.py",
        CODE / "c_Recursion/ns_global_osp_block.py",
        Path(__file__).with_name("ns_pillow_elliptic_audit.py"),
    )
    return {str(path.relative_to(CODE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def load_pbw_ledger(path, args, b, external, internal, keys):
    if path is None:
        return None, None
    ledger = json.loads(path.read_text())
    if not (ledger["passed"] and ledger["mode"] == "numeric"
            and ledger["points"] == args.points
            and ledger["total_physical_degree"] == args.degree
            and ledger["case"] == args.case and ledger["dps"] >= args.dps):
        raise ValueError("PBW ledger does not match the requested audit or precision")
    if (mp.mpf(ledger["b"]) != b
            or tuple(map(mp.mpf, ledger["external"])) != external
            or tuple(map(mp.mpf, ledger["internal"])) != internal):
        raise ValueError("PBW ledger parameters differ")
    coefficients = {tuple(row["twice_levels"]): mp.mpf(row["pbw_H"])
                    for row in ledger["coefficients"]}
    if set(coefficients) != set(keys):
        raise ValueError("PBW ledger has a different coefficient cutoff")
    metadata = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dps": ledger["dps"], "reuse": "Previously computed PBW H values only; not a seed or input to either recursion"}
    return coefficients, metadata


def run(args):
    start = time.monotonic()
    mp.mp.dps = args.dps
    m = args.points - 3
    btext, ds, hs = CASES[args.case]
    b = mp.mpf(btext)
    external = tuple(map(mp.mpf, ds[:args.points-2] + ds[-2:]))
    internal = tuple(map(mp.mpf, hs[:m]))
    c = mp.mpf("1.5") + 3*(b+1/b)**2
    keys = sorted(indices(m, 2*args.degree), key=lambda k: (sum(k), k))
    pbw, pbw_metadata = load_pbw_ledger(args.pbw_ledger, args, b, external, internal, keys)
    c_block = CachedLiteratureCRecursion(
        central_charge=c, external_weights=external, internal_weights=internal,
        vertex_sectors=(0,)*(m+1), working_precision=args.dps,
    )
    plane = {}
    for key in keys:
        plane[key] = c_block.coefficient_for_parity(key)
        if not any(key[1:]):
            print("C-RECURSION", key, f"{time.monotonic()-start:.2f}s", flush=True)
    c_seconds = time.monotonic() - start
    pulled = PillowMap(m, args.degree).pullback(plane, c, external, internal)
    pullback_seconds = time.monotonic() - start - c_seconds
    print("Pillow pullback ready", f"{time.monotonic()-start:.2f}s", flush=True)
    h_block = NSEllipticRecursion(b, external, internal)
    rows, parity_errors, degree_errors = [], {}, {}
    maximum = maximum_pbw = maximum_imaginary = mp.mpf(0)
    worst_key = None
    tolerance = mp.mpf(args.tolerance)
    passed = True
    for key in keys:
        left, right = pulled.get(key, 0), h_block.coefficient(key)
        error = scaled_error(left, right)
        if error > maximum:
            maximum, worst_key = error, key
        maximum_imaginary = max(maximum_imaginary, abs(mp.im(left))/max(1,abs(left)))
        parity = "".join(str(n%2) for n in key)
        degree = str(mp.mpf(sum(key))/2)
        parity_errors[parity] = max(parity_errors.get(parity, 0), error)
        degree_errors[degree] = max(degree_errors.get(degree, 0), error)
        ok = bool(mp.isfinite(error) and error < tolerance)
        row = {"twice_levels": key, "c_recursion_plane": number(plane[key], args.dps),
               "c_recursion_H": number(left, args.dps),
               "h_recursion_H": number(right, args.dps),
               "scaled_error": mp.nstr(error, 12)}
        if pbw is not None:
            pbw_error = scaled_error(left, pbw[key])
            maximum_pbw = max(maximum_pbw, pbw_error)
            ok = ok and bool(mp.isfinite(pbw_error) and pbw_error < tolerance)
            row.update(pbw_H=number(pbw[key], args.dps),
                       c_vs_pbw_scaled_error=mp.nstr(pbw_error, 12))
        passed = passed and ok
        rows.append(row)
        if not any(key[1:]) or not ok:
            print("CHECK", key, "PASS" if ok else "FAIL", f"{time.monotonic()-start:.2f}s", flush=True)
    return {
        "passed": passed, "points": args.points, "case": args.case,
        "total_physical_degree": args.degree, "twice_level_bound": 2*args.degree,
        "coefficient_count": len(keys), "parity_sector_count": len(parity_errors),
        "dps": args.dps, "tolerance": args.tolerance,
        "b": mp.nstr(b, args.dps), "c": mp.nstr(c, args.dps),
        "external": list(map(str, external)), "internal": list(map(str, internal)),
        "literature": "Belavin--Geiko, arXiv:1806.09563, eqs. (3.6)--(3.18), Appendix A",
        "normalization": "Human note: F = Lambda^(c) prod(varrho_i^(h_i-c/24)) C_NS H",
        "central_charge_conversion": "c_human = (3/2) c_BG; pole and Jacobian both converted",
        "c_recursion_seed": "osp(1|2) global sewing; no pillow seed, no PBW coefficient input",
        "c_child": "c -> c_rs(h_k), only h_k -> h_k+rs/2; odd rs toggles adjacent sectors",
        "h_child": "fixed c, common-weight pole shift on all h_i, then h_k += rs/2",
        "coordinate_conversion": "Plane nested ratios x_i to elliptic p_i via arbitrary-order PillowMap, including full conformal factor",
        "maximum_scaled_error": mp.nstr(maximum, 12), "worst_twice_levels": worst_key,
        "maximum_error_by_parity": {k: mp.nstr(v, 12) for k,v in parity_errors.items()},
        "maximum_error_by_total_degree": {k: mp.nstr(v, 12) for k,v in degree_errors.items()},
        "maximum_scaled_imaginary_part": mp.nstr(maximum_imaginary, 12),
        "maximum_c_vs_pbw_scaled_error": mp.nstr(maximum_pbw, 12) if pbw is not None else None,
        "pbw_reference": pbw_metadata,
        "minimum_h_recursion_denominator": mp.nstr(h_block.minimum_denominator, 12),
        "c_coefficient_cache_entries": len(c_block._coefficient_cache),
        "c_edge_cache": c_block._edge_residue.cache_info()._asdict(),
        "c_recursion_seconds": c_seconds, "pullback_seconds": pullback_seconds,
        "seconds": time.monotonic()-start, "source_sha256": source_hashes(),
        "coefficients": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, choices=(4,5,6), default=6)
    parser.add_argument("--case", choices=tuple(CASES), default="A")
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--tolerance", default="1e-50")
    parser.add_argument("--pbw-ledger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.degree < 0 or args.dps < 20:
        parser.error("degree must be nonnegative and dps at least 20")
    if not mp.isfinite(mp.mpf(args.tolerance)) or mp.mpf(args.tolerance) <= 0:
        parser.error("tolerance must be positive and finite")
    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "coefficients"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
