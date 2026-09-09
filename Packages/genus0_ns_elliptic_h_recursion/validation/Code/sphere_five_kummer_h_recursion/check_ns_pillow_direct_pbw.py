#!/usr/bin/env python3
"""Independent PBW audit of the NS sphere elliptic h-recursion proposal.

Physical total degree N includes all twice-level tuples with sum <= 2*N.
Both integer and half-integer edge powers, hence every parity sector, are
included. The direct side uses only the NS algebra and primary Ward identities.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import mpmath as mp
import sympy as sp

from ns_pillow_direct_pbw import DirectNSSpherePBW, NSModule, NSPrimaryWard, basis
from ns_pillow_elliptic_audit import PillowMap, NSEllipticRecursion, indices


CASES = {
    "A": ("1.27", (".31", ".42", ".53", ".37", ".47", ".28"), (".73", "1.10", "1.37")),
    "B": ("1.43", (".22", ".61", ".39", ".58", ".74", ".45"), ("1.13", ".85", "1.69")),
    "C": (".83", (".17", ".83", "1.21", ".46", ".34", ".67"), (".19", "1.47", ".82")),
    "D": ("1.61", ("1.12", ".26", ".79", "1.33", ".58", ".91"), ("2.31", ".64", "1.09")),
}


def symbolic_residue_checks(keys,pulled,b,c,external,internal,start):
    """Exact rational-identity certificate without expanding the whole RHS.

Induction on total degree: the PBW coefficient has only the listed simple
common-weight poles, each residue equals the proposed kernel times the
lower-degree PBW coefficient, and its polynomial part is the unit seed.
These conditions uniquely determine the same rational function as the
expanded recursion. All checks are algebraic with generic symbols.
"""
    central = sp.Rational(3,2)+3*(b+1/b)**2
    domain = sp.QQ.frac_field(b)
    recursion = NSEllipticRecursion(b,external,internal,True)
    fractions,rows = {},[]
    large = sp.Symbol("large_weight")
    m = len(internal)
    for key in keys:
        reduced = sp.cancel(pulled.get(key,0))
        numerator,denominator = sp.together(reduced.subs(c,central)).as_numer_denom()
        fractions[key] = (numerator,denominator)
        poles = [(edge,r,s) for edge,n in enumerate(key) for r in range(1,n+1)
                 for s in range(1,n//r+1) if (r+s)%2==0]
        allowed = sp.prod(internal[edge]-recursion.pole(r,s) for edge,r,s in poles)
        dpoly = sp.Poly(denominator,*internal,domain=domain)
        pole_ok = sp.Poly(allowed,*internal,domain=domain).rem(dpoly).is_zero
        shift = {h:h+large for h in internal}
        npoly = sp.Poly(numerator.subs(shift,simultaneous=True),large)
        qpoly = sp.Poly(denominator.subs(shift,simultaneous=True),large)
        seed = 1 if not any(key) else 0
        regular_ok = npoly.degree()<qpoly.degree() and seed==0
        if npoly.degree()==qpoly.degree():
            regular_ok = sp.cancel(npoly.LC()/qpoly.LC()-seed)==0
        checks = []
        parity = [n%2 for n in key]
        labels = [parity[0]]+[a^b for a,b in zip(parity,parity[1:])]+[parity[-1]]
        for edge,r,s in poles:
            h = internal[edge]
            pole = recursion.pole(r,s)
            if sp.cancel(denominator.subs(h,pole))==0:
                residue = numerator.subs(h,pole)/sp.diff(denominator,h).subs(h,pole)
            else:
                residue = sp.S.Zero
            left = (external[0],external[1]) if edge==0 else (internal[edge-1],external[edge+1])
            right = (external[-1],external[-2]) if edge==m-1 else (internal[edge+1],external[edge+2])
            kernel = ((-1)**(r*s)*recursion.norm_slope(r,s)
                      *recursion.fusion(r,s,labels[edge],*left)
                      *recursion.fusion(r,s,labels[edge+1],*right))
            child_key = list(key)
            child_key[edge] -= r*s
            cn,cd = fractions[tuple(child_key)]
            child = (cn/cd).subs(h,pole+sp.Rational(r*s,2))
            factor = 4**(r*s) if m==1 else (2**(r*s) if edge in (0,m-1) else 1)
            difference = sp.cancel(residue-factor*kernel*child)
            checks.append({"edge":edge+1,"r":r,"s":s,"exact_zero":difference==0})
        ok = bool(pole_ok and regular_ok and all(item["exact_zero"] for item in checks))
        rows.append({"twice_levels":key,"exact_zero":ok,"simple_pole_catalog_passed":bool(pole_ok),
                     "regular_part_passed":bool(regular_ok),"residue_checks":checks})
        print("RESIDUE CERTIFICATE",key,"PASS" if ok else "FAIL",f"{time.monotonic()-start:.2f}s",flush=True)
    return rows


def exact_oracle_check():
    """Cross-check the independent algebra engine, not just final blocks."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"c_Recursion"))
    from ns_genus2_symbolic_low_order import ExactNSVermaModule, ExactNSDescendantThreeForm
    c, h, d, k = sp.symbols("c h d k")
    module = NSModule(c,h,True)
    oracle = ExactNSVermaModule(c=c,weight=h)
    ward = NSPrimaryWard(c,(h,d,k),True)
    oracle_ward = ExactNSDescendantThreeForm(c=c,weights=(h,d,k))
    gram_count = ward_count = 0
    for n in range(7):
        for bra in basis(n):
            for ket in basis(n):
                assert sp.expand(module.inner(bra,ket)-oracle.inner_product(bra,ket)) == 0
                gram_count += 1
        for p in range(7-n):
            for bra in basis(n):
                for ket in basis(p):
                    for upper in (0,1):
                        if n+p+upper>6:
                            continue
                        middle = (("G",-1),) if upper else ()
                        assert sp.cancel(ward.value(bra,upper,ket)-oracle_ward.value(bra,middle,ket)) == 0
                        ward_count += 1
    return {"passed":True,"gram_entries":gram_count,"ward_entries":ward_count,
            "maximum_total_level":3,"parameters":"independent symbols c,h,d,k"}


def run(args):
    start = time.monotonic()
    if args.mode == "oracle":
        return {"mode":"oracle", **exact_oracle_check(), "seconds":time.monotonic()-start}
    symbolic = args.mode == "symbolic"
    m = args.points-3
    mp.mp.dps = args.dps
    if symbolic:
        b, c = sp.symbols("b c")
        external = sp.symbols(f"d1:{args.points+1}")
        internal = sp.symbols(f"h1:{m+1}")
        central_charge = sp.Rational(3,2)+3*(b+1/b)**2
    else:
        btext, ds, hs = CASES[args.case]
        b = mp.mpf(btext)
        # Remove the second mobile weight when restricting the six-point fixture.
        external = tuple(map(mp.mpf,ds[:args.points-2]+ds[-2:]))
        internal = tuple(map(mp.mpf,hs[:m]))
        c = central_charge = mp.mpf("1.5")+3*(b+1/b)**2
    keys = sorted(indices(m,2*args.degree),key=lambda k:(sum(k),k))
    direct = DirectNSSpherePBW(c,external,internal,symbolic)
    plane = {}
    for key in keys:
        plane[key] = direct.coefficient(key)
        if symbolic or not any(key[1:]):
            print("PBW",key,f"{time.monotonic()-start:.2f}s",flush=True)
    pbw_seconds = time.monotonic()-start
    geometry = PillowMap(m,args.degree,symbolic)
    pulled = geometry.pullback(plane,c,external,internal)
    print("Pillow pullback ready",f"{time.monotonic()-start:.2f}s",flush=True)
    recursion = NSEllipticRecursion(b,external,internal,symbolic)
    rows, parity_errors, degree_errors = [], {}, {}
    maximum = mp.mpf(0)
    passed = True
    tolerance = mp.mpf(args.tolerance)
    if symbolic and args.symbolic_method=="residues":
        rows = symbolic_residue_checks(keys,pulled,b,c,external,internal,start)
        passed = all(row["exact_zero"] for row in rows)
    for key in ([] if symbolic and args.symbolic_method=="residues" else keys):
        lhs = pulled.get(key,0)
        rhs = recursion.coefficient(key)
        if symbolic:
            # Simplify in c first to keep the independent PBW matrices compact.
            lhs = sp.cancel(lhs).subs(c,central_charge)
            difference = sp.cancel(lhs-rhs)
            ok = difference == 0
            row = {"twice_levels":key,"exact_zero":ok}
            if not ok:
                row["difference"] = str(difference)
        else:
            error = abs(lhs-rhs)/max(1,abs(lhs),abs(rhs))
            maximum = max(maximum,error)
            parity = "".join(str(n%2) for n in key)
            degree = str(mp.mpf(sum(key))/2)
            parity_errors[parity] = max(parity_errors.get(parity,0),error)
            degree_errors[degree] = max(degree_errors.get(degree,0),error)
            ok = error<tolerance
            row = {"twice_levels":key,"pbw_H":mp.nstr(lhs,args.dps),
                   "recursive_H":mp.nstr(rhs,args.dps),"scaled_error":mp.nstr(error,12)}
        passed = passed and ok
        rows.append(row)
        if symbolic or not any(key[1:]) or not ok:
            print("CHECK",key,"PASS" if ok else "FAIL",f"{time.monotonic()-start:.2f}s",flush=True)
    result = {
        "mode":args.mode,"points":args.points,"total_physical_degree":args.degree,
        "twice_level_bound":2*args.degree,"coefficient_count":len(keys),
        "all_parity_sectors":True,"passed":passed,
        "normalization":"Human note: F = Lambda^(c) prod(varrho_i^(h_i-c/24)) C_NS H",
        "direct_method":"NS PBW Gram solves and primary Ward sewing; no c/h recursion",
        "central_charge_relation":"c = 3/2 + 3*(b+1/b)^2",
        "b":str(b),"c":str(central_charge),
        "external":list(map(str,external)),"internal":list(map(str,internal)),
        "pbw_seconds":pbw_seconds,"seconds":time.monotonic()-start,
        "maximum_gram_dimension":len(basis(2*args.degree)),"coefficients":rows,
    }
    if not symbolic:
        result.update(case=args.case,dps=args.dps,tolerance=args.tolerance,
                      maximum_scaled_error=mp.nstr(maximum,12),
                      maximum_error_by_parity={k:mp.nstr(v,12) for k,v in parity_errors.items()},
                      maximum_error_by_total_degree={k:mp.nstr(v,12) for k,v in degree_errors.items()},
                      minimum_visited_recursion_denominator=mp.nstr(recursion.minimum_denominator,12))
    else:
        result["symbolic_method"] = args.symbolic_method
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("oracle","symbolic","numeric"),required=True)
    parser.add_argument("--points",type=int,choices=(4,5,6),default=5)
    parser.add_argument("--degree",type=int,default=3)
    parser.add_argument("--case",choices=tuple(CASES),default="A")
    parser.add_argument("--dps",type=int,default=80)
    parser.add_argument("--tolerance",default="1e-50")
    parser.add_argument("--symbolic-method",choices=("expanded","residues"),default="expanded")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="coefficients"},indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
