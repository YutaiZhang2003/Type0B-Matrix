"""Production assembly using CCY blocks and direct fermion sewing.

The physical PBW reference is deliberately absent from this module.
The enlarged block uses recursively computed primary coefficients and
punctured CCY descendants. The selected auxiliary backend uses the fermion
definition. The earlier, level-five Ising backend remains explicitly selectable.
Universal large-c factors are restored after the convolution division.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import mpmath as mp

from outer_branching import OuterBranching, middle_pairs, ramond_level, ns_labels
from middle_branching import MiddleBranching
from punctured_ccy import PuncturedCCY, downward_indices, theta_vacuum_seed
from series_algebra import ZERO, scalar_product, recover_minus, theta_sign
from compute_target import ns_norm_squared, ramond_norm_squared

HERE = Path(__file__).resolve().parent


def numeric(value):
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator)/value.denominator
    return mp.mpc(value)


def encoded(value):
    value = numeric(value)
    return {"real": mp.nstr(value.real, mp.mp.dps),
            "imag": mp.nstr(value.imag, mp.mp.dps)}


def encode_series(series):
    return [{"exponents": key, "values": [encoded(value) for value in vector]}
            for key, vector in sorted(series.items(), key=lambda item: (sum(item[0]), item[0]))]


def source_hashes():
    files = list(HERE.glob("*.py")) + [
        HERE.parent/"full_ramond_block_runtime"/"compute_full_block.py",
        HERE.parent/"ramond_branching_recursion"/"compute_target.py",
        HERE.parent/"ramond_branching_recursion"/"direct_state_check.py",
    ]
    return {str(path.relative_to(HERE.parent.parent)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def weights(b, momentum, n):
    q = b+1/b
    return ((q*q/4-(momentum+2*numeric(n)*b)**2)/(2*(1-b*b)),
            (q*q/4-(momentum+2*numeric(n)/b)**2)/(2*(1-1/(b*b))))


def lift_scalar(series):
    return {(2*a,b,c,2*d): value for (a,b,c,d), value in series.items()}


def multiply_scalar_parity(scalar, parity, cutoff):
    answer = {}
    for a, factor in scalar.items():
        for b, vector in parity.items():
            if sum(a)+sum(b) > cutoff:
                continue
            key = tuple(u+v for u,v in zip(a,b))
            previous = answer.setdefault(key, [mp.mpc(0)]*8)
            for j,value in enumerate(vector):
                previous[j] += factor*value
    return {key:tuple(value) for key,value in answer.items()}


def reduced_numerator(b, momenta, level, *, f=0, p=0, etas=(1,-1),
                      dps=60, ward_dps=0, progress=None):
    started = time.perf_counter()
    if progress:
        progress({"stage": "outer branching", "status": "starting", "level": level})
    outer = OuterBranching(b, momenta, level, f=f, p=p, dps=ward_dps)
    outer.prepare(etas)
    outer_seconds = time.perf_counter()-started
    if progress:
        progress({"stage": "outer branching", "status": "complete", "seconds": outer_seconds})
    # Exact rational benchmark weights are used in CCY; outer anchor precision
    # remains that of the explicitly recorded legacy primary oracle.
    b_mp = numeric(b)
    momenta_mp = tuple(map(numeric, momenta))
    middle = MiddleBranching(b_mp, momenta_mp[1], dps=dps)
    q = b_mp+1/b_mp
    central = (1+3*q*q/(1-b_mp*b_mp), 1+3*q*q/(1-1/(b_mp*b_mp)))
    external = (-(1+2*b_mp*b_mp)/(2*(1-b_mp*b_mp)),
                (b_mp*b_mp+2)/(2*(1-b_mp*b_mp)))
    answer, ccy_records = {}, []
    cases = 0
    middle_seconds = 0.0
    ccy_seconds = 0.0
    pairs = middle_pairs(level)
    r3_labels = tuple(n for n in outer.grid.r if ramond_level(n) <= level)
    for n1 in ns_labels(level):
        for incoming,outgoing in pairs:
            for n3 in r3_labels:
                shift = (int(4*n1*n1), ramond_level(incoming),
                         ramond_level(outgoing), 2*ramond_level(n3))
                remaining = 2*level-sum(shift)
                if remaining < 0:
                    continue
                factors = []
                for alpha in (0,1):
                    gamma = (f-int(2*n1)-alpha) % 2
                    tick = time.perf_counter()
                    inserted = numeric(middle.raw(outgoing,incoming,alpha))
                    middle_seconds += time.perf_counter()-tick
                    denominator = (ns_norm_squared(n1,b_mp,momenta_mp[0])
                                   *ramond_norm_squared(incoming,alpha,b_mp,momenta_mp[1])
                                   *ramond_norm_squared(outgoing,alpha,b_mp,momenta_mp[1])
                                   *ramond_norm_squared(n3,gamma,b_mp,momenta_mp[2]))
                    factor = (numeric(outer.raw(n1,incoming,n3,alpha,gamma,etas[0]))
                              *numeric(outer.raw(n1,outgoing,n3,alpha,gamma,etas[1]))
                              *inserted/denominator)
                    index = ((int(2*n1)+p)%2) | (alpha<<1) | (gamma<<2)
                    factor *= (-1)**int(2*n1)*theta_sign(index)
                    factors.append((index,factor))
                if not any(value for _,value in factors):
                    continue
                indices = downward_indices(remaining,degree_weights=(2,1,1,2))
                edge_weights = (weights(b_mp,momenta_mp[0],n1),
                                weights(b_mp,momenta_mp[1],incoming),
                                weights(b_mp,momenta_mp[1],outgoing),
                                weights(b_mp,momenta_mp[2],n3))
                tick = time.perf_counter()
                blocks = []
                for copy in (0,1):
                    engine = PuncturedCCY(central[copy],
                                         tuple(pair[copy] for pair in edge_weights),
                                         external[copy],dps=dps)
                    blocks.append(lift_scalar(engine.reduced_series(indices=indices)))
                    ccy_records.append({"labels":list(map(str,(n1,incoming,outgoing,n3))),
                                        "copy":copy+1,"cache":engine.cache_info()})
                    engine.clear_caches()
                product = scalar_product(blocks[0],blocks[1],remaining)
                ccy_seconds += time.perf_counter()-tick
                for descendant,coefficient in product.items():
                    key = tuple(a+b for a,b in zip(shift,descendant))
                    vector = answer.setdefault(key,[mp.mpc(0)]*8)
                    for index,factor in factors:
                        vector[index] += factor*coefficient
                cases += 1
                if progress and (cases == 1 or cases % 20 == 0):
                    progress({"stage":"numerator", "branches":cases,
                              "labels":list(map(str,(n1,incoming,outgoing,n3))),
                              "seconds":time.perf_counter()-started})
    diagnostics = {"outer":outer.diagnostics,"outer_seconds":outer_seconds,
                   "middle_seconds":middle_seconds,"ccy_seconds":ccy_seconds,
                   "branch_cases":cases,"ccy":ccy_records,
                   "middle_recursion_count":len(middle.recursion_records)}
    return {key:tuple(vector) for key,vector in answer.items()}, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level",type=int,default=5)
    parser.add_argument("--dps",type=int,default=100)
    parser.add_argument("--ward-dps",type=int,default=70)
    parser.add_argument("--p",type=int,choices=(0,1),default=0)
    parser.add_argument("--f",type=int,choices=(0,1),default=0)
    parser.add_argument("--eta",type=int,choices=(-1,1),default=1)
    parser.add_argument("--auxiliary", choices=("direct", "ising"), default="direct",
                        help="The direct fermion definition is the user-selected production backend")
    parser.add_argument("--ising-regulator",choices=("production","linear","kac_centered","level5_null_cycles"),
                        default="production",
                        help="Non-production choices explicitly diagnose an unproved Ising limit in the requested end-to-end check")
    parser.add_argument("--json",type=Path,required=True)
    args = parser.parse_args()
    if args.level > 5 and args.auxiliary == "ising":
        parser.error("The c-recursive Ising quotient is established only through level 5; select the direct auxiliary backend for higher levels")
    mp.mp.dps = args.dps
    b = Fraction(7,5)
    momenta = tuple(map(Fraction,("11/23","13/29","17/31")))
    started = time.perf_counter()
    report = {"status":"running", "total_q_level":args.level,
              "b":str(b),"momenta":list(map(str,momenta)),
              "p":args.p,"f":args.f,"etas":[args.eta,-args.eta],
              "auxiliary_backend": args.auxiliary,
              "ccy_dps":args.dps,"ward_dps":args.ward_dps,
              "source_hashes":source_hashes()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    report["reduced_numerator"],diagnostics = None,None
    def progress(value):
        print(json.dumps(value),flush=True)
    numerator,diagnostics = reduced_numerator(
        b,momenta,args.level,f=args.f,p=args.p,etas=(args.eta,-args.eta),
        dps=args.dps,ward_dps=args.ward_dps,progress=progress)
    report["reduced_numerator"] = encode_series(numerator)
    report["numerator_diagnostics"] = diagnostics
    args.json.write_text(json.dumps(report,indent=2,default=str)+"\n")
    tick = time.perf_counter()
    if args.auxiliary == "direct":
        from direct_fermion import DirectFermion
        auxiliary_engine = DirectFermion(Q=numeric(b)+1/numeric(b), dps=args.dps)
        auxiliary = auxiliary_engine.series(args.level, progress=progress)
        auxiliary_key = "full_auxiliary"
        seed_power = 2
    else:
        from ising_fermion import IsingFermion, RegulatedIsingCCY, LevelFiveNullCycleIsingCCY
        if args.ising_regulator == "production":
            auxiliary_engine = IsingFermion(Q=numeric(b)+1/numeric(b),dps=args.dps)
        elif args.ising_regulator == "level5_null_cycles":
            auxiliary_engine = LevelFiveNullCycleIsingCCY(
                Q=numeric(b)+1/numeric(b),dps=args.dps,allow_unproved_limit=True)
        else:
            auxiliary_engine = RegulatedIsingCCY(
                Q=numeric(b)+1/numeric(b),dps=args.dps,
                regulator=args.ising_regulator,allow_unproved_limit=True)
        auxiliary = auxiliary_engine.series(args.level,reduced=True)
        auxiliary_key = "reduced_auxiliary"
        seed_power = 1
    report["auxiliary_seconds"] = time.perf_counter()-tick
    report["auxiliary_provenance"] = auxiliary_engine.diagnostics
    report[auxiliary_key] = encode_series(auxiliary)
    report["universal_seed_power_after_recovery"] = seed_power
    args.json.write_text(json.dumps(report,indent=2,default=str)+"\n")
    tick = time.perf_counter()
    try:
        quotient = recover_minus(numerator,auxiliary,2*args.level,tolerance=1e-8)
    except ArithmeticError as exc:
        report["status"] = "failed during recovery"
        report["error"] = str(exc)
        report["total_seconds"] = time.perf_counter()-started
        args.json.write_text(json.dumps(report,indent=2,default=str)+"\n")
        raise
    seed = {(2*a,b,b,2*c):numeric(value)
            for (a,b,c),value in theta_vacuum_seed(args.level).items()}
    if seed_power == 2:
        # numerator=hatF/U^2, while the direct auxiliary is the full FF.
        # Therefore the recovered quotient must be multiplied by U^2.
        seed = scalar_product(seed, seed, 2*args.level)
    physical = multiply_scalar_parity(seed,quotient,2*args.level)
    report["recovery_and_seed_seconds"] = time.perf_counter()-tick
    report["total_seconds"] = time.perf_counter()-started
    report["coefficients"] = encode_series(physical)
    report["status"] = "computed; validation pending"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_resident_memory_mib"] = peak/(1024**2 if sys.platform == "darwin" else 1024)
    report["environment"] = {"python": sys.version, "platform": platform.system(),
                             "machine": platform.machine()}
    args.json.write_text(json.dumps(report,indent=2,default=str)+"\n")
    progress({"stage":"physical block computed", "seconds":report["total_seconds"]})


if __name__ == "__main__":
    main()
