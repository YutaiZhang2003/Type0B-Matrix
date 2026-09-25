"""Bounded algebra/precision report. Passing is NOT five-point crossing."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import sympy as sp

from so7e8_fivepoint_blocks import double_virasoro_fivepoint,direct_fivepoint_oracle
from so7e8_fivepoint_uniformization import fivepoint_b1_series,to_nome_series


def exact_C_22():
    """One nontrivial level-(2,2) coefficient with exact algebraic inputs."""
    from ramond_algebra.ramond_sca import gram_matrix,ground_state
    from ramond_algebra.nrr_three_point_tensor import rr_three_point_from_ground_tensor,hjs_to_polynomial_ground_tensor
    c=sp.Rational(27,2)
    a,b=sp.Rational(61,100),sp.Rational(83,100)
    p1,p2,p3,p4,p5=[sp.Rational(x,100) for x in (21,32,43,54,65)]
    hr=lambda p:c/24+p*p/2
    hn=lambda p:sp.Rational(1,2)+p*p/2
    beta=lambda p:sp.I*p/sp.sqrt(2)
    def basis(p):
        states,g=gram_matrix(4,h=hr(p),c=c)
        indices=[i for i,s in enumerate(states) if s.parity==0]
        return [states[i] for i in indices],g.extract(indices,indices).inv()
    aa,ia=basis(a)
    bb,ib=basis(b)
    zero=ground_state(0)
    def tensor(out,inn,pout,pin,pns,sign):
        return rr_three_point_from_ground_tensor(out,(),inn,h_infinity=hr(pout),h_ns=hn(pns),h_zero=hr(pin),c=c,
            ground_tensor=hjs_to_polynomial_ground_tensor(beta(pout),beta(pin),structure_sign=sign),simplify=False)
    right=sp.Matrix([tensor(s,zero,a,p1,p2,-1) for s in aa])
    middle=sp.Matrix([[tensor(s,v,b,a,p3,1) for v in aa] for s in bb])
    left=sp.Matrix([[tensor(zero,s,p5,b,p4,-1) for s in bb]])
    return sp.simplify((left*ib*middle*ia*right)[0])


def run():
    started=time.perf_counter()
    moments=(.21,.32,.43,.54,.65)
    base=dict(internal_momenta=(.61,.83),external_momenta=moments,maximum_twice_levels=4)
    worst=0.
    count=0
    cases=0
    for channel,b,pa in product('ABC',(.83,1.11),product((0,1),repeat=2)):
        signs=((1,1,s) for s in (-1,1)) if channel=='A' else (
            (s,t,1) for s,t in product((-1,1),repeat=2)) if channel=='B' else product((-1,1),repeat=3)
        for sign in signs:
            options=dict(**base,channel=channel,b=b,edge_parities=pa,structure_signs=sign)
            direct=direct_fivepoint_oracle(**options)
            result=double_virasoro_fivepoint(**options)
            for k,v in result.coefficients.items():
                worst=max(worst,abs(v-direct.coefficients[k])/max(1.,abs(v),abs(direct.coefficients[k])))
                count+=1
            cases+=1
    finite=[]
    exact=exact_C_22()
    for channel in 'ABC':
        previous=None
        for n in (32,48):
            options=dict(**base,channel=channel,edge_parities=(0,0),structure_signs=(-1,1,-1))
            result,diag=fivepoint_b1_series(**options,samples=n)
            oracle=direct_fivepoint_oracle(**options,b=1.)
            error=max(abs(v-oracle.coefficients[k])/max(1.,abs(v),abs(oracle.coefficients[k])) for k,v in result.coefficients.items())
            recovered=to_nome_series(result).inverse_coefficients()
            roundtrip=max(abs(v-result.coefficients.get(k,0))/max(1.,abs(v)) for k,v in recovered.items())
            finite.append(dict(channel=channel,samples=n,maximum_pbw_defect=error,
                maximum_radius_defect=max(d.absolute_error/max(1.,abs(d.value),abs(d.check_value)) for d in diag.values()),
                change_from_previous=None if previous is None else max(abs(v-previous[k])/max(1.,abs(v),abs(previous[k])) for k,v in result.coefficients.items()),
                lambda_roundtrip_defect=roundtrip,
                exact_70_digit_reference_defect=abs(result.coefficients[4,4]-complex(sp.N(exact,70))) if channel=='C' else None))
            previous=result.coefficients
            print('preflight',channel,n,'PBW',error,flush=True)
    files=(Path(__file__).name,'so7e8_fivepoint_blocks.py','so7e8_fivepoint_uniformization.py','virasoro_fivepoint_c_recursion.py')
    return dict(status='passed' if worst<1e-9 and all(r['maximum_pbw_defect']<1e-9 and r['maximum_radius_defect']<1e-7 and r['lambda_roundtrip_defect']<1e-9 for r in finite) else 'failed',
        generic_pbw_cases=cases,generic_pbw_coefficients=count,maximum_generic_pbw_defect=worst,
        physical_PBW_ceiling=[2,2],exact_C_22=str(exact),exact_C_22_70_digits=str(sp.N(exact,70)),
        finite_part=finite,seconds=time.perf_counter()-started,
        sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in files},
        higher_precision_scope='one exact rational mixed level-(2,2) PBW reference evaluated at 70 digits; production branch sums remain binary64',
        full_correlator_crossing_test=False,physical_amplitude=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status'],flush=True)
