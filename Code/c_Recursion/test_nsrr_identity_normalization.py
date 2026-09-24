#!/usr/bin/env python3
"""Check the identity residue of the imported NSRR Upsilon coefficients.

The NS leg at alpha=epsilon is shared by the NSNSNS and NSRR vertices and
cancels in their ratio. Thus no analytic continuation of the positive-leg
helper, no fitted constant, and no plumbing block enters this test.
"""
import argparse
import json
from pathlib import Path
import time
import mpmath as mp
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants
from super_liouville_structure_constants import ns_structure_constant, rr_ns_structure_constants


def identity_ratios(engine, momentum, epsilon, separation=0):
    engine.special._set_precision()
    b=mp.mpf(engine.b); Q=b+1/b
    p=mp.mpf(momentum); eps=mp.mpf(epsilon); p2=p+eps*separation
    def upsilon(x):
        # Shift away from the edge of the convergence strip before quadrature.
        if mp.re(x)>Q/2:
            x=Q-x
        step=min(b,1/b)
        if mp.re(x)<step/2:
            if b<=1:
                shift=(mp.loggamma(b*x)-mp.loggamma(1-b*x)+(1-2*b*x)*mp.log(b))
            else:
                shift=(mp.loggamma(x/b)-mp.loggamma(1-x/b)+(2*x/b-1)*mp.log(b))
            return mp.exp(engine.special.log_upsilon(x+step)-shift)
        return engine.special.upsilon(x)
    def ns(x):return upsilon(x/2)*upsilon((x+Q)/2)
    def rr(x):return upsilon((x+b)/2)*upsilon((x+1/b)/2)
    x0=eps+1j*(p+p2); x3=Q-eps+1j*(p+p2)
    x1=eps+1j*(p2-p); x2=eps-1j*(p2-p)
    metric=(engine._normalized_leg(p,"R")*engine._normalized_leg(p2,"R") /
            (b*b*engine._normalized_leg(p,"NS")*engine._normalized_leg(p2,"NS")))
    return metric*ns(x0)*ns(x3)/(rr(x0)*rr(x3)),metric*ns(x1)*ns(x2)/(rr(x1)*rr(x2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter();rows=[];b1_error=0
    for b in (.8,1.,1.4):
        engine=GenericSuperLiouvilleConstants(b,dps=40)
        for p in ("0.23","0.67"):
            for eps in ("0.01","0.0001","0.000001"):
                for offset in ((-1,0,1) if eps=="0.000001" else (0,)):
                    even,odd=identity_ratios(engine,p,eps,offset)
                    current=(even+odd,even-odd)
                    row=dict(b=b,p=p,epsilon=eps,separation_in_epsilon=offset,
                        imported_E_over_NS=[mp.nstr(mp.re(even),35),mp.nstr(mp.im(even),35)],
                        imported_O_over_NS=[mp.nstr(mp.re(odd),35),mp.nstr(mp.im(odd),35)],
                        old_family_to_NS=[float(mp.re(x/2)) for x in current],
                        unit_identity_family_to_NS=[float(mp.re(x)) for x in current])
                    rows.append(row)
                    if eps=="0.000001":
                        assert max(abs(x-1) for x in current)<mp.mpf("0.00001"),row
                    if b==1 and eps=="0.01":
                        q=1j*(1-float(eps)); c=ns_structure_constant(float(p),float(p),q,50)
                        e,o=rr_ns_structure_constants(float(p),float(p),q,50)
                        b1_error=max(b1_error,float(abs(even-e/c)),float(abs(odd-o/c)))
        print(f"identity-residue checks at b={b} passed",flush=True)
    assert b1_error<1e-12,b1_error
    result=dict(passed=True,working_digits=40,rows=rows,
        b1_independent_BarnesG_maximum_absolute_error=b1_error,
        old_limit="(E+/-O)/(2 C_NS) -> 1/2",
        corrected_limit="(E+/-O)/C_NS -> 1",
        derivation="shared NS external leg cancels; Upsilon_s(Q-x)=Upsilon_s(x); relative leg metrics cancel exactly",
        seconds=time.perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ('passed','b1_independent_BarnesG_maximum_absolute_error','seconds')}))


if __name__=="__main__":main()
