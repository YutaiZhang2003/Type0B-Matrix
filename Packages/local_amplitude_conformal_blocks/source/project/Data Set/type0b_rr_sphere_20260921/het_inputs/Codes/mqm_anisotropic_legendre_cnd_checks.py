"""Exact rational certificate: an anisotropic quartic Legendre symbol is not CND.

Every root is bracketed using Fraction arithmetic; the final strict sign
is an exact integer/rational comparison. Decimal output is descriptive.
"""
from fractions import Fraction as F
from decimal import Decimal, getcontext
from pathlib import Path
import json

getcontext().prec=75
M=(F(1,32),F(40))
POINTS=((F(4,100),F(-21)),(F(6,100),F(-12)),
        (F(-3,100),F(-2)),(F(-5,100),F(-13)),
        (F(-3,100),F(5)),(F(9,100),F(-3)))
COEFFICIENTS=(-289,498,-615,298,366,-258)


def decimal(q):
    return str(Decimal(q.numerator)/Decimal(q.denominator))


def energy_bracket(p,steps=140):
    if all(x==0 for x in p):
        return F(0),F(0)
    lo=F(0);hi=1+4*sum(x*x for x in p)
    def rhs(s):
        return 4*sum(p[k]**2/(M[k]+s)**2 for k in range(2))
    assert lo<=rhs(lo) and hi>=rhs(hi)
    for _ in range(steps):
        mid=(lo+hi)/2
        if mid<rhs(mid):lo=mid
        else:hi=mid
    assert lo<=rhs(lo) and hi>=rhs(hi)
    # This positive function decreases with s, and equals T at the root.
    def energy(s):
        w2=[p[k]**2/(M[k]+s)**2 for k in range(2)]
        return sum(M[k]*w2[k] for k in range(2))/2+3*sum(w2)**2
    return energy(hi),energy(lo)


def main():
    assert sum(COEFFICIENTS)==0
    lower=F(0);upper=F(0);second_moment_bound=F(0)
    pairs=[]
    for i in range(len(POINTS)):
        for j in range(i):
            p=tuple(POINTS[i][k]-POINTS[j][k] for k in range(2))
            tl,tu=energy_bracket(p)
            factor=2*COEFFICIENTS[i]*COEFFICIENTS[j]
            lower+=factor*(tl if factor>0 else tu)
            upper+=factor*(tu if factor>0 else tl)
            second_moment_bound+=abs(factor)*tu*tu
            pairs.append({"pair":[j,i],"T_lower":decimal(tl),"T_upper":decimal(tu)})
    assert lower>82 and upper<83
    assert upper-lower<F(1,10**32)

    # e^-x=1-x+R with 0<=R<=x²/2 for x>=0.
    time=F(1,10**8)
    schoenberg_upper=-time*lower+time*time*second_moment_bound/2
    assert schoenberg_upper<0

    # Hess T <= M^-1. Gaussian momentum variance h/2 gives
    # 0 <= AW_h T-T <= h tr(M^-1)/4 at every momentum.
    hbar=F(1,10**6)
    error_bound=hbar*sum(1/m for m in M)/4*sum(abs(a) for a in COEFFICIENTS)**2
    aw_lower=lower-error_bound
    assert aw_lower>0

    out={
      "M":[str(m) for m in M],"quartic_coefficient":"1",
      "points":[[str(x) for x in p] for p in POINTS],
      "zero_sum_integer_coefficients":list(COEFFICIENTS),
      "rational_bisections_per_pair":140,"pair_intervals":pairs,
      "CND_quadratic_form_lower":decimal(lower),
      "CND_quadratic_form_upper":decimal(upper),
      "certificate_interval_width":decimal(upper-lower),
      "Schoenberg_time":str(time),
      "Schoenberg_quadratic_form_upper_bound":decimal(schoenberg_upper),
      "AW_hbar":str(hbar),"AW_CND_quadratic_form_lower_bound":decimal(aw_lower),
      "status":"Exact rational strict-sign certificates pass.",
      "scope":"Disproves the universal anisotropic CND claim; does not determine every actual MQM fibre."
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_anisotropic_legendre_cnd_results.json').write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="pair_intervals"},indent=2))


if __name__=="__main__":
    main()
