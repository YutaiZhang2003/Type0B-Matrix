"""Independent exact combinatorial checks for the continuum m-leg bound."""
from pathlib import Path
from itertools import combinations
import json
import sympy as s

checks=[]


def equal(name,a,b=0):
    residual=s.expand(a-b)
    if residual != 0:
        raise AssertionError((name,residual))
    checks.append(name)


def compositions(total,parts):
    if parts==1:
        if total>=1:
            yield (total,)
        return
    for first in range(1,total-parts+2):
        for rest in compositions(total-first,parts-1):
            yield (first,)+rest


for m in range(2,11):
    ratios=[]
    for r in range(1,m):
        q=m-r
        for alpha in compositions(m-1,r):
            ratio=s.Rational(s.prod(alpha),s.factorial(q))
            if not 0<ratio<=1:
                raise AssertionError((m,r,alpha,ratio))
            ratios.append(ratio)
    checks.append(f"all coefficient ratios bounded at degree {m} ({len(ratios)} compositions)")

for n in range(1,5):
    frequencies=s.symbols(f"w0:{n}",positive=True)
    E=sum(frequencies)
    for m in range(2,8):
        P=0
        for r in range(1,min(m-1,n)+1):
            q=m-r
            for subset in combinations(frequencies,r):
                P+=s.prod(subset)*sum(subset)**(q-1)/(s.factorial(q)*s.factorial(q-1))
        polynomial=s.Poly(s.expand(P),*frequencies)
        for powers,coefficient in polynomial.terms():
            alpha=[power for power in powers if power]
            q=m-len(alpha)
            predicted=1/(s.factorial(q)*s.prod(s.factorial(power-1) for power in alpha))
            if s.simplify(coefficient-predicted)!=0:
                raise AssertionError((n,m,powers,coefficient,predicted))
        checks.append(f"general row-polynomial coefficients n{n} m{m}")
        margin=s.Poly(s.expand(E**(m-1)/s.factorial(m-1)-P),*frequencies)
        if any(coefficient<0 for coefficient in margin.coeffs()):
            raise AssertionError(("negative coefficient in bound",n,m))
        checks.append(f"coefficientwise Schur bound n{n} m{m}")
        if m==4:
            equal(f"quartic sharpened identity n{n}",P,E**3/6-sum(w**3 for w in frequencies)/12)

for n in range(1,7):
    ks=s.symbols(f"k0:{n}",positive=True)
    K=sum(ks)
    lattice=sum(k*(k-1)*(k-2)/12 for k in ks)
    lattice+=sum(a*b*(a+b-1)/2 for a,b in combinations(ks,2))
    lattice+=sum(a*b*c for a,b,c in combinations(ks,3))
    equal(f"quartic lattice Schur polynomial n{n}",lattice,
          (2*K**3-3*K**2+2*K-sum(k**3 for k in ks))/12)

a,b,c=s.symbols("a b c",real=True)
ks=(a,b,c,-a-b-c)
equal("all-singlet quartic quadratic-term rewrite",
      -sum(k*k for k in ks)/2,sum(x*y for x,y in combinations(ks,2)))

result={
    "passed":True,"checks":len(checks),"named_checks":checks,
    "bound":"||G_m||_(H0<=E) <= H_m(E) d^(m/2) E^(m-1)/(m-1)! in the declared Lebesgue convention.",
    "scope":"Finite-degree tensor estimate; no all-degree growth or infinite-series convergence is claimed."
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_continuum_polynomial_results.json').write_text(
    json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
