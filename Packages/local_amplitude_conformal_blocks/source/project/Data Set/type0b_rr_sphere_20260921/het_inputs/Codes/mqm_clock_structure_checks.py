"""Independent classical-clock algebra and finite-current vacuum regularity.

No reported amplitude kernels are used here. The quantum calculation tests
the fixed gap-oscillator vacuum, not every representation or completion.
"""
from pathlib import Path
import json
import math
from collections import defaultdict
import numpy as np
import sympy as s

checks = []
def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append(name)

# Exterior two-form coefficients: basis dT wedge dK.
T, K, gp, F, Fp = s.symbols('T K Gprime F Fprime', nonzero=True)
q = gp / (1-gp*(K-F))
defect = s.factor(-q+T*q*q*Fp)
check('general symplectic defect and rigidity equation',
      s.simplify(defect-q*(gp*(K-F+T*Fp)-1)/(1-gp*(K-F))) == 0)
g = s.symbols('g', positive=True)
check('linear stress source logarithmic clock is canonical',
      s.simplify(defect.subs({F:g*g*T,Fp:g*g,gp:1/K})) == 0)
zeta, w = s.symbols('zeta w', real=True)
R = (1+s.I*w*(1-zeta))/(1+s.I*w)
check('unit modulus background condition',
      s.simplify(R*s.conjugate(R)-1-zeta*(zeta-2)*w*w/(1+w*w)) == 0)

# Exact vacuum vector from the normally ordered finite oscillator polynomial.
# Occupation states are UNNORMALIZED monomials; scalar products include n!.
def vacuum_vector(N, flavors, coupling):
    out = defaultdict(lambda:s.S.Zero)
    for m in range(1,N+1):
        out[((0,m),)] += s.sqrt(2)*coupling*s.sqrt(m)/(1-s.I*m)
    for a in range(flavors):
        for m in range(1,N+1):
            for n in range(1,N+1):
                key=tuple(sorted(((a,m),(a,n))))
                out[key] += coupling**2*s.sqrt(m*n)/(2*(1-s.I*(m+n)))
    return out

def norm2(vector):
    total=s.S.Zero
    for key, value in vector.items():
        counts={mode:key.count(mode) for mode in set(key)}
        weight=math.prod(math.factorial(n) for n in counts.values())
        total += weight*value*s.conjugate(value)
    return s.simplify(s.expand_complex(total))

for N in range(1,6):
    for flavors in (1,3,24):
        actual=norm2(vacuum_vector(N,flavors,s.Rational(2,5)))
        linear=sum(s.Rational(m,1+m*m) for m in range(1,N+1))
        pair=sum(s.Rational(m*n,1+(m+n)**2)
                 for m in range(1,N+1) for n in range(1,N+1))
        predicted=2*s.Rational(2,5)**2*linear+flavors*s.Rational(2,5)**4*pair/2
        check(f'independent occupation norm N{N} flavors{flavors}',actual==predicted)

def pair_count(q,N):
    lo,hi=max(1,q-N),min(N,q-1)
    if lo>hi:return 0
    first=(hi*(hi+1)-(lo-1)*lo)//2
    second=(hi*(hi+1)*(2*hi+1)-(lo-1)*lo*(2*lo-1))//6
    return q*first-second

for N in range(1,12):
    for qq in range(2,2*N+1):
        check(f'pair shell N{N} q{qq}',
              pair_count(qq,N)==sum(m*(qq-m) for m in range(1,N+1) if 1<=qq-m<=N))

x,y=s.symbols('x y',positive=True)
integral=s.integrate(s.integrate(x*y/(x+y)**2,(y,0,1)),(x,0,1))
check('continuum leading variance integral',s.simplify(integral-s.log(2)+s.Rational(1,2))==0)
asymptotic=[]
limit=math.log(2)-.5
for N in (32,128,512,2048,8192):
    pair=sum(pair_count(qq,N)/(1+qq*qq) for qq in range(2,2*N+1))
    asymptotic.append({'N':N,'pair_sum':pair,'pair_sum_over_N2':pair/N**2})
check('pair variance leading coefficient convergence',
      abs(asymptotic[-1]['pair_sum_over_N2']-limit)<1e-4)

def smeared_variance(N,tau,coupling=.4,flavors=24):
    linear=sum(m/(1+m*m)*math.exp(-tau*tau*m*m) for m in range(1,N+1))
    pair=sum(pair_count(qq,N)/(1+qq*qq)*math.exp(-tau*tau*qq*qq)
             for qq in range(2,2*N+1))
    return 2*coupling**2*linear+flavors*coupling**4*pair/2

smearing=[]
for tau in (.1,.3,1.):
    vals=[smeared_variance(N,tau) for N in (40,80,160)]
    check(f'fixed smooth smearing converges tau{tau}',abs(vals[-1]-vals[-2])<1e-12)
    smearing.append({'tau':tau,'N40_N80_N160':vals})

result={'passed':True,'checks':len(checks),'named_checks':checks,
        'pair_variance_asymptotic':asymptotic,'coefficient':'log(2)-1/2',
        'gaussian_smearing':smearing,
        'scope':'Classical ansatz and fixed-Fock-vacuum clock regularity only; no MQM Hamiltonian or global quantum clock derived.'}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_structure_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='named_checks'},indent=2))
