"""Exact bounded tests prompted by the independent review; no amplitude inputs."""
import json
from pathlib import Path
import sympy as s

checks = []
def check(name, value):
    ok = bool(value)
    checks.append({"name": name, "passed": ok})
    if not ok:
        raise AssertionError(name)

rows = []
for p in range(12):
    for q in range(12-p):
        mu = [2]*p + [1]*q + [0]*(11-p-q)
        cas = sum(w*(w+23-2*(i+1)) for i,w in enumerate(mu))
        closed = 2*p*(24-p)+q*(23-2*p-q)
        if q % 2 == 0:
            L, ell = 12-p-q//2, q//2
        else:
            L = (q+1)//2
            ell = 12-p-L
        K = L*L+ell*(ell+1)
        check(f"Casimir/formula/{p}/{q}", cas == closed)
        check(f"Casimir/radial/{p}/{q}", cas == 2*(144-K))
        check(f"energy/{p}/{q}", L+ell+2 == 14-p)
        rows.append({"p":p,"q":q,"L":L,"ell":ell,"C":cas})
check("78_flavors", len(rows)==78)

for coupling in [s.Rational(0),s.Rational(1,10),s.Rational(2,13),s.Rational(1,5),s.Rational(3)]:
    es = [14-r['p']+coupling*r['C']/4 for r in rows]
    expected = min(s.Integer(14),3+s.Rational(143,2)*coupling)
    check(f"stable/envelope/{coupling}", min(es)==expected)
    selected = [(r['p'],r['q']) for r,e in zip(rows,es) if e==min(es)]
    wanted = [(11,0)] if coupling < s.Rational(2,13) else ([(0,0)] if coupling > s.Rational(2,13) else [(0,0),(11,0)])
    check(f"stable/minimizers/{coupling}", sorted(selected)==sorted(wanted))

z = s.symbols('z', real=True)
for r in rows:
    K = r['L']**2+r['ell']*(r['ell']+1)
    check(f"cutoff/envelope/{r['p']}/{r['q']}", s.expand(s.Rational(K,2)+z*r['C']/4-(72*z+(1-z)*K/2))==0)
check("cutoff/vector-gap", s.expand((s.Rational(133,2)+z*s.Rational(22,4))-72-s.Rational(11,2)*(z-1))==0)

x,y1,y2,M,g,y = s.symbols('x y1 y2 M g y', real=True)
J = s.Matrix([[0,1],[-1,0]])
Y = s.Matrix([[y1,y2],[y2,-y1]])/s.sqrt(2)
comm = x*J*Y-Y*x*J
check("fast/commutator_norm", s.expand(s.trace(comm*comm)-4*x*x*(y1*y1+y2*y2))==0)
nu=s.sqrt(M*M+4*g*g*x*x)
check("fast/log_derivative", s.simplify(s.diff(nu,x)/nu-4*g*g*x/(M*M+4*g*g*x*x))==0)
for angular in range(-6,7):
    n=12-2*angular
    check(f"fast/Gauss/{angular}", 0<=n<=24 and n%2==0)
    a=abs(angular)+1
    # r^2 has Gamma(shape=a, rate=nu) in the normalized lowest radial state.
    mean=s.gamma(a+1)/s.gamma(a)
    second=s.gamma(a+2)/s.gamma(a)
    check(f"fast/Born_Huang/{angular}", s.simplify(second-mean**2-a)==0)
for nu0, yx in [(s.Rational(3),s.Rational(1)),(s.Rational(3),s.Rational(2)),(s.Rational(3),s.Rational(-2)),(s.Rational(3),s.Rational(3,2))]:
    energies={m:(abs(m)+1)*nu0-2*yx*m for m in range(-6,7)}
    selected=[m for m,e in energies.items() if e==min(energies.values())]
    wanted=[0] if 2*abs(yx)<nu0 else ([6*s.sign(yx)] if 2*abs(yx)>nu0 else list(range(7)))
    check(f"fast/frozen_ground/{nu0}/{yx}", selected==wanted)

q0,V=s.symbols('q0 V', positive=True)
a=2*q0/(q0*q0+V)
check("clock/maximum_stationary", s.simplify(s.diff(a,q0).subs(q0,s.sqrt(V)))==0)
check("clock/maximum_value", s.simplify(a.subs(q0,s.sqrt(V))-1/s.sqrt(V))==0)
check("clock/global_bound", s.factor(1/V-a*a)==(q0*q0-V)**2/(V*(q0*q0+V)**2))
k=s.symbols('k', integer=True, positive=True)
m=s.symbols('m', integer=True)
check("clock/pair_frequency_weight", s.simplify(s.summation(m*(k-m),(m,1,k-1))-k*(k*k-1)/6)==0)

result={"scope":"Exact algebra for N4 Casimir diagnostic, rank2 frozen Gaussian matrix band, and positive bare-clock normalization. No continuum-state or amplitude claim.","passed":sum(c['passed'] for c in checks),"total":len(checks),"checks":checks}
path=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_external_review_routes_results.json')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({"passed":result['passed'],"total":result['total'],"output":str(path)}))
