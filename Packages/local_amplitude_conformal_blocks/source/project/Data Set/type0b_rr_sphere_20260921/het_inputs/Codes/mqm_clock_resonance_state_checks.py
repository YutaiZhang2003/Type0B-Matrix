"""Exact checks of the linear clock-mirror resonance-state audit.

These verify a quadratic boundary-mode prescription and its precise
noncommuting-source limitation, not a nonlinear quantum mirror dual.
"""
from pathlib import Path
import json
import sympy as s

checks = []


def equal(name, a, b=0):
    residual = s.simplify(s.expand_trig(a-b))
    if residual != 0:
        residual = s.simplify(residual.rewrite(s.exp))
    if residual != 0:
        raise AssertionError((name, residual))
    checks.append(name)


x, y, w, k = s.symbols("x y omega k", positive=True)
f = s.cos(w*x)+w*s.sin(w*x)
b = s.exp(-x)
equal("real-frequency bulk equation", -s.diff(f,x,2), w*w*f)
equal("real-frequency boundary equation", s.diff(f,x).subs(x,0), w*w*f.subs(x,0))
equal("bound spatial eigenvalue", -s.diff(b,x,2), -b)
equal("bound spatial boundary eigenvalue", s.diff(b,x).subs(x,0), -b.subs(x,0))
equal("bound kinetic bilinear", s.integrate(b*b,(x,0,s.oo))-1, -s.Rational(1,2))
equal("continuum-bound kinetic orthogonality",
      s.integrate(b*f,(x,0,s.oo))-1)
equal("cross principal-value cancellation",
      k*k/(k*k-w*w)+w*w/(w*w-k*k),1)
fy=s.cos(w*y)+w*s.sin(w*y)
equal("continuum completeness integrand decomposition",
      f*fy/(1+w*w),
      s.sin(w*x)*s.sin(w*y)+
      (s.cos(w*(x+y))+w*s.sin(w*(x+y)))/(1+w*w))
r=(1-s.I*w)/(1+s.I*w)
equal("unit-modulus real-frequency reflection", r*s.conjugate(r),1)

A,P,q,p,T=s.symbols("A P q p T",real=True)
qof=(A-4*P)/(2*s.sqrt(2)); pof=(A+4*P)/(2*s.sqrt(2))
equal("canonical dilation bracket",
      s.diff(qof,A)*s.diff(pof,P)-s.diff(qof,P)*s.diff(pof,A),1)
equal("bound Hamiltonian becomes dilation", qof*pof, -2*P*P+A*A/8)
At=s.sqrt(2)*(s.exp(T)*q+s.exp(-T)*p)
Pt=-s.diff(At,T)/4
equal("bound canonical evolution preserves bracket",
      s.diff(At,q)*s.diff(Pt,p)-s.diff(At,p)*s.diff(Pt,q),1)
equal("bound mode equation", s.diff(At,T,2),At)
equal("bound piece cancels continuum CCR defect", s.exp(-x)*(-2*s.exp(-y)),
      -2*s.exp(-x-y))

t=s.symbols("t",positive=True)
Gp=-s.exp(-t)/2
Gm=-s.exp(t)/2
equal("bounded Green equation on positive half-line",s.diff(Gp,t,2)-Gp)
equal("bounded Green equation on negative half-line",s.diff(Gm,t,2)-Gm)
equal("bounded Green derivative jump",s.diff(Gp,t).subs(t,0)-s.diff(Gm,t).subs(t,0),1)
equal("bounded is not half-retarded-plus-advanced",
      Gp-s.sinh(t)/2,-s.cosh(t)/2)

t1,t2,a0,a1=s.symbols("t1 t2 a0 a1",real=True)
alpha=s.sqrt(2)*(a0*s.exp(t1)+a1*s.exp(t2))
beta=s.sqrt(2)*(a0*s.exp(-t1)+a1*s.exp(-t2))
magnus=-2*a0*a1*s.sinh(t2-t1)
equal("two c-number pulses give bounded phase",
      s.expand_trig(alpha*beta/2+magnus).rewrite(s.exp),
      a0*a0+a1*a1+2*a0*a1*s.exp(t1-t2))
comm=2*(s.exp(t2-t1)-s.exp(t1-t2))
equal("A commutator coefficient", comm,4*s.sinh(t2-t1))

theta=s.symbols("theta",real=True)
Id=s.eye(2)
sx=s.Matrix([[0,1],[1,0]])
sy=s.Matrix([[0,-s.I],[s.I,0]])
sz=s.diag(1,-1)
K=s.zeros(2)
for u in (-1,1):
    for v in (-1,1):
        K+=s.exp(s.I*theta*u*v)*(Id+u*sx)*(Id+v*sz)/4
pred=s.cos(theta)*Id+s.sin(theta)*sy
res=s.simplify(s.expand_complex(K-pred))
equal("exact ordered Pauli-source resonance matrix",sum(abs(z)**2 for z in res))
res=s.simplify(s.expand_complex(K.conjugate().T*K-(Id+s.sin(2*theta)*sy)))
equal("exact nonunitarity matrix",sum(abs(z)**2 for z in res))
equal("Pauli-source rank collapse at pi over four",pred.subs(theta,s.pi/4).det())
equal("Pauli-source nonzero trace at rank collapse",pred.subs(theta,s.pi/4).trace(),s.sqrt(2))
equal("Pauli-source transition is identity at zero pulse strength",pred.subs(theta,0).trace(),2)

result={
    "passed":True,
    "checks":len(checks),
    "named_checks":checks,
    "scoped_result":"The bounded resonance prescription is a pure phase for c-number or commuting sources, but is not automatically unitary for noncommuting matter sources.",
    "not_claimed":"No failure of the actual mirror stress coupling or no-go theorem for a nonlinear continuum completion is established."
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_resonance_state_results.json').write_text(
    json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
