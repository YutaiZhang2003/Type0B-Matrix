"""Exact checks of the full coupling Hamiltonian and its norm majorant.

The continuum operator proof is analytic, in the companion memo.
These checks audit its contraction identity, local coefficient
reductions, positive majorant, and right-ordered quantum convention.
"""
from pathlib import Path
import json
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
checks = []

def equal(name, a, b=0):
    diff = s.factor(s.cancel(a-b))
    if diff != 0:
        raise AssertionError((name, diff))
    checks.append(name)

def truth(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

g, T, Tp, K, Kp, Kg, Kgp, qp, J0 = s.symbols(
    "g T Tp K Kp Kg Kgp qp J0", nonzero=True)
eta = Kg/(g*g*T)
etap = Kgp/(g*g*T)-Kg*Tp/(g*g*T*T)
coefficient = (eta-etap-eta*Tp/T-qp*J0*Kp/(K*T))/(g*g)
equal("exact pulled-back symplectic coefficient",
      coefficient.subs({Kp:K-g*g*T, Kgp:Kg-2*g*T-g*g*qp*J0}),
      2/g**3+qp*J0/K)

u,j2 = s.symbols("u j2")
raw = 2*(K-1-(1+g*u/s.sqrt(2))*s.log(K))/g**3
regular = j2*s.log(K)/(2*g)+(2*(K-1)-(1+K)*s.log(K))/g**3
equal("regular form differs by explicit total derivative",
      (raw-regular).subs(u,(K-Kp-1-g*g*j2/2)/(s.sqrt(2)*g)),
      Kp*s.log(K)/g**3)

# Differential-polynomial tests of B0=G3, B1=2G4, and the compact B2.
sig = s.symbols("s0:8")
aux = s.symbols("b0:8")
jet_sets = (sig,aux)
def deriv(expr):
    return s.expand(sum(s.diff(expr, jets[k])*jets[k+1]
                        for jets in jet_sets for k in range(len(jets)-1)))
def euler(expr,jets):
    answer=0
    for k,j in enumerate(jets):
        piece=s.diff(expr,j)
        for _ in range(k):
            piece=-deriv(piece)
        answer+=piece
    return s.expand(answer)

F=aux[0]-aux[2]
rf=aux[0]+aux[1]
a=s.sqrt(2)*sig[0]
b=(sig[0]**2+rf)/2
jsq=sig[0]**2-2*sig[0]*sig[1]+F
B0=jsq*a/2-a**3/6
B1=jsq*(b-a*a/2)/2-a*a*b/2+a**4/6
B2=jsq*(a**3/3-a*b)/2-a*b*b/2+2*a**3*b/3-3*a**5/20
G3=s.sqrt(2)/2*(sig[0]**3/3+sig[0]*F)
G4=F*aux[0]/8-sig[0]**2*F/4-sig[0]**4/24
B2compact=s.sqrt(2)*(sig[0]**5/40+sig[0]**3*F/4
                     -sig[0]*F*rf/4-sig[0]*rf*rf/8)
for label, density in (("B0-G3",B0-G3),("B1-2G4",B1-2*G4),
                        ("B2 compact reduction",B2-B2compact)):
    for field,jets in (("singlet",sig),("auxiliary",aux)):
        equal(f"Euler identity {label} {field}",euler(density,jets))

# Exact coefficients of the scalar absolute majorant.
z,W = s.symbols("z W")
w=s.sqrt(2)*z+z*z/2
cutoff=14
signed=s.series(2*W-(2+W)*s.log(1+W),W,0,cutoff+1).removeO().expand()
for n in range(3,cutoff+1):
    equal(f"exact signed coefficient n{n}",signed.coeff(W,n),
          s.Rational((-1)**n*(n-2),n*(n-1)))
L=sum(w**n/s.Integer(n) for n in range(1,cutoff+1))
C=s.Poly(s.expand(z*z*L/2+
      sum(s.Rational(n-2,n*(n-1))*w**n for n in range(3,cutoff+1))),z)
coeffs={m:s.simplify(C.coeff_monomial(z**m)) for m in range(cutoff+1)}
for m in range(3):
    equal(f"majorant has no degree {m}",coeffs[m])
for m in range(3,cutoff+1):
    truth(f"positive coefficient c{m}",coeffs[m].is_positive)
equal("cubic majorant coefficient",coeffs[3],5*s.sqrt(2)/6)
rho=2-s.sqrt(2)
equal("positive convergence radius solves w=1",w.subs(z,rho),1)
truth("convergence radius positive",rho.is_positive)

# The symmetric tensor factorial cancels the Schur factorial.
for m in range(3,cutoff+1):
    equal(f"tensor-to-operator factorial m{m}",
          s.factorial(m)*coeffs[m]/s.factorial(m-1),m*coeffs[m])

# Independent finite-matrix check of the RIGHT coupling ordering sign.
sx=s.Matrix([[0,1],[1,0]])
sy=s.Matrix([[0,-s.I],[s.I,0]])
sz=s.diag(1,-1)
A0,A1,A2=sx,sz,sy
Bsecond=3*A2+s.I*(A0*A1-A1*A0)/2
U0=s.eye(2)
U1=-s.I*A0
U2=-s.I*A1-A0*A0/2
U3=-s.I*A2-(A0*A1+A1*A0)/2+s.I*A0**3/6
for degree, left, right in (
    (0,U1,-s.I*A0),
    (1,2*U2,-s.I*(2*A1+U1*A0)),
    (2,3*U3,-s.I*(Bsecond+2*U1*A1+U2*A0)),
):
    for i in range(2):
        for j in range(2):
            equal(f"right-ordered coupling equation g{degree} entry{i}{j}",
                  left[i,j],right[i,j])

result={
    "passed":True,
    "checks":len(checks),
    "named_checks":checks,
    "majorant_coefficients":{str(m):str(coeffs[m]) for m in range(3,cutoff+1)},
    "proven_window":"|g| E sqrt(d/(2pi)) < 2-sqrt(2)",
    "scope":"Exact classical coupling Hamiltonian and analytic quantum series on its bounded-energy window; no arbitrary-energy or heterotic quantum identification.",
}
(ROOT/'data_exports/mqm/mqm_clock_coupling_isotopy_results.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="named_checks"},indent=2))
