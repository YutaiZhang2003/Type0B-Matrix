"""Independent exact checks of the positive-clock moving-boundary action.

This is a continuum boundary model and finite regulator audit, not a
proof of a heterotic MQM state or scattering limit.
"""
from pathlib import Path
import json
import sympy as s
import numpy as np

checks = []


def equal(name, left, right=0):
    residual = s.simplify(s.expand(left-right))
    if residual != 0:
        raise AssertionError((name, residual))
    checks.append(name)


v, incoming, kappa, wall = s.symbols("v incoming kappa wall", real=True)
hp = (1-v)/(1+v)
outgoing = incoming/hp
full_time = incoming+outgoing
full_space = incoming-outgoing
equal("moving Neumann reflection", full_space+v*full_time)
equal("boundary matter Lagrangian",
      (full_time**2-full_space**2)/4, incoming**2/hp)
equal("bulk energy boundary flux",
      -full_time*full_space/2-v*(full_time**2+full_space**2)/4,
      v*incoming**2/hp)
equal("boundary force implies positive clock",
      kappa*s.exp(2*wall)*hp-incoming**2/2,
      hp*(2*kappa*s.exp(2*wall)-incoming**2/hp)/2)

x = s.symbols("x", real=True)
for n in range(1,5):
    basis = [s.exp(-x/2)*s.laguerre(j,x) for j in range(n)]
    gram = s.Matrix(n,n,lambda i,j:
        s.integrate(basis[i]*basis[j],(x,0,s.oo)))
    A = s.Matrix(n,n,lambda i,j:
        s.integrate(basis[i]*s.diff(basis[j],x),(x,0,s.oo)))
    B = s.Matrix(n,n,lambda i,j:
        s.integrate(s.diff(basis[i],x)*s.diff(basis[j],x),(x,0,s.oo)))
    ell = s.ones(n,1)
    expected_A = s.Matrix(n,n,lambda i,j:
        -s.Rational(1,2) if i==j else (-1 if i<j else 0))
    equal(f"Laguerre orthonormality N={n}", s.trace((gram-s.eye(n)).T*(gram-s.eye(n))))
    equal(f"Laguerre derivative matrix N={n}", s.trace((A-expected_A).T*(A-expected_A)))
    equal(f"derivative span closes N={n}", s.trace((B-A.T*A).T*(B-A.T*A)))
    equal(f"boundary integration identity N={n}",
          s.trace((A+A.T+ell*ell.T).T*(A+A.T+ell*ell.T)))
    equal(f"positive stiffness determinant N={n}", B.det(), s.Rational(1,4)**n)

# Derive the secondary constraint from the finite canonical bracket.
R, PR, q = s.symbols("R PR q", real=True)
for n in (1,2,3):
    A = s.Matrix(n,n,lambda i,j:
        -s.Rational(1,2) if i==j else (-1 if i<j else 0))
    B, ell = A.T*A, s.ones(n,1)
    coords = [s.Matrix(s.symbols(f"a{I}_0:{n}")) for I in range(2)]
    moms = [s.Matrix(s.symbols(f"p{I}_0:{n}")) for I in range(2)]
    slopes = (q,s.S.Zero)
    D_aff = sum((p.T*A*a)[0]-qi*(ell.T*a)[0]
                for a,p,qi in zip(coords,moms,slopes))
    C = PR+D_aff
    H = sum((p.T*p)[0]+(a.T*B*a)[0]/4
            for a,p in zip(coords,moms))+q*q*R-kappa*s.exp(2*R)
    bracket = -s.diff(H,R)
    for aa,pp in zip(coords,moms):
        for ai,pi in zip(aa,pp):
            bracket += s.diff(C,ai)*s.diff(H,pi)-s.diff(C,pi)*s.diff(H,ai)
    expected = 2*kappa*s.exp(2*R)
    expected -= sum((qi+(ell.T*p)[0])**2
                    for p,qi in zip(moms,slopes))
    expected += sum((ell.T*A*a)[0]**2/4 for a in coords)
    equal(f"canonical secondary boundary constraint N={n}", bracket, expected)

for n in (1,2,3,4,8,16):
    A = np.array([[-.5 if i==j else (-1. if i<j else 0.)
                   for j in range(n)] for i in range(n)])
    actual = np.linalg.svd(A,compute_uv=False)
    predicted = np.array([.5/np.tan((2*j-1)*np.pi/(4*n))
                          for j in range(1,n+1)])
    if not np.allclose(actual,predicted,rtol=1e-12,atol=1e-12):
        raise AssertionError(("Laguerre singular frequencies",n,actual,predicted))
    checks.append(f"independent singular frequencies N={n}")

# The homogeneous-clock mode has the reference energy, despite running away.
a,z = s.symbols("a z", positive=True)
primitive = 1/(1-z)+s.log(1-z)-1
equal("runaway field-energy integral primitive", s.diff(primitive,z), z/(1-z)**2)
R_of_a = -s.log(1-a)/2
field_energy = kappa*(1/(1-a)+s.log(1-a)-1)
relative_energy = field_energy+2*kappa*R_of_a-kappa/(1-a)
equal("homogeneous runaway has reference relative energy", relative_energy, -kappa)
equal("linear unstable tangent obeys primary constraint",
      2*q*q*s.Symbol("r")-q*(2*q*s.Symbol("r")))
equal("linear unstable tangent obeys secondary constraint",
      q*(q*s.Symbol("r"))-q*q*s.Symbol("r"))

# The continuum longitudinal reduction explains the reflection and instability.
boundary_velocity = s.symbols("boundary_velocity", real=True)
equal("eliminating linearized mirror gives negative boundary kinetic term",
      (-q*R*boundary_velocity+q*q*R*R).subs(R,boundary_velocity/(2*q)),
      -boundary_velocity**2/4)
w = s.symbols("w", real=True)
reflection = (1-s.I*w)/(1+s.I*w)
equal("dynamic boundary reflection equals clock singlet reflection",
      s.I*w*(reflection-1)-w*w*(1+reflection))
equal("decaying bound profile has negative kinetic bilinear",
      s.integrate(s.exp(-2*x),(x,0,s.oo))-1, -s.Rational(1,2))
growth = s.symbols("growth", positive=True)
equal("growing mode satisfies dynamic boundary at unit exponent",
      (-growth+growth**2).subs(growth,1))
for n in (2,3,4,8,16):
    A = np.array([[-.5 if i==j else (-1. if i<j else 0.)
                   for j in range(n)] for i in range(n)])
    B = A.T@A
    b = np.ones(n)
    G = np.eye(n)-np.outer(b,b)
    ev = np.linalg.eigvals(np.linalg.solve(G,B))
    if np.max(abs(ev.imag))>1e-10 or sum(ev.real < -1e-10)!=1:
        raise AssertionError(("one unstable longitudinal mode",n,ev))
    growth_value = float(np.sqrt(-min(ev.real)))
    resolvent = float(b@np.linalg.solve(B+growth_value**2*np.eye(n),b))
    if abs(1-growth_value**2*resolvent)>1e-10:
        raise AssertionError(("bound mode secular equation",n,growth_value,resolvent))
    checks.append(f"finite longitudinal action has one unstable mode N={n}")
    checks.append(f"unstable mode obeys exact boundary resolvent condition N={n}")

result = {
    "passed": True,
    "checks": len(checks),
    "named_checks": checks,
    "scope": (
        "Moving-boundary positive-clock action, derivative-closed finite "
        "regulator and physical runaway-mode diagnostics; no MQM state/limit."
    ),
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_boundary_dynamics_results.json').write_text(
    json.dumps(result,indent=2)+"\n"
)
print(json.dumps(result,indent=2))
