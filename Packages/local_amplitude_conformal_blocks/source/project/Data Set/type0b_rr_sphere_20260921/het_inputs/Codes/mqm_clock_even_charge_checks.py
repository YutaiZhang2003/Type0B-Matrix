"""Exact boundary and zero-mode tests for geometric clock charges.

These tests do not establish a heterotic charge dictionary.  They check
concrete examples used in the independent local-one-form classification.
"""
from pathlib import Path
import json
import sympy as s

t, a, c, q = s.symbols("t a c q", real=True)
checks = []


def equal(name, left, right):
    residual = s.simplify(s.expand_trig(left - right))
    if residual != 0:
        raise AssertionError((name, residual))
    checks.append(name)


L = 2*s.pi
x, y = s.cos(t), s.sin(t)
integrate_circle = lambda f: s.integrate(f, (t, 0, L))

area = integrate_circle((x*s.diff(y,t)-y*s.diff(x,t))/2)
equal("transverse oriented area", area, s.pi)
equal("area invariant under transverse translation",
      integrate_circle(((x+c)*s.diff(y,t)-y*s.diff(x+c,t))/2), area)
equal("area invariant under integration start",
      s.integrate((x*s.diff(y,t)-y*s.diff(x,t))/2, (t,a,a+L)), area)

nonlinear = x**2*s.diff(y,t)
equal("nonlinear curve moment on centered circle",
      integrate_circle(nonlinear), 0)
equal("nonlinear curve moment changes under scalar translation",
      integrate_circle((x+c)**2*s.diff(y,t)-nonlinear), 2*s.pi*c)
equal("nonlinear moment Hamiltonian x endpoint defect",
      integrate_circle(-2*x*s.diff(y,t)), -2*s.pi)
equal("nonlinear moment Hamiltonian y endpoint defect",
      integrate_circle(2*x*s.diff(x,t)), 0)

mixed = q*t*s.diff(x,t)
mixed_integral = s.integrate(mixed, (t,a,a+L))
equal("mixed area depends on integration start", mixed_integral,
      2*s.pi*q*s.cos(a))
equal("mixed area start derivative",
      s.diff(mixed_integral,a), -2*s.pi*q*s.sin(a))
equal("mixed area invariant under scalar x translation",
      s.integrate(q*t*s.diff(x+c,t)-mixed, (t,a,a+L)), 0)
equal("mixed area invariant under scalar longitudinal translation",
      s.integrate((q*t+c)*s.diff(x,t)-mixed, (t,a,a+L)), 0)
equal("periodic mixed representative fails scalar quotient",
      integrate_circle(-(x+c)*q+x*q), -2*s.pi*q*c)
equal("mixed area exact improvement is not zero",
      ((q*t*x).subs(t,a+L)-(q*t*x).subs(t,a)),
      2*s.pi*q*s.cos(a))

x_line, y_line = s.exp(-t*t), t*s.exp(-t*t)
area_line = s.integrate(x_line*s.diff(y_line,t), (t,-s.oo,s.oo))
equal("localized line has nonzero transverse area",
      area_line, s.sqrt(s.pi)/(2*s.sqrt(2)))
equal("localized nonlinear moment is finite",
      s.integrate(x_line**2*s.diff(y_line,t), (t,-s.oo,s.oo)),
      2*s.sqrt(s.pi)/(3*s.sqrt(3)))
equal("localized nonlinear moment changes under constant translation",
      s.integrate(((x_line+c)**2-x_line**2)*s.diff(y_line,t),
                  (t,-s.oo,s.oo)),
      c*s.sqrt(s.pi)/s.sqrt(2))
equal("localized nonlinear generator leaves fixed endpoint sector",
      s.integrate(-2*x_line*s.diff(y_line,t), (t,-s.oo,s.oo)),
      -s.sqrt(s.pi)/s.sqrt(2))

# A transverse rotation preserves the slope and the local stress.
f0, f1, f2, j0, j1, j2 = s.symbols("f0 f1 f2 j0 j1 j2", real=True)
B = s.Matrix([[0,0,0],[0,0,1],[0,-1,0]])
background = s.Matrix([q,0,0])
current = s.Matrix([j0,j1,j2])
equal("transverse rotation preserves slope", (B*background).norm()**2, 0)
equal("transverse rotation preserves stress", (current.T*B*current)[0], 0)
equal("wall stabilizer dimension", s.binomial(23,2), 253)
equal("missing mixed rotations", s.binomial(24,2)-s.binomial(23,2), 23)

# A nonlocal polynomial invariant survives outside the one-form ansatz.
M1, M2, M3 = s.symbols("M1 M2 M3", real=True)
invariant = M1*M3-s.Rational(3,4)*M2**2
shifted_invariant = M1*(M3+3*c*M2+3*c*c*M1)-s.Rational(3,4)*(M2+2*c*M1)**2
equal("degree-six nonlocal invariant survives scalar x shift",
      shifted_invariant, invariant)
equal("degree-six Hamiltonian x gradient has zero mean",
      -(3*M1*M2-3*M2*M1), 0)
coordinate = s.symbols("coordinate", real=True)
C = M3+3*M1*coordinate**2-3*M2*coordinate
primitive = M3*coordinate+M1*coordinate**3-s.Rational(3,2)*M2*coordinate**2
equal("degree-six Hamiltonian y gradient is an exact derivative",
      s.diff(primitive,coordinate), C)

R, S = s.symbols("R S", positive=True)
xe, ye = R*s.cos(t), S*s.sin(t)
ellipse_moments = [
    integrate_circle(xe**r*s.diff(ye,t)) for r in (1,2,3)
]
equal("ellipse first moment", ellipse_moments[0], s.pi*R*S)
equal("ellipse second moment", ellipse_moments[1], 0)
equal("ellipse third moment", ellipse_moments[2], s.Rational(3,4)*s.pi*R**3*S)
ellipse_invariant = ellipse_moments[0]*ellipse_moments[2]-s.Rational(3,4)*ellipse_moments[1]**2
equal("ellipse degree-six invariant", ellipse_invariant,
      s.Rational(3,4)*s.pi**2*R**4*S**2)
equal("two ellipses have the same rotation area",
      ellipse_moments[0].subs({R:2,S:s.Rational(1,2)}),
      ellipse_moments[0].subs({R:1,S:1}))
equal("degree-six charge distinguishes equal-area ellipses",
      ellipse_invariant.subs({R:2,S:s.Rational(1,2)})
      -ellipse_invariant.subs({R:1,S:1}),
      s.Rational(9,4)*s.pi**2)

result = {
    "passed": True,
    "checks": len(checks),
    "named_checks": checks,
    "scope": (
        "Geometric polynomial one-form charges on fixed-slope circle and "
        "pinned localized line; no complete heterotic charge identification."
    ),
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_even_charge_results.json').write_text(
    json.dumps(result, indent=2)+"\n"
)
print(json.dumps(result, indent=2))
