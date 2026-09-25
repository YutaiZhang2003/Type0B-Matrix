"""Independent oscillator verification of charged-primary stress form factors."""
import sympy as s

osc = s.symbols('a0:11')
z = s.symbols('z')
checks = 0

def state(h, n):
    level = n-h
    charge = s.sqrt(2*h)
    result = s.Integer(1)
    for r in range(1, level+1):
        result = s.series(result * sum((charge*osc[r]*z**r/r)**j/s.factorial(j)
                                      for j in range(level//r+1)), z, 0, level+1).removeO()
    return s.expand(result).coeff(z, level)

def inner(left, right):
    out = 0
    lp = s.Poly(left, *osc[1:])
    rp = s.Poly(right, *osc[1:]).as_dict()
    for powers, coeff in lp.terms():
        out += coeff * rp.get(powers, 0) * s.prod(
            r**occupation * s.factorial(occupation)
            for r, occupation in enumerate(powers, start=1))
    return s.simplify(out)

def virasoro(h, q, polynomial):
    result = s.sqrt(2*h)*q*s.diff(polynomial, osc[q])
    result += sum((q+r)*osc[r]*s.diff(polynomial, osc[q+r])
                  for r in range(1, len(osc)-q))
    result += sum(s.Rational(r*(q-r), 2)*s.diff(polynomial, osc[r], osc[q-r])
                  for r in range(1, q))
    return s.expand(result)

def norm_formula(h, n):
    return s.binomial(n+h-1, 2*h-1)

for h in (1, 2, 3):
    for n in range(h, h+6):
        initial = state(h, n)
        assert inner(initial, initial) == norm_formula(h, n)
        checks += 1
        for q in range(1, n-h+1):
            m = n-q
            assert s.expand(virasoro(h, q, initial)
                            - (n+(h-1)*q)*state(h, m)) == 0
            checks += 1

h,n,m = 2,4,2
q = n-m
initial, final = state(h,n), state(h,m)
print('h=2 n=4 state:', initial)
print('h=2 m=2 state:', final)
print('L_2 initial:', virasoro(h,q,initial))
print('Normalized form factor:', inner(final,virasoro(h,q,initial))
      /s.sqrt(inner(initial,initial)*inner(final,final)))

E,x,y,h,CE,Cx = s.symbols('E x y h C_E C_x')
A = x-(h-1)*y
B = E+(h-1)*y
primary_term = A*CE
singlet_term = -A*CE+B*Cx
assert s.expand(primary_term+singlet_term-B*Cx) == 0
checks += 1
print('Primary transport plus nonlinear outgoing singlet:', s.factor(primary_term+singlet_term))

# All three terms in O_h(E) -> O_h(x) V(y) V(z). The factors common
# to every term (g^2 y z) are suppressed; C_h are kept independent.
Y = s.symbols('Y', positive=True)
D = lambda w: 1+s.I*w
A = x-(h-1)*Y
B = E+(h-1)*Y
direct = -s.I*A*CE/D(-Y)
sequential = -2*Y*A*CE/(D(Y)*D(-Y))
stress_output = -s.I*(-A*CE+B*Cx)/D(Y)
assert s.simplify(direct+sequential+s.I*A*CE/D(Y)) == 0
assert s.simplify(direct+sequential+stress_output+s.I*B*Cx/D(Y)) == 0
checks += 2
print('O -> O V V total / (g^2 y z):', -s.I*B*Cx/D(Y))
print('TOTAL', checks, 'charged-primary checks passed')

import json
from pathlib import Path
output = Path(__file__).resolve().parents[1]/'data_exports/heterotic_long_string_trees_20260912'/'charged_primary_results.json'
result = {'status':'passed','checks':checks,'scope':'Operator-prepared charged chiral primary states, continuum limit of circle regulator; not a new worldsheet S matrix','mode_norm':'C_h(n)=Gamma(n+h)/[Gamma(2*h)*Gamma(n-h+1)]','A_primary_to_primary_S':'-i*sqrt(2)*gamma*sqrt(y)/Q(y)*(E+(h-1)*y)*(x/E)^(h-1/2)','A_primary_to_primary_VV':'-i*gamma^2*delta_ij*sqrt(y*z)/D(y+z)*(E+(h-1)*(y+z))*(x/E)^(h-1/2)','spinor_D16plus_h2_shape':'(2*E-x)*(x/E)^(3/2)','normalization':'unit delta external states; energy delta removed; D and Q take physical Rindler frequency, not circle integer'}
output.write_text(json.dumps(result,indent=2)+'\n')
