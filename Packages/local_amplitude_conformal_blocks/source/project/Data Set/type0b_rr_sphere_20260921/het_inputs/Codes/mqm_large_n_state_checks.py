"""Exact diagnostics of the SO(m)-zonal state in ST^N(C^{m+1}).

Run with /Users/sam/miniconda3/bin/python mqm_large_n_state_checks.py.
These checks concern the regulated ground spin tensor, not an inverted sea.
"""
import json
from pathlib import Path
import sympy as s

checks = []


def check(name, lhs, rhs):
    assert s.simplify(lhs - rhs) == 0, (name, lhs, rhs)
    checks.append(name)


def dim_st(n, d):
    return s.binomial(n + d - 1, n) - (s.binomial(n + d - 3, n - 2) if n >= 2 else 0)


def coeff(n, m, j):
    return (-1)**j * s.factorial(n) / (s.factorial(n - 2*j) * 4**j * s.factorial(j) * s.rf(s.Rational(m, 2), j))


def weight(n, m, j):
    return s.factorial(n)**2 / (s.factorial(n - 2*j) * 4**j * s.factorial(j) * s.rf(s.Rational(m, 2), j))


def fock_norm(poly, variables):
    return sum(c * s.conjugate(c) * s.prod(s.factorial(k) for k in powers)
               for powers, c in s.Poly(s.expand(poly), *variables).terms())


for m in (2, 3, 5, 23):
    nu = s.Rational(m-1, 2)
    for n in range(2, 13):
        weights = [weight(n, m, j) for j in range(n//2+1)]
        z = sum(weights)
        check(f"normalization_m{m}_n{n}", z, s.factorial(n)*2**n*s.rf(nu, n)/s.rf(2*nu, n))
        mean = sum((n-2*j)*w for j, w in enumerate(weights))/z
        moment2 = sum((n-2*j)*(n-2*j-1)*w for j, w in enumerate(weights))/z
        p0 = s.Rational(n+m-2, 2*n+m-3)
        pa = s.Rational(n-1, m*(2*n+m-3))
        check(f"occupation_m{m}_n{n}", mean, n*p0)
        check(f"second_moment_m{m}_n{n}", moment2,
              s.Rational(n*(n-1)*(n+m-2)*(n+m-3), (2*n+m-3)*(2*n+m-5)))
        check(f"variance_m{m}_n{n}", moment2+mean-mean**2,
              s.Rational(2*n*(n-1)*(n+m-3)*(n+m-2), (2*n+m-5)*(2*n+m-3)**2))
        a = p0+pa
        b = s.Rational(2*(n+m-2), m*(2*n+m-3))
        check(f"casimir_gram_m{m}_n{n}", n*a+n*(n-1)*b, s.Rational(n*(n+m-1), m))
        check(f"gram_difference_m{m}_n{n}", a-b,
              s.Rational((m-1)*(n+m-3), m*(2*n+m-3)))
        check(f"branching_dimension_m{m}_n{n}", dim_st(n,m+1), sum(dim_st(k,m) for k in range(n+1)))

# Expanded polynomials provide an independent check of the sector weights
# and generator norm; no use of the Casimir formula in this calculation.
for m in (2, 3):
    variables = s.symbols(f"z0:{m+1}", real=True)
    t = variables[0]
    sigma = sum(z*z for z in variables[1:])
    for n in range(2, 7):
        h = sum(coeff(n,m,j)*t**(n-2*j)*sigma**j for j in range(n//2+1))
        check(f"harmonic_m{m}_n{n}", sum(s.diff(h,z,2) for z in variables), 0)
        z = fock_norm(h, variables)
        check(f"expanded_norm_m{m}_n{n}", z, sum(weight(n,m,j) for j in range(n//2+1)))
        rotation = t*s.diff(h,variables[1])-variables[1]*s.diff(h,t)
        check(f"expanded_rotation_m{m}_n{n}", fock_norm(rotation, variables)/z,
              s.Rational(n*(n+m-1),m))
        # Spherical moments reproduce every coefficient of the coherent average.
        for j in range(n//2+1):
            sphere_moment = s.factorial(2*j)/(4**j*s.factorial(j)*s.rf(s.Rational(m,2),j))
            check(f"coherent_average_m{m}_n{n}_j{j}", s.binomial(n,2*j)*(-1)**j*sphere_moment, coeff(n,m,j))

n, m = s.symbols("N m", positive=True)
p0 = (n+m-2)/(2*n+m-3)
pa = (n-1)/(m*(2*n+m-3))
a = p0+pa
b = 2*p0/m
for name, expr, value in (("p0_limit",p0,s.Rational(1,2)),
                          ("pa_limit",pa,1/(2*m)),
                          ("a_limit",a,(m+1)/(2*m)),
                          ("b_limit",b,1/m),
                          ("difference_limit",a-b,(m-1)/(2*m)),
                          ("purity_limit",p0**2+m*pa**2,(m+1)/(4*m))):
    check(name,s.limit(expr,n,s.oo),value)

table = []
for nvalue in (2, 3, 10, 100, 1000, 1000000):
    substitutions = {n:nvalue,m:23}
    table.append({"N":nvalue, "p0":float(p0.subs(substitutions)),
                  "pa":float(pa.subs(substitutions)), "A":float(a.subs(substitutions)),
                  "B":float(b.subs(substitutions)), "A_minus_B":float((a-b).subs(substitutions))})
output = {"checks_passed":len(checks), "scope":"Exact zonal spin state; no inverted-sea assumption", "SO23_table":table}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_large_n_state_results.json').write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,indent=2))
