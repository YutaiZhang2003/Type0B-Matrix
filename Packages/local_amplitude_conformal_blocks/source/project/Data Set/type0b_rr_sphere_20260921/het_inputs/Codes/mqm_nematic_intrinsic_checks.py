#!/usr/bin/env python3
"""Leading intrinsic nematic sigma metric and scoped quartic checks."""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def coherent_checks():
    kap = sp.symbols("kappa", positive=True)
    u, v, ut, vt, at, bt = sp.symbols("u v ut vt at bt", real=True)
    n = sp.Matrix([1, 0, 0])
    ndot = sp.Matrix([0, at, bt])
    m = sp.Matrix([0, u, v])
    mdot = sp.Matrix([-u*at-v*bt, ut, vt])
    den = 1+(m.T*m)[0]
    z = (n+sp.I*m)/sp.sqrt(den)
    zdot = (ndot+sp.I*mdot)/sp.sqrt(den) - z*(m.T*mdot)[0]/den
    berry = sp.simplify(sp.I*kap*(sp.conjugate(z).T*zdot)[0])
    assert sp.simplify(berry-2*kap*(m.T*ndot)[0]/den) == 0
    R = (n*n.T+m*m.T)/den
    assert sp.simplify(1-sp.trace(R*R)-2*(m.T*m)[0]/den**2) == 0
    a, b, U, V, eps = sp.symbols("a b U V eps", real=True)
    nx = sp.Matrix([0, a, b])
    mx = sp.Matrix([-a*u-b*v, U, V])
    Rx = (nx*n.T+n*nx.T+mx*m.T+m*mx.T)/den \
        - R*2*(m.T*mx)[0]/den
    trace = sp.trace(Rx*Rx).subs(
        {u: eps*u, v: eps*v, U: eps*U, V: eps*V})
    constant = sp.simplify(trace.subs(eps, 0))
    quadratic = sp.simplify(sp.diff(trace, eps, 2).subs(eps, 0)/2)
    expected = -4*((u*u+v*v)*(a*a+b*b)+(u*a+v*b)**2)
    assert constant == 2*(a*a+b*b)
    assert sp.simplify(quadratic-expected) == 0
    return {"Berry": str(berry),
            "quadrupole_gradient_at_m0": str(constant),
            "quadrupole_gradient_m2": str(sp.factor(quadratic))}


def eliminate_m_and_metric():
    kap, J, rho = sp.symbols("kappa J rho", positive=True)
    U, speed = sp.symbols("U speed", real=True)
    uniform = 2*kap*rho*U*speed-4*J*kap*rho**3*U**2
    ustar = sp.solve(sp.diff(uniform, U), U)[0]
    uniform_eff = sp.factor(uniform.subs(U, ustar))
    assert uniform_eff == kap*speed**2/(4*J*rho)
    R, t = sp.symbols("R t", positive=True)
    q1, q2, v1, v2 = sp.symbols("q1 q2 v1 v2", real=True)
    # Coordinate chart n=(sqrt(1-Q²/R²), Q/R).
    n0 = sp.sqrt(1-(q1*q1+q2*q2)/R**2)
    n0dot = sp.diff(n0, q1)*v1+sp.diff(n0, q2)*v2
    kinetic = sp.simplify(R**2*(n0dot*n0dot+(v1*v1+v2*v2)/R**2)/2)
    expected = (v1*v1+v2*v2)/2 + (q1*v1+q2*v2)**2/(2*(R**2-q1*q1-q2*q2))
    assert sp.simplify(kinetic-expected) == 0
    scaled = kinetic.subs({q1: t*q1, q2: t*q2, v1: t*v1, v2: t*v2})
    quartic = sp.factor(sp.diff(scaled, t, 4).subs(t, 0)/sp.factorial(4))
    assert sp.simplify(quartic-(q1*v1+q2*v2)**2/(2*R**2)) == 0
    return {"uniform_auxiliary_solution": str(ustar),
            "uniform_effective_kinetic": str(uniform_eff),
            "quartic_metric_kinetic": str(quartic),
            "pure_four_time_derivative_term": "absent in uniform coherent action"}


def contact_checks():
    w1, w2, w3, tau = sp.symbols("w1 w2 w3 tau", positive=True)
    w0, q, s, d = w1+w2+w3, w2+w3, 2*w1+w2+w3, w2-w3
    A = sp.cos(w0*tau)*sp.cos(w1*tau)
    B = sp.cos(w2*tau)*sp.cos(w3*tau)
    raw = q*q*A*B-sp.diff(A, tau)*sp.diff(B, tau)
    terms = []
    reconstructed = 0
    for alpha in (q, s):
        for beta in (q, d):
            for frequency, coeff in (
                (alpha-beta, (q*q-alpha*beta)/8),
                (alpha+beta, (q*q+alpha*beta)/8)):
                frequency, coeff = sp.expand(frequency), sp.factor(coeff)
                if frequency == 0:
                    assert coeff == 0
                    continue
                assert all(frequency.coeff(w) >= 0 for w in (w1, w2, w3))
                assert frequency != 0
                terms.append({"frequency": str(frequency), "coefficient": str(coeff)})
                reconstructed += coeff*sp.cos(frequency*tau)
    # Independent exponentials avoid a trigonometric simplifier depending
    # on the same product-to-sum decomposition.
    test = sp.expand((raw-reconstructed).rewrite(sp.exp))
    assert sp.simplify(test) == 0
    epsilon = sp.symbols("epsilon", positive=True)
    finite_abel = sum(sp.sympify(row["coefficient"])*epsilon /
                     (epsilon**2+sp.sympify(row["frequency"])**2) for row in terms)
    assert sp.limit(finite_abel, epsilon, 0, dir="+") == 0
    return {"nonzero_cosine_frequencies": terms,
            "zero_frequency_coefficient": "0",
            "Abel_1to3_contact_at_strictly_positive_energies": "0"}


def scaling_checks():
    a, J, kap, rho, v0 = sp.symbols("a J kappa rho v0", positive=True)
    R2 = sp.sqrt(2)*kap
    # κ/(4Jρ³ R⁴) in x, multiplied by dx=v0 dτ and two
    # spatial derivatives ∂x=v0^-1∂τ.
    coefficient = kap/(4*J*rho**3*R2**2)*v0/v0**2
    coefficient = sp.simplify(coefficient.subs({rho: v0/a, J: a/(2*sp.sqrt(2))}))
    target = a*a/(2*sp.sqrt(2)*kap*v0**4)
    assert sp.simplify(coefficient-target) == 0
    return {"first_mixed_four_derivative_coefficient": str(coefficient),
            "static_wall_tail_at_v0_proportional_sinh_tau": "sinh(tau)^(-4)",
            "scope": "The displayed term is not the full fourth-gradient action."}


def main():
    results = {
        "scope": "Leading coherent large-spin, smooth-gradient sigma action and one intrinsic mixed correction; not full wall scattering.",
        "coherent_geometry": coherent_checks(),
        "metric": eliminate_m_and_metric(),
        "four_vector_contact": contact_checks(),
        "higher_gradient_term": scaling_checks(),
    }
    out = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_nematic_intrinsic_results.json')
    out.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
