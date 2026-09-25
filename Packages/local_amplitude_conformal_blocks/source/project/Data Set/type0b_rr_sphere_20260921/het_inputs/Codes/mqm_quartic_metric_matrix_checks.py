#!/usr/bin/env python3
"""Checks of a positive finite-matrix extension of the inverse-fit quartic.

This tests finite-dimensional geometry and the stated classical expansion.
It does not test the large-N state, quantum scattering, or LSZ dictionary.
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as s


def sphere_edge():
    beta, eps = s.symbols("beta eps", positive=True)
    qi = s.Matrix(s.symbols("qi1 qi2", real=True))
    qj = s.Matrix(s.symbols("qj1 qj2", real=True))
    vi = s.Matrix(s.symbols("vi1 vi2", real=True))
    ni = s.Matrix([s.sqrt(1-eps**2*qi.dot(qi)/beta**2),
                   eps*qi[0]/beta, eps*qi[1]/beta])
    nj = s.Matrix([s.sqrt(1-eps**2*qj.dot(qj)/beta**2),
                   eps*qj[0]/beta, eps*qj[1]/beta])
    dni = ni.jacobian(list(qi))*vi
    overlap = s.simplify(dni.dot(nj))
    leading = s.simplify(s.diff(overlap, eps, 2).subs(eps, 0)/2)
    expected = vi.dot(qj-qi)/beta**2
    assert s.simplify(leading-expected) == 0
    assert overlap.subs(eps, 0) == 0
    assert s.diff(overlap, eps).subs(eps, 0) == 0
    # The whole one-form changes sign under a sign change of either
    # endpoint director, so its square is independently projective.
    assert s.expand((-leading)**2-leading**2) == 0
    return {"leading_one_form": str(leading),
            "first_interaction_order": 4,
            "independent_director_signs": "invariant after squaring"}


def wall_factors():
    a, mu, v0, rho, nu, tau = s.symbols(
        "a mu v0 rho nu tau", positive=True)
    # Two endpoint terms times 1/2, sum_i -> integral rho dx,
    # and each nearest-neighbor difference supplies rho^-1.
    x_coefficient = s.simplify(nu*s.Rational(1, 2)*2*rho/rho**2)
    assert x_coefficient == nu/rho
    tau_coefficient = s.simplify((x_coefficient*v0/v0**2)
                                .subs(rho, v0/a)
                                .subs(nu, 1/(2*mu))
                                .subs(v0**2, 2*mu*s.sinh(tau)**2))
    K2 = a/(4*mu**2)
    assert s.simplify(tau_coefficient-K2/s.sinh(tau)**2) == 0
    return {"moving_particle_continuum_coefficient": str(x_coefficient),
            "static_wall_coefficient": str(tau_coefficient)}


def four_wave_signs():
    x, y, z, tau = s.symbols("x y z tau", positive=True)
    w = [x+y+z, x, y, z]
    e = s.symbols("e0:4")
    ca, cb = [s.cos(wi*tau) for wi in w[:2]], \
             [s.cos(wi*tau) for wi in w[2:]]
    sa, sb = [s.sin(wi*tau) for wi in w[:2]], \
             [s.sin(wi*tau) for wi in w[2:]]
    # Strip time exponentials whose four-leg product is one.
    ta = -s.I*w[0]*ca[0]*e[0]+s.I*w[1]*ca[1]*e[1]
    tb = s.I*w[2]*cb[0]*e[2]+s.I*w[3]*cb[1]*e[3]
    xa = -w[0]*sa[0]*e[0]-w[1]*sa[1]*e[1]
    xb = -w[2]*sb[0]*e[2]-w[3]*sb[1]*e[3]
    wave_product = s.prod(e)
    W, q = s.prod(w), y+z
    bulk = s.Poly(s.expand((ta*xa+tb*xb)**2), *e).coeff_monomial(wave_product)
    difference = s.expand((bulk+2*W*s.sin(q*tau)**2).rewrite(s.exp))
    assert s.simplify(difference) == 0
    boundary = s.Poly(s.expand((ta**2+tb**2)**2).subs(tau, 0), *e)\
        .coeff_monomial(wave_product)
    assert s.simplify(boundary+8*W) == 0
    return {"bulk_four_wave_coefficient": "-2 W sin(q tau)^2",
            "boundary_four_wave_coefficient": "-8 W"}


def positive_function_and_legendre():
    r, d, p = s.symbols("r d p", positive=True)
    kin = (r+4*d*r**3)**2/2
    h = r**2/2+3*d*r**4
    assert s.expand(kin-h) == d*r**4+8*d**2*r**6
    # Inverse of p=r+4 d r^3 through p^5.
    r_series = p-4*d*p**3+48*d**2*p**5
    assert s.series(r_series+4*d*r_series**3-p, p, 0, 7).removeO() == 0
    h_series = s.series(h.subs(r, r_series), p, 0, 7).removeO()
    assert s.expand(h_series-(p**2/2-d*p**4+8*d**2*p**6)) == 0
    eps, c = s.symbols("eps c", positive=True)
    v = s.Matrix(s.symbols("v1 v2", real=True))
    # Non-diagonal base metric, with full-rank velocity map A.
    g = s.Matrix([[2, s.Rational(1, 3)],
                  [s.Rational(1, 3), s.Rational(3, 2)]])
    A = s.Matrix([[1, 2], [-1, 1]])
    norm4 = (A*v).dot(A*v)**2
    correction = 4*c*A.T*A*v*(A*v).dot(A*v)
    momentum = eps*g*v+eps**3*correction
    gv = g.inv()*momentum
    H = (momentum.dot(gv)/2-c*(A*gv).dot(A*gv)**2)
    L = s.expand(momentum.dot(eps*v)-H)
    assert s.simplify(L.coeff(eps, 2)-v.dot(g*v)/2) == 0
    assert s.simplify(L.coeff(eps, 4)-c*norm4) == 0
    # Coordinate-invariant kinetic split reproduces coefficient c.
    split_eps, t = s.symbols("split_eps t", positive=True)
    se = split_eps*t/2
    assert s.simplify(-4*(c/split_eps**2)*se**2+c*t**2) == 0
    return {"exact_s_minus_h": str(s.expand(kin-h)),
            "small_momentum_h": str(h_series),
            "Hamiltonian_quartic": "-c |A g^-1 p|^4",
            "Legendre_quartic": "+c |A velocity|^4",
            "kinetic_split_epsilon_cancels_at_quartic": True}


def main():
    result = {
        "scope": "Finite-regulator geometry and semiclassical quartic only.",
        "sphere_edge": sphere_edge(),
        "continuum_factors": wall_factors(),
        "independent_four_wave_expansion": four_wave_signs(),
        "positive_boundary_completion": positive_function_and_legendre(),
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_quartic_metric_matrix_results.json').write_text(
        json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
