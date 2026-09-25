#!/usr/bin/env python3
"""Finite matrix-moment and droplet checks; no string-scattering claim."""

import json
from pathlib import Path

import sympy as s


def matrix_moment_check():
    # Two normal matrix phase-space points, already on the singlet Gauss surface.
    n = 2
    z = s.diag(1, -1)
    w = s.diag(1 + s.I, -1 - s.I)
    moment = lambda mat, m: s.trace(mat**m) / s.sqrt(n)**m
    for m in (1, 2):
        assert s.simplify(moment(w, m) - s.I * m * moment(z, m)) == 0

    def gram(mat):
        return s.Matrix(2, 2, lambda i, j: s.simplify(
            (i + 1) * (j + 1) * s.trace(mat**i * mat.adjoint()**j)
            / s.sqrt(n)**(i + j + 2)))

    a, b = gram(z), gram(w)
    ell = s.diag(s.I, 2 * s.I)
    constraint = b + ell * a * ell.adjoint()
    reduced = s.simplify(a - a * ell.adjoint() * constraint.inv() * ell * a)
    assert a == s.diag(1, 2)
    assert b == s.diag(1, 4)
    assert constraint == s.diag(2, 12)
    assert reduced == s.diag(s.Rational(1, 2), s.Rational(2, 3))
    assert reduced[1, 1] != s.Rational(2, 5)
    return {
        "finite_N": n,
        "constraints_checked": [1, 2],
        "constraint_gram": str(constraint),
        "reduced_positive_mode_bracket_over_minus_i": str(reduced),
        "not_identical_to_central_droplet_bracket": True,
    }


def droplet_checks():
    n, h = s.symbols('N h', positive=True)
    for m in range(1, 9):
        expr = ((n + h)**(1 + s.Rational(m, 2))
                / (1 + s.Rational(m, 2)) / n**s.Rational(m, 2))
        assert s.simplify(s.diff(expr, h).subs(h, 0) - 1) == 0
        assert s.simplify(s.diff(expr, h, 2).subs(h, 0) / 2
                          - s.Rational(m, 4) / n) == 0
        assert s.simplify(s.diff(expr, h, 3).subs(h, 0) / 6
                          - s.Rational(m * (m - 2), 24) / n**2) == 0
    z = s.symbols('z')
    a, ac, b, bc = s.symbols('a ac b bc')
    u = a*z + ac/z + b*z**2 + bc/z**2
    derivative = lambda f: s.expand(s.I * z * s.diff(f, z))

    def abs_derivative(f):
        return s.expand(sum(abs(term.as_powers_dict().get(z, 0)) * term
                            for term in s.Add.make_args(s.expand(f))))

    w0 = derivative(u)
    w1 = s.expand((derivative(abs_derivative(u*u))
                   - abs_derivative(w0*w0)) / 4)
    residue = s.expand(w1 + abs_derivative(w0*w0)/4
                       - derivative(abs_derivative(u*u))/4)
    assert residue == 0
    cubic = s.expand(w0*w1).coeff(z, 0)
    expected = ((s.Rational(5, 2) - 2*s.I)*a*a*bc
                + (s.Rational(5, 2) + 2*s.I)*ac*ac*b)
    assert s.expand(cubic - expected) == 0
    return {
        "moment_expansions_checked": 8,
        "quadratic_constraint_correction":
            "w1=(D|D|(u^2)-|D|((Du)^2))/4",
        "induced_cubic_in_noncanonical_u_coordinates": str(cubic),
        "interpretation": "Harmonic flow remains exact translation; this term is not an independent scattering vertex.",
    }


def flavor_checks():
    dimension = 4
    identity = s.eye(dimension)
    products = [identity]
    for a in range(1, dimension):
        mat = s.zeros(dimension)
        mat[0, a] = mat[a, 0] = 1
        products.append(mat)
    commutator = products[1]*products[2] - products[2]*products[1]
    expected = s.zeros(dimension)
    expected[1, 2], expected[2, 1] = 1, -1
    assert commutator == expected

    # Minimal omitted-current test: two Clifford directions generate a third.
    sx = s.Matrix([[0, 1], [1, 0]])
    sy = s.Matrix([[0, -s.I], [s.I, 0]])
    sz = s.diag(1, -1)
    assert sz*sx - sx*sz == 2*s.I*sy
    assert sy*sx - sx*sy == -2*s.I*sz
    assert sy*sz - sz*sy == 2*s.I*sx
    return {
        "spin_factor_multiplications_do_not_commute": True,
        "commutator": str(commutator),
        "omitted_current_zero_mode_rotates_retained_currents": True,
    }


def main():
    results = {
        "matrix_moments": matrix_moment_check(),
        "droplet_expansion": droplet_checks(),
        "flavor_obstructions": flavor_checks(),
        "scope": "Exact finite matrix and symbolic edge-algebra tests, not a quantum constrained MQM or a heterotic amplitude.",
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_dirac_matrix_origin_results.json').write_text(
        json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
