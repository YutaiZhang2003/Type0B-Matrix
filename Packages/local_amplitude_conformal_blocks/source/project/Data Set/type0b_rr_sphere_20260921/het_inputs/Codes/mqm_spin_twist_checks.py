#!/usr/bin/env python3
"""Exact spin-twist norms and a conditional far-sea integral diagnostic.

No large-N state or density factorization is established by these checks.
The density used in the numerical tail test is explicitly an assumption.
"""
import itertools
import json
from pathlib import Path

import mpmath as mp
import sympy as sp

from mqm_one_vector_checks import add, contract, permute, rotate, scale, tensor_inner


def main():
    d, rank = 24, 3
    tensor = {(0, 0, 0): sp.Integer(1)}
    for label in range(1, d):
        for indices in set(itertools.permutations((0, label, label))):
            tensor[indices] = -sp.Rational(1, d - 1)
    norm = tensor_inner(tensor, tensor)
    rotated = [rotate(tensor, i, 1) for i in range(rank)]
    a = tensor_inner(rotated[0], rotated[0]) / norm
    b = tensor_inner(rotated[0], rotated[1]) / norm
    expected_a = sp.Rational(24 * rank + 482, 23 * (2 * rank + 20))
    expected_b = sp.Rational(2 * (rank + 21), 23 * (2 * rank + 20))
    assert a == expected_a and b == expected_b
    coefficients = [sp.Integer(2), 1 + sp.I, -3 + 2 * sp.I]
    state = add(*(scale(t, f) for t, f in zip(rotated, coefficients)))
    expected_norm = (a - b) * sum(abs(f)**2 for f in coefficients) + b * abs(sum(coefficients))**2
    assert sp.simplify(tensor_inner(state, state) / norm - expected_norm) == 0
    pair_checks = []
    for i, j in itertools.combinations(range(rank), 2):
        action = add(state, scale(permute(state, i, j), -1), contract(state, i, j, d))
        expected = (a - b) * abs(coefficients[i] - coefficients[j])**2
        assert sp.simplify(tensor_inner(state, action) / norm - expected) == 0
        pair_checks.append({"pair": [i, j], "quadratic_spin_penalty": str(expected)})

    # lambda*rho(y) = sqrt(y^2 - wall^2)/pi for the assumed one-sided sea.
    # A support point is fixed; both integration endpoints are far from it.
    mp.mp.dps = 40
    wall, point = mp.mpf(2), mp.mpf(3)
    tails = []
    previous_error = None
    for lower in [100, 1000, 10000]:
        value = mp.quad(lambda y: mp.sqrt(y*y-wall*wall)/(mp.pi*(y-point)**2),
                        [lower, 10*lower])
        expected = mp.log(10)/mp.pi
        error = abs(value-expected)
        if previous_error is not None:
            assert error < previous_error / 9
        previous_error = error
        tails.append({"interval": [lower, 10*lower], "integral": float(value),
                      "logarithmic_limit": float(expected), "absolute_error": float(error)})
    assert previous_error < mp.mpf("0.0002")
    results = {
        "all_checks_passed": True,
        "normalized_gram_A": str(a), "normalized_gram_B": str(b),
        "complex_coefficient_norm_identity": True,
        "pair_checks": pair_checks,
        "far_sea_integral": tails,
        "scope": "Exact rank-three spin identities; numerical integral for an assumed semiclassical density. This does not construct a many-body sea or exclude other state/operator dictionaries."
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_spin_twist_results.json').write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
