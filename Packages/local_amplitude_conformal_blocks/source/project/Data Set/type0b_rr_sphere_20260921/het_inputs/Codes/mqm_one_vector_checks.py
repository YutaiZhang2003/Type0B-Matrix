#!/usr/bin/env python3
"""Finite harmonic spin-tensor identities; no inverted-sea approximation.

Checks local SO(d) rotations of an SO(d-1)-invariant ST^3 tensor using
explicit sparse tensors, and the Jastrow-conjugated one-vector operator.
The operator acts on coefficient functions through an intertwiner; the
coefficient labels are not asserted to be orthonormal physical modes.
"""

import itertools
import json
from pathlib import Path

import sympy as sp


def clean(tensor):
    return {key: sp.simplify(value) for key, value in tensor.items() if sp.simplify(value) != 0}


def add(*tensors):
    result = {}
    for tensor in tensors:
        for key, value in tensor.items():
            result[key] = result.get(key, 0) + value
    return clean(result)


def scale(tensor, factor):
    return clean({key: factor * value for key, value in tensor.items()})


def permute(tensor, i, j):
    result = {}
    for key, value in tensor.items():
        permuted = list(key)
        permuted[i], permuted[j] = permuted[j], permuted[i]
        result[tuple(permuted)] = value
    return clean(result)


def contract(tensor, i, j, d):
    result = {}
    for key, value in tensor.items():
        if key[i] != key[j]:
            continue
        for label in range(d):
            contracted = list(key)
            contracted[i] = contracted[j] = label
            output = tuple(contracted)
            result[output] = result.get(output, 0) + value
    return clean(result)


def rotate(tensor, site, a):
    """Hermitian generator L^(0a)=i(|a><0|-|0><a|)."""
    result = {}
    for key, value in tensor.items():
        if key[site] not in (0, a):
            continue
        output = list(key)
        if key[site] == 0:
            output[site], factor = a, sp.I
        else:
            output[site], factor = 0, -sp.I
        output = tuple(output)
        result[output] = result.get(output, 0) + factor * value
    return clean(result)


def tensor_inner(left, right):
    return sp.simplify(sum(sp.conjugate(value) * right.get(key, 0)
                           for key, value in left.items()))


def sparse_tensor_checks(d=24):
    # The rank-three zonal harmonic is t^3 - 3 t |y|^2/(d-1).
    # Tensor contraction with z^tensor3 multiplies a 0aa tensor entry by 3.
    tensor = {(0, 0, 0): sp.Integer(1)}
    for a in range(1, d):
        for indices in set(itertools.permutations((0, a, a))):
            tensor[indices] = -sp.Rational(1, d - 1)
    pairs = list(itertools.combinations(range(3), 2))
    for i, j in pairs:
        assert permute(tensor, i, j) == tensor
        assert contract(tensor, i, j, d) == {}
    rotated = [rotate(tensor, site, 1) for site in range(3)]
    for site in range(3):
        for i, j in pairs:
            assert contract(rotated[site], i, j, d) == {}
            permuted_site = j if site == i else i if site == j else site
            assert permute(rotated[site], i, j) == rotated[permuted_site]
    gram = sp.Matrix(3, 3, lambda i, j: tensor_inner(rotated[i], rotated[j]))
    assert all(eigenvalue > 0 for eigenvalue in gram.eigenvals())
    total = add(*rotated)
    assert total
    return {
        "dimension": d,
        "rank": 3,
        "zonal_polynomial": f"t^3 - 3 t |y|^2/{d-1}",
        "symmetric_traceless": True,
        "local_rotation_Q_annihilation": True,
        "local_rotation_P_covariance": True,
        "local_rotation_gram": [[str(gram[i, j]) for j in range(3)] for i in range(3)],
        "gram_eigenvalues": {str(value): multiplicity for value, multiplicity in gram.eigenvals().items()},
        "global_rotation_nonzero": True,
    }


def coefficient_operator_checks(n=3):
    x = sp.symbols(f"x0:{n}", real=True)
    lam, omega = sp.symbols("lambda Omega", positive=True)
    log_grad = [lam * sum(1 / (x[k] - x[j]) for j in range(n) if j != k) - omega * x[k]
                for k in range(n)]

    def effective(functions):
        return [sp.factor(
            -sum(sp.diff(functions[i], coordinate, 2) for coordinate in x) / 2
            -sum(log_grad[k] * sp.diff(functions[i], x[k]) for k in range(n))
            +lam * sum((functions[i] - functions[j]) / (x[i] - x[j])**2
                       for j in range(n) if j != i)
        ) for i in range(n)]

    assert effective([sp.Integer(1)] * n) == [0] * n
    linear = effective(list(x))
    assert all(sp.simplify(linear[i] - omega * x[i]) == 0 for i in range(n))
    center_of_mass = effective([sum(x)] * n)
    assert all(sp.simplify(value - omega * sum(x)) == 0 for value in center_of_mass)
    return {
        "particle_count": n,
        "constant_coefficients": "zero relative energy (global spin rotation)",
        "fi_equals_xi": "Omega relative energy; drift and exchange singularities cancel",
        "fi_equals_center_of_mass_sum": "Omega relative energy",
        "collision_scope": "Smooth permutation-equivariant coefficients have fi-fj=O(xi-xj), hence antisymmetric branch lambda+1.",
        "physics_scope": "Finite confining oscillator identities, not an inverted Fermi sea or string amplitude.",
    }


def main():
    results = {"tensor": sparse_tensor_checks(), "operator": coefficient_operator_checks()}
    path = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_one_vector_checks.json')
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
