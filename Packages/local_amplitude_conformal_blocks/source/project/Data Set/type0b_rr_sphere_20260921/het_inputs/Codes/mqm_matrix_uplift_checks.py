#!/usr/bin/env python3
"""Exact finite oscillator checks for the engineered Hermitian-matrix uplift.

Checks on N=2,d=3 suffice to test the color covariance, normal ordering,
Cartan occupation constraint and Weyl sign. General-N identities are
proved in mqm_matrix_uplift.md. This is not a heterotic scattering test.
"""

import itertools
import json
from pathlib import Path

import sympy as sp

N, d = 2, 3
mode_count = N * d
pairs = list(itertools.combinations_with_replacement(range(mode_count), 2))
basis = []
for first, second in pairs:
    occupation = [0] * mode_count
    occupation[first] += 1
    occupation[second] += 1
    basis.append(tuple(occupation))
lookup = {occupation: index for index, occupation in enumerate(basis)}
dimension = len(basis)


def word_matrix(operations):
    result = sp.zeros(dimension)
    for column, occupation in enumerate(basis):
        current = list(occupation)
        coefficient = sp.Integer(1)
        for creation, mode in reversed(operations):
            if creation:
                coefficient *= sp.sqrt(current[mode] + 1)
                current[mode] += 1
            elif current[mode] == 0:
                coefficient = 0
                break
            else:
                coefficient *= sp.sqrt(current[mode])
                current[mode] -= 1
        if coefficient:
            result[lookup[tuple(current)], column] += coefficient
    return result


def representation(color_unitary):
    mode_unitary = sp.kronecker_product(color_unitary, sp.eye(d))
    result = sp.zeros(dimension)
    for column, (j, k) in enumerate(pairs):
        for m in range(mode_count):
            for n in range(mode_count):
                coefficient = (mode_unitary[m, j] * mode_unitary[n, k]
                               * sp.sqrt(1 + int(m == n)) / sp.sqrt(1 + int(j == k)))
                if coefficient:
                    result[pairs.index(tuple(sorted((m, n)))), column] += coefficient
    return (result / color_unitary.det()).applyfunc(sp.simplify)


def spectral_spin(projector_i, projector_j, contraction):
    result = sp.zeros(dimension)
    for m, n, p, q in itertools.product(range(N), repeat=4):
        coefficient = projector_i[m, p] * projector_j[n, q]
        if coefficient == 0:
            continue
        for a, b in itertools.product(range(d), repeat=2):
            if contraction:
                operations = [(True, m*d+a), (True, n*d+a), (False, p*d+b), (False, q*d+b)]
            else:
                operations = [(True, m*d+a), (True, n*d+b), (False, p*d+b), (False, q*d+a)]
            result += coefficient * word_matrix(operations)
    return result.applyfunc(sp.simplify)


def require_zero(matrix, name):
    assert matrix.applyfunc(sp.simplify) == sp.zeros(*matrix.shape), name


projector_0, projector_1 = sp.diag(1, 0), sp.diag(0, 1)
p_full = spectral_spin(projector_0, projector_1, False)
q_full = spectral_spin(projector_0, projector_1, True)
physical_indices = [index for index, occupation in enumerate(basis)
                    if sum(occupation[:d]) == sum(occupation[d:]) == 1]
assert len(physical_indices) == d*d
restrict = lambda matrix: matrix.extract(physical_indices, physical_indices)
p, q = restrict(p_full), restrict(q_full)
require_zero(p*p-sp.eye(d*d), "P^2")
require_zero(q*q-d*q, "Q^2")
require_zero(p*q-q, "P Q")

e01 = sum((word_matrix([(True, a), (False, d+a)]) for a in range(d)), sp.zeros(dimension))
e10 = e01.conjugate().T
require_zero(restrict(e01*e10)-sp.eye(d*d)-p, "E01 E10=I+P")
require_zero(restrict((e01*e10+e10*e01)/2)-sp.eye(d*d)-p, "Angular coefficient")

color_rotation = sp.Matrix([[1, sp.I], [sp.I, 1]]) / sp.sqrt(2)
rho = representation(color_rotation)
require_zero(rho*rho.conjugate().T-sp.eye(dimension), "Unitary oscillator representation")
rotated_projectors = [color_rotation*projector*color_rotation.conjugate().T
                      for projector in (projector_0, projector_1)]
for is_contraction, original, label in [(False, p_full, "P"), (True, q_full, "Q")]:
    rotated = spectral_spin(*rotated_projectors, is_contraction)
    require_zero(rotated-rho*original*rho.conjugate().T, f"{label} spectral covariance")

weyl = representation(sp.Matrix([[0, 1], [1, 0]]))
require_zero(restrict(weyl)+p, "det^-1 gives Weyl -P")
cartan = representation(sp.diag(sp.I, 1))
require_zero(restrict(cartan)-sp.eye(d*d), "One-per-site Cartan invariance")
assert sum(cartan[index, index] == 1 for index in range(dimension)) == d*d

lam = sp.symbols("lambda", real=True)
engineered = (lam**2-1)*sp.eye(d*d)-(lam+1)*p+lam*q
target = lam**2*sp.eye(d*d)-lam*p+lam*q
require_zero(sp.eye(d*d)+p+engineered-target, "Exact target pair coupling")

x = sp.symbols("x0:3", real=True)
vandermonde = sp.prod(x[j]-x[i] for i in range(3) for j in range(i+1, 3))
assert sp.simplify(sum(sp.diff(vandermonde, coordinate, 2) for coordinate in x)) == 0

results = {
    "checked_N": N,
    "checked_d": d,
    "oscillator_sector_dimension": dimension,
    "Cartan_invariant_dimension": len(physical_indices),
    "checks": ["P,Q projector algebra", "normal-ordered EijEji=I+P",
               "standard angular factor", "spectral P gauge covariance",
               "spectral Q gauge covariance", "determinant-twisted Weyl sign",
               "Cartan occupation constraint", "engineered potential exact reduction",
               "harmonic Vandermonde measure cancellation"],
    "all_passed": True,
    "scope": "Finite oscillator/matrix reduction identities, not duality or scattering.",
}
destination = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_matrix_uplift_checks.json')
destination.write_text(json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
