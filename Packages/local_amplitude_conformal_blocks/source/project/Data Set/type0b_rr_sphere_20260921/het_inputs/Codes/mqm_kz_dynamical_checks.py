#!/usr/bin/env python3
"""Exact low-site KZ and dynamical-parent tests; no closed-string fit."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import sympy as s


def canon(pairing):
    return tuple(sorted(tuple(sorted(pair)) for pair in pairing))


def brauer_pairing_matrices(n):
    basis = [
        canon(((0, 1), (2, 3))),
        canon(((0, 2), (1, 3))),
        canon(((0, 3), (1, 2))),
    ]
    idx = {p: k for k, p in enumerate(basis)}
    perms, contractions = {}, {}
    for i, j in itertools.combinations(range(4), 2):
        pm, qm = s.zeros(3), s.zeros(3)
        swap = lambda a: j if a == i else i if a == j else a
        for col, pairing in enumerate(basis):
            out = canon(tuple((swap(a), swap(b)) for a, b in pairing))
            pm[idx[out], col] = 1
            partners = {a: b for edge in pairing for a, b in (edge, edge[::-1])}
            if partners[i] == j:
                qm[col, col] = n
            else:
                out = canon(((i, j), (partners[i], partners[j])))
                qm[idx[out], col] = 1
        perms[i, j] = pm
        contractions[i, j] = qm
    return basis, perms, contractions


def zero(mat, label):
    mat = mat.applyfunc(s.factor)
    if mat != s.zeros(*mat.shape):
        raise AssertionError((label, mat))


def main():
    n = s.symbols("n", integer=True, positive=True)
    alpha = 1 / (n - 1)
    x = s.symbols("x0:4", real=True)
    basis, perms, contractions = brauer_pairing_matrices(n)
    omega = {pair: perms[pair] - contractions[pair] for pair in perms}
    gram = s.Matrix(3, 3, lambda i, j: n**2 if i == j else n)
    for pair in perms:
        P, Q, O = perms[pair], contractions[pair], omega[pair]
        zero(P**2 - s.eye(3), "P squared")
        zero(Q**2 - n * Q, "Q squared")
        zero(P * Q - Q, "PQ")
        zero(O**2 - s.eye(3) - (n - 2) * Q, "Omega squared")
        zero(gram * O - O.T * gram, "physical Hermiticity")
    zero(sum(omega.values(), s.zeros(3)) + 2 * (n - 1) * s.eye(3),
         "four-site singlet Casimir")
    coeff = s.Matrix([
        1 / ((x[0] - x[1]) * (x[2] - x[3])),
        -1 / ((x[0] - x[2]) * (x[1] - x[3])),
        1 / ((x[0] - x[3]) * (x[1] - x[2])),
    ])
    # Wick signs and all flavor contractions are encoded in this independent
    # invariant-tensor basis; it remains linearly independent at n=23,24.
    connection = []
    for i in range(4):
        A = alpha * sum(
            (omega[tuple(sorted((i, j)))] / (x[i] - x[j])
             for j in range(4) if j != i),
            s.zeros(3),
        )
        connection.append(A)
        zero(coeff.diff(x[i]) - A * coeff, f"N4 KZ at site {i}")
    for i, j in itertools.combinations(range(4), 2):
        zero(-connection[j].diff(x[i]) + connection[i].diff(x[j])
             + connection[i] * connection[j] - connection[j] * connection[i],
             "flat KZ connection")

    vandermonde = s.prod(x[j] - x[i] for i, j in itertools.combinations(range(4), 2))
    polynomial = (vandermonde * coeff).applyfunc(s.cancel)
    for value in polynomial:
        assert s.Poly(value, *x).total_degree() == 4
    norm_polynomial = s.Poly(s.expand((polynomial.T * gram * polynomial)[0]), *x)
    gaussian_norm_over_pi_squared = s.Integer(0)
    for powers, coefficient in norm_polynomial.terms():
        if any(power % 2 for power in powers):
            continue
        moment = s.prod(s.factorial2(power - 1) / 2**(power // 2)
                        if power else s.Integer(1) for power in powers)
        gaussian_norm_over_pi_squared += coefficient * moment
    gaussian_norm_over_pi_squared = s.factor(gaussian_norm_over_pi_squared)
    for dim in [23, 24]:
        assert gaussian_norm_over_pi_squared.subs(n, dim) > 0
    outputs = {}
    sample = dict(zip(x, [0, 1, 3, 6]))
    for g in [s.Integer(0), s.Integer(1)]:
        As = [
            connection[i] + g * sum(
                (1 / (x[i] - x[j]) for j in range(4) if j != i), s.Integer(0)
            ) * s.eye(3) for i in range(4)
        ]
        # Keep the KZ identity symbolic in all four coordinates above.
        # Check the separately expanded Hamiltonian at exact rational sites,
        # with flavor dimension still symbolic, to avoid huge redundant
        # multivariate rational-expression expansion.
        potential = sum((A.diff(x[i]).subs(sample) + A.subs(sample)**2
                         for i, A in enumerate(As)), s.zeros(3)) / 2
        pair_potential = s.zeros(3)
        for (i, j), O in omega.items():
            B = g * s.eye(3) + alpha * O
            pair_potential += (B * B - B) / (sample[x[i]] - sample[x[j]])**2
        three = (potential - pair_potential).applyfunc(s.factor)
        direct_three = s.zeros(3)
        for i in range(4):
            for j, k in itertools.combinations([a for a in range(4) if a != i], 2):
                Bj = g * s.eye(3) + alpha * omega[tuple(sorted((i, j)))]
                Bk = g * s.eye(3) + alpha * omega[tuple(sorted((i, k)))]
                direct_three += (Bj * Bk + Bk * Bj) / (
                    2 * (sample[x[i]] - sample[x[j]])
                    * (sample[x[i]] - sample[x[k]]))
        zero(three - direct_three, "three-site decomposition")
        for dim in [23, 24]:
            at = three.subs(n, dim)
            assert at != s.zeros(3)
            zero(gram.subs(n, dim) * at - at.T * gram.subs(n, dim),
                 "three-site Hermiticity")
            outputs[f"g={g},n={dim}"] = {
                "three_site_matrix_x_0_1_3_6": [[str(v) for v in row]
                                              for row in at.tolist()],
                "three_site_nonzero_witness_00": str(at[0, 0]),
            }
        if g == 0:
            zero(-sum((coeff.diff(xi, 2).subs(sample) for xi in x),
                      s.zeros(3, 1)) / 2
                 + potential * coeff.subs(sample), "formal unweighted zero energy")
        else:
            for i in range(4):
                zero(polynomial.diff(x[i]) - As[i] * polynomial,
                     "flattened polynomial KZ state")
            zero(-sum((polynomial.diff(xi, 2).subs(sample) for xi in x),
                      s.zeros(3, 1)) / 2
                 + potential * polynomial.subs(sample), "flat matrix-measure zero energy")
            # Gaussian state e^{-w |x|^2/2} has homogeneous polynomial
            # degree 4, hence oscillator energy w(4+N/2)=6w under
            # H0+w²|x|²/2, equivalently zero under the covariant square.
            euler = sum((x[i] * polynomial.diff(x[i]) for i in range(4)),
                        s.zeros(3, 1))
            zero(euler - 4 * polynomial, "polynomial homogeneity")
            total_x_A = sum((x[i] * As[i] for i in range(4)), s.zeros(3))
            zero(total_x_A - 4 * s.eye(3), "four-site covariant trap shift")

    pair_channels = {}
    for dim in [23, 24]:
        a = s.Rational(1, dim - 1)
        pair_channels[dim] = {}
        for g in [0, 1]:
            powers = {"symmetric_traceless": g + a,
                      "antisymmetric": g - a, "trace": g - 1}
            pair_channels[dim][g] = {
                name: {"D_zero_power": str(b),
                       "inverse_square_coefficient": str(b * (b - 1))}
                for name, b in powers.items()
            }
        # N2: alpha Omega times delta is -delta, independent of n.
        assert a * (1 - dim) == -1
    result = {
        "all_checks_passed": True,
        "basis": str(basis),
        "n4_wick_coefficients": [str(v) for v in coeff],
        "n4_vandermonde_polynomial": [str(s.factor(v)) for v in polynomial],
        "n4_gaussian_norm_full_R4_over_pi_squared":
            str(gaussian_norm_over_pi_squared),
        "n4_gaussian_norm_ordered_chamber_over_pi_squared":
            str(s.factor(gaussian_norm_over_pi_squared / s.factorial(4))),
        "symbolic_n_KZ": "all four derivatives identically zero",
        "symbolic_n_curvature": "all six commutators identically zero",
        "KZ_couplings": {"n23": "1/22", "n24": "1/23"},
        "Crampe_couplings": {"n23": "2/19", "n24": "1/10"},
        "pair_channels": pair_channels,
        "three_site": outputs,
        "scope": "Fixed level-one organizing principle; no Liouville wall, sea, or closed-string amplitudes derived.",
    }
    target = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_dynamical_results.json')
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "output": str(target),
                      "three_site_witnesses": {
                          k: v["three_site_nonzero_witness_00"] for k, v in outputs.items()
                      }}, indent=2))


if __name__ == "__main__":
    main()
