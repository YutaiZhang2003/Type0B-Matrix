#!/usr/bin/env python3
"""Ramond local-field projection and the signed-NS reflection test.

This is a symbolic bookkeeping check, not a Ward-identity evaluator.  It
starts from the local-field parity vectors obtained by applying the paired
holomorphic/antiholomorphic chi strings to

    Phi^(+1/2) = sigma^+ Sigma^+ + sigma^- Sigma^- .

Together with the exact discrete dependence of the chiral restrictions, it
derives the quadratic combination of the two Ramond parity-copy masters that
is visible to a local Phi correlator.  It also checks that changing the sign
of any local Ramond representative only multiplies that combination by an
overall sign.  Finally it records why Phi^(+k_1) and Phi^(-k_1), for integral
k_1, do not give two equations for the same chiral masters: they belong to
the two reflected NS branches n_1 and -n_1.
"""

from __future__ import annotations

import sympy as sp


I = sp.I
T = (1 + I) / sp.sqrt(2)  # exp(i*pi/4)


def local_ramond_vector(sign: int, mode_count: int) -> sp.Matrix:
    """Coefficient vector in the (epsilon=0,epsilon=1) branch basis.

    ``sign`` is the sign of k=sign*(M+1/2).  One common normalization of the
    complete local leg is suppressed, since it multiplies both entries and
    cannot change the rank of the projection.
    """

    sign = int(sign)
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    mode_count = int(mode_count)
    parity = mode_count % 2
    answer = [sp.Integer(0), sp.Integer(0)]
    answer[parity] = -sign * T
    answer[1 - parity] = 2 * sign * (-1) ** mode_count / T
    return sp.Matrix(answer)


def restriction_matrix(mode_count_2: int, mode_count_3: int, eta: int, f: int):
    """Return A_f[e_2,e_3]/R_{e_2} for the four parity-copy choices."""

    mode_count_2 = int(mode_count_2)
    mode_count_3 = int(mode_count_3)
    eta = int(eta)
    f = int(f)
    r3 = sp.Pow(2, sp.Rational((-1) ** (mode_count_3 + 1), 2))
    answer = sp.zeros(2, 2)
    for epsilon2 in (0, 1):
        phase = eta * (-1) ** (mode_count_2 + 1 + epsilon2) / T
        for epsilon3 in (0, 1):
            answer[epsilon2, epsilon3] = (
                r3**epsilon3
                * (-1) ** (epsilon3 * f)
                * phase**f
            )
    return answer


def quadratic_weights(mode_count_2: int, mode_count_3: int, sign2=1, sign3=1):
    """Coefficients of ((R_0)^2,(R_1)^2) in the local correlator."""

    c2 = local_ramond_vector(sign2, mode_count_2)
    c3 = local_ramond_vector(sign3, mode_count_3)
    # eta drops out after the Hadamard squares, so eta=+1 is sufficient.
    matrices = [
        restriction_matrix(mode_count_2, mode_count_3, 1, f)
        for f in (0, 1)
    ]
    weights = []
    for epsilon2 in (0, 1):
        coefficient = 0
        for epsilon3 in (0, 1):
            coefficient += c2[epsilon2] * c3[epsilon3] * sum(
                matrix[epsilon2, epsilon3] ** 2 for matrix in matrices
            )
        weights.append(sp.simplify(sp.expand_complex(coefficient)))
    return sp.Matrix(weights)


def canonical_row(mode_count_2: int, mode_count_3: int) -> sp.Matrix:
    """Return the row in the phase convention used in the accompanying note."""

    rows = {
        (0, 0): sp.Matrix([2 * I, -4]),
        (0, 1): sp.Matrix([4, 8 * I]),
        (1, 0): sp.Matrix([4, 2 * I]),
        (1, 1): sp.Matrix([-8 * I, 4]),
    }
    return rows[(int(mode_count_2) % 2, int(mode_count_3) % 2)]


def proportional(first: sp.Matrix, second: sp.Matrix) -> bool:
    """Exact proportionality test for nonzero two-component columns."""

    return sp.simplify(first[0] * second[1] - first[1] * second[0]) == 0


def audit() -> None:
    for mode_count_2 in (0, 1, 2, 3):
        for mode_count_3 in (0, 1, 2, 3):
            direct = quadratic_weights(mode_count_2, mode_count_3)
            expected = canonical_row(mode_count_2, mode_count_3)
            if not proportional(direct, expected):
                raise AssertionError(
                    (mode_count_2, mode_count_3, direct, expected)
                )

            # c_{-,M}=-c_{+,M}; changing either signed local Ramond
            # representative therefore changes only the common scale.
            signed_rows = [
                quadratic_weights(mode_count_2, mode_count_3, sign2, sign3)
                for sign2 in (-1, 1)
                for sign3 in (-1, 1)
            ]
            if not all(proportional(direct, row) for row in signed_rows):
                raise AssertionError("A signed Ramond field changed the row")

            print(
                f"(M2 mod 2,M3 mod 2)="
                f"({mode_count_2 % 2},{mode_count_3 % 2}) "
                f"row={tuple(expected)} signed-rank=1"
            )

    print(
        "signed NS representatives: Phi^(+2n) uses W_n(P), while "
        "Phi^(-2n)=Phi_(Q-alpha)^(+2n) uses W_n(-P)=W_-n(P); "
        "they evaluate two reflected branch summands and do not supply "
        "a second row at fixed (n,P)."
    )


if __name__ == "__main__":
    audit()
