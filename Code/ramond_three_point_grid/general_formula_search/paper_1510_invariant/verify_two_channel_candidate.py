#!/usr/bin/env python3
"""Exact checks for the two-channel structure suggested by arXiv:1510.01773.

The paper itself is not a Ramond-block paper.  Equations (2.9), (2.10),
(3.21), (4.21), and (4.23) nevertheless isolate a two-dimensional space of
three-linear invariants and show that a branch shift either preserves or
exchanges its two Fourier components.  This script records that matrix
algebra and checks the momentum-dependent lift required by the first hard
NS--R--R master, (0, 3/4, 3/4).

Nothing here calls the PBW/Ward evaluator.  All certificates are polynomial
identities over Q[Q,P1,P2,P3].
"""

from __future__ import annotations

import sympy as sp


Q, P1, P2, P3 = sp.symbols("Q P1 P2 P3")


def paper_channel_matrices(total_branch: int):
    """Return the shift in the epsilon and chamber bases.

    In the ordered paper basis (S_0,S_{1/2}), Eq. (2.10) shifts epsilon by
    total_branch/2 modulo one.  Equation (2.9) says

        (S_0,S_{1/2})^T = [[1,1],[-1,1]] (S_A,S_B)^T.
    """

    fourier_from_chambers = sp.Matrix(((1, 1), (-1, 1)))
    exchange = sp.Matrix(((0, 1), (1, 0))) ** (total_branch % 2)
    chamber_shift = sp.simplify(
        fourier_from_chambers.inv() * exchange * fourier_from_chambers
    )
    expected_chamber = sp.diag((-1) ** total_branch, 1)
    if chamber_shift != expected_chamber:
        raise AssertionError((chamber_shift, expected_chamber))
    return exchange, chamber_shift


def hard_polynomials():
    """Return the factorized and crossed hard numerators K and H."""

    x_plus_plus = Q / 2 + P1 + P2 + P3
    x_minus_minus = Q / 2 + P1 - P2 - P3

    # 2^(-1/8) ell(x,3)=x^2+Q*x+1 and ell(x,-3)=ell(Q-x,3).
    reflected = Q - x_minus_minus
    factorized = sp.expand(
        (x_plus_plus**2 + Q * x_plus_plus + 1)
        * (reflected**2 + Q * reflected + 1)
    )

    crossed = sp.expand(x_plus_plus * (x_minus_minus - Q))
    even_second = Q + 2 * P2
    even_third = Q + 2 * P3
    odd_second = even_second**2 + Q * even_second + 1
    odd_third = even_third**2 + Q * even_third + 1
    hard = sp.expand(
        crossed**2
        + 2 * crossed * (1 + even_second * even_third)
        + odd_second * odd_third
    )
    return factorized, hard, crossed, even_second, even_third, odd_second, odd_third


def hard_channel_kernel():
    """Return the exact two-state kernel for the crossed hard numerator."""

    _, hard, crossed, even_second, even_third, odd_second, odd_third = (
        hard_polynomials()
    )
    kernel = sp.Matrix(
        (
            (odd_second * odd_third, 1 + even_second * even_third),
            (1 + even_second * even_third, 1),
        )
    )
    boundary = sp.Matrix((1, crossed))
    residual = sp.expand((boundary.T * kernel * boundary)[0] - hard)
    if residual != 0:
        raise AssertionError(residual)
    return kernel


def hard_holonomy_matrix():
    """Hadamard-transform the two fixed-eta hard numerators.

    The identification of the chamber basis with geometric holonomies is a
    candidate, not a statement in arXiv:1510.01773.  If it is made, the two
    fixed-eta eigenvalues K and H give this symmetric holonomy matrix.
    """

    factorized, hard, *_ = hard_polynomials()
    hadamard = sp.Matrix(((1, 1), (1, -1))) / sp.sqrt(2)
    matrix = sp.simplify(
        hadamard * sp.diag(factorized, hard) * hadamard.T
    )
    diagonalized = sp.simplify(hadamard.T * matrix * hadamard)
    if diagonalized != sp.diag(factorized, hard):
        raise AssertionError(diagonalized)
    return matrix


def main() -> None:
    even_fourier, even_chamber = paper_channel_matrices(2)
    odd_fourier, odd_chamber = paper_channel_matrices(3)
    kernel = hard_channel_kernel()
    holonomy = hard_holonomy_matrix()

    print("paper even shift in (S_0,S_1/2) basis:")
    sp.pprint(even_fourier)
    print("paper even shift in (S_A,S_B) basis:")
    sp.pprint(even_chamber)
    print("paper odd shift in (S_0,S_1/2) basis:")
    sp.pprint(odd_fourier)
    print("paper odd shift in (S_A,S_B) basis:")
    sp.pprint(odd_chamber)
    print("hard crossed two-state kernel (exact residual 0):")
    sp.pprint(kernel)
    factorized, hard, *_ = hard_polynomials()
    expected = sp.Matrix(
        (((factorized + hard) / 2, (factorized - hard) / 2),
         ((factorized - hard) / 2, (factorized + hard) / 2))
    )
    if sp.simplify(holonomy - expected) != sp.zeros(2):
        raise AssertionError("compact holonomy form failed")
    print("hard holonomy matrix: [[(K+H)/2,(K-H)/2],"
          "[(K-H)/2,(K+H)/2]] (exact residual 0)")


if __name__ == "__main__":
    main()
