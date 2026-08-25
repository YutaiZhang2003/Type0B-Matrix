#!/usr/bin/env python3
"""Exact N=2 parafermionic Selberg products.

This module implements the formulas labelled ``Int-g-rep``,
``Int-g-rep-dual``, and ``C-k-answer`` by Bershtein--Fateev--Litvinov,
arXiv:1011.4090.
It is the screening-integral backend needed by the NS--R--R branching
algorithm.  It deliberately contains no super-Virasoro states and imports no
Ward evaluator.

For the NS--R--R order-field labels (k1,k2,k3)=(0,1,1), an odd number
``n=2m-1`` of screenings uses ``J_primal`` with k=1, while an even number
``n=2m`` uses ``J_dual`` with k=0.  These are the two chiral Ising channels
which are summed in the local super-Liouville three-point function.
"""

from __future__ import annotations

import sympy as sp


def G(index: int, rank: int, x, g):
    """The finite Gamma product G_index^(rank)(x), equation (3.19)."""

    index = int(index)
    rank = int(rank)
    if not 0 <= index <= rank:
        raise ValueError((index, rank))
    answer = sp.Integer(1)
    for j in range(1, rank - index + 1):
        answer *= sp.gamma(x + (j - 1) * g)
    for j in range(rank - index + 1, rank + 1):
        answer *= sp.gamma(1 + x + (j - 1) * g)
    return answer


def C(m: int, rank: int, deficit: int, g):
    """The normalization C_m^(rank,deficit)(g), formula C-k-answer."""

    m, rank, deficit = map(int, (m, rank, deficit))
    if m < 1 or not 0 <= deficit < rank:
        raise ValueError((m, rank, deficit))
    n = m * rank - deficit
    answer = sp.factorial(n) * sp.sqrt(
        sp.Rational(sp.factorial(rank), rank**n)
        / (sp.factorial(deficit) * sp.factorial(rank - deficit))
    )
    answer *= G(0, rank - deficit, 1 + g, g) / sp.gamma(1 + g) ** n
    for p in range(1, m):
        answer *= G(
            rank - deficit,
            rank,
            p + (p * rank - deficit + 1) * g,
            g,
        )
    return answer


def J_primal(m: int, rank: int, labels, A, B, g):
    """The contour integral J_m, formula Int-g-rep."""

    m, rank = int(m), int(rank)
    k1, k2, k3 = map(int, labels)
    total = k1 + k2 + k3
    if total % 2:
        raise ValueError("The primal labels must have even sum.")
    deficit = total // 2
    if not all(0 <= item <= deficit for item in (k1, k2, k3)):
        raise ValueError((labels, deficit))
    if not deficit <= rank:
        raise ValueError((deficit, rank))

    answer = C(m, rank, deficit, g)
    for p in range(0, m - 1):
        common = 1 + A + B + ((m + p) * rank - deficit - 1) * g + m + p
        answer *= G(k1, rank, 1 + A + p * rank * g + p, g)
        answer *= G(k2, rank, 1 + B + p * rank * g + p, g)
        answer /= G(deficit - k3, rank, common, g)

    tail_A = m + A + (m - 1) * rank * g
    tail_B = m + B + (m - 1) * rank * g
    tail_AB = 2 * m + A + B + ((2 * m - 1) * rank - deficit - 1) * g
    answer *= G(0, rank - deficit, tail_A, g)
    answer *= G(0, rank - deficit, tail_B, g)
    answer /= G(0, rank - deficit, tail_AB, g)
    return answer


def J_dual(m: int, rank: int, labels, A, B, g):
    """The dual contour integral, formula Int-g-rep-dual."""

    m, rank = int(m), int(rank)
    k1, k2, k3 = map(int, labels)
    shifted = k1 + k2 + k3 - rank
    if shifted % 2:
        raise ValueError("The dual labels have the wrong parity.")
    deficit = shifted // 2
    if not all(deficit <= item <= rank for item in (k1, k2, k3)):
        raise ValueError((labels, deficit))
    if not 0 <= deficit < rank:
        raise ValueError((deficit, rank))

    answer = C(m, rank, deficit, g)
    for p in range(0, m - 1):
        common = 1 + A + B + ((m + p) * rank - deficit - 1) * g + m + p
        answer *= G(k1, rank, 1 + A + p * rank * g + p, g)
        answer *= G(k2, rank, 1 + B + p * rank * g + p, g)
        answer /= G(rank - k3 + deficit, rank, common, g)

    reduced_rank = rank - deficit
    tail_A = m + A + (m - 1) * rank * g
    tail_B = m + B + (m - 1) * rank * g
    tail_AB = 2 * m + A + B + ((2 * m - 1) * rank - deficit - 1) * g
    answer *= G(k1 - deficit, reduced_rank, tail_A, g)
    answer *= G(k2 - deficit, reduced_rank, tail_B, g)
    answer /= G(rank - k3, reduced_rank, tail_AB, g)
    return answer


def nsrr_order_channel(screenings: int, A, B, g):
    """Return the N=2 NS--R--R order-field screening integral.

    Odd screening number selects the primal Ising channel; even screening
    number selects the dual channel.  The zero-screening normalization is 1.
    """

    screenings = int(screenings)
    if screenings < 0:
        raise ValueError(screenings)
    if screenings == 0:
        return sp.Integer(1)
    labels = (0, 1, 1)
    if screenings % 2:
        return J_primal((screenings + 1) // 2, 2, labels, A, B, g)
    return J_dual(screenings // 2, 2, labels, A, B, g)


def _self_test():
    A, B, g = sp.symbols("A B g")
    one_screening = sp.simplify(nsrr_order_channel(1, A, B, g))
    expected = sp.gamma(1 + A) * sp.gamma(1 + B) / sp.gamma(2 + A + B)
    assert sp.simplify(one_screening - expected) == 0
    # For two screenings the dual N=2 polynomial is
    # 1-(t_1+t_2)/2.  Its Selberg average is elementary and supplies an
    # independent check of every shift in the dual formula.
    selberg_two = (
        sp.gamma(1 + A)
        * sp.gamma(1 + A + g)
        * sp.gamma(1 + B)
        * sp.gamma(1 + B + g)
        * sp.gamma(1 + 2 * g)
        / (
            sp.gamma(2 + A + B + g)
            * sp.gamma(2 + A + B + 2 * g)
            * sp.gamma(1 + g)
        )
    )
    expected_two = selberg_two * (1 + B + g) / (2 + A + B + 2 * g)
    two_screenings = nsrr_order_channel(2, A, B, g)
    assert sp.simplify(sp.expand_func(two_screenings / expected_two) - 1) == 0
    # Exercise both channels at higher screening number without expanding
    # the Gamma products.
    assert nsrr_order_channel(3, A, B, g) != 0
    assert nsrr_order_channel(4, A, B, g) != 0
    print("N=2 Selberg backend: primal/dual checks passed")


if __name__ == "__main__":
    _self_test()
