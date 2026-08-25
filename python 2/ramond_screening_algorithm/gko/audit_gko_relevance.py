#!/usr/bin/env python3
"""Exact scope audit for the two papers confused in the working notes.

``arXiv:2404.14350`` is *Highest-weight vectors and three-point
functions in GKO coset decomposition*.  Its Theorem 4.5 is implemented by
``gko_ratio`` below.  ``arXiv:2505.23122`` is instead the Ramond
super-minimal-Liouville-gravity paper.  Its two-channel contraction is
implemented by ``physical_two_channel``.

The checks deliberately use only the published finite products and the
closed hard-level polynomials already recorded in the repository.  No Ward
recursion is imported here.
"""

from __future__ import annotations

import sympy as sp


def triangle(index, alpha, epsilon_1, epsilon_2):
    """The integral-triangle product t_n^{epsilon_1,epsilon_2}(alpha)."""

    index = int(index)
    if index > 0:
        return sp.prod(
            alpha - i * epsilon_1 - j * epsilon_2
            for i in range(index)
            for j in range(index - i)
        )
    if index < 0:
        return sp.prod(
            alpha + i * epsilon_1 + j * epsilon_2
            for i in range(1, -index)
            for j in range(1, -index - i + 1)
        )
    return sp.Integer(1)


def gko_ratio(m, n, ell, mu, nu, lam, kappa):
    """Unsigned product in Theorem 4.5 of arXiv:2404.14350.

    The theorem's exact sign is separate from this rational product.  The
    unsigned part is sufficient for the channel-count and factorization
    audit performed here.
    """

    e1 = sp.Integer(1)
    e2 = -1 / kappa
    numerator = (
        triangle(-ell - m - n, (2 + lam + mu + nu) / (-2 * kappa), e1, e2)
        * triangle(-ell + m - n, (lam - mu + nu) / (-2 * kappa), e1, e2)
        * triangle(-ell - m + n, (lam + mu - nu) / (-2 * kappa), e1, e2)
        * triangle(ell - m - n, (-lam + mu + nu) / (-2 * kappa), e1, e2)
    )
    denominator = (
        triangle(-2 * ell, (lam + 1) / (-kappa), e1, e2)
        * triangle(-2 * m, (mu + 1) / (-kappa), e1, e2)
        * triangle(-2 * n, (nu + 1) / (-kappa), e1, e2)
    )
    return sp.factor(sp.cancel(numerator / denominator))


def full_triangle(x, index, b):
    """Full b,b^{-1} lattice triangle with positive index."""

    return sp.prod(
        x + r * b + s / b
        for r in range(index)
        for s in range(index - r)
    )


def ell_product(x, index, b):
    """The checkerboard ell product, omitting its conventional 2^(1/8)."""

    parity = index % 2
    return sp.prod(
        x + r * b + s / b
        for r in range(index)
        for s in range(index - r)
        if (r + s) % 2 == parity
    )


def checkerboard_identity(max_index=9):
    """Verify T_m=ell_m ell_{m+1} without the odd 2^(1/8) convention."""

    x, b = sp.symbols("x b", nonzero=True)
    for index in range(1, max_index + 1):
        residual = sp.cancel(
            full_triangle(x, index, b)
            - ell_product(x, index, b) * ell_product(x, index + 1, b)
        )
        if residual != 0:
            raise AssertionError((index, sp.factor(residual)))


def physical_two_channel(a_l, b_l, a_m, b_m):
    """Equation (3.30) of arXiv:2505.23122 in A/B channel variables."""

    c_l_plus = (a_l + b_l) / 2
    c_l_minus = (a_l - b_l) / 2
    c_m_plus = (a_m + b_m) / 2
    c_m_minus = (a_m - b_m) / 2
    return sp.expand(c_l_plus * c_m_minus - c_l_minus * c_m_plus)


def hard_channel_obstruction():
    """Compare the one-channel product with the first crossed master.

    K is the factorized eta=+ numerator and H is the eta=- numerator at
    (n_1,n_2,n_3)=(0,3/4,3/4), with the common leg denominators removed.
    """

    q, p1, p2, p3 = sp.symbols("Q P1 P2 P3")
    x_plus = q / 2 + p1 + p2 + p3
    x_reflected = q / 2 - p1 + p2 + p3
    k_factorized = sp.expand(
        (x_plus**2 + q * x_plus + 1)
        * (x_reflected**2 + q * x_reflected + 1)
    )
    leg2 = q + 2 * p2
    leg3 = q + 2 * p3
    d2 = leg2**2 + q * leg2 + 1
    d3 = leg3**2 + q * leg3 + 1
    crossed_line = sp.expand(x_plus * (x_reflected - q))
    h_crossed = sp.expand(
        crossed_line**2
        + 2 * crossed_line * (1 + leg2 * leg3)
        + d2 * d3
    )
    if not sp.Poly(k_factorized, p1, extension=True).is_sqf:
        raise AssertionError("unexpected repeated factor in K")
    if not sp.Poly(h_crossed, q, p1, p2, p3).is_irreducible:
        raise AssertionError("the crossed hard numerator unexpectedly factors")
    return (q, p1, p2, p3), sp.factor(k_factorized), h_crossed, d2, d3


def hard_gko_example():
    """Evaluate the GKO product at labels matching the hard branch lattice.

    This is not asserted to be a physical momentum map.  It records the
    exact discrete identification m=2*n1, n=2*n2, ell=2*n3 and confirms
    that the published formula is a single scalar rational product.
    """

    b, p1, p2, p3 = sp.symbols("b P1 P2 P3", nonzero=True)
    kappa = -b**2
    mu = -1 + 2 * b * p1
    nu = -1 + 2 * b * p2
    lam = -1 + 2 * b * p3
    value = gko_ratio(0, sp.Rational(3, 2), sp.Rational(3, 2), mu, nu, lam, kappa)
    return (b, p1, p2, p3), sp.factor(value)


def natural_map_mismatch():
    """Prove that the most direct GKO-to-super-Liouville map is not a match.

    The lattice spacings force kappa=-b^2 (up to exchanging b and b^{-1}).
    The affine Weyl-shift convention then suggests

        (mu,nu,lambda)=(-1+2 b P1,-1+2 b P2,-1+2 b P3).

    At the first crossed level the result is not even the factorized eta=+
    master: their quotient depends on P1.  This prevents using the GKO
    theorem verbatim, before any discussion of the missing eta=- channel.
    """

    (b, p1, p2, p3), gko_value = hard_gko_example()
    (q, hp1, hp2, hp3), factorized, _, d2, d3 = hard_channel_obstruction()
    expected = sp.cancel(
        factorized.subs({q: b + 1 / b, hp1: p1, hp2: p2, hp3: p3})
        / (d2 * d3).subs({q: b + 1 / b, hp2: p2, hp3: p3})
    )
    quotient = sp.factor(sp.cancel(gko_value / expected))
    derivative = sp.factor(sp.diff(quotient, p1))
    if derivative == 0:
        raise AssertionError("the natural-map mismatch unexpectedly disappeared")
    return quotient, derivative


def main():
    checkerboard_identity()
    a_l, b_l, a_m, b_m = sp.symbols("A_L B_L A_M B_M")
    physical = physical_two_channel(a_l, b_l, a_m, b_m)
    expected = (b_l * a_m - a_l * b_m) / 2
    assert sp.expand(physical - expected) == 0
    _, factorized, crossed, _, _ = hard_channel_obstruction()
    _, gko_hard = hard_gko_example()
    _, mismatch_derivative = natural_map_mismatch()
    print("checkerboard/full-triangle identities: exact through m=9")
    print("2505.23122 physical contraction:", physical)
    print("hard eta=+ numerator factor count:", len(sp.factor_list(factorized)[1]))
    print("hard eta=- numerator irreducible:", sp.Poly(crossed).is_irreducible)
    print("2404.14350 hard-label GKO product:")
    print(gko_hard)
    print("natural-map quotient has nonzero P1 derivative:", mismatch_derivative != 0)
    print("scope: one GKO scalar channel cannot determine four Ramond masters")


if __name__ == "__main__":
    main()
