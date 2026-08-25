#!/usr/bin/env python3
"""Linear-step recurrence for the scalar GKO three-point coefficient.

This is the proposition labelled ``lrecurrence`` in the arXiv source of
arXiv:2404.14350, in the notation used by ``two_step_probe.py``.  It is
the scalable part of the GKO construction: changing one branch label by
one half requires six products over one lattice segment, rather than a
new highest-weight vector or a Gram-matrix inversion.

The recurrence determines the scalar/factorized Ramond channel.  It does
not supply the missing extension-space (Pfaffian/Uglov) component; see
``affine_fundamental_fusion.py`` for the exact rank-one obstruction.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from two_step_probe import minus_one_power, signed_gko_ratio  # noqa: E402


def segment(index, alpha, epsilon_1, epsilon_2):
    """Return the integral-segment product ``s_index(alpha)``.

    This is ``t_index/t_(index-1)`` written directly as a boundary
    product.  Its arithmetic cost is ``abs(index)`` factors.
    """

    index = int(index)
    if index > 0:
        return sp.prod(
            alpha - i * epsilon_1 - (index - 1 - i) * epsilon_2
            for i in range(index)
        )
    if index < 0:
        size = -index
        return sp.prod(
            alpha + i * epsilon_1 + (size + 1 - i) * epsilon_2
            for i in range(1, size + 1)
        ) ** -1
    return sp.Integer(1)


def ell_recurrence_ratio(m, n, ell, mu, nu, lam, kappa):
    r"""Return ``C_(ell-1/2)/C_(ell+1/2)`` exactly.

    The coefficient is the normalized, signed affine/GKO three-point
    function of Theorem 4.5.  All branch labels are half-integral and the
    displayed sign exponent must therefore be integral on an allowed
    three-point lattice.
    """

    m, n, ell = map(sp.Rational, (m, n, ell))
    epsilon_1 = sp.Integer(1)
    epsilon_2 = -1 / kappa
    sign = minus_one_power(-ell + m - n - sp.Rational(1, 2))

    numerator = (
        segment(
            -ell - m - n + sp.Rational(1, 2),
            -(2 + lam + mu + nu) / (2 * kappa),
            epsilon_1,
            epsilon_2,
        )
        * segment(
            -ell + m - n + sp.Rational(1, 2),
            -(lam - mu + nu) / (2 * kappa),
            epsilon_1,
            epsilon_2,
        )
        * segment(
            -ell - m + n + sp.Rational(1, 2),
            -(lam + mu - nu) / (2 * kappa),
            epsilon_1,
            epsilon_2,
        )
    )
    denominator = (
        segment(
            ell - m - n + sp.Rational(1, 2),
            -(-lam + mu + nu) / (2 * kappa),
            epsilon_1,
            epsilon_2,
        )
        * segment(
            -2 * ell + 1,
            -(lam + 1) / kappa,
            epsilon_1,
            epsilon_2,
        )
        * segment(
            -2 * ell,
            -(lam + 1) / kappa,
            epsilon_1,
            epsilon_2,
        )
    )
    return sp.factor(sp.cancel(sign * numerator / denominator))


def direct_ratio(m, n, ell, mu, nu, lam, kappa):
    """The same half-step ratio evaluated from the full triangle formula."""

    lower = signed_gko_ratio(
        m, n, sp.Rational(ell) - sp.Rational(1, 2), mu, nu, lam, kappa
    )
    upper = signed_gko_ratio(
        m, n, sp.Rational(ell) + sp.Rational(1, 2), mu, nu, lam, kappa
    )
    return sp.factor(sp.cancel(lower / upper))


def audit() -> None:
    """Check positive, zero, and negative segment indices exactly."""

    test_cases = (
        # (m,n,ell,mu,nu,lambda,kappa)
        (
            1,
            sp.Rational(1, 2),
            1,
            sp.Rational(2, 7),
            sp.Rational(3, 5),
            sp.Rational(4, 9),
            sp.Rational(7, 3),
        ),
        (
            sp.Rational(3, 2),
            1,
            1,
            sp.Rational(-2, 5),
            sp.Rational(5, 8),
            sp.Rational(7, 11),
            sp.Rational(8, 3),
        ),
        (
            -1,
            sp.Rational(1, 2),
            0,
            sp.Rational(4, 7),
            sp.Rational(-3, 8),
            sp.Rational(5, 12),
            sp.Rational(11, 4),
        ),
    )
    for case in test_cases:
        recurrence = ell_recurrence_ratio(*case)
        direct = direct_ratio(*case)
        residual = sp.factor(sp.cancel(recurrence - direct))
        if residual != 0:
            raise AssertionError((case, recurrence, direct, residual))
    print("scalar GKO half-step recurrence: exact on all audit cases")
    print("cost per half-step: O(max(|m|,|n|,|ell|)) arithmetic factors")


if __name__ == "__main__":
    audit()
