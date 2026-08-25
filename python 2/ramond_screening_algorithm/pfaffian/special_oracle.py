"""Primary NS--R--R screening products and the maximal-screening test.

This module is independent of the super-Virasoro Ward evaluator.  It fixes
the puncture ordering and isolates precisely what the primary BFL integral
does and does not compute for a branched external state.
"""

from __future__ import annotations

import sympy as sp

from ..n2_selberg import J_dual, J_primal


def ordinary_selberg(screenings, A, B, g):
    """The Selberg integral over ``[0,1]**screenings``."""

    screenings = int(screenings)
    answer = sp.Integer(1)
    for index in range(screenings):
        answer *= (
            sp.gamma(1 + (index + 1) * g)
            * sp.gamma(1 + A + index * g)
            * sp.gamma(1 + B + index * g)
        )
        answer /= sp.gamma(1 + g) * sp.gamma(
            2 + A + B + (screenings + index - 1) * g
        )
    return answer


def physical_nsrr_selberg(screenings, A, B, g):
    """The BFL two-spin integral in the repository puncture order.

    The order-field labels are ``(1,1,0)`` because the punctures
    ``(0,1,infinity)`` contain ``(R,R,NS)`` states.  Odd screening number
    is the primal ``k=1`` sequence; even screening number is the dual
    ``k=0`` sequence.
    """

    screenings = int(screenings)
    if screenings < 0:
        raise ValueError(screenings)
    if screenings == 0:
        return sp.Integer(1)
    if screenings % 2:
        return J_primal((screenings + 1) // 2, 2, (1, 1, 0), A, B, g)
    return J_dual(screenings // 2, 2, (1, 1, 0), A, B, g)


def primary_shift_ratio(n1, n2, n3, b, p2, p3):
    """Return the tempting but generally incomplete shifted-primary ratio.

    It is evaluated on the maximal-screening hyperplane

      Q/2 + P1 + P2 + P3 = -2(n1+n2+n3)b.

    For the ground state and the first crossed test this is the complete
    free-field value in one spin channel.  Starting with longer external
    staircases, an additional Pfaffian minor (an Uglov/staircase insertion)
    is required.  The function is named ``primary_shift_ratio`` rather than
    ``branching_oracle`` to prevent that missing datum from being hidden.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    screenings = 2 * (n1 + n2 + n3)
    if not screenings.is_integer:
        raise ValueError("the maximal screening number is not integral")
    screenings = int(screenings)
    q = b + 1 / b
    A = -b * (q / 2 + p3) - sp.Rational(1, 2)
    B = -b * (q / 2 + p2) - sp.Rational(1, 2)
    g = -b * q / 2
    numerator = ordinary_selberg(
        screenings,
        -b * (q / 2 + p3) - 2 * n3,
        -b * (q / 2 + p2) - 2 * n2,
        g + 1,
    )
    denominator = physical_nsrr_selberg(screenings, A, B, g)
    return sp.factor(sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True))


def audit_ground_and_hard():
    """Ward-free exact checks at ``(0,1/4,1/4)`` and ``(0,3/4,3/4)``."""

    b, p2, p3 = sp.symbols("b P_2 P_3", nonzero=True)
    q = b + 1 / b

    # Direct one- and two-screening checks of the (1,1,0) placement.
    A, B, g = sp.symbols("A B g")
    beta = sp.gamma(1 + A) * sp.gamma(1 + B) / sp.gamma(2 + A + B)
    assert sp.factor(sp.cancel(sp.expand_func(physical_nsrr_selberg(1, A, B, g) / beta) - 1)) == 0
    selberg_two = ordinary_selberg(2, A, B, g)
    mean_t = (1 + A + g) / (2 + A + B + 2 * g)
    mean_product = ordinary_selberg(2, A + 1, B, g) / selberg_two
    direct_two = selberg_two * (mean_t - mean_product)
    assert sp.factor(
        sp.cancel(sp.expand_func(physical_nsrr_selberg(2, A, B, g) / direct_two) - 1)
    ) == 0

    ground = primary_shift_ratio(0, sp.Rational(1, 4), sp.Rational(1, 4), b, p2, p3)
    assert sp.factor(ground - 1) == 0

    ratio = primary_shift_ratio(0, sp.Rational(3, 4), sp.Rational(3, 4), b, p2, p3)
    p1 = -3 * b - q / 2 - p2 - p3
    xpp = q / 2 + p1 + p2 + p3
    xmm = q / 2 + p1 - p2 - p3

    # Only the fixed odd normalization 2^(1/8) enters ell(x,3).
    ell3 = lambda x: sp.Pow(2, sp.Rational(1, 8)) * (x + b) * (x + 1 / b)
    real_eta_plus = -ell3(xpp) * ell3(q - xmm) / (
        ell3(q + 2 * p2) * ell3(q + 2 * p3)
    )
    assert sp.factor(sp.cancel(ratio + 2 * real_eta_plus)) == 0
    print("ground maximal-screening ratio: 1")
    print("hard maximal-screening ratio: -2 R_0^(+)/(1+i)")


if __name__ == "__main__":
    audit_ground_and_hard()
