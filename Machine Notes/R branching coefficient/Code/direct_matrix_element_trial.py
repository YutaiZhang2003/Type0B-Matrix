#!/usr/bin/env python3
"""Direct free-field matrix-element trial for the first crossed R channel.

The object evaluated here is the Coulomb-gas matrix element of the explicit
double-Virasoro branch fields, not a PBW expansion of an abstract SCA
three-form.  The external labels are

    (n1,n2,n3) = (0,3/4,3/4),  epsilon2=epsilon3=0.

For zero and two b-screenings the physical chiral structure is eta=-1; at
the natural three-screening node it is eta=+1.  The two-screening integrand
is obtained from the literal chi strings and the two-spin-field Pfaffian.
It is integrated by elementary-symmetric Selberg moments.  The resulting
closed functions are then compared with the independently organized
two-channel Coulomb polynomials H and K.

No SCA descendant three-form or PBW transition is used to obtain the
screened values in ``two_screening_matrix_element`` and
``three_screening_matrix_element``.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE_ROOT = ROOT / "python 2"


def _install_repository_namespace() -> None:
    """Expose the historical ``python 2`` directory as package ``python``."""

    if "python" in sys.modules:
        return
    package = types.ModuleType("python")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["python"] = package


_install_repository_namespace()

from python.ramond_screening_algorithm.pfaffian.hard_two_chart_certificate import (  # noqa: E402
    P1,
    P2,
    P3,
    Q,
    charge_channel_numerators,
    one_leg_coulomb_matrix,
)
from python.ramond_screening_algorithm.pfaffian.native_hard_screening import (  # noqa: E402
    hard_contour_polynomial,
    hard_screening_value,
)
from python.ramond_screening_algorithm.pfaffian.selberg_elementary import (  # noqa: E402
    normalized_elementary_product,
)
from python.ramond_screening_algorithm.pfaffian.special_oracle import (  # noqa: E402
    ordinary_selberg,
    physical_nsrr_selberg,
)


I = sp.I
SQRT2 = sp.sqrt(2)
R = sp.Rational


def two_screening_contour_polynomial(form_parity: int = 0) -> sp.Expr:
    """Denominator-cleared two-screening chi/Pfaffian insertion.

    The variables are ordered as returned by ``hard_contour_polynomial``.
    For ``f=0`` the displayed formula is

      (1-i)/2 (2 e2-e1) (e2^2-e1 e2+e1^2-3 e2).

    The odd-form answer differs by its fixed zero-mode phase.
    """

    variables, polynomial = hard_contour_polynomial(
        2, int(form_parity), -1
    )
    t0, t1 = variables
    e1 = t0 + t1
    e2 = t0 * t1
    core = (2 * e2 - e1) * (e2**2 - e1 * e2 + e1**2 - 3 * e2)
    phase = (1 - I) / 2 if int(form_parity) == 0 else SQRT2 * I / 2
    expected = sp.expand(phase * core)
    residual = sp.expand(polynomial - expected)
    if residual != 0:
        raise AssertionError(residual)
    return expected


def three_screening_contour_polynomial(form_parity: int = 0) -> sp.Expr:
    """The natural three-screening insertion is a pure Vandermonde square."""

    variables, polynomial = hard_contour_polynomial(
        3, int(form_parity), 1
    )
    delta = sp.prod(
        variables[left] - variables[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    if int(form_parity) == 0:
        coefficient = -SQRT2 * (1 + I) / 4
    elif int(form_parity) == 1:
        coefficient = -sp.Rational(1, 2)
    else:
        raise ValueError("form_parity must be 0 or 1")
    expected = sp.expand(coefficient * delta**2)
    residual = sp.expand(polynomial - expected)
    if residual != 0:
        raise AssertionError(residual)
    return expected


def _two_screening_normalized_moment(A, B, g) -> sp.Expr:
    """Selberg average of the phase-free two-screening polynomial."""

    # Expansion of
    # (2 e2-e1)(e2^2-e1 e2+e1^2-3 e2).
    elementary_terms = (
        (2, (2, 2, 2)),
        (-3, (1, 2, 2)),
        (3, (1, 1, 2)),
        (-1, (1, 1, 1)),
        (-6, (2, 2)),
        (3, (1, 2)),
    )
    return sp.factor(
        sum(
            coefficient
            * normalized_elementary_product(indices, 2, A, B, g)
            for coefficient, indices in elementary_terms
        )
    )


def two_screening_remainder(b, p2, p3) -> sp.Expr:
    """The phase-free eta=- matrix element on the N=2 neutrality plane."""

    b, p2, p3 = map(sp.sympify, (b, p2, p3))
    q = b + 1 / b
    A = -b * (q / 2 + p3) - R(1, 2)
    B = -b * (q / 2 + p2) - R(1, 2)
    g = -b * q / 2
    average = _two_screening_normalized_moment(A - 1, B - 1, g)
    # The phase-free core has normalized integral ``-2 R_2``.  The actual
    # f=0 contour prefactor is ``(1-i)/2``, giving ``(-1+i) R_2``.
    ratio = -sp.Rational(1, 2) * (
        average
        * ordinary_selberg(2, A - 1, B - 1, g)
        / physical_nsrr_selberg(2, A, B, g)
    )
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(ratio)), force=True)
    )


def two_screening_matrix_element(b, p2, p3, form_parity: int = 0) -> sp.Expr:
    """Direct normalized matrix element at N=2 and eta=-1."""

    remainder = two_screening_remainder(b, p2, p3)
    if int(form_parity) == 0:
        return sp.factor((-1 + I) * remainder)
    if int(form_parity) == 1:
        return sp.factor(-SQRT2 * I * remainder)
    raise ValueError("form_parity must be 0 or 1")


def three_screening_remainder(b, p2, p3) -> sp.Expr:
    """Phase-free natural-node value at N=3 and eta=+1."""

    b, p2, p3 = map(sp.sympify, (b, p2, p3))
    numerator = (
        b**2
        * (3 * b**2 - 1)
        * (2 * b**2 + b * p2 + b * p3 + 1)
        * (5 * b**2 + 2 * b * p2 + 2 * b * p3 + 1)
    )
    denominator = (
        (b**2 + 2 * b * p2 + 2)
        * (b**2 + 2 * b * p3 + 2)
        * (2 * b**2 + 2 * b * p2 + 1)
        * (2 * b**2 + 2 * b * p3 + 1)
    )
    return sp.factor(numerator / denominator)


def three_screening_matrix_element(
    b, p2, p3, form_parity: int = 0
) -> sp.Expr:
    """Direct normalized matrix element at N=3 and eta=+1."""

    remainder = three_screening_remainder(b, p2, p3)
    if int(form_parity) == 0:
        return sp.factor(-4 * (1 + I) * remainder)
    if int(form_parity) == 1:
        return sp.factor(-4 * SQRT2 * remainder)
    raise ValueError("form_parity must be 0 or 1")


def hard_channel_restrictions(b, p2, p3) -> tuple[sp.Expr, sp.Expr]:
    """Return H/(d2 d3) at N=2 and K/(d2 d3) at N=3."""

    b, p2, p3 = map(sp.sympify, (b, p2, p3))
    q = b + 1 / b
    ordinary, crossed = charge_channel_numerators()
    d2 = one_leg_coulomb_matrix(P2)[0, 0]
    d3 = one_leg_coulomb_matrix(P3)[0, 0]
    common = {Q: q, P2: p2, P3: p3}
    n2_plane = {
        **common,
        P1: -q / 2 - p2 - p3 - 2 * b,
    }
    n3_plane = {
        **common,
        P1: -q / 2 - p2 - p3 - 3 * b,
    }
    crossed_n2 = sp.factor(
        sp.cancel((crossed / (d2 * d3)).subs(n2_plane, simultaneous=True))
    )
    ordinary_n3 = sp.factor(
        sp.cancel((ordinary / (d2 * d3)).subs(n3_plane, simultaneous=True))
    )
    return crossed_n2, ordinary_n3


def audit() -> None:
    """Run the free-field and two-channel identities exactly."""

    b, p2, p3 = sp.symbols("b p2 p3", nonzero=True)
    two_screening_contour_polynomial(0)
    two_screening_contour_polynomial(1)
    three_screening_contour_polynomial(0)
    three_screening_contour_polynomial(1)

    direct_n2 = two_screening_remainder(b, p2, p3)
    direct_n3 = three_screening_remainder(b, p2, p3)
    crossed_n2, ordinary_n3 = hard_channel_restrictions(b, p2, p3)
    if sp.factor(sp.cancel(direct_n2 - crossed_n2)) != 0:
        raise AssertionError("the N=2 contour value did not reproduce H")
    if sp.factor(sp.cancel(4 * direct_n3 - ordinary_n3)) != 0:
        raise AssertionError("the N=3 contour value did not reproduce K")

    # Independent literal-Pfaffian checks at one exact rational point.
    sample = (R(3, 2), R(2, 5), R(3, 7))
    for screenings, eta, closed in (
        (2, -1, two_screening_matrix_element),
        (3, 1, three_screening_matrix_element),
    ):
        for form_parity in (0, 1):
            literal = hard_screening_value(
                screenings,
                form_parity,
                eta,
                *sample,
            )
            expected = closed(*sample, form_parity=form_parity)
            if sp.factor(sp.cancel(literal - expected)) != 0:
                raise AssertionError(
                    (screenings, form_parity, literal, expected)
                )

    print("direct N=2 matrix element: elementary Selberg evaluation exact")
    print("N=2 eta=- restriction: H/(d2*d3) residual zero")
    print("direct N=3 natural matrix element: determinant/Selberg exact")
    print("N=3 contour polynomial: a pure Delta^2 crossing singlet")
    print("N=3 eta=+ restriction: K/(d2*d3) residual zero")
    print("scope: two genuine screening hyperplanes, not generic interpolation")


if __name__ == "__main__":
    audit()
