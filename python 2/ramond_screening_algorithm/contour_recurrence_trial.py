#!/usr/bin/env python3
"""Finite contour-orbit audit for the first crossed Ramond polynomial.

The two operations needed by the Dotsenko--Fateev chamber argument are

    R: t_i -> 1-t_i,
    I: F(t) -> (product_i t_i)^3 F(1/t).

For the denominator-cleared two-screening crossed insertion these generate
three, not two, independent polynomials.  ``R`` and ``I`` act as adjacent
transpositions, hence the orbit is the three-dimensional permutation
representation of S_3.  Its sum-zero subspace is a two-dimensional standard
representation and is the natural candidate for the physical two-form
system.  This file proves only the polynomial statement.
"""

from __future__ import annotations

import sympy as sp

from .pfaffian.selberg_elementary import normalized_elementary_product
from .pfaffian.special_oracle import ordinary_selberg
from .physical_matrix_recurrence_trial import first_physical_transfer


t1, t2 = sp.symbols("t_1 t_2")


def crossed_polynomial() -> sp.Expr:
    """Phase-free polynomial of the direct N=2 crossed matrix element."""

    e1 = t1 + t2
    e2 = t1 * t2
    return sp.expand(
        (2 * e2 - e1) * (e2**2 - e1 * e2 + e1**2 - 3 * e2)
    )


def endpoint_reflection(polynomial) -> sp.Expr:
    """Apply ``t_i -> 1-t_i`` simultaneously."""

    return sp.expand(
        sp.sympify(polynomial).subs(
            {t1: 1 - t1, t2: 1 - t2}, simultaneous=True
        )
    )


def inversion(polynomial) -> sp.Expr:
    """Apply the degree-three reciprocal transformation in each variable."""

    return sp.expand(
        (t1 * t2) ** 3
        * sp.sympify(polynomial).subs(
            {t1: 1 / t1, t2: 1 / t2}, simultaneous=True
        )
    )


def contour_orbit() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``(bulk,zero,one)`` in the S_3 permutation basis."""

    bulk = crossed_polynomial()
    zero = inversion(bulk)
    one = endpoint_reflection(zero)
    return tuple(map(sp.expand, (bulk, zero, one)))


def coefficient_rank(polynomials) -> int:
    """Rank of a polynomial list over the rational numbers."""

    polynomials = tuple(map(sp.expand, polynomials))
    monomials = sorted(
        set().union(
            *(set(sp.Poly(item, t1, t2).monoms()) for item in polynomials)
        )
    )
    matrix = sp.Matrix(
        [
            [
                sp.Poly(item, t1, t2).coeff_monomial(monomial)
                for item in polynomials
            ]
            for monomial in monomials
        ]
    )
    return int(matrix.rank())


def endpoint_order(polynomial, endpoint: int) -> int:
    """Common radial order when both variables approach one endpoint."""

    tau, xi = sp.symbols("tau xi")
    if int(endpoint) == 0:
        substitution = {t1: tau * xi, t2: tau * (1 - xi)}
    elif int(endpoint) == 1:
        substitution = {t1: 1 - tau * xi, t2: 1 - tau * (1 - xi)}
    else:
        raise ValueError("endpoint must be zero or one")
    scaled = sp.Poly(
        sp.expand(
            sp.sympify(polynomial).subs(substitution, simultaneous=True)
        ),
        tau,
    )
    return min(int(power[0]) for power, coefficient in scaled.terms())


def doublet_matrices() -> tuple[sp.Matrix, sp.Matrix]:
    r"""Matrices of ``I`` and ``R`` on ``(bulk-zero,zero-one)``."""

    inversion_matrix = sp.Matrix(((-1, 1), (0, 1)))
    reflection_matrix = sp.Matrix(((1, 0), (1, -1)))
    return inversion_matrix, reflection_matrix


def physical_doublet_change() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    r"""Conjugate the contour transposition to the physical eta frame.

    Returns ``(change, physical_reflection, universal_transfer)``.  The last
    matrix is the momentum-independent factor extracted from the PBW hard
    transfer.
    """

    inversion_matrix, _ = doublet_matrices()
    change = sp.Matrix(((1, 0), (sp.I, -sp.I)))
    physical_reflection = sp.simplify(
        change * inversion_matrix * change.inv()
    )
    _, _, universal_transfer = first_physical_transfer()
    return change, physical_reflection, universal_transfer


def _elementary_average(terms, A, B, g) -> sp.Expr:
    """Average ``sum c e1**a e2**b`` in the two-variable Selberg measure."""

    return sp.factor(
        sum(
            coefficient
            * normalized_elementary_product(
                tuple([1] * power_e1 + [2] * power_e2), 2, A, B, g
            )
            for coefficient, power_e1, power_e2 in terms
        )
    )


def hard_orbit_selberg_averages(A, B, g) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Normalized Selberg averages in the ``(bulk,zero,one)`` basis."""

    bulk_terms = (
        (2, 0, 3),
        (-3, 1, 2),
        (3, 2, 1),
        (-1, 3, 0),
        (-6, 0, 2),
        (3, 1, 1),
    )
    zero_terms = (
        (-1, 3, 0),
        (3, 2, 0),
        (-3, 1, 0),
        (2, 0, 0),
        (3, 1, 1),
        (-6, 0, 1),
    )
    one_terms = ((1, 3, 0), (-3, 1, 1))
    return tuple(
        _elementary_average(terms, A, B, g)
        for terms in (bulk_terms, zero_terms, one_terms)
    )


def hard_orbit_integrals(A, B, g) -> sp.Matrix:
    """Unnormalized two-screening integrals in the permutation basis."""

    selberg = ordinary_selberg(2, A, B, g)
    return sp.Matrix(
        [sp.factor(selberg * value) for value in hard_orbit_selberg_averages(A, B, g)]
    )


def dotsenko_fateev_two_screen_factor(A, B, g) -> sp.Expr:
    """The fixed-N chamber factor for two screening variables."""

    return sp.factor(
        sp.sin(sp.pi * (A + B + 2 + g))
        * sp.sin(sp.pi * (A + B + 2 + 2 * g))
        / (
            sp.sin(sp.pi * (A + 1))
            * sp.sin(sp.pi * (A + 1 + g))
        )
    )


def hard_orbit_functional_equation(A, B, g) -> tuple[sp.Matrix, sp.Matrix]:
    r"""Return both sides of the exact vector contour functional equation.

    Every orbit polynomial has reciprocal degree three in each variable, so

    ``A_dual = -A-B-2-2g-3``.

    The permutation matrix is the action of ``I`` on
    ``(bulk,zero,one)``.
    """

    dual_A = -A - B - 2 * g - 5
    permutation = sp.Matrix(((0, 1, 0), (1, 0, 0), (0, 0, 1)))
    left = hard_orbit_integrals(A, B, g)
    right = (
        dotsenko_fateev_two_screen_factor(A, B, g)
        * permutation
        * hard_orbit_integrals(dual_A, B, g)
    )
    return left, right


def audit() -> None:
    bulk, zero, one = contour_orbit()
    if coefficient_rank((bulk, zero, one)) != 3:
        raise AssertionError("the contour orbit did not have rank three")

    if endpoint_reflection(bulk) != bulk:
        raise AssertionError("the crossed polynomial should be R-invariant")
    if inversion(bulk) != zero or inversion(zero) != bulk:
        raise AssertionError("inversion did not swap bulk and zero")
    if inversion(one) != one:
        raise AssertionError("the one-endpoint polynomial should be I-invariant")
    if endpoint_reflection(zero) != one or endpoint_reflection(one) != zero:
        raise AssertionError("reflection did not swap the endpoint polynomials")

    orders_zero = tuple(endpoint_order(item, 0) for item in (bulk, zero, one))
    orders_one = tuple(endpoint_order(item, 1) for item in (bulk, zero, one))
    if orders_zero != (3, 0, 3) or orders_one != (3, 3, 0):
        raise AssertionError((orders_zero, orders_one))

    inversion_matrix, reflection_matrix = doublet_matrices()
    identity = sp.eye(2)
    if inversion_matrix**2 != identity or reflection_matrix**2 != identity:
        raise AssertionError("the doublet generators are not reflections")
    if (inversion_matrix * reflection_matrix) ** 3 != identity:
        raise AssertionError("the doublet generators do not satisfy S_3")
    change, physical_reflection, universal_transfer = physical_doublet_change()
    expected_physical_reflection = sp.Matrix(((0, sp.I), (-sp.I, 0)))
    if physical_reflection != expected_physical_reflection:
        raise AssertionError((change, physical_reflection))
    if sp.simplify(
        universal_transfer
        - (-sp.Rational(3, 2) * identity + physical_reflection / 2)
    ) != sp.zeros(2):
        raise AssertionError((universal_transfer, physical_reflection))

    A, B, g = sp.symbols("A B g")
    left, right = hard_orbit_functional_equation(A, B, g)
    for left_entry, right_entry in zip(left, right):
        ratio = sp.factor(
            sp.powsimp(
                sp.cancel(sp.expand_func(left_entry / right_entry)), force=True
            )
        )
        # Reflection formula certificate.  Written in this paired form,
        # SymPy reduces each x sin(pi x) Gamma(-x) Gamma(x) to -pi.
        pair = lambda x: x * sp.sin(sp.pi * x) * sp.gamma(-x) * sp.gamma(x)
        certificate = sp.cancel(
            pair(A)
            * pair(A + g)
            / (pair(A + B + g) * pair(A + B + 2 * g))
        )
        if sp.factor(ratio - certificate) != 0 or sp.simplify(certificate) != 1:
            raise AssertionError((ratio, certificate))

    print("hard N=2 contour orbit: exact rank 3")
    print("I=(bulk zero), R=(zero one): S_3 permutation action")
    print("endpoint radial orders: zero=(3,0,3), one=(3,3,0)")
    print("sum-zero quotient: exact two-dimensional S_3 doublet")
    print("PBW universal mixing: -3/2 identity + 1/2 contour reflection")
    print("vector Dotsenko--Fateev functional equation: all 3 entries exact")
    print("scope: polynomial closure only; the physical singlet projection is conjectural")


if __name__ == "__main__":
    audit()
