"""Exact obstruction to a uniform Schur-width bound at Coulomb nodes.

The consecutive Ramond ``chi`` strings have a distinguished screening
number

    N0 = 2 (n1+n2+n3).

At ``N=N0`` the unreflected fermionic numerator is a rectangular
external--screening determinant.  After the endpoint poles are cleared its
symmetric insertion is ``C*Delta(t)**2``.  The Schur remainder after moving
``Delta**2`` into the Selberg weight therefore has width zero.

Momentum interpolation at fixed ``b,P2,P3`` cannot keep this screening
number: the same-parity nodes use ``N=N0+2*j``.  At node ``j`` there are
``j`` unavoidable screening--screening Wick pairs.  This file records the
first exact obstruction.  It audits the *unreflected native Coulomb
integrand*, not a fitted branching polynomial and not the missing reflected
Ramond vertex.

The obstruction is specifically to the finite-Schur-reconstruction bound.
It does not rule out a separate BFL/Uglov or holonomic integral oracle.  In
fact the smallest surplus example is exactly the standard BFL spin
polynomial, which is why a raw Schur count is not by itself a lower bound on
all possible algorithms.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, log2

import sympy as sp

from .ising_polynomial import spin_pfaffian_polynomial
from .screening_pfaffian import (
    _combined_pfaffian,
    contour_polynomial,
    external_rows,
    natural_ramond_parity,
    vandermonde,
)
from .selberg_jack import _jack_transition, monomial_coefficients


R = sp.Rational


def schur_coefficients(polynomial, variables):
    """Return the exact Schur expansion, grouped degree by degree."""

    variables = tuple(variables)
    monomial = monomial_coefficients(polynomial, variables)
    answer = {}
    for degree in sorted({sum(partition) for partition in monomial}):
        partitions, transition = _jack_transition(
            len(variables), degree, sp.Integer(1)
        )
        values = sp.Matrix(
            [monomial.get(partition, 0) for partition in partitions]
        )
        # At Jack parameter alpha=1 the monic Jack polynomials are Schur
        # polynomials.  Columns of ``transition`` are their monomial
        # coefficients.
        coefficients = transition.inv() * values
        for partition, coefficient in zip(partitions, coefficients):
            coefficient = sp.factor(coefficient)
            if coefficient != 0:
                answer[partition] = coefficient
    return answer


def smallest_surplus_certificate():
    """Check ``N0=1`` and its first same-parity surplus node ``N=3``."""

    labels = (0, R(1, 4), R(1, 4))
    epsilon2 = natural_ramond_parity(labels[1])
    epsilon3 = natural_ramond_parity(labels[2])

    variables0, minimal, _, _ = contour_polynomial(
        *labels,
        0,
        1,
        screenings=1,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
    )
    delta0 = vandermonde(variables0)
    minimal_remainder = sp.factor(minimal / delta0**2)
    if set(variables0) & minimal_remainder.free_symbols:
        raise AssertionError(minimal_remainder)

    variables1, surplus, _, _ = contour_polynomial(
        *labels,
        0,
        1,
        screenings=3,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
    )
    delta1 = vandermonde(variables1)
    _, delta_remainder = sp.div(
        sp.Poly(surplus, *variables1),
        sp.Poly(delta1**2, *variables1),
    )
    if delta_remainder.is_zero:
        raise AssertionError("the first surplus node unexpectedly kept Delta^2")

    schur = schur_coefficients(surplus, variables1)
    expected_support = {
        (2, 1),
        (1, 1, 1),
        (2, 2),
        (2, 1, 1),
    }
    if set(schur) != expected_support:
        raise AssertionError((set(schur), expected_support))
    width = max(partition[0] for partition in schur)
    if width != 2:
        raise AssertionError(width)

    # This first obstruction is nevertheless a known integrable object:
    # it is one standard NS--R--R BFL polynomial.  Thus the check rules out
    # uniform Schur width, not every possible structured integral oracle.
    bfl = spin_pfaffian_polynomial(variables1)
    ratio = sp.factor(sp.cancel(surplus / bfl))
    if set(variables1) & ratio.free_symbols:
        raise AssertionError(ratio)
    return minimal_remainder, tuple(sorted(schur)), width, ratio


def _univariate_cleared_degree(labels, screenings):
    """Degree in one screening variable without multivariate expansion."""

    labels = tuple(map(sp.Rational, labels))
    screenings = int(screenings)
    infinity, one, zero = external_rows(
        *labels,
        natural_ramond_parity(labels[1]),
        natural_ramond_parity(labels[2]),
    )
    rows = infinity + one + zero
    variable = sp.symbols("audit_t")
    # Exact, distinct rational values for all other screening variables.
    values = (variable,) + tuple(
        R(index + 2, screenings + 3) for index in range(screenings - 1)
    )
    objects = (
        tuple(("external", row) for row in infinity + one)
        + tuple(("screening", value) for value in values)
        + tuple(("external", row) for row in zero)
    )
    external_parity = len(rows) % 2
    correlator = _combined_pfaffian(
        objects,
        external_parity,  # form_parity=0
        screenings % 2,
        1,
    )
    shift_a = int(2 * labels[2] - R(1, 2))
    shift_b = int(2 * labels[1] - R(1, 2))
    cleared = sp.cancel(
        vandermonde(values)
        * sp.prod(
            value**shift_a * (1 - value) ** shift_b for value in values
        )
        * correlator
    )
    numerator, denominator = sp.fraction(cleared)
    numerator = sp.Poly(numerator, variable)
    denominator = sp.Poly(denominator, variable)
    if denominator.degree() != 0:
        raise AssertionError(denominator)
    return numerator.degree(), sp.factor(numerator.LC() / denominator.LC())


def hard_first_surplus_certificate():
    """Show loss of ``Delta^2`` at the first hard interpolation node."""

    labels = (0, R(3, 4), R(3, 4))
    minimal_degree, minimal_lead = _univariate_cleared_degree(labels, 3)
    surplus_degree, surplus_lead = _univariate_cleared_degree(labels, 5)
    if (minimal_degree, surplus_degree) != (4, 6):
        raise AssertionError((minimal_degree, surplus_degree))
    if minimal_lead == 0 or surplus_lead == 0:
        raise AssertionError((minimal_lead, surplus_lead))

    # Delta_N^2 has degree 2(N-1) in each variable.  The nonzero surplus
    # insertion has degree six, strictly below eight, and hence cannot be
    # divisible by Delta_5^2.
    if surplus_degree >= 2 * (5 - 1):
        raise AssertionError(surplus_degree)
    return minimal_degree, surplus_degree, minimal_lead, surplus_lead


def interpolation_rectangle_size(n1, n2, n3):
    """Raw Schur rectangle forced by the final same-parity node.

    This is an operation-count warning, not a claim that a BFL/Uglov oracle
    cannot do better.  The degree bound and nodes are the ones used by the
    current two-chart interpolation driver.
    """

    labels = tuple(Fraction(value) for value in (n1, n2, n3))
    degree = int(
        (2 * labels[0]) ** 2
        + (2 * labels[1]) ** 2
        + (2 * labels[2]) ** 2
        - Fraction(1, 2)
    )
    n0 = int(2 * sum(abs(value) for value in labels))
    n_max = n0 + 2 * degree
    # For the NS-primary hard family the exact one-variable degree is
    # N-1+A+B.  We report that certified raw width, rather than pretending
    # that Delta^2 can still be removed at surplus nodes.
    endpoint_degree = int(2 * abs(labels[1]) - Fraction(1, 2)) + int(
        2 * abs(labels[2]) - Fraction(1, 2)
    )
    width = n_max - 1 + endpoint_degree
    states = comb(n_max + width, width)
    return degree, n0, n_max, width, states


def audit():
    minimal, support, width, bfl_ratio = smallest_surplus_certificate()
    degrees = hard_first_surplus_certificate()
    rectangle = interpolation_rectangle_size(0, R(3, 4), R(3, 4))
    print(
        "minimal node: remainder width 0; "
        f"first surplus support={support}, width={width}"
    )
    print(f"smallest surplus / BFL_110 = {bfl_ratio}")
    print(
        "hard node univariate cleared degrees: "
        f"N=3 -> {degrees[0]}, N=5 -> {degrees[1]} "
        "(Delta_5^2 degree is 8)"
    )
    degree, n0, nmax, raw_width, states = rectangle
    print(
        "hard full interpolation warning: "
        f"D={degree}, N0={n0}, Nmax={nmax}, "
        f"raw_width={raw_width}, rectangle_states={states}, "
        f"log2(states)={log2(states):.3f}"
    )
    # Keep ``minimal`` live so that an accidental zero normalization is
    # caught even if the printed output is changed later.
    if minimal == 0:
        raise AssertionError(minimal)
    print("surplus-screening Schur-width obstruction: exact checks passed")


if __name__ == "__main__":
    audit()
