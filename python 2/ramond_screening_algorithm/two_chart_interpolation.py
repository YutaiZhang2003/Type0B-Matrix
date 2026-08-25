#!/usr/bin/env python3
"""Exact interpolation layer for two Ramond Coulomb charge charts.

The input to this module is deliberately a pair of *independent* exact
charge-chart evaluators.  At a neutrality node each evaluator must perform
its own native ground-resolved Pfaffian and Selberg integral.  Supplying a
closed expression already reconstructed from the desired branching
coefficient would be circular and is outside this API.

This layer supplies the parts which are already rigorous at arbitrary branch
labels:

* the degree bound of arXiv:1312.4520;
* distinct same-parity screening nodes;
* exact polynomial interpolation over any characteristic-zero field;
* the constant two-dimensional ground-space map after both analytic charge
  charts have independently been reconstructed;
* precise conditional complexity estimates.

The repository currently has the native Majorana Pfaffian and the
charge-preserving projected Selberg callback.  It does *not* yet have a
verified full analytic ``C_positive`` callback, nor an arbitrary-mode
complementary ``C_signed`` callback.  Therefore ``reconstruct_two_charts`` is
a complete exact orchestration routine, not yet an end-to-end all-level
branching command.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Callable, Iterable, Sequence

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)
ExactEvaluator = Callable[[sp.Expr, int], sp.Expr]


def momentum_degree_bound(n1, n2, n3) -> int:
    """Degree after clearing the known one-leg denominators."""

    labels = tuple(Fraction(value) for value in (n1, n2, n3))
    k1, k2, k3 = (2 * value for value in labels)
    degree = k1 * k1 + k2 * k2 + k3 * k3 - Fraction(1, 2)
    if degree.denominator != 1 or degree < 0:
        raise ValueError((n1, n2, n3, degree))
    return degree.numerator


def natural_screening_start(n1, n2, n3) -> int:
    """A nonnegative screening count with the natural branch parity.

    For the all-positive consecutive representatives this is the maximal
    screening number ``2(|n1|+|n2|+|n3|)``.  Adding pairs of screenings
    leaves the Ramond chiral structure unchanged.
    """

    labels = tuple(abs(Fraction(value)) for value in (n1, n2, n3))
    count = 2 * sum(labels)
    if count.denominator != 1:
        raise ValueError((n1, n2, n3, count))
    return count.numerator


def neutrality_momentum(
    q,
    b,
    p2,
    p3,
    screenings_b,
    screenings_dual=0,
    signs=(1, 1, 1),
):
    r"""Solve one Coulomb neutrality equation for ``P1``.

    The equation is

    ``Q/2+s1*P1+s2*P2+s3*P3=-r*b-s/b``.

    Here ``r=screenings_b`` and ``s=screenings_dual``.  All arithmetic is
    exact and no choice of square root or floating point approximation is
    made.
    """

    sign1, sign2, sign3 = (int(value) for value in signs)
    if any(value not in (-1, 1) for value in (sign1, sign2, sign3)):
        raise ValueError("all charge-chart signs must be +/-1")
    screenings_b = int(screenings_b)
    screenings_dual = int(screenings_dual)
    if screenings_b < 0 or screenings_dual < 0:
        raise ValueError("screening numbers must be nonnegative")
    return sp.factor(
        sign1
        * (
            -sp.sympify(q) / 2
            - sign2 * sp.sympify(p2)
            - sign3 * sp.sympify(p3)
            - screenings_b * sp.sympify(b)
            - screenings_dual / sp.sympify(b)
        )
    )


def same_parity_nodes(
    degree,
    q,
    b,
    p2,
    p3,
    first_screening,
    signs=(1, 1, 1),
):
    """Return ``degree+1`` distinct nodes with fixed screening parity.

    The screening numbers are ``N_j=N_0+2j``.  Keeping their parity fixed
    prevents the Ramond ground form from alternating between the two chiral
    structures.  Distinctness is checked symbolically.
    """

    degree = int(degree)
    first_screening = int(first_screening)
    if degree < 0 or first_screening < 0:
        raise ValueError((degree, first_screening))
    pairs = tuple(
        (
            first_screening + 2 * index,
            neutrality_momentum(
                q,
                b,
                p2,
                p3,
                first_screening + 2 * index,
                0,
                signs,
            ),
        )
        for index in range(degree + 1)
    )
    nodes = tuple(value for _, value in pairs)
    for left, value_left in enumerate(nodes):
        for value_right in nodes[left + 1 :]:
            if sp.simplify(value_left - value_right) == 0:
                raise ValueError("neutrality nodes are not distinct")
    return pairs


def exact_interpolate(variable, nodes_and_values, degree):
    """Interpolate and certify one denominator-cleared chart polynomial."""

    variable = sp.sympify(variable)
    nodes_and_values = tuple(
        (sp.sympify(node), sp.sympify(value))
        for node, value in nodes_and_values
    )
    degree = int(degree)
    if len(nodes_and_values) != degree + 1:
        raise ValueError(f"expected {degree + 1} values")
    polynomial = sp.interpolate(nodes_and_values, variable)
    polynomial = sp.Poly(sp.expand(polynomial), variable)
    if polynomial.degree() > degree:
        raise AssertionError((polynomial.degree(), degree))
    for node, value in nodes_and_values:
        residual = sp.factor(sp.cancel(polynomial.as_expr().subs(variable, node) - value))
        if residual != 0:
            raise AssertionError((node, residual))
    return polynomial.as_expr()


@dataclass(frozen=True)
class ChartReconstruction:
    """The two independently reconstructed analytic chart polynomials."""

    degree: int
    positive: sp.Expr
    signed: sp.Expr
    positive_nodes: tuple[tuple[int, sp.Expr], ...]
    signed_nodes: tuple[tuple[int, sp.Expr], ...]


def reconstruct_two_charts(
    variable,
    degree,
    positive_nodes,
    signed_nodes,
    evaluate_positive: ExactEvaluator,
    evaluate_signed: ExactEvaluator,
):
    """Evaluate, interpolate, and return two independent charge charts.

    Each callback receives ``(P1_node, screening_number)``.  It is invoked
    exactly once per node.  In particular no values are copied between the
    two charts.
    """

    degree = int(degree)
    positive_nodes = tuple(positive_nodes)
    signed_nodes = tuple(signed_nodes)
    if len(positive_nodes) != degree + 1 or len(signed_nodes) != degree + 1:
        raise ValueError("each chart needs degree+1 neutrality nodes")
    positive_values = tuple(
        (node, evaluate_positive(node, screenings))
        for screenings, node in positive_nodes
    )
    signed_values = tuple(
        (node, evaluate_signed(node, screenings))
        for screenings, node in signed_nodes
    )
    return ChartReconstruction(
        degree=degree,
        positive=exact_interpolate(variable, positive_values, degree),
        signed=exact_interpolate(variable, signed_values, degree),
        positive_nodes=positive_nodes,
        signed_nodes=signed_nodes,
    )


def ground_change_matrix(form_parity, boundary_side="right"):
    """Coordinates on ``(canonical, canonical-with-boundary-Z)`` forms.

    This constant matrix may be used only *after* both reconstructed objects
    are known to be SCA trilinear forms.  Its constancy then follows because
    the generic NS--R--R trilinear-form space is two-dimensional and its
    coordinates are fixed on the Ramond ground doublets.  ``boundary-Z``
    names the ground matrix of the second genuine form; applying endpoint Z
    to raw excited chi paths does not construct that form.  A signed nonzero
    callback still needs the 2013 reflection operator.
    """

    form_parity = int(form_parity)
    boundary_side = str(boundary_side)
    if boundary_side not in ("left", "right"):
        raise ValueError("boundary_side must be 'left' or 'right'")
    if form_parity == 0:
        return sp.Matrix(
            (
                ((1 + I) / 2, (1 - I) / 2),
                ((1 - I) / 2, (1 + I) / 2),
            )
        )
    if form_parity == 1:
        c = -(1 - I) / SQRT2
        matrix = sp.Matrix(
            (
                (c * (1 - I) / 2, -c * (1 + I) / 2),
                (c * (1 + I) / 2, -c * (1 - I) / 2),
            )
        )
        # ZJ=-JZ.  Moving the signed boundary from the right Ramond leg to
        # the left therefore reverses the second odd-form column.
        if boundary_side == "left":
            matrix[:, 1] *= -1
        return matrix
    raise ValueError("form_parity must be 0 or 1")


def resolve_eta_forms(
    reconstruction: ChartReconstruction,
    form_parity=0,
    boundary_side="right",
):
    """Return the two fixed-eta denominator-cleared polynomials."""

    charts = sp.Matrix((reconstruction.positive, reconstruction.signed))
    return sp.simplify(
        ground_change_matrix(form_parity, boundary_side) * charts
    )


@dataclass(frozen=True)
class ComplexityEstimate:
    degree: int
    interpolation_nodes_per_chart: int
    maximum_screenings: int
    external_chi_modes: int
    schur_width: int
    schur_samples_per_node: int
    pfaffian_field_ops_upper: int
    schur_transform_field_ops_upper: int
    interpolation_field_ops_upper: int


def conditional_complexity(
    n1,
    n2,
    n3,
    external_chi_modes,
    first_screening,
    schur_width,
):
    """Exact operation-size bounds under a proved Schur-width bound.

    A ground-resolved callback uses at most sixteen Pfaffians.  If the
    symmetric screening remainder lies in an ``N by w`` rectangle, it needs
    ``M=binomial(N+w,w)`` black-box samples.  The returned counts retain all
    dependence on ``w``; polynomial all-level complexity follows only when
    ``w`` is bounded independently of the branch labels.
    """

    degree = momentum_degree_bound(n1, n2, n3)
    nodes = degree + 1
    maximum_screenings = int(first_screening) + 2 * degree
    external_chi_modes = int(external_chi_modes)
    width = int(schur_width)
    if min(maximum_screenings, external_chi_modes, width) < 0:
        raise ValueError("complexity parameters must be nonnegative")
    schur_samples = comb(maximum_screenings + width, width)
    pfaffian_size = maximum_screenings + external_chi_modes + 2
    # Two charts, all nodes, at most sixteen boundary Pfaffians per sample.
    pfaffian_ops = 2 * nodes * schur_samples * 16 * pfaffian_size**3
    # The compound-Vandermonde inverse uses M^2 determinants of size w.
    transform_ops = 2 * nodes * schur_samples**2 * max(1, width**3)
    interpolation_ops = 2 * nodes**2
    return ComplexityEstimate(
        degree=degree,
        interpolation_nodes_per_chart=nodes,
        maximum_screenings=maximum_screenings,
        external_chi_modes=external_chi_modes,
        schur_width=width,
        schur_samples_per_node=schur_samples,
        pfaffian_field_ops_upper=pfaffian_ops,
        schur_transform_field_ops_upper=transform_ops,
        interpolation_field_ops_upper=interpolation_ops,
    )


def _audit():
    """Exact orchestration audit with independent polynomial callbacks."""

    x = sp.symbols("P1")
    degree = 4
    q = sp.Rational(13, 6)
    b = sp.Rational(3, 2)
    p2 = sp.Rational(2, 5)
    p3 = sp.Rational(3, 7)
    first_screening = natural_screening_start(
        0, Fraction(3, 4), Fraction(3, 4)
    )
    positive_nodes = same_parity_nodes(
        degree, q, b, p2, p3, first_screening
    )
    # The ground matrix below uses a Z insertion on the right/third Ramond
    # boundary, hence this audit reflects the third momentum.
    signed_nodes = same_parity_nodes(
        degree, q, b, p2, p3, first_screening, (1, 1, -1)
    )
    positive_polynomial = x**4 + 3 * x**3 - 2 * x + 5
    signed_polynomial = 2 * x**4 - x**2 + 7
    positive_calls = []
    signed_calls = []

    def positive(node, screenings):
        positive_calls.append((node, screenings))
        return positive_polynomial.subs(x, node)

    def signed(node, screenings):
        signed_calls.append((node, screenings))
        return signed_polynomial.subs(x, node)

    reconstructed = reconstruct_two_charts(
        x,
        degree,
        positive_nodes,
        signed_nodes,
        positive,
        signed,
    )
    if sp.expand(reconstructed.positive - positive_polynomial) != 0:
        raise AssertionError("positive chart interpolation failed")
    if sp.expand(reconstructed.signed - signed_polynomial) != 0:
        raise AssertionError("signed chart interpolation failed")
    if len(positive_calls) != 5 or len(signed_calls) != 5:
        raise AssertionError("a chart callback was reused or skipped")
    if ground_change_matrix(0).det() != I:
        raise AssertionError("the even-form chart map is singular")
    if sp.simplify(ground_change_matrix(1).det()) == 0:
        raise AssertionError("the odd-form chart map is singular")
    if sp.simplify(ground_change_matrix(1, "left").det()) == 0:
        raise AssertionError("the left-boundary odd-form chart map is singular")
    estimate = conditional_complexity(0, Fraction(3, 4), Fraction(3, 4), 4, 3, 2)
    print("two-chart interpolation: 5+5 independent exact callbacks passed")
    print("ground maps: f=0 and f=1 are nonsingular")
    print(
        "hard degree/node audit: "
        f"D={estimate.degree}, nodes/chart={estimate.interpolation_nodes_per_chart}"
    )
    print("complexity is polynomial only conditional on a uniform Schur-width bound")


if __name__ == "__main__":
    _audit()
