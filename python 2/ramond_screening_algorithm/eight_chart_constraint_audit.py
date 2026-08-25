"""Exact rank audit for the proposed eight-chart Coulomb reconstruction.

The 2013 ``Psi_-A`` argument uses all eight choices of reflected or
unreflected external representatives.  For a sign triple ``sigma`` put

    m_sigma = 2*(sigma1*n1 + sigma2*n2 + sigma3*n3).

At a neutrality point with ``N=r+s`` screenings, the elementary counting
argument proves a zero only when ``m_sigma>0`` and ``N<m_sigma`` (with the
required screening parity).  Opposite sign triples have opposite ``m``;
therefore only one member of each global-sign pair contributes zeros.

This script constructs every such equation, without reading a Ward value.
It also gives the suggested shortcut its strongest possible interpretation:
for every orientation with ``m_sigma>=0`` it *grants* the complete diagonal
``r+s=m_sigma`` as known nonzero values.  The latter are rank rows only; the
right-hand sides are deliberately absent.  In reality arXiv:1312.4520 uses
reflected representatives only to prove zeros and does not supply this
mixed reflected Coulomb evaluator.

Even under that generous grant the system does not have uniformly bounded
corank.  For ``(n1,n2,n3)=(1,5/4,5/4)`` it has 34 unknown coefficients and
rank only 25.  Along

    (n1,n2,n3)=(k/2,(2k+1)/4,(2k+1)/4)

the nullity grows quadratically.  Thus deficient and saturation planes do
not yield the claimed two-constant reconstruction, independently of the
cost of evaluating a reflected chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools

import sympy as sp

from .zero_constraint_system import (
    chart_covector,
    ell_lattice,
    momentum_degree_bound,
    polynomial_evaluation_row,
)


SIGNS = tuple(itertools.product((1, -1), repeat=3))


def signed_length(labels, signs):
    """Return the integral excess of unreflected over reflected chi rows."""

    labels = tuple(Fraction(value) for value in labels)
    signs = tuple(int(value) for value in signs)
    value = 2 * sum(sign * label for sign, label in zip(signs, labels))
    if value.denominator != 1:
        raise ValueError((labels, signs, value))
    return value.numerator


def orientation_covector(form_parity, signs):
    """Ground covector after quotienting by simultaneous reflection."""

    sign1, sign2, sign3 = map(int, signs)
    return chart_covector(
        int(form_parity), sign1 * sign2, sign1 * sign3
    )


def neutrality_p1(q, b, p2, p3, signs, r, s):
    """Solve ``Q/2+sum sigma_i P_i=-r*b-s/b`` for ``P1``."""

    sign1, sign2, sign3 = map(int, signs)
    return sp.factor(
        sign1
        * (
            -sp.sympify(q) / 2
            - sign2 * sp.sympify(p2)
            - sign3 * sp.sympify(p3)
            - int(r) * sp.sympify(b)
            - int(s) / sp.sympify(b)
        )
    )


@dataclass(frozen=True)
class ChartRow:
    kind: str
    signs: tuple[int, int, int]
    length: int
    screenings: tuple[int, int]
    momentum: sp.Expr
    covector: tuple[sp.Expr, sp.Expr]


def all_eight_rows(labels, form_parity, q, b, p2, p3):
    """Return all proved zero rows and all conditionally granted boundary rows.

    ``kind='zero'`` is exactly the 2013 rank-deficiency statement.
    ``kind='saturation'`` denotes the conjectural nonzero callback on the
    equality boundary.  Including it can only increase rank, so failure
    after including every such row is an unconditional counting
    obstruction to the proposed reconstruction from these planes.
    """

    zero_rows = []
    saturation_rows = []
    for signs in SIGNS:
        length = signed_length(labels, signs)
        covector = orientation_covector(form_parity, signs)
        if length > 0:
            for r, s in ell_lattice(length):
                zero_rows.append(
                    ChartRow(
                        "zero",
                        signs,
                        length,
                        (r, s),
                        neutrality_p1(q, b, p2, p3, signs, r, s),
                        covector,
                    )
                )
        if length >= 0:
            for r in range(length + 1):
                s = length - r
                saturation_rows.append(
                    ChartRow(
                        "saturation",
                        signs,
                        length,
                        (r, s),
                        neutrality_p1(q, b, p2, p3, signs, r, s),
                        covector,
                    )
                )
    return tuple(zero_rows), tuple(saturation_rows)


def row_matrix(degree, rows):
    return sp.Matrix(
        [
            polynomial_evaluation_row(
                degree, row.momentum, row.covector
            )
            for row in rows
        ]
    )


def covariance_direction(row):
    """The two chart covector lines are labelled by ``tau2*tau3``."""

    sign1, sign2, sign3 = row.signs
    return (sign1 * sign2) * (sign1 * sign3)


def evaluation_rank(degree, rows, form_parity):
    """Exact Vandermonde rank, without expensive symbolic elimination.

    For either form parity the eight ground covectors span exactly two
    lines, according to ``tau2*tau3=+/-1``; representatives of the two
    lines are independent.  Within one line the rows are ordinary
    polynomial evaluation rows, so their rank is the number of distinct
    nodes capped at ``degree+1``.
    """

    representatives = {
        direction: chart_covector(form_parity, 1, direction)
        for direction in (1, -1)
    }
    determinant = sp.factor(
        sp.Matrix(
            (representatives[1], representatives[-1])
        ).det()
    )
    if determinant == 0:
        raise AssertionError((form_parity, representatives))

    nodes = {1: set(), -1: set()}
    for row in rows:
        direction = covariance_direction(row)
        reference = representatives[direction]
        if sp.factor(
            sp.Matrix((reference, row.covector)).det()
        ) != 0:
            raise AssertionError((row, reference))
        nodes[direction].add(sp.cancel(row.momentum))
    return sum(min(int(degree) + 1, len(nodes[key])) for key in (1, -1))


@dataclass(frozen=True)
class EightChartRank:
    labels: tuple[sp.Rational, sp.Rational, sp.Rational]
    degree: int
    unknowns: int
    zero_rows: int
    zero_rank: int
    saturation_rows: int
    combined_rank: int
    combined_nullity: int


def rank_audit(labels, form_parity=0):
    labels = tuple(sp.Rational(value) for value in labels)
    degree = momentum_degree_bound(*labels)
    b = sp.Rational(3, 2)
    q = b + 1 / b
    p2, p3 = sp.Rational(2, 5), sp.Rational(3, 7)
    zero_rows, saturation_rows = all_eight_rows(
        labels, form_parity, q, b, p2, p3
    )
    unknowns = 2 * (degree + 1)
    zero_rank = evaluation_rank(degree, zero_rows, form_parity)
    combined_rank = evaluation_rank(
        degree, zero_rows + saturation_rows, form_parity
    )
    return EightChartRank(
        labels=labels,
        degree=degree,
        unknowns=unknowns,
        zero_rows=len(zero_rows),
        zero_rank=zero_rank,
        saturation_rows=len(saturation_rows),
        combined_rank=combined_rank,
        combined_nullity=unknowns - combined_rank,
    )


def asymptotic_row_bound(k):
    """Closed count for the balanced family used in the obstruction."""

    k = int(k)
    if k < 1:
        raise ValueError(k)
    labels = (
        Fraction(k, 2),
        Fraction(2 * k + 1, 4),
        Fraction(2 * k + 1, 4),
    )
    degree = momentum_degree_bound(*labels)
    lengths = tuple(
        abs(2 * (labels[0] + tau2 * labels[1] + tau3 * labels[2]))
        for tau2, tau3 in itertools.product((1, -1), repeat=2)
    )
    if any(value.denominator != 1 for value in lengths):
        raise AssertionError(lengths)
    lengths = tuple(value.numerator for value in lengths)
    # One positive orientation per nonzero global-sign pair.  If a length
    # vanishes, both orientations lie on the equality boundary.
    saturation = sum(value + 1 if value else 2 for value in lengths)
    return labels, degree, lengths, saturation


def audit():
    expected = {
        (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4)):
            (4, 10, 4, 4, 12, 9, 1),
        (sp.Integer(1), sp.Rational(5, 4), sp.Rational(5, 4)):
            (16, 34, 16, 16, 18, 25, 9),
        (sp.Rational(3, 2), sp.Rational(7, 4), sp.Rational(7, 4)):
            (33, 68, 33, 33, 24, 46, 22),
    }
    for form_parity in (0, 1):
        for labels, target in expected.items():
            result = rank_audit(labels, form_parity)
            observed = (
                result.degree,
                result.unknowns,
                result.zero_rows,
                result.zero_rank,
                result.saturation_rows,
                result.combined_rank,
                result.combined_nullity,
            )
            if observed != target:
                raise AssertionError((form_parity, result, target))

    labels, degree, lengths, saturation = asymptotic_row_bound(3)
    if (degree, lengths, saturation) != (33, (10, 3, 3, 4), 24):
        raise AssertionError((labels, degree, lengths, saturation))

    print("all-eight deficient rows: exact count and rank equal D")
    print(
        "hard (0,3/4,3/4): granting 12 saturation rows still leaves "
        "nullity 1"
    )
    print(
        "stored (1,5/4,5/4): 34 unknowns; zero+saturation rank 25; "
        "nullity 9"
    )
    print(
        "balanced (3/2,7/4,7/4): 68 unknowns; "
        "zero+saturation rank 46; nullity 22"
    )
    print(
        "verdict: the eight deficient/saturation charts do not reduce the "
        "Ramond two-polynomial problem to two constants"
    )


if __name__ == "__main__":
    audit()
