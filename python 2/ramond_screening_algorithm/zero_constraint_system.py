"""Vector-valued 2013 zero constraints for the Ramond two-form problem.

The NS proof of arXiv:1312.4520 reconstructs one scalar polynomial.  Its
free-field rank deficiencies supply as many scalar zeros as the momentum
degree.  In the NS--R--R problem the denominator-cleared object has two
chiral components.  A charge chart therefore supplies a *covector* on the
two components, not two scalar zeros.

This module constructs those equations without consulting a Ward value or
the known hard polynomial.  It uses only

* the four fusion-polynomial lengths

      m_(tau2,tau3) = 2 (n1 + tau2*n2 + tau3*n3),

* the screening lattice in the definition of ``ell(x,m)``, and
* the exact Ramond ground matrices on the reflected zero loci.

The result is an obstruction to a tempting shortcut.  The four root
families contain exactly ``D`` roots, where

    D=(2*n1)^2+(2*n2)^2+(2*n3)^2-1/2.

They give at most ``D`` homogeneous equations for ``2*(D+1)`` unknown
polynomial coefficients.  Two minimal-screening normalizations can raise
the rank by at most two.  Hence the 2013 zero data plus two normalizations
leave nullity at least ``D``.  For the first irreducible case
``(0,3/4,3/4)`` the exact generic ranks attain these bounds: nullity six
before normalization and four afterwards.

These reflected representatives are used only to identify homogeneous
rank-deficiency equations.  The endpoint ``Z`` on their ground matrices is
not a reflected SCA-state identity and must never be used to manufacture a
nonzero chart value.  Such a value requires the 2013 reflection operator.

This does not say that no fast Ramond algorithm exists.  It says precisely
that a new all-level matrix/fusion recurrence is required for the second
channel; it cannot be replaced by the scalar zero count of the NS proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools

import sympy as sp

from .pfaffian.native_spin_kernel import (
    GROUND_Z,
    canonical_ground_matrix,
    scblock_fock_ground_matrix,
)


I = sp.I


def momentum_degree_bound(n1, n2, n3):
    labels = tuple(Fraction(value) for value in (n1, n2, n3))
    degree = sum((2 * value) ** 2 for value in labels) - Fraction(1, 2)
    if degree.denominator != 1 or degree < 0:
        raise ValueError((labels, degree))
    return degree.numerator


def fusion_lengths(n1, n2, n3):
    """Return the four integral second arguments of the mixed ``ell``'s."""

    n1, n2, n3 = map(Fraction, (n1, n2, n3))
    answer = {}
    for tau2, tau3 in itertools.product((1, -1), repeat=2):
        value = 2 * (n1 + tau2 * n2 + tau3 * n3)
        if value.denominator != 1:
            raise ValueError((n1, n2, n3, tau2, tau3, value))
        answer[(tau2, tau3)] = value.numerator
    return answer


def ell_lattice(m):
    """The ``(r,s)`` roots in ``ell(x,m)``, with ``m`` possibly negative."""

    magnitude = abs(int(m))
    return tuple(
        (r, diagonal - r)
        for diagonal in range(magnitude)
        if diagonal % 2 == magnitude % 2
        for r in range(diagonal + 1)
    )


def root_count_identity(n1, n2, n3):
    """Return ``(sum root counts,D)`` and verify their equality."""

    count = sum(len(ell_lattice(m)) for m in fusion_lengths(n1, n2, n3).values())
    degree = momentum_degree_bound(n1, n2, n3)
    if count != degree:
        raise AssertionError(((n1, n2, n3), count, degree))
    return count, degree


def chart_covector(form_parity, tau2, tau3):
    """Ground covector of one reflected homogeneous zero condition.

    At the primary ground level, a minus sign on a Ramond momentum puts the
    endpoint bookkeeping operator ``Z`` on that ground index.  We express

        Z_left^a K_f Z_right^b

    in the two SCblock ground matrices and use the result only as the
    covector multiplying a vanishing fusion polynomial.  This routine does
    not construct the reflected descendant form, and its result is not a
    signed nonzero-value callback.  No excited-state ground rotation is made
    here.
    """

    form_parity = int(form_parity)
    tau2, tau3 = int(tau2), int(tau3)
    if form_parity not in (0, 1) or tau2 not in (-1, 1) or tau3 not in (-1, 1):
        raise ValueError((form_parity, tau2, tau3))
    left = GROUND_Z if tau2 < 0 else sp.eye(2)
    right = GROUND_Z if tau3 < 0 else sp.eye(2)
    chart = left * canonical_ground_matrix(form_parity) * right
    basis = sp.Matrix.hstack(
        scblock_fock_ground_matrix(form_parity, 1).reshape(4, 1),
        scblock_fock_ground_matrix(form_parity, -1).reshape(4, 1),
    )
    solution = sp.linsolve((basis, chart.reshape(4, 1)))
    values = tuple(solution)
    if len(values) != 1 or len(values[0]) != 2:
        raise AssertionError((chart, solution))
    return tuple(sp.factor(value) for value in values[0])


@dataclass(frozen=True)
class ZeroConstraint:
    chart: tuple[int, int]
    fusion_length: int
    screenings: tuple[int, int]
    momentum: sp.Expr
    covector: tuple[sp.Expr, sp.Expr]


def zero_constraints(n1, n2, n3, form_parity, q, b, p2, p3):
    """Enumerate every scalar rank-deficiency equation in the four charts."""

    q, b, p2, p3 = map(sp.sympify, (q, b, p2, p3))
    answer = []
    for (tau2, tau3), length in fusion_lengths(n1, n2, n3).items():
        covector = chart_covector(form_parity, tau2, tau3)
        for r, s in ell_lattice(length):
            # For m>0, ell(x,m)=0 at x=-r*b-s/b.  For m<0 the definition
            # ell(x,-M)~ell(Q-x,M) gives x=Q+r*b+s/b.
            root = -r * b - s / b if length > 0 else q + r * b + s / b
            p1 = sp.factor(root - q / 2 - tau2 * p2 - tau3 * p3)
            answer.append(
                ZeroConstraint(
                    chart=(tau2, tau3),
                    fusion_length=length,
                    screenings=(r, s),
                    momentum=p1,
                    covector=covector,
                )
            )
    expected = momentum_degree_bound(n1, n2, n3)
    if len(answer) != expected:
        raise AssertionError((len(answer), expected))
    return tuple(answer)


def polynomial_evaluation_row(degree, momentum, covector):
    """One row for ``u_+ B_+(P)+u_- B_-(P)=0``."""

    degree = int(degree)
    momentum = sp.sympify(momentum)
    plus, minus = map(sp.sympify, covector)
    powers = tuple(momentum**power for power in range(degree + 1))
    return tuple(plus * value for value in powers) + tuple(
        minus * value for value in powers
    )


def constraint_matrix(n1, n2, n3, form_parity, q, b, p2, p3):
    degree = momentum_degree_bound(n1, n2, n3)
    constraints = zero_constraints(
        n1, n2, n3, form_parity, q, b, p2, p3
    )
    matrix = sp.Matrix(
        [
            polynomial_evaluation_row(
                degree, constraint.momentum, constraint.covector
            )
            for constraint in constraints
        ]
    )
    return constraints, matrix


def append_normalization_rows(matrix, degree, rows):
    """Append value constraints; right-hand-side values do not affect rank."""

    extra = sp.Matrix(
        [
            polynomial_evaluation_row(degree, momentum, covector)
            for momentum, covector in rows
        ]
    )
    return matrix.col_join(extra)


@dataclass(frozen=True)
class RankAudit:
    labels: tuple[sp.Rational, sp.Rational, sp.Rational]
    degree: int
    equations: int
    unknowns: int
    rank: int
    nullity: int
    rank_after_two_normalizations: int
    nullity_after_two_normalizations: int


def rank_audit(n1, n2, n3, form_parity=0):
    """Exact generic rational audit, with no target three-point values."""

    labels = tuple(map(sp.Rational, (n1, n2, n3)))
    degree = momentum_degree_bound(*labels)
    # Q and b obey Q=b+1/b.  P2,P3 are generic rational samples used only
    # to audit the structural rank of the evaluation matrix.
    b = sp.Rational(3, 2)
    q = b + 1 / b
    p2, p3 = sp.Rational(2, 5), sp.Rational(3, 7)
    _, matrix = constraint_matrix(
        *labels, form_parity, q, b, p2, p3
    )
    rank = matrix.rank()
    unknowns = 2 * (degree + 1)

    # Give the shortcut every possible advantage: add two independent
    # chart evaluations at generic momenta.  Their numerical values are
    # irrelevant to uniqueness and are deliberately absent.
    normalization_rows = (
        (sp.Rational(5, 11), chart_covector(form_parity, 1, 1)),
        (sp.Rational(7, 13), chart_covector(form_parity, 1, -1)),
    )
    normalized = append_normalization_rows(
        matrix, degree, normalization_rows
    )
    normalized_rank = normalized.rank()
    return RankAudit(
        labels=labels,
        degree=degree,
        equations=matrix.rows,
        unknowns=unknowns,
        rank=rank,
        nullity=unknowns - rank,
        rank_after_two_normalizations=normalized_rank,
        nullity_after_two_normalizations=unknowns - normalized_rank,
    )


def audit():
    # The root-count identity is exact throughout the stored positive grid.
    checked = 0
    for labels in itertools.product(
        (sp.Integer(0), sp.Rational(1, 2), sp.Integer(1)),
        (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4)),
        (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4)),
    ):
        root_count_identity(*labels)
        checked += 1

    hard_even = rank_audit(0, sp.Rational(3, 4), sp.Rational(3, 4), 0)
    hard_odd = rank_audit(0, sp.Rational(3, 4), sp.Rational(3, 4), 1)
    for result in (hard_even, hard_odd):
        if (
            result.degree,
            result.equations,
            result.unknowns,
            result.rank,
            result.nullity,
            result.rank_after_two_normalizations,
            result.nullity_after_two_normalizations,
        ) != (4, 4, 10, 4, 6, 6, 4):
            raise AssertionError(result)

    print(f"mixed ell root count: {checked} exact label triples passed")
    print(
        "hard zero system (both form parities): "
        "D=4, equations=4, unknowns=10, rank=4, nullity=6"
    )
    print(
        "after two independent minimal-screening normalization rows: "
        "rank=6, nullity=4"
    )
    print(
        "verdict: scalar 2013 zeros plus two normalizations do not "
        "reconstruct the Ramond two-form polynomial"
    )


if __name__ == "__main__":
    audit()
