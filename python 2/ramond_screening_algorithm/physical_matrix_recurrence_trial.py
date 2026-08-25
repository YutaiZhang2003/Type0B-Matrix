#!/usr/bin/env python3
"""First exact physical-matrix and Ramond-collision recurrence trial.

This module contains two independent finite certificates.

1.  Put the two fixed NS--R--R chiral forms ``eta=+,-`` in the rows and
    the two Ramond multiplicity copies on the second leg in the columns.
    The known ground and first hard PBW matrix elements then determine one
    exact left transfer matrix.  The calculation proves that its scalar
    ell-product part, crossed correction, and copy/ground mixing separate.

2.  The radical-free two-spin Majorana polynomial and its endpoint-reflected
    partner obey a diagonal collision rule.  At the genuine Selberg pole the
    two endpoint residues are complementary rank-one projectors.  The audit
    checks the first nontrivial polynomial identities and the exact projector
    seed.  It also records why a tempting extra half-shift from radial power
    counting is only a would-be pole: its angular beta factor vanishes.

The first certificate uses the already independently checked hard
polynomials K and H.  It is therefore evidence for the form of a recurrence,
not a derivation of K or H at general branch label.
"""

from __future__ import annotations

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)


def pfaffian(matrix):
    """Recursive Pfaffian, sufficient for the N=2,4 identity audit."""

    size = len(matrix)
    if size == 0:
        return sp.Integer(1)
    if size % 2:
        raise ValueError("the Pfaffian matrix must have even size")
    answer = sp.Integer(0)
    for column in range(1, size):
        keep = [index for index in range(1, size) if index != column]
        minor = [[matrix[row][entry] for entry in keep] for row in keep]
        answer += (-1) ** (column + 1) * matrix[0][column] * pfaffian(minor)
    return sp.expand(answer)


def radical_free_spin_polynomial(variables):
    r"""Return ``2^(-m) Delta Pf((t_i+t_j)/(t_i-t_j))`` for N=2m."""

    variables = tuple(map(sp.sympify, variables))
    size = len(variables)
    if size % 2:
        raise ValueError("the even two-spin polynomial needs N=2m")
    matrix = [
        [
            sp.Integer(0)
            if row == column
            else (variables[row] + variables[column])
            / (variables[row] - variables[column])
            for column in range(size)
        ]
        for row in range(size)
    ]
    vandermonde = sp.prod(
        variables[row] - variables[column]
        for row in range(size)
        for column in range(row + 1, size)
    )
    return sp.factor(
        sp.cancel(vandermonde * pfaffian(matrix) / 2 ** (size // 2))
    )


def radical_free_bordered_spin_polynomial(variables):
    r"""Odd-N bordered two-spin polynomial, normalized to ``P_hat_1=1``.

    The last Pfaffian row is the radical-free Ramond zero-mode row.  This is
    the odd-screening analogue of :func:`radical_free_spin_polynomial`.
    """

    variables = tuple(map(sp.sympify, variables))
    size = len(variables)
    if size % 2 != 1:
        raise ValueError("the bordered two-spin polynomial needs odd N")
    matrix = []
    for row in range(size + 1):
        current = []
        for column in range(size + 1):
            if row == column:
                entry = sp.Integer(0)
            elif row == size:
                entry = -sp.Integer(1)
            elif column == size:
                entry = sp.Integer(1)
            else:
                entry = (variables[row] + variables[column]) / (
                    variables[row] - variables[column]
                )
            current.append(entry)
        matrix.append(current)
    vandermonde = sp.prod(
        variables[row] - variables[column]
        for row in range(size)
        for column in range(row + 1, size)
    )
    return sp.factor(
        sp.cancel(vandermonde * pfaffian(matrix) / 2 ** ((size - 1) // 2))
    )


def hard_polynomials():
    """Return the independently defined first factorized/crossed data."""

    q, p1, p2, p3 = sp.symbols("Q P_1 P_2 P_3")
    e2 = q + 2 * p2
    e3 = q + 2 * p3
    d2 = e2**2 + q * e2 + 1
    d3 = e3**2 + q * e3 + 1
    x_pp = q / 2 + p1 + p2 + p3
    x_2 = q / 2 - p1 + p2 + p3
    x_mm = q / 2 + p1 - p2 - p3
    factorized = sp.expand(
        (x_pp**2 + q * x_pp + 1) * (x_2**2 + q * x_2 + 1)
    )
    line = sp.expand(x_pp * (x_mm - q))
    crossed = sp.expand(
        line**2 + 2 * line * (e2 * e3 + 1) + d2 * d3
    )
    return (q, p1, p2, p3), factorized, crossed, d2, d3


def ground_physical_matrix():
    r"""Rows ``eta=(+,-)``, columns ``epsilon_2=(0,1)`` at (0,1/4,1/4)."""

    return sp.Matrix(
        [
            [1 + I, -(1 - I) / SQRT2],
            [1 - I, -(1 + I) / SQRT2],
        ]
    )


def hard_physical_matrix():
    r"""Rows ``eta=(+,-)``, columns ``epsilon_2=(0,1)`` at (0,3/4,3/4)."""

    _, factorized, crossed, d2, d3 = hard_polynomials()
    r_plus = -(1 + I) * factorized / (d2 * d3)
    r_minus = -(1 - I) * crossed / (d2 * d3)
    return sp.Matrix(
        [
            [r_plus, I * SQRT2 * r_plus],
            [r_minus, -I * SQRT2 * r_minus],
        ]
    )


def first_physical_transfer():
    """Return ``M_hard M_ground^{-1}`` and its channel-separated form."""

    _, factorized, crossed, d2, d3 = hard_polynomials()
    transfer = (hard_physical_matrix() * ground_physical_matrix().inv()).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    universal = sp.Matrix([[-3, I], [-I, -3]]) / 2
    separated = (
        sp.diag(factorized, crossed) * universal / (d2 * d3)
    ).applyfunc(lambda entry: sp.factor(sp.cancel(entry)))
    return transfer, separated, universal


def selberg_two_channel_seed(A, B, g):
    r"""Normalized N=2 averages of P(t) and P(1-t).

    Since ``P_2=(t_1+t_2)/2``, the first value is the elementary Aomoto
    average.  Its reflected partner is obtained by ``t -> 1-t``.
    """

    denominator = A + B + 2 + 2 * g
    return (
        sp.factor((A + 1 + g) / denominator),
        sp.factor((B + 1 + g) / denominator),
    )


def endpoint_collision_data(screenings, g):
    r"""Return the two endpoint projectors and their common residue factor.

    The component order is ``(P(t),P(1-t))``.  At ``A=-g-1`` only the
    reflected component survives; at ``B=-g-1`` only the unreflected one
    survives.  The scalar is the Hadasz--Jaskolski/BFL two-screen cluster
    factor for an integral symmetric in all screening variables.
    """

    screenings = int(screenings)
    if screenings < 2:
        raise ValueError("the collision recurrence needs at least two screenings")
    scalar = (
        sp.Rational(screenings * (screenings - 1), 2)
        * sp.gamma(-g)
        * sp.gamma(1 + 2 * g)
        / sp.gamma(1 + g)
    )
    at_zero = sp.diag(0, 1)
    at_one = sp.diag(1, 0)
    return scalar, at_zero, at_one


def audit() -> None:
    transfer, separated, universal = first_physical_transfer()
    if sp.simplify(transfer - separated) != sp.zeros(2):
        raise AssertionError((transfer, separated))
    if sp.factor(ground_physical_matrix().det() + 2 * SQRT2 * I) != 0:
        raise AssertionError("ground physical matrix is unexpectedly singular")
    if tuple(sorted(universal.eigenvals())) != (-2, -1):
        raise AssertionError(universal.eigenvals())
    copy_eigenvalues = sp.diag(-1, -2)
    if sp.simplify(
        universal * ground_physical_matrix()
        - ground_physical_matrix() * copy_eigenvalues
    ) != sp.zeros(2):
        raise AssertionError("the two Ramond copies did not diagonalize the mixing")

    t1, t2, t3, t4, x = sp.symbols("t_1 t_2 t_3 t_4 x")
    p2 = radical_free_spin_polynomial((t3, t4))
    p4 = radical_free_spin_polynomial((t1, t2, t3, t4))
    cluster = sp.factor(
        p4.subs({t1: x, t2: x})
        - x * (x - t3) ** 2 * (x - t4) ** 2 * p2
    )
    if cluster != 0:
        raise AssertionError(cluster)
    reflected2 = p2.subs({t3: 1 - t3, t4: 1 - t4})
    reflected4 = p4.subs(
        {t1: 1 - t1, t2: 1 - t2, t3: 1 - t3, t4: 1 - t4},
        simultaneous=True,
    )
    reflected_cluster = sp.factor(
        reflected4.subs({t1: x, t2: x})
        - (1 - x) * (x - t3) ** 2 * (x - t4) ** 2 * reflected2
    )
    if reflected_cluster != 0:
        raise AssertionError(reflected_cluster)

    p1 = radical_free_bordered_spin_polynomial((t3,))
    p3 = radical_free_bordered_spin_polynomial((t1, t2, t3))
    bordered_cluster = sp.factor(
        p3.subs({t1: x, t2: x}) - x * (x - t3) ** 2 * p1
    )
    if p1 != 1 or bordered_cluster != 0:
        raise AssertionError((p1, bordered_cluster))

    A, B, g = sp.symbols("A B g")
    seed, reflected_seed = selberg_two_channel_seed(A, B, g)
    if sp.factor(seed + reflected_seed - 1) != 0:
        raise AssertionError((seed, reflected_seed))

    # At the true Selberg pole A=-g-1 the first normalized seed vanishes and
    # the reflected one equals one.  The endpoint B=-g-1 reverses them.
    collision_pole = -g - 1
    at_zero_seed = (sp.factor(seed.subs(A, collision_pole)), sp.factor(reflected_seed.subs(A, collision_pole)))
    at_one_seed = (sp.factor(seed.subs(B, collision_pole)), sp.factor(reflected_seed.subs(B, collision_pole)))
    if at_zero_seed != (0, 1) or at_one_seed != (1, 0):
        raise AssertionError((at_zero_seed, at_one_seed))
    remaining_exponent = sp.factor(collision_pole + 4 * g + 2)
    if sp.factor(remaining_exponent - 3 * g - 1) != 0:
        raise AssertionError(remaining_exponent)
    scalar, projector_zero, projector_one = endpoint_collision_data(4, g)
    if projector_zero + projector_one != sp.eye(2):
        raise AssertionError((projector_zero, projector_one))
    expected_scalar = 6 * sp.gamma(-g) * sp.gamma(1 + 2 * g) / sp.gamma(1 + g)
    if sp.simplify(scalar - expected_scalar) != 0:
        raise AssertionError(scalar)

    # The unreflected polynomial has a radial factor tau at the zero
    # endpoint.  Naive radial counting suggests A=-g-3/2, but the angular
    # coefficient is proportional to B(g+1/2,-g-1/2)=0 because of the
    # reciprocal Gamma(0).  Hence there is no additional meromorphic pole.
    would_be_pole = -g - sp.Rational(3, 2)
    angular_zero = 1 / sp.gamma(
        (g + sp.Rational(1, 2)) + (would_be_pole + 1)
    )
    if angular_zero != 0:
        raise AssertionError(angular_zero)

    print("physical hard transfer: exact 2 by 2 factorization")
    print("universal copy/ground eigenvalues: (-2,-1)")
    print("ground epsilon_2 columns are the exact (-1,-2) eigenvectors")
    print("Ramond spin/reflected clustering: N=4 -> 2 and N=3 -> 1 exact")
    print("N=2 endpoint residues are complementary rank-one projectors")
    print(
        "collision recurrence shifts: "
        "A_pole=-g-1, A_remaining=3g+1"
    )
    print("the would-be A=-g-3/2 pole is killed by the angular Gamma zero")
    print("scope: first physical step and ground collision kernel, not all-level proof")


if __name__ == "__main__":
    audit()
