"""Direct Selberg evaluation of the Ramond contour alternant.

``screening_pfaffian.alternant_schur_coefficients`` performs the fermionic
calculation with univariate coefficient matrices and maximal minors.  This
module performs the remaining integral.  It intentionally does not expand
the Vandermonde or any polynomial in all screening variables.

If

    insertion(t) = Delta(t)^2 sum_lambda c_lambda s_lambda(t),

then the Vandermonde square changes the Selberg coupling from ``g`` to
``g+1``.  Every Schur remainder is integrated by the bounded-width dual
Cauchy oracle in ``selberg_elementary``.  For support of width ``k`` the
finite state count is ``binomial(N+k,k)`` rather than the number of PBW
states or unrestricted partitions at the total descendant grade.
"""

from __future__ import annotations

import time

import sympy as sp

from .screening_pfaffian import (
    SQRT2,
    alternant_schur_coefficients,
    alternant_selberg_ratio,
    natural_ramond_parity,
    selberg_ratio,
)
from .selberg_elementary import (
    normalized_bounded_width_schur,
    rectangle_partition_count,
)
from .special_oracle import ordinary_selberg, physical_nsrr_selberg


def normalized_schur_sum(coefficients, screenings, A, B, g):
    """Average a sparse bounded-width Schur expansion exactly."""

    answer = sp.Integer(0)
    for partition, coefficient in coefficients.items():
        if coefficient:
            answer += coefficient * normalized_bounded_width_schur(
                partition, screenings, A, B, g
            )
    return sp.factor(sp.cancel(answer))


def bounded_alternant_selberg_ratio(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    A,
    B,
    g,
    epsilon2=None,
    epsilon3=None,
):
    """Fast maximal-screening contour integral / primary Ising integral."""

    count, shift_A, shift_B, coefficients = alternant_schur_coefficients(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
    )
    shifted_A = A - shift_A
    shifted_B = B - shift_B
    shifted_g = g + 1
    numerator = ordinary_selberg(count, shifted_A, shifted_B, shifted_g)
    numerator *= normalized_schur_sum(
        coefficients, count, shifted_A, shifted_B, shifted_g
    )
    denominator = physical_nsrr_selberg(count, A, B, g)
    if count % 2:
        denominator /= SQRT2
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True)
    )


def support_bound(coefficients):
    """Return ``(largest_width, total_rectangle_states_by_degree_bound)``."""

    if not coefficients:
        return 0, 0
    width = max((partition[0] if partition else 0) for partition in coefficients)
    # This is the size of the full rectangle generating system.  A request
    # at one homogeneous degree normally visits fewer partitions.
    return width


def audit():
    """Check the bounded-width route against two independent exact routes."""

    A, B, g = sp.symbols("A B g", nonzero=True)
    cases = (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(3, 4)),
        (1, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(7, 4), sp.Rational(1, 4)),
    )
    for labels in cases:
        epsilon2 = natural_ramond_parity(labels[1])
        epsilon3 = natural_ramond_parity(labels[2])
        calculated = bounded_alternant_selberg_ratio(
            *labels,
            0,
            1,
            A,
            B,
            g,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        generic = alternant_selberg_ratio(
            *labels,
            0,
            1,
            A,
            B,
            g,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        if sp.factor(sp.cancel(calculated - generic)) != 0:
            raise AssertionError((labels, calculated, generic))
    print(f"bounded alternant: {len(cases)} exact generic-Jack checks passed")

    # For N<=3 also compare with the literal multivariate Pfaffian,
    # Vandermonde multiplication, polynomial expansion, and Selberg average.
    # This independently checks every coefficient-matrix minor and sign.
    numerical_parameters = (sp.Rational(2, 7), sp.Rational(3, 11), sp.Rational(5, 13))
    direct_cases = (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(3, 4)),
    )
    for labels in direct_cases:
        epsilon2 = natural_ramond_parity(labels[1])
        epsilon3 = natural_ramond_parity(labels[2])
        calculated = bounded_alternant_selberg_ratio(
            *labels,
            0,
            1,
            *numerical_parameters,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        expanded = selberg_ratio(
            *labels,
            0,
            1,
            *numerical_parameters,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        if sp.factor(sp.cancel(calculated - expanded)) != 0:
            raise AssertionError(("expanded Pfaffian", labels, calculated, expanded))
    print(f"bounded alternant: {len(direct_cases)} literal-Pfaffian checks passed")


def benchmark():
    """Run the former ``v_2`` / ``W_7/4`` bottleneck state-free."""

    A, B, g = sp.Rational(2, 7), sp.Rational(3, 11), sp.Rational(5, 13)
    cases = (
        (2, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(7, 4), sp.Rational(1, 4)),
        (2, sp.Rational(7, 4), sp.Rational(7, 4)),
    )
    for labels in cases:
        epsilon2 = natural_ramond_parity(labels[1])
        epsilon3 = natural_ramond_parity(labels[2])
        started = time.perf_counter()
        count, _, _, coefficients = alternant_schur_coefficients(
            *labels,
            0,
            1,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        value = bounded_alternant_selberg_ratio(
            *labels,
            0,
            1,
            A,
            B,
            g,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )
        elapsed = time.perf_counter() - started
        width = support_bound(coefficients)
        states = rectangle_partition_count(count, width)
        print(
            f"labels={labels}, N={count}, Schur_terms={len(coefficients)}, "
            f"width={width}, rectangle_states={states}, seconds={elapsed:.3f}, "
            f"digits={len(str(value))}"
        )


if __name__ == "__main__":
    audit()
    benchmark()
