"""Polynomial-size Selberg moments for bounded-width alternants.

The consecutive Ramond ``chi`` contours produce an alternant after the
common one-body denominator is cleared.  If its row polynomials have degree
at most ``N+k-1``, division by the Vandermonde gives only Schur polynomials
of width at most ``k``.  Jacobi--Trudi in conjugate form reduces those
Schurs to products of at most ``k`` elementary symmetric functions.

This file evaluates precisely those products without constructing a Jack
basis in ``N`` screening variables.  The dual Jack Cauchy identity is

    prod[a=1..k] prod[i=1..N] (1 + x_a t_i)
      = sum_lambda P_lambda^(1/g)(t) P_(lambda')^g(x),

where the sum is over ``lambda`` contained in the ``N by k`` rectangle.
Taking the normalized Selberg average and applying Kadell's one-Jack
formula leaves Jack polynomials in only ``k`` variables.  Therefore the
number of terms is ``binomial(N+k,k)`` for the complete generating
polynomial, and is polynomial in ``N`` whenever the width ``k`` is fixed.

No superconformal state, PBW basis, or numerical interpolation occurs here.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
import time

import sympy as sp

from .selberg_jack import (
    jack_polynomial,
    kadell_normalized_jack,
    normalized_selberg_average,
    partitions,
    variables,
)


def conjugate_partition(partition):
    """Return the conjugate Young diagram as a tuple."""

    partition = tuple(int(part) for part in partition if part)
    if not partition:
        return ()
    return tuple(
        sum(row_length >= column for row_length in partition)
        for column in range(1, partition[0] + 1)
    )


def rectangle_partition_count(screenings: int, width: int) -> int:
    """Number of partitions contained in an ``screenings x width`` box."""

    screenings, width = int(screenings), int(width)
    if screenings < 0 or width < 0:
        raise ValueError((screenings, width))
    return comb(screenings + width, width)


@lru_cache(None)
def _dual_jack_monomial_coefficient(partition, exponents, alpha):
    """Coefficient needed on the dual side of the Cauchy identity."""

    partition = tuple(map(int, partition))
    exponents = tuple(map(int, exponents))
    width = len(exponents)
    dual = conjugate_partition(partition)
    polynomial = sp.Poly(jack_polynomial(dual, alpha, width), *variables(width))
    return polynomial.coeff_monomial(exponents)


def normalized_elementary_product(indices, screenings, A, B, g):
    r"""Return the normalized Selberg moment of ``prod_a e_{indices[a]}``.

    The Selberg weight is

    ``prod_i t_i**A (1-t_i)**B prod_{i<j}(t_i-t_j)**(2*g)``.

    Zero indices are discarded.  An index outside ``0,...,screenings``
    makes the elementary symmetric function vanish.  The calculation uses
    only Jack polynomials in ``len(indices)`` variables.
    """

    screenings = int(screenings)
    if screenings < 0:
        raise ValueError(screenings)
    indices = tuple(sorted((int(index) for index in indices if index), reverse=True))
    if any(index < 0 for index in indices):
        raise ValueError(indices)
    if any(index > screenings for index in indices):
        return sp.Integer(0)
    if not indices:
        return sp.Integer(1)

    width = len(indices)
    degree = sum(indices)
    answer = sp.Integer(0)
    dual_alpha = sp.sympify(g)

    # Only partitions of the requested homogeneous degree can contribute.
    # The maximum-part and length restrictions are the N by k rectangle.
    for partition in partitions(degree, maximum=width, length=screenings):
        coefficient = _dual_jack_monomial_coefficient(
            partition, indices, dual_alpha
        )
        if coefficient:
            answer += coefficient * kadell_normalized_jack(
                partition, screenings, A, B, g
            )
    return sp.factor(sp.cancel(answer))


def normalized_bounded_width_schur(partition, screenings, A, B, g):
    r"""Average a Schur polynomial using conjugate Jacobi--Trudi.

    This routine is intended for bounded width.  If ``lambda'`` has length
    ``k``, the determinant has ``k!`` terms and every entry is an elementary
    symmetric function.  In the Ramond application ``k`` is fixed by the
    number of exceptional zero-mode/contact rows (at most a small constant),
    not by the descendant level.
    """

    partition = tuple(int(part) for part in partition if part)
    if not partition:
        return sp.Integer(1)
    if len(partition) > int(screenings):
        return sp.Integer(0)
    dual = conjugate_partition(partition)
    width = len(dual)
    matrix = sp.zeros(width)
    # We expand the small determinant ourselves so that each term can be
    # sent directly to the elementary-product oracle.
    for row in range(width):
        for column in range(width):
            matrix[row, column] = dual[row] - row + column

    answer = sp.Integer(0)
    from itertools import permutations

    for permutation in permutations(range(width)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(width)
            for right in range(left + 1, width)
        )
        indices = tuple(matrix[row, permutation[row]] for row in range(width))
        # e_r=0 for r<0 and e_0=1.
        if any(index < 0 for index in indices):
            continue
        term = normalized_elementary_product(indices, screenings, A, B, g)
        answer += (-1) ** inversions * term
    return sp.factor(sp.cancel(answer))


def audit():
    """Exact low-rank checks against Aomoto and the full N-variable backend."""

    A, B, g = sp.symbols("A B g", nonzero=True)

    # Width one is Aomoto's elementary-symmetric product formula.
    N = 4
    for degree in range(N + 1):
        calculated = normalized_elementary_product((degree,), N, A, B, g)
        expected = sp.binomial(N, degree)
        for index in range(1, degree + 1):
            expected *= A + 1 + (N - index) * g
            expected /= A + B + 2 + (2 * N - index - 1) * g
        if sp.factor(sp.cancel(calculated - expected)) != 0:
            raise AssertionError((degree, calculated, expected))

    # Products and bounded-width Schurs agree with the independent generic
    # Jack decomposition in the N screening variables.
    N = 3
    xs = variables(N)
    elementary = lambda degree: sp.Poly.from_dict(
        {
            tuple(1 if index in chosen else 0 for index in range(N)): 1
            for chosen in __import__("itertools").combinations(range(N), degree)
        },
        xs,
    ).as_expr()
    for indices in ((1, 1), (2, 1), (2, 2), (3, 2, 1)):
        polynomial = sp.prod(elementary(degree) for degree in indices)
        direct = normalized_selberg_average(polynomial, xs, A, B, g)
        dual = normalized_elementary_product(indices, N, A, B, g)
        if sp.factor(sp.cancel(direct - dual)) != 0:
            raise AssertionError((indices, direct, dual))

    for partition in ((2,), (2, 1), (2, 2), (3, 1), (3, 2, 1)):
        schur = jack_polynomial(partition, sp.Integer(1), N)
        direct = normalized_selberg_average(schur, xs, A, B, g)
        bounded = normalized_bounded_width_schur(partition, N, A, B, g)
        if sp.factor(sp.cancel(direct - bounded)) != 0:
            raise AssertionError((partition, direct, bounded))

    print("dual-Cauchy Selberg oracle: exact width<=3 checks passed")


def benchmark():
    """Demonstrate polynomial scaling on levels beyond the PBW bottleneck."""

    # Rational parameters keep this a reproducible exact-arithmetic timing,
    # while the algorithm and identities are parameter-independent.
    A, B, g = sp.Rational(2, 7), sp.Rational(3, 11), sp.Rational(5, 13)
    cases = ((5, (5,)), (8, (7, 5)), (12, (10, 7)), (12, (9, 7, 4)))
    for screenings, indices in cases:
        started = time.perf_counter()
        value = normalized_elementary_product(indices, screenings, A, B, g)
        elapsed = time.perf_counter() - started
        width = len(indices)
        print(
            f"N={screenings}, width={width}, rectangle_states="
            f"{rectangle_partition_count(screenings, width)}, "
            f"seconds={elapsed:.3f}, digits={len(str(value))}"
        )


if __name__ == "__main__":
    audit()
    benchmark()
