"""The two-spin Ising polynomial in the repository puncture order.

The punctures are ``(0,1,infinity)=(R,R,NS)``.  Consequently the order
labels in arXiv:1011.4090 are ``(k1,k2,k3)=(1,1,0)``.  This is easy to get
wrong by reading "NS--R--R" as a position-ordered list.

Multiplication of the two-spin Majorana correlator by the square-root
prefactor and by the Vandermonde removes every radical.  The result below
is computed by one Pfaffian.  ``bfl_110_polynomial`` implements the
independent symmetrized definition of the same polynomial.
"""

from __future__ import annotations

import itertools

import sympy as sp

from .core import pfaffian


def _vandermonde(variables):
    answer = sp.Integer(1)
    for first in range(len(variables)):
        for second in range(first + 1, len(variables)):
            answer *= variables[first] - variables[second]
    return answer


def spin_pfaffian_polynomial(variables):
    """Return the radical-free two-spin Majorana Pfaffian polynomial.

    For even screening number the matrix is the even two-spin kernel.  For
    odd screening number it is augmented by the one-fermion kernel.  The
    latter is the standard Pfaffian representation of an odd Gaussian
    correlator.
    """

    variables = tuple(map(sp.sympify, variables))
    count = len(variables)
    size = count if count % 2 == 0 else count + 1
    matrix = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    for first in range(count):
        for second in range(first + 1, count):
            left, right = variables[first], variables[second]
            value = (left + right - 2 * left * right) / (2 * (left - right))
            matrix[first][second] = value
            matrix[second][first] = -value
    if count % 2:
        for index in range(count):
            matrix[index][count] = 1 / sp.sqrt(2)
            matrix[count][index] = -1 / sp.sqrt(2)
    return sp.factor(_vandermonde(variables) * pfaffian(matrix))


def bfl_110_polynomial(variables):
    """Return the normalized BFL polynomial for labels ``(1,1,0)``.

    This specializes equations (3.7)--(3.10) and (A.36)--(A.39) of
    arXiv:1011.4090 to ``N=2``.  The sum is used only as an independent
    source-formula audit; production evaluation uses the cubic Pfaffian.
    """

    variables = tuple(map(sp.sympify, variables))
    count = len(variables)
    indices = tuple(range(count))
    answer = sp.Integer(0)

    if count % 2:  # primal: count=2m-1, k=1
        m = (count + 1) // 2
        for small_indices in itertools.combinations(indices, m - 1):
            small = tuple(variables[index] for index in small_indices)
            large = tuple(variables[index] for index in indices if index not in small_indices)
            answer += (
                _vandermonde(small) ** 2
                * _vandermonde(large) ** 2
                * sp.prod(value * (value - 1) for value in small)
            )
        # The projective limit (3.6) contributes
        # (-1)^((m-1) k1)=(-1)^(m-1) for (k1,k2,k3)=(1,1,0).
        return sp.factor((-1) ** (m - 1) * answer / 2 ** (m - 1))

    # dual: count=2m, k=0
    m = count // 2
    for first_indices in itertools.combinations(indices, m):
        first = tuple(variables[index] for index in first_indices)
        second = tuple(variables[index] for index in indices if index not in first_indices)
        answer += (
            _vandermonde(first) ** 2
            * _vandermonde(second) ** 2
            * sp.prod(first)
            * sp.prod(value - 1 for value in second)
        )
    # The dual projective limit (A.35) contributes
    # (-1)^(m(N-k1))=(-1)^m.
    return sp.factor((-1) ** m * answer / 2**m)


def audit(max_screenings=5):
    """Check the Pfaffian against the independent BFL symmetrization."""

    ratios = []
    for count in range(1, int(max_screenings) + 1):
        variables = sp.symbols(f"t0:{count}")
        pf = spin_pfaffian_polynomial(variables)
        bfl = bfl_110_polynomial(variables)
        ratio = sp.factor(pf / bfl)
        if any(variable in ratio.free_symbols for variable in variables):
            raise AssertionError((count, ratio))
        if sp.factor(pf - ratio * bfl) != 0:
            raise AssertionError((count, ratio))
        ratios.append(ratio)
        print(f"screenings={count}: Pfaffian/BFL={ratio}")
    return tuple(ratios)


if __name__ == "__main__":
    audit()
