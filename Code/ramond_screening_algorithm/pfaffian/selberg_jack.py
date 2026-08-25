"""Exact Selberg averages of finite symmetric polynomials via Jack basis.

This is a state-free fallback for staircase/Pfaffian insertions which do
not collapse to one product.  A symmetric polynomial is decomposed into
monomial symmetric functions, converted to monic Jack polynomials by the
Laplace--Beltrami triangular recurrence, and averaged with Kadell's
formula.  No numerical quadrature or interpolation is used.

The implementation is intended for the relatively sparse polynomials
produced by the external-chi Pfaffian.  Its cost is polynomial in the
number of monomials once the partitions appearing at each degree have
been generated; unlike PBW transport it never constructs an SCA Verma
basis.
"""

from __future__ import annotations

from functools import lru_cache
import itertools

import sympy as sp


def partitions(total: int, maximum: int | None = None, length: int | None = None):
    """Yield partitions of ``total`` in reverse lexicographic order."""

    total = int(total)
    if total < 0:
        return
    if maximum is None:
        maximum = total
    maximum = min(int(maximum), total)
    if length is None:
        length = total if total else 0
    length = int(length)
    if total == 0:
        yield ()
        return
    if length <= 0:
        return
    for first in range(maximum, 0, -1):
        for tail in partitions(total - first, first, length - 1):
            yield (first,) + tail


def _distinct_permutations(values):
    """Yield distinct permutations without materializing all duplicates."""

    values = tuple(values)
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = tuple(sorted(counts, reverse=True))

    def visit(prefix):
        if len(prefix) == len(values):
            yield tuple(prefix)
            return
        for value in ordered:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            prefix.append(value)
            yield from visit(prefix)
            prefix.pop()
            counts[value] += 1

    yield from visit([])


@lru_cache(None)
def variables(count: int):
    return sp.symbols(f"x0:{int(count)}")


@lru_cache(None)
def monomial_symmetric(partition: tuple[int, ...], variable_count: int):
    """Return the monomial symmetric polynomial ``m_partition``."""

    variable_count = int(variable_count)
    partition = tuple(map(int, partition))
    if len(partition) > variable_count:
        return sp.Integer(0)
    exponents = partition + (0,) * (variable_count - len(partition))
    xs = variables(variable_count)
    return sp.Add(
        *(
            sp.prod(x**exponent for x, exponent in zip(xs, permutation))
            for permutation in _distinct_permutations(exponents)
        )
    )


def laplace_beltrami(polynomial, alpha, xs):
    """Jack Laplace--Beltrami operator in ``len(xs)`` variables."""

    answer = sp.Integer(0)
    for x in xs:
        answer += alpha * x**2 * sp.diff(polynomial, x, 2) / 2
    for i, x in enumerate(xs):
        derivative = sp.diff(polynomial, x)
        for j, y in enumerate(xs):
            if i == j:
                continue
            answer += x**2 * derivative / (x - y)
    return sp.Poly(sp.cancel(answer), *xs).as_expr()


def jack_eigenvalue(partition, alpha, variable_count):
    return sp.expand(
        alpha
        * sum(sp.Rational(part * (part - 1), 2) for part in partition)
        + sum((variable_count - index - 1) * part for index, part in enumerate(partition))
    )


@lru_cache(None)
def _jack_transition(variable_count: int, degree: int, alpha):
    """Return partitions and the monomial columns of monic Jack P's."""

    variable_count = int(variable_count)
    degree = int(degree)
    alpha = sp.sympify(alpha)
    parts = tuple(partitions(degree, length=variable_count))
    if degree == 0:
        return parts, sp.Matrix(((1,),))
    xs = variables(variable_count)
    monomials = tuple(monomial_symmetric(part, variable_count) for part in parts)
    canonical = tuple(part + (0,) * (variable_count - len(part)) for part in parts)

    # Columns are D(m_lambda) in the m_mu basis.  Reverse lexicographic
    # order refines dominance, so this matrix is lower triangular.
    operator = sp.zeros(len(parts))
    for column, monomial in enumerate(monomials):
        image = sp.Poly(laplace_beltrami(monomial, alpha, xs), *xs)
        for row, exponent in enumerate(canonical):
            operator[row, column] = image.coeff_monomial(exponent)

    transition = sp.zeros(len(parts))
    for start, part in enumerate(parts):
        eigenvalue = jack_eigenvalue(part, alpha, variable_count)
        coefficients = [sp.Integer(0)] * len(parts)
        coefficients[start] = sp.Integer(1)
        for row in range(start + 1, len(parts)):
            numerator = sum(
                operator[row, column] * coefficients[column]
                for column in range(start, row)
            )
            denominator = operator[row, row] - eigenvalue
            if denominator == 0:
                # Generic alpha has a simple spectrum.  A zero here means
                # that the caller specialized alpha too early to an
                # accidental degeneracy; retaining alpha symbolically is
                # the exact remedy.
                if numerator != 0:
                    raise ZeroDivisionError(
                        f"degenerate Jack spectrum at {part}, row {row}"
                    )
                coefficients[row] = 0
            else:
                coefficients[row] = sp.cancel(-numerator / denominator)
        transition[:, start] = sp.Matrix(coefficients)
    return parts, transition


def jack_polynomial(partition, alpha, variable_count):
    partition = tuple(map(int, partition))
    degree = sum(partition)
    parts, transition = _jack_transition(variable_count, degree, sp.sympify(alpha))
    column = parts.index(partition)
    return sp.expand(
        sum(
            transition[row, column] * monomial_symmetric(part, variable_count)
            for row, part in enumerate(parts)
        )
    )


def jack_at_ones(partition, alpha, variable_count):
    """Evaluation of the monic Jack ``P_partition`` at ``1^N``."""

    partition = tuple(map(int, partition))
    alpha = sp.sympify(alpha)
    answer = sp.Integer(1)
    for row, row_length in enumerate(partition, start=1):
        for column in range(1, row_length + 1):
            arm_length = row_length - column
            leg_length = sum(
                other_length >= column
                for other_length in partition[row:]
            )
            arm_colength = column - 1
            leg_colength = row - 1
            answer *= (
                variable_count + alpha * arm_colength - leg_colength
            ) / (alpha * arm_length + leg_length + 1)
    return sp.factor(answer)


def kadell_normalized_jack(partition, variable_count, A, B, g):
    """Normalized Selberg average of one monic Jack polynomial."""

    partition = tuple(map(int, partition))
    alpha = sp.Integer(1) / g
    answer = jack_at_ones(partition, alpha, variable_count)
    for index, row_length in enumerate(partition, start=1):
        answer *= sp.rf(A + 1 + (variable_count - index) * g, row_length)
        answer /= sp.rf(
            A + B + 2 + (2 * variable_count - index - 1) * g,
            row_length,
        )
    return sp.factor(answer)


def monomial_coefficients(polynomial, xs):
    """Decompose a symmetric polynomial into monomial symmetric functions."""

    poly = sp.Poly(sp.expand(polynomial), *xs)
    answer: dict[tuple[int, ...], sp.Expr] = {}
    for exponent, coefficient in poly.terms():
        partition = tuple(sorted((item for item in exponent if item), reverse=True))
        canonical = partition + (0,) * (len(xs) - len(partition))
        if exponent != canonical:
            continue
        answer[partition] = coefficient

    # Rebuild exactly.  This both checks symmetry and catches a missing
    # orbit caused by an incorrect contour convention.
    canonical_xs = variables(len(xs))
    rename = dict(zip(canonical_xs, xs))
    rebuilt = sum(
        coefficient
        * monomial_symmetric(partition, len(xs)).xreplace(rename)
        for partition, coefficient in answer.items()
    )
    if sp.Poly(sp.expand(poly.as_expr() - rebuilt), *xs) != sp.Poly(0, *xs):
        raise ValueError("the insertion is not a symmetric polynomial")
    return answer


def normalized_selberg_average(polynomial, xs, A, B, g):
    """Return ``<polynomial>_Selberg / <1>_Selberg`` exactly."""

    xs = tuple(xs)
    by_monomial = monomial_coefficients(polynomial, xs)
    by_degree: dict[int, dict[tuple[int, ...], sp.Expr]] = {}
    for partition, coefficient in by_monomial.items():
        by_degree.setdefault(sum(partition), {})[partition] = coefficient

    answer = sp.Integer(0)
    alpha = sp.cancel(1 / g)
    for degree, coefficients in by_degree.items():
        parts, transition = _jack_transition(len(xs), degree, alpha)
        monomial_vector = sp.Matrix(
            [coefficients.get(partition, 0) for partition in parts]
        )
        # m-coefficients = transition * Jack-coefficients.
        jack_coefficients = transition.inv() * monomial_vector
        for coefficient, partition in zip(jack_coefficients, parts):
            if coefficient:
                answer += coefficient * kadell_normalized_jack(
                    partition, len(xs), A, B, g
                )
    return sp.factor(sp.cancel(answer))


def normalized_selberg_average_schur(coefficients, variable_count, A, B, g):
    """Average a sparse Schur expansion without expanding its variables.

    ``coefficients`` maps partitions to coefficients.  At ``alpha=1`` the
    monic Jack ``P`` is the Schur polynomial, so the first Jack transition
    gives its monomial vector.  A second triangular solve converts that
    vector to the Selberg Jack parameter ``alpha=1/g``.
    """

    variable_count = int(variable_count)
    by_degree: dict[int, dict[tuple[int, ...], sp.Expr]] = {}
    for partition, coefficient_value in coefficients.items():
        partition = tuple(map(int, partition))
        if len(partition) > variable_count:
            raise ValueError((partition, variable_count))
        by_degree.setdefault(sum(partition), {})[partition] = sp.sympify(
            coefficient_value
        )

    answer = sp.Integer(0)
    for degree, degree_coefficients in by_degree.items():
        parts, schur_transition = _jack_transition(
            variable_count, degree, sp.Integer(1)
        )
        _, selberg_transition = _jack_transition(
            variable_count, degree, sp.cancel(sp.Integer(1) / g)
        )
        monomial_vector = sp.zeros(len(parts), 1)
        for partition, coefficient_value in degree_coefficients.items():
            column = parts.index(partition)
            monomial_vector += coefficient_value * schur_transition[:, column]
        jack_coefficients = selberg_transition.inv() * monomial_vector
        for coefficient_value, partition in zip(jack_coefficients, parts):
            if coefficient_value:
                answer += coefficient_value * kadell_normalized_jack(
                    partition, variable_count, A, B, g
                )
    return sp.factor(sp.cancel(answer))


def audit() -> None:
    A, B, g = sp.symbols("A B g", nonzero=True)
    xs = variables(3)
    mean_sum = normalized_selberg_average(sum(xs), xs, A, B, g)
    expected_sum = 3 * (A + 1 + 2 * g) / (A + B + 2 + 4 * g)
    assert sp.factor(sp.cancel(mean_sum - expected_sum)) == 0

    alpha = sp.Integer(1) / g
    for partition in ((2,), (1, 1), (3,), (2, 1), (1, 1, 1)):
        jack = jack_polynomial(partition, alpha, 3)
        direct = normalized_selberg_average(jack, xs, A, B, g)
        expected = kadell_normalized_jack(partition, 3, A, B, g)
        if sp.factor(sp.cancel(direct - expected)) != 0:
            raise AssertionError((partition, sp.factor(direct - expected)))
    print("Jack/Kadell Selberg backend: exact degree<=3 checks passed")


if __name__ == "__main__":
    audit()
