#!/usr/bin/env python3
"""Small trivariate series algebra with twice-level exponents.

An exponent ``(e1,e2,e3)`` represents
``q1**(e1/2) q2**(e2/2) q3**(e3/2)``.  Keeping integer exponents is useful
for NS blocks because it makes the half-integer grading exact.
"""

from __future__ import annotations

from collections.abc import Mapping


Exponent = tuple[int, int, int]
Series = dict[Exponent, complex]


ZERO: Exponent = (0, 0, 0)


def total_degree(exponent: Exponent) -> int:
    return sum(exponent)


def clean(series: Mapping[Exponent, complex], tolerance: float = 0.0) -> Series:
    return {
        exponent: complex(coefficient)
        for exponent, coefficient in series.items()
        if abs(coefficient) > tolerance
    }


def add(*series: Mapping[Exponent, complex]) -> Series:
    result: Series = {}
    for current in series:
        for exponent, coefficient in current.items():
            result[exponent] = result.get(exponent, 0.0j) + coefficient
    return clean(result)


def scale(series: Mapping[Exponent, complex], coefficient: complex) -> Series:
    return clean({exponent: coefficient * value for exponent, value in series.items()})


def shift(
    series: Mapping[Exponent, complex],
    exponent_shift: Exponent,
    max_total_twice_level: int,
) -> Series:
    result: Series = {}
    for exponent, coefficient in series.items():
        shifted = tuple(exponent[index] + exponent_shift[index] for index in range(3))
        if total_degree(shifted) <= max_total_twice_level:
            result[shifted] = result.get(shifted, 0.0j) + coefficient
    return clean(result)


def multiply(
    left: Mapping[Exponent, complex],
    right: Mapping[Exponent, complex],
    max_total_twice_level: int,
) -> Series:
    result: Series = {}
    for exponent_left, coefficient_left in left.items():
        for exponent_right, coefficient_right in right.items():
            exponent = tuple(
                exponent_left[index] + exponent_right[index] for index in range(3)
            )
            if total_degree(exponent) > max_total_twice_level:
                continue
            result[exponent] = (
                result.get(exponent, 0.0j)
                + coefficient_left * coefficient_right
            )
    return clean(result)


def theta_cross_exponent(left: Exponent, right: Exponent) -> int:
    """Polarization of the theta quadratic parity sign.

    If ``p_i=left_i+right_i`` modulo two, then

    ``sum_{i<j} p_i p_j``

    is the sum of the two separate quadratic signs and this cross exponent.
    """

    left_parity = tuple(value % 2 for value in left)
    right_parity = tuple(value % 2 for value in right)
    return sum(
        left_parity[first] * right_parity[second]
        + right_parity[first] * left_parity[second]
        for first in range(3)
        for second in range(first + 1, 3)
    ) % 2


def theta_multiply(
    left: Mapping[Exponent, complex],
    right: Mapping[Exponent, complex],
    max_total_twice_level: int,
) -> Series:
    """Multiply with the polarized theta-graph Koszul sign."""

    result: Series = {}
    for exponent_left, coefficient_left in left.items():
        for exponent_right, coefficient_right in right.items():
            exponent = tuple(
                exponent_left[index] + exponent_right[index] for index in range(3)
            )
            if total_degree(exponent) > max_total_twice_level:
                continue
            sign = (-1) ** theta_cross_exponent(exponent_left, exponent_right)
            result[exponent] = (
                result.get(exponent, 0.0j)
                + sign * coefficient_left * coefficient_right
            )
    return clean(result)


def inverse(series: Mapping[Exponent, complex], max_total_twice_level: int) -> Series:
    """Return the truncated multiplicative inverse of a series."""

    constant = complex(series.get(ZERO, 0.0j))
    if constant == 0:
        raise ZeroDivisionError("a formal series is invertible only with nonzero constant term")
    result: Series = {ZERO: 1.0 / constant}
    for total in range(1, max_total_twice_level + 1):
        for first in range(total + 1):
            for second in range(total - first + 1):
                exponent = (first, second, total - first - second)
                convolution = 0.0j
                for source_exponent, source_coefficient in series.items():
                    if source_exponent == ZERO:
                        continue
                    remainder = tuple(
                        exponent[index] - source_exponent[index] for index in range(3)
                    )
                    if min(remainder) < 0:
                        continue
                    convolution += source_coefficient * result.get(remainder, 0.0j)
                result[exponent] = -convolution / constant
    return clean(result)


def theta_inverse(
    series: Mapping[Exponent, complex], max_total_twice_level: int
) -> Series:
    """Inverse for :func:`theta_multiply`."""

    constant = complex(series.get(ZERO, 0.0j))
    if constant == 0:
        raise ZeroDivisionError("a formal series is invertible only with nonzero constant term")
    result: Series = {ZERO: 1.0 / constant}
    for total in range(1, max_total_twice_level + 1):
        for first in range(total + 1):
            for second in range(total - first + 1):
                exponent = (first, second, total - first - second)
                convolution = 0.0j
                for source_exponent, source_coefficient in series.items():
                    if source_exponent == ZERO:
                        continue
                    remainder = tuple(
                        exponent[index] - source_exponent[index] for index in range(3)
                    )
                    if min(remainder) < 0:
                        continue
                    sign = (-1) ** theta_cross_exponent(source_exponent, remainder)
                    convolution += (
                        sign * source_coefficient * result.get(remainder, 0.0j)
                    )
                result[exponent] = -convolution / constant
    return clean(result)


def evaluate(series: Mapping[Exponent, complex], q_values: tuple[complex, complex, complex]) -> complex:
    result = 0.0j
    square_roots = tuple(value**0.5 for value in q_values)
    for exponent, coefficient in series.items():
        term = coefficient
        for edge in range(3):
            term *= square_roots[edge] ** exponent[edge]
        result += term
    return result
