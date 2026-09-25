#!/usr/bin/env python3
"""Constant-size theta-polarized star algebra for pointwise blocks.

Parity components are indexed by ``p0 + 2*p1 + 4*p_infinity``.  The theta
quadratic form is

    Q(p) = p0*p1 + p0*p_infinity + p1*p_infinity  (mod 2),

and its polarization is the cocycle in the human-note ``star`` product.
Because that cocycle is the coboundary of ``Q``, a quadratic sign twist turns
``star`` multiplication into ordinary XOR convolution.  An eight-point
Walsh--Hadamard transform then diagonalizes the algebra.
"""

from __future__ import annotations

from typing import Sequence


PARITY_DIMENSION = 8


def theta_quadratic_sign(index: int) -> int:
    p0 = int(index) & 1
    p1 = (int(index) >> 1) & 1
    pinfinity = (int(index) >> 2) & 1
    exponent = p0 * p1 + p0 * pinfinity + p1 * pinfinity
    return -1 if exponent % 2 else 1


THETA_TWIST = tuple(
    theta_quadratic_sign(index) for index in range(PARITY_DIMENSION)
)


def _validate(values: Sequence[complex]) -> list[complex]:
    if len(values) != PARITY_DIMENSION:
        raise ValueError("the genus-two parity algebra has eight components")
    return [complex(value) for value in values]


def fwht(values: Sequence[complex]) -> list[complex]:
    """Unnormalized length-eight Walsh--Hadamard transform."""

    transformed = _validate(values)
    stride = 1
    while stride < PARITY_DIMENSION:
        for start in range(0, PARITY_DIMENSION, 2 * stride):
            for offset in range(stride):
                left = start + offset
                right = left + stride
                a = transformed[left]
                b = transformed[right]
                transformed[left] = a + b
                transformed[right] = a - b
        stride *= 2
    return transformed


def inverse_fwht(values: Sequence[complex]) -> list[complex]:
    return [value / PARITY_DIMENSION for value in fwht(values)]


def star_spectrum(components: Sequence[complex]) -> list[complex]:
    """Map parity components to the diagonal star-product basis."""

    values = _validate(components)
    return fwht(
        [THETA_TWIST[index] * values[index] for index in range(8)]
    )


def from_star_spectrum(spectrum: Sequence[complex]) -> list[complex]:
    """Map the diagonal basis back to parity components."""

    untwisted = inverse_fwht(spectrum)
    return [THETA_TWIST[index] * untwisted[index] for index in range(8)]


def star_multiply(
    left: Sequence[complex], right: Sequence[complex]
) -> list[complex]:
    left_spectrum = star_spectrum(left)
    right_spectrum = star_spectrum(right)
    return from_star_spectrum(
        [a * b for a, b in zip(left_spectrum, right_spectrum)]
    )


def star_divide(
    numerator: Sequence[complex],
    denominator: Sequence[complex],
    *,
    zero_tolerance: float = 1.0e-14,
) -> list[complex]:
    """Pointwise star quotient in three parity bits."""

    numerator_spectrum = star_spectrum(numerator)
    denominator_spectrum = star_spectrum(denominator)
    scale = max(1.0, *(abs(value) for value in denominator_spectrum))
    if any(abs(value) <= zero_tolerance * scale for value in denominator_spectrum):
        raise ZeroDivisionError("the auxiliary block is singular in the star algebra")
    return from_star_spectrum(
        [
            numerator_value / denominator_value
            for numerator_value, denominator_value in zip(
                numerator_spectrum, denominator_spectrum
            )
        ]
    )


def direct_star_multiply(
    left: Sequence[complex], right: Sequence[complex]
) -> list[complex]:
    """Quadratic reference implementation used only in tests."""

    a = _validate(left)
    b = _validate(right)
    result = [0.0j] * 8
    for left_index, left_value in enumerate(a):
        lp = tuple((left_index >> bit) & 1 for bit in range(3))
        for right_index, right_value in enumerate(b):
            rp = tuple((right_index >> bit) & 1 for bit in range(3))
            exponent = sum(
                lp[i] * rp[j] + rp[i] * lp[j]
                for i in range(3)
                for j in range(i + 1, 3)
            )
            result[left_index ^ right_index] += (
                (-1) ** exponent * left_value * right_value
            )
    return result
