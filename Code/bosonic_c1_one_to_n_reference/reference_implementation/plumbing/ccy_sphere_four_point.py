#!/usr/bin/env python3
"""CCY central-charge recursion for a sphere four-point Virasoro block.

The insertions are ordered as ``(h1,h2,h3,h4)`` at ``(0,z,1,infinity)``
and ``h_internal`` propagates in the ``(12)(34)`` channel.  The returned
series contains descendant powers only,

``F(z) = sum_n F_n z^n``.

The conventional primary propagation factor ``z**(h_internal-h1-h2)`` and
the two three-point structure constants are not included.  This is the
one-internal-edge punctured specialization of the graph-level CCY recursion.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import cmath
import math
import numpy as np

try:
    from ccy_genus2_block import (
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )

try:
    from torus_two_point_blocks import modular_lambda_series, power_composition_matrix
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.torus_two_point_blocks import (
        modular_lambda_series,
        power_composition_matrix,
    )


def _rising(value: complex, order: int) -> complex:
    product = 1.0 + 0.0j
    for offset in range(int(order)):
        product *= complex(value) + offset
    return product


def sphere_four_point_global_coefficient(
    level: int,
    *,
    external_weights: Sequence[complex],
    internal_weight: complex,
) -> complex:
    r"""Return the level-``n`` coefficient of the global four-point block.

    In the stated insertion convention the large-``c`` block is

    ``2F1(h+h2-h1, h+h3-h4; 2h; z)``.
    """

    level = _validate_order(level)
    if len(external_weights) != 4:
        raise ValueError("external_weights must contain (h1,h2,h3,h4)")
    h1, h2, h3, h4 = (complex(weight) for weight in external_weights)
    h_internal = complex(internal_weight)
    denominator = _rising(2.0 * h_internal, level)
    for integer in range(1, level + 1):
        denominator *= integer
    if abs(denominator) == 0.0:
        raise ZeroDivisionError("global block has a singular internal weight")
    return complex(
        _rising(h_internal + h2 - h1, level)
        * _rising(h_internal + h3 - h4, level)
        / denominator
    )


def sphere_four_point_residue_prefactor(
    r: int,
    s: int,
    *,
    external_weights: Sequence[complex],
    internal_weight: complex,
) -> complex:
    r"""Return the two-vertex null-vector residue in the four-point channel."""

    if len(external_weights) != 4:
        raise ValueError("external_weights must contain (h1,h2,h3,h4)")
    h1, h2, h3, h4 = (complex(weight) for weight in external_weights)
    h_internal = complex(internal_weight)
    b_pole = b_from_c_rs_h(int(r), int(s), h_internal)
    left = fusion_polynomial_for_weights(int(r), int(s), b_pole, h4, h3)
    right = fusion_polynomial_for_weights(int(r), int(s), b_pole, h1, h2)
    return complex(
        minus_dc_dh_times_a_rs(int(r), int(s), h_internal)
        * left
        * right
    )


def sphere_four_point_ccy_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weight: complex,
    order: int,
    pole_tolerance: float = 1.0e-12,
) -> tuple[complex, ...]:
    r"""Return ``F_0,...,F_order`` from the CCY ``c``-recursion."""

    order = _validate_order(order)
    if len(external_weights) != 4:
        raise ValueError("external_weights must contain (h1,h2,h3,h4)")
    weights = tuple(complex(weight) for weight in external_weights)
    c_value = complex(central_charge)
    h_value = complex(internal_weight)
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    @lru_cache(maxsize=None)
    def coefficient(level: int, current_c: complex, current_h: complex) -> complex:
        total = sphere_four_point_global_coefficient(
            level,
            external_weights=weights,
            internal_weight=current_h,
        )
        for r in range(2, level + 1):
            for s in range(1, level // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, current_h)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "sphere four-point c-recursion encountered a degenerate pole "
                        f"at level {null_level}: c={current_c!r}, h={current_h!r}"
                    )
                residue = sphere_four_point_residue_prefactor(
                    r,
                    s,
                    external_weights=weights,
                    internal_weight=current_h,
                )
                total += (
                    residue
                    / denominator
                    * coefficient(
                        level - null_level,
                        pole_c,
                        current_h + null_level,
                    )
                )
        return complex(total)

    return tuple(coefficient(level, c_value, h_value) for level in range(order + 1))


def sphere_four_point_ccy_block(
    z: complex,
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weight: complex,
    order: int,
    include_primary_power: bool = False,
    pole_tolerance: float = 1.0e-12,
) -> complex:
    """Evaluate the truncated four-point block in the plane cross-ratio."""

    z = complex(z)
    coefficients = sphere_four_point_ccy_coefficients(
        central_charge=central_charge,
        external_weights=external_weights,
        internal_weight=internal_weight,
        order=order,
        pole_tolerance=pole_tolerance,
    )
    descendant_block = sum(
        coefficient * z**level
        for level, coefficient in enumerate(coefficients)
    )
    if not include_primary_power:
        return complex(descendant_block)
    h1, h2, _, _ = (complex(weight) for weight in external_weights)
    return complex(z ** (complex(internal_weight) - h1 - h2) * descendant_block)


def sphere_four_point_coefficients_in_elliptic_nome(
    coefficients: Sequence[complex],
    output_order: int | None = None,
) -> tuple[complex, ...]:
    r"""Re-expand a plane descendant block in the elliptic nome.

    If ``F(z)=sum_n a_n z^n`` and ``z=lambda(q)``, this returns the
    coefficients of ``F(lambda(q))``.  This is deliberately the complete
    plane descendant block rather than Zamolodchikov's stripped ``H(q)``;
    the two representations are algebraically equivalent, while this form
    lets the already checked CCY recursion supply the coefficients.
    """

    plane = np.asarray(tuple(complex(value) for value in coefficients), dtype=complex)
    if plane.ndim != 1 or plane.size == 0:
        raise ValueError("coefficients must be a nonempty one-dimensional sequence")
    input_order = plane.size - 1
    if output_order is None:
        output_order = input_order
    output_order = _validate_order(output_order)
    transform = power_composition_matrix(
        modular_lambda_series(output_order),
        input_order,
        output_order,
    )
    return tuple(complex(value) for value in plane @ transform)


def _truncated_product(
    left: np.ndarray,
    right: np.ndarray,
    order: int,
) -> np.ndarray:
    return np.convolve(left, right)[: order + 1]


def _unit_series_power(series: np.ndarray, exponent: complex, order: int) -> np.ndarray:
    """Raise a power series with constant coefficient one to a complex power."""

    series = np.asarray(series, dtype=complex)[: order + 1]
    if abs(series[0] - 1.0) > 1.0e-12:
        raise ValueError("the powered series must have constant coefficient one")
    displacement = series.copy()
    displacement[0] -= 1.0
    result = np.zeros(order + 1, dtype=complex)
    result[0] = 1.0
    displacement_power = np.zeros(order + 1, dtype=complex)
    displacement_power[0] = 1.0
    binomial = 1.0 + 0.0j
    for power in range(1, order + 1):
        displacement_power = _truncated_product(
            displacement_power,
            displacement,
            order,
        )
        binomial *= (complex(exponent) - power + 1.0) / power
        result += binomial * displacement_power
    return result


def _divide_unit_series(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return the formal quotient of equally truncated unit-denominator series."""

    numerator = np.asarray(numerator, dtype=complex)
    denominator = np.asarray(denominator, dtype=complex)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("series must be one-dimensional and have the same order")
    if abs(denominator[0]) == 0.0:
        raise ZeroDivisionError("series denominator has zero constant coefficient")
    quotient = np.zeros_like(numerator)
    for level in range(len(numerator)):
        quotient[level] = (
            numerator[level]
            - sum(
                denominator[index] * quotient[level - index]
                for index in range(1, level + 1)
            )
        ) / denominator[0]
    return quotient


def sphere_four_point_elliptic_h_coefficients(
    plane_coefficients: Sequence[complex],
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weight: complex,
) -> tuple[complex, ...]:
    r"""Convert checked plane coefficients to Zamolodchikov ``H(q)``.

    The conversion factors out the universal elliptic prefactor before
    truncation.  Consequently the returned series converges throughout the
    six-channel fundamental domain, unlike the unstripped composition
    ``F(lambda(q))`` whose nearest crossed-channel singularity is much closer.
    """

    if len(external_weights) != 4:
        raise ValueError("external_weights must contain four weights")
    plane = np.asarray(tuple(complex(value) for value in plane_coefficients), dtype=complex)
    order = len(plane) - 1
    if order < 0:
        raise ValueError("plane_coefficients must be nonempty")
    h1, h2, h3, h4 = (complex(value) for value in external_weights)
    internal_weight = complex(internal_weight)
    delta = (complex(central_charge) - 1.0) / 24.0

    lambda_full = modular_lambda_series(order + 1)
    lambda_series = lambda_full[: order + 1]
    composed = np.asarray(
        sphere_four_point_coefficients_in_elliptic_nome(plane, order),
        dtype=complex,
    )
    lambda_over_16q = np.zeros(order + 1, dtype=complex)
    if order == 0:
        lambda_over_16q[0] = 1.0
    else:
        lambda_over_16q[:] = lambda_full[1 : order + 2] / 16.0
    inverse_lambda_ratio = _divide_unit_series(
        np.r_[1.0 + 0.0j, np.zeros(order, dtype=complex)],
        lambda_over_16q,
    )
    one_minus_lambda = -lambda_series
    one_minus_lambda[0] += 1.0
    theta3 = np.zeros(order + 1, dtype=complex)
    theta3[0] = 1.0
    for integer in range(1, math.isqrt(order) + 1):
        theta3[integer * integer] += 2.0

    prefactor = _unit_series_power(
        inverse_lambda_ratio,
        internal_weight - delta,
        order,
    )
    prefactor = _truncated_product(
        prefactor,
        _unit_series_power(one_minus_lambda, delta - h2 - h3, order),
        order,
    )
    prefactor = _truncated_product(
        prefactor,
        _unit_series_power(
            theta3,
            (complex(central_charge) - 1.0) / 2.0
            - 4.0 * (h1 + h2 + h3 + h4),
            order,
        ),
        order,
    )
    return tuple(complex(value) for value in _divide_unit_series(composed, prefactor))


def sphere_four_point_elliptic_descendant_block(
    z: complex,
    h_coefficients: Sequence[complex],
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weight: complex,
    nome: complex,
) -> complex:
    """Evaluate the descendant block from the stripped elliptic series."""

    if len(external_weights) != 4:
        raise ValueError("external_weights must contain four weights")
    z = complex(z)
    nome = complex(nome)
    h1, h2, h3, h4 = (complex(value) for value in external_weights)
    delta = (complex(central_charge) - 1.0) / 24.0
    theta3 = 1.0 + 0.0j
    for integer in range(1, 10000):
        term = 2.0 * nome ** (integer * integer)
        theta3 += term
        if abs(term) < 1.0e-16:
            break
    prefactor = (
        (16.0 * nome / z) ** (complex(internal_weight) - delta)
        * (1.0 - z) ** (delta - h2 - h3)
        * theta3
        ** (
            (complex(central_charge) - 1.0) / 2.0
            - 4.0 * (h1 + h2 + h3 + h4)
        )
    )
    descendant_h = sum(
        complex(value) * nome**level
        for level, value in enumerate(h_coefficients)
    )
    return complex(prefactor * descendant_h)
