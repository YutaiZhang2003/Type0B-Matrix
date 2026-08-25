#!/usr/bin/env python3
"""CCY recursions for the sphere six-point linear Virasoro block.

The external primaries ``(d1,...,d6)`` are inserted at

``(0, z1, z2, z3, 1, infinity)``,

and the three internal weights ``(h1,h2,h3)`` propagate in the comb channel

``((d1 d2)->h1, (h1 d3)->h2, (h2 d4)->h3, (h3 d5 d6))``.

The plumbing variables are ``q1=z1/z2``, ``q2=z2/z3``, and ``q3=z3``.
All coefficient functions return only the descendant series.  Liouville
structure constants and the primary-coordinate powers are excluded.

Three independent constructions are supplied:

* the defining Verma-module descendant contraction;
* fixed-internal-weight central-charge recursion;
* the CCY common-weight h-recursion, regulated at c=25 by b=exp(eta).

The conventions are the direct six-point extension of
``ccy_sphere_five_point.py``.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from virasoro_blocks import (
        central_charge_to_b,
        degenerate_weight,
        fusion_polynomial_for_weights as fixed_c_fusion_polynomial_for_weights,
        zamolodchikov_a_rs,
    )
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from plumbing.virasoro_blocks import (
        central_charge_to_b,
        degenerate_weight,
        fusion_polynomial_for_weights as fixed_c_fusion_polynomial_for_weights,
        zamolodchikov_a_rs,
    )
    from plumbing.virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )


CoefficientTable = dict[tuple[int, int, int], complex]


def _validate_orders(
    order1: int,
    order2: int | None,
    order3: int | None,
) -> tuple[int, int, int]:
    order1 = int(order1)
    order2 = order1 if order2 is None else int(order2)
    order3 = order1 if order3 is None else int(order3)
    if min(order1, order2, order3) < 0:
        raise ValueError("block orders must be non-negative")
    return order1, order2, order3


def _weights(
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> tuple[tuple[complex, ...], tuple[complex, complex, complex]]:
    if len(external_weights) != 6:
        raise ValueError("external_weights must contain (d1,d2,d3,d4,d5,d6)")
    if len(internal_weights) != 3:
        raise ValueError("internal_weights must contain (h1,h2,h3)")
    return (
        tuple(complex(value) for value in external_weights),
        tuple(complex(value) for value in internal_weights),
    )


def _rising(value: complex, order: int) -> complex:
    out = 1.0 + 0.0j
    for offset in range(int(order)):
        out *= complex(value) + offset
    return out


def _factorial_rising_norm(weight: complex, level: int) -> complex:
    return math.factorial(int(level)) * _rising(2.0 * complex(weight), int(level))


def sphere_six_point_global_coefficient(
    level1: int,
    level2: int,
    level3: int,
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> complex:
    """Return one coefficient of the global six-point comb block."""

    level1, level2, level3 = _validate_orders(level1, level2, level3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    h1, h2, h3 = internal
    left = _rising(h1 + d2 - d1, level1)
    middle12 = rho_primary_descendants(
        (1,) * level2,
        (),
        (1,) * level1,
        h2,
        d3,
        h1,
        0.0,
    )
    middle23 = rho_primary_descendants(
        (1,) * level3,
        (),
        (1,) * level2,
        h3,
        d4,
        h2,
        0.0,
    )
    right = _rising(h3 + d5 - d6, level3)
    denominator = math.prod(
        (
            _factorial_rising_norm(h1, level1),
            _factorial_rising_norm(h2, level2),
            _factorial_rising_norm(h3, level3),
        )
    )
    if denominator == 0.0:
        raise ZeroDivisionError("global block has a singular internal weight")
    return complex(left * middle12 * middle23 * right / denominator)


def sphere_six_point_global_coefficients(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
) -> CoefficientTable:
    """Return a rectangular or total-degree global coefficient table."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    if max_total_order is not None and int(max_total_order) < 0:
        raise ValueError("max_total_order must be non-negative")
    return {
        (n1, n2, n3): sphere_six_point_global_coefficient(
            n1,
            n2,
            n3,
            external_weights=external_weights,
            internal_weights=internal_weights,
        )
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        for n3 in range(order3 + 1)
        if max_total_order is None or n1 + n2 + n3 <= int(max_total_order)
    }


def sphere_six_point_direct_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
) -> CoefficientTable:
    """Evaluate the defining three-edge descendant contraction."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    h1, h2, h3 = internal
    c_value = complex(central_charge)
    out: CoefficientTable = {}

    grams = (
        {
            level: inverse_verma_gram_matrix(level, h1, c_value)[:2]
            for level in range(order1 + 1)
        },
        {
            level: inverse_verma_gram_matrix(level, h2, c_value)[:2]
            for level in range(order2 + 1)
        },
        {
            level: inverse_verma_gram_matrix(level, h3, c_value)[:2]
            for level in range(order3 + 1)
        },
    )

    for n1 in range(order1 + 1):
        for n2 in range(order2 + 1):
            for n3 in range(order3 + 1):
                if max_total_order is not None and n1 + n2 + n3 > int(max_total_order):
                    continue
                basis1, inverse1 = grams[0][n1]
                basis2, inverse2 = grams[1][n2]
                basis3, inverse3 = grams[2][n3]
                coefficient = 0.0 + 0.0j
                for a1_index, a1 in enumerate(basis1):
                    for b1_index, b1 in enumerate(basis1):
                        metric1 = inverse1[a1_index, b1_index]
                        if metric1 == 0.0:
                            continue
                        left = rho_primary_descendants(
                            b1, (), (), h1, d2, d1, c_value
                        )
                        if left == 0.0:
                            continue
                        for a2_index, a2 in enumerate(basis2):
                            for b2_index, b2 in enumerate(basis2):
                                metric2 = inverse2[a2_index, b2_index]
                                if metric2 == 0.0:
                                    continue
                                middle12 = rho_primary_descendants(
                                    b2, (), a1, h2, d3, h1, c_value
                                )
                                if middle12 == 0.0:
                                    continue
                                for a3_index, a3 in enumerate(basis3):
                                    right = rho_primary_descendants(
                                        (), (), a3, d6, d5, h3, c_value
                                    )
                                    if right == 0.0:
                                        continue
                                    for b3_index, b3 in enumerate(basis3):
                                        metric3 = inverse3[a3_index, b3_index]
                                        if metric3 == 0.0:
                                            continue
                                        middle23 = rho_primary_descendants(
                                            b3, (), a2, h3, d4, h2, c_value
                                        )
                                        coefficient += (
                                            metric1
                                            * metric2
                                            * metric3
                                            * left
                                            * middle12
                                            * middle23
                                            * right
                                        )
                out[(n1, n2, n3)] = complex(coefficient)
    return out


def sphere_six_point_c_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
    pole_tolerance: float = 1.0e-12,
) -> CoefficientTable:
    """Return coefficients from fixed-weight central-charge recursion."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    initial_h1, initial_h2, initial_h3 = internal
    c_value = complex(central_charge)
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    @lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        n3: int,
        current_c: complex,
        h1: complex,
        h2: complex,
        h3: complex,
    ) -> complex:
        total = sphere_six_point_global_coefficient(
            n1,
            n2,
            n3,
            external_weights=external,
            internal_weights=(h1, h2, h3),
        )
        edge_data = (
            (n1, h1, (d1, d2), (h2, d3), 0),
            (n2, h2, (h1, d3), (h3, d4), 1),
            (n3, h3, (h2, d4), (d6, d5), 2),
        )
        for edge_order, edge_weight, left_pair, right_pair, edge_index in edge_data:
            for r in range(2, edge_order + 1):
                for s in range(1, edge_order // r + 1):
                    null_level = r * s
                    pole_c = c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"six-point c-recursion encountered an edge-{edge_index + 1} pole collision"
                        )
                    b_pole = b_from_c_rs_h(r, s, edge_weight)
                    residue = (
                        minus_dc_dh_times_a_rs(r, s, edge_weight)
                        * fusion_polynomial_for_weights(r, s, b_pole, *left_pair)
                        * fusion_polynomial_for_weights(r, s, b_pole, *right_pair)
                    )
                    next_levels = [n1, n2, n3]
                    next_levels[edge_index] -= null_level
                    next_weights = [h1, h2, h3]
                    next_weights[edge_index] += null_level
                    total += residue / denominator * coefficient(
                        *next_levels,
                        pole_c,
                        *next_weights,
                    )
        return complex(total)

    return {
        (n1, n2, n3): coefficient(
            n1,
            n2,
            n3,
            c_value,
            initial_h1,
            initial_h2,
            initial_h3,
        )
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        for n3 in range(order3 + 1)
        if max_total_order is None or n1 + n2 + n3 <= int(max_total_order)
    }


def sphere_six_point_h_coefficients(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
    central_charge: complex | None = None,
    b: complex | None = None,
    pole_tolerance: float = 1.0e-12,
) -> CoefficientTable:
    """Return coefficients from the CCY common-weight h-recursion."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    h1_initial, h2_initial, h3_initial = internal
    if central_charge is not None and b is not None:
        raise ValueError("supply central_charge or b, not both")
    if b is None:
        if central_charge is None:
            raise ValueError("central_charge or b is required")
        b_value = central_charge_to_b(complex(central_charge))
    else:
        b_value = complex(b)
    if b_value == 0.0:
        raise ValueError("b must be nonzero")
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    a2_initial = h2_initial - h1_initial
    a3_initial = h3_initial - h1_initial
    e1_initial = d1 - h1_initial
    e6_initial = d6 - h1_initial

    @lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        n3: int,
        h1: complex,
        a2: complex,
        a3: complex,
        e1: complex,
        e6: complex,
    ) -> complex:
        total = 1.0 + 0.0j if n1 == n2 == n3 == 0 else 0.0 + 0.0j

        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                null_level = r * s
                pole_h = degenerate_weight(r, s, b_value)
                denominator = h1 - pole_h
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "six-point h-recursion encountered an edge-1 Kac pole"
                    )
                residue = (
                    zamolodchikov_a_rs(r, s, b_value)
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h + e1, d2
                    )
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h + a2, d3
                    )
                )
                total += residue / denominator * coefficient(
                    n1 - null_level,
                    n2,
                    n3,
                    pole_h + null_level,
                    a2 - null_level,
                    a3 - null_level,
                    e1 - null_level,
                    e6 - null_level,
                )

        for r in range(1, n2 + 1):
            for s in range(1, n2 // r + 1):
                null_level = r * s
                pole_h = degenerate_weight(r, s, b_value)
                denominator = h1 + a2 - pole_h
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "six-point h-recursion encountered an edge-2 Kac pole"
                    )
                residue = (
                    zamolodchikov_a_rs(r, s, b_value)
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a2, d3
                    )
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a2 + a3, d4
                    )
                )
                total += residue / denominator * coefficient(
                    n1,
                    n2 - null_level,
                    n3,
                    pole_h - a2,
                    a2 + null_level,
                    a3,
                    e1,
                    e6,
                )

        for r in range(1, n3 + 1):
            for s in range(1, n3 // r + 1):
                null_level = r * s
                pole_h = degenerate_weight(r, s, b_value)
                denominator = h1 + a3 - pole_h
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "six-point h-recursion encountered an edge-3 Kac pole"
                    )
                residue = (
                    zamolodchikov_a_rs(r, s, b_value)
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a3 + a2, d4
                    )
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a3 + e6, d5
                    )
                )
                total += residue / denominator * coefficient(
                    n1,
                    n2,
                    n3 - null_level,
                    pole_h - a3,
                    a2,
                    a3 + null_level,
                    e1,
                    e6,
                )
        return complex(total)

    return {
        (n1, n2, n3): coefficient(
            n1,
            n2,
            n3,
            h1_initial,
            a2_initial,
            a3_initial,
            e1_initial,
            e6_initial,
        )
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        for n3 in range(order3 + 1)
        if max_total_order is None or n1 + n2 + n3 <= int(max_total_order)
    }


def sphere_six_point_h_c25_limit(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
    regulator_etas: Sequence[float] = (0.16, 0.13, 0.10, 0.075, 0.055),
    polynomial_degree: int = 3,
) -> tuple[CoefficientTable, CoefficientTable]:
    """Return the regulated c=25 h-recursion and a fit-shift estimate."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    etas = tuple(float(value) for value in regulator_etas)
    if len(etas) < 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in etas
    ):
        raise ValueError("regulator_etas must contain at least three positive finite values")
    degree = int(polynomial_degree)
    if degree < 1 or degree >= len(etas):
        raise ValueError("polynomial_degree must lie between one and len(regulator_etas)-1")

    samples = [
        sphere_six_point_h_coefficients(
            external_weights=external_weights,
            internal_weights=internal_weights,
            order1=order1,
            order2=order2,
            order3=order3,
            max_total_order=max_total_order,
            b=cmath.exp(eta),
        )
        for eta in etas
    ]
    keys = tuple(samples[0])
    x_values = np.asarray([eta * eta for eta in etas], dtype=float)
    values: CoefficientTable = {}
    errors: CoefficientTable = {}
    for key in keys:
        y_values = np.asarray([sample[key] for sample in samples], dtype=complex)
        high = np.polynomial.polynomial.polyfit(x_values, y_values, degree)
        low = np.polynomial.polynomial.polyfit(x_values, y_values, degree - 1)
        values[key] = complex(high[0])
        errors[key] = complex(high[0] - low[0])
    return values, errors


def evaluate_sphere_six_point_series(
    q1: complex,
    q2: complex,
    q3: complex,
    coefficients: Mapping[tuple[int, int, int], complex],
) -> complex:
    """Evaluate a supplied trivariate descendant coefficient table."""

    q1, q2, q3 = complex(q1), complex(q2), complex(q3)
    return complex(
        sum(
            complex(value)
            * q1 ** int(levels[0])
            * q2 ** int(levels[1])
            * q3 ** int(levels[2])
            for levels, value in coefficients.items()
        )
    )


def sphere_six_point_primary_factor(
    z1: complex,
    z2: complex,
    z3: complex,
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> complex:
    """Return the primary-coordinate power in the stated comb channel."""

    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, _, _ = external
    h1, h2, h3 = internal
    return complex(
        complex(z1) ** (h1 - d1 - d2)
        * complex(z2) ** (h2 - d3 - h1)
        * complex(z3) ** (h3 - d4 - h2)
    )

