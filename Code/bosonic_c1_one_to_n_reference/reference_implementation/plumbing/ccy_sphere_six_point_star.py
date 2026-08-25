#!/usr/bin/env python3
"""Direct and c-recursive sphere six-point star-channel Virasoro block.

The four trivalent vertices consist of a central pair of pants joined to
three external cherries.  External weights are paired as ``(d1,d2)``,
``(d3,d4)``, and ``(d5,d6)``.  The internal weights ``(h1,h2,h3)`` occupy the
zero, one, and infinity slots of the central vertex, respectively.

Only the descendant series in the three plumbing variables is returned.
The primary sewing powers and four CFT structure constants are excluded.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Mapping, Sequence

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
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
        raise ValueError("external_weights must contain six values")
    if len(internal_weights) != 3:
        raise ValueError("internal_weights must contain three values")
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


def sphere_six_point_star_global_coefficient(
    level1: int,
    level2: int,
    level3: int,
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> complex:
    """Return one global star-channel coefficient."""

    level1, level2, level3 = _validate_orders(level1, level2, level3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    h1, h2, h3 = internal
    outer1 = _rising(h1 + d2 - d1, level1)
    outer2 = _rising(h2 + d4 - d3, level2)
    outer3 = _rising(h3 + d5 - d6, level3)
    central = rho_primary_descendants(
        (1,) * level3,
        (1,) * level2,
        (1,) * level1,
        h3,
        h2,
        h1,
        0.0,
    )
    denominator = math.prod(
        (
            _factorial_rising_norm(h1, level1),
            _factorial_rising_norm(h2, level2),
            _factorial_rising_norm(h3, level3),
        )
    )
    if denominator == 0.0:
        raise ZeroDivisionError("global star block has a singular internal weight")
    return complex(outer1 * outer2 * outer3 * central / denominator)


def sphere_six_point_star_direct_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    order3: int | None = None,
    max_total_order: int | None = None,
) -> CoefficientTable:
    """Evaluate the defining three-edge star descendant contraction."""

    order1, order2, order3 = _validate_orders(order1, order2, order3)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5, d6 = external
    h1, h2, h3 = internal
    c_value = complex(central_charge)
    grams = tuple(
        {
            level: inverse_verma_gram_matrix(level, weight, c_value)[:2]
            for level in range(order + 1)
        }
        for weight, order in zip(internal, (order1, order2, order3))
    )
    out: CoefficientTable = {}
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
                        outer1 = rho_primary_descendants(b1, (), (), h1, d2, d1, c_value)
                        if outer1 == 0.0:
                            continue
                        for a2_index, a2 in enumerate(basis2):
                            for b2_index, b2 in enumerate(basis2):
                                metric2 = inverse2[a2_index, b2_index]
                                if metric2 == 0.0:
                                    continue
                                outer2 = rho_primary_descendants(
                                    b2, (), (), h2, d4, d3, c_value
                                )
                                if outer2 == 0.0:
                                    continue
                                for a3_index, a3 in enumerate(basis3):
                                    central = rho_primary_descendants(
                                        a3, a2, a1, h3, h2, h1, c_value
                                    )
                                    if central == 0.0:
                                        continue
                                    for b3_index, b3 in enumerate(basis3):
                                        metric3 = inverse3[a3_index, b3_index]
                                        if metric3 == 0.0:
                                            continue
                                        outer3 = rho_primary_descendants(
                                            b3, (), (), h3, d5, d6, c_value
                                        )
                                        coefficient += (
                                            metric1
                                            * metric2
                                            * metric3
                                            * outer1
                                            * outer2
                                            * outer3
                                            * central
                                        )
                out[(n1, n2, n3)] = complex(coefficient)
    return out


def sphere_six_point_star_c_coefficients(
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
    """Return star coefficients from fixed-weight c-recursion."""

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
        total = sphere_six_point_star_global_coefficient(
            n1,
            n2,
            n3,
            external_weights=external,
            internal_weights=(h1, h2, h3),
        )
        edge_data = (
            (n1, h1, (d1, d2), (h3, h2), 0),
            (n2, h2, (d3, d4), (h3, h1), 1),
            (n3, h3, (d6, d5), (h2, h1), 2),
        )
        for edge_order, edge_weight, outer_pair, central_pair, edge_index in edge_data:
            for r in range(2, edge_order + 1):
                for s in range(1, edge_order // r + 1):
                    null_level = r * s
                    pole_c = c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"six-point star c-recursion hit an edge-{edge_index + 1} pole"
                        )
                    b_pole = b_from_c_rs_h(r, s, edge_weight)
                    residue = (
                        minus_dc_dh_times_a_rs(r, s, edge_weight)
                        * fusion_polynomial_for_weights(r, s, b_pole, *outer_pair)
                        * fusion_polynomial_for_weights(r, s, b_pole, *central_pair)
                    )
                    # The zero-slot edge uses the standard CCY local
                    # coordinate.  Moving a null state to the one or infinity
                    # slot of the central pair of pants contributes the BPZ
                    # phase (-1)^(rs) in the plane plumbing frame.
                    if edge_index in (1, 2) and null_level % 2:
                        residue = -residue
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


def evaluate_sphere_six_point_star_series(
    q1: complex,
    q2: complex,
    q3: complex,
    coefficients: Mapping[tuple[int, int, int], complex],
) -> complex:
    """Evaluate a trivariate star descendant table."""

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


__all__ = [
    "evaluate_sphere_six_point_star_series",
    "sphere_six_point_star_c_coefficients",
    "sphere_six_point_star_direct_coefficients",
    "sphere_six_point_star_global_coefficient",
]
