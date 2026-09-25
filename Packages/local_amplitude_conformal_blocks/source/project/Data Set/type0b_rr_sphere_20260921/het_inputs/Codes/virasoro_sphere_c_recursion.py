"""Ordinary Virasoro four-point sewing coefficients from c-recursion.

The external order is (0, z, 1, infinity).  No superconformal large-c
assumption is made here: this module is used for the two *bosonic* Virasoro
factors of F x SVir.  The residue primitives are shared with the torus
implementation; the sphere seed is the ordinary SL(2) hypergeometric block.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Sequence

from spin23_virasoro_torus_recursion import (
    _b_square_rs_from_h,
    _c_rs_from_h,
    _fusion_polynomial,
    _minus_dc_dh_times_a_rs,
)


def _rising(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for k in range(order):
        result *= value + k
    return result


def sphere_c_coefficients(
    *, c: complex, h: complex, external_weights: Sequence[complex], order: int,
    pole_tolerance: float = 1.0e-12,
) -> tuple[complex, ...]:
    """Return F_0,...,F_order, excluding z**(h-h_0-h_z).

    Generic Verma modules only. Genuine or unresolved coincident poles raise;
    there is no Gram-matrix completion or numerical pole displacement.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    weights = tuple(map(complex, external_weights))
    if len(weights) != 4:
        raise ValueError("external_weights must contain (h_0,h_z,h_1,h_inf)")
    if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
        raise ValueError("pole_tolerance must be finite and positive")
    h0, hz, h1, hinf = weights

    @lru_cache(maxsize=None)
    def coefficient(level: int, current_c: complex, current_h: complex) -> complex:
        denominator = math.factorial(level) * _rising(2 * current_h, level)
        if denominator == 0:
            raise ZeroDivisionError("singular ordinary global-block weight")
        total = (_rising(current_h + hz - h0, level)
                 * _rising(current_h + h1 - hinf, level) / denominator)
        for r in range(2, level + 1):
            for s in range(1, level // r + 1):
                null_level = r * s
                pole_c = _c_rs_from_h(r, s, current_h)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        f"ordinary sphere c-recursion pole: (r,s)=({r},{s}), "
                        f"c={current_c}, h={current_h}")
                b_pole = cmath.sqrt(_b_square_rs_from_h(r, s, current_h))
                residue = (_minus_dc_dh_times_a_rs(r, s, current_h)
                           * _fusion_polynomial(r, s, b_pole, hinf, h1)
                           * _fusion_polynomial(r, s, b_pole, h0, hz))
                total += residue / denominator * coefficient(
                    level - null_level, pole_c, current_h + null_level)
        if not (math.isfinite(total.real) and math.isfinite(total.imag)):
            raise ArithmeticError("nonfinite ordinary sphere coefficient")
        return complex(total)

    return tuple(coefficient(n, complex(c), complex(h)) for n in range(order + 1))
