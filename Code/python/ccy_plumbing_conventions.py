#!/usr/bin/env python3
"""Cho-Collier-Yin plumbing-frame sewing conventions.

CCY build higher-genus Virasoro blocks by plumbing holed spheres with
``SL(2,C)`` maps and contracting descendant three-point functions with inverse
Gram matrices.  For an internal primary of weight ``h`` and descendant level
``N``, the sewing operator contributes ``q^(h+N)``.  Their block convention
keeps the descendant power ``q^N`` inside the Virasoro block and separates the
primary factor ``q^h`` as an overall prefactor.

This module contains only that bookkeeping.  It deliberately does not include
cylinder Casimir factors or any conformal-frame anomaly.
"""

from __future__ import annotations

import cmath
from collections.abc import Iterable


def ccy_primary_propagator(q_values: Iterable[complex], weights: Iterable[complex]) -> complex:
    """Return the CCY separated primary sewing factor ``prod_e q_e^h_e``."""
    q_tuple = tuple(q_values)
    weight_tuple = tuple(weights)
    if len(q_tuple) != len(weight_tuple):
        raise ValueError("q_values and weights must have the same length")
    propagator = 1.0 + 0.0j
    for q_value, weight in zip(q_tuple, weight_tuple):
        propagator *= complex(q_value) ** complex(weight)
    return propagator


def ccy_raw_sewing_propagator(
    q_values: Iterable[complex],
    weights: Iterable[complex],
    *,
    diagnostic_shift: float = 0.0,
    log_q_values: Iterable[complex] | None = None,
) -> complex:
    """Return the raw CCY primary propagator, optionally with an explicit diagnostic shift.

    ``diagnostic_shift`` multiplies the CCY factor by ``prod_e q_e^(-shift)``.
    It is not part of the CCY plumbing-frame definition; it exists only for
    controlled normalization diagnostics.
    """
    q_tuple = tuple(complex(q_value) for q_value in q_values)
    weight_tuple = tuple(complex(weight) for weight in weights)
    if len(q_tuple) != len(weight_tuple):
        raise ValueError("q_values and weights must have the same length")
    if log_q_values is not None:
        log_q_tuple = tuple(complex(value) for value in log_q_values)
        if len(log_q_tuple) != len(weight_tuple):
            raise ValueError("log_q_values and weights must have the same length")
        exponent = sum(
            (weight - float(diagnostic_shift)) * log_q
            for weight, log_q in zip(weight_tuple, log_q_tuple)
        )
        return cmath.exp(exponent)
    propagator = ccy_primary_propagator(q_tuple, weight_tuple)
    if diagnostic_shift == 0.0:
        return propagator
    shift_factor = 1.0 + 0.0j
    for q_value in q_tuple:
        shift_factor *= q_value ** (-float(diagnostic_shift))
    return propagator * shift_factor


def liouville_threshold_weight(b: float) -> float:
    """Return the lower edge ``Q^2/4`` of the continuous Liouville weights."""
    q_background = float(b) + 1.0 / float(b)
    return 0.25 * q_background * q_background


def liouville_threshold_modulus_factor(q_values: Iterable[complex], *, b: float) -> float:
    """Return ``|prod_e q_e^(Q^2/4)|^2`` for the diagonal raw CCY integrand."""
    exponent = 2.0 * liouville_threshold_weight(b)
    factor = 1.0
    for q_value in q_values:
        factor *= abs(complex(q_value)) ** exponent
    return factor
