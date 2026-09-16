#!/usr/bin/env python3
"""Legacy candidate NSRR sewing in the Human/HJS chiral basis.

This module is NOT a verified fixed-spin interacting sewing adapter. The
free auxiliary-fermion identity below does not establish the Ramond
vertex/BPZ conversion of an interacting nonchiral amplitude. Physical
comparisons must use the certificate boundary in spin_structure.py.

BRY's pair ``(C_even, C_odd)`` gives the ordered HJS coefficient pair

    (c_+, c_-) = (C_even/2, C_odd/2).

The historical candidate implemented here ASSUMED the following reduction
of the physical small Ramond representation, for k = eta_left eta_right:

                  1
        K(k) =     - [[1, -i k], [i k, 1]].
                  4

Equivalently, its contribution of a fixed HJS-sign pair is

    (1/4) |F_0 + i eta_left eta_right F_1|^2.

This candidate cancels the NS half-level term that is nonzero in the
independent radial-reflection state sum. Its identity limit does not detect
that error. The Human Note's bilinear prescription instead requires an
independently specified antiholomorphic block and physical Ramond dual;
neither this matrix nor the reflection replacement establishes that map.
The physical descendant BPZ dual and literal Human matrix are now derived
in nsrr_bilinear_sewing.py; use that module for new bilinear contractions.
See NSRR_BILINEAR_PAIRING_2026-09-15.md. Public names
are retained only so historical datasets can be reproduced.

The auxiliary free-Majorana test for the marked source spin [11|00] uses
the independently bosonization-checked combination

    (F_(+,+,+) + F_(+,-,+))/sqrt(2)

in geometry edge order (R at zero, R at one, NS at infinity).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


Channel = tuple[int, int, int]
Lift = tuple[int, int, int]

CHANNELS: tuple[Channel, ...] = tuple(
    (form_parity, eta_left, eta_right)
    for form_parity in (0, 1)
    for eta_left in (1, -1)
    for eta_right in (1, -1)
)

# Geometry order is (R_zero, R_one, NS_infinity).
SOURCE_MARKED_CHARACTERISTIC = ((1, 1), (0, 0))
SOURCE_FIXED_SPIN_LIFTS: tuple[Lift, Lift] = ((1, 1, 1), (1, -1, 1))
PRODUCT_SPACE_KERNEL_NORMALIZATION = 0.25
PHYSICAL_SEWING_NORMALIZATION = PRODUCT_SPACE_KERNEL_NORMALIZATION


def bry_to_hjs_coefficients(
    bry_constants: Sequence[complex],
) -> dict[int, complex]:
    """Map BRY ``(C_even,C_odd)`` to Suchanek/HJS ``c_eta``."""

    if len(bry_constants) != 2:
        raise ValueError("exactly two BRY constants are required")
    values = tuple(complex(value) for value in bry_constants)
    if not all(math.isfinite(value.real) and math.isfinite(value.imag) for value in values):
        raise ValueError("BRY constants must be finite")
    return {1: values[0] / 2.0, -1: values[1] / 2.0}


def physical_form_matrix(eta_left: int, eta_right: int) -> np.ndarray:
    """Return the historical candidate, not a verified physical kernel."""

    if eta_left not in (1, -1) or eta_right not in (1, -1):
        raise ValueError("HJS signs must be +/-1")
    k = eta_left * eta_right
    return PHYSICAL_SEWING_NORMALIZATION * np.asarray(
        [[1.0, -1.0j * k], [1.0j * k, 1.0]], dtype=np.complex128
    )


def physical_form_bilinear(
    even_block: complex,
    odd_block: complex,
    eta_left: int,
    eta_right: int,
) -> float:
    """Evaluate ``(F0,F1) M (bar(F0),bar(F1))^T`` stably."""

    if eta_left not in (1, -1) or eta_right not in (1, -1):
        raise ValueError("HJS signs must be +/-1")
    physical_block = complex(even_block) + 1.0j * eta_left * eta_right * complex(odd_block)
    return float(PHYSICAL_SEWING_NORMALIZATION * abs(physical_block) ** 2)


def project_source_fixed_spin(
    amplitudes_by_lift: Mapping[Lift, Mapping[Channel, complex]],
) -> dict[Channel, complex]:
    """Legacy two-lift projection; its interacting spin interpretation is unverified."""

    missing_lifts = set(SOURCE_FIXED_SPIN_LIFTS) - set(amplitudes_by_lift)
    if missing_lifts:
        raise ValueError(f"missing fixed-spin plumbing lifts: {sorted(missing_lifts)}")
    for lift in SOURCE_FIXED_SPIN_LIFTS:
        if set(amplitudes_by_lift[lift]) != set(CHANNELS):
            raise ValueError(f"lift {lift} does not contain all eight NSRR channels")
    normalization = math.sqrt(2.0)
    return {
        channel: sum(
            complex(amplitudes_by_lift[lift][channel])
            for lift in SOURCE_FIXED_SPIN_LIFTS
        )
        / normalization
        for channel in CHANNELS
    }


def contract_physical_blocks(
    blocks: Mapping[Channel, complex],
    bry_constants: Sequence[complex],
    *,
    reality_tolerance: float = 1.0e-10,
) -> dict:
    """Evaluate the legacy candidate parity form, without a spin certificate.

    ``blocks`` must already contain the full chiral propagation amplitude,
    including its primary plumbing power.  The returned terms do not include
    the continuum quadrature measure.
    """

    if set(blocks) != set(CHANNELS):
        raise ValueError("all eight (f,eta_left,eta_right) channels are required")
    coefficients = bry_to_hjs_coefficients(bry_constants)
    terms: dict[tuple[int, int], float] = {}
    diagonal_terms: dict[tuple[int, int], float] = {}
    interference_terms: dict[tuple[int, int], float] = {}
    maximum_coefficient_imaginary_part = 0.0
    for eta_left in (1, -1):
        for eta_right in (1, -1):
            key = (eta_left, eta_right)
            coefficient = coefficients[eta_left] * coefficients[eta_right]
            maximum_coefficient_imaginary_part = max(
                maximum_coefficient_imaginary_part, abs(coefficient.imag)
            )
            scale = max(1.0, abs(coefficient))
            if abs(coefficient.imag) > reality_tolerance * scale:
                raise ArithmeticError(
                    "the real-momentum BRY coefficient product is unexpectedly complex"
                )
            f0 = complex(blocks[0, eta_left, eta_right])
            f1 = complex(blocks[1, eta_left, eta_right])
            diagonal = PHYSICAL_SEWING_NORMALIZATION * (
                abs(f0) ** 2 + abs(f1) ** 2
            )
            physical = physical_form_bilinear(f0, f1, eta_left, eta_right)
            weight = float(coefficient.real)
            diagonal_terms[key] = weight * diagonal
            terms[key] = weight * physical
            interference_terms[key] = terms[key] - diagonal_terms[key]
    return {
        "terms": terms,
        "diagonal_terms": diagonal_terms,
        "interference_terms": interference_terms,
        "diagonal": math.fsum(diagonal_terms.values()),
        "interference": math.fsum(interference_terms.values()),
        "total": math.fsum(terms.values()),
        "maximum_coefficient_imaginary_part": maximum_coefficient_imaginary_part,
    }
