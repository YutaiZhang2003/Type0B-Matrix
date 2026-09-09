#!/usr/bin/env python3
"""Physical fixed-spin NSRR sewing in the Human/HJS chiral basis.

There are two convention changes which must be made before the saved NSRR
chiral blocks can be used in a nonchiral partition function.

First, BRY's pair ``(C_even, C_odd)`` is not Suchanek's pair of chiral
three-form coefficients.  Use the corrected conventions of
arXiv:1012.2974, equations (28)--(31) and (36)--(37); the 2008 paper's
R--NS block recursion used a conjugation assumption corrected there.  The
three-form pair is

    (c_+, c_-) = (C_even/2, C_odd/2).

Second, the small Ramond representation does not give a diagonal norm in
the chiral form-parity label.  Suchanek's physical ``R+`` vertex contains
``e tensor e - i o tensor o`` while ``R-`` contains the two crossed terms,
with a relative sign for the minus three-form.  Sewing the normalized
physical Ramond subspace gives, for k = eta_left eta_right,

                  1
        K(k) =     - [[1, -i k], [i k, 1]].
                  4

Equivalently, the physical contribution of a fixed HJS-sign pair is

    (1/4) |F_0 + i eta_left eta_right F_1|^2.

The matrix is positive and rank one.  The Human Note's block is quadratic,
but that fact does not multiply the restricted nonchiral completeness tensor
by four.  If the two Ramond edges use the full two-family restricted
completeness tensor, the identity-NS degeneration contracts to ``2``; the
unscaled modulus gives ``8`` whereas this kernel gives ``2``.  This is an
internal consistency check of that explicitly chosen sewing tensor, not an
a priori trace formula for a genus-two partition function and not an
independent way to normalize the tensor.  A GSO projector, parity defect, or
global spin-sum weight must be derived separately.  This is not the diagonal
``|F_0|^2+|F_1|^2`` ansatz used by the earlier diagnostic run.

The marked source spin [11|00] is selected from the auxiliary plumbing
lifts by the independently bosonization-checked combination

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
    """Return the normalized physical two-form-parity sewing kernel."""

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
    """Project full chiral amplitudes onto the marked source spin [11|00]."""

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
    """Contract one projected fixed-spin node with the physical parity form.

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
