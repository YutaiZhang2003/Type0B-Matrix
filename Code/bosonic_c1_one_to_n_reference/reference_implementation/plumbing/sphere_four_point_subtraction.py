#!/usr/bin/env python3
"""Executable BRY sphere-four power-subtraction primitives.

This module isolates the baby step in BRY equations (3.9)--(3.15). It does
not assume a functional form for the amplitude. At fixed complex energies
and fixed Liouville momentum it expands the regular chiral OPE factor,
retains spin-zero levels below the power-divergence threshold, and subtracts
those monomials from the moduli integrand.

The global BRY subtraction and the equivalent collar finite part are both
provided so their equality can be tested directly.
"""

from __future__ import annotations

import cmath
import math
from typing import Mapping, Sequence


def bry_divergent_levels(
    channel_energy: complex,
    momentum: float,
    maximum_level: int,
) -> tuple[int, ...]:
    """Return levels n < Re(kappa^2)/4-P^2 from BRY equation (3.12)."""

    momentum = float(momentum)
    maximum_level = int(maximum_level)
    if momentum < 0.0 or maximum_level < 0:
        raise ValueError("momentum and maximum_level must be non-negative")
    threshold = 0.25 * (complex(channel_energy) ** 2).real - momentum**2
    return tuple(level for level in range(maximum_level + 1) if level < threshold)


def bry_radial_power(channel_energy: complex, momentum: float, level: int) -> complex:
    """Return x=2(P^2+n-kappa^2/4) in BRY equation (3.9)."""

    momentum = float(momentum)
    level = int(level)
    if momentum < 0.0 or level < 0:
        raise ValueError("momentum and level must be non-negative")
    return complex(2.0 * (momentum**2 + level) - 0.5 * complex(channel_energy) ** 2)


def bry_analytic_disk_integral(radial_power: complex) -> complex:
    """Return the meromorphic continuation of the unit-disk radial integral."""

    radial_power = complex(radial_power)
    if radial_power == 0.0:
        raise ZeroDivisionError("the logarithmic threshold requires an i-epsilon")
    return complex(2.0 * math.pi / radial_power)


def bry_collar_finite_part(
    radial_power: complex,
    collar_radius: float,
) -> complex:
    """Evaluate the same finite part using a radius-rho collar."""

    radial_power = complex(radial_power)
    collar_radius = float(collar_radius)
    if radial_power == 0.0:
        raise ZeroDivisionError("the logarithmic threshold requires an i-epsilon")
    if not math.isfinite(collar_radius) or not 0.0 < collar_radius < 1.0:
        raise ValueError("collar_radius must lie strictly between zero and one")
    rho_power = cmath.exp(radial_power * math.log(collar_radius))
    outer = 2.0 * math.pi * (1.0 - rho_power) / radial_power
    inner_analytic = 2.0 * math.pi * rho_power / radial_power
    return complex(outer + inner_analytic)


def generalized_binomial_series(exponent: complex, order: int) -> tuple[complex, ...]:
    """Return coefficients of (1-z)^exponent through the requested order."""

    exponent = complex(exponent)
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    coefficients = [1.0 + 0.0j]
    for level in range(1, order + 1):
        coefficients.append(
            -coefficients[-1] * (exponent - level + 1.0) / level
        )
    return tuple(coefficients)


def convolve_truncated(
    left: Sequence[complex],
    right: Sequence[complex],
    order: int,
) -> tuple[complex, ...]:
    """Multiply two univariate series through the requested order."""

    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    result = []
    for level in range(order + 1):
        result.append(
            sum(
                complex(left[index]) * complex(right[level - index])
                for index in range(level + 1)
                if index < len(left) and level - index < len(right)
            )
        )
    return tuple(result)


def bry_regular_chiral_coefficients(
    block_coefficients: Sequence[complex],
    crossed_timelike_exponent: complex,
    order: int,
) -> tuple[complex, ...]:
    """Return the regular chiral coefficients entering the BRY projector.

    Near z=0 the regular chiral factor is
    (1-z)^(omega_in*omega_2/2) F_desc(z). The non-chiral spin-zero
    coefficient at level n is a_n^2. Analytic continuation uses the same
    coefficients in both chiral blocks, not their complex conjugates.
    """

    timelike = generalized_binomial_series(crossed_timelike_exponent, order)
    return convolve_truncated(block_coefficients, timelike, order)


def evaluate_series(coefficients: Sequence[complex], value: complex) -> complex:
    """Evaluate a series by Horner's rule."""

    total = 0.0 + 0.0j
    for coefficient in reversed(tuple(coefficients)):
        total = total * complex(value) + complex(coefficient)
    return complex(total)


def bry_s_channel_projector(
    z: complex,
    *,
    channel_energy: complex,
    momentum: float,
    regular_chiral_coefficients: Sequence[complex],
) -> complex:
    """Evaluate the fixed-momentum BRY counterterm in equation (3.13)."""

    z = complex(z)
    if z == 0.0:
        raise ZeroDivisionError("the OPE projector is singular at z=0")
    coefficients = tuple(complex(value) for value in regular_chiral_coefficients)
    levels = set(
        bry_divergent_levels(
            channel_energy,
            momentum,
            len(coefficients) - 1,
        )
    )
    base_power = -2.0 + bry_radial_power(channel_energy, momentum, 0)
    logarithmic_radius = math.log(abs(z))
    return complex(
        sum(
            coefficient**2
            * cmath.exp((base_power + 2.0 * level) * logarithmic_radius)
            for level, coefficient in enumerate(coefficients)
            if level in levels
        )
    )


def bry_subtracted_fixed_momentum_density(
    z: complex,
    *,
    channel_energy: complex,
    momentum: float,
    regular_chiral_coefficients: Sequence[complex],
) -> complex:
    """Return the truncated OPE density minus its BRY power projector."""

    z = complex(z)
    if z == 0.0:
        raise ZeroDivisionError("the OPE density is singular at z=0")
    coefficients = tuple(complex(value) for value in regular_chiral_coefficients)
    regular_holomorphic = evaluate_series(coefficients, z)
    regular_antiholomorphic = evaluate_series(coefficients, z.conjugate())
    base_power = -2.0 + bry_radial_power(channel_energy, momentum, 0)
    original = cmath.exp(base_power * math.log(abs(z))) * (
        regular_holomorphic * regular_antiholomorphic
    )
    return complex(
        original
        - bry_s_channel_projector(
            z,
            channel_energy=channel_energy,
            momentum=momentum,
            regular_chiral_coefficients=coefficients,
        )
    )


def equal_one_to_three_frequencies(
    outgoing_energy: float,
    epsilon: float,
) -> tuple[complex, complex, complex, complex]:
    """Return BRY-compatible equal 1-to-3 complex frequencies."""

    outgoing_energy = float(outgoing_energy)
    epsilon = float(epsilon)
    if outgoing_energy <= 0.0 or epsilon <= 0.0:
        raise ValueError("outgoing_energy and epsilon must be positive")
    outgoing = complex(outgoing_energy, epsilon / 3.0)
    incoming = 3.0 * outgoing
    return (incoming, outgoing, outgoing, outgoing)


def channel_thresholds(
    channel_energy: complex,
    maximum_level: int,
) -> Mapping[int, float]:
    """Return every positive BRY momentum endpoint for one channel."""

    threshold = 0.25 * (complex(channel_energy) ** 2).real
    return {
        level: math.sqrt(threshold - level)
        for level in range(int(maximum_level) + 1)
        if threshold - level > 0.0
    }
