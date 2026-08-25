#!/usr/bin/env python3
"""Boundary-stratum bookkeeping for the sphere five-point finite part.

``Mbar_{0,5}`` has ten boundary divisors and fifteen codimension-two corners.
A divisor is represented by the two-element side (the ``cherry``) of its
stable 2+3 partition.  Two such divisors meet precisely when their cherries
are disjoint.  This representation makes the complete subtraction forest
finite and explicit.

The local sewing integrand on a divisor has spin-zero terms

``d^2q |q|^(-2-kappa^2/2+2P^2+2n)``,

where ``kappa`` is the signed target-time energy through the tube.  The term
is power divergent when ``P^2+n < Re(kappa^2)/4``.  Its analytically
continued radial integral over ``|q|<rho`` is

``pi * rho**(2*alpha) / alpha``,
``alpha = P^2+n-kappa^2/4``.

At a corner the two commuting spin-zero projectors give the forest
combination ``I-S1 I-S2 I+S1 S2 I``.  Face and corner primitives must still
be restored; see :func:`forest_finite_part_terms` and the companion recipe.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

try:
    from sphere_four_point_subtraction import (
        bry_divergent_levels,
        generalized_binomial_series,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_subtraction import (
        bry_divergent_levels,
        generalized_binomial_series,
    )


Particle = int


@dataclass(frozen=True, order=True)
class BoundaryDivisor:
    """A stable 2+3 partition of five labels, named by its two-label side."""

    cherry: tuple[Particle, Particle]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(int(label) for label in self.cherry))
        if len(normalized) != 2 or normalized[0] == normalized[1]:
            raise ValueError("a boundary cherry must contain two distinct labels")
        object.__setattr__(self, "cherry", normalized)

    def is_compatible_with(self, other: "BoundaryDivisor") -> bool:
        """Return whether the two divisors meet in ``Mbar_{0,5}``."""

        return set(self.cherry).isdisjoint(other.cherry)


@dataclass(frozen=True, order=True)
class BoundaryCorner:
    """A trivalent five-leaf tree, equivalently two disjoint cherries."""

    divisors: tuple[BoundaryDivisor, BoundaryDivisor]
    middle_label: Particle


@dataclass(frozen=True)
class VisibleBoundaryChart:
    """The ten divisors visible after blowing up the ``(z1,z2)`` chart."""

    incoming: Particle
    moving_outgoing: Particle
    fixed_zero: Particle
    fixed_one: Particle
    fixed_infinity: Particle
    by_locus: Mapping[str, BoundaryDivisor]


@dataclass(frozen=True)
class ForestFinitePartTerms:
    """Algebraic pieces of the two-boundary finite-part operator."""

    bulk_remainder: complex
    face1_remainder: complex
    face2_remainder: complex
    corner_coefficient: complex


@dataclass(frozen=True)
class PlumbingForestEvaluation:
    """Fixed-(P1,P2) five-point density and its two-edge forest."""

    original: complex
    face1: complex
    face2: complex
    corner: complex
    remainder: complex
    levels1: tuple[int, ...]
    levels2: tuple[int, ...]


def canonical_divisor_ordering(
    divisor: BoundaryDivisor,
    labels: Sequence[Particle] = range(5),
) -> tuple[Particle, Particle, Particle, Particle, Particle]:
    """Choose one fixed linear frame with ``divisor`` as its left cherry."""

    normalized = tuple(sorted(int(label) for label in labels))
    if len(normalized) != 5 or len(set(normalized)) != 5:
        raise ValueError("labels must contain five distinct particles")
    left = tuple(sorted(divisor.cherry))
    remaining = tuple(label for label in normalized if label not in left)
    if len(remaining) != 3:
        raise ValueError("the divisor labels are not contained in labels")
    return (left[0], left[1], remaining[0], remaining[1], remaining[2])


def canonical_corner_ordering(
    corner: BoundaryCorner,
) -> tuple[Particle, Particle, Particle, Particle, Particle]:
    """Choose one fixed oriented representative of a boundary corner."""

    left, right = sorted(corner.divisors, key=lambda value: value.cherry)
    return (
        left.cherry[0],
        left.cherry[1],
        int(corner.middle_label),
        right.cherry[0],
        right.cherry[1],
    )


def five_point_face_sector_orderings(
    labels: Sequence[Particle] = range(5),
) -> tuple[tuple[BoundaryDivisor, tuple[Particle, ...]], ...]:
    """Return the six four-point crossing cells on each of ten faces."""

    normalized = tuple(sorted(int(label) for label in labels))
    sectors: list[tuple[BoundaryDivisor, tuple[Particle, ...]]] = []
    for divisor in five_point_boundary_divisors(normalized):
        remaining = tuple(label for label in normalized if label not in divisor.cherry)
        for permutation in itertools.permutations(remaining):
            sectors.append(
                (
                    divisor,
                    (
                        divisor.cherry[0],
                        divisor.cherry[1],
                        *permutation,
                    ),
                )
            )
    return tuple(sectors)


def five_point_boundary_divisors(labels: Sequence[Particle] = range(5)) -> tuple[BoundaryDivisor, ...]:
    """Return all ten boundary divisors of ``Mbar_{0,5}``."""

    normalized = tuple(int(label) for label in labels)
    if len(normalized) != 5 or len(set(normalized)) != 5:
        raise ValueError("labels must contain five distinct particle labels")
    return tuple(BoundaryDivisor(pair) for pair in itertools.combinations(normalized, 2))


def five_point_boundary_corners(labels: Sequence[Particle] = range(5)) -> tuple[BoundaryCorner, ...]:
    """Return all fifteen compatible divisor pairs (five-leaf trees)."""

    normalized = tuple(int(label) for label in labels)
    divisors = five_point_boundary_divisors(normalized)
    label_set = set(normalized)
    corners: list[BoundaryCorner] = []
    for left_index, left in enumerate(divisors):
        for right in divisors[left_index + 1 :]:
            if not left.is_compatible_with(right):
                continue
            remaining = label_set - set(left.cherry) - set(right.cherry)
            if len(remaining) != 1:
                raise AssertionError("compatible five-point cherries must leave one label")
            corners.append(
                BoundaryCorner(
                    divisors=(left, right),
                    middle_label=next(iter(remaining)),
                )
            )
    return tuple(corners)


def visible_boundary_chart(
    *,
    incoming: Particle = 0,
    moving_outgoing: Particle = 1,
    fixed_zero: Particle = 2,
    fixed_one: Particle = 3,
    fixed_infinity: Particle = 4,
) -> VisibleBoundaryChart:
    """Return the user's seven visible loci plus the three exceptional divisors."""

    labels = (incoming, moving_outgoing, fixed_zero, fixed_one, fixed_infinity)
    if len(set(labels)) != 5:
        raise ValueError("the five chart labels must be distinct")
    return VisibleBoundaryChart(
        incoming=int(incoming),
        moving_outgoing=int(moving_outgoing),
        fixed_zero=int(fixed_zero),
        fixed_one=int(fixed_one),
        fixed_infinity=int(fixed_infinity),
        by_locus={
            "z1->0": BoundaryDivisor((incoming, fixed_zero)),
            "z1->1": BoundaryDivisor((incoming, fixed_one)),
            "z1->infinity": BoundaryDivisor((incoming, fixed_infinity)),
            "z2->0": BoundaryDivisor((moving_outgoing, fixed_zero)),
            "z2->1": BoundaryDivisor((moving_outgoing, fixed_one)),
            "z2->infinity": BoundaryDivisor((moving_outgoing, fixed_infinity)),
            "z1->z2": BoundaryDivisor((incoming, moving_outgoing)),
            # A triple collision is the same stable partition as the cherry
            # formed by the two labels on the opposite component.
            "z1,z2->0": BoundaryDivisor((fixed_one, fixed_infinity)),
            "z1,z2->1": BoundaryDivisor((fixed_zero, fixed_infinity)),
            "z1,z2->infinity": BoundaryDivisor((fixed_zero, fixed_one)),
        },
    )


def signed_channel_energy(
    divisor: BoundaryDivisor,
    signed_energies: Mapping[Particle, complex] | Sequence[complex],
) -> complex:
    """Return the signed target-time energy flowing through a divisor."""

    if isinstance(signed_energies, Mapping):
        return complex(sum(complex(signed_energies[label]) for label in divisor.cherry))
    return complex(sum(complex(signed_energies[label]) for label in divisor.cherry))


def equal_outgoing_signed_energies(omega: complex) -> tuple[complex, ...]:
    """Return ``(+4 omega,-omega,-omega,-omega,-omega)`` for ``1->4``."""

    omega = complex(omega)
    return (4.0 * omega, -omega, -omega, -omega, -omega)


def divergent_spin_zero_levels(channel_energy: complex) -> tuple[int, ...]:
    """Return descendant levels whose momentum interval contains divergences."""

    threshold = 0.25 * complex(channel_energy) ** 2
    real_threshold = threshold.real
    if real_threshold <= 0.0:
        return ()
    # Strict inequality is required.  A level exactly at the endpoint has a
    # zero-length P interval and therefore contributes nothing.
    upper = max(0, int(math.ceil(real_threshold) - 1))
    return tuple(level for level in range(upper + 1) if level < real_threshold)


def divergent_momentum_endpoint(channel_energy: complex, level: int) -> float:
    """Return ``sqrt(Re(kappa^2)/4-level)`` for a divergent level."""

    level = int(level)
    if level < 0:
        raise ValueError("level must be non-negative")
    radicand = 0.25 * (complex(channel_energy) ** 2).real - level
    if radicand <= 0.0:
        return 0.0
    return math.sqrt(radicand)


def radial_exponent(channel_energy: complex, momentum: float, level: int) -> complex:
    """Return ``alpha=P^2+n-kappa^2/4`` for one spin-zero OPE term."""

    momentum = float(momentum)
    level = int(level)
    if momentum < 0.0 or level < 0:
        raise ValueError("momentum and level must be non-negative")
    return complex(momentum * momentum + level - 0.25 * complex(channel_energy) ** 2)


def radial_finite_part(
    alpha: complex,
    collar_radius: float,
    *,
    logarithmic_tolerance: float = 1.0e-13,
) -> complex:
    r"""Analytically continue ``int_{|q|<rho} d^2q |q|^{-2+2 alpha}``.

    The meromorphic answer is ``pi*rho^(2 alpha)/alpha``.  At an exact
    logarithmic pole the finite term after subtracting ``pi/alpha`` is
    ``2*pi*log(rho)``; this branch is useful for symbolic checks, while the
    physical ``i epsilon`` prescription normally keeps ``alpha`` nonzero.
    """

    alpha = complex(alpha)
    collar_radius = float(collar_radius)
    if not math.isfinite(collar_radius) or not 0.0 < collar_radius <= 1.0:
        raise ValueError("collar_radius must lie in (0,1]")
    if abs(alpha) < float(logarithmic_tolerance):
        return complex(2.0 * math.pi * math.log(collar_radius))
    return complex(
        math.pi * cmath.exp(2.0 * alpha * math.log(collar_radius)) / alpha
    )


def forest_subtracted_integrand(
    integrand: complex,
    face1: complex,
    face2: complex,
    corner: complex,
) -> complex:
    """Return ``I-S1 I-S2 I+S1 S2 I`` in a two-boundary chart."""

    return complex(integrand) - complex(face1) - complex(face2) + complex(corner)


def forest_finite_part_terms(
    integrand: complex,
    face1: complex,
    face2: complex,
    corner: complex,
) -> ForestFinitePartTerms:
    r"""Return the four algebraic sectors of ``FP_1 FP_2``.

    After integrating the returned pieces, the full finite part is

    ``int int bulk_remainder``
    ``+ A1 int face1_remainder + A2 int face2_remainder``
    ``+ A1 A2 corner_coefficient``,

    where ``Ai`` applies :func:`radial_finite_part` term-by-term to the
    corresponding spin-zero OPE series.
    """

    value = complex(integrand)
    first = complex(face1)
    second = complex(face2)
    overlap = complex(corner)
    return ForestFinitePartTerms(
        bulk_remainder=value - first - second + overlap,
        face1_remainder=first - overlap,
        face2_remainder=second - overlap,
        corner_coefficient=overlap,
    )


def integrate_spin_zero_counterterm(
    channel_energy: complex,
    level: int,
    collar_radius: float,
    coefficient: Callable[[float], complex],
    quadrature: Callable[[Callable[[float], complex], float, float], complex],
) -> complex:
    """Integrate one OPE counterterm over momentum after radial continuation."""

    endpoint = divergent_momentum_endpoint(channel_energy, level)
    if endpoint == 0.0:
        return 0.0 + 0.0j

    def integrand(momentum: float) -> complex:
        alpha = radial_exponent(channel_energy, momentum, level)
        return complex(coefficient(momentum)) * radial_finite_part(alpha, collar_radius) / math.pi

    return complex(quadrature(integrand, 0.0, endpoint))


def _multiply_bivariate_series(
    left: Mapping[tuple[int, int], complex],
    right: Mapping[tuple[int, int], complex],
    order1: int,
    order2: int,
) -> dict[tuple[int, int], complex]:
    """Multiply two bivariate series with rectangular truncation."""

    result: dict[tuple[int, int], complex] = {}
    for (left1, left2), left_value in left.items():
        for (right1, right2), right_value in right.items():
            level1 = int(left1) + int(right1)
            level2 = int(left2) + int(right2)
            if level1 > int(order1) or level2 > int(order2):
                continue
            key = (level1, level2)
            result[key] = result.get(key, 0.0 + 0.0j) + (
                complex(left_value) * complex(right_value)
            )
    return result


def five_point_regular_factor_coefficients(
    block_coefficients: Mapping[tuple[int, int], complex],
    ordered_signed_energies: Sequence[complex],
    *,
    order1: int,
    order2: int,
) -> dict[tuple[int, int], complex]:
    """Include timelike factors regular at a linear-channel corner.

    In the ordered frame with positions (0,q1*q2,q2,1,infinity), the regular
    chiral multiplier is the product of
    (1-q1)^(-k_b*k_c/2), (1-q1*q2)^(-k_b*k_d/2), and
    (1-q2)^(-k_c*k_d/2). Multiplying this by the CCY block gives the
    coefficients used by both face projectors and their corner overlap.
    """

    energies = tuple(complex(value) for value in ordered_signed_energies)
    if len(energies) != 5:
        raise ValueError("ordered_signed_energies must contain five values")
    order1 = int(order1)
    order2 = int(order2)
    if order1 < 0 or order2 < 0:
        raise ValueError("orders must be non-negative")
    _, k_b, k_c, k_d, _ = energies
    first = generalized_binomial_series(-0.5 * k_b * k_c, order1)
    diagonal = generalized_binomial_series(
        -0.5 * k_b * k_d,
        min(order1, order2),
    )
    second = generalized_binomial_series(-0.5 * k_c * k_d, order2)
    regular = {
        (int(level1), int(level2)): complex(value)
        for (level1, level2), value in block_coefficients.items()
        if int(level1) <= order1 and int(level2) <= order2
    }
    regular = _multiply_bivariate_series(
        regular,
        {(level, 0): value for level, value in enumerate(first)},
        order1,
        order2,
    )
    regular = _multiply_bivariate_series(
        regular,
        {(level, level): value for level, value in enumerate(diagonal)},
        order1,
        order2,
    )
    regular = _multiply_bivariate_series(
        regular,
        {(0, level): value for level, value in enumerate(second)},
        order1,
        order2,
    )
    return regular


def _evaluate_bivariate_series(
    coefficients: Mapping[tuple[int, int], complex],
    q1: complex,
    q2: complex,
) -> complex:
    return complex(
        sum(
            complex(value) * complex(q1) ** int(level1) * complex(q2) ** int(level2)
            for (level1, level2), value in coefficients.items()
        )
    )


def _evaluate_fixed_first_level(
    coefficients: Mapping[tuple[int, int], complex],
    first_level: int,
    q2: complex,
) -> complex:
    return complex(
        sum(
            complex(value) * complex(q2) ** int(level2)
            for (level1, level2), value in coefficients.items()
            if int(level1) == int(first_level)
        )
    )


def _evaluate_fixed_second_level(
    coefficients: Mapping[tuple[int, int], complex],
    second_level: int,
    q1: complex,
) -> complex:
    return complex(
        sum(
            complex(value) * complex(q1) ** int(level1)
            for (level1, level2), value in coefficients.items()
            if int(level2) == int(second_level)
        )
    )


def five_point_plumbing_channel_energies(
    ordered_signed_energies: Sequence[complex],
) -> tuple[complex, complex]:
    """Return the signed energies carried by q1 and q2 in a linear frame."""

    energies = tuple(complex(value) for value in ordered_signed_energies)
    if len(energies) != 5:
        raise ValueError("ordered_signed_energies must contain five values")
    first = energies[0] + energies[1]
    second = energies[3] + energies[4]
    return complex(first), complex(second)


def five_point_plumbing_radial_exponents(
    ordered_signed_energies: Sequence[complex],
    momentum1: complex | float,
    momentum2: complex | float,
) -> tuple[complex, complex]:
    """Return the two non-chiral exponents including the moduli Jacobian."""

    channel1, channel2 = five_point_plumbing_channel_energies(
        ordered_signed_energies
    )
    return (
        complex(-2.0 - 0.5 * channel1**2 + 2.0 * complex(momentum1) ** 2),
        complex(-2.0 - 0.5 * channel2**2 + 2.0 * complex(momentum2) ** 2),
    )


def five_point_fixed_momenta_forest(
    q1: complex,
    q2: complex,
    *,
    ordered_signed_energies: Sequence[complex],
    momentum1: float,
    momentum2: float,
    regular_coefficients: Mapping[tuple[int, int], complex],
) -> PlumbingForestEvaluation:
    """Apply the explicit BRY two-edge forest at fixed internal momenta.

    This returns I-S1 I-S2 I+S1 S2 I. Face coefficients retain the complete
    dependence on the other plumbing modulus. The overlap is the bivariate
    spin-zero coefficient, so a simultaneous degeneration is subtracted
    exactly once.
    """

    q1 = complex(q1)
    q2 = complex(q2)
    if q1 == 0.0 or q2 == 0.0:
        raise ZeroDivisionError("the forest is evaluated away from the boundary")
    coefficients = {
        (int(level1), int(level2)): complex(value)
        for (level1, level2), value in regular_coefficients.items()
    }
    maximum1 = max((key[0] for key in coefficients), default=0)
    maximum2 = max((key[1] for key in coefficients), default=0)
    channel1, channel2 = five_point_plumbing_channel_energies(
        ordered_signed_energies
    )
    levels1 = bry_divergent_levels(channel1, momentum1, maximum1)
    levels2 = bry_divergent_levels(channel2, momentum2, maximum2)
    exponent1, exponent2 = five_point_plumbing_radial_exponents(
        ordered_signed_energies,
        momentum1,
        momentum2,
    )
    log_radius1 = math.log(abs(q1))
    log_radius2 = math.log(abs(q2))
    primary = cmath.exp(
        exponent1 * log_radius1 + exponent2 * log_radius2
    )
    holomorphic = _evaluate_bivariate_series(coefficients, q1, q2)
    antiholomorphic = _evaluate_bivariate_series(
        coefficients,
        q1.conjugate(),
        q2.conjugate(),
    )
    original = primary * holomorphic * antiholomorphic

    face1 = 0.0 + 0.0j
    for level1 in levels1:
        holomorphic_row = _evaluate_fixed_first_level(
            coefficients,
            level1,
            q2,
        )
        antiholomorphic_row = _evaluate_fixed_first_level(
            coefficients,
            level1,
            q2.conjugate(),
        )
        face1 += cmath.exp(
            (exponent1 + 2.0 * level1) * log_radius1
            + exponent2 * log_radius2
        ) * holomorphic_row * antiholomorphic_row

    face2 = 0.0 + 0.0j
    for level2 in levels2:
        holomorphic_column = _evaluate_fixed_second_level(
            coefficients,
            level2,
            q1,
        )
        antiholomorphic_column = _evaluate_fixed_second_level(
            coefficients,
            level2,
            q1.conjugate(),
        )
        face2 += cmath.exp(
            exponent1 * log_radius1
            + (exponent2 + 2.0 * level2) * log_radius2
        ) * holomorphic_column * antiholomorphic_column

    corner = 0.0 + 0.0j
    for level1 in levels1:
        for level2 in levels2:
            coefficient = coefficients.get((level1, level2), 0.0 + 0.0j)
            corner += coefficient**2 * cmath.exp(
                (exponent1 + 2.0 * level1) * log_radius1
                + (exponent2 + 2.0 * level2) * log_radius2
            )
    remainder = original - face1 - face2 + corner
    return PlumbingForestEvaluation(
        original=complex(original),
        face1=complex(face1),
        face2=complex(face2),
        corner=complex(corner),
        remainder=complex(remainder),
        levels1=tuple(levels1),
        levels2=tuple(levels2),
    )


def five_point_fixed_momenta_face_finite_part(
    remaining_modulus: complex,
    *,
    ordered_signed_energies: Sequence[complex],
    momentum1: float,
    momentum2: float,
    regular_coefficients: Mapping[tuple[int, int], complex],
    collar_radius: float,
    edge: int = 1,
) -> complex:
    """Integrate one plumbing radius analytically at fixed momenta.

    The unintegrated modulus retains its full holomorphic row or column.  All
    spin-zero levels are continued by :func:`radial_finite_part`, including
    the power-divergent ones; this is the five-point analogue of the verified
    four-point collar finite part.
    """

    remaining_modulus = complex(remaining_modulus)
    if remaining_modulus == 0.0:
        raise ZeroDivisionError("the face modulus lies on a corner")
    edge = int(edge)
    if edge not in (1, 2):
        raise ValueError("edge must be 1 or 2")
    coefficients = {
        (int(level1), int(level2)): complex(value)
        for (level1, level2), value in regular_coefficients.items()
    }
    channel1, channel2 = five_point_plumbing_channel_energies(
        ordered_signed_energies
    )
    exponent1, exponent2 = five_point_plumbing_radial_exponents(
        ordered_signed_energies,
        momentum1,
        momentum2,
    )
    log_remaining = math.log(abs(remaining_modulus))
    total = 0.0 + 0.0j
    if edge == 1:
        maximum = max((key[0] for key in coefficients), default=0)
        other_primary = cmath.exp(exponent2 * log_remaining)
        for level in range(maximum + 1):
            row = _evaluate_fixed_first_level(coefficients, level, remaining_modulus)
            row_bar = _evaluate_fixed_first_level(
                coefficients,
                level,
                remaining_modulus.conjugate(),
            )
            alpha = radial_exponent(channel1, momentum1, level)
            total += radial_finite_part(alpha, collar_radius) * other_primary * row * row_bar
    else:
        maximum = max((key[1] for key in coefficients), default=0)
        other_primary = cmath.exp(exponent1 * log_remaining)
        for level in range(maximum + 1):
            column = _evaluate_fixed_second_level(coefficients, level, remaining_modulus)
            column_bar = _evaluate_fixed_second_level(
                coefficients,
                level,
                remaining_modulus.conjugate(),
            )
            alpha = radial_exponent(channel2, momentum2, level)
            total += radial_finite_part(alpha, collar_radius) * other_primary * column * column_bar
    return complex(total)


def five_point_fixed_momenta_corner_finite_part(
    *,
    ordered_signed_energies: Sequence[complex],
    momentum1: float,
    momentum2: float,
    regular_coefficients: Mapping[tuple[int, int], complex],
    collar_radius1: float,
    collar_radius2: float,
) -> complex:
    """Analytically continue both radial integrals in a corner bidisc."""

    channel1, channel2 = five_point_plumbing_channel_energies(
        ordered_signed_energies
    )
    return complex(
        sum(
            complex(coefficient) ** 2
            * radial_finite_part(
                radial_exponent(channel1, momentum1, int(level1)),
                collar_radius1,
            )
            * radial_finite_part(
                radial_exponent(channel2, momentum2, int(level2)),
                collar_radius2,
            )
            for (level1, level2), coefficient in regular_coefficients.items()
        )
    )
