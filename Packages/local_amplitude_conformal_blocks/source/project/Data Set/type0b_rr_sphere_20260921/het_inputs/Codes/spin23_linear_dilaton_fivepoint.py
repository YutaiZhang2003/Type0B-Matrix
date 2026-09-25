#!/usr/bin/env python3
r"""Screened super-linear-dilaton representative of the first five-point resonance.

At the first nonzero five-point wall resonance,

``P_in = 2 i`` and ``sum(P_out) = 2 i``, 

the residue of the super-Liouville zero-mode pole is a free-field correlator
with one integrated wall screening.  With external insertions

``(z1,z2,z3,z4,z5) = (0,z2,z3,1,infinity)``

and the screening at ``w``, the common bosonic factor is

.. math::

   B=\prod_{1\le i<j\le4}|z_i-z_j|^{-2(k_i k_j+a_i a_j)}
     \prod_{i=1}^4|w-z_i|^{-2a_i},\qquad a_i=1+iP_i.

The three zero-picture vertices are on legs 2, 3, 4.  Their four nonzero
PCO branches are summed by one holomorphic Pfaffian with polarizations
``p_i=(a_i,k_i)`` and ``p_w=(1,0)``.  The antiholomorphic Pfaffian uses
``a_i e_phi`` for a singlet, ``e_flavor`` for a vector, and ``e_phi`` for
the screening.  This gives every allowed S/V channel without a hafnian
ansatz.

The function :func:`screened_integrand` is the *meromorphic free-field
representative*, including the repository's factor ``2**-3`` for the three
PCOs.  It deliberately omits the common zero-mode residue, cosmological
constant, screening normalization, delta-normalized Liouville leg factors,
sphere normalization, and in/out reflection phases.

Crucially, this module does not define the integral by deleting collision
disks.  A zero-picture singlet gives a double chiral screening collision and
the all-singlet channel has no absolutely convergent chamber on the resonance
simplex.  The integral must be supplied by meromorphic continuation (for
example twisted cycles/KLT after IBP) or by an equivalent supersymmetric
contact-term prescription.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Callable, Hashable, Mapping, Sequence

from spin23_sphere_fivepoint_amplitude import (
    SphereFivePointState,
    validate_fivepoint_states,
)


Label = int | str
SCREEN_LABEL = "w"
FIRST_SCREENING_MOMENTUM = 2j
DEFAULT_ZERO_PICTURE_LEGS = (2, 3, 4)


def liouville_charge(momentum: complex) -> complex:
    r"""Return the free-field exponential charge ``a(P)=1+i P`` at ``b=1``."""

    return 1.0 + 1j * complex(momentum)


def _positions(z_2: complex, z_3: complex, w: complex) -> dict[Label, complex | None]:
    positions: dict[Label, complex | None] = {
        1: 0.0j,
        2: complex(z_2),
        3: complex(z_3),
        4: 1.0 + 0.0j,
        5: None,
        SCREEN_LABEL: complex(w),
    }
    finite = [positions[label] for label in (1, 2, 3, 4, SCREEN_LABEL)]
    if any(
        not math.isfinite(part)
        for value in finite
        for part in (complex(value).real, complex(value).imag)
    ):
        raise ValueError("puncture and screening positions must be finite")
    if len(set(finite)) != len(finite):
        raise ValueError("punctures and the screening must not collide")
    return positions


def _fermion_kernel(
    earlier: Label,
    later: Label,
    positions: Mapping[Label, complex | None],
) -> complex:
    earlier_position = positions[earlier]
    later_position = positions[later]
    if earlier_position is None:
        if later_position is None:
            raise ValueError("two fermions cannot both be at infinity")
        return 1.0 + 0.0j
    if later_position is None:
        raise ValueError("the infinity insertion must occur first in the ordering")
    return 1.0 / (earlier_position - later_position)


def _pfaffian_wick(
    ordered_labels: tuple[Label, ...],
    positions: Mapping[Label, complex | None],
    contraction: Callable[[Label, Label], complex],
) -> complex:
    """Evaluate one free-fermion Pfaffian in a declared global ordering."""

    if len(ordered_labels) % 2:
        return 0.0j
    if not ordered_labels:
        return 1.0 + 0.0j
    first = ordered_labels[0]
    total = 0.0j
    for offset, second in enumerate(ordered_labels[1:], start=1):
        coefficient = complex(contraction(first, second))
        if coefficient == 0:
            continue
        rest = ordered_labels[1:offset] + ordered_labels[offset + 1 :]
        sign = -1 if (offset + 1) % 2 else 1
        total += (
            sign
            * coefficient
            * _fermion_kernel(first, second, positions)
            * _pfaffian_wick(rest, positions, contraction)
        )
    return total


@dataclass(frozen=True)
class FermionMatchingTerm:
    """One signed perfect matching before its coordinate kernels are applied."""

    coefficient: complex
    edges: tuple[tuple[Label, Label], ...]

    def evaluate(self, positions: Mapping[Label, complex | None]) -> complex:
        result = complex(self.coefficient)
        for earlier, later in self.edges:
            result *= _fermion_kernel(earlier, later, positions)
        return result


def _matching_terms(
    ordered_labels: tuple[Label, ...],
    contraction: Callable[[Label, Label], complex],
) -> tuple[FermionMatchingTerm, ...]:
    """Expand a Pfaffian into signed denominator-edge terms."""

    if len(ordered_labels) % 2:
        return ()
    if not ordered_labels:
        return (FermionMatchingTerm(1.0 + 0.0j, ()),)
    first = ordered_labels[0]
    terms: list[FermionMatchingTerm] = []
    for offset, second in enumerate(ordered_labels[1:], start=1):
        pair_coefficient = complex(contraction(first, second))
        if pair_coefficient == 0:
            continue
        rest = ordered_labels[1:offset] + ordered_labels[offset + 1 :]
        sign = -1 if (offset + 1) % 2 else 1
        for subterm in _matching_terms(rest, contraction):
            terms.append(
                FermionMatchingTerm(
                    coefficient=sign * pair_coefficient * subterm.coefficient,
                    edges=((first, second),) + subterm.edges,
                )
            )
    return tuple(terms)


def _validate_first_resonance(
    states: Sequence[SphereFivePointState],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[SphereFivePointState, ...]:
    normalized = validate_fivepoint_states(states)
    incoming = normalized[4]
    if abs(complex(incoming.liouville_momentum) - FIRST_SCREENING_MOMENTUM) > tolerance:
        raise ValueError("the first five-point resonance requires incoming P=2i")
    outgoing_total = sum(
        complex(state.liouville_momentum) for state in normalized[:4]
    )
    if abs(outgoing_total - FIRST_SCREENING_MOMENTUM) > tolerance:
        raise ValueError("the outgoing Liouville momenta must sum to 2i")
    charges = tuple(
        liouville_charge(state.liouville_momentum) for state in normalized
    )
    if abs(sum(charges) + 1.0 - 2.0) > tolerance:
        raise ValueError("external charges plus one screening are not neutral")
    return normalized


def _validate_soft_regulator(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[SphereFivePointState, ...]:
    """Validate the six-point soft-singlet definition of the wall insertion."""

    normalized = tuple(states)
    if len(normalized) != 5 or any(
        not isinstance(state, SphereFivePointState) for state in normalized
    ):
        raise ValueError("exactly five physical SphereFivePointState objects are required")
    soft = complex(soft_momentum)
    incoming = normalized[4]
    if abs(complex(incoming.liouville_momentum) - FIRST_SCREENING_MOMENTUM) > tolerance:
        raise ValueError("the regulated family keeps the incoming momentum at 2i")
    if abs(sum(complex(state.time_momentum) for state in normalized) + soft) > tolerance:
        raise ValueError("the five physical time momenta plus the soft momentum must sum to zero")
    outgoing_total = sum(
        complex(state.liouville_momentum) for state in normalized[:4]
    )
    if abs(outgoing_total + soft - FIRST_SCREENING_MOMENTUM) > tolerance:
        raise ValueError("the outgoing physical and soft Liouville momenta must sum to 2i")
    for state in normalized:
        p = complex(state.liouville_momentum)
        k = complex(state.time_momentum)
        if abs(p * p - k * k) > tolerance * max(1.0, abs(p * p), abs(k * k)):
            raise ValueError("every physical state must remain on shell")
    charges = tuple(
        liouville_charge(state.liouville_momentum) for state in normalized
    )
    if abs(sum(charges) + liouville_charge(soft) - 2.0) > tolerance:
        raise ValueError("the six free-field exponentials are not charge neutral")
    return normalized


def _holomorphic_contraction(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex = 0.0j,
) -> Callable[[Label, Label], complex]:
    charges = {
        leg: liouville_charge(states[leg - 1].liouville_momentum)
        for leg in range(1, 6)
    }
    times = {
        leg: complex(states[leg - 1].time_momentum) for leg in range(1, 6)
    }
    soft_charge = liouville_charge(soft_momentum)
    soft_time = complex(soft_momentum)

    def contraction(first: Label, second: Label) -> complex:
        if first == SCREEN_LABEL:
            first, second = second, first
        if second == SCREEN_LABEL:
            return (
                charges[int(first)] * soft_charge
                + times[int(first)] * soft_time
            )
        return (
            charges[int(first)] * charges[int(second)]
            + times[int(first)] * times[int(second)]
        )

    return contraction


def _antiholomorphic_contraction(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex = 0.0j,
) -> Callable[[Label, Label], complex]:
    charges = {
        leg: liouville_charge(states[leg - 1].liouville_momentum)
        for leg in range(1, 6)
    }
    soft_charge = liouville_charge(soft_momentum)

    def contraction(first: Label, second: Label) -> complex:
        if first == SCREEN_LABEL:
            first, second = second, first
        if second == SCREEN_LABEL:
            state = states[int(first) - 1]
            return (
                charges[int(first)] * soft_charge
                if state.kind == "singlet"
                else 0.0j
            )
        first_state = states[int(first) - 1]
        second_state = states[int(second) - 1]
        if first_state.kind != second_state.kind:
            return 0.0j
        if first_state.kind == "singlet":
            return charges[int(first)] * charges[int(second)]
        return 1.0 if first_state.flavor == second_state.flavor else 0.0j

    return contraction


def fermion_matching_terms(
    states: Sequence[SphereFivePointState],
    *,
    zero_picture_legs: Sequence[int] = DEFAULT_ZERO_PICTURE_LEGS,
) -> tuple[tuple[FermionMatchingTerm, ...], tuple[FermionMatchingTerm, ...]]:
    """Return holomorphic and antiholomorphic Pfaffian term expansions.

    The screening is placed last in the declared global fermion ordering.
    This is the ordering inherited from expanding the wall interaction.
    """

    normalized = _validate_first_resonance(states)
    zero_legs = tuple(int(leg) for leg in zero_picture_legs)
    if len(zero_legs) != 3 or len(set(zero_legs)) != 3:
        raise ValueError("exactly three distinct zero-picture legs are required")
    if any(leg not in (1, 2, 3, 4) for leg in zero_legs):
        raise ValueError("the zero-picture legs must be finite external legs")
    holomorphic_order = tuple(sorted(zero_legs, reverse=True)) + (SCREEN_LABEL,)
    antiholomorphic_order: tuple[Label, ...] = (5, 4, 3, 2, 1, SCREEN_LABEL)
    return (
        _matching_terms(
            holomorphic_order, _holomorphic_contraction(normalized)
        ),
        _matching_terms(
            antiholomorphic_order,
            _antiholomorphic_contraction(normalized),
        ),
    )


def regulated_fermion_matching_terms(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    *,
    zero_picture_legs: Sequence[int] = DEFAULT_ZERO_PICTURE_LEGS,
) -> tuple[tuple[FermionMatchingTerm, ...], tuple[FermionMatchingTerm, ...]]:
    """Return matching terms for a generic sixth outgoing soft singlet.

    The sixth singlet is in picture zero and is inserted at ``w``.  At
    ``soft_momentum=0`` these Pfaffians coincide with the Yukawa-screening
    Pfaffians; only the overall picture factor differs.
    """

    normalized = _validate_soft_regulator(states, soft_momentum)
    zero_legs = tuple(int(leg) for leg in zero_picture_legs)
    if len(zero_legs) != 3 or len(set(zero_legs)) != 3:
        raise ValueError("exactly three physical zero-picture legs are required")
    if any(leg not in (1, 2, 3, 4) for leg in zero_legs):
        raise ValueError("the physical zero-picture legs must be finite")
    holomorphic_order = tuple(sorted(zero_legs, reverse=True)) + (SCREEN_LABEL,)
    antiholomorphic_order: tuple[Label, ...] = (5, 4, 3, 2, 1, SCREEN_LABEL)
    return (
        _matching_terms(
            holomorphic_order,
            _holomorphic_contraction(normalized, soft_momentum),
        ),
        _matching_terms(
            antiholomorphic_order,
            _antiholomorphic_contraction(normalized, soft_momentum),
        ),
    )


def _absolute_power(distance: float, exponent: complex) -> complex:
    if distance <= 0:
        raise ValueError("free-field insertion points collide")
    return cmath.exp(complex(exponent) * math.log(distance))


def screened_bosonic_factor(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    w: complex,
) -> complex:
    """Return the common exponential Koba--Nielsen factor at ``s=1``."""

    normalized = _validate_first_resonance(states)
    positions = _positions(z_2, z_3, w)
    charges = tuple(
        liouville_charge(state.liouville_momentum) for state in normalized
    )
    result = 1.0 + 0.0j
    for first in range(1, 5):
        for second in range(first + 1, 5):
            distance = abs(
                complex(positions[first]) - complex(positions[second])
            )
            exponent = -2.0 * (
                complex(normalized[first - 1].time_momentum)
                * complex(normalized[second - 1].time_momentum)
                + charges[first - 1] * charges[second - 1]
            )
            result *= _absolute_power(distance, exponent)
    for leg in range(1, 5):
        distance = abs(complex(positions[leg]) - complex(positions[SCREEN_LABEL]))
        result *= _absolute_power(distance, -2.0 * charges[leg - 1])
    return result


def regulated_soft_singlet_bosonic_factor(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    z_2: complex,
    z_3: complex,
    w: complex,
) -> complex:
    """Return the neutral six-exponential factor before the soft limit."""

    normalized = _validate_soft_regulator(states, soft_momentum)
    positions = _positions(z_2, z_3, w)
    charges = tuple(
        liouville_charge(state.liouville_momentum) for state in normalized
    )
    soft_charge = liouville_charge(soft_momentum)
    soft_time = complex(soft_momentum)
    result = 1.0 + 0.0j
    for first in range(1, 5):
        for second in range(first + 1, 5):
            distance = abs(
                complex(positions[first]) - complex(positions[second])
            )
            exponent = -2.0 * (
                complex(normalized[first - 1].time_momentum)
                * complex(normalized[second - 1].time_momentum)
                + charges[first - 1] * charges[second - 1]
            )
            result *= _absolute_power(distance, exponent)
    for leg in range(1, 5):
        distance = abs(complex(positions[leg]) - complex(positions[SCREEN_LABEL]))
        exponent = -2.0 * (
            charges[leg - 1] * soft_charge
            + complex(normalized[leg - 1].time_momentum) * soft_time
        )
        result *= _absolute_power(distance, exponent)
    return result


@dataclass(frozen=True)
class ScreenedIntegrandEvaluation:
    """Factorized value of the first-resonance free-field representative."""

    value: complex
    bosonic_factor: complex
    holomorphic_pfaffian: complex
    antiholomorphic_pfaffian: complex
    picture_factor: float


def evaluate_screened_integrand(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    w: complex,
    *,
    zero_picture_legs: Sequence[int] = DEFAULT_ZERO_PICTURE_LEGS,
    include_picture_factor: bool = True,
) -> ScreenedIntegrandEvaluation:
    """Evaluate the raw screened free-field representative at one point."""

    normalized = _validate_first_resonance(states)
    positions = _positions(z_2, z_3, w)
    zero_legs = tuple(int(leg) for leg in zero_picture_legs)
    if len(zero_legs) != 3 or len(set(zero_legs)) != 3:
        raise ValueError("exactly three distinct zero-picture legs are required")
    if any(leg not in (1, 2, 3, 4) for leg in zero_legs):
        raise ValueError("the zero-picture legs must be finite external legs")
    holomorphic_order = tuple(sorted(zero_legs, reverse=True)) + (SCREEN_LABEL,)
    antiholomorphic_order: tuple[Label, ...] = (5, 4, 3, 2, 1, SCREEN_LABEL)
    holomorphic = _pfaffian_wick(
        holomorphic_order,
        positions,
        _holomorphic_contraction(normalized),
    )
    antiholomorphic_positions = {
        label: (None if value is None else complex(value).conjugate())
        for label, value in positions.items()
    }
    antiholomorphic = _pfaffian_wick(
        antiholomorphic_order,
        antiholomorphic_positions,
        _antiholomorphic_contraction(normalized),
    )
    bosonic = screened_bosonic_factor(normalized, z_2, z_3, w)
    picture_factor = 0.5 ** len(zero_legs) if include_picture_factor else 1.0
    return ScreenedIntegrandEvaluation(
        value=picture_factor * bosonic * holomorphic * antiholomorphic,
        bosonic_factor=bosonic,
        holomorphic_pfaffian=holomorphic,
        antiholomorphic_pfaffian=antiholomorphic,
        picture_factor=picture_factor,
    )


def evaluate_regulated_soft_singlet_integrand(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    z_2: complex,
    z_3: complex,
    w: complex,
    *,
    zero_picture_legs: Sequence[int] = DEFAULT_ZERO_PICTURE_LEGS,
    include_picture_factor: bool = True,
) -> ScreenedIntegrandEvaluation:
    r"""Evaluate the ordinary six-point correlator defining the soft limit.

    The extra outgoing singlet at ``w`` is also in picture zero, so the
    picture factor is ``2**-4``.  At zero soft momentum the separated-point
    value is exactly one half of :func:`evaluate_screened_integrand`, because
    the Yukawa screening lacks the extra singlet vertex's factor ``1/2``.
    Defining the wall residue as the meromorphic ``soft_momentum -> 0`` limit
    is what retains collision/contact contributions.
    """

    normalized = _validate_soft_regulator(states, soft_momentum)
    positions = _positions(z_2, z_3, w)
    zero_legs = tuple(int(leg) for leg in zero_picture_legs)
    if len(zero_legs) != 3 or len(set(zero_legs)) != 3:
        raise ValueError("exactly three physical zero-picture legs are required")
    if any(leg not in (1, 2, 3, 4) for leg in zero_legs):
        raise ValueError("the physical zero-picture legs must be finite")
    holomorphic_order = tuple(sorted(zero_legs, reverse=True)) + (SCREEN_LABEL,)
    antiholomorphic_order: tuple[Label, ...] = (5, 4, 3, 2, 1, SCREEN_LABEL)
    holomorphic = _pfaffian_wick(
        holomorphic_order,
        positions,
        _holomorphic_contraction(normalized, soft_momentum),
    )
    antiholomorphic_positions = {
        label: (None if value is None else complex(value).conjugate())
        for label, value in positions.items()
    }
    antiholomorphic = _pfaffian_wick(
        antiholomorphic_order,
        antiholomorphic_positions,
        _antiholomorphic_contraction(normalized, soft_momentum),
    )
    bosonic = regulated_soft_singlet_bosonic_factor(
        normalized, soft_momentum, z_2, z_3, w
    )
    picture_factor = 0.5 ** (len(zero_legs) + 1) if include_picture_factor else 1.0
    return ScreenedIntegrandEvaluation(
        value=picture_factor * bosonic * holomorphic * antiholomorphic,
        bosonic_factor=bosonic,
        holomorphic_pfaffian=holomorphic,
        antiholomorphic_pfaffian=antiholomorphic,
        picture_factor=picture_factor,
    )


def screened_integrand(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    w: complex,
    *,
    zero_picture_legs: Sequence[int] = DEFAULT_ZERO_PICTURE_LEGS,
    include_picture_factor: bool = True,
) -> complex:
    """Return only the value from :func:`evaluate_screened_integrand`."""

    return evaluate_screened_integrand(
        states,
        z_2,
        z_3,
        w,
        zero_picture_legs=zero_picture_legs,
        include_picture_factor=include_picture_factor,
    ).value


def screen_collision_radial_power(
    state: SphereFivePointState,
    *,
    zero_picture: bool,
) -> float:
    r"""Return the worst real radial power near ``w=z_i``.

    The local measure is ``r dr``.  Absolute convergence therefore requires
    the returned power to be strictly greater than ``-2``.  The result is
    exact for imaginary momentum and reports the real part for a general
    complex continuation.
    """

    base = (-2.0 * liouville_charge(state.liouville_momentum)).real
    chiral_poles = int(bool(zero_picture)) + int(state.kind == "singlet")
    return base - chiral_poles


__all__ = [
    "DEFAULT_ZERO_PICTURE_LEGS",
    "FIRST_SCREENING_MOMENTUM",
    "FermionMatchingTerm",
    "SCREEN_LABEL",
    "ScreenedIntegrandEvaluation",
    "evaluate_screened_integrand",
    "evaluate_regulated_soft_singlet_integrand",
    "fermion_matching_terms",
    "liouville_charge",
    "screen_collision_radial_power",
    "regulated_fermion_matching_terms",
    "regulated_soft_singlet_bosonic_factor",
    "screened_bosonic_factor",
    "screened_integrand",
]
