#!/usr/bin/env python3
r"""Free-field representatives for the ``s=3`` five-point resonance.

The second nonzero genus-zero five-point resonance has incoming momentum
``P0=3i``.  Charge three can be supplied by either

``YYY``
    three Yukawa screenings ``psi phi * psibar phi * exp(phi)``;

``YB``
    one Yukawa screening and one auxiliary-field (bosonic) screening
    ``exp(2 phi)``.

Both components are required by the component super-Liouville action.  In
the conventions used by this repository their expansion coefficients are

``(-2 i mu)**3 / 3!`` and ``(-2 i mu) * (-2 pi mu**2)``.

This module supplies the exact separated-point integrands.  It intentionally
does *not* assign a finite part to their collision singularities.  The three
Yukawa representative can equivalently be obtained as the soft limit of an
ordinary neutral eight-point correlator with three additional picture-zero
singlets.  At separated points it is eight times that correlator, because a
wall Yukawa insertion does not carry the soft vertex's factor ``1/2``.

The equality at separated points is not permission to drop ``YB``.  The
``YB`` term fixes the local extension of the distribution on pairwise
screening diagonals.  A numerical implementation must therefore combine the
two action components before removing the common soft regulator.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from itertools import combinations
from typing import Callable, Hashable, Mapping, Sequence

from spin23_fivepoint_resonance_channels import FivePointChannelProjection
from spin23_sphere_fivepoint_amplitude import SphereFivePointState


Label = int | str
S3_RESONANCE_MOMENTUM = 3j
SOFT_LABELS: tuple[str, str, str] = ("w1", "w2", "w3")
PHYSICAL_ZERO_PICTURE_LABELS: tuple[int, int, int] = (7, 3, 2)


def liouville_charge(momentum: complex) -> complex:
    """Return the ``b=1`` free-field charge ``a(P)=1+iP``."""

    return 1.0 + 1j * complex(momentum)


@dataclass(frozen=True)
class SoftS3Family:
    """A neutral eight-point regulator of one five-point ``s=3`` state."""

    physical_states: tuple[SphereFivePointState, ...]
    soft_momenta: tuple[complex, complex, complex]

    def __post_init__(self) -> None:
        if len(self.physical_states) != 5:
            raise ValueError("five physical states are required")
        if len(self.soft_momenta) != 3:
            raise ValueError("three soft momenta are required")

    @property
    def soft_states(self) -> tuple[SphereFivePointState, ...]:
        return tuple(
            SphereFivePointState.singlet(momentum, momentum)
            for momentum in self.soft_momenta
        )

    @property
    def all_states_by_label(self) -> dict[Label, SphereFivePointState]:
        out1, out2, out3, out4, incoming = self.physical_states
        return {
            1: out1,
            2: out2,
            3: out3,
            "w1": self.soft_states[0],
            "w2": self.soft_states[1],
            "w3": self.soft_states[2],
            7: out4,
            8: incoming,
        }


def soft_s3_family(
    projection: FivePointChannelProjection,
    resonant_outgoing_momenta: Sequence[complex],
    soft_momenta: Sequence[complex],
    *,
    tolerance: float = 1.0e-10,
) -> SoftS3Family:
    r"""Build a neutral eight-point family approaching the ``s=3`` wall.

    The first physical outgoing momentum is shifted by
    ``-sum(soft_momenta)``.  Thus the five physical states and three soft
    singlets conserve timelike momentum, while their exponential charges sum
    to the background charge two at every nonzero regulator value.
    """

    outgoing = tuple(complex(value) for value in resonant_outgoing_momenta)
    soft = tuple(complex(value) for value in soft_momenta)
    if len(outgoing) != 4 or abs(sum(outgoing) - S3_RESONANCE_MOMENTUM) > tolerance:
        raise ValueError("four resonant outgoing momenta with sum 3i are required")
    if len(soft) != 3:
        raise ValueError("three soft momenta are required")
    shifted = (outgoing[0] - sum(soft),) + outgoing[1:]

    kinds = projection.incoming_first_kinds
    flavors = projection.flavors
    incoming_first_momenta = (S3_RESONANCE_MOMENTUM,) + shifted
    incoming_first: list[SphereFivePointState] = []
    for label, (kind, momentum, flavor) in enumerate(
        zip(kinds, incoming_first_momenta, flavors)
    ):
        time_momentum = -momentum if label == 0 else momentum
        if kind == "S":
            state = SphereFivePointState.singlet(momentum, time_momentum)
        else:
            state = SphereFivePointState.vector(
                momentum, time_momentum, int(flavor)
            )
        incoming_first.append(state)
    physical = tuple(incoming_first[1:] + incoming_first[:1])
    family = SoftS3Family(physical, soft)  # type: ignore[arg-type]

    all_states = tuple(family.all_states_by_label.values())
    if abs(sum(complex(state.time_momentum) for state in all_states)) > tolerance:
        raise AssertionError("the regulated family failed timelike neutrality")
    if abs(
        sum(liouville_charge(state.liouville_momentum) for state in all_states)
        - 2.0
    ) > tolerance:
        raise AssertionError("the regulated family failed charge neutrality")
    return family


def _positions(
    z2: complex,
    z3: complex,
    w1: complex,
    w2: complex,
    w3: complex,
) -> dict[Label, complex | None]:
    result: dict[Label, complex | None] = {
        1: 0.0j,
        2: complex(z2),
        3: complex(z3),
        "w1": complex(w1),
        "w2": complex(w2),
        "w3": complex(w3),
        7: 1.0 + 0.0j,
        8: None,
    }
    finite = tuple(value for value in result.values() if value is not None)
    if any(
        not math.isfinite(part)
        for value in finite
        for part in (complex(value).real, complex(value).imag)
    ):
        raise ValueError("all finite insertion points must be finite")
    if len(set(finite)) != len(finite):
        raise ValueError("free-field insertion points must not collide")
    return result


def _kernel(
    earlier: Label,
    later: Label,
    positions: Mapping[Label, complex | None],
) -> complex:
    first, second = positions[earlier], positions[later]
    if first is None:
        if second is None:
            raise ValueError("two fermions cannot both be at infinity")
        return 1.0 + 0.0j
    if second is None:
        raise ValueError("the infinity fermion must be first in the ordering")
    return 1.0 / (first - second)


def _pfaffian(
    labels: tuple[Label, ...],
    positions: Mapping[Label, complex | None],
    contraction: Callable[[Label, Label], complex],
) -> complex:
    if len(labels) % 2:
        return 0.0j
    if not labels:
        return 1.0 + 0.0j
    first = labels[0]
    total = 0.0j
    for offset, second in enumerate(labels[1:], start=1):
        pair = complex(contraction(first, second))
        if pair == 0:
            continue
        rest = labels[1:offset] + labels[offset + 1 :]
        sign = -1 if (offset + 1) % 2 else 1
        total += (
            sign
            * pair
            * _kernel(first, second, positions)
            * _pfaffian(rest, positions, contraction)
        )
    return total


def _holomorphic_contraction(
    family: SoftS3Family,
) -> Callable[[Label, Label], complex]:
    states = family.all_states_by_label

    def contraction(first: Label, second: Label) -> complex:
        left, right = states[first], states[second]
        return (
            liouville_charge(left.liouville_momentum)
            * liouville_charge(right.liouville_momentum)
            + complex(left.time_momentum) * complex(right.time_momentum)
        )

    return contraction


def _antiholomorphic_contraction(
    family: SoftS3Family,
) -> Callable[[Label, Label], complex]:
    states = family.all_states_by_label

    def contraction(first: Label, second: Label) -> complex:
        left, right = states[first], states[second]
        if left.kind != right.kind:
            return 0.0j
        if left.kind == "singlet":
            return (
                liouville_charge(left.liouville_momentum)
                * liouville_charge(right.liouville_momentum)
            )
        return 1.0 if left.flavor == right.flavor else 0.0j

    return contraction


def _absolute_power(distance: float, exponent: complex) -> complex:
    if distance <= 0:
        raise ValueError("free-field insertions collide")
    return cmath.exp(complex(exponent) * math.log(distance))


def soft_eightpoint_bosonic_factor(
    family: SoftS3Family,
    z2: complex,
    z3: complex,
    w1: complex,
    w2: complex,
    w3: complex,
) -> complex:
    """Return the neutral eight-exponential Koba--Nielsen factor."""

    positions = _positions(z2, z3, w1, w2, w3)
    states = family.all_states_by_label
    finite_labels: tuple[Label, ...] = (1, 2, 3, "w1", "w2", "w3", 7)
    result = 1.0 + 0.0j
    for index, first in enumerate(finite_labels):
        for second in finite_labels[index + 1 :]:
            left, right = states[first], states[second]
            exponent = -2.0 * (
                liouville_charge(left.liouville_momentum)
                * liouville_charge(right.liouville_momentum)
                + complex(left.time_momentum) * complex(right.time_momentum)
            )
            distance = abs(complex(positions[first]) - complex(positions[second]))
            result *= _absolute_power(distance, exponent)
    return result


@dataclass(frozen=True)
class S3IntegrandEvaluation:
    """Factorized value of the regulated soft-eight-point integrand."""

    value: complex
    bosonic_factor: complex
    holomorphic_pfaffian: complex
    antiholomorphic_pfaffian: complex
    picture_factor: float


@dataclass(frozen=True)
class S3Component:
    r"""One component-action contribution of total wall charge three.

    ``external_contacts`` uses the worldsheet labels ``2,3,7``.  Only a
    picture-zero singlet can be replaced by its auxiliary-field contact.
    Charge conservation is

    ``len(external_contacts) + yukawa_count + 2*bosonic_count = 3``.
    """

    external_contacts: tuple[int, ...]
    yukawa_count: int
    bosonic_count: int

    def __post_init__(self) -> None:
        contacts = tuple(sorted(int(label) for label in self.external_contacts))
        if len(set(contacts)) != len(contacts) or any(
            label not in PHYSICAL_ZERO_PICTURE_LABELS for label in contacts
        ):
            raise ValueError("contacts must be distinct labels among 2,3,7")
        if self.yukawa_count < 0 or self.bosonic_count < 0:
            raise ValueError("screening counts must be nonnegative")
        if len(contacts) + self.yukawa_count + 2 * self.bosonic_count != 3:
            raise ValueError("an s=3 component must carry total wall charge three")
        object.__setattr__(self, "external_contacts", contacts)

    @property
    def integrated_screen_count(self) -> int:
        return self.yukawa_count + self.bosonic_count

    @property
    def moduli_dimension(self) -> int:
        """Return the complex dimension including the two physical moduli."""

        return 2 + self.integrated_screen_count


@dataclass(frozen=True)
class S3ComponentEvaluation:
    """A bare component integrand and its component-action coefficient."""

    component: S3Component
    value: complex
    action_coefficient: complex
    weighted_value: complex
    bosonic_factor: complex
    holomorphic_pfaffian: complex
    antiholomorphic_pfaffian: complex
    picture_factor: float


def s3_components(
    projection: FivePointChannelProjection,
) -> tuple[S3Component, ...]:
    """Enumerate every nonzero component-action term at charge three.

    This includes external-vertex contacts.  Omitting them gives an
    incomplete resonance even if all pairwise screening collisions are
    treated correctly.
    """

    worldsheet_kind = {
        2: projection.incoming_first_kinds[2],
        3: projection.incoming_first_kinds[3],
        7: projection.incoming_first_kinds[4],
    }
    eligible = tuple(
        label for label in PHYSICAL_ZERO_PICTURE_LABELS
        if worldsheet_kind[label] == "S"
    )
    result: list[S3Component] = []
    for contact_count in range(min(3, len(eligible)) + 1):
        remaining_charge = 3 - contact_count
        for contacts in combinations(eligible, contact_count):
            for bosonic_count in range(remaining_charge // 2 + 1):
                yukawa_count = remaining_charge - 2 * bosonic_count
                result.append(
                    S3Component(contacts, yukawa_count, bosonic_count)
                )
    return tuple(result)


def s3_component_action_coefficient(
    component: S3Component,
    physical_states: Sequence[SphereFivePointState],
    *,
    mu: complex = 1.0,
) -> complex:
    r"""Return the coefficient of one charge-three component.

    Eliminating the auxiliary field in the convention that gives

    ``2 i mu b^2 psibar psi exp(b phi) + 2 pi mu^2 b^2 exp(2b phi)``

    yields

    ``W_a = a^2 psibar psi exp(a phi) - 2 pi i mu a b exp((a+b)phi)``.

    At ``b=1`` this gives the external factors below.  The factorials are
    those of the action expansion for labelled integration variables.
    """

    states = tuple(physical_states)
    if len(states) != 5:
        raise ValueError("five physical states are required")
    by_label = {2: states[1], 3: states[2], 7: states[3]}
    value = 1.0 + 0.0j
    for label in component.external_contacts:
        state = by_label[label]
        if state.kind != "singlet":
            raise ValueError("only a picture-zero singlet has an external contact")
        value *= -2.0j * math.pi * liouville_charge(state.liouville_momentum)
    value *= (-2.0j) ** component.yukawa_count / math.factorial(
        component.yukawa_count
    )
    value *= (-2.0 * math.pi) ** component.bosonic_count / math.factorial(
        component.bosonic_count
    )
    return value * complex(mu) ** 3


def evaluate_s3_component_integrand(
    projection: FivePointChannelProjection,
    resonant_outgoing_momenta: Sequence[complex],
    component: S3Component,
    z2: complex,
    z3: complex,
    *,
    yukawa_positions: Sequence[complex] = (),
    bosonic_positions: Sequence[complex] = (),
    mu: complex = 1.0,
) -> S3ComponentEvaluation:
    r"""Evaluate any separated component of the full ``s=3`` residue.

    The exponential charge of a contacted external field is increased by
    one and both of that field's fermions are removed.  A Yukawa screen has
    charge one and two fermions; a bosonic screen has charge two and no
    fermions.  All three physical picture-zero factors remain present,
    including on a contact branch.

    Components have different integration dimensions.  ``weighted_value``
    is consequently a diagnostic integrand, not something to add pointwise
    to the other components.  Their *integrals* must be combined before the
    common soft/contact regulator is removed.
    """

    y_positions = tuple(complex(value) for value in yukawa_positions)
    b_positions = tuple(complex(value) for value in bosonic_positions)
    if len(y_positions) != component.yukawa_count:
        raise ValueError("yukawa_positions disagrees with the component")
    if len(b_positions) != component.bosonic_count:
        raise ValueError("bosonic_positions disagrees with the component")
    family = soft_s3_family(
        projection, resonant_outgoing_momenta, (0.0j, 0.0j, 0.0j)
    )
    physical = family.physical_states
    out1, out2, out3, out4, incoming = physical
    states: dict[Label, SphereFivePointState] = {
        1: out1,
        2: out2,
        3: out3,
        7: out4,
        8: incoming,
    }
    positions: dict[Label, complex | None] = {
        1: 0.0j,
        2: complex(z2),
        3: complex(z3),
        7: 1.0 + 0.0j,
        8: None,
    }
    y_labels = tuple(f"y{index + 1}" for index in range(len(y_positions)))
    b_labels = tuple(f"b{index + 1}" for index in range(len(b_positions)))
    positions.update(zip(y_labels, y_positions))
    positions.update(zip(b_labels, b_positions))
    finite_positions = tuple(value for value in positions.values() if value is not None)
    if any(
        not math.isfinite(part)
        for value in finite_positions
        for part in (complex(value).real, complex(value).imag)
    ):
        raise ValueError("all finite insertion points must be finite")
    if len(set(finite_positions)) != len(finite_positions):
        raise ValueError("free-field insertion points must not collide")

    charges: dict[Label, complex] = {
        label: liouville_charge(state.liouville_momentum)
        + (1.0 if label in component.external_contacts else 0.0)
        for label, state in states.items()
        if label != 8
    }
    times: dict[Label, complex] = {
        label: complex(state.time_momentum)
        for label, state in states.items()
        if label != 8
    }
    charges.update({label: 1.0 + 0.0j for label in y_labels})
    charges.update({label: 2.0 + 0.0j for label in b_labels})
    times.update({label: 0.0j for label in y_labels + b_labels})
    finite_labels: tuple[Label, ...] = (1, 2, 3, *y_labels, *b_labels, 7)
    bosonic = 1.0 + 0.0j
    for index, first in enumerate(finite_labels):
        for second in finite_labels[index + 1 :]:
            exponent = -2.0 * (
                charges[first] * charges[second]
                + times[first] * times[second]
            )
            distance = abs(complex(positions[first]) - complex(positions[second]))
            bosonic *= _absolute_power(distance, exponent)

    hol_labels: tuple[Label, ...] = tuple(
        label for label in PHYSICAL_ZERO_PICTURE_LABELS
        if label not in component.external_contacts
    ) + y_labels

    def hol_contraction(first: Label, second: Label) -> complex:
        first_is_y, second_is_y = first in y_labels, second in y_labels
        if first_is_y and second_is_y:
            return 1.0 + 0.0j
        if first_is_y:
            first, second = second, first
            first_is_y, second_is_y = second_is_y, first_is_y
        if second_is_y:
            return liouville_charge(states[first].liouville_momentum)
        left, right = states[first], states[second]
        return (
            liouville_charge(left.liouville_momentum)
            * liouville_charge(right.liouville_momentum)
            + complex(left.time_momentum) * complex(right.time_momentum)
        )

    physical_anti_order: tuple[Label, ...] = tuple(
        label for label in (8, 7, 3, 2, 1)
        if label not in component.external_contacts
    )
    anti_labels = physical_anti_order + y_labels

    def anti_contraction(first: Label, second: Label) -> complex:
        first_is_y, second_is_y = first in y_labels, second in y_labels
        if first_is_y and second_is_y:
            return 1.0 + 0.0j
        if first_is_y:
            first, second = second, first
            first_is_y, second_is_y = second_is_y, first_is_y
        if second_is_y:
            state = states[first]
            return (
                liouville_charge(state.liouville_momentum)
                if state.kind == "singlet"
                else 0.0j
            )
        left, right = states[first], states[second]
        if left.kind != right.kind:
            return 0.0j
        if left.kind == "singlet":
            return (
                liouville_charge(left.liouville_momentum)
                * liouville_charge(right.liouville_momentum)
            )
        return 1.0 if left.flavor == right.flavor else 0.0j

    holomorphic = _pfaffian(hol_labels, positions, hol_contraction)
    barred_positions = {
        label: None if value is None else complex(value).conjugate()
        for label, value in positions.items()
    }
    antiholomorphic = _pfaffian(
        anti_labels, barred_positions, anti_contraction
    )
    picture_factor = 2.0 ** -3
    bare = picture_factor * bosonic * holomorphic * antiholomorphic
    coefficient = s3_component_action_coefficient(
        component, physical, mu=mu
    )
    return S3ComponentEvaluation(
        component=component,
        value=bare,
        action_coefficient=coefficient,
        weighted_value=coefficient * bare,
        bosonic_factor=bosonic,
        holomorphic_pfaffian=holomorphic,
        antiholomorphic_pfaffian=antiholomorphic,
        picture_factor=picture_factor,
    )


def evaluate_soft_eightpoint_integrand(
    family: SoftS3Family,
    z2: complex,
    z3: complex,
    w1: complex,
    w2: complex,
    w3: complex,
    *,
    include_picture_factor: bool = True,
) -> S3IntegrandEvaluation:
    """Evaluate the ordinary neutral eight-point soft regulator."""

    positions = _positions(z2, z3, w1, w2, w3)
    holomorphic_order: tuple[Label, ...] = (
        *PHYSICAL_ZERO_PICTURE_LABELS,
        *SOFT_LABELS,
    )
    antiholomorphic_order: tuple[Label, ...] = (
        8,
        7,
        3,
        2,
        1,
        *SOFT_LABELS,
    )
    holomorphic = _pfaffian(
        holomorphic_order, positions, _holomorphic_contraction(family)
    )
    barred_positions = {
        label: None if value is None else complex(value).conjugate()
        for label, value in positions.items()
    }
    antiholomorphic = _pfaffian(
        antiholomorphic_order,
        barred_positions,
        _antiholomorphic_contraction(family),
    )
    bosonic = soft_eightpoint_bosonic_factor(
        family, z2, z3, w1, w2, w3
    )
    picture_factor = 2.0 ** -6 if include_picture_factor else 1.0
    return S3IntegrandEvaluation(
        value=picture_factor * bosonic * holomorphic * antiholomorphic,
        bosonic_factor=bosonic,
        holomorphic_pfaffian=holomorphic,
        antiholomorphic_pfaffian=antiholomorphic,
        picture_factor=picture_factor,
    )


def evaluate_yyy_integrand(
    projection: FivePointChannelProjection,
    resonant_outgoing_momenta: Sequence[complex],
    z2: complex,
    z3: complex,
    w1: complex,
    w2: complex,
    w3: complex,
) -> S3IntegrandEvaluation:
    r"""Evaluate the separated three-Yukawa component at ``q_r=0``.

    The result includes only the three physical PCO magnitudes ``2**-3``.
    It equals eight times the ordinary soft-eight-point integrand.
    """

    family = soft_s3_family(
        projection, resonant_outgoing_momenta, (0.0j, 0.0j, 0.0j)
    )
    ordinary = evaluate_soft_eightpoint_integrand(
        family, z2, z3, w1, w2, w3, include_picture_factor=True
    )
    return S3IntegrandEvaluation(
        value=8.0 * ordinary.value,
        bosonic_factor=ordinary.bosonic_factor,
        holomorphic_pfaffian=ordinary.holomorphic_pfaffian,
        antiholomorphic_pfaffian=ordinary.antiholomorphic_pfaffian,
        picture_factor=2.0 ** -3,
    )


def _yb_contractions(
    physical_states: Sequence[SphereFivePointState],
) -> tuple[
    Callable[[Label, Label], complex],
    Callable[[Label, Label], complex],
]:
    out1, out2, out3, out4, incoming = tuple(physical_states)
    states: dict[Label, SphereFivePointState] = {
        1: out1,
        2: out2,
        3: out3,
        7: out4,
        8: incoming,
    }

    def hol(first: Label, second: Label) -> complex:
        if first == "y":
            first, second = second, first
        if second == "y":
            state = states[first]
            return liouville_charge(state.liouville_momentum)
        left, right = states[first], states[second]
        return (
            liouville_charge(left.liouville_momentum)
            * liouville_charge(right.liouville_momentum)
            + complex(left.time_momentum) * complex(right.time_momentum)
        )

    def anti(first: Label, second: Label) -> complex:
        if first == "y":
            first, second = second, first
        if second == "y":
            state = states[first]
            return (
                liouville_charge(state.liouville_momentum)
                if state.kind == "singlet"
                else 0.0j
            )
        left, right = states[first], states[second]
        if left.kind != right.kind:
            return 0.0j
        if left.kind == "singlet":
            return (
                liouville_charge(left.liouville_momentum)
                * liouville_charge(right.liouville_momentum)
            )
        return 1.0 if left.flavor == right.flavor else 0.0j

    return hol, anti


def evaluate_yb_integrand(
    projection: FivePointChannelProjection,
    resonant_outgoing_momenta: Sequence[complex],
    z2: complex,
    z3: complex,
    yukawa_position: complex,
    bosonic_position: complex,
) -> S3IntegrandEvaluation:
    r"""Evaluate the separated ``YB`` component at the ``s=3`` wall.

    ``B`` has exponential charge two and no fermions.  The result includes
    the three physical PCO magnitudes ``2**-3``.
    """

    family = soft_s3_family(
        projection, resonant_outgoing_momenta, (0.0j, 0.0j, 0.0j)
    )
    positions: dict[Label, complex | None] = {
        1: 0.0j,
        2: complex(z2),
        3: complex(z3),
        "y": complex(yukawa_position),
        "b": complex(bosonic_position),
        7: 1.0 + 0.0j,
        8: None,
    }
    finite = tuple(value for value in positions.values() if value is not None)
    if len(set(finite)) != len(finite):
        raise ValueError("free-field insertion points must not collide")
    physical = family.physical_states
    out1, out2, out3, out4, _incoming = physical
    physical_by_label = {1: out1, 2: out2, 3: out3, 7: out4}
    charges: dict[Label, complex] = {
        label: liouville_charge(state.liouville_momentum)
        for label, state in physical_by_label.items()
    }
    times: dict[Label, complex] = {
        label: complex(state.time_momentum)
        for label, state in physical_by_label.items()
    }
    charges.update({"y": 1.0 + 0.0j, "b": 2.0 + 0.0j})
    times.update({"y": 0.0j, "b": 0.0j})
    finite_labels: tuple[Label, ...] = (1, 2, 3, "y", "b", 7)
    bosonic = 1.0 + 0.0j
    for index, first in enumerate(finite_labels):
        for second in finite_labels[index + 1 :]:
            exponent = -2.0 * (
                charges[first] * charges[second]
                + times[first] * times[second]
            )
            distance = abs(complex(positions[first]) - complex(positions[second]))
            bosonic *= _absolute_power(distance, exponent)

    hol_contraction, anti_contraction = _yb_contractions(physical)
    holomorphic = _pfaffian(
        (*PHYSICAL_ZERO_PICTURE_LABELS, "y"),
        positions,
        hol_contraction,
    )
    barred_positions = {
        label: None if value is None else complex(value).conjugate()
        for label, value in positions.items()
    }
    antiholomorphic = _pfaffian(
        (8, 7, 3, 2, 1, "y"),
        barred_positions,
        anti_contraction,
    )
    picture_factor = 2.0 ** -3
    return S3IntegrandEvaluation(
        value=picture_factor * bosonic * holomorphic * antiholomorphic,
        bosonic_factor=bosonic,
        holomorphic_pfaffian=holomorphic,
        antiholomorphic_pfaffian=antiholomorphic,
        picture_factor=picture_factor,
    )


def yyy_action_coefficient(mu: complex = 1.0) -> complex:
    """Return ``(-2 i mu)^3/3!`` from the component action expansion."""

    return (-2j * complex(mu)) ** 3 / math.factorial(3)


def yb_action_coefficient(mu: complex = 1.0) -> complex:
    """Return ``(-2 i mu)(-2 pi mu^2)`` from the component action."""

    value = complex(mu)
    return (-2j * value) * (-2.0 * math.pi * value**2)


__all__ = [
    "PHYSICAL_ZERO_PICTURE_LABELS",
    "S3Component",
    "S3ComponentEvaluation",
    "S3IntegrandEvaluation",
    "S3_RESONANCE_MOMENTUM",
    "SOFT_LABELS",
    "SoftS3Family",
    "evaluate_soft_eightpoint_integrand",
    "evaluate_s3_component_integrand",
    "evaluate_yb_integrand",
    "evaluate_yyy_integrand",
    "liouville_charge",
    "soft_eightpoint_bosonic_factor",
    "soft_s3_family",
    "s3_component_action_coefficient",
    "s3_components",
    "yb_action_coefficient",
    "yyy_action_coefficient",
]
