#!/usr/bin/env python3
r"""NS continuum four-point data for the two-dimensional SO(7) x E8 string.

The interacting Liouville wall distinguishes its antiholomorphic
super-Liouville fermion from the seven spectator Majorana fermions.  Thus the
exact NS continuum is

``V^a``
    an SO(7) vector, ``a=1,...,7``, made with one spectator fermion; and
``S``
    an SO(7) x E8 singlet made with the super-Liouville descendant
    ``G_{-1/2} V_omega``.

Both vertices are tensored with the E8 level-one vacuum.  The E8 currents
have weight one and give zero-momentum discrete gauge states, not another
generic-energy NS continuum particle.

For fixed external indices, all spectator-fermion Wick contractions and all
super-Liouville blocks are identical to the corresponding SO(23) calculation.
Only the allowed vector-index range changes from 23 to 7.  This module records
that reduction and exposes the closed-form candidates in the raw-descendant
normalization of :mod:`spin23_genuine_formulas`.

The omitted factors are the common sphere normalization, heterotic coupling,
energy-conservation delta function, and asymptotic reflection phases.  The
closed forms remain candidates wherever their SO(23) parents are candidates;
the equality of the *worldsheet integral representations* is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from numbers import Number
from typing import Literal, Sequence

from spin23_genuine_formulas import (
    s_to_sss_raw_candidate as _s_to_sss_raw_candidate,
    s_to_sss_unit_descendants_candidate as _s_to_sss_unit_candidate,
    s_to_svv_raw_candidate as _s_to_svv_raw_candidate,
    s_to_svv_unit_descendants_candidate as _s_to_svv_unit_candidate,
    singlet_descendant_norm,
    v_to_vss_raw_candidate as _v_to_vss_raw_candidate,
    v_to_vss_unit_descendants_candidate as _v_to_vss_unit_candidate,
)


SO7_VECTOR_DIMENSION = 7
SO7_ADJOINT_DIMENSION = 21
E8_ADJOINT_DIMENSION = 248

Particle = Literal["S", "V"]


@dataclass(frozen=True)
class NSState:
    """One physical NS state family in the exact wall theory."""

    name: str
    so7_representation: str
    e8_representation: str
    propagating: bool
    generic_energy: bool
    matter_insertion: str


NS_CONTINUUM_STATES: tuple[NSState, ...] = (
    NSState(
        name="V",
        so7_representation="7",
        e8_representation="1",
        propagating=True,
        generic_energy=True,
        matter_insertion="lambda^a V_omega",
    ),
    NSState(
        name="S",
        so7_representation="1",
        e8_representation="1",
        propagating=True,
        generic_energy=True,
        matter_insertion="G_{-1/2} V_omega",
    ),
)


NS_DISCRETE_STATES: tuple[NSState, ...] = (
    NSState(
        name="A_SO7",
        so7_representation="21",
        e8_representation="1",
        propagating=False,
        generic_energy=False,
        matter_insertion="J lambda^a lambda^b",
    ),
    NSState(
        name="A_E8",
        so7_representation="1",
        e8_representation="248",
        propagating=False,
        generic_energy=False,
        matter_insertion="J J_E8^A",
    ),
    NSState(
        name="G",
        so7_representation="1",
        e8_representation="1",
        propagating=False,
        generic_energy=False,
        matter_insertion="J Jbar",
    ),
)


@dataclass(frozen=True)
class FourPointProcess:
    """One independent fixed-incoming generic-energy NS process."""

    incoming: Particle
    outgoing: tuple[Particle, Particle, Particle]
    tensor_structures: tuple[str, ...]
    external_sld_pattern: str


FOUR_POINT_PROCESSES: tuple[FourPointProcess, ...] = (
    FourPointProcess("S", ("S", "S", "S"), ("1",), "A"),
    FourPointProcess("S", ("S", "V", "V"), ("delta(a2,a3)",), "O"),
    FourPointProcess("V", ("V", "S", "S"), ("delta(a0,a1)",), "M"),
    FourPointProcess(
        "V",
        ("V", "V", "V"),
        (
            "delta(a0,a3) delta(a1,a2)",
            "delta(a0,a2) delta(a1,a3)",
            "delta(a0,a1) delta(a2,a3)",
        ),
        "P",
    ),
)


def allowed_ns_assignments() -> tuple[tuple[Particle, ...], ...]:
    """Return the eight nonzero S/V assignments in leg order ``(0,1,2,3)``.

    SO(7) has no invariant tensor with one or three vector indices.  At four
    points the rank-seven epsilon tensor cannot contribute, so precisely the
    assignments with an even number of vectors survive.
    """

    assignments: list[tuple[Particle, ...]] = []
    for values in product(("S", "V"), repeat=4):
        assignment = tuple(values)
        if assignment.count("V") % 2 == 0:
            assignments.append(assignment)  # type: ignore[arg-type]
    return tuple(assignments)


def s_to_sss_raw_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> complex:
    """Raw reduced ``S -> S S S`` candidate."""

    return _s_to_sss_raw_candidate(omega1, omega2, omega3)


def s_to_svv_raw_candidate(
    omega_singlet: complex,
    omega_vector1: complex,
    omega_vector2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Raw reduced ``S -> S V^a V^b`` candidate."""

    return _s_to_svv_raw_candidate(
        omega_singlet,
        omega_vector1,
        omega_vector2,
        delta_ab=delta_ab,
    )


def v_to_vss_raw_candidate(
    omega_vector: complex,
    omega_singlet1: complex,
    omega_singlet2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Raw reduced fixed-incoming ``V^a -> V^b S S`` candidate."""

    return _v_to_vss_raw_candidate(
        omega_vector,
        omega_singlet1,
        omega_singlet2,
        delta_ab=delta_ab,
    )


@dataclass(frozen=True)
class FourVectorCoefficients:
    r"""Coefficients in the ordered SO(7)-invariant four-vector basis.

    The full tensor is

    ``A delta(a0,a3)delta(a1,a2)``
    ``+ B delta(a0,a2)delta(a1,a3)``
    ``+ C delta(a0,a1)delta(a2,a3)``.
    """

    A: complex
    B: complex
    C: complex

    def as_tuple(self) -> tuple[complex, complex, complex]:
        return self.A, self.B, self.C


def v_to_vvv_raw_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> FourVectorCoefficients:
    r"""Return the reduced ``V -> V V V`` tensor coefficients.

    With ``omega0=omega1+omega2+omega3`` and
    ``W=omega0*omega1*omega2*omega3``, the phase convention inherited from
    the SO(23) block calculation is

    ``(A,B,C) = -pi*W * (1/d12, 1/d13, 1/d23)``,

    where ``dij=1+i*(omegai+omegaj)``.
    """

    omega1, omega2, omega3 = map(complex, (omega1, omega2, omega3))
    omega0 = omega1 + omega2 + omega3
    common = -math.pi * omega0 * omega1 * omega2 * omega3
    denominators = (
        1 + 1j * (omega1 + omega2),
        1 + 1j * (omega1 + omega3),
        1 + 1j * (omega2 + omega3),
    )
    if any(value == 0 for value in denominators):
        raise ZeroDivisionError("the candidate lies exactly on a pair-channel pole")
    return FourVectorCoefficients(
        *(common / value for value in denominators)
    )


def contract_v_to_vvv_raw_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
    *,
    indices: Sequence[int],
) -> complex:
    """Contract the four-vector candidate for indices ``(a0,a1,a2,a3)``."""

    if len(indices) != 4:
        raise ValueError("indices must contain (a0,a1,a2,a3)")
    if any(index < 0 or index >= SO7_VECTOR_DIMENSION for index in indices):
        raise ValueError("SO(7) vector indices must lie in range(7)")
    a0, a1, a2, a3 = indices
    coefficients = v_to_vvv_raw_candidate(omega1, omega2, omega3)
    return (
        coefficients.A * (a0 == a3) * (a1 == a2)
        + coefficients.B * (a0 == a2) * (a1 == a3)
        + coefficients.C * (a0 == a1) * (a2 == a3)
    )


def s_to_sss_unit_descendants_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> complex:
    """Unit-descendant version of :func:`s_to_sss_raw_candidate`."""

    return _s_to_sss_unit_candidate(omega1, omega2, omega3)


def s_to_svv_unit_descendants_candidate(
    omega_singlet: complex,
    omega_vector1: complex,
    omega_vector2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Unit-descendant version of :func:`s_to_svv_raw_candidate`."""

    return _s_to_svv_unit_candidate(
        omega_singlet,
        omega_vector1,
        omega_vector2,
        delta_ab=delta_ab,
    )


def v_to_vss_unit_descendants_candidate(
    omega_vector: complex,
    omega_singlet1: complex,
    omega_singlet2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Unit-descendant version of :func:`v_to_vss_raw_candidate`."""

    return _v_to_vss_unit_candidate(
        omega_vector,
        omega_singlet1,
        omega_singlet2,
        delta_ab=delta_ab,
    )


__all__ = [
    "E8_ADJOINT_DIMENSION",
    "FOUR_POINT_PROCESSES",
    "FourPointProcess",
    "FourVectorCoefficients",
    "NS_CONTINUUM_STATES",
    "NS_DISCRETE_STATES",
    "NSState",
    "SO7_ADJOINT_DIMENSION",
    "SO7_VECTOR_DIMENSION",
    "allowed_ns_assignments",
    "contract_v_to_vvv_raw_candidate",
    "s_to_sss_raw_candidate",
    "s_to_sss_unit_descendants_candidate",
    "s_to_svv_raw_candidate",
    "s_to_svv_unit_descendants_candidate",
    "singlet_descendant_norm",
    "v_to_vss_raw_candidate",
    "v_to_vss_unit_descendants_candidate",
    "v_to_vvv_raw_candidate",
]
