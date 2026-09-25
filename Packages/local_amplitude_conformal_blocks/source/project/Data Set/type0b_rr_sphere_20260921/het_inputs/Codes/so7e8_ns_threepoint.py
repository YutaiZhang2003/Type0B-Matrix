#!/usr/bin/env python3
r"""Exact generic-energy NS three-point amplitudes for SO(7) x E8.

The super-Liouville and picture-changing calculation is identical to the
SO(23) calculation at fixed spectator indices.  This module gives it an
SO(7)-specific process inventory and includes the fixed-incoming
``V -> V S`` crossing explicitly.

All functions omit the common sphere normalization, ``g_H**3``, the energy
delta function, and external reflection phases.  Singlets are raw
``G_-1/2 V_p`` descendants unless the function name contains ``unit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Literal

from spin23_three_point_amplitudes import (
    ns_weight,
    odd_structure_constant_on_shell,
    s_to_ss_raw as _s_to_ss_raw,
    s_to_ss_unit_descendants as _s_to_ss_unit,
    s_to_vv_raw as _s_to_vv_raw,
    s_to_vv_unit_descendants as _s_to_vv_unit,
    singlet_descendant_norm,
)


Particle = Literal["S", "V"]


@dataclass(frozen=True)
class NSOneToTwoProcess:
    incoming: Particle
    outgoing: tuple[Particle, Particle]
    tensor: str


NS_ONE_TO_TWO_PROCESSES: tuple[NSOneToTwoProcess, ...] = (
    NSOneToTwoProcess("S", ("S", "S"), "1"),
    NSOneToTwoProcess("S", ("V", "V"), "delta^{ab}"),
    NSOneToTwoProcess("V", ("V", "S"), "delta^{ab}"),
)


def s_to_ss_raw(p1: complex, p2: complex) -> complex:
    """Return the exact raw ``S -> S S`` reduced amplitude."""

    return _s_to_ss_raw(p1, p2)


def s_to_vv_raw(
    p1: complex,
    p2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Return the exact raw ``S -> V^a V^b`` reduced amplitude."""

    return _s_to_vv_raw(p1, p2, delta_ab=delta_ab)


def v_to_vs_raw(
    p_vector: complex,
    p_singlet: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    r"""Return the exact raw ``V^a -> V^b S`` crossing.

    With ``p0=p_vector+p_singlet`` the result is
    ``delta_ab*p0*p_vector*p_singlet``.
    """

    return _s_to_vv_raw(p_vector, p_singlet, delta_ab=delta_ab)


def s_to_ss_unit_descendants(p1: complex, p2: complex) -> complex:
    """Return ``S -> S S`` with all three singlets unit normalized."""

    return _s_to_ss_unit(p1, p2)


def s_to_vv_unit_descendants(
    p1: complex,
    p2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Return ``S -> V V`` with the incoming singlet unit normalized."""

    return _s_to_vv_unit(p1, p2, delta_ab=delta_ab)


def v_to_vs_unit_descendants(
    p_vector: complex,
    p_singlet: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Return ``V -> V S`` with the outgoing singlet unit normalized."""

    return v_to_vs_raw(
        p_vector,
        p_singlet,
        delta_ab=delta_ab,
    ) / singlet_descendant_norm(p_singlet)


__all__ = [
    "NS_ONE_TO_TWO_PROCESSES",
    "NSOneToTwoProcess",
    "ns_weight",
    "odd_structure_constant_on_shell",
    "s_to_ss_raw",
    "s_to_ss_unit_descendants",
    "s_to_vv_raw",
    "s_to_vv_unit_descendants",
    "singlet_descendant_norm",
    "v_to_vs_raw",
    "v_to_vs_unit_descendants",
]
