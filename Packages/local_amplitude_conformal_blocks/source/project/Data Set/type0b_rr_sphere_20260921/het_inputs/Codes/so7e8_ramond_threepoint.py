#!/usr/bin/env python3
r"""Tree-level two-Ramond/one-NS data in the SO(7) x E8 theory.

The exact Liouville-wall theory has two GSO-resolved Ramond continuum
families.  Both transform as the eight-dimensional spinor of
``so(7)``; the auxiliary ``S8`` and ``C8`` labels below only remember their
diagonal-GSO origin.  There are two distinct physical crossing loci:

``R(p_in) -> R(p_out) + B(p_b)``
    has ``p_in=p_out+p_b`` and uses ``C_odd=p_b/2``;

``B(p_b) -> R(p_1) + R(p_2)``
    has ``p_b=p_1+p_2`` and uses ``C_even=p_b/2``.

The second locus is essential for four-point factorization with a bosonic
incoming leg.  It must not be discarded by imposing the auxiliary
``Spin(8)`` fusion labels as an exact symmetry of the Liouville-wall theory.

Use the ordered three-punctured sphere

``(R(p_in) at infinity, NS(p_boson) at 1, R(p_out) at 0)``.

For the physical positive-energy branch,

``p_in = p_out + p_boson``.

In this ordering the relevant delta-normalized super-Liouville coefficient
is ``C_odd(p_in, p_out; p_boson)``.  At ``b=1`` it obeys the exact resonance
identity

``C_odd(p_out + p_boson, p_out; p_boson) = p_boson / 2``.

The vector coefficient also contains the conventional spin-field OPE factor
``1/sqrt(2)``.  For the singlet, the external matter state is
``G_{-1/2} V_p``.  The HJS middle-slot Ward identities are

``rho_-(w+, G_{-1/2} V, w+) = -(1-i)(p_in+p_out)/2``,

``rho_+(w+, G_{-1/2} V, w+) = +(1-i)(p_1-p_2)/2``.

Each formula below includes ``1/2`` from the two normalized GSO-resolved
Ramond vertices.  It omits the common sphere normalization, ``g_H**3``, the
time-momentum delta function, and external reflection phases.  The Ramond
primaries and the NS primary are delta normalized.  No target-space spinor
wavefunction or LSZ convention is silently included.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from numbers import Number
from typing import Literal

from spin23_super_liouville_data import rr_ns_structure_constants


SO7_SPINOR_DIMENSION = 8
E8_REPRESENTATION = "1"

RamondClass = Literal["S8", "C8"]
Particle = Literal["S", "V", "Psi", "Psi_tilde"]


@dataclass(frozen=True)
class RamondContinuumState:
    """One exact Ramond family, with its auxiliary GSO label retained."""

    name: str
    auxiliary_gso_class: RamondClass
    so7_representation: str
    e8_representation: str
    allowed_time_momentum_sign: int


RAMOND_CONTINUUM_STATES: tuple[RamondContinuumState, ...] = (
    RamondContinuumState("Psi", "S8", "8", E8_REPRESENTATION, +1),
    RamondContinuumState("Psi_tilde", "C8", "8", E8_REPRESENTATION, -1),
)


@dataclass(frozen=True)
class RamondOneToTwoProcess:
    """One nonzero fixed-positive-energy process containing two R legs."""

    incoming: Particle
    outgoing: tuple[Particle, Particle]
    tensor: str


RAMOND_ONE_TO_TWO_PROCESSES: tuple[RamondOneToTwoProcess, ...] = (
    RamondOneToTwoProcess("Psi", ("Psi_tilde", "V"), "C gamma^a"),
    RamondOneToTwoProcess("Psi", ("Psi_tilde", "S"), "C"),
    RamondOneToTwoProcess("V", ("Psi_tilde", "Psi_tilde"), "C gamma^a"),
    RamondOneToTwoProcess("S", ("Psi_tilde", "Psi_tilde"), "C"),
)


GSO_PAIR_FACTOR = 0.5
SPIN7_VECTOR_OPE_FACTOR = 1.0 / math.sqrt(2.0)


def ramond_pair_allowed(first: RamondClass, second: RamondClass) -> bool:
    r"""Return whether an exact-wall RRNS coupling is kinematically allowed.

    Both equal and unequal auxiliary classes occur.  Equal classes select the
    HJS-even structure on the boson-incoming locus; unequal classes select the
    HJS-odd structure on the Ramond-incoming locus.  The auxiliary labels are
    useful bookkeeping, not an exact unbroken Spin(8) selection rule.
    """

    if first not in ("S8", "C8") or second not in ("S8", "C8"):
        raise ValueError("Ramond classes must be 'S8' or 'C8'")
    return True


def ramond_pair_structure(first: RamondClass, second: RamondClass) -> Literal[
    "even", "odd"
]:
    """Return the RRNS structure selected by the two auxiliary classes."""

    if first not in ("S8", "C8") or second not in ("S8", "C8"):
        raise ValueError("Ramond classes must be 'S8' or 'C8'")
    return "even" if first == second else "odd"


def rr_ns_even_coefficient(
    p_one: complex,
    p_two: complex,
    p_boson: complex,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the exact ``C_even(p_one,p_two;p_boson)`` coefficient."""

    even, _ = rr_ns_structure_constants(
        p_one,
        p_two,
        p_boson,
        precision=precision,
    )
    return even


def rr_ns_odd_coefficient(
    p_in: complex,
    p_out: complex,
    p_boson: complex,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the exact ``C_odd(p_in,p_out;p_boson)`` coefficient."""

    _, odd = rr_ns_structure_constants(
        p_in,
        p_out,
        p_boson,
        precision=precision,
    )
    return odd


def rr_ns_odd_resonance_value(p_boson: complex) -> complex:
    r"""Return ``C_odd`` on ``p_in=p_out+p_boson``, namely ``p_boson/2``."""

    return complex(p_boson) / 2.0


def rr_ns_even_boson_in_resonance_value(p_boson: complex) -> complex:
    r"""Return ``C_even`` on ``p_boson=p_one+p_two``: ``p_boson/2``."""

    return complex(p_boson) / 2.0


def hjs_minus_singlet_ward_factor(
    p_in: complex,
    p_out: complex,
) -> complex:
    r"""Return the ordered HJS-minus form for ``G_-1/2 V``.

    Both Ramond ground labels are ``w+``.  Other ground-label choices differ
    only by the documented HJS phases.
    """

    return -(1.0 - 1.0j) * (complex(p_in) + complex(p_out)) / 2.0


def hjs_plus_singlet_ward_factor(
    p_one: complex,
    p_two: complex,
) -> complex:
    r"""Return the HJS-plus middle-descendant factor for two R legs."""

    return (1.0 - 1.0j) * (complex(p_one) - complex(p_two)) / 2.0


def v_psi_psitilde_raw(
    p_in: complex,
    p_out: complex,
    p_vector: complex,
    *,
    gamma_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return the reduced exact ``V^a Psi Psi_tilde`` coefficient.

    ``gamma_component`` is the selected component of
    ``(C gamma^a)_{alpha beta}``.
    """

    coefficient = rr_ns_odd_coefficient(
        p_in,
        p_out,
        p_vector,
        precision=precision,
    )
    return (
        complex(gamma_component)
        * GSO_PAIR_FACTOR
        * SPIN7_VECTOR_OPE_FACTOR
        * coefficient
    )


def s_psi_psitilde_raw_hjs(
    p_in: complex,
    p_out: complex,
    p_singlet: complex,
    *,
    charge_conjugation_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return the reduced ``S Psi Psi_tilde`` coefficient in HJS phase.

    ``charge_conjugation_component`` is the selected component of the
    SO(7) spinor metric ``C_{alpha beta}``.
    """

    coefficient = rr_ns_odd_coefficient(
        p_in,
        p_out,
        p_singlet,
        precision=precision,
    )
    return (
        complex(charge_conjugation_component)
        * GSO_PAIR_FACTOR
        * hjs_minus_singlet_ward_factor(p_in, p_out)
        * coefficient
    )


def strip_hjs_singlet_phase(value: complex) -> complex:
    r"""Remove the conventional HJS singlet phase ``-exp(-i*pi/4)``."""

    return -cmath.exp(0.25j * math.pi) * complex(value)


def s_psi_psitilde_raw(
    p_in: complex,
    p_out: complex,
    p_singlet: complex,
    *,
    charge_conjugation_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return ``S Psi Psi_tilde`` in the phase-stripped real convention."""

    return strip_hjs_singlet_phase(
        s_psi_psitilde_raw_hjs(
            p_in,
            p_out,
            p_singlet,
            charge_conjugation_component=charge_conjugation_component,
            precision=precision,
        )
    )


def v_psi_psitilde_on_shell(
    p_out: complex,
    p_vector: complex,
    *,
    gamma_component: Number = 1,
) -> complex:
    r"""Return the closed on-shell vector coefficient.

    The result is ``gamma_component*p_vector/(4*sqrt(2))``.
    """

    return (
        complex(gamma_component)
        * complex(p_vector)
        / (4.0 * math.sqrt(2.0))
    )


def s_psi_psitilde_on_shell(
    p_out: complex,
    p_singlet: complex,
    *,
    charge_conjugation_component: Number = 1,
) -> complex:
    r"""Return the closed phase-stripped on-shell singlet coefficient.

    With ``p_in=p_out+p_singlet``, the result is

    ``C_ab*(p_in+p_out)*p_singlet/(4*sqrt(2))``.
    """

    p_out = complex(p_out)
    p_singlet = complex(p_singlet)
    p_in = p_out + p_singlet
    return (
        complex(charge_conjugation_component)
        * (p_in + p_out)
        * p_singlet
        / (4.0 * math.sqrt(2.0))
    )


def singlet_descendant_norm(momentum: complex) -> complex:
    r"""Return ``sqrt(1+p**2)`` for ``G_-1/2 V_p``."""

    return cmath.sqrt(1.0 + complex(momentum) ** 2)


def s_psi_psitilde_unit_singlet_on_shell(
    p_out: complex,
    p_singlet: complex,
    *,
    charge_conjugation_component: Number = 1,
) -> complex:
    r"""Return the on-shell result with the external singlet normalized."""

    return s_psi_psitilde_on_shell(
        p_out,
        p_singlet,
        charge_conjugation_component=charge_conjugation_component,
    ) / singlet_descendant_norm(p_singlet)


def v_to_psipsi_raw(
    p_one: complex,
    p_two: complex,
    *,
    gamma_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return the reduced exact ``V -> Psi Psi`` coefficient.

    This is the boson-incoming, equal-GSO-family crossing.  The incoming
    vector momentum is ``p_vector=p_one+p_two``.
    """

    p_vector = complex(p_one) + complex(p_two)
    coefficient = rr_ns_even_coefficient(
        p_one,
        p_two,
        p_vector,
        precision=precision,
    )
    return (
        complex(gamma_component)
        * GSO_PAIR_FACTOR
        * SPIN7_VECTOR_OPE_FACTOR
        * coefficient
    )


def v_to_psipsi_on_shell(
    p_one: complex,
    p_two: complex,
    *,
    gamma_component: Number = 1,
) -> complex:
    r"""Return ``V -> Psi Psi`` as ``(p_one+p_two)/(4*sqrt(2))``."""

    return (
        complex(gamma_component)
        * (complex(p_one) + complex(p_two))
        / (4.0 * math.sqrt(2.0))
    )


def strip_hjs_plus_singlet_phase(value: complex) -> complex:
    r"""Remove the HJS-plus descendant phase ``exp(-i*pi/4)``."""

    return cmath.exp(0.25j * math.pi) * complex(value)


def s_to_psipsi_raw_hjs(
    p_one: complex,
    p_two: complex,
    *,
    charge_conjugation_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return boson-incoming ``S -> Psi Psi`` in the HJS phase."""

    p_singlet = complex(p_one) + complex(p_two)
    coefficient = rr_ns_even_coefficient(
        p_one,
        p_two,
        p_singlet,
        precision=precision,
    )
    return (
        complex(charge_conjugation_component)
        * GSO_PAIR_FACTOR
        * hjs_plus_singlet_ward_factor(p_one, p_two)
        * coefficient
    )


def s_to_psipsi_raw(
    p_one: complex,
    p_two: complex,
    *,
    charge_conjugation_component: Number = 1,
    precision: int = 40,
) -> complex:
    r"""Return phase-stripped boson-incoming ``S -> Psi Psi``."""

    return strip_hjs_plus_singlet_phase(
        s_to_psipsi_raw_hjs(
            p_one,
            p_two,
            charge_conjugation_component=charge_conjugation_component,
            precision=precision,
        )
    )


def s_to_psipsi_on_shell(
    p_one: complex,
    p_two: complex,
    *,
    charge_conjugation_component: Number = 1,
) -> complex:
    r"""Return ``S -> Psi Psi`` in the phase-stripped real convention.

    The coefficient is antisymmetric under exchange of the two identical
    outgoing fermions:

    ``C_ab*(p_one-p_two)*(p_one+p_two)/(4*sqrt(2))``.
    """

    p_one = complex(p_one)
    p_two = complex(p_two)
    return (
        complex(charge_conjugation_component)
        * (p_one - p_two)
        * (p_one + p_two)
        / (4.0 * math.sqrt(2.0))
    )


__all__ = [
    "E8_REPRESENTATION",
    "GSO_PAIR_FACTOR",
    "RAMOND_CONTINUUM_STATES",
    "RAMOND_ONE_TO_TWO_PROCESSES",
    "SO7_SPINOR_DIMENSION",
    "SPIN7_VECTOR_OPE_FACTOR",
    "RamondContinuumState",
    "RamondOneToTwoProcess",
    "hjs_plus_singlet_ward_factor",
    "hjs_minus_singlet_ward_factor",
    "ramond_pair_allowed",
    "ramond_pair_structure",
    "rr_ns_even_boson_in_resonance_value",
    "rr_ns_even_coefficient",
    "rr_ns_odd_coefficient",
    "rr_ns_odd_resonance_value",
    "s_to_psipsi_on_shell",
    "s_to_psipsi_raw",
    "s_to_psipsi_raw_hjs",
    "s_psi_psitilde_on_shell",
    "s_psi_psitilde_raw",
    "s_psi_psitilde_raw_hjs",
    "s_psi_psitilde_unit_singlet_on_shell",
    "singlet_descendant_norm",
    "strip_hjs_singlet_phase",
    "strip_hjs_plus_singlet_phase",
    "v_to_psipsi_on_shell",
    "v_to_psipsi_raw",
    "v_psi_psitilde_on_shell",
    "v_psi_psitilde_raw",
]
