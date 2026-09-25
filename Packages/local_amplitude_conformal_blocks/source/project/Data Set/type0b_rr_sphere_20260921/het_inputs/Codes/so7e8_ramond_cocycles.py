#!/usr/bin/env python3
r"""One explicit cocycle convention for SO(7) x E8 Ramond amplitudes.

This module supplies only the finite-dimensional free-field change of basis
which sits outside the super-Liouville conformal blocks.  It does not perform
the internal-momentum or modulus integrals.

Let ``a=0,1`` label the HJS ``(w+,w-)`` ground state, ``r=0,1`` the
time-Ising ``(sigma0,mu0)`` component, and ``b=0,1`` the antiholomorphic
super-Liouville Ising component.  The physical vertices are represented by

``Psi``
    ``(w+ wbar+) sigma0 + i (w- wbar+) mu0``,

``Psi_tilde``
    ``(w+ wbar-) sigma0 - i (w- wbar-) mu0``,

with an overall factor ``1/sqrt(2)``.  The Spin(8) ``8c`` index is identified
with the SO(7) spinor by the eighth gamma matrix; its phase is chosen so that
the normalized ``V Psi Psi_tilde`` spectator coefficient is ``+1/sqrt(2)``.

Only convention phases are conjugated at the infinity slot.  Analytically
continued momenta are never conjugated.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Literal


RamondFamily = Literal["Psi", "Psi_tilde"]
PCOTimeBranch = Literal["G", "psi0"]


@dataclass(frozen=True)
class RamondFreeFieldComponent:
    """One term in a physical Ramond vertex."""

    hjs_ground_parity: int
    time_ising_parity: int
    antiholomorphic_ising_parity: int
    coefficient: complex


@dataclass(frozen=True)
class TwoRamondComponentWeight:
    """One external-ground component of a two-Ramond correlator."""

    zero_hjs_ground_parity: int
    infinity_hjs_ground_parity: int
    zero_antiholomorphic_ground_parity: int
    infinity_antiholomorphic_ground_parity: int
    coefficient: complex


def _family_sign(family: RamondFamily) -> int:
    if family == "Psi":
        return 1
    if family == "Psi_tilde":
        return -1
    raise ValueError("family must be 'Psi' or 'Psi_tilde'")


def physical_ramond_components(
    family: RamondFamily,
) -> tuple[RamondFreeFieldComponent, RamondFreeFieldComponent]:
    r"""Return the two terms of one normalized physical Ramond vertex.

    Equivalently, for ``eta=+1`` (Psi) or ``eta=-1`` (Psi_tilde),

    .. math::

       U_{\eta;arb}=2^{-1/2}\delta_{r,a}
       \delta_{b,(1-\eta)/2}(1,i\eta)_a.
    """

    eta = _family_sign(family)
    anti = (1 - eta) // 2
    scale = 1.0 / math.sqrt(2.0)
    return (
        RamondFreeFieldComponent(0, 0, anti, scale),
        RamondFreeFieldComponent(1, 1, anti, 1.0j * eta * scale),
    )


def time_ising_two_point(infinity_parity: int, zero_parity: int) -> complex:
    """Return the normalized time-Ising two-spin metric."""

    if infinity_parity not in (0, 1) or zero_parity not in (0, 1):
        raise ValueError("time-Ising parities must be zero or one")
    return 1.0 + 0.0j if infinity_parity == zero_parity else 0.0j


def time_ising_psi0_three_point(
    infinity_parity: int,
    zero_parity: int,
    z: complex,
) -> complex:
    r"""Return ``<tau_inf(infinity) psi0(z) tau_zero(0)>``.

    In the ``(sigma0,mu0)`` basis the matrix is

    .. math::

       {z^{-1/2}\over\sqrt2}
       \begin{pmatrix}0&e^{-i\pi/4}\\e^{i\pi/4}&0\end{pmatrix}.

    The principal logarithm fixes the displayed local-coordinate branch.
    Other branches must be obtained by continuing the complete correlator.
    """

    if infinity_parity not in (0, 1) or zero_parity not in (0, 1):
        raise ValueError("time-Ising parities must be zero or one")
    z_value = complex(z)
    if z_value == 0:
        raise ValueError("the time-Ising three-point function is singular at z=0")
    if infinity_parity == zero_parity:
        return 0.0j
    root = cmath.exp(-0.5 * cmath.log(z_value)) / math.sqrt(2.0)
    phase = cmath.exp(-0.25j * math.pi)
    return root * (phase if (infinity_parity, zero_parity) == (0, 1) else 1 / phase)


def two_ramond_component_weights(
    zero_family: RamondFamily,
    infinity_family: RamondFamily,
    *,
    branch: PCOTimeBranch,
    z: complex,
) -> tuple[TwoRamondComponentWeight, ...]:
    r"""Return the physical external-ground weights for one PCO branch.

    ``branch='G'`` contains no explicit time fermion and uses the normalized
    time-Ising two-point function.  ``branch='psi0'`` includes its full
    ``z^-1/2`` three-point function.  The infinity coefficient is conjugated
    because it is a BPZ bra; momenta are not involved in this operation.
    """

    if branch not in ("G", "psi0"):
        raise ValueError("branch must be 'G' or 'psi0'")
    zero_components = physical_ramond_components(zero_family)
    infinity_components = physical_ramond_components(infinity_family)
    result: list[TwoRamondComponentWeight] = []
    for infinity in infinity_components:
        for zero in zero_components:
            if branch == "G":
                time_factor = time_ising_two_point(
                    infinity.time_ising_parity, zero.time_ising_parity
                )
            else:
                time_factor = time_ising_psi0_three_point(
                    infinity.time_ising_parity,
                    zero.time_ising_parity,
                    z,
                )
            coefficient = infinity.coefficient.conjugate() * zero.coefficient * time_factor
            if coefficient == 0:
                continue
            result.append(
                TwoRamondComponentWeight(
                    zero_hjs_ground_parity=zero.hjs_ground_parity,
                    infinity_hjs_ground_parity=infinity.hjs_ground_parity,
                    zero_antiholomorphic_ground_parity=(
                        zero.antiholomorphic_ising_parity
                    ),
                    infinity_antiholomorphic_ground_parity=(
                        infinity.antiholomorphic_ising_parity
                    ),
                    coefficient=coefficient,
                )
            )
    return tuple(result)


def hjs_ground_entry(
    structure_sign: int,
    infinity_parity: int,
    zero_parity: int,
    *,
    antiholomorphic: bool = False,
) -> complex:
    """Return one normalized holomorphic or antiholomorphic HJS entry."""

    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")
    if infinity_parity not in (0, 1) or zero_parity not in (0, 1):
        raise ValueError("HJS parities must be zero or one")
    sign = complex(structure_sign)
    if antiholomorphic:
        matrix = ((1.0 + 0.0j, 1.0 + 0.0j), (-1.0j * sign, sign))
    else:
        matrix = ((1.0 + 0.0j, 1.0 + 0.0j), (1.0j * sign, sign))
    return matrix[infinity_parity][zero_parity]


def rrns_primary_projection_coefficient(
    infinity_family: RamondFamily,
    zero_family: RamondFamily,
    structure_sign: int,
) -> complex:
    r"""Contract a primary RRNS ground tensor with two physical vertices.

    This is the calibration used by the certified ``V Psi Psi_tilde``
    amplitude.  ``Psi`` at infinity and ``Psi_tilde`` at zero gives one for
    the HJS-minus structure and zero for HJS-plus.  Equal families select
    HJS-plus.  Reversing the two unequal families carries the ordered spinor
    phase ``+i`` in this convention.
    """

    total = 0.0j
    for row in two_ramond_component_weights(
        zero_family,
        infinity_family,
        branch="G",
        z=1.0,
    ):
        total += (
            row.coefficient
            * hjs_ground_entry(
                structure_sign,
                row.infinity_hjs_ground_parity,
                row.zero_hjs_ground_parity,
            )
            * hjs_ground_entry(
                structure_sign,
                row.infinity_antiholomorphic_ground_parity,
                row.zero_antiholomorphic_ground_parity,
                antiholomorphic=True,
            )
        )
    return total


__all__ = [
    "PCOTimeBranch",
    "RamondFamily",
    "RamondFreeFieldComponent",
    "TwoRamondComponentWeight",
    "hjs_ground_entry",
    "physical_ramond_components",
    "rrns_primary_projection_coefficient",
    "time_ising_psi0_three_point",
    "time_ising_two_point",
    "two_ramond_component_weights",
]
