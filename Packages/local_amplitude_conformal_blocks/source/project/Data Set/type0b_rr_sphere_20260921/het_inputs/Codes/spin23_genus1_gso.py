#!/usr/bin/env python3
r"""Diagonal HO GSO projection at genus one.

The projection is fixed by the original two-dimensional heterotic-string
construction (Davis--Larsen--Seiberg, arXiv:hep-th/0505081):

.. math::

   (-1)^{F_L+F_R}=(-1)^{f_L+f_R}=1.

Equivalently, the covariant conjugacy classes are glued as

.. math::

   (O_8,O_{24})\oplus(V_8,V_{24})
   \oplus(S_8,C_{24})\oplus(C_8,S_{24}).

Expanding these characters gives projector coefficients ``+1/2`` in the
three even spin structures and ``-1/2`` in the odd ``R_tilde`` trace.  The
familiar effective even-spin signs ``(+,-,-)`` are *not* three independent
GSO choices: the latter two minus signs come from the chiral
``(psi^0, psi^phi, beta, gamma)`` trace.  This distinction is retained in
the table below so that those signs are not counted twice.

For the two- and three-point Spin(23) amplitudes, the odd spin structure
vanishes before its coefficient is used: fewer than 23 insertions of the
free ``lambda^a`` fields cannot saturate their 23 constant zero modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from spin23_genus1_free_fields import dedekind_eta, theta_characteristic
from spin23_genus1_spin import SpinLabel, TorusSpinStructure, torus_spin_structures


@dataclass(frozen=True)
class DiagonalGSOEntry:
    """One row of the diagonal genus-one HO projection."""

    spin_structure: TorusSpinStructure
    theta_name: str
    projector_coefficient: float
    even_superghost_trace_phase: int | None

    @property
    def effective_even_coefficient(self) -> float | None:
        """Return projector times the even ``psi psi beta gamma`` phase."""

        if self.even_superghost_trace_phase is None:
            return None
        return self.projector_coefficient * self.even_superghost_trace_phase


def diagonal_gso_table() -> tuple[DiagonalGSOEntry, ...]:
    r"""Return the convention-checked HO spin table in trace order.

    The odd row has no ``even_superghost_trace_phase`` because the odd
    ``beta-gamma`` system has zero modes and must be treated with its PCO
    prescription rather than by the even determinant formula.
    """

    ns, ns_tilde, r, r_tilde = torus_spin_structures()
    return (
        DiagonalGSOEntry(ns, "theta_3", 0.5, 1),
        DiagonalGSOEntry(ns_tilde, "theta_4", 0.5, -1),
        DiagonalGSOEntry(r, "theta_2", 0.5, -1),
        DiagonalGSOEntry(r_tilde, "theta_1", -0.5, None),
    )


def diagonal_projector_coefficient(spin_structure: TorusSpinStructure) -> float:
    """Return the character-gluing coefficient for one spin structure."""

    for entry in diagonal_gso_table():
        if entry.spin_structure == spin_structure:
            return entry.projector_coefficient
    raise ValueError("unknown torus spin structure")


def diagonal_projector_sum(
    contributions: Mapping[SpinLabel, complex],
) -> complex:
    r"""Sum fixed-spin contributions that already include ghost factors.

    Every supplied contribution must use the trace convention encoded by
    :class:`~spin23_genus1_spin.TorusSpinStructure`.  In particular, do not
    pre-multiply the even rows by ``(+,-,-)``; those phases belong to the
    explicit superghost/free-fermion factor.
    """

    expected = {entry.spin_structure.label for entry in diagonal_gso_table()}
    unknown = set(contributions) - expected
    if unknown:
        raise ValueError(f"unknown spin labels: {sorted(unknown)!r}")
    return sum(
        entry.projector_coefficient
        * complex(contributions.get(entry.spin_structure.label, 0.0j))
        for entry in diagonal_gso_table()
    )


def ho_vacuum_fermion_character(
    tau: complex,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the HO vacuum character, exactly equal to ``24``.

    This is the direct numerical form of

    .. math::

       \frac12\left[
       \left(\frac{\theta_3}{\eta}\right)^{12}
       -\left(\frac{\theta_4}{\eta}\right)^{12}
       -\left(\frac{\theta_2}{\eta}\right)^{12}\right].
    """

    eta = dedekind_eta(tau, precision=precision)
    theta_3 = theta_characteristic(0.0, 0.0, 0.0, tau, precision=precision)
    theta_4 = theta_characteristic(0.0, 0.5, 0.0, tau, precision=precision)
    theta_2 = theta_characteristic(0.5, 0.0, 0.0, tau, precision=precision)
    return 0.5 * (
        (theta_3 / eta) ** 12
        - (theta_4 / eta) ** 12
        - (theta_2 / eta) ** 12
    )


__all__ = [
    "DiagonalGSOEntry",
    "diagonal_gso_table",
    "diagonal_projector_coefficient",
    "diagonal_projector_sum",
    "ho_vacuum_fermion_character",
]
