#!/usr/bin/env python3
r"""Spin-structure and PCO bookkeeping for genus-one NS amplitudes.

This module records exact kinematic bookkeeping; it does not guess the
Spin(23) heterotic GSO coefficients.  A torus spin structure is specified by
the fermion periodicities around the spatial and Euclidean-time cycles.  The
spatial periodicity selects the NS or Ramond Hilbert space and temporal
periodicity selects an ordinary trace or a trace with :math:`(-1)^F`.

For NS punctures, Xi Yin's nonsingular PCO gauge inserts the *total* matter
supercurrent.  This module records only the super-Liouville word in the
branch where the supercurrent acts on the Liouville primary: every insertion
carries :math:`G^{\rm SL}_{-1/2}` for an even spin structure, whereas an odd
spin structure has one distinguished :math:`G^{\rm SL}_{-3/2}` insertion.
The complementary free-:math:`X^0` supercurrent branches are assembled in
``spin23_genus1_amplitude.py``.  For the current two- and three-point states,
the complete odd-spin contribution vanishes by the 23 free-fermion zero-mode
rule before the unresolved total-:math:`G_{-3/2}^{\rm m}` expansion is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ns_algebra.ns_sca import G, Word


Sector = Literal["NS", "R"]
TraceInsertion = Literal["identity", "fermion_parity"]
SpinLabel = Literal["NS", "NS_tilde", "R", "R_tilde"]


@dataclass(frozen=True)
class TorusSpinStructure:
    """One chiral torus spin structure and its operator interpretation."""

    spatial_periodic: bool
    temporal_periodic: bool

    @property
    def sector(self) -> Sector:
        """Return the Hilbert-space sector selected around the spatial cycle."""

        return "R" if self.spatial_periodic else "NS"

    @property
    def trace_insertion(self) -> TraceInsertion:
        """Return the operator inserted in the Euclidean-time trace."""

        return "fermion_parity" if self.temporal_periodic else "identity"

    @property
    def lift_sign(self) -> int:
        """Return ``-1`` precisely when the trace contains :math:`(-1)^F`."""

        return -1 if self.temporal_periodic else 1

    @property
    def arf_invariant(self) -> int:
        """Return the genus-one Arf invariant (zero even, one odd)."""

        return int(self.spatial_periodic and self.temporal_periodic)

    @property
    def label(self) -> SpinLabel:
        """Return the conventional trace label for this spin structure."""

        if self.sector == "NS":
            return "NS_tilde" if self.temporal_periodic else "NS"
        return "R_tilde" if self.temporal_periodic else "R"


@dataclass(frozen=True)
class GenusOnePCOPrescription:
    """Super-Liouville branch of the descendants in one PCO gauge choice."""

    spin_structure: TorusSpinStructure
    external_words: tuple[Word, ...]
    distinguished_puncture: int | None

    @property
    def is_odd(self) -> bool:
        """Return whether the underlying spin structure is odd."""

        return bool(self.spin_structure.arf_invariant)


def torus_spin_structures() -> tuple[TorusSpinStructure, ...]:
    """Return all four chiral torus spin structures in trace order."""

    return (
        TorusSpinStructure(False, False),
        TorusSpinStructure(False, True),
        TorusSpinStructure(True, False),
        TorusSpinStructure(True, True),
    )


def genus_one_ns_pco_prescription(
    n_punctures: int,
    spin_structure: TorusSpinStructure,
    *,
    distinguished_puncture: int = 0,
) -> GenusOnePCOPrescription:
    r"""Return the super-Liouville PCO branch for external NS states.

    This is not a complete total-matter-supercurrent insertion.  In
    particular, the even-spin physical correlator must also sum terms in
    which selected :math:`G^{X^0}_{-1/2}` factors produce ``psi0`` fields.
    The routine assigns neither those components nor a GSO coefficient.
    """

    if not isinstance(n_punctures, int):
        raise TypeError("n_punctures must be an integer")
    if n_punctures <= 0:
        raise ValueError("n_punctures must be positive")
    if not isinstance(spin_structure, TorusSpinStructure):
        raise TypeError("spin_structure must be a TorusSpinStructure")
    if not 0 <= distinguished_puncture < n_punctures:
        raise ValueError("distinguished_puncture is outside the puncture range")

    half_mode: Word = (G(Fraction(-1, 2)),)
    if spin_structure.arf_invariant == 0:
        return GenusOnePCOPrescription(
            spin_structure=spin_structure,
            external_words=tuple(half_mode for _ in range(n_punctures)),
            distinguished_puncture=None,
        )

    words = [half_mode for _ in range(n_punctures)]
    words[distinguished_puncture] = (G(Fraction(-3, 2)),)
    return GenusOnePCOPrescription(
        spin_structure=spin_structure,
        external_words=tuple(words),
        distinguished_puncture=distinguished_puncture,
    )


__all__ = [
    "GenusOnePCOPrescription",
    "Sector",
    "SpinLabel",
    "TorusSpinStructure",
    "TraceInsertion",
    "genus_one_ns_pco_prescription",
    "torus_spin_structures",
]
