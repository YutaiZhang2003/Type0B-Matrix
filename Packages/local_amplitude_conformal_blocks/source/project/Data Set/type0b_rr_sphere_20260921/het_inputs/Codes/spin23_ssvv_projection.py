#!/usr/bin/env python3
"""Legacy 24-component tensor projection for a putative ``S -> S V V`` amplitude.

The undeformed heterotic continuum vertex carries a Spin(24) vector index.
After choosing direction 24 for the tachyon wall, define

``S = T^24`` and ``V^a = T^a`` for ``a=1,...,23``.

For the puncture ordering used by the numerical four-vector calculation,

``(leg 1, leg 2, leg 3, leg 4) = (S_out, V_out, V_out, S_in)``,

the only surviving Spin(24) contraction is the coefficient ``C`` in

``A delta_03 delta_12 + B delta_02 delta_13 + C delta_01 delta_23``.

This construction predates the explicit super-Liouville-descendant
calculation.  Its candidate formula fails that genuine-singlet calculation
at order one and is retained only as a falsified diagnostic and for
reproducibility of the earlier projection study.  The physical candidate is
implemented in :mod:`spin23_genuine_formulas`.

This module concerns the raw reduced worldsheet coefficient.  It does not
insert an additional asymptotic singlet leg phase in the tachyon-wall
background.  Such a phase is physically distinct from the tensor projection
and must be supplied only after its incoming/outgoing convention is fixed.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SSVVKinematics:
    """Energies for ``S(omega0) -> S(omega1) V(omega2) V(omega3)``."""

    omega1: complex
    omega2: complex
    omega3: complex

    @property
    def omega0(self) -> complex:
        """Incoming energy fixed by energy conservation."""

        return self.omega1 + self.omega2 + self.omega3

    @property
    def vector_pair_sum(self) -> complex:
        """Sum of the two outgoing vector energies."""

        return self.omega2 + self.omega3

    @property
    def energy_product(self) -> complex:
        """Product of all four external energies."""

        return self.omega0 * self.omega1 * self.omega2 * self.omega3


def project_vvvv_coefficients(coefficients_abc: Sequence[complex]) -> complex:
    """Return the raw ``S -> S V V`` coefficient from ``(A, B, C)``.

    The result multiplies ``delta^{a_2 a_3}`` after setting the incoming and
    first outgoing Spin(24) indices to 24.
    """

    if len(coefficients_abc) != 3:
        raise ValueError("coefficients_abc must contain exactly (A, B, C)")
    return complex(coefficients_abc[2])


def project_scan_coefficients(coefficients_m123: Sequence[complex]) -> complex:
    """Return the same projection from the scan ordering ``(M1,M2,M3)``.

    The scan adapter uses ``(M1,M2,M3)=(C,B,A)``, hence this projection is
    stored directly as ``M1``.
    """

    if len(coefficients_m123) != 3:
        raise ValueError("coefficients_m123 must contain exactly (M1, M2, M3)")
    return complex(coefficients_m123[0])


def raw_ssvv_candidate(kinematics: SSVVKinematics) -> complex:
    r"""Return the legacy VVVV-projection hypothesis.

    .. math::

       \mathcal M^{\rm raw}_{S\to SVV}
       =-\pi\frac{\omega_0\omega_1\omega_2\omega_3}
       {1+i(\omega_2+\omega_3)}.

    This is not the genuine-singlet ``S -> SVV`` formula.  A pole is reported
    explicitly rather than regularized numerically.
    """

    denominator = 1.0 + 1j * kinematics.vector_pair_sum
    if denominator == 0:
        raise ZeroDivisionError("the SSVV candidate is on its vector-pair pole")
    return -math.pi * kinematics.energy_product / denominator


def normalized_inverse(
    amplitude: complex,
    kinematics: SSVVKinematics,
    *,
    zero_tolerance: float = 0.0,
) -> complex:
    r"""Return the inverse quantity used for the affine formula test.

    .. math::

       Y=-\pi\omega_0\omega_1\omega_2\omega_3/\mathcal M.

    The candidate predicts ``Y = 1 + i (omega2 + omega3)``.
    """

    amplitude = complex(amplitude)
    if abs(amplitude) <= zero_tolerance:
        raise ZeroDivisionError("the SSVV amplitude is too small to invert")
    return -math.pi * kinematics.energy_product / amplitude


def raw_resonance_coefficient(kinematics: SSVVKinematics) -> complex:
    r"""Return the raw resonance value ``pi*omega2*omega3``.

    The resonance condition is ``omega0=i``.  This check applies to the raw
    worldsheet projection and excludes any separate singlet scattering-state
    leg factors.
    """

    if not cmath.isclose(kinematics.omega0, 1j, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise ValueError("resonance requires omega0=omega1+omega2+omega3=i")
    return math.pi * kinematics.omega2 * kinematics.omega3
