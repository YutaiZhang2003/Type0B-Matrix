#!/usr/bin/env python3
"""Spin(23) tensor projection for the raw ``S -> S S S`` amplitude.

Writing the Spin(24) continuum state selected by the tachyon wall as
``S=T^24``, set every external tensor index to 24.  Each of the three pair
contractions then equals one, so the raw reduced coefficient is ``A+B+C`` or,
in the scan convention, ``M1+M2+M3``.

No common sphere normalization, energy delta function, or asymptotic singlet
leg phases are inserted here.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SSSSKinematics:
    """Energies for ``S(omega0) -> S(omega1) S(omega2) S(omega3)``."""

    omega1: complex
    omega2: complex
    omega3: complex

    @property
    def omega0(self) -> complex:
        return self.omega1 + self.omega2 + self.omega3

    @property
    def energy_product(self) -> complex:
        return self.omega0 * self.omega1 * self.omega2 * self.omega3

    @property
    def pair_sums(self) -> tuple[complex, complex, complex]:
        """Return pair sums in the ``(12,13,23)`` channel order."""

        return (
            self.omega1 + self.omega2,
            self.omega1 + self.omega3,
            self.omega2 + self.omega3,
        )


def project_vvvv_coefficients(coefficients_abc: Sequence[complex]) -> complex:
    """Return ``A+B+C`` after setting all four tensor indices to 24."""

    if len(coefficients_abc) != 3:
        raise ValueError("coefficients_abc must contain exactly (A, B, C)")
    return sum((complex(value) for value in coefficients_abc), 0.0j)


def project_scan_coefficients(coefficients_m123: Sequence[complex]) -> complex:
    """Return ``M1+M2+M3`` in the scan ordering ``(C,B,A)``."""

    if len(coefficients_m123) != 3:
        raise ValueError("coefficients_m123 must contain exactly (M1, M2, M3)")
    return sum((complex(value) for value in coefficients_m123), 0.0j)


def channel_features(kinematics: SSSSKinematics) -> tuple[complex, complex, complex]:
    """Return the three reciprocal pair-channel denominators."""

    denominators = tuple(1.0 + 1j * pair_sum for pair_sum in kinematics.pair_sums)
    if any(value == 0 for value in denominators):
        raise ZeroDivisionError("the SSSS candidate is on a pair-channel pole")
    return tuple(1.0 / value for value in denominators)  # type: ignore[return-value]


def raw_ssss_candidate(kinematics: SSSSKinematics) -> complex:
    r"""Return the symmetric unit-weight candidate for the raw amplitude.

    .. math::

       \mathcal M^{\rm raw}_{S\to SSS}
       =-\pi\omega_0\omega_1\omega_2\omega_3
       \sum_{1\leq i<j\leq3}\frac{1}{1+i(\omega_i+\omega_j)}.
    """

    return -math.pi * kinematics.energy_product * sum(channel_features(kinematics))


def normalized_amplitude(amplitude: complex, kinematics: SSSSKinematics) -> complex:
    """Remove the common ``-pi*product(omega)`` prefactor."""

    denominator = -math.pi * kinematics.energy_product
    if denominator == 0:
        raise ZeroDivisionError("an external energy vanishes")
    return complex(amplitude) / denominator


def raw_resonance_coefficient(kinematics: SSSSKinematics) -> complex:
    r"""Return the raw ``omega0=i`` resonance polynomial.

    .. math::

       \pi(\omega_1\omega_2+\omega_1\omega_3+\omega_2\omega_3).
    """

    if not cmath.isclose(kinematics.omega0, 1j, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise ValueError("resonance requires omega0=omega1+omega2+omega3=i")
    return math.pi * (
        kinematics.omega1 * kinematics.omega2
        + kinematics.omega1 * kinematics.omega3
        + kinematics.omega2 * kinematics.omega3
    )
