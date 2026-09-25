#!/usr/bin/env python3
"""Generic (unequal-energy) kinematic assembly for the Spin(23) heterotic
1->3 continuum amplitude.

This module removes the equal-split assumption from the earlier numerical
prototype.  It does not prescribe a particular quadrature implementation;
instead it provides the exact energy-dependent pieces that a block/moduli
integrator needs at every internal momentum P and cross ratio z.

Positions and channel convention
--------------------------------
  outgoing 1 at 0, outgoing 2 at z, outgoing 3 at 1, incoming 4 at infinity.

The reduced amplitude is decomposed as

  A/N = M1 delta(a4,a1) delta(a2,a3)
      + M2 delta(a4,a2) delta(a1,a3)
      + M3 delta(a4,a3) delta(a1,a2).

With the spectator-fermion ordering used here, the anti-holomorphic Wick
factors are respectively

  r1 = 1/(1-zbar),  r2 = -1,  r3 = 1/zbar.

The picture-raised matter combination is

  K(z,zbar) + omega2*omega3/(1-z) G(z,zbar),

where G=<V4 V3 V2 V1> and K=<V4 Lambda3 Lambda2 V1>.  A global convention
sign can be supplied through ``k_sign`` or absorbed into the universal sphere
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cmath
import math


ComplexFunction3 = Callable[[complex, complex, complex], complex]


@dataclass(frozen=True)
class Kinematics:
    omega1: complex
    omega2: complex
    omega3: complex

    @property
    def omega(self) -> complex:
        return self.omega1 + self.omega2 + self.omega3

    @property
    def weights(self) -> tuple[complex, complex, complex, complex]:
        return tuple((1 + w * w) / 2 for w in (self.omega1, self.omega2, self.omega3, self.omega))  # type: ignore[return-value]

    @property
    def pole_margin(self) -> float:
        ims = [self.omega1.imag, self.omega2.imag, self.omega3.imag]
        return 1.0 - (sum(ims) + max(ims))

    def time_factor(self, z: complex) -> complex:
        """Single-valued timelike free-boson factor in the 12->34 channel."""
        zbar = z.conjugate()
        return cmath.exp(
            -self.omega1 * self.omega2 * (cmath.log(z) + cmath.log(zbar))
            -self.omega2 * self.omega3 * (cmath.log(1 - z) + cmath.log(1 - zbar))
        )

    def structure_products(
        self,
        p_internal: complex,
        C: ComplexFunction3,
        C_tilde: ComplexFunction3,
    ) -> tuple[complex, complex]:
        """Return C12P*C34P and Ctilde12P*Ctilde34P."""
        cc = C(self.omega1, self.omega2, p_internal) * C(
            self.omega3, self.omega, p_internal
        )
        tt = C_tilde(self.omega1, self.omega2, p_internal) * C_tilde(
            self.omega3, self.omega, p_internal
        )
        return cc, tt

    def invariant_coordinates(self, paired_leg: int) -> tuple[complex, complex, complex]:
        """Coordinates (S,p,v) for fitting channel M_i.

        S is total incoming energy, p is the outgoing energy paired with the
        incoming Spin(23) index, and v is the product of the other two
        outgoing energies.  Bose symmetry implies the reduced channel can be
        represented as a function F(S,p,v).
        """
        ws = (self.omega1, self.omega2, self.omega3)
        if paired_leg not in (1, 2, 3):
            raise ValueError("paired_leg must be 1, 2, or 3")
        i = paired_leg - 1
        rest = [ws[j] for j in range(3) if j != i]
        return self.omega, ws[i], rest[0] * rest[1]

    def soft_factor(self, paired_leg: int) -> complex:
        """The resonance-motivated product omega_j*omega_k for channel M_i."""
        return self.invariant_coordinates(paired_leg)[2]


@dataclass(frozen=True)
class BlockValues:
    """Holomorphic and anti-holomorphic block values at fixed P,z.

    ``even_star`` and ``odd_star`` have level-1/2 descendants at external
    positions 3 and 2 on the picture-changed chirality.
    """

    even: complex
    odd: complex
    even_bar: complex
    odd_bar: complex
    even_star: complex
    odd_star: complex


@dataclass(frozen=True)
class ChannelDensities:
    G: complex
    K: complex
    common: complex
    M1: complex
    M2: complex
    M3: complex



def assemble_channel_densities(
    kin: Kinematics,
    z: complex,
    cc: complex,
    tt: complex,
    blocks: BlockValues,
    *,
    k_sign: complex = 1,
) -> ChannelDensities:
    """Assemble the three tensor-channel integrands at fixed P and z.

    The returned quantities still need the measure dP/pi d^2z and the chosen
    OPE/moduli regularization.  This function is the direct unequal-energy
    replacement for an equal-split expression with w=omega/3.
    """
    zbar = z.conjugate()

    G = (
        cc * blocks.even * blocks.even_bar
        + tt * blocks.odd * blocks.odd_bar
    )

    K = k_sign * (
        cc * blocks.odd_star * blocks.even_bar
        + tt * blocks.even_star * blocks.odd_bar
    )

    common = kin.time_factor(z) * (
        K + kin.omega2 * kin.omega3 * G / (1 - z)
    )

    return ChannelDensities(
        G=G,
        K=K,
        common=common,
        M1=common / (1 - zbar),
        M2=-common,
        M3=common / zbar,
    )



def reduced_channels(
    kin: Kinematics,
    M1: complex,
    M2: complex,
    M3: complex,
    *,
    include_pi: bool = True,
) -> tuple[complex, complex, complex]:
    """Return Ri=Mi/(pi*omega_j*omega_k), useful for model discovery."""
    divisor = math.pi if include_pi else 1.0
    return tuple(
        M / (divisor * kin.soft_factor(i))
        for i, M in enumerate((M1, M2, M3), start=1)
    )  # type: ignore[return-value]



def permute_kinematics(kin: Kinematics, permutation: Sequence[int]) -> Kinematics:
    """Permute outgoing legs; ``permutation`` uses zero-based indices."""
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("permutation must contain 0,1,2 exactly once")
    ws = (kin.omega1, kin.omega2, kin.omega3)
    return Kinematics(*(ws[i] for i in permutation))



def expected_permuted_channels(
    channels: Sequence[complex], permutation: Sequence[int]
) -> tuple[complex, complex, complex]:
    """Expected channel values after the same outgoing-leg permutation.

    If omega'_r=omega_{permutation[r]}, then M'_r=M_{permutation[r]}.
    """
    if len(channels) != 3 or sorted(permutation) != [0, 1, 2]:
        raise ValueError("three channels and a permutation of 0,1,2 are required")
    return tuple(channels[i] for i in permutation)  # type: ignore[return-value]


if __name__ == "__main__":
    # Lightweight convention/self-consistency check.
    kin = Kinematics(0.10 + 0.15j, 0.17 + 0.18j, 0.23 + 0.20j)
    assert kin.omega == kin.omega1 + kin.omega2 + kin.omega3
    assert kin.pole_margin > 0
    assert expected_permuted_channels((1, 2, 3), (2, 0, 1)) == (3, 1, 2)
    print("Generic kinematics self-test passed.")
