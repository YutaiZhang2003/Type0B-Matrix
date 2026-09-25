#!/usr/bin/env python3
r"""Four-Ramond tensor and component sewing for the SO(7) x E8 theory.

This module contains the finite-dimensional part of the four-Ramond sphere
integrand.  It deliberately does not pretend that a tensor pole ansatz is a
Liouville correlator.  Its responsibilities are:

* fix a factorial-normalized Spin(7) four-spinor basis and its exact Fierz
  matrices;
* enumerate the physical Ramond Fourier/cocycle components at
  ``(0,z,1,infinity)``;
* enforce the diagonal NS-channel GSO parity condition; and
* construct holomorphic and anti-HJS four-Ramond/internal-NS blocks for
  arbitrary external Ramond ground components.

The last item is needed because the physical ``Psi -> Psi_tilde^3`` vertex
does not have four ``w+`` super-Liouville components.  In the
antiholomorphic path only convention phases are conjugated.  Analytically
continued momenta are inserted unchanged.

The strict free-boson cocycle frame fixes the remaining cyclic time-Ising
product to ``tau=1``.  Public low-level helpers still accept an explicit
phase, and the fixed-P kernel permits an override, so alternative framed
transport conventions remain testable rather than being silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from itertools import product
from typing import Literal, Mapping, Sequence

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import (
    Word as NSWord,
    fermion_parity as ns_fermion_parity,
    gram_matrix as ns_gram_matrix,
)
from ramond_algebra.ns_rr_three_point_tensor import (
    hjs_ns_rr_polynomial_ground_tensor,
    ns_rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_sca import ground_state as ramond_ground_state
from so7e8_ramond_cocycles import (
    RamondFamily,
    physical_ramond_components,
)
from so7e8_four_ramond_elliptic_recursion import (
    FourRamondEllipticBlockSeries,
    general_four_ramond_sld_elliptic_h_series,
)
from spin23_ramond_blocks import ramond_weight
from spin23_super_liouville_data import ns_weight
from spin23_super_liouville_data import rr_ns_chiral_structure_constant


FourRamondComponent = Literal["even", "odd"]
Chirality = Literal["holomorphic", "antiholomorphic"]
Spin7AffineChannel = Literal["vacuum", "vector"]
TimeIsingChannel = Literal["identity", "fermion"]
SLDBlockBackend = Literal["inverse_gram", "elliptic_recursion"]

_HOL_PHASE = (sp.S.One + sp.I) / sp.sqrt(2)

# Constant part of
#
#   <tau_r(infinity) psi(1) tau_s(0)>
#       = 2^(-1/2) TIME_ISING_RNSR_PHASE_MATRIX[r,s].
#
# This R--NS--R orientation is fixed in so7e8_ramond_cocycles.py.  A four-R
# block needs the cyclically turned NS--R--R and R--R--NS vertices instead;
# the product of those two turns is the one phase not fixed by the physical
# S/Psi/Psi_tilde and V/Psi/Psi_tilde cubics.
TIME_ISING_RNSR_PHASE_MATRIX = np.asarray(
    (
        (0.0 + 0.0j, cmath.exp(-0.25j * math.pi)),
        (cmath.exp(0.25j * math.pi), 0.0 + 0.0j),
    ),
    dtype=np.complex128,
)


def bosonized_time_ising_turning_phase() -> complex:
    r"""Return the cyclic product ``tau=1`` in the repository cocycle frame.

    Double the real time Ising fermion and bosonize the resulting complex
    fermion as ``psi_+=exp(iH)``, with charged twist fields
    ``S_±=exp(±iH/2)``.  All nonzero charged three-point functions then have
    unit coefficient in radial order.  The two cyclic changes needed by the
    four-spin sewing carry inverse square-root monodromies, so their product
    is one.  Transforming from ``S_±`` to the real ``tau_0,tau_1`` basis is
    exactly the unitary transformation which gives
    :data:`TIME_ISING_RNSR_PHASE_MATRIX`; it cannot change the common cyclic
    product.  Thus the already calibrated R--NS--R cocycle fixes ``tau=1``.
    """

    return 1.0 + 0.0j


def four_ramond_superghost_factor(z: complex) -> complex:
    r"""Return ``<e^-phi/2(0)e^-phi/2(z)e^-phi/2(1)e^-phi/2(inf)>``.

    The normalized infinity limit leaves
    ``z**(-1/4)*(1-z)**(-1/4)`` on principal holomorphic branches.
    """

    value = complex(z)
    if value in (0, 1):
        raise ValueError("the superghost factor is singular at 0 and 1")
    return cmath.exp(-0.25 * (cmath.log(value) + cmath.log(1-value)))


# The basis is
#
#   T_r = 1/r! (C gamma^{a1...ar})_01
#                    (C gamma^{a1...ar})_23,  r=0,1,2,3,
#
# with every repeated vector index summed.  Equivalently one sums once over
# strictly increasing index sets.  This convention is the one in which the
# fully alternating Cayley direction is 3*T_1 + T_2.
SPIN7_TENSOR_RANKS = (0, 1, 2, 3)
SPIN7_BILINEAR_EXCHANGE_SIGNS = np.asarray((1, -1, -1, 1), dtype=np.int64)

# F[r,s] is defined by
#
#   T_s^(02|13) = sum_r F[r,s] T_r^(01|23).
#
# It is also the coefficient map for exchanging spinor slots 1 and 2.  The
# entries were derived from an explicit Cl(7) representation and are exact.
SPIN7_FIERZ_02_TO_01 = np.asarray(
    (
        (1 / 8, -7 / 8, 21 / 8, -35 / 8),
        (-1 / 8, -5 / 8, -9 / 8, -5 / 8),
        (1 / 8, -3 / 8, 1 / 8, 5 / 8),
        (-1 / 8, -1 / 8, 3 / 8, 3 / 8),
    ),
    dtype=np.float64,
)

# Spin(7)_1 Knizhnik--Zamolodchikov matrices in the same tensor basis.  The
# spinor quadratic Casimir is 21/4 in this normalization and k+g=6.
SPIN7_KZ_PAIR_0Z = np.diag((-21 / 4, -9 / 4, -1 / 4, 3 / 4)).astype(float)
SPIN7_KZ_PAIR_Z1 = np.asarray(
    (
        (0, 0, 21 / 4, 0),
        (0, -3 / 2, 0, 15 / 4),
        (1 / 4, 0, -5 / 2, -5 / 2),
        (0, 3 / 4, -3 / 2, -3),
    ),
    dtype=np.float64,
)


def spin7_pairing_change_matrix(
    target_pairing: Literal["01|23", "02|13", "03|12"],
) -> np.ndarray:
    r"""Return the exact coefficient map into the ``(01)(23)`` basis.

    If ``F = spin7_pairing_change_matrix('02|13')``, then column ``s``
    contains the coefficients of :math:`T_s^{(02|13)}` in the fixed
    :math:`T_r^{(01|23)}` basis.  The ``03|12`` orientation is obtained by
    the corresponding adjacent-slot permutation, including the bilinear
    transpose signs.
    """

    identity = np.eye(4, dtype=np.float64)
    if target_pairing == "01|23":
        return identity
    crossing = SPIN7_FIERZ_02_TO_01.copy()
    if target_pairing == "02|13":
        return crossing
    if target_pairing == "03|12":
        exchange = np.diag(SPIN7_BILINEAR_EXCHANGE_SIGNS.astype(float))
        return exchange @ crossing @ exchange
    raise ValueError("target_pairing must be '01|23', '02|13', or '03|12'")


def spin7_reexpress_pairing_coefficients(
    coefficients: Sequence[complex],
    *,
    source_pairing: Literal["01|23", "02|13", "03|12"],
    target_pairing: Literal["01|23", "02|13", "03|12"],
) -> np.ndarray:
    """Reexpress one rank-basis coefficient vector in another pairing."""

    values = np.asarray(tuple(coefficients), dtype=np.complex128)
    if values.shape != (4,):
        raise ValueError("coefficients must contain the four ranks 0,1,2,3")
    source_to_fixed = spin7_pairing_change_matrix(source_pairing)
    target_to_fixed = spin7_pairing_change_matrix(target_pairing)
    return np.linalg.solve(target_to_fixed, source_to_fixed @ values)


def spin7_cayley_contact_direction() -> np.ndarray:
    r"""Return the alternating Cayley tensor ``3*T_1 + T_2``."""

    return np.asarray((0.0, 3.0, 1.0, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class Spin7SpectatorBlockSeries:
    r"""One local ``Spin(7)_1`` four-spin-field affine block.

    ``coefficients[n]`` is a four-vector in the factorial-normalized gamma
    basis.  The block is

    ``z**leading_exponent * sum_n coefficients[n] * z**n``.

    The vacuum block is normalized to leading tensor ``T_0`` and the vector
    block to leading tensor ``T_1``.  RR--NS trinion normalizations (and the
    relative orientation phase between the two affine channels) are not
    included.
    """

    channel: Spin7AffineChannel
    leading_exponent: float
    coefficients: Mapping[int, tuple[complex, complex, complex, complex]]
    maximum_order: int

    def value(self, z: complex) -> np.ndarray:
        z_value = complex(z)
        if z_value == 0:
            raise ValueError("the spin-field block is singular at z=0")
        polynomial = np.zeros(4, dtype=np.complex128)
        for order, coefficient in self.coefficients.items():
            polynomial += np.asarray(coefficient, dtype=np.complex128) * z_value**order
        return cmath.exp(self.leading_exponent * cmath.log(z_value)) * polynomial


def spin7_spectator_block_series(
    channel: Spin7AffineChannel,
    *,
    maximum_order: int,
    compatibility_tolerance: float = 1.0e-12,
) -> Spin7SpectatorBlockSeries:
    r"""Build an exact local Spin(7) four-spin affine block by KZ recursion.

    The coefficient vector obeys

    .. math::

       \partial_z f=\left({\Omega_{0z}\over6z}
       +{\Omega_{z1}\over6(z-1)}\right)f.

    At level one the vacuum Frobenius exponent is resonant with the
    non-integrable rank-three finite-Lie-algebra solution.  Its free
    coefficient is set to zero; this is precisely the level-one affine
    integrability condition, not a numerical regularization.
    """

    if channel not in ("vacuum", "vector"):
        raise ValueError("channel must be 'vacuum' or 'vector'")
    if not isinstance(maximum_order, int) or maximum_order < 0:
        raise ValueError("maximum_order must be a nonnegative integer")
    a_matrix = SPIN7_KZ_PAIR_0Z / 6.0
    b_matrix = SPIN7_KZ_PAIR_Z1 / 6.0
    exponents = np.diag(a_matrix)
    leading_exponent = -7 / 8 if channel == "vacuum" else -3 / 8
    leading_rank = 0 if channel == "vacuum" else 1
    leading = np.zeros(4, dtype=np.complex128)
    leading[leading_rank] = 1.0
    vectors: list[np.ndarray] = [leading]

    for order in range(1, maximum_order + 1):
        right = -b_matrix @ sum(vectors, np.zeros(4, dtype=np.complex128))
        vector = np.zeros(4, dtype=np.complex128)
        for rank, exponent in enumerate(exponents):
            denominator = leading_exponent + order - exponent
            if abs(denominator) <= compatibility_tolerance:
                if abs(right[rank]) > compatibility_tolerance:
                    raise ArithmeticError(
                        "incompatible resonant Spin(7) KZ Frobenius equation"
                    )
                # Exclude the non-integrable finite-algebra solution.
                vector[rank] = 0.0
            else:
                vector[rank] = right[rank] / denominator
        vectors.append(vector)

    return Spin7SpectatorBlockSeries(
        channel=channel,
        leading_exponent=leading_exponent,
        coefficients={
            order: tuple(complex(value) for value in vector)  # type: ignore[arg-type]
            for order, vector in enumerate(vectors)
        },
        maximum_order=maximum_order,
    )


def normalized_time_ising_scalar_block(
    channel: TimeIsingChannel,
    z: complex,
) -> complex:
    r"""Return the unit-leading Ising twist block in the ``z -> 0`` channel.

    The identity block starts as ``z**(-1/8)`` and the fermion block as
    ``z**(3/8)``.  The latter is twice the commonly written local Ising
    block because the two normalized RR--fermion trinions supply a product
    ``1/2``.  Their relative orientation phase is deliberately not included.
    """

    if channel not in ("identity", "fermion"):
        raise ValueError("channel must be 'identity' or 'fermion'")
    z_value = complex(z)
    if z_value in (0, 1):
        raise ValueError("the Ising twist block is singular at z=0 or z=1")
    root = cmath.sqrt(1.0 - z_value)
    common = cmath.exp(-0.125 * cmath.log(z_value * (1.0 - z_value)))
    if channel == "identity":
        return common * cmath.sqrt((1.0 + root) / 2.0)
    return 2.0 * common * cmath.sqrt((1.0 - root) / 2.0)


def calibrated_spin7_rrns_trinion_product(
    channel: Spin7AffineChannel,
) -> complex:
    r"""Return the two-trinion coefficient for a unit-leading Spin(7) block.

    The vacuum coefficient is the two-spin metric and is normalized to one.
    Each vector trinion has magnitude ``1/sqrt(2)``, so the vector product
    has magnitude ``1/2``.  In the fixed KZ tensor basis its coefficient is
    ``-i/2`` in the literal HJS convention used by the four-R blocks.

    This phase is not guessed.  The unequal-family and equal-family raw-HJS
    singlet cubics have phases ``-exp(-i*pi/4)`` and
    ``+exp(-i*pi/4)``; their product is ``+i``.  The corresponding vector
    cubics are real.  The certified four-R residue has a plus sign between
    the singlet and vector tensors, so cyclically turning the vector
    spectator trinion supplies ``+i`` in the physical bilinear ordering.
    However, in slot order ``(0,z,1,infinity)``,

    ``B1=(C gamma)_(infinity,1)(C gamma)_(0,z)=-T1``

    because ``C gamma^a`` is antisymmetric.  Hence the coefficient of the
    unit-leading fixed-basis ``T1`` block is ``-i/2``.  A common phase of
    the complete four-R amplitude remains conventional.
    """

    if channel == "vacuum":
        return 1.0 + 0.0j
    if channel == "vector":
        return -0.5j
    raise ValueError("channel must be 'vacuum' or 'vector'")


def time_ising_rrns_trinion_product(
    parities: Sequence[int],
    channel: TimeIsingChannel,
    *,
    fermion_turning_phase: complex,
    phase_tolerance: float = 1.0e-12,
) -> complex:
    r"""Return the componentwise two-trinion time-Ising coefficient.

    ``parities`` follows ``(0,z,1,infinity)``.  The recorded two-point and
    R--NS--R tensors use one Ramond state as a BPZ bra at infinity.  In the
    four-point right trinion both Ramond states are finite kets.  Transporting
    the former bra to that finite slot gives the Ramond basis metric

    .. math::

       Z=\operatorname{diag}(1,-1).

    This is channel-independent.  Thus the right identity tensor is ``Z``
    (whereas the left one is the identity), and the right fermion tensor is
    ``Z M``.  Omitting ``Z`` makes two equal finite Ramond families select
    the HJS-odd rather than the certified HJS-even boson-in cubic.

    For the fermion channel, write

    .. math::

       \widehat M=\begin{pmatrix}
       0&e^{-i\pi/4}\\ e^{i\pi/4}&0
       \end{pmatrix}.

    Relative to the two local R--NS--R reference tensors fixed by
    :func:`so7e8_ramond_cocycles.time_ising_psi0_three_point`, the product is

    .. math::

       {\tau\over2}\,
       \widehat M_{r_1 r_\infty}(Z\widehat M)_{r_0 r_z}.

    The factor ``1/2`` is the product of the two normalized RR--fermion OPE
    coefficients.  It is external to
    :func:`normalized_time_ising_scalar_block`, whose fermion block was
    rescaled to have unit leading coefficient.

    ``tau`` is the product of the two *cyclic turning* phases which convert
    the recorded R--NS--R tensor into the NS--R--R and R--R--NS tensors
    required by four-point sewing.  The repository fixes the matrix
    ``M``.  The three-point cubics alone do not determine it, but explicit
    doubled-Ising bosonization fixes their product to one in the repository's
    strict cocycle frame.  It remains an argument here to expose framing
    dependence and must be a unit phase.
    """

    values = tuple(parities)
    if len(values) != 4 or any(value not in (0, 1) for value in values):
        raise ValueError("parities must contain four zeros or ones")
    if channel not in ("identity", "fermion"):
        raise ValueError("channel must be 'identity' or 'fermion'")
    if not time_ising_channel_has_ground_support(values, channel):
        return 0.0 + 0.0j
    r_zero, r_z, r_one, r_infinity = values
    finite_pair_transport = -1.0 if r_zero else 1.0
    if channel == "identity":
        return finite_pair_transport + 0.0j

    turning = complex(fermion_turning_phase)
    if not math.isfinite(turning.real) or not math.isfinite(turning.imag):
        raise ValueError("fermion_turning_phase must be finite")
    if abs(abs(turning) - 1.0) > phase_tolerance:
        raise ValueError("fermion_turning_phase must have unit magnitude")
    # On fermion support, with rows r_one and columns r_zero, inclusion of
    # the finite-pair transport gives [[-i,-1],[1,-i]] before tau/2.
    left = TIME_ISING_RNSR_PHASE_MATRIX[r_one, r_infinity]
    right = TIME_ISING_RNSR_PHASE_MATRIX[r_zero, r_z]
    return 0.5 * turning * left * finite_pair_transport * right


def four_ramond_free_sector_trinion_product(
    parities: Sequence[int],
    *,
    time_channel: TimeIsingChannel,
    spin7_channel: Spin7AffineChannel,
    fermion_turning_phase: complex,
) -> complex:
    r"""Return the calibrated free-sector coefficient of one component.

    This multiplies the unit-leading time-Ising and Spin(7) blocks.  It is
    normalized in the literal HJS convention.  The Spin(7) factor is fixed
    by singlet/vector factorization; the required ``fermion_turning_phase``
    exposes the remaining time-sector datum those primary residues cannot
    determine.  This is separate from finite-component braid data needed
    when changing four-point sewing frames.
    """

    return time_ising_rrns_trinion_product(
        parities,
        time_channel,
        fermion_turning_phase=fermion_turning_phase,
    ) * calibrated_spin7_rrns_trinion_product(spin7_channel)


@dataclass(frozen=True)
class FourRamondPolarizationComponent:
    """One term in the four-fold physical Ramond Fourier expansion.

    Slot order is ``(0,z,1,infinity)``.  The coefficient in the infinity
    slot is BPZ conjugated; momenta never enter this finite operation.
    """

    families: tuple[RamondFamily, RamondFamily, RamondFamily, RamondFamily]
    holomorphic_ground_parities: tuple[int, int, int, int]
    time_ising_parities: tuple[int, int, int, int]
    antiholomorphic_ground_parities: tuple[int, int, int, int]
    coefficient: complex


def physical_four_ramond_components(
    families: Sequence[RamondFamily],
) -> tuple[FourRamondPolarizationComponent, ...]:
    r"""Expand four physical Ramond vertices into HJS/Ising components.

    ``families`` follows ``(0,z,1,infinity)``.  There are exactly sixteen
    terms.  This is the four-leg extension of
    :func:`so7e8_ramond_cocycles.two_ramond_component_weights`.
    """

    ordered_families = tuple(families)
    if len(ordered_families) != 4:
        raise ValueError("families must follow the four slots (0,z,1,infinity)")
    choices = tuple(physical_ramond_components(family) for family in ordered_families)
    result: list[FourRamondPolarizationComponent] = []
    for selected in product(*choices):
        coefficient = complex(selected[3].coefficient).conjugate()
        coefficient *= selected[0].coefficient
        coefficient *= selected[1].coefficient
        coefficient *= selected[2].coefficient
        holomorphic = tuple(row.hjs_ground_parity for row in selected)
        time = tuple(row.time_ising_parity for row in selected)
        antiholomorphic = tuple(
            row.antiholomorphic_ising_parity for row in selected
        )
        result.append(
            FourRamondPolarizationComponent(
                families=ordered_families,  # type: ignore[arg-type]
                holomorphic_ground_parities=holomorphic,  # type: ignore[arg-type]
                time_ising_parities=time,  # type: ignore[arg-type]
                antiholomorphic_ground_parities=antiholomorphic,  # type: ignore[arg-type]
                coefficient=coefficient,
            )
        )
    return tuple(result)


def time_ising_channel_has_ground_support(
    parities: Sequence[int],
    channel: TimeIsingChannel,
) -> bool:
    r"""Return whether both RR--NS time-Ising trinions are nonzero.

    Identity sewing pairs equal Ramond ground labels at ``(0,z)`` and
    ``(1,infinity)``.  Fermion sewing pairs unequal labels on both sides.
    This function records support only; it intentionally does not assign the
    unresolved relative RR--NS orientation phase.
    """

    values = tuple(parities)
    if len(values) != 4 or any(value not in (0, 1) for value in values):
        raise ValueError("parities must contain four zeros or ones")
    right_flip = values[0] ^ values[1]
    left_flip = values[2] ^ values[3]
    expected = 0 if channel == "identity" else 1 if channel == "fermion" else None
    if expected is None:
        raise ValueError("channel must be 'identity' or 'fermion'")
    return right_flip == expected and left_flip == expected


@dataclass(frozen=True)
class FourRamondParitySewing:
    r"""One diagonal-GSO-compatible internal NS parity assignment."""

    holomorphic_sld_parity: int
    time_ising_parity: int
    antiholomorphic_sld_parity: int
    spin7_fermion_parity: int

    @property
    def supersymmetric_ns_class(self) -> int:
        # The NS matter vacuum has (-1)^F=-1 in the supersymmetric GSO
        # convention.  Hence its character class is one plus the raw number
        # of SLD/time fermion excitations.
        return 1 ^ self.holomorphic_sld_parity ^ self.time_ising_parity

    @property
    def gauge_ns_class(self) -> int:
        return self.antiholomorphic_sld_parity ^ self.spin7_fermion_parity

    @property
    def spin7_channel(self) -> Spin7AffineChannel:
        return "vacuum" if self.spin7_fermion_parity == 0 else "vector"

    @property
    def time_channel(self) -> TimeIsingChannel:
        return "identity" if self.time_ising_parity == 0 else "fermion"


def allowed_four_ramond_parity_sewings() -> tuple[FourRamondParitySewing, ...]:
    r"""Return the eight internal sewings allowed by diagonal GSO.

    The exact condition is

    ``1 + q_SLD + q_time = q_antiSLD + q_Spin7 (mod 2)``.

    The offset is the standard minus fermion parity of the supersymmetric NS
    vacuum.  It is fixed by the known physical primaries: a vector has
    ``(q_h,q_t;q_anti,q_7)=(0,0;0,1)`` and a singlet has
    ``(0,0;1,0)``.  An equality of the four raw parities would incorrectly
    remove both states.

    It is incorrect to sum the holomorphic, antiholomorphic, and spectator
    channels independently and multiply the three sums afterwards.
    """

    result = []
    for q_h, q_t, q_a, q_7 in product((0, 1), repeat=4):
        row = FourRamondParitySewing(q_h, q_t, q_a, q_7)
        if row.supersymmetric_ns_class == row.gauge_ns_class:
            result.append(row)
    return tuple(result)


@dataclass(frozen=True)
class FourRamondComponentSewingTerm:
    """One finite term before scalar conformal blocks are multiplied."""

    polarization: FourRamondPolarizationComponent
    parity: FourRamondParitySewing
    left_structure_sign: int
    right_structure_sign: int
    coefficient: complex


def four_ramond_component_sewing_terms(
    families: Sequence[RamondFamily],
) -> tuple[FourRamondComponentSewingTerm, ...]:
    r"""Enumerate the correlated physical four-Ramond component sum.

    Every retained two-trinion HJS branch carries ``1/4`` because the public
    RRNS constants are ``C_even`` and ``C_odd``, while the local HJS
    coefficients are half of them.  Terms with a vanishing time-Ising ground
    trinion are removed.  No resonance assumption is made.
    """

    result: list[FourRamondComponentSewingTerm] = []
    for polarization in physical_four_ramond_components(families):
        for parity in allowed_four_ramond_parity_sewings():
            if not time_ising_channel_has_ground_support(
                polarization.time_ising_parities,
                parity.time_channel,
            ):
                continue
            for left_sign, right_sign in product((-1, 1), repeat=2):
                result.append(
                    FourRamondComponentSewingTerm(
                        polarization=polarization,
                        parity=parity,
                        left_structure_sign=left_sign,
                        right_structure_sign=right_sign,
                        coefficient=0.25 * polarization.coefficient,
                    )
                )
    return tuple(result)


@dataclass(frozen=True)
class GeneralFourRamondNSBlockSeries:
    r"""A normalized four-Ramond/internal-NS HJS component block."""

    coefficients: Mapping[int, complex]
    component: FourRamondComponent
    chirality: Chirality
    c: complex
    h_internal: complex
    external_betas: tuple[complex, complex, complex, complex]
    external_weights: tuple[complex, complex, complex, complex]
    external_ground_parities: tuple[int, int, int, int]
    left_structure_sign: int
    right_structure_sign: int
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def descendant_value(self, z: complex) -> complex:
        z_value = complex(z)
        if z_value == 0:
            return complex(self.coefficients.get(0, 0.0j))
        log_z = cmath.log(z_value)
        return sum(
            value * cmath.exp(0.5 * level * log_z)
            for level, value in self.coefficients.items()
        )

    def value(self, z: complex, *, include_primary_power: bool = True) -> complex:
        descendant = self.descendant_value(z)
        if not include_primary_power:
            return descendant
        if z == 0:
            raise ValueError("the primary four-Ramond block is singular at z=0")
        h1, h2, _, _ = self.external_weights
        return cmath.exp(
            (self.h_internal - h1 - h2) * cmath.log(complex(z))
        ) * descendant


def _validate_sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _finite_complex(value: sp.Expr | complex, digits: int) -> complex:
    numeric = complex(sp.N(sp.sympify(value), digits))
    if not math.isfinite(numeric.real) or not math.isfinite(numeric.imag):
        raise ArithmeticError(f"non-finite four-Ramond component {numeric!r}")
    return numeric


def _external_hjs_factor(
    parity: int,
    beta: complex,
) -> complex:
    """Holomorphic polynomial/HJS factor for one external ground state."""

    if parity not in (0, 1):
        raise ValueError("external ground parities must be zero or one")
    if parity == 0:
        return 1.0 + 0.0j
    beta_value = complex(beta)
    if beta_value == 0:
        raise ValueError("an odd HJS external state is singular at beta=0")
    phase = complex(sp.N(_HOL_PHASE, 17))
    return phase * beta_value


def general_four_ramond_ns_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_betas: Sequence[complex],
    external_ground_parities: Sequence[int],
    maximum_twice_level: int,
    component: FourRamondComponent,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> GeneralFourRamondNSBlockSeries:
    r"""Construct one arbitrary-ground four-Ramond HJS component block.

    External order is ``(R1,R2,R3,R4)=(0,z,1,infinity)``.  The right
    trinion is NS--R2--R1.  The left RR4--R3--NS form is evaluated through
    the certified NS--R3--R4 orientation.  Holomorphic odd internal words
    carry ``-i`` and antiholomorphic odd words carry ``+i``.  An odd
    infinity Ramond ground state contributes a further minus sign on odd
    internal words, as required by HJS 0810.1203v2, equation (4.10).
    """

    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    if component not in ("even", "odd"):
        raise ValueError("component must be 'even' or 'odd'")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("chirality must be 'holomorphic' or 'antiholomorphic'")
    left_sign = _validate_sign(left_structure_sign, "left_structure_sign")
    right_sign = _validate_sign(right_structure_sign, "right_structure_sign")
    betas = tuple(complex(value) for value in external_betas)
    parities = tuple(external_ground_parities)
    if len(betas) != 4:
        raise ValueError("external_betas must contain four entries")
    if len(parities) != 4 or any(parity not in (0, 1) for parity in parities):
        raise ValueError("external_ground_parities must contain four zeros or ones")

    central_charge = complex(c)
    internal_weight = complex(h_internal)
    weights = tuple(ramond_weight(central_charge, beta) for beta in betas)

    if chirality == "antiholomorphic":
        # Conjugating only the terminal HJS matrix is insufficient for odd
        # external ground states: the ordered NS--R Ward recursion itself
        # contains convention-dependent i's.  Apply the convention
        # involution to the complete holomorphic coefficient instead.
        #
        # This is F_bar(beta,h,c) = conjugate(F_hol(conjugate(beta),
        # conjugate(h),conjugate(c))).  It is an analytic function of the
        # *original* beta values; it does not replace a continued physical
        # momentum by its numerical conjugate.  The public metadata below
        # consequently retains the supplied momenta unchanged.
        holomorphic_on_involuted_data = general_four_ramond_ns_block_series(
            c=central_charge.conjugate(),
            h_internal=internal_weight.conjugate(),
            external_betas=tuple(beta.conjugate() for beta in betas),
            external_ground_parities=parities,
            maximum_twice_level=maximum_twice_level,
            component=component,
            left_structure_sign=left_sign,
            right_structure_sign=right_sign,
            chirality="holomorphic",
            digits=digits,
            condition_limit=condition_limit,
        )
        return GeneralFourRamondNSBlockSeries(
            coefficients={
                level: coefficient.conjugate()
                for level, coefficient in holomorphic_on_involuted_data.coefficients.items()
            },
            component=component,
            chirality="antiholomorphic",
            c=central_charge,
            h_internal=internal_weight,
            external_betas=betas,  # type: ignore[arg-type]
            external_weights=weights,  # type: ignore[arg-type]
            external_ground_parities=parities,  # type: ignore[arg-type]
            left_structure_sign=left_sign,
            right_structure_sign=right_sign,
            maximum_twice_level=maximum_twice_level,
            gram_condition_numbers=(
                holomorphic_on_involuted_data.gram_condition_numbers
            ),
        )

    beta1, beta2, beta3, beta4 = betas
    a1, a2, a3, a4 = parities

    right_terminal = hjs_ns_rr_polynomial_ground_tensor(
        sp.sympify(beta2), sp.sympify(beta1), structure_sign=right_sign
    )
    left_terminal = hjs_ns_rr_polynomial_ground_tensor(
        sp.sympify(beta3), sp.sympify(beta4), structure_sign=left_sign
    )
    orientation = -1.0j

    external_factor = (
        _external_hjs_factor(a1, beta1)
        * _external_hjs_factor(a2, beta2)
        * _external_hjs_factor(a3, beta3)
        * _external_hjs_factor(a4, beta4)
    )
    states = tuple(ramond_ground_state(parity) for parity in parities)
    parity_offset = 0 if component == "even" else 1
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}

    for twice_level in range(parity_offset, maximum_twice_level + 1, 2):
        basis, gram = ns_gram_matrix(
            twice_level,
            h=sp.sympify(internal_weight),
            c=sp.sympify(central_charge),
        )
        gram_numeric = np.asarray(
            [
                [
                    _finite_complex(gram[row, column], digits)
                    for column in range(gram.cols)
                ]
                for row in range(gram.rows)
            ],
            dtype=np.complex128,
        )
        condition = float(np.linalg.cond(gram_numeric))
        if not math.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"NS Gram matrix at level {twice_level}/2 has condition "
                f"number {condition:.3e}, above {condition_limit:.3e}"
            )
        left_values: list[complex] = []
        right_values: list[complex] = []
        for word in basis:
            right = ns_rr_three_point_from_ground_tensor(
                word,
                states[1],
                states[0],
                h_infinity=sp.sympify(internal_weight),
                h_middle=sp.sympify(weights[1]),
                h_zero=sp.sympify(weights[0]),
                c=sp.sympify(central_charge),
                ground_tensor=right_terminal,
            )
            left_as_ns_rr = ns_rr_three_point_from_ground_tensor(
                word,
                states[2],
                states[3],
                h_infinity=sp.sympify(internal_weight),
                h_middle=sp.sympify(weights[2]),
                h_zero=sp.sympify(weights[3]),
                c=sp.sympify(central_charge),
                ground_tensor=left_terminal,
            )
            left = sp.sympify(orientation * (-1) ** a4) ** ns_fermion_parity(word)
            left *= left_as_ns_rr
            left_values.append(_finite_complex(left, digits))
            right_values.append(_finite_complex(right, digits))
        left_vector = np.asarray(left_values, dtype=np.complex128)
        right_vector = np.asarray(right_values, dtype=np.complex128)
        coefficients[twice_level] = complex(
            left_vector @ np.linalg.solve(gram_numeric, right_vector)
        ) / external_factor
        conditions[twice_level] = condition

    return GeneralFourRamondNSBlockSeries(
        coefficients=coefficients,
        component=component,
        chirality=chirality,
        c=central_charge,
        h_internal=internal_weight,
        external_betas=betas,  # type: ignore[arg-type]
        external_weights=weights,  # type: ignore[arg-type]
        external_ground_parities=parities,  # type: ignore[arg-type]
        left_structure_sign=left_sign,
        right_structure_sign=right_sign,
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def general_four_ramond_sld_chiral_block(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    external_ground_parities: Sequence[int],
    maximum_twice_level: int,
    component: FourRamondComponent,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> GeneralFourRamondNSBlockSeries:
    """Specialize the arbitrary-ground block to self-dual SLD data."""

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain four entries")
    betas = tuple(1.0j * momentum / math.sqrt(2.0) for momentum in momenta)
    return general_four_ramond_ns_block_series(
        c=13.5,
        h_internal=ns_weight(internal_momentum),
        external_betas=betas,
        external_ground_parities=external_ground_parities,
        maximum_twice_level=maximum_twice_level,
        component=component,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        chirality=chirality,
        digits=digits,
        condition_limit=condition_limit,
    )


FourRamondSLDBlock = (
    GeneralFourRamondNSBlockSeries | FourRamondEllipticBlockSeries
)


@dataclass(frozen=True)
class FixedPFourRamondKernelTerm:
    polarization: FourRamondPolarizationComponent
    parity: FourRamondParitySewing
    left_structure_sign: int
    right_structure_sign: int
    coefficient: complex
    holomorphic_sld: FourRamondSLDBlock
    antiholomorphic_sld: FourRamondSLDBlock
    time_channel: TimeIsingChannel
    spin7_block: Spin7SpectatorBlockSeries


@dataclass(frozen=True)
class FixedPFourRamondKernel:
    """Fixed internal-NS momentum four-R tensor kernel.

    The returned rank coefficients multiply the factorial-normalized
    ``T_0,...,T_3`` basis.  They omit ``dP/pi``, the common sphere
    normalization, coupling, and external reflection phases.
    """

    families: tuple[RamondFamily, RamondFamily, RamondFamily, RamondFamily]
    internal_momentum: complex
    time_momenta: tuple[complex, complex, complex, complex]
    fermion_turning_phase: complex
    sld_block_backend: SLDBlockBackend
    terms: tuple[FixedPFourRamondKernelTerm, ...]

    def evaluate(self, z: complex, zbar: complex | None = None) -> np.ndarray:
        zv = complex(z)
        zb = zv.conjugate() if zbar is None else complex(zbar)
        if zv in (0, 1) or zb in (0, 1):
            raise ValueError("the modulus must avoid 0 and 1")
        k1, k2, k3, _ = self.time_momenta
        tx = cmath.exp(
            -k1 * k2 * (cmath.log(zv) + cmath.log(zb))
            -k2 * k3 * (cmath.log(1-zv) + cmath.log(1-zb))
        )
        superghost = four_ramond_superghost_factor(zv)
        result = np.zeros(4, dtype=np.complex128)
        for term in self.terms:
            scalar = (
                term.coefficient
                * term.holomorphic_sld.value(zv)
                * term.antiholomorphic_sld.value(zb)
                * normalized_time_ising_scalar_block(term.time_channel, zv)
            )
            result += scalar * term.spin7_block.value(zb)
        return tx * superghost * result

    def evaluate_many(self, z_values: Sequence[complex]) -> np.ndarray:
        return np.stack([self.evaluate(z) for z in z_values], axis=0)


def build_fixed_p_four_ramond_kernel(
    internal_momentum: complex,
    *,
    external_liouville_momenta: Sequence[complex],
    time_momenta: Sequence[complex],
    families: Sequence[RamondFamily],
    maximum_twice_level: int = 1,
    spin7_maximum_order: int = 2,
    fermion_turning_phase: complex | None = None,
    sld_block_backend: SLDBlockBackend = "inverse_gram",
    digits: int = 40,
    condition_limit: float = 1.0e13,
) -> FixedPFourRamondKernel:
    """Build the complete finite-dimensional fixed-P four-R kernel."""

    momenta = tuple(complex(x) for x in external_liouville_momenta)
    times = tuple(complex(x) for x in time_momenta)
    ordered_families = tuple(families)
    if len(momenta) != 4 or len(times) != 4 or len(ordered_families) != 4:
        raise ValueError("momenta, times, and families must each have length four")
    if abs(sum(times)) > 1.0e-10:
        raise ValueError("signed time momenta must sum to zero")
    tau = (
        bosonized_time_ising_turning_phase()
        if fermion_turning_phase is None
        else complex(fermion_turning_phase)
    )
    if abs(abs(tau) - 1.0) > 1.0e-12:
        raise ValueError("fermion_turning_phase must have unit magnitude")
    if sld_block_backend not in ("inverse_gram", "elliptic_recursion"):
        raise ValueError(
            "sld_block_backend must be 'inverse_gram' or 'elliptic_recursion'"
        )
    p1, p2, p3, p4 = momenta
    structures = {
        (sl, sr): 0.25
        * rr_ns_chiral_structure_constant(
            p4, p3, internal_momentum, structure_sign=sl, precision=digits
        )
        * rr_ns_chiral_structure_constant(
            p2, p1, -complex(internal_momentum),
            structure_sign=sr, precision=digits
        )
        for sl, sr in product((-1, 1), repeat=2)
    }
    spin_blocks = {
        channel: spin7_spectator_block_series(
            channel, maximum_order=spin7_maximum_order
        )
        for channel in ("vacuum", "vector")
    }
    cache: dict[tuple, FourRamondSLDBlock] = {}
    block_builder = (
        general_four_ramond_sld_chiral_block
        if sld_block_backend == "inverse_gram"
        else general_four_ramond_sld_elliptic_h_series
    )
    terms: list[FixedPFourRamondKernelTerm] = []
    for sewing in four_ramond_component_sewing_terms(ordered_families):
        pol = sewing.polarization
        parity = sewing.parity
        signs = (sewing.left_structure_sign, sewing.right_structure_sign)
        h_component: FourRamondComponent = (
            "even" if parity.holomorphic_sld_parity == 0 else "odd"
        )
        a_component: FourRamondComponent = (
            "even" if parity.antiholomorphic_sld_parity == 0 else "odd"
        )
        hkey = ("h", h_component, signs, pol.holomorphic_ground_parities)
        akey = ("a", a_component, signs, pol.antiholomorphic_ground_parities)
        if hkey not in cache:
            cache[hkey] = block_builder(
                internal_momentum, external_momenta=momenta,
                external_ground_parities=pol.holomorphic_ground_parities,
                maximum_twice_level=maximum_twice_level,
                component=h_component, left_structure_sign=signs[0],
                right_structure_sign=signs[1], chirality="holomorphic",
                digits=digits, condition_limit=condition_limit,
            )
        if akey not in cache:
            cache[akey] = block_builder(
                internal_momentum, external_momenta=momenta,
                external_ground_parities=pol.antiholomorphic_ground_parities,
                maximum_twice_level=maximum_twice_level,
                component=a_component, left_structure_sign=signs[0],
                right_structure_sign=signs[1], chirality="antiholomorphic",
                digits=digits, condition_limit=condition_limit,
            )
        free = four_ramond_free_sector_trinion_product(
            pol.time_ising_parities,
            time_channel=parity.time_channel,
            spin7_channel=parity.spin7_channel,
            fermion_turning_phase=tau,
        )
        coefficient = pol.coefficient * structures[signs] * free
        if coefficient:
            terms.append(FixedPFourRamondKernelTerm(
                pol, parity, signs[0], signs[1], coefficient,
                cache[hkey], cache[akey], parity.time_channel,
                spin_blocks[parity.spin7_channel],
            ))
    return FixedPFourRamondKernel(
        families=ordered_families,
        internal_momentum=complex(internal_momentum),
        time_momenta=times,
        fermion_turning_phase=tau,
        sld_block_backend=sld_block_backend,
        terms=tuple(terms),
    )


__all__ = [
    "Chirality",
    "FourRamondComponent",
    "FourRamondComponentSewingTerm",
    "FourRamondParitySewing",
    "FourRamondPolarizationComponent",
    "GeneralFourRamondNSBlockSeries",
    "FixedPFourRamondKernel",
    "FixedPFourRamondKernelTerm",
    "SPIN7_BILINEAR_EXCHANGE_SIGNS",
    "SPIN7_FIERZ_02_TO_01",
    "SPIN7_KZ_PAIR_0Z",
    "SPIN7_KZ_PAIR_Z1",
    "SPIN7_TENSOR_RANKS",
    "TIME_ISING_RNSR_PHASE_MATRIX",
    "Spin7AffineChannel",
    "SLDBlockBackend",
    "Spin7SpectatorBlockSeries",
    "TimeIsingChannel",
    "allowed_four_ramond_parity_sewings",
    "bosonized_time_ising_turning_phase",
    "build_fixed_p_four_ramond_kernel",
    "calibrated_spin7_rrns_trinion_product",
    "four_ramond_free_sector_trinion_product",
    "four_ramond_superghost_factor",
    "four_ramond_component_sewing_terms",
    "general_four_ramond_ns_block_series",
    "general_four_ramond_sld_chiral_block",
    "physical_four_ramond_components",
    "normalized_time_ising_scalar_block",
    "spin7_cayley_contact_direction",
    "spin7_pairing_change_matrix",
    "spin7_reexpress_pairing_coefficients",
    "spin7_spectator_block_series",
    "time_ising_channel_has_ground_support",
    "time_ising_rrns_trinion_product",
]
