#!/usr/bin/env python3
r"""Factorization data and explicitly qualified pole lifts for SO(7) x E8.

The functions in this module return residues and declared off-divisor lifts.
They do *not* supply the regular contact terms of the integrated four-point
amplitudes.  The Ramond-incoming NS-pair residue is certified directly from
the repository's tested continuum cubics.  Mixed first-discrete-R residues
remain conditional on separately declared discrete trinion data which have
not yet been independently derived in this repository.

Conventions
-----------
The fixed-incoming two-Ramond ordering is

``B0(p0) -> B1(p1) + R2(p2) + R3(p3)``,

with ``p0=p1+p2+p3``.  The first NS and mixed-Ramond divisors are

``d23 = 1+i*(p2+p3)``,
``D12 = 1/2+i*(p1+p2)``, and
``D13 = 1/2+i*(p1+p3)``.

The scalar functions multiply

``SSRR: C F_SS``,
``SVRR: (C gamma^a) F_SV``,
``VSRR: (C gamma^a) F_VS``, and
``VVRR: delta^{ab} C F_0 + (C gamma^{ab}) F_2``.

All formulas use the raw singlet-descendant convention and omit the common
sphere normalization, coupling, energy delta function, and reflection
phases, exactly as the neighboring SO(7) amplitude modules do.

There is a second, inequivalent fixed-incoming convention used by
``evaluate_two_ramond_liouville_convergent``:

``R4(p4) -> R1(p1) + N2(p2) + N3(p3)``,

with punctures ``(R1,N2,N3,R4)=(0,z,1,infinity)`` and signed time momenta
``(p1,p2,p3,-p4)``.  Its pole data are returned by the explicitly named
``ramond_incoming_kernel_*`` functions below.  They are not obtained by
feeding ``(p4,p1,p2,p3)`` to the boson-incoming functions: the NS-pair
channel then contains a Ramond-incoming RRNS trinion, and the exterior
mixed channel is an NS-leg exchange, not an exchange of the two Ramond
legs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from so7e8_four_ramond_assembly import (
    SPIN7_BILINEAR_EXCHANGE_SIGNS,
    spin7_reexpress_pairing_coefficients,
)


KAPPA = 1.0 / (4.0 * math.sqrt(2.0))

# The conditional screened-ground-ring ansatz takes O_12 O_21=mu.  In its
# renormalized mu=1 convention, the two declared vertices and inverse BPZ
# pairing give one common holomorphic sewing factor.  A simultaneous
# rephasing is exposed as ``ramond_phase`` below.
DECLARED_MIXED_RAMOND_SEWING_SCALE = 8.0 * math.pi * KAPPA**2  # pi/4

# In the reduced normalization used throughout this repository, a local
# degeneration gives
#
#   integral d^2w |w|^(-2+2D) = pi/D + regular.
#
# The declared normalization takes the first discrete-R trinions to be one
# half of the ``a_X`` coefficients below.  Their product would then carry
# pi*(1/2)^2=pi/4.  This
# is the magnitude ``DECLARED_MIXED_RAMOND_SEWING_SCALE``.  Its plumbing
# phase/sign is part of the unresolved discrete-trinion qualification.
LOCAL_MODULUS_POLE_SCALE = math.pi
MIXED_DISCRETE_TRINION_NORMALIZATION = 0.5
RAMOND_INCOMING_NS_PAIR_STATUS = "certified_from_tested_continuum_cubics"
RAMOND_INCOMING_MIXED_RESIDUE_STATUS = (
    "unresolved_strong_full_liouville_evidence_for_zero_but_no_analytic_gso_proof"
)


@dataclass(frozen=True)
class TwoRamondScalarFunctions:
    """The five fixed-incoming coefficients in the two-Ramond sector."""

    F_SS: complex
    F_SV: complex
    F_VS: complex
    F_0: complex
    F_2: complex

    def as_tuple(self) -> tuple[complex, complex, complex, complex, complex]:
        return self.F_SS, self.F_SV, self.F_VS, self.F_0, self.F_2


@dataclass(frozen=True)
class FourRamondPairResidue:
    r"""Coefficients of the rank-one and rank-zero pairing tensors.

    For outgoing pair ``(i,j)`` and remaining outgoing leg ``k``, the
    residue is

    ``vector * (C gamma^a)_(0k) (C gamma^a)_(ij)``
    ``+ scalar * C_(0k) C_(ij)``.
    """

    vector: complex
    scalar: complex


EvaluatorPairing = Literal["01|23", "02|13", "03|12"]


@dataclass(frozen=True)
class FourRamondNSPairPoleTerm:
    r"""One certified NS-pair residue in the evaluator's fixed rank basis.

    Evaluator legs are ``(R1,R2,R3,R4)=(0,z,1,infinity)``.  ``pair`` and
    ``remaining_leg`` use the finite one-based labels ``1,2,3``; leg four
    is the incoming Ramond state.  ``fixed_basis_residue`` multiplies the
    factorial-normalized ranks ``T_0,...,T_3`` in pairing ``(01)(23)``.
    """

    pair: tuple[int, int]
    remaining_leg: int
    divisor: complex
    source_pairing: EvaluatorPairing
    source_orientation_signs: tuple[int, int, int, int]
    local_residue: FourRamondPairResidue
    fixed_basis_residue: tuple[complex, complex, complex, complex]


@dataclass(frozen=True)
class FourRamondNSPairPoleLift:
    """The declared off-divisor sum of all three certified pair poles."""

    external_momenta: tuple[complex, complex, complex, complex]
    terms: tuple[
        FourRamondNSPairPoleTerm,
        FourRamondNSPairPoleTerm,
        FourRamondNSPairPoleTerm,
    ]
    coefficients: tuple[complex, complex, complex, complex]


@dataclass(frozen=True)
class FourRamondNSPairPoleSubtraction:
    """Numerical coefficient vector split into certified poles plus contact."""

    numerical_coefficients: tuple[complex, complex, complex, complex]
    pole_lift: FourRamondNSPairPoleLift
    contact_remainder: tuple[complex, complex, complex, complex]


@dataclass(frozen=True)
class ConditionalRamondIncomingKernelPoleData:
    r"""Conditional first-pole data in the numerical kernel ordering.

    The ordering is ``(R1,N2,N3,R4)=(0,z,1,infinity)`` with ``R4``
    incoming.  ``z0`` is the mixed ``R1 N2`` channel, ``z1`` is the NS
    ``N2 N3`` channel, and ``zinfinity`` is the complementary mixed
    ``R1 N3`` channel.  All tensor coefficients use the numerical
    evaluator's convention

    ``C_(infinity,zero)``, ``(C gamma^a)_(infinity,zero)``,
    ``delta^(ab) C_(infinity,zero)``, and
    ``(C gamma^(ab))_(infinity,zero)``,

    where ``a`` labels the NS field at ``z`` and ``b`` the one at ``1``.
    """

    z0_divisor: complex
    z1_divisor: complex
    zinfinity_divisor: complex
    z0: TwoRamondScalarFunctions
    z1: TwoRamondScalarFunctions
    zinfinity: TwoRamondScalarFunctions

    def pole_lift(self) -> TwoRamondScalarFunctions:
        """Return the conditional sum of the three simple-pole terms."""

        if (
            self.z0_divisor == 0
            or self.z1_divisor == 0
            or self.zinfinity_divisor == 0
        ):
            raise ZeroDivisionError("the pole lift is undefined on a divisor")
        return TwoRamondScalarFunctions(
            *(
                r0 / self.z0_divisor
                + r1 / self.z1_divisor
                + rinf / self.zinfinity_divisor
                for r0, r1, rinf in zip(
                    self.z0.as_tuple(),
                    self.z1.as_tuple(),
                    self.zinfinity.as_tuple(),
                    strict=True,
                )
            )
        )


def declared_mixed_singlet_trinion(momentum: complex) -> complex:
    r"""Return the unverified singlet--R--discrete-R coefficient."""

    momentum = complex(momentum)
    return 2.0j * math.sqrt(14.0) * (momentum + 1.0j) / 3.0


def declared_mixed_spinor8_trinion(momentum: complex) -> complex:
    r"""Return the unverified SO(7) spinor-8 vector trinion coefficient."""

    momentum = complex(momentum)
    if momentum == 0:
        raise ZeroDivisionError("the spinor-8 coefficient is singular at p=0")
    return 4.0 * (2.0 + 7.0j * momentum) / (
        3.0 * math.sqrt(2.0) * momentum
    )


def declared_mixed_spinor48_trinion(momentum: complex) -> complex:
    r"""Return the unverified gamma-traceless-48 trinion coefficient."""

    momentum = complex(momentum)
    if momentum == 0:
        raise ZeroDivisionError("the spinor-48 coefficient is singular at p=0")
    return -4.0 / momentum


def ns_pair_divisor(momentum_one: complex, momentum_two: complex) -> complex:
    """Return the first NS-channel divisor ``1+i*(p_one+p_two)``."""

    return 1.0 + 1.0j * (complex(momentum_one) + complex(momentum_two))


def mixed_ramond_divisor(
    momentum_ns: complex,
    momentum_ramond: complex,
) -> complex:
    """Return the first mixed R--NS divisor ``1/2+i*(p_NS+p_R)``."""

    return 0.5 + 1.0j * (complex(momentum_ns) + complex(momentum_ramond))


def ramond_incoming_kernel_divisors(
    p_r_zero: complex,
    p_ns_z: complex,
    p_ns_one: complex,
) -> tuple[complex, complex, complex]:
    r"""Return the ``(z0,z1,zinfinity)`` divisors of the numerical kernel.

    The infinity divisor is written using the complementary finite pair
    ``R1 N3``.  On ``p4=p1+p2+p3`` this is the physical continuation of
    the local ``N2 R4`` degeneration.
    """

    p_r_zero, p_ns_z, p_ns_one = map(
        complex, (p_r_zero, p_ns_z, p_ns_one)
    )
    return (
        mixed_ramond_divisor(p_r_zero, p_ns_z),
        ns_pair_divisor(p_ns_z, p_ns_one),
        mixed_ramond_divisor(p_r_zero, p_ns_one),
    )


def ramond_incoming_kernel_mixed_radial_denominator(
    p_ramond: complex,
    p_ns: complex,
) -> complex:
    r"""Return the exact level-one mixed-channel radial denominator.

    At the degenerate internal super-Liouville momentum ``P=3i/2``, the
    internal Ramond weight is ``-9/16``.  Adding level one in both chiral
    blocks and the external PCO/spectator shifts gives

    ``chi = -(p_R+p_NS)^2 - 1/4 = -D*(1-D)``,

    where ``D=1/2+i*(p_R+p_NS)``.  Thus the first divisor used by the
    evaluator is exactly ``D=0``.  This identity fixes the pole location,
    but not its coefficient: the required nonchiral degenerate projector is
    not yet certified.
    """

    total = complex(p_ramond) + complex(p_ns)
    return -(total * total) - 0.25


def ramond_incoming_kernel_ns_pair_residues(
    p_r_zero: complex,
    p_ns_z: complex,
    p_ns_one: complex,
    p_r_infinity: complex,
) -> TwoRamondScalarFunctions:
    r"""Return the ``z -> 1`` residues for an incoming Ramond leg.

    Set ``q=p_ns_z+p_ns_one=i`` on the NS first-pole divisor.  The two
    reduced trinions are an NS ``B(q)->N2+N3`` amplitude and the certified
    Ramond-incoming ``R4->R1+B(q)`` amplitude.  Multiplication by the local
    radial residue ``pi`` gives

    ``(+pi*kappa*(p2*p3)^2*(p4+p1), -pi*kappa*p2*p3,``
    `` -pi*kappa*p2*p3, -pi*kappa*p2*p3*(p4+p1), 0)``.

    The plus sign in ``F_SS`` follows from the raw-singlet Ward factor
    ``h(q)+h(p2)+h(p3)-1/2=-p2*p3`` at ``q=i``.  In particular, the
    Ramond factor is ``p4+p1``; the ``p2-p3`` factor in
    :func:`boson_incoming_rr_pair_residues` belongs to a boson-incoming RR
    trinion and is not applicable here.
    """

    p_r_zero, p_ns_z, p_ns_one, p_r_infinity = map(
        complex, (p_r_zero, p_ns_z, p_ns_one, p_r_infinity)
    )
    common = LOCAL_MODULUS_POLE_SCALE * KAPPA
    ns_product = p_ns_z * p_ns_one
    ramond_sum = p_r_infinity + p_r_zero
    return TwoRamondScalarFunctions(
        F_SS=common * ns_product**2 * ramond_sum,
        F_SV=-common * ns_product,
        F_VS=-common * ns_product,
        F_0=-common * ns_product * ramond_sum,
        F_2=0.0j,
    )


def conditional_ramond_incoming_kernel_mixed_z0_residues(
    p_ns_z: complex,
    p_ns_one: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    r"""Return conditional first mixed-R residues at ``z -> 0``.

    This is conditional trinion sewing in the tensor order of the numerical
    kernel.  The asserted ``a_S,a_8,a_48`` coefficients have no independent
    descendant or ground-ring derivation in this repository, and the direct
    nonchiral projector needed to extract them from the spectral integrand is
    uncertified.  If those coefficients and their plumbing phase are
    nevertheless assumed, the physical discrete trinions are ``a_X/2`` and
    the vector-spinor quotient gives

    ``F_SS = pi*a_S(p2)*a_S(p3)/4``,
    ``F_SV = pi*a_S(p2)*a_8(p3)/(4*sqrt(7))``,
    ``F_VS = pi*a_8(p2)*a_S(p3)/(4*sqrt(7))``,

    and the standard ``8 plus 48`` projector combinations for ``F_0,F_2``.
    """

    p_ns_z, p_ns_one = map(complex, (p_ns_z, p_ns_one))
    if p_ns_z == 0 or p_ns_one == 0:
        raise ZeroDivisionError(
            "the displayed vector discrete trinions require p_ns_z*p_ns_one != 0"
        )
    scale = complex(ramond_phase) * DECLARED_MIXED_RAMOND_SEWING_SCALE
    a_s_z = declared_mixed_singlet_trinion(p_ns_z)
    a_s_one = declared_mixed_singlet_trinion(p_ns_one)
    a_8_z = declared_mixed_spinor8_trinion(p_ns_z)
    a_8_one = declared_mixed_spinor8_trinion(p_ns_one)
    a_48_z = declared_mixed_spinor48_trinion(p_ns_z)
    a_48_one = declared_mixed_spinor48_trinion(p_ns_one)
    return TwoRamondScalarFunctions(
        F_SS=scale * a_s_z * a_s_one,
        F_SV=scale * a_s_z * a_8_one / math.sqrt(7.0),
        F_VS=scale * a_8_z * a_s_one / math.sqrt(7.0),
        F_0=scale * (a_8_z * a_8_one + 6.0 * a_48_z * a_48_one) / 7.0,
        F_2=scale * (a_8_z * a_8_one - a_48_z * a_48_one) / 7.0,
    )


def conditional_ramond_incoming_kernel_mixed_zinfinity_residues(
    p_ns_z: complex,
    p_ns_one: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    r"""Return conditional first mixed-R residues at ``z -> infinity``.

    Scaling by ``1/z`` turns this channel into the ``z -> 0`` channel with
    NS legs 2 and 3 exchanged while the Ramond endpoints stay fixed.  The
    scalar, one-vector, and ``delta C`` tensors are unchanged.  Since
    ``gamma^(ba)=-gamma^(ab)``, only the displayed ``F_2`` coefficient
    changes sign.  This is *not*
    :func:`conditional_boson_incoming_mixed_pair_13_residues`, which
    exchanges two outgoing Ramond fermions in the boson-incoming convention.
    """

    swapped = conditional_ramond_incoming_kernel_mixed_z0_residues(
        p_ns_one,
        p_ns_z,
        ramond_phase=ramond_phase,
    )
    return TwoRamondScalarFunctions(
        F_SS=swapped.F_SS,
        F_SV=swapped.F_VS,
        F_VS=swapped.F_SV,
        F_0=swapped.F_0,
        F_2=-swapped.F_2,
    )


def conditional_ramond_incoming_kernel_pole_data(
    p_r_zero: complex,
    p_ns_z: complex,
    p_ns_one: complex,
    p_r_infinity: complex,
    *,
    ramond_phase: complex = 1.0,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> ConditionalRamondIncomingKernelPoleData:
    r"""Return conditional first-pole data in the evaluator's ordering.

    The ``z1`` field is certified; ``z0`` and ``zinfinity`` are conditional
    on the unverified discrete trinions.  The condition
    ``p_r_infinity=p_r_zero+p_ns_z+p_ns_one`` is required by default,
    matching :func:`evaluate_two_ramond_liouville_convergent`.  The residues
    are restrictions to their respective divisors; the returned formulas
    are one polynomial/rational continuation away from those divisors.
    """

    p1, p2, p3, p4 = map(
        complex, (p_r_zero, p_ns_z, p_ns_one, p_r_infinity)
    )
    if check_energy_conservation:
        expected = p1 + p2 + p3
        scale = max(1.0, abs(expected), abs(p4))
        if abs(p4 - expected) > tolerance * scale:
            raise ValueError(
                "the kernel ordering requires p_r_infinity="
                "p_r_zero+p_ns_z+p_ns_one"
            )
    d0, d1, dinfinity = ramond_incoming_kernel_divisors(p1, p2, p3)
    return ConditionalRamondIncomingKernelPoleData(
        z0_divisor=d0,
        z1_divisor=d1,
        zinfinity_divisor=dinfinity,
        z0=conditional_ramond_incoming_kernel_mixed_z0_residues(
            p2, p3, ramond_phase=ramond_phase
        ),
        z1=ramond_incoming_kernel_ns_pair_residues(p1, p2, p3, p4),
        zinfinity=conditional_ramond_incoming_kernel_mixed_zinfinity_residues(
            p2, p3, ramond_phase=ramond_phase
        ),
    )


def conditional_ramond_incoming_kernel_pole_lift(
    p_r_zero: complex,
    p_ns_z: complex,
    p_ns_one: complex,
    p_r_infinity: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    r"""Return the conditional three-pole lift in the evaluator ordering."""

    return conditional_ramond_incoming_kernel_pole_data(
        p_r_zero,
        p_ns_z,
        p_ns_one,
        p_r_infinity,
        ramond_phase=ramond_phase,
    ).pole_lift()


def boson_incoming_rr_pair_residues(
    p0: complex,
    p1: complex,
    p2: complex,
    p3: complex,
) -> TwoRamondScalarFunctions:
    r"""Return coefficients of ``1/d23`` from continuum cubic sewing.

    The values are restrictions to ``d23=0``.  ``p2`` and ``p3`` are kept
    separate because the scalar and singlet residues are antisymmetric under
    exchange of the two outgoing fermions.
    """

    p0, p1, p2, p3 = map(complex, (p0, p1, p2, p3))
    common = -math.pi * KAPPA
    return TwoRamondScalarFunctions(
        F_SS=common * (p0 * p1) ** 2 * (p2 - p3),
        F_SV=common * p0 * p1,
        F_VS=common * p0 * p1,
        F_0=common * p0 * p1 * (p2 - p3),
        F_2=0.0j,
    )


def conditional_boson_incoming_mixed_pair_12_residues(
    p0: complex,
    p1: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    r"""Return conditional boson-incoming coefficients of ``1/D12``.

    The SO(7) level-one BRST projection contains a spinor ``8`` and a
    gamma-traceless ``48``.  The expressions below assume the declared
    discrete trinion coefficients and plumbing scale above; neither has an
    independent executable derivation here.  ``ramond_phase`` is one common
    cocycle/BPZ phase and cannot alter relative tensor coefficients.
    """

    p0, p1 = map(complex, (p0, p1))
    if p0 == 0 or p1 == 0:
        raise ZeroDivisionError("the displayed residue coordinates require p0*p1 != 0")
    scale = complex(ramond_phase) * DECLARED_MIXED_RAMOND_SEWING_SCALE
    return TwoRamondScalarFunctions(
        F_SS=scale * (-56.0 / 9.0) * (p0 + 1.0j) * (p1 + 1.0j),
        F_SV=(
            scale
            * (8.0j / 9.0)
            * (p0 + 1.0j)
            * (2.0 + 7.0j * p1)
            / p1
        ),
        F_VS=(
            scale
            * (8.0j / 9.0)
            * (p1 + 1.0j)
            * (2.0 + 7.0j * p0)
            / p0
        ),
        F_0=(
            scale
            * 8.0
            / (9.0 * p0 * p1)
            * (16.0 + 2.0j * (p0 + p1) - 7.0 * p0 * p1)
        ),
        F_2=(
            scale
            * 8.0
            / (9.0 * p0 * p1)
            * (-2.0 + 2.0j * (p0 + p1) - 7.0 * p0 * p1)
        ),
    )


def conditional_boson_incoming_mixed_pair_13_residues(
    p0: complex,
    p1: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    """Return coefficients of ``1/D13`` in the same fixed tensor ordering.

    Exchanging the two outgoing fermions flips ``F_SS`` and ``F_0`` and
    leaves ``F_SV``, ``F_VS``, and ``F_2`` unchanged.
    """

    direct = conditional_boson_incoming_mixed_pair_12_residues(
        p0, p1, ramond_phase=ramond_phase
    )
    return TwoRamondScalarFunctions(
        F_SS=-direct.F_SS,
        F_SV=direct.F_SV,
        F_VS=direct.F_VS,
        F_0=-direct.F_0,
        F_2=direct.F_2,
    )


def conditional_boson_incoming_two_ramond_pole_lift(
    p0: complex,
    p1: complex,
    p2: complex,
    p3: complex,
    *,
    ramond_phase: complex = 1.0,
) -> TwoRamondScalarFunctions:
    r"""Return a conditional boson-incoming lift of the first-pole data.

    This is the sum ``R23/d23 + R12/D12 + R13/D13``.  The mixed residues
    use the unverified discrete trinions declared in this module.  Extending
    a residue away from its divisor is also not unique: changing this lift
    changes the regular contact term.  Accordingly this function is
    explicitly named ``conditional`` and is not an amplitude.
    """

    p0, p1, p2, p3 = map(complex, (p0, p1, p2, p3))
    d23 = ns_pair_divisor(p2, p3)
    d12 = mixed_ramond_divisor(p1, p2)
    d13 = mixed_ramond_divisor(p1, p3)
    if d23 == 0 or d12 == 0 or d13 == 0:
        raise ZeroDivisionError("the pole lift is undefined exactly on a divisor")
    rr = boson_incoming_rr_pair_residues(p0, p1, p2, p3)
    r12 = conditional_boson_incoming_mixed_pair_12_residues(
        p0, p1, ramond_phase=ramond_phase
    )
    r13 = conditional_boson_incoming_mixed_pair_13_residues(
        p0, p1, ramond_phase=ramond_phase
    )
    return TwoRamondScalarFunctions(
        *(a / d23 + b / d12 + c / d13 for a, b, c in zip(
            rr.as_tuple(), r12.as_tuple(), r13.as_tuple(), strict=True
        ))
    )


def minimal_two_ramond_contact_ansatz(
    p0: complex,
    p1: complex,
    p2: complex,
    p3: complex,
    *,
    c_ss: complex,
    c_sv: complex,
    c_0: complex,
    c_2: complex,
) -> TwoRamondScalarFunctions:
    r"""Return the lowest-degree soft and fermion-exchange contact ansatz.

    This is an optional reconstruction assumption, not a derived amplitude.
    It imposes regularity, a factor ``p0*p1`` for the two NS legs, and the
    required symmetry under ``R2 <-> R3``.  The mixed fixed-incoming
    crossings share ``c_sv``.  More general regular symmetric momentum
    functions are not excluded by factorization.
    """

    p0, p1, p2, p3 = map(complex, (p0, p1, p2, p3))
    soft = p0 * p1
    difference = p2 - p3
    return TwoRamondScalarFunctions(
        F_SS=complex(c_ss) * soft * difference,
        F_SV=complex(c_sv) * soft,
        F_VS=complex(c_sv) * soft,
        F_0=complex(c_0) * soft * difference,
        F_2=complex(c_2) * soft,
    )


def _epsilon_123(i: int, j: int, k: int) -> int:
    if sorted((i, j, k)) != [1, 2, 3]:
        raise ValueError("(i,j,k) must be a permutation of outgoing labels (1,2,3)")
    inversions = sum(
        first > second
        for offset, first in enumerate((i, j, k))
        for second in (i, j, k)[offset + 1 :]
    )
    return -1 if inversions % 2 else 1


def four_ramond_pair_residue(
    external_momenta: Sequence[complex],
    *,
    pair: tuple[int, int],
) -> FourRamondPairResidue:
    r"""Return the first NS-pair pole residue of the four-R amplitude.

    ``external_momenta`` is ``(p0,p1,p2,p3)``.  ``pair=(i,j)`` is an
    ordered pair of outgoing labels; ``k`` is the remaining label.  In the
    raw three-point normalization,

    ``Res = -pi/32 epsilon_ijk [B1 + (p_i-p_j)(p0+p_k) B0]``.
    """

    momenta = tuple(map(complex, external_momenta))
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p0,p1,p2,p3)")
    i, j = pair
    if i == j or i not in (1, 2, 3) or j not in (1, 2, 3):
        raise ValueError("pair must contain two distinct outgoing labels")
    k = next(label for label in (1, 2, 3) if label not in pair)
    sign = _epsilon_123(i, j, k)
    common = -math.pi * sign / 32.0
    return FourRamondPairResidue(
        vector=common,
        scalar=common * (momenta[i] - momenta[j]) * (momenta[0] + momenta[k]),
    )


_EVALUATOR_CANONICAL_PAIRINGS: dict[
    EvaluatorPairing,
    tuple[tuple[int, int], tuple[int, int]],
] = {
    # The names are the zero-based names used by the Spin(7) assembly; the
    # ordered pairs below use the evaluator's one-based R1,...,R4 labels.
    # Importantly, the assembly's ``03|12`` matrix is D F D and therefore
    # represents the oriented tensor (R4,R1)(R2,R3), not the literal
    # (R1,R4)(R2,R3), whose matrix would instead be D F.
    "01|23": ((1, 2), (3, 4)),
    "02|13": ((1, 3), (2, 4)),
    "03|12": ((4, 1), (2, 3)),
}


def _four_ramond_evaluator_momenta(
    external_momenta: Sequence[complex],
    *,
    check_energy_conservation: bool,
    tolerance: float,
) -> tuple[complex, complex, complex, complex]:
    momenta = tuple(map(complex, external_momenta))
    if len(momenta) != 4:
        raise ValueError(
            "external_momenta must contain evaluator-ordered (p1,p2,p3,p4)"
        )
    if any(
        not (math.isfinite(momentum.real) and math.isfinite(momentum.imag))
        for momentum in momenta
    ):
        raise ValueError("external_momenta must be finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if check_energy_conservation:
        expected = momenta[0] + momenta[1] + momenta[2]
        scale = max(1.0, abs(expected), abs(momenta[3]))
        if abs(momenta[3] - expected) > tolerance * scale:
            raise ValueError(
                "the evaluator ordering requires incoming p4=p1+p2+p3"
            )
    return momenta  # type: ignore[return-value]


def _evaluator_pairing_and_orientation(
    pair: tuple[int, int],
) -> tuple[int, EvaluatorPairing, tuple[int, int, int, int]]:
    """Map ``(R4,Rk)(Ri,Rj)`` to one canonical evaluator pairing."""

    i, j = pair
    if i == j or i not in (1, 2, 3) or j not in (1, 2, 3):
        raise ValueError("pair must contain two distinct finite labels from 1,2,3")
    k = next(label for label in (1, 2, 3) if label not in pair)
    residue_pairs = ((4, k), (i, j))
    residue_partition = {frozenset(entry) for entry in residue_pairs}

    source_pairing: EvaluatorPairing | None = None
    canonical_pairs: tuple[tuple[int, int], tuple[int, int]] | None = None
    for name, candidate in _EVALUATOR_CANONICAL_PAIRINGS.items():
        if {frozenset(entry) for entry in candidate} == residue_partition:
            source_pairing = name
            canonical_pairs = candidate
            break
    if source_pairing is None or canonical_pairs is None:
        raise AssertionError("the four external legs did not define a pairing")

    reversals = 0
    for ordered_pair in residue_pairs:
        canonical = next(
            entry
            for entry in canonical_pairs
            if frozenset(entry) == frozenset(ordered_pair)
        )
        if ordered_pair == canonical:
            continue
        if ordered_pair == canonical[::-1]:
            reversals += 1
            continue
        raise AssertionError("an oriented bilinear did not match its pairing")
    orientation = tuple(
        int(sign) ** reversals for sign in SPIN7_BILINEAR_EXCHANGE_SIGNS
    )
    return k, source_pairing, orientation  # type: ignore[return-value]


def four_ramond_ns_pair_residue_in_evaluator_basis(
    external_momenta: Sequence[complex],
    *,
    pair: tuple[int, int],
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> FourRamondNSPairPoleTerm:
    r"""Return one tested pair residue in the evaluator's rank basis.

    The input order is ``(p1,p2,p3,p4)`` for
    ``(R1,R2,R3,R4)=(0,z,1,infinity)``, with ``R4`` incoming.  The tested
    process is ``Psi(R4) -> Psi_tilde(R1) Psi_tilde(R2) Psi_tilde(R3)``
    (or its simultaneous CP-reflected orientation).  The tested
    three-point helper instead calls the incoming momentum ``p0`` and the
    three finite outgoing legs ``p1,p2,p3``.  This function performs that
    map, the bilinear-orientation signs, and the exact Spin(7) Fierz change.

    Reversing ``pair`` gives the same fixed-basis residue: the tested vector
    coefficient and its antisymmetric pair bilinear both change sign.
    """

    momenta = _four_ramond_evaluator_momenta(
        external_momenta,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    k, source_pairing, orientation = _evaluator_pairing_and_orientation(pair)
    p1, p2, p3, p4 = momenta
    local = four_ramond_pair_residue(
        (p4, p1, p2, p3),
        pair=pair,
    )
    source_coefficients = np.asarray(
        (local.scalar, local.vector, 0.0j, 0.0j),
        dtype=np.complex128,
    ) * np.asarray(orientation, dtype=np.complex128)
    fixed = spin7_reexpress_pairing_coefficients(
        source_coefficients,
        source_pairing=source_pairing,
        target_pairing="01|23",
    )
    i, j = pair
    return FourRamondNSPairPoleTerm(
        pair=(i, j),
        remaining_leg=k,
        divisor=ns_pair_divisor(momenta[i - 1], momenta[j - 1]),
        source_pairing=source_pairing,
        source_orientation_signs=orientation,
        local_residue=local,
        fixed_basis_residue=tuple(complex(value) for value in fixed),  # type: ignore[arg-type]
    )


def four_ramond_ns_pair_pole_lift(
    external_momenta: Sequence[complex],
    *,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> FourRamondNSPairPoleLift:
    r"""Return the sum of the three certified first NS-pair poles.

    The cyclic representatives are ``(1,2)``, ``(2,3)``, and ``(3,1)``.
    Extending each on-divisor residue by the polynomial returned by
    :func:`four_ramond_pair_residue` is a declared lift, not a unique
    separation of pole and contact terms.  No regular tensor is added.
    """

    momenta = _four_ramond_evaluator_momenta(
        external_momenta,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    terms = tuple(
        four_ramond_ns_pair_residue_in_evaluator_basis(
            momenta,
            pair=pair,
            check_energy_conservation=False,
        )
        for pair in ((1, 2), (2, 3), (3, 1))
    )
    if any(term.divisor == 0 for term in terms):
        raise ZeroDivisionError("the four-R pole lift is undefined on a pair divisor")
    coefficients = sum(
        (
            np.asarray(term.fixed_basis_residue, dtype=np.complex128)
            / term.divisor
            for term in terms
        ),
        np.zeros(4, dtype=np.complex128),
    )
    return FourRamondNSPairPoleLift(
        external_momenta=momenta,
        terms=terms,  # type: ignore[arg-type]
        coefficients=tuple(complex(value) for value in coefficients),  # type: ignore[arg-type]
    )


def subtract_four_ramond_ns_pair_poles(
    numerical_coefficients: Sequence[complex],
    external_momenta: Sequence[complex],
    *,
    check_energy_conservation: bool = True,
    tolerance: float = 1.0e-12,
) -> FourRamondNSPairPoleSubtraction:
    r"""Subtract all certified pair poles from a numerical rank vector.

    The returned contact remainder is simply ``numerical - pole_lift``.
    It is not projected onto the Cayley direction and no contact coefficient
    is assumed or fitted.
    """

    values = np.asarray(tuple(numerical_coefficients), dtype=np.complex128)
    if values.shape != (4,):
        raise ValueError(
            "numerical_coefficients must contain ranks 0,1,2,3 in (01)(23)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("numerical_coefficients must be finite")
    lift = four_ramond_ns_pair_pole_lift(
        external_momenta,
        check_energy_conservation=check_energy_conservation,
        tolerance=tolerance,
    )
    remainder = values - np.asarray(lift.coefficients, dtype=np.complex128)
    return FourRamondNSPairPoleSubtraction(
        numerical_coefficients=tuple(complex(value) for value in values),  # type: ignore[arg-type]
        pole_lift=lift,
        contact_remainder=tuple(complex(value) for value in remainder),  # type: ignore[arg-type]
    )


# Matter-level data for the mixed R channel.  The basis is
# A=alpha^X_-1 Omega, B=L^SL_-1 Omega, C=L^Spin7_-1 Omega.
MIXED_LEVEL_ONE_METRIC = np.diag([1.0, -9.0 / 8.0, 7.0 / 8.0])
MIXED_LEVEL_ONE_EXACT_NULL = np.asarray([0.5, 1.0, 1.0])
MIXED_LEVEL_ONE_UNIT_8 = np.asarray([9.0, 4.0, 0.0]) / math.sqrt(63.0)
MIXED_LEVEL_ONE_UNIT_8_EQUIVALENT = (
    np.asarray([7.0, 0.0, -4.0]) / math.sqrt(63.0)
)

# Coefficient of the confluent double pole of the self-dual Ramond inverse
# Gram matrix: lim_(h -> -9/16) (h+9/16)^2 B(h)^(-1).
SELF_DUAL_RAMOND_INVERSE_GRAM_DOUBLE_POLE = np.asarray(
    [
        [9.0 / 16.0, 0.0, 0.0, -3.0 / 8.0],
        [0.0, -1.0 / 2.0, -3.0 / 8.0, 0.0],
        [0.0, -3.0 / 8.0, -9.0 / 32.0, 0.0],
        [-3.0 / 8.0, 0.0, 0.0, 1.0 / 4.0],
    ]
)


__all__ = [
    "ConditionalRamondIncomingKernelPoleData",
    "DECLARED_MIXED_RAMOND_SEWING_SCALE",
    "FourRamondPairResidue",
    "FourRamondNSPairPoleLift",
    "FourRamondNSPairPoleSubtraction",
    "FourRamondNSPairPoleTerm",
    "KAPPA",
    "LOCAL_MODULUS_POLE_SCALE",
    "MIXED_DISCRETE_TRINION_NORMALIZATION",
    "MIXED_LEVEL_ONE_EXACT_NULL",
    "MIXED_LEVEL_ONE_METRIC",
    "MIXED_LEVEL_ONE_UNIT_8",
    "MIXED_LEVEL_ONE_UNIT_8_EQUIVALENT",
    "RAMOND_INCOMING_MIXED_RESIDUE_STATUS",
    "RAMOND_INCOMING_NS_PAIR_STATUS",
    "SELF_DUAL_RAMOND_INVERSE_GRAM_DOUBLE_POLE",
    "TwoRamondScalarFunctions",
    "boson_incoming_rr_pair_residues",
    "conditional_boson_incoming_mixed_pair_12_residues",
    "conditional_boson_incoming_mixed_pair_13_residues",
    "conditional_boson_incoming_two_ramond_pole_lift",
    "conditional_ramond_incoming_kernel_mixed_z0_residues",
    "conditional_ramond_incoming_kernel_mixed_zinfinity_residues",
    "conditional_ramond_incoming_kernel_pole_data",
    "conditional_ramond_incoming_kernel_pole_lift",
    "declared_mixed_singlet_trinion",
    "declared_mixed_spinor8_trinion",
    "declared_mixed_spinor48_trinion",
    "four_ramond_pair_residue",
    "four_ramond_ns_pair_pole_lift",
    "four_ramond_ns_pair_residue_in_evaluator_basis",
    "minimal_two_ramond_contact_ansatz",
    "mixed_ramond_divisor",
    "ns_pair_divisor",
    "ramond_incoming_kernel_divisors",
    "ramond_incoming_kernel_mixed_radial_denominator",
    "ramond_incoming_kernel_ns_pair_residues",
    "subtract_four_ramond_ns_pair_poles",
]
