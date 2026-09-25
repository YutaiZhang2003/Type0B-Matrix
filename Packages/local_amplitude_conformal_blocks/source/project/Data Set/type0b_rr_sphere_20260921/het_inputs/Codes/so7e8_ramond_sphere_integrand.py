#!/usr/bin/env python3
r"""Fixed-P direct-channel SO(7) x E8 two-R/two-NS sphere integrand.

The puncture order is ``(R1,N2,N3,R4)=(0,z,1,infinity)``.  Leg 2 is in
picture zero and leg 3 in picture -1.  This module performs the complete
finite-dimensional component sum at fixed internal Ramond momentum ``P``:

* both picture-zero terms;
* the physical Ramond ``U_eta`` and time-Ising matrices;
* the HJS small-representation projector;
* the correlated pair of HJS structure signs, with the ``1/4`` conversion
  from public ``C_even,C_odd`` constants to ``C^(+),C^(-)``; and
* the SO(7) spectator tensor.

It deliberately does not integrate over ``P`` or the sphere modulus.  It
also does not regularize degeneration boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Any, Literal, Mapping, Sequence

import sympy as sp

from ns_algebra.ns_sca import G
from ramond_algebra.ramond_sca import ground_state
from so7e8_ramond_cocycles import (
    PCOTimeBranch,
    RamondFamily,
    TwoRamondComponentWeight,
    two_ramond_component_weights,
)
from so7e8_ramond_fourpoint import (
    antiholomorphic_ramond_sld_chiral_branch,
    ramond_sld_structure_product,
    spin7_one_vector_correlator,
    spin7_two_vector_correlator,
    two_ramond_internal_ns_sld_chiral_block,
)
from so7e8_mixed_elliptic_recursion import (
    MixedEllipticBlockSeries,
    TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER,
    two_star_internal_ns_sld_cached_elliptic_h_series,
    two_ramond_internal_ns_sld_elliptic_h_series,
)
from ramond_sphere_uniformization import EllipticPrefactoredRamondSphereSeries
from spin23_ramond_blocks import hjs_ramond_sphere_block_series
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_chiral_structure_constant,
)
from ramond_local_series import LocalHalfLevelSeries
from sphere_block_uniformization import CutSide
from so7e8_sphere_branches import sphere_channel_coordinates


NSKind = Literal["S", "V"]
PictureZeroPosition = Literal["z", "one"]
InternalLocalRamond = Literal["R_plus", "R_minus"]
InternalNSComponent = Literal["even", "odd"]
CrossedBlockBackend = Literal["inverse_gram", "elliptic_recursion", "hybrid", "double_virasoro"]
DirectBlockBackend = Literal["inverse_gram", "double_virasoro"]

EMPTY_WORD: tuple = ()
G_WORD = (G(sp.Rational(-1, 2)),)


@dataclass(frozen=True)
class SmallRepresentationProjection:
    """One HJS local-R projector term."""

    internal_local_field: InternalLocalRamond
    holomorphic_form_parities: tuple[int, int]
    antiholomorphic_form_parities: tuple[int, int]
    phase: complex


@dataclass(frozen=True)
class FixedPComponentTerm:
    """One fully resolved summand of the fixed-P density."""

    pco_branch: PCOTimeBranch
    left_structure_sign: int
    right_structure_sign: int
    external_weight: TwoRamondComponentWeight
    projector: SmallRepresentationProjection
    holomorphic_value: complex
    antiholomorphic_value: complex
    structure_product: complex
    pco_coefficient: complex
    scalar_value_before_spectator: complex


@dataclass(frozen=True)
class FixedPTwoRamondIntegrand:
    """SO(7)-tensor coefficients at fixed ``P,z``.

    Exactly one of ``F_SS``, ``F_SV``, ``F_VS`` or the pair ``F0,F2`` is
    populated, according to ``ns_at_z,ns_at_one``.  The coefficients omit
    the integration measure ``dP/pi`` and the common sphere normalization.
    """

    ns_at_z: NSKind
    ns_at_one: NSKind
    zero_ramond_family: RamondFamily
    infinity_ramond_family: RamondFamily
    internal_momentum: complex
    z: complex
    zbar: complex
    time_factor: complex
    F_SS: complex | None
    F_SV: complex | None
    F_VS: complex | None
    F0: complex | None
    F2: complex | None
    terms: tuple[FixedPComponentTerm, ...]


@dataclass(frozen=True)
class CrossedFixedPComponentTerm:
    """One resolved summand in the crossed NS-internal channel."""

    pco_branch: PCOTimeBranch
    rr_structure_sign: int
    ns_structure_parity: int
    holomorphic_internal_component: InternalNSComponent
    antiholomorphic_internal_component: InternalNSComponent
    fixed_form_phase: complex
    external_weight: TwoRamondComponentWeight
    holomorphic_value: complex
    antiholomorphic_value: complex
    rr_structure_constant: complex
    ns_structure_constant: complex
    pco_coefficient: complex
    scalar_value_before_spectator: complex


@dataclass(frozen=True)
class CrossedFixedPTwoRamondIntegrand:
    """SO(7)-tensor coefficients in the fixed-P crossed NS channel."""

    ns_at_z: NSKind
    ns_at_one: NSKind
    zero_ramond_family: RamondFamily
    infinity_ramond_family: RamondFamily
    internal_momentum: complex
    w: complex
    wbar: complex
    time_factor: complex
    F_SS: complex | None
    F_SV: complex | None
    F_VS: complex | None
    F0: complex | None
    F2: complex | None
    terms: tuple[CrossedFixedPComponentTerm, ...]


def _validate_ns_kind(value: str, name: str) -> NSKind:
    if value not in ("S", "V"):
        raise ValueError(f"{name} must be 'S' or 'V'")
    return value  # type: ignore[return-value]


def ns_descendant_words(
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    branch: PCOTimeBranch,
    picture_zero_at: PictureZeroPosition = "z",
) -> tuple[tuple[tuple, tuple], tuple[tuple, tuple]]:
    r"""Return ``((hol2,hol3),(anti2,anti3))`` for one PCO branch."""

    state2 = _validate_ns_kind(ns_at_z, "ns_at_z")
    state3 = _validate_ns_kind(ns_at_one, "ns_at_one")
    if branch not in ("G", "psi0"):
        raise ValueError("branch must be 'G' or 'psi0'")
    if picture_zero_at not in ("z", "one"):
        raise ValueError("picture_zero_at must be 'z' or 'one'")
    hol2 = G_WORD if branch == "G" and picture_zero_at == "z" else EMPTY_WORD
    hol3 = G_WORD if branch == "G" and picture_zero_at == "one" else EMPTY_WORD
    anti2 = G_WORD if state2 == "S" else EMPTY_WORD
    anti3 = G_WORD if state3 == "S" else EMPTY_WORD
    return (hol2, hol3), (anti2, anti3)


def hjs_small_representation_projectors(
    left_structure_sign: int,
    right_structure_sign: int,
) -> tuple[SmallRepresentationProjection, ...]:
    r"""Return the eight terms of the retained, uncertified local-R ansatz.

    This is not a complete-basis derivation of the NS bulk vertex acting
    on a Ramond small representation. Its use for the mixed direct channel
    failed the picture/overlap checks. ``graded_sphere_sewing`` separates
    the actual inverse metric, trinion exchange signs and physical embedding;
    none of its signs should be bolted onto these already phase-dressed terms.

    HJS equations (4.13)--(4.15) give

    ``R+ : E Ebar - i O Obar`` and
    ``R- : E Obar + O Ebar``.

    Applying this at both trinions gives phase ``(-i)**(fL+fR)`` for the
    ``R+`` intermediate field.  For ``R-`` the coefficient of the HJS-minus
    structure changes sign at each trinion, hence the two-vertex phase
    ``sL*sR``.
    """

    if left_structure_sign not in (-1, 1) or right_structure_sign not in (-1, 1):
        raise ValueError("HJS structure signs must be +1 or -1")
    result: list[SmallRepresentationProjection] = []
    for left_form in (0, 1):
        for right_form in (0, 1):
            hol = (left_form, right_form)
            result.append(
                SmallRepresentationProjection(
                    "R_plus",
                    hol,
                    hol,
                    (-1.0j) ** (left_form + right_form),
                )
            )
            result.append(
                SmallRepresentationProjection(
                    "R_minus",
                    hol,
                    (1 - left_form, 1 - right_form),
                    complex(left_structure_sign * right_structure_sign),
                )
            )
    return tuple(result)


def suchanek_rr_vertex_form_routing(
    base_parity: int,
    holomorphic_ns_parity: int,
    antiholomorphic_ns_parity: int,
) -> tuple[int, int, complex]:
    r"""Retained trial routing for one RR--NS bulk-vertex component.

    This is not used by the production builder. The vertex-operator parity
    in this ansatz must not be identified with a homogeneous trilinear-form
    label without translating ground/dual frames and the descendant Ward
    convention. See SO7E8_GRADED_SPHERE_SEWING.md.

    Suchanek's equation ``phi_RR`` is ``V_e Vbar_e-i V_o Vbar_o``.
    Acting with ``S_-1/2`` flips the holomorphic form parity.  Acting with
    ``Sbar_-1/2`` flips the antiholomorphic parity and crosses a holomorphic
    vertex of parity ``base_parity``, producing its Koszul sign.  Therefore

    ``f_h=b xor q_h``, ``f_bar=b xor q_bar`` and
    ``phase=(-i)**b*(-1)**(b*q_bar)``.
    """

    if base_parity not in (0, 1):
        raise ValueError("base_parity must be zero or one")
    if holomorphic_ns_parity not in (0, 1) or antiholomorphic_ns_parity not in (0, 1):
        raise ValueError("NS parities must be zero or one")
    return (
        base_parity ^ holomorphic_ns_parity,
        base_parity ^ antiholomorphic_ns_parity,
        (-1.0j) ** base_parity
        * (-1.0) ** (base_parity * antiholomorphic_ns_parity),
    )


def internal_ramond_state_projectors(
    left_structure_sign: int,
    right_structure_sign: int,
    holomorphic_form_parities: tuple[int, int],
    holomorphic_ground_parities: tuple[int, int],
    antiholomorphic_ground_parities: tuple[int, int],
    holomorphic_word_parities: tuple[int, int],
    antiholomorphic_word_parities: tuple[int, int],
) -> tuple[SmallRepresentationProjection, SmallRepresentationProjection]:
    """Legacy uncertified routing by internal Ramond parities.

    Ground tuples are ordered ``(leg1,leg4)`` while form and word tuples are
    ordered ``(left,right)`` and ``(leg2,leg3)`` respectively.
    The returned structure-sign-dependent phases mix vertex-basis data with
    a purported state projection: they are not an independently derived
    inverse BPZ Gram. Kept for reproducing the failed direct-channel audit;
    the complete-state derivation is in ``graded_sphere_sewing`` and
    SO7E8_GRADED_SPHERE_SEWING.md. Physical dual-frame translation is pending.
    """

    fl, fr = holomorphic_form_parities
    gh1, gh4 = holomorphic_ground_parities
    gb1, gb4 = antiholomorphic_ground_parities
    qh2, qh3 = holomorphic_word_parities
    qb2, qb3 = antiholomorphic_word_parities
    p_left = fl ^ gh4 ^ qh3
    p_right = fr ^ gh1 ^ qh2

    def anti_forms(opposite: int) -> tuple[int, int]:
        return (
            (p_left ^ opposite) ^ gb4 ^ qb3,
            (p_right ^ opposite) ^ gb1 ^ qb2,
        )

    return (
        SmallRepresentationProjection(
            "R_plus", holomorphic_form_parities, anti_forms(0),
            (-1.0j) ** (p_left + p_right),
        ),
        SmallRepresentationProjection(
            "R_minus", holomorphic_form_parities, anti_forms(1),
            complex(left_structure_sign * right_structure_sign),
        ),
    )


def physical_structure_sign_pairs(
    zero_family: RamondFamily,
    infinity_family: RamondFamily,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two physical ``(sL,sR)`` pairs in the +P,+P convention."""

    eta1 = 1 if zero_family == "Psi" else -1
    eta4 = 1 if infinity_family == "Psi" else -1
    required_product = -eta4 * eta1
    pairs = tuple(
        (left, right)
        for left in (-1, 1)
        for right in (-1, 1)
        if left * right == required_product
    )
    return pairs  # type: ignore[return-value]


def timelike_exponential_factor(
    time_momenta: Sequence[complex],
    z: complex,
    zbar: complex | None = None,
    *,
    tolerance: float = 1.0e-10,
) -> complex:
    r"""Return the fixed-position timelike exponential correlator.

    For a geometric sphere slice ``zbar=conjugate(z)`` this is
    ``|z|^(-2*k1*k2)|1-z|^(-2*k2*k3)``.  The split-power form below also
    permits independent complex ``z,zbar`` continuation.
    """

    momenta = tuple(complex(value) for value in time_momenta)
    if len(momenta) != 4:
        raise ValueError("time_momenta must contain (k1,k2,k3,k4)")
    if abs(sum(momenta)) > tolerance:
        raise ValueError("signed time momenta must sum to zero")
    z_value = complex(z)
    zbar_value = z_value.conjugate() if zbar is None else complex(zbar)
    if z_value in (0, 1) or zbar_value in (0, 1):
        raise ValueError("the modulus must avoid 0 and 1")
    k1, k2, k3, _ = momenta
    return cmath.exp(
        -k1 * k2 * (cmath.log(z_value) + cmath.log(zbar_value))
        - k2
        * k3
        * (cmath.log(1 - z_value) + cmath.log(1 - zbar_value))
    )


def superghost_position_factor(
    z: complex,
    picture_zero_at: PictureZeroPosition,
) -> complex:
    r"""Return the nonconstant finite-position superghost correlator.

    With Ramond pictures ``(-1/2,-1/2)``, the NS operator which remains in
    picture ``-1`` contributes ``e^{-phi}``.  The superghost factor is a
    constant when that operator is fixed at one (``picture_zero_at='z'``),
    but is ``z**(-1/2)`` when it is the moving operator
    (``picture_zero_at='one'``).  The principal logarithm is the same branch
    used by the time-Ising and spectator square roots.
    """

    if picture_zero_at not in ("z", "one"):
        raise ValueError("picture_zero_at must be 'z' or 'one'")
    if picture_zero_at == "z":
        return 1.0 + 0.0j
    z_value = complex(z)
    if z_value == 0:
        raise ValueError("the moving picture-minus-one superghost is singular at z=0")
    return cmath.exp(-0.5 * cmath.log(z_value))


def ordered_pco_coefficient(
    branch: PCOTimeBranch,
    time_momentum: complex,
    picture_zero_at: PictureZeroPosition,
) -> complex:
    """PCO coefficient in the ordered HJS/time-Ising convention.

    The supercurrent Ward identity for R(0), NS(z), NS(1), R(infinity)
    fixes the local combination G V - k psi0 V. Moving the picture raise
    from z to one supplies an overall minus in this ordered convention.
    The coordinate-dependent square root is supplied separately by the
    time-Ising/superghost factors. See the mixed Ward regression test.
    """
    if branch not in ("G", "psi0") or picture_zero_at not in ("z", "one"):
        raise ValueError("invalid PCO branch or position")
    local = 1.0 if branch == "G" else -complex(time_momentum)
    return local if picture_zero_at == "z" else -local


def _holomorphic_block_value(
    internal_momentum: complex,
    external_momenta: tuple[complex, complex, complex, complex],
    maximum_twice_level: int,
    signs: tuple[int, int],
    ground_parities: tuple[int, int],
    words: tuple[tuple, tuple],
    forms: tuple[int, int],
    z: complex,
    digits: int,
    condition_limit: float,
) -> complex:
    p1, p2, p3, p4 = external_momenta
    scale = math.sqrt(2.0)
    block = hjs_ramond_sphere_block_series(
        c=13.5,
        beta_internal=1.0j * internal_momentum / scale,
        beta_one=1.0j * p1 / scale,
        beta_four=1.0j * p4 / scale,
        h_two=ns_weight(p2),
        h_three=ns_weight(p3),
        maximum_twice_level=maximum_twice_level,
        form_parities=forms,
        left_structure_sign=signs[0],
        right_structure_sign=signs[1],
        ramond_states=(ground_state(ground_parities[0]), ground_state(ground_parities[1])),
        ns_words=words,
        digits=digits,
        condition_limit=condition_limit,
    )
    return block.value(z)


def _holomorphic_block_series(
    internal_momentum: complex,
    external_momenta: tuple[complex, complex, complex, complex],
    maximum_twice_level: int,
    signs: tuple[int, int],
    ground_parities: tuple[int, int],
    words: tuple[tuple, tuple],
    forms: tuple[int, int],
    digits: int,
    condition_limit: float,
) -> Any:
    """Build the direct chiral series without evaluating its modulus."""

    p1, p2, p3, p4 = external_momenta
    scale = math.sqrt(2.0)
    return hjs_ramond_sphere_block_series(
        c=13.5,
        beta_internal=1.0j * internal_momentum / scale,
        beta_one=1.0j * p1 / scale,
        beta_four=1.0j * p4 / scale,
        h_two=ns_weight(p2),
        h_three=ns_weight(p3),
        maximum_twice_level=maximum_twice_level,
        form_parities=forms,
        left_structure_sign=signs[0],
        right_structure_sign=signs[1],
        ramond_states=(ground_state(ground_parities[0]), ground_state(ground_parities[1])),
        ns_words=words,
        digits=digits,
        condition_limit=condition_limit,
    )


def _antiholomorphic_block_series(
    internal_momentum: complex,
    external_momenta: tuple[complex, complex, complex, complex],
    maximum_twice_level: int,
    signs: tuple[int, int],
    ground_parities: tuple[int, int],
    words: tuple[tuple, tuple],
    forms: tuple[int, int],
    digits: int,
    condition_limit: float,
) -> Any:
    """Build the antiholomorphic direct series without evaluating it."""

    return antiholomorphic_ramond_sld_chiral_branch(
        internal_momentum,
        external_momenta=external_momenta,
        maximum_twice_level=maximum_twice_level,
        left_structure_sign=signs[0],
        right_structure_sign=signs[1],
        ns_words=words,
        form_parities=forms,
        external_ramond_ground_parities=ground_parities,
        digits=digits,
        condition_limit=condition_limit,
    )


def _direct_sld_block_series(
    internal_momentum: complex, external_momenta, maximum_twice_level: int,
    signs, ground_parities, words, forms, digits: int, condition_limit: float,
    *, chirality: str, block_backend: DirectBlockBackend,
    double_virasoro_limit_options: Mapping[str, Any] | None = None,
) -> Any:
    """Internal-R chiral layer; does not approve its nonchiral projector."""
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("invalid chirality")
    if block_backend == "double_virasoro":
        from so7e8_internal_ramond_double_virasoro import ramond_sld_double_virasoro_h_series

        return ramond_sld_double_virasoro_h_series(
            internal_momentum, external_momenta=external_momenta,
            maximum_twice_level=maximum_twice_level,
            left_structure_sign=signs[0], right_structure_sign=signs[1],
            ns_words=words, form_parities=forms,
            external_ramond_ground_parities=ground_parities, chirality=chirality,
            **dict(double_virasoro_limit_options or {}))
    if block_backend != "inverse_gram":
        raise ValueError("direct_block_backend must be 'inverse_gram' or 'double_virasoro'")
    builder = _holomorphic_block_series if chirality == "holomorphic" else _antiholomorphic_block_series
    return builder(internal_momentum, external_momenta, maximum_twice_level, signs,
                   ground_parities, words, forms, digits, condition_limit)


def _antiholomorphic_block_value(
    internal_momentum: complex,
    external_momenta: tuple[complex, complex, complex, complex],
    maximum_twice_level: int,
    signs: tuple[int, int],
    ground_parities: tuple[int, int],
    words: tuple[tuple, tuple],
    forms: tuple[int, int],
    zbar: complex,
    digits: int,
    condition_limit: float,
) -> complex:
    block = antiholomorphic_ramond_sld_chiral_branch(
        internal_momentum,
        external_momenta=external_momenta,
        maximum_twice_level=maximum_twice_level,
        left_structure_sign=signs[0],
        right_structure_sign=signs[1],
        ns_words=words,
        form_parities=forms,
        external_ramond_ground_parities=ground_parities,
        digits=digits,
        condition_limit=condition_limit,
    )
    return block.value(zbar)


def assemble_fixed_p_two_ramond_integrand(
    internal_momentum: complex,
    *,
    external_liouville_momenta: Sequence[complex],
    time_momenta: Sequence[complex],
    z: complex,
    zbar: complex | None = None,
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    zero_ramond_family: RamondFamily,
    infinity_ramond_family: RamondFamily,
    picture_zero_at: PictureZeroPosition = "z",
    maximum_twice_level: int = 0,
    digits: int = 40,
    condition_limit: float = 1.0e13,
    allow_uncertified_direct_projector: bool = False,
) -> FixedPTwoRamondIntegrand:
    r"""Assemble the experimental direct-channel fixed-P density.

    The nonchiral internal-R projector has not passed the integrated
    direct/crossed overlap test.  Callers must opt in explicitly so this
    diagnostic routine cannot be mistaken for the certified crossed path.
    """

    if not allow_uncertified_direct_projector:
        raise ValueError(
            "the direct internal-R projector is uncertified; pass "
            "allow_uncertified_direct_projector=True only for diagnostics"
        )

    state2 = _validate_ns_kind(ns_at_z, "ns_at_z")
    state3 = _validate_ns_kind(ns_at_one, "ns_at_one")
    momenta = tuple(complex(value) for value in external_liouville_momenta)
    if len(momenta) != 4:
        raise ValueError(
            "external_liouville_momenta must contain (p1,p2,p3,p4)"
        )
    times = tuple(complex(value) for value in time_momenta)
    if len(times) != 4:
        raise ValueError("time_momenta must contain (k1,k2,k3,k4)")
    z_value = complex(z)
    zbar_value = z_value.conjugate() if zbar is None else complex(zbar)
    time_factor = timelike_exponential_factor(times, z_value, zbar_value)
    picture_index = 1 if picture_zero_at == "z" else 2
    pco_time_momentum = times[picture_index]
    pco_position = z_value if picture_zero_at == "z" else 1.0
    terms: list[FixedPComponentTerm] = []

    hol_cache: dict[tuple, complex] = {}
    anti_cache: dict[tuple, complex] = {}
    structure_cache: dict[tuple[int, int], complex] = {}

    for signs in physical_structure_sign_pairs(
        zero_ramond_family, infinity_ramond_family
    ):
        structure_cache[signs] = ramond_sld_structure_product(
            internal_momentum,
            external_momenta=momenta,
            left_structure_sign=signs[0],
            right_structure_sign=signs[1],
            precision=digits,
        )
        for branch in ("G", "psi0"):
            pco_branch: PCOTimeBranch = branch
            hol_words, anti_words = ns_descendant_words(
                state2, state3, pco_branch, picture_zero_at
            )
            pco_coefficient = ordered_pco_coefficient(branch, pco_time_momentum, picture_zero_at)
            if pco_coefficient == 0:
                continue
            external_rows = two_ramond_component_weights(
                zero_ramond_family,
                infinity_ramond_family,
                branch=pco_branch,
                z=pco_position,
            )
            for external in external_rows:
                hol_grounds = (
                    external.zero_hjs_ground_parity,
                    external.infinity_hjs_ground_parity,
                )
                anti_grounds = (
                    external.zero_antiholomorphic_ground_parity,
                    external.infinity_antiholomorphic_ground_parity,
                )
                for projector in hjs_small_representation_projectors(*signs):
                    hol_key = (
                        signs,
                        hol_grounds,
                        hol_words,
                        projector.holomorphic_form_parities,
                    )
                    if hol_key not in hol_cache:
                        hol_cache[hol_key] = _holomorphic_block_value(
                            complex(internal_momentum),
                            momenta,
                            maximum_twice_level,
                            signs,
                            hol_grounds,
                            hol_words,
                            projector.holomorphic_form_parities,
                            z_value,
                            digits,
                            condition_limit,
                        )
                    anti_key = (
                        signs,
                        anti_grounds,
                        anti_words,
                        projector.antiholomorphic_form_parities,
                    )
                    if anti_key not in anti_cache:
                        anti_cache[anti_key] = _antiholomorphic_block_value(
                            complex(internal_momentum),
                            momenta,
                            maximum_twice_level,
                            signs,
                            anti_grounds,
                            anti_words,
                            projector.antiholomorphic_form_parities,
                            zbar_value,
                            digits,
                            condition_limit,
                        )
                    value = (
                        0.25
                        * external.coefficient
                        * projector.phase
                        * structure_cache[signs]
                        * hol_cache[hol_key]
                        * anti_cache[anti_key]
                        * pco_coefficient
                    )
                    if value == 0:
                        continue
                    terms.append(
                        FixedPComponentTerm(
                            pco_branch=pco_branch,
                            left_structure_sign=signs[0],
                            right_structure_sign=signs[1],
                            external_weight=external,
                            projector=projector,
                            holomorphic_value=hol_cache[hol_key],
                            antiholomorphic_value=anti_cache[anti_key],
                            structure_product=structure_cache[signs],
                            pco_coefficient=pco_coefficient,
                            scalar_value_before_spectator=value,
                        )
                    )

    sld_sum = sum((term.scalar_value_before_spectator for term in terms), 0.0j)
    common = (
        time_factor
        * superghost_position_factor(z_value, picture_zero_at)
        * sld_sum
    )
    f_ss = f_sv = f_vs = f0 = f2 = None
    if (state2, state3) == ("S", "S"):
        f_ss = common
    elif (state2, state3) == ("S", "V"):
        f_sv = common * spin7_one_vector_correlator(
            zbar_value, vector_position="one"
        )
    elif (state2, state3) == ("V", "S"):
        f_vs = common * spin7_one_vector_correlator(
            zbar_value, vector_position="z"
        )
    else:
        spectator = spin7_two_vector_correlator(zbar_value)
        f0 = common * spectator.scalar
        f2 = common * spectator.bivector

    return FixedPTwoRamondIntegrand(
        ns_at_z=state2,
        ns_at_one=state3,
        zero_ramond_family=zero_ramond_family,
        infinity_ramond_family=infinity_ramond_family,
        internal_momentum=complex(internal_momentum),
        z=z_value,
        zbar=zbar_value,
        time_factor=time_factor,
        F_SS=f_ss,
        F_SV=f_sv,
        F_VS=f_vs,
        F0=f0,
        F2=f2,
        terms=tuple(terms),
    )


def crossed_ns_component_routing(
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    branch: PCOTimeBranch,
    ns_structure_parity: int,
    picture_zero_at: PictureZeroPosition = "z",
) -> tuple[InternalNSComponent, InternalNSComponent, complex]:
    r"""Return the two crossed NS components and fixed-form phase.

    In standardized order ``(NS_zero,NS_w)=(leg3,leg2)``, let ``r`` be the
    nonchiral NSNSNS structure parity: ``r=0`` selects ``C_NS`` and ``r=1``
    selects ``Ctilde_NS``.  The authoritative fixed-form routing is

    ``qint_h = r xor qh3 xor qh2`` and
    ``qint_bar = r xor qb3 xor qb2``.

    Each chirality contributes the ordered right-trinion conversion phase
    ``(-1)**(q_w*(1-qint))``.  No RRNS phase is inserted here: its crossed
    HJS orientation is already part of the block wrapper.
    """

    if ns_structure_parity not in (0, 1):
        raise ValueError("ns_structure_parity must be zero or one")
    hol_words, anti_words = ns_descendant_words(
        ns_at_z, ns_at_one, branch, picture_zero_at
    )
    qh2, qh3 = (1 if word else 0 for word in hol_words)
    qb2, qb3 = (1 if word else 0 for word in anti_words)
    qh_internal = ns_structure_parity ^ qh3 ^ qh2
    qb_internal = ns_structure_parity ^ qb3 ^ qb2
    phase_h = -1 if qh2 * (1 - qh_internal) else 1
    phase_bar = -1 if qb2 * (1 - qb_internal) else 1
    return (
        "odd" if qh_internal else "even",
        "odd" if qb_internal else "even",
        complex(phase_h * phase_bar),
    )


def assemble_crossed_fixed_p_two_ramond_integrand(
    internal_momentum: complex,
    *,
    external_liouville_momenta: Sequence[complex],
    time_momenta: Sequence[complex],
    z: complex,
    zbar: complex | None = None,
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    zero_ramond_family: RamondFamily,
    infinity_ramond_family: RamondFamily,
    picture_zero_at: PictureZeroPosition = "z",
    maximum_twice_level: int = 0,
    digits: int = 40,
    condition_limit: float = 1.0e13,
) -> CrossedFixedPTwoRamondIntegrand:
    r"""Assemble the fixed-P ``w=1-z`` NS-internal component density.

    One public RRNS constant is converted to its HJS coefficient by the
    factor ``1/2``.  The NSNSNS constants need no additional factor.
    Holomorphic and antiholomorphic internal NS components are routed
    independently according to :func:`crossed_ns_component_routing`.
    """

    state2 = _validate_ns_kind(ns_at_z, "ns_at_z")
    state3 = _validate_ns_kind(ns_at_one, "ns_at_one")
    momenta = tuple(complex(value) for value in external_liouville_momenta)
    if len(momenta) != 4:
        raise ValueError(
            "external_liouville_momenta must contain (p1,p2,p3,p4)"
        )
    times = tuple(complex(value) for value in time_momenta)
    if len(times) != 4:
        raise ValueError("time_momenta must contain (k1,k2,k3,k4)")
    z_value = complex(z)
    zbar_value = z_value.conjugate() if zbar is None else complex(zbar)
    w = 1.0 - z_value
    wbar = 1.0 - zbar_value
    if w == 0 or wbar == 0:
        raise ValueError("the crossed modulus must avoid w=0")
    time_factor = timelike_exponential_factor(times, z_value, zbar_value)
    picture_index = 1 if picture_zero_at == "z" else 2
    pco_time_momentum = times[picture_index]
    pco_position = z_value if picture_zero_at == "z" else 1.0
    p1, p2, p3, p4 = momenta
    ns_constants = ns_structure_constants(
        p3, p2, complex(internal_momentum), precision=digits
    )
    rr_constants = {
        sign: rr_ns_chiral_structure_constant(
            p4,
            p1,
            complex(internal_momentum),
            structure_sign=sign,
            precision=digits,
        )
        for sign in (-1, 1)
    }
    hol_cache: dict[tuple, complex] = {}
    anti_cache: dict[tuple, complex] = {}
    terms: list[CrossedFixedPComponentTerm] = []

    for branch_name in ("G", "psi0"):
        branch: PCOTimeBranch = branch_name
        hol_words, anti_words = ns_descendant_words(
            state2, state3, branch, picture_zero_at
        )
        pco_coefficient = ordered_pco_coefficient(branch, pco_time_momentum, picture_zero_at)
        if pco_coefficient == 0:
            continue
        external_rows = two_ramond_component_weights(
            zero_ramond_family,
            infinity_ramond_family,
            branch=branch,
            z=pco_position,
        )
        for structure_parity in (0, 1):
            hol_component, anti_component, fixed_phase = (
                crossed_ns_component_routing(
                    state2, state3, branch, structure_parity, picture_zero_at
                )
            )
            ns_constant = ns_constants[structure_parity]
            for rr_sign in (-1, 1):
                for external in external_rows:
                    hol_grounds = (
                        external.zero_hjs_ground_parity,
                        external.infinity_hjs_ground_parity,
                    )
                    anti_grounds = (
                        external.zero_antiholomorphic_ground_parity,
                        external.infinity_antiholomorphic_ground_parity,
                    )
                    hol_key = (
                        hol_component,
                        rr_sign,
                        hol_words,
                        hol_grounds,
                    )
                    if hol_key not in hol_cache:
                        hol_cache[hol_key] = two_ramond_internal_ns_sld_chiral_block(
                            complex(internal_momentum),
                            external_momenta=momenta,
                            maximum_twice_level=maximum_twice_level,
                            component=hol_component,
                            rr_structure_sign=rr_sign,
                            chirality="holomorphic",
                            ns_words=hol_words,
                            ramond_ground_parities=hol_grounds,
                            digits=digits,
                            condition_limit=condition_limit,
                        ).value(w)
                    anti_key = (
                        anti_component,
                        rr_sign,
                        anti_words,
                        anti_grounds,
                    )
                    if anti_key not in anti_cache:
                        anti_cache[anti_key] = two_ramond_internal_ns_sld_chiral_block(
                            complex(internal_momentum),
                            external_momenta=momenta,
                            maximum_twice_level=maximum_twice_level,
                            component=anti_component,
                            rr_structure_sign=rr_sign,
                            chirality="antiholomorphic",
                            ns_words=anti_words,
                            ramond_ground_parities=anti_grounds,
                            digits=digits,
                            condition_limit=condition_limit,
                        ).value(wbar)
                    value = (
                        0.5
                        * external.coefficient
                        * fixed_phase
                        * ns_constant
                        * rr_constants[rr_sign]
                        * hol_cache[hol_key]
                        * anti_cache[anti_key]
                        * pco_coefficient
                    )
                    if value == 0:
                        continue
                    terms.append(
                        CrossedFixedPComponentTerm(
                            pco_branch=branch,
                            rr_structure_sign=rr_sign,
                            ns_structure_parity=structure_parity,
                            holomorphic_internal_component=hol_component,
                            antiholomorphic_internal_component=anti_component,
                            fixed_form_phase=fixed_phase,
                            external_weight=external,
                            holomorphic_value=hol_cache[hol_key],
                            antiholomorphic_value=anti_cache[anti_key],
                            rr_structure_constant=rr_constants[rr_sign],
                            ns_structure_constant=ns_constant,
                            pco_coefficient=pco_coefficient,
                            scalar_value_before_spectator=value,
                        )
                    )

    sld_sum = sum((term.scalar_value_before_spectator for term in terms), 0.0j)
    common = (
        time_factor
        * superghost_position_factor(z_value, picture_zero_at)
        * sld_sum
    )
    f_ss = f_sv = f_vs = f0 = f2 = None
    if (state2, state3) == ("S", "S"):
        f_ss = common
    elif (state2, state3) == ("S", "V"):
        f_sv = common * spin7_one_vector_correlator(
            zbar_value, vector_position="one"
        )
    elif (state2, state3) == ("V", "S"):
        f_vs = common * spin7_one_vector_correlator(
            zbar_value, vector_position="z"
        )
    else:
        spectator = spin7_two_vector_correlator(zbar_value)
        f0 = common * spectator.scalar
        f2 = common * spectator.bivector

    return CrossedFixedPTwoRamondIntegrand(
        ns_at_z=state2,
        ns_at_one=state3,
        zero_ramond_family=zero_ramond_family,
        infinity_ramond_family=infinity_ramond_family,
        internal_momentum=complex(internal_momentum),
        w=w,
        wbar=wbar,
        time_factor=time_factor,
        F_SS=f_ss,
        F_SV=f_sv,
        F_VS=f_vs,
        F0=f0,
        F2=f2,
        terms=tuple(terms),
    )


@dataclass(frozen=True)
class FixedPSeriesKernelTerm:
    """A z-independent nonchiral summand retaining both chiral series."""

    pco_branch: PCOTimeBranch
    coefficient: complex
    holomorphic_series: Any
    antiholomorphic_series: Any
    labels: Mapping[str, object]


@dataclass(frozen=True)
class LocalKernelProduct:
    """One factorized term ready for the local meromorphic integrators."""

    coefficient: complex
    holomorphic: LocalHalfLevelSeries
    antiholomorphic: LocalHalfLevelSeries
    labels: Mapping[str, object]


def _series_local(series: Any) -> LocalHalfLevelSeries:
    if isinstance(series, EllipticPrefactoredRamondSphereSeries):
        # The double-Virasoro H adapter retains its exact known sewing
        # coefficients, so collision/OPE work need not invert a truncated H.
        series = series.source
    h0, hw, _, _ = series.external_weights
    if hasattr(series, "ramond_states"):
        state0, _ = series.ramond_states
        wordw, _ = series.ns_words
        from ramond_algebra.ramond_sca import twice_level as r_twice_level
        from ns_algebra.ns_sca import twice_level as ns_twice_level

        effective0 = h0 + 0.5 * r_twice_level(state0.word)
        effectivew = hw + (0.5 * ns_twice_level(wordw) if wordw else 0.0)
    else:
        from ns_algebra.ns_sca import twice_level as ns_twice_level

        word0, wordw = series.ns_words
        effective0 = h0 + (0.5 * ns_twice_level(word0) if word0 else 0.0)
        effectivew = hw + (0.5 * ns_twice_level(wordw) if wordw else 0.0)
    return LocalHalfLevelSeries(
        complex(series.h_internal) - complex(effective0) - complex(effectivew),
        series.coefficients,
    )


@dataclass(frozen=True)
class FixedPTwoRamondKernel:
    """Reusable fixed-P kernel in either the direct-z or crossed-w channel.

    ``terms`` contain no modulus-dependent factors.  In particular the
    ``psi0`` row stores the constant part of ``K_psi``.  When picture zero is
    at the moving puncture its factor ``z**(-1/2)`` is inserted by
    :meth:`evaluate`; in the crossed local expansion this is
    ``(1-w)**(-1/2)``, not ``w**(-1/2)``.  At the fixed puncture one,
    ``K_psi(1)`` is constant, but the moving picture-minus-one NS vertex
    contributes the same ``z**(-1/2)`` through its superghost correlator.
    """

    channel: Literal["direct", "crossed"]
    ns_at_z: NSKind
    ns_at_one: NSKind
    zero_ramond_family: RamondFamily
    infinity_ramond_family: RamondFamily
    picture_zero_at: PictureZeroPosition
    internal_momentum: complex
    time_momenta: tuple[complex, complex, complex, complex]
    terms: tuple[FixedPSeriesKernelTerm, ...]

    def evaluate(self, z: complex, zbar: complex | None = None, *,
                 cut_side: CutSide = "upper") -> Mapping[str, complex]:
        frame = sphere_channel_coordinates(z, zbar=zbar, channel=self.channel, cut_side=cut_side)
        zv, zb, local, local_bar = frame.z, frame.zbar, frame.local, frame.local_bar
        scalar = 0.0j
        for term in self.terms:
            # If picture zero is at z, only the time-Ising psi0 branch has a
            # moving z**(-1/2).  If picture zero is fixed at one, the moving
            # picture-minus-one NS superghost supplies that factor to both
            # branches (while K_psi(1) itself is constant).
            moving_root = term.pco_branch == "psi0" or self.picture_zero_at == "one"
            kpsi = zv ** (-0.5) if moving_root else 1.0
            if isinstance(term.holomorphic_series, (MixedEllipticBlockSeries, EllipticPrefactoredRamondSphereSeries)):
                holomorphic_value = term.holomorphic_series.value(
                    local,
                    cut_side=frame.holomorphic_cut_side,
                )
            else:
                holomorphic_value = term.holomorphic_series.value(local)
            if isinstance(term.antiholomorphic_series, (MixedEllipticBlockSeries, EllipticPrefactoredRamondSphereSeries)):
                antiholomorphic_value = term.antiholomorphic_series.value(
                    local_bar,
                    cut_side=frame.antiholomorphic_cut_side,
                )
            else:
                antiholomorphic_value = term.antiholomorphic_series.value(local_bar)
            scalar += (
                term.coefficient
                * kpsi
                * holomorphic_value
                * antiholomorphic_value
            )
        common = timelike_exponential_factor(self.time_momenta, zv, zb) * scalar
        if (self.ns_at_z, self.ns_at_one) == ("S", "S"):
            return {"F_SS": common}
        if (self.ns_at_z, self.ns_at_one) == ("S", "V"):
            return {"F_SV": common * spin7_one_vector_correlator(zb, vector_position="one")}
        if (self.ns_at_z, self.ns_at_one) == ("V", "S"):
            return {"F_VS": common * spin7_one_vector_correlator(zb, vector_position="z")}
        spectator = spin7_two_vector_correlator(zb)
        return {"F0": common * spectator.scalar, "F2": common * spectator.bivector}

    def evaluate_many(
        self, z_values: Sequence[complex], zbar_values: Sequence[complex] | None = None
    ) -> tuple[Mapping[str, complex], ...]:
        zs = tuple(z_values)
        if zbar_values is None:
            return tuple(self.evaluate(z) for z in zs)
        zbs = tuple(zbar_values)
        if len(zs) != len(zbs):
            raise ValueError("z_values and zbar_values must have equal length")
        return tuple(self.evaluate(z, zb) for z, zb in zip(zs, zbs))

    def local_expansion(self, *, maximum_integer_order: int = 0) -> tuple[LocalKernelProduct, ...]:
        """Return block+time+PCO local products (spectator kept in labels).

        Direct terms use local coordinate ``z``; crossed terms use ``w=1-z``.
        The SO(7) spectator is intentionally not folded into these products,
        because VV has two distinct tensor coefficients.  ``labels`` records
        ``spectator_kind`` so the caller can expand the desired tensor.
        """

        k1, k2, k3, _ = self.time_momenta
        result = []
        for term in self.terms:
            if isinstance(
                term.holomorphic_series,
                MixedEllipticBlockSeries,
            ) or isinstance(
                term.antiholomorphic_series,
                MixedEllipticBlockSeries,
            ):
                raise NotImplementedError(
                    "local z-series expansion is unavailable for the "
                    "elliptic-recursion kernel; evaluate its H(t) series directly"
                )
            hol = _series_local(term.holomorphic_series)
            anti = _series_local(term.antiholomorphic_series)
            moving_root = term.pco_branch == "psi0" or self.picture_zero_at == "one"
            if self.channel == "direct":
                hol = hol.shifted(-k1 * k2).multiply_binomial(k2 * k3, maximum_integer_order=maximum_integer_order)
                anti = anti.shifted(-k1 * k2).multiply_binomial(k2 * k3, maximum_integer_order=maximum_integer_order)
                if moving_root:
                    hol = hol.shifted(-0.5)
            else:
                hol = hol.shifted(-k2 * k3).multiply_binomial(k1 * k2, maximum_integer_order=maximum_integer_order)
                anti = anti.shifted(-k2 * k3).multiply_binomial(k1 * k2, maximum_integer_order=maximum_integer_order)
                if moving_root:
                    hol = hol.multiply_binomial(0.5, maximum_integer_order=maximum_integer_order)
            labels = dict(term.labels)
            labels["pco_branch"] = term.pco_branch
            labels["local_coordinate"] = "z" if self.channel == "direct" else "w=1-z"
            labels["spectator_kind"] = self.ns_at_z + self.ns_at_one
            result.append(LocalKernelProduct(term.coefficient, hol, anti, labels))
        return tuple(result)


def _crossed_sld_block_series(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    component: InternalNSComponent,
    rr_structure_sign: int,
    chirality: Literal["holomorphic", "antiholomorphic"],
    ns_words: Sequence[tuple],
    ramond_ground_parities: Sequence[int],
    digits: int,
    condition_limit: float,
    block_backend: CrossedBlockBackend,
    double_virasoro_limit_options: Mapping[str, Any] | None = None,
) -> tuple[Any, Literal["inverse_gram", "elliptic_recursion", "double_virasoro"]]:
    """Build one crossed chiral block with an explicit backend policy.

    The supplied words retain the original ``(NS2,NS3)`` ordering.  The
    elliptic-recursion wrapper performs the standardized crossed swap
    ``(NS2,NS3)->(NS3,NS2)`` exactly once.
    """

    if block_backend not in ("inverse_gram", "elliptic_recursion", "hybrid", "double_virasoro"):
        raise ValueError(
            "crossed_block_backend must be 'inverse_gram', "
            "'elliptic_recursion', 'hybrid', or 'double_virasoro'"
        )
    if block_backend == "double_virasoro":
        from so7e8_mixed_double_virasoro import two_ramond_internal_ns_sld_double_virasoro_h_series

        return (
            two_ramond_internal_ns_sld_double_virasoro_h_series(
                internal_momentum, external_momenta=external_momenta,
                maximum_twice_level=maximum_twice_level, component=component,
                rr_structure_sign=rr_structure_sign, chirality=chirality,
                ns_words=ns_words, ramond_ground_parities=ramond_ground_parities,
                **dict(double_virasoro_limit_options or {})),
            "double_virasoro",
        )
    two_star = all(bool(tuple(word)) for word in ns_words)
    two_star_above_certified_order = (
        two_star
        and maximum_twice_level > TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER
    )
    if block_backend == "elliptic_recursion" and two_star_above_certified_order:
        raise NotImplementedError(
            "the pure elliptic-recursion backend certifies two-star blocks "
            "only through t**7; use crossed_block_backend='hybrid' or "
            "'inverse_gram' above that order"
        )
    use_recursion = block_backend == "elliptic_recursion" or (
        block_backend == "hybrid" and not two_star_above_certified_order
    )
    common = dict(
        external_momenta=external_momenta,
        maximum_twice_level=maximum_twice_level,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        ns_words=ns_words,
        ramond_ground_parities=ramond_ground_parities,
        digits=digits,
        condition_limit=condition_limit,
    )
    if use_recursion:
        return (
            two_ramond_internal_ns_sld_elliptic_h_series(
                internal_momentum,
                **common,
            ),
            "elliptic_recursion",
        )
    if block_backend == "hybrid" and two_star_above_certified_order:
        cached_common = dict(common)
        cached_common.pop("ns_words")
        return (
            two_star_internal_ns_sld_cached_elliptic_h_series(
                internal_momentum,
                **cached_common,
            ),
            "inverse_gram",
        )
    return (
        two_ramond_internal_ns_sld_chiral_block(
            internal_momentum,
            **common,
        ),
        "inverse_gram",
    )


def build_fixed_p_two_ramond_kernel(
    internal_momentum: complex,
    *,
    external_liouville_momenta: Sequence[complex],
    time_momenta: Sequence[complex],
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    zero_ramond_family: RamondFamily,
    infinity_ramond_family: RamondFamily,
    channel: Literal["direct", "crossed"] = "direct",
    picture_zero_at: PictureZeroPosition = "z",
    maximum_twice_level: int = 0,
    digits: int = 40,
    condition_limit: float = 1.0e13,
    allow_uncertified_direct_projector: bool = False,
    crossed_block_backend: CrossedBlockBackend = "inverse_gram",
    direct_block_backend: DirectBlockBackend = "double_virasoro",
    double_virasoro_limit_options: Mapping[str, Any] | None = None,
) -> FixedPTwoRamondKernel:
    """Build a z-independent fixed-P kernel.

    ``crossed_block_backend`` leaves the inverse-Gram construction available
    as the reference, selects the elliptic recursion explicitly, or uses the
    hybrid policy (recursion except for two-star blocks above ``t**7``).
    The explicit ``double_virasoro`` option uses the enlarged-theory branch
    sum, ordinary Virasoro c-recursion, and a checked b=1 contour limit,
    followed by elliptic-H truncation. It has no inverse-Gram fallback.
    ``double_virasoro_limit_options`` explicitly controls the two-radius
    finite-part extraction (radius, check_radius, samples, tolerance).
    Defaults and failure checks are unchanged; non-DV backends reject it.
    The internal-R chiral layer defaults to ``direct_block_backend=
    'double_virasoro'`` *only after* the separate direct-projector opt-in;
    ``inverse_gram`` remains an explicit low-order benchmark choice.
    The crossed channel is the locally audited reference path; this does
    not certify finite picture transport or the integrated amplitude. The direct
    nonchiral internal-R routing has failed the integrated direct/crossed
    overlap test and is available only after the explicit experimental
    opt-in ``allow_uncertified_direct_projector=True``.
    """

    if channel == "direct" and not allow_uncertified_direct_projector:
        raise ValueError(
            "the direct internal-R projector is uncertified; pass "
            "allow_uncertified_direct_projector=True only for diagnostics"
        )
    if direct_block_backend not in ("double_virasoro", "inverse_gram"):
        raise ValueError("direct_block_backend must be 'inverse_gram' or 'double_virasoro'")
    limit_options = dict(double_virasoro_limit_options or {})
    if set(limit_options) - {"radius", "check_radius", "samples", "tolerance"}:
        raise ValueError("unknown double-Virasoro finite-part option")
    selected_backend = direct_block_backend if channel == "direct" else crossed_block_backend
    if limit_options and selected_backend != "double_virasoro":
        raise ValueError("finite-part options require the double_virasoro backend")
    if channel == "direct" and crossed_block_backend != "inverse_gram":
        raise ValueError(
            "crossed_block_backend applies only when channel='crossed'"
        )
    if crossed_block_backend not in (
        "inverse_gram",
        "elliptic_recursion",
        "hybrid",
        "double_virasoro",
    ):
        raise ValueError(
            "crossed_block_backend must be 'inverse_gram', "
            "'elliptic_recursion', 'hybrid', or 'double_virasoro'"
        )

    state2 = _validate_ns_kind(ns_at_z, "ns_at_z")
    state3 = _validate_ns_kind(ns_at_one, "ns_at_one")
    momenta = tuple(complex(x) for x in external_liouville_momenta)
    times = tuple(complex(x) for x in time_momenta)
    if len(momenta) != 4 or len(times) != 4:
        raise ValueError("external and time momentum lists must each have length four")
    # Momentum conservation is checked here, before expensive block construction.
    if abs(sum(times)) > 1.0e-10:
        raise ValueError("signed time momenta must sum to zero")
    if picture_zero_at not in ("z", "one"):
        raise ValueError("picture_zero_at must be 'z' or 'one'")
    pco_time_momentum = times[1 if picture_zero_at == "z" else 2]
    built: list[FixedPSeriesKernelTerm] = []
    holomorphic_cache: dict[tuple, Any] = {}
    antiholomorphic_cache: dict[tuple, Any] = {}
    if channel == "direct":
        p1, p2, p3, p4 = momenta
        structures = {
            (sl, sr): 0.25
            * rr_ns_chiral_structure_constant(
                p4, internal_momentum, p3, structure_sign=sl, precision=digits
            )
            * rr_ns_chiral_structure_constant(
                -complex(internal_momentum), p1, p2,
                structure_sign=sr, precision=digits
            )
            for sl in (-1, 1)
            for sr in (-1, 1)
        }
        for signs, structure in structures.items():
            # The right trinion is oriented with internal momentum -P in the
            # structure constant, whereas ``hjs_ramond_sphere_block_series``
            # builds both chiral tensors with the +P Ramond ground-state
            # convention.  Under P -> -P the HJS label flips, so the block
            # signs are (epsilon_L,-epsilon_R).  Keep ``signs`` itself for
            # the structure constants and for the nonchiral R-/R+ sewing
            # phase, which are still labelled in the oriented convention.
            block_signs = (signs[0], -signs[1])
            for branch in ("G", "psi0"):
                pco = ordered_pco_coefficient(branch, pco_time_momentum, picture_zero_at)
                if pco == 0:
                    continue
                hw, aw = ns_descendant_words(state2, state3, branch, picture_zero_at)
                # z=1 extracts exactly the constant matrix part of K_psi.
                for external in two_ramond_component_weights(zero_ramond_family, infinity_ramond_family, branch=branch, z=1.0):
                    hg = (external.zero_hjs_ground_parity, external.infinity_hjs_ground_parity)
                    ag = (external.zero_antiholomorphic_ground_parity, external.infinity_antiholomorphic_ground_parity)
                    hword_parities = tuple(int(bool(word)) for word in hw)
                    aword_parities = tuple(int(bool(word)) for word in aw)
                    for fh_left in (0, 1):
                        for fh_right in (0, 1):
                            hforms = (fh_left, fh_right)
                            for projector in internal_ramond_state_projectors(
                                signs[0], signs[1], hforms, hg, ag,
                                hword_parities, aword_parities,
                            ):
                                aforms = projector.antiholomorphic_form_parities
                                phase = projector.phase
                                hkey = (block_signs, hg, hw, hforms)
                                if hkey not in holomorphic_cache:
                                    holomorphic_cache[hkey] = _direct_sld_block_series(
                                        internal_momentum, momenta,
                                        maximum_twice_level, block_signs, hg,
                                        hw, hforms, digits, condition_limit,
                                        chirality="holomorphic", block_backend=direct_block_backend,
                                        double_virasoro_limit_options=limit_options,
                                    )
                                akey = (block_signs, ag, aw, aforms)
                                if akey not in antiholomorphic_cache:
                                    antiholomorphic_cache[akey] = (
                                        _direct_sld_block_series(
                                            internal_momentum, momenta,
                                            maximum_twice_level, block_signs,
                                            ag, aw, aforms, digits,
                                            condition_limit,
                                            chirality="antiholomorphic", block_backend=direct_block_backend,
                                            double_virasoro_limit_options=limit_options,
                                        )
                                    )
                                hs = holomorphic_cache[hkey]
                                ans = antiholomorphic_cache[akey]
                                coefficient = (
                                    external.coefficient * phase
                                    * structure * pco
                                )
                                if coefficient:
                                    built.append(FixedPSeriesKernelTerm(
                                        branch, coefficient, hs, ans,
                                        {"signs": signs, "block_signs": block_signs, "internal_local_field": projector.internal_local_field, "holomorphic_form_parities": hforms, "antiholomorphic_form_parities": aforms, "holomorphic_block_backend": direct_block_backend, "antiholomorphic_block_backend": direct_block_backend, "external": external},
                                    ))
    elif channel == "crossed":
        p1, p2, p3, p4 = momenta
        nsc = ns_structure_constants(p3, p2, internal_momentum, precision=digits)
        rrc = {s: rr_ns_chiral_structure_constant(p4, p1, internal_momentum, structure_sign=s, precision=digits) for s in (-1, 1)}
        for branch in ("G", "psi0"):
            pco = ordered_pco_coefficient(branch, pco_time_momentum, picture_zero_at)
            if pco == 0:
                continue
            hw, aw = ns_descendant_words(state2, state3, branch, picture_zero_at)
            for r in (0, 1):
                hc, ac, phase = crossed_ns_component_routing(
                    state2, state3, branch, r, picture_zero_at
                )
                for sign in (-1, 1):
                    for external in two_ramond_component_weights(zero_ramond_family, infinity_ramond_family, branch=branch, z=1.0):
                        hg = (external.zero_hjs_ground_parity, external.infinity_hjs_ground_parity)
                        ag = (external.zero_antiholomorphic_ground_parity, external.infinity_antiholomorphic_ground_parity)
                        hkey = (hc, sign, hw, hg)
                        if hkey not in holomorphic_cache:
                            holomorphic_cache[hkey] = _crossed_sld_block_series(
                                internal_momentum,
                                external_momenta=momenta,
                                maximum_twice_level=maximum_twice_level,
                                component=hc,
                                rr_structure_sign=sign,
                                chirality="holomorphic",
                                ns_words=hw,
                                ramond_ground_parities=hg,
                                digits=digits,
                                condition_limit=condition_limit,
                                block_backend=crossed_block_backend,
                                double_virasoro_limit_options=limit_options,
                            )
                        akey = (ac, sign, aw, ag)
                        if akey not in antiholomorphic_cache:
                            antiholomorphic_cache[akey] = _crossed_sld_block_series(
                                internal_momentum,
                                external_momenta=momenta,
                                maximum_twice_level=maximum_twice_level,
                                component=ac,
                                rr_structure_sign=sign,
                                chirality="antiholomorphic",
                                ns_words=aw,
                                ramond_ground_parities=ag,
                                digits=digits,
                                condition_limit=condition_limit,
                                block_backend=crossed_block_backend,
                                double_virasoro_limit_options=limit_options,
                            )
                        hs, holomorphic_backend = holomorphic_cache[hkey]
                        ans, antiholomorphic_backend = antiholomorphic_cache[akey]
                        coefficient = 0.5 * external.coefficient * phase * nsc[r] * rrc[sign] * pco
                        if coefficient:
                            built.append(FixedPSeriesKernelTerm(branch, coefficient, hs, ans, {"rr_sign": sign, "ns_parity": r, "hol_component": hc, "anti_component": ac, "holomorphic_block_backend": holomorphic_backend, "antiholomorphic_block_backend": antiholomorphic_backend, "external": external}))
    else:
        raise ValueError("channel must be 'direct' or 'crossed'")
    return FixedPTwoRamondKernel(
        channel, state2, state3, zero_ramond_family, infinity_ramond_family,
        picture_zero_at, complex(internal_momentum), times, tuple(built)
    )


__all__ = [
    "CrossedBlockBackend",
    "DirectBlockBackend",
    "CrossedFixedPComponentTerm",
    "CrossedFixedPTwoRamondIntegrand",
    "FixedPComponentTerm",
    "FixedPSeriesKernelTerm",
    "FixedPTwoRamondIntegrand",
    "FixedPTwoRamondKernel",
    "InternalLocalRamond",
    "InternalNSComponent",
    "NSKind",
    "PictureZeroPosition",
    "SmallRepresentationProjection",
    "assemble_crossed_fixed_p_two_ramond_integrand",
    "assemble_fixed_p_two_ramond_integrand",
    "build_fixed_p_two_ramond_kernel",
    "crossed_ns_component_routing",
    "hjs_small_representation_projectors",
    "internal_ramond_state_projectors",
    "ns_descendant_words",
    "physical_structure_sign_pairs",
    "suchanek_rr_vertex_form_routing",
    "timelike_exponential_factor",
    "superghost_position_factor",
    "LocalKernelProduct",
]
