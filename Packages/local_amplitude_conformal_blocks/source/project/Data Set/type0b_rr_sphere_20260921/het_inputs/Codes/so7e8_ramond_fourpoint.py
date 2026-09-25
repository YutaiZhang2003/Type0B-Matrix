#!/usr/bin/env python3
r"""Exact ingredients for SO(7) x E8 sphere amplitudes containing Ramond legs.

This module deliberately separates three levels of information.

* ``ALLOWED_ONE_TO_THREE_PROCESSES`` is the exact-wall state and tensor
  inventory.  It uses only spacetime fermion parity and SO(7), not an
  auxiliary Spin(8) selection rule.
* The spectator-spin correlators below are closed free-field expressions.
* ``ramond_sld_chiral_branch`` and ``ramond_sld_structure_product`` assemble
  one exact R--NS--NS--R super-Liouville branch.
* ``antiholomorphic_ramond_sld_chiral_branch`` uses the conjugate HJS tensor
  phases with the same analytically continued momenta.  It exposes all four
  external Ramond ground components needed by the local ``R+`` expansion.
* ``two_ramond_internal_ns_sld_chiral_block`` supplies the crossed
  ``NS--NS | R--R`` expansion about ``w=1-z=0``.
* ``hjs_four_ramond_ns_block_series`` implements the HJS four-external-
  Ramond chiral block with an internal NS module.  It keeps the two HJS
  trinion signs and the even/odd NS components separate.

The physical GSO sum, the PCO component sum where NS legs are present, the
internal-momentum contour, and the regularized modulus integral remain
separate operations.

Consequently this is an exact integrand-level implementation, not a claimed
closed formula for the integrated four-point amplitudes.  The four-Ramond
routine is a normalized chiral block, not yet the local nonchiral heterotic
correlator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import cmath
from itertools import product
import math
from typing import Literal, Mapping, Sequence

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import (
    Word as NSWord,
    fermion_parity as ns_fermion_parity,
    gram_matrix as ns_gram_matrix,
    twice_level as ns_twice_level,
)
from ns_algebra.ns_three_point_tensor import ns_three_point
from ramond_algebra.ns_rr_three_point_tensor import (
    hjs_ns_rr_polynomial_ground_tensor,
    ns_rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_sca import ground_state as ramond_ground_state
from spin23_ramond_blocks import (
    RamondSphereBlockSeries,
    direct_ramond_sphere_block_series,
    hjs_ramond_sphere_block_series,
    ramond_weight,
)
from spin23_super_liouville_data import (
    ns_weight,
    rr_ns_chiral_structure_constant,
)


NSParticle = Literal["S", "V"]
Particle = Literal["S", "V", "Psi", "Psi_tilde"]
FourRamondComponent = Literal["even", "odd"]
Chirality = Literal["holomorphic", "antiholomorphic"]

_HOL_HJS_PHASE = (sp.S.One + sp.I) / sp.sqrt(2)
_ANTIHOL_HJS_PHASE = (sp.S.One - sp.I) / sp.sqrt(2)

SO7_VECTOR_DIMENSION = 7
SO7_SPINOR_DIMENSION = 8
SPHERE_REQUIRED_PICTURE = -2.0

# The first physical Ramond-channel discrete state has time momentum i/2,
# whereas its super-Liouville factor is the self-dual (1,2)/(2,1)
# degenerate representation at P=3i/2.  These must not be identified.
FIRST_RAMOND_CHANNEL_ENERGY = 0.5j
FIRST_RAMOND_INTERNAL_DEGENERATE_MOMENTUM = 1.5j

# Four external canonical Ramond vertices need no PCO.  The screening-free
# term fails the spin/H-charge selection rule.  The first nonzero wall
# resonance has one Yukawa screening and p_in=i(s+2)/2=3i/2.
FIRST_FOUR_RAMOND_SCREENING_NUMBER = 1
FIRST_FOUR_RAMOND_RESONANT_INCOMING_MOMENTUM = 1.5j


@dataclass(frozen=True)
class RamondFourPointProcess:
    """One fixed-incoming process class allowed by exact conserved charges."""

    incoming: Particle
    outgoing: tuple[Particle, Particle, Particle]
    tensor_structures: tuple[str, ...]
    ramond_count: int
    pco_count: int


# The displayed Ramond labels are the physical positive-energy orientation:
# an incoming Psi and outgoing Psi_tilde.  The CP-reflected orientation is
# obtained by exchanging the two labels.
ALLOWED_ONE_TO_THREE_PROCESSES: tuple[RamondFourPointProcess, ...] = (
    RamondFourPointProcess("S", ("S", "Psi_tilde", "Psi_tilde"), ("C",), 2, 1),
    RamondFourPointProcess(
        "S", ("V", "Psi_tilde", "Psi_tilde"), ("C gamma^a",), 2, 1
    ),
    RamondFourPointProcess(
        "V", ("S", "Psi_tilde", "Psi_tilde"), ("C gamma^a",), 2, 1
    ),
    RamondFourPointProcess(
        "V",
        ("V", "Psi_tilde", "Psi_tilde"),
        ("delta^{ab} C", "C gamma^{ab}"),
        2,
        1,
    ),
    RamondFourPointProcess("Psi", ("Psi_tilde", "S", "S"), ("C",), 2, 1),
    RamondFourPointProcess(
        "Psi", ("Psi_tilde", "S", "V"), ("C gamma^a",), 2, 1
    ),
    RamondFourPointProcess(
        "Psi",
        ("Psi_tilde", "V", "V"),
        ("delta^{ab} C", "C gamma^{ab}"),
        2,
        1,
    ),
    RamondFourPointProcess(
        "Psi",
        ("Psi_tilde", "Psi_tilde", "Psi_tilde"),
        (
            "C_{alpha beta} C_{gamma delta}",
            "(C gamma^a)_{alpha beta}(C gamma^a)_{gamma delta}",
            "(C gamma^{ab})_{alpha beta}(C gamma^{ab})_{gamma delta}",
            "(C gamma^{abc})_{alpha beta}(C gamma^{abc})_{gamma delta}",
        ),
        4,
        0,
    ),
)


@dataclass(frozen=True)
class IndependentRamondCorrelator:
    """An analytic external-species class before choosing an incoming leg."""

    external_ns: tuple[NSParticle, NSParticle] | None
    ramond_count: int
    tensor_structures: tuple[str, ...]


INDEPENDENT_RAMOND_CORRELATORS: tuple[IndependentRamondCorrelator, ...] = (
    IndependentRamondCorrelator(("S", "S"), 2, ("C",)),
    IndependentRamondCorrelator(("S", "V"), 2, ("C gamma^a",)),
    IndependentRamondCorrelator(
        ("V", "V"), 2, ("delta^{ab} C", "C gamma^{ab}")
    ),
    IndependentRamondCorrelator(
        None,
        4,
        (
            "C C",
            "(C gamma^a)(C gamma^a)",
            "(C gamma^{ab})(C gamma^{ab})",
            "(C gamma^{abc})(C gamma^{abc})",
        ),
    ),
)


def required_pco_count(ramond_count: int, ns_count: int) -> int:
    r"""Return the number of picture raises from canonical external pictures.

    Each R vertex starts in picture ``-1/2`` and each NS vertex in picture
    ``-1``.  The sphere total must be ``-2``.
    """

    if ramond_count < 0 or ns_count < 0:
        raise ValueError("external-state counts must be nonnegative")
    initial_picture = -0.5 * ramond_count - ns_count
    raises = SPHERE_REQUIRED_PICTURE - initial_picture
    if abs(raises - round(raises)) > 1.0e-12 or raises < 0:
        raise ValueError("the external pictures cannot be raised to sphere picture -2")
    return int(round(raises))


@dataclass(frozen=True)
class Spin7TwoVectorCorrelator:
    r"""Coefficients of ``delta^{ab} C`` and ``C gamma^{ab}``.

    The ordering is

    ``<Sigma_alpha(infinity) lambda^b(1) lambda^a(z) Sigma_beta(0)>``.
    """

    scalar: complex
    bivector: complex


@dataclass(frozen=True)
class LocalRamondSLDBranch:
    r"""One branch of the local ``R+--NS--NS--R+`` SLD correlator.

    The public structure-constant functions return ``C_even`` and ``C_odd``
    rather than HJS ``C^(+)`` and ``C^(-)``.  Since
    ``C^(+)=C_even/2`` and ``C^(-)=C_odd/2``, every two-trinion branch has
    the coefficient ``1/4`` recorded here.
    """

    left_structure_sign: int
    right_structure_sign: int
    coefficient: float


@dataclass(frozen=True)
class NoScreenVVBranchCoefficients:
    r"""Beta-reduced coefficients at the screening-free NS resonance.

    The entries multiply ``delta^{ab} C`` and ``C gamma^{ab}`` before the
    holomorphic PCO coefficient and heterotic external-polarization/cocycle
    layer are attached.
    """

    scalar: complex
    bivector: complex


@dataclass(frozen=True)
class FourRamondNSBlockSeries:
    r"""One normalized HJS four-Ramond chiral block.

    The puncture order is ``(R1,R2,R3,R4)=(0,z,1,infinity)`` and the
    internal module is NS.  ``coefficients[2*f]`` stores the coefficient at
    NS descendant level ``f``.  Integer levels form the even component and
    half-integer levels form the odd component.

    This object contains neither super-Liouville structure constants nor the
    physical nonchiral GSO/polarization sum.
    """

    coefficients: Mapping[int, complex]
    component: FourRamondComponent
    c: complex
    h_internal: complex
    external_betas: tuple[complex, complex, complex, complex]
    external_weights: tuple[complex, complex, complex, complex]
    left_structure_sign: int
    right_structure_sign: int
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def descendant_value(self, z: complex) -> complex:
        """Evaluate the descendant series, without its primary power."""

        z = complex(z)
        if z == 0:
            return complex(self.coefficients.get(0, 0.0j))
        log_z = cmath.log(z)
        return sum(
            coefficient * cmath.exp(0.5 * twice_level * log_z)
            for twice_level, coefficient in self.coefficients.items()
        )

    def value(self, z: complex, *, include_primary_power: bool = True) -> complex:
        r"""Evaluate :math:`z^{h_p-h_1-h_2}` times the descendant series."""

        descendant = self.descendant_value(z)
        if not include_primary_power:
            return descendant
        if z == 0:
            raise ValueError("the primary four-Ramond block power is singular at z=0")
        h1, h2, _, _ = self.external_weights
        return cmath.exp(
            (self.h_internal - h1 - h2) * cmath.log(complex(z))
        ) * descendant


@dataclass(frozen=True)
class TwoRamondInternalNSBlockSeries:
    r"""One crossed two-Ramond block with an internal NS module.

    The standardized puncture order is

    ``(NS1,NS2,R3,R4)=(0,w,1,infinity)``.

    Thus the right trinion is NS--NS--NS and the left trinion is
    R--R--NS.  The latter is evaluated through the ordered NS--R--R Ward
    tensor after the HJS orientation transformation.  ``coefficients[2*f]``
    stores the coefficient at internal NS level ``f``.

    This is a normalized chiral block.  Super-Liouville structure constants,
    the nonchiral GSO sum, free-field factors, and modulus integration are not
    included.
    """

    coefficients: Mapping[int, complex]
    component: FourRamondComponent
    chirality: Chirality
    c: complex
    h_internal: complex
    external_weights: tuple[complex, complex, complex, complex]
    ns_words: tuple[NSWord, NSWord]
    ramond_ground_parities: tuple[int, int]
    rr_structure_sign: int | None
    rr_orientation_phase: complex
    external_ground_basis: Literal["polynomial", "hjs"]
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def descendant_value(self, w: complex) -> complex:
        """Evaluate the descendant series without its leading OPE power."""

        w = complex(w)
        if w == 0:
            return complex(self.coefficients.get(0, 0.0j))
        log_w = cmath.log(w)
        return sum(
            coefficient * cmath.exp(0.5 * twice_level * log_w)
            for twice_level, coefficient in self.coefficients.items()
        )

    def value(self, w: complex, *, include_primary_power: bool = True) -> complex:
        r"""Evaluate :math:`w^{h_p-h_1-h_2}` times the descendant series."""

        descendant = self.descendant_value(w)
        if not include_primary_power:
            return descendant
        if w == 0:
            raise ValueError("the primary crossed-block power is singular at w=0")
        h_zero, h_w, _, _ = self.external_weights
        word_zero, word_w = self.ns_words
        effective_zero = h_zero + (
            0.5 * ns_twice_level(word_zero) if word_zero else 0.0
        )
        effective_w = h_w + (
            0.5 * ns_twice_level(word_w) if word_w else 0.0
        )
        return cmath.exp(
            (self.h_internal - effective_zero - effective_w)
            * cmath.log(complex(w))
        ) * descendant


def _validate_hjs_structure_sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def antiholomorphic_hjs_ground_tensor(structure_sign: int) -> sp.Matrix:
    r"""Return the antiholomorphic HJS Ramond ground tensor.

    Rows and columns are ordered ``(+,-)``.  With ``s=structure_sign`` the
    tensor is

    .. math::

       \bar H_s=\begin{pmatrix}1&1\\-s i&s\end{pmatrix}.

    In particular the odd lower-left entry is ``-s*i`` rather than the
    holomorphic ``+s*i``.  This operation conjugates convention phases only;
    it does not complex-conjugate an analytically continued momentum.
    """

    sign = _validate_hjs_structure_sign(structure_sign, "structure_sign")
    s = sp.Integer(sign)
    return sp.Matrix([[1, 1], [-s * sp.I, s]])


def antiholomorphic_hjs_to_polynomial_ground_tensor(
    beta_infinity: sp.Expr | complex,
    beta_zero: sp.Expr | complex,
    *,
    structure_sign: int,
) -> sp.Matrix:
    r"""Convert one antiholomorphic HJS R--NS--R tensor to polynomial basis.

    The exact conversion is

    .. math::

       \operatorname{diag}(1,-i e^{-i\pi/4}\beta_\infty)\,
       \bar H_s\,
       \operatorname{diag}(1,e^{-i\pi/4}\beta_0).

    The first factor includes the anti-linear infinity-slot convention.  The
    supplied ``beta`` values are used literally: callers performing analytic
    continuation must pass the same momenta to both chiralities, not their
    numerical complex conjugates.
    """

    beta_inf = sp.sympify(beta_infinity)
    beta_0 = sp.sympify(beta_zero)
    left = sp.diag(1, -sp.I * _ANTIHOL_HJS_PHASE * beta_inf)
    right = sp.diag(1, _ANTIHOL_HJS_PHASE * beta_0)
    return (
        left
        * antiholomorphic_hjs_ground_tensor(structure_sign)
        * right
    ).applyfunc(lambda entry: sp.factor(sp.expand(entry)))


def antiholomorphic_crossed_ns_rr_polynomial_ground_tensor(
    beta_one: sp.Expr | complex,
    beta_infinity: sp.Expr | complex,
    *,
    structure_sign: int,
) -> sp.Matrix:
    r"""Return the anti-HJS tensor used by the crossed RR--NS form.

    The crossed block is standardized as

    ``(NS_zero,NS_w,R_one,R_infinity)=(0,w,1,infinity)``.

    Its left RR--NS form is evaluated through an auxiliary ordered NS--R--R
    tensor whose Ramond slots are ``(middle,zero)=(R_one,R_infinity)``.  After
    convention conjugation of the HJS phases, the exact polynomial-basis
    conversion is

    .. math::

       \operatorname{diag}(1,e^{-i\pi/4}\beta_1)\,
       \bar H_s\,
       \operatorname{diag}(1,-e^{-i\pi/4}\beta_4).

    The minus sign in the auxiliary zero slot is the crossed-channel image of
    the anti-linear physical infinity slot.  Together with the orientation
    factor ``(+i)**fermion_parity``, it makes every normalized
    antiholomorphic coefficient the convention conjugate of the holomorphic
    one while leaving analytically continued beta values untouched.
    """

    sign = _validate_hjs_structure_sign(structure_sign, "structure_sign")
    beta_1 = sp.sympify(beta_one)
    beta_4 = sp.sympify(beta_infinity)
    middle = sp.diag(1, _ANTIHOL_HJS_PHASE * beta_1)
    zero = sp.diag(1, -_ANTIHOL_HJS_PHASE * beta_4)
    return (
        middle * antiholomorphic_hjs_ground_tensor(sign) * zero
    ).applyfunc(lambda entry: sp.factor(sp.expand(entry)))


def _antiholomorphic_hjs_external_state_factor(
    ground_parity: int,
    beta: complex,
    *,
    slot: Literal["infinity", "zero"],
) -> complex:
    """Return the polynomial/HJS factor for one anti-HJS ground state."""

    if ground_parity not in (0, 1):
        raise ValueError("ground_parity must be zero or one")
    if ground_parity == 0:
        return 1.0 + 0.0j
    beta_value = complex(beta)
    if beta_value == 0:
        raise ValueError("an odd anti-HJS ground state is singular at beta=0")
    phase = complex(sp.N(_ANTIHOL_HJS_PHASE, 17))
    if slot == "infinity":
        return -1.0j * phase * beta_value
    if slot == "zero":
        return phase * beta_value
    raise ValueError("slot must be 'infinity' or 'zero'")


def hjs_four_ramond_ns_trinion_values(
    internal_word: NSWord,
    *,
    c: sp.Expr | complex,
    h_internal: sp.Expr | complex,
    external_betas: Sequence[sp.Expr | complex],
    left_structure_sign: int,
    right_structure_sign: int,
) -> tuple[sp.Expr, sp.Expr]:
    r"""Return the exact HJS left and right trinion entries.

    The right entry is literally the ordered NS--R--R form

    .. math::

       \rho_{NR}^{(s_R)}(\nu_A,w_2^+,w_1^+).

    The left HJS form is intrinsically ordered R--R--NS.  HJS equations
    (5.15)--(5.16) relate it to the existing ordered NS--R--R evaluator:

    .. math::

       \rho_{RN,e}^{(s_L)}(w_4^+,w_3^+,\nu_A)
         &=\rho_{NR,e}^{(s_L)}(\nu_A,w_3^+,w_4^+),\\
       \rho_{RN,o}^{(s_L)}(w_4^+,w_3^+,\nu_A)
         &=-i\,\rho_{NR,o}^{(s_L)}(\nu_A,w_3^+,w_4^+).

    Hence the orientation factor is ``1`` for an even NS word and ``-i``
    for an odd NS word.  It is not ``(-1)**level``.  The infinity-slot Ward
    recursion already contains the BPZ/local-coordinate mode transformation,
    so no further word sign is inserted.
    """

    word = tuple(internal_word)
    betas = tuple(sp.sympify(value) for value in external_betas)
    if len(betas) != 4:
        raise ValueError("external_betas must contain (beta1,beta2,beta3,beta4)")
    left_sign = _validate_hjs_structure_sign(
        left_structure_sign, "left_structure_sign"
    )
    right_sign = _validate_hjs_structure_sign(
        right_structure_sign, "right_structure_sign"
    )
    central_charge = sp.sympify(c)
    internal_weight = sp.sympify(h_internal)
    weights = tuple(central_charge / 24 - beta**2 for beta in betas)
    beta1, beta2, beta3, beta4 = betas
    h1, h2, h3, h4 = weights
    ramond_plus = ramond_ground_state(0)

    right_ground = hjs_ns_rr_polynomial_ground_tensor(
        beta2,
        beta1,
        structure_sign=right_sign,
    )
    right = ns_rr_three_point_from_ground_tensor(
        word,
        ramond_plus,
        ramond_plus,
        h_infinity=internal_weight,
        h_middle=h2,
        h_zero=h1,
        c=central_charge,
        ground_tensor=right_ground,
    )

    left_ground = hjs_ns_rr_polynomial_ground_tensor(
        beta3,
        beta4,
        structure_sign=left_sign,
    )
    left_as_nr = ns_rr_three_point_from_ground_tensor(
        word,
        ramond_plus,
        ramond_plus,
        h_infinity=internal_weight,
        h_middle=h3,
        h_zero=h4,
        c=central_charge,
        ground_tensor=left_ground,
    )
    orientation = (-sp.I) ** ns_fermion_parity(word)
    return sp.factor(orientation * left_as_nr), sp.factor(right)


def _finite_complex(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite four-Ramond block value {value!r}")
    return value


def hjs_four_ramond_ns_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_betas: Sequence[complex],
    maximum_twice_level: int,
    component: FourRamondComponent,
    left_structure_sign: int,
    right_structure_sign: int,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> FourRamondNSBlockSeries:
    r"""Construct one direct inverse-Gram four-Ramond HJS block.

    This implements HJS equations (5.1)--(5.2), rather than an elliptic
    recursion ansatz.  The four external Ramond states are the normalized
    ``w+`` ground states.  ``external_betas`` follow the puncture order
    ``(beta1,beta2,beta3,beta4)=(0,z,1,infinity)``.
    """

    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    if component not in ("even", "odd"):
        raise ValueError("component must be 'even' or 'odd'")
    left_sign = _validate_hjs_structure_sign(
        left_structure_sign, "left_structure_sign"
    )
    right_sign = _validate_hjs_structure_sign(
        right_structure_sign, "right_structure_sign"
    )
    betas = tuple(complex(value) for value in external_betas)
    if len(betas) != 4:
        raise ValueError("external_betas must contain four entries")
    central_charge = complex(c)
    internal_weight = complex(h_internal)
    weights = tuple(ramond_weight(central_charge, beta) for beta in betas)
    parity = 0 if component == "even" else 1
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}

    for twice_level in range(parity, maximum_twice_level + 1, 2):
        basis, gram = ns_gram_matrix(
            twice_level,
            h=sp.sympify(internal_weight),
            c=sp.sympify(central_charge),
        )
        gram_numeric = np.asarray(
            [
                [
                    _finite_complex(gram[row, column], int(digits))
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
            left, right = hjs_four_ramond_ns_trinion_values(
                word,
                c=central_charge,
                h_internal=internal_weight,
                external_betas=betas,
                left_structure_sign=left_sign,
                right_structure_sign=right_sign,
            )
            left_values.append(_finite_complex(left, int(digits)))
            right_values.append(_finite_complex(right, int(digits)))
        left_vector = np.asarray(left_values, dtype=np.complex128)
        right_vector = np.asarray(right_values, dtype=np.complex128)
        coefficients[twice_level] = complex(
            left_vector @ np.linalg.solve(gram_numeric, right_vector)
        )
        conditions[twice_level] = condition

    return FourRamondNSBlockSeries(
        coefficients=coefficients,
        component=component,
        c=central_charge,
        h_internal=internal_weight,
        external_betas=betas,  # type: ignore[arg-type]
        external_weights=weights,  # type: ignore[arg-type]
        left_structure_sign=left_sign,
        right_structure_sign=right_sign,
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def direct_two_ramond_internal_ns_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    rr_ground_tensor: Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase,
    maximum_twice_level: int,
    component: FourRamondComponent,
    rr_orientation_phase: sp.Expr | complex = -1.0j,
    chirality: Chirality = "holomorphic",
    ns_words: Sequence[NSWord] = ((), ()),
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> TwoRamondInternalNSBlockSeries:
    r"""Construct the direct crossed block with an internal NS module.

    External data follow the standardized order

    ``(NS_zero,NS_w,R_one,R_infinity)=(0,w,1,infinity)``.

    ``rr_ground_tensor`` is the polynomial-basis terminal tensor for the
    auxiliary ordered NS--R--R form with Ramond slots
    ``(middle,zero)=(R_one,R_infinity)``.  The physical left RR--NS form is
    obtained by multiplying a level-``A`` entry by
    ``(rr_orientation_phase*(-1)**a_infinity)**fermion_parity(A)``.
    HJS uses ``-i`` in the
    holomorphic chirality and ``+i`` in the antiholomorphic chirality.
    """

    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    if component not in ("even", "odd"):
        raise ValueError("component must be 'even' or 'odd'")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("chirality must be 'holomorphic' or 'antiholomorphic'")
    weights = tuple(complex(value) for value in external_weights)
    if len(weights) != 4:
        raise ValueError(
            "external_weights must contain (h_NS_zero,h_NS_w,h_R_one,h_R_infinity)"
        )
    words = tuple(tuple(word) for word in ns_words)
    if len(words) != 2:
        raise ValueError("ns_words must contain (word_NS_zero,word_NS_w)")
    ground_parities = tuple(ramond_ground_parities)
    if len(ground_parities) != 2 or any(
        parity not in (0, 1) for parity in ground_parities
    ):
        raise ValueError(
            "ramond_ground_parities must contain the parities at (one,infinity)"
        )
    terminal = sp.Matrix(rr_ground_tensor)
    if terminal.shape != (2, 2):
        raise ValueError("rr_ground_tensor must be a two-by-two matrix")

    central_charge = complex(c)
    internal_weight = complex(h_internal)
    h_ns_zero, h_ns_w, h_r_one, h_r_infinity = weights
    state_one = ramond_ground_state(ground_parities[0])
    state_infinity = ramond_ground_state(ground_parities[1])
    orientation_phase = sp.sympify(rr_orientation_phase)
    parity = 0 if component == "even" else 1
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}

    for twice_descendant_level in range(
        parity,
        maximum_twice_level + 1,
        2,
    ):
        basis, gram = ns_gram_matrix(
            twice_descendant_level,
            h=sp.sympify(internal_weight),
            c=sp.sympify(central_charge),
        )
        gram_numeric = np.asarray(
            [
                [
                    _finite_complex(gram[row, column], int(digits))
                    for column in range(gram.cols)
                ]
                for row in range(gram.rows)
            ],
            dtype=np.complex128,
        )
        # Equilibrate the descendant norms before solving. This is an exact
        # congruence transformation and keeps the condition limit unchanged.
        norms = np.abs(np.diag(gram_numeric))
        norms = np.where(norms > 0, norms, np.max(np.abs(gram_numeric), axis=1))
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise np.linalg.LinAlgError("NS Gram basis has an invalid norm scale")
        basis_scales = np.sqrt(norms)
        equilibrated_gram = gram_numeric / basis_scales[:, None] / basis_scales[None, :]
        condition = float(np.linalg.cond(equilibrated_gram))
        if not math.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"Equilibrated NS Gram matrix at level {twice_descendant_level}/2 has condition "
                f"number {condition:.3e}, above {condition_limit:.3e}"
            )

        left_values: list[complex] = []
        right_values: list[complex] = []
        for internal_word in basis:
            left_as_ns_rr = ns_rr_three_point_from_ground_tensor(
                internal_word,
                state_one,
                state_infinity,
                h_infinity=sp.sympify(internal_weight),
                h_middle=sp.sympify(h_r_one),
                h_zero=sp.sympify(h_r_infinity),
                c=sp.sympify(central_charge),
                ground_tensor=terminal,
            )
            # Native RN Ward identity, HJS 0810.1203v2 (4.10): the
            # infinity Ramond parity crosses every odd internal NS word.
            left = (orientation_phase * (-1)**ground_parities[1]) ** ns_fermion_parity(internal_word)
            left *= left_as_ns_rr
            right = ns_three_point(
                internal_word,
                words[1],
                words[0],
                h_infinity=sp.sympify(internal_weight),
                h_middle=sp.sympify(h_ns_w),
                h_zero=sp.sympify(h_ns_zero),
                c=sp.sympify(central_charge),
            )
            left_values.append(_finite_complex(left, int(digits)))
            right_values.append(_finite_complex(right, int(digits)))
        left_vector = np.asarray(left_values, dtype=np.complex128)
        right_vector = np.asarray(right_values, dtype=np.complex128)
        coefficients[twice_descendant_level] = complex(
            (left_vector / basis_scales)
            @ np.linalg.solve(equilibrated_gram, right_vector / basis_scales)
        )
        conditions[twice_descendant_level] = condition

    return TwoRamondInternalNSBlockSeries(
        coefficients=coefficients,
        component=component,
        chirality=chirality,
        c=central_charge,
        h_internal=internal_weight,
        external_weights=weights,  # type: ignore[arg-type]
        ns_words=words,  # type: ignore[arg-type]
        ramond_ground_parities=ground_parities,  # type: ignore[arg-type]
        rr_structure_sign=None,
        rr_orientation_phase=complex(sp.N(orientation_phase, int(digits))),
        external_ground_basis="polynomial",
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def _crossed_hjs_external_ground_factor(
    beta_one: complex,
    beta_infinity: complex,
    *,
    ramond_ground_parities: tuple[int, int],
    chirality: Chirality,
) -> complex:
    """Return the polynomial/HJS factor for the crossed external R pair."""

    parity_one, parity_infinity = ramond_ground_parities
    if parity_one and beta_one == 0:
        raise ValueError("the odd HJS state at one is singular at beta_one=0")
    if parity_infinity and beta_infinity == 0:
        raise ValueError(
            "the odd HJS state at infinity is singular at beta_infinity=0"
        )
    if chirality == "holomorphic":
        phase = complex(sp.N(_HOL_HJS_PHASE, 17))
        one_factor = phase * beta_one if parity_one else 1.0 + 0.0j
        infinity_factor = (
            phase * beta_infinity if parity_infinity else 1.0 + 0.0j
        )
    else:
        phase = complex(sp.N(_ANTIHOL_HJS_PHASE, 17))
        one_factor = phase * beta_one if parity_one else 1.0 + 0.0j
        infinity_factor = (
            -phase * beta_infinity if parity_infinity else 1.0 + 0.0j
        )
    return one_factor * infinity_factor


def hjs_two_ramond_internal_ns_block_series(
    *,
    c: complex,
    h_internal: complex,
    beta_one: complex,
    beta_infinity: complex,
    h_ns_zero: complex,
    h_ns_w: complex,
    maximum_twice_level: int,
    component: FourRamondComponent,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ns_words: Sequence[NSWord] = ((), ()),
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> TwoRamondInternalNSBlockSeries:
    r"""Construct an HJS-normalized two-R/internal-NS crossed block.

    Momenta are not numerically conjugated in the antiholomorphic branch.
    Only convention phases and the HJS odd tensor are conjugated.
    """

    sign = _validate_hjs_structure_sign(rr_structure_sign, "rr_structure_sign")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("chirality must be 'holomorphic' or 'antiholomorphic'")
    parities = tuple(ramond_ground_parities)
    if len(parities) != 2 or any(parity not in (0, 1) for parity in parities):
        raise ValueError(
            "ramond_ground_parities must contain the parities at (one,infinity)"
        )
    beta_1 = complex(beta_one)
    beta_4 = complex(beta_infinity)
    if chirality == "holomorphic":
        terminal = hjs_ns_rr_polynomial_ground_tensor(
            sp.sympify(beta_1),
            sp.sympify(beta_4),
            structure_sign=sign,
        )
        orientation_phase: complex = -1.0j
    else:
        terminal = antiholomorphic_crossed_ns_rr_polynomial_ground_tensor(
            sp.sympify(beta_1),
            sp.sympify(beta_4),
            structure_sign=sign,
        )
        orientation_phase = 1.0j

    direct = direct_two_ramond_internal_ns_block_series(
        c=c,
        h_internal=h_internal,
        external_weights=(
            h_ns_zero,
            h_ns_w,
            ramond_weight(c, beta_1),
            ramond_weight(c, beta_4),
        ),
        rr_ground_tensor=terminal,
        maximum_twice_level=maximum_twice_level,
        component=component,
        rr_orientation_phase=orientation_phase,
        chirality=chirality,
        ns_words=ns_words,
        ramond_ground_parities=parities,
        digits=digits,
        condition_limit=condition_limit,
    )
    external_factor = _crossed_hjs_external_ground_factor(
        beta_1,
        beta_4,
        ramond_ground_parities=parities,  # type: ignore[arg-type]
        chirality=chirality,
    )
    return replace(
        direct,
        coefficients={
            level: coefficient / external_factor
            for level, coefficient in direct.coefficients.items()
        },
        rr_structure_sign=sign,
        external_ground_basis="hjs",
    )


def two_ramond_internal_ns_sld_chiral_block(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    component: FourRamondComponent,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ns_words: Sequence[NSWord] = ((), ()),
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> TwoRamondInternalNSBlockSeries:
    r"""Build the crossed SLD block for an original R--NS--NS--R ordering.

    Input momenta and NS words follow

    ``(R1,NS2,NS3,R4)=(0,z,1,infinity)``.

    With ``w=1-z`` the standardized crossed order is
    ``(NS3,NS2,R1,R4)=(0,w,1,infinity)``; the wrapper performs exactly this
    permutation.  It does not attach either RRNS/NSNSNS structure constant.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p_R1,p_NS2,p_NS3,p_R4)")
    words = tuple(tuple(word) for word in ns_words)
    if len(words) != 2:
        raise ValueError("ns_words must contain the words on original legs (2,3)")
    p_r1, p_ns2, p_ns3, p_r4 = momenta
    scale = math.sqrt(2.0)
    return hjs_two_ramond_internal_ns_block_series(
        c=13.5,
        h_internal=ns_weight(internal_momentum),
        beta_one=1.0j * p_r1 / scale,
        beta_infinity=1.0j * p_r4 / scale,
        h_ns_zero=ns_weight(p_ns3),
        h_ns_w=ns_weight(p_ns2),
        maximum_twice_level=maximum_twice_level,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        ns_words=(words[1], words[0]),
        ramond_ground_parities=ramond_ground_parities,
        digits=digits,
        condition_limit=condition_limit,
    )


def spin7_zero_vector_correlator() -> complex:
    r"""Coefficient of ``C_{alpha beta}`` for two spin fields."""

    return 1.0 + 0.0j


def spin7_one_vector_correlator(
    z: complex,
    *,
    vector_position: Literal["z", "one"],
) -> complex:
    r"""Coefficient of ``(C gamma^a)`` for two spin fields and one fermion."""

    z = complex(z)
    if z == 0:
        raise ValueError("the spin-field OPE power is singular at z=0")
    if vector_position == "z":
        return cmath.exp(-0.5 * cmath.log(z)) / math.sqrt(2.0)
    if vector_position == "one":
        return 1.0 / math.sqrt(2.0)
    raise ValueError("vector_position must be 'z' or 'one'")


def spin7_two_vector_correlator(z: complex) -> Spin7TwoVectorCorrelator:
    r"""Return the exact two-spin/two-fermion spectator correlator.

    With ``gamma^{ab}=[gamma^a,gamma^b]/2``, the result is

    ``z^-1/2 * ((1+z)/(2(1-z)) delta^{ab} C - 1/2 C gamma^{ab})``.
    """

    z = complex(z)
    if z in (0, 1):
        raise ValueError("the correlator is singular at z=0 or z=1")
    root_factor = cmath.exp(-0.5 * cmath.log(z))
    return Spin7TwoVectorCorrelator(
        scalar=root_factor * (1.0 + z) / (2.0 * (1.0 - z)),
        bivector=-0.5 * root_factor,
    )


def local_r_plus_sld_branches() -> tuple[LocalRamondSLDBranch, ...]:
    r"""Return the four exact branches of a local ``R+`` SLD correlator.

    The antiholomorphic block carries the same two HJS signs as its
    holomorphic partner.  Under analytic continuation its momenta are not
    numerically complex-conjugated.
    """

    return tuple(
        LocalRamondSLDBranch(left, right, 0.25)
        for left, right in product((-1, 1), repeat=2)
    )


def standard_heterotic_structure_sign_pairs(
    external_polarization_product: int,
) -> tuple[tuple[int, int], ...]:
    r"""Return the conditional order/disorder Fourier projection.

    If ``eta4*eta1`` is the product of the two external Ramond
    polarizations, the repository HJS signs obey

    ``s_left*s_right = -eta4*eta1``.

    The minus sign comes from rewriting Suchanek's right trinion from
    internal momentum ``-P`` to ``+P``.  This is the standard type-0B-like
    polarization convention.  A finite-wall heterotic vertex/cocycle
    convention is still required to certify relative PCO and spectator
    phases.
    """

    if external_polarization_product not in (-1, 1):
        raise ValueError("external_polarization_product must be +1 or -1")
    required_product = -external_polarization_product
    return tuple(
        (left, right)
        for left, right in product((-1, 1), repeat=2)
        if left * right == required_product
    )


def no_screen_vv_sld_spectator_coefficients(
    p_ramond_at_zero: complex,
    p_ns_at_z: complex,
) -> NoScreenVVBranchCoefficients:
    r"""Return the exact beta-reduced ``V V R R`` resonance branch.

    The ordering is ``(R1,NS2,NS_in,R4)=(0,z,1,infinity)`` and the
    screening-free resonance imposes ``p_in=i`` and
    ``p4=i-p1-p2``.  The surviving pinch is the mixed
    ``C_even(left)*C_odd(right)`` branch.  Its contour residue, complex beta
    integrals, and two-vector Spin(7) correlator reduce exactly to

    ``scalar   = -pi/2 * (1+i*(2*p1+p2))``
    ``bivector = -i*pi*p2/2``.

    This is a convention-independent checkpoint for the interacting-wall
    calculation, not by itself the complete heterotic amplitude.
    """

    p1 = complex(p_ramond_at_zero)
    p2 = complex(p_ns_at_z)
    return NoScreenVVBranchCoefficients(
        scalar=-0.5 * math.pi * (1.0 + 1.0j * (2.0 * p1 + p2)),
        bivector=-0.5j * math.pi * p2,
    )


def no_screen_minus_pco_coefficient() -> complex:
    r"""Return the conventional HJS-minus PCO sum at ``p_in=i``.

    With the standard free-Ising cocycle, the SLD-descendant and time-
    fermion terms combine to ``-(1+i)/2``.  The repository does not yet
    contain the finite-wall heterotic vertex convention needed to promote
    this conditional phase choice to a convention-independent statement.
    """

    return -(1.0 + 1.0j) / 2.0


def standard_projected_v_to_vrr_no_screen_resonance(
    p_ramond_at_zero: complex,
    p_vector_at_z: complex,
) -> NoScreenVVBranchCoefficients:
    r"""Attach the standard conditional GSO/PCO factors to ``V V R R``.

    The factor is ``C^(+)C^(-)/(C_even*C_odd)=1/4`` times the conventional
    HJS-minus PCO coefficient.  This helper is intentionally named
    ``standard_projected`` rather than ``physical`` because the heterotic
    finite-wall cocycle phase has not been fixed in the repository.
    """

    branch = no_screen_vv_sld_spectator_coefficients(
        p_ramond_at_zero,
        p_vector_at_z,
    )
    factor = 0.25 * no_screen_minus_pco_coefficient()
    return NoScreenVVBranchCoefficients(
        scalar=factor * branch.scalar,
        bivector=factor * branch.bivector,
    )


def standard_projected_v_to_srr_no_screen_resonance(
    p_singlet_at_z: complex,
) -> complex:
    r"""Return the conditional first-resonance ``V -> S R R`` coefficient.

    The result multiplies ``(C gamma^a)`` and equals
    ``pi*p_s*(p_s-i)/(8*sqrt(2))`` in the standard cocycle convention.
    """

    p_s = complex(p_singlet_at_z)
    return math.pi * p_s * (p_s - 1.0j) / (8.0 * math.sqrt(2.0))


def ramond_sld_structure_product(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    left_structure_sign: int,
    right_structure_sign: int,
    precision: int = 40,
) -> complex:
    r"""Return the two exact RRNS constants for an R--NS--NS--R channel.

    External order is ``(R1,NS2,NS3,R4)=(0,z,1,infinity)``.  No complex
    conjugation is made when the momenta are analytically continued.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p1,p2,p3,p4)")
    p1, p2, p3, p4 = momenta
    internal = complex(internal_momentum)
    left = rr_ns_chiral_structure_constant(
        p4,
        internal,
        p3,
        structure_sign=left_structure_sign,
        precision=precision,
    )
    right = rr_ns_chiral_structure_constant(
        internal,
        p1,
        p2,
        structure_sign=right_structure_sign,
        precision=precision,
    )
    return left * right


def ramond_sld_chiral_branch(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    left_structure_sign: int,
    right_structure_sign: int,
    ns_words: Sequence[NSWord] = ((), ()),
    form_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondSphereBlockSeries:
    r"""Build one exact finite-level HJS R--NS--NS--R chiral block."""

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p1,p2,p3,p4)")
    if len(tuple(ns_words)) != 2:
        raise ValueError("ns_words must contain the words on legs (2,3)")
    p1, p2, p3, p4 = momenta
    scale = math.sqrt(2.0)
    return hjs_ramond_sphere_block_series(
        c=13.5,
        beta_internal=1j * complex(internal_momentum) / scale,
        beta_one=1j * p1 / scale,
        beta_four=1j * p4 / scale,
        h_two=ns_weight(p2),
        h_three=ns_weight(p3),
        maximum_twice_level=maximum_twice_level,
        form_parities=form_parities,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        ns_words=ns_words,
        digits=digits,
        condition_limit=condition_limit,
    )


def antiholomorphic_ramond_sld_chiral_branch(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    left_structure_sign: int,
    right_structure_sign: int,
    ns_words: Sequence[NSWord] = ((), ()),
    form_parities: Sequence[int] = (0, 0),
    external_ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondSphereBlockSeries:
    r"""Build the exact anti-HJS R--NS--NS--R chiral block.

    The puncture order is ``(R1,NS2,NS3,R4)=(0,z,1,infinity)``.  This is a
    separate antiholomorphic tensor path, not a numerical conjugation of
    :func:`ramond_sld_chiral_branch`: the same analytically continued
    momenta are inserted while HJS convention phases are conjugated.

    ``external_ramond_ground_parities`` selects ``(+,-)`` HJS ground states
    on legs ``(1,4)``.  This permits the four components required when the
    two local ``R+`` polarizations are expanded.  ``ns_words`` independently
    selects the primary/descendant patterns needed for the mixed ``V/S``
    components.  The returned object is still one chiral integrand block;
    it contains no GSO sum, spectator factor, or modulus integration.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p1,p2,p3,p4)")
    words = tuple(tuple(word) for word in ns_words)
    if len(words) != 2:
        raise ValueError("ns_words must contain the words on legs (2,3)")
    ground_parities = tuple(external_ramond_ground_parities)
    if len(ground_parities) != 2 or any(
        parity not in (0, 1) for parity in ground_parities
    ):
        raise ValueError(
            "external_ramond_ground_parities must contain two zeros or ones"
        )
    left_sign = _validate_hjs_structure_sign(
        left_structure_sign, "left_structure_sign"
    )
    right_sign = _validate_hjs_structure_sign(
        right_structure_sign, "right_structure_sign"
    )

    p1, p2, p3, p4 = momenta
    scale = math.sqrt(2.0)
    beta_internal = 1.0j * complex(internal_momentum) / scale
    beta_one = 1.0j * p1 / scale
    beta_four = 1.0j * p4 / scale
    h_internal = ramond_weight(13.5, beta_internal)
    h_one = ramond_weight(13.5, beta_one)
    h_four = ramond_weight(13.5, beta_four)
    states = tuple(ramond_ground_state(parity) for parity in ground_parities)
    right_tensor = antiholomorphic_hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_internal),
        sp.sympify(beta_one),
        structure_sign=right_sign,
    )
    left_tensor = antiholomorphic_hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_four),
        sp.sympify(beta_internal),
        structure_sign=left_sign,
    )
    series = direct_ramond_sphere_block_series(
        c=13.5,
        h_internal=h_internal,
        external_weights=(h_one, ns_weight(p2), ns_weight(p3), h_four),
        left_ground_tensor=left_tensor,
        right_ground_tensor=right_tensor,
        maximum_twice_level=maximum_twice_level,
        form_parities=form_parities,
        ramond_states=states,
        ns_words=words,
        digits=digits,
        condition_limit=condition_limit,
    )
    external_factor = _antiholomorphic_hjs_external_state_factor(
        ground_parities[0], beta_one, slot="zero"
    ) * _antiholomorphic_hjs_external_state_factor(
        ground_parities[1], beta_four, slot="infinity"
    )
    return replace(
        series,
        coefficients={
            level: coefficient / external_factor
            for level, coefficient in series.coefficients.items()
        },
        external_ground_basis="hjs",
    )


def four_ramond_sld_chiral_block(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    component: FourRamondComponent,
    left_structure_sign: int,
    right_structure_sign: int,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> FourRamondNSBlockSeries:
    r"""Build one four-Ramond, internal-NS super-Liouville chiral block.

    At ``b=1`` the HJS Ramond parameters are ``beta_i=i*p_i/sqrt(2)`` and
    the internal NS weight is :func:`spin23_super_liouville_data.ns_weight`.
    Structure constants and the local nonchiral GSO assembly are deliberately
    not included.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p1,p2,p3,p4)")
    scale = math.sqrt(2.0)
    betas = tuple(1.0j * momentum / scale for momentum in momenta)
    return hjs_four_ramond_ns_block_series(
        c=13.5,
        h_internal=ns_weight(complex(internal_momentum)),
        external_betas=betas,
        maximum_twice_level=maximum_twice_level,
        component=component,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        digits=digits,
        condition_limit=condition_limit,
    )


def four_ramond_resonance_incoming_momentum(screening_number: int) -> complex:
    r"""Return ``i*(s+2)/2`` for a four-Ramond wall resonance.

    The four physical Ramond polarization considered here has nonzero
    free-field charge balance only for odd ``s``.  In particular ``s=0`` is
    the vanishing screening-free term and ``s=1`` gives the first nonzero
    resonance, ``p_in=3i/2``.
    """

    if not isinstance(screening_number, int) or screening_number < 0:
        raise ValueError("screening_number must be a nonnegative integer")
    return 0.5j * (screening_number + 2)


def four_ramond_resonance_can_be_nonzero(screening_number: int) -> bool:
    """Return whether the Ramond spin/H-charge rule permits this resonance."""

    if not isinstance(screening_number, int) or screening_number < 0:
        raise ValueError("screening_number must be a nonnegative integer")
    return screening_number % 2 == 1


def ns_pair_channel_divisor(momentum_one: complex, momentum_two: complex) -> complex:
    r"""Return ``1+i*(p_one+p_two)``, the first NS pair divisor."""

    return 1.0 + 1.0j * (complex(momentum_one) + complex(momentum_two))


def ramond_pair_channel_divisor(
    momentum_ns: complex,
    momentum_ramond: complex,
) -> complex:
    r"""Return ``1/2+i*(p_NS+p_R)``, the first Ramond pair divisor.

    The channel energy is ``i/2`` on this divisor.  The pinched SLD module
    instead has momentum ``P=3i/2``.
    """

    return 0.5 + 1.0j * (
        complex(momentum_ns) + complex(momentum_ramond)
    )


def pair_channel_divisor(momentum_one: complex, momentum_two: complex) -> complex:
    r"""Backward-compatible alias for :func:`ns_pair_channel_divisor`.

    Mixed R--NS channels must use :func:`ramond_pair_channel_divisor`.
    """

    return ns_pair_channel_divisor(momentum_one, momentum_two)


__all__ = [
    "ALLOWED_ONE_TO_THREE_PROCESSES",
    "Chirality",
    "FIRST_FOUR_RAMOND_RESONANT_INCOMING_MOMENTUM",
    "FIRST_FOUR_RAMOND_SCREENING_NUMBER",
    "FIRST_RAMOND_CHANNEL_ENERGY",
    "FIRST_RAMOND_INTERNAL_DEGENERATE_MOMENTUM",
    "FourRamondNSBlockSeries",
    "INDEPENDENT_RAMOND_CORRELATORS",
    "IndependentRamondCorrelator",
    "LocalRamondSLDBranch",
    "NoScreenVVBranchCoefficients",
    "RamondFourPointProcess",
    "SO7_SPINOR_DIMENSION",
    "SO7_VECTOR_DIMENSION",
    "SPHERE_REQUIRED_PICTURE",
    "Spin7TwoVectorCorrelator",
    "TwoRamondInternalNSBlockSeries",
    "antiholomorphic_crossed_ns_rr_polynomial_ground_tensor",
    "antiholomorphic_hjs_ground_tensor",
    "antiholomorphic_hjs_to_polynomial_ground_tensor",
    "antiholomorphic_ramond_sld_chiral_branch",
    "direct_two_ramond_internal_ns_block_series",
    "four_ramond_resonance_can_be_nonzero",
    "four_ramond_resonance_incoming_momentum",
    "four_ramond_sld_chiral_block",
    "hjs_four_ramond_ns_block_series",
    "hjs_four_ramond_ns_trinion_values",
    "hjs_two_ramond_internal_ns_block_series",
    "local_r_plus_sld_branches",
    "no_screen_minus_pco_coefficient",
    "no_screen_vv_sld_spectator_coefficients",
    "ns_pair_channel_divisor",
    "pair_channel_divisor",
    "ramond_pair_channel_divisor",
    "ramond_sld_chiral_branch",
    "ramond_sld_structure_product",
    "required_pco_count",
    "standard_heterotic_structure_sign_pairs",
    "standard_projected_v_to_srr_no_screen_resonance",
    "standard_projected_v_to_vrr_no_screen_resonance",
    "spin7_one_vector_correlator",
    "spin7_two_vector_correlator",
    "spin7_zero_vector_correlator",
    "two_ramond_internal_ns_sld_chiral_block",
]
