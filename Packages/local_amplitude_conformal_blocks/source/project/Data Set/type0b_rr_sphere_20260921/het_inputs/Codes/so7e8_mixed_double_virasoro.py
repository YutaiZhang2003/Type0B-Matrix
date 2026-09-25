"""Genus-zero NS NS | R R blocks via F x SVir -> Vir x Vir.

The auxiliary NS legs are identities. Thus the auxiliary vacuum is the
only state supported by their trinion, and the enlarged sphere block equals
the physical mixed block (the auxiliary R two-point normalization is one).
No genus-two auxiliary-series division is needed.

Finite PBW *branching vectors* are constructed with the existing embedding
engine. Internal descendant towers are evaluated by ordinary Virasoro
c-recursion, never by full superconformal Gram inversion.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import G, fermion_parity
from ramond_algebra.ramond_sca import PBWState
from ramond_algebra.ns_rr_three_point_tensor import ns_rr_three_point_from_ground_tensor
from so7e8_ramond_fourpoint import TwoRamondInternalNSBlockSeries
from spin23_genus1_recursion import FinitePartDiagnostics, dictionary_finite_part
from spin23_ns_torus_two_virasoro import _oriented_ns_branching_coefficient
from spin23_two_virasoro_ramond import (
    AuxiliaryFermionState, TensorBasisState,
    auxiliary_ns_r_r_three_point, embedded_branch_state,
)
from virasoro_sphere_c_recursion import sphere_c_coefficients


@dataclass(frozen=True)
class MixedDoubleVirasoroResult:
    series: TwoRamondInternalNSBlockSeries
    branch_numbers: tuple[Fraction, ...]
    max_eigen_residual: float
    min_spectral_gap: float
    max_external_resolution_residual: float


@dataclass(frozen=True)
class SelfDualMixedDoubleVirasoroResult:
    series: TwoRamondInternalNSBlockSeries
    diagnostics: dict[int, FinitePartDiagnostics]
    generic_evaluations: int
    max_eigen_residual: float
    min_spectral_gap: float


_RR_PARAMETERS = sp.symbols("h h1 h4 c t00 t01 t10 t11")


@lru_cache(maxsize=4096)
def _rr_ward_template(ns_word, one_state, infinity_state):
    """Compile a finite branching Ward polynomial once, not at each P or b."""
    h, h1, h4, c, *terminal = _RR_PARAMETERS
    expression = ns_rr_three_point_from_ground_tensor(
        ns_word, one_state, infinity_state, h_infinity=h, h_middle=h1,
        h_zero=h4, c=c, ground_tensor=sp.Matrix(2, 2, terminal),factor_result=False)
    # The Ward reducer already returns the exact polynomial. Multivariate
    # algebraic factorization becomes very expensive at the new n=3/2
    # branch (level 9/2), and is unnecessary for numerical compilation.
    return sp.lambdify(_RR_PARAMETERS, expression, modules="numpy", cse=False,
                       docstring_limit=0)


def _sphere_rr_product_form(ns_state, one_state, infinity_state, *,
                            h, h1, h4, c, terminal, form_parity):
    """NS-at-infinity oriented restriction, auxiliary form even.

    This is the left restriction in _theta_elementary_product_form, not
    the genus-two sewing sign. Its Koszul exponent is
    (p_super + nu_one)*mu_infinity + mu_internal. The exact Ward polynomial
    is compiled once and reused over the contour and momentum samples.
    """
    mu = tuple(state.auxiliary.parity for state in (ns_state, one_state, infinity_state))
    nu = (fermion_parity(ns_state.super_state), one_state.super_state.parity,
          infinity_state.super_state.parity)
    if sum(mu) % 2 or sum(nu) % 2 != form_parity:
        return 0j
    auxiliary = auxiliary_ns_r_r_three_point(
        ns_state.auxiliary, one_state.auxiliary, infinity_state.auxiliary, form_parity=0)
    if auxiliary == 0:
        return 0j
    super_value = _rr_ward_template(ns_state.super_state, one_state.super_state,
                                    infinity_state.super_state)(h, h1, h4, c, *terminal)
    return complex((-1) ** ((form_parity + nu[1]) * mu[2] + mu[0])
                   * auxiliary * super_value)


@lru_cache(maxsize=4096)
def _external_resolution(b: complex, momentum: complex, sector: str, parity: int):
    if sector == "NS":
        labels = (Fraction(0),) if parity == 0 else (Fraction(-1, 2), Fraction(1, 2))
        target = TensorBasisState(AuxiliaryFermionState("NS"), () if parity == 0 else (G(Fraction(-1, 2)),))
    else:
        labels = (Fraction(-1, 4), Fraction(1, 4))
        target = TensorBasisState(AuxiliaryFermionState("R"), PBWState((), parity))
    branches = tuple(embedded_branch_state(
        b=b, sector=sector, physical_momentum=momentum, branch_number=n,
        parity=parity) for n in labels)
    basis = branches[0].basis
    if any(branch.basis != basis for branch in branches):
        raise AssertionError("external branch bases differ")
    vector = np.array([complex(state == target) for state in basis])
    matrix = np.array([branch.coefficients for branch in branches]).T
    coefficients = np.linalg.solve(matrix, vector)
    residual = float(np.linalg.norm(matrix @ coefficients - vector))
    if residual > 1.0e-10:
        raise ArithmeticError("external double-Virasoro resolution failed")
    return tuple(zip(coefficients, branches)), residual


@lru_cache(maxsize=16384)
def _rr_branch_form(internal, one, infinity, structure_sign, form_parity):
    total = 0j
    beta_one = 1j * one.physical_momentum / math.sqrt(2)
    beta_infinity = 1j * infinity.physical_momentum / math.sqrt(2)
    phase = cmath.exp(1j * math.pi / 4)
    terminal = (1, phase * beta_infinity, structure_sign * 1j * phase * beta_one,
                structure_sign * 1j * beta_one * beta_infinity)
    for (a, state_a), (b, state_b), (d, state_d) in itertools.product(
        zip(internal.coefficients, internal.basis),
        zip(one.coefficients, one.basis),
        zip(infinity.coefficients, infinity.basis),
    ):
        total += a * b * d * _sphere_rr_product_form(
            state_a, state_b, state_d, h=internal.super_weight,
            h1=one.super_weight, h4=infinity.super_weight,
            c=internal.super_central_charge, terminal=terminal, form_parity=form_parity)
    return complex(total)


@lru_cache(maxsize=16384)
def _double_virasoro_descendants(internal_parameters, external_parameters, order):
    factors = tuple(sphere_c_coefficients(
        c=getattr(internal_parameters, f"c_{copy}"),
        h=getattr(internal_parameters, f"h_{copy}"),
        external_weights=tuple(getattr(parameters, f"h_{copy}")
                               for parameters in external_parameters),
        order=order) for copy in (1, 2))
    return tuple(sum(factors[0][j] * factors[1][level - j]
                     for j in range(level + 1)) for level in range(order + 1))


def mixed_double_virasoro_coefficients(
    *, b: complex, internal_momentum: complex, external_momenta,
    maximum_twice_level: int, component: str, rr_structure_sign: int,
    ns_stars=(False, False), ramond_ground_parities=(0, 0),
    chirality: str = "holomorphic",
) -> MixedDoubleVirasoroResult:
    """Generic-b HJS block in order (NS_0,NS_z,R_1,R_inf).

    NS stars mean G_{-1/2}, with its unrescaled physical normalization.
    The parameters are Liouville momenta: h_NS=Q^2/8+P^2/2 and
    h_R=c/24+P^2/2. b=1 is a singular embedding, not a valid input here.
    """
    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    if component not in ("even", "odd") or rr_structure_sign not in (-1, 1):
        raise ValueError("invalid parity component or RR structure sign")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("invalid chirality")
    momenta = tuple(map(complex, external_momenta))
    stars = tuple(ns_stars)
    parities = tuple(ramond_ground_parities)
    if len(momenta) != 4 or len(stars) != 2 or len(parities) != 2:
        raise ValueError("expected four momenta, two NS stars, two R parities")
    if any(p not in (0, 1) for p in stars + parities):
        raise ValueError("stars and parities must be zero or one")
    stars = tuple(map(int, stars))
    b = complex(b)
    if b == 0 or abs(b * b - 1) < 1.0e-12:
        raise ValueError("the generic-b embedding requires b != 0 and b^2 != 1")
    if any(not (math.isfinite(z.real) and math.isfinite(z.imag))
           for z in (b, complex(internal_momentum)) + momenta):
        raise ValueError("b and all momenta must be finite")
    q_background = b + 1 / b
    c = 1.5 + 3 * q_background ** 2
    h = q_background ** 2 / 8 + complex(internal_momentum) ** 2 / 2
    resolutions = tuple(_external_resolution(b, p, sector, parity)
                        for p, sector, parity in zip(momenta, ("NS", "NS", "R", "R"), stars + parities))
    internal_parity = int(component == "odd")
    bound = math.isqrt(maximum_twice_level)
    labels = tuple(Fraction(k, 2) for k in range(-bound, bound + 1) if k % 2 == internal_parity)
    coefficients = dict.fromkeys(range(internal_parity, maximum_twice_level + 1, 2), 0j)
    branches_seen = [branch for resolution, _ in resolutions for _, branch in resolution]
    for n in labels:
        internal = embedded_branch_state(
            b=b, sector="NS", physical_momentum=internal_momentum, branch_number=n)
        branches_seen.append(internal)
        onset = int((2 * n) ** 2)
        order = (maximum_twice_level - onset) // 2
        for external in itertools.product(*(resolution for resolution, _ in resolutions)):
            external_coefficient = math.prod(coefficient for coefficient, _ in external)
            zero, middle, one, infinity = (branch for _, branch in external)
            right = _oriented_ns_branching_coefficient(
                internal, middle, zero, orientation="left",
                super_form_parity=(internal_parity + sum(stars)) % 2)
            left = _rr_branch_form(internal, one, infinity, rr_structure_sign,
                                   (internal_parity + sum(parities)) % 2)
            weight = external_coefficient * left * right / internal.norm
            if weight == 0:
                continue
            descendants = _double_virasoro_descendants(
                internal.parameters, tuple(branch.parameters for _, branch in external), order)
            for level, value in enumerate(descendants):
                coefficients[onset + 2 * level] += weight * value
    phase = cmath.exp(1j * math.pi / 4)
    external_ground_factor = math.prod(
        (phase * 1j * momentum / math.sqrt(2)) if parity else 1
        for momentum, parity in zip(momenta[2:], parities))
    if external_ground_factor == 0:
        raise ValueError("odd HJS ground states are singular at zero momentum")
    orientation = (-1j * (-1) ** parities[1]) ** internal_parity
    if chirality == "antiholomorphic":
        # The crossed anti-HJS polynomial tensor equals the holomorphic
        # tensor in the even homogeneous form, and i times it in the odd
        # form. Include the +i internal orientation and external basis
        # factors. This changes convention phases, NOT complex momenta.
        form_parity = (internal_parity + sum(parities)) % 2
        orientation *= ((-1) ** internal_parity * 1j ** form_parity
                        / ((-1j) ** parities[0] * 1j ** parities[1]))
    series = TwoRamondInternalNSBlockSeries(
        coefficients={level: complex(value * orientation / external_ground_factor) for level, value in coefficients.items()},
        component=component, chirality=chirality, c=c, h_internal=h,
        external_weights=tuple((q_background ** 2 / 8 if index < 2 else c / 24) + p ** 2 / 2 for index, p in enumerate(momenta)),
        ns_words=tuple((G(Fraction(-1, 2)),) if star else () for star in stars),
        ramond_ground_parities=parities, rr_structure_sign=rr_structure_sign,
        rr_orientation_phase=(-1j if chirality == "holomorphic" else 1j), external_ground_basis="hjs",
        maximum_twice_level=maximum_twice_level, gram_condition_numbers={})
    return MixedDoubleVirasoroResult(
        series, labels, max(branch.eigen_residual for branch in branches_seen),
        min(branch.spectral_gap for branch in branches_seen),
        max(residual for _, residual in resolutions))


def self_dual_mixed_double_virasoro_coefficients(
    *, internal_momentum: complex, external_momenta, maximum_twice_level: int,
    component: str, rr_structure_sign: int, ns_stars=(False, False),
    ramond_ground_parities=(0, 0), chirality: str = "holomorphic",
    radius: float = 0.04, check_radius: float = 0.05, samples: int = 24,
    tolerance: float = 1.0e-8,
) -> SelfDualMixedDoubleVirasoroResult:
    """Physical c=27/2 limit, projecting the *assembled* branch sum.

    Liouville momenta are held fixed on b=exp(t). A two-radius Cauchy
    constant-term projection removes the removable b=1 embedding singularity.
    A failed radius check raises; it never substitutes PBW coefficients.
    The radius diagnostic is not an all-orders error proof: sample-count
    refinement and independent PBW tests are also required in applications.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    momenta = tuple(map(complex, external_momenta))
    records = []

    def evaluate(b):
        result = mixed_double_virasoro_coefficients(
            b=b, internal_momentum=internal_momentum, external_momenta=momenta,
            maximum_twice_level=maximum_twice_level, component=component,
            rr_structure_sign=rr_structure_sign, ns_stars=ns_stars,
            ramond_ground_parities=ramond_ground_parities, chirality=chirality)
        records.append(result)
        return result.series.coefficients

    keys = tuple(range(int(component == "odd"), maximum_twice_level + 1, 2))
    coefficients, diagnostics = dictionary_finite_part(
        evaluate, keys=keys, radius=radius, check_radius=check_radius,
        samples=samples)
    for level, diagnostic in diagnostics.items():
        if diagnostic.absolute_error > tolerance * max(1, abs(diagnostic.value)):
            raise ArithmeticError(
                f"self-dual double-Virasoro radius check failed at level {level}/2: "
                f"difference={diagnostic.absolute_error:.3e}")
    series = replace(
        records[0].series, coefficients=coefficients, c=13.5 + 0j,
        h_internal=0.5 + complex(internal_momentum) ** 2 / 2,
        external_weights=tuple((0.5 if i < 2 else 9 / 16) + p ** 2 / 2
                               for i, p in enumerate(momenta)))
    return SelfDualMixedDoubleVirasoroResult(
        series, diagnostics, len(records),
        max(record.max_eigen_residual for record in records),
        min(record.min_spectral_gap for record in records))


def mixed_double_virasoro_elliptic_h_series(*, b: complex = 1.0, **kwargs):
    """Generate by double Vir + c-recursion, then re-expand/truncate in q.

    Returns the existing prefactored-H type, with t=sqrt(q). Unknown sewing
    coefficients are not inferred: the q series stops at the same known order.
    The physical cross-ratio here is the standardized mixed NS-channel w.
    """
    from ramond_sphere_uniformization import elliptically_prefactor_two_ramond_internal_ns_series

    builder = (self_dual_mixed_double_virasoro_coefficients if complex(b) == 1
               else mixed_double_virasoro_coefficients)
    result = builder(**kwargs) if complex(b) == 1 else builder(b=b, **kwargs)
    if not result.series.coefficients:
        # A cutoff below the first odd level denotes the zero series, not a
        # request to manufacture an unknown leading coefficient. Construct
        # the standard prefactor metadata but retain no H coefficients.
        placeholder = replace(result.series, coefficients={1: 0j})
        empty_h = elliptically_prefactor_two_ramond_internal_ns_series(placeholder)
        return replace(empty_h, source=result.series, h_coefficients=(),
                       known_through_twice_level=result.series.maximum_twice_level)
    return elliptically_prefactor_two_ramond_internal_ns_series(result.series)


def two_ramond_internal_ns_sld_double_virasoro_h_series(
    internal_momentum: complex, *, external_momenta, maximum_twice_level: int,
    component: str, rr_structure_sign: int, chirality: str = "holomorphic",
    ns_words=((), ()), ramond_ground_parities=(0, 0), **limit_options,
):
    """Amplitude adapter: (R_0,NS_z,NS_1,R_inf) -> (NS_1,NS_z,R_0,R_inf).

    Unlike the coefficient API, this uses the amplitude's original external
    order and swaps both NS words and momenta exactly once. Only primary and
    G_{-1/2} NS states are supported; other descendants are not reinterpreted.
    """
    momenta = tuple(external_momenta)
    words = tuple(tuple(word) for word in ns_words)
    star_word = (G(Fraction(-1, 2)),)
    if len(momenta) != 4 or len(words) != 2:
        raise ValueError("expected four external momenta and two NS words")
    if any(word not in ((), star_word) for word in words):
        raise NotImplementedError("double Virasoro supports only NS primaries and G_-1/2")
    return mixed_double_virasoro_elliptic_h_series(
        internal_momentum=internal_momentum,
        external_momenta=(momenta[2], momenta[1], momenta[0], momenta[3]),
        ns_stars=(bool(words[1]), bool(words[0])),
        maximum_twice_level=maximum_twice_level, component=component,
        rr_structure_sign=rr_structure_sign, chirality=chirality,
        ramond_ground_parities=ramond_ground_parities, **limit_options)
