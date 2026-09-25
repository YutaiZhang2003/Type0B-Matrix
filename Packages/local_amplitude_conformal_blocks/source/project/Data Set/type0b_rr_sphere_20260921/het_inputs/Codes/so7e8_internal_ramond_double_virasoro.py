"""Internal-R sphere blocks from F x SVir -> Vir x Vir.

External order: (R_0, NS_z, NS_1, R_infinity). NS auxiliary identities
project the auxiliary internal R line onto its selected ground state.
Only finite branch vectors use PBW data; all descendant towers use ordinary
Virasoro c-recursion. This chiral layer does not certify a nonchiral GSO
projector or enable the currently quarantined internal-R amplitude channel.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

from ns_algebra.ns_sca import G
from ramond_algebra.ramond_sca import ground_state
from ramond_algebra.nrr_three_point_tensor import hjs_to_polynomial_ground_tensor
from so7e8_ramond_fourpoint import antiholomorphic_hjs_to_polynomial_ground_tensor
from so7e8_mixed_double_virasoro import _external_resolution, _double_virasoro_descendants
from spin23_genus1_recursion import FinitePartDiagnostics, dictionary_finite_part
from spin23_ramond_blocks import RamondSphereBlockSeries
from spin23_two_virasoro_ramond import embedded_branch_state, oriented_branching_coefficient


@dataclass(frozen=True)
class InternalRamondDoubleVirasoroResult:
    series: RamondSphereBlockSeries
    branch_numbers: tuple[Fraction, ...]
    internal_parity: int | None
    chirality: str
    structure_signs: tuple[int, int]
    max_eigen_residual: float
    min_spectral_gap: float
    max_external_resolution_residual: float


@dataclass(frozen=True)
class SelfDualInternalRamondDoubleVirasoroResult:
    series: RamondSphereBlockSeries
    diagnostics: dict[int, FinitePartDiagnostics]
    generic_evaluations: int
    max_eigen_residual: float
    min_spectral_gap: float


def ramond_branch_numbers(maximum_twice_level: int) -> tuple[Fraction, ...]:
    """All n in 1/4 + Z/2 with twice-onset 4*n**2-1/4 <= cutoff."""
    if (not isinstance(maximum_twice_level, int)
            or isinstance(maximum_twice_level, bool) or maximum_twice_level < 0):
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    positive = [Fraction(2 * k + 1, 4) for k in range(math.isqrt(maximum_twice_level) + 1)
                if k * (k + 1) <= maximum_twice_level]
    return tuple(sorted([-n for n in positive] + positive))


@lru_cache(maxsize=16384)
def _sphere_rnsr_branch_form(infinity, middle, zero, structure_sign, form_parity):
    # Both are literal local R-infinity, NS-at-1, R-at-0 forms. No torus
    # trace twist or genus-two sewing sign is used on this sphere edge.
    return oriented_branching_coefficient(
        infinity, middle, zero, orientation="left", structure_sign=structure_sign,
        super_form_parity=form_parity, auxiliary_form_parity=0, auxiliary_parity_twist=1)


def internal_ramond_double_virasoro_coefficients(
    *, b: complex, internal_momentum: complex, external_momenta,
    maximum_twice_level: int, left_structure_sign: int = 1,
    right_structure_sign: int = 1, form_parities=(0, 0),
    ns_stars=(False, False), ramond_ground_parities=(0, 0),
    chirality: str = "holomorphic",
) -> InternalRamondDoubleVirasoroResult:
    """Generic-b HJS coefficients of an R--NS | NS--R sphere block.

    The two homogeneous form parities are (left,right), independently of
    the two HJS structure signs. Zero/one/two NS G_-1/2 insertions retain
    their original normalization. Parameters are Liouville momenta, not
    beta: beta=i*P/sqrt(2). Incompatible endpoint parities give a zero block.
    """
    labels = ramond_branch_numbers(maximum_twice_level)
    cutoff = maximum_twice_level - maximum_twice_level % 2
    momenta = tuple(map(complex, external_momenta))
    stars, parities, forms = tuple(ns_stars), tuple(ramond_ground_parities), tuple(form_parities)
    if len(momenta) != 4 or any(len(x) != 2 for x in (stars, parities, forms)):
        raise ValueError("expected four momenta and pairs of stars, R parities, and form parities")
    if any(x not in (0, 1) for x in stars + parities + forms):
        raise ValueError("stars and parities must be zero or one")
    if left_structure_sign not in (-1, 1) or right_structure_sign not in (-1, 1):
        raise ValueError("structure signs must be +1 or -1")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("invalid chirality")
    b, internal_momentum = complex(b), complex(internal_momentum)
    if b == 0 or abs(b * b - 1) < 1e-12:
        raise ValueError("generic-b embedding requires b != 0 and b^2 != 1")
    if any(not (math.isfinite(x.real) and math.isfinite(x.imag)) for x in (b, internal_momentum) + momenta):
        raise ValueError("b and all momenta must be finite")
    if internal_momentum == 0 or momenta[0] == 0 or momenta[3] == 0:
        raise ValueError("zero-momentum Ramond branch collisions require a separate limit")
    stars, parities, forms = tuple(map(int, stars)), tuple(map(int, parities)), tuple(map(int, forms))
    q_background = b + 1 / b
    c = 1.5 + 3 * q_background ** 2
    h = c / 24 + internal_momentum ** 2 / 2
    weights = tuple((c / 24 if i in (0, 3) else q_background ** 2 / 8) + p ** 2 / 2
                    for i, p in enumerate(momenta))
    betas = tuple(1j * p / math.sqrt(2) for p in (momenta[0], internal_momentum, momenta[3]))
    tensor_builder = (hjs_to_polynomial_ground_tensor if chirality == "holomorphic"
                      else antiholomorphic_hjs_to_polynomial_ground_tensor)
    left_tensor = tensor_builder(betas[2], betas[1], structure_sign=left_structure_sign)
    right_tensor = tensor_builder(betas[1], betas[0], structure_sign=right_structure_sign)
    coefficients = dict.fromkeys(range(0, cutoff + 1, 2), 0j)
    selected_right = (forms[1] + stars[0] + parities[0]) % 2
    selected_left = (forms[0] + stars[1] + parities[1]) % 2
    branches_seen, residuals = [], []
    if selected_left == selected_right:
        resolutions = tuple(_external_resolution(b, p, sector, parity)
                            for p, sector, parity in zip(momenta, ("R", "NS", "NS", "R"),
                                                         (parities[0], *stars, parities[1])))
        branches_seen.extend(branch for resolution, _ in resolutions for _, branch in resolution)
        residuals.extend(residual for _, residual in resolutions)
        for n in labels:
            internal = embedded_branch_state(b=b, sector="R", physical_momentum=internal_momentum,
                                             branch_number=n, parity=selected_right)
            branches_seen.append(internal)
            onset = int(4 * n * n - Fraction(1, 4))
            order = (cutoff - onset) // 2
            for external in product(*(resolution for resolution, _ in resolutions)):
                external_coefficient = math.prod(coefficient for coefficient, _ in external)
                zero, moving, one, infinity = (branch for _, branch in external)
                right = _sphere_rnsr_branch_form(internal, moving, zero, right_structure_sign, forms[1])
                left = _sphere_rnsr_branch_form(infinity, one, internal, left_structure_sign, forms[0])
                weight = external_coefficient * left * right / internal.norm
                if weight == 0:
                    continue
                descendants = _double_virasoro_descendants(
                    internal.parameters, tuple(branch.parameters for _, branch in external), order)
                for level, value in enumerate(descendants):
                    coefficients[onset + 2 * level] += weight * value
    phase = cmath.exp(1j * math.pi / 4)
    divisor = (phase * betas[0]) ** parities[0] * (1j * phase * betas[2]) ** parities[1]
    convention = 1 / divisor
    if chirality == "antiholomorphic":
        # Each odd homogeneous RNSR tensor changes by -i; the external
        # zero/infinity basis factors change by -i/+i, respectively.
        convention *= (-1j) ** sum(forms) / ((-1j) ** parities[0] * 1j ** parities[1])
    series = RamondSphereBlockSeries(
        coefficients={level: complex(value * convention) for level, value in coefficients.items()},
        c=c, h_internal=h, external_weights=weights,
        ramond_states=tuple(ground_state(parity) for parity in parities),
        ns_words=tuple((G(Fraction(-1, 2)),) if star else () for star in stars),
        left_ground_tensor=tuple(tuple(left_tensor[i, j] for j in range(2)) for i in range(2)),
        right_ground_tensor=tuple(tuple(right_tensor[i, j] for j in range(2)) for i in range(2)),
        form_parities=forms, external_ground_basis="hjs", maximum_twice_level=cutoff,
        gram_condition_numbers={})
    return InternalRamondDoubleVirasoroResult(
        series, labels if branches_seen else (), selected_right if branches_seen else None,
        chirality, (left_structure_sign, right_structure_sign),
        max((branch.eigen_residual for branch in branches_seen), default=0.),
        min((branch.spectral_gap for branch in branches_seen), default=1.),
        max(residuals, default=0.))


def self_dual_internal_ramond_double_virasoro_coefficients(
    *, internal_momentum: complex, external_momenta, maximum_twice_level: int,
    left_structure_sign: int = 1, right_structure_sign: int = 1,
    form_parities=(0, 0), ns_stars=(False, False), ramond_ground_parities=(0, 0),
    chirality: str = "holomorphic", radius: float = .04, check_radius: float = .05,
    samples: int = 24, tolerance: float = 1e-8,
) -> SelfDualInternalRamondDoubleVirasoroResult:
    """Assembled b=1 coefficient limit, at fixed physical Liouville momenta.

    Uses two Cauchy contours in log(b), as for the internal-NS implementation.
    The two-radius discrepancy is a diagnostic, not a rigorous error bound.
    Branch/pole/contour failures raise; no full PBW fallback is used.
    """
    ramond_branch_numbers(maximum_twice_level)  # Validate cutoff before contours.
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    momenta = tuple(map(complex, external_momenta))
    records = []

    def evaluate(b):
        result = internal_ramond_double_virasoro_coefficients(
            b=b, internal_momentum=internal_momentum, external_momenta=momenta,
            maximum_twice_level=maximum_twice_level, left_structure_sign=left_structure_sign,
            right_structure_sign=right_structure_sign, form_parities=form_parities,
            ns_stars=ns_stars, ramond_ground_parities=ramond_ground_parities, chirality=chirality)
        records.append(result)
        return result.series.coefficients

    keys = tuple(range(0, maximum_twice_level + 1, 2))
    coefficients, diagnostics = dictionary_finite_part(
        evaluate, keys=keys, radius=radius, check_radius=check_radius, samples=samples)
    for level, diagnostic in diagnostics.items():
        values = (diagnostic.value, diagnostic.check_value)
        if (any(not (math.isfinite(v.real) and math.isfinite(v.imag)) for v in values)
                or diagnostic.absolute_error > tolerance * max(1., *(abs(v) for v in values))):
            raise ArithmeticError(f"internal-R double-Virasoro radius check failed at level {level}/2")
    series = replace(records[0].series, coefficients=coefficients, c=13.5 + 0j,
                     h_internal=9 / 16 + complex(internal_momentum) ** 2 / 2,
                     external_weights=tuple((9 / 16 if i in (0, 3) else .5) + p ** 2 / 2
                                            for i, p in enumerate(momenta)))
    return SelfDualInternalRamondDoubleVirasoroResult(
        series, diagnostics, len(records), max(r.max_eigen_residual for r in records),
        min(r.min_spectral_gap for r in records))


def internal_ramond_double_virasoro_elliptic_h_series(*, b: complex = 1., **kwargs):
    """Re-expand the computed sewing series into the internal-R elliptic H.

    q=exp[-pi K(1-z)/K(z)] and t=sqrt(q). Uses the R-channel prefactor,
    not the crossed NS-channel one. The source sewing coefficients remain
    attached for OPE work; no unknown higher coefficient is supplied.
    """
    from ramond_sphere_uniformization import elliptically_prefactor_internal_ramond_series

    if complex(b) == 1:
        result = self_dual_internal_ramond_double_virasoro_coefficients(**kwargs)
    else:
        result = internal_ramond_double_virasoro_coefficients(b=b, **kwargs)
    return elliptically_prefactor_internal_ramond_series(result.series)


def ramond_sld_double_virasoro_h_series(
    internal_momentum: complex, *, external_momenta, maximum_twice_level: int,
    left_structure_sign: int = 1, right_structure_sign: int = 1,
    ns_words=((), ()), form_parities=(0, 0), external_ramond_ground_parities=(0, 0),
    chirality: str = "holomorphic", **limit_options,
):
    """Chiral amplitude adapter, already in (R0,NSz,NS1,Rinf) order.

    Does not permute external legs, sum nonchiral projectors, or change the
    signed momentum used by either three-point structure constant.
    """
    words = tuple(tuple(word) for word in ns_words)
    star_word = (G(Fraction(-1, 2)),)
    if len(words) != 2:
        raise ValueError("ns_words must contain the two NS external words")
    if any(word not in ((), star_word) for word in words):
        raise NotImplementedError("internal-R double Virasoro supports only NS primaries and G_-1/2")
    return internal_ramond_double_virasoro_elliptic_h_series(
        internal_momentum=internal_momentum, external_momenta=external_momenta,
        maximum_twice_level=maximum_twice_level, left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign, form_parities=form_parities,
        ns_stars=tuple(bool(word) for word in words), ramond_ground_parities=external_ramond_ground_parities,
        chirality=chirality, **limit_options)
