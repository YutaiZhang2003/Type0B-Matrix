#!/usr/bin/env python3
r"""Pointwise and spectral genus-zero five-point SO(23) ingredients.

This module is the next layer above :mod:`spin23_sphere_fivepoint`.  It
assembles the established heterotic sphere conventions at fixed
``(z2,z3,P1,P2)``:

* five external vector/singlet states at ``(0,z2,z3,1,infinity)``;
* three picture-zero vertices and the four nonzero total-supercurrent PCO
  branches (zero or two timelike-fermion insertions);
* the timelike-boson Koba--Nielsen factor;
* the flavored SO(23) free-fermion Wick factor;
* three delta-normalized super-Liouville structure constants;
* the diagonal two-edge NS block contraction, including component phases;
* and the two continuum resolutions ``dP1/pi dP2/pi``.

The conformal blocks with external ``G_-1/2`` words currently use the direct
finite-level oracle.  The primary ``h``-recursion is not silently substituted
for these physical PCO components.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_genlaguerre
from scipy.stats import qmc

from ns_algebra.ns_sca import Word, fermion_parity
from spin23_sphere_fivepoint import (
    EMPTY_WORD,
    G_MINUS_HALF,
    FormWeights,
    SphereFivePointSeries,
    direct_ns_sphere_fivepoint_series,
    ns_fivepoint_structure_weights,
    ns_liouville_weight,
)


StateKind = Literal["vector", "singlet"]
QuadratureMethod = Literal["gauss_laguerre", "gauss_legendre"]
LevelPair = tuple[int, int]


def _normalize_cutoffs(value: int | Sequence[int]) -> LevelPair:
    if isinstance(value, bool):
        raise TypeError("descendant cutoffs must be integers")
    result = (value, value) if isinstance(value, int) else tuple(value)
    if len(result) != 2 or any(
        isinstance(entry, bool) or not isinstance(entry, int) or entry < 0
        for entry in result
    ):
        raise ValueError("two nonnegative integer descendant cutoffs are required")
    return int(result[0]), int(result[1])


@dataclass(frozen=True)
class SphereFivePointState:
    r"""One on-shell external NS state.

    ``liouville_momentum`` labels the positive/reflected Liouville dressing;
    ``time_momentum`` is signed in ``exp(i*k*X0)``.  They are kept separate
    because the incoming leg has opposite signed time momentum.
    """

    kind: StateKind
    liouville_momentum: complex
    time_momentum: complex
    flavor: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("vector", "singlet"):
            raise ValueError("kind must be 'vector' or 'singlet'")
        values = (
            complex(self.liouville_momentum),
            complex(self.time_momentum),
        )
        if any(
            not math.isfinite(part)
            for value in values
            for part in (value.real, value.imag)
        ):
            raise ValueError("external momenta must be finite")
        if self.kind == "vector":
            if not isinstance(self.flavor, int) or not 0 <= self.flavor < 23:
                raise ValueError("a vector flavor must be an integer in [0,23)")
        elif self.flavor is not None:
            raise ValueError("a singlet has no SO(23) flavor")

    @classmethod
    def vector(
        cls,
        liouville_momentum: complex,
        time_momentum: complex,
        flavor: int,
    ) -> "SphereFivePointState":
        return cls("vector", liouville_momentum, time_momentum, flavor)

    @classmethod
    def singlet(
        cls,
        liouville_momentum: complex,
        time_momentum: complex,
    ) -> "SphereFivePointState":
        return cls("singlet", liouville_momentum, time_momentum)


def validate_fivepoint_states(
    states: Sequence[SphereFivePointState],
    *,
    neutrality_tolerance: float = 1.0e-10,
    mass_shell_tolerance: float = 1.0e-10,
) -> tuple[
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
]:
    """Validate five ordered external states and return a tuple."""

    normalized = tuple(states)
    if len(normalized) != 5 or any(
        not isinstance(state, SphereFivePointState) for state in normalized
    ):
        raise ValueError("exactly five SphereFivePointState objects are required")
    time_total = sum(complex(state.time_momentum) for state in normalized)
    if abs(time_total) > neutrality_tolerance:
        raise ValueError("signed timelike momenta must sum to zero")
    for state in normalized:
        p = complex(state.liouville_momentum)
        k = complex(state.time_momentum)
        scale = max(1.0, abs(p * p), abs(k * k))
        if abs(p * p - k * k) > mass_shell_tolerance * scale:
            raise ValueError("every external state must satisfy P**2=k**2")
    return normalized  # type: ignore[return-value]


def _validate_moduli(z_2: complex, z_3: complex) -> tuple[complex, complex]:
    z_2, z_3 = complex(z_2), complex(z_3)
    values = (z_2, z_3)
    if any(
        not math.isfinite(part)
        for value in values
        for part in (value.real, value.imag)
    ):
        raise ValueError("five-point moduli must be finite")
    if z_2 in (0, 1) or z_3 in (0, 1) or z_2 == z_3:
        raise ValueError("five-point insertions must not collide")
    q_1, q_2 = z_2 / z_3, z_3
    if not (0 < abs(q_1) < 1 and 0 < abs(q_2) < 1):
        raise ValueError("the selected linear channel requires 0<|q1|,|q2|<1")
    return q_1, q_2


def _positions(z_2: complex, z_3: complex) -> Mapping[int, complex | None]:
    return {1: 0.0j, 2: complex(z_2), 3: complex(z_3), 4: 1.0 + 0.0j, 5: None}


def _sphere_fermion_kernel(
    earlier_leg: int,
    later_leg: int,
    positions: Mapping[int, complex | None],
) -> complex:
    """Return the radial-order sphere Majorana kernel.

    Legs are supplied in descending radial/operator order.  The conformal
    limit of a weight-1/2 fermion at infinity gives unit contraction with a
    finite fermion.
    """

    earlier = positions[earlier_leg]
    later = positions[later_leg]
    if earlier is None:
        if later is None:
            raise ValueError("two insertions cannot both be at infinity")
        return 1.0 + 0.0j
    if later is None:
        raise ValueError("fermion legs must be supplied in radial order")
    return 1.0 / (earlier - later)


def _pfaffian_wick(
    ordered_legs: tuple[int, ...],
    positions: Mapping[int, complex | None],
    pair_allowed: Callable[[int, int], bool],
) -> complex:
    if len(ordered_legs) % 2:
        return 0.0j
    if not ordered_legs:
        return 1.0 + 0.0j
    first = ordered_legs[0]
    total = 0.0j
    for offset, second in enumerate(ordered_legs[1:], start=1):
        if not pair_allowed(first, second):
            continue
        rest = ordered_legs[1:offset] + ordered_legs[offset + 1 :]
        sign = -1 if (offset + 1) % 2 else 1
        total += (
            sign
            * _sphere_fermion_kernel(first, second, positions)
            * _pfaffian_wick(rest, positions, pair_allowed)
        )
    return total


def flavored_sphere_fermion_wick_factor(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
) -> complex:
    """Return the antiholomorphic SO(23) flavored-fermion correlator."""

    normalized = tuple(states)
    vector_legs = tuple(
        leg
        for leg in range(5, 0, -1)
        if normalized[leg - 1].kind == "vector"
    )
    flavors = {
        leg: int(normalized[leg - 1].flavor) for leg in vector_legs
    }
    return _pfaffian_wick(
        vector_legs,
        _positions(z_2, z_3),
        lambda first, second: flavors[first] == flavors[second],
    )


def timelike_boson_factor(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
) -> complex:
    r"""Return the five-point timelike Koba--Nielsen factor.

    The incoming leg is at infinity.  Complex external energies are not
    conjugated; geometric distances enter through real logarithms.
    """

    normalized = tuple(states)
    positions = _positions(z_2, z_3)
    exponent = 0.0j
    for first in range(1, 5):
        for second in range(first + 1, 5):
            distance = abs(
                complex(positions[second]) - complex(positions[first])
            )
            if distance == 0:
                raise ValueError("timelike vertex insertions collide")
            exponent -= (
                2.0
                * complex(normalized[first - 1].time_momentum)
                * complex(normalized[second - 1].time_momentum)
                * math.log(distance)
            )
    return cmath.exp(exponent)


@dataclass(frozen=True)
class SphereFivePointPCOComponent:
    """One nonzero term in the three-PCO total-supercurrent expansion."""

    time_fermion_legs: tuple[int, ...]
    holomorphic_liouville_words: tuple[Word, Word, Word, Word, Word]
    momentum_coefficient: complex

    def time_fermion_wick(self, z_2: complex, z_3: complex) -> complex:
        ordered = tuple(sorted(self.time_fermion_legs, reverse=True))
        return _pfaffian_wick(
            ordered,
            _positions(z_2, z_3),
            lambda _first, _second: True,
        )


def fivepoint_pco_components(
    states: Sequence[SphereFivePointState],
    *,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
) -> tuple[SphereFivePointPCOComponent, ...]:
    r"""Return the four even-``psi0`` terms from three picture-zero legs."""

    normalized = tuple(states)
    legs = tuple(zero_picture_legs)
    if len(legs) != 3 or len(set(legs)) != 3:
        raise ValueError("exactly three distinct zero-picture legs are required")
    if any(
        isinstance(leg, bool) or not isinstance(leg, int) or leg not in range(1, 5)
        for leg in legs
    ):
        raise ValueError("the current frame requires finite zero-picture legs 1..4")
    components: list[SphereFivePointPCOComponent] = []
    for count in (0, 2):
        for selected in combinations(legs, count):
            selected_set = set(selected)
            words = tuple(
                G_MINUS_HALF
                if leg in legs and leg not in selected_set
                else EMPTY_WORD
                for leg in range(1, 6)
            )
            coefficient = math.prod(
                complex(normalized[leg - 1].time_momentum) for leg in selected
            )
            components.append(
                SphereFivePointPCOComponent(
                    time_fermion_legs=tuple(selected),
                    holomorphic_liouville_words=words,  # type: ignore[arg-type]
                    momentum_coefficient=coefficient,
                )
            )
    return tuple(components)


def antiholomorphic_liouville_words(
    states: Sequence[SphereFivePointState],
) -> tuple[Word, Word, Word, Word, Word]:
    """Return primaries for vectors and ``G_-1/2`` for genuine singlets."""

    return tuple(
        EMPTY_WORD if state.kind == "vector" else G_MINUS_HALF
        for state in states
    )  # type: ignore[return-value]


def _word_parities(words: Sequence[Word]) -> tuple[int, int, int, int, int]:
    if len(words) != 5:
        raise ValueError("five external words are required")
    return tuple(fermion_parity(tuple(word)) for word in words)  # type: ignore[return-value]


def fivepoint_component_phase(
    words: Sequence[Word],
    edge_parities: Sequence[int],
) -> int:
    r"""Return the fixed-form to physical-superfield component phase.

    Each ordered trinion ``(infinity,middle,zero)`` contributes
    ``(-1)**(p_middle*(1-p_infinity))``.  The three infinity-slot parities
    are respectively the first edge, the second edge, and external leg 5.
    """

    p1, p2, p3, p4, p5 = _word_parities(words)
    routing = tuple(edge_parities)
    if len(routing) != 2 or any(value not in (0, 1) for value in routing):
        raise ValueError("edge_parities must be a pair of zeroes or ones")
    r1, r2 = routing
    exponent = p2 * (1 - r1) + p3 * (1 - r2) + p4 * (1 - p5)
    return -1 if exponent % 2 else 1


def _routing_for_structure_labels(
    words: Sequence[Word],
    first_form: int,
    third_form: int,
) -> tuple[tuple[int, int], int]:
    p1, p2, p3, p4, p5 = _word_parities(words)
    r1 = first_form ^ p1 ^ p2
    r2 = third_form ^ p4 ^ p5
    central_form = r1 ^ r2 ^ p3
    return (r1, r2), central_form


def diagonal_ns_fivepoint_contraction(
    holomorphic: SphereFivePointSeries,
    antiholomorphic: SphereFivePointSeries,
    q_1: complex,
    q_2: complex,
    form_weights: Sequence[Sequence[complex]],
    *,
    maximum_twice_levels: int | Sequence[int] | None = None,
) -> complex:
    r"""Contract two NS blocks with one physical structure product.

    The same three even/odd structure labels are used in both chiral halves.
    External descendants can make the corresponding internal edge parities
    differ between the two halves.  This routing is why multiplying two
    independently structure-weighted full blocks would be incorrect.
    """

    if holomorphic.sector != "NS" or antiholomorphic.sector != "NS":
        raise ValueError("the diagonal contraction requires two NS series")
    weights = tuple(tuple(complex(value) for value in pair) for pair in form_weights)
    if len(weights) != 3 or any(len(pair) != 2 for pair in weights):
        raise ValueError("one (even,odd) structure pair is required per trinion")
    holomorphic_parities = _word_parities(holomorphic.external_words)
    antiholomorphic_parities = _word_parities(antiholomorphic.external_words)
    if sum(holomorphic_parities) % 2 != sum(antiholomorphic_parities) % 2:
        return 0.0j

    total = 0.0j
    for first_form in (0, 1):
        for third_form in (0, 1):
            holomorphic_routing, central_form = _routing_for_structure_labels(
                holomorphic.external_words, first_form, third_form
            )
            antiholomorphic_routing, anti_central = _routing_for_structure_labels(
                antiholomorphic.external_words, first_form, third_form
            )
            if anti_central != central_form:
                continue
            structure = (
                weights[0][first_form]
                * weights[1][central_form]
                * weights[2][third_form]
            )
            holomorphic_value = holomorphic.component_value(
                q_1,
                q_2,
                holomorphic_routing,
                maximum_levels=maximum_twice_levels,
            )
            antiholomorphic_value = antiholomorphic.component_value(
                complex(q_1).conjugate(),
                complex(q_2).conjugate(),
                antiholomorphic_routing,
                maximum_levels=maximum_twice_levels,
            )
            total += (
                structure
                * fivepoint_component_phase(
                    holomorphic.external_words, holomorphic_routing
                )
                * fivepoint_component_phase(
                    antiholomorphic.external_words, antiholomorphic_routing
                )
                * holomorphic_value
                * antiholomorphic_value
            )
    return total


@dataclass(frozen=True)
class FixedNodePCOContribution:
    """One PCO contribution at fixed moduli and internal momenta."""

    component: SphereFivePointPCOComponent
    liouville_contraction: complex
    time_fermion_wick: complex
    value_before_common_factor: complex


@dataclass(frozen=True)
class FivePointFixedNodeEvaluation:
    """Complete reduced fixed-node five-point kernel."""

    value: complex
    common_factor: complex
    time_boson_factor: complex
    flavored_fermion_factor: complex
    picture_raising_factor: float
    pco_contributions: tuple[FixedNodePCOContribution, ...]
    q_parameters: tuple[complex, complex]
    internal_momenta: tuple[float, float]
    maximum_gram_condition: float

    @property
    def cancellation_ratio(self) -> float:
        absolute_sum = abs(self.common_factor) * sum(
            abs(component.value_before_common_factor)
            for component in self.pco_contributions
        )
        return absolute_sum / max(abs(self.value), 1.0e-300)


@dataclass(frozen=True)
class FivePointFixedMomentumWorkspace:
    """Reusable blocks and structure constants at fixed internal momenta."""

    states: tuple[
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
    ]
    internal_momenta: tuple[float, float]
    form_weights: FormWeights
    antiholomorphic: SphereFivePointSeries
    components: tuple[SphereFivePointPCOComponent, ...]
    holomorphic_blocks: Mapping[
        tuple[Word, Word, Word, Word, Word], SphereFivePointSeries
    ]
    maximum_twice_levels: LevelPair
    subtract_maximum_twice_levels: LevelPair | None
    zero_picture_legs: tuple[int, int, int]
    include_picture_raising_factor: bool
    maximum_gram_condition: float

    def evaluate(
        self,
        z_2: complex,
        z_3: complex,
    ) -> FivePointFixedNodeEvaluation:
        """Evaluate all cached PCO components at one moduli point."""

        q_1, q_2 = _validate_moduli(z_2, z_3)
        evaluations: list[FixedNodePCOContribution] = []
        for component in self.components:
            holomorphic = self.holomorphic_blocks[
                component.holomorphic_liouville_words
            ]
            liouville = diagonal_ns_fivepoint_contraction(
                holomorphic,
                self.antiholomorphic,
                q_1,
                q_2,
                self.form_weights,
                maximum_twice_levels=self.maximum_twice_levels,
            )
            if self.subtract_maximum_twice_levels is not None:
                liouville -= diagonal_ns_fivepoint_contraction(
                    holomorphic,
                    self.antiholomorphic,
                    q_1,
                    q_2,
                    self.form_weights,
                    maximum_twice_levels=self.subtract_maximum_twice_levels,
                )
            time_wick = component.time_fermion_wick(z_2, z_3)
            before_common = (
                component.momentum_coefficient * time_wick * liouville
            )
            evaluations.append(
                FixedNodePCOContribution(
                    component=component,
                    liouville_contraction=liouville,
                    time_fermion_wick=time_wick,
                    value_before_common_factor=before_common,
                )
            )

        time_factor = timelike_boson_factor(self.states, z_2, z_3)
        flavor_factor = flavored_sphere_fermion_wick_factor(
            self.states, z_2, z_3
        )
        picture_factor = 0.5 ** len(self.zero_picture_legs)
        if not self.include_picture_raising_factor:
            picture_factor = 1.0
        common = picture_factor * time_factor * flavor_factor
        value = common * sum(
            contribution.value_before_common_factor
            for contribution in evaluations
        )
        return FivePointFixedNodeEvaluation(
            value=value,
            common_factor=common,
            time_boson_factor=time_factor,
            flavored_fermion_factor=flavor_factor,
            picture_raising_factor=picture_factor,
            pco_contributions=tuple(evaluations),
            q_parameters=(q_1, q_2),
            internal_momenta=self.internal_momenta,
            maximum_gram_condition=self.maximum_gram_condition,
        )


def build_fivepoint_fixed_momentum_workspace(
    states: Sequence[SphereFivePointState],
    internal_momenta: Sequence[float],
    *,
    maximum_twice_levels: int | Sequence[int] = 2,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    include_picture_raising_factor: bool = True,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> FivePointFixedMomentumWorkspace:
    """Build the moduli-independent data for one spectral node."""

    normalized = validate_fivepoint_states(states)
    internal = tuple(float(value) for value in internal_momenta)
    if len(internal) != 2 or any(
        not math.isfinite(value) or value < 0 for value in internal
    ):
        raise ValueError("two finite nonnegative internal momenta are required")
    high_cutoffs = _normalize_cutoffs(maximum_twice_levels)
    low_cutoffs = None
    if subtract_maximum_twice_levels is not None:
        low_cutoffs = _normalize_cutoffs(subtract_maximum_twice_levels)
        if any(low > high for low, high in zip(low_cutoffs, high_cutoffs)):
            raise ValueError("subtraction cutoffs must not exceed construction cutoffs")
    external_momenta = tuple(
        complex(state.liouville_momentum) for state in normalized
    )
    external_weights = tuple(
        ns_liouville_weight(momentum, 1.0) for momentum in external_momenta
    )
    internal_weights = tuple(
        ns_liouville_weight(momentum, 1.0) for momentum in internal
    )
    form_weights = ns_fivepoint_structure_weights(
        internal_momenta=internal,
        external_momenta=external_momenta,
        precision=structure_precision,
    )
    anti_words = antiholomorphic_liouville_words(normalized)
    antiholomorphic = direct_ns_sphere_fivepoint_series(
        c=13.5,
        internal_weights=internal_weights,
        external_weights=external_weights,
        maximum_twice_levels=high_cutoffs,
        external_words=anti_words,
        digits=block_digits,
        condition_limit=condition_limit,
        vertex_backend="template",
    )
    legs = tuple(zero_picture_legs)
    components = fivepoint_pco_components(
        normalized, zero_picture_legs=legs
    )
    holomorphic_blocks: dict[
        tuple[Word, Word, Word, Word, Word], SphereFivePointSeries
    ] = {}
    maximum_condition = max(
        antiholomorphic.gram_condition_numbers.values(), default=1.0
    )
    for component in components:
        words = component.holomorphic_liouville_words
        if words in holomorphic_blocks:
            continue
        block = direct_ns_sphere_fivepoint_series(
            c=13.5,
            internal_weights=internal_weights,
            external_weights=external_weights,
            maximum_twice_levels=high_cutoffs,
            external_words=words,
            digits=block_digits,
            condition_limit=condition_limit,
            vertex_backend="template",
        )
        holomorphic_blocks[words] = block
        maximum_condition = max(
            maximum_condition,
            max(block.gram_condition_numbers.values(), default=1.0),
        )

    return FivePointFixedMomentumWorkspace(
        states=normalized,
        internal_momenta=internal,  # type: ignore[arg-type]
        form_weights=form_weights,
        antiholomorphic=antiholomorphic,
        components=components,
        holomorphic_blocks=holomorphic_blocks,
        maximum_twice_levels=high_cutoffs,
        subtract_maximum_twice_levels=low_cutoffs,
        zero_picture_legs=legs,  # type: ignore[arg-type]
        include_picture_raising_factor=include_picture_raising_factor,
        maximum_gram_condition=maximum_condition,
    )


def evaluate_fixed_fivepoint_node(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    internal_momenta: Sequence[float],
    *,
    maximum_twice_levels: int | Sequence[int] = 2,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    include_picture_raising_factor: bool = True,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> FivePointFixedNodeEvaluation:
    """Evaluate the reduced SO(23) kernel at one spectral and moduli node.

    If ``subtract_maximum_twice_levels`` is supplied, the returned value is
    the high-cutoff block minus that lower rectangular truncation.  Both are
    evaluated from the same coefficient tables.
    """

    workspace = build_fivepoint_fixed_momentum_workspace(
        states,
        internal_momenta,
        maximum_twice_levels=maximum_twice_levels,
        zero_picture_legs=zero_picture_legs,
        structure_precision=structure_precision,
        block_digits=block_digits,
        condition_limit=condition_limit,
        include_picture_raising_factor=include_picture_raising_factor,
        subtract_maximum_twice_levels=subtract_maximum_twice_levels,
    )
    return workspace.evaluate(z_2, z_3)


@dataclass(frozen=True)
class TwoMomentumQuadratureDiagnostics:
    """Two-dimensional continuum quadrature and one refinement check."""

    value: complex
    refined_value: complex | None
    order: int
    refined_order: int | None
    method: QuadratureMethod
    endpoint_powers: tuple[float, float]
    evaluation_count: int
    refined_evaluation_count: int

    @property
    def absolute_error(self) -> float:
        if self.refined_value is None:
            return math.nan
        return abs(self.refined_value - self.value)

    @property
    def relative_error(self) -> float:
        if self.refined_value is None:
            return math.nan
        return self.absolute_error / max(
            abs(self.value), abs(self.refined_value), 1.0e-300
        )


def _validate_order(order: int, name: str) -> int:
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"{name} must be a positive integer")
    return order


def integrate_two_momentum_density(
    density: Callable[[float, float], complex],
    q_1: complex,
    q_2: complex,
    *,
    order: int,
    refined_order: int | None = None,
    method: QuadratureMethod = "gauss_laguerre",
    p_max: float | None = None,
    refined_p_max: float | None = None,
    endpoint_powers: Sequence[float] = (0.0, 0.0),
) -> TwoMomentumQuadratureDiagnostics:
    r"""Integrate ``density(P1,P2) dP1/pi dP2/pi``.

    For generalized Laguerre quadrature, ``density`` must contain the block's
    natural Gaussian ``|q1|**P1**2 |q2|**P2**2``.  The transformation removes
    that Gaussian.  If the residual density has a certified factor
    ``P_e**k_e``, pass ``endpoint_powers=(k1,k2)``; the matched generalized
    Laguerre exponents are then ``alpha_e=(k_e-1)/2``.
    """

    order = _validate_order(order, "order")
    if refined_order is not None:
        refined_order = _validate_order(refined_order, "refined_order")
        if refined_order <= order:
            raise ValueError("refined_order must exceed order")
    q_values = (complex(q_1), complex(q_2))
    if any(not 0 < abs(value) < 1 for value in q_values):
        raise ValueError("momentum quadrature requires 0<|q1|,|q2|<1")
    if method not in ("gauss_laguerre", "gauss_legendre"):
        raise ValueError("unknown two-momentum quadrature method")
    powers = tuple(float(value) for value in endpoint_powers)
    if len(powers) != 2 or any(
        not math.isfinite(value) or value < 0 for value in powers
    ):
        raise ValueError("endpoint_powers must contain two finite nonnegative values")

    def evaluate(current_order: int, current_p_max: float | None) -> complex:
        if method == "gauss_laguerre":
            a1, a2 = (-math.log(abs(value)) for value in q_values)
            nodes1, weights1 = roots_genlaguerre(
                current_order, 0.5 * (powers[0] - 1.0)
            )
            nodes2, weights2 = roots_genlaguerre(
                current_order, 0.5 * (powers[1] - 1.0)
            )
            p1 = np.sqrt(nodes1 / a1)
            p2 = np.sqrt(nodes2 / a2)
            factor = 1.0 / (
                4.0
                * math.pi**2
                * a1 ** (0.5 * (powers[0] + 1.0))
                * a2 ** (0.5 * (powers[1] + 1.0))
            )
            total = 0.0j
            for index1, momentum1 in enumerate(p1):
                for index2, momentum2 in enumerate(p2):
                    endpoint_factor = (
                        momentum1 ** powers[0] * momentum2 ** powers[1]
                    )
                    total += (
                        weights1[index1]
                        * weights2[index2]
                        * complex(density(float(momentum1), float(momentum2)))
                        * math.exp(nodes1[index1] + nodes2[index2])
                        / endpoint_factor
                    )
            return factor * total

        if current_p_max is None or not math.isfinite(current_p_max) or current_p_max <= 0:
            raise ValueError("positive p_max is required for Gauss-Legendre")
        raw_nodes, raw_weights = leggauss(current_order)
        momenta = 0.5 * current_p_max * (raw_nodes + 1.0)
        weights = 0.5 * current_p_max * raw_weights / math.pi
        return sum(
            weights[index1]
            * weights[index2]
            * complex(density(float(momentum1), float(momentum2)))
            for index1, momentum1 in enumerate(momenta)
            for index2, momentum2 in enumerate(momenta)
        )

    value = evaluate(order, p_max)
    refined_value = None
    if refined_order is not None:
        next_p_max = p_max if refined_p_max is None else refined_p_max
        refined_value = evaluate(refined_order, next_p_max)
    return TwoMomentumQuadratureDiagnostics(
        value=value,
        refined_value=refined_value,
        order=order,
        refined_order=refined_order,
        method=method,
        endpoint_powers=powers,  # type: ignore[arg-type]
        evaluation_count=order * order,
        refined_evaluation_count=0 if refined_order is None else refined_order**2,
    )


@dataclass(frozen=True)
class FivePointSpectralIntegral:
    """Physical spectral integral plus block-conditioning diagnostics."""

    quadrature: TwoMomentumQuadratureDiagnostics
    maximum_gram_condition: float
    maximum_node_cancellation_ratio: float


@dataclass(frozen=True)
class LinearChamberQMCDiagnostics:
    r"""Independent-scramble diagnostic for one truncated plumbing chamber.

    This chamber is ``|q1|<R1``, ``|q2|<R2`` with
    ``z2=q1*q2, z3=q2``.  The complex Jacobian is ``|q2|**2``.  It is one
    certified local chart, not yet a crossing-complete sphere-five-point
    atlas.
    """

    value: complex
    replicate_values: tuple[complex, ...]
    sample_power: int
    samples_per_replicate: int
    replicates: int
    q_maxima: tuple[float, float]
    real_standard_error: float
    imaginary_standard_error: float

    @property
    def combined_standard_error(self) -> float:
        return math.hypot(
            self.real_standard_error, self.imaginary_standard_error
        )

    @property
    def relative_standard_error(self) -> float:
        return self.combined_standard_error / max(abs(self.value), 1.0e-300)


def integrate_linear_chamber_qmc(
    integrand: Callable[[complex, complex], complex],
    *,
    q_maxima: Sequence[float] = (0.5, 0.5),
    sample_power: int = 4,
    replicates: int = 4,
    seed: int = 1729,
) -> LinearChamberQMCDiagnostics:
    r"""Integrate one ``(q1,q2)`` plumbing polydisk with its exact Jacobian.

    Each complex disk is sampled uniformly in area via
    ``q=R*sqrt(u)*exp(2*pi*i*v)``.  The returned integral includes
    ``d2z2 d2z3 = |q2|**2 d2q1 d2q2``.
    """

    maxima = tuple(float(value) for value in q_maxima)
    if len(maxima) != 2 or any(
        not math.isfinite(value) or not 0 < value < 1 for value in maxima
    ):
        raise ValueError("q_maxima must contain two finite values in (0,1)")
    if (
        isinstance(sample_power, bool)
        or not isinstance(sample_power, int)
        or sample_power < 0
    ):
        raise ValueError("sample_power must be a nonnegative integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    sample_count = 2**sample_power
    disk_volume = math.pi**2 * maxima[0] ** 2 * maxima[1] ** 2
    values: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=4, scramble=True, seed=seed + replicate)
        points = sampler.random_base2(sample_power)
        q_1 = maxima[0] * np.sqrt(points[:, 0]) * np.exp(
            2j * math.pi * points[:, 1]
        )
        q_2 = maxima[1] * np.sqrt(points[:, 2]) * np.exp(
            2j * math.pi * points[:, 3]
        )
        total = sum(
            abs(second) ** 2 * complex(integrand(first, second))
            for first, second in zip(q_1, q_2)
        )
        values.append(disk_volume * total / sample_count)

    array = np.asarray(values, dtype=np.complex128)
    value = complex(np.mean(array))
    real_error = float(np.std(array.real, ddof=1) / math.sqrt(replicates))
    imaginary_error = float(
        np.std(array.imag, ddof=1) / math.sqrt(replicates)
    )
    return LinearChamberQMCDiagnostics(
        value=value,
        replicate_values=tuple(values),
        sample_power=sample_power,
        samples_per_replicate=sample_count,
        replicates=replicates,
        q_maxima=maxima,  # type: ignore[arg-type]
        real_standard_error=real_error,
        imaginary_standard_error=imaginary_error,
    )


def integrate_fivepoint_linear_chamber(
    states: Sequence[SphereFivePointState],
    *,
    q_maxima: Sequence[float] = (0.35, 0.35),
    sample_power: int = 2,
    replicates: int = 2,
    seed: int = 1729,
    maximum_twice_levels: int | Sequence[int] = 2,
    spectral_order: int = 15,
    spectral_method: QuadratureMethod = "gauss_legendre",
    spectral_p_max: float | None = 3.0,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> LinearChamberQMCDiagnostics:
    """Integrate the physical spectral kernel over one truncated comb chart.

    With the recommended fixed Gauss--Legendre momentum grid, the loops are
    ordered by spectral node.  Each fixed-momentum workspace is therefore
    built once and evaluated at every Sobol moduli point, instead of rebuilding
    the same conformal blocks inside the moduli loop.
    """

    normalized = validate_fivepoint_states(states)

    if spectral_method == "gauss_legendre":
        spectral_order = _validate_order(spectral_order, "spectral_order")
        if (
            spectral_p_max is None
            or not math.isfinite(spectral_p_max)
            or spectral_p_max <= 0
        ):
            raise ValueError("positive spectral_p_max is required")
        maxima = tuple(float(value) for value in q_maxima)
        if len(maxima) != 2 or any(
            not math.isfinite(value) or not 0 < value < 1
            for value in maxima
        ):
            raise ValueError("q_maxima must contain two finite values in (0,1)")
        if (
            isinstance(sample_power, bool)
            or not isinstance(sample_power, int)
            or sample_power < 0
        ):
            raise ValueError("sample_power must be a nonnegative integer")
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates < 2
        ):
            raise ValueError("replicates must be an integer of at least two")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        sample_count = 2**sample_power
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for replicate in range(replicates):
            sampler = qmc.Sobol(d=4, scramble=True, seed=seed + replicate)
            points = sampler.random_base2(sample_power)
            q_1 = maxima[0] * np.sqrt(points[:, 0]) * np.exp(
                2j * math.pi * points[:, 1]
            )
            q_2 = maxima[1] * np.sqrt(points[:, 2]) * np.exp(
                2j * math.pi * points[:, 3]
            )
            samples.append((q_1, q_2))

        raw_nodes, raw_weights = leggauss(spectral_order)
        momenta = 0.5 * spectral_p_max * (raw_nodes + 1.0)
        momentum_weights = (
            0.5 * spectral_p_max * raw_weights / math.pi
        )
        replicate_sums = np.zeros(replicates, dtype=np.complex128)
        for index1, momentum1 in enumerate(momenta):
            for index2, momentum2 in enumerate(momenta):
                workspace = build_fivepoint_fixed_momentum_workspace(
                    normalized,
                    (float(momentum1), float(momentum2)),
                    maximum_twice_levels=maximum_twice_levels,
                    zero_picture_legs=zero_picture_legs,
                    structure_precision=structure_precision,
                    block_digits=block_digits,
                    condition_limit=condition_limit,
                    subtract_maximum_twice_levels=(
                        subtract_maximum_twice_levels
                    ),
                )
                spectral_weight = (
                    momentum_weights[index1] * momentum_weights[index2]
                )
                for replicate, (q_1, q_2) in enumerate(samples):
                    moduli_sum = sum(
                        abs(second) ** 2
                        * workspace.evaluate(first * second, second).value
                        for first, second in zip(q_1, q_2)
                    )
                    replicate_sums[replicate] += (
                        spectral_weight * moduli_sum
                    )

        disk_volume = math.pi**2 * maxima[0] ** 2 * maxima[1] ** 2
        replicate_values = tuple(
            complex(disk_volume * value / sample_count)
            for value in replicate_sums
        )
        array = np.asarray(replicate_values, dtype=np.complex128)
        value = complex(np.mean(array))
        real_error = float(
            np.std(array.real, ddof=1) / math.sqrt(replicates)
        )
        imaginary_error = float(
            np.std(array.imag, ddof=1) / math.sqrt(replicates)
        )
        return LinearChamberQMCDiagnostics(
            value=value,
            replicate_values=replicate_values,
            sample_power=sample_power,
            samples_per_replicate=sample_count,
            replicates=replicates,
            q_maxima=maxima,  # type: ignore[arg-type]
            real_standard_error=real_error,
            imaginary_standard_error=imaginary_error,
        )

    def integrand(q_1: complex, q_2: complex) -> complex:
        z_3 = q_2
        z_2 = q_1 * q_2
        return integrate_fivepoint_spectral_kernel(
            normalized,
            z_2,
            z_3,
            maximum_twice_levels=maximum_twice_levels,
            order=spectral_order,
            method=spectral_method,
            p_max=spectral_p_max,
            zero_picture_legs=zero_picture_legs,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
            subtract_maximum_twice_levels=subtract_maximum_twice_levels,
        ).quadrature.value

    return integrate_linear_chamber_qmc(
        integrand,
        q_maxima=q_maxima,
        sample_power=sample_power,
        replicates=replicates,
        seed=seed,
    )


def integrate_fivepoint_spectral_kernel(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    *,
    maximum_twice_levels: int | Sequence[int] = 2,
    order: int = 15,
    refined_order: int | None = None,
    method: QuadratureMethod = "gauss_legendre",
    p_max: float | None = 3.0,
    refined_p_max: float | None = None,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> FivePointSpectralIntegral:
    """Integrate the certified fixed-node kernel over both internal momenta."""

    normalized = validate_fivepoint_states(states)
    q_1, q_2 = _validate_moduli(z_2, z_3)
    maximum_condition = 1.0
    maximum_cancellation = 1.0

    def density(momentum1: float, momentum2: float) -> complex:
        nonlocal maximum_condition, maximum_cancellation
        evaluation = evaluate_fixed_fivepoint_node(
            normalized,
            z_2,
            z_3,
            (momentum1, momentum2),
            maximum_twice_levels=maximum_twice_levels,
            zero_picture_legs=zero_picture_legs,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
            subtract_maximum_twice_levels=subtract_maximum_twice_levels,
        )
        maximum_condition = max(
            maximum_condition, evaluation.maximum_gram_condition
        )
        maximum_cancellation = max(
            maximum_cancellation, evaluation.cancellation_ratio
        )
        return evaluation.value

    quadrature = integrate_two_momentum_density(
        density,
        q_1,
        q_2,
        order=order,
        refined_order=refined_order,
        method=method,
        p_max=p_max,
        refined_p_max=refined_p_max,
        endpoint_powers=(2.0, 2.0),
    )
    return FivePointSpectralIntegral(
        quadrature=quadrature,
        maximum_gram_condition=maximum_condition,
        maximum_node_cancellation_ratio=maximum_cancellation,
    )


__all__ = [
    "FivePointFixedMomentumWorkspace",
    "FivePointFixedNodeEvaluation",
    "FivePointSpectralIntegral",
    "FixedNodePCOContribution",
    "LinearChamberQMCDiagnostics",
    "SphereFivePointPCOComponent",
    "SphereFivePointState",
    "TwoMomentumQuadratureDiagnostics",
    "antiholomorphic_liouville_words",
    "build_fivepoint_fixed_momentum_workspace",
    "diagonal_ns_fivepoint_contraction",
    "evaluate_fixed_fivepoint_node",
    "fivepoint_component_phase",
    "fivepoint_pco_components",
    "flavored_sphere_fermion_wick_factor",
    "integrate_fivepoint_spectral_kernel",
    "integrate_fivepoint_linear_chamber",
    "integrate_linear_chamber_qmc",
    "integrate_two_momentum_density",
    "timelike_boson_factor",
    "validate_fivepoint_states",
]
