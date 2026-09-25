#!/usr/bin/env python3
r"""Full-integrand Cauchy localization at the first five-point resonance.

This is a focused diagnostic for the ``s=1`` super-Liouville global pole.
It repairs one precise defect of the earlier center-evaluation code in
``spin23_sphere_fivepoint_resonance``.  At ``b=1`` several generic-``b``
crossing poles coalesce.  The resulting internal-momentum poles can be of
higher order, so

``Res[C(P1,P2)] * F(P1_center,P2_center)``

is not the residue of ``C*F``: derivatives of the complete descendant block
``F`` contribute.  The evaluator below instead performs the two Cauchy
projections on the *full fixed-moduli integrand* at every contour node.

For each nonzero global regulator ``eta`` it

1. deforms the five on-shell momenta coherently;
2. encloses each of the three one-screening pole clusters;
3. evaluates the structure product, both descendant blocks, every PCO
   component, and its routing phase at each two-torus node;
4. sums the three clusters before multiplying by ``eta``; and
5. extrapolates the assembled result to ``eta=0``.

The returned number is a fixed-moduli, reduced global-pole coefficient in
the exact super-Liouville descendant representation.  In that limited sense
it contains the local collision/contact information encoded by the exact
superconformal correlator.  It is *not* by itself a proof of PCO/vertical
completeness for the string amplitude: the PCOs remain tied to external
vertices, the two complex moduli still have to be integrated, and picture
assignment invariance plus all boundary/vertical terms must be checked on
that integrated result.  Common sphere, wall, cosmological-constant, and
in/out phase conventions are also omitted, as in the existing five-point
resonance code.

The coalesced contour is equivalent to a generic-``b`` split-cluster limit
provided no pole crosses its boundary during the homotopy from generic
``b`` to one and the complete cluster (not an individual split pole) is
enclosed.  This is a standard consequence of Cauchy homotopy invariance, but
those two hypotheses must be checked numerically by contour-radius and
contour-order scans; this module reports the data needed for that check.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from spin23_sphere_fivepoint import (
    SphereFivePointSeries,
    direct_ns_sphere_fivepoint_series,
    ns_liouville_weight,
)
from spin23_sphere_fivepoint_amplitude import (
    SphereFivePointPCOComponent,
    SphereFivePointState,
    antiholomorphic_liouville_words,
    fivepoint_component_phase,
    fivepoint_pco_components,
    flavored_sphere_fermion_wick_factor,
    timelike_boson_factor,
    validate_fivepoint_states,
)
from spin23_sphere_fivepoint_resonance import (
    FIRST_FIVEPOINT_SCREENING_NUMBER,
    ResonanceResidueSettings,
    _deformed_external_momenta,
    _extrapolated_intercept,
    _normalize_cutoffs,
    _routing_for_forms,
    _states_with_deformed_momenta,
    _structure_product,
    fivepoint_resonance_momentum,
    localized_internal_momenta,
    screening_allocations,
)


@dataclass(frozen=True)
class FullClusterAllocationResult:
    """One full-integrand two-torus residue at fixed nonzero ``eta``."""

    allocation: tuple[int, int, int]
    center: tuple[complex, complex]
    value: complex
    maximum_gram_condition: float


@dataclass(frozen=True)
class FullClusterRadiusResult:
    """The three allocation clusters assembled at one regulator radius."""

    radius_parameter: float
    eta: complex
    allocation_results: tuple[FullClusterAllocationResult, ...]
    unscaled_value: complex
    eta_scaled_value: complex
    maximum_gram_condition: float


@dataclass(frozen=True)
class FullClusterPointwiseResult:
    """Extrapolated fixed-moduli coefficient and regulator diagnostics."""

    value: complex
    fit_spread: float
    z_2: complex
    z_3: complex
    maximum_twice_levels: tuple[int, int]
    radius_results: tuple[FullClusterRadiusResult, ...]
    maximum_gram_condition: float


@dataclass(frozen=True)
class FullSphereBootstrapComparison:
    """Comparison record for an already moduli-integrated resonance value."""

    integrated_value: complex
    bootstrap_value: complex
    normalization_scale: complex
    scaled_bootstrap_value: complex
    difference: complex
    relative_difference: float


@dataclass(frozen=True)
class FullClusterCauchyNode:
    """Moduli-independent data at one internal-momentum contour node."""

    internal_momenta: tuple[complex, complex]
    differential_weight: complex
    structure_product: complex
    antiholomorphic: SphereFivePointSeries
    holomorphic_blocks: Mapping[tuple, SphereFivePointSeries]
    maximum_gram_condition: float


@dataclass(frozen=True)
class FullClusterAllocationWorkspace:
    """All two-torus nodes enclosing one screening-allocation cluster."""

    allocation: tuple[int, int, int]
    center: tuple[complex, complex]
    nodes: tuple[FullClusterCauchyNode, ...]
    maximum_gram_condition: float


@dataclass(frozen=True)
class FullClusterRegulatorWorkspace:
    """Reusable coefficient tables at one nonzero global regulator."""

    radius_parameter: float
    eta: complex
    states: tuple[
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
    ]
    components: tuple[SphereFivePointPCOComponent, ...]
    antiholomorphic_words: tuple
    allocations: tuple[FullClusterAllocationWorkspace, ...]
    zero_picture_legs: tuple[int, int, int]
    maximum_twice_levels: tuple[int, int]
    maximum_gram_condition: float


@dataclass(frozen=True)
class S1FullClusterWorkspace:
    r"""Full-integrand Cauchy data reusable throughout one moduli atlas.

    This is the scalable interface: structure constants and finite-level
    descendant coefficient tables are built once per regulator, allocation,
    and contour node.  Evaluating a new ``(z2,z3)`` then only contracts those
    cached tables and forms the global-regulator extrapolation.
    """

    regulated: tuple[FullClusterRegulatorWorkspace, ...]
    extrapolation_degree: int
    maximum_twice_levels: tuple[int, int]
    maximum_gram_condition: float

    def evaluate(
        self,
        z_2: complex,
        z_3: complex,
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> FullClusterPointwiseResult:
        z_2, z_3 = complex(z_2), complex(z_3)
        q_1, q_2 = z_2 / z_3, z_3
        if not (0 < abs(q_1) < 1 and 0 < abs(q_2) < 1):
            raise ValueError("the selected comb requires 0<|z2/z3|,|z3|<1")
        cutoffs = (
            self.maximum_twice_levels
            if maximum_twice_levels is None
            else _normalize_cutoffs(maximum_twice_levels)
        )
        if any(
            selected > available
            for selected, available in zip(cutoffs, self.maximum_twice_levels)
        ):
            raise ValueError(
                "evaluation cutoffs cannot exceed the constructed cutoffs"
            )

        radius_results: list[FullClusterRadiusResult] = []
        for regulator in self.regulated:
            allocation_results: list[FullClusterAllocationResult] = []
            for allocation_workspace in regulator.allocations:
                value = 0.0j
                for node in allocation_workspace.nodes:
                    block_factor = _cached_node_block_factor(
                        node,
                        regulator.antiholomorphic_words,
                        regulator.components,
                        allocation_workspace.allocation,
                        q_1,
                        q_2,
                        cutoffs,
                    )
                    value += (
                        node.differential_weight
                        * node.structure_product
                        * block_factor
                    )
                allocation_results.append(
                    FullClusterAllocationResult(
                        allocation=allocation_workspace.allocation,
                        center=allocation_workspace.center,
                        value=value,
                        maximum_gram_condition=(
                            allocation_workspace.maximum_gram_condition
                        ),
                    )
                )

            common = (
                (-2j) ** 2
                * 0.5 ** len(regulator.zero_picture_legs)
                * timelike_boson_factor(regulator.states, z_2, z_3)
                * flavored_sphere_fermion_wick_factor(
                    regulator.states, z_2, z_3
                )
            )
            unscaled = common * sum(
                (entry.value for entry in allocation_results), 0.0j
            )
            radius_results.append(
                FullClusterRadiusResult(
                    radius_parameter=regulator.radius_parameter,
                    eta=regulator.eta,
                    allocation_results=tuple(allocation_results),
                    unscaled_value=unscaled,
                    eta_scaled_value=regulator.eta * unscaled,
                    maximum_gram_condition=regulator.maximum_gram_condition,
                )
            )

        radii = np.asarray(
            [result.radius_parameter for result in radius_results],
            dtype=np.float64,
        )
        values = np.asarray(
            [result.eta_scaled_value for result in radius_results],
            dtype=np.complex128,
        )
        value, fit_spread = _extrapolated_intercept(
            radii, values, self.extrapolation_degree
        )
        return FullClusterPointwiseResult(
            value=value,
            fit_spread=fit_spread,
            z_2=z_2,
            z_3=z_3,
            maximum_twice_levels=cutoffs,
            radius_results=tuple(radius_results),
            maximum_gram_condition=self.maximum_gram_condition,
        )


def compare_integrated_s1_with_bootstrap(
    integrated_value: complex,
    bootstrap_value: complex,
    *,
    normalization_scale: complex = 1.0,
) -> FullSphereBootstrapComparison:
    r"""Compare a completed atlas integral with a soft-bootstrap prediction.

    This deliberately accepts values rather than a pointwise workspace.  The
    caller is responsible for supplying a crossing-complete, converged
    moduli integral; a fixed-moduli value from :meth:`S1FullClusterWorkspace.evaluate`
    must never be passed off as this comparison.  ``normalization_scale`` is
    exposed because the present residue omits the common sphere/wall
    convention that the factorization bootstrap can later calibrate.
    """

    integrated = complex(integrated_value)
    bootstrap = complex(bootstrap_value)
    scale = complex(normalization_scale)
    scaled = scale * bootstrap
    difference = integrated - scaled
    return FullSphereBootstrapComparison(
        integrated_value=integrated,
        bootstrap_value=bootstrap,
        normalization_scale=scale,
        scaled_bootstrap_value=scaled,
        difference=difference,
        relative_difference=abs(difference) / max(abs(scaled), 1.0e-300),
    )


def nested_torus_cauchy_residue(
    evaluator: Callable[[complex, complex], complex],
    center: Sequence[complex],
    *,
    outer_radius: float,
    inner_radius: float,
    order: int,
    outer_offset: float = 0.173,
    inner_offset: float = 0.319,
) -> complex:
    r"""Return ``Res_P1 Res_P2 evaluator`` on a product of two circles.

    The discrete trapezoid formula includes the two differentials, hence the
    factor ``delta_1*delta_2`` at every node.  It is exact for the Laurent
    modes resolved by the selected order and converges exponentially for a
    meromorphic integrand whose other poles stay outside the annulus.
    """

    selected_center = tuple(complex(value) for value in center)
    if len(selected_center) != 2:
        raise ValueError("two internal-momentum centers are required")
    if (
        not math.isfinite(outer_radius)
        or not math.isfinite(inner_radius)
        or not 0 < inner_radius < outer_radius
    ):
        raise ValueError("Cauchy radii must obey 0<inner<outer")
    if isinstance(order, bool) or not isinstance(order, int) or order < 4:
        raise ValueError("Cauchy order must be an integer of at least four")

    total = 0.0j
    for first_index in range(order):
        angle_1 = 2.0 * math.pi * (first_index + outer_offset) / order
        delta_1 = outer_radius * cmath.exp(1j * angle_1)
        for second_index in range(order):
            angle_2 = 2.0 * math.pi * (second_index + inner_offset) / order
            delta_2 = inner_radius * cmath.exp(1j * angle_2)
            total += (
                complex(
                    evaluator(
                        selected_center[0] + delta_1,
                        selected_center[1] + delta_2,
                    )
                )
                * delta_1
                * delta_2
            )
    return total / order**2


def _block_series(
    internal_momenta: tuple[complex, complex],
    external_weights: tuple[complex, ...],
    external_words: tuple,
    cutoffs: tuple[int, int],
    *,
    block_digits: int,
    condition_limit: float,
) -> SphereFivePointSeries:
    internal_weights = tuple(
        ns_liouville_weight(momentum, 1.0) for momentum in internal_momenta
    )
    return direct_ns_sphere_fivepoint_series(
        c=13.5,
        internal_weights=internal_weights,
        external_weights=external_weights,
        maximum_twice_levels=cutoffs,
        external_words=external_words,
        digits=block_digits,
        condition_limit=condition_limit,
        vertex_backend="template",
    )


def _maximum_condition(series: SphereFivePointSeries) -> float:
    return max(series.gram_condition_numbers.values(), default=1.0)


def _allocation_block_factor(
    internal_momenta: tuple[complex, complex],
    external_weights: tuple[complex, ...],
    anti_words: tuple,
    components: tuple[SphereFivePointPCOComponent, ...],
    allocation: tuple[int, int, int],
    q_1: complex,
    q_2: complex,
    cutoffs: tuple[int, int],
    *,
    block_digits: int,
    condition_limit: float,
) -> tuple[complex, float]:
    """Evaluate both chiral descendant blocks before taking the residue."""

    first_form, central_form, third_form = allocation
    anti_routing, anti_central = _routing_for_forms(
        anti_words, first_form, third_form
    )
    if anti_central != central_form % 2:
        return 0.0j, 1.0

    antiholomorphic = _block_series(
        internal_momenta,
        external_weights,
        anti_words,
        cutoffs,
        block_digits=block_digits,
        condition_limit=condition_limit,
    )
    maximum_condition = _maximum_condition(antiholomorphic)
    anti_value = antiholomorphic.component_value(
        q_1.conjugate(),
        q_2.conjugate(),
        anti_routing,
        maximum_levels=cutoffs,
    )
    anti_value *= fivepoint_component_phase(anti_words, anti_routing)

    cache: dict[tuple, SphereFivePointSeries] = {}
    holomorphic_sum = 0.0j
    for component in components:
        routing, selected_central = _routing_for_forms(
            component.holomorphic_liouville_words,
            first_form,
            third_form,
        )
        if selected_central != central_form % 2:
            continue
        words = component.holomorphic_liouville_words
        if words not in cache:
            cache[words] = _block_series(
                internal_momenta,
                external_weights,
                words,
                cutoffs,
                block_digits=block_digits,
                condition_limit=condition_limit,
            )
            maximum_condition = max(
                maximum_condition, _maximum_condition(cache[words])
            )
        block_value = cache[words].component_value(
            q_1,
            q_2,
            routing,
            maximum_levels=cutoffs,
        )
        holomorphic_sum += (
            component.momentum_coefficient
            * component.time_fermion_wick(q_1 * q_2, q_2)
            * fivepoint_component_phase(words, routing)
            * block_value
        )
    return anti_value * holomorphic_sum, maximum_condition


def _cached_node_block_factor(
    node: FullClusterCauchyNode,
    anti_words: tuple,
    components: tuple[SphereFivePointPCOComponent, ...],
    allocation: tuple[int, int, int],
    q_1: complex,
    q_2: complex,
    cutoffs: tuple[int, int],
) -> complex:
    """Contract one cached coefficient table at a new moduli point."""

    first_form, central_form, third_form = allocation
    anti_routing, anti_central = _routing_for_forms(
        anti_words, first_form, third_form
    )
    if anti_central != central_form % 2:
        return 0.0j
    anti_value = node.antiholomorphic.component_value(
        q_1.conjugate(),
        q_2.conjugate(),
        anti_routing,
        maximum_levels=cutoffs,
    )
    anti_value *= fivepoint_component_phase(anti_words, anti_routing)

    holomorphic_sum = 0.0j
    for component in components:
        routing, selected_central = _routing_for_forms(
            component.holomorphic_liouville_words,
            first_form,
            third_form,
        )
        if selected_central != central_form % 2:
            continue
        words = component.holomorphic_liouville_words
        block = node.holomorphic_blocks.get(words)
        if block is None:
            raise AssertionError("a routed holomorphic block was not cached")
        holomorphic_sum += (
            component.momentum_coefficient
            * component.time_fermion_wick(q_1 * q_2, q_2)
            * fivepoint_component_phase(words, routing)
            * block.component_value(
                q_1,
                q_2,
                routing,
                maximum_levels=cutoffs,
            )
        )
    return anti_value * holomorphic_sum


def _allocation_has_support(
    anti_words: tuple,
    components: tuple[SphereFivePointPCOComponent, ...],
    allocation: tuple[int, int, int],
) -> bool:
    first_form, central_form, third_form = allocation
    _anti_routing, anti_central = _routing_for_forms(
        anti_words, first_form, third_form
    )
    if anti_central != central_form % 2:
        return False
    return any(
        _routing_for_forms(
            component.holomorphic_liouville_words,
            first_form,
            third_form,
        )[1]
        == central_form % 2
        for component in components
    )


def build_s1_full_cluster_workspace(
    states: Sequence[SphereFivePointState],
    *,
    maximum_twice_levels: int | Sequence[int] = 0,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    residue_settings: ResonanceResidueSettings = ResonanceResidueSettings(),
    block_digits: int = 40,
    condition_limit: float = 1.0e13,
) -> S1FullClusterWorkspace:
    """Build full-residue coefficient tables once for repeated moduli use."""

    target_states = validate_fivepoint_states(states)
    target_external = tuple(
        complex(state.liouville_momentum) for state in target_states
    )
    resonance = fivepoint_resonance_momentum(FIRST_FIVEPOINT_SCREENING_NUMBER)
    if (
        abs(target_external[4] - resonance) > 1.0e-10
        or abs(sum(target_external[:4]) - resonance) > 1.0e-10
    ):
        raise ValueError("the states must lie on the first five-point resonance")
    cutoffs = _normalize_cutoffs(maximum_twice_levels)
    picture_legs = tuple(int(value) for value in zero_picture_legs)
    if len(picture_legs) != 3 or len(set(picture_legs)) != 3:
        raise ValueError("exactly three distinct zero-picture legs are required")

    regulated: list[FullClusterRegulatorWorkspace] = []
    overall_maximum_condition = 1.0
    order = residue_settings.cauchy_order
    for radius_parameter in residue_settings.regulator_radii:
        eta = float(radius_parameter) * complex(
            residue_settings.regulator_direction
        )
        external = _deformed_external_momenta(target_external, eta)
        deformed_states = _states_with_deformed_momenta(target_states, external)
        components = fivepoint_pco_components(
            deformed_states, zero_picture_legs=picture_legs
        )
        anti_words = antiholomorphic_liouville_words(deformed_states)
        external_weights = tuple(
            ns_liouville_weight(momentum, 1.0) for momentum in external
        )
        base_radius = abs(eta)
        outer_radius = residue_settings.outer_radius_ratio * base_radius
        inner_radius = residue_settings.inner_radius_ratio * base_radius
        allocation_workspaces: list[FullClusterAllocationWorkspace] = []

        for allocation in screening_allocations(
            FIRST_FIVEPOINT_SCREENING_NUMBER
        ):
            center = localized_internal_momenta(external, allocation)
            nodes: list[FullClusterCauchyNode] = []
            allocation_maximum_condition = 1.0
            if _allocation_has_support(anti_words, components, allocation):
                first_form, central_form, third_form = allocation
                for first_index in range(order):
                    angle_1 = 2.0 * math.pi * (first_index + 0.173) / order
                    delta_1 = outer_radius * cmath.exp(1j * angle_1)
                    for second_index in range(order):
                        angle_2 = (
                            2.0 * math.pi * (second_index + 0.319) / order
                        )
                        delta_2 = inner_radius * cmath.exp(1j * angle_2)
                        internal = (
                            center[0] + delta_1,
                            center[1] + delta_2,
                        )
                        antiholomorphic = _block_series(
                            internal,
                            external_weights,
                            anti_words,
                            cutoffs,
                            block_digits=block_digits,
                            condition_limit=condition_limit,
                        )
                        node_maximum_condition = _maximum_condition(
                            antiholomorphic
                        )
                        holomorphic_blocks: dict[
                            tuple, SphereFivePointSeries
                        ] = {}
                        for component in components:
                            _routing, selected_central = _routing_for_forms(
                                component.holomorphic_liouville_words,
                                first_form,
                                third_form,
                            )
                            if selected_central != central_form % 2:
                                continue
                            words = component.holomorphic_liouville_words
                            if words in holomorphic_blocks:
                                continue
                            holomorphic_blocks[words] = _block_series(
                                internal,
                                external_weights,
                                words,
                                cutoffs,
                                block_digits=block_digits,
                                condition_limit=condition_limit,
                            )
                            node_maximum_condition = max(
                                node_maximum_condition,
                                _maximum_condition(holomorphic_blocks[words]),
                            )
                        nodes.append(
                            FullClusterCauchyNode(
                                internal_momenta=internal,
                                differential_weight=(
                                    delta_1 * delta_2 / order**2
                                ),
                                structure_product=_structure_product(
                                    internal[0],
                                    internal[1],
                                    external,
                                    allocation,
                                    residue_settings.structure_precision,
                                ),
                                antiholomorphic=antiholomorphic,
                                holomorphic_blocks=holomorphic_blocks,
                                maximum_gram_condition=node_maximum_condition,
                            )
                        )
                        allocation_maximum_condition = max(
                            allocation_maximum_condition,
                            node_maximum_condition,
                        )
            allocation_workspaces.append(
                FullClusterAllocationWorkspace(
                    allocation=allocation,
                    center=center,
                    nodes=tuple(nodes),
                    maximum_gram_condition=allocation_maximum_condition,
                )
            )
            overall_maximum_condition = max(
                overall_maximum_condition, allocation_maximum_condition
            )

        regulated.append(
            FullClusterRegulatorWorkspace(
                radius_parameter=float(radius_parameter),
                eta=eta,
                states=deformed_states,
                components=components,
                antiholomorphic_words=anti_words,
                allocations=tuple(allocation_workspaces),
                zero_picture_legs=picture_legs,  # type: ignore[arg-type]
                maximum_twice_levels=cutoffs,
                maximum_gram_condition=max(
                    (
                        item.maximum_gram_condition
                        for item in allocation_workspaces
                    ),
                    default=1.0,
                ),
            )
        )
    return S1FullClusterWorkspace(
        regulated=tuple(regulated),
        extrapolation_degree=residue_settings.extrapolation_degree,
        maximum_twice_levels=cutoffs,
        maximum_gram_condition=overall_maximum_condition,
    )


def evaluate_s1_full_cluster_pointwise(
    states: Sequence[SphereFivePointState],
    z_2: complex,
    z_3: complex,
    *,
    maximum_twice_levels: int | Sequence[int] = 0,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    residue_settings: ResonanceResidueSettings = ResonanceResidueSettings(),
    block_digits: int = 40,
    condition_limit: float = 1.0e13,
) -> FullClusterPointwiseResult:
    r"""Evaluate the full exact-SL fixed-moduli ``s=1`` pole cluster.

    ``states`` use the worldsheet order ``(0,z2,z3,1,infinity)``.  The
    function is intentionally a pointwise pilot.  A physical five-point
    amplitude requires a separately converged integral over ``z2,z3`` and a
    picture/vertical-completion audit of that integrated result.
    """

    workspace = build_s1_full_cluster_workspace(
        states,
        maximum_twice_levels=maximum_twice_levels,
        zero_picture_legs=zero_picture_legs,
        residue_settings=residue_settings,
        block_digits=block_digits,
        condition_limit=condition_limit,
    )
    return workspace.evaluate(z_2, z_3)


__all__ = [
    "FullClusterAllocationResult",
    "FullClusterAllocationWorkspace",
    "FullClusterCauchyNode",
    "FullClusterPointwiseResult",
    "FullClusterRadiusResult",
    "FullClusterRegulatorWorkspace",
    "FullSphereBootstrapComparison",
    "S1FullClusterWorkspace",
    "build_s1_full_cluster_workspace",
    "compare_integrated_s1_with_bootstrap",
    "evaluate_s1_full_cluster_pointwise",
    "nested_torus_cauchy_residue",
]
