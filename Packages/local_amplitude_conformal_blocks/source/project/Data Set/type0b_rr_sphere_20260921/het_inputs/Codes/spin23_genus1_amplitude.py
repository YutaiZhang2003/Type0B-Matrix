#!/usr/bin/env python3
r"""Pointwise genus-one Spin(23) heterotic NS amplitude integrands.

The functions in this module assemble the convention-checked ingredients at
fixed ``(tau, z_i)`` in one convergent necklace channel.  They do not perform
the remaining integration over the torus modulus or puncture positions.

For every external NS state, picture changing acts with the *total* matter
supercurrent.  In the sphere convention already used by this repository,

.. math::

   \frac12 G^{\rm m}_{-1/2}
   \left(e^{ikX^0}V_P\right)
   =\frac12 e^{ikX^0}
    \left(G^{\rm SL}_{-1/2}V_P+k\psi^0V_P\right).

The code therefore sums all even-cardinality subsets of ``psi^0`` terms.
The free time-fermion is written in the convention in which its Wick kernel
is the ordinary Szego kernel and the combined timelike/OPE phases are carried
by the signed momenta ``k``.  This is the convention that gives the positive
``omega_2*omega_3`` term in the established sphere calculation.

The odd torus spin structure vanishes for the two- and three-point processes
implemented here because their Spin(23) fermion insertions cannot saturate
all 23 constant ``lambda^a`` zero modes.  The three even spin structures are
summed with the diagonal HO projector.  The returned value includes

* the nonchiral super-Liouville spectral integral;
* ``X^0``, ``psi^0``, and the 23 free ``lambda^a`` fields;
* the even ``beta-gamma`` determinant;
* ``bc`` ghosts and the fixed-puncture ``d^2 tau/2`` density;
* one exact factor ``1/2`` per picture-raised vertex; and
* the genus-one diagonal GSO coefficient.

It excludes the target-time zero-mode volume (equivalently, the momentum
delta function), the string coupling, and the overall phase ``i**n`` unless
``include_string_phase=True``.  The spectral error estimates are convergence
diagnostics, not rigorous error bounds.
"""

from __future__ import annotations

import cmath
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
import multiprocessing as mp
import os
from typing import Callable, Literal, Mapping, Sequence

from ns_algebra.ns_sca import G, Word
from spin23_genus1_free_fields import (
    even_superghost_chiral_partition,
    flavored_majorana_wick_factor,
    heterotic_fixed_puncture_measure_density,
    majorana_chiral_partition,
    majorana_wick_factor,
    noncompact_boson_vertex_correlator,
)
from spin23_genus1_gso import diagonal_gso_table
from spin23_genus1_spectral import (
    BlockBackend,
    SpectralIntegralDiagnostics,
    integrate_liouville_necklace,
    integrate_liouville_necklace_batch,
)
from spin23_genus1_spin import SpinLabel, TorusSpinStructure
from spin23_super_liouville_data import ns_weight


StateKind = Literal["vector", "singlet"]
LiouvilleEvaluator = Callable[..., SpectralIntegralDiagnostics]
LiouvilleBatchEvaluator = Callable[
    ...,
    tuple[SpectralIntegralDiagnostics, ...],
]

_MAX_FINITE_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")

EMPTY_WORD: Word = ()
G_MINUS_HALF: Word = (G(Fraction(-1, 2)),)


@dataclass(frozen=True)
class GenusOneNSState:
    r"""One external continuum NS state.

    ``liouville_momentum`` labels the super-Liouville primary, whereas
    ``time_momentum`` is the signed momentum in ``exp(i*k*X0)``.  For an
    on-shell state at ``alpha_prime=2`` they obey ``P**2=k**2``.  They remain
    separate because an incoming state generally has ``P=-k``.
    """

    kind: StateKind
    liouville_momentum: complex
    time_momentum: complex
    flavor: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("vector", "singlet"):
            raise ValueError("kind must be 'vector' or 'singlet'")
        p = complex(self.liouville_momentum)
        k = complex(self.time_momentum)
        if not all(
            math.isfinite(part)
            for part in (p.real, p.imag, k.real, k.imag)
        ):
            raise ValueError("external momenta must be finite")
        if self.kind == "vector":
            if not isinstance(self.flavor, int) or not 0 <= self.flavor < 23:
                raise ValueError("a vector flavor must be an integer in [0, 23)")
        elif self.flavor is not None:
            raise ValueError("a singlet does not carry a Spin(23) flavor")

    @classmethod
    def vector(
        cls,
        liouville_momentum: complex,
        time_momentum: complex,
        flavor: int,
    ) -> "GenusOneNSState":
        """Construct a Spin(23)-vector continuum state."""

        return cls("vector", liouville_momentum, time_momentum, flavor)

    @classmethod
    def singlet(
        cls,
        liouville_momentum: complex,
        time_momentum: complex,
    ) -> "GenusOneNSState":
        """Construct a continuum singlet state."""

        return cls("singlet", liouville_momentum, time_momentum)


@dataclass(frozen=True)
class NecklaceCoordinates:
    """One ordered additive torus lift and its annulus plumbing variables."""

    tau: complex
    additive_points: tuple[complex, ...]
    multiplicative_points: tuple[complex, ...]
    plumbing_parameters: tuple[complex, ...]


@dataclass(frozen=True)
class EvenPCOComponent:
    """One term in the total-supercurrent expansion."""

    time_fermion_indices: tuple[int, ...]
    holomorphic_liouville_words: tuple[Word, ...]
    momentum_coefficient: complex


@dataclass(frozen=True)
class ComponentEvaluation:
    """One PCO component and its spectral diagnostic."""

    component: EvenPCOComponent
    value: complex
    multiplier: complex
    liouville: SpectralIntegralDiagnostics


@dataclass(frozen=True)
class FixedSpinEvaluation:
    """One complete even-spin contribution before the GSO coefficient."""

    spin_label: SpinLabel
    value: complex
    estimated_absolute_error: float | None
    common_free_field_factor: complex
    components: tuple[ComponentEvaluation, ...]

    @property
    def absolute_component_scale(self) -> float:
        """Return the sum of absolute PCO-component contributions."""

        return float(sum(abs(component.value) for component in self.components))

    @property
    def cancellation_factor(self) -> float:
        """Return component scale divided by the completed fixed-spin value."""

        scale = self.absolute_component_scale
        if self.value == 0:
            return _MAX_FINITE_FLOAT if scale else 1.0
        return min(scale / abs(self.value), _MAX_FINITE_FLOAT)


@dataclass(frozen=True)
class GenusOneIntegrandEvaluation:
    """The diagonal even-spin sum at one moduli-space point."""

    value: complex
    estimated_absolute_error: float | None
    coordinates: NecklaceCoordinates
    fixed_spin: Mapping[SpinLabel, FixedSpinEvaluation]
    picture_raising_factor: float
    includes_string_phase: bool

    @property
    def absolute_spin_scale(self) -> float:
        """Return the absolute scale before cancellations in the GSO sum."""

        coefficients = {
            entry.spin_structure.label: entry.projector_coefficient
            for entry in diagonal_gso_table()
            if not entry.spin_structure.arf_invariant
        }
        return float(
            sum(
                abs(coefficients[label] * evaluation.value)
                for label, evaluation in self.fixed_spin.items()
            )
        )

    @property
    def cancellation_factor(self) -> float:
        """Return absolute spin scale divided by the completed GSO sum."""

        scale = self.absolute_spin_scale
        if self.value == 0:
            return _MAX_FINITE_FLOAT if scale else 1.0
        return min(scale / abs(self.value), _MAX_FINITE_FLOAT)


@dataclass(frozen=True)
class GenusOneFixedGeometryWorkspace:
    r"""Reusable ordered necklace geometry for an external-energy sweep.

    :meth:`evaluate` keeps the quadrature momentum outermost and evaluates
    every supplied state set at that node before advancing.  Geometry-only
    Gram factors and polynomially whitened Ward tensors are therefore shared
    without changing the finite block or quadrature definitions.
    """

    coordinates: NecklaceCoordinates

    @classmethod
    def from_points(
        cls,
        tau: complex,
        points: Sequence[complex],
    ) -> "GenusOneFixedGeometryWorkspace":
        """Construct one workspace from an ordered additive torus lift."""

        return cls(ordered_necklace_coordinates(tau, points))

    def evaluate(
        self,
        state_sets: Sequence[Sequence[GenusOneNSState]],
        **options: object,
    ) -> tuple[GenusOneIntegrandEvaluation, ...]:
        """Evaluate all state sets with the exact batched-energy assembler."""

        return _evaluate_genus_one_integrand_batch_at_coordinates(
            state_sets,
            self.coordinates,
            **options,
        )


def ordered_necklace_coordinates(
    tau: complex,
    points: Sequence[complex],
    *,
    tolerance: float = 1.0e-12,
) -> NecklaceCoordinates:
    r"""Convert an ordered additive torus lift to annulus plumbing data.

    The first point is translated to zero.  The remaining lifts must obey

    .. math::

       0<\operatorname{Im}z_2<\cdots<\operatorname{Im}z_n<\tau_2.

    The corresponding necklace parameters are

    .. math::

       q_i=e^{2\pi i(z_{i+1}-z_i)},\qquad
       q_n=e^{2\pi i(\tau+z_1-z_n)},

    so that ``prod(q_i)=exp(2*pi*i*tau)``.  This routine defines one channel;
    it does not choose an optimal channel for arbitrary unordered punctures.
    """

    tau = complex(tau)
    if not math.isfinite(tau.real) or not math.isfinite(tau.imag) or tau.imag <= 0:
        raise ValueError("tau must lie in the upper half-plane")
    original = tuple(complex(point) for point in points)
    if not original:
        raise ValueError("at least one puncture is required")
    origin = original[0]
    additive = tuple(point - origin for point in original)
    increments = tuple(
        additive[index + 1] - additive[index]
        for index in range(len(additive) - 1)
    ) + (tau - additive[-1],)
    if any(increment.imag <= tolerance for increment in increments):
        raise ValueError(
            "the selected additive lifts are not strictly ordered in a "
            "convergent annulus channel"
        )
    plumbing = tuple(cmath.exp(2j * math.pi * increment) for increment in increments)
    if any(not 0 < abs(value) < 1 for value in plumbing):
        raise AssertionError("ordered annulus increments produced invalid plumbing data")
    multiplicative = tuple(
        cmath.exp(2j * math.pi * point)
        for point in additive
    )
    expected_nome = cmath.exp(2j * math.pi * tau)
    if abs(math.prod(plumbing) - expected_nome) > tolerance * max(1.0, abs(expected_nome)):
        raise AssertionError("the necklace plumbing parameters do not close")
    return NecklaceCoordinates(
        tau=tau,
        additive_points=additive,
        multiplicative_points=multiplicative,
        plumbing_parameters=plumbing,
    )


def validate_external_states(
    states: Sequence[GenusOneNSState],
    *,
    mass_shell_tolerance: float = 1.0e-10,
    neutrality_tolerance: float = 1.0e-10,
) -> tuple[GenusOneNSState, ...]:
    """Validate the two- or three-point continuum kinematics."""

    normalized = tuple(states)
    if len(normalized) not in (2, 3):
        raise ValueError("the current physical assembler supports two or three states")
    if any(not isinstance(state, GenusOneNSState) for state in normalized):
        raise TypeError("every external state must be a GenusOneNSState")
    if abs(sum(complex(state.time_momentum) for state in normalized)) > neutrality_tolerance:
        raise ValueError("signed timelike momenta must sum to zero")
    for state in normalized:
        p = complex(state.liouville_momentum)
        k = complex(state.time_momentum)
        if abs(p * p - k * k) > mass_shell_tolerance * max(1.0, abs(p * p), abs(k * k)):
            raise ValueError("each state must satisfy P**2=k**2 at alpha_prime=2")
    return normalized


def _validate_external_state_batch(
    state_sets: Sequence[Sequence[GenusOneNSState]],
) -> tuple[tuple[GenusOneNSState, ...], ...]:
    """Validate fixed-operator state sets for one energy sweep."""

    normalized = tuple(validate_external_states(states) for states in state_sets)
    if not normalized:
        raise ValueError("state_sets must not be empty")
    operator_labels = tuple(
        (state.kind, state.flavor) for state in normalized[0]
    )
    if any(
        tuple((state.kind, state.flavor) for state in states)
        != operator_labels
        for states in normalized[1:]
    ):
        raise ValueError(
            "all state sets in a fixed-geometry sweep must have the same "
            "operator kinds and vector flavors"
        )
    return normalized


def even_pco_components(
    states: Sequence[GenusOneNSState],
) -> tuple[EvenPCOComponent, ...]:
    r"""Return all nonzero even-``psi0`` terms in ``prod_i G_m V_i``."""

    normalized = tuple(states)
    components: list[EvenPCOComponent] = []
    for count in range(0, len(normalized) + 1, 2):
        for selected in combinations(range(len(normalized)), count):
            selected_set = set(selected)
            components.append(
                EvenPCOComponent(
                    time_fermion_indices=selected,
                    holomorphic_liouville_words=tuple(
                        EMPTY_WORD if index in selected_set else G_MINUS_HALF
                        for index in range(len(normalized))
                    ),
                    momentum_coefficient=math.prod(
                        complex(normalized[index].time_momentum)
                        for index in selected
                    ),
                )
            )
    return tuple(components)


def antiholomorphic_liouville_words(
    states: Sequence[GenusOneNSState],
) -> tuple[Word, ...]:
    """Return primary words for vectors and ``G_-1/2`` for singlets."""

    return tuple(
        EMPTY_WORD if state.kind == "vector" else G_MINUS_HALF
        for state in states
    )


def _word_weight_shift(word: Word) -> float:
    if word == EMPTY_WORD:
        return 0.0
    if word == G_MINUS_HALF:
        return 0.5
    raise NotImplementedError(
        "only primary and G_-1/2 external components transform as primaries "
        "in the current even-spin assembler"
    )


def liouville_annulus_to_additive_factor(
    coordinates: NecklaceCoordinates,
    external_momenta: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
) -> complex:
    r"""Return the external-coordinate Jacobian from annulus to additive frame.

    The lifted additive coordinates are used directly in the exponent, which
    fixes the branch of ``w_i**h_i`` throughout the selected necklace chart.
    ``G_-1/2 V`` is a Virasoro primary of weight ``h+1/2`` when ``V`` is an NS
    superprimary.  The anti-holomorphic weights are analytically continued
    with the same Liouville momenta and are not complex-conjugated.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    holomorphic = tuple(tuple(word) for word in holomorphic_words)
    antiholomorphic = tuple(tuple(word) for word in antiholomorphic_words)
    if not (
        len(momenta)
        == len(holomorphic)
        == len(antiholomorphic)
        == len(coordinates.additive_points)
    ):
        raise ValueError("coordinate-factor inputs must have equal length")
    log_holomorphic_constant = cmath.log(2j * math.pi)
    log_antiholomorphic_constant = cmath.log(-2j * math.pi)
    exponent = 0.0j
    for point, momentum, hol_word, anti_word in zip(
        coordinates.additive_points,
        momenta,
        holomorphic,
        antiholomorphic,
    ):
        base_weight = complex(ns_weight(momentum))
        hol_weight = base_weight + _word_weight_shift(hol_word)
        anti_weight = base_weight + _word_weight_shift(anti_word)
        exponent += hol_weight * (
            log_holomorphic_constant + 2j * math.pi * point
        )
        exponent += anti_weight * (
            log_antiholomorphic_constant
            - 2j * math.pi * point.conjugate()
        )
    return cmath.exp(exponent)


def evaluate_fixed_even_spin(
    states: Sequence[GenusOneNSState],
    coordinates: NecklaceCoordinates,
    spin_structure: TorusSpinStructure,
    *,
    maximum_twice_levels: int | Sequence[int],
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    free_field_precision: int = 40,
    spectral_method: str = "gauss_legendre",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
    component_indices: Sequence[int] | None = None,
    liouville_evaluator: LiouvilleEvaluator = integrate_liouville_necklace,
) -> FixedSpinEvaluation:
    r"""Evaluate one even spin structure before its GSO coefficient.

    ``component_indices`` may select a disjoint subset of the PCO expansion
    for exact distributed evaluation.  The default evaluates every component.
    Partial results are linear contributions and must be recombined before
    they are interpreted as a complete fixed-spin amplitude.

    When ``subtract_maximum_twice_levels`` is supplied, every Liouville
    component is the difference between ``maximum_twice_levels`` and the
    specified lower rectangular truncation, evaluated from one coefficient
    table.  The returned fixed-spin value is then a truncation correction,
    not the complete integrand at either level.

    ``spectral_workers`` parallelizes independent momentum nodes inside each
    component.  It changes neither the quadrature rule nor the component
    summation order.
    """

    normalized = validate_external_states(states)
    if spin_structure.arf_invariant:
        raise ValueError("evaluate_fixed_even_spin does not accept the odd structure")
    if len(coordinates.additive_points) != len(normalized):
        raise ValueError("one puncture position is required per external state")

    external_momenta = tuple(
        complex(state.liouville_momentum) for state in normalized
    )
    time_momenta = tuple(complex(state.time_momentum) for state in normalized)
    anti_words = antiholomorphic_liouville_words(normalized)
    sector = spin_structure.sector
    temporal_lift_sign = spin_structure.lift_sign

    time_boson = noncompact_boson_vertex_correlator(
        coordinates.additive_points,
        time_momenta,
        coordinates.tau,
        alpha_prime=2.0,
        target_signature=-1,
        include_partition_function=True,
        precision=free_field_precision,
    )
    time_fermion_partition = majorana_chiral_partition(
        coordinates.tau,
        spin_structure,
        n_fermions=1,
        precision=free_field_precision,
    )
    vector_indices = tuple(
        index for index, state in enumerate(normalized) if state.kind == "vector"
    )
    lambda_factor = flavored_majorana_wick_factor(
        tuple(coordinates.additive_points[index] for index in vector_indices),
        tuple(int(normalized[index].flavor) for index in vector_indices),
        coordinates.tau,
        spin_structure,
        n_fermions=23,
        include_even_partition=True,
        precision=free_field_precision,
    ).conjugate()
    superghost = even_superghost_chiral_partition(
        coordinates.tau,
        spin_structure,
        precision=free_field_precision,
    )
    measure_ghost = heterotic_fixed_puncture_measure_density(
        coordinates.tau,
        precision=free_field_precision,
    )
    picture_raising_factor = 0.5 ** len(normalized)
    common = (
        picture_raising_factor
        * time_boson
        * time_fermion_partition
        * lambda_factor
        * superghost
        * measure_ghost
    )

    all_components = even_pco_components(normalized)
    if component_indices is None:
        selected_component_indices = tuple(range(len(all_components)))
    else:
        selected_component_indices = tuple(component_indices)
        if not selected_component_indices:
            raise ValueError("component_indices must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in selected_component_indices
        ):
            raise TypeError("component indices must be integers")
        if len(set(selected_component_indices)) != len(
            selected_component_indices
        ):
            raise ValueError("component_indices must not contain duplicates")
        if any(
            not 0 <= index < len(all_components)
            for index in selected_component_indices
        ):
            raise IndexError(
                "component index lies outside the PCO expansion"
            )
        # Preserve the canonical expansion order independently of how a
        # distributed caller orders its requested shard labels.
        selected_component_indices = tuple(sorted(selected_component_indices))

    component_evaluations: list[ComponentEvaluation] = []
    total = 0.0j
    error_terms: list[float] = []
    for component_index in selected_component_indices:
        component = all_components[component_index]
        selected_points = tuple(
            coordinates.additive_points[index]
            for index in component.time_fermion_indices
        )
        normalized_wick = majorana_wick_factor(
            selected_points,
            coordinates.tau,
            spin_structure,
            precision=free_field_precision,
        )
        frame_factor = liouville_annulus_to_additive_factor(
            coordinates,
            external_momenta,
            component.holomorphic_liouville_words,
            anti_words,
        )
        multiplier = (
            common
            * component.momentum_coefficient
            * normalized_wick
            * frame_factor
        )
        liouville = liouville_evaluator(
            sector=sector,
            external_momenta=external_momenta,
            plumbing_parameters=coordinates.plumbing_parameters,
            holomorphic_words=component.holomorphic_liouville_words,
            antiholomorphic_words=anti_words,
            maximum_twice_levels=maximum_twice_levels,
            temporal_lift_sign=temporal_lift_sign,
            p_max=p_max,
            quadrature_order=quadrature_order,
            refined_order=refined_order,
            extended_p_max=extended_p_max,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
            spectral_method=spectral_method,
            spectral_relative_tolerance=spectral_relative_tolerance,
            block_backend=block_backend,
            recursion_radius=recursion_radius,
            recursion_check_radius=recursion_check_radius,
            recursion_samples=recursion_samples,
            recursion_finite_part_tolerance=recursion_finite_part_tolerance,
            spectral_workers=spectral_workers,
            subtract_maximum_twice_levels=subtract_maximum_twice_levels,
        )
        value = multiplier * liouville.value
        total += value
        if liouville.estimated_absolute_error is not None:
            error_terms.append(abs(multiplier) * liouville.estimated_absolute_error)
        component_evaluations.append(
            ComponentEvaluation(
                component=component,
                value=value,
                multiplier=multiplier,
                liouville=liouville,
            )
        )
    return FixedSpinEvaluation(
        spin_label=spin_structure.label,
        value=total,
        estimated_absolute_error=(sum(error_terms) if error_terms else None),
        common_free_field_factor=common,
        components=tuple(component_evaluations),
    )


def evaluate_fixed_even_spin_batch(
    state_sets: Sequence[Sequence[GenusOneNSState]],
    coordinates: NecklaceCoordinates,
    spin_structure: TorusSpinStructure,
    *,
    maximum_twice_levels: int | Sequence[int],
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    free_field_precision: int = 40,
    spectral_method: str = "gauss_laguerre_adaptive",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct_fast",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
    component_indices: Sequence[int] | None = None,
    liouville_batch_evaluator: LiouvilleBatchEvaluator = (
        integrate_liouville_necklace_batch
    ),
) -> tuple[FixedSpinEvaluation, ...]:
    """Evaluate one spin structure for many energies at fixed geometry."""

    normalized_batch = _validate_external_state_batch(state_sets)
    if spin_structure.arf_invariant:
        raise ValueError(
            "evaluate_fixed_even_spin_batch does not accept the odd structure"
        )
    state_count = len(normalized_batch[0])
    if len(coordinates.additive_points) != state_count:
        raise ValueError("one puncture position is required per external state")

    external_batch = tuple(
        tuple(complex(state.liouville_momentum) for state in states)
        for states in normalized_batch
    )
    time_batch = tuple(
        tuple(complex(state.time_momentum) for state in states)
        for states in normalized_batch
    )
    anti_words = antiholomorphic_liouville_words(normalized_batch[0])
    sector = spin_structure.sector
    temporal_lift_sign = spin_structure.lift_sign

    time_bosons = tuple(
        noncompact_boson_vertex_correlator(
            coordinates.additive_points,
            time_momenta,
            coordinates.tau,
            alpha_prime=2.0,
            target_signature=-1,
            include_partition_function=True,
            precision=free_field_precision,
        )
        for time_momenta in time_batch
    )
    time_fermion_partition = majorana_chiral_partition(
        coordinates.tau,
        spin_structure,
        n_fermions=1,
        precision=free_field_precision,
    )
    vector_indices = tuple(
        index
        for index, state in enumerate(normalized_batch[0])
        if state.kind == "vector"
    )
    lambda_factor = flavored_majorana_wick_factor(
        tuple(coordinates.additive_points[index] for index in vector_indices),
        tuple(
            int(normalized_batch[0][index].flavor)
            for index in vector_indices
        ),
        coordinates.tau,
        spin_structure,
        n_fermions=23,
        include_even_partition=True,
        precision=free_field_precision,
    ).conjugate()
    superghost = even_superghost_chiral_partition(
        coordinates.tau,
        spin_structure,
        precision=free_field_precision,
    )
    measure_ghost = heterotic_fixed_puncture_measure_density(
        coordinates.tau,
        precision=free_field_precision,
    )
    picture_raising_factor = 0.5**state_count
    fixed_common = (
        picture_raising_factor
        * time_fermion_partition
        * lambda_factor
        * superghost
        * measure_ghost
    )
    common_factors = tuple(fixed_common * value for value in time_bosons)

    components_by_energy = tuple(
        even_pco_components(states) for states in normalized_batch
    )
    component_shapes = tuple(
        (
            component.time_fermion_indices,
            component.holomorphic_liouville_words,
        )
        for component in components_by_energy[0]
    )
    if any(
        tuple(
            (
                component.time_fermion_indices,
                component.holomorphic_liouville_words,
            )
            for component in components
        )
        != component_shapes
        for components in components_by_energy[1:]
    ):
        raise AssertionError("the fixed operator batch changed its PCO expansion")
    component_count = len(components_by_energy[0])
    if component_indices is None:
        selected_indices = tuple(range(component_count))
    else:
        selected_indices = tuple(component_indices)
        if not selected_indices:
            raise ValueError("component_indices must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in selected_indices
        ):
            raise TypeError("component indices must be integers")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("component_indices must not contain duplicates")
        if any(not 0 <= index < component_count for index in selected_indices):
            raise IndexError("component index lies outside the PCO expansion")
        selected_indices = tuple(sorted(selected_indices))

    totals = [0.0j] * len(normalized_batch)
    error_terms: list[list[float]] = [
        [] for _ in normalized_batch
    ]
    evaluated_components: list[list[ComponentEvaluation]] = [
        [] for _ in normalized_batch
    ]
    for component_index in selected_indices:
        reference_component = components_by_energy[0][component_index]
        selected_points = tuple(
            coordinates.additive_points[index]
            for index in reference_component.time_fermion_indices
        )
        normalized_wick = majorana_wick_factor(
            selected_points,
            coordinates.tau,
            spin_structure,
            precision=free_field_precision,
        )
        frame_factors = tuple(
            liouville_annulus_to_additive_factor(
                coordinates,
                external_momenta,
                reference_component.holomorphic_liouville_words,
                anti_words,
            )
            for external_momenta in external_batch
        )
        multipliers = tuple(
            common
            * components[component_index].momentum_coefficient
            * normalized_wick
            * frame
            for common, components, frame in zip(
                common_factors,
                components_by_energy,
                frame_factors,
            )
        )
        liouville_batch = liouville_batch_evaluator(
            sector=sector,
            external_momenta_batch=external_batch,
            plumbing_parameters=coordinates.plumbing_parameters,
            holomorphic_words=(
                reference_component.holomorphic_liouville_words
            ),
            antiholomorphic_words=anti_words,
            maximum_twice_levels=maximum_twice_levels,
            temporal_lift_sign=temporal_lift_sign,
            p_max=p_max,
            quadrature_order=quadrature_order,
            refined_order=refined_order,
            extended_p_max=extended_p_max,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
            spectral_method=spectral_method,
            spectral_relative_tolerance=spectral_relative_tolerance,
            block_backend=block_backend,
            recursion_radius=recursion_radius,
            recursion_check_radius=recursion_check_radius,
            recursion_samples=recursion_samples,
            recursion_finite_part_tolerance=(
                recursion_finite_part_tolerance
            ),
            spectral_workers=spectral_workers,
            subtract_maximum_twice_levels=(
                subtract_maximum_twice_levels
            ),
        )
        if len(liouville_batch) != len(normalized_batch):
            raise AssertionError("the Liouville batch returned the wrong size")
        for energy_index, (component, multiplier, liouville) in enumerate(
            zip(
                (components[component_index] for components in components_by_energy),
                multipliers,
                liouville_batch,
            )
        ):
            value = multiplier * liouville.value
            totals[energy_index] += value
            if liouville.estimated_absolute_error is not None:
                error_terms[energy_index].append(
                    abs(multiplier) * liouville.estimated_absolute_error
                )
            evaluated_components[energy_index].append(
                ComponentEvaluation(
                    component=component,
                    value=value,
                    multiplier=multiplier,
                    liouville=liouville,
                )
            )

    return tuple(
        FixedSpinEvaluation(
            spin_label=spin_structure.label,
            value=totals[index],
            estimated_absolute_error=(
                sum(error_terms[index]) if error_terms[index] else None
            ),
            common_free_field_factor=common_factors[index],
            components=tuple(evaluated_components[index]),
        )
        for index in range(len(normalized_batch))
    )


def _evaluate_genus_one_integrand_batch_at_coordinates(
    state_sets: Sequence[Sequence[GenusOneNSState]],
    coordinates: NecklaceCoordinates,
    *,
    maximum_twice_levels: int | Sequence[int],
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    free_field_precision: int = 40,
    spectral_method: str = "gauss_laguerre_adaptive",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct_fast",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    spin_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
    spin_labels: Sequence[SpinLabel] | None = None,
    component_indices: Sequence[int] | None = None,
    include_string_phase: bool = False,
    liouville_batch_evaluator: LiouvilleBatchEvaluator = (
        integrate_liouville_necklace_batch
    ),
) -> tuple[GenusOneIntegrandEvaluation, ...]:
    """Evaluate a fixed-geometry state batch after coordinates are built."""

    normalized_batch = _validate_external_state_batch(state_sets)
    if isinstance(spin_workers, bool) or not isinstance(spin_workers, int):
        raise TypeError("spin_workers must be an integer")
    if not 1 <= spin_workers <= 3:
        raise ValueError("spin_workers must lie between one and three")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    requested_cpus = spin_workers * spectral_workers
    if slurm_cpus is not None and requested_cpus > int(slurm_cpus):
        raise ValueError(
            f"nested spin/spectral workers ({requested_cpus}) exceed the "
            f"Slurm cpus-per-task allocation ({slurm_cpus})"
        )

    all_entries = tuple(
        entry
        for entry in diagonal_gso_table()
        if not entry.spin_structure.arf_invariant
    )
    if spin_labels is None:
        selected_labels = tuple(
            entry.spin_structure.label for entry in all_entries
        )
    else:
        selected_labels = tuple(spin_labels)
        allowed_labels = {
            entry.spin_structure.label for entry in all_entries
        }
        if not selected_labels or any(
            label not in allowed_labels for label in selected_labels
        ):
            raise ValueError(
                "spin_labels must be a nonempty subset of the even labels"
            )
        if len(set(selected_labels)) != len(selected_labels):
            raise ValueError("spin_labels must not contain duplicates")
    entries = tuple(
        entry
        for entry in all_entries
        if entry.spin_structure.label in selected_labels
    )
    if spin_workers > len(entries):
        raise ValueError("spin_workers cannot exceed the selected spin count")

    spin_options = {
        "maximum_twice_levels": maximum_twice_levels,
        "p_max": p_max,
        "quadrature_order": quadrature_order,
        "refined_order": refined_order,
        "extended_p_max": extended_p_max,
        "structure_precision": structure_precision,
        "block_digits": block_digits,
        "condition_limit": condition_limit,
        "free_field_precision": free_field_precision,
        "spectral_method": spectral_method,
        "spectral_relative_tolerance": spectral_relative_tolerance,
        "block_backend": block_backend,
        "recursion_radius": recursion_radius,
        "recursion_check_radius": recursion_check_radius,
        "recursion_samples": recursion_samples,
        "recursion_finite_part_tolerance": (
            recursion_finite_part_tolerance
        ),
        "spectral_workers": spectral_workers,
        "subtract_maximum_twice_levels": (
            subtract_maximum_twice_levels
        ),
        "component_indices": component_indices,
        "liouville_batch_evaluator": liouville_batch_evaluator,
    }
    if spin_workers == 1:
        evaluations_by_spin = tuple(
            evaluate_fixed_even_spin_batch(
                normalized_batch,
                coordinates,
                entry.spin_structure,
                **spin_options,
            )
            for entry in entries
        )
    else:
        if os.name != "posix":
            raise RuntimeError("parallel spin evaluation requires POSIX fork")
        with ProcessPoolExecutor(
            max_workers=spin_workers,
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = tuple(
                executor.submit(
                    evaluate_fixed_even_spin_batch,
                    normalized_batch,
                    coordinates,
                    entry.spin_structure,
                    **spin_options,
                )
                for entry in entries
            )
            evaluations_by_spin = tuple(
                future.result() for future in futures
            )

    batch_size = len(normalized_batch)
    if any(len(evaluations) != batch_size for evaluations in evaluations_by_spin):
        raise AssertionError("a fixed-spin energy batch returned the wrong size")
    phase = (1j ** len(normalized_batch[0])) if include_string_phase else 1.0
    results: list[GenusOneIntegrandEvaluation] = []
    for energy_index in range(batch_size):
        fixed: dict[SpinLabel, FixedSpinEvaluation] = {}
        total = 0.0j
        error_terms: list[float] = []
        for entry, evaluations in zip(entries, evaluations_by_spin):
            evaluation = evaluations[energy_index]
            fixed[entry.spin_structure.label] = evaluation
            total += entry.projector_coefficient * evaluation.value
            if evaluation.estimated_absolute_error is not None:
                error_terms.append(
                    abs(entry.projector_coefficient)
                    * evaluation.estimated_absolute_error
                )
        results.append(
            GenusOneIntegrandEvaluation(
                value=phase * total,
                estimated_absolute_error=(
                    sum(error_terms) if error_terms else None
                ),
                coordinates=coordinates,
                fixed_spin=fixed,
                picture_raising_factor=0.5 ** len(normalized_batch[0]),
                includes_string_phase=include_string_phase,
            )
        )
    return tuple(results)


def evaluate_genus_one_integrand_batch(
    state_sets: Sequence[Sequence[GenusOneNSState]],
    tau: complex,
    points: Sequence[complex],
    **options: object,
) -> tuple[GenusOneIntegrandEvaluation, ...]:
    r"""Evaluate many energies in one exact fixed-:math:`(\tau,z)` sweep.

    The state sets must have identical operator kinds and vector flavors.
    Numerical options are the same as
    :func:`evaluate_genus_one_integrand`, except that only the generalized
    Gauss--Laguerre spectral methods support batched evaluation.
    """

    workspace = GenusOneFixedGeometryWorkspace.from_points(tau, points)
    return workspace.evaluate(state_sets, **options)


def evaluate_genus_one_integrand(
    states: Sequence[GenusOneNSState],
    tau: complex,
    points: Sequence[complex],
    *,
    maximum_twice_levels: int | Sequence[int],
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    free_field_precision: int = 40,
    spectral_method: str = "gauss_legendre",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    spin_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
    spin_labels: Sequence[SpinLabel] | None = None,
    component_indices: Sequence[int] | None = None,
    include_string_phase: bool = False,
    liouville_evaluator: LiouvilleEvaluator = integrate_liouville_necklace,
) -> GenusOneIntegrandEvaluation:
    r"""Return the diagonal even-spin sum at one ordered moduli point.

    The calculation includes every selected PCO component, free-field and
    ghost factor, and diagonal HO projector coefficient, but no integration
    over ``tau`` or the unfixed puncture positions.  If
    ``subtract_maximum_twice_levels`` is supplied, the result is the
    corresponding block-truncation correction rather than a complete
    fixed-level integrand.

    ``spin_workers`` distributes independent spin structures and
    ``spectral_workers`` distributes momentum nodes inside each spin task.
    Their product is checked against ``SLURM_CPUS_PER_TASK`` when that
    allocation is available.
    """

    normalized = validate_external_states(states)
    coordinates = ordered_necklace_coordinates(tau, points)
    if isinstance(spin_workers, bool) or not isinstance(spin_workers, int):
        raise TypeError("spin_workers must be an integer")
    if not 1 <= spin_workers <= 3:
        raise ValueError("spin_workers must lie between one and three")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    requested_cpus = spin_workers * spectral_workers
    if slurm_cpus is not None and requested_cpus > int(slurm_cpus):
        raise ValueError(
            f"nested spin/spectral workers ({requested_cpus}) exceed the "
            f"Slurm cpus-per-task allocation ({slurm_cpus})"
        )

    all_entries = tuple(
        entry for entry in diagonal_gso_table()
        if not entry.spin_structure.arf_invariant
    )
    if spin_labels is None:
        selected_labels = tuple(
            entry.spin_structure.label for entry in all_entries
        )
    else:
        selected_labels = tuple(spin_labels)
        allowed_labels = {
            entry.spin_structure.label for entry in all_entries
        }
        if not selected_labels or any(
            label not in allowed_labels for label in selected_labels
        ):
            raise ValueError(
                "spin_labels must be a nonempty subset of the even labels"
            )
        if len(set(selected_labels)) != len(selected_labels):
            raise ValueError("spin_labels must not contain duplicates")
    entries = tuple(
        entry for entry in all_entries
        if entry.spin_structure.label in selected_labels
    )
    if spin_workers > len(entries):
        raise ValueError("spin_workers cannot exceed the selected spin count")
    spin_options = {
        "maximum_twice_levels": maximum_twice_levels,
        "p_max": p_max,
        "quadrature_order": quadrature_order,
        "refined_order": refined_order,
        "extended_p_max": extended_p_max,
        "structure_precision": structure_precision,
        "block_digits": block_digits,
        "condition_limit": condition_limit,
        "free_field_precision": free_field_precision,
        "spectral_method": spectral_method,
        "spectral_relative_tolerance": spectral_relative_tolerance,
        "block_backend": block_backend,
        "recursion_radius": recursion_radius,
        "recursion_check_radius": recursion_check_radius,
        "recursion_samples": recursion_samples,
        "recursion_finite_part_tolerance": recursion_finite_part_tolerance,
        "spectral_workers": spectral_workers,
        "subtract_maximum_twice_levels": subtract_maximum_twice_levels,
        "component_indices": component_indices,
        "liouville_evaluator": liouville_evaluator,
    }
    if spin_workers == 1:
        evaluations = tuple(
            evaluate_fixed_even_spin(
                normalized,
                coordinates,
                entry.spin_structure,
                **spin_options,
            )
            for entry in entries
        )
    else:
        if os.name != "posix":
            raise RuntimeError("parallel spin evaluation requires POSIX fork")
        with ProcessPoolExecutor(
            max_workers=spin_workers,
            mp_context=mp.get_context("fork"),
        ) as executor:
            futures = tuple(
                executor.submit(
                    evaluate_fixed_even_spin,
                    normalized,
                    coordinates,
                    entry.spin_structure,
                    **spin_options,
                )
                for entry in entries
            )
            evaluations = tuple(future.result() for future in futures)

    fixed: dict[SpinLabel, FixedSpinEvaluation] = {}
    total = 0.0j
    error_terms: list[float] = []
    for entry, evaluation in zip(entries, evaluations):
        fixed[entry.spin_structure.label] = evaluation
        total += entry.projector_coefficient * evaluation.value
        if evaluation.estimated_absolute_error is not None:
            error_terms.append(
                abs(entry.projector_coefficient)
                * evaluation.estimated_absolute_error
            )

    phase = (1j ** len(normalized)) if include_string_phase else 1.0
    return GenusOneIntegrandEvaluation(
        value=phase * total,
        estimated_absolute_error=(sum(error_terms) if error_terms else None),
        coordinates=coordinates,
        fixed_spin=fixed,
        picture_raising_factor=0.5 ** len(normalized),
        includes_string_phase=include_string_phase,
    )


__all__ = [
    "ComponentEvaluation",
    "EMPTY_WORD",
    "EvenPCOComponent",
    "FixedSpinEvaluation",
    "G_MINUS_HALF",
    "GenusOneIntegrandEvaluation",
    "GenusOneFixedGeometryWorkspace",
    "GenusOneNSState",
    "NecklaceCoordinates",
    "antiholomorphic_liouville_words",
    "evaluate_fixed_even_spin",
    "evaluate_fixed_even_spin_batch",
    "evaluate_genus_one_integrand",
    "evaluate_genus_one_integrand_batch",
    "even_pco_components",
    "liouville_annulus_to_additive_factor",
    "ordered_necklace_coordinates",
    "validate_external_states",
]
