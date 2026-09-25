#!/usr/bin/env python3
r"""Odd global resonances of the heterotic sphere five-point function.

For the physical ``1 -> 4`` branch, write the four outgoing Liouville
momenta as ``P_j`` and the incoming momentum as ``P_0=sum(P_j)``.  The
nonzero NS resonances are

``P_0 = i (3+s)/2``, with ``s=1,3,5,...``.

The ``s`` super-Liouville screenings can be distributed among the three
trinions in ``binomial(s+2,2)`` ways. Analytic continuation of the two
continuum momentum contours localizes them at the corresponding crossing
poles. This module evaluates the resulting *global-pole coefficient*. It
deliberately does not evaluate the ordinary continuum integral at the pinch.

The continuation is parameterized by

``P_j(eta) = (1 + eta/R) P_j``, ``R=(3+s)/2``

so that ``sum(P_j(eta)) = i (R + eta)``. At fixed nonzero ``eta`` a nested
Cauchy integral extracts the two contour residues.  A small polynomial
extrapolation of ``eta * residue`` gives the coefficient of the global
``1/eta`` pole.  The nesting radii encode the continuation from the original
positive-real internal-momentum contours.  The usual two contour-crossing
factor is ``(-2 i)^2``.

All vector/singlet channels allowed by the even-vector free-fermion selection
rule are assembled here.  The returned value uses the
same reduced raw-descendant, picture-raising, timelike, and flavored-fermion conventions as
``spin23_sphere_fivepoint_amplitude``.  The common sphere normalization,
coupling, energy delta function, wall phases, and the conventional
cosmological-constant factor multiplying the resonance residue remain
omitted.

.. warning::

   The workspaces in this module Cauchy-project the product of structure
   constants and evaluate conformal blocks afterward at the pole center.
   That operation omits conformal-block derivatives when poles coalesce at
   ``b=1`` and is therefore diagnostic, not the physical ``s=1`` residue.
   Use :mod:`spin23_s1_full_cluster` for the full-integrand fixed-moduli
   Cauchy pilot.  A completed amplitude additionally requires a converged
   moduli integral and a picture/vertical-completion audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from ns_algebra.ns_sca import Word, fermion_parity
from spin23_sphere_fivepoint import (
    SphereFivePointSeries,
    direct_ns_sphere_fivepoint_series,
    ns_liouville_weight,
)
from spin23_sphere_fivepoint_amplitude import (
    SphereFivePointPCOComponent,
    SphereFivePointState,
    antiholomorphic_liouville_words,
    flavored_sphere_fermion_wick_factor,
    fivepoint_component_phase,
    fivepoint_pco_components,
    timelike_boson_factor,
    validate_fivepoint_states,
)
from spin23_super_liouville_data import ns_structure_constants


FIRST_FIVEPOINT_SCREENING_NUMBER = 1
FIRST_FIVEPOINT_INCOMING_MOMENTUM = 2j
SCREENING_ALLOCATIONS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


def _validate_screening_number(screening_number: int) -> int:
    if (
        isinstance(screening_number, bool)
        or not isinstance(screening_number, int)
        or screening_number < 1
        or screening_number % 2 == 0
    ):
        raise ValueError(
            "five NS external states require a positive odd screening number"
        )
    return screening_number


def screening_allocations(
    screening_number: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return all weak compositions of ``screening_number`` into three parts."""

    _validate_screening_number(screening_number)
    if screening_number == FIRST_FIVEPOINT_SCREENING_NUMBER:
        return SCREENING_ALLOCATIONS
    return tuple(
        (first, second, screening_number - first - second)
        for first in range(screening_number + 1)
        for second in range(screening_number - first + 1)
    )


def _screening_number_from_incoming(momentum: complex) -> int:
    value = complex(momentum)
    candidate = int(round(2.0 * value.imag - 3.0))
    _validate_screening_number(candidate)
    expected = fivepoint_resonance_momentum(candidate)
    if not cmath.isclose(value, expected, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise ValueError(
            "incoming momentum is not on a nonzero five-point NS resonance"
        )
    return candidate


@dataclass(frozen=True)
class ResonanceResidueSettings:
    """Numerical controls for the nested structure-constant residues."""

    regulator_radii: tuple[float, ...] = (0.024, 0.016, 0.010, 0.006)
    regulator_direction: complex = 1.0 + 0.37j
    cauchy_order: int = 8
    outer_radius_ratio: float = 0.08
    inner_radius_ratio: float = 0.025
    extrapolation_degree: int = 2
    structure_precision: int = 50

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.regulator_radii)
        if len(radii) < 2 or any(
            not math.isfinite(value) or value <= 0 for value in radii
        ):
            raise ValueError("at least two positive regulator radii are required")
        if len(set(radii)) != len(radii):
            raise ValueError("regulator radii must be distinct")
        if complex(self.regulator_direction) == 0:
            raise ValueError("regulator_direction must be nonzero")
        if (
            isinstance(self.cauchy_order, bool)
            or not isinstance(self.cauchy_order, int)
            or self.cauchy_order < 4
        ):
            raise ValueError("cauchy_order must be an integer of at least four")
        if not 0 < self.inner_radius_ratio < self.outer_radius_ratio < 0.25:
            raise ValueError(
                "Cauchy ratios must obey 0<inner<outer<0.25"
            )
        if (
            isinstance(self.extrapolation_degree, bool)
            or not isinstance(self.extrapolation_degree, int)
            or self.extrapolation_degree < 1
            or self.extrapolation_degree >= len(radii)
        ):
            raise ValueError(
                "extrapolation_degree must be positive and smaller than the sample count"
            )
        if (
            isinstance(self.structure_precision, bool)
            or not isinstance(self.structure_precision, int)
            or self.structure_precision < 30
        ):
            raise ValueError("structure_precision must be an integer of at least 30")


@dataclass(frozen=True)
class ScreeningResidueEstimate:
    """Leading global-pole weight for one screening allocation."""

    allocation: tuple[int, int, int]
    value: complex
    fit_spread: float
    regulator_values: tuple[complex, ...]
    double_residues: tuple[complex, ...]
    localized_internal_momenta: tuple[complex, complex]


@dataclass(frozen=True)
class ResonantBlockTerm:
    """One localized comb block and its extrapolated structure weight."""

    residue: ScreeningResidueEstimate
    antiholomorphic: SphereFivePointSeries
    holomorphic: tuple[tuple[SphereFivePointPCOComponent, SphereFivePointSeries], ...]


@dataclass(frozen=True)
class ResonantFivePointWorkspace:
    """Reusable localized S/V workspace at one ordered kinematic point."""

    states: tuple[
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
    ]
    terms: tuple[ResonantBlockTerm, ...]
    antiholomorphic_words: tuple[Word, Word, Word, Word, Word]
    zero_picture_legs: tuple[int, int, int]
    maximum_twice_levels: tuple[int, int]
    maximum_gram_condition: float

    def evaluate(
        self,
        z_2: complex,
        z_3: complex,
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Evaluate the reduced coefficient of the global resonance pole."""

        z_2, z_3 = complex(z_2), complex(z_3)
        q_1, q_2 = z_2 / z_3, z_3
        if not (0 < abs(q_1) < 1 and 0 < abs(q_2) < 1):
            raise ValueError("the selected comb requires 0<|q1|,|q2|<1")
        cutoffs = (
            self.maximum_twice_levels
            if maximum_twice_levels is None
            else _normalize_cutoffs(maximum_twice_levels)
        )
        if any(low > high for low, high in zip(cutoffs, self.maximum_twice_levels)):
            raise ValueError("evaluation cutoffs exceed the constructed block order")
        total = 0.0j
        for term in self.terms:
            first_form, central_form, third_form = term.residue.allocation
            anti_routing, anti_central = _routing_for_forms(
                self.antiholomorphic_words, first_form, third_form
            )
            if anti_central != central_form % 2:
                continue
            anti_value = term.antiholomorphic.component_value(
                q_1.conjugate(),
                q_2.conjugate(),
                anti_routing,
                maximum_levels=cutoffs,
            )
            anti_phase = fivepoint_component_phase(
                self.antiholomorphic_words, anti_routing
            )
            component_sum = 0.0j
            for component, block in term.holomorphic:
                routing, selected_central = _routing_for_forms(
                    component.holomorphic_liouville_words,
                    first_form,
                    third_form,
                )
                if selected_central != central_form % 2:
                    continue
                component_sum += (
                    component.momentum_coefficient
                    * component.time_fermion_wick(z_2, z_3)
                    * fivepoint_component_phase(
                        component.holomorphic_liouville_words, routing
                    )
                    * block.component_value(
                        q_1, q_2, routing, maximum_levels=cutoffs
                    )
                )
            total += (
                term.residue.value
                * anti_phase
                * anti_value
                * component_sum
            )

        # One factor -2i for each crossed dP/pi contour, and 1/2 per PCO.
        common = (
            (-2j) ** 2
            * 0.5 ** len(self.zero_picture_legs)
            * timelike_boson_factor(self.states, z_2, z_3)
            * flavored_sphere_fermion_wick_factor(self.states, z_2, z_3)
        )
        return common * total


@dataclass(frozen=True)
class ResonantFullSphereDiagnostics:
    """Nested-QMC diagnostics for the crossing-complete resonance residue."""

    values: tuple[complex, ...]
    standard_errors: tuple[float, ...]
    relative_standard_errors: tuple[float, ...]
    descendant_increments: tuple[complex, ...] | None
    descendant_increment_relative_sizes: tuple[float, ...] | None
    sample_powers: tuple[int, ...]
    replicates: int
    radial_scale: float
    moduli_sampler: str
    chart_evaluations: int
    maximum_gram_condition: float
    maximum_structure_fit_spread: float


@dataclass(frozen=True)
class RegulatedResonantBlockTerm:
    """One screening allocation before the global-pole limit is taken."""

    allocation: tuple[int, int, int]
    double_residue: complex
    antiholomorphic: SphereFivePointSeries
    holomorphic: tuple[tuple[SphereFivePointPCOComponent, SphereFivePointSeries], ...]


@dataclass(frozen=True)
class RegulatedResonantFivePointWorkspace:
    """Complete localized coefficient at one nonzero global regulator."""

    eta: complex
    radius_parameter: float
    states: tuple[
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
        SphereFivePointState,
    ]
    terms: tuple[RegulatedResonantBlockTerm, ...]
    antiholomorphic_words: tuple[Word, Word, Word, Word, Word]
    zero_picture_legs: tuple[int, int, int]
    maximum_twice_levels: tuple[int, int]
    maximum_gram_condition: float

    def evaluate(
        self,
        z_2: complex,
        z_3: complex,
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Return ``eta`` times the fully assembled localized correlator."""

        z_2, z_3 = complex(z_2), complex(z_3)
        q_1, q_2 = z_2 / z_3, z_3
        if not (0 < abs(q_1) < 1 and 0 < abs(q_2) < 1):
            raise ValueError("the selected comb requires 0<|q1|,|q2|<1")
        cutoffs = (
            self.maximum_twice_levels
            if maximum_twice_levels is None
            else _normalize_cutoffs(maximum_twice_levels)
        )
        if any(low > high for low, high in zip(cutoffs, self.maximum_twice_levels)):
            raise ValueError("evaluation cutoffs exceed the constructed block order")
        total = 0.0j
        for term in self.terms:
            first_form, central_form, third_form = term.allocation
            anti_routing, anti_central = _routing_for_forms(
                self.antiholomorphic_words, first_form, third_form
            )
            if anti_central != central_form % 2:
                continue
            anti_value = term.antiholomorphic.component_value(
                q_1.conjugate(),
                q_2.conjugate(),
                anti_routing,
                maximum_levels=cutoffs,
            )
            anti_phase = fivepoint_component_phase(
                self.antiholomorphic_words, anti_routing
            )
            holomorphic_sum = 0.0j
            for component, block in term.holomorphic:
                routing, selected_central = _routing_for_forms(
                    component.holomorphic_liouville_words,
                    first_form,
                    third_form,
                )
                if selected_central != central_form % 2:
                    continue
                holomorphic_sum += (
                    component.momentum_coefficient
                    * component.time_fermion_wick(z_2, z_3)
                    * fivepoint_component_phase(
                        component.holomorphic_liouville_words, routing
                    )
                    * block.component_value(
                        q_1, q_2, routing, maximum_levels=cutoffs
                    )
                )
            total += (
                term.double_residue
                * anti_phase
                * anti_value
                * holomorphic_sum
            )
        return (
            self.eta
            * (-2j) ** 2
            * 0.5 ** len(self.zero_picture_legs)
            * timelike_boson_factor(self.states, z_2, z_3)
            * flavored_sphere_fermion_wick_factor(self.states, z_2, z_3)
            * total
        )


@dataclass(frozen=True)
class AssembledResonantFivePointWorkspace:
    """Regulator family in which all pole sectors are combined before fitting."""

    regulated: tuple[RegulatedResonantFivePointWorkspace, ...]
    extrapolation_degree: int
    maximum_twice_levels: tuple[int, int]
    maximum_gram_condition: float

    def evaluate_with_spread(
        self,
        z_2: complex,
        z_3: complex,
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> tuple[complex, float]:
        """Return the assembled pole coefficient and lower-degree fit spread."""

        values = np.asarray(
            [
                workspace.evaluate(
                    z_2,
                    z_3,
                    maximum_twice_levels=maximum_twice_levels,
                )
                for workspace in self.regulated
            ],
            dtype=np.complex128,
        )
        radii = np.asarray(
            [workspace.radius_parameter for workspace in self.regulated],
            dtype=np.float64,
        )
        return _extrapolated_intercept(
            radii, values, self.extrapolation_degree
        )

    def evaluate(
        self,
        z_2: complex,
        z_3: complex,
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> complex:
        return self.evaluate_with_spread(
            z_2,
            z_3,
            maximum_twice_levels=maximum_twice_levels,
        )[0]


def fivepoint_resonance_momentum(screening_number: int) -> complex:
    r"""Return the all-plus global ``1 -> 4`` resonance momentum.

    Fermion parity permits odd nonnegative screening number only.
    """

    _validate_screening_number(screening_number)
    return 0.5j * (3 + screening_number)


def all_singlet_resonance_states(
    outgoing_momenta: Sequence[complex],
    *,
    screening_number: int,
    tolerance: float = 1.0e-10,
) -> tuple[
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
]:
    """Construct all-singlet states on one odd global resonance plane."""

    _validate_screening_number(screening_number)
    outgoing = tuple(complex(value) for value in outgoing_momenta)
    if len(outgoing) != 4:
        raise ValueError("four outgoing momenta are required")
    incoming = sum(outgoing)
    target = fivepoint_resonance_momentum(screening_number)
    if not cmath.isclose(
        incoming,
        target,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"screening number {screening_number} requires sum(P_j)={target}"
        )
    states = tuple(
        SphereFivePointState.singlet(momentum, momentum)
        for momentum in outgoing
    ) + (SphereFivePointState.singlet(incoming, -incoming),)
    return validate_fivepoint_states(states)


def all_singlet_first_resonance_states(
    outgoing_momenta: Sequence[complex],
    *,
    tolerance: float = 1.0e-10,
) -> tuple[
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
]:
    """Backward-compatible constructor for the one-screening plane."""

    return all_singlet_resonance_states(
        outgoing_momenta,
        screening_number=FIRST_FIVEPOINT_SCREENING_NUMBER,
        tolerance=tolerance,
    )


def localized_internal_momenta(
    external_momenta: Sequence[complex],
    allocation: Sequence[int],
) -> tuple[complex, complex]:
    r"""Return the two crossing poles for one screening allocation.

    External momenta are ordered at ``(0,z2,z3,1,infinity)``.  Starting from
    the infinity end of the comb gives

    ``p2=P5+P4-i(1+s3)``,
    ``p1=p2+P3-i(1+s2)``.
    """

    momenta = tuple(complex(value) for value in external_momenta)
    selected = tuple(int(value) for value in allocation)
    if len(momenta) != 5:
        raise ValueError("five external momenta are required")
    if (
        len(selected) != 3
        or any(value < 0 for value in selected)
        or sum(selected) < 1
        or sum(selected) % 2 == 0
    ):
        raise ValueError(
            "allocation must distribute a positive odd screening number"
        )
    _s1, s2, s3 = selected
    p2 = momenta[4] + momenta[3] - 1j * (1 + s3)
    p1 = p2 + momenta[2] - 1j * (1 + s2)
    return p1, p2


def _routing_for_forms(
    words: Sequence[Word],
    first_form: int,
    third_form: int,
) -> tuple[tuple[int, int], int]:
    first_form %= 2
    third_form %= 2
    parities = tuple(fermion_parity(tuple(word)) for word in words)
    p1, p2, p3, p4, p5 = parities
    first_edge = first_form ^ p1 ^ p2
    second_edge = third_form ^ p4 ^ p5
    central_form = first_edge ^ second_edge ^ p3
    return (first_edge, second_edge), central_form


def _deformed_external_momenta(
    target: Sequence[complex], eta: complex
) -> tuple[complex, complex, complex, complex, complex]:
    incoming = complex(target[4])
    screening_number = _screening_number_from_incoming(incoming)
    resonance_coefficient = 0.5 * (3 + screening_number)
    outgoing = tuple(
        complex(value) * (1.0 + complex(eta) / resonance_coefficient)
        for value in target[:4]
    )
    incoming = sum(outgoing)
    return (*outgoing, incoming)


def _states_with_deformed_momenta(
    target_states: Sequence[SphereFivePointState],
    external_momenta: Sequence[complex],
) -> tuple[
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
    SphereFivePointState,
]:
    """Preserve S/V labels and flavors while deforming resonance momenta."""

    states = validate_fivepoint_states(target_states)
    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 5:
        raise ValueError("five deformed external momenta are required")
    result: list[SphereFivePointState] = []
    for index, (state, momentum) in enumerate(zip(states, momenta)):
        time_momentum = momentum if index < 4 else -momentum
        if state.kind == "vector":
            result.append(
                SphereFivePointState.vector(
                    momentum,
                    time_momentum,
                    int(state.flavor),
                )
            )
        else:
            result.append(
                SphereFivePointState.singlet(momentum, time_momentum)
            )
    return validate_fivepoint_states(result)


def _structure_product(
    internal_1: complex,
    internal_2: complex,
    external: Sequence[complex],
    allocation: tuple[int, int, int],
    precision: int,
) -> complex:
    p1, p2, p3, p4, p5 = tuple(complex(value) for value in external)
    first, central, third = allocation
    return (
        ns_structure_constants(
            internal_1, p2, p1, precision=precision
        )[first % 2]
        * ns_structure_constants(
            internal_2, p3, internal_1, precision=precision
        )[central % 2]
        * ns_structure_constants(
            p5, p4, internal_2, precision=precision
        )[third % 2]
    )


def _nested_double_residue(
    external: Sequence[complex],
    allocation: tuple[int, int, int],
    eta: complex,
    settings: ResonanceResidueSettings,
) -> complex:
    center_1, center_2 = localized_internal_momenta(external, allocation)
    base_radius = abs(complex(eta))
    radius_1 = settings.outer_radius_ratio * base_radius
    radius_2 = settings.inner_radius_ratio * base_radius
    order = settings.cauchy_order
    total = 0.0j
    for first_index in range(order):
        angle_1 = 2 * math.pi * (first_index + 0.173) / order
        delta_1 = radius_1 * cmath.exp(1j * angle_1)
        for second_index in range(order):
            angle_2 = 2 * math.pi * (second_index + 0.319) / order
            delta_2 = radius_2 * cmath.exp(1j * angle_2)
            total += (
                _structure_product(
                    center_1 + delta_1,
                    center_2 + delta_2,
                    external,
                    allocation,
                    settings.structure_precision,
                )
                * delta_1
                * delta_2
            )
    return total / order**2


def _extrapolated_intercept(
    radii: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> tuple[complex, float]:
    design = np.vander(radii, N=degree + 1, increasing=True)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    value = complex(coefficients[0])
    lower_degree = max(0, degree - 1)
    lower_design = np.vander(radii, N=lower_degree + 1, increasing=True)
    lower_coefficients, *_ = np.linalg.lstsq(
        lower_design, values, rcond=None
    )
    spread = float(abs(value - complex(lower_coefficients[0])))
    return value, spread


def estimate_screening_residue(
    target_external_momenta: Sequence[complex],
    allocation: Sequence[int],
    *,
    settings: ResonanceResidueSettings = ResonanceResidueSettings(),
) -> ScreeningResidueEstimate:
    """Extrapolate the coefficient of the global pole for one allocation."""

    target = tuple(complex(value) for value in target_external_momenta)
    selected = tuple(int(value) for value in allocation)
    if len(target) != 5:
        raise ValueError("five target external momenta are required")
    screening_number = sum(selected)
    if screening_number != FIRST_FIVEPOINT_SCREENING_NUMBER:
        raise NotImplementedError(
            "higher screenings require a generic-b or Coulomb-gas residue; "
            "naive b=1 pole sectors have coalescing Upsilon zeros"
        )
    if selected not in screening_allocations(screening_number):
        raise ValueError("invalid odd-screening allocation")
    resonance_momentum = fivepoint_resonance_momentum(screening_number)
    if not cmath.isclose(
        sum(target[:4]), resonance_momentum, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise ValueError("target outgoing momenta do not obey the resonance")
    if not cmath.isclose(
        target[4], resonance_momentum, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise ValueError("target incoming momentum does not obey the resonance")

    regulator_values: list[complex] = []
    double_residues: list[complex] = []
    scaled_values: list[complex] = []
    for radius in settings.regulator_radii:
        eta = radius * complex(settings.regulator_direction)
        external = _deformed_external_momenta(target, eta)
        double_residue = _nested_double_residue(
            external, selected, eta, settings
        )
        regulator_values.append(eta)
        double_residues.append(double_residue)
        scaled_values.append(eta * double_residue)

    radii = np.asarray(settings.regulator_radii, dtype=np.float64)
    values = np.asarray(scaled_values, dtype=np.complex128)
    value, spread = _extrapolated_intercept(
        radii, values, settings.extrapolation_degree
    )
    return ScreeningResidueEstimate(
        allocation=selected,  # type: ignore[arg-type]
        value=value,
        fit_spread=spread,
        regulator_values=tuple(regulator_values),
        double_residues=tuple(double_residues),
        localized_internal_momenta=localized_internal_momenta(target, selected),
    )


def _normalize_cutoffs(
    maximum_twice_levels: int | Sequence[int],
) -> tuple[int, int]:
    if isinstance(maximum_twice_levels, int):
        values = (maximum_twice_levels, maximum_twice_levels)
    else:
        values = tuple(int(value) for value in maximum_twice_levels)
    if len(values) != 2 or any(value < 0 for value in values):
        raise ValueError("two nonnegative twice-level cutoffs are required")
    return values  # type: ignore[return-value]


def build_resonant_fivepoint_workspace(
    states: Sequence[SphereFivePointState],
    *,
    maximum_twice_levels: int | Sequence[int] = 4,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    residue_settings: ResonanceResidueSettings = ResonanceResidueSettings(),
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
) -> ResonantFivePointWorkspace:
    """Build the localized S/V workspace for one comb ordering."""

    normalized = validate_fivepoint_states(states)
    screening_number = _screening_number_from_incoming(
        normalized[4].liouville_momentum
    )
    resonance_momentum = fivepoint_resonance_momentum(screening_number)
    if not cmath.isclose(
        sum(state.liouville_momentum for state in normalized[:4]),
        resonance_momentum,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise ValueError("the outgoing Liouville momenta must obey the resonance")

    cutoffs = _normalize_cutoffs(maximum_twice_levels)
    picture_legs = tuple(int(value) for value in zero_picture_legs)
    components = fivepoint_pco_components(
        normalized, zero_picture_legs=picture_legs
    )
    anti_words = antiholomorphic_liouville_words(normalized)
    external_momenta = tuple(
        complex(state.liouville_momentum) for state in normalized
    )
    external_weights = tuple(
        ns_liouville_weight(momentum, 1.0) for momentum in external_momenta
    )

    terms: list[ResonantBlockTerm] = []
    maximum_condition = 1.0
    for allocation in screening_allocations(screening_number):
        residue = estimate_screening_residue(
            external_momenta, allocation, settings=residue_settings
        )
        internal_weights = tuple(
            ns_liouville_weight(momentum, 1.0)
            for momentum in residue.localized_internal_momenta
        )
        antiholomorphic = direct_ns_sphere_fivepoint_series(
            c=13.5,
            internal_weights=internal_weights,
            external_weights=external_weights,
            maximum_twice_levels=cutoffs,
            external_words=anti_words,
            digits=block_digits,
            condition_limit=condition_limit,
            vertex_backend="template",
        )
        maximum_condition = max(
            maximum_condition,
            max(antiholomorphic.gram_condition_numbers.values(), default=1.0),
        )
        holomorphic: list[
            tuple[SphereFivePointPCOComponent, SphereFivePointSeries]
        ] = []
        cached: dict[tuple[Word, Word, Word, Word, Word], SphereFivePointSeries] = {}
        for component in components:
            words = component.holomorphic_liouville_words
            if words not in cached:
                cached[words] = direct_ns_sphere_fivepoint_series(
                    c=13.5,
                    internal_weights=internal_weights,
                    external_weights=external_weights,
                    maximum_twice_levels=cutoffs,
                    external_words=words,
                    digits=block_digits,
                    condition_limit=condition_limit,
                    vertex_backend="template",
                )
                maximum_condition = max(
                    maximum_condition,
                    max(cached[words].gram_condition_numbers.values(), default=1.0),
                )
            holomorphic.append((component, cached[words]))
        terms.append(
            ResonantBlockTerm(
                residue=residue,
                antiholomorphic=antiholomorphic,
                holomorphic=tuple(holomorphic),
            )
        )

    return ResonantFivePointWorkspace(
        states=normalized,
        terms=tuple(terms),
        antiholomorphic_words=anti_words,
        zero_picture_legs=picture_legs,  # type: ignore[arg-type]
        maximum_twice_levels=cutoffs,
        maximum_gram_condition=maximum_condition,
    )


def build_assembled_resonant_fivepoint_workspace(
    states: Sequence[SphereFivePointState],
    *,
    maximum_twice_levels: int | Sequence[int] = 4,
    zero_picture_legs: Sequence[int] = (2, 3, 4),
    residue_settings: ResonanceResidueSettings = ResonanceResidueSettings(),
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
) -> AssembledResonantFivePointWorkspace:
    """Build the pole family, assembling all allocations before extrapolation.

    This is the production implementation.  In contrast to
    :func:`build_resonant_fivepoint_workspace`, it never extrapolates an
    individual screening allocation.  Negative Laurent powers can and do
    cancel only in the sum over allocations.
    """

    target_states = validate_fivepoint_states(states)
    target_external = tuple(
        complex(state.liouville_momentum) for state in target_states
    )
    screening_number = _screening_number_from_incoming(target_external[4])
    if screening_number != FIRST_FIVEPOINT_SCREENING_NUMBER:
        raise NotImplementedError(
            "higher screenings require a generic-b or Coulomb-gas residue; "
            "naive b=1 pole sectors have coalescing Upsilon zeros"
        )
    resonance_momentum = fivepoint_resonance_momentum(screening_number)
    if not cmath.isclose(
        sum(target_external[:4]), resonance_momentum, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError("the outgoing momenta must obey the selected resonance")
    cutoffs = _normalize_cutoffs(maximum_twice_levels)
    picture_legs = tuple(int(value) for value in zero_picture_legs)
    regulated_workspaces: list[RegulatedResonantFivePointWorkspace] = []
    overall_maximum_condition = 1.0

    for radius in residue_settings.regulator_radii:
        eta = radius * complex(residue_settings.regulator_direction)
        external = _deformed_external_momenta(target_external, eta)
        deformed_states = _states_with_deformed_momenta(
            target_states, external
        )
        components = fivepoint_pco_components(
            deformed_states, zero_picture_legs=picture_legs
        )
        anti_words = antiholomorphic_liouville_words(deformed_states)
        external_weights = tuple(
            ns_liouville_weight(momentum, 1.0) for momentum in external
        )
        terms: list[RegulatedResonantBlockTerm] = []
        maximum_condition = 1.0
        for allocation in screening_allocations(screening_number):
            internal = localized_internal_momenta(external, allocation)
            internal_weights = tuple(
                ns_liouville_weight(momentum, 1.0) for momentum in internal
            )
            double_residue = _nested_double_residue(
                external, allocation, eta, residue_settings
            )
            antiholomorphic = direct_ns_sphere_fivepoint_series(
                c=13.5,
                internal_weights=internal_weights,
                external_weights=external_weights,
                maximum_twice_levels=cutoffs,
                external_words=anti_words,
                digits=block_digits,
                condition_limit=condition_limit,
                vertex_backend="template",
            )
            maximum_condition = max(
                maximum_condition,
                max(
                    antiholomorphic.gram_condition_numbers.values(),
                    default=1.0,
                ),
            )
            cached: dict[
                tuple[Word, Word, Word, Word, Word], SphereFivePointSeries
            ] = {}
            holomorphic: list[
                tuple[SphereFivePointPCOComponent, SphereFivePointSeries]
            ] = []
            for component in components:
                words = component.holomorphic_liouville_words
                if words not in cached:
                    cached[words] = direct_ns_sphere_fivepoint_series(
                        c=13.5,
                        internal_weights=internal_weights,
                        external_weights=external_weights,
                        maximum_twice_levels=cutoffs,
                        external_words=words,
                        digits=block_digits,
                        condition_limit=condition_limit,
                        vertex_backend="template",
                    )
                    maximum_condition = max(
                        maximum_condition,
                        max(
                            cached[words].gram_condition_numbers.values(),
                            default=1.0,
                        ),
                    )
                holomorphic.append((component, cached[words]))
            terms.append(
                RegulatedResonantBlockTerm(
                    allocation=allocation,
                    double_residue=double_residue,
                    antiholomorphic=antiholomorphic,
                    holomorphic=tuple(holomorphic),
                )
            )
        regulated_workspaces.append(
            RegulatedResonantFivePointWorkspace(
                eta=eta,
                radius_parameter=float(radius),
                states=deformed_states,
                terms=tuple(terms),
                antiholomorphic_words=anti_words,
                zero_picture_legs=picture_legs,  # type: ignore[arg-type]
                maximum_twice_levels=cutoffs,
                maximum_gram_condition=maximum_condition,
            )
        )
        overall_maximum_condition = max(
            overall_maximum_condition, maximum_condition
        )
    return AssembledResonantFivePointWorkspace(
        regulated=tuple(regulated_workspaces),
        extrapolation_degree=residue_settings.extrapolation_degree,
        maximum_twice_levels=cutoffs,
        maximum_gram_condition=overall_maximum_condition,
    )


def sobol_resonance_partitions(
    count: int,
    *,
    screening_number: int = FIRST_FIVEPOINT_SCREENING_NUMBER,
    seed: int = 230824,
    floor: float = 0.08,
    ceiling: float | None = None,
) -> tuple[tuple[complex, complex, complex, complex], ...]:
    """Generate deterministic unequal partitions of one resonance momentum."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    _validate_screening_number(screening_number)
    total = fivepoint_resonance_momentum(screening_number).imag
    if not math.isfinite(floor) or not 0 <= floor < total / 4:
        raise ValueError("floor must lie below one fourth of the resonance total")
    selected_ceiling = (
        total - 3 * floor if ceiling is None else float(ceiling)
    )
    if (
        not math.isfinite(selected_ceiling)
        or selected_ceiling <= total / 4
        or selected_ceiling > total - 3 * floor
    ):
        raise ValueError(
            "ceiling must lie between one fourth of the total and total-3*floor"
        )
    # An exponential transform of a scrambled Sobol net gives Dirichlet(1)
    # marginals while retaining deterministic low-discrepancy coverage.
    power = int(math.ceil(math.log2(max(8, 8 * count))))
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    unit = sampler.random_base2(power)
    raw = -np.log(np.maximum(unit, np.finfo(float).tiny))
    simplex = raw / np.sum(raw, axis=1, keepdims=True)
    weights = floor + (total - 4.0 * floor) * simplex
    weights = weights[np.max(weights, axis=1) <= selected_ceiling]
    if len(weights) < count:
        raise RuntimeError("the deterministic Sobol candidate batch was too small")
    weights = weights[:count]
    return tuple(
        tuple(1j * float(value) for value in row)  # type: ignore[misc]
        for row in weights
    )


def singlet_resonance_collision_margin(
    imaginary_coefficients: Sequence[float],
) -> float:
    r"""Return the smallest elementary pair-collision convergence margin.

    Leg 1 is the finite ``-1``-picture outgoing leg.  Legs 2--4 are in
    picture zero.  A pair among legs 2--4 can carry a holomorphic and an
    antiholomorphic fermion pole, giving margin ``2(t_i+t_j-1)`` above the
    radial ``r^-1`` boundary.  A pair involving leg 1 carries only the
    antiholomorphic pole and has margin ``2(t_1+t_j)-1``.

    A positive value is necessary for absolute convergence at all elementary
    outgoing-pair divisors.  It is not, by itself, a proof for every nested
    boundary stratum of compactified ``M_0,5``.
    """

    values = tuple(float(value) for value in imaginary_coefficients)
    if len(values) != 4 or any(not math.isfinite(value) for value in values):
        raise ValueError("four finite imaginary coefficients are required")
    t1, t2, t3, t4 = values
    margins = [2 * (t1 + value) - 1 for value in (t2, t3, t4)]
    margins.extend(
        2 * (first + second - 1)
        for first, second in ((t2, t3), (t2, t4), (t3, t4))
    )
    return min(margins)


def sobol_convergent_resonance_partitions(
    count: int,
    *,
    screening_number: int = FIRST_FIVEPOINT_SCREENING_NUMBER,
    seed: int = 230824,
    first_leg_interval: tuple[float, float] = (0.12, 0.42),
    minimum_margin: float = 0.05,
) -> tuple[tuple[complex, complex, complex, complex], ...]:
    """Generate first-resonance partitions strictly inside the pair wedge.

    The construction sets ``t_2,t_3,t_4 > 1/2`` and distributes their excess
    over a three-simplex after choosing ``t_1``. It then filters on the full
    set of elementary collision margins.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    _validate_screening_number(screening_number)
    total = fivepoint_resonance_momentum(screening_number).imag
    lower, upper = (float(value) for value in first_leg_interval)
    if not 0 < lower < upper or total - 1.5 - upper <= 0:
        raise ValueError(
            "first_leg_interval must be positive and leave room above three halves"
        )
    if not math.isfinite(minimum_margin) or minimum_margin <= 0:
        raise ValueError("minimum_margin must be positive")
    power = int(math.ceil(math.log2(max(8, 4 * count))))
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    unit = sampler.random_base2(power)
    t1 = lower + (upper - lower) * unit[:, 0]
    raw = -np.log(np.maximum(unit[:, 1:], np.finfo(float).tiny))
    simplex = raw / np.sum(raw, axis=1, keepdims=True)
    remaining = total - 1.5 - t1
    other = 0.5 + remaining[:, None] * simplex
    rows = np.column_stack((t1, other))
    margins = np.asarray(
        [singlet_resonance_collision_margin(row) for row in rows]
    )
    rows = rows[margins >= minimum_margin]
    if len(rows) < count:
        raise RuntimeError("the deterministic convergence-wedge batch was too small")
    rows = rows[:count]
    return tuple(
        tuple(1j * float(value) for value in row)  # type: ignore[misc]
        for row in rows
    )


def integrate_full_sphere_resonance(
    states: Sequence[SphereFivePointState],
    *,
    maximum_twice_levels: int | Sequence[int] = 4,
    lower_maximum_twice_levels: int | Sequence[int] | None = 2,
    sample_powers: Sequence[int] = (5, 7),
    replicates: int = 4,
    radial_scale: float = 1.0,
    base_seed: int = 230824,
    moduli_sampler: str = "radial_charts",
    residue_settings: ResonanceResidueSettings = ResonanceResidueSettings(),
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
) -> ResonantFullSphereDiagnostics:
    """Integrate the localized coefficient over the six radial charts.

    ``radial_charts`` samples the two comb parameters directly in unit disks
    and sums the six orderings. This bounded representation is the default.
    ``global_planes`` retains the older independent compactified-plane audit.
    """

    # Imported lazily to keep the pointwise module independent of the global
    # sampler at import time.
    from spin23_sphere_fivepoint_atlas import RADIAL_CHARTS
    from spin23_sphere_fivepoint_full_integral import generate_full_sphere_samples

    normalized = validate_fivepoint_states(states)
    high_cutoffs = _normalize_cutoffs(maximum_twice_levels)
    low_cutoffs = (
        None
        if lower_maximum_twice_levels is None
        else _normalize_cutoffs(lower_maximum_twice_levels)
    )
    if low_cutoffs is not None and any(
        low > high for low, high in zip(low_cutoffs, high_cutoffs)
    ):
        raise ValueError("lower cutoffs must not exceed construction cutoffs")
    powers = tuple(int(value) for value in sample_powers)
    if not powers or tuple(sorted(set(powers))) != powers or powers[0] < 0:
        raise ValueError("sample_powers must be nonnegative and strictly increasing")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("at least two independent replicates are required")
    if moduli_sampler not in ("radial_charts", "global_planes"):
        raise ValueError("moduli_sampler must be 'radial_charts' or 'global_planes'")
    batch = None
    if moduli_sampler == "global_planes":
        batch = generate_full_sphere_samples(
            sample_powers=powers,
            replicates=replicates,
            radial_scale=radial_scale,
            base_seed=base_seed,
        )
        permutations = {
            sample.chart.permutation
            for replicate_samples in batch.samples
            for sample in replicate_samples
        }
    else:
        permutations = set(RADIAL_CHARTS)
    workspaces: dict[
        tuple[int, int, int, int, int], AssembledResonantFivePointWorkspace
    ] = {}
    maximum_condition = 1.0
    maximum_fit_spread = 0.0
    for permutation in permutations:
        ordered_states = tuple(normalized[index] for index in permutation)
        workspace = build_assembled_resonant_fivepoint_workspace(
            ordered_states,
            maximum_twice_levels=high_cutoffs,
            residue_settings=residue_settings,
            block_digits=block_digits,
            condition_limit=condition_limit,
        )
        workspaces[permutation] = workspace
        maximum_condition = max(
            maximum_condition, workspace.maximum_gram_condition
        )

    shape = (len(powers), replicates)
    values = np.zeros(shape, dtype=np.complex128)
    increments = (
        np.zeros(shape, dtype=np.complex128)
        if low_cutoffs is not None
        else None
    )
    prefix_counts = tuple(2**power for power in powers)
    maximum_sample_count = prefix_counts[-1]
    if batch is not None:
        sample_replicates = batch.samples
    else:
        radial_batches = []
        for replicate in range(replicates):
            sampler = qmc.Sobol(d=4, scramble=True, seed=base_seed + replicate)
            points = sampler.random_base2(powers[-1])
            radii_1 = np.sqrt(points[:, 0])
            radii_2 = np.sqrt(points[:, 2])
            q_1 = radii_1 * np.exp(2j * math.pi * points[:, 1])
            q_2 = radii_2 * np.exp(2j * math.pi * points[:, 3])
            radial_batches.append(tuple(zip(q_1, q_2)))
        sample_replicates = tuple(radial_batches)

    for replicate, replicate_samples in enumerate(sample_replicates):
        high_density = np.empty(maximum_sample_count, dtype=np.complex128)
        low_density = (
            np.empty(maximum_sample_count, dtype=np.complex128)
            if low_cutoffs is not None
            else None
        )
        if batch is not None:
            for index, sample in enumerate(replicate_samples):
                chart = sample.chart
                workspace = workspaces[chart.permutation]
                factor = sample.total_measure_jacobian
                high_value, fit_spread = workspace.evaluate_with_spread(
                    chart.chart_z_2,
                    chart.chart_z_3,
                    maximum_twice_levels=high_cutoffs,
                )
                maximum_fit_spread = max(maximum_fit_spread, fit_spread)
                high_density[index] = factor * high_value
                if low_density is not None and low_cutoffs is not None:
                    low_density[index] = factor * workspace.evaluate(
                        chart.chart_z_2,
                        chart.chart_z_3,
                        maximum_twice_levels=low_cutoffs,
                    )
        else:
            for index, (q_1, q_2) in enumerate(replicate_samples):
                z_2, z_3 = q_1 * q_2, q_2
                high_sum = 0.0j
                low_sum = 0.0j
                for permutation, workspace in workspaces.items():
                    high_value, fit_spread = workspace.evaluate_with_spread(
                        z_2,
                        z_3,
                        maximum_twice_levels=high_cutoffs,
                    )
                    maximum_fit_spread = max(maximum_fit_spread, fit_spread)
                    high_sum += high_value
                    if low_density is not None and low_cutoffs is not None:
                        low_sum += workspace.evaluate(
                            z_2,
                            z_3,
                            maximum_twice_levels=low_cutoffs,
                        )
                # d^2 z_2 d^2 z_3 = |q_2|^2 d^2q_1 d^2q_2,
                # and each uniformly sampled unit disk contributes pi.
                factor = math.pi**2 * abs(q_2) ** 2
                high_density[index] = factor * high_sum
                if low_density is not None:
                    low_density[index] = factor * low_sum
        high_prefix = np.cumsum(high_density, dtype=np.complex128)
        low_prefix = (
            None
            if low_density is None
            else np.cumsum(low_density, dtype=np.complex128)
        )
        for power_index, count in enumerate(prefix_counts):
            values[power_index, replicate] = high_prefix[count - 1] / count
            if increments is not None and low_prefix is not None:
                increments[power_index, replicate] = (
                    high_prefix[count - 1] - low_prefix[count - 1]
                ) / count

    means: list[complex] = []
    errors: list[float] = []
    relative_errors: list[float] = []
    for row in values:
        mean = complex(np.mean(row))
        error = math.hypot(
            float(np.std(row.real, ddof=1) / math.sqrt(len(row))),
            float(np.std(row.imag, ddof=1) / math.sqrt(len(row))),
        )
        means.append(mean)
        errors.append(error)
        relative_errors.append(error / max(abs(mean), 1.0e-300))
    increment_means = None
    increment_relative_sizes = None
    if increments is not None:
        increment_means = tuple(complex(np.mean(row)) for row in increments)
        increment_relative_sizes = tuple(
            abs(increment) / max(abs(value), 1.0e-300)
            for increment, value in zip(increment_means, means)
        )
    return ResonantFullSphereDiagnostics(
        values=tuple(means),
        standard_errors=tuple(errors),
        relative_standard_errors=tuple(relative_errors),
        descendant_increments=increment_means,
        descendant_increment_relative_sizes=increment_relative_sizes,
        sample_powers=powers,
        replicates=replicates,
        radial_scale=float(radial_scale),
        moduli_sampler=moduli_sampler,
        chart_evaluations=(
            replicates * maximum_sample_count * len(permutations)
            if moduli_sampler == "radial_charts"
            else replicates * maximum_sample_count
        ),
        maximum_gram_condition=maximum_condition,
        maximum_structure_fit_spread=maximum_fit_spread,
    )


__all__ = [
    "FIRST_FIVEPOINT_INCOMING_MOMENTUM",
    "FIRST_FIVEPOINT_SCREENING_NUMBER",
    "SCREENING_ALLOCATIONS",
    "AssembledResonantFivePointWorkspace",
    "ResonanceResidueSettings",
    "ResonantFullSphereDiagnostics",
    "ResonantFivePointWorkspace",
    "ScreeningResidueEstimate",
    "all_singlet_resonance_states",
    "all_singlet_first_resonance_states",
    "build_assembled_resonant_fivepoint_workspace",
    "build_resonant_fivepoint_workspace",
    "estimate_screening_residue",
    "fivepoint_resonance_momentum",
    "integrate_full_sphere_resonance",
    "localized_internal_momenta",
    "screening_allocations",
    "singlet_resonance_collision_margin",
    "sobol_convergent_resonance_partitions",
    "sobol_resonance_partitions",
]
