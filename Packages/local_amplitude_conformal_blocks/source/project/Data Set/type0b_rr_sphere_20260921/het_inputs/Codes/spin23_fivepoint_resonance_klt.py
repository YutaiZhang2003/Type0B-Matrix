#!/usr/bin/env python3
r"""Numerical KLT evaluator for the first five-point wall resonance.

The object evaluated here is the meromorphic screened free-field
representative.  It is deliberately kept separate from the still-unfixed
Liouville zero-mode, cosmological-constant, wall-interaction, external-leg,
and in/out phase normalization.

Two regulators are removed only after a complete KLT contraction:

* a sixth, soft picture-zero singlet regulates collisions with the wall
  insertion; and
* a generic twist deformation regulates endpoint poles of individual real
  periods which cancel against the sine kernel and the other chambers.

The returned ``wall_value`` includes all four picture-zero factors in the
ordinary six-point regulator and the factor of two relating its zero-momentum
soft vertex to the separated Yukawa-screening representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mpmath as mp
import numpy as np

from spin23_fivepoint_klt import (
    DEFAULT_TWIST_REGULATOR_DIRECTION,
    KLT_WORDS,
    OrderedSimplexTerm as KLTOrderedSimplexTerm,
    contract_klt_periods,
    klt_sine_kernel,
    negative_period_terms,
    positive_period_terms,
    regulated_twist_exponents,
)
from spin23_fivepoint_resonance_channels import (
    FivePointChannelProjection,
    soft_regulated_worldsheet_states,
)
from spin23_twisted_periods import (
    OrderedSimplexIntegral,
    OrderedSimplexTerm,
    OrderedSimplexVectorIntegral,
    integrate_term_components,
    integrate_terms,
)


@dataclass(frozen=True)
class RegulatedFivePointKLTResult:
    """One fully contracted value at nonzero soft and twist regulators."""

    projection_label: str
    soft_momentum: complex
    twist_regulator: complex
    sixpoint_value: complex
    wall_value: complex
    absolute_error_bound: float
    left_periods: tuple[complex, ...]
    right_periods: tuple[complex, ...]
    left_errors: tuple[float, ...]
    right_errors: tuple[float, ...]
    kernel_condition_number: float
    cubature_calls: int
    subdivisions: int
    periods_converged: bool


@dataclass(frozen=True)
class LaurentFit:
    """A linear Laurent fit in one real regulator radius."""

    powers: tuple[int, ...]
    coefficients: tuple[complex, ...]
    finite_part: complex
    residual_norm: float
    condition_number: float


def s1_outgoing_leg_product(
    resonant_outgoing_momenta: Sequence[complex],
) -> complex:
    r"""Return the four outgoing delta-normalized Liouville leg factors.

    With ``x_i=i P_i`` and ``sum(x_i)=-2``, the product is

    ``G4 = product_i Gamma(-x_i) / Gamma(x_i)``.

    The common zero-mode/screening normalization is intentionally not part of
    this function.
    """

    outgoing = tuple(complex(value) for value in resonant_outgoing_momenta)
    if len(outgoing) != 4 or abs(sum(outgoing) - 2j) > 1.0e-10:
        raise ValueError("four first-resonance momenta with sum 2i are required")
    result = mp.mpc(1)
    for momentum in outgoing:
        x_value = mp.mpc(1j * momentum)
        result *= mp.gamma(-x_value) / mp.gamma(x_value)
    return complex(result)


def s1_delta_normalized_value(
    result: RegulatedFivePointKLTResult,
    resonant_outgoing_momenta: Sequence[complex],
    *,
    pco_magnitudes_stripped: bool = False,
    wall_mu: complex | None = None,
) -> complex:
    r"""Attach the audited first-resonance leg and zero-mode factor.

    ``result.wall_value`` contains the physical magnitude ``1/8`` from three
    PCOs.  The canonical screening-stripped answer is ``-2 i G4 I``.  Set
    ``pco_magnitudes_stripped=True`` for the repository's reduced convention,
    which multiplies this by eight.  If ``wall_mu`` is supplied, the formal
    Yukawa-wall coefficient ``-2 i mu`` is included as well.

    A common sphere/cosmological normalization and the wall-fermion ordering
    sign remain conventional; this helper does not pretend to fix them.
    """

    value = (
        -2j
        * s1_outgoing_leg_product(resonant_outgoing_momenta)
        * result.wall_value
    )
    if pco_magnitudes_stripped:
        value *= 8.0
    if wall_mu is not None:
        value *= -2j * complex(wall_mu)
    return value


def _backend_terms(
    terms: Sequence[KLTOrderedSimplexTerm],
) -> tuple[OrderedSimplexTerm, ...]:
    return tuple(
        OrderedSimplexTerm(
            divisor_powers=dict(term.powers),
            coefficient=term.coefficient,
        )
        for term in terms
    )


def _integrate_period(
    terms: Sequence[KLTOrderedSimplexTerm],
    *,
    rtol: float,
    atol: float,
    rule: str,
    max_subdivisions: int,
) -> OrderedSimplexIntegral:
    backend = _backend_terms(terms)
    if len(backend) == 1:
        return integrate_terms(
            backend,
            rtol=rtol,
            atol=atol,
            rule=rule,
            max_subdivisions=max_subdivisions,
        )

    # Keep Wick monomials as separate output components during IBP.  Combining
    # them symbolically first is mathematically equivalent but makes the
    # all-singlet Pfaffian expand into an enormous differentiated rational
    # expression.  A common vector continuation preserves all cancellations.
    diagnostic = integrate_term_components(
        tuple((term,) for term in backend),
        rtol=rtol,
        atol=atol,
        rule=rule,
        max_subdivisions=max_subdivisions,
    )
    return OrderedSimplexIntegral(
        value=sum(diagnostic.values, 0.0j),
        absolute_error=sum(diagnostic.absolute_errors),
        sector_values=tuple(
            sum(sector, 0.0j) for sector in diagnostic.sector_values
        ),
        sector_errors=tuple(sum(sector) for sector in diagnostic.sector_errors),
        cubature_calls=diagnostic.cubature_calls,
        subdivisions=diagnostic.subdivisions,
        converged=diagnostic.converged,
    )


def _integrate_period_components(
    components: Sequence[Sequence[KLTOrderedSimplexTerm]],
    *,
    rtol: float,
    atol: float,
    rule: str,
    max_subdivisions: int,
) -> OrderedSimplexVectorIntegral:
    """Integrate several right-moving periods on one shared cubature grid."""

    backend_components = tuple(_backend_terms(terms) for terms in components)
    term_components = tuple(
        (term,) for component in backend_components for term in component
    )
    diagnostic = integrate_term_components(
        term_components,
        rtol=rtol,
        atol=atol,
        rule=rule,
        max_subdivisions=max_subdivisions,
    )
    offsets = [0]
    for component in backend_components:
        offsets.append(offsets[-1] + len(component))

    def aggregate(entries: Sequence[complex | float]) -> tuple:
        return tuple(
            sum(entries[offsets[index] : offsets[index + 1]])
            for index in range(len(backend_components))
        )

    return OrderedSimplexVectorIntegral(
        values=tuple(complex(value) for value in aggregate(diagnostic.values)),
        absolute_errors=tuple(
            float(value) for value in aggregate(diagnostic.absolute_errors)
        ),
        sector_values=tuple(
            tuple(complex(value) for value in aggregate(sector))
            for sector in diagnostic.sector_values
        ),
        sector_errors=tuple(
            tuple(float(value) for value in aggregate(sector))
            for sector in diagnostic.sector_errors
        ),
        cubature_calls=diagnostic.cubature_calls,
        subdivisions=diagnostic.subdivisions,
        converged=diagnostic.converged,
    )


def evaluate_regulated_klt_projection(
    projection: FivePointChannelProjection,
    resonant_outgoing_momenta: Sequence[complex],
    *,
    soft_momentum: complex,
    twist_regulator: complex,
    twist_direction: Mapping[
        tuple[int, int], complex
    ] = DEFAULT_TWIST_REGULATOR_DIRECTION,
    rtol: float = 2.0e-4,
    atol: float = 2.0e-7,
    rule: str = "gk15",
    max_subdivisions: int = 10_000,
) -> RegulatedFivePointKLTResult:
    r"""Evaluate one tensor projection at a common nonzero regulator.

    ``resonant_outgoing_momenta`` must sum to ``2i``.  The first outgoing
    momentum is shifted by minus the supplied soft momentum so that the five
    physical states plus the sixth soft singlet remain neutral.
    """

    return evaluate_regulated_klt_projections(
        (projection,),
        resonant_outgoing_momenta,
        soft_momentum=soft_momentum,
        twist_regulator=twist_regulator,
        twist_direction=twist_direction,
        rtol=rtol,
        atol=atol,
        rule=rule,
        max_subdivisions=max_subdivisions,
    )[0]


def evaluate_regulated_klt_projections(
    projections: Sequence[FivePointChannelProjection],
    resonant_outgoing_momenta: Sequence[complex],
    *,
    soft_momentum: complex,
    twist_regulator: complex,
    twist_direction: Mapping[
        tuple[int, int], complex
    ] = DEFAULT_TWIST_REGULATOR_DIRECTION,
    rtol: float = 2.0e-4,
    atol: float = 2.0e-7,
    rule: str = "gk15",
    max_subdivisions: int = 10_000,
) -> tuple[RegulatedFivePointKLTResult, ...]:
    """Evaluate several projections while reusing their common left periods.

    Right-moving components are compiled in small bounded batches.  Very
    large SymPy vector expressions otherwise exceed Python's compiler
    recursion depth long before numerical cubature begins.
    """

    selected = tuple(projections)
    if not selected:
        raise ValueError("at least one channel projection is required")
    resonant = tuple(complex(value) for value in resonant_outgoing_momenta)
    if len(resonant) != 4 or abs(sum(resonant) - 2j) > 1.0e-10:
        raise ValueError("four resonant outgoing momenta with sum 2i are required")
    soft = complex(soft_momentum)
    outgoing = (resonant[0] - soft,) + resonant[1:]
    common_states = soft_regulated_worldsheet_states(selected[0], outgoing, soft)
    twist = regulated_twist_exponents(
        common_states,
        soft,
        twist_regulator,
        direction=twist_direction,
    )

    left_diagnostics: list[OrderedSimplexIntegral] = []
    for word in KLT_WORDS:
        left_diagnostics.append(
            _integrate_period(
                positive_period_terms(
                    common_states,
                    soft,
                    word,
                    "holomorphic",
                    exponents=twist,
                ),
                rtol=rtol,
                atol=atol,
                rule=rule,
                max_subdivisions=max_subdivisions,
            )
        )

    states_by_projection = []
    for projection in selected:
        states = soft_regulated_worldsheet_states(projection, outgoing, soft)
        if any(
            abs(value - twist[pair]) > 1.0e-12
            for pair, value in regulated_twist_exponents(
                states,
                soft,
                twist_regulator,
                direction=twist_direction,
            ).items()
        ):
            raise AssertionError("S/V labels unexpectedly changed the bosonic twist")
        states_by_projection.append(states)

    # A singleton call keeps the simpler scalar path (and its more detailed
    # diagnostics).  For an S/V scan, each of the six antiholomorphic periods
    # is instead evaluated as one vector.  The expensive meromorphic IBP and
    # cubature nodes are then shared by every tensor projection.
    right_values: list[list[complex]] = [[] for _ in selected]
    right_errors_by_projection: list[list[float]] = [[] for _ in selected]
    right_converged = True
    right_cubature_calls = 0
    right_subdivisions = 0
    if len(selected) == 1:
        right_diagnostics: list[OrderedSimplexIntegral] = []
        for word in KLT_WORDS:
            diagnostic = _integrate_period(
                negative_period_terms(
                    states_by_projection[0],
                    soft,
                    word,
                    "antiholomorphic",
                    exponents=twist,
                ),
                rtol=rtol,
                atol=atol,
                rule=rule,
                max_subdivisions=max_subdivisions,
            )
            right_diagnostics.append(diagnostic)
            right_values[0].append(diagnostic.value)
            right_errors_by_projection[0].append(diagnostic.absolute_error)
        right_converged = all(item.converged for item in right_diagnostics)
        right_cubature_calls = sum(item.cubature_calls for item in right_diagnostics)
        right_subdivisions = sum(item.subdivisions for item in right_diagnostics)
    else:
        right_vector_diagnostics: list[OrderedSimplexVectorIntegral] = []
        for word in KLT_WORDS:
            word_values: list[complex] = []
            word_errors: list[float] = []
            for batch_start in range(0, len(states_by_projection), 4):
                batch = states_by_projection[batch_start : batch_start + 4]
                diagnostic = _integrate_period_components(
                    tuple(
                        negative_period_terms(
                            states,
                            soft,
                            word,
                            "antiholomorphic",
                            exponents=twist,
                        )
                        for states in batch
                    ),
                    rtol=rtol,
                    atol=atol,
                    rule=rule,
                    max_subdivisions=max_subdivisions,
                )
                right_vector_diagnostics.append(diagnostic)
                word_values.extend(diagnostic.values)
                word_errors.extend(diagnostic.absolute_errors)
            for index, (value, error) in enumerate(zip(word_values, word_errors)):
                right_values[index].append(value)
                right_errors_by_projection[index].append(error)
        right_converged = all(item.converged for item in right_vector_diagnostics)
        right_cubature_calls = sum(
            item.cubature_calls for item in right_vector_diagnostics
        )
        right_subdivisions = sum(
            item.subdivisions for item in right_vector_diagnostics
        )

    left = tuple(item.value for item in left_diagnostics)
    left_errors = tuple(item.absolute_error for item in left_diagnostics)
    kernel = klt_sine_kernel(twist)
    kernel_array = np.asarray(kernel, dtype=np.complex128)
    condition = float(np.linalg.cond(kernel_array))
    results: list[RegulatedFivePointKLTResult] = []
    for projection_index, projection in enumerate(selected):
        right = tuple(right_values[projection_index])
        right_errors = tuple(right_errors_by_projection[projection_index])
        sixpoint = contract_klt_periods(left, right, kernel) / 16.0

        # Conservative first-order propagation of period integration errors.
        contraction_error = 0.0
        for sigma in range(6):
            for gamma in range(6):
                weight = abs(kernel[gamma][sigma])
                contraction_error += weight * (
                    abs(right[gamma]) * left_errors[sigma]
                    + abs(left[sigma]) * right_errors[gamma]
                    + left_errors[sigma] * right_errors[gamma]
                )
        sixpoint_error = contraction_error / 16.0
        results.append(
            RegulatedFivePointKLTResult(
                projection_label=projection.label,
                soft_momentum=soft,
                twist_regulator=complex(twist_regulator),
                sixpoint_value=sixpoint,
                wall_value=2.0 * sixpoint,
                absolute_error_bound=2.0 * sixpoint_error,
                left_periods=left,
                right_periods=right,
                left_errors=left_errors,
                right_errors=right_errors,
                kernel_condition_number=condition,
                cubature_calls=(
                    sum(item.cubature_calls for item in left_diagnostics)
                    + right_cubature_calls
                ),
                subdivisions=(
                    sum(item.subdivisions for item in left_diagnostics)
                    + right_subdivisions
                ),
                periods_converged=(
                    all(item.converged for item in left_diagnostics)
                    and right_converged
                ),
            )
        )
    return tuple(results)


def fit_laurent_series(
    radii: Sequence[float],
    values: Sequence[complex],
    *,
    powers: Sequence[int] = (-2, -1, 0, 1, 2),
) -> LaurentFit:
    """Fit a complex Laurent series and return its coefficient at power zero."""

    x = np.asarray(tuple(float(value) for value in radii), dtype=np.float64)
    y = np.asarray(tuple(complex(value) for value in values), dtype=np.complex128)
    selected_powers = tuple(int(power) for power in powers)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("radii and values must be equal-length one-dimensional data")
    if len(x) < len(selected_powers) or any(value <= 0 for value in x):
        raise ValueError("positive radii must provide at least one row per Laurent power")
    if len(set(selected_powers)) != len(selected_powers) or 0 not in selected_powers:
        raise ValueError("Laurent powers must be distinct and include zero")
    matrix = np.column_stack([x**power for power in selected_powers])
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = matrix @ coefficients - y
    return LaurentFit(
        powers=selected_powers,
        coefficients=tuple(complex(value) for value in coefficients),
        finite_part=complex(coefficients[selected_powers.index(0)]),
        residual_norm=float(np.linalg.norm(residual)),
        condition_number=float(np.linalg.cond(matrix)),
    )


__all__ = [
    "LaurentFit",
    "RegulatedFivePointKLTResult",
    "evaluate_regulated_klt_projection",
    "evaluate_regulated_klt_projections",
    "fit_laurent_series",
    "s1_delta_normalized_value",
    "s1_outgoing_leg_product",
]
