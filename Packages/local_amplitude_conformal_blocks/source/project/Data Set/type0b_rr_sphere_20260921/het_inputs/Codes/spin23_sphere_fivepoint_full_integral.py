#!/usr/bin/env python3
r"""Full reduced sphere-five-point integral over the six-chart radial atlas.

The global coordinates ``z2,z3`` range over two copies of the complex plane.
Each plane is compactified with

``z=s*sqrt(u/(1-u))*exp(2*pi*i*v)``

and its exact area Jacobian.  Every point is then transported to the unique
radial comb chart selected by :mod:`spin23_sphere_fivepoint_atlas`.  At a
fixed tensor Gauss--Legendre momentum node, conformal-block workspaces are
reused over the entire randomized-QMC batch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc

from spin23_sphere_fivepoint_amplitude import (
    FivePointFixedMomentumWorkspace,
    SphereFivePointState,
    build_fivepoint_fixed_momentum_workspace,
    validate_fivepoint_states,
)
from spin23_sphere_fivepoint_atlas import (
    ChartPermutation,
    RadialChartPoint,
    global_to_radial_chart,
)


@dataclass(frozen=True)
class CompactifiedModuliSample:
    """One global QMC point and its owner comb coordinates."""

    chart: RadialChartPoint
    compactification_jacobian: float

    @property
    def total_measure_jacobian(self) -> float:
        return (
            self.compactification_jacobian * self.chart.measure_jacobian
        )


@dataclass(frozen=True)
class FullSphereSampleBatch:
    """Nested Sobol prefixes for independent scrambled replicates."""

    samples: tuple[tuple[CompactifiedModuliSample, ...], ...]
    sample_powers: tuple[int, ...]
    radial_scale: float
    base_seed: int

    @property
    def maximum_sample_count(self) -> int:
        return 2 ** self.sample_powers[-1]

    @property
    def replicates(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class FullSphereSpectralNodePartial:
    """Weighted contribution of one internal-momentum quadrature node."""

    values: np.ndarray
    descendant_increments: np.ndarray | None
    maximum_gram_condition: float
    maximum_pco_cancellation_ratio: float
    momentum: tuple[float, float]
    momentum_weight: float


@dataclass(frozen=True)
class FullSphereIntegralDiagnostics:
    """Reduced integral estimates for nested QMC sample counts."""

    values: tuple[complex, ...]
    standard_errors: tuple[float, ...]
    relative_standard_errors: tuple[float, ...]
    descendant_increments: tuple[complex, ...] | None
    descendant_increment_relative_sizes: tuple[float, ...] | None
    sample_powers: tuple[int, ...]
    replicates: int
    spectral_order: int
    spectral_p_max: float
    radial_scale: float
    maximum_gram_condition: float
    maximum_pco_cancellation_ratio: float


def _validate_sample_powers(values: Sequence[int]) -> tuple[int, ...]:
    powers = tuple(int(value) for value in values)
    if not powers or tuple(sorted(set(powers))) != powers:
        raise ValueError("sample_powers must be a strictly increasing sequence")
    if powers[0] < 0:
        raise ValueError("sample powers must be nonnegative")
    return powers


def generate_full_sphere_samples(
    *,
    sample_powers: Sequence[int] = (6, 8),
    replicates: int = 4,
    radial_scale: float = 1.0,
    base_seed: int = 230824,
) -> FullSphereSampleBatch:
    """Generate deterministic compactified-plane samples and owner charts."""

    powers = _validate_sample_powers(sample_powers)
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("replicates must be an integer")
    if replicates < 2:
        raise ValueError("at least two independent replicates are required")
    if not math.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError("radial_scale must be finite and positive")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")

    batches: list[tuple[CompactifiedModuliSample, ...]] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=4, scramble=True, seed=base_seed + replicate)
        points = sampler.random_base2(powers[-1])
        radii_2 = radial_scale * np.sqrt(points[:, 0] / (1.0 - points[:, 0]))
        radii_3 = radial_scale * np.sqrt(points[:, 2] / (1.0 - points[:, 2]))
        z_2 = radii_2 * np.exp(2j * math.pi * points[:, 1])
        z_3 = radii_3 * np.exp(2j * math.pi * points[:, 3])
        jacobians = (
            math.pi**2
            * radial_scale**4
            / ((1.0 - points[:, 0]) ** 2 * (1.0 - points[:, 2]) ** 2)
        )
        batches.append(
            tuple(
                CompactifiedModuliSample(
                    chart=global_to_radial_chart(first, second),
                    compactification_jacobian=float(jacobian),
                )
                for first, second, jacobian in zip(z_2, z_3, jacobians)
            )
        )
    return FullSphereSampleBatch(
        samples=tuple(batches),
        sample_powers=powers,
        radial_scale=float(radial_scale),
        base_seed=base_seed,
    )


def _workspace_key(
    states: Sequence[SphereFivePointState],
) -> tuple[tuple[str, complex, complex, int | None], ...]:
    return tuple(
        (
            state.kind,
            complex(state.liouville_momentum),
            complex(state.time_momentum),
            state.flavor,
        )
        for state in states
    )


def evaluate_full_sphere_spectral_node(
    states: Sequence[SphereFivePointState],
    momenta: Sequence[float],
    momentum_weight: float,
    samples: FullSphereSampleBatch,
    *,
    maximum_twice_levels: int | Sequence[int],
    lower_maximum_twice_levels: int | Sequence[int] | None = None,
    structure_precision: int = 32,
    block_digits: int = 40,
    condition_limit: float = 1.0e13,
) -> FullSphereSpectralNodePartial:
    """Evaluate one weighted spectral node over every QMC replicate/prefix."""

    normalized = validate_fivepoint_states(states)
    internal = tuple(float(value) for value in momenta)
    if len(internal) != 2:
        raise ValueError("two internal momenta are required")
    weight = float(momentum_weight)
    if not math.isfinite(weight):
        raise ValueError("momentum_weight must be finite")

    permutations = {
        sample.chart.permutation
        for replicate_samples in samples.samples
        for sample in replicate_samples
    }
    workspaces_by_state: dict[
        tuple[tuple[str, complex, complex, int | None], ...],
        FivePointFixedMomentumWorkspace,
    ] = {}
    workspaces: dict[ChartPermutation, FivePointFixedMomentumWorkspace] = {}
    increment_workspaces: dict[
        ChartPermutation, FivePointFixedMomentumWorkspace
    ] = {}
    maximum_condition = 1.0
    for permutation in permutations:
        ordered_states = tuple(normalized[index] for index in permutation)
        key = _workspace_key(ordered_states)
        if key not in workspaces_by_state:
            workspaces_by_state[key] = build_fivepoint_fixed_momentum_workspace(
                ordered_states,
                internal,
                maximum_twice_levels=maximum_twice_levels,
                structure_precision=structure_precision,
                block_digits=block_digits,
                condition_limit=condition_limit,
            )
        workspace = workspaces_by_state[key]
        workspaces[permutation] = workspace
        maximum_condition = max(
            maximum_condition, workspace.maximum_gram_condition
        )
        if lower_maximum_twice_levels is not None:
            increment_workspaces[permutation] = replace(
                workspace,
                subtract_maximum_twice_levels=(
                    int(lower_maximum_twice_levels)
                    if isinstance(lower_maximum_twice_levels, int)
                    else tuple(int(value) for value in lower_maximum_twice_levels)
                ),
            )

    shape = (len(samples.sample_powers), samples.replicates)
    values = np.zeros(shape, dtype=np.complex128)
    increments = (
        np.zeros(shape, dtype=np.complex128)
        if lower_maximum_twice_levels is not None
        else None
    )
    maximum_cancellation = 1.0
    prefix_counts = tuple(2**power for power in samples.sample_powers)
    for replicate, replicate_samples in enumerate(samples.samples):
        high_density = np.empty(samples.maximum_sample_count, dtype=np.complex128)
        increment_density = (
            np.empty(samples.maximum_sample_count, dtype=np.complex128)
            if increments is not None
            else None
        )
        for index, sample in enumerate(replicate_samples):
            chart = sample.chart
            evaluation = workspaces[chart.permutation].evaluate(
                chart.chart_z_2, chart.chart_z_3
            )
            maximum_cancellation = max(
                maximum_cancellation, evaluation.cancellation_ratio
            )
            high_density[index] = (
                sample.total_measure_jacobian * evaluation.value
            )
            if increment_density is not None:
                increment = increment_workspaces[chart.permutation].evaluate(
                    chart.chart_z_2, chart.chart_z_3
                )
                increment_density[index] = (
                    sample.total_measure_jacobian * increment.value
                )
        high_prefix = np.cumsum(high_density, dtype=np.complex128)
        increment_prefix = (
            None
            if increment_density is None
            else np.cumsum(increment_density, dtype=np.complex128)
        )
        for power_index, count in enumerate(prefix_counts):
            values[power_index, replicate] = (
                weight * high_prefix[count - 1] / count
            )
            if increments is not None and increment_prefix is not None:
                increments[power_index, replicate] = (
                    weight * increment_prefix[count - 1] / count
                )
    return FullSphereSpectralNodePartial(
        values=values,
        descendant_increments=increments,
        maximum_gram_condition=maximum_condition,
        maximum_pco_cancellation_ratio=maximum_cancellation,
        momentum=internal,  # type: ignore[arg-type]
        momentum_weight=weight,
    )


def tensor_legendre_momentum_rule(
    order: int,
    p_max: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return ``(P1,P2,w1*w2)`` for ``dP1/pi dP2/pi``."""

    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if not math.isfinite(p_max) or p_max <= 0:
        raise ValueError("p_max must be finite and positive")
    raw_nodes, raw_weights = leggauss(order)
    momenta = 0.5 * p_max * (raw_nodes + 1.0)
    weights = 0.5 * p_max * raw_weights / math.pi
    return tuple(
        (float(first), float(second), float(weights[i] * weights[j]))
        for i, first in enumerate(momenta)
        for j, second in enumerate(momenta)
    )


def _summarize_replicates(values: np.ndarray) -> tuple[complex, float, float]:
    mean = complex(np.mean(values))
    error = math.hypot(
        float(np.std(values.real, ddof=1) / math.sqrt(len(values))),
        float(np.std(values.imag, ddof=1) / math.sqrt(len(values))),
    )
    return mean, error, error / max(abs(mean), 1.0e-300)


def integrate_full_sphere_fivepoint(
    states: Sequence[SphereFivePointState],
    *,
    maximum_twice_levels: int | Sequence[int] = 6,
    lower_maximum_twice_levels: int | Sequence[int] | None = 4,
    spectral_order: int = 9,
    spectral_p_max: float = 3.0,
    sample_powers: Sequence[int] = (5, 7),
    replicates: int = 4,
    radial_scale: float = 1.0,
    base_seed: int = 230824,
    structure_precision: int = 32,
    block_digits: int = 40,
    condition_limit: float = 1.0e13,
) -> FullSphereIntegralDiagnostics:
    """Evaluate the complete six-chart reduced amplitude without sharding."""

    batch = generate_full_sphere_samples(
        sample_powers=sample_powers,
        replicates=replicates,
        radial_scale=radial_scale,
        base_seed=base_seed,
    )
    total = np.zeros(
        (len(batch.sample_powers), batch.replicates), dtype=np.complex128
    )
    total_increment = (
        np.zeros_like(total)
        if lower_maximum_twice_levels is not None
        else None
    )
    maximum_condition = 1.0
    maximum_cancellation = 1.0
    for momentum1, momentum2, weight in tensor_legendre_momentum_rule(
        spectral_order, spectral_p_max
    ):
        partial = evaluate_full_sphere_spectral_node(
            states,
            (momentum1, momentum2),
            weight,
            batch,
            maximum_twice_levels=maximum_twice_levels,
            lower_maximum_twice_levels=lower_maximum_twice_levels,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
        )
        total += partial.values
        if total_increment is not None:
            if partial.descendant_increments is None:
                raise AssertionError("missing descendant increment")
            total_increment += partial.descendant_increments
        maximum_condition = max(
            maximum_condition, partial.maximum_gram_condition
        )
        maximum_cancellation = max(
            maximum_cancellation,
            partial.maximum_pco_cancellation_ratio,
        )

    summaries = tuple(_summarize_replicates(row) for row in total)
    values = tuple(summary[0] for summary in summaries)
    errors = tuple(summary[1] for summary in summaries)
    relative_errors = tuple(summary[2] for summary in summaries)
    increment_values = None
    increment_relative_sizes = None
    if total_increment is not None:
        increment_values = tuple(
            complex(np.mean(row)) for row in total_increment
        )
        increment_relative_sizes = tuple(
            abs(increment) / max(abs(value), 1.0e-300)
            for increment, value in zip(increment_values, values)
        )
    return FullSphereIntegralDiagnostics(
        values=values,
        standard_errors=errors,
        relative_standard_errors=relative_errors,
        descendant_increments=increment_values,
        descendant_increment_relative_sizes=increment_relative_sizes,
        sample_powers=batch.sample_powers,
        replicates=batch.replicates,
        spectral_order=spectral_order,
        spectral_p_max=spectral_p_max,
        radial_scale=radial_scale,
        maximum_gram_condition=maximum_condition,
        maximum_pco_cancellation_ratio=maximum_cancellation,
    )


__all__ = [
    "CompactifiedModuliSample",
    "FullSphereIntegralDiagnostics",
    "FullSphereSampleBatch",
    "FullSphereSpectralNodePartial",
    "evaluate_full_sphere_spectral_node",
    "generate_full_sphere_samples",
    "integrate_full_sphere_fivepoint",
    "tensor_legendre_momentum_rule",
]
