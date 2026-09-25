#!/usr/bin/env python3
r"""Nonchiral super-Liouville spectral integrals in a torus necklace frame.

The delta-normalized continuum resolution of the identity used here is

.. math::

   \int_0^\infty \frac{dP}{\pi}\,|P\rangle\langle P|.

Consequently an ``n``-edge necklace carries ``n`` independent factors
``dP_e / pi``.  The numerical integral is truncated to
``0 <= P_e <= p_max``.  The module provides both independent
Gauss--Legendre rules and embedded Gauss--Kronrod 3/7 and 7/15 rules.  It also
uses the exact plumbing Gaussian to provide generalized Gauss--Laguerre rules
on the full half-line, including a component-adaptive order sequence.  In the
embedded rules, the quadrature diagnostic reuses every lower-order
evaluation, while the cutoff diagnostic integrates only the disjoint added
momentum boxes.  The returned diagnostics keep numerical effects separate;
none of the reported differences is a rigorous error bound.

On POSIX systems, parallel Laguerre evaluation retains one fork pool across
all related quadrature orders.  This preserves process-local conformal-block
caches without changing nodes, weights, or deterministic reduction order.
For a requested block-truncation difference, the current and lower
rectangular truncations are read from the same coefficient table before they
are subtracted.  This is an exact multilevel evaluation, not a surrogate for
either block.

The fixed-momentum NS assembly sums the diagonal nonchiral pairing of the
even and odd NS three-point forms.  The Ramond assembly follows the BRY/HJS
pairing checked by the trusted Type-0B one- and two-point implementations:
compatible HJS sign assignments obey ``product(signs)=temporal_lift_sign``
and carry one overall factor of two from the closed long-R ground fiber.

No target-space, free-fermion, ghost, GSO, moduli-measure, or string-coupling
factor is included in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
import math
import multiprocessing as mp
import os
import time
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.special import roots_genlaguerre

from ns_algebra.ns_sca import Word, fermion_parity
from spin23_genus1_blocks import (
    TorusNecklaceBlockSeries,
    b1_ramond_liouville_necklace_series,
    direct_ns_torus_necklace_series,
)
from spin23_genus1_recursive_blocks import (
    G_MINUS_HALF as RECURSIVE_G_MINUS_HALF,
    RecursiveTorusBlockResult,
    recursive_ns_torus_two_point_series,
    recursive_ramond_torus_two_point_series,
)
from spin23_ns_fast import (
    direct_fast_b1_ns_necklace_series,
    generated_ns_rectangle_available,
    prewarm_direct_fast_b1_ns,
)
from spin23_ramond_fast import (
    GENERATED_MAX_TWICE_LEVEL,
    direct_fast_b1_ramond_necklace_series,
    direct_fast_b1_ramond_two_sign_series,
    generated_direct_rectangle_available,
    prewarm_direct_fast_b1_ramond,
)
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_structure_constants,
)


MomentumIntegrand = Callable[[tuple[float, ...]], complex]
MomentumVectorIntegrand = Callable[
    [tuple[float, ...]],
    tuple[complex, ...],
]
MomentumEvaluation = complex | tuple[complex, ...]
BlockBackend = Literal[
    "direct",
    "direct_fast",
    "recursion",
    "hybrid",
    "auto",
]


# The direct/recursive crossover is a measured policy parameter, not a
# statement about formal feature support.  It is capped by the statically
# generated range so ``auto`` never triggers runtime symbolic compilation.
_FAST_DIRECT_MAX_TWICE_LEVEL = min(8, GENERATED_MAX_TWICE_LEVEL)


@dataclass(frozen=True)
class SpectralIntegralDiagnostics:
    """Value and independent numerical diagnostics for one spectral integral."""

    value: complex
    refined_value: complex | None
    extended_value: complex | None
    quadrature_absolute_error: float | None
    tail_absolute_error: float | None
    estimated_absolute_error: float | None
    dimension: int
    p_max: float | None
    quadrature_order: int
    refined_order: int | None
    extended_p_max: float | None
    function_evaluations: int
    quadrature_method: str = "gauss_legendre"
    coarse_value: complex | None = None
    coarse_order: int | None = None
    target_relative_tolerance: float | None = None
    converged: bool | None = None
    orders_evaluated: tuple[int, ...] = ()
    endpoint_powers: tuple[int, ...] = ()
    block_backend_requested: BlockBackend | None = None
    recursive_block_evaluations: int = 0
    direct_block_evaluations: int = 0
    fast_direct_block_evaluations: int = 0
    recursion_fallbacks: int = 0
    worker_processes: int = 1
    runtime_seconds: float | None = None

    @property
    def estimated_relative_error(self) -> float | None:
        """Return the conservative error estimate divided by ``abs(value)``."""

        if self.estimated_absolute_error is None:
            return None
        if self.value == 0:
            return math.inf if self.estimated_absolute_error else 0.0
        return self.estimated_absolute_error / abs(self.value)


@dataclass
class _BlockBackendStats:
    """Mutable counters local to one spectral integral."""

    recursive_block_evaluations: int = 0
    direct_block_evaluations: int = 0
    fast_direct_block_evaluations: int = 0
    recursion_fallbacks: int = 0


# These globals are populated immediately before creating a POSIX ``fork``
# pool. Forked workers inherit the otherwise unpicklable local integrand
# closure without serializing it for every quadrature node. They are reset as
# soon as the pool exits, and spectral integrations are not nested.
_FORK_MOMENTUM_INTEGRAND: (
    Callable[[tuple[float, ...]], MomentumEvaluation] | None
) = None
_FORK_BACKEND_STATS: tuple[_BlockBackendStats, ...] = ()


def _validate_worker_processes(workers: int) -> int:
    """Validate a positive worker count against the active Slurm task."""

    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError("workers must be an integer")
    if workers < 1:
        raise ValueError("workers must be positive")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None and workers > int(slurm_cpus):
        raise ValueError(
            f"spectral workers ({workers}) exceed the Slurm cpus-per-task "
            f"allocation ({slurm_cpus})"
        )
    return workers


def _backend_stat_tuple(
    stats: _BlockBackendStats | None,
) -> tuple[int, int, int, int]:
    if stats is None:
        return (0, 0, 0, 0)
    return (
        stats.recursive_block_evaluations,
        stats.direct_block_evaluations,
        stats.fast_direct_block_evaluations,
        stats.recursion_fallbacks,
    )


def _fork_momentum_worker(
    momenta: tuple[float, ...],
) -> tuple[MomentumEvaluation, tuple[tuple[int, int, int, int], ...]]:
    """Evaluate one inherited momentum integrand and return counter deltas."""

    if _FORK_MOMENTUM_INTEGRAND is None:
        raise RuntimeError("the forked momentum integrand was not initialized")
    before = tuple(_backend_stat_tuple(stats) for stats in _FORK_BACKEND_STATS)
    raw_value = _FORK_MOMENTUM_INTEGRAND(momenta)
    value: MomentumEvaluation = (
        tuple(complex(entry) for entry in raw_value)
        if isinstance(raw_value, tuple)
        else complex(raw_value)
    )
    after = tuple(_backend_stat_tuple(stats) for stats in _FORK_BACKEND_STATS)
    deltas = tuple(
        tuple(right - left for left, right in zip(before_row, after_row))
        for before_row, after_row in zip(before, after)
    )
    return value, deltas


class _ForkMomentumBatchExecutor:
    """Keep one fork pool alive across related quadrature orders.

    The conformal-block backends build substantial process-local caches.
    Recreating the pool for every independent Laguerre rule discards those
    caches even though all rules evaluate the same integrand.  This executor
    fixes the inherited closure once, then reuses the workers without changing
    any quadrature node or summation weight.
    """

    def __init__(
        self,
        integrand: Callable[[tuple[float, ...]], MomentumEvaluation],
        *,
        workers: int,
        maximum_tasks: int,
        backend_stats: (
            _BlockBackendStats | Sequence[_BlockBackendStats] | None
        ) = None,
    ) -> None:
        """Prepare a serial executor or one bounded persistent fork pool."""

        if not isinstance(maximum_tasks, int) or maximum_tasks < 1:
            raise ValueError("maximum_tasks must be a positive integer")
        self.integrand = integrand
        self.worker_count = min(
            _validate_worker_processes(workers),
            maximum_tasks,
        )
        if backend_stats is None:
            self.backend_stats = ()
        elif isinstance(backend_stats, _BlockBackendStats):
            self.backend_stats = (backend_stats,)
        else:
            self.backend_stats = tuple(backend_stats)
        self.pool = None

    def __enter__(self) -> "_ForkMomentumBatchExecutor":
        """Start the pool once and install the inherited integrand closure."""

        if self.worker_count <= 1:
            return self
        if os.name != "posix":
            raise RuntimeError(
                "parallel spectral integration currently requires POSIX fork"
            )

        global _FORK_MOMENTUM_INTEGRAND, _FORK_BACKEND_STATS
        if _FORK_MOMENTUM_INTEGRAND is not None:
            raise RuntimeError("nested parallel spectral integration is unsupported")
        _FORK_MOMENTUM_INTEGRAND = self.integrand
        _FORK_BACKEND_STATS = self.backend_stats
        try:
            self.pool = mp.get_context("fork").Pool(
                processes=self.worker_count
            )
        except BaseException:
            _FORK_MOMENTUM_INTEGRAND = None
            _FORK_BACKEND_STATS = ()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Close or terminate workers and clear inherited global state."""

        global _FORK_MOMENTUM_INTEGRAND, _FORK_BACKEND_STATS
        try:
            if self.pool is not None:
                if exception_type is None:
                    self.pool.close()
                else:
                    self.pool.terminate()
                self.pool.join()
        finally:
            self.pool = None
            _FORK_MOMENTUM_INTEGRAND = None
            _FORK_BACKEND_STATS = ()

    def evaluate(
        self,
        momenta: Sequence[tuple[float, ...]],
    ) -> tuple[MomentumEvaluation, ...]:
        """Evaluate one batch while retaining process-local block caches."""

        if not momenta:
            return ()
        if self.pool is None:
            evaluated = []
            for point in momenta:
                raw_value = self.integrand(point)
                evaluated.append(
                    tuple(complex(entry) for entry in raw_value)
                    if isinstance(raw_value, tuple)
                    else complex(raw_value)
                )
            return tuple(evaluated)

        chunksize = max(1, len(momenta) // (4 * self.worker_count))
        evaluated = self.pool.map(
            _fork_momentum_worker,
            momenta,
            chunksize=chunksize,
        )
        if self.backend_stats:
            for _, deltas in evaluated:
                for stats, delta in zip(self.backend_stats, deltas):
                    stats.recursive_block_evaluations += delta[0]
                    stats.direct_block_evaluations += delta[1]
                    stats.fast_direct_block_evaluations += delta[2]
                    stats.recursion_fallbacks += delta[3]
        return tuple(value for value, _ in evaluated)


def _evaluate_momentum_batch(
    integrand: MomentumIntegrand,
    momenta: Sequence[tuple[float, ...]],
    *,
    workers: int,
    backend_stats: _BlockBackendStats | None = None,
) -> tuple[complex, ...]:
    """Evaluate independent momentum nodes serially or in a fork pool."""

    if not momenta:
        return ()
    with _ForkMomentumBatchExecutor(
        integrand,
        workers=workers,
        maximum_tasks=len(momenta),
        backend_stats=backend_stats,
    ) as executor:
        evaluated = executor.evaluate(momenta)
    if any(isinstance(value, tuple) for value in evaluated):
        raise AssertionError("scalar momentum integrand returned a vector")
    return tuple(complex(value) for value in evaluated)


def gauss_legendre_spectral_rule(
    p_max: float,
    quadrature_order: int,
) -> tuple[tuple[float, float], ...]:
    r"""Return ``(P, weight)`` nodes for :math:`\int_0^{P_{\max}}dP/\pi`."""

    p_max = float(p_max)
    if not math.isfinite(p_max) or p_max <= 0:
        raise ValueError("p_max must be finite and positive")
    if not isinstance(quadrature_order, int):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    midpoint = 0.5 * p_max
    return tuple(
        (
            midpoint * (float(node) + 1.0),
            midpoint * float(weight) / math.pi,
        )
        for node, weight in zip(nodes, weights)
    )


_G3_ABSCISSA = math.sqrt(3.0 / 5.0)
_GAUSS_3_RULE = (
    (-_G3_ABSCISSA, 5.0 / 9.0),
    (0.0, 8.0 / 9.0),
    (_G3_ABSCISSA, 5.0 / 9.0),
)
_KRONROD_7_RULE = (
    (-0.9604912687080203, 0.10465622602646727),
    (-_G3_ABSCISSA, 0.26848808986833344),
    (-0.43424374934680254, 0.4013974147759622),
    (0.0, 0.45091653865847414),
    (0.43424374934680254, 0.4013974147759622),
    (_G3_ABSCISSA, 0.26848808986833344),
    (0.9604912687080203, 0.10465622602646727),
)
_GAUSS_7_RULE = (
    (-0.9491079123427585, 0.1294849661688697),
    (-0.7415311855993945, 0.2797053914892766),
    (-0.4058451513773972, 0.3818300505051189),
    (0.0, 0.4179591836734694),
    (0.4058451513773972, 0.3818300505051189),
    (0.7415311855993945, 0.2797053914892766),
    (0.9491079123427585, 0.1294849661688697),
)
_KRONROD_15_RULE = (
    (-0.9914553711208126, 0.02293532201052922),
    (-0.9491079123427585, 0.06309209262997855),
    (-0.8648644233597691, 0.1047900103222502),
    (-0.7415311855993945, 0.1406532597155259),
    (-0.5860872354676911, 0.1690047266392679),
    (-0.4058451513773972, 0.1903505780647854),
    (-0.2077849550078985, 0.2044329400752989),
    (0.0, 0.2094821410847278),
    (0.2077849550078985, 0.2044329400752989),
    (0.4058451513773972, 0.1903505780647854),
    (0.5860872354676911, 0.1690047266392679),
    (0.7415311855993945, 0.1406532597155259),
    (0.8648644233597691, 0.1047900103222502),
    (0.9491079123427585, 0.06309209262997855),
    (0.9914553711208126, 0.02293532201052922),
)


def _map_spectral_rule(
    rule: Sequence[tuple[float, float]],
    lower: float,
    upper: float,
) -> tuple[tuple[float, float], ...]:
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return tuple(
        (
            midpoint + half_width * abscissa,
            half_width * weight / math.pi,
        )
        for abscissa, weight in rule
    )


def _tensor_box_value(
    integrand: MomentumIntegrand,
    *,
    intervals: Sequence[tuple[float, float]],
    rule: Sequence[tuple[float, float]],
    cache: dict[tuple[float, ...], complex],
) -> complex:
    channels = tuple(
        _map_spectral_rule(rule, lower, upper)
        for lower, upper in intervals
    )
    total = 0.0j
    for selected in product(*channels):
        momenta = tuple(channel[0] for channel in selected)
        if momenta not in cache:
            value = complex(integrand(momenta))
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ArithmeticError(
                    "the spectral integrand returned a non-finite value"
                )
            cache[momenta] = value
        total += math.prod(channel[1] for channel in selected) * cache[momenta]
    return total


def _embedded_gauss_kronrod_spectral_integral(
    integrand: MomentumIntegrand,
    *,
    dimension: int,
    p_max: float,
    extended_p_max: float | None = None,
    gauss_rule: Sequence[tuple[float, float]],
    kronrod_rule: Sequence[tuple[float, float]],
    gauss_order: int,
    kronrod_order: int,
    method_name: str,
) -> SpectralIntegralDiagnostics:
    """Evaluate one tensor embedded rule and optional disjoint tail boxes."""

    if not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    p_max = float(p_max)
    if not math.isfinite(p_max) or p_max <= 0:
        raise ValueError("p_max must be finite and positive")
    if extended_p_max is not None:
        extended_p_max = float(extended_p_max)
        if not math.isfinite(extended_p_max) or extended_p_max <= p_max:
            raise ValueError("extended_p_max must be finite and exceed p_max")

    cache: dict[tuple[float, ...], complex] = {}
    core_intervals = ((0.0, p_max),) * dimension
    coarse = _tensor_box_value(
        integrand,
        intervals=core_intervals,
        rule=gauss_rule,
        cache=cache,
    )
    value = _tensor_box_value(
        integrand,
        intervals=core_intervals,
        rule=kronrod_rule,
        cache=cache,
    )
    quadrature_error = abs(value - coarse)

    extended_value: complex | None = None
    tail_error: float | None = None
    if extended_p_max is not None:
        tail = 0.0j
        for region in product((0, 1), repeat=dimension):
            if not any(region):
                continue
            intervals = tuple(
                (p_max, extended_p_max) if is_tail else (0.0, p_max)
                for is_tail in region
            )
            tail += _tensor_box_value(
                integrand,
                intervals=intervals,
                rule=gauss_rule,
                cache=cache,
            )
        extended_value = value + tail
        tail_error = abs(tail)

    available_errors = tuple(
        error for error in (quadrature_error, tail_error) if error is not None
    )
    return SpectralIntegralDiagnostics(
        value=value,
        refined_value=None,
        extended_value=extended_value,
        quadrature_absolute_error=quadrature_error,
        tail_absolute_error=tail_error,
        estimated_absolute_error=max(available_errors),
        dimension=dimension,
        p_max=p_max,
        quadrature_order=kronrod_order,
        refined_order=None,
        extended_p_max=extended_p_max,
        function_evaluations=len(cache),
        quadrature_method=method_name,
        coarse_value=coarse,
        coarse_order=gauss_order,
    )


def gauss_kronrod_3_7_spectral_integral(
    integrand: MomentumIntegrand,
    *,
    dimension: int,
    p_max: float,
    extended_p_max: float | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Evaluate the embedded tensor Gauss--Kronrod 3/7 rule.

    The selected value is the seven-node Kronrod rule on the core box.  The
    embedded three-node Gauss rule supplies the quadrature diagnostic without
    additional core evaluations.  Optional disjoint tail boxes use the
    three-node rule and do not move any core node.
    """

    return _embedded_gauss_kronrod_spectral_integral(
        integrand,
        dimension=dimension,
        p_max=p_max,
        extended_p_max=extended_p_max,
        gauss_rule=_GAUSS_3_RULE,
        kronrod_rule=_KRONROD_7_RULE,
        gauss_order=3,
        kronrod_order=7,
        method_name="gauss_kronrod_3_7",
    )


def gauss_kronrod_7_15_spectral_integral(
    integrand: MomentumIntegrand,
    *,
    dimension: int,
    p_max: float,
    extended_p_max: float | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Evaluate the embedded tensor Gauss--Kronrod 7/15 rule.

    The selected value is the fifteen-node Kronrod rule on the core box.  The
    embedded seven-node Gauss rule supplies the quadrature diagnostic without
    additional core evaluations.  Optional disjoint tail boxes use the
    seven-node rule and do not move any core node.
    """

    return _embedded_gauss_kronrod_spectral_integral(
        integrand,
        dimension=dimension,
        p_max=p_max,
        extended_p_max=extended_p_max,
        gauss_rule=_GAUSS_7_RULE,
        kronrod_rule=_KRONROD_15_RULE,
        gauss_order=7,
        kronrod_order=15,
        method_name="gauss_kronrod_7_15",
    )


def _tensor_generalized_laguerre_value(
    reduced_integrand: MomentumIntegrand,
    *,
    gaussian_scales: Sequence[float],
    quadrature_order: int,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: _BlockBackendStats | None = None,
    _batch_executor: _ForkMomentumBatchExecutor | None = None,
) -> tuple[complex, int]:
    r"""Integrate ``exp(-sum(a_e P_e**2)) * reduced_integrand(P)``.

    If the endpoint behavior is factored as ``F(P)=P**m*H(P)``, the exact
    substitution ``y=a*P**2`` gives generalized Laguerre parameter
    ``alpha=(m-1)/2`` and evaluates ``H=F/P**m``.  The default ``m=0`` is the
    ordinary half-line Gaussian measure.  The returned tensor rule covers
    every ``P_e`` in ``[0,infinity)`` without a momentum cutoff.
    """

    if not isinstance(quadrature_order, int):
        raise TypeError("quadrature_order must be an integer")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    scales = tuple(float(scale) for scale in gaussian_scales)
    if not scales or any(
        not math.isfinite(scale) or scale <= 0.0 for scale in scales
    ):
        raise ValueError("gaussian_scales must be finite and positive")

    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )
    if len(powers) != len(scales):
        raise ValueError("endpoint_powers must have one entry per momentum")
    if any(not isinstance(power, int) or power < 0 for power in powers):
        raise ValueError("endpoint powers must be nonnegative integers")

    channels = tuple(
        tuple(
            (
                math.sqrt(float(node) / scale),
                float(weight)
                / (2.0 * math.pi * scale ** ((power + 1.0) / 2.0)),
            )
            for node, weight in zip(
                *roots_genlaguerre(
                    quadrature_order,
                    (power - 1.0) / 2.0,
                )
            )
        )
        for scale, power in zip(scales, powers)
    )
    selected_nodes = tuple(product(*channels))
    momenta = tuple(
        tuple(entry[0] for entry in selected)
        for selected in selected_nodes
    )
    values = (
        _evaluate_momentum_batch(
            reduced_integrand,
            momenta,
            workers=workers,
            backend_stats=_backend_stats,
        )
        if _batch_executor is None
        else _batch_executor.evaluate(momenta)
    )
    total = 0.0j
    for selected, point, value in zip(selected_nodes, momenta, values):
        endpoint_factor = math.prod(
            momentum**power
            for momentum, power in zip(point, powers)
        )
        value /= endpoint_factor
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ArithmeticError(
                "the Gaussian-reduced spectral integrand returned a "
                "non-finite value"
            )
        total += math.prod(entry[1] for entry in selected) * value
    if not math.isfinite(total.real) or not math.isfinite(total.imag):
        raise ArithmeticError("the spectral integral returned a non-finite value")
    return total, len(momenta)


def _tensor_generalized_laguerre_values(
    reduced_integrand: MomentumVectorIntegrand,
    *,
    output_size: int,
    gaussian_scales: Sequence[float],
    quadrature_order: int,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: Sequence[_BlockBackendStats] = (),
    _batch_executor: _ForkMomentumBatchExecutor | None = None,
) -> tuple[tuple[complex, ...], int]:
    """Vector-valued counterpart of the generalized Laguerre tensor rule."""

    if not isinstance(output_size, int) or output_size < 1:
        raise ValueError("output_size must be a positive integer")
    if not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer of at least two")
    scales = tuple(float(scale) for scale in gaussian_scales)
    if not scales or any(
        not math.isfinite(scale) or scale <= 0.0 for scale in scales
    ):
        raise ValueError("gaussian_scales must be finite and positive")
    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )
    if len(powers) != len(scales):
        raise ValueError("endpoint_powers must have one entry per momentum")
    if any(not isinstance(power, int) or power < 0 for power in powers):
        raise ValueError("endpoint powers must be nonnegative integers")

    channels = tuple(
        tuple(
            (
                math.sqrt(float(node) / scale),
                float(weight)
                / (2.0 * math.pi * scale ** ((power + 1.0) / 2.0)),
            )
            for node, weight in zip(
                *roots_genlaguerre(
                    quadrature_order,
                    (power - 1.0) / 2.0,
                )
            )
        )
        for scale, power in zip(scales, powers)
    )
    selected_nodes = tuple(product(*channels))
    momenta = tuple(
        tuple(entry[0] for entry in selected)
        for selected in selected_nodes
    )
    evaluated = (
        _batch_executor.evaluate(momenta)
        if _batch_executor is not None
        else _evaluate_momentum_vector_batch(
            reduced_integrand,
            momenta,
            workers=workers,
            backend_stats=_backend_stats,
        )
    )
    totals = np.zeros(output_size, dtype=np.complex128)
    for selected, point, value in zip(selected_nodes, momenta, evaluated):
        if not isinstance(value, tuple) or len(value) != output_size:
            raise ValueError(
                "the vector momentum integrand returned the wrong output size"
            )
        endpoint_factor = math.prod(
            momentum**power for momentum, power in zip(point, powers)
        )
        normalized = np.asarray(value, dtype=np.complex128) / endpoint_factor
        if not np.all(np.isfinite(normalized)):
            raise ArithmeticError(
                "the Gaussian-reduced spectral integrand returned a "
                "non-finite value"
            )
        totals += math.prod(entry[1] for entry in selected) * normalized
    if not np.all(np.isfinite(totals)):
        raise ArithmeticError("the spectral integral returned a non-finite value")
    return tuple(complex(value) for value in totals), len(momenta)


def _evaluate_momentum_vector_batch(
    integrand: MomentumVectorIntegrand,
    momenta: Sequence[tuple[float, ...]],
    *,
    workers: int,
    backend_stats: Sequence[_BlockBackendStats] = (),
) -> tuple[tuple[complex, ...], ...]:
    """Evaluate vector-valued momentum nodes serially or in one fork pool."""

    if not momenta:
        return ()
    with _ForkMomentumBatchExecutor(
        integrand,
        workers=workers,
        maximum_tasks=len(momenta),
        backend_stats=backend_stats,
    ) as executor:
        evaluated = executor.evaluate(momenta)
    if any(not isinstance(value, tuple) for value in evaluated):
        raise AssertionError("vector momentum integrand returned a scalar")
    return tuple(tuple(complex(entry) for entry in value) for value in evaluated)


def gauss_laguerre_spectral_integral_batch(
    reduced_integrand: MomentumVectorIntegrand,
    *,
    output_size: int,
    gaussian_scales: Sequence[float],
    quadrature_order: int,
    coarse_order: int | None = None,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: Sequence[_BlockBackendStats] = (),
) -> tuple[SpectralIntegralDiagnostics, ...]:
    """Integrate several energies on one exact generalized Laguerre grid."""

    scales = tuple(float(scale) for scale in gaussian_scales)
    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )
    if coarse_order is not None and not (
        isinstance(coarse_order, int)
        and 2 <= coarse_order < quadrature_order
    ):
        raise ValueError(
            "coarse_order must be at least two and below quadrature_order"
        )
    with _ForkMomentumBatchExecutor(
        reduced_integrand,
        workers=workers,
        maximum_tasks=quadrature_order ** len(scales),
        backend_stats=_backend_stats,
    ) as executor:
        values, evaluations = _tensor_generalized_laguerre_values(
            reduced_integrand,
            output_size=output_size,
            gaussian_scales=scales,
            quadrature_order=quadrature_order,
            endpoint_powers=powers,
            workers=workers,
            _backend_stats=_backend_stats,
            _batch_executor=executor,
        )
        coarse_values: tuple[complex, ...] | None = None
        if coarse_order is not None:
            coarse_values, count = _tensor_generalized_laguerre_values(
                reduced_integrand,
                output_size=output_size,
                gaussian_scales=scales,
                quadrature_order=coarse_order,
                endpoint_powers=powers,
                workers=workers,
                _backend_stats=_backend_stats,
                _batch_executor=executor,
            )
            evaluations += count

    return tuple(
        SpectralIntegralDiagnostics(
            value=value,
            refined_value=None,
            extended_value=None,
            quadrature_absolute_error=(
                None
                if coarse_values is None
                else abs(value - coarse_values[index])
            ),
            tail_absolute_error=None,
            estimated_absolute_error=(
                None
                if coarse_values is None
                else abs(value - coarse_values[index])
            ),
            dimension=len(scales),
            p_max=None,
            quadrature_order=quadrature_order,
            refined_order=None,
            extended_p_max=None,
            function_evaluations=evaluations,
            quadrature_method="gauss_laguerre_batch",
            coarse_value=(
                None if coarse_values is None else coarse_values[index]
            ),
            coarse_order=coarse_order,
            orders_evaluated=(
                (quadrature_order,)
                if coarse_order is None
                else (coarse_order, quadrature_order)
            ),
            endpoint_powers=powers,
            worker_processes=min(
                workers,
                quadrature_order ** len(scales),
            ),
        )
        for index, value in enumerate(values)
    )


def adaptive_gauss_laguerre_spectral_integral_batch(
    reduced_integrand: MomentumVectorIntegrand,
    *,
    output_size: int,
    gaussian_scales: Sequence[float],
    quadrature_orders: Sequence[int],
    relative_tolerance: float,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: Sequence[_BlockBackendStats] = (),
) -> tuple[SpectralIntegralDiagnostics, ...]:
    """Adapt one shared Laguerre grid until every energy stabilizes."""

    orders = tuple(int(order) for order in quadrature_orders)
    if len(orders) < 2 or any(order < 2 for order in orders) or any(
        right <= left for left, right in zip(orders, orders[1:])
    ):
        raise ValueError(
            "adaptive quadrature orders must be strictly increasing and "
            "contain at least two entries"
        )
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    scales = tuple(float(scale) for scale in gaussian_scales)
    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )
    values_by_order: list[tuple[complex, ...]] = []
    evaluations = 0
    converged = (False,) * output_size
    with _ForkMomentumBatchExecutor(
        reduced_integrand,
        workers=workers,
        maximum_tasks=max(orders) ** len(scales),
        backend_stats=_backend_stats,
    ) as executor:
        for order in orders:
            values, count = _tensor_generalized_laguerre_values(
                reduced_integrand,
                output_size=output_size,
                gaussian_scales=scales,
                quadrature_order=order,
                endpoint_powers=powers,
                workers=workers,
                _backend_stats=_backend_stats,
                _batch_executor=executor,
            )
            values_by_order.append(values)
            evaluations += count
            if len(values_by_order) >= 2:
                converged = tuple(
                    abs(current - previous)
                    / max(abs(current), np.finfo(float).tiny)
                    <= tolerance
                    for current, previous in zip(
                        values_by_order[-1],
                        values_by_order[-2],
                    )
                )
                if all(converged):
                    break

    selected_order = orders[len(values_by_order) - 1]
    coarse_order = orders[len(values_by_order) - 2]
    selected = values_by_order[-1]
    coarse = values_by_order[-2]
    return tuple(
        SpectralIntegralDiagnostics(
            value=value,
            refined_value=None,
            extended_value=None,
            quadrature_absolute_error=abs(value - coarse[index]),
            tail_absolute_error=None,
            estimated_absolute_error=abs(value - coarse[index]),
            dimension=len(scales),
            p_max=None,
            quadrature_order=selected_order,
            refined_order=None,
            extended_p_max=None,
            function_evaluations=evaluations,
            quadrature_method="gauss_laguerre_adaptive_batch",
            coarse_value=coarse[index],
            coarse_order=coarse_order,
            target_relative_tolerance=tolerance,
            converged=converged[index],
            orders_evaluated=orders[: len(values_by_order)],
            endpoint_powers=powers,
            worker_processes=min(
                workers,
                selected_order ** len(scales),
            ),
        )
        for index, value in enumerate(selected)
    )


def gauss_laguerre_spectral_integral(
    reduced_integrand: MomentumIntegrand,
    *,
    gaussian_scales: Sequence[float],
    quadrature_order: int,
    coarse_order: int | None = None,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: _BlockBackendStats | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Integrate an exactly Gaussian-weighted continuum on ``[0,infinity)``.

    This routine expects the supplied function with the known factor
    ``exp(-sum(a_e*P_e**2))`` removed.  If ``coarse_order`` is supplied, an
    independent lower-order rule provides a convergence diagnostic.  The two
    generalized Laguerre rules are not nested, so their function evaluations
    are counted separately.
    """

    scales = tuple(float(scale) for scale in gaussian_scales)
    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )
    if coarse_order is not None:
        if not isinstance(coarse_order, int):
            raise TypeError("coarse_order must be an integer or None")
        if coarse_order < 2 or coarse_order >= quadrature_order:
            raise ValueError(
                "coarse_order must be at least two and below quadrature_order"
            )

    with _ForkMomentumBatchExecutor(
        reduced_integrand,
        workers=workers,
        maximum_tasks=quadrature_order ** len(scales),
        backend_stats=_backend_stats,
    ) as executor:
        value, evaluations = _tensor_generalized_laguerre_value(
            reduced_integrand,
            gaussian_scales=scales,
            quadrature_order=quadrature_order,
            endpoint_powers=powers,
            workers=workers,
            _backend_stats=_backend_stats,
            _batch_executor=executor,
        )
        coarse_value: complex | None = None
        quadrature_error: float | None = None
        if coarse_order is not None:
            coarse_value, count = _tensor_generalized_laguerre_value(
                reduced_integrand,
                gaussian_scales=scales,
                quadrature_order=coarse_order,
                endpoint_powers=powers,
                workers=workers,
                _backend_stats=_backend_stats,
                _batch_executor=executor,
            )
            evaluations += count
            quadrature_error = abs(value - coarse_value)

    return SpectralIntegralDiagnostics(
        value=value,
        refined_value=None,
        extended_value=None,
        quadrature_absolute_error=quadrature_error,
        tail_absolute_error=None,
        estimated_absolute_error=quadrature_error,
        dimension=len(scales),
        p_max=None,
        quadrature_order=quadrature_order,
        refined_order=None,
        extended_p_max=None,
        function_evaluations=evaluations,
        quadrature_method="gauss_laguerre",
        coarse_value=coarse_value,
        coarse_order=coarse_order,
        orders_evaluated=(
            (quadrature_order,)
            if coarse_order is None
            else (coarse_order, quadrature_order)
        ),
        endpoint_powers=powers,
        worker_processes=min(workers, quadrature_order ** len(scales)),
    )


def adaptive_gauss_laguerre_spectral_integral(
    reduced_integrand: MomentumIntegrand,
    *,
    gaussian_scales: Sequence[float],
    quadrature_orders: Sequence[int],
    relative_tolerance: float,
    endpoint_powers: Sequence[int] | None = None,
    workers: int = 1,
    _backend_stats: _BlockBackendStats | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Increase Gaussian-weighted tensor order until the value stabilizes.

    Successive generalized Laguerre rules are independent rather than nested.
    The stopping test is

    ``abs(I_n-I_previous) / max(abs(I_n), tiny) <= relative_tolerance``.

    This is a falsifiable convergence diagnostic, not a rigorous quadrature
    bound.  If the final requested order still fails, ``converged`` is false
    and the final difference is retained as the estimated error.
    """

    orders = tuple(int(order) for order in quadrature_orders)
    if len(orders) < 2:
        raise ValueError("at least two adaptive quadrature orders are required")
    if any(order < 2 for order in orders) or any(
        right <= left for left, right in zip(orders, orders[1:])
    ):
        raise ValueError(
            "adaptive quadrature orders must be strictly increasing and at "
            "least two"
        )
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    scales = tuple(float(scale) for scale in gaussian_scales)
    powers = (
        (0,) * len(scales)
        if endpoint_powers is None
        else tuple(endpoint_powers)
    )

    values: list[complex] = []
    evaluations = 0
    converged = False
    with _ForkMomentumBatchExecutor(
        reduced_integrand,
        workers=workers,
        maximum_tasks=max(orders) ** len(scales),
        backend_stats=_backend_stats,
    ) as executor:
        for order in orders:
            value, count = _tensor_generalized_laguerre_value(
                reduced_integrand,
                gaussian_scales=scales,
                quadrature_order=order,
                endpoint_powers=powers,
                workers=workers,
                _backend_stats=_backend_stats,
                _batch_executor=executor,
            )
            values.append(value)
            evaluations += count
            if len(values) >= 2:
                change = abs(values[-1] - values[-2])
                relative_change = change / max(
                    abs(values[-1]),
                    np.finfo(float).tiny,
                )
                if relative_change <= tolerance:
                    converged = True
                    break

    selected_order = orders[len(values) - 1]
    coarse_order = orders[len(values) - 2]
    quadrature_error = abs(values[-1] - values[-2])
    return SpectralIntegralDiagnostics(
        value=values[-1],
        refined_value=None,
        extended_value=None,
        quadrature_absolute_error=quadrature_error,
        tail_absolute_error=None,
        estimated_absolute_error=quadrature_error,
        dimension=len(scales),
        p_max=None,
        quadrature_order=selected_order,
        refined_order=None,
        extended_p_max=None,
        function_evaluations=evaluations,
        quadrature_method="gauss_laguerre_adaptive",
        coarse_value=values[-2],
        coarse_order=coarse_order,
        target_relative_tolerance=tolerance,
        converged=converged,
        orders_evaluated=orders[: len(values)],
        endpoint_powers=powers,
        worker_processes=min(workers, selected_order ** len(scales)),
    )


def _tensor_rule_value(
    integrand: MomentumIntegrand,
    *,
    dimension: int,
    p_max: float,
    quadrature_order: int,
) -> tuple[complex, int]:
    channels = gauss_legendre_spectral_rule(p_max, quadrature_order)
    total = 0.0j
    evaluations = 0
    for selected in product(channels, repeat=dimension):
        momenta = tuple(channel[0] for channel in selected)
        weight = math.prod(channel[1] for channel in selected)
        total += weight * complex(integrand(momenta))
        evaluations += 1
    if not math.isfinite(total.real) or not math.isfinite(total.imag):
        raise ArithmeticError("the spectral integral returned a non-finite value")
    return total, evaluations


def tensor_spectral_integral(
    integrand: MomentumIntegrand,
    *,
    dimension: int,
    p_max: float,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Evaluate a tensor ``dP/pi`` integral and optional error probes.

    ``refined_order`` estimates finite quadrature-order error at fixed
    ``p_max``.  ``extended_p_max`` estimates the omitted continuum tail at
    fixed quadrature order.  These differences are diagnostics rather than
    rigorous bounds; both must be varied before reporting precision.
    """

    if not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if refined_order is not None:
        if not isinstance(refined_order, int):
            raise TypeError("refined_order must be an integer or None")
        if refined_order <= quadrature_order:
            raise ValueError("refined_order must exceed quadrature_order")
    if extended_p_max is not None:
        extended_p_max = float(extended_p_max)
        if extended_p_max <= p_max:
            raise ValueError("extended_p_max must exceed p_max")

    value, evaluations = _tensor_rule_value(
        integrand,
        dimension=dimension,
        p_max=p_max,
        quadrature_order=quadrature_order,
    )
    refined_value: complex | None = None
    quadrature_error: float | None = None
    if refined_order is not None:
        refined_value, count = _tensor_rule_value(
            integrand,
            dimension=dimension,
            p_max=p_max,
            quadrature_order=refined_order,
        )
        evaluations += count
        quadrature_error = abs(refined_value - value)

    extended_value: complex | None = None
    tail_error: float | None = None
    if extended_p_max is not None:
        extended_value, count = _tensor_rule_value(
            integrand,
            dimension=dimension,
            p_max=extended_p_max,
            quadrature_order=quadrature_order,
        )
        evaluations += count
        tail_error = abs(extended_value - value)

    available_errors = tuple(
        error for error in (quadrature_error, tail_error) if error is not None
    )
    estimated_error = max(available_errors) if available_errors else None
    return SpectralIntegralDiagnostics(
        value=value,
        refined_value=refined_value,
        extended_value=extended_value,
        quadrature_absolute_error=quadrature_error,
        tail_absolute_error=tail_error,
        estimated_absolute_error=estimated_error,
        dimension=dimension,
        p_max=float(p_max),
        quadrature_order=quadrature_order,
        refined_order=refined_order,
        extended_p_max=extended_p_max,
        function_evaluations=evaluations,
    )


def _validate_necklace_inputs(
    internal_momenta: Sequence[float],
    external_momenta: Sequence[complex],
    plumbing_parameters: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
) -> tuple[
    tuple[float, ...],
    tuple[complex, ...],
    tuple[complex, ...],
    tuple[Word, ...],
    tuple[Word, ...],
]:
    internal = tuple(float(value) for value in internal_momenta)
    external = tuple(complex(value) for value in external_momenta)
    plumbing = tuple(complex(value) for value in plumbing_parameters)
    holomorphic = tuple(tuple(word) for word in holomorphic_words)
    antiholomorphic = tuple(tuple(word) for word in antiholomorphic_words)
    size = len(internal)
    if size == 0 or not all(
        len(values) == size
        for values in (external, plumbing, holomorphic, antiholomorphic)
    ):
        raise ValueError("all necklace inputs must have the same positive length")
    if any(momentum < 0 or not math.isfinite(momentum) for momentum in internal):
        raise ValueError("internal Liouville momenta must be finite and nonnegative")
    if any(not 0 < abs(value) < 1 for value in plumbing):
        raise ValueError("every plumbing parameter must satisfy 0 < abs(q) < 1")
    return internal, external, plumbing, holomorphic, antiholomorphic


def _edge_lifts(size: int, temporal_lift_sign: int) -> tuple[int, ...]:
    if temporal_lift_sign not in (-1, 1):
        raise ValueError("temporal_lift_sign must be +1 or -1")
    return (temporal_lift_sign,) + (1,) * (size - 1)


def _validate_block_backend(block_backend: str) -> BlockBackend:
    if block_backend not in (
        "direct",
        "direct_fast",
        "recursion",
        "hybrid",
        "auto",
    ):
        raise ValueError(
            "block_backend must be 'direct', 'direct_fast', 'recursion', "
            "'hybrid', or 'auto'"
        )
    return block_backend  # type: ignore[return-value]


def _maximum_twice_level_tuple(
    maximum_twice_levels: int | Sequence[int],
    size: int,
) -> tuple[int, ...]:
    if isinstance(maximum_twice_levels, int) and not isinstance(
        maximum_twice_levels,
        bool,
    ):
        cutoffs = (maximum_twice_levels,) * size
    else:
        cutoffs = tuple(maximum_twice_levels)  # type: ignore[arg-type]
    if len(cutoffs) != size:
        raise ValueError("one descendant cutoff is required per edge")
    return tuple(int(value) for value in cutoffs)


def _splice_hybrid_series(
    direct_base: TorusNecklaceBlockSeries,
    recursive_full: TorusNecklaceBlockSeries,
) -> TorusNecklaceBlockSeries:
    """Use direct coefficients in one base rectangle and recursion outside."""

    coefficients = dict(recursive_full.coefficients)
    for levels, coefficient in direct_base.coefficients.items():
        if all(
            level <= cutoff
            for level, cutoff in zip(levels, direct_base.maximum_twice_levels)
        ):
            coefficients[levels] = coefficient
    conditions = dict(recursive_full.gram_condition_numbers)
    conditions.update(direct_base.gram_condition_numbers)
    return replace(
        recursive_full,
        coefficients=coefficients,
        gram_condition_numbers=conditions,
    )


def _two_internal_momenta_are_confluent(
    internal_momenta: Sequence[float],
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    """Return whether a two-edge block lies on the recursion's confluent locus."""

    if len(internal_momenta) != 2:
        return False
    left, right = (float(value) for value in internal_momenta)
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _checked_recursive_series(
    result: RecursiveTorusBlockResult,
    tolerance: float,
) -> TorusNecklaceBlockSeries:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "recursion_finite_part_tolerance must be finite and positive"
        )
    if result.scaled_finite_part_error > tolerance:
        raise ArithmeticError(
            "the self-dual recursion failed its two-radius finite-part "
            f"diagnostic: {result.scaled_finite_part_error:.3e} > "
            f"{tolerance:.3e}"
        )
    return result.series


def _ramond_endpoint_powers(
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
) -> tuple[int, ...]:
    r"""Return the verified two-point long-R endpoint powers.

    For a two-vertex necklace, equal combined chiral parities at the two
    vertices imply one common endpoint class.  Even combined parity gives
    the observed linear zero on each internal momentum; odd combined parity
    gives a nonzero endpoint.  A general multi-vertex derivation has not yet
    been supplied, so higher-point necklaces conservatively retain the
    always-exact unfactored rule ``m=0``.
    """

    holomorphic = tuple(tuple(word) for word in holomorphic_words)
    antiholomorphic = tuple(tuple(word) for word in antiholomorphic_words)
    if len(holomorphic) != len(antiholomorphic):
        raise ValueError("holomorphic and antiholomorphic words must match")
    if len(holomorphic) != 2:
        return (0,) * len(holomorphic)
    combined_parities = tuple(
        (fermion_parity(hol_word) + fermion_parity(anti_word)) % 2
        for hol_word, anti_word in zip(holomorphic, antiholomorphic)
    )
    if combined_parities[0] != combined_parities[1]:
        return (0, 0)
    power = 1 - combined_parities[0]
    return (power, power)


def _paired_series_value(
    holomorphic_series: TorusNecklaceBlockSeries,
    antiholomorphic_series: TorusNecklaceBlockSeries,
    plumbing: tuple[complex, ...],
    *,
    sector: str,
    strip_internal_gaussian: bool,
    maximum_twice_levels: int | Sequence[int] | None = None,
) -> complex:
    """Evaluate a nonchiral block pair with an optional exact Gaussian strip."""

    conjugate_plumbing = tuple(value.conjugate() for value in plumbing)
    if not strip_internal_gaussian:
        return complex(
            holomorphic_series.value(
                plumbing,
                maximum_twice_levels=maximum_twice_levels,
            )
            * antiholomorphic_series.value(
                conjugate_plumbing,
                maximum_twice_levels=maximum_twice_levels,
            )
        )

    # At c=27/2, h_NS(P)-c/24=P^2/2-1/16 and
    # h_R(P)-c/24=P^2/2.  Removing exp(P^2 log|q|) from the nonchiral pair
    # therefore leaves only |q|^(-1/8) per NS edge and no R primary factor.
    constant_primary = (
        math.prod(abs(value) ** (-1.0 / 8.0) for value in plumbing)
        if sector == "NS"
        else 1.0
    )
    return complex(
        constant_primary
        * holomorphic_series.descendant_value(
            plumbing,
            maximum_twice_levels=maximum_twice_levels,
        )
        * antiholomorphic_series.descendant_value(
            conjugate_plumbing,
            maximum_twice_levels=maximum_twice_levels,
        )
    )


def ns_liouville_necklace_momentum_integrand(
    internal_momenta: Sequence[float],
    *,
    external_momenta: Sequence[complex],
    plumbing_parameters: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
    maximum_twice_levels: int | Sequence[int],
    temporal_lift_sign: int = 1,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    strip_internal_gaussian: bool = False,
    block_backend: BlockBackend = "direct",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    _backend_stats: _BlockBackendStats | None = None,
    _lower_maximum_twice_levels: int | Sequence[int] | None = None,
) -> complex | tuple[complex, complex]:
    r"""Return one fixed-momentum nonchiral NS-necklace integrand.

    The anti-holomorphic block is evaluated at ``conjugate(q_e)`` with the
    *same* analytically continued momenta and weights.  It is not obtained by
    conjugating those parameters.
    """

    internal, external, plumbing, holomorphic, antiholomorphic = (
        _validate_necklace_inputs(
            internal_momenta,
            external_momenta,
            plumbing_parameters,
            holomorphic_words,
            antiholomorphic_words,
        )
    )
    size = len(internal)
    backend = _validate_block_backend(block_backend)
    lifts = _edge_lifts(size, temporal_lift_sign)
    requested_cutoffs = _maximum_twice_level_tuple(
        maximum_twice_levels,
        size,
    )
    internal_weights = tuple(ns_weight(momentum) for momentum in internal)
    external_weights = tuple(ns_weight(momentum) for momentum in external)
    if size == 2 and external[0] == external[1]:
        # NS three-point constants are symmetric in all three momenta.  The
        # second necklace vertex only exchanges the two internal arguments.
        shared_vertex_constants = ns_structure_constants(
            internal[1],
            external[0],
            internal[0],
            precision=structure_precision,
        )
        vertex_constants = (
            shared_vertex_constants,
            shared_vertex_constants,
        )
    else:
        vertex_constants = tuple(
            ns_structure_constants(
                internal[(vertex - 1) % size],
                external[vertex],
                internal[vertex],
                precision=structure_precision,
            )
            for vertex in range(size)
        )

    total = 0.0j
    lower_total = 0.0j
    for parities in product((0, 1), repeat=size):
        form_parity_sum = sum(parities) % 2
        if form_parity_sum != sum(
            fermion_parity(word) for word in holomorphic
        ) % 2 or form_parity_sum != sum(
            fermion_parity(word) for word in antiholomorphic
        ) % 2:
            # On a closed necklace, every internal-state parity occurs at
            # two adjacent vertices and cancels.  Hence the xor of the local
            # three-form parities must equal the total external parity in
            # each chiral half.  The omitted terms vanish identically.
            continue
        form_weights = tuple(
            (1.0, 0.0) if parity == 0 else (0.0, 1.0)
            for parity in parities
        )
        coefficient = math.prod(
            vertex_constants[vertex][parity]
            for vertex, parity in enumerate(parities)
        )
        if coefficient == 0:
            continue

        @lru_cache(maxsize=2)
        def build_series(words: tuple[Word, ...]) -> TorusNecklaceBlockSeries:
            equal_external_momenta = (
                size == 2
                and abs(external[0] - external[1])
                <= 1.0e-12
                * max(1.0, abs(external[0]), abs(external[1]))
            )
            recursively_supported = (
                size == 2
                and parities[0] == parities[1]
                and (
                    words == ((), ())
                    or (
                        equal_external_momenta
                        and words
                        == (
                            RECURSIVE_G_MINUS_HALF,
                            RECURSIVE_G_MINUS_HALF,
                        )
                    )
                )
            )
            if backend == "recursion" and not recursively_supported:
                raise NotImplementedError(
                    "the certified NS recursion supports only a two-point "
                    "PP block, or an equal-momentum GG block, with equal "
                    "homogeneous form parities"
                )

            static_fast_available = generated_ns_rectangle_available(
                requested_cutoffs,
                words,
            )
            if backend == "direct_fast" or (
                backend in ("auto", "hybrid") and static_fast_available
            ):
                if _backend_stats is not None:
                    _backend_stats.fast_direct_block_evaluations += 1
                return direct_fast_b1_ns_necklace_series(
                    internal_momenta=internal,
                    external_ns_momenta=external,
                    maximum_twice_levels=maximum_twice_levels,
                    edge_lift_signs=lifts,
                    external_words=words,
                    form_weights=form_weights,
                    condition_limit=min(condition_limit, 1.0e11),
                )

            # Equal internal weights are a confluent-pole locus of the
            # ordinary-c recursion but are regular in the inverse-Gram
            # definition.  ``auto`` avoids exception-driven work there;
            # strict ``recursion`` retains the failure as a diagnostic.
            recursion_is_stable = not _two_internal_momenta_are_confluent(
                internal
            )
            if recursively_supported and (
                backend == "recursion"
                or (backend == "auto" and recursion_is_stable)
            ):
                try:
                    recursive = _checked_recursive_series(
                        recursive_ns_torus_two_point_series(
                            internal_momenta=internal,
                            external_ns_momenta=external,
                            maximum_twice_levels=maximum_twice_levels,
                            edge_lift_signs=lifts,
                            external_words=words,
                            form_parity=parities[0],
                            radius=recursion_radius,
                            check_radius=recursion_check_radius,
                            samples=recursion_samples,
                        ),
                        recursion_finite_part_tolerance,
                    )
                except (ArithmeticError, np.linalg.LinAlgError):
                    if backend == "recursion":
                        raise
                    if _backend_stats is not None:
                        _backend_stats.recursion_fallbacks += 1
                else:
                    if _backend_stats is not None:
                        _backend_stats.recursive_block_evaluations += 1
                    return recursive
            if _backend_stats is not None:
                _backend_stats.direct_block_evaluations += 1
            return direct_ns_torus_necklace_series(
                c=13.5,
                internal_weights=internal_weights,
                external_weights=external_weights,
                maximum_twice_levels=maximum_twice_levels,
                edge_lift_signs=lifts,
                external_words=words,
                form_weights=form_weights,
                digits=block_digits,
                condition_limit=condition_limit,
                vertex_backend="template",
            )

        holomorphic_series = build_series(holomorphic)
        antiholomorphic_series = build_series(antiholomorphic)
        total += coefficient * _paired_series_value(
            holomorphic_series,
            antiholomorphic_series,
            plumbing,
            sector="NS",
            strip_internal_gaussian=strip_internal_gaussian,
        )
        if _lower_maximum_twice_levels is not None:
            lower_total += coefficient * _paired_series_value(
                holomorphic_series,
                antiholomorphic_series,
                plumbing,
                sector="NS",
                strip_internal_gaussian=strip_internal_gaussian,
                maximum_twice_levels=_lower_maximum_twice_levels,
            )
    if _lower_maximum_twice_levels is None:
        return complex(total)
    return complex(total), complex(lower_total)


def ramond_liouville_necklace_momentum_integrand(
    internal_momenta: Sequence[float],
    *,
    external_momenta: Sequence[complex],
    plumbing_parameters: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
    maximum_twice_levels: int | Sequence[int],
    temporal_lift_sign: int = 1,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    strip_internal_gaussian: bool = False,
    block_backend: BlockBackend = "direct",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    _backend_stats: _BlockBackendStats | None = None,
    _lower_maximum_twice_levels: int | Sequence[int] | None = None,
) -> complex | tuple[complex, complex]:
    r"""Return one fixed-momentum nonchiral long-R necklace integrand."""

    internal, external, plumbing, holomorphic, antiholomorphic = (
        _validate_necklace_inputs(
            internal_momenta,
            external_momenta,
            plumbing_parameters,
            holomorphic_words,
            antiholomorphic_words,
        )
    )
    size = len(internal)
    backend = _validate_block_backend(block_backend)
    lifts = _edge_lifts(size, temporal_lift_sign)
    if size == 2 and external[0] == external[1]:
        # The R-R-NS constants are symmetric in their first two (Ramond)
        # momenta, so the two equal-external-momentum vertices share both HJS
        # structure coefficients.
        shared_vertex_constants = rr_ns_structure_constants(
            internal[1],
            internal[0],
            external[0],
            precision=structure_precision,
        )
        vertex_constants = (
            shared_vertex_constants,
            shared_vertex_constants,
        )
    else:
        vertex_constants = tuple(
            rr_ns_structure_constants(
                internal[(vertex - 1) % size],
                internal[vertex],
                external[vertex],
                precision=structure_precision,
            )
            for vertex in range(size)
        )

    requested_cutoffs = _maximum_twice_level_tuple(
        maximum_twice_levels,
        size,
    )
    recursion_is_stable = not _two_internal_momenta_are_confluent(internal)

    def recursion_supports(words: tuple[Word, ...]) -> bool:
        return size == 2 and lifts[0] == lifts[1] and words in (
            ((), ()),
            (RECURSIVE_G_MINUS_HALF, RECURSIVE_G_MINUS_HALF),
        )

    static_direct_available = (
        generated_direct_rectangle_available(requested_cutoffs, holomorphic)
        and generated_direct_rectangle_available(
            requested_cutoffs,
            antiholomorphic,
        )
    )
    use_fused_fast = size == 2 and (
        backend == "direct_fast"
        or (
            backend in ("auto", "hybrid")
            and (
                static_direct_available
                or not recursion_supports(holomorphic)
                or not recursion_supports(antiholomorphic)
                or not recursion_is_stable
            )
        )
    )
    if use_fused_fast:
        @lru_cache(maxsize=2)
        def build_fused(
            words: tuple[Word, ...],
        ) -> dict[int, TorusNecklaceBlockSeries]:
            if _backend_stats is not None:
                _backend_stats.fast_direct_block_evaluations += 1
            return direct_fast_b1_ramond_two_sign_series(
                internal_momenta=internal,
                external_ns_momenta=external,
                temporal_lift_sign=temporal_lift_sign,
                maximum_twice_levels=maximum_twice_levels,
                edge_lift_signs=lifts,
                external_words=words,
                condition_limit=min(condition_limit, 1.0e11),
            )

        holomorphic_by_sign = build_fused(holomorphic)
        antiholomorphic_by_sign = build_fused(antiholomorphic)
        total = 0.0j
        lower_total = 0.0j
        for first_sign in (-1, 1):
            signs = (first_sign, temporal_lift_sign * first_sign)
            coefficient = 2.0 * math.prod(
                vertex_constants[vertex][0 if sign == 1 else 1]
                for vertex, sign in enumerate(signs)
            )
            total += coefficient * _paired_series_value(
                holomorphic_by_sign[first_sign],
                antiholomorphic_by_sign[first_sign],
                plumbing,
                sector="R",
                strip_internal_gaussian=strip_internal_gaussian,
            )
            if _lower_maximum_twice_levels is not None:
                lower_total += coefficient * _paired_series_value(
                    holomorphic_by_sign[first_sign],
                    antiholomorphic_by_sign[first_sign],
                    plumbing,
                    sector="R",
                    strip_internal_gaussian=strip_internal_gaussian,
                    maximum_twice_levels=_lower_maximum_twice_levels,
                )
        if _lower_maximum_twice_levels is None:
            return complex(total)
        return complex(total), complex(lower_total)

    total = 0.0j
    lower_total = 0.0j
    for signs in product((-1, 1), repeat=size):
        if math.prod(signs) != temporal_lift_sign:
            continue
        coefficient = 2.0 * math.prod(
            vertex_constants[vertex][0 if sign == 1 else 1]
            for vertex, sign in enumerate(signs)
        )

        @lru_cache(maxsize=2)
        def build_series(words: tuple[Word, ...]) -> TorusNecklaceBlockSeries:
            recursively_supported = (
                size == 2
                and lifts[0] == lifts[1]
                and words
                in (
                    ((), ()),
                    (RECURSIVE_G_MINUS_HALF, RECURSIVE_G_MINUS_HALF),
                )
            )
            if backend == "recursion" and not recursively_supported:
                raise NotImplementedError(
                    "the certified Ramond recursion supports only two-point "
                    "PP or GG external components"
                )

            def build_direct_oracle() -> TorusNecklaceBlockSeries:
                if _backend_stats is not None:
                    _backend_stats.direct_block_evaluations += 1
                return b1_ramond_liouville_necklace_series(
                    internal_momenta=internal,
                    external_ns_momenta=external,
                    structure_signs=signs,
                    maximum_twice_levels=maximum_twice_levels,
                    edge_lift_signs=lifts,
                    external_words=words,
                    include_structure_constants=False,
                    precision=structure_precision,
                    digits=block_digits,
                    condition_limit=condition_limit,
                )

            def build_direct_fast(
                cutoffs: int | Sequence[int] = maximum_twice_levels,
            ) -> TorusNecklaceBlockSeries:
                if _backend_stats is not None:
                    _backend_stats.fast_direct_block_evaluations += 1
                return direct_fast_b1_ramond_necklace_series(
                    internal_momenta=internal,
                    external_ns_momenta=external,
                    structure_signs=signs,
                    maximum_twice_levels=cutoffs,
                    edge_lift_signs=lifts,
                    external_words=words,
                    include_structure_constants=False,
                    structure_precision=structure_precision,
                    condition_limit=min(condition_limit, 1.0e11),
                )

            if backend == "direct":
                return build_direct_oracle()
            if backend == "direct_fast":
                return build_direct_fast()

            # The same confluent-pole restriction applies to the Ramond
            # two-Virasoro recursion.  The automatic policy also keeps all
            # certified low-level rectangles on the faster regular b=1 path.
            automatic_low_level = generated_direct_rectangle_available(
                requested_cutoffs,
                words,
            )
            if backend in ("auto", "hybrid") and (
                automatic_low_level
                or not recursively_supported
                or not recursion_is_stable
            ):
                return build_direct_fast()

            if recursively_supported and (
                recursion_is_stable or backend == "recursion"
            ):
                try:
                    recursive = _checked_recursive_series(
                        recursive_ramond_torus_two_point_series(
                            internal_momenta=internal,
                            external_ns_momenta=external,
                            structure_signs=signs,
                            maximum_twice_levels=maximum_twice_levels,
                            edge_lift_signs=lifts,
                            external_words=words,
                            radius=recursion_radius,
                            check_radius=recursion_check_radius,
                            samples=recursion_samples,
                        ),
                        recursion_finite_part_tolerance,
                    )
                except (ArithmeticError, np.linalg.LinAlgError):
                    if backend == "recursion":
                        raise
                    if _backend_stats is not None:
                        _backend_stats.recursion_fallbacks += 1
                    return build_direct_fast()
                else:
                    if _backend_stats is not None:
                        _backend_stats.recursive_block_evaluations += 1
                    if backend == "recursion":
                        return recursive
                    base_cutoffs = tuple(
                        min(cutoff, _FAST_DIRECT_MAX_TWICE_LEVEL)
                        for cutoff in requested_cutoffs
                    )
                    direct_base = build_direct_fast(base_cutoffs)
                    return _splice_hybrid_series(direct_base, recursive)

            if backend == "recursion":
                raise NotImplementedError(
                    "the certified Ramond recursion does not support this block"
                )
            return build_direct_fast()

        holomorphic_series = build_series(holomorphic)
        antiholomorphic_series = build_series(antiholomorphic)
        total += coefficient * _paired_series_value(
            holomorphic_series,
            antiholomorphic_series,
            plumbing,
            sector="R",
            strip_internal_gaussian=strip_internal_gaussian,
        )
        if _lower_maximum_twice_levels is not None:
            lower_total += coefficient * _paired_series_value(
                holomorphic_series,
                antiholomorphic_series,
                plumbing,
                sector="R",
                strip_internal_gaussian=strip_internal_gaussian,
                maximum_twice_levels=_lower_maximum_twice_levels,
            )
    if _lower_maximum_twice_levels is None:
        return complex(total)
    return complex(total), complex(lower_total)


def _prewarm_gaussian_direct_backend(
    *,
    sector: str,
    block_backend: BlockBackend,
    maximum_twice_levels: int | Sequence[int],
    necklace_size: int,
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
) -> None:
    """Prepare generated or fallback templates before a spectral fork."""

    backend_policy = _validate_block_backend(block_backend)
    requested_cutoffs = _maximum_twice_level_tuple(
        maximum_twice_levels,
        necklace_size,
    )
    if sector == "NS" and backend_policy == "direct_fast":
        if not (
            generated_ns_rectangle_available(
                requested_cutoffs,
                holomorphic_words,
            )
            and generated_ns_rectangle_available(
                requested_cutoffs,
                antiholomorphic_words,
            )
        ):
            prewarm_direct_fast_b1_ns(
                requested_cutoffs,
                holomorphic_words,
            )
            if tuple(antiholomorphic_words) != tuple(holomorphic_words):
                prewarm_direct_fast_b1_ns(
                    requested_cutoffs,
                    antiholomorphic_words,
                )
    if sector == "R" and backend_policy in (
        "direct_fast",
        "hybrid",
        "auto",
    ):
        prewarm_cutoffs = (
            requested_cutoffs
            if backend_policy == "direct_fast"
            or (
                generated_direct_rectangle_available(
                    requested_cutoffs,
                    holomorphic_words,
                )
                and generated_direct_rectangle_available(
                    requested_cutoffs,
                    antiholomorphic_words,
                )
            )
            else tuple(
                min(cutoff, _FAST_DIRECT_MAX_TWICE_LEVEL)
                for cutoff in requested_cutoffs
            )
        )
        prewarm_direct_fast_b1_ramond(
            prewarm_cutoffs,
            holomorphic_words,
        )
        if tuple(antiholomorphic_words) != tuple(holomorphic_words):
            prewarm_direct_fast_b1_ramond(
                prewarm_cutoffs,
                antiholomorphic_words,
            )


def integrate_liouville_necklace(
    *,
    sector: str,
    external_momenta: Sequence[complex],
    plumbing_parameters: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
    maximum_twice_levels: int | Sequence[int],
    temporal_lift_sign: int,
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    spectral_method: str = "gauss_legendre",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> SpectralIntegralDiagnostics:
    r"""Integrate one NS or R nonchiral necklace over all edge momenta.

    If ``subtract_maximum_twice_levels`` is supplied, the returned integrand
    is the difference between the constructed block and its lower rectangular
    truncation.  Both values are read from the same coefficient tables at
    every momentum node.  This exact multilevel correction avoids a second
    block construction; it does not approximate the conformal block.
    """

    started = time.perf_counter()
    external = tuple(complex(value) for value in external_momenta)
    spectral_workers = _validate_worker_processes(spectral_workers)
    if sector not in ("NS", "R"):
        raise ValueError("sector must be 'NS' or 'R'")
    evaluator = (
        ns_liouville_necklace_momentum_integrand
        if sector == "NS"
        else ramond_liouville_necklace_momentum_integrand
    )
    backend_stats = _BlockBackendStats()

    def finalize(
        result: SpectralIntegralDiagnostics,
    ) -> SpectralIntegralDiagnostics:
        return replace(
            result,
            block_backend_requested=_validate_block_backend(block_backend),
            recursive_block_evaluations=(
                backend_stats.recursive_block_evaluations
            ),
            direct_block_evaluations=backend_stats.direct_block_evaluations,
            fast_direct_block_evaluations=(
                backend_stats.fast_direct_block_evaluations
            ),
            recursion_fallbacks=backend_stats.recursion_fallbacks,
            runtime_seconds=time.perf_counter() - started,
        )

    def momentum_integrand(
        momenta: tuple[float, ...],
        *,
        strip_internal_gaussian: bool = False,
    ) -> complex:
        evaluated = evaluator(
            momenta,
            external_momenta=external,
            plumbing_parameters=plumbing_parameters,
            holomorphic_words=holomorphic_words,
            antiholomorphic_words=antiholomorphic_words,
            maximum_twice_levels=maximum_twice_levels,
            temporal_lift_sign=temporal_lift_sign,
            structure_precision=structure_precision,
            block_digits=block_digits,
            condition_limit=condition_limit,
            strip_internal_gaussian=strip_internal_gaussian,
            block_backend=block_backend,
            recursion_radius=recursion_radius,
            recursion_check_radius=recursion_check_radius,
            recursion_samples=recursion_samples,
            recursion_finite_part_tolerance=recursion_finite_part_tolerance,
            _backend_stats=backend_stats,
            _lower_maximum_twice_levels=subtract_maximum_twice_levels,
        )
        if subtract_maximum_twice_levels is None:
            if isinstance(evaluated, tuple):
                raise AssertionError("unexpected paired block evaluation")
            return complex(evaluated)
        if not isinstance(evaluated, tuple):
            raise AssertionError("missing lower block evaluation")
        current, lower = evaluated
        return complex(current - lower)

    if spectral_method == "gauss_legendre":
        if p_max is None:
            raise ValueError("gauss_legendre requires a finite p_max")
        return finalize(
            tensor_spectral_integral(
                momentum_integrand,
                dimension=len(external),
                p_max=p_max,
                quadrature_order=quadrature_order,
                refined_order=refined_order,
                extended_p_max=extended_p_max,
            )
        )
    if spectral_method == "gauss_kronrod_3_7":
        if p_max is None:
            raise ValueError("gauss_kronrod_3_7 requires a finite p_max")
        if quadrature_order != 7 or refined_order is not None:
            raise ValueError(
                "gauss_kronrod_3_7 requires quadrature_order=7 and "
                "refined_order=None"
            )
        return finalize(
            gauss_kronrod_3_7_spectral_integral(
                momentum_integrand,
                dimension=len(external),
                p_max=p_max,
                extended_p_max=extended_p_max,
            )
        )
    if spectral_method == "gauss_kronrod_7_15":
        if p_max is None:
            raise ValueError("gauss_kronrod_7_15 requires a finite p_max")
        if quadrature_order != 15 or refined_order is not None:
            raise ValueError(
                "gauss_kronrod_7_15 requires quadrature_order=15 and "
                "refined_order=None"
            )
        return finalize(
            gauss_kronrod_7_15_spectral_integral(
                momentum_integrand,
                dimension=len(external),
                p_max=p_max,
                extended_p_max=extended_p_max,
            )
        )
    if spectral_method in (
        "gauss_laguerre",
        "gauss_laguerre_7_15",
        "gauss_laguerre_adaptive",
    ):
        if p_max is not None or extended_p_max is not None:
            raise ValueError(
                "Gaussian-weighted spectral integration requires "
                "p_max=None and extended_p_max=None"
            )
        if (
            spectral_method != "gauss_laguerre_adaptive"
            and refined_order is not None
        ):
            raise ValueError(
                "Gaussian-weighted spectral integration uses coarse_order "
                "internally and requires refined_order=None"
            )
        if spectral_method == "gauss_laguerre_7_15" and quadrature_order != 15:
            raise ValueError(
                "gauss_laguerre_7_15 requires quadrature_order=15"
            )
        scales = tuple(
            -math.log(abs(complex(value))) for value in plumbing_parameters
        )
        endpoint_powers = (
            (0,) * len(scales)
            if sector == "NS"
            else _ramond_endpoint_powers(
                holomorphic_words,
                antiholomorphic_words,
            )
        )

        _prewarm_gaussian_direct_backend(
            sector=sector,
            block_backend=block_backend,
            maximum_twice_levels=maximum_twice_levels,
            necklace_size=len(external),
            holomorphic_words=holomorphic_words,
            antiholomorphic_words=antiholomorphic_words,
        )

        def gaussian_reduced_integrand(
            momenta: tuple[float, ...],
        ) -> complex:
            return momentum_integrand(
                momenta,
                strip_internal_gaussian=True,
            )

        if spectral_method == "gauss_laguerre_adaptive":
            if refined_order is None or refined_order <= quadrature_order:
                raise ValueError(
                    "gauss_laguerre_adaptive requires refined_order above "
                    "quadrature_order"
                )
            if spectral_relative_tolerance is None:
                raise ValueError(
                    "gauss_laguerre_adaptive requires "
                    "spectral_relative_tolerance"
                )
            intermediate_order = 2 * quadrature_order + 1
            orders = [quadrature_order]
            if intermediate_order < refined_order:
                orders.append(intermediate_order)
            orders.append(refined_order)
            return finalize(
                adaptive_gauss_laguerre_spectral_integral(
                    gaussian_reduced_integrand,
                    gaussian_scales=scales,
                    quadrature_orders=orders,
                    relative_tolerance=spectral_relative_tolerance,
                    endpoint_powers=endpoint_powers,
                    workers=spectral_workers,
                    _backend_stats=backend_stats,
                )
            )
        if spectral_relative_tolerance is not None:
            raise ValueError(
                "spectral_relative_tolerance is only used by "
                "gauss_laguerre_adaptive"
            )

        return finalize(
            gauss_laguerre_spectral_integral(
                gaussian_reduced_integrand,
                gaussian_scales=scales,
                quadrature_order=quadrature_order,
                coarse_order=(
                    7 if spectral_method == "gauss_laguerre_7_15" else None
                ),
                endpoint_powers=endpoint_powers,
                workers=spectral_workers,
                _backend_stats=backend_stats,
            )
        )
    raise ValueError(
        "spectral_method must be 'gauss_legendre', 'gauss_kronrod_3_7', "
        "'gauss_kronrod_7_15', 'gauss_laguerre', "
        "'gauss_laguerre_7_15', or 'gauss_laguerre_adaptive'"
    )


def integrate_liouville_necklace_batch(
    *,
    sector: str,
    external_momenta_batch: Sequence[Sequence[complex]],
    plumbing_parameters: Sequence[complex],
    holomorphic_words: Sequence[Word],
    antiholomorphic_words: Sequence[Word],
    maximum_twice_levels: int | Sequence[int],
    temporal_lift_sign: int,
    p_max: float | None,
    quadrature_order: int,
    refined_order: int | None = None,
    extended_p_max: float | None = None,
    structure_precision: int = 40,
    block_digits: int = 50,
    condition_limit: float = 1.0e13,
    spectral_method: str = "gauss_laguerre_adaptive",
    spectral_relative_tolerance: float | None = None,
    block_backend: BlockBackend = "direct_fast",
    recursion_radius: float = 0.04,
    recursion_check_radius: float = 0.05,
    recursion_samples: int = 24,
    recursion_finite_part_tolerance: float = 1.0e-7,
    spectral_workers: int = 1,
    subtract_maximum_twice_levels: int | Sequence[int] | None = None,
) -> tuple[SpectralIntegralDiagnostics, ...]:
    r"""Integrate several external energies on one fixed momentum grid.

    The quadrature nodes are outermost and the external-energy index is
    innermost.  Thus every energy uses exactly the same nodes and weights,
    while Gram factorizations and polynomially whitened Ward tensors remain
    reusable inside each worker.  This routine is exact relative to the
    selected finite descendant rectangle and supports the Gaussian-weighted
    production methods; it performs no interpolation in momentum or energy.
    """

    started = time.perf_counter()
    external_batch = tuple(
        tuple(complex(value) for value in external_momenta)
        for external_momenta in external_momenta_batch
    )
    if not external_batch:
        raise ValueError("external_momenta_batch must not be empty")
    necklace_size = len(external_batch[0])
    if necklace_size < 1 or any(
        len(external) != necklace_size for external in external_batch
    ):
        raise ValueError(
            "every external momentum tuple must have the same positive length"
        )
    if sector not in ("NS", "R"):
        raise ValueError("sector must be 'NS' or 'R'")
    if spectral_method not in (
        "gauss_laguerre",
        "gauss_laguerre_7_15",
        "gauss_laguerre_adaptive",
    ):
        raise ValueError(
            "batched energy evaluation currently requires a Gauss-Laguerre "
            "spectral method"
        )
    if p_max is not None or extended_p_max is not None:
        raise ValueError(
            "Gaussian-weighted spectral integration requires p_max=None and "
            "extended_p_max=None"
        )
    if spectral_method != "gauss_laguerre_adaptive" and refined_order is not None:
        raise ValueError(
            "non-adaptive Gaussian integration requires refined_order=None"
        )
    if spectral_method == "gauss_laguerre_7_15" and quadrature_order != 15:
        raise ValueError("gauss_laguerre_7_15 requires quadrature_order=15")

    spectral_workers = _validate_worker_processes(spectral_workers)
    evaluator = (
        ns_liouville_necklace_momentum_integrand
        if sector == "NS"
        else ramond_liouville_necklace_momentum_integrand
    )
    backend_stats = tuple(_BlockBackendStats() for _ in external_batch)
    plumbing = tuple(complex(value) for value in plumbing_parameters)
    holomorphic = tuple(tuple(word) for word in holomorphic_words)
    antiholomorphic = tuple(tuple(word) for word in antiholomorphic_words)
    if not (
        len(plumbing)
        == len(holomorphic)
        == len(antiholomorphic)
        == necklace_size
    ):
        raise ValueError("batch necklace inputs must have equal length")

    _prewarm_gaussian_direct_backend(
        sector=sector,
        block_backend=block_backend,
        maximum_twice_levels=maximum_twice_levels,
        necklace_size=necklace_size,
        holomorphic_words=holomorphic,
        antiholomorphic_words=antiholomorphic,
    )

    def momentum_integrand(
        momenta: tuple[float, ...],
    ) -> tuple[complex, ...]:
        values: list[complex] = []
        for external, stats in zip(external_batch, backend_stats):
            evaluated = evaluator(
                momenta,
                external_momenta=external,
                plumbing_parameters=plumbing,
                holomorphic_words=holomorphic,
                antiholomorphic_words=antiholomorphic,
                maximum_twice_levels=maximum_twice_levels,
                temporal_lift_sign=temporal_lift_sign,
                structure_precision=structure_precision,
                block_digits=block_digits,
                condition_limit=condition_limit,
                strip_internal_gaussian=True,
                block_backend=block_backend,
                recursion_radius=recursion_radius,
                recursion_check_radius=recursion_check_radius,
                recursion_samples=recursion_samples,
                recursion_finite_part_tolerance=(
                    recursion_finite_part_tolerance
                ),
                _backend_stats=stats,
                _lower_maximum_twice_levels=(
                    subtract_maximum_twice_levels
                ),
            )
            if subtract_maximum_twice_levels is None:
                if isinstance(evaluated, tuple):
                    raise AssertionError("unexpected paired block evaluation")
                values.append(complex(evaluated))
            else:
                if not isinstance(evaluated, tuple):
                    raise AssertionError("missing lower block evaluation")
                current, lower = evaluated
                values.append(complex(current - lower))
        return tuple(values)

    scales = tuple(-math.log(abs(value)) for value in plumbing)
    endpoint_powers = (
        (0,) * necklace_size
        if sector == "NS"
        else _ramond_endpoint_powers(holomorphic, antiholomorphic)
    )
    if spectral_method == "gauss_laguerre_adaptive":
        if refined_order is None or refined_order <= quadrature_order:
            raise ValueError(
                "gauss_laguerre_adaptive requires refined_order above "
                "quadrature_order"
            )
        if spectral_relative_tolerance is None:
            raise ValueError(
                "gauss_laguerre_adaptive requires spectral_relative_tolerance"
            )
        intermediate_order = 2 * quadrature_order + 1
        orders = [quadrature_order]
        if intermediate_order < refined_order:
            orders.append(intermediate_order)
        orders.append(refined_order)
        results = adaptive_gauss_laguerre_spectral_integral_batch(
            momentum_integrand,
            output_size=len(external_batch),
            gaussian_scales=scales,
            quadrature_orders=orders,
            relative_tolerance=spectral_relative_tolerance,
            endpoint_powers=endpoint_powers,
            workers=spectral_workers,
            _backend_stats=backend_stats,
        )
    else:
        if spectral_relative_tolerance is not None:
            raise ValueError(
                "spectral_relative_tolerance is only used by the adaptive method"
            )
        results = gauss_laguerre_spectral_integral_batch(
            momentum_integrand,
            output_size=len(external_batch),
            gaussian_scales=scales,
            quadrature_order=quadrature_order,
            coarse_order=(
                7 if spectral_method == "gauss_laguerre_7_15" else None
            ),
            endpoint_powers=endpoint_powers,
            workers=spectral_workers,
            _backend_stats=backend_stats,
        )

    runtime = time.perf_counter() - started
    validated_backend = _validate_block_backend(block_backend)
    return tuple(
        replace(
            result,
            block_backend_requested=validated_backend,
            recursive_block_evaluations=stats.recursive_block_evaluations,
            direct_block_evaluations=stats.direct_block_evaluations,
            fast_direct_block_evaluations=stats.fast_direct_block_evaluations,
            recursion_fallbacks=stats.recursion_fallbacks,
            runtime_seconds=runtime,
        )
        for result, stats in zip(results, backend_stats)
    )


__all__ = [
    "SpectralIntegralDiagnostics",
    "adaptive_gauss_laguerre_spectral_integral",
    "adaptive_gauss_laguerre_spectral_integral_batch",
    "gauss_legendre_spectral_rule",
    "gauss_kronrod_3_7_spectral_integral",
    "gauss_kronrod_7_15_spectral_integral",
    "gauss_laguerre_spectral_integral",
    "gauss_laguerre_spectral_integral_batch",
    "integrate_liouville_necklace",
    "integrate_liouville_necklace_batch",
    "ns_liouville_necklace_momentum_integrand",
    "ramond_liouville_necklace_momentum_integrand",
    "tensor_spectral_integral",
]
