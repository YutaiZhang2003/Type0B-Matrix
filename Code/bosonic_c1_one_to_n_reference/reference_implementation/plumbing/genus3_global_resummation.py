#!/usr/bin/env python3
"""Channel-adapted all-level global blocks for all genus-three channels.

Each of the five marked plumbing graphs is contracted in its native frame:

* tadpole edges are reduced to one-index loop kernels;
* parallel edge pairs are reduced to bivariate double-edge kernels;
* ``opposite-double-edge-cycle`` becomes two double-edge kernels;
* ``tetrahedron`` additionally sums the complete ``q12`` tower as ``2F1``.

The remaining edge sums are cubically capped and increased adaptively.  The
cap is a numerical convergence control, independent of the Virasoro pole
recursion order used by :mod:`ccy_plumbing_graph`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

try:
    from ccy_genus2_block import rho_lminus1_two_edge
    from genus3_plumbing_channels import (
        Genus3PlumbingChannel,
        genus3_channel_by_name,
        genus3_channel_q_values,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import rho_lminus1_two_edge
    from plumbing.genus3_plumbing_channels import (
        Genus3PlumbingChannel,
        genus3_channel_by_name,
        genus3_channel_q_values,
    )


SUPPORTED_CHANNELS = frozenset(
    {
        "one-tadpole-double-triangle",
        "opposite-double-edge-cycle",
        "tetrahedron",
        "three-tadpole-star",
        "two-tadpoles-double-bridge",
    }
)


@dataclass(frozen=True)
class Genus3ResummedGlobalBlockResult:
    """One adaptively converged channel-specific global block."""

    value: complex
    channel: str
    method: str
    cap: int
    relative_tail_estimate: float
    tolerance: float
    exact_resummed_edges: tuple[str, ...]


def _resolve_channel(
    channel: str | Genus3PlumbingChannel,
) -> Genus3PlumbingChannel:
    resolved = genus3_channel_by_name(channel) if isinstance(channel, str) else channel
    if resolved.name not in SUPPORTED_CHANNELS:
        raise ValueError(
            f"channel-specific genus-three resummation is not implemented for "
            f"{resolved.name!r}"
        )
    return resolved


def _edge_weights(
    channel: Genus3PlumbingChannel,
    values: Sequence[complex] | Mapping[str, complex],
) -> tuple[complex, ...]:
    if isinstance(values, Mapping):
        missing = [name for name in channel.edge_names if name not in values]
        extra = sorted(set(values) - set(channel.edge_names))
        if missing or extra:
            raise ValueError(
                "edge_weights keys do not match channel edges: "
                f"missing={missing}, extra={extra}"
            )
        ordered = tuple(complex(values[name]) for name in channel.edge_names)
    else:
        if len(values) != len(channel.edge_names):
            raise ValueError(
                f"edge_weights must contain {len(channel.edge_names)} entries"
            )
        ordered = tuple(complex(value) for value in values)
    if any(
        not math.isfinite(value.real) or not math.isfinite(value.imag)
        for value in ordered
    ):
        raise ValueError("edge_weights must be finite")
    return ordered


def _validate_controls(
    *,
    tolerance: float,
    minimum_cap: int,
    maximum_cap: int,
    cap_step: int,
) -> tuple[float, int, int, int]:
    tolerance = float(tolerance)
    minimum_cap = int(minimum_cap)
    maximum_cap = int(maximum_cap)
    cap_step = int(cap_step)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("global-block tolerance must lie strictly between zero and one")
    if minimum_cap < 0 or maximum_cap < minimum_cap:
        raise ValueError("global-block caps must satisfy 0 <= minimum <= maximum")
    if cap_step <= 0:
        raise ValueError("global-block cap_step must be positive")
    if maximum_cap - minimum_cap < 2 * cap_step:
        raise ValueError(
            "global-block cap range must contain at least three successive caps"
        )
    return tolerance, minimum_cap, maximum_cap, cap_step


@lru_cache(maxsize=8192)
def _propagator_table(weight: complex, q_value: complex, cap: int) -> np.ndarray:
    """Return ``q**n / (n! (2h)_n)`` without forming factorials."""

    result = np.empty(cap + 1, dtype=np.complex128)
    result[0] = 1.0 + 0.0j
    for level in range(1, cap + 1):
        denominator = level * (2.0 * weight + level - 1.0)
        if denominator == 0.0:
            raise ZeroDivisionError("global descendant norm is singular")
        result[level] = result[level - 1] * q_value / denominator
    result.setflags(write=False)
    return result


@lru_cache(maxsize=65536)
def _rho_two_edge_table(
    h_infinity: complex,
    h_one: complex,
    h_zero: complex,
    cap: int,
) -> np.ndarray:
    """Return ``rho(i,0,k)`` for all ``0 <= i,k <= cap``."""

    dimension = cap + 1
    result = np.empty((dimension, dimension), dtype=np.complex128)
    for infinity_level in range(dimension):
        for zero_level in range(dimension):
            result[infinity_level, zero_level] = rho_lminus1_two_edge(
                infinity_level,
                zero_level,
                h_infinity,
                h_one,
                h_zero,
            )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=4096)
def _rho_tensor(
    h_infinity: complex,
    h_one: complex,
    h_zero: complex,
    cap: int,
) -> np.ndarray:
    r"""Return the full ``rho(i,j,k)`` tensor by a middle-slot recurrence."""

    base = _rho_two_edge_table(h_infinity, h_one, h_zero, cap)
    dimension = cap + 1
    result = np.empty((dimension, dimension, dimension), dtype=np.complex128)
    result[:, 0, :] = base
    infinity_levels = np.arange(dimension, dtype=np.float64)[:, None]
    zero_levels = np.arange(dimension, dtype=np.float64)[None, :]
    middle_parameter = (
        h_one + h_zero + zero_levels - h_infinity - infinity_levels
    )
    prefactor = np.ones((dimension, dimension), dtype=np.complex128)
    for middle_level in range(1, dimension):
        prefactor *= -(middle_parameter + middle_level - 1.0)
        result[:, middle_level, :] = base * prefactor
    result.setflags(write=False)
    return result


def _vectorized_gauss_2f1(
    a: np.ndarray,
    b: np.ndarray,
    c: complex,
    z: complex,
    *,
    tolerance: float,
    max_terms: int = 10000,
) -> np.ndarray:
    """Evaluate a broadcast array of Gauss series in ``|z| < 1``."""

    if abs(z) >= 1.0:
        raise ValueError("the plumbing-frame Gauss series requires |z| < 1")
    shape = np.broadcast_shapes(a.shape, b.shape)
    total = np.ones(shape, dtype=np.complex128)
    if z == 0.0:
        return total
    term = np.ones(shape, dtype=np.complex128)
    converged_terms = 0
    for level in range(1, int(max_terms) + 1):
        denominator = (c + level - 1.0) * level
        if denominator == 0.0:
            raise ZeroDivisionError("2F1 has a singular lower parameter")
        term *= (
            (a + level - 1.0)
            * (b + level - 1.0)
            * z
            / denominator
        )
        total += term
        if not np.all(np.isfinite(total)) or not np.all(np.isfinite(term)):
            raise ArithmeticError("vectorized 2F1 produced a non-finite value")
        converged = np.all(
            np.abs(term) <= tolerance * np.maximum(1.0, np.abs(total))
        )
        converged_terms = converged_terms + 1 if converged else 0
        if converged_terms >= 3:
            return total
    raise ArithmeticError(
        f"vectorized 2F1 did not converge in {max_terms} terms at z={z!r}"
    )


def _opposite_cycle_value_at_cap(
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    cap: int,
) -> complex:
    """Contract the two native double-edge kernels at one common cap."""

    # Edge order: q02=a, q03_1=b, q03_2=c, q12_1=d, q12_2=e, q13=f.
    h_a, h_b, h_c, h_d, h_e, h_f = weights
    q_a, q_b, q_c, q_d, q_e, q_f = q_values
    p_a, p_b, p_c, p_d, p_e, p_f = (
        _propagator_table(weight, q_value, cap)
        for weight, q_value in zip(weights, q_values)
    )

    vertex_0 = _rho_tensor(h_c, h_b, h_a, cap)  # (c,b,a)
    vertex_1 = _rho_tensor(h_f, h_e, h_d, cap)  # (f,e,d)
    vertex_2 = _rho_tensor(h_e, h_d, h_a, cap)  # (e,d,a)
    vertex_3 = _rho_tensor(h_f, h_c, h_b, cap)  # (f,c,b)

    kernel_03 = np.einsum(
        "cba,fcb,b,c->af",
        vertex_0,
        vertex_3,
        p_b,
        p_c,
        optimize="greedy",
    )
    kernel_12 = np.einsum(
        "fed,eda,d,e->af",
        vertex_1,
        vertex_2,
        p_d,
        p_e,
        optimize="greedy",
    )
    value = np.einsum(
        "af,af,a,f->",
        kernel_03,
        kernel_12,
        p_a,
        p_f,
        optimize="greedy",
    )
    return complex(value)


def _one_tadpole_double_triangle_value_at_cap(
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    cap: int,
) -> complex:
    """Contract one loop kernel and the doubled 0--3 edge pair."""

    # Edge order: q02=a, q03_1=b, q03_2=c, q11=d, q12=e, q23=f.
    h_a, h_b, h_c, h_d, h_e, h_f = weights
    p_a, p_b, p_c, p_d, p_e, p_f = (
        _propagator_table(weight, q_value, cap)
        for weight, q_value in zip(weights, q_values)
    )
    vertex_0 = _rho_tensor(h_c, h_b, h_a, cap)  # (c,b,a)
    vertex_1 = _rho_tensor(h_e, h_d, h_d, cap)  # (e,d,d)
    vertex_2 = _rho_tensor(h_f, h_e, h_a, cap)  # (f,e,a)
    vertex_3 = _rho_tensor(h_f, h_c, h_b, cap)  # (f,c,b)

    loop_11 = np.einsum("edd,d->e", vertex_1, p_d, optimize="greedy")
    kernel_03 = np.einsum(
        "cba,fcb,b,c->af",
        vertex_0,
        vertex_3,
        p_b,
        p_c,
        optimize="greedy",
    )
    value = np.einsum(
        "af,fea,e,a,e,f->",
        kernel_03,
        vertex_2,
        loop_11,
        p_a,
        p_e,
        p_f,
        optimize="greedy",
    )
    return complex(value)


def _three_tadpole_star_value_at_cap(
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    cap: int,
) -> complex:
    """Reduce all three tadpoles to vectors around the central trinion."""

    # Edge order: q01=a, q02=b, q03=c, q11=d, q22=e, q33=f.
    h_a, h_b, h_c, h_d, h_e, h_f = weights
    p_a, p_b, p_c, p_d, p_e, p_f = (
        _propagator_table(weight, q_value, cap)
        for weight, q_value in zip(weights, q_values)
    )
    vertex_0 = _rho_tensor(h_c, h_b, h_a, cap)  # (c,b,a)
    vertex_1 = _rho_tensor(h_d, h_d, h_a, cap)  # (d,d,a)
    vertex_2 = _rho_tensor(h_e, h_e, h_b, cap)  # (e,e,b)
    vertex_3 = _rho_tensor(h_f, h_f, h_c, cap)  # (f,f,c)

    loop_11 = np.einsum("dda,d->a", vertex_1, p_d, optimize="greedy")
    loop_22 = np.einsum("eeb,e->b", vertex_2, p_e, optimize="greedy")
    loop_33 = np.einsum("ffc,f->c", vertex_3, p_f, optimize="greedy")
    value = np.einsum(
        "cba,a,b,c,a,b,c->",
        vertex_0,
        loop_11,
        loop_22,
        loop_33,
        p_a,
        p_b,
        p_c,
        optimize="greedy",
    )
    return complex(value)


def _two_tadpoles_double_bridge_value_at_cap(
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    cap: int,
) -> complex:
    """Contract two loop vectors through the doubled 0--3 edge pair."""

    # Edge order: q02=a, q03_1=b, q03_2=c, q11=d, q13=e, q22=f.
    h_a, h_b, h_c, h_d, h_e, h_f = weights
    p_a, p_b, p_c, p_d, p_e, p_f = (
        _propagator_table(weight, q_value, cap)
        for weight, q_value in zip(weights, q_values)
    )
    vertex_0 = _rho_tensor(h_c, h_b, h_a, cap)  # (c,b,a)
    vertex_1 = _rho_tensor(h_e, h_d, h_d, cap)  # (e,d,d)
    vertex_2 = _rho_tensor(h_f, h_f, h_a, cap)  # (f,f,a)
    vertex_3 = _rho_tensor(h_e, h_c, h_b, cap)  # (e,c,b)

    loop_11 = np.einsum("edd,d->e", vertex_1, p_d, optimize="greedy")
    loop_22 = np.einsum("ffa,f->a", vertex_2, p_f, optimize="greedy")
    kernel_03 = np.einsum(
        "cba,ecb,b,c->ae",
        vertex_0,
        vertex_3,
        p_b,
        p_c,
        optimize="greedy",
    )
    value = np.einsum(
        "ae,a,e,a,e->",
        kernel_03,
        loop_22,
        loop_11,
        p_a,
        p_e,
        optimize="greedy",
    )
    return complex(value)


def _tetrahedron_value_at_cap(
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    cap: int,
    *,
    hypergeometric_tolerance: float,
) -> complex:
    r"""Contract K4 while summing the complete ``q12`` tower as ``2F1``."""

    # Edge order: q01=a, q02=b, q03=c, q12=d, q13=e, q23=f.
    h_a, h_b, h_c, h_d, h_e, h_f = weights
    q_a, q_b, q_c, _q_d, q_e, q_f = q_values
    p_a = _propagator_table(h_a, q_a, cap)
    p_b = _propagator_table(h_b, q_b, cap)
    p_c = _propagator_table(h_c, q_c, cap)
    p_e = _propagator_table(h_e, q_e, cap)
    p_f = _propagator_table(h_f, q_f, cap)

    vertex_0 = _rho_tensor(h_c, h_b, h_a, cap)  # (c,b,a)
    vertex_3 = _rho_tensor(h_f, h_e, h_c, cap)  # (f,e,c)
    rho_1 = _rho_two_edge_table(h_e, h_d, h_a, cap)  # (e,a)
    rho_2 = _rho_two_edge_table(h_f, h_d, h_b, cap)  # (f,b)

    levels = np.arange(cap + 1, dtype=np.float64)
    parameter_1 = (
        h_d
        + h_a
        + levels[None, :]
        - h_e
        - levels[:, None]
    )  # (e,a)
    parameter_2 = (
        h_d
        + h_b
        + levels[None, :]
        - h_f
        - levels[:, None]
    )  # (f,b)
    hypergeometric = _vectorized_gauss_2f1(
        parameter_1[:, :, None, None],
        parameter_2[None, None, :, :],
        2.0 * h_d,
        q_values[3],
        tolerance=hypergeometric_tolerance,
    )  # (e,a,f,b)

    bridge = np.einsum(
        "cba,c,fec->abef",
        vertex_0,
        p_c,
        vertex_3,
        optimize="greedy",
    )
    value = np.einsum(
        "abef,ea,fb,eafb,a,b,e,f->",
        bridge,
        rho_1,
        rho_2,
        hypergeometric,
        p_a,
        p_b,
        p_e,
        p_f,
        optimize="greedy",
    )
    return complex(value)


@lru_cache(maxsize=65536)
def _resummed_cached(
    channel_name: str,
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    tolerance: float,
    minimum_cap: int,
    maximum_cap: int,
    cap_step: int,
) -> Genus3ResummedGlobalBlockResult:
    if channel_name == "one-tadpole-double-triangle":
        evaluator = lambda cap: _one_tadpole_double_triangle_value_at_cap(
            weights,
            q_values,
            cap,
        )
        method = "one-loop-double-edge-kernel-adaptive"
        exact_edges: tuple[str, ...] = ()
    elif channel_name == "opposite-double-edge-cycle":
        evaluator = lambda cap: _opposite_cycle_value_at_cap(weights, q_values, cap)
        method = "double-edge-kernel-adaptive"
        exact_edges = ()
    elif channel_name == "tetrahedron":
        hypergeometric_tolerance = min(1.0e-13, 0.01 * tolerance)
        evaluator = lambda cap: _tetrahedron_value_at_cap(
            weights,
            q_values,
            cap,
            hypergeometric_tolerance=hypergeometric_tolerance,
        )
        method = "q12-2f1-five-edge-adaptive"
        exact_edges = ("q12",)
    elif channel_name == "three-tadpole-star":
        evaluator = lambda cap: _three_tadpole_star_value_at_cap(
            weights,
            q_values,
            cap,
        )
        method = "three-loop-star-adaptive"
        exact_edges = ()
    elif channel_name == "two-tadpoles-double-bridge":
        evaluator = lambda cap: _two_tadpoles_double_bridge_value_at_cap(
            weights,
            q_values,
            cap,
        )
        method = "two-loop-double-edge-kernel-adaptive"
        exact_edges = ()
    else:  # pragma: no cover - guarded by _resolve_channel
        raise AssertionError(f"unsupported resummed channel {channel_name!r}")

    previous: complex | None = None
    converged_steps = 0
    relative_tail = math.inf
    for cap in range(minimum_cap, maximum_cap + 1, cap_step):
        value = evaluator(cap)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ArithmeticError("resummed global block is non-finite")
        if previous is not None:
            relative_tail = abs(value - previous) / max(abs(value), 1.0e-300)
            converged_steps = converged_steps + 1 if relative_tail <= tolerance else 0
            if converged_steps >= 2:
                return Genus3ResummedGlobalBlockResult(
                    value=value,
                    channel=channel_name,
                    method=method,
                    cap=cap,
                    relative_tail_estimate=float(relative_tail),
                    tolerance=tolerance,
                    exact_resummed_edges=exact_edges,
                )
        previous = value
    raise ArithmeticError(
        f"{channel_name} resummed global block did not converge through cap "
        f"{maximum_cap}; final relative step={relative_tail:.6e}, "
        f"tolerance={tolerance:.6e}"
    )


def genus3_channel_global_sl2_block_resummed(
    channel: str | Genus3PlumbingChannel,
    *,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    tolerance: float = 1.0e-9,
    minimum_cap: int = 8,
    maximum_cap: int = 24,
    cap_step: int = 2,
) -> Genus3ResummedGlobalBlockResult:
    """Return the best native-frame global-block resummation for one channel."""

    resolved = _resolve_channel(channel)
    weights = _edge_weights(resolved, edge_weights)
    q_tuple = genus3_channel_q_values(resolved, q_values)
    if any(
        not math.isfinite(value.real)
        or not math.isfinite(value.imag)
        or abs(value) >= 1.0
        for value in q_tuple
    ):
        raise ValueError("q_values must be finite and satisfy |q| < 1")
    controls = _validate_controls(
        tolerance=tolerance,
        minimum_cap=minimum_cap,
        maximum_cap=maximum_cap,
        cap_step=cap_step,
    )
    return _resummed_cached(
        resolved.name,
        weights,
        q_tuple,
        *controls,
    )


def clear_genus3_global_resummation_caches() -> None:
    """Release per-block tensor caches before moving to new momenta."""

    _resummed_cached.cache_clear()
    _rho_tensor.cache_clear()
    _rho_two_edge_table.cache_clear()
    _propagator_table.cache_clear()


__all__ = [
    "Genus3ResummedGlobalBlockResult",
    "SUPPORTED_CHANNELS",
    "clear_genus3_global_resummation_caches",
    "genus3_channel_global_sl2_block_resummed",
]
