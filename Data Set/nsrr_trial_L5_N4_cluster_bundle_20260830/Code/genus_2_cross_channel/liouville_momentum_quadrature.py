#!/usr/bin/env python3
"""Momentum quadrature helpers for Liouville plumbing integrals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.special import roots_genlaguerre


@dataclass(frozen=True)
class MomentumQuadratureRule:
    """Quadrature rule for one Liouville momentum edge."""

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    split_point: float | None
    gaussian_width: float


def normalize_quadrature_orders(
    q_values: Sequence[complex],
    quadrature_order: int | Sequence[int],
) -> tuple[int, ...]:
    """Return one positive quadrature order per plumbing edge."""

    edge_count = len(q_values)
    if isinstance(quadrature_order, Integral):
        orders = (int(quadrature_order),) * edge_count
    else:
        orders = tuple(int(order) for order in quadrature_order)
        if len(orders) != edge_count:
            raise ValueError(
                f"expected {edge_count} edge quadrature orders, got {len(orders)}"
            )
    if any(order <= 0 for order in orders):
        raise ValueError("quadrature orders must be positive")
    return orders


def gaussian_width_from_q(q_value: complex, *, log_q_abs: float | None = None) -> float:
    """Return the near-threshold width from ``exp(-2 |log q| P^2)``."""

    if log_q_abs is not None:
        log_q_abs = float(log_q_abs)
        if not math.isfinite(log_q_abs) or log_q_abs >= 0.0:
            raise ValueError("log|q| must be finite and negative")
        return float(1.0 / math.sqrt(-2.0 * log_q_abs))
    q_abs = abs(complex(q_value))
    if not 0.0 < q_abs < 1.0:
        raise ValueError("momentum width requires 0<|q|<1")
    return float(1.0 / math.sqrt(2.0 * abs(math.log(q_abs))))


def _interval_rule(start: float, stop: float, order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    if stop <= start:
        return (), ()
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    midpoint = 0.5 * (float(start) + float(stop))
    half_width = 0.5 * (float(stop) - float(start))
    mapped_nodes = tuple(float(midpoint + half_width * node) for node in nodes)
    mapped_weights = tuple(float(half_width * weight / math.pi) for weight in weights)
    return mapped_nodes, mapped_weights


def uniform_momentum_rule(p_max: float, quadrature_order: int) -> MomentumQuadratureRule:
    """Return the original global Gauss rule on ``[0, p_max]``."""

    if p_max <= 0.0:
        raise ValueError("p_max must be positive")
    nodes, weights = _interval_rule(0.0, float(p_max), int(quadrature_order))
    return MomentumQuadratureRule(
        nodes=nodes,
        weights=weights,
        split_point=None,
        gaussian_width=float("nan"),
    )


def edge_scaled_momentum_rule(
    q_value: complex,
    p_max: float,
    near_order: int,
    *,
    tail_order: int | None = None,
    split_widths: float = 4.0,
    log_q_abs: float | None = None,
) -> MomentumQuadratureRule:
    """Return a two-piece rule adapted to the edge's primary propagator.

    The primary part of the nonchiral integrand contains
    ``exp(-2 |log |q|| P^2)``.  For very small plumbing parameters, most of the
    mass sits close to ``P=0``; a single low-order Gauss rule on ``[0,p_max]``
    can miss it.  This rule places one interval on ``[0, split]`` with
    ``split = split_widths / sqrt(2 |log |q||)`` and a coarser tail interval.
    """

    if p_max <= 0.0:
        raise ValueError("p_max must be positive")
    if split_widths <= 0.0:
        raise ValueError("split_widths must be positive")
    width = gaussian_width_from_q(q_value, log_q_abs=log_q_abs)
    split = min(float(p_max), float(split_widths) * width)
    if split >= float(p_max) * (1.0 - 1.0e-14):
        nodes, weights = _interval_rule(0.0, float(p_max), int(near_order))
        return MomentumQuadratureRule(nodes=nodes, weights=weights, split_point=None, gaussian_width=width)

    if tail_order is None:
        tail_order = max(2, int(near_order) // 2)
    near_nodes, near_weights = _interval_rule(0.0, split, int(near_order))
    tail_nodes, tail_weights = _interval_rule(split, float(p_max), int(tail_order))
    return MomentumQuadratureRule(
        nodes=near_nodes + tail_nodes,
        weights=near_weights + tail_weights,
        split_point=float(split),
        gaussian_width=width,
    )


def primary_gaussian_momentum_rule(
    q_value: complex,
    quadrature_order: int,
    *,
    log_q_abs: float | None = None,
) -> MomentumQuadratureRule:
    r"""Return an infinite-interval rule matched to the primary propagator.

    With ``sigma = 1/sqrt(2 |log |q||)`` and ``P = sigma sqrt(u)``, the
    nonchiral primary propagator contributes ``exp(-u)``.  Generalized
    Gauss-Laguerre quadrature with ``alpha=-1/2`` therefore integrates the
    narrow threshold Gaussian directly on ``[0, infinity)``.  The returned
    weights include ``exp(u)`` so callers continue to pass the complete
    integrand, including its primary propagator.
    """

    if quadrature_order <= 0:
        raise ValueError("quadrature order must be positive")
    width = gaussian_width_from_q(q_value, log_q_abs=log_q_abs)
    u_nodes, u_weights = roots_genlaguerre(int(quadrature_order), -0.5)
    nodes = tuple(float(width * math.sqrt(float(node))) for node in u_nodes)
    weights = tuple(
        float(width * float(weight) * math.exp(float(node)) / (2.0 * math.pi))
        for node, weight in zip(u_nodes, u_weights)
    )
    return MomentumQuadratureRule(
        nodes=nodes,
        weights=weights,
        split_point=None,
        gaussian_width=width,
    )


def momentum_quadrature_rules(
    q_values: Sequence[complex],
    *,
    p_max: float,
    quadrature_order: int | Sequence[int],
    quadrature_scheme: str = "uniform",
    tail_order: int | None = None,
    split_widths: float = 4.0,
    log_q_abs_values: Sequence[float] | None = None,
) -> tuple[MomentumQuadratureRule, ...]:
    """Return one momentum quadrature rule per plumbing edge."""

    orders = normalize_quadrature_orders(q_values, quadrature_order)
    if log_q_abs_values is None:
        log_abs_values: tuple[float | None, ...] = (None,) * len(q_values)
    else:
        supplied = tuple(float(value) for value in log_q_abs_values)
        if len(supplied) != len(q_values):
            raise ValueError(
                f"expected {len(q_values)} log|q| values, got {len(supplied)}"
            )
        log_abs_values = supplied

    if quadrature_scheme == "uniform":
        return tuple(
            uniform_momentum_rule(float(p_max), order) for order in orders
        )
    if quadrature_scheme == "edge-scaled":
        return tuple(
            edge_scaled_momentum_rule(
                value,
                float(p_max),
                order,
                tail_order=tail_order,
                split_widths=split_widths,
                log_q_abs=log_q_abs,
            )
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    if quadrature_scheme == "primary-gaussian":
        return tuple(
            primary_gaussian_momentum_rule(value, order, log_q_abs=log_q_abs)
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    raise ValueError(f"unsupported quadrature scheme {quadrature_scheme!r}")
