#!/usr/bin/env python3
"""Checks for Liouville momentum quadrature helpers."""

from __future__ import annotations

import math

try:
    from liouville_momentum_quadrature import (
        edge_scaled_momentum_rule,
        gaussian_width_from_q,
        momentum_quadrature_rules,
        primary_gaussian_momentum_rule,
        uniform_momentum_rule,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.liouville_momentum_quadrature import (
        edge_scaled_momentum_rule,
        gaussian_width_from_q,
        momentum_quadrature_rules,
        primary_gaussian_momentum_rule,
        uniform_momentum_rule,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _gaussian_integral(rule, coefficient: float) -> float:
    return sum(weight * math.exp(-coefficient * node * node) for node, weight in zip(rule.nodes, rule.weights))


def check_tiny_q_gaussian_peak() -> None:
    q_value = 3.0e-9
    p_max = 3.5
    coefficient = 2.0 * abs(math.log(abs(q_value)))
    exact = math.sqrt(math.pi) * math.erf(math.sqrt(coefficient) * p_max) / (2.0 * math.sqrt(coefficient) * math.pi)
    uniform = _gaussian_integral(uniform_momentum_rule(p_max, 3), coefficient)
    edge_scaled = _gaussian_integral(
        edge_scaled_momentum_rule(q_value, p_max, 10, tail_order=5, split_widths=4.0),
        coefficient,
    )
    print("tiny-q Gaussian peak")
    print(f"  width={gaussian_width_from_q(q_value):.12e}")
    print(f"  exact={exact:.12e}")
    print(f"  uniform/order3={uniform:.12e} ({uniform / exact:.6e} of exact)")
    print(f"  edge-scaled/order10+5={edge_scaled:.12e} ({edge_scaled / exact:.6e} of exact)")
    _require(uniform / exact < 0.05, "uniform order-3 unexpectedly resolved the tiny-q peak")
    _require(abs(edge_scaled / exact - 1.0) < 1.0e-7, "edge-scaled rule failed the tiny-q Gaussian model")


def check_uniform_rule_normalization() -> None:
    p_max = 2.75
    rule = uniform_momentum_rule(p_max, 5)
    integral = sum(rule.weights)
    expected = p_max / math.pi
    print("\nuniform rule normalization")
    print(f"  integral={integral:.12e}")
    print(f"  expected={expected:.12e}")
    _require(abs(integral - expected) < 1.0e-14, "uniform rule does not integrate constants")


def check_primary_gaussian_rule() -> None:
    q_value = 3.0e-9
    coefficient = 2.0 * abs(math.log(abs(q_value)))
    rule = primary_gaussian_momentum_rule(q_value, 4)
    gaussian = _gaussian_integral(rule, coefficient)
    exact_gaussian = 1.0 / (2.0 * math.sqrt(math.pi * coefficient))
    quadratic = sum(
        weight * node * node * math.exp(-coefficient * node * node)
        for node, weight in zip(rule.nodes, rule.weights)
    )
    exact_quadratic = 1.0 / (4.0 * math.sqrt(math.pi) * coefficient**1.5)
    print("\nprimary-Gaussian infinite rule")
    print(f"  Gaussian relative error={abs(gaussian / exact_gaussian - 1.0):.6e}")
    print(f"  quadratic relative error={abs(quadratic / exact_quadratic - 1.0):.6e}")
    _require(abs(gaussian / exact_gaussian - 1.0) < 1.0e-14, "primary-Gaussian rule missed its weight")
    _require(abs(quadratic / exact_quadratic - 1.0) < 1.0e-14, "primary-Gaussian rule missed a moment")


def check_anisotropic_orders() -> None:
    rules = momentum_quadrature_rules(
        (1.0e-8, 0.6, 2.0e-7),
        p_max=4.0,
        quadrature_order=(6, 12, 8),
        quadrature_scheme="primary-gaussian",
    )
    node_counts = tuple(len(rule.nodes) for rule in rules)
    print("\nanisotropic primary-Gaussian orders")
    print(f"  node counts={node_counts}")
    _require(node_counts == (6, 12, 8), "edge-specific quadrature orders were not preserved")


def check_logarithmic_underflow_edge() -> None:
    log_q_abs = -3606.0
    surrogate_q = math.exp(-690.0)
    coefficient = -2.0 * log_q_abs
    rule = primary_gaussian_momentum_rule(
        surrogate_q,
        4,
        log_q_abs=log_q_abs,
    )
    gaussian = _gaussian_integral(rule, coefficient)
    exact = 1.0 / (2.0 * math.sqrt(math.pi * coefficient))
    expected_width = 1.0 / math.sqrt(coefficient)
    _require(
        abs(rule.gaussian_width / expected_width - 1.0) < 1.0e-14,
        "logarithmic q did not set the true cusp width",
    )
    _require(
        abs(gaussian / exact - 1.0) < 1.0e-14,
        "logarithmic underflow rule missed its Gaussian weight",
    )


def main() -> None:
    check_tiny_q_gaussian_peak()
    check_uniform_rule_normalization()
    check_primary_gaussian_rule()
    check_anisotropic_orders()
    check_logarithmic_underflow_edge()
    print("\nliouville momentum quadrature checks passed")


if __name__ == "__main__":
    main()
