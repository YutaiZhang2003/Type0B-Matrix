#!/usr/bin/env python3
"""Checks for Liouville momentum quadrature helpers."""

from __future__ import annotations

import math

try:
    from liouville_momentum_quadrature import (
        edge_scaled_momentum_rule,
        cft_asymptotic_momentum_rule,
        gaussian_width_from_q,
        momentum_quadrature_rules,
        primary_gaussian_momentum_rule,
        threshold_polar_momentum_pair_rule,
        threshold_gaussian_momentum_rule,
        uniform_momentum_rule,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.liouville_momentum_quadrature import (
        edge_scaled_momentum_rule,
        cft_asymptotic_momentum_rule,
        gaussian_width_from_q,
        momentum_quadrature_rules,
        primary_gaussian_momentum_rule,
        threshold_polar_momentum_pair_rule,
        threshold_gaussian_momentum_rule,
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
    _require(
        rule.momentum_domain == "[0,p_max]"
        and rule.p_max_used
        and rule.p_max == p_max,
        "uniform rule lost its active cutoff metadata",
    )


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
    _require(
        rule.momentum_domain == "[0,infinity)"
        and not rule.p_max_used
        and rule.p_max is None,
        "primary-Gaussian rule falsely reports a finite cutoff",
    )
    _require(abs(gaussian / exact_gaussian - 1.0) < 1.0e-14, "primary-Gaussian rule missed its weight")
    _require(abs(quadratic / exact_quadratic - 1.0) < 1.0e-14, "primary-Gaussian rule missed a moment")


def check_cft_adapted_rules() -> None:
    q_value = 0.3
    primary_coefficient = 2.0 * abs(math.log(q_value))
    threshold = threshold_gaussian_momentum_rule(q_value, 4)
    threshold_quadratic = sum(
        weight * node * node * math.exp(-primary_coefficient * node * node)
        for node, weight in zip(threshold.nodes, threshold.weights)
    )
    exact_quadratic = 1.0 / (
        4.0 * math.sqrt(math.pi) * primary_coefficient**1.5
    )
    _require(
        abs(threshold_quadratic / exact_quadratic - 1.0) < 1.0e-14,
        "threshold-Gaussian rule missed its P^2-weighted Gaussian",
    )
    _require(
        threshold.laguerre_alpha == 0.5,
        "threshold-Gaussian rule did not record alpha=+1/2",
    )

    adapted = cft_asymptotic_momentum_rule(q_value, 4)
    coefficient = float(adapted.effective_decay_coefficient)
    adapted_quadratic = sum(
        weight * node * node * math.exp(-coefficient * node * node)
        for node, weight in zip(adapted.nodes, adapted.weights)
    )
    exact_adapted = 1.0 / (4.0 * math.sqrt(math.pi) * coefficient**1.5)
    _require(
        abs(adapted_quadratic / exact_adapted - 1.0) < 1.0e-14,
        "CFT-asymptotic rule missed its adapted Gaussian",
    )
    _require(
        float(adapted.effective_q_abs) > q_value
        and coefficient < primary_coefficient
        and not adapted.asymptotic_tempered,
        "CFT-asymptotic rule did not broaden the positive-q tail",
    )

    tempered = cft_asymptotic_momentum_rule(0.85, 4)
    _require(
        tempered.asymptotic_tempered
        and math.isclose(
            float(tempered.effective_decay_coefficient),
            0.75 * 2.0 * abs(math.log(0.85)),
        ),
        "CFT-asymptotic decay floor was not applied",
    )


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
    _require(
        all(
            rule.momentum_domain == "[0,infinity)"
            and not rule.p_max_used
            and rule.p_max is None
            for rule in rules
        ),
        "anisotropic primary-Gaussian rules falsely use p_max",
    )


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


def check_threshold_polar_pair_rule() -> None:
    q_first = 0.71
    q_second = 0.83
    first_decay = -2.0 * math.log(q_first)
    second_decay = -2.0 * math.log(q_second)
    rule = threshold_polar_momentum_pair_rule(
        q_first,
        q_second,
        radial_order=3,
        angular_order=3,
    )
    value = sum(
        weight
        * first * first
        * second * second
        * math.exp(-first_decay * first * first - second_decay * second * second)
        for first, second, weight in zip(
            rule.first_nodes, rule.second_nodes, rule.weights
        )
    )
    exact = 1.0 / (
        16.0
        * math.pi
        * first_decay**1.5
        * second_decay**1.5
    )
    print("\nthreshold-polar correlated pair rule")
    print(f"  node count={len(rule.weights)}")
    print(f"  relative error={abs(value / exact - 1.0):.6e}")
    _require(
        abs(value / exact - 1.0) < 2.0e-14,
        "threshold-polar rule missed the two-edge threshold Gaussian",
    )
    _require(
        rule.radial_laguerre_alpha == 2.0
        and rule.angular_jacobi_alpha == 0.5
        and rule.angular_jacobi_beta == 0.5,
        "threshold-polar rule recorded the wrong reference weights",
    )


def main() -> None:
    check_tiny_q_gaussian_peak()
    check_uniform_rule_normalization()
    check_primary_gaussian_rule()
    check_cft_adapted_rules()
    check_anisotropic_orders()
    check_logarithmic_underflow_edge()
    check_threshold_polar_pair_rule()
    print("\nliouville momentum quadrature checks passed")


if __name__ == "__main__":
    main()
