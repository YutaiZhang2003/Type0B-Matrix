#!/usr/bin/env python3
"""Checks for the CCY plumbing-frame sewing convention."""

from __future__ import annotations

import cmath

try:
    from ccy_plumbing_conventions import (
        ccy_primary_propagator,
        ccy_raw_sewing_propagator,
        liouville_threshold_modulus_factor,
        liouville_threshold_weight,
    )
    from liouville_genus2_ccy import liouville_genus2_ccy_partition, liouville_weight_from_momentum
    from liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing_algorithms import local_coordinate_map, plumbing_transition
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_plumbing_conventions import (
        ccy_primary_propagator,
        ccy_raw_sewing_propagator,
        liouville_threshold_modulus_factor,
        liouville_threshold_weight,
    )
    from plumbing.liouville_genus2_ccy import liouville_genus2_ccy_partition, liouville_weight_from_momentum
    from plumbing.liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing.plumbing_algorithms import local_coordinate_map, plumbing_transition


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_primary_propagator_matches_liouville_sample() -> None:
    b = 1.0
    q_values = (0.03 + 0.01j, 0.025 - 0.004j, 0.012 + 0.002j)
    result = liouville_genus2_ccy_partition(
        b=b,
        q1=q_values[0],
        q2=q_values[1],
        q3=q_values[2],
        block_order=0,
        p_max=0.7,
        quadrature_order=1,
        dps=22,
        include_vacuum_seed=False,
        store_samples=True,
    )
    sample = result.samples[0]
    weights = (
        liouville_weight_from_momentum(b, sample.p1),
        liouville_weight_from_momentum(b, sample.p2),
        liouville_weight_from_momentum(b, sample.p3),
    )
    expected = ccy_primary_propagator(q_values, weights)
    relative_error = abs(sample.propagator - expected) / max(abs(expected), 1.0e-300)

    print("CCY separated primary propagator")
    print(f"  sample propagator={sample.propagator!r}")
    print(f"  expected product={expected!r}")
    print(f"  relative error={relative_error:.6e}")
    require(relative_error < 1.0e-14, "Liouville wrapper is not using the CCY q^h primary propagator")


def check_literal_plumbing_coordinate() -> None:
    """Check that geometry and CCY propagation use the same ``u v=q``."""

    q_value = 0.013 - 0.004j
    source_global = 0.27 + 0.19j
    for source, target in (
        ("zero", "one"),
        ("one", "infty"),
        ("infty", "zero"),
    ):
        source_local = local_coordinate_map(source)(source_global)
        target_global = plumbing_transition(source, target, q_value)(source_global)
        target_local = local_coordinate_map(target)(target_global)
        require(
            abs(source_local * target_local / q_value - 1.0) < 2.0e-14,
            f"{source}->{target} transition does not satisfy u v=q",
        )

    q_values = (0.011, 0.009, 0.007)
    weights_a = (1.0, 1.0, 1.0)
    weights_b = (1.17, 1.31, 1.43)
    ratio_four_threshold = ccy_primary_propagator(
        tuple(4.0 * q for q in q_values),
        weights_a,
    ) / ccy_primary_propagator(q_values, weights_a)
    ratio_four_generic = ccy_primary_propagator(
        tuple(4.0 * q for q in q_values),
        weights_b,
    ) / ccy_primary_propagator(q_values, weights_b)
    ratio_sixteen_threshold = ccy_primary_propagator(
        tuple(16.0 * q for q in q_values),
        weights_a,
    ) / ccy_primary_propagator(q_values, weights_a)

    print("\nliteral CCY plumbing coordinate")
    print(
        "  nonchiral threshold ratio under q->4q  = "
        f"{abs(ratio_four_threshold) ** 2:.12g}"
    )
    print(
        "  nonchiral generic-weight ratio q->4q   = "
        f"{abs(ratio_four_generic) ** 2:.12g}"
    )
    print(
        "  nonchiral threshold ratio under q->16q = "
        f"{abs(ratio_sixteen_threshold) ** 2:.12g}"
    )
    require(
        abs(abs(ratio_four_threshold) ** 2 / (2.0**12) - 1.0) < 1.0e-14,
        "a hypothetical q->4q on three threshold edges should expose 2^12",
    )
    require(
        abs(abs(ratio_sixteen_threshold) ** 2 / (2.0**24) - 1.0) < 1.0e-14,
        "q->16q would give 2^24, not the observed nonchiral factor",
    )
    require(
        abs(abs(ratio_four_generic / ratio_four_threshold) ** 2 - 1.0) > 1.0,
        "q->4q should be visibly momentum dependent away from threshold",
    )


def check_diagnostic_shift_is_extra_factor() -> None:
    q_values = (0.15 + 0.0j, 0.04 + 0.01j, 0.03 - 0.005j)
    weights = (1.2, 1.7, 2.1)
    shift = 0.9
    raw = ccy_raw_sewing_propagator(q_values, weights)
    shifted = ccy_raw_sewing_propagator(q_values, weights, diagnostic_shift=shift)
    expected_ratio = 1.0 + 0.0j
    for q_value in q_values:
        expected_ratio *= q_value ** (-shift)
    relative_error = abs((shifted / raw) - expected_ratio) / max(abs(expected_ratio), 1.0e-300)

    print("\nCCY diagnostic shift bookkeeping")
    print(f"  relative error={relative_error:.6e}")
    require(relative_error < 1.0e-14, "diagnostic shift is not a pure extra factor on top of CCY sewing")


def check_liouville_threshold_factor() -> None:
    q_values = (0.15 + 0.0j, 0.15 + 0.0j, 0.15 + 0.0j)
    threshold = liouville_threshold_weight(1.0)
    factor = liouville_threshold_modulus_factor(q_values, b=1.0)
    expected = abs(q_values[0] * q_values[1] * q_values[2]) ** (2.0 * threshold)

    print("\nLiouville threshold factor")
    print(f"  Q^2/4={threshold:.12g}")
    print(f"  factor={factor:.12e}")
    require(threshold == 1.0, "b=1 Liouville threshold should be one")
    require(abs(factor - expected) < 1.0e-18, "threshold modulus factor does not match |prod q^(Q^2/4)|^2")


def check_logarithmic_underflow_propagator() -> None:
    logs = (-3.0 + 0.2j, -3606.0 + 0.4j, -4.0 - 0.3j)
    q_values = tuple(cmath.exp(complex(max(value.real, -690.0), value.imag)) for value in logs)
    weights = (1.02, 1.001, 1.03)
    got = ccy_raw_sewing_propagator(
        q_values,
        weights,
        diagnostic_shift=1.0,
        log_q_values=logs,
    )
    expected = cmath.exp(sum((weight - 1.0) * log_q for weight, log_q in zip(weights, logs)))
    require(abs(got / expected - 1.0) < 1.0e-14, "logarithmic propagator used the surrogate q")


def check_complex_dozz_phase_is_preserved() -> None:
    common = dict(
        b=0.9,
        block_order=0,
        p_max=0.6,
        quadrature_order=1,
        dps=22,
        mu=0.8 + 0.3j,
        include_cosmological_prefactor=True,
        include_vacuum_seed=False,
        store_samples=True,
    )
    theta = liouville_genus2_ccy_partition(
        **common,
        q1=0.025 + 0.003j,
        q2=0.021 - 0.002j,
        q3=0.018 + 0.001j,
    )
    theta_sample = theta.samples[0]
    theta_expected = (
        theta_sample.measure_weight
        * (theta_sample.structure_constant**2)
        * abs(theta_sample.propagator * theta_sample.block) ** 2
    )
    theta_abs_weight = (
        theta_sample.measure_weight
        * abs(theta_sample.structure_constant) ** 2
        * abs(theta_sample.propagator * theta_sample.block) ** 2
    )

    glasses = liouville_genus2_glasses_partition(
        **common,
        q_left=0.025 + 0.003j,
        q_right=0.021 - 0.002j,
        q_bridge=0.018 + 0.001j,
    )
    glasses_sample = glasses.samples[0]
    glasses_expected = (
        glasses_sample.measure_weight
        * glasses_sample.structure_left
        * glasses_sample.structure_right
        * abs(glasses_sample.propagator * glasses_sample.block) ** 2
    )
    glasses_abs_weight = (
        glasses_sample.measure_weight
        * abs(glasses_sample.structure_left * glasses_sample.structure_right)
        * abs(glasses_sample.propagator * glasses_sample.block) ** 2
    )

    print("\ncomplex DOZZ phase preservation")
    print(f"  theta contribution={theta_sample.contribution!r}")
    print(f"  glasses contribution={glasses_sample.contribution!r}")
    require(abs(theta_sample.contribution - theta_expected) < 1.0e-28, "theta wrapper dropped the DOZZ phase")
    require(abs(glasses_sample.contribution - glasses_expected) < 1.0e-28, "glasses wrapper dropped the DOZZ phase")
    require(
        abs(theta_expected - theta_abs_weight) > 1.0e-20,
        "theta complex-mu sample did not produce a visible phase test",
    )
    require(
        abs(glasses_expected - glasses_abs_weight) > 1.0e-20,
        "glasses complex-mu sample did not produce a visible phase test",
    )


def run() -> None:
    check_primary_propagator_matches_liouville_sample()
    check_literal_plumbing_coordinate()
    check_diagnostic_shift_is_extra_factor()
    check_liouville_threshold_factor()
    check_logarithmic_underflow_propagator()
    check_complex_dozz_phase_is_preserved()
    print("\nall CCY plumbing convention checks passed")


if __name__ == "__main__":
    run()
