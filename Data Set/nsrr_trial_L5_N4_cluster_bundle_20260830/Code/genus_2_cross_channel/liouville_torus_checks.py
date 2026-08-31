#!/usr/bin/env python3
"""Checks for Liouville DOZZ data and torus one-point integration."""

from __future__ import annotations

import cmath
import math

try:
    from ccy_plumbing_conventions import liouville_threshold_modulus_factor
    from liouville_torus import (
        LiouvilleTorusOnePointQuadrature,
        UpsilonB,
        dozz_structure_constant_alpha,
        dozz_structure_constant_lambda,
        liouville_torus_one_point,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from liouville_torus_plumbing_modular_check import run_torus_plumbing_modular_check
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_plumbing_conventions import liouville_threshold_modulus_factor
    from plumbing.liouville_torus import (
        LiouvilleTorusOnePointQuadrature,
        UpsilonB,
        dozz_structure_constant_alpha,
        dozz_structure_constant_lambda,
        liouville_torus_one_point,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from plumbing.liouville_torus_plumbing_modular_check import run_torus_plumbing_modular_check


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_upsilon_identities() -> None:
    special = UpsilonB(0.8, dps=30)
    b = special.b
    q_background = special.q_background
    x = 0.73 + 0.21j

    reflection_error = abs(special.upsilon(x) - special.upsilon(q_background - x))
    shift_lhs = special.upsilon(x + b)
    shift_rhs = special.gamma_ratio(b * x) * (b ** (1 - 2 * b * x)) * special.upsilon(x)
    shift_error = abs(shift_lhs - shift_rhs)

    print("Upsilon_b identities")
    print(f"  reflection error: {float(reflection_error):.6e}")
    print(f"  b-shift error:    {float(shift_error):.6e}")
    require(reflection_error < 1e-12, "Upsilon reflection identity failed")
    require(shift_error < 1e-12, "Upsilon b-shift identity failed")


def check_dozz_lambda_matches_alpha() -> None:
    special = UpsilonB(0.8, dps=30)
    q_background = special.q_background
    external_lambda = 0.37 + 0.11j
    internal_lambda = 0.42j

    lambda_value = dozz_structure_constant_lambda(
        special,
        external_lambda,
        internal_lambda,
        include_cosmological_prefactor=False,
    )
    alpha_value = dozz_structure_constant_alpha(
        special,
        0.5 * (q_background + internal_lambda),
        0.5 * (q_background + external_lambda),
        0.5 * (q_background - internal_lambda),
        include_cosmological_prefactor=False,
    )
    relative_error = abs(lambda_value - alpha_value) / max(1.0, abs(alpha_value))

    print("\nDOZZ convention conversion")
    print(f"  relative error: {relative_error:.6e}")
    require(relative_error < 1e-12, "lambda-space DOZZ does not match alpha-space DOZZ")


def check_xi_resonance_normalization() -> None:
    special = UpsilonB(1.0, dps=30)
    p1 = 0.17
    p2 = 0.23
    p3 = p1 + p2
    value = yin_structure_constant_momentum(special, p3, p1, p2)
    expected = (2 * p3) * (2 * p1) * (2 * p2)

    print("\nBalthazar-Rodriguez-Yin b=1 normalization")
    print(f"  C(P1+P2,P1,P2): {value!r}")
    print(f"  expected:        {expected!r}")
    require(abs(value - expected) < 1e-12, "Xi convention resonance normalization failed")


def check_bry_b1_simplified_formula() -> None:
    """Compare the generic implementation directly with BRY equation (2.9)."""

    special = UpsilonB(1.0, dps=40)
    samples = (
        (0.13, 0.29, 0.41),
        (0.07, 0.22, 0.36),
        (0.31, 0.18, 0.12),
        (-0.13, 0.29, 0.41),
        (0.13, -0.29, 0.41),
        (0.13, 0.29, -0.41),
    )
    largest_relative_error = 0.0
    for momenta in samples:
        total = sum(momenta)
        expected = 1.0 / special.upsilon(1.0 + 1j * total)
        for momentum in momenta:
            expected *= (
                2.0
                * momentum
                * special.upsilon(1.0 + 2j * momentum)
                / special.upsilon(1.0 + 1j * (total - 2.0 * momentum))
            )
        expected = complex(expected)
        value = yin_structure_constant_momentum(special, *momenta)
        relative_error = abs(value - expected) / max(abs(expected), 1.0e-300)
        largest_relative_error = max(largest_relative_error, relative_error)

    print("\nBRY equation (2.9) at generic b=1 momenta")
    print(f"  largest relative error: {largest_relative_error:.6e}")
    require(
        largest_relative_error < 1.0e-12,
        "generic momentum-space DOZZ coefficient does not match BRY equation (2.9)",
    )


def check_bry_cosmological_prefactor() -> None:
    """Check the momentum-independent multiplier in BRY equation (2.5)."""

    b = 0.8
    mu = 0.73
    special = UpsilonB(b, dps=40)
    momenta = (0.13, 0.29, 0.41)
    renormalized = yin_structure_constant_momentum(
        special,
        *momenta,
        mu=mu,
        include_cosmological_prefactor=False,
    )
    full = yin_structure_constant_momentum(
        special,
        *momenta,
        mu=mu,
        include_cosmological_prefactor=True,
    )
    q_background = b + 1.0 / b
    base = (
        math.pi
        * mu
        * complex(special.gamma_ratio(b * b))
        * b ** (2.0 - 2.0 * b * b)
    )
    expected_ratio = base ** (-q_background / (2.0 * b))
    relative_error = abs(full / renormalized - expected_ratio) / abs(expected_ratio)

    print("\nBRY equation (2.5) cosmological prefactor")
    print(f"  relative error: {relative_error:.6e}")
    require(
        relative_error < 1.0e-12,
        "DOZZ cosmological prefactor does not match BRY equation (2.5)",
    )

    try:
        yin_structure_constant_momentum(
            UpsilonB(1.0, dps=30),
            *momenta,
            include_cosmological_prefactor=True,
        )
    except ValueError as exc:
        require("singular at b=1" in str(exc), "unexpected b=1 prefactor error")
    else:
        raise AssertionError("the singular bare b=1 cosmological prefactor was accepted")


def check_resonant_b_guard() -> None:
    print("\nresonant b guard")
    try:
        validate_nonresonant_b_for_block(1.0, 2)
    except ValueError as exc:
        print(f"  expected guard: {exc}")
        return
    raise AssertionError("b=1 should be flagged as resonant for the generic recursion")


def check_torus_integral_sample() -> None:
    q = cmath.exp(2j * cmath.pi * 0.9j)
    coarse = liouville_torus_one_point(
        b=0.8,
        external_momentum=0.2,
        q=q,
        block_order=2,
        p_max=2.2,
        quadrature_order=10,
        dps=25,
    )
    fine = liouville_torus_one_point(
        b=0.8,
        external_momentum=0.2,
        q=q,
        block_order=2,
        p_max=2.2,
        quadrature_order=12,
        dps=25,
    )
    difference = abs(fine.value - coarse.value)

    print("\nLiouville torus one-point sample")
    print(f"  coarse: {coarse.value!r}")
    print(f"  fine:   {fine.value!r}")
    print(f"  |fine-coarse|: {difference:.6e}")
    require(abs(fine.value) > 0, "sample integral unexpectedly vanished")
    require(difference < 2e-5, "sample integral is not stable under quadrature refinement")


def check_plumbing_s_t_modular_sample() -> None:
    result = run_torus_plumbing_modular_check(
        tau=0.2 + 0.9j,
        b=0.8,
        external_momentum=0.2,
        block_order=2,
        quadrature_order=14,
        p_max=None,
        dps=22,
        form="full",
        collocation_order=10,
        collocation_samples=96,
    )
    print("\ntorus plumbing S/T modular sample")
    print(f"  tau plumbing error mod Z: {result.plumbing_tau_error_mod_z:.6e}")
    print(f"  S tau error mod Z:       {result.s_tau_error_mod_z:.6e}")
    print(f"  S relative error:        {abs(result.s_relative_error):.6e}")
    print(f"  T lift difference:       {result.t_tau_lift_difference!r}")
    print(f"  T relative error:        {abs(result.t_relative_error):.6e}")
    require(result.plumbing_tau_error_mod_z < 1e-12, "plumbing did not reconstruct tau")
    require(result.s_tau_error_mod_z < 1e-12, "plumbing did not reconstruct S tau")
    require(abs(result.s_relative_error) < 1e-4, "S covariance failed for torus one-point sample")
    require(abs(result.t_tau_lift_difference + 1.0) < 1e-12, "T lift should differ by one period")
    require(abs(result.t_relative_error) < 1e-12, "T invariance failed for torus one-point sample")


def check_raw_qh_is_not_modular_object() -> None:
    """Show that raw q^h sewing is not the modular-covariant torus object."""
    tau = 0.2 + 0.9j
    s_tau = -1.0 / tau
    q = cmath.exp(2j * cmath.pi * tau)
    s_q = cmath.exp(2j * cmath.pi * s_tau)
    quadrature = LiouvilleTorusOnePointQuadrature.for_q_values(
        b=0.8,
        external_momentum=0.2,
        q_values=(q, s_q),
        block_order=2,
        quadrature_order=14,
        dps=22,
    )

    def raw_qh_value(q_value: complex) -> complex:
        threshold_factor = liouville_threshold_modulus_factor((q_value,), b=quadrature.b)
        return threshold_factor * quadrature.hjs_stripped_integral(q_value)

    hjs_q = quadrature.hjs_stripped_integral(q)
    hjs_s_q = quadrature.hjs_stripped_integral(s_q)
    hjs_expected = (abs(tau) ** (2.0 * quadrature.external_weight.real + 1.0)) * hjs_q
    raw_q = raw_qh_value(q)
    raw_s_q = raw_qh_value(s_q)
    raw_expected_if_stripped = (abs(tau) ** (2.0 * quadrature.external_weight.real + 1.0)) * raw_q
    threshold_ratio = (raw_s_q / hjs_s_q) / (raw_q / hjs_q)
    predicted_ratio = (abs(s_q) / abs(q)) ** (0.5 * quadrature.q_background * quadrature.q_background)

    print("\nraw q^h torus normalization guard")
    print(f"  HJS stripped S relative error: {abs((hjs_s_q - hjs_expected) / hjs_expected):.6e}")
    print(f"  raw q^h S relative error:      {abs((raw_s_q - raw_expected_if_stripped) / raw_expected_if_stripped):.6e}")
    print(f"  threshold ratio:               {threshold_ratio!r}")
    print(f"  predicted threshold ratio:     {predicted_ratio!r}")
    require(abs((hjs_s_q - hjs_expected) / hjs_expected) < 1e-4, "HJS stripped torus covariance failed")
    require(abs(threshold_ratio - predicted_ratio) < 1e-12, "raw q^h threshold factor was not reproduced")
    require(
        abs((raw_s_q - raw_expected_if_stripped) / raw_expected_if_stripped) > 1e-1,
        "raw q^h unexpectedly behaved like a modular-covariant stripped torus object",
    )


def run() -> None:
    check_upsilon_identities()
    check_dozz_lambda_matches_alpha()
    check_bry_b1_simplified_formula()
    check_bry_cosmological_prefactor()
    check_xi_resonance_normalization()
    check_resonant_b_guard()
    check_torus_integral_sample()
    check_plumbing_s_t_modular_sample()
    check_raw_qh_is_not_modular_object()
    print("\nall Liouville torus checks passed")


if __name__ == "__main__":
    run()
