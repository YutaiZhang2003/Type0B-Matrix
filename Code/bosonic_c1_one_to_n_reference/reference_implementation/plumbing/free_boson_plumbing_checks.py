#!/usr/bin/env python3
"""Checks for the genus-two free-boson plumbing diagnostic."""

from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from compare_free_boson_weyl_overlap import DEFAULT_GLASSES_Q, DEFAULT_THETA_Q
    from free_boson_plumbing import (
        bergman_petersson_norm_delta2,
        free_boson_chiral_log_factor,
        glasses_separating_F_asymptotic,
        glasses_separating_raw_oscillator_asymptotic,
        glasses_free_boson_product,
        glasses_free_boson_powered_convergence,
        igusa_chi10_genus2,
        igusa_chi10_evaluation_genus2,
        igusa_chi10_log_abs_genus2,
        long_tube_normalized_frame_factor,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_loop_momentum_log_factor,
        noncompact_scalar_zero_mode_factor,
        plumbing_over_bergman_frame_factor,
        theta_maximal_F_asymptotic,
        theta_maximal_raw_oscillator_asymptotic,
        theta_free_boson_product,
        theta_free_boson_powered_convergence,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from plumbing_algorithms import generators_for_glasses, schottky_glasses_period_matrix, schottky_period_matrix_cross_ratio, schottky_theta_period_matrix_cross_ratio
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.compare_free_boson_weyl_overlap import DEFAULT_GLASSES_Q, DEFAULT_THETA_Q
    from plumbing.free_boson_plumbing import (
        bergman_petersson_norm_delta2,
        free_boson_chiral_log_factor,
        glasses_separating_F_asymptotic,
        glasses_separating_raw_oscillator_asymptotic,
        glasses_free_boson_product,
        glasses_free_boson_powered_convergence,
        igusa_chi10_genus2,
        igusa_chi10_evaluation_genus2,
        igusa_chi10_log_abs_genus2,
        long_tube_normalized_frame_factor,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_loop_momentum_log_factor,
        noncompact_scalar_zero_mode_factor,
        plumbing_over_bergman_frame_factor,
        theta_maximal_F_asymptotic,
        theta_maximal_raw_oscillator_asymptotic,
        theta_free_boson_product,
        theta_free_boson_powered_convergence,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from plumbing.plumbing_algorithms import generators_for_glasses, schottky_glasses_period_matrix, schottky_period_matrix_cross_ratio, schottky_theta_period_matrix_cross_ratio


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _dedekind_eta(tau: complex, max_mode: int = 200) -> complex:
    q = cmath.exp(2j * math.pi * complex(tau))
    value = cmath.exp(1j * math.pi * complex(tau) / 12.0)
    q_power = q
    for _ in range(1, max_mode + 1):
        value *= 1.0 - q_power
        q_power *= q
    return value


def check_single_multiplier_product() -> None:
    multiplier = 0.17 + 0.03j
    got, tail = free_boson_chiral_log_factor(multiplier, max_mode=40, tolerance=1.0e-16)
    expected = sum(-cmath.log(1.0 - multiplier**mode) for mode in range(1, 41))
    _require(abs(got - expected) < 1.0e-14, "single-multiplier product disagrees with direct sum")
    _require(tail > 0.0, "tail estimate should be positive for nonzero multiplier")


def check_genus_g_loop_momentum_normalization() -> None:
    alpha_prime = 1.7
    for imaginary_diagonal in ((1.3,), (1.3, 0.9), (1.3, 0.9, 1.7)):
        omega = 1j * np.diag(imaginary_diagonal)
        genus = len(imaginary_diagonal)
        expected_dimensionless = math.prod(imaginary_diagonal) ** -0.5
        observed_dimensionless = noncompact_scalar_loop_momentum_factor(omega)
        _require(
            abs(observed_dimensionless / expected_dimensionless - 1.0) < 1.0e-14,
            f"genus-{genus} dimensionless handle Gaussian is misnormalized",
        )
        observed_xi = xi_noncompact_scalar_loop_momentum_factor(
            omega,
            alpha_prime=alpha_prime,
        )
        expected_xi = expected_dimensionless / (
            (2.0 * math.pi * math.sqrt(alpha_prime)) ** genus
        )
        _require(
            abs(observed_xi / expected_xi - 1.0) < 1.0e-14,
            f"genus-{genus} physical-momentum handle Gaussian is misnormalized",
        )


def check_single_multiplier_fixed_cutoff_tail() -> None:
    multiplier = 0.5 + 0.0j
    got, tail = free_boson_chiral_log_factor(
        multiplier,
        max_mode=40,
        tolerance=0.1,
    )
    expected = sum(
        (-cmath.log(1.0 - multiplier**mode) for mode in range(1, 41)),
        0.0j,
    )
    observed_remainder = sum(
        (-math.log(1.0 - multiplier.real**mode) for mode in range(41, 400)),
        0.0,
    )
    _require(
        abs(got - expected) < 1.0e-15,
        "free-boson product did not use the requested fixed cutoff",
    )
    _require(
        tail >= observed_remainder,
        "free-boson tail does not cover modes above the fixed cutoff",
    )


def check_period_matched_bergman_norm() -> None:
    omega_glasses = schottky_glasses_period_matrix(*DEFAULT_GLASSES_Q, max_word_len=8, b_order=600)
    omega_theta = schottky_theta_period_matrix_cross_ratio(*DEFAULT_THETA_Q, max_word_len=8)
    _, _, norm_glasses = bergman_petersson_norm_delta2(omega_glasses, theta_nmax=8)
    _, _, norm_theta = bergman_petersson_norm_delta2(omega_theta, theta_nmax=8)
    relative = abs(norm_theta - norm_glasses) / abs(norm_glasses)
    _require(relative < 5.0e-6, f"Bergman norm mismatch too large: {relative:.6e}")


def check_overlap_frame_ratio() -> None:
    glasses = glasses_free_boson_product(*DEFAULT_GLASSES_Q, max_word_length=8, max_mode=80)
    theta = theta_free_boson_product(*DEFAULT_THETA_Q, max_word_length=8, max_mode=80)
    oscillator_ratio = theta.nonchiral_value / glasses.nonchiral_value
    omega_glasses = schottky_glasses_period_matrix(*DEFAULT_GLASSES_Q, max_word_len=8, b_order=600)
    omega_theta = schottky_theta_period_matrix_cross_ratio(*DEFAULT_THETA_Q, max_word_len=8)
    zero_mode_ratio = noncompact_scalar_zero_mode_factor(omega_theta) / noncompact_scalar_zero_mode_factor(
        omega_glasses
    )
    full_ratio = oscillator_ratio * zero_mode_ratio
    _require(
        abs(oscillator_ratio - 0.4661608566149985) < 5.0e-8,
        f"unexpected free-boson oscillator ratio {oscillator_ratio:.16e}",
    )
    _require(abs(zero_mode_ratio - 0.08928555539163692) < 5.0e-8, "unexpected scalar zero-mode ratio")
    _require(abs(full_ratio - 0.041621430976911) < 5.0e-8, "unexpected full free-boson ratio")


def check_power_25_accuracy_budget() -> None:
    """Check the reported c=25 tail against higher word cutoffs."""

    q_values = (0.1 + 0.0j, 0.1 + 0.0j, 0.1 + 0.0j)
    for name, product_function in (
        ("theta", theta_free_boson_product),
        ("glasses", glasses_free_boson_product),
    ):
        truncated = product_function(
            *q_values,
            max_word_length=8,
            max_mode=100,
            tolerance=1.0e-16,
        )
        reference = product_function(
            *q_values,
            max_word_length=10,
            max_mode=120,
            tolerance=1.0e-16,
        )
        estimated_log_error = truncated.powered_log_error_estimate(25.0)
        estimated_relative_error = truncated.powered_relative_error_estimate(25.0)
        observed_log_step = 25.0 * abs(
            reference.nonchiral_log_value - truncated.nonchiral_log_value
        )
        _require(estimated_log_error is not None, f"{name} c=25 tail is unavailable")
        _require(
            estimated_log_error >= observed_log_step,
            f"{name} c=25 tail estimate does not cover the word-8 to word-10 step",
        )
        _require(
            estimated_relative_error is not None
            and estimated_relative_error < 1.0e-6,
            f"{name} c=25 relative tail is too large at |q|=0.1",
        )


def check_streamed_powered_convergence() -> None:
    """Early stopping must equal an independent run at the reached cap."""

    q_values = (0.1 + 0.0j, 0.1 + 0.0j, 0.1 + 0.0j)
    for name, convergence_function, product_function in (
        ("theta", theta_free_boson_powered_convergence, theta_free_boson_product),
        ("glasses", glasses_free_boson_powered_convergence, glasses_free_boson_product),
    ):
        streamed = convergence_function(
            *q_values,
            initial_word_length=8,
            maximum_word_length=14,
            word_length_increment=2,
            max_mode=100,
            tolerance=1.0e-16,
            power=25.0,
            powered_relative_tolerance=1.0e-6,
            confirmation_steps=2,
        )
        _require(streamed.converged, f"{name} streamed scalar did not converge")
        reached = streamed.product.max_word_length
        _require(reached < 14, f"{name} streamed scalar did not stop before its cap")
        fixed = product_function(
            *q_values,
            max_word_length=reached,
            max_mode=100,
            tolerance=1.0e-16,
        )
        _require(
            streamed.product.nonchiral_log_value == fixed.nonchiral_log_value,
            f"{name} streamed scalar changed the fixed-cap log product",
        )
        _require(
            streamed.product.primitive_count == fixed.primitive_count,
            f"{name} streamed scalar changed the fixed-cap primitive count",
        )


def check_stable_loop_momentum_determinant() -> None:
    omega = np.asarray(
        [[0.0 + 1.0e-8j, 0.0 + 0.99999999e-8j],
         [0.0 + 0.99999999e-8j, 0.0 + 1.0e-8j]],
        dtype=np.complex128,
    )
    log_factor = noncompact_scalar_loop_momentum_log_factor(omega)
    factor = noncompact_scalar_zero_mode_factor(omega)
    _require(math.isfinite(log_factor), "loop-momentum log factor is not finite")
    _require(
        abs(math.log(factor) - log_factor) < 2.0e-15 * max(1.0, abs(log_factor)),
        "loop-momentum factor is inconsistent with its stable logarithm",
    )


def check_glasses_long_tube_normalization() -> None:
    q_left = 0.05 + 0.0j
    q_right = 0.07 + 0.0j
    q_bridge = 1.0e-4 + 0.0j
    omega = schottky_period_matrix_cross_ratio(
        generators_for_glasses(q_left, q_right, q_bridge),
        max_word_len=8,
    )
    product = glasses_free_boson_product(q_left, q_right, q_bridge, max_word_length=8, max_mode=80)
    _det_im, _chi10, exact_F = bergman_petersson_norm_delta2(omega, theta_nmax=10)
    asymptotic_F = glasses_separating_F_asymptotic(q_left, q_right, q_bridge)
    raw_asymptotic = glasses_separating_raw_oscillator_asymptotic(q_left, q_right)
    normalized = long_tube_normalized_frame_factor(
        product.nonchiral_value,
        exact_F,
        raw_asymptotic,
        asymptotic_F,
    )
    direct_frame = plumbing_over_bergman_frame_factor(product.nonchiral_value, exact_F)
    direct_asymptotic = plumbing_over_bergman_frame_factor(raw_asymptotic, asymptotic_F)
    _require(abs(exact_F / asymptotic_F - 1.0) < 5.0e-4, "Bergman F separating asymptotic failed")
    _require(
        abs(normalized - direct_frame / direct_asymptotic) < 1.0e-15,
        "normalized frame factor should equal direct frame/asymptotic ratio",
    )
    _require(abs(normalized - 1.0) < 5.0e-4, "long-tube normalized frame factor should tend to one")


def check_theta_maximal_F_asymptotic() -> None:
    q_values = (1.0e-4 + 0.0j, 2.0e-4 + 0.0j, 3.0e-4 + 0.0j)
    omega = schottky_theta_period_matrix_cross_ratio(*q_values, max_word_len=8)
    _det_im, _chi10, exact_F = bergman_petersson_norm_delta2(omega, theta_nmax=10)
    asymptotic_F = theta_maximal_F_asymptotic(*q_values)
    _require(abs(exact_F / asymptotic_F - 1.0) < 5.0e-4, "theta maximal F asymptotic failed")
    raw_asymptotic = theta_maximal_raw_oscillator_asymptotic(*q_values)
    _require(raw_asymptotic == 1.0, "theta raw maximal oscillator asymptotic should be one")


def check_chi10_logarithmic_cusp() -> None:
    bulk = np.asarray([[1.0j, 0.2j], [0.2j, 1.2j]], dtype=np.complex128)
    bulk_raw = igusa_chi10_genus2(bulk)
    bulk_note = igusa_chi10_genus2(bulk, normalization="string_note_2^-12")
    bulk_legacy_alias = igusa_chi10_genus2(bulk, normalization="igusa_2^-12")
    _require(
        abs(bulk_note / bulk_raw - 2.0**-12) < 1.0e-16,
        "the string-note chi10 convention has the wrong sign or power of two",
    )
    _require(
        bulk_legacy_alias == bulk_note,
        "the legacy 2^-12 normalization alias disagrees with the string-note form",
    )
    bulk_log = igusa_chi10_log_abs_genus2(bulk)
    _require(abs(bulk_log - math.log(abs(bulk_raw))) < 1.0e-13, "log chi10 changed the bulk value")
    bulk_combined = igusa_chi10_evaluation_genus2(bulk)
    _require(
        abs(bulk_combined.value / bulk_raw - 1.0) < 2.0e-14,
        "combined chi10 changed the bulk complex value",
    )
    _require(
        abs(bulk_combined.log_abs - bulk_log) < 2.0e-13,
        "combined chi10 changed the bulk logarithmic value",
    )

    cusp = np.asarray([[0.2 + 1.7j, -0.08 + 0.7j], [-0.08 + 0.7j, 0.1 + 575.0j]])
    cusp_raw = igusa_chi10_genus2(cusp)
    cusp_log = igusa_chi10_log_abs_genus2(cusp)
    cusp_combined = igusa_chi10_evaluation_genus2(cusp)
    _require(cusp_raw == 0.0, "the cusp regression no longer exercises product underflow")
    _require(math.isfinite(cusp_log) and cusp_log < -3000.0, "log chi10 did not retain the long cusp")
    _require(cusp_combined.value == 0.0, "combined chi10 lost cusp underflow compatibility")
    _require(
        abs(cusp_combined.log_abs - cusp_log) < 2.0e-12,
        "combined chi10 changed the long-cusp logarithm",
    )


def check_string_note_chi10_separating_factorization() -> None:
    tau_left = 0.17 + 1.13j
    tau_right = -0.21 + 0.91j
    eta_product = _dedekind_eta(tau_left) ** 24 * _dedekind_eta(tau_right) ** 24
    errors = []
    for epsilon in (0.01, 0.005, 0.0025):
        omega = np.asarray(
            [[tau_left, epsilon], [epsilon, tau_right]],
            dtype=np.complex128,
        )
        q_separating = 2j * math.pi * epsilon
        chi10_note = igusa_chi10_genus2(
            omega,
            nmax=12,
            normalization="string_note_2^-12",
        )
        ratio = chi10_note / (q_separating**2 * eta_product)
        errors.append(abs(ratio - 1.0))
    _require(errors[-1] < errors[0], "the string-note chi10 limit did not converge")
    _require(
        errors[-1] < 3.0e-5,
        "the theta convention does not reproduce chi10_note ~ q^2 eta_1^24 eta_2^24",
    )


def main() -> None:
    check_single_multiplier_product()
    check_genus_g_loop_momentum_normalization()
    check_single_multiplier_fixed_cutoff_tail()
    check_period_matched_bergman_norm()
    check_overlap_frame_ratio()
    check_power_25_accuracy_budget()
    check_streamed_powered_convergence()
    check_stable_loop_momentum_determinant()
    check_glasses_long_tube_normalization()
    check_theta_maximal_F_asymptotic()
    check_chi10_logarithmic_cusp()
    check_string_note_chi10_separating_factorization()
    print("free_boson_plumbing checks passed")


if __name__ == "__main__":
    main()
