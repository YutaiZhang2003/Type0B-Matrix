#!/usr/bin/env python3
"""Self-checks for the holomorphic-form q-to-Omega audit."""

from __future__ import annotations

import cmath

import numpy as np

try:
    from audit_q_to_omega_accuracy import (
        collocation_orders,
        collocation_period_from_q,
        period_max_residual,
        validate_or_refine_period_map,
    )
    from genus2_hybrid_period_map import (
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        hybrid_period_matrix,
    )
except ImportError:  # pragma: no cover
    from plumbing.audit_q_to_omega_accuracy import (
        collocation_orders,
        collocation_period_from_q,
        period_max_residual,
        validate_or_refine_period_map,
    )
    from plumbing.genus2_hybrid_period_map import (
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        hybrid_period_matrix,
    )


def run_checks() -> None:
    if collocation_orders("theta", (0.31, 0.2, 0.1))[2:] != (44, 176, 48, 192):
        raise AssertionError("large-q theta certificate did not use the raised basis")
    if collocation_orders("glasses", (0.31, 0.2, 0.1))[2:] != (44, 264, 48, 288):
        raise AssertionError("large-q glasses certificate did not use the raised basis")

    slow_glasses_seed = (
        -0.1551822226618 + 0.1467239521383j,
        0.34110408903 - 0.00365837214843j,
        -0.1918326222726 - 0.09555789100111j,
    )
    # Retain the difficult phases and large handle, but keep the standard
    # glasses sewing disks disjoint so this is a valid chart.
    slow_glasses_q = (
        slow_glasses_seed[0],
        slow_glasses_seed[1],
        0.15 * slow_glasses_seed[2] / abs(slow_glasses_seed[2]),
    )
    slow_glasses_target, _, _ = collocation_period_from_q(
        "glasses",
        slow_glasses_q,
        basis_order=56,
        samples_per_seam=336,
    )
    slow_glasses_certificate = validate_or_refine_period_map(
        "glasses",
        slow_glasses_target,
        slow_glasses_q,
        word_length=8,
        word_step=1,
        tolerance=1.0e-6,
        reinverse_validation_word_length=10,
        reinverse_max_nfev=80,
    )
    if (
        slow_glasses_certificate.high_order < 40
        or slow_glasses_certificate.seam_residual > 1.0e-6
    ):
        raise AssertionError("slow glasses seam did not trigger adaptive basis escalation")

    left = np.asarray([[1.2j, 0.1j], [0.1j, 1.4j]])
    right = left + np.asarray([[1.0, -2.0], [-2.0, 3.0]])
    if period_max_residual(left, right) > 1.0e-14:
        raise AssertionError("integral period-matrix branch was not removed")

    true_q = (0.025 + 0.003j, -0.031 + 0.002j, 0.041 - 0.004j)
    initial_q = tuple(value * (1.0 + 2.0e-4) for value in true_q)
    _, _, _, _, validation_order, validation_samples = collocation_orders("theta", true_q)
    direct_target, _, _ = collocation_period_from_q(
        "theta",
        true_q,
        basis_order=validation_order,
        samples_per_seam=validation_samples,
    )
    certificate = validate_or_refine_period_map(
        "theta",
        direct_target,
        true_q,
        word_length=7,
        word_step=1,
        tolerance=2.0e-8,
        reinverse_validation_word_length=8,
        reinverse_max_nfev=60,
    )
    if certificate.period_algorithm != "holomorphic-form-collocation":
        raise AssertionError("bulk validation did not use normalized holomorphic one-forms")
    if certificate.refined or certificate.final_residual > 2.0e-8:
        raise AssertionError("an already accurate q was unnecessarily re-inverted")

    corrected = validate_or_refine_period_map(
        "theta",
        direct_target,
        initial_q,
        word_length=7,
        word_step=1,
        tolerance=2.0e-8,
        reinverse_validation_word_length=8,
        reinverse_max_nfev=60,
    )
    if not corrected.refined or corrected.final_residual > 2.0e-8:
        raise AssertionError("an inaccurate q did not trigger successful re-inversion")

    cusp_q = (0.05 + 0.0j, 0.04 + 0.0j, 1.0e-13 + 0.0j)
    cusp_config = HybridPeriodMapConfig(
        tolerance=1.0e-6,
        agreement_tolerance=1.0e-6,
        minimum_schottky_word=7,
        maximum_schottky_word=10,
    )
    cusp_target = hybrid_period_matrix("theta", cusp_q, config=cusp_config).omega
    cusp_certificate = validate_or_refine_period_map(
        "theta",
        cusp_target,
        cusp_q,
        word_length=8,
        word_step=1,
        tolerance=1.0e-6,
        reinverse_validation_word_length=10,
        reinverse_max_nfev=40,
    )
    if (
        cusp_certificate.period_algorithm != SCHOTTKY_ALGORITHM
        or cusp_certificate.refined
        or cusp_certificate.final_residual > 1.0e-6
    ):
        raise AssertionError("deep cusp was not certified by adaptive Schottky words")

    underflow_logs = (-6.5 + 0.2j, -3606.0 + 0.4j, -4.7 - 0.3j)
    underflow_q = (cmath.exp(underflow_logs[0]), 0j, cmath.exp(underflow_logs[2]))
    underflow_config = HybridPeriodMapConfig(
        tolerance=1.0e-7,
        agreement_tolerance=1.0e-7,
        minimum_schottky_word=4,
        maximum_schottky_word=7,
    )
    underflow_target = hybrid_period_matrix(
        "theta",
        underflow_q,
        config=underflow_config,
        log_q_values=underflow_logs,
    ).omega
    underflow_certificate = validate_or_refine_period_map(
        "theta",
        underflow_target,
        underflow_q,
        initial_log_q=underflow_logs,
        word_length=5,
        word_step=1,
        tolerance=1.0e-7,
        reinverse_validation_word_length=6,
        reinverse_max_nfev=20,
    )
    if (
        underflow_certificate.period_algorithm != SCHOTTKY_ALGORITHM
        or underflow_certificate.log_q != underflow_logs
        or underflow_certificate.final_residual > 1.0e-7
    ):
        raise AssertionError("logarithmic underflow cusp lost hybrid coverage")

    print("audit_q_to_omega_accuracy checks passed")


if __name__ == "__main__":
    run_checks()
