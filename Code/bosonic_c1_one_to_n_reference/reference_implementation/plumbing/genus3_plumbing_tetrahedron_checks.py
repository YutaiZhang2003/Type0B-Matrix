#!/usr/bin/env python3
"""Focused checks for the marked genus-three tetrahedral plumbing map."""

from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from genus3_plumbing_tetrahedron import (
        TETRAHEDRON_EDGES,
        TETRAHEDRON_HALF_TWIST,
        generators_for_tetrahedron,
        invert_tetrahedron_period_matrix,
        period_difference_mod_symmetric_integer,
        tetrahedron_leading_period_matrix,
        tetrahedron_leading_q_from_omega,
        tetrahedron_schottky_forward,
        tetrahedron_transition_maps,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus3_plumbing_tetrahedron import (
        TETRAHEDRON_EDGES,
        TETRAHEDRON_HALF_TWIST,
        generators_for_tetrahedron,
        invert_tetrahedron_period_matrix,
        period_difference_mod_symmetric_integer,
        tetrahedron_leading_period_matrix,
        tetrahedron_leading_q_from_omega,
        tetrahedron_schottky_forward,
        tetrahedron_transition_maps,
    )


TEST_Q = (
    0.012 * cmath.exp(0.07j),
    0.010 * cmath.exp(-0.05j),
    0.014 * cmath.exp(0.03j),
    0.011 * cmath.exp(0.06j),
    0.009 * cmath.exp(-0.04j),
    0.013 * cmath.exp(0.02j),
)

POSITIVE_TEST_Q = (0.008, 0.010, 0.012, 0.009, 0.011, 0.013)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_transition_inverses() -> None:
    transitions = tetrahedron_transition_maps(TEST_Q)
    probes = (0.27 + 0.31j, -0.41 + 0.23j, 1.37 + 0.19j)
    for left, right in TETRAHEDRON_EDGES:
        forward = transitions[(left, right)]
        backward = transitions[(right, left)]
        for probe in probes:
            image = forward(probe)
            _require(image is not None, "finite transition probe mapped to infinity")
            recovered = backward(image)
            _require(recovered is not None, "inverse transition probe mapped to infinity")
            _require(abs(recovered - probe) < 2.0e-12, "oriented plumbing transitions are not inverse")


def check_tropical_round_trip() -> None:
    omega = tetrahedron_leading_period_matrix(TEST_Q)
    recovered = tetrahedron_leading_q_from_omega(omega)
    error = max(abs(left - right) for left, right in zip(TEST_Q, recovered))
    _require(error < 2.0e-14, "tetrahedral tropical period map did not invert exactly")

    # Word length zero contains only generator multipliers and pairwise fixed-
    # point cross ratios.  In the plumbing limit it must approach the tropical
    # expression, including the coordinate-induced half twists, modulo an
    # integral B-shift.
    # Values much below 1e-6 make the double-precision fixed-point quadratic
    # ill-conditioned, so this check stays in a small but numerically resolved
    # plumbing regime.
    tiny_q = tuple(1.0e-4 * cmath.exp(0.01j * index) for index in range(6))
    exact_zero = tetrahedron_schottky_forward(tiny_q, word_length=0)
    tropical = tetrahedron_leading_period_matrix(tiny_q)
    difference, _ = period_difference_mod_symmetric_integer(exact_zero.omega, tropical)
    _require(
        float(np.max(np.abs(difference))) < 1.0e-4,
        "tetrahedral half-twist convention does not match the Schottky plumbing limit",
    )
    _require(
        np.array_equal(2 * TETRAHEDRON_HALF_TWIST.real, np.rint(2 * TETRAHEDRON_HALF_TWIST.real)),
        "tetrahedral twist entries are not half integral",
    )


def check_generator_health_and_forward_stability() -> None:
    generators = generators_for_tetrahedron(TEST_Q)
    _require(len(generators) == 3, "tetrahedral chart did not produce rank three")
    for generator in generators:
        _require(0.0 < abs(generator.multiplier) < 1.0e-4, "unexpected tetrahedral generator multiplier")

    low = tetrahedron_schottky_forward(TEST_Q, word_length=3)
    high = tetrahedron_schottky_forward(TEST_Q, word_length=4)
    difference, _ = period_difference_mod_symmetric_integer(high.omega, low.omega)
    stability = float(np.max(np.abs(difference)))
    _require(low.minimum_imaginary_eigenvalue > 0.0, "word-3 period is not in Siegel space")
    _require(high.minimum_imaginary_eigenvalue > 0.0, "word-4 period is not in Siegel space")
    _require(high.symmetry_error < 1.0e-9, "word-4 period symmetry defect is too large")
    _require(stability < 1.0e-8, "rank-three Schottky word sum has not stabilized")


def check_omega_q_omega_round_trip() -> None:
    # The target is generated at the validation order, but the inverse optimizer
    # uses the lower order.  Passing therefore requires more than a same-function
    # identity: the omitted word correction must also be stable.
    for q_truth in (TEST_Q, POSITIVE_TEST_Q):
        target = tetrahedron_schottky_forward(q_truth, word_length=4)
        inverse = invert_tetrahedron_period_matrix(
            target.omega,
            word_length=3,
            validation_word_step=1,
            max_nfev=240,
            period_tolerance=1.0e-8,
            stability_tolerance=1.0e-8,
            symmetry_tolerance=1.0e-9,
        )
        _require(inverse.success, f"tetrahedral inverse optimizer failed: {inverse.message}")
        _require(inverse.certified, "Omega -> q -> Omega did not pass certification")
        _require(inverse.period_max_residual < 1.0e-8, "round-trip period residual is too large")
        _require(inverse.word_stability < 1.0e-8, "round-trip word stability is too large")
        _require(math.isfinite(inverse.max_q_abs), "round-trip returned nonfinite q")
        relative_q_error = max(
            abs(recovered - truth) / abs(truth)
            for recovered, truth in zip(inverse.q_values, q_truth)
        )
        _require(relative_q_error < 1.0e-6, "local inverse did not recover the input plumbing point")


def check_inverse_rejects_nonsymmetric_target() -> None:
    target = tetrahedron_schottky_forward(TEST_Q, word_length=4).omega.copy()
    target[0, 1] += 0.2
    target[1, 0] -= 0.2
    try:
        invert_tetrahedron_period_matrix(target, symmetry_tolerance=1.0e-8)
    except ValueError as error:
        _require(
            "symmetry defect" in str(error),
            "nonsymmetric inverse target raised an unrelated error",
        )
    else:
        raise AssertionError("nonsymmetric inverse target was silently projected")


def run_checks() -> None:
    check_transition_inverses()
    check_tropical_round_trip()
    check_generator_health_and_forward_stability()
    check_omega_q_omega_round_trip()
    check_inverse_rejects_nonsymmetric_target()
    print("genus3 tetrahedral plumbing checks passed")


if __name__ == "__main__":
    run_checks()
