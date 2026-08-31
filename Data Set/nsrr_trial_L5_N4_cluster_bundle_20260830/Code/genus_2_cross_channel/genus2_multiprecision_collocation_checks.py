#!/usr/bin/env python3
"""Numerical checks for the rescaled mixed-cusp holomorphic solver."""

from __future__ import annotations

import cmath
import math
from unittest import mock

import numpy as np

try:
    from genus2_hybrid_period_map import _collocation_at_order, period_max_residual
    from genus2_multiprecision_collocation import (
        BACKEND_READY,
        evaluate_multiprecision_holomorphic_period_map,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_hybrid_period_map import _collocation_at_order, period_max_residual
    from plumbing.genus2_multiprecision_collocation import (
        BACKEND_READY,
        evaluate_multiprecision_holomorphic_period_map,
    )


TOLERANCE = 1.0e-11


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def evaluate(topology: str, q: tuple[complex, complex, complex]):
    return evaluate_multiprecision_holomorphic_period_map(
        topology,
        q,
        log_q_values=tuple(cmath.log(value) for value in q),
        tolerance=TOLERANCE,
        maximum_basis=64,
    )


def check_one_small_edge() -> None:
    cases = (
        ("theta", (1.0e-12 + 0.0j, 0.10 + 0.0j, 0.16 + 0.0j)),
        ("glasses", (1.0e-12 + 0.0j, 0.10 + 0.0j, 0.08 + 0.0j)),
    )
    for topology, q in cases:
        result = evaluate(topology, q)
        require(result.converged, f"{topology} one-edge cusp did not converge")
        require(result.used_multiprecision, f"{topology} cusp did not restore periods at MP")
        require(result.error_estimate <= TOLERANCE, f"{topology} cusp missed its error bar")
        require(
            float(np.min(np.linalg.eigvalsh(result.omega.imag))) > 0.0,
            f"{topology} cusp result is not a Riemann matrix",
        )


def check_two_small_edges() -> None:
    for topology, finite in (("theta", 0.16), ("glasses", 0.08)):
        q = (
            1.0e-12 * cmath.exp(0.3j),
            2.0e-11 * cmath.exp(-0.2j),
            finite * cmath.exp(0.1j),
        )
        result = evaluate(topology, q)
        require(result.converged, f"{topology} two-edge cusp did not converge")
        require(result.error_estimate <= TOLERANCE, f"{topology} two-edge cusp missed its bar")


def check_against_direct_overlap() -> None:
    cases = (
        ("theta", (1.0e-10 + 0.0j, 0.10 + 0.0j, 0.16 + 0.0j), 2.0e-11),
        ("glasses", (1.0e-10 + 0.0j, 0.10 + 0.0j, 0.08 + 0.0j), 2.0e-11),
    )
    for topology, q, bar in cases:
        result = evaluate(topology, q)
        direct, _, _ = _collocation_at_order(topology, q, 40)
        residual = period_max_residual(result.omega, direct)
        require(residual <= bar, f"{topology} rescaled/direct overlap residual {residual:.3e}")


def check_failed_production_row_regressions() -> None:
    cases = (
        (
            "theta",
            (
                9.246960340975649e-02 - 2.3479497994756718e-01j,
                9.861784402437108e-08 + 1.3386881417361418e-07j,
                4.227764052232719e-04 - 9.501376141275798e-05j,
            ),
        ),
        (
            "glasses",
            (
                1.9079928632813205e-15 + 9.816290706456654e-15j,
                -9.306739045167904e-03 - 2.096200783246549e-02j,
                5.920263558100136e-01 - 1.7562261859443384e-01j,
            ),
        ),
        (
            "theta",
            (
                4.99996393782919e-01 + 8.364699785892941e-02j,
                -4.723596131341934e-15 + 8.814059200389527e-15j,
                6.254361714218839e-01 + 1.639550973278829e-02j,
            ),
        ),
        (
            "theta",
            (
                2.660035492285044e-01 - 2.072682617061456e-01j,
                3.771206539698527e-15 + 9.261641389890607e-15j,
                7.305069461584899e-01 - 9.928133658936936e-02j,
            ),
        ),
    )
    for topology, q in cases:
        result = evaluate_multiprecision_holomorphic_period_map(
            topology,
            q,
            log_q_values=tuple(cmath.log(value) for value in q),
            tolerance=TOLERANCE,
            maximum_basis=224,
        )
        require(result.converged, f"{topology} production retry regression did not converge")
        require(result.high_order <= 224, f"{topology} retry exceeded its recorded ceiling")
        require(result.error_estimate <= TOLERANCE, f"{topology} retry missed its error bar")
        require(
            "scaled" in result.algorithm,
            f"{topology} retry did not use logarithmic basis scaling",
        )
        require(
            float(np.min(np.linalg.eigvalsh(result.omega.imag))) > 0.0,
            f"{topology} retry result is not a Riemann matrix",
        )


def check_cluster_svd_failure_uses_pivoted_qr() -> None:
    q = (
        4.99996393782919e-01 + 8.364699785892941e-02j,
        -4.723596131341934e-15 + 8.814059200389527e-15j,
        6.254361714218839e-01 + 1.639550973278829e-02j,
    )
    with mock.patch.object(
        np.linalg,
        "lstsq",
        side_effect=np.linalg.LinAlgError("forced cluster SVD nonconvergence"),
    ):
        result = evaluate_multiprecision_holomorphic_period_map(
            "theta",
            q,
            log_q_values=tuple(cmath.log(value) for value in q),
            tolerance=1.0e-6,
            maximum_basis=64,
        )
    require(result.converged, "pivoted-QR fallback did not certify the cluster failure")
    require(result.error_estimate <= 1.0e-6, "pivoted-QR fallback missed its error bar")


def run() -> None:
    require(BACKEND_READY, "multiprecision backend is not marked ready")
    check_one_small_edge()
    check_two_small_edges()
    check_against_direct_overlap()
    check_failed_production_row_regressions()
    check_cluster_svd_failure_uses_pivoted_qr()
    print("multiprecision mixed-cusp collocation checks passed")


if __name__ == "__main__":
    run()
