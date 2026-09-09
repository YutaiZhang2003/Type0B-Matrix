#!/usr/bin/env python3
"""Checks for the genus-zero 1 -> 2 normalization audit."""

from __future__ import annotations

import math

try:
    from audit_genus0_one_to_two_amplitude import build_audit
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_genus0_one_to_two_amplitude import build_audit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_samples() -> None:
    for omega_1, omega_2, alpha_prime, g_s_xi in (
        (0.34, 0.46, 1.0, 0.037),
        (0.73, 1.11, 2.3, 0.019),
        (1.20, 0.40, 0.57, 0.061),
    ):
        audit = build_audit(
            omega_1,
            omega_2,
            alpha_prime=alpha_prime,
            g_s_xi=g_s_xi,
            dps=40,
        )
        require(bool(audit["passed"]), "normalization audit did not pass")
        factors = audit["factor_ledger"]
        require(
            math.isclose(
                float(factors["zero_mode_coefficient_without_i"]),
                4.0 * math.pi,
                rel_tol=1.0e-15,
            ),
            "the alpha'-independent 4*pi sphere factor failed",
        )
        bry = audit["bry_cross_check"]
        require(
            math.isclose(
                float(bry["g_s_BRY_over_g_s_Xi"]),
                2.0,
                rel_tol=1.0e-15,
            ),
            "BRY/Xi coupling ratio failed",
        )


def run() -> None:
    check_samples()
    print("genus-zero 1->2 amplitude checks passed")


if __name__ == "__main__":
    run()
