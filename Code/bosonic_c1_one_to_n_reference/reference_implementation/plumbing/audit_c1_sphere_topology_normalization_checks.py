#!/usr/bin/env python3
"""Checks for the c=1 sphere-topology normalization audit."""

from __future__ import annotations

import math

try:
    from audit_c1_sphere_topology_normalization import build_audit
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_c1_sphere_topology_normalization import build_audit


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_alpha_prime_values() -> None:
    for alpha_prime in (0.5, 1.0, 1.7, 2.0):
        audit = build_audit(alpha_prime)
        require(bool(audit["all_checks_pass"]), "analytic ledger failed")
        replacement = audit["critical_to_c1_topology_replacement"]
        require(
            float(replacement["genus_one_value"]) == 1.0,
            "the genus-one vacuum must be insensitive to the sphere normalization",
        )
        require(
            math.isclose(
                float(replacement["value"]),
                4.0 * math.pi / alpha_prime,
                rel_tol=2.0e-15,
            ),
            "critical-to-c=1 topology ratio failed",
        )
        current = audit["current_code"]
        require(
            math.isclose(
                float(current["sphere_normalized_pre_brst_kernel_multiplier"]),
                4.0,
                rel_tol=2.0e-15,
            ),
            "required pre-BRST c=1 positive-real coefficient failed",
        )
        require(
            math.isclose(
                float(current["complete_production_v5_kernel_multiplier"]),
                64.0 * math.pi**4,
                rel_tol=2.0e-15,
            ),
            "complete v5 c=1 coefficient failed",
        )


def run() -> None:
    check_alpha_prime_values()
    print("c=1 sphere-topology normalization checks passed")


if __name__ == "__main__":
    run()
