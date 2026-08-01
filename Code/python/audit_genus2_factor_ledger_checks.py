#!/usr/bin/env python3
"""Regression checks for the factor-by-factor genus-two normalization audit."""

from __future__ import annotations

import math

try:
    from audit_genus2_factor_ledger import build_audit
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.audit_genus2_factor_ledger import build_audit


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> None:
    payload = build_audit()
    factors = {str(item["id"]): item for item in payload["factors"]}

    _require(len(factors) == 15, "the explicit factor ledger is incomplete")
    _require(
        all(bool(item["passed"]) for item in factors.values()),
        "at least one explicit factor test failed",
    )
    _require(
        factors["F13"]["status"] == "source_consistent_not_independent",
        "the shared-convention Liouville sewing test was overstated",
    )
    _require(
        abs(float(factors["F05"]["value"]) - 2.0 / math.pi) < 2.0e-15,
        "the final local multiplier is not 2/pi",
    )
    _require(
        float(factors["F06"]["value"]) == 2.0**24,
        "the raw-product Mumford conversion is not 2^24",
    )
    _require(
        float(factors["F14"]["value"]) == 0.5,
        "the genus-two stack weight is not 1/2",
    )
    _require(
        payload["comparison"]["status"] == "external_comparison_failed",
        "the failed absolute comparison was hidden",
    )
    _require(
        payload["absolute_normalization_certified"] is False,
        "the audit incorrectly certified the absolute normalization",
    )
    _require(
        payload["matrix_model_genus2_value_used_to_set_worldsheet_factors"] is False,
        "the matrix-model target leaked into the worldsheet factor tests",
    )
    print("genus-two factor ledger checks passed")


if __name__ == "__main__":
    run()
