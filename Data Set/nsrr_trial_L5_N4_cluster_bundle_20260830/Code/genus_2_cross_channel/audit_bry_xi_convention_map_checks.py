#!/usr/bin/env python3
"""Exact checks for the BRY-to-Xi convention audit."""

from __future__ import annotations

import math

try:
    from audit_bry_xi_convention_map import build_audit
    from genus2_integrand_normalization import bry_xi_bare_convention_map
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_bry_xi_convention_map import build_audit
    from plumbing.genus2_integrand_normalization import bry_xi_bare_convention_map


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    sphere = bry_xi_bare_convention_map(0, 4)
    torus = bry_xi_bare_convention_map(1, 2)
    genus_two = bry_xi_bare_convention_map(2, 0)
    audit = build_audit(2.7)

    _require(sphere.complex_moduli_dimension == 1, "sphere dimension is wrong")
    _require(sphere.string_coupling_power == 2, "sphere coupling power is wrong")
    _require(sphere.xi_over_bry_known_product == 0.5, "sphere bare map is wrong")
    _require(torus.complex_moduli_dimension == 2, "torus dimension is wrong")
    _require(torus.string_coupling_power == 2, "torus coupling power is wrong")
    _require(torus.xi_over_bry_known_product == 1.0, "torus map must cancel")
    _require(genus_two.complex_moduli_dimension == 3, "genus-two dimension is wrong")
    _require(genus_two.string_coupling_power == 2, "genus-two coupling power is wrong")
    _require(genus_two.xi_over_bry_real_measure_factor == 8.0, "measure factor is wrong")
    _require(genus_two.xi_over_bry_coupling_weight == 0.25, "coupling factor is wrong")
    _require(genus_two.xi_over_bry_known_product == 2.0, "genus-two bare map is wrong")

    _require(audit["intrinsic_liouville_map"]["factor"] == 1.0, "Liouville map is not one")
    _require(
        math.isclose(
            audit["string_coupling_map"]["g_s_BRY_over_g_s_Xi"],
            2.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "string-coupling map is not two",
    )
    _require(
        math.isclose(
            audit["genus_two_displayed_coefficients"][
                "BRY_extrapolated_over_Xi_positive_real"
            ],
            2.0,
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ),
        "the exposed genus-two coefficient ratio is not two",
    )
    _require(
        not audit["current_code_path"]["apply_g_s_BRY_over_g_s_Xi_to_current_kernel"],
        "the current Xi kernel must not receive a BRY coupling correction",
    )
    _require(
        not audit["two_to_twelve_verdict"]["derived_from_BRY_to_Xi_map"],
        "the BRY/Xi map must not claim to derive 2^12",
    )
    print("BRY-to-Xi convention-map checks passed")


if __name__ == "__main__":
    main()
