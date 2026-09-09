#!/usr/bin/env python3
"""Audit the convention map from BRY to Xi's string notes.

This file deliberately separates intrinsic Liouville CFT data from a full
gauge-fixed string amplitude.  The former agree literally.  The latter also
contains the string coupling, moduli differential, scalar/ghost state metric,
and moduli-stack convention, so it cannot be converted by multiplying the
Liouville block by a power of two.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

try:
    from genus2_integrand_normalization import (
        BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED,
        BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
        BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        bry_genus2_relative_topology_normalization,
        bry_string_coupling_from_mqm_fermi_level,
        bry_xi_bare_convention_map,
        string_note_genus2_full_kernel_multiplier,
        xi_string_coupling_from_mqm_fermi_level,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus2_integrand_normalization import (
        BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED,
        BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
        BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        bry_genus2_relative_topology_normalization,
        bry_string_coupling_from_mqm_fermi_level,
        bry_xi_bare_convention_map,
        string_note_genus2_full_kernel_multiplier,
        xi_string_coupling_from_mqm_fermi_level,
    )


def _map_payload(genus: int, punctures: int) -> dict[str, object]:
    result = bry_xi_bare_convention_map(genus, punctures)
    return {
        "genus": result.genus,
        "punctures": result.punctures,
        "complex_moduli_dimension": result.complex_moduli_dimension,
        "string_coupling_power": result.string_coupling_power,
        "xi_over_bry_real_measure_factor": result.xi_over_bry_real_measure_factor,
        "xi_over_bry_coupling_weight": result.xi_over_bry_coupling_weight,
        "xi_over_bry_known_product": result.xi_over_bry_known_product,
        "scope": (
            "measure and genus-counting coupling only; excludes state metrics, "
            "critical-to-c=1 replacement constants, and automorphism quotients"
        ),
    }


def build_audit(fermi_level: float = 2.7) -> dict[str, object]:
    """Return the reviewer-facing BRY/Xi convention ledger."""

    mu = float(fermi_level)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("fermi_level must be positive and finite")

    g_xi = xi_string_coupling_from_mqm_fermi_level(mu)
    g_bry = bry_string_coupling_from_mqm_fermi_level(mu)
    bry_genus2_extrapolation = bry_genus2_relative_topology_normalization(g_bry)
    xi_complex_form_coefficient = g_xi**2 / (8.0 * math.pi)
    xi_positive_real_coefficient = string_note_genus2_full_kernel_multiplier(g_xi)

    return {
        "scope": "BRY arXiv:1705.07151 versus Xi string-note equations 4.58-4.122",
        "intrinsic_liouville_map": {
            "factor": 1.0,
            "certified": BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
            "matching_data": [
                "V_P=S(P)^(-1/2)V_in",
                "<V_P V_P'>=pi delta(P-P')",
                "completeness measure dP/pi",
                "the b=1 DOZZ coefficient with Upsilon_1(1)=1",
                "the literal plumbing relation u*v=q",
            ],
            "conclusion": (
                "No A_L multiplies the BRY Liouville partition when it is used "
                "inside Xi's cutting-and-sewing construction."
            ),
        },
        "string_coupling_map": {
            "mu": mu,
            "g_s_BRY": g_bry,
            "g_s_Xi": g_xi,
            "g_s_BRY_over_g_s_Xi": g_bry / g_xi,
            "BRY_dictionary": "mu^-1=2*pi*g_s_BRY",
            "Xi_dictionary": "mu^-1=4*pi*g_s_Xi",
            "certified": BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        },
        "bare_measure_and_coupling_checks": {
            "sphere_four_point": _map_payload(0, 4),
            "torus_two_point": _map_payload(1, 2),
            "genus_two_vacuum": _map_payload(2, 0),
            "lower_genus_anchor": (
                "For the torus two-point amplitude Xi has two complex moduli: "
                "2^2 from i dz wedge dbar(z) cancels "
                "(g_s_Xi/g_s_BRY)^2=1/4."
            ),
        },
        "genus_two_displayed_coefficients": {
            "BRY_extrapolated_C_Sigma2": bry_genus2_extrapolation,
            "BRY_extrapolation_assumption": "C_Sigma2*C_S2=C_T2^2",
            "Xi_complex_six_form_coefficient": xi_complex_form_coefficient,
            "Xi_positive_real_coefficient": xi_positive_real_coefficient,
            "BRY_extrapolated_over_Xi_positive_real": (
                bry_genus2_extrapolation / xi_positive_real_coefficient
            ),
            "status": "not yet an equality of identically normalized full amplitudes",
            "reason": (
                "BRY do not write the genus-two vacuum amplitude. Their displayed "
                "lower-genus constants and Xi's critical-boson coefficient bundle "
                "different state-metric, scalar/ghost, and automorphism conventions."
            ),
        },
        "current_code_path": {
            "liouville_data": "BRY/Xi common intrinsic CFT normalization",
            "moduli_differential": "Xi positive real period-coordinate convention",
            "string_coupling": "Xi, with mu^-1=4*pi*g_s_Xi",
            "generic_genus_two_stack_weight": "applied separately by the integrator",
            "apply_g_s_BRY_over_g_s_Xi_to_current_kernel": False,
            "apply_bare_measure_map_to_current_kernel": False,
            "reason": (
                "The current kernel is not a full BRY amplitude. It only imports "
                "Liouville CFT data, whose BRY-to-Xi factor is one."
            ),
        },
        "full_partition_bridge": {
            "general_formula": "I_Xi/I_code=A_crit*A_XR/A_X^26",
            "common_scalar_formula": "I_Xi/I_code=A_crit/A_X^25 when A_XR=A_X",
            "liouville_factor": 1.0,
            "derived_value": 1.0,
            "certified": BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED,
            "independent_cross_check": (
                "Sew the normalized free-boson plus bc-ghost system in the same "
                "pants coordinates and compare it directly with Xi equation 4.105; "
                "the genus-one-anchored separating audit already fixes the bridge."
            ),
        },
        "two_to_twelve_verdict": {
            "derived_from_BRY_to_Xi_map": False,
            "known_genus_two_measure_factor": 8.0,
            "known_genus_two_coupling_weight": 0.25,
            "known_product": 2.0,
            "conclusion": (
                "The verified BRY/Xi coupling and differential-form changes do not "
                "produce 2^12. Applying either change to the current kernel would "
                "double-count conventions already implemented."
            ),
        },
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the BRY-to-Xi convention map.")
    parser.add_argument("--mu", type=float, default=2.7)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("plumbing/results/genus2_c1_moduli_mc/bry_xi_convention_map.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = build_audit(args.mu)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("BRY-to-Xi convention audit")
    print("  intrinsic Liouville factor: 1")
    print("  g_s^BRY/g_s^Xi: 2")
    print("  genus-two bare measure factor: 8")
    print("  genus-two coupling weight: 1/4")
    print("  apply these factors to current kernel: no")
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
