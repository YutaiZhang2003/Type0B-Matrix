#!/usr/bin/env python3
"""Audit the sphere-topology normalization in the genus-two c=1 kernel.

This is an analytic convention audit.  It does not use the matrix-model
genus-two free energy and does not infer a constant from numerical data.

The critical seed carries Xi's critical-string sphere metric

    K_S2 = 8*pi/alpha'

through the genus-two coefficient g_s^2/K_S2.  The c=1 scattering convention
instead normalizes the time-like zero mode by

    K_tilde_S2 = 2/sqrt(alpha').

After compactification, separate the complete sphere metric into

    K_res_c1 * (2*pi*r),
    K_res_c1 = sqrt(alpha')*K_tilde_S2 = 2.

The explicit compact theta-graph density already contains the inverse
Fourier metric ``1/(2*pi*r)``.  Topology sewing therefore replaces the
critical complex-form coefficient ``g_s^2/K_S2`` by the fixed
``g_s^2/K_res_c1``.
The ratio is ``4*pi/alpha'``.  The positive-real six-form Jacobian is common
to both theories and does not alter this ratio.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

try:
    from genus2_integrand_normalization import (
        GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT,
        c1_genus_topology_correction,
        c1_genus2_topology_correction,
        c1_reduced_sphere_metric,
        c1_residual_topology_metric,
        c1_sphere_normalized_genus2_kernel_multiplier,
        c1_timelike_sphere_constant,
        sphere_state_metric_normalization,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus2_integrand_normalization import (
        GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT,
        c1_genus_topology_correction,
        c1_genus2_topology_correction,
        c1_reduced_sphere_metric,
        c1_residual_topology_metric,
        c1_sphere_normalized_genus2_kernel_multiplier,
        c1_timelike_sphere_constant,
        sphere_state_metric_normalization,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "c1_sphere_topology_normalization_audit.json"
)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) / float(target) - 1.0)


def build_audit(alpha_prime: float = 1.0) -> dict[str, object]:
    """Return the critical-to-c=1 topology normalization ledger."""

    alpha_prime = _positive_finite("alpha_prime", alpha_prime)

    critical_sphere_metric = sphere_state_metric_normalization(alpha_prime)
    timelike_sphere_constant = c1_timelike_sphere_constant(alpha_prime)
    energy_delta_jacobian = math.sqrt(alpha_prime)
    energy_delta_sphere_metric = c1_reduced_sphere_metric(alpha_prime)
    residual_topology_metric = c1_residual_topology_metric(alpha_prime)

    critical_complex_coefficient = 1.0 / critical_sphere_metric
    c1_complex_coefficient = 1.0 / residual_topology_metric
    topology_ratio = c1_genus2_topology_correction(alpha_prime)

    real_form_factor = string_note_genus2_complex_form_real_factor()
    critical_positive_coefficient = (
        real_form_factor * critical_complex_coefficient
    )
    c1_positive_coefficient = real_form_factor * c1_complex_coefficient

    inherited_critical_kernel_multiplier = string_note_genus2_kernel_multiplier(
        alpha_prime
    )
    sphere_normalized_pre_brst_multiplier = (
        inherited_critical_kernel_multiplier * topology_ratio
    )
    complete_v5_kernel_multiplier = (
        c1_sphere_normalized_genus2_kernel_multiplier(alpha_prime)
    )
    required_over_inherited = (
        sphere_normalized_pre_brst_multiplier
        / inherited_critical_kernel_multiplier
    )

    # Independent BRY cross-check.  Their real-measure genus-two topology
    # coefficient is (g_s^BRY)^2/(2*pi).  Since g_s^BRY=2*g_s^Xi, stripping
    # (g_s^Xi)^2 gives 2/pi.
    bry_over_xi_coupling = 2.0
    bry_positive_coefficient_in_xi_coupling = (
        bry_over_xi_coupling**2 / (2.0 * math.pi)
    )

    scalar_measure_conversion = xi_full_replacement_over_dimensionless(alpha_prime)
    scalar_conversion_as_k_tilde = timelike_sphere_constant / (4.0 * math.pi)

    checks = {
        "critical_coefficient_is_inverse_K_S2": _relative_error(
            critical_complex_coefficient,
            alpha_prime / (8.0 * math.pi),
        )
        < 2.0e-15,
        "c1_energy_delta_sphere_metric_is_4pi": _relative_error(
            energy_delta_sphere_metric,
            4.0 * math.pi,
        )
        < 2.0e-15,
        "c1_residual_topology_metric_is_2": _relative_error(
            residual_topology_metric,
            2.0,
        )
        < 2.0e-15,
        "topology_ratio_is_4pi_over_alpha_prime": _relative_error(
            topology_ratio,
            4.0 * math.pi / alpha_prime,
        )
        < 2.0e-15,
        "genus_one_vacuum_topology_ratio_is_one": (
            c1_genus_topology_correction(1, alpha_prime) == 1.0
        ),
        "inherited_kernel_is_critical_coefficient": _relative_error(
            inherited_critical_kernel_multiplier,
            critical_positive_coefficient,
        )
        < 2.0e-15,
        "required_c1_positive_coefficient_is_4": _relative_error(
            sphere_normalized_pre_brst_multiplier,
            4.0,
        )
        < 2.0e-15,
        "no_extra_global_modular_weight": GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT == 1.0,
    }

    return {
        "scope": (
            "absolute topology coefficient multiplying the already normalized "
            "local genus-two c=1 CFT density"
        ),
        "matrix_model_genus2_value_used": False,
        "numerical_fit_used": False,
        "alpha_prime": alpha_prime,
        "critical_string": {
            "K_S2": critical_sphere_metric,
            "complex_form_coefficient_with_gs2_stripped": (
                critical_complex_coefficient
            ),
            "positive_real_coefficient_with_gs2_stripped": (
                critical_positive_coefficient
            ),
            "identity": "g_s^2*alpha'/(8*pi)=g_s^2/K_S2",
        },
        "c1_string": {
            "K_tilde_S2": timelike_sphere_constant,
            "delta_k0_to_delta_omega_jacobian": energy_delta_jacobian,
            "energy_delta_sphere_metric": energy_delta_sphere_metric,
            "energy_delta_sphere_metric_identity": (
                "Khat_c1=2*pi*sqrt(alpha')*K_tilde_S2=4*pi"
            ),
            "residual_topology_metric": residual_topology_metric,
            "residual_topology_metric_identity": (
                "K_res_c1=sqrt(alpha')*K_tilde_S2=2"
            ),
            "complex_form_coefficient_with_gs2_stripped": c1_complex_coefficient,
            "positive_real_coefficient_with_gs2_stripped": c1_positive_coefficient,
        },
        "critical_to_c1_topology_replacement": {
            "genus_g_formula": "(4*pi/alpha')^(g-1)",
            "genus_one_value": c1_genus_topology_correction(1, alpha_prime),
            "genus_one_note": (
                "No multiplicative torus-vacuum factor. A constant g_s--mu "
                "conversion may shift only the additive part of log(mu), not "
                "its universal coefficient."
            ),
            "formula": (
                "Lambda_top=K_S2/Khat_c1="
                "K_S2/(sqrt(alpha')*K_tilde_S2)"
            ),
            "value": topology_ratio,
            "closed_form": "4*pi/alpha'",
        },
        "current_code": {
            "inherited_critical_kernel_multiplier": (
                inherited_critical_kernel_multiplier
            ),
            "inherited_kernel_multiplier_origin": (
                "critical string-note coefficient alpha'/pi"
            ),
            "sphere_normalized_pre_brst_kernel_multiplier": (
                sphere_normalized_pre_brst_multiplier
            ),
            "sphere_topology_over_inherited": required_over_inherited,
            "sphere_topology_over_inherited_closed_form": "4*pi/alpha'",
            "complete_production_v5_kernel_multiplier": (
                complete_v5_kernel_multiplier
            ),
            "fixed_puncture_brst_multiplier_applied_separately": (
                complete_v5_kernel_multiplier
                / sphere_normalized_pre_brst_multiplier
            ),
            "fundamental_domain_weight": (
                GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT
            ),
        },
        "scalar_measure_conversion_warning": {
            "xi_full_replacement_over_dimensionless": scalar_measure_conversion,
            "same_number_as_K_tilde_over_4pi": scalar_conversion_as_k_tilde,
            "equal": _relative_error(
                scalar_measure_conversion,
                scalar_conversion_as_k_tilde,
            )
            < 2.0e-15,
            "interpretation": (
                "This equality is not a topology audit: the code derives this "
                "factor from loop-momentum measures and the compact zero mode. "
                "The separate external kernel multiplier still uses K_S2."
            ),
        },
        "historical_bry_comparison_not_used": {
            "g_s_BRY_over_g_s_Xi": bry_over_xi_coupling,
            "C_Sigma2_over_g_s_Xi_squared": (
                bry_positive_coefficient_in_xi_coupling
            ),
            "identity": (
                "(g_s^BRY)^2/(2*pi)="
                "2*(g_s^Xi)^2/pi"
            ),
            "note": (
                "This coefficient uses BRY's real-measure convention and is "
                "not the compact Xi marked-plumbing coefficient."
            ),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "verdict": (
            "The inherited alpha'/pi multiplier is the critical-string topology "
            "coefficient. The compact sphere-normalized marked coefficient is 4, "
            "larger by 4*pi/alpha'. The separate fixed-puncture BRST audit then "
            "converts it to the complete v5 coefficient."
        ),
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit the c=1 sphere normalization in the genus-two kernel."
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_audit(args.alpha_prime)
    if not payload["all_checks_pass"]:
        failed = [
            name for name, passed in payload["checks"].items() if not passed
        ]
        raise RuntimeError(f"sphere-topology normalization audit failed: {failed}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    replacement = payload["critical_to_c1_topology_replacement"]
    code = payload["current_code"]
    print("c=1 sphere-topology normalization audit")
    print("  no matrix-model genus-two input: yes")
    print("  K_S2 critical             = 8*pi/alpha'")
    print("  Khat_S2 c=1               = 4*pi")
    print("  fixed K_res,S2 c=1        = 2")
    print(f"  Lambda_top                = {replacement['value']:.16e}")
    print(
        "  inherited critical kernel = "
        f"{code['inherited_critical_kernel_multiplier']:.16e}"
    )
    print(
        "  sphere-normalized pre-BRST = "
        f"{code['sphere_normalized_pre_brst_kernel_multiplier']:.16e}"
    )
    print(
        "  sphere topology/inherited = "
        f"{code['sphere_topology_over_inherited']:.16e}"
    )
    print(
        "  complete production v5    = "
        f"{code['complete_production_v5_kernel_multiplier']:.16e}"
    )
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
