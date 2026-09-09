#!/usr/bin/env python3
"""Audit every multiplicative factor in the genus-two c=1 kernel.

The purpose of this script is diagnostic, not cosmetic.  In particular it
distinguishes three logically different outcomes:

``passed``
    An algebraic or numerical test fixes the factor without using the
    genus-two matrix-model answer.

``source_consistent_not_independent``
    The production code implements Xi's stated CFT convention, but the test
    compares two uses of the same convention and therefore cannot exclude a
    common overall rescaling.

``external_comparison``
    The item belongs to the matrix-model comparison rather than to the
    worldsheet integrand.

The audited kernel is

    K_2 = (2/pi) 2^24 (2 pi r/sqrt(alpha')) Theta_r
          (det Im Omega)^-13 |Psi_10|^-2 Z_L/(Z_X,p^code)^25,

with a separate stack weight 1/2 in the moduli integral.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import mpmath as mp
import numpy as np

try:
    from audit_c1_sphere_topology_normalization import build_audit as sphere_audit
    from audit_genus0_one_to_two_amplitude import build_audit as genus0_audit
    from audit_genus2_hyperbolic_volume_sampling import audit_design
    from genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        integration_kernel_scale_to_current,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_kernel_multiplier,
        worldsheet_gauge_fixing_normalization,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
    from liouville_momentum_quadrature import primary_gaussian_momentum_rule
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.audit_c1_sphere_topology_normalization import (
        build_audit as sphere_audit,
    )
    from plumbing.audit_genus0_one_to_two_amplitude import build_audit as genus0_audit
    from plumbing.audit_genus2_hyperbolic_volume_sampling import audit_design
    from plumbing.genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        integration_kernel_scale_to_current,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_kernel_multiplier,
        worldsheet_gauge_fixing_normalization,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
    from plumbing.liouville_momentum_quadrature import primary_gaussian_momentum_rule
    from plumbing.liouville_torus import UpsilonB, yin_structure_constant_momentum


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256"
)
DEFAULT_OUTPUT = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/absolute_normalization_factor_ledger.json"
)
LONG_TUBE_AUDIT = ROOT / (
    "plumbing/results/free_boson_long_tube_normalization/normalization_audit.json"
)
LOCAL_SEWING_AUDIT = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/full_c1_separating_factorization_audit.json"
)
GENUS1_ANCHOR_AUDIT = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/genus1_anchored_sewing_audit.json"
)
RELEASE_SUMMARY = ROOT / "output/data/genus2_c1_free_energy_direct_39/summary.json"


def _relative_error(value: complex | float, target: complex | float) -> float:
    return float(abs(complex(value) / complex(target) - 1.0))


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required audit artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _literal_xi_structure_constant(
    special: UpsilonB,
    momenta: tuple[float, float, float],
) -> complex:
    """Evaluate Xi (4.119) literally, independently of the production wrapper."""

    special._set_precision()
    p_values = tuple(mp.mpf(value) for value in momenta)
    total = sum(p_values)
    value = 1 / special.upsilon(1 + 1j * total)
    for momentum in p_values:
        value *= (
            2
            * momentum
            * special.upsilon(1 + 2j * momentum)
            / special.upsilon(1 + 1j * (total - 2 * momentum))
        )
    return complex(float(mp.re(value)), float(mp.im(value)))


def _dozz_source_audit() -> dict[str, object]:
    special = UpsilonB(1.0, dps=38)
    upsilon_one = complex(special.upsilon(1))
    triples = ((0.17, 0.23, 0.31), (0.11, 0.29, 0.47), (0.19, 0.27, 0.46))
    rows: list[dict[str, object]] = []
    for momenta in triples:
        literal = _literal_xi_structure_constant(special, momenta)
        production = yin_structure_constant_momentum(special, *momenta)
        rows.append(
            {
                "momenta": list(momenta),
                "literal_xi_4_119": [literal.real, literal.imag],
                "production": [production.real, production.imag],
                "relative_error": _relative_error(production, literal),
            }
        )
    maximum_error = max(float(row["relative_error"]) for row in rows)
    return {
        "upsilon_1_of_1": [upsilon_one.real, upsilon_one.imag],
        "upsilon_normalization_error": abs(upsilon_one - 1),
        "generic_nonresonant_points": rows,
        "maximum_relative_error": maximum_error,
        "passed": abs(upsilon_one - 1) < 2.0e-13 and maximum_error < 2.0e-13,
        "scope": (
            "Pointwise implementation of Xi (4.119), including Upsilon_1(1)=1; "
            "the bare singular Liouville cosmological prefactor is not part of "
            "Xi's renormalized b=1 coefficient."
        ),
    }


def _liouville_quadrature_measure_audit() -> dict[str, object]:
    q_value = 1.0e-7
    coefficient = -2.0 * math.log(q_value)
    rule = primary_gaussian_momentum_rule(q_value, 12)
    numerical = math.fsum(
        weight * math.exp(-coefficient * node * node)
        for node, weight in zip(rule.nodes, rule.weights)
    )
    exact = 1.0 / (2.0 * math.sqrt(math.pi * coefficient))
    error = _relative_error(numerical, exact)
    return {
        "integral": "int_0^infinity dP/pi exp(-a P^2)",
        "a": coefficient,
        "numerical": numerical,
        "exact": exact,
        "relative_error": error,
        "passed": error < 2.0e-14,
    }


def _scalar_measure_audit(alpha_prime: float, radius: float) -> dict[str, object]:
    y = np.asarray([[1.31, 0.27], [0.27, 0.94]], dtype=np.float64)
    determinant = float(np.linalg.det(y))
    bare_p_gaussian = determinant**-0.5
    xi_k_gaussian = 1.0 / (
        4.0 * math.pi**2 * alpha_prime * math.sqrt(determinant)
    )
    conversion = xi_genus2_scalar_over_dimensionless(alpha_prime)
    compact_zero_mode = xi_compact_target_zero_mode(radius, alpha_prime)
    net_replacement = xi_full_replacement_over_dimensionless(alpha_prime)
    checks = {
        "code_p_to_xi_k": _relative_error(
            conversion * bare_p_gaussian,
            xi_k_gaussian,
        )
        < 2.0e-15,
        "compact_zero_mode_is_target_length": _relative_error(
            compact_zero_mode,
            2.0 * math.pi * math.sqrt(alpha_prime) * radius,
        )
        < 2.0e-15,
        "correlated_26_minus_25_conversion": _relative_error(
            net_replacement,
            1.0 / (2.0 * math.pi * math.sqrt(alpha_prime)),
        )
        < 2.0e-15,
    }
    return {
        "test_matrix_Y": y.tolist(),
        "det_Y": determinant,
        "bare_code_p_gaussian": bare_p_gaussian,
        "xi_k_gaussian": xi_k_gaussian,
        "Z_X_xi_over_Z_X_code": conversion,
        "compact_zero_mode": compact_zero_mode,
        "full_replacement_over_old_dimensionless_convention": net_replacement,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _matrix_model_coefficient_audit(radius: float) -> dict[str, object]:
    # This derives f_2 from the t^4 term of the universal MQM proper-time
    # kernel; it does not use the numerical worldsheet integral.
    coefficient_t4 = (
        7.0 + 10.0 / radius**2 + 7.0 / radius**4
    ) / 5760.0
    f2 = radius * coefficient_t4
    target_over_gs2 = 16.0 * math.pi**2 * f2
    return {
        "proper_time_t4_coefficient": coefficient_t4,
        "f2": f2,
        "formula": "f2=(7 R^2+10+7 R^-2)/(5760 R)",
        "mu_dictionary_input": "mu^-1=4*pi*g_s^Xi from the 1->2 amplitude",
        "connected_logZ_target_over_gs_squared": target_over_gs2,
        "passed": True,
        "worldsheet_genus2_value_used": False,
    }


def build_audit(
    *,
    alpha_prime: float = 1.0,
    radius: float = 1.0,
    design_dir: Path = DEFAULT_DESIGN,
) -> dict[str, object]:
    alpha_prime = float(alpha_prime)
    radius = float(radius)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")

    sphere = sphere_audit(alpha_prime)
    tree = genus0_audit(alpha_prime=alpha_prime, dps=38)
    dozz = _dozz_source_audit()
    quadrature = _liouville_quadrature_measure_audit()
    scalar = _scalar_measure_audit(alpha_prime, radius)
    volume = audit_design(design_dir)
    long_tube = _load_json(LONG_TUBE_AUDIT)
    local_sewing = _load_json(LOCAL_SEWING_AUDIT)
    genus1_anchor = _load_json(GENUS1_ANCHOR_AUDIT)
    release = _load_json(RELEASE_SUMMARY)
    matrix = _matrix_model_coefficient_audit(radius)

    release_normalization = release["normalization"]
    release_self_dual = release["self_dual_radius"]
    source_convention = str(
        release_normalization["source_integration_kernel_convention"]
    )
    expected_release_scale = integration_kernel_scale_to_current(
        source_convention,
        alpha_prime,
    )
    stored_release_scale = float(release_normalization["source_to_current_kernel_scale"])

    n20 = worldsheet_gauge_fixing_normalization(2, 0)
    real_form_factor = string_note_genus2_complex_form_real_factor()
    critical_multiplier = string_note_genus2_kernel_multiplier(alpha_prime)
    topology_factor = c1_genus2_topology_correction(alpha_prime)
    final_multiplier = c1_sphere_normalized_genus2_kernel_multiplier(alpha_prime)

    last_tube_point = min(
        long_tube["points"],
        key=lambda row: float(row["q_bridge_abs"]),
    )
    long_tube_constant = float(
        long_tube["boundary_extrapolated_normalization_constant"]
    )

    factors: list[dict[str, object]] = [
        {
            "id": "F01",
            "factor": "q = 2*pi*i*Omega_12 + O(Omega_12^3)",
            "role": "separating plumbing coordinate and its Jacobian",
            "value": "2*pi*i",
            "test": "period matrix reconstructed from literal uv=q plumbing",
            "evidence": {
                "smallest_abs_q": last_tube_point["q_bridge_abs"],
                "q_from_period_over_q_input": last_tube_point["q_bridge_period_ratio"],
            },
            "status": "passed",
            "passed": abs(float(last_tube_point["q_bridge_period_ratio"]) - 1.0) < 1.0e-5,
        },
        {
            "id": "F02",
            "factor": "N_2,0 * (-1) * (-8i)",
            "role": "gauge phase, Wick-rotation sign, complex-six-form Jacobian",
            "value": real_form_factor,
            "evidence": {
                "N_2_0": [n20.real, n20.imag],
                "expected_positive_real_product": 8.0,
            },
            "status": "passed",
            "passed": n20 == -1j and _relative_error(real_form_factor, 8.0) < 2.0e-15,
        },
        {
            "id": "F03",
            "factor": "alpha'/(8*pi) times F02",
            "role": "critical-string positive-real external coefficient",
            "value": critical_multiplier,
            "expected": alpha_prime / math.pi,
            "status": "passed",
            "passed": _relative_error(critical_multiplier, alpha_prime / math.pi) < 2.0e-15,
        },
        {
            "id": "F04",
            "factor": "K_S2^crit/Khat_S2^c1 = 2/alpha'",
            "role": "critical-to-c=1 topology replacement at genus two",
            "value": topology_factor,
            "evidence": sphere,
            "status": "passed",
            "passed": bool(sphere["all_checks_pass"]),
        },
        {
            "id": "F05",
            "factor": "(alpha'/pi)*(2/alpha') = 2/pi",
            "role": "final local external kernel multiplier",
            "value": final_multiplier,
            "status": "passed",
            "passed": _relative_error(final_multiplier, 2.0 / math.pi) < 2.0e-15,
        },
        {
            "id": "F06",
            "factor": "2^24",
            "role": "raw Psi_10 to unit nonchiral Mumford residue",
            "value": RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
            "evidence": genus1_anchor["critical_separating_limit"][-1],
            "status": "passed",
            "passed": bool(genus1_anchor["checks"]["critical_residue_is_unit"]),
        },
        {
            "id": "F07",
            "factor": "2*pi*sqrt(alpha')*r and scalar momentum conversions",
            "role": "compact connected zero mode plus the correlated 26-25 scalar bridge",
            "value": scalar,
            "status": "passed",
            "passed": bool(scalar["passed"]),
        },
        {
            "id": "F08",
            "factor": "Theta_r^(2)",
            "role": "compact momentum/winding lattice",
            "value": "harmonic-map sum with action pi*r^2*(m+bar(Omega)n)^T Y^-1 (m+Omega n)",
            "evidence": genus1_anchor["scalar_state_sewing"],
            "status": "passed",
            "passed": bool(genus1_anchor["checks"]["compact_scalar_sewing_is_unit"]),
        },
        {
            "id": "F09",
            "factor": "(det Im Omega)^-13",
            "role": "26 handle-momentum Gaussians",
            "value": "[det(Im Omega)^-1/2]^26",
            "evidence": {
                "scalar_sewing": genus1_anchor["scalar_state_sewing"],
                "absolute_long_tube_constant": long_tube_constant,
            },
            "status": "passed",
            "passed": (
                bool(genus1_anchor["checks"]["noncompact_scalar_sewing_is_unit"])
                and abs(long_tube_constant - 1.0) < 5.0e-10
            ),
        },
        {
            "id": "F10",
            "factor": "|Psi_10|^-2",
            "role": "critical matter-plus-ghost oscillator determinant",
            "value": "raw product of ten even theta constants squared",
            "evidence": genus1_anchor["critical_separating_limit"][-1],
            "status": "passed",
            "passed": bool(genus1_anchor["checks"]["critical_residue_is_unit"]),
        },
        {
            "id": "F11",
            "factor": "C(P1,P2,P3)",
            "role": "one Xi-normalized b=1 Liouville coefficient per pair of pants",
            "value": "Xi (4.119), Upsilon_1(1)=1",
            "evidence": {"generic_points": dozz, "resonance_and_tree": tree},
            "status": "passed",
            "passed": bool(dozz["passed"]) and bool(tree["passed"]),
        },
        {
            "id": "F12",
            "factor": "dP/pi on each Liouville edge",
            "role": "inverse of <V_P V_P'>=pi*delta(P-P')",
            "value": "pi^-3 for a three-edge genus-two pants decomposition",
            "evidence": quadrature,
            "status": "passed",
            "passed": bool(quadrature["passed"]),
        },
        {
            "id": "F13",
            "factor": "Z_L/(Z_X,p^code)^25",
            "role": "Weyl-invariant same-frame c=1 matter replacement",
            "value": "local separating residue 1",
            "evidence": local_sewing,
            "status": "source_consistent_not_independent",
            "passed": bool(local_sewing["all_checks_pass"]),
            "limitation": (
                "The factorization test uses the same Xi three-point coefficient "
                "on both sides.  It certifies implementation and inverse metrics, "
                "but by itself cannot detect a common rescaling of all pants.  F11 "
                "supplies the separate source-anchored pointwise test."
            ),
        },
        {
            "id": "F14",
            "factor": "1/2",
            "role": "generic genus-two hyperelliptic stabilizer/stack weight",
            "value": GENUS2_GENERIC_STACK_WEIGHT,
            "evidence": volume,
            "status": "passed",
            "passed": (
                GENUS2_GENERIC_STACK_WEIGHT == 0.5
                and abs(float(volume["coarse_domain_z_score"])) < 5.0
            ),
        },
        {
            "id": "F15",
            "factor": "saved-kernel convention migration",
            "role": "apply old-to-v3 normalization exactly once",
            "value": stored_release_scale,
            "expected": expected_release_scale,
            "evidence": {
                "source_convention": source_convention,
                "target_convention": release_normalization["integration_kernel_convention"],
            },
            "status": "passed",
            "passed": (
                _relative_error(stored_release_scale, expected_release_scale) < 2.0e-15
                and release_normalization["integration_kernel_convention"]
                == STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
            ),
        },
    ]

    failed = [item["id"] for item in factors if not bool(item["passed"])]
    if failed:
        raise AssertionError(f"factor audits failed: {failed}")

    worldsheet_value = float(
        release_self_dual["connected_logZ_genus2_over_gs_squared"]
    )
    worldsheet_error = float(release_self_dual["connected_logZ_standard_error"])
    matrix_target = float(matrix["connected_logZ_target_over_gs_squared"])
    comparison = {
        "worldsheet_connected_logZ_over_gs_squared": worldsheet_value,
        "worldsheet_standard_error": worldsheet_error,
        "matrix_model_target_over_gs_squared": matrix_target,
        "target_over_worldsheet": matrix_target / worldsheet_value,
        "target_over_worldsheet_standard_error": (
            matrix_target * worldsheet_error / worldsheet_value**2
        ),
        "matrix_coefficient_audit": matrix,
        "status": "external_comparison_failed",
        "interpretation": (
            "Every explicit multiplicative factor in the implemented worldsheet "
            "kernel passes its stated test.  The absolute genus-two comparison "
            "does not.  No test in this ledger derives the discrepancy as an "
            "allowed constant multiplier."
        ),
    }

    return {
        "kernel": (
            "K2=(2/pi)*2^24*(2*pi*r/sqrt(alpha'))*Theta_r*"
            "det(Y)^-13*|Psi10|^-2*Z_L/(Z_X,p^code)^25"
        ),
        "moduli_integral": "F2/g_s^2=(1/2)*int_coarse d^3X d^3Y K2",
        "alpha_prime": alpha_prime,
        "radius": radius,
        "matrix_model_genus2_value_used_to_set_worldsheet_factors": False,
        "all_explicit_factor_tests_pass": True,
        "absolute_normalization_certified": False,
        "why_not_certified": (
            "The worldsheet-to-MQM identification of the complete connected "
            "vacuum functional has not been independently closed at genus two, "
            "and the finite CFT evaluation can still carry a systematic error."
        ),
        "factors": factors,
        "comparison": comparison,
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_audit(
        alpha_prime=args.alpha_prime,
        radius=args.radius,
        design_dir=args.design_dir,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Genus-two factor-by-factor normalization audit")
    for item in payload["factors"]:
        print(f"  {item['id']} {item['status']}: {item['factor']}")
    print("  all explicit factor tests: passed")
    print("  absolute matrix-model comparison: failed")
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
