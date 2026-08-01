#!/usr/bin/env python3
"""Genus-two normalization audit anchored to the normalized genus-one theory.

This audit deliberately does not use the matrix-model genus-two coefficient.
It compares the separating limit of the critical genus-two Mumford density
with the two once-punctured torus densities that actually occur in plumbing.
It also checks the scalar state metric, compact zero mode, Liouville inverse
metric, Polyakov normalization recurrence, and the single global stack weight.

The important distinction is that a separating genus-two surface produces two
once-punctured tori.  The unpunctured genus-one vacuum density contains an
extra translation-CKV factor ``1/tau2`` and a global stack factor ``1/2``;
using two copies of it in the sewing relation is therefore incorrect.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        igusa_chi10_genus2,
        noncompact_scalar_zero_mode_factor,
    )
    from genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from genus2_integrand_normalization import (
        CODE_SCALAR_OVER_DHP_PER_VOLUME,
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        critical_prefactor_ratio_to_dhp,
        genus1_critical_prefactor_ratio_to_dhp,
        sphere_state_metric_normalization,
        worldsheet_gauge_fixing_normalization,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        igusa_chi10_genus2,
        noncompact_scalar_zero_mode_factor,
    )
    from plumbing.genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from plumbing.genus2_integrand_normalization import (
        CODE_SCALAR_OVER_DHP_PER_VOLUME,
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        critical_prefactor_ratio_to_dhp,
        genus1_critical_prefactor_ratio_to_dhp,
        sphere_state_metric_normalization,
        worldsheet_gauge_fixing_normalization,
    )
    from plumbing.liouville_torus import UpsilonB, yin_structure_constant_momentum


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "genus1_anchored_sewing_audit.json"
)


def _relative_error(value: float | complex, target: float | complex) -> float:
    return float(abs(complex(value) / complex(target) - 1.0))


def _eta_abs(tau: complex) -> float:
    return dedekind_eta_abs_from_q(np.exp(2j * math.pi * complex(tau)))


def once_punctured_torus_critical_density(tau: complex) -> float:
    r"""Critical ``26 X + bc`` coefficient on a once-punctured torus.

    This is the coefficient appropriate to separating plumbing.  It does not
    divide the translation CKV volume and does not apply a vacuum stack weight.
    """

    tau = complex(tau)
    return float(tau.imag**-13.0 * _eta_abs(tau) ** -48.0)


def vacuum_torus_critical_stack_density(tau: complex) -> float:
    r"""Unpunctured critical torus vacuum coefficient on the physical stack."""

    tau = complex(tau)
    return float(
        GENUS2_GENERIC_STACK_WEIGHT
        * tau.imag**-14.0
        * _eta_abs(tau) ** -48.0
    )


def critical_separating_residue(
    tau_left: complex,
    tau_right: complex,
    epsilon: complex,
    *,
    theta_nmax: int,
) -> dict[str, float]:
    r"""Strip ``|dq/q^2|^2`` from the genus-two critical density.

    The raw theta product is used in the code, followed by the exact ``2^24``
    nonchiral conversion.  With ``q=2*pi*i*epsilon``, changing the transverse
    coordinate from ``epsilon`` to ``q`` contributes
    ``|d epsilon/dq|^2=1/(4*pi^2)``.
    """

    omega = np.asarray(
        [[complex(tau_left), complex(epsilon)], [complex(epsilon), complex(tau_right)]],
        dtype=np.complex128,
    )
    y = np.imag(omega)
    det_y = float(np.linalg.det(y))
    raw_product = complex(
        igusa_chi10_genus2(
            omega,
            nmax=int(theta_nmax),
            tol=1.0e-14,
            normalization="product",
        )
    )
    q_bridge = 2j * math.pi * complex(epsilon)
    density_in_epsilon = (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION
        * abs(2j * math.pi) ** 2
        * det_y**-13.0
        / abs(raw_product) ** 2
    )
    density_in_q = density_in_epsilon / (4.0 * math.pi**2)
    stripped_density = density_in_q * abs(q_bridge) ** 4

    punctured_product = (
        once_punctured_torus_critical_density(tau_left)
        * once_punctured_torus_critical_density(tau_right)
    )
    vacuum_product = (
        vacuum_torus_critical_stack_density(tau_left)
        * vacuum_torus_critical_stack_density(tau_right)
    )
    naive_vacuum_expected_ratio = 4.0 * complex(tau_left).imag * complex(tau_right).imag
    return {
        "epsilon_abs": abs(complex(epsilon)),
        "q_bridge_abs": abs(q_bridge),
        "det_im_omega": det_y,
        "stripped_genus2_density": stripped_density,
        "punctured_torus_product": punctured_product,
        "ratio_to_punctured_tori": stripped_density / punctured_product,
        "vacuum_torus_stack_product": vacuum_product,
        "ratio_to_two_vacuum_stack_densities": stripped_density / vacuum_product,
        "naive_vacuum_expected_ratio": naive_vacuum_expected_ratio,
        "naive_vacuum_ratio_residual": (
            stripped_density / vacuum_product / naive_vacuum_expected_ratio - 1.0
        ),
    }


def _compact_winding_sum_genus1(tau: complex, radius: float, nmax: int) -> float:
    integers = range(-int(nmax), int(nmax) + 1)
    return float(
        sum(
            math.exp(
                -math.pi
                * radius**2
                * abs(m + complex(tau) * n) ** 2
                / complex(tau).imag
            )
            for m in integers
            for n in integers
        )
    )


def scalar_state_sewing(
    tau_left: complex,
    tau_right: complex,
    *,
    radius: float,
    lattice_nmax: int,
) -> dict[str, float | str]:
    """Check noncompact and compact scalar separating coefficients."""

    omega = np.asarray(
        [[complex(tau_left), 0.0], [0.0, complex(tau_right)]],
        dtype=np.complex128,
    )
    scalar_genus2 = noncompact_scalar_zero_mode_factor(omega)
    scalar_tori = (complex(tau_left).imag * complex(tau_right).imag) ** -0.5

    theta_genus2 = compact_boson_winding_sum_genus2(
        omega,
        radius,
        lattice_nmax=lattice_nmax,
    )
    theta_left = _compact_winding_sum_genus1(tau_left, radius, lattice_nmax)
    theta_right = _compact_winding_sum_genus1(tau_right, radius, lattice_nmax)
    compact_genus2 = radius * scalar_genus2 * theta_genus2
    compact_left = radius * complex(tau_left).imag**-0.5 * theta_left
    compact_right = radius * complex(tau_right).imag**-0.5 * theta_right
    sewn_compact = compact_left * compact_right / radius
    return {
        "noncompact_state_metric": "<p|p'>=delta(p-p'), completeness dp",
        "compact_vacuum_metric": "<0|0>=R, inverse metric 1/R",
        "noncompact_gaussian_ratio": scalar_genus2 / scalar_tori,
        "winding_sum_ratio": theta_genus2 / (theta_left * theta_right),
        "compact_partition_ratio": compact_genus2 / sewn_compact,
    }


def liouville_state_sewing() -> dict[str, float | str | bool]:
    """Check the BRY/Xi Liouville normalization used on every sewn edge."""

    special = UpsilonB(1.0, dps=36)
    p1 = 0.17
    p2 = 0.23
    p3 = p1 + p2
    value = yin_structure_constant_momentum(special, p3, p1, p2)
    expected = (2.0 * p1) * (2.0 * p2) * (2.0 * p3)
    error = _relative_error(value, expected)
    return {
        "two_point_metric": "<V_P V_P'>=pi delta(P-P')",
        "inverse_metric": "dP/pi",
        "dozz_resonance_value_real": value.real,
        "dozz_resonance_value_imag": value.imag,
        "dozz_resonance_expected": expected,
        "dozz_resonance_relative_error": error,
        "passed": error < 1.0e-12,
    }


def polyakov_topology_sewing(alpha_prime: float) -> dict[str, object]:
    """Check Xi equations (4.69)--(4.71) for the separating channel."""

    sphere_metric = sphere_state_metric_normalization(alpha_prime)
    n_20 = worldsheet_gauge_fixing_normalization(2, 0)
    n_11 = worldsheet_gauge_fixing_normalization(1, 1)
    inverse_sphere_metric_factor = 8.0 * math.pi / (alpha_prime * sphere_metric)
    recurrence_right = -1j * n_20 * inverse_sphere_metric_factor
    return {
        "N_2_0": {"real": n_20.real, "imag": n_20.imag},
        "N_1_1": {"real": n_11.real, "imag": n_11.imag},
        "sphere_state_metric_K_S2": sphere_metric,
        "inverse_sphere_metric_factor_8pi_over_alpha_K": inverse_sphere_metric_factor,
        "N_1_1_squared": {"real": (n_11**2).real, "imag": (n_11**2).imag},
        "recurrence_right": {
            "real": recurrence_right.real,
            "imag": recurrence_right.imag,
        },
        "relative_error": _relative_error(n_11**2, recurrence_right),
    }


def critical_scalar_convention_bridge() -> dict[str, float | str]:
    """Combine the critical and 26-scalar convention changes in one ratio."""

    genus1_critical = genus1_critical_prefactor_ratio_to_dhp()
    genus2_critical = critical_prefactor_ratio_to_dhp()
    scalar_26 = CODE_SCALAR_OVER_DHP_PER_VOLUME**26
    return {
        "one_scalar_repo_over_dhp_per_volume": CODE_SCALAR_OVER_DHP_PER_VOLUME,
        "twenty_six_scalar_ratio": scalar_26,
        "critical_ratio_genus1": genus1_critical,
        "critical_ratio_genus2": genus2_critical,
        "genus1_critical_over_scalar26": genus1_critical / scalar_26,
        "genus2_critical_over_scalar26": genus2_critical / scalar_26,
        "interpretation": (
            "The critical Mumford coefficient and the 26 scalar determinants "
            "must be converted together; their convention bridge is one."
        ),
    }


def build_audit(
    *,
    tau_left: complex,
    tau_right: complex,
    epsilon_values: tuple[complex, ...],
    radius: float,
    theta_nmax: int,
    lattice_nmax: int,
    alpha_prime: float,
) -> dict[str, object]:
    critical = [
        critical_separating_residue(
            tau_left,
            tau_right,
            epsilon,
            theta_nmax=theta_nmax,
        )
        for epsilon in epsilon_values
    ]
    punctured_errors = [abs(float(row["ratio_to_punctured_tori"]) - 1.0) for row in critical]
    scalar = scalar_state_sewing(
        tau_left,
        tau_right,
        radius=radius,
        lattice_nmax=lattice_nmax,
    )
    liouville = liouville_state_sewing()
    polyakov = polyakov_topology_sewing(alpha_prime)
    convention = critical_scalar_convention_bridge()

    checks = {
        "critical_residue_converges": punctured_errors[-1] < punctured_errors[0],
        "critical_residue_is_unit": punctured_errors[-1] < 2.0e-6,
        "naive_vacuum_formula_fails_as_predicted": abs(
            float(critical[-1]["naive_vacuum_ratio_residual"])
        )
        < 2.0e-6,
        "noncompact_scalar_sewing_is_unit": abs(
            float(scalar["noncompact_gaussian_ratio"]) - 1.0
        )
        < 2.0e-14,
        "compact_scalar_sewing_is_unit": abs(
            float(scalar["compact_partition_ratio"]) - 1.0
        )
        < 2.0e-12,
        "liouville_inverse_metric_is_unit": bool(liouville["passed"]),
        "polyakov_topology_recurrence_is_unit": float(polyakov["relative_error"])
        < 2.0e-15,
        "critical_scalar_convention_bridge_is_unit": abs(
            float(convention["genus2_critical_over_scalar26"]) - 1.0
        )
        < 2.0e-15,
        "single_global_genus2_stack_weight": GENUS2_GENERIC_STACK_WEIGHT == 0.5,
    }
    all_pass = all(checks.values())
    if not all_pass:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"genus-one anchored sewing audit failed: {failed}")

    return {
        "scope": (
            "Matrix-model-independent separating-sewing normalization of the "
            "current genus-two critical-boson replacement, anchored to genus one."
        ),
        "source_equations": {
            "genus1_vacuum": "Xi string notes (4.90)-(4.93)",
            "plumbing_and_state_metric": "Xi string notes (4.56)-(4.71)",
            "genus2_critical_measure": "Xi string notes (4.103)-(4.109)",
        },
        "parameters": {
            "tau_left": [complex(tau_left).real, complex(tau_left).imag],
            "tau_right": [complex(tau_right).real, complex(tau_right).imag],
            "epsilon_abs_values": [abs(value) for value in epsilon_values],
            "radius": radius,
            "theta_nmax": theta_nmax,
            "lattice_nmax": lattice_nmax,
            "alpha_prime": alpha_prime,
        },
        "critical_separating_limit": critical,
        "scalar_state_sewing": scalar,
        "liouville_state_sewing": liouville,
        "polyakov_topology_sewing": polyakov,
        "critical_scalar_convention_bridge": convention,
        "stack_ledger": {
            "genus1_vacuum_stack_weight": 0.5,
            "genus2_global_stack_weight": GENUS2_GENERIC_STACK_WEIGHT,
            "component_vacuum_stack_weights_used_in_sewing": False,
            "reason": (
                "The components are once-punctured tori. Their translation CKV "
                "and unpunctured-vacuum stack factors are not part of the local "
                "sewing residue; the smooth genus-two stack weight is applied once."
            ),
        },
        "checks": checks,
        "all_checks_pass": all_pass,
        "matrix_model_used": False,
        "result": {
            "xi_code_local_bridge": 1.0,
            "lambda_full_in_declared_common_state_convention": 1.0,
            "extra_factor_2_to_12_derived": False,
            "conclusion": (
                "The genus-one-anchored separating sewing supplies no missing "
                "overall multiplier. The current critical/scalar replacement "
                "normalization is unity; two unpunctured genus-one vacuum "
                "densities are the wrong factorization objects."
            ),
        },
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit genus-two normalization from genus-one separating sewing."
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit = build_audit(
        tau_left=0.17 + 1.13j,
        tau_right=-0.21 + 0.91j,
        epsilon_values=(
            0.025j,
            0.0125j,
            0.00625j,
            0.003125j,
            0.0015625j,
            0.00078125j,
            0.000390625j,
        ),
        radius=1.23,
        theta_nmax=12,
        lattice_nmax=9,
        alpha_prime=1.0,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    final_critical = audit["critical_separating_limit"][-1]
    print("Genus-two normalization from genus-one separating sewing")
    print(
        "  critical / punctured-torus product = "
        f"{float(final_critical['ratio_to_punctured_tori']):.16e}"
    )
    print(
        "  critical / two vacuum densities    = "
        f"{float(final_critical['ratio_to_two_vacuum_stack_densities']):.16e}"
    )
    print(
        "  predicted naive-vacuum ratio        = "
        f"{float(final_critical['naive_vacuum_expected_ratio']):.16e}"
    )
    print(
        "  compact scalar sewing ratio         = "
        f"{float(audit['scalar_state_sewing']['compact_partition_ratio']):.16e}"
    )
    print(
        "  critical/scalar convention bridge   = "
        f"{float(audit['critical_scalar_convention_bridge']['genus2_critical_over_scalar26']):.16e}"
    )
    print(
        "  Polyakov recurrence error           = "
        f"{float(audit['polyakov_topology_sewing']['relative_error']):.3e}"
    )
    print("  lambda_full                          = 1")
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
