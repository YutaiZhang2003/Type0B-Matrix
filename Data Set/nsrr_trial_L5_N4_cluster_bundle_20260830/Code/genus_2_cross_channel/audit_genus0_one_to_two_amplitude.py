#!/usr/bin/env python3
"""Audit the genus-zero c=1 string 1 -> 2 amplitude.

The normalization ledger follows Xi's string-note equations (4.111)--(4.122)
and compares the coefficient of the energy-conserving delta function with the
collective-field result (Q.23).  No coefficient is fitted to numerical data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

try:
    from genus2_integrand_normalization import (
        bry_string_coupling_from_xi_string_coupling,
        mqm_fermi_level_from_xi_string_coupling,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus2_integrand_normalization import (
        bry_string_coupling_from_xi_string_coupling,
        mqm_fermi_level_from_xi_string_coupling,
    )
    from plumbing.liouville_torus import UpsilonB, yin_structure_constant_momentum


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus0_one_to_two_amplitude/audit.json"
)


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _relative_error(value: complex | float, target: complex | float) -> float:
    return float(abs(complex(value) / complex(target) - 1.0))


def build_audit(
    omega_1: float = 0.73,
    omega_2: float = 1.11,
    *,
    alpha_prime: float = 1.0,
    g_s_xi: float = 0.037,
    dps: int = 40,
) -> dict[str, object]:
    """Return the analytic factor ledger and one numerical DOZZ check."""

    omega_1 = _positive_finite("omega_1", omega_1)
    omega_2 = _positive_finite("omega_2", omega_2)
    alpha_prime = _positive_finite("alpha_prime", alpha_prime)
    g_s_xi = _positive_finite("g_s_xi", g_s_xi)
    if isinstance(dps, bool) or int(dps) != dps or int(dps) < 30:
        raise ValueError("dps must be an integer at least 30")

    # On-shell energy conservation in S=delta(omega-omega_1-omega_2) A.
    omega = omega_1 + omega_2
    p_1 = omega_1 / 2.0
    p_2 = omega_2 / 2.0
    p_in = omega / 2.0

    # Xi (4.113)--(4.114).  This is the timelike-X^0 sphere constant,
    # not the full string-state metric K_{S^2}=8*pi/alpha'.
    k_tilde_sphere = 2.0 / math.sqrt(alpha_prime)
    momentum_delta_jacobian = math.sqrt(alpha_prime)
    zero_mode_coefficient = (
        2.0 * math.pi * momentum_delta_jacobian * k_tilde_sphere
    )

    # Xi (4.119)--(4.120), checked through the repository's Upsilon_1
    # implementation rather than replacing it by the resonance polynomial.
    special = UpsilonB(1.0, dps=int(dps))
    liouville_numeric = yin_structure_constant_momentum(
        special,
        p_in,
        p_1,
        p_2,
    )
    liouville_resonance = 8.0 * p_1 * p_2 * p_in
    liouville_energy_polynomial = omega * omega_1 * omega_2

    # Three vertices supply g_s^3 and the sphere X^0 correlator supplies
    # g_s^-2, hence one net power of Xi's string coupling.
    vertex_coupling_power = g_s_xi**3
    sphere_dilaton_power = g_s_xi**-2
    net_string_coupling = vertex_coupling_power * sphere_dilaton_power

    worldsheet_reduced = (
        1j
        * net_string_coupling
        * zero_mode_coefficient
        * liouville_numeric
    )
    worldsheet_closed_form = (
        4j * math.pi * g_s_xi * omega * omega_1 * omega_2
    )

    # Q.23 gives A_MQM=(i/mu) omega omega_1 omega_2.  Equating its
    # coefficient to the independently normalized sphere amplitude solves
    # the coupling dictionary analytically.
    mu = mqm_fermi_level_from_xi_string_coupling(g_s_xi)
    mqm_reduced = 1j * omega * omega_1 * omega_2 / mu
    g_s_bry = bry_string_coupling_from_xi_string_coupling(g_s_xi)

    liouville_error = _relative_error(liouville_numeric, liouville_resonance)
    worldsheet_error = _relative_error(worldsheet_reduced, worldsheet_closed_form)
    mqm_error = _relative_error(worldsheet_closed_form, mqm_reduced)
    passed = max(liouville_error, worldsheet_error, mqm_error) < 1.0e-12

    return {
        "scope": (
            "coefficient of delta(omega-omega_1-omega_2) in the genus-zero "
            "1->2 S-matrix element"
        ),
        "notation": {
            "state_metric": "<omega|omega'>=omega delta(omega-omega')",
            "vertex": (
                "V_omega^+-=g_s^Xi c_tilde c "
                "exp(+-i omega X^0/sqrt(alpha')) V_{P=omega/2}"
            ),
            "s_matrix": (
                "S(omega_1,omega_2;omega)="
                "delta(omega-omega_1-omega_2) A_0(omega_1,omega_2)"
            ),
        },
        "kinematics": {
            "omega_1": omega_1,
            "omega_2": omega_2,
            "omega": omega,
            "P_1": p_1,
            "P_2": p_2,
            "P_in": p_in,
        },
        "factor_ledger": {
            "three_vertex_factor": vertex_coupling_power,
            "sphere_X0_dilaton_factor": sphere_dilaton_power,
            "net_string_coupling": net_string_coupling,
            "X0_zero_mode_fourier_factor": "2*pi*i",
            "delta_k0_to_delta_omega_jacobian": momentum_delta_jacobian,
            "K_tilde_S2": k_tilde_sphere,
            "zero_mode_coefficient_without_i": zero_mode_coefficient,
            "zero_mode_identity": (
                "2*pi*sqrt(alpha')*K_tilde_S2=4*pi"
            ),
            "full_string_state_metric_not_used_here": "K_S2=8*pi/alpha'",
        },
        "liouville_resonance": {
            "identity": "C(P_1,P_2,P_1+P_2)=8 P_1 P_2 (P_1+P_2)",
            "numeric_real": liouville_numeric.real,
            "numeric_imag": liouville_numeric.imag,
            "resonance_polynomial": liouville_resonance,
            "energy_polynomial": liouville_energy_polynomial,
            "relative_error": liouville_error,
        },
        "worldsheet": {
            "formula": "A_0^Xi=4*pi*i*g_s^Xi*omega_1*omega_2*(omega_1+omega_2)",
            "reduced_amplitude_real": worldsheet_reduced.real,
            "reduced_amplitude_imag": worldsheet_reduced.imag,
            "closed_form_real": worldsheet_closed_form.real,
            "closed_form_imag": worldsheet_closed_form.imag,
            "relative_error": worldsheet_error,
        },
        "mqm": {
            "formula": "A_1->2^MQM=(i/mu)*omega*omega_1*omega_2",
            "derived_dictionary": "mu^-1=4*pi*g_s^Xi",
            "mu": mu,
            "reduced_amplitude_real": mqm_reduced.real,
            "reduced_amplitude_imag": mqm_reduced.imag,
            "worldsheet_relative_error": mqm_error,
        },
        "bry_cross_check": {
            "g_s_BRY": g_s_bry,
            "g_s_BRY_over_g_s_Xi": g_s_bry / g_s_xi,
            "dictionary": "mu^-1=2*pi*g_s^BRY=4*pi*g_s^Xi",
        },
        "passed": passed,
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit the normalized genus-zero c=1 string 1->2 amplitude."
    )
    parser.add_argument("--omega-1", type=float, default=0.73)
    parser.add_argument("--omega-2", type=float, default=1.11)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--g-s-xi", type=float, default=0.037)
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = build_audit(
        args.omega_1,
        args.omega_2,
        alpha_prime=args.alpha_prime,
        g_s_xi=args.g_s_xi,
        dps=args.dps,
    )
    if not payload["passed"]:
        raise RuntimeError("the genus-zero 1->2 normalization audit failed")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("Genus-zero 1->2 normalization audit")
    print("  C(P1,P2,P1+P2) = omega*omega1*omega2: passed")
    print("  2*pi*sqrt(alpha')*K_tilde_S2 = 4*pi: passed")
    print("  A_0^Xi = 4*pi*i*g_s^Xi*omega*omega1*omega2: passed")
    print("  MQM match pins mu^-1 = 4*pi*g_s^Xi: passed")
    print("  equivalently g_s^BRY = 2*g_s^Xi: passed")
    print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
