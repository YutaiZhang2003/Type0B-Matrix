#!/usr/bin/env python3
"""Pointwise genus-two separating match anchored to the genus-one integrand.

This audit starts from the tested string-note genus-one vacuum density and
does not use the matrix-model genus-two coefficient.  It compares three
normalization layers at

    Omega = [[tau_left, epsilon], [epsilon, tau_right]],
    q_bridge = 2 pi i epsilon -> 0.

The layers are:

1. the previously proposed sewing formula made from integrated torus
   one-point amplitudes J_11;
2. the fixed-puncture CFT blocks B_11, including the inverse compact-vacuum
   metric in the dimensionless sewing convention;
3. the stack-weighted, factorization-normalized Mumford density.

All calculations use alpha'=1.  The Liouville torus one-point functions and
their bridge integral use the Xi/BRY measure dP/pi.  No fitted normalization
or matrix-model input appears.
"""

from __future__ import annotations

import argparse
import cmath
import math
from typing import Iterable

import numpy as np

try:
    from free_boson_plumbing import dedekind_eta_abs_from_q
    from genus2_c1_string_integrand import (
        SameFrameMatterPartitions,
        genus2_c1_string_integrand_density,
    )
    from integrate_tau_compact_liouville import (
        string_note_genus1_integrand_per_liouville_volume,
    )
    from liouville_genus2 import liouville_genus2_pair_of_tori
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.free_boson_plumbing import dedekind_eta_abs_from_q
    from plumbing.genus2_c1_string_integrand import (
        SameFrameMatterPartitions,
        genus2_c1_string_integrand_density,
    )
    from plumbing.integrate_tau_compact_liouville import (
        string_note_genus1_integrand_per_liouville_volume,
    )
    from plumbing.liouville_genus2 import liouville_genus2_pair_of_tori


ALPHA_PRIME = 1.0


def _tau_from_q(q_value: complex) -> complex:
    q_value = complex(q_value)
    if not 0.0 < abs(q_value) < 1.0:
        raise ValueError("handle nomes must have modulus strictly between zero and one")
    return cmath.log(q_value) / (2j * math.pi)


def _canonical_scalar_torus(tau: complex) -> float:
    r"""Return ``tau2^-1/2 |eta(tau)|^-2`` in the dimensionless convention."""

    q_value = cmath.exp(2j * math.pi * complex(tau))
    eta_abs = dedekind_eta_abs_from_q(q_value)
    return float(complex(tau).imag**-0.5 * eta_abs**-2)


def _bry_liouville_vacuum_trace(tau: complex) -> float:
    r"""Return ``int_0^inf dP/pi |q|^(2 P^2)/|eta|^2``."""

    q_value = cmath.exp(2j * math.pi * complex(tau))
    eta_abs = dedekind_eta_abs_from_q(q_value)
    return float(1.0 / (4.0 * math.pi * math.sqrt(complex(tau).imag) * eta_abs**2))


def evaluate_point(
    *,
    q_left: complex,
    q_right: complex,
    q_bridge: float,
    radius: float,
    bridge_p_max: float,
    handle_p_max: float,
    bridge_quadrature_order: int,
    handle_quadrature_order: int,
    theta_cutoff: int,
    theta_nmax: int,
    dps: int,
) -> dict[str, float]:
    """Evaluate one point in the separating family."""

    tau_left = _tau_from_q(q_left)
    tau_right = _tau_from_q(q_right)
    q_bridge_complex = complex(float(q_bridge), 0.0)
    epsilon = q_bridge_complex / (2j * math.pi)
    omega = np.asarray(
        [[tau_left, epsilon], [epsilon, tau_right]],
        dtype=np.complex128,
    )

    liouville = liouville_genus2_pair_of_tori(
        b=1.0,
        q1=q_left,
        q2=q_right,
        q_bridge=q_bridge_complex,
        block_order=1,
        bridge_p_max=bridge_p_max,
        handle_p_max=handle_p_max,
        bridge_quadrature_order=bridge_quadrature_order,
        handle_quadrature_order=handle_quadrature_order,
        dps=dps,
        include_bridge_vacuum_energy=True,
        include_cosmological_prefactor=False,
    )
    liouville_value = float(liouville.value.real)
    if liouville_value <= 0.0 or abs(liouville.value.imag) > 1.0e-10 * liouville_value:
        raise ValueError("Liouville bridge integral is not positive real")

    # The canonical scalar and Liouville quantities contain the same tube
    # Casimir factor |q_bridge|^-1/12.  It cancels in Z_L/Z_X^25 together
    # with the corresponding critical seed powers.
    scalar_genus2 = (
        _canonical_scalar_torus(tau_left)
        * _canonical_scalar_torus(tau_right)
        * abs(q_bridge_complex) ** (-1.0 / 12.0)
    )
    full = genus2_c1_string_integrand_density(
        omega,
        radius,
        alpha_prime=ALPHA_PRIME,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame="plumbing:glasses",
            liouville_partition=liouville_value,
            noncompact_scalar_partition=scalar_genus2,
        ),
        lattice_tolerance=1.0e-14,
        theta_nmax=theta_nmax,
        theta_tolerance=1.0e-14,
        chi10_normalization="product",
    )

    i1_left = string_note_genus1_integrand_per_liouville_volume(
        tau_left,
        radius=radius,
        theta_cutoff=theta_cutoff,
        alpha_prime=ALPHA_PRIME,
    )
    i1_right = string_note_genus1_integrand_per_liouville_volume(
        tau_right,
        radius=radius,
        theta_cutoff=theta_cutoff,
        alpha_prime=ALPHA_PRIME,
    )
    zl0_left = _bry_liouville_vacuum_trace(tau_left)
    zl0_right = _bry_liouville_vacuum_trace(tau_right)

    fixed_block_integral = 0.0
    direct_fixed_block_integral = 0.0
    for sample in liouville.samples:
        p_bridge = sample.bridge_momentum
        bridge_primary = abs(q_bridge_complex) ** (2.0 * p_bridge * p_bridge)

        # This is the requested expression in terms of the tested genus-one
        # density.  At alpha'=1,
        # B_11(P;tau)=tau2 I_1^note(tau) T_P(tau)/Z_L,0^BRY(tau).
        b_left = (
            tau_left.imag
            * i1_left
            * float(sample.left_torus_one_point.real)
            / zl0_left
        )
        b_right = (
            tau_right.imag
            * i1_right
            * float(sample.right_torus_one_point.real)
            / zl0_right
        )
        fixed_block_integral += (
            sample.bridge_measure_weight * bridge_primary * b_left * b_right
        )

        # Independent component formula, used only as a guard on the I_1
        # rewriting above: B_11=r Theta_R |eta|^2 T_P/sqrt(tau2).
        eta_left = dedekind_eta_abs_from_q(q_left)
        eta_right = dedekind_eta_abs_from_q(q_right)
        theta_left = (
            4.0 * math.pi * tau_left.imag**2 * i1_left / radius
        )
        theta_right = (
            4.0 * math.pi * tau_right.imag**2 * i1_right / radius
        )
        b_left_direct = (
            radius
            * theta_left
            * eta_left**2
            * float(sample.left_torus_one_point.real)
            / math.sqrt(tau_left.imag)
        )
        b_right_direct = (
            radius
            * theta_right
            * eta_right**2
            * float(sample.right_torus_one_point.real)
            / math.sqrt(tau_right.imag)
        )
        direct_fixed_block_integral += (
            sample.bridge_measure_weight
            * bridge_primary
            * b_left_direct
            * b_right_direct
        )

    q_abs_squared = abs(q_bridge_complex) ** 2
    propagator_coefficient = ALPHA_PRIME / (4.0 * math.pi * q_abs_squared)
    q_to_epsilon_jacobian = 4.0 * math.pi**2

    # Previously proposed formula: use the integrated torus one-point
    # amplitude J_11=4 pi^2 B_11, with no compact inverse metric.
    integrated_amplitude_integral = (4.0 * math.pi**2) ** 2 * fixed_block_integral
    prior_k_sep_q = propagator_coefficient * integrated_amplitude_integral
    prior_k_sep_epsilon = q_to_epsilon_jacobian * prior_k_sep_q

    # Fixed-puncture CFT sewing in the repository's dimensionless compact
    # state convention.  The compact vacuum metric is <0|0>=r.
    fixed_block_k_sep_q = (
        propagator_coefficient * fixed_block_integral / radius
    )
    fixed_block_k_sep_epsilon = q_to_epsilon_jacobian * fixed_block_k_sep_q

    k_note = full.string_note_kernel_density
    mumford = full.factorization_normalized_density
    prior_ratio = k_note / prior_k_sep_epsilon
    fixed_block_ratio = k_note / fixed_block_k_sep_epsilon
    mumford_stack_ratio = 0.5 * mumford / fixed_block_k_sep_epsilon
    expected_prior_ratio = 1.0 / (8.0 * math.pi**5 * radius)

    return {
        "q_bridge": float(q_bridge),
        "epsilon_abs": float(abs(epsilon)),
        "i1_left": float(i1_left),
        "i1_right": float(i1_right),
        "k2_note_epsilon": float(k_note),
        "mumford_density_epsilon": float(mumford),
        "fixed_block_integral": float(fixed_block_integral),
        "i1_fixed_block_rewrite_ratio": float(
            fixed_block_integral / direct_fixed_block_integral
        ),
        "prior_k_sep_epsilon": float(prior_k_sep_epsilon),
        "prior_ratio": float(prior_ratio),
        "expected_prior_ratio": float(expected_prior_ratio),
        "prior_ratio_over_expected": float(prior_ratio / expected_prior_ratio),
        "fixed_block_k_sep_epsilon": float(fixed_block_k_sep_epsilon),
        "k_note_over_fixed_block_sewing": float(fixed_block_ratio),
        "expected_k_note_over_fixed_block_sewing": float(2.0 / math.pi),
        "stacked_mumford_over_fixed_block_sewing": float(mumford_stack_ratio),
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pointwise genus-two/genus-one separating normalization match."
    )
    parser.add_argument("--q-left", type=complex, default=0.003 + 0.0j)
    parser.add_argument("--q-right", type=complex, default=0.0025 + 0.0j)
    parser.add_argument(
        "--bridge-values",
        default="0.02,0.01,0.005,0.0025,0.00125",
    )
    parser.add_argument("--radius", type=float, default=1.31)
    parser.add_argument("--bridge-p-max", type=float, default=1.5)
    parser.add_argument("--handle-p-max", type=float, default=1.5)
    parser.add_argument("--bridge-quadrature-order", type=int, default=6)
    parser.add_argument("--handle-quadrature-order", type=int, default=8)
    parser.add_argument("--theta-cutoff", type=int, default=16)
    parser.add_argument("--theta-nmax", type=int, default=10)
    parser.add_argument("--dps", type=int, default=28)
    args = parser.parse_args(list(argv) if argv is not None else None)

    bridge_values = tuple(
        sorted(
            (
                float(piece.strip())
                for piece in args.bridge_values.split(",")
                if piece.strip()
            ),
            reverse=True,
        )
    )
    rows = [
        evaluate_point(
            q_left=args.q_left,
            q_right=args.q_right,
            q_bridge=q_bridge,
            radius=args.radius,
            bridge_p_max=args.bridge_p_max,
            handle_p_max=args.handle_p_max,
            bridge_quadrature_order=args.bridge_quadrature_order,
            handle_quadrature_order=args.handle_quadrature_order,
            theta_cutoff=args.theta_cutoff,
            theta_nmax=args.theta_nmax,
            dps=args.dps,
        )
        for q_bridge in bridge_values
    ]

    print("Genus-two pointwise match anchored to the genus-one integrand")
    print(f"  radius r={args.radius:.12g}, alpha'=1")
    print(
        "  columns: q, prior-ratio, prior/predicted, "
        "Knote/fixed-B, (1/2 Mumford)/fixed-B, I1->B"
    )
    for row in rows:
        print(
            f"  {row['q_bridge']:.8g}  "
            f"{row['prior_ratio']:.12e}  "
            f"{row['prior_ratio_over_expected']:.12e}  "
            f"{row['k_note_over_fixed_block_sewing']:.12e}  "
            f"{row['stacked_mumford_over_fixed_block_sewing']:.12e}  "
            f"{row['i1_fixed_block_rewrite_ratio']:.12e}"
        )
    print("  limiting predictions")
    print(f"    prior ratio             = 1/(8 pi^5 r) = {1.0 / (8.0 * math.pi**5 * args.radius):.12e}")
    print(f"    Knote/fixed-B sewing    = 2/pi         = {2.0 / math.pi:.12e}")
    print("    (1/2 Mumford)/fixed-B   = 1")

    final = rows[-1]
    checks = {
        "I1 rewrite": abs(final["i1_fixed_block_rewrite_ratio"] - 1.0) < 2.0e-12,
        "prior limiting constant": abs(final["prior_ratio_over_expected"] - 1.0) < 1.0e-5,
        "fixed-block 2/pi limit": abs(
            final["k_note_over_fixed_block_sewing"] / (2.0 / math.pi) - 1.0
        )
        < 1.0e-5,
        "stacked Mumford factorization": abs(
            final["stacked_mumford_over_fixed_block_sewing"] - 1.0
        )
        < 1.0e-5,
    }
    print(f"  all limiting checks pass={all(checks.values())}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    run()
