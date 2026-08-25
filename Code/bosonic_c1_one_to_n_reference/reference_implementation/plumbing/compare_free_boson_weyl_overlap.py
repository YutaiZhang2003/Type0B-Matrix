#!/usr/bin/env python3
"""Compare direct c=1 free-boson plumbing sums at a theta/glasses overlap."""

from __future__ import annotations

import argparse
import math
from typing import Iterable

try:
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        glasses_free_boson_product,
        glasses_separating_F_asymptotic,
        glasses_separating_raw_oscillator_asymptotic,
        long_tube_normalized_frame_factor,
        noncompact_scalar_zero_mode_factor,
        plumbing_over_bergman_frame_factor,
        theta_free_boson_product,
        theta_maximal_F_asymptotic,
        theta_maximal_raw_oscillator_asymptotic,
    )
    from liouville_genus2 import format_complex, parse_complex
    from plumbing_algorithms import schottky_glasses_period_matrix, schottky_theta_period_matrix_cross_ratio
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from plumbing.free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        glasses_free_boson_product,
        glasses_separating_F_asymptotic,
        glasses_separating_raw_oscillator_asymptotic,
        long_tube_normalized_frame_factor,
        noncompact_scalar_zero_mode_factor,
        plumbing_over_bergman_frame_factor,
        theta_free_boson_product,
        theta_maximal_F_asymptotic,
        theta_maximal_raw_oscillator_asymptotic,
    )
    from plumbing.liouville_genus2 import format_complex, parse_complex
    from plumbing.plumbing_algorithms import schottky_glasses_period_matrix, schottky_theta_period_matrix_cross_ratio


DEFAULT_GLASSES_Q = (0.15 + 0.0j, 0.15 + 0.0j, 0.15 + 0.0j)
DEFAULT_THETA_Q = (
    0.1602332247082 - 2.777592429258e-7j,
    2.998106672314e-9 + 1.261814452401e-14j,
    4.251284599701e-9 - 1.550732812259e-14j,
)


def _fmt_q(values: tuple[complex, complex, complex]) -> str:
    return "(" + ", ".join(format_complex(value) for value in values) + ")"


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute direct free-boson plumbing/Bergman factors at a period-matched overlap."
    )
    parser.add_argument("--glasses-q", type=parse_complex, nargs=3, default=DEFAULT_GLASSES_Q)
    parser.add_argument("--theta-q", type=parse_complex, nargs=3, default=DEFAULT_THETA_Q)
    parser.add_argument("--max-word-length", type=int, default=10)
    parser.add_argument("--max-mode", type=int, default=80)
    parser.add_argument("--direct-level", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1.0e-14)
    parser.add_argument("--period-word-length", type=int, default=8)
    parser.add_argument("--period-b-order", type=int, default=600)
    parser.add_argument("--theta-nmax", type=int, default=8)
    parser.add_argument("--theta-tol", type=float, default=1.0e-12)
    parser.add_argument(
        "--common-omega-from",
        choices=("glasses", "theta"),
        default="glasses",
        help="which period matrix to use as the single canonical/Bergman comparison point",
    )
    parser.add_argument(
        "--determinant-exponent",
        type=float,
        default=-0.5,
        help="power of det' Delta_B; -1/2 is one real scalar and cancels in the same-Omega ratio",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    glasses_q = tuple(complex(value) for value in args.glasses_q)
    theta_q = tuple(complex(value) for value in args.theta_q)

    glasses_schottky = glasses_free_boson_product(
        *glasses_q,
        max_word_length=args.max_word_length,
        max_mode=args.max_mode,
        tolerance=args.tolerance,
    )
    theta_schottky = theta_free_boson_product(
        *theta_q,
        max_word_length=args.max_word_length,
        max_mode=args.max_mode,
        tolerance=args.tolerance,
    )

    omega_glasses = schottky_glasses_period_matrix(
        *glasses_q,
        max_word_len=args.period_word_length,
        b_order=args.period_b_order,
    )
    omega_theta = schottky_theta_period_matrix_cross_ratio(*theta_q, max_word_len=args.period_word_length)
    bergman_period_glasses_direct = bergman_scalar_partition_candidate(
        omega_glasses,
        determinant_exponent=args.determinant_exponent,
        theta_nmax=args.theta_nmax,
        theta_tol=args.theta_tol,
    )
    bergman_period_theta_direct = bergman_scalar_partition_candidate(
        omega_theta,
        determinant_exponent=args.determinant_exponent,
        theta_nmax=args.theta_nmax,
        theta_tol=args.theta_tol,
    )
    bergman_common = (
        bergman_period_glasses_direct if args.common_omega_from == "glasses" else bergman_period_theta_direct
    )

    glasses_direct = glasses_heisenberg_plumbing_partition(
        *glasses_q,
        max_total_level=args.direct_level,
    )
    theta_direct = theta_heisenberg_plumbing_partition(
        *theta_q,
        max_total_level=args.direct_level,
    )
    direct_ratio = theta_direct.nonchiral_value / glasses_direct.nonchiral_value
    glasses_zero_mode = noncompact_scalar_zero_mode_factor(omega_glasses)
    theta_zero_mode = noncompact_scalar_zero_mode_factor(omega_theta)
    zero_mode_ratio = theta_zero_mode / glasses_zero_mode
    full_scalar_glasses = glasses_direct.nonchiral_value * glasses_zero_mode
    full_scalar_theta = theta_direct.nonchiral_value * theta_zero_mode
    full_scalar_ratio = full_scalar_theta / full_scalar_glasses
    schottky_ratio = theta_schottky.nonchiral_value / glasses_schottky.nonchiral_value
    bergman_period_ratio = (
        bergman_period_theta_direct.partition_candidate / bergman_period_glasses_direct.partition_candidate
    )
    common_bergman_ratio = 1.0
    glasses_frame = plumbing_over_bergman_frame_factor(
        full_scalar_glasses,
        bergman_common.petersson_norm_delta2,
        determinant_exponent=args.determinant_exponent,
    )
    theta_frame = plumbing_over_bergman_frame_factor(
        full_scalar_theta,
        bergman_common.petersson_norm_delta2,
        determinant_exponent=args.determinant_exponent,
    )
    frame_ratio = theta_frame / glasses_frame
    glasses_raw_asymptotic = glasses_separating_raw_oscillator_asymptotic(glasses_q[0], glasses_q[1])
    glasses_F_asymptotic = glasses_separating_F_asymptotic(*glasses_q)
    glasses_long_tube_normalized = long_tube_normalized_frame_factor(
        glasses_schottky.nonchiral_value,
        bergman_common.petersson_norm_delta2,
        glasses_raw_asymptotic,
        glasses_F_asymptotic,
        determinant_exponent=args.determinant_exponent,
    )
    theta_raw_asymptotic = theta_maximal_raw_oscillator_asymptotic(*theta_q)
    theta_F_asymptotic = theta_maximal_F_asymptotic(*theta_q)
    theta_long_tube_normalized = long_tube_normalized_frame_factor(
        theta_schottky.nonchiral_value,
        bergman_common.petersson_norm_delta2,
        theta_raw_asymptotic,
        theta_F_asymptotic,
        determinant_exponent=args.determinant_exponent,
    )

    print("Free-boson c=1 direct plumbing/Bergman comparison")
    print(f"  glasses q = {_fmt_q(glasses_q)}")
    print(f"  theta q   = {_fmt_q(theta_q)}")
    print()
    print("Direct Heisenberg pair-of-pants sums")
    print(f"  total sewing level = {args.direct_level}")
    print(f"  glasses = {glasses_direct.nonchiral_value:.16e}")
    print(f"  theta   = {theta_direct.nonchiral_value:.16e}")
    print(f"  oscillator theta/glasses = {direct_ratio:.16e}")
    print(f"  glasses zero mode = {glasses_zero_mode:.16e}")
    print(f"  theta zero mode   = {theta_zero_mode:.16e}")
    print(f"  zero-mode theta/glasses = {zero_mode_ratio:.16e}")
    print(f"  full scalar theta/glasses = {full_scalar_ratio:.16e}")
    print()
    print("Independent plumbing-frame primitive-word oscillator product (m >= 1)")
    print(f"  glasses = {glasses_schottky.nonchiral_value:.16e}")
    print(f"  theta   = {theta_schottky.nonchiral_value:.16e}")
    print(f"  theta/glasses = {schottky_ratio:.16e}")
    print(
        "  direct/oscillator-product relative ratio error = "
        f"{abs(direct_ratio - schottky_ratio) / abs(schottky_ratio):.6e}"
    )
    print()
    print("Bergman determinant candidates")
    print(f"  determinant exponent = {args.determinant_exponent:+.6g}")
    print(f"  common Omega from = {args.common_omega_from}")
    print(f"  common Z_Bergman = {bergman_common.partition_candidate:.16e}")
    print(f"  common F = {bergman_common.petersson_norm_delta2:.16e}")
    print(f"  separate F_glasses = {bergman_period_glasses_direct.petersson_norm_delta2:.16e}")
    print(f"  separate F_theta   = {bergman_period_theta_direct.petersson_norm_delta2:.16e}")
    print(
        "  separate relative F mismatch = "
        f"{abs(bergman_period_theta_direct.petersson_norm_delta2 - bergman_period_glasses_direct.petersson_norm_delta2) / max(abs(bergman_period_glasses_direct.petersson_norm_delta2), 1.0e-300):.6e}"
    )
    print(f"  common Bergman theta/glasses = {common_bergman_ratio:.16e}")
    print(f"  separate Bergman theta/glasses diagnostic = {bergman_period_ratio:.16e}")
    print()
    print("Full-scalar plumbing/Bergman quotients (common factors cancel only in the ratio)")
    print(f"  glasses, c=1 = {glasses_frame:.16e}")
    print(f"  theta, c=1   = {theta_frame:.16e}")
    print(f"  theta/glasses, c=1 = {frame_ratio:.16e}")
    print(f"  theta/glasses, c=25 = {math.exp(25.0 * math.log(frame_ratio)):.16e}")
    print("  (the common Bergman determinant cancels in these two relative numbers)")
    print()
    print("Long-tube oscillator finite-part diagnostic (not the relative Weyl factor)")
    print(f"  glasses raw asymptotic = {glasses_raw_asymptotic:.16e}")
    print(f"  glasses F asymptotic   = {glasses_F_asymptotic:.16e}")
    print(f"  glasses normalized     = {glasses_long_tube_normalized:.16e}")
    print(f"  theta raw asymptotic   = {theta_raw_asymptotic:.16e}")
    print(f"  theta F asymptotic     = {theta_F_asymptotic:.16e}")
    print(f"  theta normalized       = {theta_long_tube_normalized:.16e}")
    print(f"  normalized theta/glasses, c=1 = {theta_long_tube_normalized / glasses_long_tube_normalized:.16e}")


if __name__ == "__main__":
    run()
