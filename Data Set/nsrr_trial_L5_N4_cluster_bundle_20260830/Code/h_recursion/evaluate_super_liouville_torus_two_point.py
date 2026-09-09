#!/usr/bin/env python3
"""Evaluate the Type-0B NS torus two-point spectral integral."""

from __future__ import annotations

import argparse

from super_liouville_torus_two_point import (
    Type0BNSTorusTwoPointHRecursionCorrelator,
    Type0BNSTorusTwoPointLeadingCorrelator,
    Type0BRamondTorusTwoPointBetaRecursionCorrelator,
)
from superconformal_torus_blocks import (
    TorusTwoPointSpinStructure,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the NS torus two-point correlator."
    )
    parser.add_argument(
        "--internal-sector",
        choices=("NS", "R"),
        default="NS",
        help="sector carried by both necklace edges",
    )
    parser.add_argument(
        "--block-method",
        choices=("auto", "h-recursion", "beta-recursion", "leading"),
        default="auto",
        help="conformal-block evaluator appropriate to the internal sector",
    )
    parser.add_argument("--external-momentum-1", type=float, default=0.33)
    parser.add_argument("--external-momentum-2", type=float, default=0.41)
    parser.add_argument("--q1", type=complex, default=0.08 + 0.0j)
    parser.add_argument("--q2", type=complex, default=0.05 + 0.0j)
    parser.add_argument("--lift1", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--lift2", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--spin-structure",
        choices=("NS", "NS_tilde", "R", "R_tilde"),
        help="physical torus spin structure appropriate to the edge sector",
    )
    parser.add_argument("--p-max", type=float, default=3.5)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--structure-precision", type=int, default=25)
    parser.add_argument("--max-twice-level-1", type=int, default=4)
    parser.add_argument("--max-twice-level-2", type=int, default=4)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument(
        "--finite-part-check-radius", type=float, default=0.05
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    parser.add_argument("--difference-radius", type=float, default=0.03)
    parser.add_argument("--difference-samples", type=int, default=16)
    parser.add_argument("--max-level-1", type=int, default=1)
    parser.add_argument("--max-level-2", type=int, default=1)
    parser.add_argument(
        "--cycle1",
        choices=("identity", "parity"),
        default="identity",
    )
    parser.add_argument(
        "--cycle2",
        choices=("identity", "parity"),
        default="identity",
    )
    args = parser.parse_args()

    if args.internal_sector == "NS":
        if args.spin_structure not in (None, "NS", "NS_tilde"):
            parser.error("NS edges require spin structure NS or NS_tilde")
        lift_2 = args.lift2
        if args.spin_structure is not None:
            temporal_sign = 1 if args.spin_structure == "NS" else -1
            lift_2 = temporal_sign * args.lift1
        spin = TorusTwoPointSpinStructure(
            "NS",
            "NS",
            ns_lift_sign_1=args.lift1,
            ns_lift_sign_2=lift_2,
        )
    else:
        if args.spin_structure not in (None, "R", "R_tilde"):
            parser.error("R edges require spin structure R or R_tilde")
        cycle_2 = args.cycle2
        if args.spin_structure == "R":
            cycle_2 = args.cycle1
        elif args.spin_structure == "R_tilde":
            cycle_2 = (
                "parity" if args.cycle1 == "identity" else "identity"
            )
        spin = TorusTwoPointSpinStructure(
            "R",
            "R",
            r_cycle_insertion_1=args.cycle1,
            r_cycle_insertion_2=cycle_2,
        )

    method = args.block_method
    if method == "auto":
        method = "h-recursion" if args.internal_sector == "NS" else (
            "beta-recursion"
        )
    if args.internal_sector == "NS" and method == "beta-recursion":
        parser.error("--block-method beta-recursion requires --internal-sector R")
    if args.internal_sector == "R" and method != "beta-recursion":
        parser.error(
            "--internal-sector R currently requires "
            "--block-method beta-recursion"
        )

    common = dict(
        external_momentum_1=args.external_momentum_1,
        external_momentum_2=args.external_momentum_2,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        structure_precision=args.structure_precision,
    )
    if method == "h-recursion":
        correlator = Type0BNSTorusTwoPointHRecursionCorrelator(
            **common,
            max_twice_level_1=args.max_twice_level_1,
            max_twice_level_2=args.max_twice_level_2,
            finite_part_radius=args.finite_part_radius,
            finite_part_check_radius=args.finite_part_check_radius,
            finite_part_samples=args.finite_part_samples,
            difference_radius=args.difference_radius,
            difference_samples=args.difference_samples,
        )
        plumbing_1, plumbing_2 = spin.plumbing_parameters(
            args.q1, args.q2
        )
    elif method == "leading":
        correlator = Type0BNSTorusTwoPointLeadingCorrelator(**common)
        plumbing_1, plumbing_2 = spin.plumbing_parameters(
            args.q1, args.q2
        )
    else:
        correlator = Type0BRamondTorusTwoPointBetaRecursionCorrelator(
            **common,
            max_level_1=args.max_level_1,
            max_level_2=args.max_level_2,
            cycle_insertion_1=args.cycle1,
            cycle_insertion_2=spin.r_cycle_insertion_2,
            finite_part_radius=args.finite_part_radius,
            finite_part_check_radius=args.finite_part_check_radius,
            finite_part_samples=args.finite_part_samples,
        )
        plumbing_1, plumbing_2 = spin.plumbing_parameters(
            args.q1, args.q2
        )
    value = correlator.evaluate(
        plumbing_1,
        plumbing_2,
    )
    print("Type-0B torus two-point necklace contribution")
    print(f"  torus nome q=q1*q2={_format(args.q1 * args.q2)}")
    print(f"  P cutoff={args.p_max:.12g}")
    print(f"  quadrature order={args.quadrature_order}")
    print(f"  internal sector={args.internal_sector}")
    print(f"  spin structure={spin.spin_label}")
    print(f"  block method={method}")
    if method == "h-recursion":
        print(
            "  twice-level cutoffs="
            f"({args.max_twice_level_1},{args.max_twice_level_2})"
        )
    elif method == "beta-recursion":
        print(
            f"  level cutoffs=({args.max_level_1},{args.max_level_2})"
        )
        print(
            "  cycle insertions="
            f"({spin.r_cycle_insertion_1},"
            f"{spin.r_cycle_insertion_2})"
        )
    print(f"  value={_format(value)}")
    if method == "h-recursion":
        print(
            "  status=truncated two-edge h-recursion with coefficient-wise "
            "b=1 finite parts"
        )
    elif method == "beta-recursion":
        print(
            "  status=sign-summed R beta recursion with a level-one "
            "direct regular seed"
        )
    else:
        print("  status=direct leading necklace block")


if __name__ == "__main__":
    main()
