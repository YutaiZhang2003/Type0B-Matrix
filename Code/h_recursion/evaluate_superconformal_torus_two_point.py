#!/usr/bin/env python3
"""Print a Type-0B NS torus two-point necklace block."""

from __future__ import annotations

import argparse

from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import TorusTwoPointSpinStructure
from superconformal_torus_two_point import NSTorusTwoPointLeadingBlock
from superconformal_torus_two_point import (
    SelfDualNSTorusTwoPointHRecursionBlock,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the NS torus two-point necklace block."
    )
    parser.add_argument(
        "--block-method",
        choices=("h-recursion", "leading"),
        default="h-recursion",
        help="conformal-block evaluator (default: h-recursion)",
    )
    parser.add_argument("--internal-momentum-1", type=float, default=0.61)
    parser.add_argument("--internal-momentum-2", type=float, default=0.74)
    parser.add_argument("--external-momentum-1", type=float, default=0.33)
    parser.add_argument("--external-momentum-2", type=float, default=0.41)
    parser.add_argument("--q1", type=complex, default=0.08 + 0.0j)
    parser.add_argument("--q2", type=complex, default=0.05 + 0.0j)
    parser.add_argument("--lift1", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--lift2", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--spin-structure",
        choices=("NS", "NS_tilde"),
        help=(
            "physical torus spin structure; when supplied, lift2 is "
            "chosen relative to lift1"
        ),
    )
    parser.add_argument("--max-twice-level-1", type=int, default=4)
    parser.add_argument("--max-twice-level-2", type=int, default=4)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument(
        "--finite-part-check-radius", type=float, default=0.05
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    parser.add_argument("--difference-radius", type=float, default=0.03)
    parser.add_argument("--difference-samples", type=int, default=16)
    args = parser.parse_args()

    if args.block_method == "h-recursion":
        block = SelfDualNSTorusTwoPointHRecursionBlock(
            internal_momentum_1=args.internal_momentum_1,
            internal_momentum_2=args.internal_momentum_2,
            external_momentum_1=args.external_momentum_1,
            external_momentum_2=args.external_momentum_2,
            radius=args.finite_part_radius,
            check_radius=args.finite_part_check_radius,
            samples=args.finite_part_samples,
            difference_radius=args.difference_radius,
            difference_samples=args.difference_samples,
        )
        coefficients = block.raw_coefficients(
            args.max_twice_level_1, args.max_twice_level_2
        )
    else:
        c = central_charge(1.0)
        block = NSTorusTwoPointLeadingBlock(
            central_charge=c,
            internal_weight_1=ns_liouville_weight(
                args.internal_momentum_1, 1.0
            ),
            internal_weight_2=ns_liouville_weight(
                args.internal_momentum_2, 1.0
            ),
            external_weight_1=ns_liouville_weight(
                args.external_momentum_1, 1.0
            ),
            external_weight_2=ns_liouville_weight(
                args.external_momentum_2, 1.0
            ),
        )
        coefficients = block.raw_coefficients()
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
    plumbing_1, plumbing_2 = spin.plumbing_parameters(args.q1, args.q2)

    print("Type-0B NS torus two-point necklace block")
    print(f"  torus nome q=q1*q2={_format(args.q1 * args.q2)}")
    print(f"  spin structure={spin.spin_label}")
    print(f"  local NS lifts=({args.lift1:+d},{lift_2:+d})")
    print(f"  block method={args.block_method}")
    print("  coefficients keyed by (2 n1,2 n2):")
    for levels, coefficient in sorted(coefficients.items()):
        print(f"    {levels}: {_format(coefficient)}")
    if args.block_method == "h-recursion":
        descendant_series = block.evaluate(
            plumbing_1,
            plumbing_2,
            args.max_twice_level_1,
            args.max_twice_level_2,
        )
        chiral_block = block.chiral_block(
            plumbing_1,
            plumbing_2,
            args.max_twice_level_1,
            args.max_twice_level_2,
        )
        diagnostics = block.coefficient_diagnostics(
            args.max_twice_level_1, args.max_twice_level_2
        )
        print(
            "  maximum two-radius finite-part discrepancy="
            f"{max(item.absolute_error for item in diagnostics.values()):.3g}"
        )
    else:
        descendant_series = block.evaluate(plumbing_1, plumbing_2)
        chiral_block = block.chiral_block(plumbing_1, plumbing_2)
    print(f"  descendant series={_format(descendant_series)}")
    print(f"  chiral block={_format(chiral_block)}")


if __name__ == "__main__":
    main()
