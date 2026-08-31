#!/usr/bin/env python3
"""Print a Ramond-handle torus two-point necklace block."""

from __future__ import annotations

import argparse

from ramond_sphere_blocks import ramond_liouville_weight
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import TorusTwoPointSpinStructure
from superconformal_torus_two_point import RamondTorusTwoPointGroundBlock
from superconformal_torus_two_point import (
    SelfDualRamondTorusTwoPointBetaRecursionBlock,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the R-sewn torus two-point block."
        )
    )
    parser.add_argument(
        "--block-method",
        choices=("beta-recursion", "ground"),
        default="beta-recursion",
    )
    parser.add_argument("--internal-momentum-1", type=float, default=0.61)
    parser.add_argument("--internal-momentum-2", type=float, default=0.74)
    parser.add_argument("--external-momentum-1", type=float, default=0.33)
    parser.add_argument("--external-momentum-2", type=float, default=0.41)
    parser.add_argument("--q1", type=complex, default=0.08 + 0.0j)
    parser.add_argument("--q2", type=complex, default=0.05 + 0.0j)
    parser.add_argument("--sign1", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--sign2", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--cycle1",
        choices=("identity", "parity", "g0", "parity_g0"),
        default="identity",
    )
    parser.add_argument(
        "--cycle2",
        choices=("identity", "parity", "g0", "parity_g0"),
        default="identity",
    )
    parser.add_argument(
        "--spin-structure",
        choices=("R", "R_tilde"),
        help=(
            "physical torus spin structure; requires identity/parity "
            "cycle insertions and chooses cycle2 relative to cycle1"
        ),
    )
    parser.add_argument("--max-level-1", type=int, default=1)
    parser.add_argument("--max-level-2", type=int, default=1)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument(
        "--finite-part-check-radius", type=float, default=0.05
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    args = parser.parse_args()

    c = central_charge(1.0)
    cycle_2 = args.cycle2
    if args.spin_structure is not None:
        if args.cycle1 not in ("identity", "parity"):
            parser.error(
                "--spin-structure cannot be combined with a G0 cycle "
                "insertion"
            )
        if args.spin_structure == "R":
            cycle_2 = args.cycle1
        else:
            cycle_2 = (
                "parity" if args.cycle1 == "identity" else "identity"
            )
    if args.block_method == "beta-recursion" and (
        args.cycle1 not in ("identity", "parity")
        or cycle_2 not in ("identity", "parity")
    ):
        parser.error(
            "positive-level beta recursion supports only identity/parity "
            "cycle insertions"
        )
    spin = TorusTwoPointSpinStructure(
        "R",
        "R",
        r_cycle_insertion_1=args.cycle1,
        r_cycle_insertion_2=cycle_2,
    )
    plumbing_1, plumbing_2 = spin.plumbing_parameters(args.q1, args.q2)

    print("Type-0B R-handle torus two-point necklace block")
    print(f"  torus nome q=q1*q2={_format(args.q1 * args.q2)}")
    print(f"  spin structure={spin.spin_label}")
    print(f"  HJS vertex signs=({args.sign1:+d},{args.sign2:+d})")
    print(f"  cycle insertions=({args.cycle1},{cycle_2})")
    print(f"  block method={args.block_method}")
    if args.block_method == "beta-recursion":
        block = SelfDualRamondTorusTwoPointBetaRecursionBlock(
            internal_momentum_1=args.internal_momentum_1,
            internal_momentum_2=args.internal_momentum_2,
            external_momentum_1=args.external_momentum_1,
            external_momentum_2=args.external_momentum_2,
            vertex_sign_1=args.sign1,
            vertex_sign_2=args.sign2,
            cycle_insertion_1=args.cycle1,
            cycle_insertion_2=cycle_2,
            radius=args.finite_part_radius,
            check_radius=args.finite_part_check_radius,
            samples=args.finite_part_samples,
        )
        coefficients = block.raw_coefficients(
            args.max_level_1, args.max_level_2
        )
        print("  coefficients keyed by (level1,level2):")
        for levels, coefficient in sorted(coefficients.items()):
            print(f"    {levels}: {_format(coefficient)}")
        diagnostics = block.coefficient_diagnostics(
            args.max_level_1, args.max_level_2
        )
        print(
            "  maximum two-radius finite-part discrepancy="
            f"{max(item.absolute_error for item in diagnostics.values()):.3g}"
        )
        normalized_chiral = block.normalized_chiral_block(
            plumbing_1,
            plumbing_2,
            args.max_level_1,
            args.max_level_2,
        )
        projected_chiral = block.chiral_block(
            plumbing_1,
            plumbing_2,
            args.max_level_1,
            args.max_level_2,
        )
        print(
            "  normalized chiral block="
            f"{_format(normalized_chiral)}"
        )
        print(
            "  cycle-projected chiral block="
            f"{_format(projected_chiral)}"
        )
        print(
            "  status=R beta-pole recursion with a direct regular seed "
            "validated through level one"
        )
    else:
        block = RamondTorusTwoPointGroundBlock(
            central_charge=c,
            internal_weight_1=ramond_liouville_weight(
                args.internal_momentum_1, 1.0
            ),
            internal_weight_2=ramond_liouville_weight(
                args.internal_momentum_2, 1.0
            ),
            external_weight_1=ns_liouville_weight(
                args.external_momentum_1, 1.0
            ),
            external_weight_2=ns_liouville_weight(
                args.external_momentum_2, 1.0
            ),
            vertex_sign_1=args.sign1,
            vertex_sign_2=args.sign2,
        )
        print(
            "  kappa^2="
            f"({_format(block.kappa_squared_1)},"
            f"{_format(block.kappa_squared_2)})"
        )
        print(
            "  ground coefficient="
            f"{_format(block.ground_coefficient(plumbing_1, plumbing_2))}"
        )
        print(
            "  chiral ground block="
            f"{_format(block.chiral_block(plumbing_1, plumbing_2))}"
        )


if __name__ == "__main__":
    main()
