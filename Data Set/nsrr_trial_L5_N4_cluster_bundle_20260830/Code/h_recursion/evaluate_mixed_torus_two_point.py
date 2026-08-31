#!/usr/bin/env python3
"""Evaluate the ground-fiber normalization of an NS--R torus necklace."""

from __future__ import annotations

import argparse

from ramond_sphere_blocks import ramond_liouville_weight
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import TorusTwoPointSpinStructure
from superconformal_torus_two_point import (
    MixedNSRamondTorusTwoPointGroundBlock,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the recursion-normalization layer of the mixed NS--R "
            "necklace with two external Ramond punctures."
        )
    )
    parser.add_argument("--internal-ns-momentum", type=float, default=0.61)
    parser.add_argument("--internal-r-momentum", type=float, default=0.74)
    parser.add_argument("--q-ns", type=complex, default=0.08 + 0.0j)
    parser.add_argument("--q-r", type=complex, default=0.05 + 0.0j)
    parser.add_argument("--ns-lift", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--r-cycle",
        choices=("identity", "parity", "g0", "parity_g0"),
        default="identity",
    )
    parser.add_argument("--sign1", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--sign2", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--form1", choices=("even", "odd"), default="even"
    )
    parser.add_argument(
        "--form2", choices=("even", "odd"), default="even"
    )
    parser.add_argument(
        "--external-ground1", choices=("+", "-"), default="+"
    )
    parser.add_argument(
        "--external-ground2", choices=("+", "-"), default="+"
    )
    args = parser.parse_args()

    spin = TorusTwoPointSpinStructure(
        "NS",
        "R",
        ns_lift_sign_1=args.ns_lift,
        r_cycle_insertion_2=args.r_cycle,
    )
    ns_plumbing, r_plumbing = spin.plumbing_parameters(
        args.q_ns, args.q_r
    )
    c = central_charge(1.0)
    block = MixedNSRamondTorusTwoPointGroundBlock(
        central_charge=c,
        internal_ns_weight=ns_liouville_weight(
            args.internal_ns_momentum, 1.0
        ),
        internal_r_weight=ramond_liouville_weight(
            args.internal_r_momentum, 1.0
        ),
        vertex_sign_1=args.sign1,
        vertex_sign_2=args.sign2,
        form_parity_1=args.form1,
        form_parity_2=args.form2,
        external_ground_1=args.external_ground1,
        external_ground_2=args.external_ground2,
    )

    print("Type-0B mixed NS--R torus two-point necklace")
    print("  external puncture sectors=(R,R)")
    print(f"  mixed spin bits={spin.mixed_spin_bits}")
    print(f"  NS lift={args.ns_lift:+d}")
    print(f"  R-cycle insertion={args.r_cycle}")
    print(
        "  RRNS forms="
        f"({args.form1},{args.form2}), signs=({args.sign1:+d},{args.sign2:+d})"
    )
    print(
        "  external chiral grounds="
        f"({args.external_ground1},{args.external_ground2})"
    )
    print(
        "  ground coefficient="
        f"{_format(block.ground_coefficient(ns_plumbing, r_plumbing))}"
    )
    print(
        "  chiral ground block="
        f"{_format(block.chiral_block(ns_plumbing, r_plumbing))}"
    )
    print(
        "  status=ground-fiber normalization for the analytic mixed "
        "h-recursion; no descendant truncation is used"
    )


if __name__ == "__main__":
    main()
