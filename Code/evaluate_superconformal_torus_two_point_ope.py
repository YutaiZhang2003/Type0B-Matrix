#!/usr/bin/env python3
"""Evaluate a torus two-point block in the sphere--torus OPE channel."""

from __future__ import annotations

import argparse

from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)
from superconformal_torus_two_point_ope import (
    SelfDualNSTorusTwoPointOPEBlock,
    SelfDualRamondHandleTorusTwoPointOPEBlock,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Type-0B sphere-three-point--torus-one-point "
            "two-point block."
        )
    )
    parser.add_argument(
        "--handle-sector", choices=("NS", "R"), default="NS"
    )
    parser.add_argument("--bridge-momentum", type=float, default=0.61)
    parser.add_argument("--handle-momentum", type=float, default=0.74)
    parser.add_argument("--external-momentum-1", type=float, default=0.33)
    parser.add_argument("--external-momentum-2", type=float, default=0.41)
    parser.add_argument(
        "--z",
        type=complex,
        default=0.8 + 0.0j,
        help="annulus-plane position; the collision coordinate is x=1-z",
    )
    parser.add_argument("--q", type=complex, default=0.004 + 0.0j)
    parser.add_argument("--lift", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--max-bridge-twice-level",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help=(
            "the verified direct bridge sewing currently permits 0, 1, or 2"
        ),
    )
    parser.add_argument("--max-handle-twice-level", type=int, default=8)
    parser.add_argument("--max-handle-level", type=int, default=4)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument(
        "--finite-part-check-radius", type=float, default=0.05
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    args = parser.parse_args()

    x = 1.0 - args.z
    common = dict(
        bridge_momentum=args.bridge_momentum,
        handle_momentum=args.handle_momentum,
        external_momentum_1=args.external_momentum_1,
        external_momentum_2=args.external_momentum_2,
        radius=args.finite_part_radius,
        check_radius=args.finite_part_check_radius,
        samples=args.finite_part_samples,
    )
    if args.handle_sector == "NS":
        block = SelfDualNSTorusTwoPointOPEBlock(**common)
        plumbing = NSPlumbingParameter(args.q, args.lift)
        coefficients = block.raw_coefficients(
            args.max_bridge_twice_level,
            args.max_handle_twice_level,
        )
        value = block.chiral_block(
            x,
            plumbing,
            args.max_bridge_twice_level,
            args.max_handle_twice_level,
        )
    else:
        block = SelfDualRamondHandleTorusTwoPointOPEBlock(
            **common, sign=1
        )
        plumbing = RamondPlumbingParameter(args.q, "identity")
        coefficients = block.normalized_raw_coefficients(
            args.max_bridge_twice_level,
            args.max_handle_level,
        )
        value = block.normalized_chiral_block(
            x,
            plumbing,
            args.max_bridge_twice_level,
            args.max_handle_level,
        )

    print("Type-0B torus two-point sphere--torus OPE block")
    print(f"  z={_format(args.z)}")
    print(f"  collision coordinate x=1-z={_format(x)}")
    print(f"  torus nome q={_format(args.q)}")
    print(f"  handle sector={args.handle_sector}")
    print("  coefficients keyed by (2 n_bridge, n_handle units):")
    for levels, coefficient in sorted(coefficients.items()):
        print(f"    {levels}: {_format(coefficient)}")
    print(f"  normalized chiral block={_format(value)}")
    if args.handle_sector == "NS":
        print(
            "  status=exact handle h-recursion with direct bridge "
            "Ward/Gram sewing through level one"
        )
    else:
        print(
            "  status=exact R-handle recursion with direct bridge "
            "Ward/Gram sewing through level one"
        )


if __name__ == "__main__":
    main()
