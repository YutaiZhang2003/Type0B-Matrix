#!/usr/bin/env python3
"""Evaluate the pilot sphere--torus OPE-channel spectral integral."""

from __future__ import annotations

import argparse

from super_liouville_torus_two_point_ope import (
    Type0BNSHandleTorusTwoPointOPECorrelator,
    Type0BRamondHandleTorusTwoPointOPECorrelator,
)
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the even-form sphere-three-point--torus-one-point "
            "Type-0B contribution."
        )
    )
    parser.add_argument(
        "--handle-sector", choices=("NS", "R"), default="NS"
    )
    parser.add_argument("--external-momentum-1", type=float, default=0.33)
    parser.add_argument("--external-momentum-2", type=float, default=0.41)
    parser.add_argument("--z", type=complex, default=0.8 + 0.0j)
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
    parser.add_argument("--p-max", type=float, default=3.5)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--structure-precision", type=int, default=25)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument(
        "--finite-part-check-radius", type=float, default=0.05
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    args = parser.parse_args()

    common = dict(
        external_momentum_1=args.external_momentum_1,
        external_momentum_2=args.external_momentum_2,
        max_bridge_twice_level=args.max_bridge_twice_level,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        structure_precision=args.structure_precision,
        finite_part_radius=args.finite_part_radius,
        finite_part_check_radius=args.finite_part_check_radius,
        finite_part_samples=args.finite_part_samples,
    )
    x = 1.0 - args.z
    if args.handle_sector == "NS":
        correlator = Type0BNSHandleTorusTwoPointOPECorrelator(
            **common,
            max_handle_twice_level=args.max_handle_twice_level,
        )
        plumbing = NSPlumbingParameter(args.q, args.lift)
    else:
        correlator = Type0BRamondHandleTorusTwoPointOPECorrelator(
            **common,
            max_handle_level=args.max_handle_level,
        )
        plumbing = RamondPlumbingParameter(args.q, "identity")

    value = correlator.evaluate(x, plumbing)
    print("Type-0B sphere--torus OPE-channel contribution")
    print(f"  z={_format(args.z)}")
    print(f"  collision coordinate x=1-z={_format(x)}")
    print(f"  torus nome q={_format(args.q)}")
    print(f"  handle sector={args.handle_sector}")
    print(f"  P cutoff={args.p_max:.12g}")
    print(f"  quadrature order={args.quadrature_order}")
    print(f"  value={_format(value)}")
    if args.handle_sector == "NS":
        print(
            "  status=even NS three-form with exact handle h-recursion "
            "and direct bridge sewing through level one"
        )
    else:
        print(
            "  status=even NS three-form; the R-handle bridge oracle is "
            "exact through level one"
        )


if __name__ == "__main__":
    main()
