"""Command-line evaluation of the BRY genus-zero four-point correlators."""

from __future__ import annotations

import argparse
from typing import Sequence

from sphere_four_point import BRYFourTachyonSphere, BRYNSFourPointCorrelator


DEFAULT_Z = (0.05 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.1j)


def _complex_value(text: str) -> complex:
    try:
        return complex(text.replace("i", "j"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid complex number: {text}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BRY's b=1 NS Liouville sphere correlators G, H, and J."
    )
    parser.add_argument("--p1", type=float, default=0.5)
    parser.add_argument("--p2", type=float, default=1.0 / 3.0)
    parser.add_argument("--p3", type=float, default=0.25)
    parser.add_argument("--p4", type=float, default=0.6)
    parser.add_argument(
        "--z",
        type=_complex_value,
        action="append",
        help="cross ratio; repeat for a grid (examples: 0.1, 0.3+0.1i)",
    )
    parser.add_argument("--block-order", type=int, default=8)
    parser.add_argument("--p-max", type=float, default=5.0)
    parser.add_argument("--quadrature-order", type=int, default=20)
    parser.add_argument(
        "--four-tachyon",
        action="store_true",
        help="also print the reduced BRY four-tachyon moduli integrand",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    correlator = BRYNSFourPointCorrelator(
        p1=args.p1,
        p2=args.p2,
        p3=args.p3,
        p4=args.p4,
        block_order=args.block_order,
    )
    sphere = None
    if args.four_tachyon:
        sphere = BRYFourTachyonSphere(
            omega=args.p4,
            omega1=args.p1,
            omega2=args.p2,
            omega3=args.p3,
            block_order=args.block_order,
        )

    z_values = tuple(args.z) if args.z else DEFAULT_Z
    columns = ["z", "G(z)", "H(z)", "J(z)"]
    if sphere is not None:
        columns.append("I_T(z)")
    print("  ".join(f"{column:>25}" for column in columns))
    for z in z_values:
        values = correlator.evaluate(
            z, p_max=args.p_max, quadrature_order=args.quadrature_order
        )
        row = [z, values.G, values.H, values.J]
        if sphere is not None:
            row.append(sphere.combine_correlators(z, values))
        print("  ".join(f"{value:>25.15g}" for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
