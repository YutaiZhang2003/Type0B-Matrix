"""Evaluate the first Type-0B NS/R torus one-point block layer."""

from __future__ import annotations

import argparse

from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
    SelfDualNSTorusOnePointBlock,
    SelfDualRamondTorusOnePointBlock,
)


def _complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def _format(value: complex) -> str:
    return f"{value.real:+.14e}{value.imag:+.14e}i"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sector", choices=("NS", "R"), required=True)
    parser.add_argument("--internal-momentum", type=_complex, required=True)
    parser.add_argument("--external-momentum", type=_complex, required=True)
    parser.add_argument("--q", type=_complex, default=0.05)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--lift-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--r-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--cycle-insertion",
        choices=("identity", "parity", "g0", "parity_g0"),
        default="identity",
    )
    parser.add_argument("--finite-part-samples", type=int, default=24)
    args = parser.parse_args()

    if args.sector == "NS":
        plumbing = NSPlumbingParameter(args.q, args.lift_sign)
        block = SelfDualNSTorusOnePointBlock(
            internal_momentum=args.internal_momentum,
            external_momentum=args.external_momentum,
            samples=args.finite_part_samples,
        )
        max_twice_level = 2 * args.order
        coefficients = block.elliptic_coefficients(max_twice_level)
        value = block.evaluate(plumbing, max_twice_level)
        print("Type-0B NS-handle torus elliptic block")
        for twice_level, coefficient in coefficients.items():
            print(f"q^{twice_level}/2\t{_format(coefficient)}")
        print(f"value\t{_format(value)}")
        return

    plumbing = RamondPlumbingParameter(
        args.q, args.cycle_insertion
    )
    block = SelfDualRamondTorusOnePointBlock(
        internal_momentum=args.internal_momentum,
        external_momentum=args.external_momentum,
        sign=args.r_sign,
        samples=args.finite_part_samples,
    )
    elliptic = block.elliptic_coefficients(args.order)
    raw = block.raw_even_coefficients(args.order)
    projected = block.cycle_projected_raw_coefficients(
        plumbing, args.order
    )
    print("Type-0B long-R-handle torus block")
    print(
        f"ground contraction\t"
        f"{_format(projected[0] / raw[0])}"
    )
    for level in range(args.order + 1):
        print(
            f"q^{level}\tH={_format(elliptic[level])}\t"
            f"F_even={_format(raw[level])}\t"
            f"projected={_format(projected[level])}"
        )


if __name__ == "__main__":
    main()
