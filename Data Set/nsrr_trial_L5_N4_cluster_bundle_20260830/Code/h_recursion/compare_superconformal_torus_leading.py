#!/usr/bin/env python3
"""Compare low-level torus descendant sewing with the elliptic recursion."""

from __future__ import annotations

import argparse

from ramond_sphere_blocks import ramond_beta
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSTorusOnePointBlock,
    RamondTorusOnePointBlock,
    SelfDualNSTorusOnePointBlock,
    SelfDualRamondTorusOnePointBlock,
)
from superconformal_torus_descendants import (
    BruteForceNSTorusOnePointBlock,
    BruteForceRamondTorusOnePointBlock,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def _print_matrix(label: str, matrix: tuple[tuple[complex, ...], ...]) -> None:
    print(label)
    for row in matrix:
        print("    " + "  ".join(f"{_format(value):>22}" for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct Gram/Ward torus coefficients with the "
            "Zamolodchikov recursion."
        )
    )
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--internal-ns-momentum", type=float, default=0.61)
    parser.add_argument("--internal-r-momentum", type=float, default=0.60)
    parser.add_argument("--external-momentum", type=float, default=0.33)
    parser.add_argument("--finite-part-samples", type=int, default=16)
    args = parser.parse_args()

    b = complex(args.b)
    c = central_charge(b)
    h_ns = ns_liouville_weight(args.internal_ns_momentum, b)
    h_r = c / 24.0 - ramond_beta(args.internal_r_momentum) ** 2
    d = ns_liouville_weight(args.external_momentum, b)

    ns_direct = BruteForceNSTorusOnePointBlock(
        internal_weight=h_ns,
        external_weight=d,
    )
    ns_raw = ns_direct.raw_coefficients()
    ns_direct_h = ns_direct.elliptic_coefficients()
    if abs(b - 1.0) < 1.0e-14:
        ns_recursive_h = SelfDualNSTorusOnePointBlock(
            internal_momentum=args.internal_ns_momentum,
            external_momentum=args.external_momentum,
            samples=args.finite_part_samples,
        ).elliptic_coefficients(2)
    else:
        ns_recursive_h = NSTorusOnePointBlock(
            b=b,
            internal_weight=h_ns,
            external_weight=d,
        ).elliptic_coefficients(2)

    print("NS handle")
    print(f"  c={_format(c)}, h={_format(h_ns)}, d={_format(d)}")
    for twice_level, level_label in ((0, "0"), (1, "1/2"), (2, "1")):
        direct = ns_direct_h[twice_level]
        recursive = ns_recursive_h[twice_level]
        print(
            f"  level {level_label:>3}: "
            f"F_direct={_format(ns_raw[twice_level])}, "
            f"H_direct={_format(direct)}, "
            f"H_rec={_format(recursive)}, "
            f"|difference|={abs(direct-recursive):.3e}"
        )

    for sign in (1, -1):
        direct_block = BruteForceRamondTorusOnePointBlock(
            central_charge=c,
            internal_weight=h_r,
            external_weight=d,
            sign=sign,
        )
        direct_raw = direct_block.raw_even_coefficients()
        if abs(b - 1.0) < 1.0e-14:
            recursive = SelfDualRamondTorusOnePointBlock(
                internal_momentum=args.internal_r_momentum,
                external_momentum=args.external_momentum,
                sign=sign,
                samples=args.finite_part_samples,
            )
        else:
            recursive = RamondTorusOnePointBlock(
                b=b,
                internal_beta=ramond_beta(args.internal_r_momentum),
                external_weight=d,
                sign=sign,
            )
        recursive_raw = recursive.raw_even_coefficients(1)
        print()
        print(f"R handle, HJS sign {sign:+d}")
        print(f"  c={_format(c)}, h={_format(h_r)}, d={_format(d)}")
        _print_matrix("  level-one Gram matrix:", direct_block.gram_matrices()[1])
        _print_matrix(
            "  level-one Ward vertex matrix:",
            direct_block.vertex_matrices()[1],
        )
        for level in (0, 1):
            print(
                f"  level {level}: "
                f"F_direct={_format(direct_raw[level])}, "
                f"F_rec={_format(recursive_raw[level])}, "
                f"|difference|={abs(direct_raw[level]-recursive_raw[level]):.3e}"
            )


if __name__ == "__main__":
    main()
