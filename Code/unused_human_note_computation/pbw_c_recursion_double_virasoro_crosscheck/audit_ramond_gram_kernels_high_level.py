#!/usr/bin/env python3
"""Higher-level exact certificate for the Ramond Gram null doublets."""

from __future__ import annotations

import argparse
import time

import sympy as sp

from ramond_pbw_generalized_ward import (
    RamondPBWModule,
    clean,
    ramond_degenerate_data,
    ramond_labels_at_level,
)


def certify_kernel(r: int, s: int, b: sp.Expr) -> tuple[int, float]:
    """Prove nullity one and Gram annihilation for chi^+ and chi^-."""

    started = time.monotonic()
    data = ramond_degenerate_data(r, s, b)
    level = int(data["level"])
    module = RamondPBWModule(data["h"], data["beta"], data["c"])
    dimension = 0
    for parity in (0, 1):
        basis, gram = module.gram_matrix(level, parity)
        dimension = len(basis)
        kernel_basis, kernel = module.gram_kernel(level, parity)
        if kernel_basis != basis or len(kernel) != 1:
            raise AssertionError(
                f"(r,s)=({r},{s}), parity={parity}: nullity {len(kernel)}"
            )
        null = module.normalized_null_vector(level, parity)
        column = sp.Matrix([null.get(state, sp.S.Zero) for state in basis])
        for residual in gram * column:
            if clean(residual) != 0:
                raise AssertionError(
                    f"(r,s)=({r},{s}), parity={parity}: nonzero Gram residual"
                )
    return dimension, time.monotonic() - started


def run(symbolic_through: int, sampled_through: int, samples: tuple[sp.Expr, ...]):
    if sampled_through < symbolic_through:
        raise ValueError("sampled-through must be at least symbolic-through")
    b = sp.symbols("b", nonzero=True)
    print("Higher-level Ramond Gram-kernel certificate")
    print(f"  symbolic generic-b levels: 1..{symbolic_through}")
    print(f"  exact sampled levels: {symbolic_through + 1}..{sampled_through}")
    print(f"  samples: {samples}")

    for level in range(1, symbolic_through + 1):
        for r, s in ramond_labels_at_level(level):
            dimension, elapsed = certify_kernel(r, s, b)
            print(
                f"PASS symbolic: level={level}, (r,s)=({r},{s}), "
                f"blocks=2x{dimension}, seconds={elapsed:.2f}"
            )

    for level in range(symbolic_through + 1, sampled_through + 1):
        for r, s in ramond_labels_at_level(level):
            for sample in samples:
                dimension, elapsed = certify_kernel(r, s, sample)
                print(
                    f"PASS exact sample: level={level}, (r,s)=({r},{s}), "
                    f"b={sample}, blocks=2x{dimension}, seconds={elapsed:.2f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic-through", type=int, default=3)
    parser.add_argument("--sampled-through", type=int, default=5)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=("2/3", "3/2"),
        help="exact SymPy rationals or algebraic expressions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.symbolic_through,
        arguments.sampled_through,
        tuple(sp.sympify(sample) for sample in arguments.samples),
    )
