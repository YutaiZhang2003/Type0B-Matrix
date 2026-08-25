#!/usr/bin/env python3
"""Reproducible checks for the Virasoro torus block recursion."""

from __future__ import annotations

import cmath

try:
    from virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        central_charge_to_b,
        partition_numbers,
        torus_one_point_chiral_block,
        torus_one_point_elliptic_block,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        central_charge_to_b,
        partition_numbers,
        torus_one_point_chiral_block,
        torus_one_point_elliptic_block,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def max_abs(values: list[complex]) -> float:
    return max(abs(value) for value in values) if values else 0.0


def check_identity_insertion() -> None:
    c = 30.0
    internal_h = 0.7
    external_h = 0.0
    order = 10
    block = TorusOnePointVirasoroBlock(c, internal_h, external_h)
    h_coeffs = block.elliptic_coefficients(order)
    descendant_coeffs = block.descendant_coefficients(order)
    partitions = partition_numbers(order)

    print("identity insertion")
    print(f"  max |H[n>0]|: {max_abs(h_coeffs[1:]):.6e}")
    print(f"  descendant coefficients: {descendant_coeffs}")
    require(abs(h_coeffs[0] - 1.0) < 1e-13, "H[0] is not normalized to one")
    require(max_abs(h_coeffs[1:]) < 1e-12, "identity insertion should have H(q)=1")
    require(
        max_abs([descendant_coeffs[n] - partitions[n] for n in range(order + 1)]) < 1e-12,
        "identity insertion should reduce to Verma character coefficients",
    )


def check_level_one_coefficient() -> None:
    c = 30.0
    internal_h = 0.8
    external_h = 0.3
    block = TorusOnePointVirasoroBlock(c, internal_h, external_h)
    h1 = block.elliptic_coefficients(1)[1]
    expected_h1 = external_h * (external_h - 1.0) / (2.0 * internal_h)
    full_level_one = block.descendant_coefficients(1)[1]
    expected_full_level_one = 1.0 + expected_h1

    print("\nlevel-one coefficient")
    print(f"  H[1]: {h1!r}")
    print(f"  expected H[1]: {expected_h1!r}")
    print(f"  full q coefficient: {full_level_one!r}")
    require(abs(h1 - expected_h1) < 1e-12, "wrong level-one elliptic coefficient")
    require(abs(full_level_one - expected_full_level_one) < 1e-12, "wrong level-one full coefficient")


def check_c_recursion_against_h_recursion() -> None:
    cases = [
        {
            "c": 30.0,
            "h": 0.83,
            "d": 0.27,
            "b": central_charge_to_b(30.0),
            "external_lambda": None,
        },
    ]
    physical_b = 0.8
    q_timelike = 1.0 / physical_b - physical_b
    external_momentum = 2.5
    internal_momentum = 0.7 + 0.05j
    cases.append(
        {
            "c": 1.0 - 6.0 * q_timelike**2,
            "h": -q_timelike**2 / 4.0 + internal_momentum**2,
            "d": -q_timelike**2 / 4.0 - external_momentum**2,
            "b": 1.0j * physical_b,
            "external_lambda": -2.0 * external_momentum,
        }
    )

    largest_relative_error = 0.0
    for case in cases:
        c_block = TorusOnePointVirasoroBlock(
            case["c"],
            case["h"],
            case["d"],
            recursion="c",
        )
        h_block = TorusOnePointVirasoroBlock(
            case["c"],
            case["h"],
            case["d"],
            b=case["b"],
            external_lambda=case["external_lambda"],
            recursion="h",
        )
        c_coefficients = c_block.elliptic_coefficients(6)
        h_coefficients = h_block.elliptic_coefficients(6)
        largest_relative_error = max(
            largest_relative_error,
            max(
                abs(c_value - h_value) / max(1.0, abs(h_value))
                for c_value, h_value in zip(c_coefficients, h_coefficients)
            ),
        )

    print("\nfixed-weight c-recursion cross-check")
    print(f"  max relative coefficient difference: {largest_relative_error:.6e}")
    require(
        largest_relative_error < 5.0e-11,
        "fixed-weight c-recursion disagrees with the independent h-recursion",
    )


def check_b_branch_invariance() -> None:
    c = 30.0
    internal_h = 0.9
    external_h = 0.4
    q = 0.025 + 0.015j
    order = 7
    b = central_charge_to_b(c)
    value_b = torus_one_point_elliptic_block(c, internal_h, external_h, q, order, b=b)
    value_inv_b = torus_one_point_elliptic_block(c, internal_h, external_h, q, order, b=1.0 / b)

    print("\nb-branch invariance")
    print(f"  |H_b-H_1/b|: {abs(value_b - value_inv_b):.6e}")
    require(abs(value_b - value_inv_b) < 1e-11, "block should be invariant under b -> 1/b")


def check_order_stability() -> None:
    c = 30.0
    internal_h = 0.83
    external_h = 0.27
    q = 0.03 * cmath.exp(0.4j)
    h6 = torus_one_point_elliptic_block(c, internal_h, external_h, q, 6)
    h9 = torus_one_point_elliptic_block(c, internal_h, external_h, q, 9)
    f6 = torus_one_point_chiral_block(c, internal_h, external_h, q, 6, include_prefactor=False)
    f9 = torus_one_point_chiral_block(c, internal_h, external_h, q, 9, include_prefactor=False)

    print("\norder stability")
    print(f"  |H_order9-H_order6|: {abs(h9 - h6):.6e}")
    print(f"  |F_order9-F_order6| without prefactor: {abs(f9 - f6):.6e}")
    require(abs(h9 - h6) < 1e-9, "elliptic block is not stable at small q")
    require(abs(f9 - f6) < 1e-8, "full descendant series is not stable at small q")


def run() -> None:
    check_identity_insertion()
    check_level_one_coefficient()
    check_c_recursion_against_h_recursion()
    check_b_branch_invariance()
    check_order_stability()
    print("\nall Virasoro block checks passed")


if __name__ == "__main__":
    run()
