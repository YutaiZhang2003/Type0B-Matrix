#!/usr/bin/env python3
"""Checks for torus one-point descendant blocks."""

from __future__ import annotations

try:
    from torus_descendant_blocks import (
        gram_matrix,
        lminus_one_power_multiplier,
        rho_descendant_external,
        torus_one_point_descendant_block,
        torus_one_point_descendant_coefficients,
    )
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.torus_descendant_blocks import (
        gram_matrix,
        lminus_one_power_multiplier,
        rho_descendant_external,
        torus_one_point_descendant_block,
        torus_one_point_descendant_coefficients,
    )
    from plumbing.virasoro_blocks import TorusOnePointVirasoroBlock


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def max_abs(values: list[complex]) -> float:
    return max(abs(value) for value in values) if values else 0.0


def check_gram_level_two() -> None:
    c = 30.0
    h = 0.8
    basis, gram = gram_matrix(h, c, 2)
    print("Gram matrix level 2")
    print(f"  basis={basis}")
    print(f"  gram={gram.tolist()}")
    require(basis == ((2,), (1, 1)), "unexpected level-2 partition basis")
    require(abs(gram[0, 0] - (4.0 * h + 0.5 * c)) < 1.0e-12, "wrong <L_-2|L_-2>")
    require(abs(gram[0, 1] - 6.0 * h) < 1.0e-12, "wrong <L_-2|L_-1^2>")
    require(abs(gram[1, 1] - 4.0 * h * (2.0 * h + 1.0)) < 1.0e-12, "wrong <L_-1^2|L_-1^2>")


def check_primary_matches_recursion() -> None:
    c = 30.0
    internal_h = 0.83
    external_h = 0.27
    order = 4
    q = 0.025 + 0.011j
    direct = torus_one_point_descendant_coefficients(c, internal_h, external_h, (), order)
    recursion = TorusOnePointVirasoroBlock(c, internal_h, external_h).descendant_coefficients(order)
    diffs = [direct[n] - recursion[n] for n in range(order + 1)]
    direct_value = torus_one_point_descendant_block(c, internal_h, external_h, (), q, order)
    recursion_value = TorusOnePointVirasoroBlock(c, internal_h, external_h).chiral_block(
        q,
        order,
        include_prefactor=False,
    )
    print("\nprimary external check")
    print(f"  max coefficient diff={max_abs(diffs):.6e}")
    print(f"  value diff={abs(direct_value - recursion_value):.6e}")
    require(max_abs(diffs) < 1.0e-10, "direct Ward sum does not match primary recursion")
    require(abs(direct_value - recursion_value) < 1.0e-11, "direct Ward block does not match primary recursion")


def check_lminus_one_tower() -> None:
    c = 30.0
    internal_h = 0.83
    external_h = 0.27
    order = 4
    primary = torus_one_point_descendant_coefficients(c, internal_h, external_h, (), order)
    for power in (1, 2, 3):
        state = (1,) * power
        multiplier = lminus_one_power_multiplier(external_h, power)
        descendant = torus_one_point_descendant_coefficients(c, internal_h, external_h, state, order)
        diffs = [descendant[n] - multiplier * primary[n] for n in range(order + 1)]
        print(f"\nL_-1^{power} external check")
        print(f"  multiplier={multiplier!r}")
        print(f"  max coefficient diff={max_abs(diffs):.6e}")
        require(max_abs(diffs) < 1.0e-10, f"L_-1^{power} multiplier check failed")


def check_lminus_two_smoke() -> None:
    c = 30.0
    internal_h = 0.83
    external_h = 0.27
    value = rho_descendant_external((), (2,), (), internal_h, external_h, c)
    expected = internal_h + external_h
    print("\nL_-2 primary matrix element")
    print(f"  rho(h,L_-2 d,h)={value!r}")
    print(f"  expected={expected!r}")
    require(abs(value - expected) < 1.0e-12, "wrong rho(h,L_-2 d,h)")


def run() -> None:
    check_gram_level_two()
    check_primary_matches_recursion()
    check_lminus_one_tower()
    check_lminus_two_smoke()
    print("\nall torus descendant block checks passed")


if __name__ == "__main__":
    run()
