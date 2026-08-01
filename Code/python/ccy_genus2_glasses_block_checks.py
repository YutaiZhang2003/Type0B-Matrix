#!/usr/bin/env python3
"""Checks for the CCY genus-two glasses-frame recursion."""

from __future__ import annotations

import numpy as np

try:
    from ccy_genus2_block import b_from_c_rs_h, dc_rs_dh, format_complex
    from ccy_genus2_glasses_block import (
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        handle_residue_prefactor,
    )
    from virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        momentum_from_weight,
        torus_one_point_residue,
    )
    from torus_descendant_blocks import gram_matrix, rho_descendant_external
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import b_from_c_rs_h, dc_rs_dh, format_complex
    from plumbing.ccy_genus2_glasses_block import (
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        handle_residue_prefactor,
    )
    from plumbing.virasoro_blocks import (
        TorusOnePointVirasoroBlock,
        momentum_from_weight,
        torus_one_point_residue,
    )
    from plumbing.torus_descendant_blocks import gram_matrix, rho_descendant_external


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_order_zero_and_one() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    order0 = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=0,
        include_vacuum_seed=False,
    ).value
    order1 = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=1,
        include_vacuum_seed=False,
    ).value
    global1 = genus2_global_glasses_sl2_block(h_left, h_right, h_bridge, q_left, q_right, q_bridge, 1)
    print("order zero/one")
    print(f"  order0={order0!r}")
    print(f"  order1={order1!r}")
    print(f"  global1={global1!r}")
    require(abs(order0 - 1.0) < 1.0e-14, "order-zero block should be one without vacuum seed")
    require(abs(order1 - global1) < 1.0e-14, "order-one block should equal the global seed")


def check_handle_residue_matches_torus_one_point() -> None:
    r, s = 3, 2
    handle_weight = 1.37
    bridge_weight = 0.82
    b = b_from_c_rs_h(r, s, handle_weight)
    external_lambda = momentum_from_weight(bridge_weight, b)
    expected = -dc_rs_dh(r, s, handle_weight) * torus_one_point_residue(
        r,
        s,
        b,
        external_lambda,
    )
    observed = handle_residue_prefactor(r, s, handle_weight, bridge_weight)
    print("\nhandle tadpole residue")
    print(f"  observed={observed!r}")
    print(f"  expected={expected!r}")
    require(abs(observed - expected) < 1.0e-10, "handle residue is not the torus one-point residue")


def check_separating_limit_matches_two_tori() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 0.002, 0.0017, 1.0e-4
    order = 3
    glasses = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=5,
        vacuum_oscillator_level_max=40,
    ).value
    left = TorusOnePointVirasoroBlock(c, h_left, h_bridge).chiral_block(
        q_left,
        order,
        include_prefactor=False,
    )
    right = TorusOnePointVirasoroBlock(c, h_right, h_bridge).chiral_block(
        q_right,
        order,
        include_prefactor=False,
    )
    expected = left * right
    rel_error = abs(glasses - expected) / max(1.0e-30, abs(expected))
    print("\nseparating degeneration")
    print(f"  glasses={format_complex(glasses)}")
    print(f"  two tori={format_complex(expected)}")
    print(f"  rel error={rel_error:.6e}")
    require(rel_error < 1.0e-4, "glasses block does not approach the product of two torus blocks")


def check_order_three_finite() -> None:
    value = ccy_genus2_glasses_block(
        c=26.215,
        h_left=0.91,
        h_right=0.97,
        h_bridge=1.03,
        q_left=0.003 + 0.001j,
        q_right=0.0025 - 0.0007j,
        q_bridge=0.0012 + 0.0003j,
        order=3,
        include_vacuum_seed=True,
        vacuum_word_len=4,
        vacuum_oscillator_level_max=30,
    ).value
    print("\norder three finite")
    print(f"  value={value!r}")
    require(abs(value) > 0.0, "order-three value vanished unexpectedly")
    require(abs(value) < 10.0, "order-three value is implausibly large for tiny q")


def direct_glasses_descendant_block(
    *,
    c: complex,
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    order: int,
) -> complex:
    """Direct total-degree descendant sum for the glasses graph."""

    gram_data = {}
    for label, weight in (("left", h_left), ("right", h_right), ("bridge", h_bridge)):
        for level in range(order + 1):
            basis, gram = gram_matrix(weight, c, level)
            gram_data[(label, level)] = (basis, np.linalg.inv(gram))

    total = 0.0 + 0.0j
    for level_left in range(order + 1):
        basis_left, inv_left = gram_data[("left", level_left)]
        for level_right in range(order + 1 - level_left):
            basis_right, inv_right = gram_data[("right", level_right)]
            for level_bridge in range(order + 1 - level_left - level_right):
                basis_bridge, inv_bridge = gram_data[("bridge", level_bridge)]
                coefficient = 0.0 + 0.0j
                for a, state_left_bra in enumerate(basis_left):
                    for b, state_left_ket in enumerate(basis_left):
                        left_metric = inv_left[a, b]
                        for c_index, state_right_bra in enumerate(basis_right):
                            for d, state_right_ket in enumerate(basis_right):
                                right_metric = inv_right[c_index, d]
                                for e, state_bridge_left in enumerate(basis_bridge):
                                    rho_left = rho_descendant_external(
                                        state_left_bra,
                                        state_bridge_left,
                                        state_left_ket,
                                        h_left,
                                        h_bridge,
                                        c,
                                    )
                                    for f, state_bridge_right in enumerate(basis_bridge):
                                        rho_right = rho_descendant_external(
                                            state_right_bra,
                                            state_bridge_right,
                                            state_right_ket,
                                            h_right,
                                            h_bridge,
                                            c,
                                        )
                                        coefficient += (
                                            left_metric
                                            * right_metric
                                            * inv_bridge[e, f]
                                            * rho_left
                                            * rho_right
                                        )
                total += (
                    q_left**level_left
                    * q_right**level_right
                    * q_bridge**level_bridge
                    * coefficient
                )
    return total


def check_against_direct_descendant_sum() -> None:
    c = 26.215
    h_left, h_right, h_bridge = 0.91, 0.97, 1.03
    q_left, q_right, q_bridge = 1.1e-4, 1.3e-4, 0.9e-4
    order = 3
    direct = direct_glasses_descendant_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
    )
    recursive = ccy_genus2_glasses_block(
        c=c,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=6,
        vacuum_oscillator_level_max=20,
    ).value
    relative = abs(recursive - direct) / abs(direct)
    print("\ndirect descendant-sum comparison")
    print(f"  direct={direct!r}")
    print(f"  recursive={recursive!r}")
    print(f"  relative difference={relative:.6e}")
    require(relative < 1.0e-10, "glasses recursion disagrees with the direct level-three descendant sum")


def run() -> None:
    check_order_zero_and_one()
    check_handle_residue_matches_torus_one_point()
    check_separating_limit_matches_two_tori()
    check_order_three_finite()
    check_against_direct_descendant_sum()
    print("\nall CCY glasses-frame checks passed")


if __name__ == "__main__":
    run()
