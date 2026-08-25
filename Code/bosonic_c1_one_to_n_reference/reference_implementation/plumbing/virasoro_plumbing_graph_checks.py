#!/usr/bin/env python3
"""Focused checks for the direct plumbing-graph Virasoro block."""

from __future__ import annotations

import math

try:
    from ccy_genus2_block import ccy_genus2_block, genus2_global_sl2_block, rho_lminus1_triple
    from ccy_genus2_glasses_block import ccy_genus2_glasses_block
    from genus2_vacuum_blocks import descendant_inner_product
    from virasoro_plumbing_graph import (
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
        rho_primary_descendants,
        verma_gram_matrix,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import ccy_genus2_block, genus2_global_sl2_block, rho_lminus1_triple
    from plumbing.ccy_genus2_glasses_block import ccy_genus2_glasses_block
    from plumbing.genus2_vacuum_blocks import descendant_inner_product
    from plumbing.virasoro_plumbing_graph import (
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
        rho_primary_descendants,
        verma_gram_matrix,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_verma_gram_levels_one_and_two() -> None:
    h = 0.91
    c = 26.215
    basis1, gram1 = verma_gram_matrix(1, h, c)
    basis2, gram2 = verma_gram_matrix(2, h, c)
    expected2 = (
        (4.0 * h + 0.5 * c, 6.0 * h),
        (6.0 * h, 4.0 * h * (2.0 * h + 1.0)),
    )
    maximum_error = max(
        abs(gram2[row][column] - expected2[row][column])
        for row in range(2)
        for column in range(2)
    )
    print("Verma Gram matrices")
    print(f"  level-one basis={basis1}, G11={gram1[0][0]!r}")
    print(f"  level-two basis={basis2}, max analytic error={maximum_error:.3e}")
    require(basis1 == ((1,),), "unexpected level-one PBW basis")
    require(basis2 == ((2,), (1, 1)), "unexpected level-two PBW basis")
    require(abs(gram1[0][0] - 2.0 * h) < 1.0e-14, "level-one Gram norm is wrong")
    require(maximum_error < 1.0e-13, "level-two Gram matrix is wrong")


def check_three_point_global_closed_form() -> None:
    c = 26.215
    h_infinity, h_one, h_zero = 0.91, 0.97, 1.03
    maximum_error = 0.0
    checked = 0
    for i_level in range(4):
        for j_level in range(4 - i_level):
            for k_level in range(4 - i_level - j_level):
                observed = rho_primary_descendants(
                    (1,) * i_level,
                    (1,) * j_level,
                    (1,) * k_level,
                    h_infinity,
                    h_one,
                    h_zero,
                    c,
                )
                expected = rho_lminus1_triple(
                    i_level,
                    j_level,
                    k_level,
                    h_infinity,
                    h_one,
                    h_zero,
                )
                maximum_error = max(maximum_error, abs(observed - expected))
                checked += 1
    print("\nthree-point descendant tensor")
    print(f"  closed L_-1 cases checked={checked}")
    print(f"  max absolute error={maximum_error:.3e}")
    require(maximum_error < 1.0e-12, "Ward recursion disagrees with the closed L_-1 formula")


def check_identity_insertion_reduces_to_inner_product() -> None:
    h = 0.83
    c = 25.7
    maximum_error = 0.0
    descendants = ((), (1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1))
    for desc_infinity in descendants:
        for desc_zero in descendants:
            observed = rho_primary_descendants(
                desc_infinity,
                (),
                desc_zero,
                h,
                0.0,
                h,
                c,
            )
            expected = descendant_inner_product(
                desc_infinity,
                desc_zero,
                h=h,
                c=c,
                vacuum=False,
            )
            maximum_error = max(maximum_error, abs(observed - expected))
    print("\nidentity insertion")
    print(f"  max |rho(desc,I,desc)-Gram|={maximum_error:.3e}")
    require(maximum_error < 1.0e-11, "primary identity insertion did not reduce to the Gram form")


def check_theta_against_genus2_results() -> None:
    c = 26.215
    weights = (0.91, 0.97, 1.03)
    q_values = (0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j)

    direct_order1 = direct_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=c,
        edge_weights=weights,
        q_values=q_values,
        max_total_level=1,
    )
    global_order1 = genus2_global_sl2_block(*weights, *q_values, order=1)

    direct_order3 = direct_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=c,
        edge_weights=weights,
        q_values=q_values,
        max_total_level=3,
    )
    recursive_order3 = ccy_genus2_block(
        c=c,
        h1=weights[0],
        h2=weights[1],
        h3=weights[2],
        q1=q_values[0],
        q2=q_values[1],
        q3=q_values[2],
        order=3,
        include_vacuum_seed=False,
    ).value
    print("\ngenus-two theta graph")
    print(f"  order-one direct/global error={abs(direct_order1.value - global_order1):.3e}")
    print(f"  order-three direct/CCY error={abs(direct_order3.value - recursive_order3):.3e}")
    print(f"  maximum Gram condition number={direct_order3.max_gram_condition_number:.3e}")
    require(
        abs(direct_order1.value - global_order1) < 1.0e-13,
        "theta block does not reduce to the global block at total level one",
    )
    require(
        abs(direct_order3.value - recursive_order3) < 1.0e-12,
        "theta block disagrees with the known genus-two result at total level three",
    )


def check_glasses_against_genus2_results() -> None:
    c = 26.215
    weights = (0.91, 0.97, 1.03)
    q_values = (1.1e-4, 1.3e-4, 0.9e-4)
    order = 3
    direct = direct_plumbing_graph_block(
        genus2_glasses_graph(),
        central_charge=c,
        edge_weights=weights,
        q_values=q_values,
        max_total_level=order,
    )
    recursive = ccy_genus2_glasses_block(
        c=c,
        h_left=weights[0],
        h_right=weights[1],
        h_bridge=weights[2],
        q_left=q_values[0],
        q_right=q_values[1],
        q_bridge=q_values[2],
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=6,
        vacuum_oscillator_level_max=20,
    ).value
    relative_error = abs(direct.value - recursive) / abs(direct.value)
    print("\ngenus-two glasses graph")
    print(f"  direct={direct.value!r}")
    print(f"  recursive={recursive!r}")
    print(f"  relative error={relative_error:.3e}")
    require(
        relative_error < 2.0e-12,
        "generic graph contraction disagrees with the genus-two glasses recursion",
    )


def check_tetrahedral_smoke() -> None:
    graph = genus3_tetrahedral_graph()
    result = direct_plumbing_graph_block(
        graph,
        central_charge=26.0,
        edge_weights=(0.81, 0.87, 0.93, 0.99, 1.05, 1.11),
        q_values=(1.0e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-4),
        max_total_level=1,
    )
    print("\ngenus-three tetrahedral smoke check")
    print(f"  genus={graph.genus}, value={result.value!r}")
    require(graph.genus == 3, "K4 graph has the wrong first Betti number")
    require(math.isfinite(abs(result.value)), "K4 order-one block is not finite")
    require(abs(result.coefficient_by_levels[(0, 0, 0, 0, 0, 0)] - 1.0) < 1.0e-14, "K4 primary term is not one")


def run() -> None:
    check_verma_gram_levels_one_and_two()
    check_three_point_global_closed_form()
    check_identity_insertion_reduces_to_inner_product()
    check_theta_against_genus2_results()
    check_glasses_against_genus2_results()
    check_tetrahedral_smoke()
    print("\nall direct plumbing-graph Virasoro checks passed")


if __name__ == "__main__":
    run()
