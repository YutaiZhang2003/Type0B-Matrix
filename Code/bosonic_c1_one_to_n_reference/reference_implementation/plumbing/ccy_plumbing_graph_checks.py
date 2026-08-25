#!/usr/bin/env python3
"""Focused regression checks for the graph-generic CCY recursion."""

from __future__ import annotations

import math

try:
    from ccy_genus2_block import (
        ccy_genus2_block,
        ccy_residue_prefactor_for_weights,
        genus2_global_sl2_block,
    )
    from ccy_genus2_glasses_block import (
        bridge_residue_prefactor,
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        handle_residue_prefactor,
    )
    from ccy_plumbing_graph import (
        ccy_genus3_channel_block,
        ccy_genus3_tetrahedral_block,
        ccy_plumbing_graph_block,
        genus3_channel_vacuum_seed_schottky,
        genus3_tetrahedral_vacuum_seed_schottky,
        global_sl2_plumbing_graph_block,
        graph_edge_residue_prefactor,
    )
    from genus2_vacuum_blocks import schottky_vacuum_block
    from genus3_plumbing_channels import enumerate_genus3_channels
    from genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from virasoro_plumbing_graph import (
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        ccy_genus2_block,
        ccy_residue_prefactor_for_weights,
        genus2_global_sl2_block,
    )
    from plumbing.ccy_genus2_glasses_block import (
        bridge_residue_prefactor,
        ccy_genus2_glasses_block,
        genus2_global_glasses_sl2_block,
        handle_residue_prefactor,
    )
    from plumbing.ccy_plumbing_graph import (
        ccy_genus3_channel_block,
        ccy_genus3_tetrahedral_block,
        ccy_plumbing_graph_block,
        genus3_channel_vacuum_seed_schottky,
        genus3_tetrahedral_vacuum_seed_schottky,
        global_sl2_plumbing_graph_block,
        graph_edge_residue_prefactor,
    )
    from plumbing.genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing.genus3_plumbing_channels import enumerate_genus3_channels
    from plumbing.genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from plumbing.virasoro_plumbing_graph import (
        direct_plumbing_graph_block,
        genus2_glasses_graph,
        genus2_theta_graph,
        genus3_tetrahedral_graph,
    )


def require_close(observed: complex, expected: complex, tolerance: float, message: str) -> None:
    error = abs(complex(observed) - complex(expected))
    if error > tolerance:
        raise AssertionError(f"{message}: error={error:.6e}, observed={observed!r}, expected={expected!r}")


def check_global_blocks() -> None:
    weights = (0.91, 0.97, 1.03)
    q_values = (0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j)
    order = 4
    theta_generic = global_sl2_plumbing_graph_block(
        genus2_theta_graph(),
        edge_weights=weights,
        q_values=q_values,
        order=order,
    )
    theta_known = genus2_global_sl2_block(*weights, *q_values, order)
    glasses_generic = global_sl2_plumbing_graph_block(
        genus2_glasses_graph(),
        edge_weights=weights,
        q_values=q_values,
        order=order,
    )
    glasses_known = genus2_global_glasses_sl2_block(*weights, *q_values, order)
    print("global graph blocks")
    print(f"  theta error={abs(theta_generic - theta_known):.3e}")
    print(f"  glasses error={abs(glasses_generic - glasses_known):.3e}")
    require_close(theta_generic, theta_known, 1.0e-13, "generic theta global block is wrong")
    require_close(glasses_generic, glasses_known, 1.0e-13, "generic glasses global block is wrong")


def check_genus3_global_regular_terms_order_ten() -> None:
    """Lock the six-edge global sum and its default CCY wiring in every channel."""

    momenta = (0.11, 0.17, 0.23, 0.29, 0.31, 0.37)
    weights = tuple(1.0 + momentum * momentum for momentum in momenta)
    q_values = (
        0.020 + 0.001j,
        0.021 - 0.001j,
        0.022 + 0.0005j,
        0.023 - 0.0007j,
        0.024 + 0.0003j,
        0.025 - 0.0004j,
    )
    expected_order_ten = {
        "one-tadpole-double-triangle": 0.9714131510292924 + 0.0009422584038636356j,
        "opposite-double-edge-cycle": 0.9801897775771755 + 0.0006265385706375538j,
        "tetrahedron": 1.0251241252265728 + 0.0005170568372796679j,
        "three-tadpole-star": 1.0102107508757596 + 0.001274326648232865j,
        "two-tadpoles-double-bridge": 0.9794717405346907 + 0.001532245725559669j,
    }

    print("\ngenus-three global-times-vacuum regular terms")
    for channel in enumerate_genus3_channels():
        order_nine = global_sl2_plumbing_graph_block(
            channel.graph,
            edge_weights=weights,
            q_values=q_values,
            order=9,
        )
        order_ten = global_sl2_plumbing_graph_block(
            channel.graph,
            edge_weights=weights,
            q_values=q_values,
            order=10,
        )
        vacuum_seed = genus3_channel_vacuum_seed_schottky(
            channel.name,
            q_values,
            max_word_len=3,
            oscillator_level_max=12,
        )
        order_one = global_sl2_plumbing_graph_block(
            channel.graph,
            edge_weights=weights,
            q_values=q_values,
            order=1,
        )
        default_block = ccy_genus3_channel_block(
            channel=channel,
            central_charge=25.0,
            edge_weights=weights,
            q_values=q_values,
            order=1,
            vacuum_word_len=3,
            vacuum_oscillator_level_max=12,
        )
        shell = abs(order_ten - order_nine)
        print(
            f"  {channel.name}: G10={order_ten!r}, "
            f"|G10-G9|={shell:.3e}, Zvac={vacuum_seed!r}"
        )
        require_close(
            order_ten,
            expected_order_ten[channel.name],
            2.0e-13,
            f"{channel.name} order-ten global block changed",
        )
        if shell > 1.0e-11:
            raise AssertionError(f"{channel.name} global block has not converged by order ten")
        require_close(
            default_block.value,
            vacuum_seed * order_one,
            2.0e-13,
            f"{channel.name} default regular term is not vacuum times global",
        )
        if default_block.regular_term_scheme != "schottky-vacuum-times-global":
            raise AssertionError(f"{channel.name} does not use the global CCY seed by default")


def check_graph_residues() -> None:
    weights = (0.91, 0.97, 1.03)
    r, s = 2, 1
    theta = genus2_theta_graph()
    theta_expected = (
        ccy_residue_prefactor_for_weights(r, s, weights[0], weights[2], weights[1]),
        ccy_residue_prefactor_for_weights(r, s, weights[1], weights[2], weights[0]),
        ccy_residue_prefactor_for_weights(r, s, weights[2], weights[0], weights[1]),
    )
    theta_observed = tuple(
        graph_edge_residue_prefactor(
            theta,
            edge_index=edge_index,
            r=r,
            s=s,
            edge_weights=weights,
        )
        for edge_index in range(3)
    )

    glasses = genus2_glasses_graph()
    glasses_expected = (
        handle_residue_prefactor(r, s, weights[0], weights[2]),
        handle_residue_prefactor(r, s, weights[1], weights[2]),
        bridge_residue_prefactor(r, s, weights[2], weights[0], weights[1]),
    )
    glasses_observed = tuple(
        graph_edge_residue_prefactor(
            glasses,
            edge_index=edge_index,
            r=r,
            s=s,
            edge_weights=weights,
        )
        for edge_index in range(3)
    )

    theta_error = max(abs(left - right) for left, right in zip(theta_observed, theta_expected))
    glasses_error = max(abs(left - right) for left, right in zip(glasses_observed, glasses_expected))
    print("\ngraph null-vector residues")
    print(f"  theta max error={theta_error:.3e}")
    print(f"  glasses max error={glasses_error:.3e}")
    if theta_error > 1.0e-12:
        raise AssertionError("generic theta residue does not reproduce the specialized CCY residue")
    if glasses_error > 1.0e-12:
        raise AssertionError("generic glasses residue does not reproduce the specialized CCY residue")


def check_specialized_recursions() -> None:
    c_value = 26.215
    weights = (0.91, 0.97, 1.03)
    order = 3

    theta_q = (0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j)
    theta_generic = ccy_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=c_value,
        edge_weights=weights,
        q_values=theta_q,
        order=order,
        include_vacuum_seed=False,
    ).value
    theta_known = ccy_genus2_block(
        c=c_value,
        h1=weights[0],
        h2=weights[1],
        h3=weights[2],
        q1=theta_q[0],
        q2=theta_q[1],
        q3=theta_q[2],
        order=order,
        include_vacuum_seed=False,
    ).value

    glasses_q = (1.1e-4, 1.3e-4, 0.9e-4)
    glasses_generic = ccy_plumbing_graph_block(
        genus2_glasses_graph(),
        central_charge=c_value,
        edge_weights=weights,
        q_values=glasses_q,
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=6,
        vacuum_oscillator_level_max=20,
    ).value
    glasses_known = ccy_genus2_glasses_block(
        c=c_value,
        h_left=weights[0],
        h_right=weights[1],
        h_bridge=weights[2],
        q_left=glasses_q[0],
        q_right=glasses_q[1],
        q_bridge=glasses_q[2],
        order=order,
        include_vacuum_seed=True,
        vacuum_word_len=6,
        vacuum_oscillator_level_max=20,
    ).value

    print("\nfull genus-two recursions")
    print(f"  theta error={abs(theta_generic - theta_known):.3e}")
    print(f"  glasses error={abs(glasses_generic - glasses_known):.3e}")
    require_close(theta_generic, theta_known, 1.0e-12, "generic theta recursion is wrong")
    require_close(glasses_generic, glasses_known, 1.0e-12, "generic glasses recursion is wrong")


def check_k4_against_direct_sum() -> None:
    graph = genus3_tetrahedral_graph()
    c_value = 26.215
    weights = (0.81, 0.87, 0.93, 0.99, 1.05, 1.11)
    q_values = (1.0e-4, 1.1e-4, 1.2e-4, 1.3e-4, 1.4e-4, 1.5e-4)
    order = 3
    direct = direct_plumbing_graph_block(
        graph,
        central_charge=c_value,
        edge_weights=weights,
        q_values=q_values,
        max_total_level=order,
    )
    recursive = ccy_genus3_tetrahedral_block(
        central_charge=c_value,
        edge_weights=weights,
        q_values=q_values,
        order=order,
        include_vacuum_seed=False,
    )
    error = abs(direct.value - recursive.value)
    print("\ngenus-three K4 recursion")
    print(f"  direct={direct.value!r}")
    print(f"  recursive={recursive.value!r}")
    print(f"  absolute error through total level {order}={error:.3e}")
    print(
        "  partial fractions="
        f"{recursive.partial_fraction_pole_count} poles, "
        f"{recursive.partial_fraction_coefficient_count} coefficients"
    )
    require_close(recursive.value, direct.value, 1.0e-12, "K4 recursion disagrees with direct contraction")


def check_k4_confluent_pole_regulator() -> None:
    """Check the first order at which equal internal weights produce a collision."""

    graph = genus3_tetrahedral_graph()
    weights = (1.04,) * 6
    q_values = (0.020, 0.021, 0.022, 0.023, 0.024, 0.025)
    direct = direct_plumbing_graph_block(
        graph,
        central_charge=25.0,
        edge_weights=weights,
        q_values=q_values,
        max_total_level=4,
    )
    regulated = ccy_genus3_channel_block(
        channel="tetrahedron",
        central_charge=25.0,
        edge_weights=weights,
        q_values=q_values,
        order=4,
        include_vacuum_seed=True,
        regular_term_scheme="pants",
    )
    alternate_direction = ccy_genus3_channel_block(
        channel="tetrahedron",
        central_charge=25.0,
        edge_weights=weights,
        q_values=q_values,
        order=4,
        include_vacuum_seed=True,
        regular_term_scheme="pants",
        collision_regulator_direction=(0.0, 2.0, 5.0, 9.0, 14.0, 20.0),
    )
    near_weights = tuple(1.04 + 1.0e-6 * index for index in range(6))
    near_direct = direct_plumbing_graph_block(
        graph,
        central_charge=25.0,
        edge_weights=near_weights,
        q_values=q_values,
        max_total_level=4,
    )
    near_recursive = ccy_genus3_channel_block(
        channel="tetrahedron",
        central_charge=25.0,
        edge_weights=near_weights,
        q_values=q_values,
        order=4,
        include_vacuum_seed=True,
        regular_term_scheme="pants",
    )

    exact_error = abs(regulated.value - direct.value)
    direction_error = abs(regulated.value - alternate_direction.value)
    near_error = abs(near_recursive.value - near_direct.value)
    print("\nK4 confluent-pole regulator")
    print(f"  direct={direct.value!r}")
    print(f"  regulated={regulated.value!r}")
    print(f"  exact collision error={exact_error:.3e}")
    print(f"  direction error={direction_error:.3e}")
    print(f"  near-collision error={near_error:.3e}")
    if not regulated.collision_regulated or not alternate_direction.collision_regulated:
        raise AssertionError("K4 exact collision did not activate the whole-block regulator")
    require_close(regulated.value, direct.value, 3.0e-13, "regulated K4 collision is wrong")
    require_close(
        alternate_direction.value,
        regulated.value,
        3.0e-13,
        "K4 collision limit depends on regulator direction",
    )
    require_close(
        near_recursive.value,
        near_direct.value,
        3.0e-13,
        "generic K4 recursion is discontinuous near the collision",
    )


def check_all_genus3_confluent_pole_regulators() -> None:
    """Verify the physical repeated-weight limit in all five graph types."""

    weights = (1.04,) * 6
    q_values = (0.020, 0.021, 0.022, 0.023, 0.024, 0.025)
    print("\nall-channel exact repeated-weight recursion")
    for channel in enumerate_genus3_channels():
        direct = direct_plumbing_graph_block(
            channel.graph,
            central_charge=25.0,
            edge_weights=weights,
            q_values=q_values,
            max_total_level=4,
        )
        regulated = ccy_genus3_channel_block(
            channel=channel,
            central_charge=25.0,
            edge_weights=weights,
            q_values=q_values,
            order=4,
            include_vacuum_seed=True,
            regular_term_scheme="pants",
            collision_regulator_direction={
                name: value
                for name, value in reversed(
                    tuple(
                        zip(
                            channel.edge_names,
                            (0.0, 2.0, 5.0, 9.0, 14.0, 20.0),
                        )
                    )
                )
            },
        )
        error = abs(regulated.value - direct.value)
        print(f"  {channel.name}: regulated={regulated.collision_regulated}, error={error:.3e}")
        if not regulated.collision_regulated:
            raise AssertionError(f"{channel.name} did not activate the collision regulator")
        require_close(
            regulated.value,
            direct.value,
            3.0e-12,
            f"{channel.name} repeated-weight recursion is wrong",
        )


def check_rank_three_vacuum_seed() -> None:
    q_values = (0.020, 0.021, 0.022, 0.023, 0.024, 0.025)
    value = genus3_tetrahedral_vacuum_seed_schottky(
        q_values,
        max_word_len=2,
        oscillator_level_max=12,
    )
    print("\nrank-three Schottky seed")
    print(f"  value={value!r}, |value-1|={abs(value - 1.0):.3e}")
    if not (abs(value) < float("inf")):
        raise AssertionError("rank-three Schottky vacuum seed is not finite")
    if abs(value - 1.0) < 1.0e-12:
        raise AssertionError("rank-three vacuum smoke test was numerically trivial")


def check_rank_reducing_vacuum_degeneration() -> None:
    """Pinch q12 and compare the rank-three seed with its rank-two limit."""

    base = (0.020, 0.021, 0.022, 0.0, 0.024, 0.025)
    epsilon_values = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
    errors: list[float] = []
    pinched_multipliers: list[float] = []
    for epsilon in epsilon_values:
        q_values = base[:3] + (epsilon,) + base[4:]
        generators = generators_for_tetrahedron(q_values)
        rank_three_result = schottky_vacuum_block(
            generators,
            max_word_length=3,
            max_mode=12,
            channel="genus3_tetrahedral",
        )
        if rank_three_result.truncation_certified:
            raise AssertionError(
                "finite-word Schottky seed falsely reports a certified tail"
            )
        if (
            rank_three_result.primitive_word_tail_estimate is not None
            and (
                not math.isfinite(
                    rank_three_result.primitive_word_tail_estimate
                )
                or rank_three_result.primitive_word_tail_estimate < 0.0
            )
        ):
            raise AssertionError(
                "finite-word Schottky seed returned an invalid empirical tail"
            )
        rank_three = rank_three_result.value
        rank_two = schottky_vacuum_block(
            generators[1:],
            max_word_length=3,
            max_mode=12,
            channel="genus2_normalization",
        ).value
        errors.append(abs(rank_three - rank_two))
        pinched_multipliers.append(abs(generators[0].multiplier))
    print("\nK4 nonseparating vacuum degeneration q12 -> 0")
    for epsilon, multiplier, error in zip(epsilon_values, pinched_multipliers, errors):
        print(f"  q12={epsilon:.1e}, |k_pinched|={multiplier:.3e}, rank3/rank2 error={error:.3e}")
    if not all(right < left for left, right in zip(errors, errors[1:])):
        raise AssertionError("rank-three Schottky seed does not approach the rank-two seed monotonically")
    if errors[-1] > 1.0e-12:
        raise AssertionError("rank-reducing Schottky degeneration did not reach the expected accuracy")


def run() -> None:
    check_global_blocks()
    check_genus3_global_regular_terms_order_ten()
    check_graph_residues()
    check_specialized_recursions()
    check_k4_against_direct_sum()
    check_k4_confluent_pole_regulator()
    check_all_genus3_confluent_pole_regulators()
    check_rank_three_vacuum_seed()
    check_rank_reducing_vacuum_degeneration()
    print("\nall graph-generic CCY recursion checks passed")


if __name__ == "__main__":
    run()
