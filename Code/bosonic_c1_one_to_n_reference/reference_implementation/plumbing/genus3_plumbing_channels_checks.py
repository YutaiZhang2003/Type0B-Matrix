#!/usr/bin/env python3
"""Checks for the five marked genus-three plumbing channels."""

from __future__ import annotations

import math

import numpy as np

try:
    from ccy_plumbing_graph import genus3_channel_vacuum_seed_schottky
    from genus3_plumbing_channels import (
        GENUS3_CHANNEL_EDGE_NAMES,
        GENUS3_CHANNEL_NAMES,
        GENUS3_CHANNEL_Q_ENDPOINT_SLOTS,
        enumerate_cubic_adjacency_matrices,
        enumerate_genus3_channels,
        generators_for_genus3_channel,
        genus3_channel_by_name,
        genus3_channel_leading_period_slope_matrix,
        genus3_channel_transition_maps,
    )
    from genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from plumbing_algorithms import plumbing_transition
    from virasoro_plumbing_graph import genus3_tetrahedral_graph
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_plumbing_graph import genus3_channel_vacuum_seed_schottky
    from plumbing.genus3_plumbing_channels import (
        GENUS3_CHANNEL_EDGE_NAMES,
        GENUS3_CHANNEL_NAMES,
        GENUS3_CHANNEL_Q_ENDPOINT_SLOTS,
        enumerate_cubic_adjacency_matrices,
        enumerate_genus3_channels,
        generators_for_genus3_channel,
        genus3_channel_by_name,
        genus3_channel_leading_period_slope_matrix,
        genus3_channel_transition_maps,
    )
    from plumbing.genus3_plumbing_tetrahedron import generators_for_tetrahedron
    from plumbing.plumbing_algorithms import plumbing_transition
    from plumbing.virasoro_plumbing_graph import genus3_tetrahedral_graph


EXPECTED_MARKINGS = {
    "one-tadpole-double-triangle": (
        (0, 1, 4),
        (2, 3, 5),
        ((0, -1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (1, -1, 0, 0, 0, 1)),
    ),
    "opposite-double-edge-cycle": (
        (0, 1, 3),
        (2, 4, 5),
        ((0, -1, 1, 0, 0, 0), (0, 0, 0, -1, 1, 0), (1, -1, 0, -1, 0, 1)),
    ),
    "tetrahedron": (
        (0, 1, 2),
        (3, 4, 5),
        ((1, -1, 0, 1, 0, 0), (1, 0, -1, 0, 1, 0), (0, 1, -1, 0, 0, 1)),
    ),
    "three-tadpole-star": (
        (0, 1, 2),
        (3, 4, 5),
        ((0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)),
    ),
    "two-tadpoles-double-bridge": (
        (0, 1, 4),
        (2, 3, 5),
        ((0, -1, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1)),
    ),
}

EXPECTED_CCY_VERTEX_EDGE_INDICES = {
    # Each vertex tuple is in descendant-tensor order
    # (infinity, one, zero), even though geometric punctures are assigned in
    # the order (zero, one, infinity) when the channel graph is constructed.
    "one-tadpole-double-triangle": (
        (2, 1, 0),
        (4, 3, 3),
        (5, 4, 0),
        (5, 2, 1),
    ),
    "opposite-double-edge-cycle": (
        (2, 1, 0),
        (5, 4, 3),
        (4, 3, 0),
        (5, 2, 1),
    ),
    "tetrahedron": (
        (2, 1, 0),
        (4, 3, 0),
        (5, 3, 1),
        (5, 4, 2),
    ),
    "three-tadpole-star": (
        (2, 1, 0),
        (3, 3, 0),
        (4, 4, 1),
        (5, 5, 2),
    ),
    "two-tadpoles-double-bridge": (
        (2, 1, 0),
        (4, 3, 3),
        (5, 5, 0),
        (4, 2, 1),
    ),
}

EXPECTED_Q_ENDPOINT_SLOTS = {
    "one-tadpole-double-triangle": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q11": ((1, "zero"), (1, "one")),
        "q12": ((1, "infty"), (2, "one")),
        "q23": ((2, "infty"), (3, "infty")),
    },
    "opposite-double-edge-cycle": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q12_1": ((1, "zero"), (2, "one")),
        "q12_2": ((1, "one"), (2, "infty")),
        "q13": ((1, "infty"), (3, "infty")),
    },
    "tetrahedron": {
        "q01": ((0, "zero"), (1, "zero")),
        "q02": ((0, "one"), (2, "zero")),
        "q03": ((0, "infty"), (3, "zero")),
        "q12": ((1, "one"), (2, "one")),
        "q13": ((1, "infty"), (3, "one")),
        "q23": ((2, "infty"), (3, "infty")),
    },
    "three-tadpole-star": {
        "q01": ((0, "zero"), (1, "zero")),
        "q02": ((0, "one"), (2, "zero")),
        "q03": ((0, "infty"), (3, "zero")),
        "q11": ((1, "one"), (1, "infty")),
        "q22": ((2, "one"), (2, "infty")),
        "q33": ((3, "one"), (3, "infty")),
    },
    "two-tadpoles-double-bridge": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q11": ((1, "zero"), (1, "one")),
        "q13": ((1, "infty"), (3, "infty")),
        "q22": ((2, "one"), (2, "infty")),
    },
}

EXPECTED_LEADING_PERIOD_SLOPES = {
    # Rows are (Omega_11, Omega_22, Omega_33, Omega_12, Omega_13, Omega_23);
    # columns are the public named-edge order for the corresponding channel.
    "one-tadpole-double-triangle": (
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (1, 1, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
    "opposite-double-edge-cycle": (
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 1, 0),
        (1, 1, 0, 1, 0, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
    ),
    "tetrahedron": (
        (1, 1, 0, 1, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (0, 1, 1, 0, 0, 1),
        (1, 0, 0, 0, 0, 0),
        (0, -1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
    ),
    "three-tadpole-star": (
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
    "two-tadpoles-double-bridge": (
        (0, 1, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
}

EXPECTED_LEADING_PERIOD_SLOPE_RANKS = {
    "one-tadpole-double-triangle": 4,
    "opposite-double-edge-cycle": 5,
    "tetrahedron": 6,
    "three-tadpole-star": 3,
    "two-tadpoles-double-bridge": 3,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mobius_array(generator) -> np.ndarray:
    gamma = generator.gamma
    return np.asarray((gamma.a, gamma.b, gamma.c, gamma.d), dtype=np.complex128)


def _transition_array(transition) -> np.ndarray:
    return np.asarray(
        (transition.a, transition.b, transition.c, transition.d),
        dtype=np.complex128,
    )


def check_channel_enumeration_and_marking() -> None:
    matrices = enumerate_cubic_adjacency_matrices()
    channels = enumerate_genus3_channels()
    require(len(matrices) == 5, "cubic multigraph enumerator did not return five classes")
    require(
        tuple(channel.name for channel in channels) == GENUS3_CHANNEL_NAMES,
        "channel ordering does not match the genus-three note",
    )
    for channel in channels:
        degrees = tuple(
            2 * channel.adjacency[vertex][vertex]
            + sum(
                channel.adjacency[vertex][other]
                for other in range(4)
                if other != vertex
            )
            for vertex in range(4)
        )
        cycle_matrix = np.asarray(channel.cycle_matrix, dtype=int)
        require(degrees == (3, 3, 3, 3), f"{channel.name} is not trivalent")
        require(channel.genus == 3, f"{channel.name} does not have genus three")
        require(len(channel.graph.edges) == 6, f"{channel.name} does not have six edges")
        require(
            channel.edge_names == GENUS3_CHANNEL_EDGE_NAMES[channel.name],
            f"{channel.name} edge order disagrees with the note",
        )
        require(
            channel.vertex_edge_indices
            == EXPECTED_CCY_VERTEX_EDGE_INDICES[channel.name],
            (
                f"{channel.name} does not map geometric "
                "(zero, one, infinity) punctures to CCY "
                "(infinity, one, zero) tensor slots"
            ),
        )
        expected_tree, expected_chords, expected_cycles = EXPECTED_MARKINGS[channel.name]
        require(
            (
                channel.tree_edges,
                channel.chord_edges,
                channel.cycle_matrix,
            )
            == (expected_tree, expected_chords, expected_cycles),
            f"{channel.name} spanning tree or cycle marking disagrees with the note",
        )
        require(
            np.linalg.matrix_rank(cycle_matrix) == 3,
            f"{channel.name} marked cycles do not have rank three",
        )
        slope = genus3_channel_leading_period_slope_matrix(channel)
        require(
            slope == EXPECTED_LEADING_PERIOD_SLOPES[channel.name],
            f"{channel.name} logarithmic period slope changed",
        )
        require(
            np.linalg.matrix_rank(np.asarray(slope, dtype=int))
            == EXPECTED_LEADING_PERIOD_SLOPE_RANKS[channel.name],
            f"{channel.name} logarithmic period slope has the wrong rank",
        )
        require(
            len({item for vertex in channel.vertex_edge_indices for item in vertex}) <= 6,
            f"{channel.name} has an invalid edge incidence",
        )
    print("five genus-three channel definitions and markings passed")


def check_named_q_endpoint_slot_passing() -> None:
    """Lock named q's to both endpoint punctures in every channel."""

    slot_punctures = ("infty", "one", "zero")
    maximum_transition_error = 0.0
    for channel in enumerate_genus3_channels():
        expected_contract = EXPECTED_Q_ENDPOINT_SLOTS[channel.name]
        require(
            GENUS3_CHANNEL_Q_ENDPOINT_SLOTS[channel.name] == expected_contract,
            f"{channel.name} public q endpoint-slot contract changed",
        )
        actual_contract = {
            edge.name: tuple(
                (endpoint.vertex, slot_punctures[endpoint.slot])
                for endpoint in edge.endpoints
            )
            for edge in channel.graph.edges
        }
        require(
            actual_contract == expected_contract,
            f"{channel.name} graph does not realize its named q endpoint slots",
        )
        expected_vertex_q_names = tuple(
            tuple(
                channel.edge_names[edge_index]
                for edge_index in vertex_edge_indices
            )
            for vertex_edge_indices in EXPECTED_CCY_VERTEX_EDGE_INDICES[channel.name]
        )
        require(
            channel.vertex_q_names == expected_vertex_q_names,
            f"{channel.name} does not pass q names in CCY (infinity, one, zero) order",
        )

        named_q = {
            edge_name: complex(0.011 + 0.002 * edge_index, 0.0003 * (edge_index + 1))
            for edge_index, edge_name in enumerate(channel.edge_names)
        }
        reversed_named_q = dict(reversed(tuple(named_q.items())))
        transitions = genus3_channel_transition_maps(channel, reversed_named_q)
        for edge, (computed, _) in zip(channel.graph.edges, transitions):
            (source_vertex, source_puncture), (
                target_vertex,
                target_puncture,
            ) = expected_contract[edge.name]
            require(
                (source_vertex, target_vertex)
                == tuple(endpoint.vertex for endpoint in edge.endpoints),
                f"{channel.name}:{edge.name} endpoint orientation changed",
            )
            expected = plumbing_transition(
                source_puncture,
                target_puncture,
                named_q[edge.name],
            )
            transition_error = float(
                np.max(np.abs(_transition_array(computed) - _transition_array(expected)))
            )
            maximum_transition_error = max(
                maximum_transition_error,
                transition_error,
            )
    print(
        "all-channel named-q endpoint-slot passing "
        f"max error={maximum_transition_error:.3e}"
    )
    require(
        maximum_transition_error < 1.0e-15,
        "a named q was passed to the wrong plumbing transition slot",
    )


def check_k4_backward_compatibility() -> None:
    channel = genus3_channel_by_name("tetrahedron")
    require(
        channel.graph == genus3_tetrahedral_graph(),
        "new tetrahedral graph differs from the original K4 graph",
    )
    q_values = (
        0.020 + 0.001j,
        0.021 - 0.001j,
        0.022 + 0.0005j,
        0.023 - 0.0007j,
        0.024 + 0.0003j,
        0.025 - 0.0004j,
    )
    original = generators_for_tetrahedron(q_values)
    generic = generators_for_genus3_channel(channel, q_values)
    errors = tuple(
        min(
            float(np.max(np.abs(_mobius_array(left) - _mobius_array(right)))),
            float(np.max(np.abs(_mobius_array(left) + _mobius_array(right)))),
        )
        for left, right in zip(original, generic)
    )
    multiplier_errors = tuple(
        abs(left.multiplier - right.multiplier)
        for left, right in zip(original, generic)
    )
    print("K4 generator backward compatibility")
    print("  Mobius errors=" + ", ".join(f"{value:.3e}" for value in errors))
    print(
        "  multiplier errors="
        + ", ".join(f"{value:.3e}" for value in multiplier_errors)
    )
    require(max(errors) < 1.0e-14, "generic K4 generators changed the old marking")
    require(
        max(multiplier_errors) < 1.0e-14,
        "generic K4 multipliers changed the old marking",
    )


def check_deep_cusp_multiplier_stability() -> None:
    """Regress multipliers that binary64 fixed-point derivatives corrupt."""

    channel = "two-tadpoles-double-bridge"
    bundled_q = (
        2.0001660216461376e-5 - 4.1066323897417868e-6j,
        -1.1648275785310041e-4 - 2.0579681658679932e-1j,
        -1.9706248381780544e-7 - 4.8581880153370849e-4j,
        -9.9980003025209018e-5 - 1.6814218798053764e-14j,
        2.5000276934260911e-5 + 1.2252023173849550e-8j,
        9.9999999989581757e-5 + 1.4973473901858005e-14j,
    )
    bundled_reference = (
        9.999999984152233e-5 - 3.449996479126316e-14j,
        1.0000000002561921e-4 + 1.6820945662704553e-14j,
        9.999999998958176e-5 + 1.4973473901858005e-14j,
    )
    symmetric_reference = (
        -9.98001003990004e-7,
        -9.980049860418684e-4,
        1.0e-3,
    )
    failure_regression_reference = (
        -4.893142407711038e-7,
        -6.990217116456434e-4,
        7.0e-4,
    )

    def maximum_relative_error(
        q_values: tuple[complex, ...],
        reference: tuple[complex, ...],
    ) -> float:
        computed = generators_for_genus3_channel(channel, q_values)
        return max(
            abs(generator.multiplier / expected - 1.0)
            for generator, expected in zip(computed, reference)
        )

    bundled_error = maximum_relative_error(bundled_q, bundled_reference)
    symmetric_error = maximum_relative_error(
        (1.0e-3,) * 6,
        symmetric_reference,
    )
    failure_regression_error = maximum_relative_error(
        (7.0e-4,) * 6,
        failure_regression_reference,
    )
    print("deep-cusp projective multiplier stability")
    print(
        "  relative errors="
        f"{bundled_error:.3e}, {symmetric_error:.3e}, "
        f"{failure_regression_error:.3e}"
    )
    require(
        max(bundled_error, symmetric_error, failure_regression_error) < 2.0e-13,
        "deep-cusp multiplier lost projective-determinant precision",
    )


def check_all_channel_generators_and_vacuum_seeds() -> None:
    base_q = (
        0.020 + 0.001j,
        0.021 - 0.001j,
        0.022 + 0.0005j,
        0.023 - 0.0007j,
        0.024 + 0.0003j,
        0.025 - 0.0004j,
    )
    print("all-channel Schottky generators and vacuum seeds")
    for channel in enumerate_genus3_channels():
        generators = generators_for_genus3_channel(channel, base_q)
        multipliers = tuple(abs(generator.multiplier) for generator in generators)
        seed = genus3_channel_vacuum_seed_schottky(
            channel.name,
            base_q,
            max_word_len=3,
            oscillator_level_max=12,
        )
        deep_seed = genus3_channel_vacuum_seed_schottky(
            channel.name,
            tuple(0.1 * value for value in base_q),
            max_word_len=3,
            oscillator_level_max=12,
        )
        print(
            f"  {channel.name}: max|k|={max(multipliers):.3e}, "
            f"|seed-1|={abs(seed - 1.0):.3e}, "
            f"deep={abs(deep_seed - 1.0):.3e}"
        )
        require(
            all(math.isfinite(value) and 0.0 < value < 1.0 for value in multipliers),
            f"{channel.name} did not produce three loxodromic generators",
        )
        require(
            math.isfinite(seed.real) and math.isfinite(seed.imag),
            f"{channel.name} vacuum seed is nonfinite",
        )
        require(
            abs(deep_seed - 1.0) < abs(seed - 1.0),
            f"{channel.name} vacuum seed does not approach one in the deep cusp",
        )


def run() -> None:
    check_channel_enumeration_and_marking()
    check_named_q_endpoint_slot_passing()
    check_k4_backward_compatibility()
    check_deep_cusp_multiplier_stability()
    check_all_channel_generators_and_vacuum_seeds()
    print("all genus-three plumbing-channel checks passed")


if __name__ == "__main__":
    run()
