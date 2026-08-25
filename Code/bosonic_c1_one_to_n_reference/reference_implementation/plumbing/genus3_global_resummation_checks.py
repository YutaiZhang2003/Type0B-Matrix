#!/usr/bin/env python3
"""Regression checks for all five channel-adapted genus-three global blocks."""

from __future__ import annotations

import itertools
import math

try:
    from ccy_genus2_block import rho_lminus1_triple, sl2_descendant_norm
    from ccy_plumbing_graph import (
        ccy_genus3_channel_block,
        global_sl2_plumbing_graph_block,
    )
    from genus3_global_resummation import (
        SUPPORTED_CHANNELS,
        _one_tadpole_double_triangle_value_at_cap,
        _opposite_cycle_value_at_cap,
        _tetrahedron_value_at_cap,
        _three_tadpole_star_value_at_cap,
        _two_tadpoles_double_bridge_value_at_cap,
        genus3_channel_global_sl2_block_resummed,
    )
    from genus3_plumbing_channels import genus3_channel_by_name
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import rho_lminus1_triple, sl2_descendant_norm
    from plumbing.ccy_plumbing_graph import (
        ccy_genus3_channel_block,
        global_sl2_plumbing_graph_block,
    )
    from plumbing.genus3_global_resummation import (
        SUPPORTED_CHANNELS,
        _one_tadpole_double_triangle_value_at_cap,
        _opposite_cycle_value_at_cap,
        _tetrahedron_value_at_cap,
        _three_tadpole_star_value_at_cap,
        _two_tadpoles_double_bridge_value_at_cap,
        genus3_channel_global_sl2_block_resummed,
    )
    from plumbing.genus3_plumbing_channels import genus3_channel_by_name


def require_close(
    observed: complex,
    expected: complex,
    tolerance: float,
    message: str,
) -> None:
    relative_error = abs(complex(observed) - complex(expected)) / max(
        abs(complex(expected)),
        1.0e-300,
    )
    if relative_error > tolerance:
        raise AssertionError(
            f"{message}: relative_error={relative_error:.6e}, "
            f"observed={observed!r}, expected={expected!r}"
        )


def _brute_capped_block(
    channel_name: str,
    weights: tuple[complex, ...],
    q_values: tuple[complex, ...],
    caps: tuple[int, ...],
) -> complex:
    channel = genus3_channel_by_name(channel_name)
    total = 0.0 + 0.0j
    for levels in itertools.product(*(range(cap + 1) for cap in caps)):
        term = 1.0 + 0.0j
        for level, weight, q_value in zip(levels, weights, q_values):
            term *= q_value**level / sl2_descendant_norm(weight, level)
        for infinity_edge, one_edge, zero_edge in channel.vertex_edge_indices:
            term *= rho_lminus1_triple(
                levels[infinity_edge],
                levels[one_edge],
                levels[zero_edge],
                weights[infinity_edge],
                weights[one_edge],
                weights[zero_edge],
            )
        total += term
    return complex(total)


def check_native_factorizations() -> None:
    weights = (1.0121, 1.0361, 1.0729, 1.1156, 1.1849, 1.2704)
    q_values = (
        0.061 + 0.003j,
        -0.052 + 0.007j,
        0.043 - 0.009j,
        -0.034 - 0.005j,
        0.025 + 0.006j,
        0.016 - 0.002j,
    )
    cap = 3

    capped_evaluators = {
        "one-tadpole-double-triangle": (
            _one_tadpole_double_triangle_value_at_cap
        ),
        "opposite-double-edge-cycle": _opposite_cycle_value_at_cap,
        "three-tadpole-star": _three_tadpole_star_value_at_cap,
        "two-tadpoles-double-bridge": (
            _two_tadpoles_double_bridge_value_at_cap
        ),
    }
    for channel_name, evaluator in capped_evaluators.items():
        observed = evaluator(weights, q_values, cap)
        expected = _brute_capped_block(
            channel_name,
            weights,
            q_values,
            (cap,) * 6,
        )
        require_close(
            observed,
            expected,
            3.0e-14,
            f"{channel_name} native factorization is wrong",
        )

    tetrahedron = _tetrahedron_value_at_cap(
        weights,
        q_values,
        cap,
        hypergeometric_tolerance=1.0e-15,
    )
    tetrahedron_brute = _brute_capped_block(
        "tetrahedron",
        weights,
        q_values,
        (cap, cap, cap, 24, cap, cap),
    )
    require_close(
        tetrahedron,
        tetrahedron_brute,
        3.0e-14,
        "tetrahedron q12 hypergeometric factorization is wrong",
    )
    print("native channel factorizations agree with brute descendant sums")


def check_low_order_coefficients() -> None:
    weights = (1.0121, 1.0361, 1.0729, 1.1156, 1.1849, 1.2704)
    base_q = (
        0.24 + 0.003j,
        -0.22 + 0.04j,
        -0.06 - 0.10j,
        -0.21 - 0.09j,
        0.04 + 0.15j,
        0.10 + 0.004j,
    )
    scale = 5.0e-3
    q_values = tuple(scale * value for value in base_q)
    for channel_name in sorted(SUPPORTED_CHANNELS):
        channel = genus3_channel_by_name(channel_name)
        truncated = global_sl2_plumbing_graph_block(
            channel.graph,
            edge_weights=weights,
            q_values=q_values,
            order=4,
        )
        resummed = genus3_channel_global_sl2_block_resummed(
            channel,
            edge_weights=weights,
            q_values=q_values,
            tolerance=1.0e-12,
            minimum_cap=4,
            maximum_cap=10,
            cap_step=2,
        )
        require_close(
            resummed.value,
            truncated,
            2.0e-12,
            f"{channel_name} resummation changed low-order coefficients",
        )
    print("resummed blocks reproduce the existing low-order series")


def check_adaptive_convergence_and_ccy_wiring() -> None:
    weights = (1.0121, 1.0361, 1.0729, 1.1156, 1.1849, 1.2704)
    q_by_channel = {
        "one-tadpole-double-triangle": (
            0.061 + 0.003j,
            -0.052 + 0.007j,
            0.043 - 0.009j,
            -0.034 - 0.005j,
            0.025 + 0.006j,
            0.016 - 0.002j,
        ),
        "opposite-double-edge-cycle": (
            0.24396463832641663 + 0.0026686660918344767j,
            -0.2293003274258914 + 0.04513166761461772j,
            -0.05821459124785343 - 0.10499700764637752j,
            -0.213499793222394 - 0.09221920667014041j,
            0.03996822757326669 + 0.15486182252831288j,
            0.10201838494197259 + 0.00452919353926955j,
        ),
        "tetrahedron": (
            0.24893416058335147 - 0.012794806009197865j,
            0.009513932580501995 - 0.013290479150757319j,
            -0.011401846538748536 + 0.0035077021198496264j,
            -0.029294931991420983 - 0.019188269788211188j,
            0.004077758113582364 + 0.01305167592392002j,
            0.24337190004398992 + 0.0062253278405502935j,
        ),
        "three-tadpole-star": (
            0.061 + 0.003j,
            -0.052 + 0.007j,
            0.043 - 0.009j,
            -0.034 - 0.005j,
            0.025 + 0.006j,
            0.016 - 0.002j,
        ),
        "two-tadpoles-double-bridge": (
            0.061 + 0.003j,
            -0.052 + 0.007j,
            0.043 - 0.009j,
            -0.034 - 0.005j,
            0.025 + 0.006j,
            0.016 - 0.002j,
        ),
    }
    for channel_name, q_values in q_by_channel.items():
        loose = genus3_channel_global_sl2_block_resummed(
            channel_name,
            edge_weights=weights,
            q_values=q_values,
            tolerance=1.0e-7,
            minimum_cap=6,
            maximum_cap=26,
            cap_step=2,
        )
        tight = genus3_channel_global_sl2_block_resummed(
            channel_name,
            edge_weights=weights,
            q_values=q_values,
            tolerance=1.0e-9,
            minimum_cap=8,
            maximum_cap=26,
            cap_step=2,
        )
        require_close(
            loose.value,
            tight.value,
            2.0e-8,
            f"{channel_name} adaptive cap is unstable",
        )
        block = ccy_genus3_channel_block(
            channel=channel_name,
            central_charge=25.0,
            edge_weights=weights,
            q_values=q_values,
            order=2,
            regular_term_scheme="schottky-resummed",
            vacuum_word_len=3,
            vacuum_oscillator_level_max=12,
            global_block_tolerance=1.0e-6,
            global_block_minimum_cap=6,
            global_block_maximum_cap=24,
        )
        if block.regular_term_scheme != (
            "schottky-vacuum-times-channel-resummed-global"
        ):
            raise AssertionError("CCY recursion lost resummed-scheme provenance")
        if not math.isfinite(block.value.real) or not math.isfinite(block.value.imag):
            raise AssertionError("resummed CCY block is non-finite")
    print("adaptive convergence and CCY resummed wiring passed")


def run() -> None:
    check_native_factorizations()
    check_low_order_coefficients()
    check_adaptive_convergence_and_ccy_wiring()
    print("all genus-three global-resummation checks passed")


if __name__ == "__main__":
    run()
