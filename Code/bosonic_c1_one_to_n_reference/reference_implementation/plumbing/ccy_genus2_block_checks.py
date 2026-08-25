#!/usr/bin/env python3
"""Checks for the Cho-Collier-Yin genus-two block implementation."""

from __future__ import annotations

import cmath
import math

try:
    from ccy_genus2_block import (
        ConfluentPoleError,
        PartialFractionInC,
        b_from_c_rs_h,
        ccy_genus2_block,
        ccy_genus2_block_partial_fraction,
        ccy_residue_prefactor_for_weights,
        c_rs_from_h,
        genus2_global_sl2_block,
        genus2_global_sl2_block_resummed,
        genus2_global_sl2_block_resummed_diagnostics,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        normalized_rho_lminus1_two_edge,
        normalized_rho_lminus1_two_edge_shells,
        normalized_rho_lminus1_two_edge_table,
        rho_lminus1_triple,
    )
    from genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing_algorithms import (
        generators_for_theta,
        theta_cusp_surviving_multipliers,
    )
    from virasoro_blocks import degenerate_weight, zamolodchikov_a_rs
    from virasoro_plumbing_graph import direct_plumbing_graph_block, genus2_theta_graph
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        ConfluentPoleError,
        PartialFractionInC,
        b_from_c_rs_h,
        ccy_genus2_block,
        ccy_genus2_block_partial_fraction,
        ccy_residue_prefactor_for_weights,
        c_rs_from_h,
        genus2_global_sl2_block,
        genus2_global_sl2_block_resummed,
        genus2_global_sl2_block_resummed_diagnostics,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        normalized_rho_lminus1_two_edge,
        normalized_rho_lminus1_two_edge_shells,
        normalized_rho_lminus1_two_edge_table,
        rho_lminus1_triple,
    )
    from plumbing.genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing.plumbing_algorithms import (
        generators_for_theta,
        theta_cusp_surviving_multipliers,
    )
    from plumbing.virasoro_blocks import degenerate_weight, zamolodchikov_a_rs
    from plumbing.virasoro_plumbing_graph import direct_plumbing_graph_block, genus2_theta_graph


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_degenerate_pole_location() -> None:
    h = 0.83
    r, s = 2, 1
    b = b_from_c_rs_h(r, s, h)
    degenerate = degenerate_weight(r, s, b)
    pole_c = c_rs_from_h(r, s, h)
    print("degenerate c-pole")
    print(f"  c_({r},{s})={pole_c!r}")
    print(f"  d_({r},{s})(c_rs)-h={degenerate - h!r}")
    require(abs(degenerate - h) < 1.0e-11, "c_rs(h) does not solve d_rs(c)=h")


def check_global_torus_level_one_identity() -> None:
    h = 0.8
    d = 0.3
    rho = rho_lminus1_triple(1, 0, 1, h, d, h)
    coefficient = rho / (2.0 * h)
    expected = 1.0 + d * (d - 1.0) / (2.0 * h)
    print("\nglobal level-one identity")
    print(f"  rho/(2h)={coefficient!r}")
    print(f"  expected={expected!r}")
    require(abs(coefficient - expected) < 1.0e-12, "global rho formula failed torus level-one check")


def check_order_zero_and_one() -> None:
    c = 26.215
    h1, h2, h3 = 0.91, 0.97, 1.03
    q1, q2, q3 = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    order0 = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=0,
        include_vacuum_seed=False,
    ).value
    order1 = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=1,
        include_vacuum_seed=False,
    ).value
    global1 = genus2_global_sl2_block(h1, h2, h3, q1, q2, q3, order=1)
    print("\norder zero/one")
    print(f"  order0={order0!r}")
    print(f"  order1={order1!r}")
    print(f"  global1={global1!r}")
    require(abs(order0 - 1.0) < 1.0e-14, "order-zero block should be one without vacuum seed")
    require(abs(order1 - global1) < 1.0e-14, "order-one c-recursion should equal the global seed")


def check_resummed_global_block() -> None:
    h1, h2, h3 = 1.2, 1.7, 2.1
    q1, q2, q3 = 0.03 + 0.01j, 0.07 - 0.015j, 0.11 + 0.02j
    resummed = genus2_global_sl2_block_resummed(
        h1,
        h2,
        h3,
        q1,
        q2,
        q3,
        tolerance=1.0e-13,
    )
    high_order = genus2_global_sl2_block(h1, h2, h3, q1, q2, q3, order=28)
    recursion_seed = ccy_genus2_block(
        c=26.215,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=0,
        include_vacuum_seed=False,
        resum_global_block=True,
        global_block_tolerance=1.0e-13,
    )
    print("\nresummed theta global block")
    print(f"  resummed={resummed!r}")
    print(f"  order-28 difference={abs(resummed - high_order):.3e}")
    require(abs(resummed - high_order) < 1.0e-13, "theta 2F1 resummation changed the global block")
    diagnostics = genus2_global_sl2_block_resummed_diagnostics(
        h1,
        h2,
        h3,
        q1,
        q2,
        q3,
        tolerance=1.0e-13,
    )
    require(diagnostics.converged, "resummed theta diagnostics lost convergence")
    require(
        abs(diagnostics.value - resummed) < 1.0e-14,
        "resummed theta diagnostics changed the global block",
    )
    require(
        len(diagnostics.shell_norms) == diagnostics.outer_order_reached + 1,
        "resummed theta diagnostics lost an outer shell",
    )
    require(
        abs(recursion_seed.value - resummed) < 1.0e-14,
        "order-zero pole recursion did not use the all-level global seed",
    )
    require(recursion_seed.global_block_resummed, "resummed global-block metadata was lost")


def check_high_level_normalized_coefficient() -> None:
    h1 = 1.090547524929354
    h2 = 1.000451517923736
    h3 = 1.302204668385813
    observed = normalized_rho_lminus1_two_edge(24, 52, h1, h2, h3)
    recurrence_table = normalized_rho_lminus1_two_edge_table(
        52,
        h1,
        h2,
        h3,
    )
    recurrent = recurrence_table[24][52]
    expected = 0.7438839049647926
    print("\nhigh-level normalized coefficient")
    print(f"  observed={observed!r}")
    print(f"  recurrent={recurrent!r}")
    print(f"  expected={expected!r}")
    require(
        abs(observed - expected) < 1.0e-13,
        "cancellation corrupted the high-level normalized coefficient",
    )
    require(
        abs(recurrent - expected) < 1.0e-13,
        "two-edge recurrence corrupted the high-level normalized coefficient",
    )


def check_normalized_coefficient_recurrence() -> None:
    h1, h2, h3 = 0.73, 0.91, 1.17
    max_order = 16
    table = normalized_rho_lminus1_two_edge_table(max_order, h1, h2, h3)
    largest_error = 0.0
    largest_swap_error = 0.0
    swapped = normalized_rho_lminus1_two_edge_table(max_order, h3, h2, h1)
    for i_level in range(max_order + 1):
        for k_level in range(max_order + 1 - i_level):
            expected = normalized_rho_lminus1_two_edge(
                i_level,
                k_level,
                h1,
                h2,
                h3,
            )
            largest_error = max(
                largest_error,
                abs(table[i_level][k_level] - expected),
            )
            largest_swap_error = max(
                largest_swap_error,
                abs(table[i_level][k_level] - swapped[k_level][i_level]),
            )
    print("\nnormalized two-edge coefficient recurrence")
    print(f"  largest closed-form difference={largest_error:.3e}")
    print(f"  largest edge-swap difference={largest_swap_error:.3e}")
    require(
        largest_error < 2.0e-12,
        "two-edge coefficient recurrence disagrees with the finite descendant sum",
    )
    require(
        largest_swap_error < 2.0e-12,
        "two-edge coefficient recurrence broke h1/h3 exchange symmetry",
    )


def check_lazy_normalized_coefficient_shells() -> None:
    """The lazy triangular recurrence must reproduce the rectangular table."""

    h1, h2, h3 = 0.73, 0.91, 1.17
    max_order = 18
    table = normalized_rho_lminus1_two_edge_table(max_order, h1, h2, h3)
    shells = tuple(
        normalized_rho_lminus1_two_edge_shells(max_order, h1, h2, h3)
    )
    largest_error = 0.0
    for outer_level, shell in enumerate(shells):
        require(
            len(shell) == outer_level + 1,
            "lazy two-edge recurrence returned a malformed shell",
        )
        for i_level, value in enumerate(shell):
            k_level = outer_level - i_level
            largest_error = max(
                largest_error,
                abs(value - table[i_level][k_level]),
            )

    print("\nlazy normalized two-edge coefficient shells")
    print(f"  largest rectangular-table difference={largest_error:.3e}")
    require(
        largest_error < 2.0e-12,
        "lazy two-edge shells disagree with the established recurrence table",
    )


def check_partial_fraction_matches_simple_recursion() -> None:
    c = 26.215
    h1, h2, h3 = 0.91, 0.97, 1.03
    q1, q2, q3 = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    collision_aware = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=3,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
    ).value
    simple = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=3,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
        collision_aware=False,
    ).value
    print("\npartial-fraction/simple recursion agreement")
    print(f"  collision-aware={collision_aware!r}")
    print(f"  simple={simple!r}")
    require(abs(collision_aware - simple) < 1.0e-13, "partial-fraction recursion changed the generic result")


def check_order_two_finite() -> None:
    c = 26.215
    h1, h2, h3 = 0.91, 0.97, 1.03
    q1, q2, q3 = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    value = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=2,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
    ).value
    print("\norder two finite")
    print(f"  value={value!r}")
    require(abs(value) > 0.0, "order-two value vanished unexpectedly")
    require(abs(value) < 10.0, "order-two value is implausibly large for tiny q")


def check_vacuum_seed_uses_theta_chart() -> None:
    seed = genus2_vacuum_seed_schottky(
        0.02,
        0.018,
        0.015,
        max_word_len=3,
        oscillator_level_max=12,
    )
    print("\ntheta-chart vacuum seed")
    print(f"  seed={seed!r}")
    print(f"  |seed-1|={abs(seed - 1.0):.6e}")
    require(abs(seed - 1.0) < 1.0e-5, "CCY vacuum seed appears to use non-theta q variables")


def check_block_reverses_ccy_slots_for_vacuum_seed() -> None:
    q_geometry = (0.023 + 0.003j, 0.017 - 0.002j, 0.011 + 0.001j)
    q_ccy = (q_geometry[2], q_geometry[1], q_geometry[0])
    expected = genus2_vacuum_seed_schottky(
        *q_geometry,
        max_word_len=4,
        oscillator_level_max=16,
    )
    observed = ccy_genus2_block(
        c=25.0,
        h1=1.03,
        h2=1.02,
        h3=1.01,
        q1=q_ccy[0],
        q2=q_ccy[1],
        q3=q_ccy[2],
        order=0,
        include_vacuum_seed=True,
        vacuum_word_len=4,
        vacuum_oscillator_level_max=16,
    ).value
    wrong = genus2_vacuum_seed_schottky(
        *q_ccy,
        max_word_len=4,
        oscillator_level_max=16,
    )
    print("\nCCY-slot/vacuum-chart ordering")
    print(f"  block seed={observed!r}")
    print(f"  geometric seed={expected!r}")
    print(f"  old-order displacement={abs(observed - wrong):.6e}")
    require(
        abs(observed - expected) < 1.0e-14,
        "CCY block did not reverse infinity/one/zero slots for the Schottky seed",
    )
    require(
        abs(observed - wrong) > 1.0e-10,
        "vacuum-order regression data do not distinguish the old wiring",
    )


def check_vacuum_seed_stable_backend_and_cache() -> None:
    q_values = (
        -1.17863764843e-22 - 1.277295602141e-21j,
        0.08568814836112 - 0.3460621574772j,
        -0.001259854601576 - 0.002936379485707j,
    )
    genus2_vacuum_seed_schottky.cache_clear()
    observed = genus2_vacuum_seed_schottky(
        *q_values,
        max_word_len=6,
        oscillator_level_max=20,
    )
    repeated = genus2_vacuum_seed_schottky(
        *q_values,
        max_word_len=6,
        oscillator_level_max=20,
    )
    expected = schottky_vacuum_block(
        generators_for_theta(*q_values),
        max_word_length=6,
        max_mode=20,
    ).value
    info = genus2_vacuum_seed_schottky.cache_info()
    print("\nstable cached theta vacuum seed")
    print(f"  observed={observed!r}")
    print(f"  cache={info}")
    require(math.isfinite(abs(observed)), "extreme theta vacuum seed is not finite")
    require(abs(observed - expected) < 1.0e-14, "CCY seed bypassed the stable Schottky backend")
    require(repeated == observed and info.hits >= 1, "theta vacuum seed was not cached")


def check_vacuum_seed_nonseparating_underflow_limit() -> None:
    q_zero = 0.01 + 0.002j
    q_one = 0.013 - 0.001j
    q_infty = 0.02 - 0.001j
    pinched = math.exp(-690.0) + 0.0j

    trace_zero_one = q_zero + q_one - 1.0
    determinant_zero_one = q_zero * q_one
    root = cmath.sqrt(trace_zero_one * trace_zero_one - 4.0 * determinant_zero_one)
    eigenvalues = (
        0.5 * (trace_zero_one + root),
        0.5 * (trace_zero_one - root),
    )
    eigenvalue_large = max(eigenvalues, key=abs)
    zero_one_multiplier = determinant_zero_one / (eigenvalue_large * eigenvalue_large)

    single_pinches = (
        (
            "zero",
            (pinched, q_one, q_infty),
            generators_for_theta(pinched, q_one, q_infty)[1].multiplier,
        ),
        ("one", (q_zero, pinched, q_infty), q_zero * q_infty),
        ("infinity", (q_zero, q_one, pinched), zero_one_multiplier),
    )
    for edge_name, q_values, expected_multiplier in single_pinches:
        surviving = theta_cusp_surviving_multipliers(*q_values)
        require(
            surviving is not None and len(surviving) == 1,
            f"{edge_name}-edge pinch did not reduce to rank one",
        )
        require(
            abs(surviving[0] / expected_multiplier - 1.0) < 1.0e-14,
            f"{edge_name}-edge pinch returned the wrong surviving multiplier",
        )
        seed = genus2_vacuum_seed_schottky(
            *q_values,
            max_word_len=3,
            oscillator_level_max=12,
        )
        expected_seed = math.prod(
            1.0 / (1.0 - expected_multiplier**mode)
            for mode in range(2, 13)
        )
        require(
            abs(seed / expected_seed - 1.0) < 1.0e-14,
            f"{edge_name}-edge pinch missed the surviving vacuum character",
        )

    for finite_edge, q_values in (
        ("zero", (q_zero, pinched, pinched)),
        ("one", (pinched, q_one, pinched)),
        ("infinity", (pinched, pinched, q_infty)),
    ):
        surviving = theta_cusp_surviving_multipliers(*q_values)
        require(
            surviving == (),
            f"double pinch with finite {finite_edge} edge did not reduce to rank zero",
        )
        seed = genus2_vacuum_seed_schottky(
            *q_values,
            max_word_len=3,
            oscillator_level_max=12,
        )
        require(
            abs(seed - 1.0) < 1.0e-14,
            f"double pinch with finite {finite_edge} edge did not reduce to one",
        )

    mixed_cusp = (
        0.2102638145829 + 0.02585167970028j,
        -0.004662146956686 - 0.003851197468554j,
        -1.73659589803e-11 - 5.772614037345e-11j,
    )
    require(
        theta_cusp_surviving_multipliers(*mixed_cusp) is None,
        "a representable rank-two mixed cusp was incorrectly reduced",
    )
    generic_seed = schottky_vacuum_block(
        generators_for_theta(*mixed_cusp),
        max_word_length=6,
        max_mode=20,
    ).value
    reduced_seed = genus2_vacuum_seed_schottky(
        *mixed_cusp,
        max_word_len=6,
        oscillator_level_max=20,
    )
    require(
        abs(reduced_seed / generic_seed - 1.0) < 1.0e-14,
        "mixed-cusp vacuum seed bypassed a representable composite cycle",
    )


def check_collision_aware_order_four_finite() -> None:
    c = 26.215
    h1, h2, h3 = 1.1340649633051987, 1.1, 1.2
    q1, q2, q3 = 0.003 + 0.001j, 0.0025 - 0.0007j, 0.0012 + 0.0003j
    value = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=4,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
    ).value
    print("\ncollision-aware order four finite")
    print(f"  value={value!r}")
    require(abs(value) > 0.0, "collision-aware order-four value vanished unexpectedly")
    require(abs(value) < 10.0, "collision-aware order-four value is implausibly large for tiny q")


def check_equal_weight_double_pole_bookkeeping() -> None:
    c = 25.0
    h1, h2, h3 = 1.04, 1.04, 1.0625
    q1, q2, q3 = 0.003, 0.0025, 0.0012
    try:
        ccy_genus2_block_partial_fraction(
            h1=h1,
            h2=h2,
            h3=h3,
            q1=q1,
            q2=q2,
            q3=q3,
            order=5,
            include_vacuum_seed=True,
            vacuum_word_len=2,
            vacuum_oscillator_level_max=6,
            pole_tolerance=1.0e-10,
        )
    except ConfluentPoleError:
        pass
    else:
        raise AssertionError("generic-weight partial fraction accepted a confluent lower pole")

    result = ccy_genus2_block(
        c=c,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=q1,
        q2=q2,
        q3=q3,
        order=5,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
        pole_tolerance=1.0e-10,
    )
    direct = direct_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=c,
        edge_weights=(h1, h2, h3),
        q_values=(q1, q2, q3),
        max_total_level=5,
    )
    print("\nequal-weight confluent-pole regulator")
    print(f"  regulated={result.collision_regulated}")
    print(f"  regulator error estimate={result.collision_regulator_error:.3e}")
    print(f"  direct error={abs(result.value - direct.value):.3e}")
    print(f"  value at c=25={result.value!r}")
    require(result.collision_regulated, "equal-weight recursion did not activate its regulator")
    require(abs(result.value - direct.value) < 3.0e-12, "regulated equal-weight block disagrees with descendants")

    try:
        ccy_genus2_block(
            c=c,
            h1=h1,
            h2=h2,
            h3=h3,
            q1=q1,
            q2=q2,
            q3=q3,
            order=5,
            include_vacuum_seed=True,
            vacuum_word_len=2,
            vacuum_oscillator_level_max=6,
            collision_aware=False,
            pole_tolerance=1.0e-10,
        )
    except ZeroDivisionError:
        return
    raise AssertionError("collision-unaware recursion should fail for the equal-weight pole collision")


def check_nearby_poles_remain_distinct() -> None:
    partial = PartialFractionInC()
    partial.add_pole_coefficient(
        1.0,
        1,
        2.0,
        pole_tolerance=1.0e-10,
    )
    partial.add_pole_coefficient(
        1.0 + 5.0e-13,
        1,
        3.0,
        pole_tolerance=1.0e-10,
    )
    print("\nnearby generic poles")
    print(f"  pole count={partial.pole_count}")
    require(partial.pole_count == 2, "nearby but distinct generic poles were merged")


def check_simplified_universal_residue_factor() -> None:
    r, s, h = 3, 2, 1.37
    b = b_from_c_rs_h(r, s, h)
    direct = -(
        (24.0 * ((b * b) ** 2 - 1.0) / ((1.0 - r * r) * (b * b) ** 2 - (1.0 - s * s)))
    ) * zamolodchikov_a_rs(r, s, b)
    simplified = minus_dc_dh_times_a_rs(r, s, h)
    print("\nsimplified universal residue factor")
    print(f"  direct={direct!r}")
    print(f"  simplified={simplified!r}")
    require(abs(direct - simplified) < 1.0e-11, "simplified universal residue factor changed generic value")


def check_resonant_residue_limit_at_c25() -> None:
    prefactor = ccy_residue_prefactor_for_weights(
        5,
        1,
        4.0,
        2.0,
        4.0,
    )
    print("\nresonant residue limit at c=25")
    print(f"  prefactor={prefactor!r}")
    require(abs(prefactor - 0.07410495) < 1.0e-6, "unexpected resonant residue finite limit")

    value = ccy_genus2_block(
        c=25.0,
        h1=2.0,
        h2=2.0,
        h3=2.0,
        q1=0.003,
        q2=0.0025,
        q3=0.0012,
        order=7,
        include_vacuum_seed=True,
        vacuum_word_len=2,
        vacuum_oscillator_level_max=6,
    ).value
    print(f"  order-seven value={value!r}")
    require(abs(value) > 0.0, "resonant order-seven value vanished unexpectedly")


def run() -> None:
    check_degenerate_pole_location()
    check_global_torus_level_one_identity()
    check_order_zero_and_one()
    check_resummed_global_block()
    check_normalized_coefficient_recurrence()
    check_lazy_normalized_coefficient_shells()
    check_high_level_normalized_coefficient()
    check_partial_fraction_matches_simple_recursion()
    check_order_two_finite()
    check_vacuum_seed_uses_theta_chart()
    check_block_reverses_ccy_slots_for_vacuum_seed()
    check_vacuum_seed_stable_backend_and_cache()
    check_vacuum_seed_nonseparating_underflow_limit()
    check_collision_aware_order_four_finite()
    check_equal_weight_double_pole_bookkeeping()
    check_nearby_poles_remain_distinct()
    check_simplified_universal_residue_factor()
    check_resonant_residue_limit_at_c25()
    print("\nall CCY genus-two block checks passed")


if __name__ == "__main__":
    run()
