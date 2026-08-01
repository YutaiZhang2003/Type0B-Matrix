#!/usr/bin/env python3
"""Checks for the Cho-Collier-Yin genus-two block implementation."""

from __future__ import annotations

import math

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        ccy_genus2_block,
        ccy_genus2_block_partial_fraction,
        ccy_residue_prefactor_for_weights,
        c_rs_from_h,
        genus2_global_sl2_block,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        rho_lminus1_triple,
    )
    from genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing_algorithms import generators_for_theta
    from virasoro_blocks import degenerate_weight, zamolodchikov_a_rs
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_genus2_block import (
        b_from_c_rs_h,
        ccy_genus2_block,
        ccy_genus2_block_partial_fraction,
        ccy_residue_prefactor_for_weights,
        c_rs_from_h,
        genus2_global_sl2_block,
        genus2_vacuum_seed_schottky,
        minus_dc_dh_times_a_rs,
        rho_lminus1_triple,
    )
    from plumbing.genus2_vacuum_blocks import schottky_vacuum_block
    from plumbing.plumbing_algorithms import generators_for_theta
    from plumbing.virasoro_blocks import degenerate_weight, zamolodchikov_a_rs


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
    q1 = 0.01 + 0.002j
    q2 = math.exp(-690.0) + 0.0j
    q3 = 0.02 - 0.001j
    seed = genus2_vacuum_seed_schottky(
        q1,
        q2,
        q3,
        max_word_len=3,
        oscillator_level_max=12,
    )
    multiplier = q1 * q3
    expected = math.prod(1.0 / (1.0 - multiplier**mode) for mode in range(2, 13))
    require(abs(seed / expected - 1.0) < 1.0e-14, "underflow vacuum seed missed the surviving handle")

    double_seed = genus2_vacuum_seed_schottky(
        math.exp(-690.0),
        math.exp(-690.0),
        q3,
        max_word_len=3,
        oscillator_level_max=12,
    )
    require(abs(double_seed - 1.0) < 1.0e-14, "double-cusp vacuum seed did not reduce to one")

    alternative_seed = genus2_vacuum_seed_schottky(
        math.exp(-690.0),
        q1,
        q3,
        max_word_len=3,
        oscillator_level_max=12,
    )
    alternative_multiplier = generators_for_theta(
        math.exp(-690.0), q1, q3
    )[1].multiplier
    alternative_expected = math.prod(
        1.0 / (1.0 - alternative_multiplier**mode) for mode in range(2, 13)
    )
    require(
        abs(alternative_seed / alternative_expected - 1.0) < 1.0e-14,
        "alternative theta cusp missed its surviving generator",
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
    partial_fraction = ccy_genus2_block_partial_fraction(
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
    value = partial_fraction.value(c, pole_tolerance=1.0e-10)
    print("\nequal-weight double-pole bookkeeping")
    print(f"  pole count={partial_fraction.pole_count}")
    print(f"  coefficient count={partial_fraction.coefficient_count}")
    print(f"  max pole order={partial_fraction.max_pole_order}")
    print(f"  value at c=25={value!r}")
    require(partial_fraction.max_pole_order >= 2, "equal h1=h2 case should generate a higher-order c-pole")
    require(abs(value) > 0.0, "collision-aware equal-weight value vanished unexpectedly")

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
    check_partial_fraction_matches_simple_recursion()
    check_order_two_finite()
    check_vacuum_seed_uses_theta_chart()
    check_vacuum_seed_stable_backend_and_cache()
    check_vacuum_seed_nonseparating_underflow_limit()
    check_collision_aware_order_four_finite()
    check_equal_weight_double_pole_bookkeeping()
    check_simplified_universal_residue_factor()
    check_resonant_residue_limit_at_c25()
    print("\nall CCY genus-two block checks passed")


if __name__ == "__main__":
    run()
