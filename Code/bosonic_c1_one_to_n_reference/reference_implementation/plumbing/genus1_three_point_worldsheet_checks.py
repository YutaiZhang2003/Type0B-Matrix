#!/usr/bin/env python3
"""Checks for the blind genus-one three-point necklace kernel."""

from __future__ import annotations

import cmath
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from genus1_three_point_worldsheet import (
        _regulated_h_recursion_coefficients,
        dedekind_eta_log_abs_squared,
        dedekind_eta_oscillator_abs_squared,
        evaluate_pade_rows,
        LiouvilleTorusThreePointNecklace,
        ordered_necklace_data,
        reduced_worldsheet_integrand_three_point,
        torus_prime_form_log_norm,
    )
    from genus1_two_point_worldsheet import MomentumRule, dedekind_eta, torus_prime_form_norm
    from torus_necklace_blocks_checks import direct_necklace_descendant_coefficients
    from torus_three_point_blocks import necklace_descendant_coefficients_three_point
    from virasoro_blocks import TorusThreePointVirasoroBlock
    from torus_two_point_blocks import (
        necklace_coefficients_in_elliptic_nomes,
        necklace_coefficients_in_elliptic_nomes_nd,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus1_three_point_worldsheet import (
        _regulated_h_recursion_coefficients,
        dedekind_eta_log_abs_squared,
        dedekind_eta_oscillator_abs_squared,
        evaluate_pade_rows,
        LiouvilleTorusThreePointNecklace,
        ordered_necklace_data,
        reduced_worldsheet_integrand_three_point,
        torus_prime_form_log_norm,
    )
    from plumbing.genus1_two_point_worldsheet import (
        MomentumRule,
        dedekind_eta,
        torus_prime_form_norm,
    )
    from plumbing.torus_necklace_blocks_checks import direct_necklace_descendant_coefficients
    from plumbing.torus_three_point_blocks import necklace_descendant_coefficients_three_point
    from plumbing.virasoro_blocks import TorusThreePointVirasoroBlock
    from plumbing.torus_two_point_blocks import (
        necklace_coefficients_in_elliptic_nomes,
        necklace_coefficients_in_elliptic_nomes_nd,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_nd_elliptic_composition() -> None:
    rng = np.random.default_rng(20260824)
    coefficients = rng.normal(size=(4, 3)) + 1.0j * rng.normal(size=(4, 3))
    old = necklace_coefficients_in_elliptic_nomes(coefficients, 3, 2)
    new = necklace_coefficients_in_elliptic_nomes_nd(coefficients, (3, 2))
    error = float(np.max(np.abs(old - new)))
    print(f"two-axis elliptic composition compatibility: {error:.3e}")
    require(error < 1.0e-13, "N-dimensional elliptic composition changed the two-point map")


def check_regulated_recursion_at_c25() -> None:
    internal = (1.02, 1.17, 1.51)
    external = (0.9775, 0.9775, 0.91)
    orders = (3, 2, 2)
    regulated = _regulated_h_recursion_coefficients(
        internal,
        external,
        orders,
        c_regulator=0.05,
    )
    exact = direct_necklace_descendant_coefficients(
        25.0,
        internal,
        external,
        orders,
    )
    error = float(np.max(np.abs(regulated - exact)))
    print(f"regulated h-recursion versus exact c=25 sewing: {error:.3e}")
    require(error < 2.0e-6, "regulated c=25 h-recursion is outside its validated window")
    production_exact = necklace_descendant_coefficients_three_point(
        25.0,
        internal,
        external,
        orders,
    )
    baseline_error = float(np.max(np.abs(production_exact - exact)))
    print(f"production exact sewing versus test baseline: {baseline_error:.3e}")
    require(baseline_error < 1.0e-13, "production descendant sewing changed the block")


def check_regulated_recursion_releases_memoized_states() -> None:
    # The recursive method uses a class-level lru_cache whose key contains the
    # block instance.  Production bank construction must clear it after every
    # independent momentum triple or a long-lived worker retains every block.
    TorusThreePointVirasoroBlock._reduced_coefficient.cache_clear()
    _regulated_h_recursion_coefficients(
        (1.02, 1.17, 1.51),
        (0.9775, 0.9775, 0.91),
        (3, 2, 2),
        c_regulator=0.05,
    )
    retained = TorusThreePointVirasoroBlock._reduced_coefficient.cache_info().currsize
    print(f"regulated h-recursion retained memoized states: {retained}")
    require(retained == 0, "regulated h-recursion leaked memoized block states")


def check_necklace_geometry() -> None:
    tau = 0.23 + 1.41j
    w1 = 2.0 * math.pi * (0.71 + 0.82 * tau)
    w2 = 2.0 * math.pi * (0.13 + 0.27 * tau)
    ordered, logs = ordered_necklace_data(w1, w2, tau)
    require(ordered == (w2, w1), "punctures were not ordered by cylinder height")
    product_q = np.prod([cmath.exp(value) for value in logs])
    expected = cmath.exp(2.0j * math.pi * tau)
    error = abs(product_q - expected)
    print(f"necklace nome product error: {error:.3e}")
    require(error < 1.0e-14, "three cylinder nomes do not multiply to the torus nome")
    require(all(value.real < 0.0 for value in logs), "a necklace cylinder is not contracting")


def check_equal_split_and_adaptive_orders() -> None:
    kernel = LiouvilleTorusThreePointNecklace(
        0.6,
        momentum_rule=MomentumRule.power_legendre(4.0, 1, 2.0),
        high_order=6,
        low_order=2,
        adaptive_tolerance=5.0e-5,
    )
    require(kernel.external_weights == (0.9775, 0.9775, 0.91), "equal-split weights are wrong")
    require(kernel.adaptive_order(0.01) == 2, "small nome should use the floor order")
    require(kernel.adaptive_order(0.10) == 4, "intermediate nome order is wrong")
    require(kernel.adaptive_order(0.25) == 6, "large nome should hit the smoke cap")
    print("equal-split kinematics and adaptive orders: passed")


def check_stable_cusp_factors() -> None:
    tau = 0.17 + 3.4j
    z = 2.0 * math.pi * (0.31 + 0.43 * tau)
    eta_ratio = (
        abs(dedekind_eta(tau)) ** 2
        * math.exp(math.pi * tau.imag / 6.0)
    )
    oscillator = dedekind_eta_oscillator_abs_squared(tau)
    log_combined = math.exp(
        dedekind_eta_log_abs_squared(tau) + math.pi * tau.imag / 6.0
    )
    require(
        abs(eta_ratio - oscillator) < 2.0e-15,
        "explicit eta-vacuum cancellation changed the eta oscillator factor",
    )
    require(
        abs(log_combined - oscillator) < 2.0e-15,
        "log-domain eta-vacuum combination changed the oscillator factor",
    )
    direct_log = math.log(torus_prime_form_norm(z, tau))
    stable_log = torus_prime_form_log_norm(z, tau)
    require(
        abs(direct_log - stable_log) < 2.0e-14,
        "prime-form log changed the compact-domain value",
    )

    deep_tau = 0.13 + 1.0e6j
    deep_z = 2.0 * math.pi * (0.27 + 0.49 * deep_tau)
    deep_log = torus_prime_form_log_norm(deep_z, deep_tau)
    require(math.isfinite(deep_log), "deep-cusp prime-form log is not finite")
    print("stable eta and prime-form cusp factors: passed")


def check_stable_reduced_integrand() -> None:
    rule = MomentumRule.power_legendre(4.0, 2, 2.0)
    kernel = LiouvilleTorusThreePointNecklace(
        0.6,
        momentum_rules=(rule, rule, rule),
        high_order=2,
        low_order=2,
        block_backend="exact-c25-descendants",
    )
    kernel.prepare()
    tau = 0.11 + 80.0j
    w1 = 2.0 * math.pi * (0.23 + 0.31 * tau)
    w2 = 2.0 * math.pi * (0.67 + 0.74 * tau)
    value = reduced_worldsheet_integrand_three_point(kernel, w1, w2, tau)
    require(
        math.isfinite(value.real) and math.isfinite(value.imag),
        "stabilized reduced integrand is not finite in the deep cusp",
    )
    print("stable deep-cusp reduced integrand: passed")


def check_vectorized_pade_rows() -> None:
    # (1+2z)/(1-z) has coefficients 1,3,3,3,... .
    rows = np.asarray([[1.0, 3.0, 3.0, 3.0], [2.0, 6.0, 6.0, 6.0]])
    value = 0.37
    evaluated = evaluate_pade_rows(rows, value, 1, 1)
    expected = np.asarray([(1.0 + 2.0 * value) / (1.0 - value), 2.0 * (1.0 + 2.0 * value) / (1.0 - value)])
    require(
        float(np.max(np.abs(evaluated - expected))) < 1.0e-13,
        "vectorized Padé evaluation changed a rational test series",
    )
    print("vectorized Padé rows: passed")


def check_restartable_bank_cache() -> None:
    rule = MomentumRule.power_legendre(3.0, 1, 2.0)
    first = LiouvilleTorusThreePointNecklace(
        0.6,
        momentum_rules=(rule, rule, rule),
        high_order=2,
        low_order=2,
        block_backend="exact-c25-descendants",
    )
    first.prepare()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "banks.npz"
        first.save_banks(path)
        second = LiouvilleTorusThreePointNecklace(
            0.6,
            momentum_rules=(rule, rule, rule),
            high_order=2,
            low_order=2,
            block_backend="exact-c25-descendants",
        )
        second.load_banks(path)
        for edge in range(3):
            require(
                np.array_equal(
                    first._banks[edge].coefficients,
                    second._banks[edge].coefficients,
                ),
                "restartable bank cache changed coefficients",
            )
        partial_path = Path(directory) / "partial_banks.npz"
        partial = LiouvilleTorusThreePointNecklace(
            0.6,
            momentum_rules=(rule, rule, rule),
            high_order=2,
            low_order=2,
            block_backend="exact-c25-descendants",
        )
        partial._banks[1] = partial._build_bank(1)
        partial.save_banks(partial_path, prepare_missing=False)
        resumed = LiouvilleTorusThreePointNecklace(
            0.6,
            momentum_rules=(rule, rule, rule),
            high_order=2,
            low_order=2,
            block_backend="exact-c25-descendants",
        )
        resumed.load_banks(partial_path)
        require(
            set(resumed._banks) == {1},
            "partial bank checkpoint did not preserve its completed edge",
        )
    print("restartable three-point bank cache: passed")


def run() -> None:
    check_nd_elliptic_composition()
    check_regulated_recursion_at_c25()
    check_regulated_recursion_releases_memoized_states()
    check_necklace_geometry()
    check_equal_split_and_adaptive_orders()
    check_stable_cusp_factors()
    check_stable_reduced_integrand()
    check_vectorized_pade_rows()
    check_restartable_bank_cache()
    print("all genus-one three-point worldsheet checks passed")


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        raise
