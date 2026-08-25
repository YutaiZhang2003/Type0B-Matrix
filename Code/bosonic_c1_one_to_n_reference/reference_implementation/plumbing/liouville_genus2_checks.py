#!/usr/bin/env python3
"""Checks for the genus-two Liouville pair-of-tori layer."""

from __future__ import annotations

import cmath

try:
    from liouville_genus2 import (
        bridge_primary_sewing_exponent,
        bridge_primary_sewing_factor,
        liouville_genus2_pair_of_tori,
    )
    from liouville_genus2_modular_check import liouville_genus2_sp4_generator_suite
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.liouville_genus2 import (
        bridge_primary_sewing_exponent,
        bridge_primary_sewing_factor,
        liouville_genus2_pair_of_tori,
    )
    from plumbing.liouville_genus2_modular_check import liouville_genus2_sp4_generator_suite


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def relative_difference(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def check_bridge_exponent() -> None:
    p = 0.37
    exponent = bridge_primary_sewing_exponent(b=0.8, bridge_momentum=p)
    expected = p * p - 1.0 / 24.0
    factor = bridge_primary_sewing_factor(0.013 * cmath.exp(0.4j), b=0.8, bridge_momentum=p)
    expected_factor = 0.013 ** (2.0 * expected)

    print("bridge primary factor")
    print(f"  exponent:        {exponent:.12e}")
    print(f"  expected:        {expected:.12e}")
    print(f"  factor error:    {abs(factor - expected_factor):.6e}")
    require(abs(exponent - expected) < 1.0e-14, "unexpected bridge primary exponent")
    require(abs(factor - expected_factor) < 1.0e-14, "unexpected bridge primary factor")


def check_q1_q2_symmetry() -> None:
    kwargs = dict(
        b=0.8,
        q1=0.018 * cmath.exp(0.15j),
        q2=0.014 * cmath.exp(-0.11j),
        q_bridge=0.009,
        block_order=1,
        bridge_p_max=1.5,
        handle_p_max=1.5,
        bridge_quadrature_order=2,
        handle_quadrature_order=3,
        dps=20,
    )
    direct = liouville_genus2_pair_of_tori(**kwargs)
    swapped = liouville_genus2_pair_of_tori(**{**kwargs, "q1": kwargs["q2"], "q2": kwargs["q1"]})
    rel = relative_difference(direct.value, swapped.value)

    print("\nq1/q2 symmetry")
    print(f"  direct:  {direct.value!r}")
    print(f"  swapped: {swapped.value!r}")
    print(f"  relative difference: {rel:.6e}")
    require(rel < 1.0e-12, "pair-of-tori result should be symmetric under q1 <-> q2")


def check_quadrature_refinement() -> None:
    common = dict(
        b=0.8,
        q1=0.016,
        q2=0.012,
        q_bridge=0.007,
        block_order=1,
        bridge_p_max=1.7,
        handle_p_max=1.7,
        dps=16,
    )
    coarse = liouville_genus2_pair_of_tori(
        **common,
        bridge_quadrature_order=7,
        handle_quadrature_order=8,
    )
    fine = liouville_genus2_pair_of_tori(
        **common,
        bridge_quadrature_order=8,
        handle_quadrature_order=9,
    )
    rel = relative_difference(coarse.value, fine.value)

    print("\nquadrature refinement")
    print(f"  coarse: {coarse.value!r}")
    print(f"  fine:   {fine.value!r}")
    print(f"  relative difference: {rel:.6e}")
    require(abs(fine.value) > 0, "genus-two sample unexpectedly vanished")
    require(rel < 1.0e-2, "sample is not stable under this small quadrature refinement")


def check_sp4_generator_honest_original_chart_suite() -> None:
    suite = liouville_genus2_sp4_generator_suite(
        b=0.8,
        q1=0.08 * cmath.exp(0.1j),
        q2=0.07 * cmath.exp(-0.05j),
        q_bridge=0.03 * cmath.exp(0.2j),
        expected_law="chiral-section",
        block_order=1,
        bridge_p_max=1.5,
        handle_p_max=1.5,
        bridge_quadrature_order=2,
        handle_quadrature_order=3,
        dps=16,
        plumbing_word_len=4,
        plumbing_b_order=180,
        inverse_max_nfev=40,
        chart_residual_tolerance=1.0e-5,
        modular_relative_tolerance=5.0e-3,
        use_direct_chart_action=False,
        target_chart="original",
    )

    print("\nSp(4,Z) generator honest original-chart suite")
    for result in suite.results:
        print(
            f"  {result.transform.name}: "
            f"DeltaOmega={result.plumbing.max_abs_residual:.3e}, "
            f"rel={result.relative_error:.3e}, "
            f"ok={result.modular_error_ok}, "
            f"source={result.plumbing.source}"
        )
    require(len(suite.bookkeeping_only) == 0, "honest suite should not use bookkeeping-only checks")
    require(len(suite.modular_failures) == 0, "a generator with a valid chart inverse had a modular mismatch")


def run() -> None:
    check_bridge_exponent()
    check_q1_q2_symmetry()
    check_quadrature_refinement()
    check_sp4_generator_honest_original_chart_suite()
    print("\nall Liouville genus-two smoke checks completed")


if __name__ == "__main__":
    run()
