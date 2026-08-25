#!/usr/bin/env python3
"""Checks for the direct Heisenberg pair-of-pants sewing sums."""

from __future__ import annotations

import math

try:
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        heisenberg_gram_norm,
        heisenberg_three_point,
        theta_heisenberg_plumbing_partition,
    )
    from free_boson_plumbing import glasses_free_boson_product, theta_free_boson_product
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        heisenberg_gram_norm,
        heisenberg_three_point,
        theta_heisenberg_plumbing_partition,
    )
    from plumbing.free_boson_plumbing import glasses_free_boson_product, theta_free_boson_product


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def torus_oscillator_chiral(q: complex, max_mode: int = 200) -> complex:
    product = 1.0 + 0.0j
    for mode in range(1, max_mode + 1):
        product /= 1.0 - complex(q) ** mode
    return product


def check_gram_and_pants_coefficients() -> None:
    print("Heisenberg Gram and pants coefficients")
    require(heisenberg_gram_norm(()) == 1, "vacuum norm should be one")
    require(heisenberg_gram_norm((1, 1, 1)) == 6, "z_(1,1,1) should be 3!")
    require(heisenberg_gram_norm((3, 2, 2)) == 24, "unexpected mixed Heisenberg norm")
    require(heisenberg_three_point((1,), (), (1,)) == 1, "two-point reduction failed at level one")
    require(heisenberg_three_point((2,), (), (2,)) == 2, "two-point reduction failed at level two")
    require(heisenberg_three_point((), (2,), (2,)) == -6, "derivative-current contraction failed")
    require(heisenberg_three_point((2,), (2,), ()) == 2, "bra-current contraction failed")
    print("  basic coefficients passed")


def check_theta_level_two() -> None:
    q1, q2, q3 = 0.017, 0.013, 0.011
    result = theta_heisenberg_plumbing_partition(q1, q2, q3, max_total_level=2)
    expected = 1.0 + q1 * q2 + q1 * q3 + q2 * q3
    print("\nTheta level-two plumbing identity")
    print(f"  direct={result.chiral_value!r}")
    print(f"  expected={expected!r}")
    require(abs(result.chiral_value - expected) < 1.0e-15, "theta level-two coefficient is wrong")


def check_glasses_separating_factorization() -> None:
    q_left, q_right = 0.017, 0.013
    order = 8
    result = glasses_heisenberg_plumbing_partition(q_left, q_right, 0.0, max_total_level=order)
    expected = 0.0 + 0.0j
    for left_level in range(order + 1):
        for right_level in range(order + 1 - left_level):
            # The coefficient is p(n_left) p(n_right).
            from_left = sum(1 for _ in _partitions_for_check(left_level))
            from_right = sum(1 for _ in _partitions_for_check(right_level))
            expected += from_left * from_right * (q_left**left_level) * (q_right**right_level)
    print("\nGlasses separating factorization")
    print(f"  direct={result.chiral_value!r}")
    print(f"  truncated two-torus product={expected!r}")
    require(abs(result.chiral_value - expected) < 1.0e-14, "glasses q_bridge=0 factorization failed")


def _partitions_for_check(total: int, max_part: int | None = None):
    if total == 0:
        yield ()
        return
    if max_part is None:
        max_part = total
    for part in range(min(max_part, total), 0, -1):
        for tail in _partitions_for_check(total - part, part):
            yield (part,) + tail


def check_against_schottky_products() -> None:
    theta_q = (0.012 + 0.001j, 0.009 - 0.0007j, 0.007 + 0.0004j)
    glasses_q = (0.018 + 0.001j, 0.015 - 0.0008j, 0.008 + 0.0003j)
    order = 7

    theta_direct = theta_heisenberg_plumbing_partition(*theta_q, max_total_level=order)
    theta_schottky = theta_free_boson_product(*theta_q, max_word_length=10, max_mode=100)
    glasses_direct = glasses_heisenberg_plumbing_partition(*glasses_q, max_total_level=order)
    glasses_schottky = glasses_free_boson_product(*glasses_q, max_word_length=10, max_mode=100)

    theta_rel = abs(theta_direct.nonchiral_value - theta_schottky.nonchiral_value) / theta_schottky.nonchiral_value
    glasses_rel = abs(glasses_direct.nonchiral_value - glasses_schottky.nonchiral_value) / glasses_schottky.nonchiral_value
    print("\nDirect sewing versus plumbing-frame primitive-word oscillator product")
    print(f"  theta direct={theta_direct.nonchiral_value:.16e}")
    print(f"  theta oscillator product={theta_schottky.nonchiral_value:.16e}")
    print(f"  theta relative difference={theta_rel:.6e}")
    print(f"  glasses direct={glasses_direct.nonchiral_value:.16e}")
    print(f"  glasses oscillator product={glasses_schottky.nonchiral_value:.16e}")
    print(f"  glasses relative difference={glasses_rel:.6e}")
    require(theta_rel < 5.0e-10, "theta direct sewing does not match the Schottky product")
    require(glasses_rel < 5.0e-9, "glasses direct sewing does not match the Schottky product")
    theta_powered = math.expm1(
        25.0
        * abs(
            math.log(theta_direct.nonchiral_value)
            - theta_schottky.nonchiral_log_value
        )
    )
    glasses_powered = math.expm1(
        25.0
        * abs(
            math.log(glasses_direct.nonchiral_value)
            - glasses_schottky.nonchiral_log_value
        )
    )
    require(theta_powered < 5.0e-10, "theta direct sewing is inaccurate after power 25")
    require(glasses_powered < 5.0e-9, "glasses direct sewing is inaccurate after power 25")


def check_period_matched_overlap_convergence() -> None:
    glasses_q = (0.15 + 0.0j, 0.15 + 0.0j, 0.15 + 0.0j)
    theta_q = (
        0.1602332247082 - 2.777592429258e-7j,
        2.998106672314e-9 + 1.261814452401e-14j,
        4.251284599701e-9 - 1.550732812259e-14j,
    )
    direct_level = 18
    glasses_direct = glasses_heisenberg_plumbing_partition(*glasses_q, max_total_level=direct_level)
    theta_direct = theta_heisenberg_plumbing_partition(*theta_q, max_total_level=direct_level)
    glasses_schottky = glasses_free_boson_product(*glasses_q, max_word_length=10, max_mode=100)
    theta_schottky = theta_free_boson_product(*theta_q, max_word_length=10, max_mode=100)
    direct_ratio = theta_direct.nonchiral_value / glasses_direct.nonchiral_value
    schottky_ratio = theta_schottky.nonchiral_value / glasses_schottky.nonchiral_value
    relative = abs(direct_ratio - schottky_ratio) / abs(schottky_ratio)
    print("\nPeriod-matched overlap convergence")
    print(f"  direct total level={direct_level}")
    print(f"  direct theta/glasses={direct_ratio:.16e}")
    print(f"  oscillator-product theta/glasses={schottky_ratio:.16e}")
    print(f"  relative difference={relative:.6e}")
    require(relative < 5.0e-7, "direct overlap plumbing ratio has not approached Schottky resummation")
    powered_relative = math.expm1(25.0 * abs(math.log(direct_ratio / schottky_ratio)))
    print(f"  relative difference after power 25={powered_relative:.6e}")
    require(powered_relative < 2.0e-5, "direct overlap is inaccurate after power 25")


def run() -> None:
    check_gram_and_pants_coefficients()
    check_theta_level_two()
    check_glasses_separating_factorization()
    check_against_schottky_products()
    check_period_matched_overlap_convergence()
    print("\nall direct Heisenberg plumbing checks passed")


if __name__ == "__main__":
    run()
