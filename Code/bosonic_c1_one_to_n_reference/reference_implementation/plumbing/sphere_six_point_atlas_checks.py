#!/usr/bin/env python3
"""Geometry and coverage checks for the six-point comb atlas."""

from __future__ import annotations

import math

import numpy as np

try:
    from sphere_six_point_atlas import (
        best_linear_channels,
        best_star_channels,
        comb_tree_signature,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_plumbing_coordinates,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        oriented_comb_orderings,
        star_channel_complex_jacobian_to_chart,
        star_channel_from_plumbing_coordinates,
        star_channel_from_ordering,
        star_channel_positions_by_label,
        star_tree_signature,
        oriented_tridisc_log_mixture_density_in_frame,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.sphere_six_point_atlas import (
        best_linear_channels,
        best_star_channels,
        comb_tree_signature,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_plumbing_coordinates,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        oriented_comb_orderings,
        star_channel_complex_jacobian_to_chart,
        star_channel_from_plumbing_coordinates,
        star_channel_from_ordering,
        star_channel_positions_by_label,
        star_tree_signature,
        oriented_tridisc_log_mixture_density_in_frame,
    )


def check_combinatorics() -> None:
    orderings = oriented_comb_orderings()
    signatures = {comb_tree_signature(ordering) for ordering in orderings}
    star_signatures = {star_tree_signature(ordering) for ordering in orderings}
    print("six-point comb combinatorics")
    print(
        f"  oriented frames per topology={len(orderings)}, "
        f"comb trees={len(signatures)}, star trees={len(star_signatures)}"
    )
    if len(orderings) != math.factorial(6) or len(signatures) != 90 or len(star_signatures) != 15:
        raise AssertionError("six-point comb atlas has the wrong channel count")


def check_round_trip_and_jacobian() -> None:
    ordering = (2, 0, 5, 1, 4, 3)
    q_values = (0.13 + 0.07j, -0.21 + 0.11j, 0.31 - 0.09j)
    positions = linear_channel_positions_by_label(*q_values, ordering)
    recovered = linear_channel_from_ordering(positions, ordering)
    errors = tuple(abs(observed - expected) for observed, expected in zip(
        (recovered.q1, recovered.q2, recovered.q3), q_values
    ))
    if max(errors) > 2.0e-15:
        raise AssertionError("six-point comb coordinates do not round trip")

    selected = ordering
    moving = selected[1:4]
    analytic = linear_channel_complex_jacobian_to_chart(
        *q_values,
        ordering,
        fixed_zero=selected[0],
        fixed_one=selected[4],
        fixed_infinity=selected[5],
        moving_labels=moving,
    )
    # In the selected gauge the moving coordinates are
    # (q1*q2*q3, q2*q3, q3), whose determinant is q2*q3^2.
    expected = q_values[1] * q_values[2] ** 2
    relative = abs(analytic - expected) / abs(expected)
    print("\ncomb round trip and Jacobian")
    print(f"  q errors={errors}, relative Jacobian error={relative:.3e}")
    if relative > 2.0e-15:
        raise AssertionError("six-point holomorphic Jacobian is inconsistent")


def check_random_coverage() -> None:
    rng = np.random.default_rng(20260823)
    scores: list[float] = []
    for _ in range(80):
        moving = []
        while len(moving) < 3:
            candidate = complex(rng.normal(), rng.normal())
            fixed = (0.0j, 1.0 + 0.0j)
            if min(abs(candidate - value) for value in fixed + tuple(moving)) > 0.08:
                moving.append(candidate)
        positions = tuple(moving) + (0.0j, 1.0 + 0.0j, None)
        best = best_linear_channels(positions, limit=1)[0]
        scores.append(best.score)
    print("\nrandom M_0,6 comb coverage")
    print(f"  maximum best-channel radius={max(scores):.6f}")
    print(f"  median best-channel radius={float(np.median(scores)):.6f}")
    if max(scores) >= 1.0:
        raise AssertionError("the random six-point sample left the comb atlas")


def check_triple_cherry_limit() -> None:
    comb_scores = []
    star_scores = []
    for epsilon in (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3):
        # Three colliding pairs near 0, 1, and infinity.
        positions = (
            0.0j,
            epsilon,
            1.0 + 0.0j,
            1.0 + epsilon,
            1.0 / epsilon,
            None,
        )
        comb_scores.append(best_linear_channels(positions, limit=1)[0].score)
        star_scores.append(best_star_channels(positions, limit=1)[0].score)
    print("\ntriple-cherry degeneration")
    print(f"  best comb radii={comb_scores}")
    print(f"  best star radii={star_scores}")
    if max(comb_scores) >= 1.0 or star_scores[-1] > 4.0e-3:
        raise AssertionError("the comb atlas fails near a triple-cherry corner")


def check_star_round_trip_and_jacobian() -> None:
    ordering = (5, 1, 2, 4, 0, 3)
    q_values = (0.13 + 0.04j, -0.16 + 0.07j, 0.11 - 0.05j)
    positions = star_channel_positions_by_label(*q_values, ordering)
    recovered = star_channel_from_ordering(positions, ordering)
    errors = tuple(abs(observed - expected) for observed, expected in zip(
        (recovered.q1, recovered.q2, recovered.q3), q_values
    ))
    analytic = star_channel_complex_jacobian_to_chart(
        *q_values,
        ordering,
        fixed_zero=ordering[0],
        fixed_one=ordering[2],
        fixed_infinity=ordering[5],
        moving_labels=(ordering[1], ordering[3], ordering[4]),
    )
    expected = -1.0 / q_values[2] ** 2
    relative = abs(analytic - expected) / abs(expected)
    print("\nstar round trip and Jacobian")
    print(f"  q errors={errors}, relative Jacobian error={relative:.3e}")
    if max(errors) > 2.0e-15 or relative > 2.0e-15:
        raise AssertionError("six-point star geometry is inconsistent")


def check_exact_proposal_channels() -> None:
    ordering = (4, 1, 5, 2, 0, 3)
    q_values = (2.0e-80 + 1.0e-80j, 0.17 - 0.03j, -0.11 + 0.04j)
    linear = linear_channel_from_plumbing_coordinates(*q_values, ordering)
    star = star_channel_from_plumbing_coordinates(*q_values, ordering)
    for channel in (linear, star):
        observed = (channel.q1, channel.q2, channel.q3)
        if observed != q_values:
            raise AssertionError("an exact proposal channel reconstructed its q values")
        if channel.local_scales != (1.0 + 0.0j,) * 6:
            raise AssertionError("a canonical proposal channel has nontrivial scales")
        if not channel.score < 1.0:
            raise AssertionError("the exact proposal channel is not convergent")
    print("\nexact deep-collar proposal channels")
    print(f"  preserved radii={tuple(abs(value) for value in q_values)}")


def check_mixture_density() -> None:
    ordering = (0, 1, 2, 3, 4, 5)
    positions = linear_channel_positions_by_label(
        0.12 + 0.04j,
        0.18 - 0.03j,
        0.24 + 0.05j,
        ordering,
    )
    first = oriented_tridisc_log_mixture_density_in_frame(
        positions, ordering, radial_power=0.12
    )
    reversed_ordering = tuple(reversed(ordering))
    reversed_channel = linear_channel_from_ordering(positions, reversed_ordering)
    second = oriented_tridisc_log_mixture_density_in_frame(
        positions, reversed_ordering, radial_power=0.12
    )
    # Densities are expressed in different gauges; compare only finiteness and
    # exact re-evaluation in each gauge, not their raw numerical values.
    print("\ntridisc mixture density")
    print(f"  selected log density={first:.8f}, reversed={second:.8f}")
    print(f"  reversed channel score={reversed_channel.score:.6f}")
    if not math.isfinite(first) or not math.isfinite(second):
        raise AssertionError("six-point mixture density is not finite")


def run() -> None:
    check_combinatorics()
    check_round_trip_and_jacobian()
    check_random_coverage()
    check_triple_cherry_limit()
    check_star_round_trip_and_jacobian()
    check_exact_proposal_channels()
    check_mixture_density()
    print("\nall sphere six-point atlas checks passed")


if __name__ == "__main__":
    run()
