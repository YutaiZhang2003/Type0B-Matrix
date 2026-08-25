#!/usr/bin/env python3
"""Fast geometry and conformal-frame checks for the three-point atlas."""

from __future__ import annotations

import cmath
import math
import tempfile
from pathlib import Path

try:
    from genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        canonical_loop_displacement,
        nearest_torus_displacement,
        pair_disc_to_flat_log_factor,
    )
    from genus1_three_point_worldsheet import LiouvilleTorusThreePointNecklace
    from genus1_two_point_worldsheet import MomentumRule
except ImportError:  # pragma: no cover
    from plumbing.genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        canonical_loop_displacement,
        nearest_torus_displacement,
        pair_disc_to_flat_log_factor,
    )
    from plumbing.genus1_three_point_worldsheet import LiouvilleTorusThreePointNecklace
    from plumbing.genus1_two_point_worldsheet import MomentumRule


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> None:
    tau = 0.17 + 1.08j
    z = 0.31 + 0.22j
    shifted = z + 2.0 * math.pi * (2 - 3 * tau)
    require(
        abs(nearest_torus_displacement(shifted, tau) - z) < 2.0e-13,
        "nearest pair displacement is not lattice invariant",
    )
    canonical = canonical_loop_displacement(shifted, tau)
    horizontal = (
        canonical.real
        - tau.real * canonical.imag / tau.imag
    ) / (2.0 * math.pi)
    vertical = canonical.imag / (2.0 * math.pi * tau.imag)
    require(0.0 <= horizontal < 1.0, "canonical horizontal coordinate left its cell")
    require(0.0 <= vertical < 1.0, "canonical vertical coordinate left its cell")

    delta = 0.21 + 0.08j
    weight = 0.91
    equal_log = pair_disc_to_flat_log_factor(delta, weight, weight)
    expected = -2.0 * weight * cmath.log(2.0 * cmath.sin(delta / 2.0))
    require(abs(equal_log - expected) < 2.0e-14, "equal-weight frame factor changed")

    d_a, d_b = 0.82, 0.93
    v = cmath.exp(-1.0j * delta) - 1.0
    derivative_product_form = (
        d_a * cmath.log(-1.0j * cmath.exp(-1.0j * delta))
        + d_b * cmath.log(-1.0j)
        - (d_a + d_b) * cmath.log(v)
    )
    direct = pair_disc_to_flat_log_factor(delta, d_a, d_b)
    # Branch choices can differ by a phase, but the nonchiral Weyl factor is
    # fixed by the real part.
    require(
        abs(direct.real - derivative_product_form.real) < 2.0e-14,
        "unequal-weight disc-to-flat Weyl factor is inconsistent",
    )

    rules = tuple(
        MomentumRule.power_legendre(4.0, 2, 2.0 + 0.137 * edge)
        for edge in range(3)
    )
    necklace = LiouvilleTorusThreePointNecklace(
        0.75,
        momentum_rules=rules,
        high_order=2,
        low_order=2,
        block_backend="exact-c25-descendants",
        special_dps=18,
    )
    h_dominant = LiouvilleTorusThreePointAtlas(
        necklace,
        necklace_qhat_threshold=0.99,
        necklace_second_qhat_threshold=0.99,
        ope_order=2,
        high_loop_order=2,
        low_loop_order=2,
        total_ope_order=2,
        comb_loop_order=2,
        special_dps=18,
    )
    near_collision = h_dominant.choose_channel(
        0.2 + 0.05j,
        2.0 + 1.5j,
        tau,
    )
    require(
        near_collision.channel == "necklace",
        "qhat-safe point did not remain in the dominant necklace channel",
    )
    ope_fallback = LiouvilleTorusThreePointAtlas(
        necklace,
        necklace_qhat_threshold=0.01,
        necklace_second_qhat_threshold=0.01,
        ope_order=2,
        high_loop_order=2,
        low_loop_order=2,
        total_ope_order=2,
        comb_loop_order=2,
        special_dps=18,
    )
    bad_necklace = ope_fallback.choose_channel(
        2.0 + 1.0j,
        4.0 + 3.0j,
        tau,
    )
    require(
        bad_necklace.channel != "necklace",
        "qhat-unsafe point did not fall back to c-recursion",
    )
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "atlas_banks.npz"
        first_atlas = LiouvilleTorusThreePointAtlas(
            necklace,
            ope_order=2,
            high_loop_order=2,
            low_loop_order=2,
            total_ope_order=2,
            comb_loop_order=2,
            special_dps=18,
            bank_cache_path=cache,
        )
        first_value = first_atlas.correlator_ope(
            0.2 + 0.05j,
            2.0 + 1.5j,
            tau,
            (0, 2),
            record_diagnostics=False,
        )
        second_atlas = LiouvilleTorusThreePointAtlas(
            necklace,
            ope_order=2,
            high_loop_order=2,
            low_loop_order=2,
            total_ope_order=2,
            comb_loop_order=2,
            special_dps=18,
            bank_cache_path=cache,
        )
        second_value = second_atlas.correlator_ope(
            0.2 + 0.05j,
            2.0 + 1.5j,
            tau,
            (0, 2),
            record_diagnostics=False,
        )
        require(abs(first_value - second_value) < 1.0e-16, "atlas bank cache changed the block")
    print("all genus-one three-point channel-atlas checks passed")


if __name__ == "__main__":
    run()
