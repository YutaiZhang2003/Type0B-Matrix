#!/usr/bin/env python3
"""Fast checks for the genus-one adaptive Liouville momentum audit."""

from __future__ import annotations

import math

import numpy as np

try:
    from genus1_two_point_adaptive_momentum import (
        AuditPoint,
        local_momentum_rules,
        local_polar_momentum_rule,
    )
    from genus1_two_point_worldsheet import MomentumRule
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_adaptive_momentum import (
        AuditPoint,
        local_momentum_rules,
        local_polar_momentum_rule,
    )
    from plumbing.genus1_two_point_worldsheet import MomentumRule


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    q_abs = 0.37
    decay = -2.0 * math.log(q_abs)

    threshold = MomentumRule.threshold_gaussian(q_abs, 4)
    threshold_value = float(
        np.dot(
            threshold.weights,
            threshold.nodes**2 * np.exp(-decay * threshold.nodes**2),
        )
    )
    threshold_exact = math.sqrt(math.pi) / (4.0 * decay**1.5)
    require(
        abs(threshold_value / threshold_exact - 1.0) < 2.0e-14,
        "threshold-Gaussian normalization is incorrect",
    )

    primary = MomentumRule.primary_gaussian(q_abs, 4)
    primary_value = float(
        np.dot(primary.weights, np.exp(-decay * primary.nodes**2))
    )
    primary_exact = math.sqrt(math.pi) / (2.0 * math.sqrt(decay))
    require(
        abs(primary_value / primary_exact - 1.0) < 2.0e-14,
        "primary-Gaussian normalization is incorrect",
    )

    tau = 0.17 + 1.08j
    necklace = AuditPoint("neck", "necklace", 0.8 + 0.5j, tau)
    necklace_rules = local_momentum_rules(necklace, 4)
    require(
        all(rule.kind == "threshold-gaussian" for rule in necklace_rules),
        "ordinary necklace edges must use threshold rules",
    )
    require(
        necklace_rules[0].q_abs > necklace_rules[1].q_abs,
        "the two local necklace widths were not distinguished",
    )

    disc = AuditPoint(
        "disc", "collision-disc", 0.0j, tau, collision_radius=0.1
    )
    disc_rules = local_momentum_rules(disc, 4)
    require(
        disc_rules[0].kind == "threshold-gaussian"
        and disc_rules[1].kind == "primary-gaussian",
        "the collision-disc threshold cancellation was not applied",
    )

    polar = local_polar_momentum_rule(necklace, 4, 5)
    first_decay = 1.0 / polar.first_gaussian_width**2
    second_decay = 1.0 / polar.second_gaussian_width**2
    polar_value = float(
        np.dot(
            polar.weights,
            polar.first_nodes**2
            * polar.second_nodes**2
            * np.exp(
                -first_decay * polar.first_nodes**2
                - second_decay * polar.second_nodes**2
            ),
        )
    )
    polar_exact = (
        math.sqrt(math.pi)
        / (4.0 * first_decay**1.5)
        * math.sqrt(math.pi)
        / (4.0 * second_decay**1.5)
    )
    require(
        abs(polar_value / polar_exact - 1.0) < 3.0e-14,
        "threshold-polar normalization is incorrect",
    )

    disc_polar = local_polar_momentum_rule(disc, 4, 5)
    disc_first_decay = 1.0 / disc_polar.first_gaussian_width**2
    disc_second_decay = 1.0 / disc_polar.second_gaussian_width**2
    disc_polar_value = float(
        np.dot(
            disc_polar.weights,
            disc_polar.first_nodes**2
            * np.exp(
                -disc_first_decay * disc_polar.first_nodes**2
                - disc_second_decay * disc_polar.second_nodes**2
            ),
        )
    )
    disc_polar_exact = (
        math.sqrt(math.pi)
        / (4.0 * disc_first_decay**1.5)
        * math.sqrt(math.pi)
        / (2.0 * math.sqrt(disc_second_decay))
    )
    require(
        abs(disc_polar_value / disc_polar_exact - 1.0) < 3.0e-14,
        "threshold-primary polar normalization is incorrect",
    )

    invalid_ope = AuditPoint("bad", "ope", 0.0 + 0.9j, tau)
    try:
        local_momentum_rules(invalid_ope, 4)
    except ValueError as exc:
        require("|v|<1" in str(exc), "unexpected invalid-OPE diagnostic")
    else:
        raise AssertionError("an OPE point with |v|>1 was accepted")

    print("genus-one two-point adaptive momentum checks: PASS")
    print(f"threshold analytic relative error = {threshold_value/threshold_exact-1:+.3e}")
    print(f"primary analytic relative error = {primary_value/primary_exact-1:+.3e}")
    print(f"polar analytic relative error = {polar_value/polar_exact-1:+.3e}")
    print(
        "disc polar analytic relative error = "
        f"{disc_polar_value/disc_polar_exact-1:+.3e}"
    )


if __name__ == "__main__":
    main()
