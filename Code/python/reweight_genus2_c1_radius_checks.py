#!/usr/bin/env python3
"""Self-checks for the genus-two compact-radius reweighting."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    from reweight_genus2_c1_radius import (
        DEFAULT_INPUT,
        c1_matrix_model_genus2_coefficient,
        conditional_strict_bry_target,
        evaluate_radius_sweep,
        load_refined_nodes,
        paired_shape_jackknife,
    )
except ImportError:  # pragma: no cover
    from plumbing.reweight_genus2_c1_radius import (
        DEFAULT_INPUT,
        c1_matrix_model_genus2_coefficient,
        conditional_strict_bry_target,
        evaluate_radius_sweep,
        load_refined_nodes,
        paired_shape_jackknife,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_checks(input_csv: Path = DEFAULT_INPUT) -> None:
    require(
        math.isclose(c1_matrix_model_genus2_coefficient(1.0), 1.0 / 240.0),
        "self-dual matrix-model coefficient is not 1/240",
    )
    require(
        math.isclose(conditional_strict_bry_target(1.0), math.pi**2 / 15.0),
        "string-note self-dual target is not pi^2/15",
    )
    for radius in (0.5, 0.8, 1.25, 2.0):
        inverse = 1.0 / radius
        require(
            math.isclose(
                conditional_strict_bry_target(radius),
                conditional_strict_bry_target(inverse) / radius**2,
                rel_tol=2.0e-15,
            ),
            f"matrix target fails T-duality at R={radius}",
        )

    base = np.array([1.0, 2.0, 4.0, 8.0])
    scaled = 3.0 * base
    shape, shape_se, mismatch, mismatch_se = paired_shape_jackknife(
        scaled,
        base,
        target_shape=3.0,
    )
    require(math.isclose(shape, 3.0), "paired shape estimator failed")
    require(shape_se < 1.0e-14, "paired shape should have zero jackknife error")
    require(math.isclose(mismatch, 1.0), "constant synthetic mismatch should be one")
    require(mismatch_se < 1.0e-14, "synthetic mismatch should have zero jackknife error")

    nodes = load_refined_nodes(input_csv)
    results, diagnostics, values = evaluate_radius_sweep(
        nodes,
        (0.5, 0.8, 1.0, 1.25, 2.0),
        lattice_tolerance=1.0e-13,
    )
    radius_one = next(row for row in results if row.radius == 1.0)
    require(
        math.isclose(
            radius_one.local_moduli_integral,
            2.552217694416e-4 / math.pi,
            rel_tol=2.0e-12,
        ),
        "R=1 reweighting does not reproduce the string-note-normalized pilot",
    )
    require(
        diagnostics["maximum_recomputed_vs_saved_radius_one_winding_relative_error"]
        < 2.0e-12,
        "recomputed R=1 winding sum disagrees with the saved sample",
    )
    require(
        diagnostics["maximum_nodewise_t_duality_relative_residual"] < 2.0e-11,
        "nodewise compact-boson reweighting fails T-duality",
    )
    require(
        np.max(np.abs(values[0.5] / (4.0 * values[2.0]) - 1.0)) < 2.0e-11,
        "R=1/2 and R=2 node values violate T-duality",
    )
    require(
        np.max(np.abs(values[0.8] / (values[1.25] / 0.8**2) - 1.0)) < 2.0e-11,
        "R=0.8 and R=1.25 node values violate T-duality",
    )
    print("Genus-two compact-radius reweighting checks passed.")


if __name__ == "__main__":
    run_checks()
