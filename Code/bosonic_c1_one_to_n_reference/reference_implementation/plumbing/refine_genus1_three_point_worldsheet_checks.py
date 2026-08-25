#!/usr/bin/env python3
"""Cheap checks for the restartable three-point refinement driver."""

from __future__ import annotations

import math

import numpy as np

try:
    from refine_genus1_three_point_worldsheet import (
        combine_saved_prefix,
        mean_and_standard_error,
        ordered_gap_importance,
        tail_height_and_jacobian,
    )
except ImportError:  # pragma: no cover
    from plumbing.refine_genus1_three_point_worldsheet import (
        combine_saved_prefix,
        mean_and_standard_error,
        ordered_gap_importance,
        tail_height_and_jacobian,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_nested_prefix_reuse() -> None:
    values = np.arange(1.0, 17.0) + 1.0j * np.arange(17.0, 33.0)
    saved = complex(np.mean(values[:4]))
    continued = combine_saved_prefix(
        saved,
        values[4:],
        previous_power=2,
        final_power=4,
    )
    require(abs(continued[2] - np.mean(values[:4])) < 1.0e-14, "saved prefix changed")
    require(abs(continued[3] - np.mean(values[:8])) < 1.0e-14, "level-three mean is wrong")
    require(abs(continued[4] - np.mean(values)) < 1.0e-14, "final nested mean is wrong")
    print("nested saved-prefix continuation: passed")


def check_tail_map() -> None:
    start = 8.0
    exponent = 1.5
    for coordinate in (0.0, 0.25, 0.75, 0.99):
        height, jacobian = tail_height_and_jacobian(
            coordinate,
            tail_start=start,
            proposal_exponent=exponent,
        )
        expected_height = start * (1.0 - coordinate) ** -2.0
        expected_jacobian = 2.0 * start * (1.0 - coordinate) ** -3.0
        require(math.isclose(height, expected_height, rel_tol=1.0e-14), "tail height is wrong")
        require(math.isclose(jacobian, expected_jacobian, rel_tol=1.0e-14), "tail Jacobian is wrong")
    print("exact infinite-tail map: passed")


def check_complex_standard_error() -> None:
    mean, error = mean_and_standard_error([1.0 + 2.0j, 3.0 + 6.0j])
    require(mean == 2.0 + 4.0j, "complex mean is wrong")
    require(math.isclose(error.real, 1.0), "real standard error is wrong")
    require(math.isclose(error.imag, 2.0), "imaginary standard error is wrong")
    print("complex replicate standard error: passed")


def check_gap_importance() -> None:
    first, second, weight = ordered_gap_importance(0.37, 0.61, alpha=1.0)
    require(0.0 < first < second < 1.0, "ordered gaps left the simplex")
    require(math.isclose(weight, 1.0, rel_tol=2.0e-14), "uniform gap weight is not one")
    _, _, boundary_weight = ordered_gap_importance(0.001, 0.999, alpha=0.5)
    require(boundary_weight > 0.0 and math.isfinite(boundary_weight), "gap weight is invalid")
    print("ordered Dirichlet gap importance: passed")


def run() -> None:
    check_nested_prefix_reuse()
    check_tail_map()
    check_complex_standard_error()
    check_gap_importance()
    print("all three-point refinement checks passed")


if __name__ == "__main__":
    run()
