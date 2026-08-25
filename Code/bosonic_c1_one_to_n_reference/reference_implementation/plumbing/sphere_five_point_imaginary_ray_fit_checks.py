#!/usr/bin/env python3
"""Regression checks for the target-free 1->4 imaginary-ray fit workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from sphere_five_point_imaginary_ray_fit import (
    FIT_STATUS,
    POINTS_STATUS,
    freeze_fit,
    freeze_points,
    sha256,
)
from sphere_five_point_matrix_comparison import compare


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_exact_quadratic_recovery() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        audit = directory / "worldsheet_audit.json"
        points = []
        for t in (0.10, 0.20, 0.30, 0.40, 0.48, 0.49):
            points.append(
                {
                    "t": t,
                    "Q": 1.3 - 2.1 * t + 0.7 * t**2,
                    "Q_standard_error": 1.0e-3,
                    "block_order": 6,
                    "sobol_power": 8,
                    "contour": "real" if t < 0.4 else "continued",
                    "residue_status": "included" if t >= 0.4 else "inactive",
                }
            )
        audit.write_text(json.dumps({"points": points}) + "\n")
        source = directory / "worldsheet_points.json"
        extracted = freeze_points(audit, source)
        _require(extracted["status"] == POINTS_STATUS, "wrong points status")
        output = directory / "fit.json"
        result = freeze_fit(source, output)
        _require(result["status"] == FIT_STATUS, "wrong frozen fit status")
        _require(
            np.allclose(
                result["primary_fit"]["coefficients_in_t"],
                [1.3, -2.1, 0.7],
                rtol=0.0,
                atol=2.0e-13,
            ),
            "quadratic coefficients were not recovered",
        )
        _require(
            len(result["diagnostic_points_excluded_from_primary_fit"]) == 1,
            "the t=0.49 diagnostic was not separated",
        )


def check_production_hash_and_fit() -> None:
    base = Path(__file__).parent / "results" / "sphere_five_point_1to4"
    source = base / "worldsheet_imaginary_ray_points_frozen.json"
    frozen_path = base / "worldsheet_imaginary_ray_fit_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    _require(frozen["status"] == FIT_STATUS, "production fit is not frozen")
    _require(
        frozen["source_worldsheet_points_sha256"] == sha256(source),
        "production source hash is stale",
    )
    _require(len(frozen["primary_points"]) == 17, "primary fit must use 17 points")
    _require(
        len(frozen["diagnostic_points_excluded_from_primary_fit"]) == 1,
        "exactly one near-wall diagnostic is expected",
    )
    expected = np.asarray([2.005754579858613, -12.032272580609972, 16.03986983226631])
    actual = np.asarray(frozen["primary_fit"]["coefficients_in_t"])
    _require(np.allclose(actual, expected, rtol=0.0, atol=2.0e-13), "fit drifted")


def check_comparison_guard_and_output() -> None:
    base = Path(__file__).parent
    results = base / "results" / "sphere_five_point_1to4"
    old_fit = results / "worldsheet_fit_frozen.json"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        try:
            compare(old_fit, directory / "rejected.json", directory / "rejected.png")
        except ValueError as error:
            _require("imaginary-ray fit" in str(error), "wrong comparison rejection")
        else:
            raise AssertionError("old convergent-only fit was accepted")

        result = compare(
            results / "worldsheet_imaginary_ray_fit_frozen.json",
            directory / "comparison.json",
            directory / "comparison.png",
        )
        _require(
            result["comparison_domain"] == "omega=i*t",
            "comparison left the imaginary-energy domain",
        )
        _require(
            not result["direct_physical_iepsilon_claimed"],
            "comparison incorrectly claimed a physical-i-epsilon result",
        )
        _require((directory / "comparison.png").stat().st_size > 10_000, "plot missing")


def main() -> None:
    check_exact_quadratic_recovery()
    check_production_hash_and_fit()
    check_comparison_guard_and_output()
    print("PASS target-free imaginary-ray quadratic fit and downstream comparison")


if __name__ == "__main__":
    main()
