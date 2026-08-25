#!/usr/bin/env python3
"""Regression and trust-boundary checks for the sphere 1->5 cubic fit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sphere_six_point_imaginary_ray_fit import (
    FIT_STATUS,
    POINTS_STATUS,
    freeze_fit,
    sha256,
)
from sphere_six_point_imaginary_ray_fit_comparison import compare


BASE = Path(__file__).parent / "results" / "sphere_six_point_1to5"
POINTS = BASE / "worldsheet_imaginary_ray_points_frozen.json"
FIT = BASE / "worldsheet_imaginary_ray_cubic_fit_frozen.json"
COMPARISON = BASE / "matrix_model_fit_comparison_16point_local.json"
FIGURE = BASE / "sphere_one_to_five_amplitude_16point_fit.png"


class SphereSixPointImaginaryRayFitChecks(unittest.TestCase):
    def test_exact_target_free_cubic_recovery(self) -> None:
        coefficients = np.asarray([1.75, -2.5, 0.875, 3.25])
        t_values = np.linspace(0.11, 0.31, 12)
        values = sum(
            coefficient * t_values**power
            for power, coefficient in enumerate(coefficients)
        )
        source = {
            "status": POINTS_STATUS,
            "target_information_present": False,
            "Q5_discretization_error": 1.0e-5,
            "points": [
                {
                    "t": float(t),
                    "Q5": float(value),
                    "Q5_qmc_standard_error": 2.0e-5,
                }
                for t, value in zip(t_values, values, strict=True)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = root / "points.json"
            output = root / "fit.json"
            points.write_text(json.dumps(source) + "\n")
            result = freeze_fit(points, output)
        self.assertEqual(result["status"], FIT_STATUS)
        self.assertFalse(result["target_information_used"])
        np.testing.assert_allclose(
            result["primary_fit"]["coefficients_in_t"], coefficients, atol=2e-10
        )

    def test_production_freeze_and_fit_regression(self) -> None:
        points = json.loads(POINTS.read_text())
        fit = json.loads(FIT.read_text())
        self.assertEqual(points["status"], POINTS_STATUS)
        self.assertFalse(points["target_information_present"])
        self.assertEqual(len(points["points"]), 16)
        self.assertTrue(all(0.0 < item["t"] < 1.0 / 3.0 for item in points["points"]))
        self.assertEqual(fit["status"], FIT_STATUS)
        self.assertFalse(fit["target_information_used"])
        self.assertEqual(fit["source_worldsheet_points_sha256"], sha256(POINTS))
        np.testing.assert_allclose(
            fit["primary_fit"]["coefficients_in_t"],
            [6.331582504679437, -58.91205388826323, 164.86491188609426, -142.96916925625098],
            rtol=0.0,
            atol=2e-12,
        )

    def test_comparison_artifacts_and_guard(self) -> None:
        comparison = json.loads(COMPARISON.read_text())
        self.assertEqual(
            comparison["status"],
            "sphere_1to5_imaginary_ray_fit_compared_after_hash_verification",
        )
        self.assertFalse(comparison["direct_physical_iepsilon_claimed"])
        self.assertEqual(comparison["verified_worldsheet_points_sha256"], sha256(POINTS))
        self.assertGreater(FIGURE.stat().st_size, 10_000)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"status": "not-a-frozen-fit"}) + "\n")
            with self.assertRaisesRegex(ValueError, "frozen target-free"):
                compare(bad, Path(directory) / "out.json", Path(directory) / "out.png")


if __name__ == "__main__":
    unittest.main()
