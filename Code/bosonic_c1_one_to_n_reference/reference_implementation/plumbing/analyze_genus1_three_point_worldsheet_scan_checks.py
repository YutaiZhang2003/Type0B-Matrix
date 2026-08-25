#!/usr/bin/env python3
"""Checks for the target-free torus three-point scan shape analysis."""

from __future__ import annotations

import json
import math
import unittest

try:
    import analyze_genus1_three_point_worldsheet_scan as analysis
except ImportError:  # pragma: no cover
    from plumbing import analyze_genus1_three_point_worldsheet_scan as analysis


class ShapeAnalysisChecks(unittest.TestCase):
    def test_current_scan_shape(self) -> None:
        path = analysis.DEFAULT_SCAN_DIR / "worldsheet_scan_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = analysis.analyze_manifest(manifest)
        self.assertFalse(result["comparison_stage_present"])
        self.assertEqual(result["discrete_maximum"]["t"], 0.75)
        peak = result["local_quadratic_peak"]
        self.assertTrue(math.isclose(peak["mean_curve_t"], 0.7123593850, abs_tol=1.0e-9))
        self.assertTrue(0.70 < peak["replicate_mean_t"] < 0.73)
        self.assertTrue(result["t_cubed_reduced_curve_is_strictly_decreasing"])
        self.assertEqual(len(result["adjacent_changes"]), 9)

    def test_rejects_comparison_manifest(self) -> None:
        with self.assertRaises(ValueError):
            analysis.analyze_manifest({"comparison_stage_present": True})


if __name__ == "__main__":
    unittest.main()
