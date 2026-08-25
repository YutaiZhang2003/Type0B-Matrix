#!/usr/bin/env python3
"""Checks for the blind torus three-point imaginary-energy scan wrapper."""

from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

try:
    import run_genus1_three_point_imaginary_scan as scan
except ImportError:  # pragma: no cover
    from plumbing import run_genus1_three_point_imaginary_scan as scan


class ThreePointScanChecks(unittest.TestCase):
    def test_frozen_design(self) -> None:
        design = scan.ScanDesign()
        self.assertEqual(scan.DEFAULT_T_VALUES[0], 0.05)
        self.assertEqual(scan.DEFAULT_T_VALUES[-1], 0.95)
        self.assertEqual(len(scan.DEFAULT_T_VALUES), 10)
        self.assertEqual(design.momentum_order, 12)
        self.assertEqual(design.block_backend, "exact-c25-descendants")
        self.assertEqual((design.high_order, design.low_order), (4, 2))
        self.assertEqual((design.sobol_power, design.tail_sobol_power), (8, 8))
        self.assertEqual(design.replicates, 4)

    def test_pole_free_grid_validation(self) -> None:
        self.assertEqual(scan.parse_t_values("0.1,0.5,0.9"), (0.1, 0.5, 0.9))
        for invalid in ("0", "1", "-0.1,0.2", "0.8,0.7"):
            with self.assertRaises(ValueError):
                scan.parse_t_values(invalid)

    def test_namespace_matches_integrator(self) -> None:
        design = scan.ScanDesign()
        namespace = scan.point_namespace(0.4, Path("point.json"), design)
        self.assertTrue(math.isclose(namespace.t, 0.4))
        self.assertEqual(namespace.momentum_order, 12)
        self.assertEqual(namespace.output, "point.json")

    def test_blind_wrapper_has_no_comparison_formula(self) -> None:
        source = inspect.getsource(scan)
        self.assertNotIn("matrix_stripped_genus1_three_point", source)
        self.assertNotIn("matrix_f1_bry_normalization", source)
        self.assertNotIn("F1_MM", source)
        self.assertNotIn("worldsheet_over_matrix", source)


if __name__ == "__main__":
    unittest.main()
