#!/usr/bin/env python3
"""Regression tests for the configurable Ramond branching target driver."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "compute_target.py"


class ComputeTargetTest(unittest.TestCase):
    def run_target(self, *arguments: str):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            process = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), *arguments, "--json", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text()) if output.exists() else None
        return process, payload

    def test_default_target_regression(self):
        process, payload = self.run_target()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["convention"], "Human Notes/SCblock.tex only")
        self.assertFalse(payload["external_ramond_convention_conversion_used"])
        self.assertFalse(payload["pbw_double_virasoro_match_certified"])
        self.assertEqual(payload["target"], {"n1": "2", "n2": "7/4", "n3": "5/4"})
        self.assertEqual(payload["ward_label_sets"]["n1"], ["0", "1", "2"])
        self.assertEqual(
            payload["ward_label_sets"]["n2"],
            ["-1/4", "3/4", "7/4"],
        )
        self.assertEqual(
            payload["ward_label_sets"]["n3"],
            ["-3/4", "1/4", "5/4"],
        )
        # Numerical convention claims are deliberately excluded here until
        # the independent PBW/double-Virasoro comparison is certified.  This
        # test covers only the recursion's internal algebra and output schema.
        for result in payload["results"]:
            for evaluation in result["evaluations"]:
                value = evaluation["B_from_recursion"]
                self.assertIsInstance(value["real"], float)
                self.assertIsInstance(value["imag"], float)

    def test_new_interior_target(self):
        process, payload = self.run_target(
            "--n1", "1", "--n2", "3/4", "--n3", "3/4"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["target"], {"n1": "1", "n2": "3/4", "n3": "3/4"})
        for result in payload["results"]:
            self.assertEqual(result["interior_nodes"], 1)
            self.assertEqual(result["boundary_terms"], 3)
        self.assertLess(payload["maximum_recursion_vs_Ward_relative_disagreement"], 1e-10)

    def test_boundary_target(self):
        process, payload = self.run_target(
            "--n1", "2", "--n2", "3/4", "--n3", "1/4"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(payload["passed"])
        self.assertIsNone(payload["minimum_recursion_denominator_absolute_value"])
        self.assertTrue(all(result["interior_nodes"] == 0 for result in payload["results"]))

    def test_negative_ramond_reflection_component(self):
        process, payload = self.run_target(
            "--n1", "0", "--n2=-7/4", "--n3", "1/4"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(payload["passed"])
        self.assertEqual(
            payload["ward_label_sets"]["n2"], ["-7/4", "-3/4", "1/4"]
        )

    def test_multiprecision_decomposition_and_ward_solve(self):
        process, payload = self.run_target(
            "--n1", "1", "--n2", "3/4", "--n3", "3/4", "--mp-dps", "50"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(payload["passed"])
        self.assertEqual(
            payload["arithmetic"], {"backend": "mpmath", "decimal_digits": 50}
        )
        self.assertLess(payload["maximum_decomposition_relative_residual"], 1e-35)
        self.assertLess(payload["maximum_finite_Ward_relative_residual"], 1e-35)
        solvers = {
            check["solver"]
            for sector in payload["decomposition_checks"].values()
            for check in sector.values()
        }
        self.assertEqual(solvers, {"mixed-precision-pivoted-refinement-50dps"})

    def test_rejects_unsupported_half_integral_ns_component(self):
        process, payload = self.run_target(
            "--n1", "3/2", "--n2", "3/4", "--n3", "3/4"
        )
        self.assertEqual(process.returncode, 2)
        self.assertIsNone(payload)
        self.assertIn("n1 must be a nonnegative integer", process.stderr)


if __name__ == "__main__":
    unittest.main()
