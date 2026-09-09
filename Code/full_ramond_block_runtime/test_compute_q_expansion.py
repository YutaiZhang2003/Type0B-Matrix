#!/usr/bin/env python3
"""Regression tests for the production Ramond q-expansion driver."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compute_q_expansion import run  # noqa: E402


class ComputeQExpansionTest(unittest.TestCase):
    def test_multiprecision_branching_matches_direct_pbw(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run(
                2,
                Path(temporary) / "q2.json",
                check_direct_pbw=True,
                primary_parity=0,
                branching_mp_dps=40,
            )

        comparison = result["diagnostics"]["direct_pbw_comparison"]
        self.assertEqual(comparison["coefficient_count"], 112)
        self.assertLess(comparison["maximum_absolute_error"], 1.0e-12)
        for diagnostic in result["diagnostics"]["ward_systems"]:
            self.assertTrue(diagnostic["full_column_rank"])
            self.assertEqual(
                diagnostic["solver"],
                "mixed-precision-qr-refinement-40dps",
            )
            self.assertLess(diagnostic["relative_residual"], 3.0e-16)


if __name__ == "__main__":
    unittest.main()
