"""Failure and provenance boundaries for native NSRR production."""

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import nsrr_cpp_backend as native


class NativeBoundaryTests(unittest.TestCase):
    def test_missing_executable_has_no_pbw_fallback(self):
        with patch.dict(os.environ, {"TYPE0B_NSRR_BINARY": "/missing/type0b/ramond"}):
            with self.assertRaisesRegex(RuntimeError, "make -C C\\+\\+"):
                native.NativeNSRR(1.4, (.2, .3, .4), 2)

    def test_failed_inserted_pipeline_is_reported_without_fallback(self):
        with patch.object(native, "executable", return_value=Path("/native/ramond")):
            runtime = native.NativeNSRR(1.4, (.2, .3, .4), 2)
        failed = subprocess.CompletedProcess([], 1, "", "Ward residual exceeded tolerance")
        with patch.object(native.subprocess, "run", return_value=failed) as run:
            with self.assertRaisesRegex(ArithmeticError, "Ward residual"):
                runtime.physical_components(0, 1, -1)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "inserted")
        self.assertEqual(command[command.index("--P1") + 1], "0,0.2")
        self.assertEqual(runtime.records, {})

    def test_missing_coefficient_is_rejected_instead_of_treated_as_zero(self):
        with self.assertRaisesRegex(ArithmeticError, "Incomplete"):
            native.physical_rows({"total_q_level": 1, "coefficients": []})

    def test_old_pbw_source_cannot_be_reused(self):
        import run_nsrr_nsnsns_offaxis_constant_scan as scan
        baseline = native.ROOT / "Data Set/nsrr_nsnsns_generic_10point_N4_20260904"
        config = scan.load(baseline / "config.json")
        config["reuse_parent_output"] = str(baseline)
        self.assertIsNone(scan.reusable_parent(config, "source", 0))
        self.assertIsNotNone(scan.reusable_parent(config, "target", 0))


if __name__ == "__main__":
    unittest.main()
