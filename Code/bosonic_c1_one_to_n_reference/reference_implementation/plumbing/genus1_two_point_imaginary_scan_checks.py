#!/usr/bin/env python3
"""Checks for the blind scan wrapper and post-freeze BRY fitter."""

from __future__ import annotations

import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import fit_genus1_two_point_bry_scan as fit_stage
    import run_genus1_two_point_imaginary_scan as blind_stage
except ImportError:  # pragma: no cover - package-style execution
    from plumbing import fit_genus1_two_point_bry_scan as fit_stage
    from plumbing import run_genus1_two_point_imaginary_scan as blind_stage


class BlindScanChecks(unittest.TestCase):
    def test_frozen_design(self) -> None:
        design = blind_stage.SmokeDesign()
        self.assertEqual(design.sobol_power, 8)
        self.assertEqual(design.tail_sobol_power, 7)
        self.assertEqual(design.replicates, 4)
        self.assertEqual(design.momentum_order, 16)
        self.assertEqual(
            (design.necklace_order_first, design.necklace_order_second),
            (6, 3),
        )
        self.assertEqual((design.ope_q_order, design.ope_z_order), (3, 8))
        self.assertEqual(blind_stage.DEFAULT_T_VALUES[0], 0.05)
        self.assertEqual(blind_stage.DEFAULT_T_VALUES[-1], 0.95)
        self.assertEqual(len(blind_stage.DEFAULT_T_VALUES), 10)

    def test_blind_wrapper_has_no_postfreeze_basis(self) -> None:
        source = inspect.getsource(blind_stage)
        self.assertNotIn("BRY_REPORTED_COEFFICIENTS", source)
        self.assertNotIn("analytic_reduced_amplitude", source)
        self.assertNotIn("bry_design_matrix", source)

    def test_current_frozen_scan_validates(self) -> None:
        manifest = blind_stage.inspect_existing_scan(
            blind_stage.DEFAULT_OUTPUT_DIR,
            blind_stage.DEFAULT_T_VALUES,
            blind_stage.SmokeDesign(),
        )
        self.assertFalse(manifest["comparison_stage_present"])
        self.assertEqual(len(manifest["points"]), 10)
        self.assertTrue(
            all(
                point["legacy_metadata_gaps"]
                == ["tail_rqmc", "special_dps"]
                for point in manifest["points"]
            )
        )


class PostFreezeFitChecks(unittest.TestCase):
    def test_bry_basis_at_imaginary_energy(self) -> None:
        t_values = np.asarray([0.2, 0.5, 0.8])
        direct = (-t_values**2 + 2.0 * t_values**4 - t_values**5) / 24.0
        np.testing.assert_allclose(
            fit_stage.analytic_reduced_amplitude(t_values),
            direct,
            rtol=0.0,
            atol=2.0e-17,
        )

    def test_current_fit_coefficients(self) -> None:
        loaded = fit_stage.load_frozen_records(fit_stage.DEFAULT_SCAN_DIR)
        result = fit_stage.fit_frozen_records(loaded)
        coefficients = result["weighted_fit"]["coefficients"]
        self.assertTrue(
            math.isclose(coefficients["a"], 1.0050417526680584, abs_tol=2.0e-13)
        )
        self.assertTrue(
            math.isclose(coefficients["b"], 1.0215772074687866, abs_tol=2.0e-13)
        )
        self.assertTrue(
            math.isclose(coefficients["c"], 1.0396699616863916, abs_tol=2.0e-13)
        )
        common = result["common_shape_fit"]
        self.assertTrue(
            math.isclose(common["kappa"], 1.000120480356741, abs_tol=2.0e-13)
        )

    def test_loader_refuses_unfrozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for tag in ("t010", "t020", "t030"):
                path = Path(temporary) / tag / "worldsheet_blind.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"blind_freeze": False}) + "\n")
            with self.assertRaisesRegex(ValueError, "non-frozen"):
                fit_stage.load_frozen_records(Path(temporary))


if __name__ == "__main__":
    unittest.main()
