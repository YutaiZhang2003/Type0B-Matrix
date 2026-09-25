#!/usr/bin/env python3
"""Focused algebraic and data-selection tests for the SSVV projection."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_spin23_ssvv_1000 import (
    _plot_results,
    build_ssvv_frame,
    fit_inverse_affine,
    select_stratified_rows,
)
from spin23_ssvv_projection import (
    SSVVKinematics,
    normalized_inverse,
    project_scan_coefficients,
    project_vvvv_coefficients,
    raw_resonance_coefficient,
    raw_ssvv_candidate,
)


class SSVVProjectionTests(unittest.TestCase):
    def test_tensor_projection_selects_c_and_m1(self) -> None:
        self.assertEqual(project_vvvv_coefficients((2, 3, 5)), 5)
        self.assertEqual(project_scan_coefficients((5, 3, 2)), 5)

    def test_outgoing_vector_exchange_symmetry(self) -> None:
        original = SSVVKinematics(0.11 + 0.14j, 0.17 + 0.16j, 0.23 + 0.18j)
        exchanged = SSVVKinematics(
            original.omega1,
            original.omega3,
            original.omega2,
        )
        self.assertAlmostEqual(raw_ssvv_candidate(original), raw_ssvv_candidate(exchanged))

    def test_resonance_limit(self) -> None:
        kinematics = SSVVKinematics(0.2j, 0.3j, 0.5j)
        self.assertAlmostEqual(
            raw_ssvv_candidate(kinematics),
            raw_resonance_coefficient(kinematics),
        )

    def test_normalized_inverse_is_affine(self) -> None:
        kinematics = SSVVKinematics(0.08 + 0.14j, 0.13 + 0.17j, 0.19 + 0.20j)
        amplitude = raw_ssvv_candidate(kinematics)
        self.assertAlmostEqual(
            normalized_inverse(amplitude, kinematics),
            1 + 1j * kinematics.vector_pair_sum,
        )

    def test_complex_affine_fit(self) -> None:
        pair_sum = np.asarray([0.1 + 0.2j, 0.3 + 0.1j, 0.4 + 0.5j])
        values = (1.0 - 0.2j) + (0.1 + 0.9j) * pair_sum
        alpha, beta = fit_inverse_affine(pair_sum, values)
        self.assertAlmostEqual(alpha, 1.0 - 0.2j)
        self.assertAlmostEqual(beta, 0.1 + 0.9j)

    def test_stratified_selector_has_requested_unique_count(self) -> None:
        rows = []
        for family, count in (("core", 8), ("structured", 5)):
            rows.extend(
                {"point_id": f"{family}-{index}", "family": family, "status": "ok"}
                for index in range(count)
            )
        frame = pd.DataFrame(rows)
        selected = select_stratified_rows(frame, {"core": 4, "structured": 3})
        self.assertEqual(len(selected), 7)
        self.assertFalse(selected["point_id"].duplicated().any())

    def test_plot_smoke(self) -> None:
        """Exercise every MathText label without production data."""

        import tempfile
        from pathlib import Path

        source = pd.DataFrame(
            {
                "omega1_re": [0.10, 0.12],
                "omega1_im": [0.14, 0.15],
                "omega2_re": [0.16, 0.18],
                "omega2_im": [0.17, 0.18],
                "omega3_re": [0.20, 0.22],
                "omega3_im": [0.19, 0.20],
            }
        )
        candidates = []
        for row in source.itertuples(index=False):
            candidates.append(
                raw_ssvv_candidate(
                    SSVVKinematics(
                        row.omega1_re + 1j * row.omega1_im,
                        row.omega2_re + 1j * row.omega2_im,
                        row.omega3_re + 1j * row.omega3_im,
                    )
                )
            )
        source["M1_re"] = np.real(candidates)
        source["M1_im"] = np.imag(candidates)
        frame = build_ssvv_frame(source)
        with tempfile.TemporaryDirectory() as temporary:
            _plot_results(frame, Path(temporary))
            self.assertTrue(
                (Path(temporary) / "spin23_ssvv_1000_diagnostics.png").exists()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
