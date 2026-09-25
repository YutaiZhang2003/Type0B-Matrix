#!/usr/bin/env python3
"""Focused checks for the raw ``S -> S S S`` projection and fit."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_spin23_ssss_1000 import (
    _plot_results,
    build_ssss_frame,
    fit_channel_weights,
)
from spin23_ssss_projection import (
    SSSSKinematics,
    channel_features,
    normalized_amplitude,
    project_scan_coefficients,
    project_vvvv_coefficients,
    raw_resonance_coefficient,
    raw_ssss_candidate,
)


class SSSSProjectionTests(unittest.TestCase):
    def test_tensor_projection_sums_all_channels(self) -> None:
        self.assertEqual(project_vvvv_coefficients((2, 3, 5)), 10)
        self.assertEqual(project_scan_coefficients((5, 3, 2)), 10)

    def test_outgoing_singlet_permutation_symmetry(self) -> None:
        original = SSSSKinematics(0.11 + 0.14j, 0.17 + 0.16j, 0.23 + 0.18j)
        values = (original.omega1, original.omega2, original.omega3)
        for permutation in ((1, 0, 2), (2, 1, 0), (1, 2, 0)):
            permuted = SSSSKinematics(*(values[index] for index in permutation))
            self.assertAlmostEqual(raw_ssss_candidate(original), raw_ssss_candidate(permuted))

    def test_resonance_limit(self) -> None:
        kinematics = SSSSKinematics(0.2j, 0.3j, 0.5j)
        self.assertAlmostEqual(
            raw_ssss_candidate(kinematics),
            raw_resonance_coefficient(kinematics),
        )

    def test_normalized_candidate_is_channel_sum(self) -> None:
        kinematics = SSSSKinematics(0.08 + 0.14j, 0.13 + 0.17j, 0.19 + 0.20j)
        self.assertAlmostEqual(
            normalized_amplitude(raw_ssss_candidate(kinematics), kinematics),
            sum(channel_features(kinematics)),
        )

    def test_unconstrained_fit_recovers_channel_weights(self) -> None:
        features = np.asarray(
            [
                [1.0 + 0.1j, 0.7 - 0.2j, 0.5 + 0.3j],
                [0.4 + 0.2j, 1.1 + 0.1j, 0.8 - 0.1j],
                [0.9 - 0.3j, 0.6 + 0.4j, 1.2 + 0.2j],
                [1.3 + 0.2j, 0.2 + 0.1j, 0.7 - 0.4j],
            ]
        )
        expected = np.asarray([1.0 + 0.1j, 0.9 - 0.2j, 1.1 + 0.05j])
        fitted = fit_channel_weights(features, features @ expected)
        np.testing.assert_allclose(fitted, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_plot_smoke(self) -> None:
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
        channels = [[], [], []]
        for row in source.itertuples(index=False):
            kin = SSSSKinematics(
                row.omega1_re + 1j * row.omega1_im,
                row.omega2_re + 1j * row.omega2_im,
                row.omega3_re + 1j * row.omega3_im,
            )
            common = -np.pi * kin.energy_product
            for index, feature in enumerate(channel_features(kin)):
                channels[index].append(common * feature)
        # Scan order is (C,B,A), while the sum is insensitive to this reversal.
        for name, values in zip(("M1", "M2", "M3"), channels[::-1], strict=True):
            source[f"{name}_re"] = np.real(values)
            source[f"{name}_im"] = np.imag(values)
        frame = build_ssss_frame(source)
        with tempfile.TemporaryDirectory() as temporary:
            _plot_results(frame, np.ones(3, dtype=complex), Path(temporary))
            self.assertTrue(
                (Path(temporary) / "spin23_ssss_1000_diagnostics.png").exists()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
