#!/usr/bin/env python3
"""Focused tests for genuine-singlet production reduction diagnostics."""

from __future__ import annotations

import unittest

from reduce_spin23_singlet_scan import _block_backend_summary, _convergence_control_summary


def _row(
    point_id: str,
    family: str,
    setting: str,
    *,
    parent_id: str = "",
    current: complex,
    lower: complex,
) -> dict[str, str]:
    result = {
        "point_id": point_id,
        "parent_id": parent_id,
        "family": family,
        "setting": setting,
    }
    for process in ("ssvv", "ssss"):
        result[f"{process}_raw_re"] = str(current.real)
        result[f"{process}_raw_im"] = str(current.imag)
        result[f"{process}_raw_lower_q_re"] = str(lower.real)
        result[f"{process}_raw_lower_q_im"] = str(lower.imag)
    return result


class SingletReductionTests(unittest.TestCase):
    def test_backend_summary_accepts_blank_gram_columns(self) -> None:
        recursive = {"block_backend": "c_recursion", "maximum_gram_condition": "",
                     "maximum_recursion_cancellation": "12.0"}
        only_recursion = _block_backend_summary([recursive])
        self.assertIsNone(only_recursion["maximum_gram_condition"])
        self.assertEqual(only_recursion["maximum_recursion_cancellation"]["max"], 12.0)
        mixed = _block_backend_summary([recursive, {"maximum_gram_condition": "1e8"}])
        self.assertEqual(mixed["maximum_gram_condition"]["max"], 1e8)
        self.assertEqual(mixed["block_backend_counts"], {"c_recursion": 1, "inverse_gram_legacy": 1})

    def test_controls_use_the_intended_parent_truncation(self) -> None:
        parent = _row(
            "parent",
            "core",
            "production",
            current=2 + 1j,
            lower=1 + 1j,
        )
        controls = [
            _row(
                "repeat",
                "convergence",
                "q_plus2",
                parent_id="parent",
                current=2 + 1j,
                lower=1 + 1j,
            ),
            _row(
                "tail",
                "convergence",
                "p_extended",
                parent_id="parent",
                current=1.1 + 1j,
                lower=0.9 + 1j,
            ),
        ]
        summary = _convergence_control_summary([parent], controls)
        profiles = summary["profiles"]
        self.assertEqual(profiles["q_plus2"]["ssvv"]["max"], 0.0)
        expected = 0.1 / abs(1.1 + 1j)
        self.assertAlmostEqual(
            profiles["p_extended"]["ssss"]["max"],
            expected,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
