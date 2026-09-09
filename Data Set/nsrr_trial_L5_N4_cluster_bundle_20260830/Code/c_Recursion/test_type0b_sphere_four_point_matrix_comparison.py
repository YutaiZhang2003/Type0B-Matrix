"""Tests for the BRY normalization and Type-0B matrix-model comparison."""

from __future__ import annotations

import math
import unittest

from compare_type0b_sphere_four_point_wall_one_to_matrix_model import (
    _chi_square_survival_even,
    bry_worldsheet_coefficient,
    compare_scan,
    matrix_model_coefficient,
)


class Type0BFourPointMatrixComparisonTests(unittest.TestCase):
    def test_bry_reduced_target_maps_to_matrix_model_coefficient(self):
        outgoing = (0.1 - 0.2j, 0.3 + 0.1j, 0.2 + 0.4j)
        incoming = sum(outgoing)
        target = (
            math.pi
            * incoming
            * outgoing[0]
            * outgoing[1]
            * outgoing[2]
            * (1.0 + 2.0j * incoming)
        )
        self.assertAlmostEqual(
            bry_worldsheet_coefficient(target),
            matrix_model_coefficient(incoming, outgoing),
            places=14,
        )

    def test_no_extra_leg_factor_enters_worldsheet_conversion(self):
        reduced = 0.17 - 0.04j
        self.assertEqual(
            bry_worldsheet_coefficient(reduced), 8.0j * reduced / math.pi
        )

    def test_even_chi_square_survival_function(self):
        self.assertAlmostEqual(_chi_square_survival_even(0.0, 20), 1.0)
        self.assertAlmostEqual(
            _chi_square_survival_even(26.37311393939577, 20),
            0.1538396762342082,
            places=14,
        )

    def test_underresolved_scan_is_not_comparison_ready(self):
        outgoing = (0.1 + 0.2j, 0.2 + 0.3j, 0.3 + 0.4j)
        incoming = sum(outgoing)
        target = (
            math.pi
            * incoming
            * outgoing[0]
            * outgoing[1]
            * outgoing[2]
            * (1.0 + 2.0j * incoming)
        )

        def encoded(value):
            return {"real": value.real, "imag": value.imag}

        payload = {
            "points": [
                {
                    "index": 0,
                    "x": 0.2,
                    "t": 0.3,
                    "status": "integrated",
                    "crossing_audit": {"relative_spread": 1.0e-4},
                    "amplitude": {
                        "mean": encoded(target),
                        "incoming_energy": encoded(incoming),
                        "outgoing_energies": [encoded(value) for value in outgoing],
                        "standard_error_real": abs(target),
                        "standard_error_imag": abs(target),
                    },
                }
            ]
        }
        comparison = compare_scan(payload)
        self.assertFalse(comparison["precision_gate_passed"])
        self.assertEqual(comparison["status"], "unconverged-moduli-scan")
        self.assertFalse(comparison["aggregate"]["valid_for_inference"])


if __name__ == "__main__":
    unittest.main()
