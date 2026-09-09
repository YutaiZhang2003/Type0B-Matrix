from __future__ import annotations

import unittest

import mpmath as mp

from genus0_elliptic_h_recursion import (
    comb_cross_ratios,
    compute_h_recursion,
    coordinates_from_segment_nomes,
    effective_plumbing_parameters,
    invert_aligned_coordinates,
    reconstruct_from_real_moduli,
)


class GeometryAndBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        mp.mp.dps = 60

    def test_effective_plumbing_endpoint_factors(self) -> None:
        self.assertEqual(effective_plumbing_parameters((mp.mpf("0.01"),)), (mp.mpf("0.16"),))
        self.assertEqual(
            effective_plumbing_parameters((mp.mpf("0.1"), mp.mpf("0.2"))),
            (mp.mpf("0.4"), mp.mpf("0.8")),
        )
        self.assertEqual(
            effective_plumbing_parameters(
                (mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("0.4"))
            ),
            (mp.mpf("0.4"), mp.mpf("0.2"), mp.mpf("0.3"), mp.mpf("1.6")),
        )

    def test_six_point_forward_inverse_map(self) -> None:
        inverse = invert_aligned_coordinates(
            "0.1075", ("0.32", "0.62"), dps=60
        )
        z, mobiles = coordinates_from_segment_nomes(inverse.segment_nomes)
        self.assertLess(abs(z - mp.mpf("0.1075")), mp.mpf("1e-50"))
        self.assertLess(abs(mobiles[0] - mp.mpf("0.32")), mp.mpf("1e-50"))
        self.assertLess(abs(mobiles[1] - mp.mpf("0.62")), mp.mpf("1e-50"))
        self.assertLess(inverse.product_residual, mp.mpf("1e-55"))
        self.assertEqual(
            comb_cross_ratios("0.1", ("0.25", "0.5")),
            (mp.mpf("0.4"), mp.mpf("0.5"), mp.mpf("0.5")),
        )

    def test_full_six_point_reconstruction_snapshot(self) -> None:
        table = compute_h_recursion(
            central_charge="26.215",
            external_weights=("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
            internal_weights=("0.9371", "1.0837", "1.3321"),
            order=6,
            dps=60,
            pole_tolerance="1e-10",
        )
        result = reconstruct_from_real_moduli(
            table,
            z="0.1075",
            mobile_positions=("0.32", "0.62"),
        )
        expected = mp.mpf(
            "1.046682032246890188577303319446838245176292106717107141440284920127"
        )
        self.assertLess(abs(result.value - expected), mp.mpf("1e-48"))

    def test_seven_point_general_engine_smoke(self) -> None:
        table = compute_h_recursion(
            central_charge="28.4",
            external_weights=(
                "0.12",
                "0.23",
                "0.34",
                "0.45",
                "0.56",
                "0.67",
                "0.78",
            ),
            internal_weights=("0.91", "1.07", "1.24", "1.43"),
            order=2,
            dps=50,
            pole_tolerance="1e-10",
        )
        self.assertEqual(table.point_count, 7)
        self.assertEqual(table.edge_count, 4)
        self.assertEqual(len(table.coefficients), 15)
        value = table.evaluate(("0.05", "0.08", "0.11", "0.14"))
        self.assertTrue(mp.isfinite(abs(value)))


if __name__ == "__main__":
    unittest.main()
