"""Spin, marking and frame contracts for the repaired NSRR chart."""
import copy
from pathlib import Path
import unittest

import numpy as np

import nsrr_human_note_geometry as geometry


class HumanGeometryTests(unittest.TestCase):
    def test_marking_and_spin_transport(self):
        M = geometry.SOURCE_REMARKING
        J = np.block([[geometry.ZERO, geometry.IDENTITY], [-geometry.IDENTITY, geometry.ZERO]])
        np.testing.assert_array_equal(M.T@J@M, J)
        self.assertEqual(geometry.scan._transport_spin_characteristic(M, geometry.scan.SOURCE_SPIN),
                         ((1, 1), (0, 0)))
        self.assertEqual(geometry.geometry_to_human_slots(("R0", "R1", "NSinf")),
                         ("NSinf", "R1", "R0"))

    def test_inverse_chart_is_resolved_not_naively_permuted(self):
        cfg = geometry.scan._load(Path(__file__).parents[1]/"config/nsrr_nsnsns_theta_omega_scan_20260830.json")
        cfg = copy.deepcopy(cfg)
        cfg["points"] = [cfg["points"][2]]
        result = geometry.build_geometry(cfg)
        point = result["points"][0]
        self.assertLess(point["high_order_forward_period_residual"], 1e-9)
        old = [complex(q) for q in cfg["points"][0]["charts"]["source_nsrr"]["q_values"]]
        new = [complex(q) for q in point["source_chart"]["q_values"]]
        self.assertGreater(max(abs(x-y) for x, y in zip(new, reversed(old))), 1e-4)
        self.assertEqual(result["geometry_edge_sectors"], ["R", "R", "NS"])
        self.assertIn("not inferred", result["physical_ramond_lift_dictionary_status"])


if __name__ == "__main__":
    unittest.main()
