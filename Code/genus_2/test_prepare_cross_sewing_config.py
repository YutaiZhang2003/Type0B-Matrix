from __future__ import annotations

import unittest
from pathlib import Path

from genus_2.prepare_cross_sewing_config import prepare_config


ROOT = Path(__file__).resolve().parents[2]


class CrossSewingConfigTests(unittest.TestCase):
    def test_prepared_config_has_one_matched_spin_design(self) -> None:
        config = prepare_config(
            ROOT / "Code/config/ns_genus2_cannon_fivepoint_r20_24_n8_12_axis.json"
        )
        self.assertEqual(
            config["convergence_designs"],
            [{"recursion_order": 24, "quadrature_order": 10}],
        )
        self.assertEqual(config["physical_lifts"]["theta"], [1, -1, 1])
        self.assertEqual(config["physical_lifts"]["glasses"], [1, 1, 1])
        for point in config["points"]:
            ledger = config["spin_transport_ledger"][point["id"]]
            self.assertEqual(ledger["theta"], {"alpha": [0, 0], "beta": [0, 0]})
            self.assertEqual(ledger["glasses"], {"alpha": [0, 0], "beta": [0, 0]})

    def test_prepared_config_can_select_an_axis_and_one_point(self) -> None:
        config = prepare_config(
            ROOT / "Code/config/ns_genus2_cannon_fivepoint_r20_24_n8_12_axis.json",
            designs=((20, 10), (22, 10), (24, 8), (24, 12)),
            point_ids=("o0243-periodmatched",),
        )
        self.assertEqual([point["id"] for point in config["points"]], ["o0243-periodmatched"])
        self.assertEqual(
            config["convergence_designs"],
            [
                {"recursion_order": 20, "quadrature_order": 10},
                {"recursion_order": 22, "quadrature_order": 10},
                {"recursion_order": 24, "quadrature_order": 8},
                {"recursion_order": 24, "quadrature_order": 12},
            ],
        )


if __name__ == "__main__":
    unittest.main()
