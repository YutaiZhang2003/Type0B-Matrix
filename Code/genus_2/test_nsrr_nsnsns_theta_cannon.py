#!/usr/bin/env python3
"""Structural tests for the order-eight NSRR/NSNSNS Cannon workflow."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent

from nsrr_nsnsns_theta_cannon import (  # noqa: E402
    _designs,
    _validate_config,
    channel_chunk_count,
    channel_task_ranges,
    decode_task,
    task_count,
)


CONFIG = HERE.parent / "config" / "nsrr_nsnsns_theta_order8_cannon_20260829.json"


class NSRRNSNSNSThetaCannonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_geometry_and_spin_transport_are_validated(self) -> None:
        ledger = _validate_config(self.config)
        self.assertEqual(
            ledger["transported_characteristic"],
            {"alpha": [0, 0], "beta": [0, 0]},
        )
        self.assertLess(ledger["period_transport_residual"], 1.0e-12)

    def test_order_eight_axis_and_task_count(self) -> None:
        designs = _designs(self.config)
        self.assertEqual({row["block_order"] for row in designs}, {8})
        self.assertEqual(
            {row["quadrature_order"] for row in designs}, {8, 10, 12}
        )
        self.assertEqual(task_count(self.config), 6480)

    def test_source_and_target_ranges_partition_all_tasks(self) -> None:
        source = channel_task_ranges(self.config, "source_nsrr")
        target = channel_task_ranges(self.config, "target_nsnsns")
        self.assertEqual(source, ((0, 511), (1024, 2023), (3024, 4751)))
        self.assertEqual(target, ((512, 1023), (2024, 3023), (4752, 6479)))
        covered = {
            task
            for start, stop in source + target
            for task in range(start, stop + 1)
        }
        self.assertEqual(covered, set(range(task_count(self.config))))
        self.assertEqual(channel_chunk_count(self.config, "source_nsrr", 768), 5)
        self.assertEqual(channel_chunk_count(self.config, "target_nsnsns", 1024), 4)

    def test_boundary_task_decoding(self) -> None:
        first, first_node = decode_task(self.config, 0)
        last, last_node = decode_task(self.config, 6479)
        self.assertEqual((first["channel"], first_node), ("source_nsrr", 0))
        self.assertEqual((last["channel"], last_node), ("target_nsnsns", 1727))


if __name__ == "__main__":
    unittest.main()
