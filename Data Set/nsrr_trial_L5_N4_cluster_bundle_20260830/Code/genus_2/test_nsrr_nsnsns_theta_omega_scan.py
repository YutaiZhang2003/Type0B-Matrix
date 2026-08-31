"""Geometry, conventions, process isolation and strict-reduction regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import nsrr_nsnsns_theta_omega_scan as scan
from compare_nsrr_nsnsns_theta import nsrr_node, _legacy_unvalidated_nsrr_node


CONFIG = Path(__file__).parents[1] / "config/nsrr_nsnsns_theta_omega_scan_20260830.json"


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text())

    def test_geometry_and_design(self):
        scan.validate_config(self.config)
        self.assertEqual(len(scan.tasks(self.config)), 560)
        self.assertEqual(len(scan.channel_indices(self.config, "source_nsrr")), 280)
        changed = copy.deepcopy(self.config)
        changed["target_recursion_order_twice_level"] = 8
        with self.assertRaises(ValueError):
            scan.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["points"][0]["charts"]["target_nsnsns"]["lifts"] = [1, 1, 1]
        with self.assertRaises(ValueError):
            scan.validate_config(changed)

    def test_uncertified_partition_production_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "nonchiral Ramond sewing"):
            scan.source_values(self.config, (.2, .3, .4), 1., None)
        with self.assertRaisesRegex(NotImplementedError, "nonchiral Ramond sewing"):
            nsrr_node()

    def test_retired_formula_cannot_be_mixed_with_corrected_blocks(self):
        with self.assertRaisesRegex(NotImplementedError, "retired"):
            scan._legacy_source_values(self.config, (.21,.37,.52), 1., None)
        with self.assertRaisesRegex(NotImplementedError, "retired"):
            _legacy_unvalidated_nsrr_node(b=1.4, q_values=(.01,.02,.03),
                lifts=(1,1,-1), block_order=1, momenta=(.21,.37,.52),
                measure=1., constants=None)

    def test_each_node_gets_fresh_process(self):
        with patch.object(scan.subprocess, "run") as run:
            scan.channel_worker(CONFIG, Path("shards"), "source_nsrr", 0, 2)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.args[0][0], scan.sys.executable)
                self.assertIn("worker", call.args[0])
                self.assertTrue(call.kwargs["check"])

    def test_incomplete_reduction_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "560 missing"):
                scan.reduce_scan(CONFIG, Path(tmp), Path(tmp) / "summary.json")
            self.assertFalse((Path(tmp) / "summary.svg").exists())

    def test_complete_reducer_pairs_points_levels_and_orders(self):
        cfg = copy.deepcopy(self.config)
        cfg["quadrature_orders"] = [1, 2]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config_path = directory / "config.json"
            scan.write_json(config_path, cfg)
            shards = directory / "shards"
            for task_index in range(len(scan.tasks(cfg))):
                channel, order, node, indices, momenta, measure = scan.node_data(cfg, task_index)
                levels = cfg["source_physical_levels"] if channel == "source_nsrr" else [8]
                values = [{"t": p["t"], "physical_level": level,
                           "sector_contributions": [p["t"] * level / order ** 3, 0.]}
                          for p in cfg["points"] for level in levels]
                scan.write_json(shards / f"task-{task_index:06d}.json", {
                    "schema": scan.SCHEMA, "config_digest": scan._digest(cfg),
                    "implementation_fingerprint": "test-only", "task_index": task_index,
                    "channel": channel, "quadrature_order": order, "node_index": node,
                    "indices": list(indices), "momenta": list(momenta), "measure": measure,
                    "values": values, "maximum_ward_residual": 1e-10 if channel == "source_nsrr" else None,
                    "runtime_seconds": 1.})
            with patch.object(scan, "fingerprint", return_value="test-only"):
                result = scan.reduce_scan(config_path, shards, directory / "summary.json")
            self.assertEqual(result["shards_validated"], 18)
            for row in result["rows"]:
                self.assertAlmostEqual(row["values"]["source_nsrr_L6"]["Z"], row["t"] * 6)
                self.assertAlmostEqual(row["values"]["target_nsnsns_L8"]["Z"], row["t"] * 8)
            for row in result["convergence_diagnostics"]:
                self.assertAlmostEqual(row["source_quadrature_relative_change"], 0.)
                self.assertAlmostEqual(row["source_level_relative_change"], .5)
            ET.parse(directory / "summary.svg")

    def test_plot_contains_raw_curves_and_ratio(self):
        rows = []
        for p in self.config["points"]:
            for order in (4, 6):
                rows.append({"t": p["t"], "quadrature_order": order,
                             "values": {"source_nsrr_L6": {"Q": 2e-7}, "target_nsnsns_L8": {"Q": 3e-7}},
                             "source_over_target_by_source_level": {"4": .65, "6": 2/3}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.svg"
            scan.plot_svg({"config": self.config, "rows": rows, "kappa": 9.94,
                           "maximum_ward_residual": 1e-9}, path)
            ET.parse(path)
            self.assertIn("Re Omega_12", path.read_text())
            self.assertIn("no fitted rescaling", path.read_text())


if __name__ == "__main__":
    unittest.main()
