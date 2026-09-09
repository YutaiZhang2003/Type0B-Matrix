"""Provenance and reduction boundary for the fresh target-only rerun."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recompute_all_ns_reference as fresh


class FreshReferenceTests(unittest.TestCase):
    def setUp(self):
        base = fresh.scan._load(fresh.ROOT / "Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json")
        self.config = {
            "schema": fresh.SCHEMA, "implementation_fingerprint": fresh.fingerprint(),
            "quadrature_order": 1, "recursion_order_twice_level": 16,
            "quadrature_reference_abs_q": base["quadrature_reference_abs_q"]["target_nsnsns"],
            "parameters": base["parameters"], "numerics": base["numerics"], "points": [],
        }
        for point in base["points"]:
            chart = point["charts"]["target_nsnsns"]
            self.config["points"].append({"t": point["t"], "q_values": chart["q_values"],
                                         "Z_free": chart["physical_free_superfield"],
                                         "forward_period_error": chart["inverse_period_residual"]})
        momenta, measure = fresh.node_data(self.config, 0)
        self.shard = {
            "schema": fresh.SCHEMA, "implementation_fingerprint": fresh.fingerprint(),
            "config_digest": fresh.scan._digest(self.config), "index": 0,
            "momenta": list(momenta), "measure": measure, "runtime_seconds": 1,
            "values": [{"t": p["t"], "sector_contributions": [2., 3.],
                        "global_nonconverged_calls": 0, "global_max_occupation_used": 4}
                       for p in self.config["points"]],
        }

    def test_reject_changed_implementation_and_node(self):
        fresh.validate_config(self.config)
        fresh.validate_shard(self.config, 0, self.shard)
        altered = copy.deepcopy(self.config)
        altered["implementation_fingerprint"] = "old"
        with self.assertRaises(ValueError):
            fresh.validate_config(altered)
        altered = copy.deepcopy(self.shard)
        altered["momenta"][0] += 1e-3
        with self.assertRaises(ValueError):
            fresh.validate_shard(self.config, 0, altered)

    def test_reject_unconverged_or_incomplete_values(self):
        altered = copy.deepcopy(self.shard)
        altered["values"][0]["global_nonconverged_calls"] = 1
        with self.assertRaises(ValueError):
            fresh.validate_shard(self.config, 0, altered)
        altered = copy.deepcopy(self.shard)
        altered["values"].pop()
        with self.assertRaises(ValueError):
            fresh.validate_shard(self.config, 0, altered)

    def test_partial_reduction_never_supplies_source_values(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            fresh.scan.write_json(out/"config.json", self.config)
            with self.assertRaises(ValueError):
                fresh.reduce_run(out/"config.json", out)
            fresh.scan.write_json(out/"shards/node-000.json", self.shard)
            with patch.object(fresh, "plot_svg"):
                result = fresh.reduce_run(out/"config.json", out)
        for row in result["rows"]:
            self.assertEqual(row["target_Z"], 5.)
            self.assertAlmostEqual(row["target_Q"], 5/row["target_Z_free"]**result["kappa"])
            self.assertIsNone(row["source_Z"])
            self.assertIsNone(row["source_Q"])
            self.assertIsNone(row["source_over_target"])


if __name__ == "__main__":
    unittest.main()
