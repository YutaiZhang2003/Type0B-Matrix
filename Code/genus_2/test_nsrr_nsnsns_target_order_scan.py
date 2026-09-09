"""Controlled-design, baseline reproduction and strict-reduction checks."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import run_nsrr_nsnsns_target_order_scan as target
from audit_nsrr_nsnsns_edge_order import audit as audit_edge_order
from audit_nsrr_nsnsns_spin_projection import audit as audit_spin_projection


class TargetOrderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = target.scan._load(Path(__file__).parents[1]/"config/nsrr_nsnsns_theta_omega_scan_20260830.json")
        self.base.update(quadrature_orders=[1, 2], target_physical_level=4,
                         target_recursion_order_twice_level=8, source_physical_levels=[3, 4])
        target.scan.write_json(self.directory/"config.json", self.base)
        rows = []
        for point in self.base["points"]:
            for n in (1, 2):
                free = point["charts"]["target_nsnsns"]["physical_free_superfield"]
                q = 1.01*point["t"]/free**9.94
                rows.append({"t": point["t"], "quadrature_order": n,
                             "values": {"source_nsrr_L4": {"Q": 2*q}, "target_nsnsns_L4": {"Q": q}}})
        target.scan.write_json(self.directory/"summary.json", {
            "config": self.base, "implementation_fingerprint": target.scan.fingerprint(),
            "kappa": 9.94, "rows": rows})
        self.config = target.make_config(self.directory, [8, 12, 16], [2])
        self.config_path = self.directory/"order-config.json"
        target.scan.write_json(self.config_path, self.config)
        for index in target.task_indices(self.config):
            channel, n, node, indices, momenta, measure = target.scan.node_data(self.base, index)
            target.scan.write_json(self.directory/"shards"/f"task-{index:06d}.json", {
                "schema": target.scan.SCHEMA, "config_digest": target.scan._digest(self.base),
                "implementation_fingerprint": target.scan.fingerprint(), "task_index": index,
                "channel": channel, "quadrature_order": n, "node_index": node,
                "indices": list(indices), "momenta": list(momenta), "measure": measure,
                "maximum_ward_residual": None, "runtime_seconds": 1.,
                "values": [{"t": p["t"], "physical_level": 4,
                            "sector_contributions": [p["t"]/n**3, .01*p["t"]/n**3]} for p in self.base["points"]]})

    def fill(self):
        def fake_node(**kwargs):
            q = kwargs["q_values"]
            point = next(p for p in self.base["points"] if tuple(complex(x) for x in p["charts"]["target_nsnsns"]["q_values"]) == q)
            factor = 1+.001*(kwargs["recursion_order"]-8)
            return point["t"]*factor/8, .01*point["t"]*factor/8
        rec = SimpleNamespace(global_max_used=12, global_nonconverged_calls=0,
                              global_worst_last_shell_relative=1e-9, confluent_max_total_cancellation_ratio=1.)
        with patch.object(target.scan, "all_ns_node", side_effect=fake_node), patch.object(
                target.scan, "NSGenus2CRecursion", return_value=rec), patch.object(
                target.scan, "GenericSuperLiouvilleConstants"), patch("builtins.print"):
            for index in target.task_indices(self.config):
                target.worker(self.config_path, self.directory/"new-shards", index)

    def test_unchanged_grid_and_order_units(self):
        self.assertEqual(len(target.task_indices(self.config)), 8)
        self.assertEqual(self.config["baseline_config"], self.base)
        for key, value in (("recursion_orders", [12, 16]), ("quadrature_orders", [3])):
            invalid = copy.deepcopy(self.config)
            invalid[key] = value
            with self.assertRaises(ValueError):
                target.validate_config(invalid)

    def test_slot_order_conversion_changes_middle_edge_coefficient(self):
        result = audit_edge_order()
        self.assertEqual(result["geometry_order"], list(reversed(result["ccy_tensor_slot_order"])))
        unchanged = result["example_q_one_global_coefficient_without_boundary_conversion"][0]
        converted = result["example_q_one_global_coefficient_with_boundary_conversion"][0]
        self.assertAlmostEqual(unchanged, 2.404545454545454)
        self.assertAlmostEqual(converted, .004545454545454533)

    def test_corrected_runtime_uses_literal_ground_spin_sum(self):
        result = audit_spin_projection()
        same = next(r for r in result["rows"] if r["form_parity"] == 0 and r["eta_left"] == r["eta_right"] == 1)
        crossed = next(r for r in result["rows"] if r["form_parity"] == 0 and r["eta_left"] == 1 and r["eta_right"] == -1)
        self.assertEqual(same["literal_fixed_lift_sum"], [0., 0.])
        self.assertEqual(same["star_character"], [2., 0.])
        self.assertEqual(crossed["literal_fixed_lift_sum"], [2., 0.])
        self.assertEqual(crossed["star_character"], [0., 0.])
        self.assertLess(max(r["runtime_minus_literal_absolute"] for r in result["rows"]), 1e-10)

    def test_incomplete_reduction_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "8 missing"):
            target.reduce_scan(self.config_path, self.directory/"empty", self.directory/"result.json")

    def test_reducer_preserves_source_and_pairs_recursion_orders(self):
        self.fill()
        result = target.reduce_scan(self.config_path, self.directory/"new-shards", self.directory/"result.json")
        self.assertEqual(result["shards_validated"], 8)
        self.assertEqual(result["baseline_sector_relative_error_max"], 0.)
        for row in result["rows"]:
            factor = 1+.001*(row["recursion_order"]-8)
            self.assertAlmostEqual(row["source_over_target"], 2/factor)
            self.assertAlmostEqual(row["target_relative_change_from_baseline"], factor-1)
        ET.parse(self.directory/"result.svg")

    def test_invalid_baseline_reproduction_rejected(self):
        index = target.task_indices(self.config)[0]
        self.fill()
        shard = target.scan._load(self.directory/"new-shards"/f"task-{index:06d}.json")
        shard["baseline_sector_relative_error_max"] = 1e-4
        with self.assertRaisesRegex(ValueError, "baseline R"):
            target.validate_shard(self.config, index, shard)
        shard["baseline_sector_relative_error_max"] = 0.
        shard["values"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            target.validate_shard(self.config, index, shard)


if __name__ == "__main__":
    unittest.main()
