"""Portable configuration and bounded-job tests; expensive L5 checks are preflights."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import nsrr_trial_cluster as cluster


class ClusterTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="nsrr-cluster-test-")
        cls.reference = cluster.trial.ROOT/"Data Set/nsrr_factorized_sign_trial_L3_N5_20260830"
        cls.configs = {n: cluster.prepare(cls.reference, Path(cls.temporary.name)/f"n{n}.json", n)
                       for n in (3, 4)}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_N3_and_N4_complete_grids(self):
        for n, config in self.configs.items():
            cluster.validate_config(config)
            self.assertEqual(len(config["reference_nodes"]), n**3)
            self.assertEqual(config["levels"], [j/2 for j in range(11)])
            self.assertEqual(len(config["channels"]), 8)

    def test_three_hour_resources_and_node_timeout(self):
        resource = self.configs[4]["resources"]
        self.assertEqual(resource["wall_seconds"], 3*3600)
        self.assertLess(resource["compute_seconds"], resource["wall_seconds"])
        self.assertEqual(resource["cpus"], 8)
        self.assertEqual(resource["node_timeout_seconds"], 900)

    def test_no_runtime_dependency_on_local_reference_paths(self):
        for config in self.configs.values():
            self.assertNotIn("baseline_config_path", config)
            self.assertNotIn("geometry_path", config)
            self.assertNotIn("/Users/", str(config))

    def test_changed_inputs_rejected(self):
        for key, value in (("b", 1.5), ("vertex_ansatz", "changed"), ("physical_Q", 1.0),
                           ("max_level", 6), ("quadrature_orders", [5])):
            config = deepcopy(self.configs[4])
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                cluster.validate_config(config)

    def test_changed_coefficient_rejected(self):
        config = deepcopy(self.configs[4])
        config["reference_nodes"][0]["C_BRY"][0][0] *= 2
        with self.assertRaises(ValueError):
            cluster.validate_config(config)

    def test_missing_node_rejected(self):
        config = deepcopy(self.configs[4])
        config["reference_nodes"].pop()
        with self.assertRaises(ValueError):
            cluster.validate_config(config)

    def test_same_frame_free_factors(self):
        self.assertLess(cluster.preflight(self.configs[4])["free_factor_max_relative_error"], 1e-12)

    def test_incomplete_reduction_rejected(self):
        with self.assertRaises(FileNotFoundError):
            cluster.reduce_run(self.configs[4], Path(self.temporary.name)/"missing")


if __name__ == "__main__":
    unittest.main()
