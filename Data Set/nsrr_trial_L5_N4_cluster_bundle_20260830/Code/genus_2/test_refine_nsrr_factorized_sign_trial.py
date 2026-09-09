"""Accuracy-extension checks; no change of the nonchiral trial assumptions."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

import refine_nsrr_factorized_sign_trial as refine
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples

trial = refine.trial
BASELINE = trial.ROOT/"Data Set/nsrr_factorized_sign_trial_L2_N3_20260830"


class RefinementConfigurationTests(unittest.TestCase):
    def setUp(self):
        base = trial.load(BASELINE/"config.json")
        with patch.object(trial, "make_config", return_value=deepcopy(base)):
            self.config = refine.make_config(BASELINE)

    def test_accuracy_design_and_original_formula(self):
        self.assertEqual(self.config["levels"], [0, .5, 1, 1.5, 2, 2.5, 3])
        self.assertEqual(len(trial.tasks(self.config)), 27+64+125)
        base = trial.load(BASELINE/"config.json")
        for key in refine.FORMULA_KEYS:
            self.assertEqual(self.config[key], base[key])

    def test_changed_assumptions_and_false_physical_labels_rejected(self):
        for key, value in (("vertex_ansatz", "different"), ("b", 1.5),
                           ("physical_Q", 1.), ("physical_lift_spin_dictionary", "guessed")):
            with self.subTest(key=key):
                config = deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    refine.validate_config(config)

    def test_bad_accuracy_or_missing_levels_rejected(self):
        for orders in ([5], [4, 3], [3, 3, 5], [3, 7]):
            config = deepcopy(self.config)
            config["quadrature_orders"] = orders
            with self.assertRaises(ValueError):
                refine.validate_config(config)
        self.config["levels"].remove(2.5)
        with self.assertRaises(ValueError):
            refine.validate_config(self.config)

    def test_shared_node_audit_detects_changed_blocks(self):
        shard = trial.load(BASELINE/"shards/node-008.json")
        self.assertEqual(refine.baseline_node_check(self.config, shard)["block_scaled_error"], 0.)
        shard["rows"][0]["blocks"][0][0] += .1
        with self.assertRaises(ArithmeticError):
            refine.baseline_node_check(self.config, shard)

    def test_reduction_reproduces_existing_N3(self):
        config = deepcopy(self.config)
        config["quadrature_orders"] = [3]
        shards = [trial.load(BASELINE/"shards"/f"node-{i:03d}.json") for i in range(8, 35)]
        actual = refine.reduced_rows(config, shards)
        previous = [r for r in trial.load(BASELINE/"summary.json")["rows"] if r["quadrature_order"] == 3]
        self.assertEqual(actual, previous)


class RefinementBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.momenta = (.31, .43, .57)
        cls.blocks, cls.checks = refine.block_components(1.4, cls.momenta, 3)

    def test_all_eight_components_through_level_three(self):
        self.assertEqual(set(self.blocks), set(trial.CHANNELS))
        self.assertEqual(self.checks["explicit_PBW_completion_calls"], 4)
        self.assertLess(self.checks["analytic_ground_half_level_max_error"], 1e-11)
        for vectors in self.blocks.values():
            self.assertEqual(set(vectors), set(level_triples(6)))

    def test_lower_coefficients_reproduce_original_runner(self):
        previous, _ = trial.block_components(1.4, self.momenta, 2)
        for channel in trial.CHANNELS:
            for exponent, vector in previous[channel].items():
                np.testing.assert_allclose(self.blocks[channel][exponent], vector, atol=1e-10, rtol=1e-10)

    def test_new_level_equal_sign_coefficients_against_independent_PBW(self):
        b = sp.Rational(7, 5)
        bg = b+1/b
        p = [sp.Rational(str(v)) for v in self.momenta]
        for f in (0, 1):
            for eta in (1, -1):
                oracle = HumanNSRRThetaOracle(
                    central_charge=sp.Rational(3, 2)+3*bg**2,
                    h_ns=bg**2/8+p[0]**2/2,
                    beta_r1=sp.I*p[1]/sp.sqrt(2), beta_r2=sp.I*p[2]/sp.sqrt(2),
                    form_parity=f, primary_parity=0, etas=(eta, eta))
                for e in level_triples(6):
                    if sum(e) <= 4:
                        continue
                    expected = oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                    np.testing.assert_allclose(self.blocks[f, eta, eta][e], expected, atol=2e-9, rtol=2e-9)

    def test_no_unbounded_completion(self):
        with self.assertRaises(ValueError):
            refine.block_components(1.4, self.momenta, 4)


if __name__ == "__main__":
    unittest.main()
