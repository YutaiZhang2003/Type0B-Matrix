"""Numerical design and spin-boundary tests; no kernel edits."""
import copy
import math
import unittest

import numpy as np

import check_nsrr_spin_quadrature as check
import audit_nsrr_comparison_spin_basis as spin


class SpinQuadratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = check.trial.load(check.DEFAULT_OUTPUT/"config.json")

    def test_protected_and_frozen_configs(self):
        check.validate(self.c)
        for channel in ("target", "source"):
            self.assertEqual(self.c[channel], check.trial.load(check.REFERENCES[channel]/"config.json"))

    def test_gaussian_rule_normalization(self):
        for q in (.04, .13):
            for n in (3, 5, 6, 7):
                momenta, weights = check.trial._rules((q, q, q), n)[0]
                integral = sum(w*math.exp(math.log(q)*p*p) for p, w in zip(momenta, weights))
                self.assertAlmostEqual(integral, 1/(2*math.sqrt(math.pi*(-math.log(q)))), places=14)

    def test_node_index_is_not_parity_index(self):
        for channel in ("source", "target"):
            for index in (0, 1, 31, 215):
                momenta, weight = check.node_data(self.c, channel, 6, index)
                self.assertTrue(all(p > 0 for p in momenta))
                self.assertGreater(weight, 0)
            with self.assertRaises(ValueError):
                check.node_data(self.c, channel, 6, 216)

    def test_saved_node_reproduction_and_cache(self):
        for channel in ("source", "target"):
            p = check.trial.load(check.DEFAULT_OUTPUT/f"{channel}_N5_reproduction.json")
            self.assertLess(p["maximum_relative_error"], 1e-12)
            check.validate_shard(self.c, channel, 5, p["node"], p["new_shard"])
            changed = copy.deepcopy(p["new_shard"])
            changed["index"] += 1
            with self.assertRaises(ValueError):
                check.validate_shard(self.c, channel, 5, p["node"], changed)
        self.assertTrue(check.trial.load(check.DEFAULT_OUTPUT/"source_N5_reproduction.json")["cache_bitwise_equal"])

    def test_reject_changed_point_and_physical_promotion(self):
        c = copy.deepcopy(self.c)
        c["target_point"]["lifts"] = [1, 1, 1]
        with self.assertRaises(ValueError):
            check.validate(c)
        c = copy.deepcopy(self.c)
        c["physical_Q_NSrr"] = .1
        with self.assertRaises(ValueError):
            check.validate(c)

    def test_branch_changes_spin_label(self):
        self.assertEqual(spin.raw_spin((-1, 1, 1), spin.TARGET_BRANCH), ((0, 0), (0, 0)))
        self.assertEqual(spin.raw_spin((1, -1, 1), spin.TARGET_BRANCH), ((0, 0), (1, 1)))

    def test_no_real_lift_relabel_removes_quadratic_sign(self):
        raw = {spin.leading_pair_signs(l) for l in spin.LIFTS}
        filtered = {spin.leading_pair_signs(l, True) for l in spin.LIFTS}
        self.assertFalse(raw & filtered)
        self.assertTrue(all(math.prod(s) == 1 for s in raw))
        self.assertTrue(all(math.prod(s) == -1 for s in filtered))

    def test_ground_trial_lift_invariance_is_not_a_spin_assignment(self):
        # The unchanged ansatz gives (E+O)^2 for every real Ramond lift:
        # swapping a lift exchanges which f/sign components carry it.
        E, O = 1.2, .7
        for lifts in spin.LIFTS:
            blocks = {key: check.trial.low_level_coefficients(
                1.4, (.31, .43, .57), *key, lifts)[0] for key in check.trial.CHANNELS}
            result = check.trial.contract(blocks, {k: z.conjugate() for k, z in blocks.items()}, (E, O))
            self.assertAlmostEqual(result["total"], (E+O)**2)

    def test_full_free_spin_basis_audit(self):
        report = spin.audit(self.c)
        self.assertEqual(report["U_squared_identity_error"], 0)
        self.assertLess(report["NSRR_trial_lift_relative_spread"], 1e-12)
        for chart in report["charts"].values():
            self.assertLess(chart["charge_period_error"], 1e-8)
            self.assertLess(chart["F_equals_U_D_max_absolute_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
