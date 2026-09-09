"""Tests for independent branching cutoffs; protected code is never changed."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nsrr_branching_cutoff_probe as probe
from compute_full_block import base_twice_level, ns_labels, ramond_labels


class BranchingCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = 1.4
        cls.momenta = (.31, .43, .57)
        cls.grid, cls.raw, cls.labels, cls.ward = probe.branch_data(cls.b, cls.momenta, 3)
        cls.products = probe.make_products(cls.grid, cls.labels, 3)
        cls.runtime = probe.trial.dv.NSRRDoubleVirasoroTheta(
            b=cls.b, physical_momenta=cls.momenta, cutoff=3, completion="none")

    def test_branch_counts(self):
        for K, count in ((3, 80), (4, 112), (5, 152), (6, 196), (8, 300), (10, 388)):
            labels = product(ns_labels(K), ramond_labels(K), ramond_labels(K))
            self.assertEqual(sum(base_twice_level(n) <= 2*K for n in labels), count)

    def test_reconstruct_protected_enlarged_series(self):
        for eta in (1, -1):
            actual = probe.formal_enlarged(self.b, self.momenta, self.raw, self.products, 3, eta)
            expected = self.runtime.enlarged_series(0, eta, eta)
            error = max(abs(actual.get(k, 0j)-expected.get(k, 0j))/max(1., abs(expected.get(k, 0j)))
                        for k in actual.keys() | expected.keys())
            self.assertLess(error, 1e-10)

    def test_numerical_assembly_keeps_above_global_L5(self):
        # This sector starts at 4.5; its first descendant must survive K=5,D=1.
        n = (Fraction(3, 2), Fraction(1, 4), Fraction(1, 4))
        q = (.1, .2, .3)
        products = {n: {(0, 0, 0): 1., (1, 0, 0): 7.}}
        with patch.object(probe, "branch_prefactors", return_value=[(0, 1.)]):
            shells = probe.numerical_shells(self.b, self.momenta, {}, products, q, (0, 1))
        self.assertAlmostEqual(shells[1, 1][9][0], .1**4.5*(1+7*.1))
        self.assertAlmostEqual(shells[1, 0][9][0], .1**4.5)
        self.assertEqual(probe.cumulative_vector(shells[1, 1], 4), [0j]*8)

    def test_fixed_D_shell_sum_matches_independent_direct_evaluation(self):
        q = (.019+.003j, .023-.004j, .029+.002j)
        shells = probe.numerical_shells(self.b, self.momenta, self.raw, self.products, q, (2, 3))
        # Total cutoff six includes all s<=3 plus descendants<=3, without loss.
        for eta in (1, -1):
            formal = probe.formal_enlarged(self.b, self.momenta, self.raw, self.products, 6, eta)
            expected = [0j]*8
            for key, z in formal.items():
                p = key[3] | key[4] << 1 | key[5] << 2
                expected[p] += z*complex(q[0])**(key[0]/2)*q[1]**(key[1]/2)*q[2]**(key[2]/2)
            actual = probe.cumulative_vector(shells[eta, 3], 3)
            self.assertLess(max(abs(a-e) for a, e in zip(actual, expected)), 1e-11)

    def test_supported_quotient_and_odd_partner(self):
        from theta_star_algebra import from_star_spectrum, star_multiply
        a = from_star_spectrum([0, 0, 1.2, 2.3, 3.4, 4.5, 0, 0])
        b = from_star_spectrum([0, 0, 2., 2.1, 2.2, 2.3, 0, 0])
        full = star_multiply(a, b)
        quotient, leakage = probe.supported_quotient(full, b)
        self.assertLess(max(abs(x-y) for x, y in zip(a, quotient)), 1e-12)
        self.assertLess(leakage, 1e-12)
        first, _ = probe.supported_quotient(probe.odd_partner(full), b)
        second = probe.odd_partner(a)
        self.assertLess(max(abs(x-y) for x, y in zip(first, second)), 1e-12)

    def test_cached_actions_identical_and_reused(self):
        with tempfile.TemporaryDirectory(prefix="nsrr-action-cache-test-") as directory:
            for repeat in (False, True):
                if repeat:
                    with patch.object(probe, "solve_ns_l1", side_effect=AssertionError("cache miss")), \
                         patch.object(probe, "solve_ramond_lminus", side_effect=AssertionError("cache miss")):
                        grid, raw, labels, residual = probe.branch_data(self.b, self.momenta, 3, cache_dir=directory)
                else:
                    grid, raw, labels, residual = probe.branch_data(self.b, self.momenta, 3, cache_dir=directory)
                self.assertEqual(labels, self.labels)
                self.assertEqual(grid.ns_actions, self.grid.ns_actions)
                self.assertEqual(grid.r_actions, self.grid.r_actions)
                self.assertEqual(raw, self.raw)

    def test_node_index_not_overwritten_by_parity_index(self):
        with tempfile.TemporaryDirectory(prefix="nsrr-node-index-test-") as directory:
            source = probe.trial.ROOT/"Data Set/nsrr_trial_L5_N3_local_20260830"
            config = probe.prepare(source, Path(directory))
            config.update(branch_cutoffs=[3, 4, 5], descendant_cutoffs=[4, 5])
            result = probe.evaluate_node(config, 0)
            self.assertEqual(result["index"], 0)
            self.assertEqual(result["momenta_slots"], tuple(probe.trial.load(
                Path(config["reference_dir"])/"shards/node-000.json")["momenta_slots"]))


if __name__ == "__main__":
    unittest.main()
