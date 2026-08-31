"""Regression checks for the explicitly hypothetical NSRR sign trial."""
import unittest

import numpy as np
import sympy as sp

import nsrr_factorized_sign_trial as trial
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
from theta_star_algebra import fwht


class NSRRSignTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b, cls.momenta = 1.4, (.31, .43, .57)
        cls.components, cls.checks = trial.block_components(cls.b, cls.momenta, 2)

    def test_explicit_completion_and_analytic_check(self):
        self.assertEqual(self.checks["explicit_PBW_completion_calls"], 4)
        self.assertLess(self.checks["analytic_ground_half_level_max_error"], 1e-12)

    def test_equal_sign_double_virasoro_against_independent_pbw(self):
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
                for e in level_triples(4):
                    expected = oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                    np.testing.assert_allclose(self.components[f, eta, eta][e], expected,
                                               atol=1e-10, rtol=1e-10)

    def test_vertex_exchange_symmetry(self):
        for f in (0, 1):
            for e in self.components[f, 1, -1]:
                np.testing.assert_allclose(self.components[f, 1, -1][e],
                                           self.components[f, -1, 1][e], atol=1e-12)

    def test_sign_and_two_i_phases_cancel_but_are_separate(self):
        blocks = {key: complex(j+1, .2*j) for j, key in enumerate(trial.CHANNELS)}
        anti = {k: z.conjugate() for k, z in blocks.items()}
        c = (1.2, .7)
        actual = trial.contract(blocks, anti, c)
        expected = sum((c[0 if eta == 1 else 1]/2)*(c[0 if ep == 1 else 1]/2)*abs(z)**2
                       for (f, eta, ep), z in blocks.items())
        self.assertAlmostEqual(actual["total"], expected)
        wrong = trial.contract(blocks, anti, c, sewing_sign=False)
        self.assertAlmostEqual(actual["even"], wrong["even"])
        self.assertAlmostEqual(actual["odd"], -wrong["odd"])

    def test_coefficients_are_multiplied_not_absolute_squared(self):
        blocks = {key: 0j for key in trial.CHANNELS}
        blocks[0, 1, 1] = 1
        actual = trial.contract(blocks, blocks, (2j, 0))["total"]
        self.assertEqual(actual, -1)

    def test_ground_trial_normalization_not_a_physical_ground_claim(self):
        blocks = {key: trial.low_level_coefficients(self.b, self.momenta, *key, (1, 1, 1))[0]
                  for key in trial.CHANNELS}
        anti = {k: z.conjugate() for k, z in blocks.items()}
        result = trial.contract(blocks, anti, (1.2, .7))
        self.assertAlmostEqual(result["total"], (1.2+.7)**2)
        self.assertAlmostEqual(result["equal"], 1.2**2+.7**2)
        self.assertAlmostEqual(result["mixed"], 2*1.2*.7)
        # The formal same-chiral-convention anti block is a DIFFERENT
        # hypothesis, explicitly retained as a control in the trial.
        formal = trial.contract(blocks, blocks, (1.2, .7))
        self.assertAlmostEqual(formal["total"], (1.2-.7)**2)

    def test_missing_mixed_blocks_are_rejected(self):
        blocks = {key: 1 for key in trial.CHANNELS if key[1] == key[2]}
        with self.assertRaises(ValueError):
            trial.contract(blocks, blocks, (1, 2))

    def test_unsupported_accuracy_is_rejected(self):
        with self.assertRaises(ValueError):
            trial.block_components(self.b, self.momenta, 3)

    def test_truncation_and_ordinary_lift_sum(self):
        q = (.01+.02j, -.02+.01j, .03-.01j)
        for lifts in ((1, 1, 1), (-1, 1, -1)):
            k = trial.dv.spin_character_index(lifts)
            for level in (0, .5, 1, 1.5, 2):
                blocks = trial.evaluate_blocks(self.components, q, lifts, level)
                for channel, vector in self.components.items():
                    direct = sum(fwht(v)[k]*np.prod([q[i]**(e[i]/2) for i in range(3)])
                                 for e, v in vector.items() if sum(e) <= 2*level)
                    self.assertAlmostEqual(blocks[channel], direct)


if __name__ == "__main__":
    unittest.main()
