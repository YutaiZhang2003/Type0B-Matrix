"""Tests independent of super-Liouville and of the double-Virasoro quotient."""
import itertools
import math
import unittest

import numpy as np

from fixed_spin_free_plumbing import (
    charged_frame, characteristic_in_charge_frame, charge_lattice_sum,
    direct_charged_fock_sum, fixed_spin_partition,
)
from physical_free_plumbing_resummation import (
    theta_boson_loop_gaussian, theta_charged_boson_resummation,
    theta_physical_fermion_fredholm,
)


Q_SOURCE = (-.03938929794343916-.02508339199638473j,
            -.04059269805965829+.02978808739157108j,
            -.03515917339490496+.025344924433414715j)
OMEGA_SOURCE = np.array([[1j, .4+.5j], [.4+.5j, -.2+1j]])
SOURCE_BRANCH = ((0, 0), (0, 1))


class FixedSpinFreeTests(unittest.TestCase):
    def test_period_from_charged_sewing(self):
        frame = charged_frame(Q_SOURCE)
        self.assertLess(np.max(abs(frame.omega_charge-OMEGA_SOURCE-SOURCE_BRANCH)), 4e-13)

    def test_charge_quadratic_reconstructs_arbitrary_block(self):
        frame = charged_frame(Q_SOURCE)
        for a, b in ((.5, -.5), (.5, 1.5), (.27, -.63)):
            actual = theta_charged_boson_resummation(
                Q_SOURCE, alpha_zero=a, alpha_one=b, max_mode=24).chiral_value
            charges = np.array([a, b])
            expected = frame.boson_chiral*np.exp(1j*math.pi*charges@frame.omega_charge@charges)
            self.assertLess(abs(actual/expected-1), 2e-14)

    def test_gaussian_is_derived_charge_measure(self):
        frame = charged_frame(Q_SOURCE)
        old = theta_boson_loop_gaussian(Q_SOURCE, max_mode=24)
        self.assertAlmostEqual(frame.loop_gaussian, old.charge_measure_gaussian, places=14)
        self.assertAlmostEqual(frame.loop_gaussian, 1/math.sqrt(3), places=12)

    def test_affine_branch_matters_for_NS_but_not_this_RR(self):
        self.assertEqual(characteristic_in_charge_frame(((0, 0), (0, 0)), SOURCE_BRANCH),
                         ((0, 0), (0, 1)))
        self.assertEqual(characteristic_in_charge_frame(((1, 1), (0, 0)), SOURCE_BRANCH),
                         ((1, 1), (0, 0)))

    def test_wrong_period_branch_fails(self):
        with self.assertRaises(ArithmeticError):
            fixed_spin_partition(Q_SOURCE, OMEGA_SOURCE, ((1, 1), (0, 0)),
                                 period_branch=((0, 0), (0, 0)))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            charged_frame((0, .01, .02))
        with self.assertRaises(ValueError):
            characteristic_in_charge_frame(((1, 1), (0, 0)), ((0, .5), (.5, 0)))

    def test_four_NS_fredholm_determinants_obey_bosonization(self):
        # Unequal q avoids accidental degeneracy of two theta constants.
        q = (.013+.008j, -.019+.009j, .021-.004j)
        frame = charged_frame(q)
        for eta0, eta1 in itertools.product((1, -1), repeat=2):
            determinant = theta_physical_fermion_fredholm(
                q, (eta0, eta1, 1), max_mode=24).determinant_values[0]
            spin = ((0, 0), (int(eta0 < 0), int(eta1 < 0)))
            lattice = charge_lattice_sum(frame.omega_charge, spin)
            self.assertLess(abs(determinant**2/(frame.boson_chiral*lattice)-1), 2e-14)

    def test_filtered_NS_sum_is_not_one_spin(self):
        q = (.013+.008j, -.019+.009j, .021-.004j)
        frame = charged_frame(q)
        filtered = theta_physical_fermion_fredholm(q, (1, 1, 1), max_mode=24)
        ds = filtered.determinant_values
        self.assertEqual(filtered.chiral_value, (-ds[0]+ds[1]+ds[2]+ds[3])/2)
        errors = [abs(filtered.chiral_value**2/(frame.boson_chiral*charge_lattice_sum(
            frame.omega_charge, ((0, 0), beta)))-1)
                  for beta in itertools.product((0, 1), repeat=2)]
        self.assertGreater(min(errors), 1e-3)

    def test_RR_direct_fractional_charge_Fock_sewing(self):
        frame = charged_frame(Q_SOURCE)
        spin = ((1, 1), (0, 0))
        exact = frame.boson_chiral*charge_lattice_sum(frame.omega_charge, spin)
        low = direct_charged_fock_sum(Q_SOURCE, spin, total_level=4)["dirac_chiral"]
        high = direct_charged_fock_sum(Q_SOURCE, spin, total_level=12)["dirac_chiral"]
        self.assertLess(abs(high/exact-1), 2e-11)
        self.assertLess(abs(high-exact), abs(low-exact)/100000)

    def test_NS_direct_integer_charge_Fock_sewing(self):
        q = (.004+.003j, -.007+.001j, .009-.002j)
        frame = charged_frame(q)
        for beta in itertools.product((0, 1), repeat=2):
            spin = ((0, 0), beta)
            exact = frame.boson_chiral*charge_lattice_sum(frame.omega_charge, spin)
            direct = direct_charged_fock_sum(q, spin, total_level=8)["dirac_chiral"]
            self.assertLess(abs(direct/exact-1), 1e-13)

    def test_marked_RR_theta_and_mode_convergence(self):
        results = [fixed_spin_partition(Q_SOURCE, OMEGA_SOURCE, ((1, 1), (0, 0)),
                                       period_branch=SOURCE_BRANCH, max_mode=n)
                   for n in (16, 24, 32)]
        self.assertLess(results[-1]["theta_absolute_relative_error"], 2e-13)
        self.assertLess(abs(results[0]["Z_free"]/results[-1]["Z_free"]-1), 1e-13)
        self.assertAlmostEqual(results[-1]["Z_free"], .5754095923717206, places=13)

    def test_all_six_odd_spins_vanish(self):
        for bits in itertools.product((0, 1), repeat=4):
            alpha, beta = bits[:2], bits[2:]
            if sum(a*b for a, b in zip(alpha, beta)) % 2:
                result = fixed_spin_partition(Q_SOURCE, OMEGA_SOURCE, (alpha, beta),
                                              period_branch=SOURCE_BRANCH)
                self.assertEqual(result["Z_free"], 0)

    def test_Ramond_ground_normalization(self):
        # Dirac charges (+1/2,-1/2) and (-1/2,+1/2) give a factor two.
        # One nonchiral Majorana takes |Z_Dirac,chiral|, not its square.
        q = (1e-12, 1.3e-12, .9e-12)
        frame = charged_frame(q, max_mode=4)
        result = fixed_spin_partition(q, frame.omega_charge, ((1, 1), (0, 0)),
                                      period_branch=((0, 0), (0, 0)), max_mode=4)
        leading = 2*abs(q[0]*q[1])**(1/8)
        self.assertLess(abs(result["Z_majorana"]/leading-1), 1.1e-6)


if __name__ == "__main__":
    unittest.main()
