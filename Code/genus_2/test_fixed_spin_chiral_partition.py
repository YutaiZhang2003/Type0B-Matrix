"""Physical phase/monodromy regressions for one unsquared Majorana."""
import cmath
import itertools
import math
import unittest

import numpy as np

from fixed_spin_free_plumbing import (
    charged_frame, fixed_spin_chiral_partition, fixed_spin_partition,
    theta_translation_phase,
)
from free_boson_plumbing import riemann_theta_constant_genus2
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from spin_structure import SpinCharacteristic


class ChiralFreeTests(unittest.TestCase):
    def test_saved_rr_keeps_pi_over_eight_phase_and_nonchiral_norm(self):
        q = (-.03938929794343916-.02508339199638473j,
             -.04059269805965829+.02978808739157108j,
             -.03515917339490496+.025344924433414715j)
        omega = np.array([[1j, .4+.5j], [.4+.5j, -.2+1j]])
        branch = ((0, 0), (0, 1))
        spin = ((1, 1), (0, 0))
        actual = fixed_spin_chiral_partition(q, omega, spin, period_branch=branch)
        paired = fixed_spin_partition(q, omega, spin, period_branch=branch)
        self.assertLess(abs(actual["theta_translation_phase"]-cmath.exp(1j*math.pi/4)), 1e-15)
        naive = cmath.sqrt(actual["boson_chiral"]*actual["marked_theta"])
        self.assertLess(abs(actual["majorana_chiral"]/naive-cmath.exp(1j*math.pi/8)), 3e-13)
        self.assertLess(abs(actual["majorana_chiral"]**2/actual["dirac_chiral_from_marked_theta"]-1), 5e-13)
        self.assertEqual(actual["Z_free_nonchiral"], paired["Z_free"])
        self.assertAlmostEqual(actual["Z_free_nonchiral"], .5754095923717206, places=13)
        self.assertLess(abs(abs(actual["majorana_chiral"])**2/paired["Z_majorana"]-1), 1e-13)

    def test_translation_phase_for_all_even_spins_and_binary_reductions(self):
        omega = np.array([[.13+1.2j, .17+.21j], [.17+.21j, -.11+1.1j]])
        for branch in (((0, 0), (0, 1)), ((-1, -1), (-1, 1)), ((2, -1), (-1, 3))):
            for a, b in itertools.product(itertools.product((0, 1), repeat=2), repeat=2):
                spin = SpinCharacteristic(a, b)
                if spin.arf:
                    continue
                before = complex(riemann_theta_constant_genus2(omega, spin.pairs, tol=1e-15))
                after = complex(riemann_theta_constant_genus2(
                    omega+np.asarray(branch), spin.charge_frame(branch).pairs, tol=1e-15))
                self.assertLess(abs(after/(theta_translation_phase(spin, branch)*before)-1), 3e-14)
        with self.assertRaises(ValueError):
            theta_translation_phase(((1, 1), (0, 0)), ((0, .5), (.5, 0)))

    def test_all_ns_unsquared_against_independent_fermion_fredholm(self):
        q = (.013+.008j, -.019+.009j, .021-.004j)
        frame = charged_frame(q, max_mode=16)
        branch = np.array([[-1, -1], [-1, 1]])
        for beta in itertools.product((0, 1), repeat=2):
            marked = SpinCharacteristic((0, 0), beta)
            charge = marked.charge_frame(branch)
            actual = fixed_spin_chiral_partition(
                q, frame.omega_charge-branch, marked.pairs, period_branch=branch, max_mode=16)
            direct = theta_physical_fermion_fredholm(
                q, charge.all_ns_determinant_lifts(), max_mode=16).determinant_values[0]
            self.assertLess(abs(actual["majorana_chiral"]/direct-1), 1e-13)

    def test_ground_normalization_for_each_ramond_edge_placement(self):
        q = (1e-8+2e-9j, -1.3e-8+1e-9j, .9e-8-2e-9j)
        frame = charged_frame(q, max_mode=4)
        for alpha in ((1, 0), (0, 1), (1, 1)):
            for beta in itertools.product((0, 1), repeat=2):
                spin = SpinCharacteristic(alpha, beta)
                if spin.arf:
                    continue
                actual = fixed_spin_chiral_partition(q, frame.omega_charge, spin.pairs,
                    period_branch=((0, 0), (0, 0)), max_mode=4, radial_steps=12)
                logs = sum(cmath.log(z) for z, s in zip(q, spin.theta_edge_sectors) if s == "R")
                leading = math.sqrt(2)*cmath.exp(logs/16)
                self.assertLess(abs(actual["majorana_chiral"]/leading-1), 2e-4)

    def test_odd_spin_zero_modes_have_no_phase(self):
        q = (.013+.008j, -.019+.009j, .021-.004j)
        frame = charged_frame(q, max_mode=8)
        for alpha, beta in itertools.product(itertools.product((0, 1), repeat=2), repeat=2):
            spin = SpinCharacteristic(alpha, beta)
            if not spin.arf:
                continue
            actual = fixed_spin_chiral_partition(q, frame.omega_charge, spin.pairs,
                period_branch=((0, 0), (0, 0)), max_mode=8, radial_steps=4)
            self.assertEqual(actual["majorana_chiral"], 0j)
            self.assertIsNone(actual["majorana_phase_radians"])

    def test_eight_ramond_turns_preserve_the_square_and_reverse_the_sign(self):
        previous = None
        initial = None
        for step in range(8*24+1):
            angle = 2*math.pi*step/24
            q = (.013*cmath.exp(1j*angle), .017, .011)
            winding = round((angle-cmath.phase(q[0]))/(2*math.pi))
            shift = np.diag([winding, 0])
            frame = charged_frame(q, max_mode=8)
            previous = fixed_spin_chiral_partition(q, frame.omega_charge+shift, ((1, 1), (0, 0)),
                period_branch=-shift, max_mode=8, previous=previous, radial_steps=8)
            if initial is None:
                initial = previous
            if step % 24 == 0:
                expected = cmath.exp(1j*math.pi*(step//24)/8)
                self.assertLess(abs(previous["majorana_chiral"]/initial["majorana_chiral"]-expected), 1e-12)
        self.assertLess(abs(previous["dirac_chiral"]/initial["dirac_chiral"]-1), 1e-12)
        self.assertLess(abs(previous["majorana_chiral"]/initial["majorana_chiral"]+1), 1e-12)

    def test_continuation_rejects_a_reset_period_marking(self):
        q0 = (.013*cmath.exp(1j*(math.pi-.1)), .017, .011)
        q1 = (.013*cmath.exp(1j*(-math.pi+.1)), .017, .011)
        f0, f1 = charged_frame(q0, max_mode=8), charged_frame(q1, max_mode=8)
        previous = fixed_spin_chiral_partition(q0, f0.omega_charge, ((1, 1), (0, 0)),
            period_branch=((0, 0), (0, 0)), max_mode=8)
        with self.assertRaisesRegex(ValueError, "period marking changed"):
            fixed_spin_chiral_partition(q1, f1.omega_charge, ((1, 1), (0, 0)),
                period_branch=((0, 0), (0, 0)), max_mode=8, previous=previous)
        shift = np.diag([1, 0])
        continued = fixed_spin_chiral_partition(q1, f1.omega_charge+shift, ((1, 1), (0, 0)),
            period_branch=-shift, max_mode=8, previous=previous)
        self.assertEqual(continued["chiral_branch"]["log_windings"], (1, 0, 0))
        self.assertLess(abs(cmath.phase(continued["majorana_chiral"]/previous["majorana_chiral"])), .1)

    def test_continuation_rejects_a_spin_change_and_undersampled_path(self):
        q = (.013, .017, .011)
        frame = charged_frame(q, max_mode=8)
        previous = fixed_spin_chiral_partition(q, frame.omega_charge, ((1, 1), (0, 0)),
            period_branch=((0, 0), (0, 0)), max_mode=8, radial_steps=8)
        with self.assertRaisesRegex(ValueError, "same marked spin"):
            fixed_spin_chiral_partition(q, frame.omega_charge, ((1, 1), (1, 1)),
                period_branch=((0, 0), (0, 0)), max_mode=8, previous=previous)
        jumped = (-.013, .017, .011)
        frame = charged_frame(jumped, max_mode=8)
        with self.assertRaisesRegex(ValueError, "more finely"):
            fixed_spin_chiral_partition(jumped, frame.omega_charge, ((1, 1), (0, 0)),
                period_branch=((0, 0), (0, 0)), max_mode=8, previous=previous)


if __name__ == "__main__":
    unittest.main()
