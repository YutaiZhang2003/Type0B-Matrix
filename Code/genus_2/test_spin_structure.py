"""Geometric spin transport and basis-boundary regression tests."""
import itertools
import cmath
import unittest

import numpy as np

from spin_structure import (ALL_LIFTS, ALL_PARITIES, SpinCharacteristic,
                            ThetaSpinFrame, ThetaLogBranch, UnverifiedSpinSewing,
                            human_theta_orientation,
                            remove_human_theta_quadratic_factor,
                            require_spin_sewing_certificate, theta_log_windings)


class SpinStructureTests(unittest.TestCase):
    def test_quadratic_refinement_and_arf_count(self):
        cycles = tuple(itertools.product((0, 1), repeat=4))
        odd = 0
        for bits in cycles:
            spin = SpinCharacteristic(bits[:2], bits[2:])
            odd += spin.arf
            for x, y in itertools.product(cycles, repeat=2):
                xy = tuple(a ^ b for a, b in zip(x, y))
                intersection = sum(x[i]*y[i+2]+x[i+2]*y[i] for i in range(2)) % 2
                self.assertEqual(spin.quadratic_refinement(xy[:2], xy[2:]),
                                 spin.quadratic_refinement(x[:2], x[2:])
                                 ^ spin.quadratic_refinement(y[:2], y[2:]) ^ intersection)
        self.assertEqual(odd, 6)

    def test_transport_composition_and_homology(self):
        eye, zero = np.eye(2, dtype=int), np.zeros((2, 2), dtype=int)
        s = np.block([[zero, -eye], [eye, zero]])
        branch = np.asarray([[1, 1], [1, 0]])
        t = np.block([[eye, branch], [zero, eye]])
        for bits in itertools.product((0, 1), repeat=4):
            spin = SpinCharacteristic(bits[:2], bits[2:])
            self.assertEqual(spin.transport(s).transport(t), spin.transport(t@s))
            self.assertEqual(spin.transport(t), spin.charge_frame(branch))
            for m in (s, t, t@s):
                target = spin.transport(m)
                a, b, c, d = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
                for v in itertools.product((0, 1), repeat=4):
                    av, bv = np.asarray(v[:2]), np.asarray(v[2:])
                    old_a, old_b = (d.T@av+b.T@bv) % 2, (c.T@av+a.T@bv) % 2
                    self.assertEqual(target.quadratic_refinement(av, bv),
                                     spin.quadratic_refinement(old_a, old_b))

    def test_invalid_transport_and_ramond_lift_shortcut_rejected(self):
        with self.assertRaises(ValueError):
            SpinCharacteristic((0, 2), (0, 0))
        with self.assertRaises(ValueError):
            SpinCharacteristic((0, 0), (0, 0)).transport(2*np.eye(4))
        r = SpinCharacteristic((1, 1), (0, 0))
        self.assertEqual(r.theta_edge_sectors, ('R', 'R', 'NS'))
        with self.assertRaisesRegex(ValueError, 'Ramond'):
            r.all_ns_determinant_lifts()

    def test_period_branch_is_explicit_and_checked(self):
        spin = SpinCharacteristic((0, 0), (0, 0))
        frame = ThetaSpinFrame(spin, (.02+.01j, .03, -.04+.02j), ((0, 0), (0, 1)))
        self.assertEqual(frame.charge_spin.pairs, ((0, 0), (0, 1)))
        self.assertEqual(frame.charge_spin.all_ns_determinant_lifts(), (1, -1, 1))
        omega = np.asarray([[1.1j, .2j], [.2j, 1.3j]])
        self.assertEqual(frame.validate_periods(omega, omega+np.diag([0, 1])), 0)
        with self.assertRaisesRegex(ValueError, 'branch mismatch'):
            frame.validate_periods(omega, omega)
        self.assertNotEqual(frame.digest, ThetaSpinFrame(spin, frame.q_values, ((0, 0), (0, 0))).digest)

    def test_continuous_roots_through_two_turns_on_each_edge(self):
        reference_lifts = (1, -1, 1)
        for edge, direction in itertools.product(range(3), (1, -1)):
            state = ThetaLogBranch((.02, .03, .04))
            for angle in np.linspace(0, direction*4*np.pi, 97)[1:]:
                q = [.02, .03, .04]
                q[edge] *= cmath.exp(1j*angle)
                state = state.advance(q)
                expected = cmath.sqrt((.02, .03, .04)[edge])*cmath.exp(1j*angle/2)
                self.assertAlmostEqual(state.square_roots[edge], expected)
                eta = state.principal_root_lifts(reference_lifts)
                self.assertAlmostEqual(eta[edge]*cmath.sqrt(q[edge]), reference_lifts[edge]*expected)
                b_current = -np.asarray(state.period_shift)
                self.assertEqual(theta_log_windings(((0, 0), (0, 0)), b_current), state.windings)
            self.assertEqual(state.windings[edge], 2*direction)
            self.assertEqual(state.root_signs, (1, 1, 1))

    def test_observed_branches_from_charge_conservation(self):
        reference = ((-1, -1), (-1, 0))
        for current, expected_w, expected_lifts in (
                (((-1, -1), (-1, 1)), (0, -1, 0), (1, 1, 1)),
                (((0, 0), (0, 1)), (0, 0, -1), (1, -1, -1))):
            w = theta_log_windings(reference, current)
            self.assertEqual(w, expected_w)
            self.assertEqual(ThetaLogBranch((.02, .03, .04), w).principal_root_lifts((1, -1, 1)),
                             expected_lifts)

    def test_root_tracking_requires_valid_and_resolved_path(self):
        with self.assertRaises(ValueError):
            ThetaLogBranch((0, .02, .03))
        with self.assertRaises(ValueError):
            ThetaLogBranch((.01, .02, .03), (0, .5, 0))
        with self.assertRaisesRegex(ValueError, 'more finely'):
            ThetaLogBranch((.01, .02, .03)).advance((-.01, .02, .03))
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            theta_log_windings(((0, 0), (0, 0)), ((0, 1), (0, 0)))

    def test_basis_conversion_on_both_vertex_parities(self):
        rng = np.random.default_rng(418)
        for sector in (0, 1):
            coefficients = {p: complex(*rng.normal(size=2)) if sum(p) % 2 == sector else 0j
                            for p in ALL_PARITIES}
            human = {e: sum(human_theta_orientation(p)*value
                            * np.prod([x**n for x, n in zip(e, p)])
                            for p, value in coefficients.items()) for e in ALL_LIFTS}
            for e in ALL_LIFTS:
                expected = sum(value*np.prod([x**n for x, n in zip(e, p)])
                               for p, value in coefficients.items())
                self.assertAlmostEqual(remove_human_theta_quadratic_factor(human, e), expected)
        with self.assertRaisesRegex(ValueError, 'all eight'):
            remove_human_theta_quadratic_factor({(1, 1, 1): 1}, (1, 1, 1))

    def test_free_test_or_eta_label_is_not_an_interacting_certificate(self):
        frame = ThetaSpinFrame(SpinCharacteristic((1, 1), (0, 0)), (.02, .03, .04), ((0, 0), (0, 1)))
        kwargs = dict(frame=frame, theory='super-Liouville', implementation_sha256='example')
        for certificate in (None, {'eta': [1, 1, 1]},
                            {'schema': 'spin-sewing-certificate-v1', 'frame_digest': frame.digest,
                             'theory': 'super-Liouville', 'implementation_sha256': 'example',
                             'status': 'verified', 'checks': {'free_field_limit': True}}):
            with self.assertRaises(UnverifiedSpinSewing):
                require_spin_sewing_certificate(certificate, **kwargs)


if __name__ == '__main__':
    unittest.main()
