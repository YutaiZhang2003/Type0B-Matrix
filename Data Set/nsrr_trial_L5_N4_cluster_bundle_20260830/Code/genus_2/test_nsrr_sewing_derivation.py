"""Exact checkpoints for the separate NSRR nonchiral derivation note.

These tests do not implement a physical partition or change any checked
kernel. The small-representation dual below is an algebraic example, not
an asserted local-coordinate BPZ identification for the production run.
"""
import itertools
import unittest

import sympy as s

from ramond_pbw_generalized_ward import GeneralizedNRRWard


I = s.I
U = (1-I)/s.sqrt(2)


class NSRRSewingDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h, cls.c, cls.b2, cls.b3 = s.symbols("h c beta_2 beta_3", real=True)
        cls.forms = {(f, eta): GeneralizedNRRWard(
            p_phi=0, form_parity=f, eta=eta, h_ns=cls.h,
            h_second=cls.c/24-cls.b2**2, h_third=cls.c/24-cls.b3**2,
            beta_second=cls.b2, beta_third=cls.b3, central_charge=cls.c)
            for f in (0, 1) for eta in (1, -1)}

    def test_low_level_Ward_components(self):
        word = (("G", -s.Rational(1, 2)),)
        for eta in (1, -1):
            v = U*(self.b3-eta*self.b2)
            expected = {(0, 0, 0): 0, (0, 0, 1): -I*v,
                        (0, 1, 0): eta*v, (0, 1, 1): 0,
                        (1, 0, 0): v, (1, 0, 1): 0,
                        (1, 1, 0): 0, (1, 1, 1): eta*v}
            for (f, a, b), value in expected.items():
                actual = self.forms[f, eta].value(word, (), a, (), b)
                self.assertEqual(s.simplify(actual-value), 0)

    def test_ground_and_half_level_for_every_sign_and_lift(self):
        # Compute directly from the protected Ward forms, inverse ground
        # BPZ pairings, and the Human Note quadratic sign.
        for eta, etap, lifts, half, f in itertools.product(
                (1, -1), (1, -1), tuple(itertools.product((1, -1), repeat=3)),
                (0, 1), (0, 1)):
            l1, l2, l3 = lifts
            word = (("G", -s.Rational(1, 2)),) if half else ()
            actual = 0
            for a, b in itertools.product((0, 1), repeat=2):
                sign = (-1)**(half*a+half*b+a*b)
                metric = (2*self.h if half else 1)*I**(a+b)
                rho = self.forms[f, eta].value(word, (), a, (), b)
                rhop = self.forms[f, etap].value(word, (), a, (), b)
                actual += sign*l1**half*l2**a*l3**b*rho*rhop/metric
            k = eta*etap
            aa = (self.b3-eta*self.b2)*(self.b3-etap*self.b2)
            expected = {
                (0, 0): 1+k*l2*l3,
                (0, 1): -I*(l3-k*l2),
                (1, 0): -l1*aa*(l3-k*l2)/(2*self.h),
                (1, 1): -I*l1*aa*(1+k*l2*l3)/(2*self.h),
            }[half, f]
            with self.subTest(eta=eta, etap=etap, lifts=lifts, half=half, f=f):
                self.assertEqual(s.simplify(actual-expected), 0)

    def test_three_distinct_structure_constant_bases(self):
        ce, co = s.symbols("C_even C_odd")
        dplus, dminus = (ce+co)/2, (ce-co)/2
        cplus, cminus = (dplus+dminus)/2, (dplus-dminus)/2
        self.assertEqual(s.expand(cplus), ce/2)
        self.assertEqual(s.expand(cminus), co/2)
        self.assertEqual(s.expand(dplus**2+dminus**2), (ce**2+co**2)/2)

    @staticmethod
    def small_ground_data():
        embedding = s.Matrix([[1, 0], [0, 1], [0, 1], [-I, 0]])/s.sqrt(2)
        pairing = s.diag(1, -I, I, -1)
        return embedding, pairing

    def test_small_ket_representation_is_invariant(self):
        embedding, _ = self.small_ground_data()
        beta, betat = s.symbols("beta beta_tilde")
        g0 = s.Matrix([[0, I*beta*s.conjugate(U)], [I*beta*U, 0]])
        gt0 = s.Matrix([[0, -I*betat*U], [-I*betat*s.conjugate(U), 0]])
        holo = s.kronecker_product(g0, s.eye(2))
        anti = s.kronecker_product(s.diag(1, -1), gt0)
        self.assertEqual(s.simplify(holo*embedding-embedding*g0), s.zeros(4, 2))
        self.assertEqual(s.simplify(anti*embedding-embedding*gt0), s.zeros(4, 2))

    def test_naive_same_embedding_BPZ_pullback_is_singular(self):
        embedding, pairing = self.small_ground_data()
        self.assertEqual(s.simplify(embedding.T*pairing*embedding), s.diag(1, 0))

    def test_dual_example_and_restricted_completeness(self):
        er, pairing = self.small_ground_data()
        el = pairing.inv()*s.conjugate(er)
        g = s.simplify(el.T*pairing*er)
        self.assertEqual(g, s.eye(2))
        inverse_pairing = s.simplify(er*g.inv()*el.T)
        projector = s.simplify(inverse_pairing*pairing)
        self.assertEqual(projector*er, er)
        self.assertEqual(s.simplify(projector**2-projector), s.zeros(4))
        self.assertEqual(projector.rank(), 2)
        total_parity = s.diag(1, -1, -1, 1)
        holo_parity = s.diag(1, 1, -1, -1)
        self.assertEqual(projector*total_parity-total_parity*projector, s.zeros(4))
        self.assertNotEqual(projector*holo_parity-holo_parity*projector, s.zeros(4))

    def test_common_lift_preserves_small_space_but_one_sided_lift_does_not(self):
        er, pairing = self.small_ground_data()
        el = pairing.inv()*s.conjugate(er)
        projector = s.simplify(er*el.T*pairing)
        common = s.diag(1, -1, -1, 1)
        one_sided = s.diag(1, 1, -1, -1)
        self.assertEqual(projector*common-common*projector, s.zeros(4))
        self.assertNotEqual(projector*one_sided-one_sided*projector, s.zeros(4))

    def test_two_sided_basis_changes_do_not_change_completeness(self):
        er, pairing = self.small_ground_data()
        el = pairing.inv()*s.conjugate(er)
        g = el.T*pairing*er
        sr, sl = s.diag(2, 3*I), s.diag(1+I, 4)
        newg = (el*sl).T*pairing*(er*sr)
        old = er*g.inv()*el.T
        new = er*sr*newg.inv()*(el*sl).T
        self.assertEqual(s.simplify(new-old), s.zeros(4))


if __name__ == "__main__":
    unittest.main()
