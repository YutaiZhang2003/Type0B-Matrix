"""Keep physical grading independent of the double-Virasoro auxiliary metric."""
import itertools
import unittest

import numpy as np

from graded_sewing_audit import graded_tensor_gram, theta_bilinear_contraction


def product_pants(holo, anti):
    # State order is (a,abar),(b,bbar),(c,cbar). A common vertex-ordering
    # Koszul sign at both pants cancels since each edge pairing is even.
    return np.einsum("abc,def->adbecf", holo, anti).reshape(
        tuple(a*b for a, b in zip(holo.shape, anti.shape)))


class GradedSewingTests(unittest.TestCase):
    def test_ground_product_metric_has_exchange_sign(self):
        # Physical grading persists even when the chiral odd norm is i.
        b = np.diag([1, 1j])
        bt = np.diag([1, -1j])
        actual = graded_tensor_gram(b, (0, 1), bt, (0, 1))
        np.testing.assert_array_equal(actual, np.diag([1, -1j, 1j, -1]))
        self.assertEqual(np.kron(b, bt)[3, 3], 1)

    def test_all_64_parity_signs_reproduce_note_factorization(self):
        # Q(p+pt)+p.pt = Q(p)+Q(pt)+(sum p)(sum pt) mod 2.
        q = lambda p: sum(p[i]*p[j] for i in range(3) for j in range(i+1, 3)) % 2
        for p in itertools.product((0, 1), repeat=3):
            for pt in itertools.product((0, 1), repeat=3):
                combined = tuple((a+b) % 2 for a, b in zip(p, pt))
                self.assertEqual((q(combined)+sum(a*b for a, b in zip(p, pt))) % 2,
                                 (q(p)+q(pt)+sum(p)*sum(pt)) % 2)

    def test_full_graded_contraction_matches_factorized_blocks(self):
        rng = np.random.default_rng(623)
        p = np.array([0, 1])
        pairings = [np.diag([1.2, .8j]), np.diag([.9, 1.1j]), np.diag([1.4, .7j])]
        anti_pairings = [g.conjugate() for g in pairings]
        full_pairings = [graded_tensor_gram(g, p, gt, p)
                         for g, gt in zip(pairings, anti_pairings)]
        full_p = ((p[:, None]+p[None, :]) % 2).ravel()
        parity = np.indices((2, 2, 2)).sum(axis=0) % 2
        for f in (0, 1):
            for ft in (0, 1):
                tensors = [rng.normal(size=(2, 2, 2))+1j*rng.normal(size=(2, 2, 2))
                           for _ in range(4)]
                l, r = [x*(parity == f) for x in tensors[:2]]
                lt, rt = [x*(parity == ft) for x in tensors[2:]]
                direct = theta_bilinear_contraction(product_pants(l, lt), product_pants(r, rt),
                                                     full_pairings, [full_p]*3)
                factorized = ((-1)**(f*ft)
                              *theta_bilinear_contraction(l, r, pairings, [p]*3)
                              *theta_bilinear_contraction(lt, rt, anti_pairings, [p]*3))
                np.testing.assert_allclose(direct, factorized, rtol=2e-13, atol=2e-13)

    def test_ungraded_replacement_fails_odd_sector(self):
        p = np.array([0, 1])
        g = np.eye(2)
        rho = np.zeros((2, 2, 2))
        rho[1, 0, 0] = 1
        full = product_pants(rho, rho)
        full_p = [0, 1, 1, 0]
        graded = theta_bilinear_contraction(full, full,
                    [graded_tensor_gram(g, p, g, p)]*3, [full_p]*3)
        ungraded = theta_bilinear_contraction(full, full, [np.eye(4)]*3, [full_p]*3)
        self.assertEqual(graded, -1)
        self.assertEqual(ungraded, 1)

    def test_covariant_basis_change_preserves_contraction(self):
        rng = np.random.default_rng(942)
        p = np.array([0, 0, 1, 1])
        grams = [np.diag([1.1, 1.4, 1, 1.3]) for _ in range(3)]
        changes = [np.array([[1, .2j, 0, 0], [.3, 1, 0, 0],
                             [0, 0, 1j, .1], [0, 0, .2j, .8j]]) for _ in range(3)]
        left = rng.normal(size=(4, 4, 4))+1j*rng.normal(size=(4, 4, 4))
        right = rng.normal(size=(4, 4, 4))+1j*rng.normal(size=(4, 4, 4))
        expected = theta_bilinear_contraction(left, right, grams, [p]*3)
        changed_grams = [s.T@g@s for s, g in zip(changes, grams)]
        transform = lambda t: np.einsum("ai,bj,ck,abc->ijk", *changes, t)
        actual = theta_bilinear_contraction(transform(left), transform(right), changed_grams, [p]*3)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        # Changing only the metric is not a change of convention.
        wrong = theta_bilinear_contraction(left, right, changed_grams, [p]*3)
        self.assertGreater(abs(wrong-expected), .1)

    def test_odd_rephasing_changes_norm_not_the_amplitude(self):
        g = np.eye(2)
        s = np.diag([1, 1j])
        changed = s.T@g@s
        np.testing.assert_array_equal(changed, np.diag([1, -1]))
        rho = np.zeros((2, 2, 2), dtype=complex)
        rho[1, 1, 0] = 1
        transformed = np.einsum("ai,bj,ck,abc->ijk", s, s, s, rho)
        self.assertEqual(theta_bilinear_contraction(rho, rho, [g]*3, [[0, 1]]*3),
                         theta_bilinear_contraction(transformed, transformed, [changed]*3, [[0, 1]]*3))

    def test_odd_coefficient_i_and_decomposition_minus_are_both_required(self):
        c_top = 1.7
        c_note = 1j*c_top
        self.assertAlmostEqual((-c_note**2).real, c_top**2)
        self.assertLess((c_note**2).real, 0)

    def test_odd_pairing_is_rejected(self):
        with self.assertRaises(ValueError):
            graded_tensor_gram([[1, .2], [.2, 1]], [0, 1], np.eye(2), [0, 1])


if __name__ == "__main__":
    unittest.main()
