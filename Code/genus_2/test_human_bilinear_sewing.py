"""Physical grading and descendant regressions for Human-Note sewing."""
import unittest
import cmath

import numpy as np

from audit_human_bilinear_sewing import (
    all_ns_states, parity_and_basis_checks, ramond_dual_obstruction,
    unrestricted_nsrr_states,
)
from human_bilinear_sewing import (
    all_ns_theta_matrix, contract_bilinear, restricted_bilinear_completeness,
)


class HumanBilinearSewingTests(unittest.TestCase):
    def test_all_ns_full_state_descendants_and_all_lifts(self):
        result = all_ns_states()
        self.assertEqual(result['checks'], 122)
        self.assertEqual(result['norm_phase_checks'], 32)
        self.assertLess(result['maximum_scaled_error'], 2e-12)

    def test_nonunitary_transport_retains_unequal_primary_factors(self):
        result = parity_and_basis_checks()
        self.assertEqual(result['exact_grading_identities'], 2048)
        self.assertLess(result['unequal_primary_transport_error'], 2e-14)

    def test_unrestricted_ramond_descendants(self):
        result = unrestricted_nsrr_states()
        self.assertEqual(result['checks'], 36)
        self.assertLess(result['maximum_scaled_error'], 3e-12)

    def test_ramond_hermitian_ket_cannot_replace_the_bilinear_dual(self):
        result = ramond_dual_obstruction()
        self.assertEqual(result['same_embedding_rank'], 1)
        self.assertEqual(result['dual_pairing'], 'identity')

    def test_ramond_restrict_then_invert_with_independent_dual(self):
        gram = np.diag([1,1j,1j,1])
        right = np.asarray([[1,0],[0,-1j],[0,1],[-1,0]])/np.sqrt(2)
        left = np.asarray([[1,0],[0,1],[0,-1j],[-1,0]])/np.sqrt(2)
        physical, completeness = restricted_bilinear_completeness(gram, right, left)
        np.testing.assert_allclose(physical, np.eye(2), atol=1e-15)
        np.testing.assert_allclose(completeness @ gram @ right, right, atol=1e-15)
        with self.assertRaisesRegex(ValueError, 'singular'):
            restricted_bilinear_completeness(gram, right, right)

    def test_bry_odd_conversion_uses_two_i_factors(self):
        left, right = (1+.3j, .4-.2j), (.7+.5j, -.6+.9j)
        matrix = all_ns_theta_matrix((left[0],1j*left[1]), (right[0],1j*right[1]))
        np.testing.assert_allclose(matrix, np.diag([left[0]*right[0],left[1]*right[1]]))
        value = contract_bilinear(matrix=matrix, holomorphic_blocks=[0,1], antiholomorphic_blocks=[0,1j])
        self.assertAlmostEqual(value, 1j*left[1]*right[1])

    def test_all_ns_node_keeps_structure_products_and_external_primary(self):
        from compare_nsrr_nsnsns_theta import all_ns_node

        class Constants:
            @staticmethod
            def ns_constants(*momenta):
                return 2j, 3j

        class Blocks:
            @staticmethod
            def block(*, sector, **kwargs):
                return (1+.2j, .3-.4j)[sector]

        q = (.03+.01j, -.02+.015j, .01-.03j)
        momenta = (.2,.3,.4)
        weights = [(1.4+1/1.4)**2/8+p*p/2 for p in momenta]
        primary = cmath.exp(sum(h*cmath.log(z) for h,z in zip(weights,q)))
        observed = all_ns_node(b=1.4, q_values=q, lifts=(1,-1,1),
                               recursion_order=1, momenta=momenta, measure=.7,
                               constants=Constants(), recursion=Blocks())
        expected = (-4*.7*abs(primary*(1+.2j))**2,
                    -9*.7*abs(primary*(.3-.4j))**2)
        np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=0)


if __name__ == '__main__':
    unittest.main()
