"""Normalization must preserve the new bilinear matrix, primary powers and phases."""
import cmath
import itertools
import unittest
from unittest.mock import patch
import numpy as np

from nsrr_bilinear_sewing import CHANNELS, nsrr_bilinear_matrix
from nsrr_normalization import (LOCAL, PROVISIONAL_FOUR, normalized_nsrr_matrix,
                                contract_normalized_nsrr)
from nsrr_resummed_sewing import resummed_integrand


class NSRRNormalizationTests(unittest.TestCase):
    def test_independent_complex_anti_data_all_signs(self):
        f = {k: (i+1)/7 + 1j*(i-2)/11 for i, k in enumerate(CHANNELS)}
        ft = {k: (9-i)/13 - 1j*(i+2)/17 for i, k in enumerate(CHANNELS)}
        left, right = (1+.3j, .7-.1j), (.2-.8j, 1.3+.4j)
        p, pt = .3+.7j, -.2+.6j
        for signs in itertools.product((1, -1), repeat=3):
            result = contract_normalized_nsrr(descendant_blocks=f, antiholomorphic_blocks=ft,
                left_bry=left, right_bry=right, physical_lifts_slots=signs,
                primary=p, antiholomorphic_primary=pt)
            local = nsrr_bilinear_matrix(left, right, physical_lifts_slots=signs)
            expected = 4*p*pt*np.array(list(f.values())) @ local @ np.array(list(ft.values()))
            self.assertAlmostEqual(result['total'], expected)
            self.assertAlmostEqual(result['diagonal']+result['interference'], expected)
            self.assertAlmostEqual(sum(result['terms'].values()), expected)
            self.assertEqual(result['normalization_status'], 'assumption')
            self.assertFalse(result['global_normalization_verified'])
            reference = contract_normalized_nsrr(normalization=LOCAL, descendant_blocks=f,
                antiholomorphic_blocks=ft, left_bry=left, right_bry=right,
                physical_lifts_slots=signs, primary=p, antiholomorphic_primary=pt)
            self.assertAlmostEqual(cmath.phase(result['total']/reference['total']), 0.)

    def test_matrix_has_factor_four_once(self):
        result = normalized_nsrr_matrix((2, 3), (5, 7), physical_lifts_slots=(1,-1,1))
        # (++), f=0 has E_L E_R/2 = 5, not /8 and not 16 times /8.
        self.assertEqual(result['matrix'][0,0], 5.)
        self.assertEqual(result['matrix'][0,4], 5.)
        self.assertEqual(result['matrix'][1,1], 0.)
        with self.assertRaises(ValueError):
            normalized_nsrr_matrix((2,3),(5,7),physical_lifts_slots=(1,-1,1),normalization='fit')

    @patch('nsrr_resummed_sewing.ResummedNSRR')
    def test_resummed_boundary_changes_integrand_only(self, factory):
        runtime = factory.return_value
        runtime.channels.return_value = {k: np.array([1.+.2j*i, .3-.1j*i]) for i,k in enumerate(CHANNELS)}
        runtime.project.side_effect = lambda vector, lift: sum(vector)
        runtime.diagnostics.return_value = {}
        args = dict(b=1.4, momenta_geometry=(.31,.47,.52),
            q_geometry=(.03+.01j,-.02+.01j,.04-.01j),
            lifts_geometry=((1,1,1),(1,-1,1)),bry_constants=(2.,3.),
            branch_level=3, recursion_order=3, physical_lifts_slots=(1,-1,1))
        reference = resummed_integrand(**args)
        assumed = resummed_integrand(normalization=PROVISIONAL_FOUR, **args)
        for key in ('blocks','descendant_blocks','primary_prefactor','primary_weights_slots','log_q_slots'):
            self.assertEqual(reference[key], assumed[key])
        for key in ('total','diagonal','interference'):
            self.assertAlmostEqual(assumed['integrand'][key], 4*reference['integrand'][key])
        self.assertEqual(assumed['normalization_status'], 'assumption')
        self.assertFalse(assumed['human_bilinear_pairing_verified'])
        self.assertTrue(assumed['local_bilinear_kernel_verified'])
        with self.assertRaises(ValueError):
            resummed_integrand(normalization=PROVISIONAL_FOUR,sewing_convention='legacy-times-four',**args)


if __name__ == '__main__':
    unittest.main()
