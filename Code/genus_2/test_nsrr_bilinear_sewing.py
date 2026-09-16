"""Regress physical NSRR normalization, BPZ descendants and relative phases."""
import itertools
import unittest
import numpy as np
import sympy as sp

from audit_nsrr_bilinear_sewing import clifford_derivation, block_coefficients
from nsrr_bilinear_sewing import CHANNELS, nsrr_bilinear_matrix, contract_nsrr_bilinear
from nsrr_bilinear_state_oracle import PhysicalNSRRBPZ


class NSRRBilinearTests(unittest.TestCase):
    def test_exact_clifford_reduction(self):
        result = clifford_derivation()
        self.assertEqual(len(result['exact_family_reductions']), 16)

    def test_bilinear_descendants_independent_anti_data(self):
        oracle = PhysicalNSRRBPZ(anti_h=sp.Rational(9, 10)+sp.I/7,
            anti_c=17-sp.I/9, anti_beta2=sp.Rational(1, 8)+sp.I/3,
            anti_beta3=sp.Rational(1, 9)+sp.I/2)
        levels = [(0,0,0),(1,0,0),(0,1,0)]
        f, ft = block_coefficients(oracle,levels), block_coefficients(oracle,levels,True)
        left, right = (1+.3j,.2-.4j), (.7-.6j,-.1+.8j)
        for n,m in ((levels[1],levels[0]),(levels[1],levels[1]),(levels[2],levels[1])):
            for s,r in itertools.product((1,-1),repeat=2):
                full=oracle.coefficient(n,m,physical_lifts_slots=(s,r,1))
                expected=sum(left[i]*right[j]/4*full[e,ep] for i,e in enumerate((1,-1)) for j,ep in enumerate((1,-1)))
                actual=contract_nsrr_bilinear(descendant_blocks=f[n],antiholomorphic_blocks=ft[m],
                    left_bry=left,right_bry=right,physical_lifts_slots=(s,r,1),
                    primary=1,antiholomorphic_primary=1)['total']
                self.assertAlmostEqual(actual,expected,places=12)

    def test_half_level_retains_its_bilinear_phase(self):
        oracle=PhysicalNSRRBPZ()
        f=block_coefficients(oracle,[(0,0,0),(1,0,0)])
        ft=block_coefficients(oracle,[(0,0,0)],True)
        # beta=i p/sqrt(2); the supplied BRY constants are E=2,O=3.
        p2,p3=[float(sp.im(beta))*2**.5 for beta in oracle.betas]
        coefficient=(4*(p3-p2)**2+9*(p3+p2)**2)/(8*float(oracle.h))
        for s in (1,-1):
            z=contract_nsrr_bilinear(descendant_blocks=f[1,0,0],antiholomorphic_blocks=ft[0,0,0],
                left_bry=(2,3),right_bry=(2,3),physical_lifts_slots=(s,-1,1),
                primary=1,antiholomorphic_primary=1)['total']
            self.assertAlmostEqual(z,1j*s*coefficient)

    def test_simultaneous_tube_sign_reversal_is_redundant(self):
        for signs in itertools.product((1,-1),repeat=3):
            m=nsrr_bilinear_matrix((1+.2j,.7),(2,-.3j),physical_lifts_slots=signs)
            reverse=nsrr_bilinear_matrix((1+.2j,.7),(2,-.3j),physical_lifts_slots=tuple(-v for v in signs))
            np.testing.assert_array_equal(m,reverse)


if __name__ == '__main__':
    unittest.main()
