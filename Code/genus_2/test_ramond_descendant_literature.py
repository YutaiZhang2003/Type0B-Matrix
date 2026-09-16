"""Ramond bra ordering and explicit HJS torus descendant coefficients."""
import unittest

import numpy as np

from audit_ramond_descendant_literature import (
    RamondVermaModule, RamondThreePointWardMatrix, hjs_torus_coefficients,
)
from ramond_genus2_direct import RRNSDescendantThreeForm


class RamondLiteratureTests(unittest.TestCase):
    def test_hjs_torus_coefficients_both_signs_through_level_two(self):
        # Published Appendix C coefficients are independent of our Ward code.
        for b,p,d in ((1.1,.73,.83),(1.27,1.11,.42),(1.2,.83+.17j,.61-.13j)):
            c=1.5+3*(b+1/b)**2;h=c/24+p*p
            module=RamondVermaModule(c=c,weight=h)
            for sign in (1,-1):
                vertex=RamondThreePointWardMatrix(left_module=module,right_module=module,
                                                  external_ns_weight=d,sign=sign)
                for n,even_expected in enumerate(hjs_torus_coefficients(b,h,d,sign)):
                    for parity in (0,1):
                        # HJS section 2.2: F_o^(sign)=sign*F_e^(sign).
                        expected=sign**parity*even_expected
                        gram=np.asarray(module.gram_matrix(n,parity))
                        inserted=np.asarray(vertex.matrix(n,n,parity))
                        actual=np.trace(np.linalg.solve(gram,inserted))
                        self.assertLess(abs(actual-expected)/max(1,abs(expected)),2e-12)

    def test_identity_insertion_is_the_entire_gram_matrix(self):
        c=16.2;h=c/24+.7**2
        module=RamondVermaModule(c=c,weight=h)
        vertex=RamondThreePointWardMatrix(left_module=module,right_module=module,
                                         external_ns_weight=0,sign=1)
        for n in range(4):
            for parity in (0,1):
                basis=module.basis(n,parity)
                for left in basis:
                    for right in basis:
                        gram=module.inner_product(left,right)
                        actual=vertex.value(left,right)
                        self.assertLess(abs(actual-gram)/max(1,abs(gram)),1e-12)

    def test_ramond_ward_sum_terminates_at_the_inserted_state_level(self):
        # [G_1,V(G_-1/2 L_-1^2 nu)] includes j=3, since
        # binomial(3/2,3) is nonzero. Truncating at j=2 gave a wrong value.
        form=RRNSDescendantThreeForm(c=37.25,left_weight=37.25/24-.67**2,
                                    ns_weight=.73,right_weight=37.25/24-.83**2,sign=1)
        vertex=RamondThreePointWardMatrix(left_module=form.left_module,right_module=form.right_module,
                                         external_ns_weight=.73,sign=1)
        left=(("L",-1),("L",-1),("G",-1),("G",0))
        right=(("G",0),)
        self.assertLess(abs(form.value(left,(("G",-1),),right)-vertex.value(left,right,1)),1e-11)

    def test_bra_respects_mixed_mode_commutator(self):
        c=16.2;hl=c/24+.7**2;hr=c/24+.4**2
        for sign in (1,-1):
            form=RRNSDescendantThreeForm(c=c,left_weight=hl,ns_weight=.7,right_weight=hr,sign=sign)
            vertex=RamondThreePointWardMatrix(left_module=form.left_module,right_module=form.right_module,
                                             external_ns_weight=.7,sign=sign)
            for ground in ((),(("G",0),)):
                lg=(("L",-1),("G",-1))+ground
                gl=(("G",-1),("L",-1))+ground
                g2=(("G",-2),)+ground
                for n in range(3):
                    for parity in (0,1):
                        for right in form.right_module.basis(n,parity):
                            for evaluate in (lambda left:vertex.value(left,right),
                                             lambda left:form.value(left,(),right)):
                                self.assertLess(abs(evaluate(lg)-evaluate(gl)-.5*evaluate(g2)),2e-12)

    def test_translation_covariance_with_descendants_on_all_legs(self):
        form=RRNSDescendantThreeForm(c=16.2,left_weight=.675+.7**2,
                                    ns_weight=.7,right_weight=.675+.4**2,sign=-1)
        for nl,nm,nr in ((2,3,1),(1,4,1),(2,1,0),(0,5,1)):
            for pl in (0,1):
                for pr in (0,1):
                    for left in form.left_module.basis(nl,pl):
                        for middle in form.ns_module.basis(nm):
                            for right in form.right_module.basis(nr,pr):
                                expected=(form.weights[0]+nl-form.weights[1]-nm/2-form.weights[2]-nr)*form.value(left,middle,right)
                                actual=form.value(left,(("L",-2),)+middle,right)
                                self.assertLess(abs(actual-expected)/max(1,abs(expected)),3e-11)


if __name__=="__main__":
    unittest.main()
