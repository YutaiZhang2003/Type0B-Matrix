"""Fast regression tests for the component research extension."""

from itertools import product
import unittest

import mpmath as mp
import sympy as sp

from ns_pillow_components import (ComponentEllipticRecursion, ExactPBWSeeds,
    InteriorUnitSeed, component_markings, component_pullback,
    component_vertex_labels, effective_weights, polynomial_part)
from ns_pillow_direct_pbw import DirectNSSpherePBW, NSPrimaryWard
from ns_pillow_elliptic_audit import NSEllipticRecursion, indices


class ComponentTests(unittest.TestCase):
    def test_eight_human_three_forms(self):
        c,h,d,k=sp.symbols('c h d k')
        ward=NSPrimaryWard(c,(h,d,k),True)
        expected={(0,0,0):1,(1,0,0):1,(0,1,0):1,(0,0,1):-1,
                  (1,1,0):h+d-k,(1,0,1):h-d+k,(0,1,1):h-d-k,
                  (1,1,1):-h-d-k+sp.Rational(1,2)}
        g=(('G',-1),)
        for beta,value in expected.items():
            self.assertEqual(sp.expand(ward.value(g if beta[0] else (),beta[1],g if beta[2] else ())-value),0)

    def test_vertex_parity_conservation_all_markings(self):
        for n in (4,5,6):
            for beta in product((0,1),repeat=n):
                for eps in product((0,1),repeat=n-3):
                    alpha=component_vertex_labels(eps,beta)
                    self.assertEqual(sum(alpha)%2,sum(beta)%2)
                    for edge in range(n-3):
                        child=list(eps);child[edge]^=1
                        actual=component_vertex_labels(child,beta)
                        expected=list(alpha);expected[edge]^=1;expected[edge+1]^=1
                        self.assertEqual(actual,tuple(expected))

    def test_component_validation(self):
        for beta in ((0,1),(0,1,0,2),(0,1,False,0)):
            with self.assertRaises(ValueError):
                component_markings(beta,4)
        self.assertEqual(effective_weights((1,2,3,4),(0,1,0,1)),(1,sp.Rational(5,2),3,sp.Rational(9,2)))

    def test_polynomial_part_includes_constant_and_linear_terms(self):
        h,a=sp.symbols('h a')
        self.assertEqual(polynomial_part(h+3+a/(h-2),h),h+3)

    def test_bottom_recursion_unchanged(self):
        with mp.workdps(50):
            b=mp.mpf('1.27');d=tuple(map(mp.mpf,('.31','.42','.53','.47','.28')));h=(mp.mpf('.73'),mp.mpf('1.1'))
            original=NSEllipticRecursion(b,d,h)
            extended=ComponentEllipticRecursion(b,d,h)
            for key in indices(2,6):
                self.assertEqual(original.coefficient(key),extended.coefficient(key))

    def test_unit_seed_rejected_at_caps(self):
        with self.assertRaises(ValueError):
            InteriorUnitSeed((0,1,1,1,0))
        with self.assertRaisesRegex(ValueError,'own regular seed'):
            ComponentEllipticRecursion(sp.Rational(127,100),(1,2,3,4),(sp.Symbol('h'),),
                                      (0,1,1,0),symbolic=True)

    def test_generic_four_point_seed_is_not_constant(self):
        b=sp.Rational(127,100);c=sp.Rational(3,2)+3*(b+1/b)**2
        d=sp.symbols('d1:5')
        seeds=ExactPBWSeeds(c,d,(0,1,1,0),1).build()
        expected=-2*seeds.base+2*(d[0]-d[1]-d[2]+d[3])
        self.assertEqual(sp.expand(seeds.expression((1,))-expected),0)
        self.assertEqual(seeds.expression((0,)),1)
        self.assertEqual(seeds.expression((2,)),0)

    def test_generic_five_point_three_upper_seed(self):
        c=sp.Symbol('c');d=sp.symbols('d1:6')
        seeds=ExactPBWSeeds(c,d,(0,1,1,1,0),1).build()
        a=seeds.differences[0];h=seeds.base
        expected={(0,0):1,(1,0):a-d[2],(0,1):-a-d[2],
                  (1,1):2*h+a+d[2]-sp.Rational(1,2)+2*(d[1]+d[3]-d[0]-d[4])}
        for key,value in expected.items():
            self.assertEqual(sp.expand(seeds.expression(key)-value),0)

    def test_compiled_seed_reuse_and_guardrails(self):
        b=sp.Rational(127,100);c=sp.Rational(3,2)+3*(b+1/b)**2
        d=tuple(map(sp.Rational,('.31','.42','.53','.47')));beta=(0,1,1,0)
        exact=ExactPBWSeeds(c,d,beta,2).build()
        with mp.workdps(60):
            seed=exact.numeric(dps=60)
            ds=tuple(mp.mpf(str(x.p))/x.q for x in d);bb=mp.mpf('1.27');hh=(mp.mpf('1.123'),)
            hblock=ComponentEllipticRecursion(bb,ds,hh,beta,seed=seed)
            cc=mp.mpf('1.5')+3*(bb+1/bb)**2
            direct=DirectNSSpherePBW(cc,ds,hh,external_descendants=beta)
            plane={key:direct.coefficient(key) for key in indices(1,4)}
            reference=component_pullback(plane,cc,ds,hh,beta,2)
            for key in indices(1,4):
                self.assertLess(abs(hblock.coefficient(key)-reference.get(key,0)),mp.mpf('1e-50'))
            with self.assertRaises(ValueError):
                seed.coefficient((5,),hh)
            with self.assertRaisesRegex(ValueError,'parameters'):
                ComponentEllipticRecursion(mp.mpf('1.3'),ds,hh,beta,seed=seed)
            with self.assertRaisesRegex(ValueError,'markings'):
                ComponentEllipticRecursion(bb,ds,hh,(0,0,0,0),seed=seed)
        with mp.workdps(70),self.assertRaisesRegex(ValueError,'recompile'):
            seed.coefficient((0,),(mp.mpf('1.123'),))

    def test_interior_upper_fast_seed_against_direct_pbw(self):
        with mp.workdps(60):
            b=mp.mpf('1.43');c=mp.mpf('1.5')+3*(b+1/b)**2
            d=tuple(map(mp.mpf,('.22','.61','.39','.74','.45')));h=(mp.mpf('1.13'),mp.mpf('.85'))
            beta=(0,0,1,0,0)
            direct=DirectNSSpherePBW(c,d,h,external_descendants=beta)
            plane={key:direct.coefficient(key) for key in indices(2,6)}
            reference=component_pullback(plane,c,d,h,beta,3)
            recursion=ComponentEllipticRecursion(b,d,h,beta,seed=InteriorUnitSeed(beta))
            for key in indices(2,6):
                self.assertLess(abs(recursion.coefficient(key)-reference.get(key,0)),mp.mpf('1e-50'))


if __name__=='__main__':
    unittest.main()
