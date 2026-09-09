"""Small regression tests for the independent NS pillow audit."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
import time

import mpmath as mp
import sympy as sp

from ns_pillow_direct_pbw import NSModule, NSPrimaryWard, DirectNSSpherePBW, basis
from ns_pillow_elliptic_audit import PillowMap, NSEllipticRecursion, indices
from check_ns_pillow_direct_pbw import symbolic_residue_checks


class NSPillowAuditTests(unittest.TestCase):
    def test_gram_and_odd_vertex_normalization(self):
        c,h,d,k = sp.symbols("c h d k")
        module = NSModule(c,h,True)
        g = (("G",-1),)
        self.assertEqual(module.inner(g,g),2*h)
        ward = NSPrimaryWard(c,(h,d,k),True)
        self.assertEqual(ward.value((),0,g),-1)
        self.assertEqual(ward.value(g,0,()),1)
        self.assertEqual(ward.value(g,0,g),h+k-d)
        self.assertEqual(len(basis(20)),161)

    def test_symbolic_leading_five_point_coefficients(self):
        b = sp.symbols("b")
        d = sp.symbols("d1:6")
        h = sp.symbols("h1:3")
        recursion = NSEllipticRecursion(b,d,h,True)
        expected = {(0,0):1,(1,0):-1/h[0],(0,1):-1/h[1],
                    (1,1):(d[2]-h[0]-h[1])/(h[0]*h[1]),
                    (2,0):-2*(d[0]-d[1])*(d[2]+h[0]-h[1])/h[0],
                    (0,2):2*(d[3]-d[4])*(d[2]-h[0]+h[1])/h[1]}
        for key,value in expected.items():
            self.assertEqual(sp.cancel(recursion.coefficient(key)-value),0)

    def test_coordinate_truncation_is_consistent(self):
        for m in (1,2,3):
            low,high = PillowMap(m,3,True),PillowMap(m,10,True)
            for left,right in zip([low.zunit,*low.yunits,low.inverse_character],
                                  [high.zunit,*high.yunits,high.inverse_character]):
                self.assertEqual(left,{k:v for k,v in right.items() if sum(k)<=6})

    def test_four_five_six_point_direct_comparison(self):
        with mp.workdps(50):
            b = mp.mpf("1.27")
            c = mp.mpf("1.5")+3*(b+1/b)**2
            ds = list(map(mp.mpf,(".31",".42",".53",".37",".47",".28")))
            hs = list(map(mp.mpf,(".73","1.10","1.37")))
            for m in (1,2,3):
                external,internal = ds[:m+1]+ds[-2:],hs[:m]
                direct = DirectNSSpherePBW(c,external,internal)
                keys = list(indices(m,6))
                pulled = PillowMap(m,3).pullback(
                    {k:direct.coefficient(k) for k in keys},c,external,internal)
                recursion = NSEllipticRecursion(b,external,internal)
                for key in keys:
                    self.assertLess(abs(pulled.get(key,0)-recursion.coefficient(key)),mp.mpf("1e-40"))

    def test_exact_certificate_detects_wrong_residue_and_regular_part(self):
        b,c,h = sp.symbols("b c h")
        ds = sp.symbols("d1:5")
        direct = DirectNSSpherePBW(c,ds,(h,),True)
        keys = [(0,),(1,),(2,)]
        pulled = PillowMap(1,1,True).pullback({k:direct.coefficient(k) for k in keys},c,ds,(h,))
        with redirect_stdout(StringIO()):
            correct = symbolic_residue_checks(keys,pulled,b,c,ds,(h,),time.monotonic())
            self.assertTrue(all(row["exact_zero"] for row in correct))
            for error in (sp.S.One,1/h,1/(h+1)):
                broken = dict(pulled)
                broken[(1,)] += error
                checked = symbolic_residue_checks(keys,broken,b,c,ds,(h,),time.monotonic())
                self.assertFalse(checked[1]["exact_zero"])


if __name__ == "__main__":
    unittest.main()
